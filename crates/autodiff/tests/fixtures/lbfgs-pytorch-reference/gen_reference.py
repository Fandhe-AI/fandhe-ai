#!/usr/bin/env python3
"""LBFGS PyTorch 参照値フィクスチャ生成スクリプト（イシュー #2197）。

`torch.optim.LBFGS` を実行して、閉形式で検証しにくい strong Wolfe line
search・two-loop recursion・履歴 FIFO を含む更新系列を実測する。
rmsprop-pytorch-reference/README.md と同じ方針（実 PyTorch 実行値・
演算順ドリフト確認・sha256 記録）。

目的関数は凸・良条件の最小二乗（`loss = mean((X @ w + b - y) ** 2)`）で、
Rust 側 parity テストの closure はホスト解析式を f32 で直接計算する
（Tape 駆動 closure は本 fixture の対象外。計画 §5）。

`lr` は 1 / 0.5 / 0.25 のみを使う（advisor 助言: 2 の冪の lr なら
`c1 * t * gtd` 等の f64 部分積が f32 へキャストされた際に純 f32 演算と
丸めが一致するため）。
"""

import hashlib
import json
import random

import torch


def make_problem(seed: int, n: int = 8, d: int = 4):
    rng = random.Random(seed)
    x = [[rng.uniform(-1.0, 1.0) for _ in range(d)] for _ in range(n)]
    true_w = [rng.uniform(-1.0, 1.0) for _ in range(d)]
    true_b = rng.uniform(-1.0, 1.0)
    y = []
    for row in x:
        v = sum(a * b for a, b in zip(row, true_w)) + true_b
        v += rng.uniform(-0.05, 0.05)
        y.append(v)
    init_w = [rng.uniform(-0.5, 0.5) for _ in range(d)]
    init_b = rng.uniform(-0.5, 0.5)
    return x, y, init_w, init_b


def run_case(case):
    x, y, init_w, init_b = make_problem(case["seed"])
    x_t = torch.tensor(x, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    w = torch.tensor(init_w, dtype=torch.float32, requires_grad=True)
    b = torch.tensor([init_b], dtype=torch.float32, requires_grad=True)

    n_calls = [0]

    def closure():
        n_calls[0] += 1
        opt.zero_grad()
        pred = x_t @ w + b
        loss = torch.mean((pred - y_t) ** 2)
        assert torch.isfinite(loss).all(), "closure produced non-finite loss"
        loss.backward()
        assert torch.isfinite(w.grad).all() and torch.isfinite(
            b.grad
        ).all(), "closure produced non-finite grad"
        return loss

    kwargs = dict(
        lr=case["lr"],
        max_iter=case["max_iter"],
        tolerance_grad=case.get("tolerance_grad", 1e-7),
        tolerance_change=case.get("tolerance_change", 1e-9),
        history_size=case.get("history_size", 100),
        line_search_fn=case.get("line_search_fn"),
    )
    opt = torch.optim.LBFGS([w, b], **kwargs)

    steps = []
    n_steps = case.get("n_steps", 1)
    for _ in range(n_steps):
        evals_before = n_calls[0]
        orig_loss = opt.step(closure)
        state = opt.state[opt.param_groups[0]["params"][0]]
        steps.append(
            {
                "w": w.detach().tolist(),
                "b": b.detach().tolist(),
                "func_evals": state.get("func_evals", 0),
                "n_iter": state.get("n_iter", 0),
                "orig_loss": orig_loss.item()
                if torch.is_tensor(orig_loss)
                else float(orig_loss),
                "closure_calls_this_step": n_calls[0] - evals_before,
            }
        )
    return {
        "x": x,
        "y": y,
        "init_w": init_w,
        "init_b": init_b,
        "lr": case["lr"],
        "max_iter": case["max_iter"],
        "tolerance_grad": kwargs["tolerance_grad"],
        "tolerance_change": kwargs["tolerance_change"],
        "history_size": kwargs["history_size"],
        "line_search_fn": kwargs["line_search_fn"],
        "steps": steps,
    }


CASES = {
    "fixed_step_default": dict(
        seed=20260926, lr=1.0, max_iter=20, n_steps=1, line_search_fn=None
    ),
    "strong_wolfe_default": dict(
        # advisor 助言（実装計画 §7）: 収束末期は loss の変化量が
        # tolerance_change（1e-9）付近まで縮小し、f32/f64 混在演算に
        # 由来する ULP 差が「打ち切り判定」の分岐そのものを反転させ
        # うる（実測: 実装計画中の分岐反転調査でイシュー #2197 実装時に
        # 確認）。`max_iter=10` はこの近接領域に到達する前（loss の
        # 変化量が 1e-9 から十分離れている段階）で打ち切ることで、この
        # 種の分岐反転を避ける。
        seed=20260927, lr=1.0, max_iter=10, n_steps=1, line_search_fn="strong_wolfe"
    ),
    "small_history": dict(
        seed=20260928,
        lr=1.0,
        max_iter=20,
        n_steps=1,
        history_size=2,
        line_search_fn="strong_wolfe",
    ),
    "multi_step_carry_fixed": dict(
        seed=20260929, lr=0.5, max_iter=4, n_steps=4, line_search_fn=None
    ),
    "multi_step_carry_wolfe": dict(
        seed=20260930, lr=0.5, max_iter=4, n_steps=4, line_search_fn="strong_wolfe"
    ),
    "tolerance_grad_early_return": dict(
        seed=20260931,
        lr=1.0,
        max_iter=20,
        n_steps=1,
        tolerance_grad=10.0,
        line_search_fn=None,
    ),
}


def main():
    torch.manual_seed(0)
    out = {"torch_version": torch.__version__, "cases": {}}
    for name, case in CASES.items():
        out["cases"][name] = run_case(case)

    with open("lbfgs_reference.json", "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
        f.write("\n")

    print("torch:", torch.__version__)
    for name, case_out in out["cases"].items():
        print(name, [s["func_evals"] for s in case_out["steps"]], [s["n_iter"] for s in case_out["steps"]])


if __name__ == "__main__":
    main()
