"""RAdam（torch.optim.RAdam）参照値生成スクリプト（イシュー #2171）。

`crates/autodiff/tests/nn_optim_radam.rs` の
`radam_matches_pytorch_reference` が読む `radam_reference.json` を
生成する。`rmsprop-pytorch-reference/gen_reference.py`（イシュー #1743）と
同型の構成。本スクリプト自体は CI では実行されない（生成済み JSON を
コミットして使う。README 参照）。

RAdam は `rho_t`（近似 SMA 長）が 5.0 を超えるかどうかで補正式の分岐が
切り替わる（`torch/optim/radam.py::_single_tensor_radam`）。実装計画
§3.4 の要求どおり、各ケース・各 step の `rho_t` を計算して JSON にも
出し、(a) 10 step 以内に両分岐を通ること、(b) どの step の `rho_t` も
5.0 から `1e-3` 以上離れていること（Rust 側が `beta2 as f64` で計算する
際の丸めで分岐が反転しないため）を assert する。
"""

import json
import random

import torch

SEED = 20260926


def gen_values(seed: int, n: int) -> list[float]:
    rng = random.Random(seed)
    return [rng.uniform(-1.0, 1.0) for _ in range(n)]


PARAM_A_SHAPE = [2, 3]
PARAM_B_SHAPE = [4]
STEPS = 10

init_a = gen_values(SEED, 6)
init_b = gen_values(SEED + 1, 4)
grads_a = [gen_values(SEED + 100 + s, 6) for s in range(STEPS)]
grads_b = [gen_values(SEED + 200 + s, 4) for s in range(STEPS)]

CASES = {
    "default": dict(lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0, decoupled_weight_decay=False),
    "beta2_small": dict(lr=1e-3, beta1=0.9, beta2=0.9, eps=1e-8, weight_decay=0.0, decoupled_weight_decay=False),
    "weight_decay": dict(lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.1, decoupled_weight_decay=False),
    "decoupled_weight_decay": dict(lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.1, decoupled_weight_decay=True),
    "all": dict(lr=0.01, beta1=0.8, beta2=0.95, eps=1e-6, weight_decay=0.02, decoupled_weight_decay=True),
}


def rho_t_series(beta2: float, steps: int) -> list[float]:
    rho_inf = 2.0 / (1.0 - beta2) - 1.0
    out = []
    for step in range(1, steps + 1):
        rho_t = rho_inf - 2.0 * step * (beta2**step) / (1.0 - beta2**step)
        out.append(rho_t)
    return out


def run_case(hp: dict) -> dict:
    a = torch.tensor(init_a, dtype=torch.float32).reshape(PARAM_A_SHAPE).clone()
    b = torch.tensor(init_b, dtype=torch.float32).reshape(PARAM_B_SHAPE).clone()
    a.requires_grad_(True)
    b.requires_grad_(True)
    opt = torch.optim.RAdam(
        [a, b],
        lr=hp["lr"],
        betas=(hp["beta1"], hp["beta2"]),
        eps=hp["eps"],
        weight_decay=hp["weight_decay"],
        decoupled_weight_decay=hp["decoupled_weight_decay"],
    )

    steps_out = []
    for s in range(STEPS):
        opt.zero_grad()
        a.grad = torch.tensor(grads_a[s], dtype=torch.float32).reshape(PARAM_A_SHAPE)
        b.grad = torch.tensor(grads_b[s], dtype=torch.float32).reshape(PARAM_B_SHAPE)
        opt.step()
        steps_out.append(
            {
                "param_a": a.detach().flatten().tolist(),
                "param_b": b.detach().flatten().tolist(),
            }
        )
    return steps_out


def main() -> None:
    out = {
        "torch_version": torch.__version__,
        "param_a_shape": PARAM_A_SHAPE,
        "param_b_shape": PARAM_B_SHAPE,
        "steps": STEPS,
        "init_a": init_a,
        "init_b": init_b,
        "grads_a": grads_a,
        "grads_b": grads_b,
        "cases": {},
    }
    for name, hp in CASES.items():
        rho_series = rho_t_series(hp["beta2"], STEPS)
        # (a) 両分岐（<=5 と >5）を 10 step 以内に通ること。
        has_le5 = any(r <= 5.0 for r in rho_series)
        has_gt5 = any(r > 5.0 for r in rho_series)
        assert has_le5 and has_gt5, (
            f"case {name}: rho_t series does not cross the rectification "
            f"threshold within {STEPS} steps: {rho_series}"
        )
        # (b) どの step の rho_t も 5.0 から 1e-3 以上離れていること。
        for step, r in enumerate(rho_series, start=1):
            assert abs(r - 5.0) > 1e-3, (
                f"case {name} step {step}: rho_t={r} is too close to the "
                "rectification boundary (5.0) for a stable f32-vs-f64 "
                "branch decision"
            )
        for r in rho_series:
            assert r == r, f"case {name}: NaN rho_t encountered"

        out["cases"][name] = {
            "hyperparams": {
                "lr": hp["lr"],
                "beta1": hp["beta1"],
                "beta2": hp["beta2"],
                "eps": hp["eps"],
                "weight_decay": hp["weight_decay"],
                "decoupled_weight_decay": hp["decoupled_weight_decay"],
            },
            "rho_t": rho_series,
            "steps": run_case(hp),
        }

    with open("radam_reference.json", "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
