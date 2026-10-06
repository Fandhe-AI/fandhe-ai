"""Lion 参照値生成スクリプト（イシュー #2656）。

**`torch.optim` 2.14.0 に Lion は存在しない**。本スクリプトは公式参照実装
（google/automl `lion/lion_pytorch.py`。Apache-2.0。
https://github.com/google/automl/blob/master/lion/lion_pytorch.py）の更新則
を、実 PyTorch 2.14.0+cpu のテンソル演算で自前記述して実行した値を
`lion_reference.json` へ書き出す。サードパーティ `lion-pytorch` は使わない。

`crates/autodiff/tests/nn_optim_lion.rs` の
`lion_matches_pytorch_reference`／`lion_matches_pytorch_reference_edge_cases`
が JSON を読む。本スクリプト自体は CI では実行されない（生成済み JSON を
コミットして使う。README 参照）。
"""

import json
import random
import struct

import torch

SEED = 20261007
SHAPES = [[2, 3], [4]]
STEPS = 10


def numel(shape):
    n = 1
    for s in shape:
        n *= s
    return n


def gen_values(seed: int, n: int) -> list[float]:
    rng = random.Random(seed)
    return [rng.uniform(-1.0, 1.0) for _ in range(n)]


init = [gen_values(SEED + k, numel(sh)) for k, sh in enumerate(SHAPES)]
grads = [
    [gen_values(SEED + 1000 + 100 * s + k, numel(sh)) for k, sh in enumerate(SHAPES)]
    for s in range(STEPS)
]
# 符号の 3 値を確実に通すため、一部要素の系列を固定する。
for s in range(STEPS):
    grads[s][0][0] = 0.4 + 0.01 * s  # 常に正
    grads[s][0][1] = 0.6 if s % 2 == 0 else -0.6  # 毎 step 符号反転
grads[0][0][2] = 0.0  # 厳密 0.0（exp_avg も 0 のため update が厳密 0）
grads[1][0][2] = -0.0
grads[0][1][3] = 0.0
grads[1][1][3] = 0.0

CASES = {
    "base": dict(lr=1e-2, beta1=0.9, beta2=0.99, weight_decay=0.0),
    "lr": dict(lr=0.1, beta1=0.9, beta2=0.99, weight_decay=0.0),
    "betas": dict(lr=1e-2, beta1=0.6, beta2=0.8, weight_decay=0.0),
    "weight_decay": dict(lr=0.05, beta1=0.9, beta2=0.99, weight_decay=0.5),
    "lr_change": dict(
        lr=1e-2, beta1=0.9, beta2=0.99, weight_decay=0.1,
        lr_change_after_step=5, lr_after=0.2,
    ),
}


def lion_step(p, g, exp_avg, lr, beta1, beta2, wd):
    """公式 lion_pytorch.py の 1 step。演算順をそのまま写す。"""
    p.mul_(1 - lr * wd)
    update = exp_avg.mul(beta1).add(g, alpha=1 - beta1)
    p.add_(update.sign_(), alpha=-lr)
    exp_avg.mul_(beta2).add_(g, alpha=1 - beta2)


def run_case(hp: dict, seen: set) -> list:
    ps = [
        torch.tensor(init[k], dtype=torch.float32).reshape(sh).clone()
        for k, sh in enumerate(SHAPES)
    ]
    ea = [torch.zeros_like(p) for p in ps]
    lr = hp["lr"]
    out = []
    for s in range(STEPS):
        if "lr_change_after_step" in hp and s == hp["lr_change_after_step"]:
            lr = hp["lr_after"]
        for k, sh in enumerate(SHAPES):
            g = torch.tensor(grads[s][k], dtype=torch.float32).reshape(sh)
            # 生成時 assert: sign が丸め差で反転しない余裕（厳密 0 または |update| >= 1e-3）。
            upd = ea[k].mul(hp["beta1"]).add(g, alpha=1 - hp["beta1"])
            for x in upd.flatten().tolist():
                assert x == 0.0 or abs(x) >= 1e-3, f"update が 0 近傍: {x} (step={s})"
                seen.add(0.0 if x == 0.0 else (1.0 if x > 0 else -1.0))
            lion_step(ps[k], g, ea[k], lr, hp["beta1"], hp["beta2"], hp["weight_decay"])
        out.append([p.flatten().tolist() for p in ps])
    return out


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def run_edge() -> dict:
    nan = float("nan")
    inf = float("inf")
    # 列 0: NaN、1: +0、2: -0、3: +inf、4: 極小値、5: 通常
    grads_e = [
        [nan, 0.0, -0.0, inf, 1e-30, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1e-30, -1.0],
        [1.0, -1.0, 0.0, -inf, 1.0, 1.0],
        [1.0, -1.0, 0.5, 1.0, -1e-30, 1.0],
    ]
    init_e = [0.5, -0.5, 0.25, 1.0, 0.0, -1.0]
    hp = dict(lr=0.01, beta1=0.9, beta2=0.99, weight_decay=0.1)
    p = torch.tensor(init_e, dtype=torch.float32)
    e = torch.zeros_like(p)
    params = []
    for g in grads_e:
        lion_step(p, torch.tensor(g, dtype=torch.float32), e, hp["lr"], hp["beta1"], hp["beta2"], hp["weight_decay"])
        params.append(p.flatten().tolist())
    return {
        "hyperparams": hp,
        "init_bits": [f32_bits(x) for x in init_e],
        "grads_bits": [[f32_bits(x) for x in g] for g in grads_e],
        "params_bits": [[f32_bits(x) for x in row] for row in params],
    }


def main() -> None:
    seen: set = set()
    out = {
        "torch_version": torch.__version__,
        "shapes": SHAPES,
        "steps": STEPS,
        "init": init,
        "grads": grads,
        "cases": {},
        "edge": run_edge(),
    }
    for name, hp in CASES.items():
        out["cases"][name] = {"hyperparams": hp, "steps": run_case(hp, seen)}
    assert seen == {-1.0, 0.0, 1.0}, f"sign の 3 値が揃わない: {seen}"
    with open("lion_reference.json", "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
