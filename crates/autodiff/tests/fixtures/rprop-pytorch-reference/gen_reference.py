"""Rprop（torch.optim.Rprop）参照値生成スクリプト（イシュー #2655）。

`crates/autodiff/tests/nn_optim_rprop.rs` の
`rprop_matches_pytorch_reference`／`rprop_matches_pytorch_reference_edge_cases`
が読む `rprop_reference.json` を生成する。`adamax-pytorch-reference/
gen_reference.py`（イシュー #2171）と同型の構成。本スクリプト自体は CI では
実行されない（生成済み JSON をコミットして使う。README 参照）。
"""

import json
import random
import struct

import torch

SEED = 20260927


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

# 3 分岐（同符号・符号反転・ゼロ）と clamp 境界（上限・下限）を確実に通すため、
# 一部要素の勾配系列を固定する。
for s in range(STEPS):
    grads_a[s][0] = 0.5 + 0.01 * s  # 常に正: 同符号が続き step_size が単調増加（上限到達用）
    grads_b[s][0] = 0.3 if s % 2 == 0 else -0.3  # 毎 step 符号反転（下限到達用）
grads_a[3][1] = 0.0  # 厳密ゼロ
grads_a[4][1] = 0.0
grads_b[5][2] = 0.0
grads_b[6][2] = 0.0

CASES = {
    "default": dict(lr=1e-2, eta_minus=0.5, eta_plus=1.2, step_size_min=1e-6, step_size_max=50.0),
    "etas": dict(lr=1e-2, eta_minus=0.3, eta_plus=1.5, step_size_min=1e-6, step_size_max=50.0),
    "tight_bounds": dict(lr=1e-2, eta_minus=0.5, eta_plus=1.2, step_size_min=5e-3, step_size_max=2e-2),
    "lr_change": dict(
        lr=1e-2,
        eta_minus=0.5,
        eta_plus=1.2,
        step_size_min=1e-6,
        step_size_max=50.0,
        lr_change_after_step=5,
        lr_after=0.5,
    ),
}


def run_case(hp: dict) -> list:
    a = torch.tensor(init_a, dtype=torch.float32).reshape(PARAM_A_SHAPE).clone()
    b = torch.tensor(init_b, dtype=torch.float32).reshape(PARAM_B_SHAPE).clone()
    a.requires_grad_(True)
    b.requires_grad_(True)
    opt = torch.optim.Rprop(
        [a, b],
        lr=hp["lr"],
        etas=(hp["eta_minus"], hp["eta_plus"]),
        step_sizes=(hp["step_size_min"], hp["step_size_max"]),
        foreach=False,
    )

    steps_out = []
    hit_min = hit_max = False
    branches = set()
    for s in range(STEPS):
        if "lr_change_after_step" in hp and s == hp["lr_change_after_step"]:
            opt.param_groups[0]["lr"] = hp["lr_after"]
        opt.zero_grad()
        ga = torch.tensor(grads_a[s], dtype=torch.float32).reshape(PARAM_A_SHAPE)
        gb = torch.tensor(grads_b[s], dtype=torch.float32).reshape(PARAM_B_SHAPE)
        # 分岐観測（step 前の prev との積の符号）
        for p, g in ((a, ga), (b, gb)):
            st = opt.state.get(p)
            if st:
                for v in (g * st["prev"]).sign().flatten().tolist():
                    branches.add(v)
        a.grad = ga
        b.grad = gb
        opt.step()
        for p in (a, b):
            ss = opt.state[p]["step_size"].flatten().tolist()
            lo32 = torch.tensor(hp["step_size_min"], dtype=torch.float32).item()
            hi32 = torch.tensor(hp["step_size_max"], dtype=torch.float32).item()
            hit_min = hit_min or any(x == lo32 for x in ss)
            hit_max = hit_max or any(x == hi32 for x in ss)
        steps_out.append(
            {
                "param_a": a.detach().flatten().tolist(),
                "param_b": b.detach().flatten().tolist(),
            }
        )
    assert branches >= {-1.0, 0.0, 1.0}, f"3 分岐が揃わない: {branches}"
    if hp["step_size_min"] > 1e-3:
        assert hit_min and hit_max, f"clamp 境界に張り付かない: min={hit_min} max={hit_max}"
    return steps_out


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def run_edge() -> dict:
    nan = float("nan")
    inf = float("inf")
    # 列 0: NaN→以降の積が NaN→sign 0 経由、1: +0、2: -0、3: +inf→-inf、
    # 4: 積がアンダーフロー（1e-30 * 1e-30）、5: 通常
    grads = [
        [nan, 0.0, -0.0, inf, 1e-30, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1e-30, -1.0],
        [nan, 1.0, 1.0, -inf, 1.0, 1.0],
        [1.0, -1.0, 0.5, 1.0, -1e-30, 1.0],
    ]
    p = torch.zeros(6, requires_grad=True)
    opt = torch.optim.Rprop([p], lr=0.01, foreach=False)
    params = []
    for g in grads:
        p.grad = torch.tensor(g, dtype=torch.float32)
        opt.step()
        params.append(p.detach().flatten().tolist())
    return {
        "hyperparams": dict(lr=0.01, eta_minus=0.5, eta_plus=1.2, step_size_min=1e-6, step_size_max=50.0),
        "init_bits": [f32_bits(0.0)] * 6,
        "grads_bits": [[f32_bits(x) for x in g] for g in grads],
        "params_bits": [[f32_bits(x) for x in row] for row in params],
    }


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
        "edge": run_edge(),
    }
    for name, hp in CASES.items():
        out["cases"][name] = {
            "hyperparams": hp,
            "steps": run_case(hp),
        }

    with open("rprop_reference.json", "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
