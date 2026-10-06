"""Adafactor（torch.optim.Adafactor）参照値生成スクリプト（イシュー #2656）。

`crates/autodiff/tests/nn_optim_adafactor.rs` の
`adafactor_matches_pytorch_reference`／
`adafactor_matches_pytorch_reference_edge_cases` が読む
`adafactor_reference.json` を生成する。`rprop-pytorch-reference/
gen_reference.py`（イシュー #2655）と同型の構成。本スクリプト自体は CI では
実行されない（生成済み JSON をコミットして使う。README 参照）。
"""

import json
import random
import struct

import torch

SEED = 20261006

# rank 2・rank 1・rank 3（バッチ付き因子分解）・rank 0 の 4 本。
SHAPES = [[2, 3], [4], [2, 3, 4], []]
STEPS = 10


def numel(shape):
    n = 1
    for s in shape:
        n *= s
    return n


def gen_values(seed: int, n: int, scale: float = 1.0) -> list[float]:
    rng = random.Random(seed)
    return [rng.uniform(-1.0, 1.0) * scale for _ in range(n)]


init = [gen_values(SEED + k, numel(sh)) for k, sh in enumerate(SHAPES)]
# grads[step][param] = flat list
grads = [
    [gen_values(SEED + 1000 + 100 * s + k, numel(sh)) for k, sh in enumerate(SHAPES)]
    for s in range(STEPS)
]

EPS1_DEFAULT = 1.1920928955078125e-07  # torch.finfo(torch.float32).eps

CASES = {
    "default": dict(lr=1e-2, beta2_decay=-0.8, eps1=None, eps2=1e-3, d=1.0, weight_decay=0.0),
    "relative_step": dict(lr=1.0, beta2_decay=-0.8, eps1=None, eps2=1e-3, d=1.0, weight_decay=0.0),
    "weight_decay": dict(lr=0.1, beta2_decay=-0.8, eps1=None, eps2=1e-3, d=1.0, weight_decay=0.3),
    "beta2_decay": dict(lr=1e-2, beta2_decay=-0.5, eps1=None, eps2=1e-3, d=1.0, weight_decay=0.0),
    "eps": dict(lr=1e-2, beta2_decay=-0.8, eps1=0.5, eps2=2.0, d=1.0, weight_decay=0.0),
    "clip_d": dict(lr=1e-2, beta2_decay=-0.8, eps1=None, eps2=1e-3, d=2.5, weight_decay=0.0),
    "lr_change": dict(
        lr=1e-2, beta2_decay=-0.8, eps1=None, eps2=1e-3, d=1.0, weight_decay=0.0,
        lr_change_after_step=5, lr_after=0.3,
    ),
}


def make_params():
    ps = []
    for k, sh in enumerate(SHAPES):
        p = torch.tensor(init[k], dtype=torch.float32).reshape(sh).clone()
        p.requires_grad_(True)
        ps.append(p)
    return ps


def make_opt(ps, hp):
    return torch.optim.Adafactor(
        ps,
        lr=hp["lr"],
        beta2_decay=hp["beta2_decay"],
        eps=(hp["eps1"], hp["eps2"]),
        d=hp["d"],
        weight_decay=hp["weight_decay"],
        foreach=False,
    )


def run_case(hp: dict, stats: dict) -> list:
    ps = make_params()
    opt = make_opt(ps, hp)
    eps1 = hp["eps1"] if hp["eps1"] is not None else EPS1_DEFAULT
    lr = hp["lr"]
    out = []
    for s in range(STEPS):
        if "lr_change_after_step" in hp and s == hp["lr_change_after_step"]:
            opt.param_groups[0]["lr"] = hp["lr_after"]
            lr = hp["lr_after"]
        pre = [p.detach().clone() for p in ps]
        gs = [
            torch.tensor(grads[s][k], dtype=torch.float32).reshape(sh)
            for k, sh in enumerate(SHAPES)
        ]
        for p, g in zip(ps, gs):
            p.grad = g
        opt.step()
        t = float(s + 1)
        w = t ** hp["beta2_decay"]
        stats["w_lt_half"] |= w < 0.5
        stats["w_ge_half"] |= w >= 0.5
        stats["rho_lr"] |= lr <= 1 / (t ** 0.5)
        stats["rho_inv"] |= 1 / (t ** 0.5) < lr
        for k, p in enumerate(ps):
            st = opt.state[p]
            rms = pre[k].norm(2).item() / (pre[k].numel() ** 0.5)
            stats["alpha_eps2"] |= hp["eps2"] > rms
            stats["alpha_rms"] |= hp["eps2"] <= rms
            g = gs[k]
            if g.dim() > 1:
                rv, cv = st["row_var"], st["col_var"]
                ve = rv @ cv
                den = rv.mean(dim=-2, keepdim=True)
                stats["row_mean_clamp"] |= bool((den < eps1).any())
                ve = ve / den.clamp(min=eps1)
            else:
                ve = st["variance"].clone()
            stats["var_clamp"] |= bool((ve < eps1 * eps1).any())
            u = ve.clamp(min=eps1 * eps1).rsqrt() * g
            denom = max(1.0, u.norm(2).item() / ((u.numel() ** 0.5) * hp["d"]))
            stats["denom_one"] |= denom == 1.0
            stats["denom_gt_one"] |= denom > 1.0
        out.append([p.detach().flatten().tolist() for p in ps])
    return out


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def run_edge() -> dict:
    nan = float("nan")
    inf = float("inf")
    shapes = [[6], [2, 3]]
    inits = [[0.5, -0.5, 0.25, 1.0, 0.0, -1.0], [0.5, -0.5, 0.25, 1.0, 0.1, -1.0]]
    # 列 0: NaN、1: +0、2: -0、3: +inf、4: 極小値、5: 通常
    grads_e = [
        [[nan, 0.0, -0.0, inf, 1e-30, 1.0], [0.3, 0.0, -0.0, 0.2, 1e-30, 1.0]],
        [[1.0, 1.0, 1.0, 1.0, 1e-30, -1.0], [nan, 1.0, 1.0, 1.0, 1e-30, -1.0]],
        [[1.0, 1.0, 1.0, -inf, 1e-30, 1.0], [0.3, 0.0, 0.0, 0.2, 0.0, 0.0]],
        [[1.0, -1.0, 0.5, 1.0, -1e-30, 1.0], [inf, 1.0, 0.5, 1.0, -1e-30, 1.0]],
    ]
    ps = [
        torch.tensor(i, dtype=torch.float32).reshape(sh).requires_grad_(True)
        for i, sh in zip(inits, shapes)
    ]
    opt = torch.optim.Adafactor(ps, lr=0.01, foreach=False)
    params = []
    for step_g in grads_e:
        for p, g, sh in zip(ps, step_g, shapes):
            p.grad = torch.tensor(g, dtype=torch.float32).reshape(sh)
        opt.step()
        params.append([p.detach().flatten().tolist() for p in ps])
    return {
        "hyperparams": dict(lr=0.01, beta2_decay=-0.8, eps1=EPS1_DEFAULT, eps2=1e-3, d=1.0, weight_decay=0.0),
        "shapes": shapes,
        "init_bits": [[f32_bits(x) for x in row] for row in inits],
        "grads_bits": [[[f32_bits(x) for x in g] for g in step_g] for step_g in grads_e],
        "params_bits": [[[f32_bits(x) for x in row] for row in step_p] for step_p in params],
    }


def main() -> None:
    stats = {
        k: False
        for k in (
            "w_lt_half", "w_ge_half", "rho_lr", "rho_inv", "alpha_eps2", "alpha_rms",
            "row_mean_clamp", "var_clamp", "denom_one", "denom_gt_one",
        )
    }
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
        hp_out = dict(hp)
        if hp_out["eps1"] is None:
            hp_out["eps1"] = EPS1_DEFAULT
        out["cases"][name] = {"hyperparams": hp_out, "steps": run_case(hp, stats)}
    missing = [k for k, v in stats.items() if not v]
    assert not missing, f"分岐が通っていない: {missing}"
    print("branch coverage:", stats)

    with open("adafactor_reference.json", "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
