"""ASGD（torch.optim.ASGD）参照値生成スクリプト（イシュー #2655）。

`crates/autodiff/tests/nn_optim_asgd.rs` の `asgd_matches_pytorch_reference`
が読む `asgd_reference.json` を生成する。各 step の `param_*` に加え、
平均化パラメータ `ax`（`opt.state[p]["ax"]`）も記録する。
`adamax-pytorch-reference/gen_reference.py`（イシュー #2171）と同型の構成。
本スクリプト自体は CI では実行されない（README 参照）。
"""

import json
import random

import torch

SEED = 20260928


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
    "default": dict(lr=1e-2, lambd=1e-4, alpha=0.75, t0=1e6, weight_decay=0.0),
    "small_t0": dict(lr=1e-2, lambd=1e-4, alpha=0.75, t0=2.0, weight_decay=0.0),
    "weight_decay": dict(lr=1e-2, lambd=1e-4, alpha=0.75, t0=1e6, weight_decay=0.1),
    "all": dict(lr=0.05, lambd=0.02, alpha=0.6, t0=3.0, weight_decay=0.05),
    "lr_change": dict(
        lr=1e-2,
        lambd=1e-2,
        alpha=0.75,
        t0=1e6,
        weight_decay=0.0,
        lr_change_after_step=5,
        lr_after=0.05,
    ),
}


def run_case(hp: dict) -> list:
    a = torch.tensor(init_a, dtype=torch.float32).reshape(PARAM_A_SHAPE).clone()
    b = torch.tensor(init_b, dtype=torch.float32).reshape(PARAM_B_SHAPE).clone()
    a.requires_grad_(True)
    b.requires_grad_(True)
    opt = torch.optim.ASGD(
        [a, b],
        lr=hp["lr"],
        lambd=hp["lambd"],
        alpha=hp["alpha"],
        t0=hp["t0"],
        weight_decay=hp["weight_decay"],
        foreach=False,
    )

    steps_out = []
    mu_one = mu_other = False
    for s in range(STEPS):
        if "lr_change_after_step" in hp and s == hp["lr_change_after_step"]:
            opt.param_groups[0]["lr"] = hp["lr_after"]
        opt.zero_grad()
        a.grad = torch.tensor(grads_a[s], dtype=torch.float32).reshape(PARAM_A_SHAPE)
        b.grad = torch.tensor(grads_b[s], dtype=torch.float32).reshape(PARAM_B_SHAPE)
        # この step の更新で使う mu（前 step の終わりに計算済みの値）
        if s > 0:
            mu = opt.state[a]["mu"].item()
            mu_one = mu_one or mu == 1
            mu_other = mu_other or mu != 1
        else:
            mu_one = True
        opt.step()
        steps_out.append(
            {
                "param_a": a.detach().flatten().tolist(),
                "param_b": b.detach().flatten().tolist(),
                "ax_a": opt.state[a]["ax"].flatten().tolist(),
                "ax_b": opt.state[b]["ax"].flatten().tolist(),
            }
        )
    if hp["t0"] < 10:
        assert mu_one and mu_other, f"mu の両分岐が通らない: one={mu_one} other={mu_other}"
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
        out["cases"][name] = {
            "hyperparams": hp,
            "steps": run_case(hp),
        }

    with open("asgd_reference.json", "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
