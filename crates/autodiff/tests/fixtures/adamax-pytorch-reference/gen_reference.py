"""Adamax（torch.optim.Adamax）参照値生成スクリプト（イシュー #2171）。

`crates/autodiff/tests/nn_optim_adamax.rs` の
`adamax_matches_pytorch_reference` が読む `adamax_reference.json` を
生成する。`rmsprop-pytorch-reference/gen_reference.py`（イシュー #1743）と
同型の構成。本スクリプト自体は CI では実行されない（生成済み JSON を
コミットして使う。README 参照）。
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
    "default": dict(lr=2e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0),
    "betas": dict(lr=2e-3, beta1=0.8, beta2=0.99, eps=1e-8, weight_decay=0.0),
    "weight_decay": dict(lr=2e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.1),
    "all": dict(lr=0.05, beta1=0.7, beta2=0.95, eps=1e-6, weight_decay=0.02),
}


def run_case(hp: dict) -> dict:
    a = torch.tensor(init_a, dtype=torch.float32).reshape(PARAM_A_SHAPE).clone()
    b = torch.tensor(init_b, dtype=torch.float32).reshape(PARAM_B_SHAPE).clone()
    a.requires_grad_(True)
    b.requires_grad_(True)
    opt = torch.optim.Adamax(
        [a, b],
        lr=hp["lr"],
        betas=(hp["beta1"], hp["beta2"]),
        eps=hp["eps"],
        weight_decay=hp["weight_decay"],
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
        out["cases"][name] = {
            "hyperparams": {
                "lr": hp["lr"],
                "beta1": hp["beta1"],
                "beta2": hp["beta2"],
                "eps": hp["eps"],
                "weight_decay": hp["weight_decay"],
            },
            "steps": run_case(hp),
        }

    with open("adamax_reference.json", "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
