"""Adadelta（torch.optim.Adadelta）参照値生成スクリプト（イシュー #2171）。

`crates/autodiff/tests/nn_optim_adadelta.rs` の
`adadelta_matches_pytorch_reference` が読む `adadelta_reference.json` を
生成する。`rmsprop-pytorch-reference/gen_reference.py`（イシュー #1743）と
同型の構成: 2 つの独立パラメータ（`param_a`: shape [2,3]、`param_b`:
shape [4]。層としての関係は持たず、Adadelta が要素ごと・スロットごとに
独立更新することを利用した単純な確認用テンソル）に対し、固定の初期値と
10 step 分の固定勾配系列を与え、各 step 後のパラメータ値を記録する。

本スクリプト自体は CI では実行されない（生成済み JSON をコミットして
使う。README 参照）。
"""

import json
import random

import torch

SEED = 20260926  # 生成日をシードに使う（本スクリプト固有の値。再実行時も同一出力）


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
    "default": dict(lr=1.0, rho=0.9, eps=1e-6, weight_decay=0.0),
    "lr_rho": dict(lr=0.5, rho=0.8, eps=1e-6, weight_decay=0.0),
    "weight_decay": dict(lr=1.0, rho=0.9, eps=1e-6, weight_decay=0.1),
    "all": dict(lr=0.3, rho=0.7, eps=1e-5, weight_decay=0.05),
}


def run_case(hp: dict) -> dict:
    a = torch.tensor(init_a, dtype=torch.float32).reshape(PARAM_A_SHAPE).clone()
    b = torch.tensor(init_b, dtype=torch.float32).reshape(PARAM_B_SHAPE).clone()
    a.requires_grad_(True)
    b.requires_grad_(True)
    opt = torch.optim.Adadelta([a, b], **hp)

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
                "rho": hp["rho"],
                "eps": hp["eps"],
                "weight_decay": hp["weight_decay"],
            },
            "steps": run_case(hp),
        }

    with open("adadelta_reference.json", "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
