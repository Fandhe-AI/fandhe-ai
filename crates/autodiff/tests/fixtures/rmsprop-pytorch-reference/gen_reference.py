"""RMSprop（torch.optim.RMSprop）参照値生成スクリプト（イシュー #1743）。

`crates/autodiff/tests/nn_optim_rmsprop.rs` の
`rmsprop_matches_pytorch_reference` が読む `rmsprop_reference.json` を
生成する。`adamw-pytorch-reference/gen_reference.py`（イシュー #194）と
同型の構成: 2 つの独立パラメータ（`param_a`: shape [2,3]、`param_b`:
shape [4]。層としての関係は持たず、RMSprop が要素ごと・スロットごとに
独立更新することを利用した単純な確認用テンソル）に対し、固定の初期値と
10 step 分の固定勾配系列を与え、各 step 後のパラメータ値を記録する。

5 ケース（既定・momentum・centered・weight_decay・全部組み合わせ）を
出す。乱数は `random.Random(seed)` で再現可能に固定する（本スクリプト
自体は CI では実行されない。生成済み JSON をコミットして使う。
README 参照）。
"""

import json
import random

import torch

SEED = 20260914  # 生成日をシードに使う（本スクリプト固有の値。再実行時も同一出力）


def gen_values(seed: int, n: int) -> list[float]:
    rng = random.Random(seed)
    return [rng.uniform(-1.0, 1.0) for _ in range(n)]


PARAM_A_SHAPE = [2, 3]
PARAM_B_SHAPE = [4]
STEPS = 10

# 初期値・勾配系列はケース間で共通（ケース間の差はハイパーパラメータのみ
# にすることで、更新式そのものの一致検証に焦点を絞る）。
init_a = gen_values(SEED, 6)
init_b = gen_values(SEED + 1, 4)
# 各 step ごとに異なる勾配（学習ループの実勾配変化を模す最小限の系列）。
grads_a = [gen_values(SEED + 100 + s, 6) for s in range(STEPS)]
grads_b = [gen_values(SEED + 200 + s, 4) for s in range(STEPS)]

CASES = {
    "default": dict(lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0.0, momentum=0.0, centered=False),
    "momentum": dict(lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0.0, momentum=0.9, centered=False),
    "centered": dict(lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0.0, momentum=0.0, centered=True),
    "weight_decay": dict(lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0.1, momentum=0.0, centered=False),
    "all": dict(lr=0.05, alpha=0.9, eps=1e-6, weight_decay=0.01, momentum=0.5, centered=True),
}


def run_case(hp: dict) -> dict:
    a = torch.tensor(init_a, dtype=torch.float32).reshape(PARAM_A_SHAPE).clone()
    b = torch.tensor(init_b, dtype=torch.float32).reshape(PARAM_B_SHAPE).clone()
    a.requires_grad_(True)
    b.requires_grad_(True)
    opt = torch.optim.RMSprop([a, b], **hp)

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
                "alpha": hp["alpha"],
                "eps": hp["eps"],
                "weight_decay": hp["weight_decay"],
                "momentum": hp["momentum"],
                "centered": hp["centered"],
            },
            "steps": run_case(hp),
        }

    with open("rmsprop_reference.json", "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
