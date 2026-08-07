"""AdamW（torch.optim.AdamW）参照値生成スクリプト（イシュー #194）。

`crates/autodiff/tests/nn_optim_adamw.rs` の
`adamw_matches_pytorch_reference` が読む `adamw_reference.json` を
生成する。2 つの独立パラメータ（`param_a`: shape [2,3]、`param_b`:
shape [4]。層としての関係は持たず、AdamW が要素ごと・スロットごとに
独立更新することを利用した単純な確認用テンソル）に対し、固定の初期値と
10 step 分の固定勾配系列を与え、各 step 後のパラメータ値を記録する。

3 ケース（既定ハイパーパラメータ・非既定・weight_decay=0）を出す。
乱数は `random.Random(seed)` で再現可能に固定する（本スクリプト自体は
CI では実行されない。生成済み JSON をコミットして使う。README 参照）。
"""

import json
import random

import torch

SEED = 20260807  # 生成日をシードに使う（本スクリプト固有の値。再実行時も同一出力）


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
    "default": dict(lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01),
    "custom": dict(lr=0.1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1),
    "weight_decay_zero": dict(lr=0.01, betas=(0.8, 0.9), eps=1e-6, weight_decay=0.0),
}


def run_case(hp: dict) -> dict:
    a = torch.tensor(init_a, dtype=torch.float32).reshape(PARAM_A_SHAPE).clone()
    b = torch.tensor(init_b, dtype=torch.float32).reshape(PARAM_B_SHAPE).clone()
    a.requires_grad_(True)
    b.requires_grad_(True)
    opt = torch.optim.AdamW([a, b], **hp)

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
                "beta1": hp["betas"][0],
                "beta2": hp["betas"][1],
                "eps": hp["eps"],
                "weight_decay": hp["weight_decay"],
            },
            "steps": run_case(hp),
        }

    with open("adamw_reference.json", "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
