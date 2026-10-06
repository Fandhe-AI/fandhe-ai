#!/usr/bin/env python3
"""PolynomialLR・ChainedScheduler の PyTorch 参照値を生成する。

イシュー #2659（親 #2657）の `crates/autodiff/tests/nn_optim_lr_scheduler_poly_chained.rs` が参照する
固定フィクスチャ `lr_scheduler_poly_chained_reference.json` の生成スクリプト。CI は Python／PyTorch に
依存せず、コミット済み JSON のみを読む（`README.md` 参照）。実行はこのディレクトリをカレントにして行う。

- 各ケースは `SGD([p], lr=base_lr)` にスケジューラを組み、`step()` を 30 回呼んで epoch 0〜30 の
  `param_groups[0]["lr"]`（Python float = f64）を記録する（#2176 の `lr_sequence` と同型）。
- 入力値（base_lr／gamma／start_factor／power）は f32 で厳密に表せる値のみを使う
  （torch は f64・Rust は f32 引数のため、丸め差が 30 step の累乗で増幅されるのを避ける）。
- `non_multiplicative_probe`: 加算項を持つ `CosineAnnealingLR` をチェーンに入れた場合の PyTorch 値。
  Rust の積の閉形式が一致しない（意図的な制限）ことを固定するための記録で、通常ケースには混ぜない。
"""

import json
import math
import platform

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import (
    ChainedScheduler,
    CosineAnnealingLR,
    ExponentialLR,
    LinearLR,
    MultiStepLR,
    PolynomialLR,
    StepLR,
)

EXPECTED_TORCH_PREFIX = "2.14.0"
assert torch.__version__.startswith(EXPECTED_TORCH_PREFIX), torch.__version__

N_EPOCHS = 30


def new_opt(base_lr):
    p = torch.nn.Parameter(torch.zeros(1))
    return SGD([p], lr=base_lr)


def run(opt, sched):
    lrs = [float(opt.param_groups[0]["lr"])]
    for _ in range(N_EPOCHS):
        opt.step()
        sched.step()
        lrs.append(float(opt.param_groups[0]["lr"]))
    assert all(math.isfinite(v) for v in lrs)
    return lrs


def poly_case(name, base_lr, total_iters, power):
    opt = new_opt(base_lr)
    sched = PolynomialLR(opt, total_iters=total_iters, power=power)
    return {
        "name": name,
        "base_lr": base_lr,
        "total_iters": total_iters,
        "power": power,
        "lrs": run(opt, sched),
    }


def chain_case(name, base_lr, members, build):
    """`build(opt)` が member scheduler 列を返す。`members` は Rust 側が再構築する仕様。"""
    opt = new_opt(base_lr)
    sched = ChainedScheduler(build(opt), optimizer=opt)
    return {"name": name, "base_lr": base_lr, "members": members, "lrs": run(opt, sched)}


def main():
    poly = [
        poly_case("p1_t5", 0.5, 5, 1.0),
        poly_case("p2_t10", 0.5, 10, 2.0),
        poly_case("p05_t7", 0.5, 7, 0.5),
        poly_case("p1_t1", 0.25, 1, 1.0),
        poly_case("p3_t40", 0.125, 40, 3.0),
        poly_case("p0_t5", 0.5, 5, 0.0),
    ]
    step = {"kind": "step", "step_size": 3, "gamma": 0.5}
    exp = {"kind": "exp", "gamma": 0.875}
    chain = [
        chain_case(
            "step_exp",
            0.5,
            [step, exp],
            lambda o: [StepLR(o, step_size=3, gamma=0.5), ExponentialLR(o, gamma=0.875)],
        ),
        chain_case(
            "exp_step_reversed",
            0.5,
            [exp, step],
            lambda o: [ExponentialLR(o, gamma=0.875), StepLR(o, step_size=3, gamma=0.5)],
        ),
        chain_case(
            "linear_multistep",
            0.25,
            [
                {"kind": "linear", "warmup_steps": 5, "start_factor": 0.25},
                {"kind": "multistep", "milestones": [4, 10, 20], "gamma": 0.5},
            ],
            lambda o: [
                LinearLR(o, start_factor=0.25, end_factor=1.0, total_iters=5),
                MultiStepLR(o, milestones=[4, 10, 20], gamma=0.5),
            ],
        ),
        chain_case(
            "linear_poly_exp",
            0.5,
            [
                {"kind": "linear", "warmup_steps": 4, "start_factor": 0.25},
                {"kind": "poly", "total_iters": 20, "power": 2.0},
                {"kind": "exp", "gamma": 0.9375},
            ],
            lambda o: [
                LinearLR(o, start_factor=0.25, end_factor=1.0, total_iters=4),
                PolynomialLR(o, total_iters=20, power=2.0),
                ExponentialLR(o, gamma=0.9375),
            ],
        ),
    ]

    opt = new_opt(0.5)
    probe_sched = ChainedScheduler(
        [CosineAnnealingLR(opt, T_max=10, eta_min=0.0), ExponentialLR(opt, gamma=0.875)],
        optimizer=opt,
    )
    probe = {
        "name": "cosine_exp_chain",
        "base_lr": 0.5,
        "t_max": 10,
        "eta_min": 0.0,
        "gamma": 0.875,
        "lrs": run(opt, probe_sched),
    }

    out = {
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "n_epochs": N_EPOCHS,
        "poly_cases": poly,
        "chain_cases": chain,
        "non_multiplicative_probe": probe,
    }
    with open("lr_scheduler_poly_chained_reference.json", "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()
