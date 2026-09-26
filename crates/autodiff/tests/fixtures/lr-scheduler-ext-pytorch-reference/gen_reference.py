#!/usr/bin/env python3
"""イシュー #2176（親 #2131）の
`tests/nn_optim_lr_scheduler_ext.rs` が参照する PyTorch 実行値
フィクスチャの生成スクリプト。`radam-pytorch-reference/gen_reference.py`
（イシュー #2171 先例）と同じ方針: `torch.optim.SGD([p], lr=base_lr)`
に対応するスケジューラを組み、引数なしの `step()` を N 回呼んで
`param_groups[0]['lr']` を epoch 0..N の系列として記録する。

このスクリプトは venv の外（repo の依存グラフ）では実行しない
（`.claude/rules/deps-policy.md`。torch は fixture 生成専用の
一時 venv にのみ導入する）。CI は生成済み JSON のみを読む。
"""

import hashlib
import json
import sys

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import (
    CyclicLR,
    ExponentialLR,
    LambdaLR,
    LinearLR,
    MultiStepLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    SequentialLR,
)

N_EPOCHS = 30


def lr_sequence(base_lr, make_scheduler):
    """epoch 0（構築直後）から N_EPOCHS 回 `step()` した後までの
    `param_groups[0]['lr']` 系列を返す（長さ N_EPOCHS + 1）。"""
    p = torch.nn.Parameter(torch.zeros(1))
    opt = SGD([p], lr=base_lr)
    sched = make_scheduler(opt)
    seq = [opt.param_groups[0]["lr"]]
    for _ in range(N_EPOCHS):
        sched.step()
        seq.append(opt.param_groups[0]["lr"])
    return seq


def multistep_case(name, base_lr, milestones, gamma):
    seq = lr_sequence(base_lr, lambda opt: MultiStepLR(opt, milestones=milestones, gamma=gamma))
    return {
        "name": name,
        "kind": "multi_step",
        "base_lr": base_lr,
        "milestones": milestones,
        "gamma": gamma,
        "lr": seq,
    }


def cawr_case(name, base_lr, t_0, t_mult, eta_min):
    seq = lr_sequence(
        base_lr,
        lambda opt: CosineAnnealingWarmRestarts(opt, T_0=t_0, T_mult=t_mult, eta_min=eta_min),
    )
    return {
        "name": name,
        "kind": "cosine_annealing_warm_restarts",
        "base_lr": base_lr,
        "t_0": t_0,
        "t_mult": t_mult,
        "eta_min": eta_min,
        "lr": seq,
    }


def cyclic_case(name, base_lr, max_lr, step_size_up, step_size_down):
    seq = lr_sequence(
        base_lr,
        lambda opt: CyclicLR(
            opt,
            base_lr=base_lr,
            max_lr=max_lr,
            step_size_up=step_size_up,
            step_size_down=step_size_down,
            mode="triangular",
            cycle_momentum=False,
        ),
    )
    return {
        "name": name,
        "kind": "cyclic",
        "base_lr": base_lr,
        "max_lr": max_lr,
        "step_size_up": step_size_up,
        "step_size_down": step_size_down,
        "lr": seq,
    }


def lambda_case(name, base_lr, lambda_kind):
    if lambda_kind == "geometric":
        fn = lambda e: 0.95 ** e
    elif lambda_kind == "harmonic":
        fn = lambda e: 1.0 / (e + 1)
    else:
        raise ValueError(lambda_kind)
    seq = lr_sequence(base_lr, lambda opt: LambdaLR(opt, lr_lambda=fn))
    return {
        "name": name,
        "kind": "lambda",
        "base_lr": base_lr,
        "lambda_kind": lambda_kind,
        "lr": seq,
    }


def sequential_case_2stage(name, base_lr):
    # LinearLR(start_factor=0.1, total_iters=3) -> ExponentialLR(0.9)、
    # milestones=[3]。
    def make(opt):
        s1 = LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=3)
        s2 = ExponentialLR(opt, gamma=0.9)
        return SequentialLR(opt, schedulers=[s1, s2], milestones=[3])

    seq = lr_sequence(base_lr, make)
    return {
        "name": name,
        "kind": "sequential_2stage",
        "base_lr": base_lr,
        "milestones": [3],
        "lr": seq,
    }


def sequential_case_3stage(name, base_lr):
    # LinearLR(0.1, total_iters=3) -> CosineAnnealingLR(T_max=5,
    # eta_min=0.01*base_lr) -> MultiStepLR(milestones=[2,4], gamma=0.5)、
    # milestones=[3, 8]。
    def make(opt):
        s1 = LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=3)
        s2 = CosineAnnealingLR(opt, T_max=5, eta_min=0.01 * base_lr)
        s3 = MultiStepLR(opt, milestones=[2, 4], gamma=0.5)
        return SequentialLR(opt, schedulers=[s1, s2, s3], milestones=[3, 8])

    seq = lr_sequence(base_lr, make)
    return {
        "name": name,
        "kind": "sequential_3stage",
        "base_lr": base_lr,
        "milestones": [3, 8],
        "cosine_t_max": 5,
        "cosine_eta_min": 0.01 * base_lr,
        "multistep_milestones": [2, 4],
        "multistep_gamma": 0.5,
        "lr": seq,
    }


def main():
    cases = [
        multistep_case("multistep_unsorted_dup_gamma_half", 0.1, [3, 3, 7, 2], 0.5),
        multistep_case("multistep_sorted_gamma_tenth", 0.2, [5, 10, 15], 0.1),
        cawr_case("cawr_t0_4_tmult_1", 0.1, 4, 1, 0.0),
        cawr_case("cawr_t0_2_tmult_2_eta_min", 0.1, 2, 2, 0.01),
        cyclic_case("cyclic_up3_down_none", 0.01, 0.1, 3, None),
        cyclic_case("cyclic_up2_down5", 0.01, 0.1, 2, 5),
        lambda_case("lambda_geometric", 0.1, "geometric"),
        lambda_case("lambda_harmonic", 0.1, "harmonic"),
        sequential_case_2stage("sequential_2stage", 0.1),
        sequential_case_3stage("sequential_3stage", 0.2),
    ]

    for case in cases:
        for v in case["lr"]:
            assert v == v, f"{case['name']}: NaN が生成された"  # NaN 検出

    out = {
        "torch_version": torch.__version__,
        "n_epochs": N_EPOCHS,
        "cases": cases,
    }
    out_path = "lr_scheduler_ext_reference.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
        f.write("\n")

    sha = hashlib.sha256(open(out_path, "rb").read()).hexdigest()
    print(f"torch_version={torch.__version__}")
    print(f"{sha}  {out_path}")


if __name__ == "__main__":
    sys.exit(main())
