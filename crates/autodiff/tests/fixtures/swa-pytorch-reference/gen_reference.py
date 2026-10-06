#!/usr/bin/env python3
"""SWA（AveragedModel・SWALR）の PyTorch 参照値を生成する。

イシュー #2658（親 #2657）の `crates/autodiff/tests/nn_swa.rs` が参照する固定フィクスチャ
`swa_reference.json` の生成スクリプト。CI は Python／PyTorch に依存せず、コミット済み JSON のみを
読む（`README.md` 参照）。実行はこのディレクトリをカレントにして行う。

- SWALR 系列: `SGD([p], lr=base_lr)` に `SWALR` を組み、`step()` を 30 回呼んで epoch 0〜30 の
  `param_groups[0]["lr"]`（Python float = f64）を記録する。学習率の入力（base_lr／swa_lr）は
  f32 で厳密に表せる値へ丸めてから torch へ渡す（Rust 側が f32 で受けるため）。
- 連結ケース: `CosineAnnealingLR` を主スケジューラにし、PyTorch ドキュメントの定番ループ
  （`epoch > swa_start` なら `swa_scheduler.step()`、それ以外は `scheduler.step()`）で系列を作る。
- AveragedModel: 各反復で `update_parameters` へ渡したパラメータ（入力）と、更新後の平均値・
  `n_averaged`（出力）の両方を記録する（Rust 側は torch の RNG を再現できないため）。
  f32 値はすべて u32 ビットパターンで保存する。
"""

import json
import math
import platform
import struct

import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.swa_utils import SWALR, AveragedModel

EXPECTED_TORCH_PREFIX = "2.14.0"
assert torch.__version__.startswith(EXPECTED_TORCH_PREFIX), torch.__version__

torch.set_num_threads(1)
SEED = 2658
GEN = torch.Generator().manual_seed(SEED)
N_EPOCHS = 30


def f32(x):
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


def to_bits(values):
    out = []
    for v in values:
        v = float(v)
        assert math.isfinite(v), v
        out.append(struct.unpack("<I", struct.pack("<f", v))[0])
    return out


def flat_bits(t):
    return to_bits(t.detach().contiguous().reshape(-1).tolist())


def randn(*shape):
    return torch.randn(*shape, generator=GEN, dtype=torch.float32)


def lr_of(opt):
    return float(opt.param_groups[0]["lr"])


def swalr_case(name, base_lr, swa_lr, anneal_epochs, strategy):
    base_lr, swa_lr = f32(base_lr), f32(swa_lr)
    p = nn.Parameter(torch.zeros(1))
    opt = SGD([p], lr=base_lr)
    sched = SWALR(opt, swa_lr=swa_lr, anneal_epochs=anneal_epochs, anneal_strategy=strategy)
    lrs = [lr_of(opt)]
    for _ in range(N_EPOCHS):
        opt.step()
        sched.step()
        lrs.append(lr_of(opt))
    return {
        "name": name,
        "base_lr": base_lr,
        "swa_lr": swa_lr,
        "anneal_epochs": anneal_epochs,
        "strategy": strategy,
        "lrs": lrs,
    }


def chain_case():
    base_lr, swa_lr, t_max, swa_start, anneal = f32(0.1), f32(0.02), 20, 9, 5
    p = nn.Parameter(torch.zeros(1))
    opt = SGD([p], lr=base_lr)
    sched = CosineAnnealingLR(opt, T_max=t_max)
    swa_sched = SWALR(opt, swa_lr=swa_lr, anneal_epochs=anneal, anneal_strategy="cos")
    lrs = []
    main_steps = 0
    for epoch in range(N_EPOCHS + 1):
        lrs.append(lr_of(opt))
        opt.step()
        if epoch > swa_start:
            swa_sched.step()
        else:
            sched.step()
            main_steps += 1
    # 主スケジューラが実際に進んだ回数 = SWA 段へ切り替わる step（epoch 番号）。
    assert main_steps == swa_start + 1
    return {
        "name": "cos_main_then_swalr",
        "base_lr": base_lr,
        "swa_lr": swa_lr,
        "t_max": t_max,
        "swa_start": swa_start,
        "switch_step": main_steps,
        "anneal_epochs": anneal,
        "strategy": "cos",
        "lrs": lrs,
    }


class Holder(nn.Module):
    def __init__(self, shapes):
        super().__init__()
        self.ps = nn.ParameterList([nn.Parameter(torch.zeros(*s)) for s in shapes])


def avg_case(name, shapes, init, updates):
    """`init`／`updates[i]` は shapes に対応するテンソル列。"""
    holder = Holder(shapes)
    with torch.no_grad():
        for p, v in zip(holder.ps, init):
            p.copy_(v)
    avg = AveragedModel(holder)
    steps = []
    for params in updates:
        with torch.no_grad():
            for p, v in zip(holder.ps, params):
                p.copy_(v)
        avg.update_parameters(holder)
        steps.append(
            {
                "params": [flat_bits(p) for p in holder.ps],
                "averaged": [flat_bits(p) for p in avg.module.ps],
                "n_averaged": int(avg.n_averaged.item()),
            }
        )
    return {
        "name": name,
        "shapes": [list(s) for s in shapes],
        "init": [flat_bits(v) for v in init],
        "steps": steps,
    }


def linear_sgd_case():
    model = nn.Linear(3, 2)
    with torch.no_grad():
        model.weight.copy_(randn(2, 3))
        model.bias.copy_(randn(2))
    x = randn(4, 3)
    opt = SGD(model.parameters(), lr=0.1)
    avg = AveragedModel(model)
    init = [flat_bits(model.weight), flat_bits(model.bias)]
    steps = []
    for _ in range(12):
        opt.zero_grad()
        (model(x) ** 2).mean().backward()
        opt.step()
        avg.update_parameters(model)
        steps.append(
            {
                "params": [flat_bits(model.weight), flat_bits(model.bias)],
                "averaged": [flat_bits(avg.module.weight), flat_bits(avg.module.bias)],
                "n_averaged": int(avg.n_averaged.item()),
            }
        )
    return {"name": "linear_sgd_12", "shapes": [[2, 3], [2]], "init": init, "steps": steps}


def main():
    swalr = [
        swalr_case("cos_e5", 0.1, 0.01, 5, "cos"),
        swalr_case("cos_e10", 0.1, 0.01, 10, "cos"),
        swalr_case("linear_e5", 0.1, 0.01, 5, "linear"),
        swalr_case("linear_e10", 0.1, 0.01, 10, "linear"),
        swalr_case("cos_swa_gt_base", 0.05, 0.2, 7, "cos"),
        swalr_case("linear_swa_gt_base", 0.05, 0.2, 7, "linear"),
        swalr_case("cos_e1", 0.1, 0.01, 1, "cos"),
        swalr_case("linear_e1", 0.1, 0.01, 1, "linear"),
        swalr_case("cos_e0", 0.1, 0.01, 0, "cos"),
        swalr_case("linear_e0", 0.1, 0.01, 0, "linear"),
    ]
    chain = chain_case()

    cases = [linear_sgd_case()]

    shapes = [(4,), (2, 3)]
    walk, cur = [], [randn(*s) for s in shapes]
    init = [c.clone() for c in cur]
    for _ in range(200):
        cur = [c + 0.3 * randn(*c.shape) for c in cur]
        walk.append([c.clone() for c in cur])
    cases.append(avg_case("random_walk_200", shapes, init, walk))

    cases.append(avg_case("single_update", [(5,)], [randn(5)], [[randn(5)]]))

    const = randn(6)
    cases.append(
        avg_case(
            "constant_updates",
            [(6,)],
            [randn(6)],
            [[const.clone()] for _ in range(6)],
        )
    )

    def mixed():
        mag = 10 ** (torch.rand(8, generator=GEN) * 12 - 6)
        sign = torch.where(torch.rand(8, generator=GEN) > 0.5, 1.0, -1.0)
        return [(mag * sign).float()]

    cases.append(avg_case("mixed_scale", [(8,)], mixed(), [mixed() for _ in range(20)]))

    out = {
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "seed": SEED,
        "n_epochs": N_EPOCHS,
        "swalr_cases": swalr,
        "swalr_chain_case": chain,
        "avg_cases": cases,
    }
    for c in swalr + [chain]:
        assert all(math.isfinite(v) for v in c["lrs"])
    with open("swa_reference.json", "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()
