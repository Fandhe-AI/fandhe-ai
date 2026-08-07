#!/usr/bin/env python3
"""SGD（momentum）PyTorch 一致 fixture 生成スクリプト（#193）。

`crates/autodiff/tests/optim_sgd.rs` の `PYTORCH_FIXTURES` に埋め込む
期待値を、`torch.optim.SGD` の更新則（PyTorch ドキュメント「Algorithm」
節）を f64 で再実装した参照実装から生成する。実装環境に PyTorch が
無いため（実装計画 §3.5 の調査結果）、既定では純 Python 実装のみを使う。
`--with-torch` を渡すと `torch.optim.SGD` 実系列との突合も行う
（PyTorch 導入済み環境での再検証・将来の再生成に備える）。

使い方:
    python3 gen_sgd_expected.py                # 純 Python 参照実装で出力
    python3 gen_sgd_expected.py --with-torch   # torch 実系列とも突合

出力は Rust の `[(f32, ...); N]` テストデータに転記しやすい JSON。
"""

from __future__ import annotations

import argparse
import json


def sgd_reference(
    p0: list[float],
    grads: list[list[float]],
    lr: float,
    momentum: float = 0.0,
    dampening: float = 0.0,
    weight_decay: float = 0.0,
    nesterov: bool = False,
) -> list[list[float]]:
    """`torch.optim.SGD` の擬似コードを f64 相当（Python float）で再実装する。

    `p0` は初期パラメータ（1 テンソル分、平坦化済み）、`grads[t]` は
    step t の勾配。各 step 後のパラメータ列を返す（`len(grads)` 個）。
    """
    p = list(p0)
    b: list[float] | None = None
    out = []
    for grad in grads:
        new_p = []
        new_b = [] if momentum != 0.0 else None
        for j in range(len(p)):
            g = grad[j]
            if weight_decay != 0.0:
                g = g + weight_decay * p[j]
            if momentum != 0.0:
                if b is None:
                    bj = g
                else:
                    bj = momentum * b[j] + (1.0 - dampening) * g
                if nesterov:
                    g = g + momentum * bj
                else:
                    g = bj
                new_b.append(bj)
            new_p.append(p[j] - lr * g)
        p = new_p
        if new_b is not None:
            b = new_b
        out.append(list(p))
    return out


CONFIGS = {
    "vanilla": dict(lr=0.1, momentum=0.0, dampening=0.0, weight_decay=0.0, nesterov=False),
    "momentum_0_9": dict(lr=0.1, momentum=0.9, dampening=0.0, weight_decay=0.0, nesterov=False),
    "momentum_dampening": dict(
        lr=0.1, momentum=0.9, dampening=0.5, weight_decay=0.0, nesterov=False
    ),
    "momentum_weight_decay": dict(
        lr=0.1, momentum=0.9, dampening=0.0, weight_decay=0.01, nesterov=False
    ),
    "nesterov": dict(lr=0.1, momentum=0.9, dampening=0.0, weight_decay=0.0, nesterov=True),
}

# 3 要素・5 step。乱数ではなく固定小規模数列（再現性の議論を単純にする）。
P0 = [1.0, -0.5, 2.0]
GRADS = [
    [0.5, -0.2, 0.1],
    [0.25, -0.1, 0.05],
    [0.125, 0.3, -0.4],
    [-0.05, 0.2, 0.15],
    [0.4, -0.3, 0.1],
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-torch", action="store_true")
    args = parser.parse_args()

    results = {}
    for name, cfg in CONFIGS.items():
        results[name] = sgd_reference(P0, GRADS, **cfg)

    if args.with_torch:
        import torch

        for name, cfg in CONFIGS.items():
            p = torch.tensor(P0, dtype=torch.float64, requires_grad=False)
            opt = torch.optim.SGD(
                [p.requires_grad_()],
                lr=cfg["lr"],
                momentum=cfg["momentum"],
                dampening=cfg["dampening"],
                weight_decay=cfg["weight_decay"],
                nesterov=cfg["nesterov"],
            )
            torch_out = []
            for g in GRADS:
                opt.zero_grad()
                p.grad = torch.tensor(g, dtype=torch.float64)
                opt.step()
                torch_out.append(p.detach().tolist())
            ref = results[name]
            for t, (a, b) in enumerate(zip(ref, torch_out)):
                for j, (x, y) in enumerate(zip(a, b)):
                    assert abs(x - y) < 1e-9, (
                        f"{name} step={t} idx={j}: python={x} torch={y}"
                    )
            print(f"[ok] {name}: python 参照実装は torch.optim.SGD と一致")

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
