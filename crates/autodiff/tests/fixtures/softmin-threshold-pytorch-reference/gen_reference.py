#!/usr/bin/env python3
"""Softmin・Tanhshrink・Threshold・RReLU の PyTorch 参照値（forward 出力と入力勾配）を生成する。

イシュー #2650 の `tests/softmin_threshold_parity.rs` が参照する固定フィクスチャ
`softmin_threshold_reference.json` の生成スクリプト。CI は Python／PyTorch に依存せず
コミット済み JSON のみを読む（`README.md` 参照）。

損失は `(out * g).sum()`（`g` は固定シードの上流勾配）。入力・`g`・出力・勾配・shape・
パラメータをすべて JSON に保存し、Rust 側で入力を再生成しない。dtype は float32。
`cases` は NaN／inf を含まず、`x == threshold`・`x == 0` ちょうどの要素も含まない
（境界は `edge_cases` で扱う）ことを生成時に assert する。RReLU 学習時は乱数列が
PyTorch と一致しないため、`F.rrelu(training=True)` を 1 回だけ計算して同じ標本の noise
（`y.sum()` の勾配）と勾配を取り、`y == x * noise` を assert したうえで noise を保存する。
Rust 側はこの noise を `rrelu_with_noise` へ渡して突合する。

`edge_cases` は NaN／inf を JSON で表せないため、要素ごとに
`{"class": nan|pos_inf|neg_inf|finite, "value": ...}` で保存する（入力・出力・勾配とも。
勾配は上流勾配 1 で測る）。
"""

import json
import math
import platform
import sys

import torch
import torch.nn.functional as F

EXPECTED_TORCH_PREFIX = "2.14.0"
assert torch.__version__.startswith(EXPECTED_TORCH_PREFIX), torch.__version__

torch.manual_seed(2650)

NAN = float("nan")
INF = float("inf")


def klass(v):
    if math.isnan(v):
        return {"class": "nan"}
    if math.isinf(v):
        return {"class": "pos_inf" if v > 0 else "neg_inf"}
    return {"class": "finite", "value": v}


def lst(t):
    return [float(v) for v in t.flatten().tolist()]


def klst(t):
    return [klass(float(v)) for v in t.flatten().tolist()]


def numel(shape):
    n = 1
    for s in shape:
        n *= s
    return n


def sample(shape, lo=-3.0, hi=3.0):
    """境界（0 ちょうど）を避けた一様乱数。"""
    x = (torch.rand(numel(shape)) * (hi - lo) + lo).reshape(shape).float()
    return x


def finite(*ts):
    for t in ts:
        assert torch.isfinite(t).all(), "cases に NaN／inf を含めない"


def run(fn, x):
    """`fn(x)` の出力と、`(out * g).sum()` の入力勾配を返す。"""
    x = x.clone().requires_grad_(True)
    out = fn(x)
    g = torch.rand_like(out) * 2.0 - 1.0
    (out * g).sum().backward()
    return x.detach(), g, out.detach(), x.grad


cases = []


def add_case(name, op, shape, params, fn, x, extra=None):
    x, g, out, gx = run(fn, x)
    finite(x, g, out, gx)
    c = {
        "name": name, "op": op, "shape": shape, "params": params,
        "x": lst(x), "g": lst(g), "out": lst(out), "grad_x": lst(gx),
    }
    if extra:
        c.update(extra)
    cases.append(c)


SHAPES = ([5], [2, 3], [2, 2, 2])

# softmin: 各軸
for shape in ([5], [2, 3], [2, 3, 4]):
    for dim in range(len(shape)):
        x = sample(shape)
        add_case(f"softmin_{'x'.join(map(str, shape))}_dim{dim}", "softmin", shape,
                 {"dim": dim}, lambda t, d=dim: F.softmin(t, dim=d), x)

# tanhshrink
for shape in ([6], [2, 3], [2, 2, 2]):
    x = sample(shape, -4.0, 4.0)
    add_case(f"tanhshrink_{'x'.join(map(str, shape))}", "tanhshrink", shape, {},
             lambda t: F.tanhshrink(t), x)

# threshold: 形状 3 種 × パラメータ 3 組（正負の threshold・value）
for shape in SHAPES:
    for (th, val) in [(0.1, 20.0), (-0.5, -1.5), (1.0, 0.0)]:
        x = sample(shape)
        assert not (x == th).any()
        add_case(f"threshold_{'x'.join(map(str, shape))}_t{th}_v{val}", "threshold", shape,
                 {"threshold": th, "value": val},
                 lambda t, a=th, b=val: F.threshold(t, a, b), x)

# rrelu 推論: 既定値・任意範囲・lower == upper
for shape in SHAPES:
    for (lo, hi) in [(1.0 / 8.0, 1.0 / 3.0), (0.05, 0.6), (0.25, 0.25)]:
        x = sample(shape)
        assert not (x == 0).any()
        add_case(f"rrelu_eval_{'x'.join(map(str, shape))}_{lo:.4f}_{hi:.4f}", "rrelu_eval", shape,
                 {"lower": lo, "upper": hi},
                 lambda t, a=lo, b=hi: F.rrelu(t, a, b, training=False), x)


def rrelu_train_case(name, shape, lo, hi, x):
    """同じ標本の noise と勾配を 1 回の forward から取る（公開 API のみ）。"""
    x = x.clone().requires_grad_(True)
    y = F.rrelu(x, lo, hi, training=True)
    (noise,) = torch.autograd.grad(y.sum(), x, retain_graph=True)
    g = torch.rand_like(y) * 2.0 - 1.0
    (grad,) = torch.autograd.grad((y * g).sum(), x)
    assert torch.equal(y.detach(), x.detach() * noise), "y == x * noise"
    xd, yd = x.detach(), y.detach()
    finite(xd, yd, g, grad, noise)
    return {
        "name": name, "op": "rrelu_train", "shape": shape,
        "params": {"lower": lo, "upper": hi},
        "x": lst(xd), "g": lst(g), "out": lst(yd), "grad_x": lst(grad), "noise": lst(noise),
    }


for shape in SHAPES:
    for (lo, hi) in [(1.0 / 8.0, 1.0 / 3.0), (0.05, 0.6)]:
        x = sample(shape)
        assert not (x == 0).any()
        cases.append(rrelu_train_case(
            f"rrelu_train_{'x'.join(map(str, shape))}_{lo:.4f}_{hi:.4f}", shape, lo, hi, x))

# edge_cases
edge_cases = []


def add_edge(name, op, params, fn, xs, extra_noise=False):
    x = torch.tensor(xs, dtype=torch.float32, requires_grad=True)
    if extra_noise:
        y = fn(x)
        (noise,) = torch.autograd.grad(y.sum(), x, retain_graph=True)
        y.sum().backward()
    else:
        y = fn(x)
        y.sum().backward()
        noise = None
    e = {
        "name": name, "op": op, "params": params, "x": [klass(v) for v in xs],
        "out": klst(y.detach()), "grad_x": klst(x.grad),
    }
    if noise is not None:
        e["noise"] = lst(noise)
    edge_cases.append(e)


add_edge("threshold_nan_inf", "threshold", {"threshold": 0.5, "value": 7.0},
         lambda t: F.threshold(t, 0.5, 7.0), [NAN, INF, -INF, 0.5, 0.4999, 0.5001, 1.0, -1.0])
add_edge("rrelu_eval_boundary", "rrelu_eval", {"lower": 0.1, "upper": 0.3},
         lambda t: F.rrelu(t, 0.1, 0.3, training=False),
         [0.0, -0.0, NAN, INF, -INF, 1.0, -2.0])
add_edge("rrelu_train_boundary", "rrelu_train", {"lower": 0.1, "upper": 0.3},
         lambda t: F.rrelu(t, 0.1, 0.3, training=True),
         [0.0, -0.0, NAN, INF, -INF, 1.0, -2.0], extra_noise=True)
add_edge("softmin_inf_row", "softmin", {"dim": 0},
         lambda t: F.softmin(t, dim=0), [INF, 1.0, 2.0])
add_edge("softmin_neg_inf_row", "softmin", {"dim": 0},
         lambda t: F.softmin(t, dim=0), [-INF, 1.0, 2.0])
add_edge("softmin_large", "softmin", {"dim": 0},
         lambda t: F.softmin(t, dim=0), [1e30, -1e30, 0.0])
add_edge("softmin_equal", "softmin", {"dim": 0},
         lambda t: F.softmin(t, dim=0), [3.0, 3.0, 3.0, 3.0])
add_edge("tanhshrink_edges", "tanhshrink", {},
         lambda t: F.tanhshrink(t), [0.0, 1e-4, -1e-4, 20.0, -20.0, INF, -INF, NAN])

fixture = {
    "torch_version": torch.__version__,
    "python_version": platform.python_version(),
    "seed": 2650,
    "cases": cases,
    "edge_cases": edge_cases,
}
json.dump(fixture, sys.stdout, indent=1, allow_nan=False)
print()
