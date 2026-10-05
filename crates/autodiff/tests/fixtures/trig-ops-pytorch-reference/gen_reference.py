#!/usr/bin/env python3
"""逆三角関数・双曲線関数 9 種の PyTorch 参照値（forward 出力と入力勾配）を生成する。

イシュー #2634 の `tests/trig_ops_parity.rs` が参照する固定フィクスチャ
`trig_ops_reference.json` の生成スクリプト。CI は Python／PyTorch に依存せず
コミット済み JSON のみを読む（`README.md` 参照）。

損失は `(out * g).sum()`（`g` は固定シードの上流勾配）。入力・`g`・出力・勾配・
shape をすべて JSON に保存し、Rust 側で入力を再生成しない。dtype は float32。
`atan2` は `torch.atan2(input, other)` の引数順で、Rust 側の `a = input = y`・
`b = other = x` と同じ。`cases` は NaN／inf を含まないことを生成時に assert する。
`edge_cases`（定義域の境界・外側・巨大値・符号付きゼロ）は NaN／inf を JSON で表せない
ため、要素ごとに `{"class": nan|pos_inf|neg_inf|finite, "value": ...}` で保存する
（勾配は上流勾配 1 で測る）。
"""

import json
import math
import platform
import sys

import torch

EXPECTED_TORCH_PREFIX = "2.14.0"
assert torch.__version__.startswith(EXPECTED_TORCH_PREFIX), torch.__version__

torch.manual_seed(2634)

UNARY = ["atan", "asin", "acos", "sinh", "cosh", "asinh", "acosh", "atanh"]


def sample(op, shape):
    n = 1
    for s in shape:
        n *= s
    if op in ("asin", "acos", "atanh"):
        x = torch.rand(n) * 1.8 - 0.9
    elif op == "acosh":
        x = torch.rand(n) * 4.0 + 1.1
    elif op in ("sinh", "cosh"):
        x = torch.rand(n) * 6.0 - 3.0
    else:
        x = torch.rand(n) * 8.0 - 4.0
    return x.reshape(shape).float()


def klass(v):
    if math.isnan(v):
        return {"class": "nan"}
    if math.isinf(v):
        return {"class": "pos_inf" if v > 0 else "neg_inf"}
    return {"class": "finite", "value": v}


def run_unary(op, x):
    x = x.clone().requires_grad_(True)
    out = getattr(torch, op)(x)
    g = torch.rand_like(out) * 2.0 - 1.0
    (out * g).sum().backward()
    return x.detach(), g, out.detach(), x.grad


def run_binary(a, b):
    a = a.clone().requires_grad_(True)
    b = b.clone().requires_grad_(True)
    out = torch.atan2(a, b)  # torch.atan2(input=y, other=x)
    g = torch.rand_like(out) * 2.0 - 1.0
    (out * g).sum().backward()
    return a.detach(), b.detach(), g, out.detach(), a.grad, b.grad


def lst(t):
    return [float(v) for v in t.flatten().tolist()]


def finite(*ts):
    for t in ts:
        assert torch.isfinite(t).all(), "cases に NaN／inf を含めない"


cases = []
for op in UNARY:
    for shape in ([6], [2, 3], [2, 2, 2]):
        x = sample(op, shape)
        x, g, out, gx = run_unary(op, x)
        finite(x, g, out, gx)
        cases.append({
            "name": f"{op}_{'x'.join(map(str, shape))}", "op": op, "shape": shape,
            "x": lst(x), "g": lst(g), "out": lst(out), "grad_x": lst(gx),
        })


def far(shape):
    n = 1
    for s in shape:
        n *= s
    mag = torch.rand(n) * 2.5 + 0.3
    sign = torch.where(torch.rand(n) < 0.5, -1.0, 1.0)
    return (mag * sign).reshape(shape).float()


# atan2: 同形状（4 象限を含む）と broadcast 2 種。原点から離す。
for name, sa, sb in [("atan2_same_6", [6], [6]), ("atan2_same_2x3", [2, 3], [2, 3]),
                     ("atan2_bcast_2x3_3", [2, 3], [3]), ("atan2_bcast_2x1_1x3", [2, 1], [1, 3])]:
    a, b = far(sa), far(sb)
    a, b, g, out, ga, gb = run_binary(a, b)
    finite(a, b, g, out, ga, gb)
    cases.append({
        "name": name, "op": "atan2", "shape_a": sa, "shape_b": sb, "out_shape": list(out.shape),
        "a": lst(a), "b": lst(b), "g": lst(g), "out": lst(out),
        "grad_a": lst(ga), "grad_b": lst(gb),
    })

# edge_cases: 定義域の境界・外側・巨大値・符号付きゼロ。
EDGE_INPUTS = {
    "atan": [0.0, -0.0, 3e38, -3e38],
    "asin": [1.0, -1.0, 1.5, -1.5, 0.0],
    "acos": [1.0, -1.0, 1.5, -1.5, 0.0],
    "sinh": [0.0, -0.0, 100.0, -100.0],
    "cosh": [0.0, -0.0, 100.0, -100.0],
    "asinh": [0.0, -0.0, 3e38, -3e38],
    "acosh": [1.0, 0.5, 0.0, -0.5, -1.0, -1.5, 3e38],
    "atanh": [1.0, -1.0, 1.5, -1.5, 0.0, -0.0],
}
edge_cases = []
for op, xs in EDGE_INPUTS.items():
    xx = torch.tensor(xs, dtype=torch.float32, requires_grad=True)
    o = getattr(torch, op)(xx)
    o.sum().backward()
    edge_cases.append({
        "name": f"edge_{op}", "op": op, "x": [float(v) for v in xs],
        "out": [klass(float(v)) for v in o.detach().tolist()],
        "grad_x": [klass(float(v)) for v in xx.grad.tolist()],
    })

ZP = [(1.0, 1.0), (1.0, -1.0), (-1.0, -1.0), (-1.0, 1.0), (0.0, 0.0), (0.0, -0.0),
      (-0.0, 0.0), (-0.0, -0.0), (1e-30, 1e-30), (1e20, 1e20), (0.0, 1.0), (1.0, 0.0)]
a = torch.tensor([p[0] for p in ZP], dtype=torch.float32, requires_grad=True)
b = torch.tensor([p[1] for p in ZP], dtype=torch.float32, requires_grad=True)
o = torch.atan2(a, b)
o.sum().backward()
edge_cases.append({
    "name": "edge_atan2", "op": "atan2",
    "a": [float(p[0]) for p in ZP], "b": [float(p[1]) for p in ZP],
    "out": [klass(float(v)) for v in o.detach().tolist()],
    "grad_a": [klass(float(v)) for v in a.grad.tolist()],
    "grad_b": [klass(float(v)) for v in b.grad.tolist()],
})

doc = {
    "torch_version": torch.__version__,
    "python_version": platform.python_version(),
    "platform": platform.platform(),
    "seed": 2634,
    "cases": cases,
    "edge_cases": edge_cases,
}
sys.stdout.write(json.dumps(doc, indent=1, allow_nan=False) + "\n")
