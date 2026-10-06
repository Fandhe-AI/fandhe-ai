#!/usr/bin/env python3
"""SELU・CELU・Softsign・Hardsigmoid・LogSigmoid の PyTorch 参照値を生成する。

イシュー #2649 の `tests/activation_scalar_ops_parity.rs` が参照する固定フィクスチャ
`activation_scalar_ops_reference.json` の生成スクリプト。CI は Python／PyTorch に
依存せずコミット済み JSON のみを読む（`README.md` 参照）。引数・環境変数・ネット
ワークは読まず、標準出力へ JSON を書くだけ。

- `cases`: 損失 `(out * g).sum()`（`g` は固定シードの上流勾配）。入力・`g`・出力・
  勾配・shape をすべて JSON に保存し、Rust 側で入力を再生成しない。dtype は float32。
  NaN／inf を含まないことを生成時に assert する。`torch.nn.*` の層出力が関数形と
  一致することも生成時に assert する。Hardsigmoid は 3 領域（`[-5,-3)`・`(-3,3)`・
  `(3,5)`）から層化サンプリングする。
- `edge_cases`: 上流勾配 1。入力は NaN／inf を表せるよう u32 ビットパターンで保存し、
  出力・勾配は要素ごとに `{"class": nan|pos_inf|neg_inf|finite, "value": ...}`。
  `±3` の 1 ulp 隣は float32 tensor の `torch.nextafter` で作る。
- `error_cases`: `F.celu` の不正 `alpha` について例外の有無・例外型・例外なしの
  場合の出力クラスを記録する。
"""

import json
import math
import platform
import struct
import sys

import torch
import torch.nn.functional as F

EXPECTED_TORCH_PREFIX = "2.14.0"
assert torch.__version__.startswith(EXPECTED_TORCH_PREFIX), torch.__version__

SEED = 2649
torch.manual_seed(SEED)
torch.set_num_threads(1)

CELU_ALPHAS = [1.0, 0.5, 2.5, -1.5]
SHAPES = [[6], [2, 3], [2, 2, 2]]


def numel(shape):
    n = 1
    for s in shape:
        n *= s
    return n


def sample(op, shape):
    n = numel(shape)
    if op == "hardsigmoid":
        # 3 領域から層化サンプリングして連結・shuffle する。
        k = n // 3
        parts = [
            torch.rand(k) * 2.0 - 5.0,          # [-5, -3)
            torch.rand(k) * 5.0 - 2.5,          # (-2.5, 2.5)
            torch.rand(n - 2 * k) * 2.0 + 3.0,  # [3, 5)
        ]
        x = torch.cat(parts)
        x = x[torch.randperm(n)]
    else:
        x = torch.rand(n) * 10.0 - 5.0
    return x.reshape(shape).float()


def fn_of(op, alpha=None):
    if op == "selu":
        return F.selu, torch.nn.SELU()
    if op == "celu":
        return (lambda t: F.celu(t, alpha=alpha)), torch.nn.CELU(alpha=alpha)
    if op == "softsign":
        return F.softsign, torch.nn.Softsign()
    if op == "hardsigmoid":
        return F.hardsigmoid, torch.nn.Hardsigmoid()
    if op == "log_sigmoid":
        return F.logsigmoid, torch.nn.LogSigmoid()
    raise ValueError(op)


def klass(v):
    if math.isnan(v):
        return {"class": "nan"}
    if math.isinf(v):
        return {"class": "pos_inf" if v > 0 else "neg_inf"}
    return {"class": "finite", "value": v}


def bits(v):
    return struct.unpack("<I", struct.pack("<f", v))[0]


def lst(t):
    return [float(v) for v in t.flatten().tolist()]


def finite(*ts):
    for t in ts:
        assert torch.isfinite(t).all(), "cases に NaN／inf を含めない"


def run(op, x, alpha=None):
    f, layer = fn_of(op, alpha)
    x = x.clone().requires_grad_(True)
    out = f(x)
    assert torch.equal(out.detach(), layer(x.detach())), f"{op}: 層と関数形が不一致"
    g = torch.rand_like(out) * 2.0 - 1.0
    (out * g).sum().backward()
    return x.detach(), g, out.detach(), x.grad


cases = []
for op in ["selu", "celu", "softsign", "hardsigmoid", "log_sigmoid"]:
    alphas = CELU_ALPHAS if op == "celu" else [None]
    for alpha in alphas:
        for shape in SHAPES:
            x = sample(op, shape)
            x, g, out, gx = run(op, x, alpha)
            finite(x, g, out, gx)
            tag = f"{op}" + (f"_a{alpha}" if alpha is not None else "")
            case = {
                "name": f"{tag}_{'x'.join(map(str, shape))}", "op": op, "shape": shape,
                "x": lst(x), "g": lst(g), "out": lst(out), "grad_x": lst(gx),
            }
            if alpha is not None:
                case["alpha"] = alpha
            cases.append(case)


def f32(v):
    return torch.tensor([v], dtype=torch.float32)


three = f32(3.0)
neg_three = f32(-3.0)
EDGE_X = [
    0.0, -0.0,
    3.0, -3.0,
    float(torch.nextafter(three, f32(0.0))[0]),       # 3 の 1 ulp 内側
    float(torch.nextafter(three, f32(10.0))[0]),      # 3 の 1 ulp 外側
    float(torch.nextafter(neg_three, f32(0.0))[0]),   # -3 の 1 ulp 内側
    float(torch.nextafter(neg_three, f32(-10.0))[0]), # -3 の 1 ulp 外側
    100.0, -100.0, 3e38, -3e38,
    float("inf"), float("-inf"), float("nan"),
]
edge_cases = []
for op, alpha in [("selu", None), ("celu", 1.0), ("celu", 2.5), ("softsign", None),
                  ("hardsigmoid", None), ("log_sigmoid", None)]:
    f, _ = fn_of(op, alpha)
    xx = torch.tensor(EDGE_X, dtype=torch.float32, requires_grad=True)
    o = f(xx)
    o.sum().backward()
    case = {
        "name": f"edge_{op}" + (f"_a{alpha}" if alpha is not None else ""), "op": op,
        "x_bits": [bits(float(v)) for v in xx.detach().tolist()],
        "out": [klass(float(v)) for v in o.detach().tolist()],
        "grad_x": [klass(float(v)) for v in xx.grad.tolist()],
    }
    if alpha is not None:
        case["alpha"] = alpha
    edge_cases.append(case)

error_cases = []
probe = torch.tensor([-1.0, 0.5], dtype=torch.float32)
for label, alpha in [("zero", 0.0), ("nan", float("nan")), ("pos_inf", float("inf")),
                     ("neg_inf", float("-inf"))]:
    rec = {"name": f"celu_alpha_{label}", "alpha_class": label}
    try:
        o = F.celu(probe, alpha=alpha)
        rec["raised"] = False
        rec["exception_type"] = None
        rec["out"] = [klass(float(v)) for v in o.tolist()]
    except Exception as e:  # noqa: BLE001 - 例外型そのものを実測する
        rec["raised"] = True
        rec["exception_type"] = type(e).__name__
        rec["out"] = []
    error_cases.append(rec)

doc = {
    "torch_version": torch.__version__,
    "python_version": platform.python_version(),
    "platform": platform.platform(),
    "seed": SEED,
    "cases": cases,
    "edge_cases": edge_cases,
    "error_cases": error_cases,
}
sys.stdout.write(json.dumps(doc, indent=1, allow_nan=False) + "\n")
