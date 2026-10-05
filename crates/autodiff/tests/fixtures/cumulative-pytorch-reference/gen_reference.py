#!/usr/bin/env python3
"""cummax／cummin／logcumsumexp の PyTorch 参照値を生成する。

イシュー #2636（親 #2625）の `tests/cumulative_parity.rs` が参照する固定
フィクスチャ `cumulative_reference.json` の生成スクリプト。CI は Python／
PyTorch に依存せず、コミット済み JSON のみを読む（`README.md` 参照）。

f32 はすべて u32 ビットパターン配列で保存する（JSON は NaN／inf を運べない
ため）。入力・上流勾配 `g`・出力・索引・入力勾配をすべて保存し、Rust 側で
入力を再生成しない。損失は `(values * g).sum()`。dtype は float32。
"""

import json
import platform
import struct
import sys

import torch

EXPECTED_TORCH_PREFIX = "2.14.0"
assert torch.__version__.startswith(EXPECTED_TORCH_PREFIX), torch.__version__

SEED = 2636
NAN = float("nan")
INF = float("inf")


def to_bits(values):
    return [struct.unpack("<I", struct.pack("<f", float(v)))[0] for v in values]


def flat_bits(t):
    return to_bits(t.detach().contiguous().reshape(-1).tolist())


def rnd(shape, gen):
    return torch.randn(*shape, generator=gen, dtype=torch.float32)


def tied(shape, gen):
    # 0.5 刻みに丸めて連続・非連続のタイを作る。
    return (torch.randn(*shape, generator=gen, dtype=torch.float32) * 2).round() / 2


def numel(shape):
    n = 1
    for s in shape:
        n *= s
    return n


# 有限ケース: (name, shape, dim, kind)。kind は入力の作り方。
FINITE_CASES = [
    ("len1", [1], 0, "randn"),
    ("len6", [6], 0, "randn"),
    ("len9_ties", [9], 0, "tied"),
    ("len8_ties", [8], 0, "tied"),
    ("mono_inc", [6], 0, "inc"),
    ("mono_dec", [6], 0, "dec"),
    ("all_equal", [5], 0, "const"),
    ("signed_zeros", [6], 0, "zeros"),
    ("neg_values", [7], 0, "neg"),
    ("batched_dim1", [3, 5], 1, "randn"),
    ("batched_dim0", [4, 3], 0, "randn"),
    ("rank3_dim0", [3, 4, 2], 0, "tied"),
    ("rank3_dim1", [3, 4, 2], 1, "tied"),
    ("rank3_dim2", [3, 4, 2], 2, "tied"),
    ("rank3_dim1_rand", [2, 5, 3], 1, "randn"),
]

# logcumsumexp 用の追加（大振幅・広ダイナミックレンジ）。
LSE_EXTRA = [
    ("lse_large_pos", [6], 0, "big50"),
    ("lse_large_neg", [6], 0, "bigneg50"),
    ("lse_huge_500", [5], 0, "big500"),
    ("lse_wide_range", [8], 0, "wide"),
]


def make_input(shape, kind, gen):
    n = numel(shape)
    if kind == "randn":
        return rnd(shape, gen)
    if kind == "tied":
        return tied(shape, gen)
    if kind == "inc":
        return torch.arange(n, dtype=torch.float32).reshape(shape) * 0.5 - 1.0
    if kind == "dec":
        return -torch.arange(n, dtype=torch.float32).reshape(shape) * 0.5 + 1.0
    if kind == "const":
        return torch.full(shape, 1.5, dtype=torch.float32)
    if kind == "zeros":
        vals = [0.0, -0.0, 0.0, -0.0, -0.0, 0.0][:n]
        return torch.tensor(vals, dtype=torch.float32).reshape(shape)
    if kind == "neg":
        return -(rnd(shape, gen).abs() + 0.1)
    if kind == "big50":
        return rnd(shape, gen) * 50.0
    if kind == "bigneg50":
        return rnd(shape, gen) * 50.0 - 100.0
    if kind == "big500":
        return rnd(shape, gen) * 500.0
    if kind == "wide":
        base = [-80.0, 30.0, -5.0, 60.0, 0.0, -120.0, 45.0, 10.0]
        return torch.tensor(base[:n], dtype=torch.float32).reshape(shape)
    raise AssertionError(kind)


def run(op, x, dim, g_gen):
    x = x.clone().requires_grad_(True)
    if op == "logcumsumexp":
        y = torch.logcumsumexp(x, dim)
        idx = None
    else:
        fn = torch.cummax if op == "cummax" else torch.cummin
        y, idx = fn(x, dim)
    g = torch.randn(*y.shape, generator=g_gen, dtype=torch.float32)
    (y * g).sum().backward()
    return x, y, idx, g


def record(op, name, shape, dim, x0, gen):
    x, y, idx, g = run(op, x0, dim, gen)
    out = {
        "name": f"{op}_{name}",
        "op": op,
        "shape": shape,
        "dim": dim,
        "x_bits": flat_bits(x),
        "g_bits": flat_bits(g),
        "out_bits": flat_bits(y),
        "grad_bits": flat_bits(x.grad),
    }
    if idx is not None:
        out["index"] = idx.reshape(-1).tolist()
    return out


# 非有限ケース: (name, shape, dim, 値列)。NaN／±inf を含む lane。
NONFINITE = [
    ("nan_first", [5], 0, [NAN, 1.0, 2.0, 0.5, 3.0]),
    ("nan_mid", [6], 0, [1.0, 3.0, NAN, 2.0, 5.0, 4.0]),
    ("nan_multi", [6], 0, [1.0, NAN, 2.0, NAN, 0.0, 7.0]),
    ("nan_last", [4], 0, [1.0, 2.0, 3.0, NAN]),
    ("neginf_all", [4], 0, [-INF, -INF, -INF, -INF]),
    ("neginf_first", [5], 0, [-INF, 1.0, -INF, 2.0, 0.0]),
    ("neginf_mid", [5], 0, [1.0, -INF, 2.0, -INF, 0.5]),
    ("posinf_mid", [5], 0, [1.0, INF, 2.0, INF, 0.5]),
    ("posinf_first", [4], 0, [INF, 1.0, 2.0, INF]),
    ("posinf_neginf", [5], 0, [INF, -INF, 1.0, -INF, INF]),
    ("nan_and_inf", [5], 0, [INF, NAN, -INF, 1.0, 2.0]),
    ("batched_nan_dim1", [2, 4], 1, [1.0, NAN, 2.0, 3.0, 0.5, 1.5, INF, -INF]),
    ("batched_nan_dim0", [3, 2], 0, [1.0, -INF, NAN, 2.0, 0.5, INF]),
]

# torch が例外を出すか否かを実測して記録する境界ケース。(name, shape, dim)
ERROR_CASES = [
    ("empty_axis", [0], 0),
    ("dim_oob", [4], 1),
    ("rank0_dim0", [], 0),
    ("rank0_dim_neg1", [], -1),
    ("neg_dim", [4], -1),
]


def make_error(op, name, shape, dim):
    x = torch.zeros(*shape) if shape else torch.tensor(1.0)
    try:
        if op == "logcumsumexp":
            r = torch.logcumsumexp(x, dim)
        else:
            fn = torch.cummax if op == "cummax" else torch.cummin
            r = fn(x, dim)[0]
        return {"name": f"{op}_{name}", "op": op, "shape": shape, "dim": dim,
                "torch_raises": False, "message": "",
                "out_shape": list(r.shape)}
    except Exception as e:  # noqa: BLE001 - 例外型・文面を実測として記録する
        return {"name": f"{op}_{name}", "op": op, "shape": shape, "dim": dim,
                "torch_raises": True, "message": f"{type(e).__name__}: {e}",
                "out_shape": []}


def main():
    gen = torch.Generator().manual_seed(SEED)
    finite = []
    for op in ("cummax", "cummin", "logcumsumexp"):
        for name, shape, dim, kind in FINITE_CASES:
            finite.append(record(op, name, shape, dim, make_input(shape, kind, gen), gen))
    for name, shape, dim, kind in LSE_EXTRA:
        finite.append(record("logcumsumexp", name, shape, dim, make_input(shape, kind, gen), gen))
    nonfinite = []
    for op in ("cummax", "cummin", "logcumsumexp"):
        for name, shape, dim, vals in NONFINITE:
            x0 = torch.tensor(vals, dtype=torch.float32).reshape(shape)
            nonfinite.append(record(op, name, shape, dim, x0, gen))
    errors = []
    for op in ("cummax", "cummin", "logcumsumexp"):
        for name, shape, dim in ERROR_CASES:
            errors.append(make_error(op, name, shape, dim))
    doc = {
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "seed": SEED,
        "finite_cases": finite,
        "nonfinite_cases": nonfinite,
        "error_cases": errors,
    }
    json.dump(doc, sys.stdout, ensure_ascii=False, separators=(",", ":"))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
