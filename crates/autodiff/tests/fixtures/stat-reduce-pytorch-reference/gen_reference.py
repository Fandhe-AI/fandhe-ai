#!/usr/bin/env python3
"""median／kthvalue／quantile／nansum／nanmean の PyTorch 参照値を生成する。

イシュー #2637（親 #2625）の `tests/stat_reduce_parity.rs` が参照する固定
フィクスチャ `stat_reduce_reference.json` の生成スクリプト。CI は Python／
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

SEED = 2637
NAN = float("nan")
INF = float("inf")
INTERPS = ["linear", "lower", "higher", "midpoint", "nearest"]


def f32(v):
    return struct.unpack("<f", struct.pack("<f", float(v)))[0]


def to_bits(values):
    return [struct.unpack("<I", struct.pack("<f", float(v)))[0] for v in values]


def flat_bits(t):
    return to_bits(t.detach().contiguous().reshape(-1).tolist())


def numel(shape):
    n = 1
    for s in shape:
        n *= s
    return n


def rnd(shape, gen):
    return torch.randn(*shape, generator=gen, dtype=torch.float32)


def tied(shape, gen):
    # 0.5 刻みに丸めてタイを作る。
    return (torch.randn(*shape, generator=gen, dtype=torch.float32) * 2).round() / 2


def apply(spec, x):
    """spec の演算を実行し (値, 索引 or None) を返す。"""
    op = spec["op"]
    dim = spec["dim"]
    if op == "median_all":
        return torch.median(x), None
    if op == "median_dim":
        r = torch.median(x, dim)
        return r.values, r.indices
    if op == "kthvalue":
        r = torch.kthvalue(x, spec["k"], dim)
        return r.values, r.indices
    if op == "quantile":
        q = torch.tensor(spec["q"], dtype=torch.float32)
        if dim is None:
            return torch.quantile(x, q, interpolation=spec["interp"]), None
        return torch.quantile(x, q, dim=dim, interpolation=spec["interp"]), None
    if op == "nansum":
        return (torch.nansum(x) if dim is None else torch.nansum(x, dim)), None
    if op == "nanmean":
        return (torch.nanmean(x) if dim is None else torch.nanmean(x, dim)), None
    raise AssertionError(op)


def spec_name(spec):
    s = spec["op"]
    if spec["op"] == "kthvalue":
        s += f"_k{spec['k']}"
    if spec["op"] == "quantile":
        s += f"_{spec['interp']}_q{spec['q']}"
    d = spec["dim"]
    return s + ("_dNone" if d is None else f"_d{d}")


def record(case, spec, x0, gen, g_override=None):
    x = x0.clone().requires_grad_(True)
    y, idx = apply(spec, x)
    if g_override is not None:
        g = g_override
    else:
        g = torch.randn(list(y.shape), generator=gen, dtype=torch.float32)
    (y * g).sum().backward()
    out = {
        "name": f"{case}__{spec_name(spec)}",
        "op": spec["op"],
        "shape": list(x0.shape),
        "dim": spec["dim"],
        "k": spec.get("k"),
        "q_bits": None if "q" not in spec else to_bits([spec["q"]])[0],
        "interp": spec.get("interp"),
        "x_bits": flat_bits(x),
        "g_bits": flat_bits(g),
        "out_shape": list(y.shape),
        "out_bits": flat_bits(y),
        "grad_bits": flat_bits(x.grad),
    }
    if idx is not None:
        out["index"] = idx.reshape(-1).tolist()
    return out


def all_specs(shape, dim, n, qs, interps=INTERPS):
    """1 つの (shape, dim) に対する演算指定の一覧。dim が None なら全要素版。"""
    specs = []
    if dim is None:
        specs.append({"op": "median_all", "dim": None})
    else:
        specs.append({"op": "median_dim", "dim": dim})
        for k in sorted({1, (n + 1) // 2, n}):
            specs.append({"op": "kthvalue", "dim": dim, "k": k})
    for interp in interps:
        for q in qs:
            specs.append({"op": "quantile", "dim": dim, "q": f32(q), "interp": interp})
    specs.append({"op": "nansum", "dim": dim})
    specs.append({"op": "nanmean", "dim": dim})
    return specs


def lane_len(shape, dim):
    return numel(shape) if dim is None else shape[dim]


QS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]

# 有限（タイなし）ケース: (name, shape, dim)。
FINITE_SHAPES = [
    ("len1", [1], 0),
    ("len5_odd", [5], 0),
    ("len6_even", [6], 0),
    ("all_len7", [7], None),
    ("batched_dim1", [3, 5], 1),
    ("batched_dim0", [4, 3], 0),
    ("rank2_all", [3, 4], None),
    ("rank3_dim0", [3, 4, 2], 0),
    ("rank3_dim1", [3, 4, 2], 1),
    ("rank3_dim2", [3, 4, 2], 2),
    ("rank3_all", [2, 3, 4], None),
]

# f32 rank 判別ケース: rank = f32(q) * (n-1) が整数ちょうどになる／.5 になる組。
RANK_CASES = [
    ("n11", 11, [0.1, 0.3, 0.7, 0.9]),
    ("n21", 21, [0.1, 0.15, 0.35, 0.65, 0.85]),
    ("n5_half", 5, [0.125, 0.375, 0.625, 0.875]),
]

# 非有限ケース: (name, shape, dim, 値列)。
NAN_CASES = [
    ("nan_first", [5], 0, [NAN, 1.0, 2.0, 0.5, 3.0]),
    ("nan_mid", [6], 0, [1.0, 3.0, NAN, 2.0, 5.0, 4.0]),
    ("nan_last", [4], 0, [1.0, 2.0, 3.0, NAN]),
    ("nan_multi", [6], 0, [1.0, NAN, 2.0, NAN, 0.0, 7.0]),
    ("nan_all", [4], 0, [NAN, NAN, NAN, NAN]),
    ("batched_one_lane_nan", [2, 4], 1, [1.0, NAN, 2.0, 3.0, 0.5, 1.5, 4.0, 2.5]),
    ("batched_dim0_nan", [3, 2], 0, [1.0, -1.0, NAN, 2.0, 0.5, 3.0]),
    ("batched_all_nan_lane", [2, 3], 1, [NAN, NAN, NAN, 1.0, 2.0, 3.0]),
    ("nan_all_dim_none", [2, 3], None, [NAN, 1.0, NAN, 2.0, 5.0, NAN]),
]

INF_CASES = [
    ("posinf_mid", [5], 0, [1.0, INF, 2.0, INF, 0.5]),
    ("neginf_mid", [5], 0, [1.0, -INF, 2.0, -INF, 0.5]),
    ("posinf_neginf", [5], 0, [INF, -INF, 1.0, -INF, INF]),
    ("single_posinf", [4], 0, [1.0, INF, 2.0, 3.0]),
    ("nan_and_inf", [5], 0, [INF, NAN, -INF, 1.0, 2.0]),
    ("batched_inf_dim1", [2, 4], 1, [1.0, INF, 2.0, 3.0, 0.5, 1.5, -INF, 4.0]),
]

TIE_CASES = [
    ("tied_len9", [9], 0, "tied"),
    ("tied_len8", [8], 0, "tied"),
    ("all_equal", [5], 0, "const"),
    ("signed_zeros", [6], 0, "zeros"),
    ("tied_batched_dim1", [3, 8], 1, "tied"),
    ("tied_all", [4, 5], None, "tied"),
]


def make_tie_input(shape, kind, gen):
    if kind == "tied":
        return tied(shape, gen)
    if kind == "const":
        return torch.full(shape, 1.5, dtype=torch.float32)
    if kind == "zeros":
        return torch.tensor([0.0, -0.0, 0.0, -0.0, -0.0, 0.0], dtype=torch.float32).reshape(shape)
    raise AssertionError(kind)


def make_error(spec, name, shape, x=None):
    if x is None:
        x = torch.zeros(*shape) if shape else torch.tensor(1.0)
    try:
        r, _ = apply(spec, x)
        return {"name": f"{name}__{spec_name(spec)}", "op": spec["op"], "shape": shape,
                "dim": spec["dim"], "k": spec.get("k"),
                "q_bits": None if "q" not in spec else to_bits([spec["q"]])[0],
                "interp": spec.get("interp"),
                "torch_raises": False, "message": "", "out_shape": list(r.shape),
                "out_bits": flat_bits(r)}
    except Exception as e:  # noqa: BLE001 - 例外型・文面を実測として記録する
        return {"name": f"{name}__{spec_name(spec)}", "op": spec["op"], "shape": shape,
                "dim": spec["dim"], "k": spec.get("k"),
                "q_bits": None if "q" not in spec else to_bits([spec["q"]])[0],
                "interp": spec.get("interp"),
                "torch_raises": True, "message": f"{type(e).__name__}: {e}",
                "out_shape": [], "out_bits": []}


def main():
    gen = torch.Generator().manual_seed(SEED)
    finite = []
    for name, shape, dim in FINITE_SHAPES:
        x0 = rnd(shape, gen)
        for spec in all_specs(shape, dim, lane_len(shape, dim), QS):
            finite.append(record(name, spec, x0, gen))
    for name, n, qs in RANK_CASES:
        x0 = rnd([n], gen)
        for interp in ("higher", "lower", "nearest", "linear"):
            for q in qs:
                spec = {"op": "quantile", "dim": 0, "q": f32(q), "interp": interp}
                finite.append(record(name, spec, x0, gen))
    ties = []
    for name, shape, dim, kind in TIE_CASES:
        x0 = make_tie_input(shape, kind, gen)
        for spec in all_specs(shape, dim, lane_len(shape, dim), [0.0, 0.3, 0.5, 1.0],
                              ["linear", "lower", "higher", "nearest"]):
            ties.append(record(name, spec, x0, gen))
    nans = []
    for name, shape, dim, vals in NAN_CASES:
        x0 = torch.tensor(vals, dtype=torch.float32).reshape(shape)
        for spec in all_specs(shape, dim, lane_len(shape, dim), [0.3, 0.5],
                              ["linear", "lower", "higher", "midpoint", "nearest"]):
            nans.append(record(name, spec, x0, gen))
    infs = []
    for name, shape, dim, vals in INF_CASES:
        x0 = torch.tensor(vals, dtype=torch.float32).reshape(shape)
        for spec in all_specs(shape, dim, lane_len(shape, dim), [0.0, 0.3, 0.5, 0.9, 1.0],
                              ["linear", "midpoint", "lower", "higher", "nearest"]):
            infs.append(record(name, spec, x0, gen))
    # nansum／nanmean に非有限の上流勾配を流す（NaN 位置の勾配が乗算で決まるかの実測）。
    upstream = []
    x_nan = torch.tensor([NAN, 2.0, 1.0, NAN, NAN, 4.0], dtype=torch.float32).reshape(3, 2)
    for gname, gvals in (("g_inf", [INF, 1.0]), ("g_nan", [NAN, 1.0]), ("g_neginf", [-INF, 0.0]),
                         ("g_zero", [0.0, 0.0])):
        g = torch.tensor(gvals, dtype=torch.float32)
        for op in ("nansum", "nanmean"):
            upstream.append(record(gname, {"op": op, "dim": 0}, x_nan, gen, g_override=g))
    x_all = torch.tensor([NAN, NAN, 3.0, 4.0], dtype=torch.float32).reshape(2, 2)
    for gname, gvals in (("g_inf", [INF, 1.0]), ("g_zero", [0.0, 0.0])):
        g = torch.tensor(gvals, dtype=torch.float32)
        for op in ("nansum", "nanmean"):
            upstream.append(record("alllane_" + gname, {"op": op, "dim": 1}, x_all, gen, g_override=g))
    errors = []
    err_specs = [
        ("empty_axis", [0], {"op": "median_dim", "dim": 0}),
        ("empty_axis", [0], {"op": "kthvalue", "dim": 0, "k": 1}),
        ("empty_axis", [0], {"op": "quantile", "dim": 0, "q": f32(0.5), "interp": "linear"}),
        ("empty_axis", [0], {"op": "nansum", "dim": 0}),
        ("empty_axis", [0], {"op": "nanmean", "dim": 0}),
        ("empty_all", [0], {"op": "median_all", "dim": None}),
        ("empty_all", [0], {"op": "quantile", "dim": None, "q": f32(0.5), "interp": "linear"}),
        ("empty_all", [0], {"op": "nansum", "dim": None}),
        ("empty_all", [0], {"op": "nanmean", "dim": None}),
        ("empty_rows", [0, 2], {"op": "nansum", "dim": 0}),
        ("empty_rows", [0, 2], {"op": "nanmean", "dim": 0}),
        ("k0", [3], {"op": "kthvalue", "dim": 0, "k": 0}),
        ("k_gt_n", [3], {"op": "kthvalue", "dim": 0, "k": 4}),
        ("q_neg", [3], {"op": "quantile", "dim": 0, "q": f32(-0.1), "interp": "linear"}),
        ("q_gt1", [3], {"op": "quantile", "dim": 0, "q": f32(1.5), "interp": "linear"}),
        ("q_nan", [3], {"op": "quantile", "dim": 0, "q": NAN, "interp": "linear"}),
        ("dim_oob", [4], {"op": "median_dim", "dim": 1}),
        ("dim_oob", [4], {"op": "kthvalue", "dim": 1, "k": 1}),
        ("dim_oob", [4], {"op": "quantile", "dim": 1, "q": f32(0.5), "interp": "linear"}),
        ("dim_oob", [4], {"op": "nansum", "dim": 1}),
        ("dim_oob", [4], {"op": "nanmean", "dim": 1}),
        ("rank0_dim0", [], {"op": "median_dim", "dim": 0}),
        ("rank0_dim0", [], {"op": "kthvalue", "dim": 0, "k": 1}),
        ("rank0_dim0", [], {"op": "nansum", "dim": 0}),
        ("rank0_all", [], {"op": "median_all", "dim": None}),
        ("rank0_all", [], {"op": "nansum", "dim": None}),
        ("neg_dim", [4], {"op": "median_dim", "dim": -1}),
        ("neg_dim", [4], {"op": "nansum", "dim": -1}),
    ]
    for name, shape, spec in err_specs:
        errors.append(make_error(spec, name, shape))
    doc = {
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "seed": SEED,
        "finite_cases": finite,
        "tie_cases": ties,
        "nan_cases": nans,
        "inf_cases": infs,
        "nonfinite_upstream_cases": upstream,
        "error_cases": errors,
    }
    json.dump(doc, sys.stdout, ensure_ascii=False, separators=(",", ":"))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
