#!/usr/bin/env python3
"""max_pool3d／avg_pool3d の PyTorch 参照値を生成する。

イシュー #2643（親 #2625）の `tests/pool3d_parity.rs` が参照する固定フィクスチャ
`pool3d_reference.json` の生成スクリプト。CI は Python／PyTorch に依存せず、
コミット済み JSON のみを読む（`README.md` 参照）。

f32 はすべて u32 ビットパターン配列で保存する（JSON は NaN／inf を運べないため）。
入力・上流勾配 `g`・出力・索引・入力勾配をすべて保存し、Rust 側で入力を再生成しない。
損失は `(out * g).sum()`。dtype は float32。

空窓になる構成（kernel=2 かつ dilation > 入力長）は torch で実行しない
（`docs/pooling-ops-design.md` §3 が torch 側の縮退動作を記録している。本実装の拒否は
Rust 側のテストだけで固定する）。
"""

import json
import platform
import struct
import sys

import torch
import torch.nn.functional as F

EXPECTED_TORCH_PREFIX = "2.14.0"
assert torch.__version__.startswith(EXPECTED_TORCH_PREFIX), torch.__version__

SEED = 2643
NAN = float("nan")
INF = float("inf")


def to_bits(values):
    return [struct.unpack("<I", struct.pack("<f", float(v)))[0] for v in values]


def flat_bits(t):
    return to_bits(t.detach().contiguous().reshape(-1).tolist())


def numel(shape):
    n = 1
    for s in shape:
        n *= s
    return n


def make_input(shape, kind, gen):
    n = numel(shape)
    if kind == "randn":
        return torch.randn(*shape, generator=gen, dtype=torch.float32)
    if kind == "tied":
        # 0.5 刻みに丸めて窓内・窓間のタイを作る。
        return (torch.randn(*shape, generator=gen, dtype=torch.float32) * 2).round() / 2
    if kind == "const":
        return torch.full(shape, 1.5, dtype=torch.float32)
    if kind == "zeros":
        vals = [0.0, -0.0] * (n // 2 + 1)
        return torch.tensor(vals[:n], dtype=torch.float32).reshape(shape)
    if kind == "neg":
        return -(torch.randn(*shape, generator=gen, dtype=torch.float32).abs() + 0.1)
    raise AssertionError(kind)


# 有限ケース: (name, in_shape, kernel, stride, padding, dilation, kind)。
# stride=None は kernel 既定。dilation は Max のみ（avg_pool3d に dilation は無い）。
FINITE_CASES = [
    ("iso_k2_default_stride", [1, 1, 4, 4, 4], [2, 2, 2], None, [0, 0, 0], [1, 1, 1], "randn"),
    ("iso_k3_s1_overlap", [1, 2, 5, 5, 5], [3, 3, 3], [1, 1, 1], [0, 0, 0], [1, 1, 1], "randn"),
    ("aniso_k_s", [2, 2, 5, 6, 4], [2, 3, 2], [1, 2, 2], [0, 0, 0], [1, 1, 1], "randn"),
    ("pad1_k3_s2", [1, 2, 5, 5, 5], [3, 3, 3], [2, 2, 2], [1, 1, 1], [1, 1, 1], "randn"),
    ("pad_half_aniso", [1, 1, 4, 5, 6], [2, 3, 4], [1, 2, 2], [1, 1, 2], [1, 1, 1], "randn"),
    ("pad_neg_values", [1, 1, 4, 4, 4], [2, 2, 2], [1, 1, 1], [1, 1, 1], [1, 1, 1], "neg"),
    ("kernel_eq_input", [1, 1, 3, 3, 3], [3, 3, 3], None, [0, 0, 0], [1, 1, 1], "randn"),
    ("multi_batch_s1", [3, 2, 4, 4, 4], [2, 2, 2], [1, 1, 1], [0, 0, 0], [1, 1, 1], "randn"),
    ("n1c1_k3_s3", [1, 1, 6, 6, 6], [3, 3, 3], [3, 3, 3], [0, 0, 0], [1, 1, 1], "randn"),
    ("stride_gt_kernel", [1, 1, 7, 7, 7], [2, 2, 2], [3, 3, 3], [0, 0, 0], [1, 1, 1], "randn"),
    ("ties_overlap", [2, 2, 4, 4, 4], [2, 2, 2], [1, 1, 1], [0, 0, 0], [1, 1, 1], "tied"),
    ("all_equal", [1, 2, 3, 3, 3], [2, 2, 2], [1, 1, 1], [0, 0, 0], [1, 1, 1], "const"),
    ("signed_zeros", [1, 1, 2, 2, 2], [2, 2, 2], None, [0, 0, 0], [1, 1, 1], "zeros"),
    ("unit_kernel", [1, 2, 3, 3, 3], [1, 1, 1], None, [0, 0, 0], [1, 1, 1], "randn"),
    ("dilation_k2_d2", [1, 1, 6, 6, 6], [2, 2, 2], [1, 1, 1], [0, 0, 0], [2, 2, 2], "randn"),
    ("dilation_k3_d2_pad1", [1, 2, 7, 7, 7], [3, 3, 3], [2, 2, 2], [1, 1, 1], [2, 2, 2], "randn"),
    ("dilation_aniso", [1, 1, 6, 7, 8], [2, 3, 2], [1, 1, 2], [0, 1, 0], [3, 1, 2], "tied"),
]

# 非有限ケース: (name, in_shape, kernel, stride, 差し込み {flat 位置: 値})。
# 入力は 0.25 刻みの単調増加列に差し込む。
NONFINITE_CASES = [
    ("nan_single", [1, 1, 3, 3, 3], [2, 2, 2], [1, 1, 1], {13: NAN}),
    ("nan_multi_in_window", [1, 1, 3, 3, 3], [2, 2, 2], [1, 1, 1], {13: NAN, 14: NAN, 4: NAN}),
    ("posinf", [1, 1, 3, 3, 3], [2, 2, 2], [1, 1, 1], {13: INF}),
    ("neginf", [1, 1, 3, 3, 3], [2, 2, 2], [1, 1, 1], {13: -INF}),
    ("neginf_whole_window", [1, 1, 2, 2, 2], [2, 2, 2], None,
     {i: -INF for i in range(8)}),
    ("nan_and_inf", [1, 1, 3, 3, 3], [2, 2, 2], [1, 1, 1], {13: NAN, 14: INF, 0: -INF}),
]

# torch が例外を出すか否かを実測して記録する境界ケース。
# (name, in_shape, kernel, stride, padding, dilation, ceil_mode)
ERROR_CASES = [
    ("padding_over_half", [1, 1, 4, 4, 4], [2, 2, 2], None, [2, 0, 0], [1, 1, 1], False),
    ("kernel_zero", [1, 1, 4, 4, 4], [0, 2, 2], None, [0, 0, 0], [1, 1, 1], False),
    ("stride_zero", [1, 1, 4, 4, 4], [2, 2, 2], [1, 0, 1], [0, 0, 0], [1, 1, 1], False),
    ("dilation_zero", [1, 1, 4, 4, 4], [2, 2, 2], None, [0, 0, 0], [1, 0, 1], False),
    ("kernel_gt_input", [1, 1, 2, 4, 4], [3, 2, 2], None, [0, 0, 0], [1, 1, 1], False),
    ("spatial_zero_d", [1, 1, 0, 4, 4], [2, 2, 2], None, [0, 0, 0], [1, 1, 1], False),
    ("spatial_zero_w", [1, 1, 4, 4, 0], [1, 1, 1], None, [0, 0, 0], [1, 1, 1], False),
    ("batch_zero", [0, 2, 4, 4, 4], [2, 2, 2], None, [0, 0, 0], [1, 1, 1], False),
    ("channel_zero", [1, 0, 4, 4, 4], [2, 2, 2], None, [0, 0, 0], [1, 1, 1], False),
    ("rank4_unbatched", [2, 4, 4, 4], [2, 2, 2], None, [0, 0, 0], [1, 1, 1], False),
    ("rank3", [4, 4, 4], [2, 2, 2], None, [0, 0, 0], [1, 1, 1], False),
    ("ceil_mode_true", [1, 1, 5, 5, 5], [2, 2, 2], None, [0, 0, 0], [1, 1, 1], True),
]


def run_max(x0, k, s, p, d):
    x = x0.clone().requires_grad_(True)
    y, idx = F.max_pool3d(x, k, s, p, d, ceil_mode=False, return_indices=True)
    g = torch.randn(*y.shape, generator=GEN, dtype=torch.float32)
    (y * g).sum().backward()
    return x, y, idx, g


def run_avg(x0, k, s, p, cip):
    x = x0.clone().requires_grad_(True)
    y = F.avg_pool3d(x, k, s, p, ceil_mode=False, count_include_pad=cip)
    g = torch.randn(*y.shape, generator=GEN, dtype=torch.float32)
    (y * g).sum().backward()
    return x, y, g


def record_max(name, shape, k, s, p, d, x0):
    x, y, idx, g = run_max(x0, k, s, p, d)
    return {
        "name": f"max_{name}", "op": "max", "in_shape": shape, "out_shape": list(y.shape),
        "kernel": k, "stride": s, "padding": p, "dilation": d, "count_include_pad": None,
        "x_bits": flat_bits(x), "g_bits": flat_bits(g), "out_bits": flat_bits(y),
        "grad_bits": flat_bits(x.grad), "index": idx.reshape(-1).tolist(),
    }


def record_avg(name, shape, k, s, p, cip, x0):
    x, y, g = run_avg(x0, k, s, p, cip)
    return {
        "name": f"avg_{'incl' if cip else 'excl'}_{name}", "op": "avg", "in_shape": shape,
        "out_shape": list(y.shape), "kernel": k, "stride": s, "padding": p,
        "dilation": [1, 1, 1], "count_include_pad": cip,
        "x_bits": flat_bits(x), "g_bits": flat_bits(g), "out_bits": flat_bits(y),
        "grad_bits": flat_bits(x.grad),
    }


def make_error(op, name, shape, k, s, p, d, ceil):
    x = torch.zeros(*shape)
    base = {"name": f"{op}_{name}", "op": op, "in_shape": shape, "kernel": k, "stride": s,
            "padding": p, "dilation": d, "ceil_mode": ceil}
    try:
        if op == "max":
            r = F.max_pool3d(x, k, s, p, d, ceil_mode=ceil)
        else:
            r = F.avg_pool3d(x, k, s, p, ceil_mode=ceil)
        return {**base, "torch_raises": False, "message": "", "out_shape": list(r.shape)}
    except Exception as e:  # noqa: BLE001 - 例外型・文面を実測として記録する
        return {**base, "torch_raises": True, "message": f"{type(e).__name__}: {e}",
                "out_shape": []}


GEN = torch.Generator().manual_seed(SEED)


def main():
    finite = []
    for name, shape, k, s, p, d, kind in FINITE_CASES:
        x0 = make_input(shape, kind, GEN)
        finite.append(record_max(name, shape, k, s, p, d, x0))
        if d == [1, 1, 1]:
            for cip in (True, False):
                finite.append(record_avg(name, shape, k, s, p, cip, x0))
    nonfinite = []
    for name, shape, k, s, inject in NONFINITE_CASES:
        n = numel(shape)
        vals = [0.25 * i for i in range(n)]
        for pos, v in inject.items():
            vals[pos] = v
        x0 = torch.tensor(vals, dtype=torch.float32).reshape(shape)
        nonfinite.append(record_max(name, shape, k, s, [0, 0, 0], [1, 1, 1], x0))
        for cip in (True, False):
            nonfinite.append(record_avg(name, shape, k, s, [0, 0, 0], cip, x0))
    errors = []
    for name, shape, k, s, p, d, ceil in ERROR_CASES:
        errors.append(make_error("max", name, shape, k, s, p, d, ceil))
        errors.append(make_error("avg", name, shape, k, s, p, [1, 1, 1], ceil))
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
