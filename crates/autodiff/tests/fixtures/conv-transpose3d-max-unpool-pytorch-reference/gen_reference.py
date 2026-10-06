#!/usr/bin/env python3
"""conv_transpose3d／max_unpool1d・2d・3d の PyTorch 参照値を生成する。

イシュー #2644（親 #2625）の `tests/conv_transpose3d_parity.rs`・`tests/max_unpool_parity.rs` が
参照する固定フィクスチャ `conv_transpose3d_max_unpool_reference.json` の生成スクリプト。
CI は Python／PyTorch に依存せず、コミット済み JSON のみを読む（`README.md` 参照）。

f32 はすべて u32 ビットパターン配列で保存する（JSON は NaN／inf を運べないため）。
入力・重み・bias・上流勾配 `g`・出力・各入力勾配をすべて保存し、Rust 側で入力を再生成しない。
損失は `(out * g).sum()`。dtype は float32。
"""

import json
import platform
import struct
import sys

import torch
import torch.nn.functional as F

EXPECTED_TORCH_PREFIX = "2.14.0"
assert torch.__version__.startswith(EXPECTED_TORCH_PREFIX), torch.__version__

# 重複索引の max_unpool は PyTorch の並列 CPU カーネルが書き込みを競合させ、実行ごとに勝者が
# 変わりうる（実測）。再生成で同一の JSON を得られるよう単一スレッドに固定する。
torch.set_num_threads(1)

SEED = 2644
NAN = float("nan")
INF = float("inf")
GEN = torch.Generator().manual_seed(SEED)


def to_bits(values):
    return [struct.unpack("<I", struct.pack("<f", float(v)))[0] for v in values]


def flat_bits(t):
    return to_bits(t.detach().contiguous().reshape(-1).tolist())


def randn(*shape):
    return torch.randn(*shape, generator=GEN, dtype=torch.float32)


def raises(fn):
    try:
        fn()
    except Exception as e:  # noqa: BLE001 - 例外種別を問わず「拒否した」ことだけを記録する
        return True, str(e).splitlines()[0][:160]
    return False, ""


# ---------------------------------------------------------------- ConvTranspose3d

# (name, in_shape, weight_shape, bias, stride, padding, output_padding, dilation, groups)
CT_CASES = [
    ("basic_k2_s1", [1, 1, 2, 2, 2], [1, 1, 2, 2, 2], False, [1, 1, 1], [0, 0, 0], [0, 0, 0], [1, 1, 1], 1),
    ("bias_cout_eq_wout", [1, 2, 2, 2, 2], [2, 3, 2, 2, 2], True, [1, 1, 1], [0, 0, 0], [0, 0, 0], [1, 1, 1], 1),
    ("s2_p1_op1", [1, 2, 3, 3, 3], [2, 3, 3, 3, 3], True, [2, 2, 2], [1, 1, 1], [1, 1, 1], [1, 1, 1], 1),
    ("aniso_all", [2, 2, 2, 3, 4], [2, 2, 2, 3, 2], True, [1, 2, 2], [0, 1, 0], [0, 1, 1], [1, 1, 2], 1),
    ("dilation2", [1, 2, 3, 3, 3], [2, 2, 2, 2, 2], False, [1, 1, 1], [0, 0, 0], [0, 0, 0], [2, 2, 2], 1),
    ("groups2", [1, 4, 2, 3, 3], [4, 3, 2, 2, 2], True, [1, 1, 1], [0, 0, 0], [0, 0, 0], [1, 1, 1], 2),
    ("depthwise", [1, 3, 2, 2, 3], [3, 1, 2, 2, 2], True, [1, 1, 1], [0, 0, 0], [0, 0, 0], [1, 1, 1], 3),
    ("batch2_cin_ne_cout", [2, 3, 2, 2, 2], [3, 2, 3, 3, 3], True, [2, 2, 2], [1, 1, 1], [0, 0, 0], [1, 1, 1], 1),
    ("k111", [1, 2, 2, 2, 2], [2, 3, 1, 1, 1], False, [1, 1, 1], [0, 0, 0], [0, 0, 0], [1, 1, 1], 1),
    ("stride3_op2", [1, 1, 2, 2, 2], [1, 2, 2, 2, 2], True, [3, 3, 3], [0, 0, 0], [2, 2, 2], [1, 1, 1], 1),
]


def ct_case(spec):
    name, in_shape, w_shape, has_bias, s, p, op, d, groups = spec
    x = randn(*in_shape).requires_grad_(True)
    w = randn(*w_shape).requires_grad_(True)
    cout = w_shape[1] * groups
    b = randn(cout).requires_grad_(True) if has_bias else None
    out = F.conv_transpose3d(x, w, b, stride=s, padding=p, output_padding=op, groups=groups, dilation=d)
    g = randn(*out.shape)
    (out * g).sum().backward()
    return {
        "name": name,
        "in_shape": in_shape,
        "weight_shape": w_shape,
        "stride": s,
        "padding": p,
        "output_padding": op,
        "dilation": d,
        "groups": groups,
        "out_shape": list(out.shape),
        "x_bits": flat_bits(x),
        "w_bits": flat_bits(w),
        "b_bits": flat_bits(b) if b is not None else None,
        "g_bits": flat_bits(g),
        "out_bits": flat_bits(out),
        "dx_bits": flat_bits(x.grad),
        "dw_bits": flat_bits(w.grad),
        "db_bits": flat_bits(b.grad) if b is not None else None,
    }


def ct_error(name, fn):
    r, msg = raises(fn)
    return {"name": name, "torch_raises": r, "message": msg}


def rt(*shape):
    return randn(*shape)


CT_ERRORS = [
    ct_error("output_padding_eq_stride", lambda: F.conv_transpose3d(rt(1, 1, 2, 2, 2), rt(1, 1, 2, 2, 2), stride=1, output_padding=1)),
    ct_error("output_padding_lt_dilation_ge_stride", lambda: F.conv_transpose3d(rt(1, 1, 2, 2, 2), rt(1, 1, 2, 2, 2), stride=1, dilation=2, output_padding=1)),
    ct_error("channel_mismatch", lambda: F.conv_transpose3d(rt(1, 2, 2, 2, 2), rt(3, 1, 2, 2, 2))),
    ct_error("groups_not_divide_cin", lambda: F.conv_transpose3d(rt(1, 3, 2, 2, 2), rt(3, 1, 2, 2, 2), groups=2)),
    ct_error("stride_zero", lambda: F.conv_transpose3d(rt(1, 1, 2, 2, 2), rt(1, 1, 2, 2, 2), stride=0)),
    ct_error("dilation_zero", lambda: F.conv_transpose3d(rt(1, 1, 2, 2, 2), rt(1, 1, 2, 2, 2), dilation=0)),
    ct_error("groups_zero", lambda: F.conv_transpose3d(rt(1, 1, 2, 2, 2), rt(1, 1, 2, 2, 2), groups=0)),
    ct_error("input_rank4_batchless", lambda: F.conv_transpose3d(rt(1, 2, 2, 2), rt(1, 1, 2, 2, 2))),
    ct_error("padding_too_large", lambda: F.conv_transpose3d(rt(1, 1, 2, 2, 2), rt(1, 1, 2, 2, 2), padding=5)),
    ct_error("bias_shape_mismatch", lambda: F.conv_transpose3d(rt(1, 1, 2, 2, 2), rt(1, 2, 2, 2, 2), torch.zeros(3))),
    ct_error("batch_zero", lambda: F.conv_transpose3d(rt(0, 1, 2, 2, 2), rt(1, 1, 2, 2, 2))),
]

# ---------------------------------------------------------------- MaxUnpool

POOL = {1: F.max_pool1d, 2: F.max_pool2d, 3: F.max_pool3d}
UNPOOL = {1: F.max_unpool1d, 2: F.max_unpool2d, 3: F.max_unpool3d}


def unpool_case(name, dim, pool_in_shape, k, s, p, output_size=None, values=None, indices=None):
    """`values`／`indices` を渡さなければ max_pool 由来（値・索引）を入力にする。"""
    if isinstance(output_size, int):
        output_size = [output_size]
    if values is None:
        px = randn(*pool_in_shape)
        v, idx = POOL[dim](px, k, s, p, return_indices=True)
        v = v.detach()
    else:
        v, idx = values, indices
    v = v.clone().requires_grad_(True)
    out = UNPOOL[dim](v, idx, k, s, p, output_size=output_size)
    g = randn(*out.shape)
    (out * g).sum().backward()
    return {
        "name": name,
        "dim": dim,
        "kernel": k,
        "stride": s,
        "padding": p,
        "output_size": output_size,
        "in_shape": list(v.shape),
        "out_shape": list(out.shape),
        "x_bits": flat_bits(v),
        "index": [int(i) for i in idx.reshape(-1).tolist()],
        "g_bits": flat_bits(g),
        "out_bits": flat_bits(out),
        "grad_bits": flat_bits(v.grad),
    }


# 非重複索引（stride >= kernel の max_pool 由来）。
UNPOOL_CASES = [
    unpool_case("1d_k2_s2", 1, [2, 3, 8], 2, 2, 0),
    unpool_case("1d_k3_s2_p1", 1, [1, 2, 9], 3, 2, 1),
    unpool_case("1d_odd_len_output_size", 1, [1, 2, 5], 2, 2, 0, output_size=5),
    unpool_case("1d_stride_gt_kernel", 1, [1, 1, 8], 2, 3, 0),
    unpool_case("2d_k2_s2", 2, [2, 2, 4, 6], 2, 2, 0),
    unpool_case("2d_k3_s2_p1", 2, [1, 2, 7, 7], 3, 2, 1),
    unpool_case("2d_aniso_output_size", 2, [1, 2, 6, 8], [2, 3], [2, 2], [0, 1], output_size=[6, 8]),
    unpool_case("2d_odd_output_size", 2, [1, 1, 5, 5], 2, 2, 0, output_size=[5, 5]),
    unpool_case("3d_k2_s2", 3, [1, 2, 4, 4, 6], 2, 2, 0),
    unpool_case("3d_k3_s2_p1", 3, [1, 1, 5, 5, 5], 3, 2, 1),
    unpool_case("3d_aniso", 3, [2, 2, 4, 6, 4], [2, 3, 2], [2, 3, 2], [0, 0, 0]),
    unpool_case("3d_odd_output_size", 3, [1, 1, 5, 5, 5], 2, 2, 0, output_size=[5, 5, 5]),
]

# 重複索引（重なり窓 k=3,s=1 の max_pool 由来）と手作りの重複索引。
UNPOOL_DUP_CASES = [
    unpool_case("1d_overlap_k3_s1", 1, [1, 2, 8], 3, 1, 0),
    unpool_case("2d_overlap_k3_s1", 2, [1, 1, 6, 6], 3, 1, 0),
    unpool_case("3d_overlap_k3_s1", 3, [1, 1, 5, 5, 5], 3, 1, 0),
    unpool_case(
        "1d_manual_dup",
        1,
        None,
        2,
        2,
        0,
        values=torch.tensor([[[1.0, 2.0, 3.0, 4.0]]]),
        indices=torch.tensor([[[1, 1, 3, 3]]]),
    ),
    unpool_case(
        "2d_manual_dup",
        2,
        None,
        2,
        2,
        0,
        values=torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]),
        indices=torch.tensor([[[[0, 0], [5, 0]]]]),
    ),
]

# NaN／inf を含む入力（コピーのみなのでビット保存を確認する）。
UNPOOL_NONFINITE_CASES = [
    unpool_case(
        "1d_nonfinite",
        1,
        None,
        2,
        2,
        0,
        values=torch.tensor([[[NAN, INF, -INF, 1.5]]]),
        indices=torch.tensor([[[1, 2, 5, 6]]]),
    ),
    unpool_case(
        "2d_nonfinite",
        2,
        None,
        2,
        2,
        0,
        values=torch.tensor([[[[NAN, INF], [-INF, -0.0]]]]),
        indices=torch.tensor([[[[0, 3], [8, 15]]]]),
    ),
    unpool_case(
        "3d_nonfinite",
        3,
        None,
        2,
        2,
        0,
        values=torch.tensor([[[[[NAN, INF], [-INF, 2.0]], [[0.0, -0.0], [3.0, NAN]]]]]),
        indices=torch.tensor([[[[[0, 9], [20, 3]], [[40, 41], [60, 7]]]]]),
    ),
]


def unpool_error(name, fn):
    r, msg = raises(fn)
    return {"name": name, "torch_raises": r, "message": msg}


V1 = torch.tensor([[[1.0, 2.0, 3.0]]])
I1 = torch.tensor([[[0, 1, 2]]])
# 1d・in=3・k=2・s=2・p=0 の既定出力長は 6。output_size は default±stride の開区間で許される。
UNPOOL_ERRORS = [
    unpool_error("index_out_of_range", lambda: F.max_unpool1d(V1, torch.tensor([[[0, 1, 6]]]), 2)),
    unpool_error("index_negative", lambda: F.max_unpool1d(V1, torch.tensor([[[0, 1, -1]]]), 2)),
    unpool_error("index_shape_mismatch", lambda: F.max_unpool1d(V1, torch.tensor([[[0, 1]]]), 2)),
    unpool_error("output_size_eq_default_minus_stride", lambda: F.max_unpool1d(V1, I1, 2, output_size=[4])),
    unpool_error("output_size_default_minus_stride_plus1", lambda: F.max_unpool1d(V1, I1, 2, output_size=[5])),
    unpool_error("output_size_default_plus_stride_minus1", lambda: F.max_unpool1d(V1, I1, 2, output_size=[7])),
    unpool_error("output_size_eq_default_plus_stride", lambda: F.max_unpool1d(V1, I1, 2, output_size=[8])),
    unpool_error("output_size_far_below", lambda: F.max_unpool1d(V1, I1, 2, output_size=[3])),
    unpool_error("batchless_input", lambda: F.max_unpool1d(V1[0], I1[0], 2)),
    unpool_error("kernel_zero", lambda: F.max_unpool1d(V1, I1, 0)),
    unpool_error("stride_zero", lambda: F.max_unpool1d(V1, I1, 2, 0)),
    unpool_error("batch_zero", lambda: F.max_unpool1d(torch.zeros(0, 1, 3), torch.zeros(0, 1, 3, dtype=torch.long), 2)),
    unpool_error("channel_zero", lambda: F.max_unpool1d(torch.zeros(1, 0, 3), torch.zeros(1, 0, 3, dtype=torch.long), 2)),
    unpool_error("rank_mismatch_2d_with_3d_input", lambda: F.max_unpool2d(torch.zeros(1, 1, 2, 2, 2), torch.zeros(1, 1, 2, 2, 2, dtype=torch.long), 2)),
]


def main():
    doc = {
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "seed": SEED,
        "conv_transpose3d_cases": [ct_case(c) for c in CT_CASES],
        "conv_transpose3d_error_cases": CT_ERRORS,
        "max_unpool_cases": UNPOOL_CASES,
        "max_unpool_dup_cases": UNPOOL_DUP_CASES,
        "max_unpool_nonfinite_cases": UNPOOL_NONFINITE_CASES,
        "max_unpool_error_cases": UNPOOL_ERRORS,
    }
    json.dump(doc, sys.stdout, separators=(",", ":"))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
