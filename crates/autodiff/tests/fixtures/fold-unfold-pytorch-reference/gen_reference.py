#!/usr/bin/env python3
"""F.unfold／F.fold の PyTorch 参照値を生成する。

イシュー #2645（親 #2625）の `tests/fold_unfold_parity.rs` が参照する固定フィクスチャ
`fold_unfold_reference.json` の生成スクリプト。CI は Python／PyTorch に依存せず、コミット済み
JSON のみを読む（`README.md` 参照）。

f32 はすべて u32 ビットパターン配列で保存する（JSON は NaN／inf を運べないため）。
入力・上流勾配 `g`・出力・入力勾配をすべて保存し、Rust 側で入力を再生成しない。
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

torch.set_num_threads(1)

SEED = 2645
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


def out_len(size, k, s, p, d):
    return (size + 2 * p - d * (k - 1) - 1) // s + 1


# (name, in_shape[N,C,H,W], kernel, stride, padding, dilation)
UNFOLD_CASES = [
    ("k2_s1", [1, 1, 3, 3], [2, 2], [1, 1], [0, 0], [1, 1]),
    ("k3_p1", [1, 2, 4, 4], [3, 3], [1, 1], [1, 1], [1, 1]),
    ("stride2", [1, 2, 5, 5], [3, 3], [2, 2], [0, 0], [1, 1]),
    ("stride_gt_kernel", [1, 1, 6, 6], [2, 2], [3, 3], [0, 0], [1, 1]),
    ("dilation2", [1, 2, 6, 6], [2, 2], [1, 1], [0, 0], [2, 2]),
    ("aniso_all", [2, 3, 7, 6], [2, 3], [2, 1], [1, 0], [2, 1]),
    ("k1", [1, 3, 3, 4], [1, 1], [1, 1], [0, 0], [1, 1]),
    ("kernel_eq_input", [1, 2, 3, 3], [3, 3], [1, 1], [0, 0], [1, 1]),
    ("big_padding", [1, 1, 3, 3], [3, 3], [1, 1], [2, 2], [1, 1]),
    ("batch3_c1", [3, 1, 4, 5], [2, 2], [1, 2], [1, 1], [1, 1]),
]

# (name, N, C, output_size, kernel, stride, padding, dilation)
FOLD_CASES = [
    ("k2_s1_overlap", 1, 1, [3, 3], [2, 2], [1, 1], [0, 0], [1, 1]),
    ("k3_p1_overlap", 1, 2, [4, 4], [3, 3], [1, 1], [1, 1], [1, 1]),
    ("stride2_overlap", 1, 2, [5, 5], [3, 3], [2, 2], [0, 0], [1, 1]),
    ("k2_s2_disjoint", 1, 2, [6, 6], [2, 2], [2, 2], [0, 0], [1, 1]),
    ("stride_gt_kernel_disjoint", 1, 1, [6, 6], [2, 2], [3, 3], [0, 0], [1, 1]),
    ("dilation2", 1, 2, [6, 6], [2, 2], [1, 1], [0, 0], [2, 2]),
    ("aniso_all", 2, 3, [7, 6], [2, 3], [2, 1], [1, 0], [2, 1]),
    ("k1_disjoint", 1, 3, [3, 4], [1, 1], [1, 1], [0, 0], [1, 1]),
    ("kernel_eq_output", 1, 2, [3, 3], [3, 3], [1, 1], [0, 0], [1, 1]),
    ("big_padding", 1, 1, [3, 3], [3, 3], [1, 1], [2, 2], [1, 1]),
    ("padded_disjoint", 1, 1, [4, 4], [2, 2], [2, 2], [1, 1], [1, 1]),
]


def unfold_case(spec):
    name, in_shape, k, s, p, d = spec
    x = randn(*in_shape).requires_grad_(True)
    out = F.unfold(x, k, dilation=d, padding=p, stride=s)
    g = randn(*out.shape)
    (out * g).sum().backward()
    return {
        "name": name,
        "in_shape": in_shape,
        "kernel": k,
        "stride": s,
        "padding": p,
        "dilation": d,
        "out_shape": list(out.shape),
        "x_bits": flat_bits(x),
        "g_bits": flat_bits(g),
        "out_bits": flat_bits(out),
        "dx_bits": flat_bits(x.grad),
    }


def fold_case(spec):
    name, n, c, osz, k, s, p, d = spec
    l = out_len(osz[0], k[0], s[0], p[0], d[0]) * out_len(osz[1], k[1], s[1], p[1], d[1])
    in_shape = [n, c * k[0] * k[1], l]
    x = randn(*in_shape).requires_grad_(True)
    out = F.fold(x, osz, k, dilation=d, padding=p, stride=s)
    g = randn(*out.shape)
    (out * g).sum().backward()
    # 各出力位置への寄与数の最大値（> 1 なら窓が重なり、f32 累積順の差が出うる）。
    cover = F.fold(torch.ones(in_shape), osz, k, dilation=d, padding=p, stride=s)
    return {
        "name": name,
        "in_shape": in_shape,
        "output_size": osz,
        "kernel": k,
        "stride": s,
        "padding": p,
        "dilation": d,
        "overlapping": bool(cover.max().item() > 1.0),
        "out_shape": list(out.shape),
        "x_bits": flat_bits(x),
        "g_bits": flat_bits(g),
        "out_bits": flat_bits(out),
        "dx_bits": flat_bits(x.grad),
    }


def nonfinite_unfold(name, in_shape, values, k, s, p, d):
    x = torch.tensor(values, dtype=torch.float32).reshape(in_shape).requires_grad_(True)
    out = F.unfold(x, k, dilation=d, padding=p, stride=s)
    g = randn(*out.shape)
    (out * g).sum().backward()
    return {
        "name": name,
        "kind": "unfold",
        "in_shape": in_shape,
        "kernel": k,
        "stride": s,
        "padding": p,
        "dilation": d,
        "out_shape": list(out.shape),
        "x_bits": flat_bits(x),
        "g_bits": flat_bits(g),
        "out_bits": flat_bits(out),
        "dx_bits": flat_bits(x.grad),
    }


def nonfinite_fold(name, in_shape, values, osz, k, s, p, d):
    x = torch.tensor(values, dtype=torch.float32).reshape(in_shape).requires_grad_(True)
    out = F.fold(x, osz, k, dilation=d, padding=p, stride=s)
    g = randn(*out.shape)
    (out * g).sum().backward()
    return {
        "name": name,
        "kind": "fold",
        "in_shape": in_shape,
        "output_size": osz,
        "kernel": k,
        "stride": s,
        "padding": p,
        "dilation": d,
        "out_shape": list(out.shape),
        "x_bits": flat_bits(x),
        "g_bits": flat_bits(g),
        "out_bits": flat_bits(out),
        "dx_bits": flat_bits(x.grad),
    }


NONFINITE = [
    nonfinite_unfold(
        "unfold_nan_inf_negzero",
        [1, 1, 3, 3],
        [NAN, INF, -INF, -0.0, 0.0, 1.0, -1.0, NAN, -0.0],
        [2, 2], [1, 1], [0, 0], [1, 1],
    ),
    nonfinite_unfold(
        "unfold_padded_negzero",
        [1, 1, 2, 2],
        [-0.0, -0.0, INF, NAN],
        [2, 2], [1, 1], [1, 1], [1, 1],
    ),
    # 窓が重ならない設定（k=2・s=2）: 各出力位置への寄与は高々 1 つ。
    nonfinite_fold(
        "fold_disjoint_nan_inf_negzero",
        [1, 4, 4],
        [NAN, INF, -INF, -0.0, 0.0, 1.0, -1.0, NAN, -0.0, 2.0, 3.0, INF, -INF, 4.0, 5.0, -0.0],
        [4, 4], [2, 2], [2, 2], [0, 0], [1, 1],
    ),
    # 重なる設定（k=2・s=1）での inf＋(-inf)→NaN の伝播。
    nonfinite_fold(
        "fold_overlap_inf_cancel",
        [1, 4, 4],
        [INF, 1.0, 2.0, 3.0, 1.0, -INF, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
        [3, 3], [2, 2], [1, 1], [0, 0], [1, 1],
    ),
]


def error(name, fn):
    r, msg = raises(fn)
    return {"name": name, "torch_raises": r, "message": msg}


def rt(*shape):
    return randn(*shape)


ERRORS = [
    error("fold_k_not_divisible", lambda: F.fold(rt(1, 5, 4), [3, 3], [2, 2])),
    error("fold_l_mismatch", lambda: F.fold(rt(1, 4, 5), [3, 3], [2, 2])),
    error("fold_output_size_too_small", lambda: F.fold(rt(1, 4, 4), [2, 2], [2, 2])),
    error("fold_kernel_zero", lambda: F.fold(rt(1, 4, 4), [3, 3], [0, 2])),
    error("fold_stride_zero", lambda: F.fold(rt(1, 4, 4), [3, 3], [2, 2], stride=[0, 1])),
    error("fold_dilation_zero", lambda: F.fold(rt(1, 4, 4), [3, 3], [2, 2], dilation=[0, 1])),
    error("fold_batchless_2d", lambda: F.fold(rt(4, 4), [3, 3], [2, 2])),
    error("fold_rank4", lambda: F.fold(rt(1, 1, 4, 4), [3, 3], [2, 2])),
    error("fold_batch_zero", lambda: F.fold(rt(0, 4, 4), [3, 3], [2, 2])),
    error("fold_c_zero", lambda: F.fold(rt(1, 0, 4), [3, 3], [2, 2])),
    error("fold_output_size_zero", lambda: F.fold(rt(1, 4, 1), [0, 3], [2, 2])),
    error("unfold_kernel_zero", lambda: F.unfold(rt(1, 1, 3, 3), [0, 2])),
    error("unfold_stride_zero", lambda: F.unfold(rt(1, 1, 3, 3), [2, 2], stride=[0, 1])),
    error("unfold_dilation_zero", lambda: F.unfold(rt(1, 1, 3, 3), [2, 2], dilation=[0, 1])),
    error("unfold_kernel_gt_input", lambda: F.unfold(rt(1, 1, 3, 3), [5, 5])),
    error("unfold_batchless_3d", lambda: F.unfold(rt(1, 3, 3), [2, 2])),
    error("unfold_rank5", lambda: F.unfold(rt(1, 1, 1, 3, 3), [2, 2])),
    error("unfold_batch_zero", lambda: F.unfold(rt(0, 1, 3, 3), [2, 2])),
    error("unfold_c_zero", lambda: F.unfold(rt(1, 0, 3, 3), [2, 2])),
]

result = {
    "torch_version": torch.__version__,
    "python_version": platform.python_version(),
    "seed": SEED,
    "unfold_cases": [unfold_case(c) for c in UNFOLD_CASES],
    "fold_cases": [fold_case(c) for c in FOLD_CASES],
    "nonfinite_cases": NONFINITE,
    "error_cases": ERRORS,
}
json.dump(result, sys.stdout, separators=(",", ":"))
sys.stdout.write("\n")
