#!/usr/bin/env python3
"""local_response_norm／weight_norm／spectral_norm の PyTorch 参照値を生成する。

イシュー #2646（親 #2625）の `tests/lrn_parity.rs`・`tests/weight_reparam_parity.rs` が参照する
固定フィクスチャ `lrn_weight_reparam_reference.json` の生成スクリプト。CI は Python／PyTorch に
依存せず、コミット済み JSON のみを読む（`README.md` 参照）。

f32 はすべて u32 ビットパターン配列で保存する（JSON は NaN／inf を運べないため）。
入力・上流勾配・出力・入力勾配をすべて保存し、Rust 側で入力を再生成しない。
損失は `(out * up).sum()`。dtype は float32。

spectral_norm は `_SpectralNorm` を直接生成し、**初期化直後の `u0`／`v0`（予備反復 15 回後）と、
forward を 1 回だけ呼んだ後の `u1`／`v1`** の両方を保存する（training モードでは forward 1 回ごとに
反復が走るため 1 記録につき forward は 1 回。初期化乱数を Rust 側で再現しない）。
"""

import json
import platform
import struct
import sys

import torch
import torch.nn.functional as F
from torch.nn.utils.parametrizations import _SpectralNorm

EXPECTED_TORCH_PREFIX = "2.14.0"
assert torch.__version__.startswith(EXPECTED_TORCH_PREFIX), torch.__version__

torch.set_num_threads(1)

SEED = 2646
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
    if kind == "randn":
        return torch.randn(*shape, generator=gen, dtype=torch.float32)
    if kind == "big":
        return torch.randn(*shape, generator=gen, dtype=torch.float32) * 4.0
    if kind == "pos":
        return torch.rand(*shape, generator=gen, dtype=torch.float32) + 0.25
    raise AssertionError(kind)


# ---------------------------------------------------------------- LocalResponseNorm

# (name, shape, size, alpha, beta, k, kind)。
LRN_CASES = [
    ("r3_odd_default_params", [2, 5, 3], 3, 1e-4, 0.75, 1.0, "randn"),
    ("r3_odd_strong", [2, 5, 3], 3, 0.5, 0.75, 2.0, "big"),
    ("r3_even2", [2, 5, 3], 2, 0.5, 0.75, 2.0, "big"),
    ("r3_even4", [2, 6, 2], 4, 0.8, 0.5, 1.0, "big"),
    ("r3_size5", [1, 7, 4], 5, 0.4, 0.75, 1.5, "randn"),
    ("r4_size5", [2, 6, 3, 3], 5, 0.3, 0.75, 2.0, "big"),
    ("r5_size3", [1, 4, 2, 2, 3], 3, 0.6, 1.0, 1.0, "randn"),
    ("r4_even6", [1, 8, 2, 2], 6, 0.9, 0.75, 1.0, "big"),
    ("size1", [2, 4, 3], 1, 0.7, 0.75, 1.0, "randn"),
    ("size_gt_c", [2, 3, 2], 7, 0.7, 0.75, 1.0, "big"),
    ("size_eq_c", [1, 4, 3], 4, 0.7, 0.75, 1.0, "big"),
    ("size_eq_c_odd", [1, 5, 3], 5, 0.7, 0.75, 1.0, "big"),
    ("c1_size3", [2, 1, 4], 3, 0.7, 0.75, 1.0, "randn"),
    ("beta1", [2, 5, 3], 3, 0.5, 1.0, 1.0, "big"),
    ("beta2", [2, 5, 3], 3, 0.5, 2.0, 1.0, "randn"),
    ("beta_half_k0", [2, 5, 3], 3, 0.5, 0.5, 0.0, "pos"),
    ("beta0", [1, 4, 3], 3, 0.5, 0.0, 1.0, "randn"),
    ("alpha0", [1, 4, 3], 3, 0.0, 0.75, 2.0, "randn"),
    ("negative_alpha", [1, 4, 3], 3, -0.05, 0.75, 2.0, "randn"),
    ("wide_c16_even8", [1, 16, 2], 8, 0.25, 0.75, 1.0, "big"),
    ("n3_c9_size4", [3, 9, 2, 2], 4, 0.2, 0.75, 2.0, "randn"),
]

# 非有限（name, shape, size, injects{pos: value}）。
LRN_NONFINITE_CASES = [
    ("nan_single", [1, 4, 2], 3, {3: NAN}),
    ("nan_two_channels", [1, 5, 2], 3, {2: NAN, 4: NAN}),
    ("inf_pos", [1, 4, 2], 3, {2: INF}),
    ("inf_neg", [1, 4, 2], 2, {2: -INF}),
    ("nan_and_inf", [1, 5, 1], 4, {1: NAN, 3: INF}),
]

# (name, shape, size, alpha, beta, k)。torch が例外を出すか否かを実測する。
LRN_ERROR_CASES = [
    ("rank2", [2, 3], 3, 1e-4, 0.75, 1.0),
    ("rank1", [3], 3, 1e-4, 0.75, 1.0),
    ("size0", [1, 3, 2], 0, 1e-4, 0.75, 1.0),
    ("n0", [0, 3, 2], 3, 1e-4, 0.75, 1.0),
    ("c0", [2, 0, 2], 3, 1e-4, 0.75, 1.0),
    ("spatial0", [2, 3, 0], 3, 1e-4, 0.75, 1.0),
    ("alpha_nan", [1, 3, 2], 3, NAN, 0.75, 1.0),
    ("beta_inf", [1, 3, 2], 3, 1e-4, INF, 1.0),
    ("k_nan", [1, 3, 2], 3, 1e-4, 0.75, NAN),
    ("k_neg_inf", [1, 3, 2], 3, 1e-4, 0.75, -INF),
]


def lrn_record(name, shape, size, alpha, beta, k, x0, gen):
    x = x0.clone().requires_grad_(True)
    out = F.local_response_norm(x, size, alpha=alpha, beta=beta, k=k)
    g = torch.randn(*out.shape, generator=gen, dtype=torch.float32)
    (out * g).sum().backward()
    return {
        "name": name,
        "shape": shape,
        "size": size,
        "alpha_bits": to_bits([alpha])[0],
        "beta_bits": to_bits([beta])[0],
        "k_bits": to_bits([k])[0],
        "x_bits": flat_bits(x0),
        "g_bits": flat_bits(g),
        "out_bits": flat_bits(out),
        "grad_bits": flat_bits(x.grad),
    }


def lrn_error(name, shape, size, alpha, beta, k):
    x = torch.randn(*shape, dtype=torch.float32)
    rec = {
        "name": name,
        "shape": shape,
        "size": size,
        "alpha_bits": to_bits([alpha])[0],
        "beta_bits": to_bits([beta])[0],
        "k_bits": to_bits([k])[0],
    }
    try:
        out = F.local_response_norm(x, size, alpha=alpha, beta=beta, k=k)
        rec["torch_raises"] = False
        rec["out_shape"] = list(out.shape)
    except Exception as e:  # noqa: BLE001 - 例外の有無を実測して記録する。
        rec["torch_raises"] = True
        rec["out_shape"] = []
        rec["torch_error"] = type(e).__name__
    return rec


# ---------------------------------------------------------------- weight_norm

# (name, v_shape, dim(None=-1), g_kind, kind)。
WEIGHT_NORM_CASES = [
    ("r1_dim0", [5], 0, "own", "randn"),
    ("r1_dim0_scaled_g", [5], 0, "scaled", "randn"),
    ("r2_dim0", [4, 3], 0, "own", "randn"),
    ("r2_dim0_scaled_g", [4, 3], 0, "scaled", "randn"),
    ("r2_dim1", [4, 3], 1, "scaled", "randn"),
    ("r2_none", [4, 3], None, "scaled", "randn"),
    ("r2_none_own", [4, 3], None, "own", "big"),
    ("r4_dim0", [3, 2, 2, 2], 0, "scaled", "randn"),
    ("r4_dim1", [3, 2, 2, 2], 1, "scaled", "randn"),
    ("r4_dim3", [3, 2, 2, 2], 3, "scaled", "big"),
    ("r4_none", [3, 2, 2, 2], None, "scaled", "randn"),
    ("axis_len1_dim0", [1, 4], 0, "scaled", "randn"),
    ("axis_len1_dim1", [3, 1], 1, "scaled", "randn"),
    ("r3_dim1_big", [2, 6, 3], 1, "own", "big"),
    ("r3_dim2_pos_g", [2, 3, 5], 2, "scaled", "pos"),
    ("tiny_g", [3, 4], 0, "tiny", "randn"),
    ("negative_g", [3, 4], 0, "neg", "randn"),
]

# 非有限・ゼロノルム（name, v_shape, dim, injects{pos: value}）。
WEIGHT_NORM_NONFINITE_CASES = [
    ("zero_norm_row", [3, 2], 0, {2: 0.0, 3: 0.0}),
    ("zero_norm_whole", [2, 2], None, {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}),
    ("nan_in_v", [3, 2], 0, {1: NAN}),
    ("inf_in_v", [3, 2], 1, {3: INF}),
]

# (name, v_shape, g_shape, dim)。
WEIGHT_NORM_ERROR_CASES = [
    ("dim_out_of_range", [3, 3], [3, 1], 5),
    ("rank0", [], [], 0),
    ("g_flat_instead_of_keepdim", [3, 4], [3], 0),
    ("g_wrong_len", [3, 4], [2, 1], 0),
    ("g_keepdim_on_wrong_axis", [3, 4], [1, 4], 0),
    ("none_with_keepdim_g", [3, 4], [1, 1], None),
]


def wn_g(v, dim, g_kind, gen):
    d = -1 if dim is None else dim
    own = torch.norm_except_dim(v, 2, d)
    if g_kind == "own":
        return own.clone()
    if g_kind == "scaled":
        return own * 1.7 + 0.3
    if g_kind == "tiny":
        return own * 1e-3
    if g_kind == "neg":
        return -own * 0.8
    raise AssertionError(g_kind)


def wn_record(name, v0, g0, dim, gen):
    d = -1 if dim is None else dim
    v = v0.clone().requires_grad_(True)
    g = g0.clone().requires_grad_(True)
    out = torch._weight_norm(v, g, d)
    up = torch.randn(*out.shape, generator=gen, dtype=torch.float32)
    (out * up).sum().backward()
    norm = torch.norm_except_dim(v0, 2, d)
    return {
        "name": name,
        "v_shape": list(v0.shape),
        "g_shape": list(g0.shape),
        "dim": dim,
        "v_bits": flat_bits(v0),
        "g_bits": flat_bits(g0),
        "up_bits": flat_bits(up),
        "out_bits": flat_bits(out),
        "dv_bits": flat_bits(v.grad),
        "dg_bits": flat_bits(g.grad),
        "norm_shape": list(norm.shape),
        "norm_bits": flat_bits(norm),
    }


def wn_error(name, v_shape, g_shape, dim):
    d = -1 if dim is None else dim
    v = torch.randn(*v_shape, dtype=torch.float32) if v_shape else torch.tensor(1.5)
    g = torch.ones(*g_shape, dtype=torch.float32) if g_shape else torch.tensor(1.0)
    rec = {"name": name, "v_shape": v_shape, "g_shape": g_shape, "dim": dim}
    try:
        out = torch._weight_norm(v, g, d)
        rec["torch_raises"] = False
        rec["out_shape"] = list(out.shape)
    except Exception as e:  # noqa: BLE001
        rec["torch_raises"] = True
        rec["out_shape"] = []
        rec["torch_error"] = type(e).__name__
    return rec


# ---------------------------------------------------------------- spectral_norm

# (name, shape, dim, n_iter, training, eps, w_scale)。
SPECTRAL_CASES = [
    ("r2_dim0_train_n1", [4, 3], 0, 1, True, 1e-12, 1.0),
    ("r2_dim0_train_n3", [4, 3], 0, 3, True, 1e-12, 1.0),
    ("r2_dim0_eval", [4, 3], 0, 1, False, 1e-12, 1.0),
    ("r2_dim1_train_n1", [4, 3], 1, 1, True, 1e-12, 1.0),
    ("r2_dim1_train_n3", [3, 5], 1, 3, True, 1e-12, 1.0),
    ("r2_dim1_eval", [3, 5], 1, 1, False, 1e-12, 1.0),
    ("r2_wide_train", [2, 7], 0, 2, True, 1e-12, 2.0),
    ("r2_tall_train", [7, 2], 0, 2, True, 1e-12, 2.0),
    ("r4_dim0_train_n1", [3, 2, 2, 2], 0, 1, True, 1e-12, 1.0),
    ("r4_dim0_train_n3", [3, 2, 2, 2], 0, 3, True, 1e-12, 1.0),
    ("r4_dim1_train_n2", [3, 4, 2, 2], 1, 2, True, 1e-12, 1.0),
    ("r4_dim3_train_n2", [2, 3, 2, 4], 3, 2, True, 1e-12, 1.0),
    ("r4_dim2_eval", [2, 3, 4, 2], 2, 1, False, 1e-12, 1.0),
    ("r3_dim1_train_n2", [2, 5, 3], 1, 2, True, 1e-12, 1.0),
    ("r2_big_scale", [4, 4], 0, 2, True, 1e-12, 50.0),
    ("r2_small_scale_eps_clamps", [4, 3], 0, 2, True, 1e-2, 1e-4),
    ("r2_1xN_dim0", [1, 5], 0, 1, True, 1e-12, 1.0),
    ("r2_Nx1_dim0", [5, 1], 0, 1, True, 1e-12, 1.0),
]

# (name, shape, dim, n_iter, eps)。
SPECTRAL_ERROR_CASES = [
    ("rank1", [4], 0, 1, 1e-12),
    ("n_iter0", [3, 3], 0, 0, 1e-12),
    ("dim_out_of_range", [3, 3], 2, 1, 1e-12),
    ("zero_elements", [0, 3], 0, 1, 1e-12),
    ("eps_negative", [3, 3], 0, 1, -1e-3),
    ("eps_nan", [3, 3], 0, 1, NAN),
]


def sp_record(name, shape, dim, n_iter, training, eps, w_scale, gen):
    w0 = torch.randn(*shape, generator=gen, dtype=torch.float32) * w_scale
    # 初期化は乱数（グローバル RNG）。値は保存して Rust 側で再現しない。
    sn = _SpectralNorm(w0, n_power_iterations=n_iter, dim=dim, eps=eps)
    u0 = sn._u.clone()
    v0 = sn._v.clone()
    # 初期化に使った重みとは別の重みで forward する（u0／v0 が収束済みでない状態を作る）。
    w1 = (w0 + torch.randn(*shape, generator=gen, dtype=torch.float32) * 0.3 * w_scale).detach()
    w1.requires_grad_(True)
    sn.train(training)
    out = sn(w1)  # forward は 1 回だけ（training では呼ぶたびに反復が走る）。
    up = torch.randn(*out.shape, generator=gen, dtype=torch.float32)
    (out * up).sum().backward()
    return {
        "name": name,
        "shape": shape,
        "dim": dim,
        "n_iter": n_iter,
        "training": training,
        "eps_bits": to_bits([eps])[0],
        "w_bits": flat_bits(w1),
        "u0_bits": flat_bits(u0),
        "v0_bits": flat_bits(v0),
        "u1_bits": flat_bits(sn._u),
        "v1_bits": flat_bits(sn._v),
        "up_bits": flat_bits(up),
        "out_bits": flat_bits(out),
        "grad_bits": flat_bits(w1.grad),
    }


def sp_error(name, shape, dim, n_iter, eps):
    w = torch.randn(*shape, dtype=torch.float32)
    rec = {
        "name": name,
        "shape": shape,
        "dim": dim,
        "n_iter": n_iter,
        "eps_bits": to_bits([eps])[0],
    }
    try:
        sn = _SpectralNorm(w, n_power_iterations=n_iter, dim=dim, eps=eps)
        out = sn(w)
        rec["torch_raises"] = False
        rec["out_shape"] = list(out.shape)
        rec["out_all_finite"] = bool(torch.isfinite(out).all().item()) if out.numel() else True
    except Exception as e:  # noqa: BLE001
        rec["torch_raises"] = True
        rec["out_shape"] = []
        rec["out_all_finite"] = False
        rec["torch_error"] = type(e).__name__
    return rec


def main():
    gen = torch.Generator().manual_seed(SEED)
    torch.manual_seed(SEED)

    lrn = []
    for name, shape, size, a, b, k, kind in LRN_CASES:
        x0 = make_input(shape, kind, gen)
        lrn.append(lrn_record(name, shape, size, a, b, k, x0, gen))
    lrn_nonfinite = []
    for name, shape, size, inject in LRN_NONFINITE_CASES:
        n = numel(shape)
        vals = [0.25 * (i + 1) for i in range(n)]
        for pos, v in inject.items():
            vals[pos] = v
        x0 = torch.tensor(vals, dtype=torch.float32).reshape(shape)
        lrn_nonfinite.append(lrn_record(name, shape, size, 0.5, 0.75, 1.0, x0, gen))
    lrn_errors = [lrn_error(*c) for c in LRN_ERROR_CASES]

    wn = []
    for name, shape, dim, g_kind, kind in WEIGHT_NORM_CASES:
        v0 = make_input(shape, kind, gen)
        g0 = wn_g(v0, dim, g_kind, gen)
        wn.append(wn_record(name, v0, g0, dim, gen))
    wn_nonfinite = []
    for name, shape, dim, inject in WEIGHT_NORM_NONFINITE_CASES:
        n = numel(shape)
        vals = [0.5 + 0.25 * i for i in range(n)]
        for pos, v in inject.items():
            vals[pos] = v
        v0 = torch.tensor(vals, dtype=torch.float32).reshape(shape)
        g0 = torch.ones(*torch.norm_except_dim(v0, 2, -1 if dim is None else dim).shape) \
            if dim is not None else torch.tensor(1.0)
        wn_nonfinite.append(wn_record(name, v0, g0, dim, gen))
    wn_errors = [wn_error(*c) for c in WEIGHT_NORM_ERROR_CASES]

    sp = []
    for name, shape, dim, n_iter, training, eps, scale in SPECTRAL_CASES:
        sp.append(sp_record(name, shape, dim, n_iter, training, eps, scale, gen))
    sp_errors = [sp_error(*c) for c in SPECTRAL_ERROR_CASES]

    doc = {
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "seed": SEED,
        "lrn_cases": lrn,
        "lrn_nonfinite_cases": lrn_nonfinite,
        "lrn_error_cases": lrn_errors,
        "weight_norm_cases": wn,
        "weight_norm_nonfinite_cases": wn_nonfinite,
        "weight_norm_error_cases": wn_errors,
        "spectral_cases": sp,
        "spectral_error_cases": sp_errors,
    }
    json.dump(doc, sys.stdout, ensure_ascii=False, separators=(",", ":"))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
