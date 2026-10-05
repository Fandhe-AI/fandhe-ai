#!/usr/bin/env python3
"""histc／bincount／searchsorted／bucketize の PyTorch 参照値を生成する。

イシュー #2638（親 #2625）の `tests/binning_parity.rs` が参照する固定フィクスチャ
`binning_reference.json` の生成スクリプト。CI は Python／PyTorch に依存せず、
コミット済み JSON のみを読む（`README.md` 参照）。

f32 はすべて u32 ビットパターン配列で保存する（JSON は NaN／inf を運べないため）。
索引・カウント（torch の int64）は整数配列で保存する。例外になる呼び出しは
`torch_raises: true` と例外文面を実測として記録する。dtype は float32。
"""

import json
import platform
import struct
import sys

import torch

EXPECTED_TORCH_PREFIX = "2.14.0"
assert torch.__version__.startswith(EXPECTED_TORCH_PREFIX), torch.__version__

SEED = 2638
NAN = float("nan")
INF = float("inf")


def f32(v):
    return struct.unpack("<f", struct.pack("<f", float(v)))[0]


def to_bits(values):
    return [struct.unpack("<I", struct.pack("<f", float(v)))[0] for v in values]


def flat_bits(t):
    return to_bits(t.detach().contiguous().reshape(-1).tolist())


def ft(vals, shape=None):
    t = torch.tensor(vals, dtype=torch.float32)
    return t if shape is None else t.reshape(shape)


def call(fn):
    try:
        return fn(), None
    except Exception as e:  # noqa: BLE001 - 例外型・文面を実測として記録する
        return None, f"{type(e).__name__}: {e}"


def histc_case(name, x, bins, mn, mx):
    out, err = call(lambda: torch.histc(x, bins=bins, min=mn, max=mx))
    rec = {
        "name": name,
        "shape": list(x.shape),
        "x_bits": flat_bits(x),
        "bins": bins,
        "min_bits": to_bits([mn])[0],
        "max_bits": to_bits([mx])[0],
        "torch_raises": err is not None,
        "message": err or "",
    }
    if out is not None:
        rec["out_bits"] = flat_bits(out)
    return rec


def bincount_case(name, inp, minlength, weights=None):
    out, err = call(lambda: torch.bincount(inp, weights=weights, minlength=minlength))
    rec = {
        "name": name,
        "shape": list(inp.shape),
        "input": inp.reshape(-1).tolist(),
        "minlength": minlength,
        "weights_bits": None if weights is None else flat_bits(weights),
        "torch_raises": err is not None,
        "message": err or "",
    }
    if out is not None:
        rec["out_dtype"] = str(out.dtype)
        if weights is None:
            rec["out"] = out.tolist()
        else:
            rec["out_bits"] = flat_bits(out)
    return rec


def ss_case(name, seq, values, right, bucket=False):
    if bucket:
        out, err = call(lambda: torch.bucketize(values, seq, right=right))
    else:
        out, err = call(lambda: torch.searchsorted(seq, values, right=right))
    rec = {
        "name": name,
        "seq_shape": list(seq.shape),
        "seq_bits": flat_bits(seq),
        "values_shape": list(values.shape),
        "values_bits": flat_bits(values),
        "right": right,
        "torch_raises": err is not None,
        "message": err or "",
    }
    if out is not None:
        rec["out_shape"] = list(out.shape)
        rec["out"] = out.reshape(-1).tolist()
        rec["out_dtype"] = str(out.dtype)
    return rec


def differentiability():
    rows = []
    x = ft([1.0, 2.0, 2.5, 3.0]).requires_grad_(True)
    b = ft([1.0, 2.0, 3.0]).requires_grad_(True)
    w = ft([1.0, 2.0, 3.0]).requires_grad_(True)
    idx = torch.tensor([0, 1, 1])

    def probe(name, fn):
        out = fn()
        row = {
            "op": name,
            "out_dtype": str(out.dtype),
            "requires_grad": bool(out.requires_grad),
            "grad_fn": None if out.grad_fn is None else type(out.grad_fn).__name__,
            "backward_raises": False,
            "backward_message": "",
        }
        if out.requires_grad:
            try:
                out.to(torch.float32).sum().backward()
            except Exception as e:  # noqa: BLE001
                row["backward_raises"] = True
                row["backward_message"] = f"{type(e).__name__}: {e}"
        rows.append(row)

    probe("histc", lambda: torch.histc(x, bins=3))
    probe("bincount_weights", lambda: torch.bincount(idx, weights=w, minlength=4))
    probe("searchsorted_seq", lambda: torch.searchsorted(b, x))
    probe("searchsorted_values", lambda: torch.searchsorted(b.detach(), x))
    probe("bucketize_input", lambda: torch.bucketize(x, b))
    return rows


def main():
    gen = torch.Generator().manual_seed(SEED)
    rnd = lambda *s: torch.randn(*s, generator=gen, dtype=torch.float32)  # noqa: E731

    histc = []
    h = lambda *a: histc.append(histc_case(*a))  # noqa: E731
    x8 = ft([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    for bins in (1, 2, 4, 5, 8):
        h(f"grid_bins{bins}", x8, bins, 0.0, 4.0)
    h("default_range_bins4", ft([1.0, 2.0, 3.0, 4.0, 2.0, 2.5]), 4, 0.0, 0.0)
    h("default_range_neg", ft([-3.0, -1.0, 0.0, 5.0]), 5, 0.0, 0.0)
    h("empty_default", ft([]), 4, 0.0, 0.0)
    h("empty_explicit", ft([]), 4, -1.0, 1.0)
    h("all_equal_default", ft([1.0, 1.0, 1.0]), 4, 0.0, 0.0)
    h("all_equal_zero_default", ft([0.0, 0.0]), 4, 0.0, 0.0)
    h("min_eq_max_nonzero", ft([2.0, 2.0]), 4, 2.0, 2.0)
    h("min_eq_max_uses_data", ft([0.0, 1.0, 0.5]), 4, 7.0, 7.0)
    h("explicit_range_out_of_range", ft([-5.0, -1.0, 0.0, 0.5, 1.0, 2.0, 9.0]), 4, -1.0, 1.0)
    h("x_eq_max_last_bin", ft([1.0, 1.0, 0.0]), 3, 0.0, 1.0)
    h("signed_zero", ft([0.0, -0.0, -0.0]), 2, -1.0, 1.0)
    h("scalar_input", ft(1.5), 3, 0.0, 3.0)
    h("multi_dim_flatten", ft([0.1, 0.9, 0.5, 0.3, 0.7, 0.2], [2, 3]), 5, 0.0, 1.0)
    h("range_overflow", ft([-3e38, 3e38]), 3, -3e38, 3e38)
    h("nan_explicit_range", ft([0.0, NAN, 2.0]), 4, 0.0, 2.0)
    h("inf_explicit_range", ft([0.0, INF, -INF, 2.0]), 4, 0.0, 2.0)
    h("nan_only_explicit", ft([NAN, NAN]), 4, 0.0, 1.0)
    for name, n, bins, mn, mx in (
        ("rand_b7_default", 3000, 7, 0.0, 0.0),
        ("rand_b100", 5000, 100, -2.0, 2.0),
        ("rand_b33", 4000, 33, -3.0, 3.0),
        ("rand_b1000", 20000, 1000, -3.5, 3.5),
        ("rand_b10_narrow", 3000, 10, -0.3, 0.7),
        ("rand_b64_default", 5000, 64, 0.0, 0.0),
    ):
        h(name, rnd(n), bins, mn, mx)
    # ビン境界判別: torch.linspace の実エッジ（f32）・その直上／直下の値を入力にする。
    for bins, mn, mx in ((3, 0.0, 1.0), (10, -1.0, 2.0), (20, 0.1, 0.7), (100, -2.0, 2.0),
                         (100, 0.0, 1.0), (257, -1.5, 3.25)):
        edges = torch.linspace(mn, mx, bins + 1, dtype=torch.float32)
        up = torch.nextafter(edges, torch.tensor(INF))
        dn = torch.nextafter(edges, torch.tensor(-INF))
        h(f"edges_b{bins}_{mn}_{mx}", torch.cat([edges, up, dn]), bins, mn, mx)
    # 倍精度演算では境界要素のビンがずれる入力（0.1 刻みの格子 × 小数の範囲）。
    grid = torch.arange(0, 101, dtype=torch.float32) / 10.0
    h("decimal_grid_b10", grid, 10, 0.0, 10.0)
    h("decimal_grid_b7", grid, 7, 0.0, 10.0)
    h("decimal_grid_b30", grid, 30, 0.0, 10.0)
    histc_errors = []
    e = lambda *a: histc_errors.append(histc_case(*a))  # noqa: E731
    e("min_gt_max", ft([0.0, 1.0, 2.0]), 4, 3.0, 1.0)
    e("bins_zero", ft([0.0, 1.0]), 0, 0.0, 1.0)
    e("nan_default_range", ft([0.0, NAN, 2.0]), 4, 0.0, 0.0)
    e("inf_default_range", ft([0.0, INF, 2.0]), 4, 0.0, 0.0)
    e("explicit_inf_max", ft([0.0, 1.0]), 4, 0.0, INF)
    e("explicit_nan_min", ft([0.0, 1.0]), 4, NAN, 1.0)

    bincount = []
    b = lambda *a, **k: bincount.append(bincount_case(*a, **k))  # noqa: E731
    b("basic", torch.tensor([0, 1, 1, 3, 3, 3]), 0)
    b("minlength_larger", torch.tensor([0, 1, 1]), 6)
    b("minlength_smaller", torch.tensor([0, 1, 4, 4]), 2)
    b("empty_minlength0", torch.tensor([], dtype=torch.int64), 0)
    b("empty_minlength3", torch.tensor([], dtype=torch.int64), 3)
    b("single_zero", torch.tensor([0]), 0)
    b("large_index", torch.tensor([0, 1000, 1000]), 0)
    b("w_basic", torch.tensor([0, 1, 1, 3]), 0, ft([1.0, 2.0, 3.0, 4.0]))
    b("w_signed", torch.tensor([0, 1, 1, 0, 2]), 5, ft([1.5, -2.0, 3.0, -0.5, 0.25]))
    b("w_cancel", torch.tensor([0, 0, 0]), 0, ft([1e8, 1.0, -1e8]))
    b("w_nonfinite", torch.tensor([0, 1, 1, 2]), 0, ft([INF, NAN, 1.0, -INF]))
    b("w_inf_cancel", torch.tensor([0, 0]), 0, ft([INF, -INF]))
    b("w_empty", torch.tensor([], dtype=torch.int64), 2, ft([]))
    ri = torch.randint(0, 40, (2000,), generator=gen)
    b("w_random", ri, 0, rnd(2000))
    b("random", ri, 50)
    bincount_errors = []
    be = lambda *a, **k: bincount_errors.append(bincount_case(*a, **k))  # noqa: E731
    be("negative", torch.tensor([0, -1, 2]), 0)
    be("rank2", torch.tensor([[0, 1], [1, 2]]), 0)
    be("rank0", torch.tensor(1), 0)
    be("weights_len_mismatch", torch.tensor([0, 1, 2]), 0, ft([1.0, 2.0]))
    b("weights_ignored_when_input_empty", torch.tensor([], dtype=torch.int64), 0, ft([1.0]))

    sorted_seq = ft([1.0, 2.0, 2.0, 2.0, 5.0, 7.0])
    vals = ft([0.0, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 8.0])
    searchsorted = []
    s = lambda *a, **k: searchsorted.append(ss_case(*a, **k))  # noqa: E731
    for right in (False, True):
        r = "right" if right else "left"
        s(f"dup_{r}", sorted_seq, vals, right)
        s(f"empty_seq_{r}", ft([]), ft([1.0, 2.0]), right)
        s(f"empty_values_{r}", sorted_seq, ft([]), right)
        s(f"scalar_values_{r}", sorted_seq, ft(2.0), right)
        s(f"values_2d_{r}", sorted_seq, vals.reshape(2, 4), right)
        s(f"single_{r}", ft([3.0]), ft([2.0, 3.0, 4.0]), right)
        s(f"batched_{r}", ft([1.0, 3.0, 5.0, 2.0, 2.0, 8.0], [2, 3]),
          ft([2.0, 3.0, 0.0, 2.0, 9.0, 5.0, 2.0, 1.0], [2, 4]), right)
        s(f"batched3d_{r}", ft([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [2, 2, 2]),
          ft([2.0, 5.0, 9.0, 0.0, 6.0, 8.0, 1.0, 3.0, 4.0, 7.0, 8.5, 2.5], [2, 2, 3]), right)
        s(f"nan_values_{r}", sorted_seq, ft([NAN, 2.0]), right)
        s(f"nan_in_seq_{r}", ft([1.0, NAN, 3.0]), ft([NAN, 2.0, 5.0, 0.0]), right)
        s(f"nan_tail_seq_{r}", ft([1.0, 2.0, NAN, NAN]), ft([NAN, 2.0, 5.0, 0.0]), right)
        s(f"inf_{r}", ft([-INF, 0.0, INF]), ft([-INF, -1.0, 0.0, INF, 1.0]), right)
        s(f"signed_zero_{r}", ft([-0.0, 0.0, 1.0]), ft([0.0, -0.0]), right)
        s(f"unsorted_{r}", ft([3.0, 1.0, 2.0, 0.0, 5.0]), ft([0.0, 1.0, 2.5, 3.0, 6.0]), right)
        big = torch.sort(rnd(1000)).values
        s(f"random_{r}", big, rnd(500), right)
        s(f"random_exact_{r}", big, big[::7].clone(), right)
        s(f"batched_random_{r}", torch.sort(rnd(4, 50), dim=1).values.contiguous(), rnd(4, 20), right)
    searchsorted_errors = []
    se = lambda *a, **k: searchsorted_errors.append(ss_case(*a, **k))  # noqa: E731
    se("seq_rank0", ft(1.0), ft([2.0]), False)
    se("batched_first_dim_mismatch", ft([1.0, 2.0, 3.0, 4.0], [2, 2]), ft([1.0, 2.0, 3.0], [3, 1]), False)
    se("batched_values_rank_mismatch", ft([1.0, 2.0, 3.0, 4.0], [2, 2]), ft([1.0, 2.0]), False)
    se("batched_scalar_values", ft([1.0, 2.0, 3.0, 4.0], [2, 2]), ft(1.0), False)

    bucketize = []
    k = lambda *a, **kw: bucketize.append(ss_case(*a, bucket=True, **kw))  # noqa: E731
    bounds = ft([1.0, 3.0, 5.0, 7.0])
    for right in (False, True):
        r = "right" if right else "left"
        k(f"basic_{r}", bounds, ft([0.0, 1.0, 2.0, 3.0, 4.0, 7.0, 8.0], [7]), right)
        k(f"input_2d_{r}", bounds, ft([0.0, 1.0, 2.0, 3.0, 4.0, 7.0], [2, 3]), right)
        k(f"input_scalar_{r}", bounds, ft(3.0), right)
        k(f"empty_boundaries_{r}", ft([]), ft([1.0, 2.0]), right)
        k(f"nan_input_{r}", bounds, ft([NAN, 3.0]), right)
        k(f"dup_boundaries_{r}", ft([1.0, 2.0, 2.0, 3.0]), ft([2.0, 0.0, 4.0]), right)
        k(f"random_{r}", torch.sort(rnd(200)).values, rnd(3, 40), right)
    bucketize_errors = []
    ke = lambda *a, **kw: bucketize_errors.append(ss_case(*a, bucket=True, **kw))  # noqa: E731
    ke("boundaries_2d", ft([1.0, 2.0, 3.0, 4.0], [2, 2]), ft([1.0, 2.0]), False)
    ke("boundaries_rank0", ft(1.0), ft([1.0, 2.0]), False)

    doc = {
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "seed": SEED,
        "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "differentiability": differentiability(),
        "histc_cases": histc,
        "histc_errors": histc_errors,
        "bincount_cases": bincount,
        "bincount_errors": bincount_errors,
        "searchsorted_cases": searchsorted,
        "searchsorted_errors": searchsorted_errors,
        "bucketize_cases": bucketize,
        "bucketize_errors": bucketize_errors,
    }
    json.dump(doc, sys.stdout, ensure_ascii=False, separators=(",", ":"))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
