#!/usr/bin/env python3
"""stft／istft の PyTorch 参照値（forward 出力と入力勾配）を生成する。

イシュー #2633（親 #2630）の `tests/stft_parity.rs` が参照する固定フィクスチャ
`stft_reference.json` の生成スクリプト。CI は Python／PyTorch に依存せず、
コミット済み JSON のみを読む（`README.md` 参照）。既存の `gen_reference.py`／
`fft_reference.json`（rfft／irfft／fft／ifft の 69 件）は変更しない。

複素 autograd の規約差を避けるため、複素数は常に `torch.view_as_real`／
`torch.view_as_complex` で末尾次元 2 の実テンソル対として扱い、損失は実数
`(out * g).sum()`（`g` は固定シードの実テンソル）にする。

- stft: 実の葉 `x`（`[L]`／`[B, L]`）→ `y = view_as_real(torch.stft(x, ...,
  return_complex=True))`・`x.grad`。
- istft: 実の葉 `xr`（`[N, T, 2]`／`[B, N, T, 2]`）→ `out = torch.istft(
  view_as_complex(xr), ...)`・`xr.grad`。

窓は `requires_grad=False`（窓への勾配は Rust 側でも流さない）。入力・窓・上流
勾配・出力・入力勾配・shape・全引数を JSON に保存し、Rust 側で再生成しない。
dtype は float32。NaN を含まないことを生成時に assert する。
"""

import json
import platform
import sys
import warnings

import torch

warnings.filterwarnings("ignore")

EXPECTED_TORCH_PREFIX = "2.14.0"
assert torch.__version__.startswith(EXPECTED_TORCH_PREFIX), torch.__version__

SEED = 20261005


def hann(n):
    return torch.hann_window(n, periodic=True, dtype=torch.float32)


def pos_window(gen, n):
    """端が 0 でない正の窓（center=False の istft でも NOLA を満たす）。"""
    return 0.5 + 0.5 * torch.rand(n, generator=gen, dtype=torch.float32)


# stft ケース: name, in_shape, n_fft, hop, win_length, window 種別, center,
# pad_mode, normalized, onesided
# window 種別: None（矩形）／"hann"／"rand"（長さ win_length の乱数窓）
STFT_CASES = [
    ("default_rect", [32], 8, None, None, None, True, "reflect", False, True),
    ("hann", [32], 8, None, None, "hann", True, "reflect", False, True),
    ("hop_non_divisor", [33], 8, 3, None, "hann", True, "reflect", False, True),
    ("odd_n_fft", [30], 7, 2, None, "rand", True, "reflect", False, True),
    ("odd_n_fft_hop3", [31], 5, 3, None, None, True, "reflect", False, True),
    ("win_length_lt_n_fft", [32], 8, 2, 6, "hann", True, "reflect", False, True),
    ("win_length_odd_diff", [32], 8, 2, 5, "rand", True, "reflect", False, True),
    ("center_false", [32], 8, 2, None, "hann", False, "reflect", False, True),
    ("pad_constant", [32], 8, 2, None, "hann", True, "constant", False, True),
    ("pad_constant_short", [3], 8, 2, None, None, True, "constant", False, True),
    ("normalized", [32], 8, 2, None, "hann", True, "reflect", True, True),
    ("onesided_false", [32], 8, 2, None, "hann", True, "reflect", False, False),
    ("onesided_false_odd", [29], 7, 2, None, "rand", True, "reflect", True, False),
    ("batched", [2, 32], 8, 2, None, "hann", True, "reflect", False, True),
    ("batched_center_false", [3, 24], 6, 3, None, "rand", False, "reflect", True, True),
    ("single_frame", [8], 8, 8, None, None, False, "reflect", False, True),
    ("small_n_fft4", [16], 4, None, None, "hann", True, "reflect", False, True),
    ("n_fft2_hop1", [10], 2, 1, None, None, True, "reflect", False, True),
    ("n_fft1_hop1", [6], 1, 1, None, None, True, "reflect", False, True),
    ("n_fft_gt_L_center", [10], 16, 4, None, "hann", True, "constant", False, True),
    ("reflect_min_L", [5], 8, 2, None, "hann", True, "reflect", False, True),
]

# istft ケース: name, in_shape(末尾 2 を含む), n_fft, hop, win_length, window 種別,
# center, normalized, onesided(None|bool), length
ISTFT_CASES = [
    ("hann_hop_quarter", [5, 17, 2], 8, None, None, "hann", True, False, None, None),
    ("center_false_pos_window", [5, 15, 2], 8, 2, None, "pos", False, False, None, None),
    ("normalized", [5, 17, 2], 8, 2, None, "hann", True, True, None, None),
    ("onesided_false_non_hermitian", [8, 17, 2], 8, 2, None, "hann", True, False, False, None),
    ("onesided_false_odd", [7, 12, 2], 7, 2, None, "pos", False, True, False, None),
    ("onesided_none_infer_full", [6, 9, 2], 6, 3, None, "pos", False, False, None, None),
    ("length_short", [5, 17, 2], 8, 2, None, "hann", True, False, None, 20),
    ("length_long", [5, 17, 2], 8, 2, None, "hann", True, False, None, 40),
    ("length_long_center_false", [5, 6, 2], 8, 2, None, "pos", False, False, None, 30),
    ("batched", [2, 5, 17, 2], 8, 2, None, "hann", True, False, None, None),
    ("batched_onesided_false", [2, 8, 10, 2], 8, 4, None, "pos", False, True, False, None),
    ("win_length_lt_n_fft", [5, 17, 2], 8, 2, 6, "hann6", True, False, None, None),
    ("win_length_rand", [5, 12, 2], 8, 3, 5, "rand5", True, False, None, None),
    ("odd_n_fft", [4, 11, 2], 6 + 1, 2, None, "pos", True, False, None, None),
    ("single_frame_rect", [5, 1, 2], 8, 8, None, None, False, False, None, None),
    ("hop_eq_win_rect", [5, 4, 2], 8, 8, None, None, False, False, None, None),
    ("small_n_fft4", [3, 9, 2], 4, None, None, "hann", True, False, None, None),
    ("n_fft2", [2, 7, 2], 2, 1, None, "pos", True, False, None, None),
    ("n_fft1_center_false", [1, 7, 2], 1, 1, None, None, False, False, None, None),
    ("n_fft1_length", [1, 7, 2], 1, 1, None, None, True, False, None, 5),
]


def make_window(kind, gen, n_fft, win_length):
    wl = win_length if win_length is not None else n_fft
    if kind is None:
        return None
    if kind == "hann":
        return hann(wl)
    if kind == "hann6":
        return hann(6)
    if kind == "pos":
        return pos_window(gen, wl)
    if kind == "rand":
        return torch.randn(wl, generator=gen, dtype=torch.float32)
    if kind == "rand5":
        return torch.randn(5, generator=gen, dtype=torch.float32)
    raise AssertionError(kind)


def rnd(shape, gen):
    return torch.randn(*shape, generator=gen, dtype=torch.float32)


def flat(t):
    out = t.detach().contiguous().reshape(-1).tolist()
    assert all(v == v for v in out), "NaN を含む"
    return out


def stft_kwargs(n_fft, hop, wl, window, center, pad_mode, normalized, onesided):
    kw = dict(
        n_fft=n_fft,
        center=center,
        pad_mode=pad_mode,
        normalized=normalized,
        onesided=onesided,
        return_complex=True,
    )
    if hop is not None:
        kw["hop_length"] = hop
    if wl is not None:
        kw["win_length"] = wl
    if window is not None:
        kw["window"] = window
    return kw


def istft_kwargs(n_fft, hop, wl, window, center, normalized, onesided, length):
    kw = dict(n_fft=n_fft, center=center, normalized=normalized)
    if hop is not None:
        kw["hop_length"] = hop
    if wl is not None:
        kw["win_length"] = wl
    if window is not None:
        kw["window"] = window
    if onesided is not None:
        kw["onesided"] = onesided
    if length is not None:
        kw["length"] = length
    return kw


def make_stft(gen, case):
    name, shape, n_fft, hop, wl, wkind, center, pad_mode, normalized, onesided = case
    window = make_window(wkind, gen, n_fft, wl)
    x = rnd(shape, gen).requires_grad_(True)
    y = torch.view_as_real(
        torch.stft(x, **stft_kwargs(n_fft, hop, wl, window, center, pad_mode, normalized, onesided))
    )
    g = rnd(list(y.shape), gen)
    (y * g).sum().backward()
    return {
        "name": "stft_" + name,
        "op": "stft",
        "in_shape": shape,
        "n_fft": n_fft,
        "hop": hop,
        "win_length": wl,
        "window": None if window is None else flat(window),
        "center": center,
        "pad_mode": pad_mode,
        "normalized": normalized,
        "onesided": onesided,
        "length": None,
        "input": flat(x),
        "grad_out": flat(g),
        "out_shape": list(y.shape),
        "output": flat(y),
        "grad_in": flat(x.grad),
    }


def make_istft(gen, case):
    name, shape, n_fft, hop, wl, wkind, center, normalized, onesided, length = case
    window = make_window(wkind, gen, n_fft, wl)
    xr = rnd(shape, gen).requires_grad_(True)
    out = torch.istft(
        torch.view_as_complex(xr),
        **istft_kwargs(n_fft, hop, wl, window, center, normalized, onesided, length),
    )
    g = rnd(list(out.shape), gen)
    (out * g).sum().backward()
    return {
        "name": "istft_" + name,
        "op": "istft",
        "in_shape": shape,
        "n_fft": n_fft,
        "hop": hop,
        "win_length": wl,
        "window": None if window is None else flat(window),
        "center": center,
        "pad_mode": "reflect",
        "normalized": normalized,
        "onesided": onesided,
        "length": length,
        "input": flat(xr),
        "grad_out": flat(g),
        "out_shape": list(out.shape),
        "output": flat(out),
        "grad_in": flat(xr.grad),
    }


def make_roundtrip(gen):
    """torch の stft 出力をそのまま istft へ入れる往復ケース（入力は stft 出力）。"""
    n_fft, hop, length = 8, 2, 32
    window = hann(n_fft)
    sig = rnd([length], gen)
    spec = torch.view_as_real(
        torch.stft(sig, n_fft, hop_length=hop, window=window, return_complex=True)
    )
    xr = spec.detach().clone().requires_grad_(True)
    out = torch.istft(
        torch.view_as_complex(xr), n_fft, hop_length=hop, window=window, length=length
    )
    g = rnd(list(out.shape), gen)
    (out * g).sum().backward()
    return {
        "name": "istft_roundtrip_from_stft",
        "op": "istft",
        "in_shape": list(xr.shape),
        "n_fft": n_fft,
        "hop": hop,
        "win_length": None,
        "window": flat(window),
        "center": True,
        "pad_mode": "reflect",
        "normalized": False,
        "onesided": None,
        "length": length,
        "input": flat(xr),
        "grad_out": flat(g),
        "out_shape": list(out.shape),
        "output": flat(out),
        "grad_in": flat(xr.grad),
    }


def raises(fn):
    try:
        fn()
    except Exception:
        return True
    return False


def rect_boundary_window(w0):
    w = torch.ones(8, dtype=torch.float32)
    w[0] = w0
    return w


# error_case: dict。op は "stft"／"istft"。window は値のリスト（None は省略）。
def stft_err(name, shape, n_fft, hop=None, wl=None, window=None, center=True,
             pad_mode="reflect", onesided=True, empty=False):
    return dict(name=name, op="stft", in_shape=shape, n_fft=n_fft, hop=hop,
                win_length=wl, window=window, center=center, pad_mode=pad_mode,
                normalized=False, onesided=onesided, length=None)


def istft_err(name, shape, n_fft, hop=None, wl=None, window=None, center=True,
              onesided=None, length=None):
    return dict(name=name, op="istft", in_shape=shape, n_fft=n_fft, hop=hop,
                win_length=wl, window=window, center=center, pad_mode="reflect",
                normalized=False, onesided=onesided, length=length)


ERROR_CASES = [
    stft_err("stft_n_fft0", [32], 0),
    stft_err("stft_n_fft_gt_L_no_center", [32], 40, center=False),
    stft_err("stft_n_fft_gt_L_center_ok", [32], 40),
    stft_err("stft_hop0", [32], 8, hop=0),
    stft_err("stft_n_fft3_default_hop", [32], 3),
    stft_err("stft_n_fft1_default_hop", [32], 1),
    stft_err("stft_win_length0", [32], 8, wl=0),
    stft_err("stft_win_length_gt_n_fft", [32], 8, wl=9),
    stft_err("stft_window_len_mismatch", [32], 8, wl=8, window=[1.0] * 7),
    stft_err("stft_window_len_vs_default_wl", [32], 8, window=[1.0] * 6),
    stft_err("stft_window_gt_n_fft", [32], 8, window=[1.0] * 9),
    stft_err("stft_reflect_pad_eq_L", [4], 8),
    stft_err("stft_reflect_pad_lt_L_ok", [5], 8),
    stft_err("stft_constant_short_ok", [3], 8, pad_mode="constant"),
    stft_err("stft_rank0", [], 8),
    stft_err("stft_rank3", [1, 2, 32], 8),
    stft_err("stft_empty_L", [0], 8),
    stft_err("stft_empty_batch", [0, 32], 8),
    stft_err("stft_hop_gt_win_ok", [32], 8, hop=9),
    istft_err("istft_hop_gt_win", [5, 17, 2], 8, hop=9, window=[1.0] * 8),
    istft_err("istft_hop0", [5, 17, 2], 8, hop=0, window=[1.0] * 8),
    istft_err("istft_bins_mismatch", [3, 17, 2], 8, window=[1.0] * 8),
    istft_err("istft_onesided_true_full_bins", [8, 17, 2], 8, onesided=True, window=[1.0] * 8),
    istft_err("istft_onesided_false_half_bins", [5, 17, 2], 8, onesided=False, window=[1.0] * 8),
    istft_err("istft_T0", [5, 0, 2], 8, window=[1.0] * 8),
    istft_err("istft_length0", [5, 17, 2], 8, window=[1.0] * 8, length=0),
    istft_err("istft_rank5", [1, 1, 5, 17, 2], 8, window=[1.0] * 8),
    istft_err("istft_rank2", [17, 2], 8, window=[1.0] * 8),
    istft_err("istft_empty_batch", [0, 5, 17, 2], 8, window=[1.0] * 8),
    istft_err("istft_window_len_mismatch", [5, 17, 2], 8, wl=8, window=[1.0] * 7),
    istft_err("istft_n_fft0", [5, 17, 2], 0),
    istft_err("istft_nola_hann_no_center", [5, 17, 2], 8, hop=2, center=False,
              window=hann(8).tolist()),
    istft_err("istft_nola_rect_hop_eq_n_ok", [5, 3, 2], 8, hop=8, center=False,
              window=[1.0] * 8),
    istft_err("istft_nola_boundary_below", [5, 3, 2], 8, hop=8, center=False,
              window=rect_boundary_window(3.1e-6).tolist()),
    istft_err("istft_nola_boundary_above", [5, 3, 2], 8, hop=8, center=False,
              window=rect_boundary_window(3.2e-6).tolist()),
    istft_err("istft_n_fft1_center_no_length", [1, 7, 2], 1, hop=1, window=[1.0]),
    istft_err("istft_T1_center_even_empty_out", [5, 1, 2], 8, hop=8, window=[1.0] * 8),
]


def run_error_case(c):
    wl = c["win_length"]
    window = None if c["window"] is None else torch.tensor(c["window"], dtype=torch.float32)
    shape = c["in_shape"]
    if c["op"] == "stft":
        x = torch.randn(*shape, dtype=torch.float32) if shape else torch.tensor(1.0)
        kw = stft_kwargs(c["n_fft"], c["hop"], wl, window, c["center"], c["pad_mode"],
                         c["normalized"], c["onesided"])
        c["torch_raises"] = raises(lambda: torch.stft(x, **kw))
    else:
        xr = torch.randn(*shape, dtype=torch.float32)
        kw = istft_kwargs(c["n_fft"], c["hop"], wl, window, c["center"], c["normalized"],
                          c["onesided"], c["length"])
        c["torch_raises"] = raises(lambda: torch.istft(torch.view_as_complex(xr), **kw))
    return c


def main():
    gen = torch.Generator().manual_seed(SEED)
    cases = [make_stft(gen, c) for c in STFT_CASES]
    cases += [make_istft(gen, c) for c in ISTFT_CASES]
    cases.append(make_roundtrip(gen))
    errors = [run_error_case(c) for c in ERROR_CASES]
    json.dump(
        {
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "seed": SEED,
            "cases": cases,
            "error_cases": errors,
        },
        sys.stdout,
        separators=(",", ":"),
    )
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
