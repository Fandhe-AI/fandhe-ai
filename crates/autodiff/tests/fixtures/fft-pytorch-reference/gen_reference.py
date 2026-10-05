#!/usr/bin/env python3
"""rfft／irfft／fft／ifft の PyTorch 参照値（forward 出力と入力勾配）を生成する。

イシュー #2631・#2632（親 #2630）の `tests/fft_parity.rs` が参照する固定
フィクスチャ `fft_reference.json` の生成スクリプト。CI は Python／PyTorch
に依存せず、コミット済み JSON のみを読む（`README.md` 参照）。

複素 autograd の規約差を避けるため、複素数は常に `torch.view_as_real`／
`torch.view_as_complex` で末尾次元 2 の実テンソル対として扱い、損失は
実数 `(out * g).sum()`（`g` は固定シードの実テンソル）にする。

- rfft: 実の葉 `x` → `y = view_as_real(rfft(x, n, dim, norm))`・`x.grad`。
- irfft: 実の葉 `xr`（`[..., m, 2]`）→ `out = irfft(view_as_complex(xr),
  n, dim, norm)`・`xr.grad`。

- fft／ifft（#2632）: 実の葉 `x`（`[..., L, ..., 2]`）→ `y = view_as_real(
  torch.fft.{fft,ifft}(view_as_complex(x), n, dim, norm))`・`x.grad`。
  複素軸の添字は Rust 側の実軸の添字と同じ（`dim` は非負）。

入力・上流勾配・出力・勾配・引数はすべて JSON に保存し、Rust 側で入力を
再生成しない。dtype は float32。NaN を含まないことを生成時に assert する。
"""

import json
import platform
import sys

import torch

EXPECTED_TORCH_PREFIX = "2.14.0"
assert torch.__version__.startswith(EXPECTED_TORCH_PREFIX), torch.__version__

# (name, 入力形状, n, dim, norm)。dim は非負（Rust 側と同じ添字）。None は既定。
RFFT_CASES = [
    ("rfft_L1", [1], None, None, "backward"),
    ("rfft_L2", [2], None, None, "backward"),
    ("rfft_L3", [3], None, None, "backward"),
    ("rfft_L4", [4], None, None, "backward"),
    ("rfft_L5", [5], None, None, "backward"),
    ("rfft_L8", [8], None, None, "backward"),
    ("rfft_L5_ortho", [5], None, None, "ortho"),
    ("rfft_L5_forward", [5], None, None, "forward"),
    ("rfft_L8_ortho", [8], None, None, "ortho"),
    ("rfft_L8_forward", [8], None, None, "forward"),
    ("rfft_pad_6_to_8", [6], 8, None, "backward"),
    ("rfft_trunc_8_to_5", [8], 5, None, "backward"),
    ("rfft_trunc_5_to_2", [5], 2, None, "ortho"),
    ("rfft_pad_3_to_7", [3], 7, None, "forward"),
    ("rfft_batched_dim1", [2, 5, 3], None, 1, "backward"),
    ("rfft_batched_dim0_pad", [2, 5, 3], 4, 0, "ortho"),
    ("rfft_batched_default_dim", [3, 6], None, None, "backward"),
    ("rfft_rank3_dim2_forward", [2, 3, 4], None, 2, "forward"),
]

# (name, 入力形状〔末尾が 2〕, n, dim, norm)。
IRFFT_CASES = [
    ("irfft_m3_default_n", [3, 2], None, None, "backward"),
    ("irfft_m5_default_n", [5, 2], None, None, "backward"),
    ("irfft_m5_ortho", [5, 2], None, None, "ortho"),
    ("irfft_m5_forward", [5, 2], None, None, "forward"),
    ("irfft_m4_odd_n5_trunc", [4, 2], 5, None, "backward"),
    ("irfft_m3_n7_pad", [3, 2], 7, None, "backward"),
    ("irfft_m2_n2", [2, 2], 2, None, "backward"),
    ("irfft_m1_n1", [1, 2], 1, None, "backward"),
    ("irfft_m1_n2", [1, 2], 2, None, "ortho"),
    ("irfft_m1_n3", [1, 2], 3, None, "forward"),
    ("irfft_m4_n4_trunc", [4, 2], 4, None, "backward"),
    ("irfft_m3_n6", [3, 2], 6, None, "ortho"),
    ("irfft_batched", [2, 4, 2], None, None, "backward"),
    ("irfft_nonlast_dim0", [3, 4, 2], 4, 0, "backward"),
    ("irfft_rank4_dim1", [2, 5, 3, 2], None, 1, "forward"),
]

# (name, 入力形状〔末尾が 2〕, n, dim, norm)。fft／ifft 共通のケース集合
# （op ごとに別々の乱数で生成する）。#2632。
C2C_CASES = [
    ("L1", [1, 2], None, None, "backward"),
    ("L2", [2, 2], None, None, "backward"),
    ("L3", [3, 2], None, None, "backward"),
    ("L4", [4, 2], None, None, "backward"),
    ("L5", [5, 2], None, None, "backward"),
    ("L8", [8, 2], None, None, "backward"),
    ("L5_ortho", [5, 2], None, None, "ortho"),
    ("L5_forward", [5, 2], None, None, "forward"),
    ("L8_ortho", [8, 2], None, None, "ortho"),
    ("L8_forward", [8, 2], None, None, "forward"),
    ("pad_6_to_8", [6, 2], 8, None, "backward"),
    ("trunc_8_to_5", [8, 2], 5, None, "backward"),
    ("trunc_5_to_2", [5, 2], 2, None, "ortho"),
    ("pad_3_to_7", [3, 2], 7, None, "forward"),
    ("batched_dim1", [2, 5, 3, 2], None, 1, "backward"),
    ("batched_dim0_n4", [2, 5, 3, 2], 4, 0, "ortho"),
    ("batched_default_dim", [3, 6, 2], None, None, "backward"),
    ("rank4_dim2_forward", [2, 3, 4, 2], None, 2, "forward"),
]

# torch が例外を出すか否かを実測して記録する境界ケース（Rust 側の拒否方針
# との突き合わせ用）。(name, op, 入力形状, n, dim)。
ERROR_CASES = [
    ("rfft_n0", "rfft", [4], 0, None),
    ("rfft_dim_oob", "rfft", [4], None, 1),
    ("rfft_empty_axis_default_n", "rfft", [0], None, None),
    ("rfft_empty_axis_n4", "rfft", [0], 4, None),
    ("irfft_m1_default_n", "irfft", [1, 2], None, None),
    ("irfft_n0", "irfft", [3, 2], 0, None),
    ("irfft_dim_oob", "irfft", [3, 2], None, 1),
    ("fft_n0", "fft", [4, 2], 0, None),
    ("fft_dim_oob", "fft", [4, 2], None, 1),
    ("fft_empty_axis_default_n", "fft", [0, 2], None, None),
    ("fft_empty_axis_n4", "fft", [0, 2], 4, None),
    ("ifft_n0", "ifft", [4, 2], 0, None),
    ("ifft_dim_oob", "ifft", [4, 2], None, 1),
    ("ifft_empty_axis_default_n", "ifft", [0, 2], None, None),
    ("ifft_empty_axis_n4", "ifft", [0, 2], 4, None),
]


def rnd(shape, gen):
    return torch.randn(*shape, generator=gen, dtype=torch.float32)


def flat(t):
    out = t.detach().contiguous().reshape(-1).tolist()
    assert all(v == v for v in out), "NaN を含む"
    return out


def kwargs(n, dim, norm):
    kw = {"norm": norm}
    if n is not None:
        kw["n"] = n
    if dim is not None:
        kw["dim"] = dim
    return kw


def make_rfft(gen, name, shape, n, dim, norm):
    x = rnd(shape, gen).requires_grad_(True)
    y = torch.view_as_real(torch.fft.rfft(x, **kwargs(n, dim, norm)))
    g = rnd(list(y.shape), gen)
    (y * g).sum().backward()
    return {
        "name": name, "op": "rfft", "in_shape": shape, "n": n, "dim": dim,
        "norm": norm, "input": flat(x), "grad_out": flat(g),
        "out_shape": list(y.shape), "output": flat(y), "grad_in": flat(x.grad),
    }


def make_irfft(gen, name, shape, n, dim, norm):
    xr = rnd(shape, gen).requires_grad_(True)
    out = torch.fft.irfft(torch.view_as_complex(xr), **kwargs(n, dim, norm))
    g = rnd(list(out.shape), gen)
    (out * g).sum().backward()
    return {
        "name": name, "op": "irfft", "in_shape": shape, "n": n, "dim": dim,
        "norm": norm, "input": flat(xr), "grad_out": flat(g),
        "out_shape": list(out.shape), "output": flat(out),
        "grad_in": flat(xr.grad),
    }


def make_c2c(gen, name, op, shape, n, dim, norm):
    x = rnd(shape, gen).requires_grad_(True)
    fn = torch.fft.fft if op == "fft" else torch.fft.ifft
    y = torch.view_as_real(fn(torch.view_as_complex(x), **kwargs(n, dim, norm)))
    g = rnd(list(y.shape), gen)
    (y * g).sum().backward()
    return {
        "name": f"{op}_{name}", "op": op, "in_shape": shape, "n": n, "dim": dim,
        "norm": norm, "input": flat(x), "grad_out": flat(g),
        "out_shape": list(y.shape), "output": flat(y), "grad_in": flat(x.grad),
    }


def make_error(name, op, shape, n, dim):
    kw = {}
    if n is not None:
        kw["n"] = n
    if dim is not None:
        kw["dim"] = dim
    try:
        if op == "rfft":
            torch.fft.rfft(torch.zeros(*shape), **kw)
        elif op in ("fft", "ifft"):
            fn = torch.fft.fft if op == "fft" else torch.fft.ifft
            fn(torch.view_as_complex(torch.zeros(*shape)), **kw)
        else:
            torch.fft.irfft(torch.view_as_complex(torch.zeros(*shape)), **kw)
        return {"name": name, "op": op, "in_shape": shape, "n": n, "dim": dim,
                "torch_raises": False, "message": ""}
    except Exception as e:  # noqa: BLE001 - 例外型・文面を実測として記録する
        return {"name": name, "op": op, "in_shape": shape, "n": n, "dim": dim,
                "torch_raises": True, "message": f"{type(e).__name__}: {e}"}


def main():
    gen = torch.Generator().manual_seed(20261005)
    cases = []
    for c in RFFT_CASES:
        cases.append(make_rfft(gen, *c))
    for c in IRFFT_CASES:
        cases.append(make_irfft(gen, *c))
    # 既存 33 ケースの乱数列を変えないよう、c2c は必ず最後に生成する（#2632）。
    for op in ("fft", "ifft"):
        for c in C2C_CASES:
            cases.append(make_c2c(gen, c[0], op, *c[1:]))
    doc = {
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "seed": 20261005,
        "cases": cases,
        "error_cases": [make_error(*c) for c in ERROR_CASES],
    }
    json.dump(doc, sys.stdout, ensure_ascii=False, separators=(",", ":"))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
