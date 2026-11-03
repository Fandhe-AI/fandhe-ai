#!/usr/bin/env python3
"""lp_pool1d／lp_pool2d／lp_pool3d の PyTorch 参照値を生成する。

イシュー #3825 の `tests/lp_pool.rs` の `parity` モジュール が参照する固定フィクスチャ
`lp_pool_reference.json` の生成スクリプト。CI は Python／PyTorch に依存せず、コミット済み JSON のみを
読む（`README.md` 参照。`adaptive-pool3d-pytorch-reference` と同方針）。

f32 はすべて u32 ビットパターン配列で保存する（JSON は NaN／inf を運べないため）。
入力・上流勾配 `g`・出力・入力勾配をすべて保存し、Rust 側で入力を再生成しない。
損失は `(out * g).sum()`。dtype は float32。PyTorch 側は `F.lp_pool{1,2,3}d`（`ceil_mode=False`）。
"""

import json
import platform
import struct

import torch
import torch.nn.functional as F

EXPECTED_TORCH_PREFIX = "2.14"
assert torch.__version__.startswith(EXPECTED_TORCH_PREFIX), torch.__version__
assert hasattr(F, "lp_pool3d") and hasattr(torch.nn, "LPPool3d")

SEED = 3825


def to_bits(values):
    return [struct.unpack("<I", struct.pack("<f", float(v)))[0] for v in values]


def flat_bits(t):
    return to_bits(t.detach().contiguous().reshape(-1).tolist())


def make_input(shape, kind, gen):
    if kind == "pos":
        return torch.rand(*shape, generator=gen, dtype=torch.float32) + 0.1
    if kind == "randn":
        return torch.randn(*shape, generator=gen, dtype=torch.float32)
    if kind == "zero_window":
        # 先頭の空間窓を丸ごと 0 にする（全零窓の勾配）。
        x = torch.rand(*shape, generator=gen, dtype=torch.float32) + 0.1
        x.reshape(-1)[: shape[-1]] = 0.0
        return x
    if kind == "has_zero":
        x = torch.rand(*shape, generator=gen, dtype=torch.float32) + 0.1
        x.reshape(-1)[0] = 0.0
        return x
    raise AssertionError(kind)


# (name, dim, in_shape, norm_type, kernel_size, stride, input kind)
CASES = [
    ("1d_p1", 1, [2, 3, 8], 1.0, 2, None, "pos"),
    ("1d_p2_stride", 1, [2, 3, 9], 2.0, 3, 2, "pos"),
    ("1d_p3", 1, [1, 2, 7], 3.0, 3, 1, "pos"),
    ("1d_p1p5", 1, [1, 2, 8], 1.5, 2, None, "pos"),
    ("1d_p2_negative_input", 1, [1, 2, 8], 2.0, 2, None, "randn"),
    ("1d_p1p5_negative_input_nan", 1, [1, 2, 8], 1.5, 2, None, "randn"),
    ("1d_p2_zero_window", 1, [1, 1, 8], 2.0, 2, None, "zero_window"),
    ("1d_p1p5_zero_window", 1, [1, 1, 8], 1.5, 2, None, "zero_window"),
    ("1d_p0p5_has_zero", 1, [1, 1, 8], 0.5, 2, None, "has_zero"),
    ("2d_p1", 2, [2, 2, 6, 6], 1.0, [2, 2], None, "pos"),
    ("2d_p2", 2, [2, 2, 6, 6], 2.0, [2, 2], None, "pos"),
    ("2d_p3_stride", 2, [1, 3, 7, 7], 3.0, [3, 2], [2, 1], "pos"),
    ("2d_p1p5", 2, [1, 2, 6, 6], 1.5, [2, 3], None, "pos"),
    ("2d_p2_negative_input", 2, [1, 2, 6, 6], 2.0, [2, 2], None, "randn"),
    ("2d_p1p5_negative_input_nan", 2, [1, 2, 6, 6], 1.5, [2, 2], None, "randn"),
    ("2d_p2_zero_window", 2, [1, 1, 4, 4], 2.0, [1, 4], None, "zero_window"),
    ("3d_p1", 3, [1, 2, 4, 4, 4], 1.0, [2, 2, 2], None, "pos"),
    ("3d_p2", 3, [1, 2, 4, 4, 4], 2.0, [2, 2, 2], None, "pos"),
    ("3d_p3_stride", 3, [2, 1, 5, 5, 5], 3.0, [2, 3, 2], [1, 2, 2], "pos"),
    ("3d_p1p5", 3, [1, 1, 4, 6, 4], 1.5, [2, 2, 2], [2, 2, 1], "pos"),
    ("3d_p2_negative_input", 3, [1, 1, 4, 4, 4], 2.0, [2, 2, 2], None, "randn"),
    ("3d_p2_zero_window", 3, [1, 1, 2, 2, 4], 2.0, [1, 1, 4], None, "zero_window"),
]


def run(dim, x, p, k, s):
    fn = {1: F.lp_pool1d, 2: F.lp_pool2d, 3: F.lp_pool3d}[dim]
    x = x.clone().requires_grad_(True)
    out = fn(x, p, k, stride=s, ceil_mode=False)
    return x, out


def main():
    gen = torch.Generator().manual_seed(SEED)
    cases = []
    for name, dim, shape, p, k, s, kind in CASES:
        x0 = make_input(shape, kind, gen)
        x, out = run(dim, x0, p, k, s)
        g = torch.rand(*out.shape, generator=gen, dtype=torch.float32) + 0.5
        (out * g).sum().backward()
        cases.append(
            {
                "name": name,
                "dim": dim,
                "norm_type": p,
                "kernel_size": k if isinstance(k, list) else [k],
                "stride": (s if isinstance(s, list) else [s]) if s is not None else None,
                "in_shape": shape,
                "out_shape": list(out.shape),
                "x_bits": flat_bits(x0),
                "g_bits": flat_bits(g),
                "out_bits": flat_bits(out),
                "grad_bits": flat_bits(x.grad),
            }
        )
    print(
        json.dumps(
            {
                "torch_version": torch.__version__,
                "python_version": platform.python_version(),
                "seed": SEED,
                "cases": cases,
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
