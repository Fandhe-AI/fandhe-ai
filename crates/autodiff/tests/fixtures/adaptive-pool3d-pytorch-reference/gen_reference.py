#!/usr/bin/env python3
"""adaptive_avg_pool3d／adaptive_max_pool3d の PyTorch 参照値を生成する。

イシュー #3823 の `tests/adaptive_pool3d_parity.rs` が参照する固定フィクスチャ
`adaptive_pool3d_reference.json` の生成スクリプト。CI は Python／PyTorch に依存せず、
コミット済み JSON のみを読む（`README.md` 参照。`pool3d-pytorch-reference` と同方針）。

f32 はすべて u32 ビットパターン配列で保存する（JSON は NaN／inf を運べないため）。
入力・上流勾配 `g`・出力・索引・入力勾配をすべて保存し、Rust 側で入力を再生成しない。
損失は `(out * g).sum()`。dtype は float32。
"""

import json
import platform
import struct
import sys

import torch
import torch.nn.functional as F

EXPECTED_TORCH_PREFIX = "2.14"
assert torch.__version__.startswith(EXPECTED_TORCH_PREFIX), torch.__version__

SEED = 3823
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
    if kind == "neg":
        return -(torch.randn(*shape, generator=gen, dtype=torch.float32).abs() + 0.1)
    raise AssertionError(kind)


# 有限ケース: (name, in_shape, output_size, kind)。
# 割り切れる・割り切れない（窓が重なる）・出力が入力より大きい・全体縮約・恒等・非等方を含む。
FINITE_CASES = [
    ("divisible_4_to_2", [1, 1, 4, 4, 4], [2, 2, 2], "randn"),
    ("non_divisible_5_to_3", [1, 2, 5, 5, 5], [3, 3, 3], "randn"),
    ("aniso_mixed", [2, 2, 5, 6, 4], [3, 4, 2], "randn"),
    ("global_to_1", [1, 3, 4, 5, 6], [1, 1, 1], "randn"),
    ("identity", [1, 2, 3, 3, 3], [3, 3, 3], "randn"),
    ("output_larger_than_input", [1, 1, 2, 3, 2], [5, 4, 3], "randn"),
    ("multi_batch", [3, 2, 4, 4, 4], [3, 2, 1], "randn"),
    ("ties_overlap", [2, 2, 5, 4, 4], [3, 3, 3], "tied"),
    ("all_equal", [1, 2, 3, 3, 3], [2, 2, 2], "const"),
    ("neg_values", [1, 1, 4, 5, 6], [2, 3, 4], "neg"),
    ("unit_spatial", [2, 3, 1, 1, 1], [2, 2, 2], "randn"),
]

# 非有限ケース: (name, in_shape, output_size, 差し込み {flat 位置: 値})。
# 入力は 0.25 刻みの単調増加列に差し込む。
NONFINITE_CASES = [
    ("nan_single", [1, 1, 3, 3, 3], [2, 2, 2], {13: NAN}),
    ("nan_multi_in_window", [1, 1, 3, 3, 3], [1, 1, 1], {13: NAN, 14: NAN, 4: NAN}),
    ("posinf", [1, 1, 3, 3, 3], [2, 2, 2], {13: INF}),
    ("neginf", [1, 1, 3, 3, 3], [2, 2, 2], {13: -INF}),
    ("neginf_whole_window", [1, 1, 2, 2, 2], [1, 1, 1], {i: -INF for i in range(8)}),
    ("nan_and_inf", [1, 1, 3, 3, 3], [2, 2, 2], {13: NAN, 14: INF, 0: -INF}),
]

# torch が例外を出すか否かを実測して記録する境界ケース: (name, in_shape, output_size)。
ERROR_CASES = [
    ("output_zero_d", [1, 1, 4, 4, 4], [0, 2, 2]),
    ("output_zero_w", [1, 1, 4, 4, 4], [2, 2, 0]),
    ("spatial_zero_d", [1, 1, 0, 4, 4], [2, 2, 2]),
    ("spatial_zero_w", [1, 1, 4, 4, 0], [1, 1, 1]),
    ("batch_zero", [0, 2, 4, 4, 4], [2, 2, 2]),
    ("channel_zero", [1, 0, 4, 4, 4], [2, 2, 2]),
    ("rank4_unbatched", [2, 4, 4, 4], [2, 2, 2]),
    ("rank3", [4, 4, 4], [2, 2, 2]),
]

GEN = torch.Generator().manual_seed(SEED)


def run_max(x0, os_):
    x = x0.clone().requires_grad_(True)
    y, idx = F.adaptive_max_pool3d(x, os_, return_indices=True)
    g = torch.randn(*y.shape, generator=GEN, dtype=torch.float32)
    (y * g).sum().backward()
    return x, y, idx, g


def run_avg(x0, os_):
    x = x0.clone().requires_grad_(True)
    y = F.adaptive_avg_pool3d(x, os_)
    g = torch.randn(*y.shape, generator=GEN, dtype=torch.float32)
    (y * g).sum().backward()
    return x, y, g


def record_max(name, shape, os_, x0):
    x, y, idx, g = run_max(x0, os_)
    return {
        "name": f"max_{name}", "op": "max", "in_shape": shape, "out_shape": list(y.shape),
        "output_size": os_,
        "x_bits": flat_bits(x), "g_bits": flat_bits(g), "out_bits": flat_bits(y),
        "grad_bits": flat_bits(x.grad), "index": idx.reshape(-1).tolist(),
    }


def record_avg(name, shape, os_, x0):
    x, y, g = run_avg(x0, os_)
    return {
        "name": f"avg_{name}", "op": "avg", "in_shape": shape, "out_shape": list(y.shape),
        "output_size": os_,
        "x_bits": flat_bits(x), "g_bits": flat_bits(g), "out_bits": flat_bits(y),
        "grad_bits": flat_bits(x.grad),
    }


def make_error(op, name, shape, os_):
    x = torch.zeros(*shape)
    base = {"name": f"{op}_{name}", "op": op, "in_shape": shape, "output_size": os_}
    try:
        if op == "max":
            r = F.adaptive_max_pool3d(x, os_)
        else:
            r = F.adaptive_avg_pool3d(x, os_)
        return {**base, "torch_raises": False, "message": "", "out_shape": list(r.shape)}
    except Exception as e:  # noqa: BLE001 - 例外型・文面を実測として記録する
        return {**base, "torch_raises": True, "message": f"{type(e).__name__}: {e}",
                "out_shape": []}


def main():
    finite = []
    for name, shape, os_, kind in FINITE_CASES:
        x0 = make_input(shape, kind, GEN)
        finite.append(record_max(name, shape, os_, x0))
        finite.append(record_avg(name, shape, os_, x0))
    nonfinite = []
    for name, shape, os_, inject in NONFINITE_CASES:
        n = numel(shape)
        vals = [0.25 * i for i in range(n)]
        for pos, v in inject.items():
            vals[pos] = v
        x0 = torch.tensor(vals, dtype=torch.float32).reshape(shape)
        nonfinite.append(record_max(name, shape, os_, x0))
        nonfinite.append(record_avg(name, shape, os_, x0))
    errors = []
    for name, shape, os_ in ERROR_CASES:
        errors.append(make_error("max", name, shape, os_))
        errors.append(make_error("avg", name, shape, os_))
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
