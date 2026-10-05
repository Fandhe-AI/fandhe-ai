#!/usr/bin/env python3
"""isnan・isinf・isfinite・nan_to_num の PyTorch 参照値を生成する。

イシュー #2635 の `tests/nonfinite_parity.rs` が参照する固定フィクスチャ
`nonfinite_reference.json` の生成スクリプト。CI は Python／PyTorch に依存せず
コミット済み JSON のみを読む（`README.md` 参照）。

JSON は NaN／inf を運べないため、f32 はすべて u32 のビットパターン配列で保存する
（符号付き NaN・`-0.0`・非正規化数・`f32::MAX` も保存できる）。torch 側もビット
パターンから float32 テンソルを組み立てる（numpy 非依存）。

`nan_to_num` の損失は `(out * g).sum()`（`g` は固定シードの有限乱数）。`cases` の
入力勾配は NaN を含まないことを生成時に assert する。上流勾配が非有限のケースは
`upstream_nonfinite_observations` に「観測記録」として別保存する（突合の合否には
使わず、決定記録の事実記載の根拠とする）。
"""

import json
import platform
import struct
import sys

import torch

EXPECTED_TORCH_PREFIX = "2.14.0"
assert torch.__version__.startswith(EXPECTED_TORCH_PREFIX), torch.__version__

SEED = 2635
torch.manual_seed(SEED)

F32_MAX_BITS = 0x7F7FFFFF
F32_MIN_BITS = 0xFF7FFFFF
POS_NAN = 0x7FC00000
NEG_NAN = 0xFFC00001
POS_INF = 0x7F800000
NEG_INF = 0xFF800000
POS_ZERO = 0x00000000
NEG_ZERO = 0x80000000
DENORM = 0x00000001  # 最小の正の非正規化数
DENORM_NEG = 0x80000123


def f2b(v):
    return struct.unpack("<I", struct.pack("<f", v))[0]


def to_tensor(bits, shape):
    signed = [b - (1 << 32) if b >= (1 << 31) else b for b in bits]
    t = torch.tensor(signed, dtype=torch.int32).view(torch.float32)
    return t.reshape(shape)


def to_bits(t):
    return [b & 0xFFFFFFFF for b in t.reshape(-1).view(torch.int32).tolist()]


def numel(shape):
    n = 1
    for s in shape:
        n *= s
    return n


def finite_rand(n, scale=4.0):
    return [f2b(v) for v in ((torch.rand(n) * 2.0 - 1.0) * scale).tolist()]


SPECIALS = [
    POS_NAN, NEG_NAN, POS_INF, NEG_INF, POS_ZERO, NEG_ZERO, DENORM, DENORM_NEG,
    F32_MAX_BITS, F32_MIN_BITS,
]


def make_input(shape):
    n = numel(shape)
    bits = []
    i = 0
    while len(bits) < n:
        if len(bits) % 3 == 0:
            bits.append(SPECIALS[i % len(SPECIALS)])
            i += 1
        else:
            bits.extend(finite_rand(1))
    return bits[:n]


SHAPES = [[0], [1], [10], [2, 5], [2, 3, 4]]


def predicate_cases():
    out = []
    for shape in SHAPES:
        bits = make_input(shape)
        x = to_tensor(bits, shape)
        out.append({
            "name": "pred_" + "x".join(str(s) for s in shape),
            "shape": shape,
            "x_bits": bits,
            "isnan": torch.isnan(x).reshape(-1).tolist(),
            "isinf": torch.isinf(x).reshape(-1).tolist(),
            "isfinite": torch.isfinite(x).reshape(-1).tolist(),
        })
    # 全特殊値を 1 行に並べた網羅ケース
    bits = SPECIALS + [f2b(1.0), f2b(-1.5), 0x00800000, 0x007FFFFF]
    x = to_tensor(bits, [len(bits)])
    out.append({
        "name": "pred_specials",
        "shape": [len(bits)],
        "x_bits": bits,
        "isnan": torch.isnan(x).tolist(),
        "isinf": torch.isinf(x).tolist(),
        "isfinite": torch.isfinite(x).tolist(),
    })
    return out


ARG_SETS = [
    ("defaults", None, None, None),
    ("nan_only", 3.5, None, None),
    ("posinf_only", None, 100.0, None),
    ("neginf_only", None, None, -100.0),
    ("all_three", 1.0, 2.0, -3.0),
    ("nonfinite_replacements", float("inf"), float("nan"), float("-inf")),
]


def arg_bits(v):
    return None if v is None else f2b(v)


def run_nan_to_num(x, nan, posinf, neginf, g=None):
    xg = x.clone().requires_grad_(True)
    kwargs = {}
    if nan is not None:
        kwargs["nan"] = nan
    if posinf is not None:
        kwargs["posinf"] = posinf
    if neginf is not None:
        kwargs["neginf"] = neginf
    out = torch.nan_to_num(xg, **kwargs)
    if g is None:
        g = torch.rand_like(out) * 2.0 - 1.0
    (out * g).sum().backward()
    return out.detach(), g, xg.grad


def nan_to_num_cases():
    cases = []
    for shape in SHAPES:
        bits = make_input(shape)
        x = to_tensor(bits, shape)
        for label, nan, posinf, neginf in ARG_SETS:
            out, g, grad = run_nan_to_num(x, nan, posinf, neginf)
            grad_bits = to_bits(grad)
            for b in grad_bits:
                exp = (b >> 23) & 0xFF
                assert not (exp == 0xFF and (b & 0x7FFFFF) != 0), "grad に NaN"
            cases.append({
                "name": "n2n_%s_%s" % (label, "x".join(str(s) for s in shape)),
                "shape": shape,
                "x_bits": bits,
                "nan_bits": arg_bits(nan),
                "posinf_bits": arg_bits(posinf),
                "neginf_bits": arg_bits(neginf),
                "out_bits": to_bits(out),
                "g_bits": to_bits(g),
                "grad_bits": grad_bits,
            })
    return cases


def upstream_observations():
    """上流勾配が非有限のときの PyTorch 実測（観測記録。合否には使わない）。"""
    x_bits = [POS_NAN, POS_INF, NEG_INF, f2b(2.0), f2b(-1.0), POS_NAN, f2b(0.5)]
    g_bits = [POS_INF, POS_INF, NEG_INF, POS_INF, POS_NAN, f2b(1.0), NEG_INF]
    x = to_tensor(x_bits, [len(x_bits)])
    g = to_tensor(g_bits, [len(g_bits)])
    out, _, grad = run_nan_to_num(x, None, None, None, g=g)
    return [{
        "name": "obs_upstream_nonfinite",
        "shape": [len(x_bits)],
        "x_bits": x_bits,
        "g_bits": g_bits,
        "out_bits": to_bits(out),
        "grad_bits": to_bits(grad),
    }]


doc = {
    "torch_version": torch.__version__,
    "python_version": platform.python_version(),
    "seed": SEED,
    "note": "f32 はすべて u32 ビットパターン。nan_to_num の損失は (out * g).sum()。",
    "predicate_cases": predicate_cases(),
    "nan_to_num_cases": nan_to_num_cases(),
    "upstream_nonfinite_observations": upstream_observations(),
}
json.dump(doc, sys.stdout, indent=1)
sys.stdout.write("\n")
