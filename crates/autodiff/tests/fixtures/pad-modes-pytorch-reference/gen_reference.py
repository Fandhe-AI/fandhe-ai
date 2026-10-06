#!/usr/bin/env python3
"""pad の reflect／replicate／circular モードの PyTorch 参照値を生成する。

イシュー #2642（親 #2625）の `tests/pad_modes_parity.rs` が参照する固定
フィクスチャ `pad_modes_reference.json` の生成スクリプト。CI は Python／
PyTorch に依存せず、コミット済み JSON のみを読む（`README.md` 参照）。

f32 はすべて u32 ビットパターン配列で保存する（JSON は NaN／inf を運べない
ため）。入力・上流勾配 `g`・出力・入力勾配をすべて保存し、Rust 側で入力を
再生成しない。損失は `(out * g).sum()`。dtype は float32。

`torch_pad` は PyTorch 形式（末尾軸から順の平坦リスト）、`pads` は Rust 形式
（先頭軸から順の `(before, after)` を rank 個。パディングしない先頭軸は
`[0, 0]`）。
"""

import json
import platform
import struct
import sys

import torch
import torch.nn.functional as F

EXPECTED_TORCH_PREFIX = "2.14.0"
assert torch.__version__.startswith(EXPECTED_TORCH_PREFIX), torch.__version__

SEED = 2642
NAN = float("nan")
INF = float("inf")
MODES = ("reflect", "replicate", "circular")


def to_bits(values):
    return [struct.unpack("<I", struct.pack("<f", float(v)))[0] for v in values]


def flat_bits(t):
    return to_bits(t.detach().contiguous().reshape(-1).tolist())


def torch_pad_of(rust_pads):
    """Rust 形式の pads から PyTorch 形式の平坦リストを作る（末尾軸から）。"""
    # 先頭側の (0,0) 軸は PyTorch 形式から省く。
    first = 0
    while first < len(rust_pads) and tuple(rust_pads[first]) == (0, 0):
        first += 1
    flat = []
    for b, a in reversed(rust_pads[first:]):
        flat += [b, a]
    return flat


def cap(mode, n):
    """モード別の「境界ちょうど」の pad 幅。"""
    if mode == "reflect":
        return n - 1
    return n  # circular・replicate は軸長ちょうど


def variants(mode, n):
    c = cap(mode, n)
    out = {
        "sym": (min(2, c), min(2, c)),
        "asym": (min(1, c), min(3, c)),
        "left0": (min(2, c), 0),
        "right0": (0, min(3, c)),
        "boundary": (c, c),
    }
    if mode == "replicate":
        out["exceed"] = (n + 1, n + 2)
    return out


# (名前, shape, パディングする末尾軸の数)
SHAPES = [
    ("r2_1ax", [2, 5], 1),
    ("r3_1ax", [2, 3, 6], 1),
    ("len1_1ax", [3, 1], 1),
    ("r3_2ax", [2, 4, 5], 2),
    ("r4_2ax", [2, 2, 4, 3], 2),
    ("r4_3ax", [2, 3, 3, 2], 3),
    ("r5_3ax", [2, 2, 3, 3, 2], 3),
]


def rust_pads_for(shape, nd, mode, variant):
    rank = len(shape)
    pads = [(0, 0)] * rank
    for ax in range(rank - nd, rank):
        v = variants(mode, shape[ax])
        if variant not in v:
            return None
        pads[ax] = v[variant]
    return pads


def run(x0, torch_pad, mode, gen):
    x = x0.clone().requires_grad_(True)
    y = F.pad(x, torch_pad, mode=mode)
    g = torch.randn(*y.shape, generator=gen, dtype=torch.float32)
    (y * g).sum().backward()
    return x, y, g


def record(name, mode, shape, rust_pads, x0, gen):
    tp = torch_pad_of(rust_pads)
    x, y, g = run(x0, tp, mode, gen)
    return {
        "name": name,
        "mode": mode,
        "shape": shape,
        "torch_pad": tp,
        "pads": [list(p) for p in rust_pads],
        "x_bits": flat_bits(x),
        "g_bits": flat_bits(g),
        "out_shape": list(y.shape),
        "out_bits": flat_bits(y),
        "grad_bits": flat_bits(x.grad),
    }


# 非有限ケース: (名前, shape, 値列, 末尾軸 pad)。forward の bit 保存確認用。
NONFINITE = [
    ("nan_inf_negzero", [1, 6], [NAN, 1.0, -0.0, INF, -INF, 0.5], (2, 2)),
    (
        "batched",
        [2, 2, 3],
        [0.0, -0.0, NAN, INF, 1.0, -INF, -0.0, 2.0, NAN, 3.0, 4.0, -INF],
        (1, 1),
    ),
]


def nonfinite_pads(mode, shape, base):
    n = shape[-1]
    c = cap(mode, n)
    pads = [(0, 0)] * len(shape)
    pads[-1] = (min(base[0], c), min(base[1], c))
    return pads


# torch が例外を出すか否かを実測して記録する境界ケース。
# (名前, モード, shape, Rust 形式 pads)。torch_pad は pads から導出する。
ERROR_CASES = [
    ("reflect_pad_eq_len", "reflect", [1, 3, 4], [(0, 0), (0, 0), (4, 0)]),
    ("reflect_pad_gt_len", "reflect", [1, 3, 4], [(0, 0), (0, 0), (0, 5)]),
    ("circular_pad_gt_len", "circular", [1, 3, 4], [(0, 0), (0, 0), (5, 0)]),
    ("circular_pad_eq_len", "circular", [1, 3, 4], [(0, 0), (0, 0), (4, 4)]),
    ("replicate_empty_axis", "replicate", [1, 3, 0], [(0, 0), (0, 0), (1, 1)]),
    ("reflect_empty_axis", "reflect", [1, 3, 0], [(0, 0), (0, 0), (1, 1)]),
    ("circular_empty_axis", "circular", [1, 3, 0], [(0, 0), (0, 0), (1, 1)]),
    ("replicate_rank1", "replicate", [5], [(1, 1)]),
    ("reflect_rank1", "reflect", [5], [(1, 1)]),
    ("circular_rank1", "circular", [5], [(1, 1)]),
    ("replicate_batch_empty", "replicate", [0, 3, 4], [(0, 0), (0, 0), (1, 1)]),
    ("replicate_leading_axis", "replicate", [3, 4, 5], [(1, 1), (0, 0), (0, 0)]),
    ("replicate_rank2_2ax", "replicate", [4, 5], [(1, 1), (1, 1)]),
    ("replicate_zero_pad", "replicate", [2, 3, 4], [(0, 0), (0, 0), (0, 0)]),
]


def make_error(name, mode, shape, rust_pads):
    x = torch.zeros(*shape)
    tp = torch_pad_of(rust_pads)
    base = {
        "name": name,
        "mode": mode,
        "shape": shape,
        "pads": [list(p) for p in rust_pads],
        "torch_pad": tp,
    }
    try:
        r = F.pad(x, tp, mode=mode)
        base.update({"torch_raises": False, "message": "", "out_shape": list(r.shape)})
    except Exception as e:  # noqa: BLE001 - 例外型・文面を実測として記録する
        base.update(
            {"torch_raises": True, "message": f"{type(e).__name__}: {e}", "out_shape": []}
        )
    return base


def main():
    gen = torch.Generator().manual_seed(SEED)
    finite = []
    for mode in MODES:
        for sname, shape, nd in SHAPES:
            for vname in ("sym", "asym", "left0", "right0", "boundary", "exceed"):
                pads = rust_pads_for(shape, nd, mode, vname)
                if pads is None:
                    continue
                if all(tuple(p) == (0, 0) for p in pads):
                    continue
                x0 = torch.randn(*shape, generator=gen, dtype=torch.float32)
                finite.append(
                    record(f"{mode}_{sname}_{vname}", mode, shape, pads, x0, gen)
                )
    nonfinite = []
    for mode in MODES:
        for name, shape, vals, base in NONFINITE:
            pads = nonfinite_pads(mode, shape, base)
            x0 = torch.tensor(vals, dtype=torch.float32).reshape(shape)
            tp = torch_pad_of(pads)
            y = F.pad(x0, tp, mode=mode)
            nonfinite.append(
                {
                    "name": f"{mode}_{name}",
                    "mode": mode,
                    "shape": shape,
                    "torch_pad": tp,
                    "pads": [list(p) for p in pads],
                    "x_bits": flat_bits(x0),
                    "out_shape": list(y.shape),
                    "out_bits": flat_bits(y),
                }
            )
    errors = [make_error(*c) for c in ERROR_CASES]
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
