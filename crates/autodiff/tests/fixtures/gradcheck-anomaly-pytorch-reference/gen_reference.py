#!/usr/bin/env python3
"""gradcheck の PyTorch 参照値を生成する。

イシュー #2671（親 #2668）の `tests/gradcheck_anomaly_parity.rs` が参照する固定フィクスチャ
`gradcheck_anomaly_reference.json` の生成スクリプト。CI は Python／PyTorch に依存せず、
コミット済み JSON のみを読む（`README.md` 参照）。

各プログラムについて `torch.autograd.functional.jacobian`（`vectorize=False`・
`create_graph=False`）を float32（G1 の期待値）と float64（G3 の観測値）で実行する。
f32 はすべて u32 ビットパターン、f64 は u64 ビットパターンで保存し、入力・定数・出力・
期待値をすべて保存して Rust 側で入力を再生成しない。乱数は固定シード 2671。

入力は中心差分のキンク回避のため、relu の前段値が 10·eps（eps=1e-3）より 0 から離れる
ことを assert で保証する。プログラム名は Rust 側（`gradcheck_anomaly_parity.rs`）の
`build_program` と 1 対 1。
"""

import json
import platform
import sys

import torch

EXPECTED_TORCH_PREFIX = "2.14.0"
assert torch.__version__.startswith(EXPECTED_TORCH_PREFIX), torch.__version__

SEED = 2671
GEN = torch.Generator().manual_seed(SEED)
KINK_MARGIN = 10 * 1e-3


def bits32(t):
    flat = t.detach().contiguous().reshape(-1)
    if flat.numel() == 0:
        return []
    return [v & 0xFFFFFFFF for v in flat.view(torch.int32).tolist()]


def bits64(t):
    flat = t.detach().contiguous().reshape(-1)
    if flat.numel() == 0:
        return []
    return [v & 0xFFFFFFFFFFFFFFFF for v in flat.view(torch.int64).tolist()]


def rnd(shape, scale=1.0):
    return torch.randn(tuple(shape), generator=GEN, dtype=torch.float32) * scale


def away_from_zero(t):
    """|t| < 0.05 の要素を符号を保って押し出す（キンク回避）。"""
    s = torch.where(t >= 0, torch.ones_like(t), -torch.ones_like(t))
    return torch.where(t.abs() < 0.05, s * 0.05, t)


def pack32(t):
    return {"shape": list(t.shape), "bits": bits32(t)}


def pack64(t):
    return {"shape": list(t.shape), "bits": bits64(t)}


def p_elementwise(xs, c):
    (x,) = xs
    return torch.tanh(x) * x + torch.exp(x)


def p_matmul_tanh(xs, c):
    (x,) = xs
    return torch.tanh(x @ c["w"])


def p_sum_reduce(xs, c):
    (x,) = xs
    return (torch.sigmoid(x) * x).sum()


def p_mlp(xs, c):
    (x,) = xs
    pre = x @ c["w1"] + c["b1"]
    assert pre.abs().min().item() > KINK_MARGIN, "relu のキンクに近い前段値"
    return torch.relu(pre) @ c["w2"]


def p_scalar_in(xs, c):
    (x,) = xs
    return torch.exp(x) * x


def p_multi_input(xs, c):
    x, y = xs
    return x * y + torch.tanh(y)


def p_relu_safe(xs, c):
    (x,) = xs
    assert x.abs().min().item() > KINK_MARGIN
    return torch.relu(x) * x


# (name, 入力 shape 列, 定数 shape, 関数, 入力にキンク回避を掛けるか)
SPECS = [
    ("elementwise", [[4]], {}, p_elementwise, False),
    ("matmul_tanh", [[2, 3]], {"w": [3, 2]}, p_matmul_tanh, False),
    ("sum_reduce", [[2, 3]], {}, p_sum_reduce, False),
    ("mlp", [[2, 3]], {"w1": [3, 4], "b1": [4], "w2": [4, 2]}, p_mlp, False),
    ("scalar_in", [[]], {}, p_scalar_in, False),
    ("multi_input", [[3], [3]], {}, p_multi_input, False),
    ("relu_safe", [[5]], {}, p_relu_safe, True),
]


def gen():
    cases = []
    for (name, xshapes, cs, fn, avoid) in SPECS:
        xs = [rnd(s) for s in xshapes]
        if avoid:
            xs = [away_from_zero(x) for x in xs]
        consts = {k: rnd(s, 0.7) for k, s in cs.items()}
        out = fn(tuple(xs), consts)
        jac32 = torch.autograd.functional.jacobian(
            lambda *a: fn(a, consts), tuple(xs), vectorize=False, create_graph=False
        )
        xs64 = tuple(x.double() for x in xs)
        c64 = {k: v.double() for k, v in consts.items()}
        out64 = fn(xs64, c64)
        jac64 = torch.autograd.functional.jacobian(
            lambda *a: fn(a, c64), xs64, vectorize=False, create_graph=False
        )
        cases.append({
            "name": name,
            "xs": [pack32(x) for x in xs],
            "consts": {k: pack32(v) for k, v in consts.items()},
            "out": pack32(out),
            "expected": [pack32(j) for j in jac32],
            "out_f64": pack64(out64),
            "expected_f64": [pack64(j) for j in jac64],
        })
    return cases


def main():
    doc = {
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "seed": SEED,
        "cases": gen(),
    }
    out = sys.argv[1] if len(sys.argv) > 1 else "gradcheck_anomaly_reference.json"
    with open(out, "w") as fh:
        json.dump(doc, fh, separators=(",", ":"))
        fh.write("\n")


if __name__ == "__main__":
    main()
