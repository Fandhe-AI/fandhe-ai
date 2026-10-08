#!/usr/bin/env python3
"""vjp・hvp・vmap の PyTorch 参照値を生成する。

イシュー #2877（親 #2841）の `tests/functional_ops_pytorch_parity.rs` が参照する固定
フィクスチャ `functional_transforms_reference.json` の生成スクリプト。CI は Python／
PyTorch に依存せず、コミット済み JSON のみを読む（`README.md` 参照）。設計の正は
`docs/autodiff-functional-transforms-design.md` §6（第三者比較層）。

- vjp: `torch.func.vjp(f, x)` の `vjp_fn(u)[0]`。
- hvp: 主参照は `torch.func.vjp(torch.func.grad(g), x)` の `vjp_fn(v)[0]`
  （fandhe 側と同じ reverse-over-reverse）。`torch.autograd.functional.hvp` の結果と
  スクリプト内で `torch.allclose` による相互検証を行う（JSON には主参照だけを保存する）。
- vmap: `torch.func.vmap(f, in_dims=k, out_dims=0)(x)`（forward 値のみ）。

f32 はすべて u32 ビットパターン配列で保存し、入力・定数・出力・期待値を保存して Rust 側で
入力を再生成しない。診断用に同じ f32 入力を float64 へ上げた結果を `expected_f64` として
保存する（ゲートには使わず、判定不能の分類根拠にだけ使う）。乱数は固定シード 2877。

プログラムは名前で識別する。Rust 側（`functional_ops_pytorch_parity.rs`）が同名のプログラムを
`Var` 演算で組む。hvp のプログラムは `Op::supports_create_graph` の対象 Op のみで組む。
"""

import json
import platform
import sys

import torch

EXPECTED_TORCH_PREFIX = "2.14.0"
assert torch.__version__.startswith(EXPECTED_TORCH_PREFIX), torch.__version__

SEED = 2877
GEN = torch.Generator().manual_seed(SEED)


def bits_of(t):
    flat = t.detach().contiguous().reshape(-1)
    if flat.numel() == 0:
        return []
    return [v & 0xFFFFFFFF for v in flat.view(torch.int32).tolist()]


def rnd(shape, scale=1.0):
    return torch.randn(tuple(shape), generator=GEN, dtype=torch.float32) * scale


def pack(t):
    return {"shape": list(t.shape), "bits": bits_of(t)}


def pack_f64(t):
    return {"shape": list(t.shape), "values": t.detach().contiguous().reshape(-1).tolist()}


# ---- vjp プログラム ----------------------------------------------------------

def p_vec_elementwise(x, c):
    return torch.tanh(x) * x + torch.exp(x)


def p_scalar_sum(x, c):
    return (torch.sigmoid(x) * x).sum()


def p_matmul_tanh(x, c):
    return torch.tanh(x @ c["w"])


def p_transpose_out(x, c):
    return x.transpose(0, 1)


def p_broadcast_out(x, c):
    return x.expand(2, 3)


def p_independent_rows(x, c):
    return torch.cat([x * x, c["k"]], 0)


def p_mean_dim(x, c):
    return torch.exp(x).mean(1)


def p_mlp(x, c):
    return torch.relu(x @ c["w1"] + c["b1"]) @ c["w2"]


def p_scalar_in(x, c):
    return torch.exp(x) * x


VJP_SPECS = [
    ("vec_elementwise", [4], {}, p_vec_elementwise),
    ("scalar_sum", [2, 3], {}, p_scalar_sum),
    ("matmul_tanh", [2, 3], {"w": [3, 2]}, p_matmul_tanh),
    ("transpose_out", [2, 3], {}, p_transpose_out),
    ("broadcast_out", [3], {}, p_broadcast_out),
    ("independent_rows", [3], {"k": [2]}, p_independent_rows),
    ("mean_dim", [2, 3], {}, p_mean_dim),
    ("mlp", [2, 3], {"w1": [3, 4], "b1": [4], "w2": [4, 2]}, p_mlp),
    ("scalar_in", [], {}, p_scalar_in),
]


# ---- hvp プログラム ----------------------------------------------------------

def h_quadratic(x, c):
    return (x * (c["a"] @ x)).sum()


def h_linear(x, c):
    return (c["k"] * x).sum()


def h_tanh_sum(x, c):
    return (torch.tanh(x) * x).sum()


def h_exp_mean(x, c):
    return (torch.exp(x) * x).mean()


def h_mlp(x, c):
    return torch.sigmoid(torch.tanh(x @ c["w1"] + c["b1"]) @ c["w2"]).sum()


def h_scalar_in(x, c):
    return torch.exp(x) * x


def h_loss_1x1(x, c):
    return (x * x).sum().reshape(1, 1)


def h_cat_transpose(x, c):
    z = torch.cat([torch.sigmoid(x), torch.tanh(x)], 1).transpose(0, 1)
    return (z * z).sum()


def h_relu_cubic(x, c):
    r = torch.relu(x)
    return (r * r * x).sum()


HVP_SPECS = [
    ("quadratic", [3, 1], {"a": [3, 3]}, h_quadratic),
    ("linear", [4], {"k": [4]}, h_linear),
    ("tanh_sum", [2, 3], {}, h_tanh_sum),
    ("exp_mean", [3], {}, h_exp_mean),
    ("mlp", [1, 3], {"w1": [3, 4], "b1": [4], "w2": [4, 1]}, h_mlp),
    ("scalar_in", [], {}, h_scalar_in),
    ("loss_1x1", [3], {}, h_loss_1x1),
    ("cat_transpose", [2, 3], {}, h_cat_transpose),
    ("relu_cubic", [4], {}, h_relu_cubic),
]


# ---- vmap プログラム（スライス 1 枚に適用する本体）-----------------------------

def m_elementwise(s, c):
    return torch.tanh(s) * s


def m_in_dim1(s, c):
    return torch.exp(s) * s + s


def m_matmul_closure(s, c):
    return torch.tanh(s @ c["w"])


def m_transpose(s, c):
    return s.transpose(0, 1)


def m_to_scalar(s, c):
    return s.sum()


def m_expand(s, c):
    return s.expand(2, 3)


def m_cat(s, c):
    return torch.cat([s, s * s], 0)


# (name, 入力 shape, in_dim, consts, fn)
VMAP_SPECS = [
    ("elementwise", [3, 4], 0, {}, m_elementwise),
    ("in_dim1", [3, 4], 1, {}, m_in_dim1),
    ("matmul_closure", [3, 2, 3], 0, {"w": [3, 2]}, m_matmul_closure),
    ("transpose", [2, 2, 3], 0, {}, m_transpose),
    ("to_scalar", [3, 4], 0, {}, m_to_scalar),
    ("expand", [4, 3], 0, {}, m_expand),
    ("cat", [3, 4], 0, {}, m_cat),
]


def scalarize(fn):
    """`torch.func.grad` はスカラー出力を要求するため、shape `[1, 1]` の loss を `reshape(())` で包む。"""
    return lambda t: fn(t).reshape(())


def hvp_ref(g, x, v):
    _, vjp_fn = torch.func.vjp(torch.func.grad(g), x)
    return vjp_fn(v)[0]


def gen_vjp():
    cases = []
    for (name, xs, cs, fn) in VJP_SPECS:
        x = rnd(xs)
        consts = {k: rnd(s, 0.7) for k, s in cs.items()}
        f = lambda t, fn=fn, consts=consts: fn(t, consts)
        out = f(x)
        u = rnd(out.shape)
        _, vjp_fn = torch.func.vjp(f, x)
        res = vjp_fn(u)[0]
        consts64 = {k: v.double() for k, v in consts.items()}
        f64 = lambda t, fn=fn, consts64=consts64: fn(t, consts64)
        _, vjp_fn64 = torch.func.vjp(f64, x.double())
        res64 = vjp_fn64(u.double())[0]
        cases.append({
            "name": name, "x": pack(x), "consts": {k: pack(v) for k, v in consts.items()},
            "out": pack(out), "u": pack(u), "expected": pack(res), "expected_f64": pack_f64(res64),
        })
    return cases


def gen_hvp():
    cases = []
    for (name, xs, cs, fn) in HVP_SPECS:
        x = rnd(xs)
        consts = {k: rnd(s, 0.7) for k, s in cs.items()}
        f = lambda t, fn=fn, consts=consts: fn(t, consts)
        out = f(x)
        v = rnd(xs)
        g = scalarize(f)
        res = hvp_ref(g, x, v)
        cross = torch.autograd.functional.hvp(g, x, v)[1]
        assert torch.allclose(res, cross), (name, res, cross)
        consts64 = {k: t.double() for k, t in consts.items()}
        f64 = lambda t, fn=fn, consts64=consts64: fn(t, consts64)
        res64 = hvp_ref(scalarize(f64), x.double(), v.double())
        cases.append({
            "name": name, "x": pack(x), "consts": {k: pack(t) for k, t in consts.items()},
            "out": pack(out), "v": pack(v), "expected": pack(res), "expected_f64": pack_f64(res64),
        })
    return cases


def gen_vmap():
    cases = []
    for (name, xs, in_dim, cs, fn) in VMAP_SPECS:
        x = rnd(xs)
        consts = {k: rnd(s, 0.7) for k, s in cs.items()}
        f = lambda t, fn=fn, consts=consts: fn(t, consts)
        res = torch.func.vmap(f, in_dims=in_dim, out_dims=0)(x)
        consts64 = {k: t.double() for k, t in consts.items()}
        f64 = lambda t, fn=fn, consts64=consts64: fn(t, consts64)
        res64 = torch.func.vmap(f64, in_dims=in_dim, out_dims=0)(x.double())
        cases.append({
            "name": name, "x": pack(x), "in_dim": in_dim,
            "consts": {k: pack(t) for k, t in consts.items()},
            "expected": pack(res), "expected_f64": pack_f64(res64),
        })
    return cases


def main():
    doc = {
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "seed": SEED,
        "api": {
            "vjp": "torch.func.vjp",
            "hvp": "torch.func.vjp(torch.func.grad(f))",
            "vmap": "torch.func.vmap",
            "torch_func_has_hvp": hasattr(torch.func, "hvp"),
        },
        "vjp_cases": gen_vjp(),
        "hvp_cases": gen_hvp(),
        "vmap_cases": gen_vmap(),
    }
    out = sys.argv[1] if len(sys.argv) > 1 else "functional_transforms_reference.json"
    with open(out, "w") as fh:
        json.dump(doc, fh, separators=(",", ":"))
        fh.write("\n")


if __name__ == "__main__":
    main()
