#!/usr/bin/env python3
"""jacobian／hessian の PyTorch 参照値を生成する。

イシュー #2670（親 #2668）の `tests/jacobian_hessian_parity.rs` が参照する固定
フィクスチャ `jacobian_hessian_reference.json` の生成スクリプト。CI は Python／
PyTorch に依存せず、コミット済み JSON のみを読む（`README.md` 参照）。

`torch.autograd.functional.jacobian`／`hessian` を `vectorize=False`・
`create_graph=False`・float32 で実行する。f32 はすべて u32 ビットパターン配列で保存し、
入力・定数・出力・期待値をすべて保存して Rust 側で入力を再生成しない。乱数は固定シード 2670。

ケースは名前でプログラムを識別する。Rust 側（`jacobian_hessian_parity.rs`）が同名の
プログラムを `Var` 演算で組む。hessian のプログラムは create_graph の対象 Op
（add／mul／relu／exp／tanh／sigmoid／sum／mean／reshape／broadcast_to／matmul／transpose／
narrow／cat）のみで組む。
"""

import json
import platform
import sys

import torch

EXPECTED_TORCH_PREFIX = "2.14.0"
assert torch.__version__.startswith(EXPECTED_TORCH_PREFIX), torch.__version__

SEED = 2670
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


# ---- jacobian プログラム -----------------------------------------------------

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


JAC_SPECS = [
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


# ---- hessian プログラム ------------------------------------------------------

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


HESS_SPECS = [
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


def gen(specs, kind):
    cases = []
    for (name, xs, cs, fn) in specs:
        x = rnd(xs)
        consts = {k: rnd(s, 0.7) for k, s in cs.items()}

        def f(t, fn=fn, consts=consts):
            return fn(t, consts)

        out = f(x)
        if kind == "jacobian":
            res = torch.autograd.functional.jacobian(f, x, vectorize=False, create_graph=False)
        else:
            res = torch.autograd.functional.hessian(f, x, vectorize=False, create_graph=False)
        cases.append({
            "name": name,
            "x": pack(x),
            "consts": {k: pack(v) for k, v in consts.items()},
            "out": pack(out),
            "expected": pack(res),
        })
    return cases


def main():
    doc = {
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "seed": SEED,
        "jacobian_cases": gen(JAC_SPECS, "jacobian"),
        "hessian_cases": gen(HESS_SPECS, "hessian"),
    }
    out = sys.argv[1] if len(sys.argv) > 1 else "jacobian_hessian_reference.json"
    with open(out, "w") as fh:
        json.dump(doc, fh, separators=(",", ":"))
        fh.write("\n")


if __name__ == "__main__":
    main()
