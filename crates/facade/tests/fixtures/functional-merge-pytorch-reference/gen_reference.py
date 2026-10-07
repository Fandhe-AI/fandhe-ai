#!/usr/bin/env python3
"""Functional API の結合ノード（Concatenate／Add／Multiply／Average）の PyTorch 参照値を生成する。

イシュー #2666（親 #2663）の `crates/facade/src/compat/functional/merge_tests.rs` が参照する
固定フィクスチャ `functional_merge_reference.json` の生成スクリプト。CI は Python／PyTorch に
依存せず、コミット済み JSON のみを読む（`README.md` 参照）。`functional-graph-pytorch-reference`
（#2665）と同じ方式で、`F.linear` と活性化・結合（`torch.cat`／`+`／`*`／`(a+b+..)/n`）を手で結線した
参照を使う。目的は結線・結合ノード経由の勾配合流・結合の出力を別の結合へ入れる連鎖の検証。

f32 はすべて u32 ビットパターン配列で保存する。入力・重み・上流勾配・出力・入力勾配をすべて保存し、
Rust 側で再生成しない。dtype は float32、乱数は固定シード 2666。

重みレイアウト: 本リポの `Linear.weight` は `[in, out]`（`x·W` 規約）、PyTorch の `nn.Linear.weight` は
`[out, in]`。この転置は本スクリプト側で行い（`F.linear(x, W.T, b)`）、JSON には本リポのレイアウトと
全ブロック通し番号キー（`"{i}.weight"`／`"{i}.bias"`。結合ノードは層を持たないため通し番号に影響しない）で
保存する。
"""

import json
import platform
import sys

import torch
import torch.nn.functional as F

EXPECTED_TORCH_PREFIX = "2.14.0"
assert torch.__version__.startswith(EXPECTED_TORCH_PREFIX), torch.__version__

SEED = 2666
BATCH = 3
GEN = torch.Generator().manual_seed(SEED)


def bits_of(t):
    flat = t.detach().contiguous().reshape(-1)
    return [v & 0xFFFFFFFF for v in flat.view(torch.int32).tolist()]


def rnd(shape, scale=1.0):
    return torch.randn(tuple(shape), generator=GEN, dtype=torch.float32) * scale


def tensor_json(t):
    return {"shape": list(t.shape), "bits": bits_of(t)}


# ノード定義: ("input", dim) | ("block", [layer...], 入力ノード添字) | ("merge", op, dim, [入力ノード添字])
# layer: ("linear", in, out) | ("relu",) | ("tanh",) | ("sigmoid",)
CASES = [
    {
        "name": "residual_add",
        "nodes": [
            ("input", 4),
            ("block", [("linear", 4, 4), ("tanh",)], 0),
            ("merge", "add", 0, [0, 1]),
            ("block", [("linear", 4, 2), ("sigmoid",)], 2),
        ],
        "inputs": [0],
        "outputs": [3],
    },
    {
        "name": "concat_two_branches_then_linear",
        "nodes": [
            ("input", 4),
            ("block", [("linear", 4, 3), ("relu",)], 0),
            ("block", [("linear", 4, 2), ("tanh",)], 0),
            ("merge", "concatenate", 1, [1, 2]),
            ("block", [("linear", 5, 2), ("sigmoid",)], 3),
        ],
        "inputs": [0],
        "outputs": [4],
    },
    {
        "name": "multiply_gate",
        "nodes": [
            ("input", 4),
            ("block", [("linear", 4, 4), ("sigmoid",)], 0),
            ("block", [("linear", 4, 4), ("tanh",)], 0),
            ("merge", "multiply", 0, [1, 2]),
        ],
        "inputs": [0],
        "outputs": [3],
    },
    {
        "name": "average_three_branches",
        "nodes": [
            ("input", 4),
            ("block", [("linear", 4, 3), ("relu",)], 0),
            ("block", [("linear", 4, 3), ("tanh",)], 0),
            ("block", [("linear", 4, 3), ("sigmoid",)], 0),
            ("merge", "average", 0, [1, 2, 3]),
        ],
        "inputs": [0],
        "outputs": [4],
    },
    {
        "name": "chained_merges_two_inputs",
        "nodes": [
            ("input", 4),
            ("input", 4),
            ("block", [("linear", 4, 4), ("relu",)], 0),
            ("block", [("linear", 4, 4), ("tanh",)], 1),
            ("merge", "add", 0, [2, 3]),
            ("merge", "multiply", 0, [4, 3]),
            ("merge", "concatenate", 1, [4, 5]),
            ("merge", "average", 0, [4, 5, 2]),
        ],
        "inputs": [0, 1],
        "outputs": [6, 7],
    },
    {
        "name": "merge_only_no_blocks",
        "nodes": [
            ("input", 4),
            ("input", 4),
            ("merge", "add", 0, [0, 1]),
            ("merge", "concatenate", 1, [2, 0]),
        ],
        "inputs": [0, 1],
        "outputs": [3],
    },
]

ACT = {"relu": torch.relu, "tanh": torch.tanh, "sigmoid": torch.sigmoid}


def do_merge(op, dim, vs):
    if op == "concatenate":
        return torch.cat(vs, dim=dim)
    acc = vs[0]
    for v in vs[1:]:
        acc = acc * v if op == "multiply" else acc + v
    if op == "average":
        acc = acc / float(len(vs))
    return acc


def run_case(case):
    nodes = case["nodes"]
    values = [None] * len(nodes)
    leaves = {}
    params = {}
    layer_base = 0
    node_defs = []
    for idx, node in enumerate(nodes):
        if node[0] == "input":
            x = rnd((BATCH, node[1])).requires_grad_(True)
            leaves[idx] = x
            values[idx] = x
            node_defs.append({"kind": "input", "dim": node[1]})
        elif node[0] == "block":
            layers, src = node[1], node[2]
            h = values[src]
            layer_defs = []
            for j, layer in enumerate(layers):
                if layer[0] == "linear":
                    w = rnd((layer[1], layer[2]), 0.5)
                    b = rnd((layer[2],), 0.1)
                    key = layer_base + j
                    params["%d.weight" % key] = tensor_json(w)
                    params["%d.bias" % key] = tensor_json(b)
                    h = F.linear(h, w.t(), b)
                    layer_defs.append({"kind": "linear", "in": layer[1], "out": layer[2]})
                else:
                    h = ACT[layer[0]](h)
                    layer_defs.append({"kind": layer[0]})
            layer_base += len(layers)
            values[idx] = h
            node_defs.append({"kind": "block", "layers": layer_defs, "input": src})
        else:
            _, op, dim, srcs = node
            values[idx] = do_merge(op, dim, [values[s] for s in srcs])
            node_defs.append({"kind": "merge", "op": op, "dim": dim, "inputs": srcs})
    outs = [values[o] for o in case["outputs"]]
    ups = [rnd(tuple(o.shape)) for o in outs]
    loss = sum((o * g).sum() for o, g in zip(outs, ups))
    in_nodes = case["inputs"]
    grads = torch.autograd.grad(loss, [leaves[i] for i in in_nodes])
    return {
        "name": case["name"],
        "nodes": node_defs,
        "inputs": in_nodes,
        "outputs": case["outputs"],
        "params": params,
        "input_values": [tensor_json(leaves[i]) for i in in_nodes],
        "upstream": [tensor_json(g) for g in ups],
        "outputs_value": [tensor_json(o) for o in outs],
        "input_grads": [tensor_json(g) for g in grads],
    }


def main():
    doc = {
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "seed": SEED,
        "cases": [run_case(c) for c in CASES],
    }
    json.dump(doc, sys.stdout, separators=(",", ":"))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
