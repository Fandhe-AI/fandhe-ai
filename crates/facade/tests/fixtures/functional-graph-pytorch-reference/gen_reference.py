#!/usr/bin/env python3
"""Functional API（多入力・多出力 DAG）の PyTorch 参照値を生成する。

イシュー #2665（親 #2663）の `crates/facade/src/compat/functional/tests.rs` が参照する
固定フィクスチャ `functional_graph_reference.json` の生成スクリプト。CI は Python／
PyTorch に依存せず、コミット済み JSON のみを読む（`README.md` 参照）。

PyTorch に Functional API は無いため、`F.linear` と活性化を手で結線した参照を使う。
層の語彙は Linear／ReLU／Tanh／Sigmoid に限る（層ごとの数値は既存テストが担保済みで、
本フィクスチャの目的は結線・出力順・入力順・fan-out の勾配合流の検証）。

f32 はすべて u32 ビットパターン配列で保存する。入力・重み・上流勾配・出力・入力勾配を
すべて保存し、Rust 側で再生成しない。dtype は float32、乱数は固定シード 2665。

重みレイアウト: 本リポの `Linear.weight` は `[in, out]`（`x·W` 規約）、PyTorch の
`nn.Linear.weight` は `[out, in]`。この転置は本スクリプト側で行い（`F.linear(x, W.T, b)`）、
JSON には本リポのレイアウトと全ブロック通し番号キー（`"{i}.weight"`／`"{i}.bias"`。
`i` は Block ノードの挿入順に層数を累積した通し番号）で保存する。
"""

import json
import platform
import sys

import torch
import torch.nn.functional as F

EXPECTED_TORCH_PREFIX = "2.14.0"
assert torch.__version__.startswith(EXPECTED_TORCH_PREFIX), torch.__version__

SEED = 2665
BATCH = 3
GEN = torch.Generator().manual_seed(SEED)


def bits_of(t):
    flat = t.detach().contiguous().reshape(-1)
    return [v & 0xFFFFFFFF for v in flat.view(torch.int32).tolist()]


def rnd(shape, scale=1.0):
    return torch.randn(tuple(shape), generator=GEN, dtype=torch.float32) * scale


def tensor_json(t):
    return {"shape": list(t.shape), "bits": bits_of(t)}


# ノード定義: ("input", dim) | ("block", [layer...], 入力ノード添字)
# layer: ("linear", in, out) | ("relu",) | ("tanh",) | ("sigmoid",)
CASES = [
    {
        "name": "two_in_two_out_disjoint",
        "nodes": [
            ("input", 4),
            ("input", 3),
            ("block", [("linear", 4, 5), ("relu",)], 0),
            ("block", [("linear", 3, 2), ("tanh",)], 1),
        ],
        "inputs": [0, 1],
        "outputs": [2, 3],
    },
    {
        "name": "fan_out",
        "nodes": [
            ("input", 4),
            ("block", [("linear", 4, 6), ("tanh",)], 0),
            ("block", [("linear", 6, 3), ("sigmoid",)], 1),
            ("block", [("linear", 6, 2), ("relu",)], 1),
        ],
        "inputs": [0],
        "outputs": [3, 2],
    },
    {
        "name": "chain_with_inner_output",
        "nodes": [
            ("input", 3),
            ("block", [("linear", 3, 4), ("relu",), ("linear", 4, 4), ("tanh",)], 0),
            ("block", [("linear", 4, 4), ("tanh",)], 1),
            ("block", [("linear", 4, 2), ("sigmoid",)], 2),
        ],
        "inputs": [0],
        "outputs": [3, 2],
    },
    {
        "name": "inputs_permuted",
        "nodes": [
            ("input", 3),
            ("input", 4),
            ("block", [("linear", 4, 3), ("tanh",)], 1),
            ("block", [("linear", 4, 2), ("relu",)], 1),
            ("block", [("linear", 3, 3), ("sigmoid",)], 0),
        ],
        "inputs": [1, 0],
        "outputs": [4, 2, 3],
    },
    {
        "name": "passthrough_output",
        "nodes": [
            ("input", 4),
            ("block", [("linear", 4, 3), ("relu",)], 0),
        ],
        "inputs": [0],
        "outputs": [1, 0],
    },
]

ACT = {"relu": torch.relu, "tanh": torch.tanh, "sigmoid": torch.sigmoid}


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
        else:
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
