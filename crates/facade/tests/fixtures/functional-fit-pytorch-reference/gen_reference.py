#!/usr/bin/env python3
"""Functional API の学習（パラメータ勾配・fit の軌跡）の PyTorch 参照値を生成する。

イシュー #2667（親 #2663）の `crates/facade/src/compat/functional/fit_parity_tests.rs` が参照する
固定フィクスチャ `functional_fit_reference.json` の生成スクリプト。CI は Python／PyTorch に
依存せず、コミット済み JSON のみを読む（`README.md` 参照）。`functional-merge-pytorch-reference`
（#2666）と同じ方式で、`F.linear` と活性化・結合（`torch.cat`／`+`／`*`／`(a+b+..)/n`）を手で結線した
参照を使い、次を保存する。

- `grad` ケース: 初期パラメータでの全パラメータ勾配（損失は出力ごとの `F.mse_loss`〈mean〉の和）。
  #2666 の fixture は入力勾配のみで、`bind` 経路が要るパラメータ勾配は本 fixture が初出。
- `fit` ケース: `torch.optim.SGD`／`torch.optim.Adam` による数 epoch の学習（`shuffle=false`・端数バッチあり）の
  epoch 損失（サンプル数重み付き平均。Rust 側の集計式 `Σ loss_b·n_b / N` と同じ。f64 で合計し 1 回 f32 化）と
  最終パラメータ。
- 分類ケース: 単一出力・`F.cross_entropy`（mean）・int64 クラス目標。

f32 はすべて u32 ビットパターン配列で保存し、Rust 側で再生成しない。dtype は float32、乱数は固定シード 2667。

重みレイアウト: 本リポの `Linear.weight` は `[in, out]`（`x·W` 規約）、PyTorch の `nn.Linear.weight` は
`[out, in]`。ここでは `[in, out]` のリーフテンソルを直接持ち `F.linear(x, W.t(), b)` で使うため、保存する
勾配・最終値も本リポのレイアウトのまま。キーは全ブロックを通した層の通し番号 `"{i}.weight"`／`"{i}.bias"`
（結合ノードは層を持たず通し番号に影響しない）。
"""

import json
import platform
import sys

import torch
import torch.nn.functional as F

EXPECTED_TORCH_PREFIX = "2.14.0"
assert torch.__version__.startswith(EXPECTED_TORCH_PREFIX), torch.__version__

SEED = 2667
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
# G1: 2 入力・fan-out（分岐 a が add／concatenate の両方へ入る）・結合 2 種・2 出力。
G1 = {
    "nodes": [
        ("input", 3),
        ("input", 2),
        ("block", [("linear", 3, 4), ("tanh",)], 0),
        ("block", [("linear", 2, 4), ("relu",)], 1),
        ("merge", "add", 0, [2, 3]),
        ("merge", "concatenate", 1, [2, 3]),
        ("block", [("linear", 4, 2)], 4),
        ("block", [("linear", 8, 1)], 5),
    ],
    "inputs": [0, 1],
    "outputs": [6, 7],
}

# G2: multiply・average・fan-out・3 入力 average を含む単一出力グラフ。
G2 = {
    "nodes": [
        ("input", 4),
        ("block", [("linear", 4, 3), ("sigmoid",)], 0),
        ("block", [("linear", 4, 3), ("tanh",)], 0),
        ("merge", "multiply", 0, [1, 2]),
        ("merge", "average", 0, [1, 2, 3]),
        ("block", [("linear", 3, 2)], 4),
    ],
    "inputs": [0],
    "outputs": [5],
}

# G3: 単一入力・単一出力の分類チェーン（CrossEntropy）。
G3 = {
    "nodes": [
        ("input", 4),
        ("block", [("linear", 4, 6), ("tanh",), ("linear", 6, 3)], 0),
    ],
    "inputs": [0],
    "outputs": [1],
}

ACT = {"relu": torch.relu, "tanh": torch.tanh, "sigmoid": torch.sigmoid}


def init_params(graph):
    """グラフの全 Linear を通し番号キーで初期化する（リーフ・勾配あり）。"""
    params = {}
    layer_base = 0
    for node in graph["nodes"]:
        if node[0] != "block":
            continue
        for j, layer in enumerate(node[1]):
            if layer[0] == "linear":
                params["%d.weight" % (layer_base + j)] = rnd((layer[1], layer[2]), 0.5).requires_grad_(True)
                params["%d.bias" % (layer_base + j)] = rnd((layer[2],), 0.1).requires_grad_(True)
        layer_base += len(node[1])
    return params


def do_merge(op, dim, vs):
    if op == "concatenate":
        return torch.cat(vs, dim=dim)
    acc = vs[0]
    for v in vs[1:]:
        acc = acc * v if op == "multiply" else acc + v
    if op == "average":
        acc = acc / float(len(vs))
    return acc


def forward(graph, params, xs):
    """`xs` は graph["inputs"] の順の入力。graph["outputs"] の順の出力を返す。"""
    nodes = graph["nodes"]
    values = [None] * len(nodes)
    for slot, idx in enumerate(graph["inputs"]):
        values[idx] = xs[slot]
    layer_base = 0
    for idx, node in enumerate(nodes):
        if node[0] == "input":
            continue
        if node[0] == "block":
            h = values[node[2]]
            for j, layer in enumerate(node[1]):
                if layer[0] == "linear":
                    key = layer_base + j
                    h = F.linear(h, params["%d.weight" % key].t(), params["%d.bias" % key])
                else:
                    h = ACT[layer[0]](h)
            layer_base += len(node[1])
            values[idx] = h
        else:
            _, op, dim, srcs = node
            values[idx] = do_merge(op, dim, [values[s] for s in srcs])
    return [values[o] for o in graph["outputs"]]


def node_defs(graph):
    defs = []
    for node in graph["nodes"]:
        if node[0] == "input":
            defs.append({"kind": "input", "dim": node[1]})
        elif node[0] == "block":
            layer_defs = []
            for layer in node[1]:
                if layer[0] == "linear":
                    layer_defs.append({"kind": "linear", "in": layer[1], "out": layer[2]})
                else:
                    layer_defs.append({"kind": layer[0]})
            defs.append({"kind": "block", "layers": layer_defs, "input": node[2]})
        else:
            defs.append({"kind": "merge", "op": node[1], "dim": node[2], "inputs": node[3]})
    return defs


def in_dims(graph):
    return [graph["nodes"][i][1] for i in graph["inputs"]]


def mse_sum(outs, ys):
    """出力ごとの mse（mean）の和。出力の指定順に左畳み込み（Rust 側の `Var::add` と同じ順）。"""
    total = None
    for o, y in zip(outs, ys):
        term = F.mse_loss(o, y)
        total = term if total is None else total + term
    return total


def case_common(name, graph, params, n, loss_kind):
    xs = [rnd((n, d)) for d in in_dims(graph)]
    return {
        "name": name,
        "nodes": node_defs(graph),
        "inputs": graph["inputs"],
        "outputs": graph["outputs"],
        "params": {k: tensor_json(v) for k, v in params.items()},
        "loss": loss_kind,
        "xs": [tensor_json(x) for x in xs],
    }, xs


def grad_case(name, graph, n, out_dims):
    params = init_params(graph)
    case, xs = case_common(name, graph, params, n, "mse")
    ys = [rnd((n, d)) for d in out_dims]
    case["ys"] = [tensor_json(y) for y in ys]
    case["target_dtype"] = "f32"
    outs = forward(graph, params, xs)
    loss = mse_sum(outs, ys)
    keys = list(params.keys())
    grads = torch.autograd.grad(loss, [params[k] for k in keys])
    case["kind"] = "grad"
    case["loss_bits"] = bits_of(loss)
    case["param_grads"] = {k: tensor_json(g) for k, g in zip(keys, grads)}
    return case


def fit_case(name, graph, n, out_dims, opt_name, opt_cfg, epochs, batch_size, ce=False):
    params = init_params(graph)
    case, xs = case_common(name, graph, params, n, "cross_entropy" if ce else "mse")
    if ce:
        ys = [torch.randint(0, out_dims[0], (n,), generator=GEN)]
        case["ys"] = [{"shape": [n], "values": [int(v) for v in ys[0].tolist()]}]
        case["target_dtype"] = "i32"
    else:
        ys = [rnd((n, d)) for d in out_dims]
        case["ys"] = [tensor_json(y) for y in ys]
        case["target_dtype"] = "f32"
    keys = list(params.keys())
    plist = [params[k] for k in keys]
    if opt_name == "sgd":
        opt = torch.optim.SGD(plist, lr=opt_cfg["lr"], momentum=opt_cfg["momentum"])
    else:
        opt = torch.optim.Adam(
            plist,
            lr=opt_cfg["lr"],
            betas=(opt_cfg["beta1"], opt_cfg["beta2"]),
            eps=opt_cfg["eps"],
            weight_decay=0.0,
        )
    epoch_losses = []
    for _ in range(epochs):
        weighted = 0.0
        for start in range(0, n, batch_size):
            xb = [x[start:start + batch_size] for x in xs]
            yb = [y[start:start + batch_size] for y in ys]
            nb = xb[0].shape[0]
            outs = forward(graph, params, xb)
            if ce:
                loss = F.cross_entropy(outs[0], yb[0])
            else:
                loss = mse_sum(outs, yb)
            weighted += float(loss.item()) * float(nb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        epoch_losses.append(torch.tensor(weighted / float(n), dtype=torch.float64).to(torch.float32))
    case["kind"] = "fit"
    case["optimizer"] = dict(opt_cfg, kind=opt_name)
    case["epochs"] = epochs
    case["batch_size"] = batch_size
    case["epoch_losses_bits"] = [bits_of(e) for e in epoch_losses]
    case["final_params"] = {k: tensor_json(params[k]) for k in keys}
    return case


def main():
    sgd = {"lr": 0.05, "momentum": 0.9}
    adam = {"lr": 0.01, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8}
    cases = [
        grad_case("grad_two_in_two_out_fan_out", G1, 5, [2, 1]),
        grad_case("grad_multiply_average_fan_out", G2, 4, [2]),
        fit_case("fit_sgd_momentum_two_in_two_out", G1, 10, [2, 1], "sgd", sgd, 3, 4),
        fit_case("fit_adam_two_in_two_out", G1, 10, [2, 1], "adam", adam, 3, 4),
        fit_case("fit_cross_entropy_chain", G3, 10, [3], "sgd", {"lr": 0.1, "momentum": 0.0}, 3, 4, ce=True),
    ]
    doc = {
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "seed": SEED,
        "cases": cases,
    }
    json.dump(doc, sys.stdout, separators=(",", ":"))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
