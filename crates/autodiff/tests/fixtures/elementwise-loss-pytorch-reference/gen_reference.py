#!/usr/bin/env python3
"""pos_weight 付き BCEWithLogits・HingeEmbedding・SoftMargin・GaussianNLL の PyTorch 参照値を生成する。

イシュー #2652 の `tests/elementwise_loss_parity.rs` が参照する固定フィクスチャ
`elementwise_loss_reference.json` の生成スクリプト。CI は Python／PyTorch に依存せず
コミット済み JSON のみを読む（`README.md` 参照）。

上流勾配 1 の `loss.backward()` で全入力勾配を取る（BCE は `target`、GaussianNLL は
`input`・`target`・`var` の 3 入力すべて）。入力・出力・勾配・shape・パラメータをすべて
JSON に保存し、Rust 側で入力を再生成しない。dtype は float32。`cases` は NaN／inf を含まず、
hinge は `x == margin` ちょうどを含まない（境界は `edge_cases` で扱う）ことを生成時に assert する。

`edge_cases` は NaN／inf を JSON で表せないため、要素ごとに
`{"class": nan|pos_inf|neg_inf|finite, "value": ...}` で保存する。`diverges` を付けるのは
決定記録に数学的理由を書ける差分（SoftMargin の大振幅で PyTorch が f32 の
`log1p(exp(.))` により inf・勾配 NaN になる領域）だけである。
"""

import json
import math
import platform
import sys

import torch
import torch.nn.functional as F

EXPECTED_TORCH_PREFIX = "2.14.0"
assert torch.__version__.startswith(EXPECTED_TORCH_PREFIX), torch.__version__

SEED = 2652
torch.manual_seed(SEED)

NAN = float("nan")
INF = float("inf")


def klass(v):
    if math.isnan(v):
        return {"class": "nan"}
    if math.isinf(v):
        return {"class": "pos_inf" if v > 0 else "neg_inf"}
    return {"class": "finite", "value": v}


def lst(t):
    return [float(v) for v in t.flatten().tolist()]


def klst(t):
    return [klass(float(v)) for v in t.flatten().tolist()]


def numel(shape):
    n = 1
    for s in shape:
        n *= s
    return n


def name_of(shape):
    return "x".join(map(str, shape))


def sample(shape, lo=-3.0, hi=3.0):
    return (torch.rand(numel(shape)) * (hi - lo) + lo).reshape(shape).float()


def pm_one(shape):
    return (torch.randint(0, 2, (numel(shape),)) * 2 - 1).reshape(shape).float()


def finite(*ts):
    for t in ts:
        if t is not None:
            assert torch.isfinite(t).all(), "cases に NaN／inf を含めない"


SHAPES = ([6], [4, 3], [2, 3, 4])
REDUCTIONS = ("mean", "sum")
cases = []
edge_cases = []


def red(r):
    return r


def case(name, op, shape, params, inputs, loss_fn, grads_for):
    """`inputs`: {名前: Tensor}。`grads_for` の名前のみ requires_grad にして勾配を保存する。"""
    ts = {}
    for k, v in inputs.items():
        t = v.clone().float()
        if k in grads_for:
            t.requires_grad_(True)
        ts[k] = t
    loss = loss_fn(ts)
    loss.backward()
    out = {
        "name": name, "op": op, "shape": shape, "params": params,
        "loss": float(loss.detach()),
    }
    for k, t in ts.items():
        out[k] = lst(t.detach())
    for k in grads_for:
        out["grad_" + k] = lst(ts[k].grad)
    finite(*[t.detach() for t in ts.values()], *[ts[k].grad for k in grads_for])
    assert math.isfinite(out["loss"])
    cases.append(out)


# --- bce_pos_weight ---------------------------------------------------------
def pw_shapes(shape):
    c = shape[-1]
    yield "C", [c]
    yield "one", [1]
    yield "full", list(shape)


for shape in SHAPES:
    for pwname, pwshape in pw_shapes(shape):
        for r in REDUCTIONS:
            x = sample(shape, -4.0, 4.0)
            soft = len(cases) % 2 == 0
            y = torch.rand(numel(shape)).reshape(shape) if soft else \
                torch.randint(0, 2, (numel(shape),)).reshape(shape).float()
            p = torch.rand(numel(pwshape)).reshape(pwshape) * 4.0 + 0.1
            case(f"bce_pw_{name_of(shape)}_{pwname}_{r}_{'soft' if soft else 'hard'}",
                 "bce_pos_weight", shape,
                 {"reduction": r, "pos_weight_shape": pwshape, "has_pos_weight": True},
                 {"x": x, "y": y, "pos_weight": p},
                 lambda t, r=r: F.binary_cross_entropy_with_logits(
                     t["x"], t["y"], pos_weight=t["pos_weight"], reduction=r),
                 ["x", "y"])

# pos_weight = ones（新規 Op 経路と既存 BCE の契約整合用）と pos_weight なし（委譲経路）
for shape in SHAPES:
    for r in REDUCTIONS:
        x = sample(shape, -4.0, 4.0)
        y = torch.rand(numel(shape)).reshape(shape)
        case(f"bce_pw_ones_{name_of(shape)}_{r}", "bce_pos_weight", shape,
             {"reduction": r, "pos_weight_shape": [1], "has_pos_weight": True},
             {"x": x, "y": y, "pos_weight": torch.ones(1)},
             lambda t, r=r: F.binary_cross_entropy_with_logits(
                 t["x"], t["y"], pos_weight=t["pos_weight"], reduction=r),
             ["x", "y"])
        case(f"bce_nopw_{name_of(shape)}_{r}", "bce_pos_weight", shape,
             {"reduction": r, "pos_weight_shape": [], "has_pos_weight": False},
             {"x": x, "y": y},
             lambda t, r=r: F.binary_cross_entropy_with_logits(t["x"], t["y"], reduction=r),
             ["x", "y"])

# --- hinge_embedding --------------------------------------------------------
for shape in SHAPES:
    for margin in (1.0, 0.5, 2.0):
        for r in REDUCTIONS:
            x = sample(shape)
            y = pm_one(shape)
            assert not (x == margin).any()
            case(f"hinge_{name_of(shape)}_m{margin}_{r}", "hinge_embedding", shape,
                 {"reduction": r, "margin": margin},
                 {"x": x, "y": y},
                 lambda t, r=r, m=margin: F.hinge_embedding_loss(
                     t["x"], t["y"], margin=m, reduction=r),
                 ["x"])

# --- soft_margin ------------------------------------------------------------
for shape in SHAPES:
    for r in REDUCTIONS:
        x = sample(shape, -8.0, 8.0)
        y = pm_one(shape)
        case(f"soft_margin_{name_of(shape)}_{r}", "soft_margin", shape,
             {"reduction": r},
             {"x": x, "y": y},
             lambda t, r=r: F.soft_margin_loss(t["x"], t["y"], reduction=r),
             ["x"])

# --- gaussian_nll -----------------------------------------------------------
for shape in SHAPES:
    for full in (False, True):
        for r in REDUCTIONS:
            x = sample(shape)
            t_ = sample(shape)
            v = torch.rand(numel(shape)).reshape(shape) * 2.0 + 0.05
            case(f"gnll_{name_of(shape)}_full{int(full)}_{r}", "gaussian_nll", shape,
                 {"reduction": r, "full": full, "eps": 1e-6},
                 {"x": x, "t": t_, "var": v},
                 lambda t, r=r, f=full: F.gaussian_nll_loss(
                     t["x"], t["t"], t["var"], full=f, reduction=r),
                 ["x", "t", "var"])

# var < eps（クランプ領域）: 既定 eps とカスタム eps
for shape in ([6], [4, 3]):
    for eps in (1e-6, 0.3):
        for r in REDUCTIONS:
            x = sample(shape)
            t_ = sample(shape)
            v = torch.rand(numel(shape)).reshape(shape) * 0.6
            v.flatten()[0] = 0.0
            v.flatten()[1] = eps * 0.5
            case(f"gnll_clamp_{name_of(shape)}_eps{eps}_{r}", "gaussian_nll", shape,
                 {"reduction": r, "full": False, "eps": eps},
                 {"x": x, "t": t_, "var": v},
                 lambda t, r=r, e=eps: F.gaussian_nll_loss(
                     t["x"], t["t"], t["var"], eps=e, reduction=r),
                 ["x", "t", "var"])


# --- edge_cases -------------------------------------------------------------
def edge(name, op, params, inputs, loss_fn, grads_for, diverges=None):
    ts = {}
    for k, v in inputs.items():
        t = torch.tensor(v, dtype=torch.float32)
        if k in grads_for:
            t.requires_grad_(True)
        ts[k] = t
    loss = loss_fn(ts)
    loss.backward()
    e = {"name": name, "op": op, "params": params,
         "loss": klass(float(loss.detach()))}
    for k, v in inputs.items():
        e[k] = [klass(a) for a in v]
    for k in grads_for:
        e["grad_" + k] = klst(ts[k].grad)
    if diverges:
        e["diverges"] = diverges
    edge_cases.append(e)


def bce_edge(name, xs, ys, ps, grads=("x", "y"), r="sum"):
    edge(name, "bce_pos_weight",
         {"reduction": r, "pos_weight_shape": [len(ps)] if len(ps) != 1 else [1],
          "has_pos_weight": True},
         {"x": xs, "y": ys, "pos_weight": ps},
         lambda t: F.binary_cross_entropy_with_logits(
             t["x"], t["y"], pos_weight=t["pos_weight"], reduction=r),
         list(grads))


bce_edge("bce_pw_big_logits", [1e4, -1e4, 30.0, -30.0], [1.0, 0.0, 1.0, 1.0], [2.0])
bce_edge("bce_pw_zero_pos_weight", [0.5, -0.5, 1.0], [1.0, 1.0, 0.0], [0.0])
bce_edge("bce_pw_inf", [INF, -INF, 0.0], [0.0, 1.0, 1.0], [2.0])
bce_edge("bce_pw_nan", [NAN, 1.0], [0.5, 0.5], [2.0])


def hinge_edge(name, xs, ys, margin):
    edge(name, "hinge_embedding", {"reduction": "sum", "margin": margin},
         {"x": xs, "y": ys},
         lambda t: F.hinge_embedding_loss(t["x"], t["y"], margin=margin, reduction="sum"),
         ["x"])


hinge_edge("hinge_boundary", [1.0, 1.0, 0.5], [-1.0, 1.0, -1.0], 1.0)
hinge_edge("hinge_nan_inf", [NAN, NAN, INF, -INF, INF, -INF],
           [1.0, -1.0, 1.0, 1.0, -1.0, -1.0], 1.0)

edge("soft_margin_zero", "soft_margin", {"reduction": "sum"},
     {"x": [0.0, 0.0], "y": [1.0, -1.0]},
     lambda t: F.soft_margin_loss(t["x"], t["y"], reduction="sum"), ["x"])
edge("soft_margin_large", "soft_margin", {"reduction": "sum"},
     {"x": [200.0, -200.0, 5.0], "y": [-1.0, 1.0, 1.0]},
     lambda t: F.soft_margin_loss(t["x"], t["y"], reduction="sum"), ["x"],
     diverges="PyTorch は f32 の log1p(exp(-y*x)) を直接評価するため -y*x が約 88.7 を超えると "
              "inf（勾配は NaN）になる。本実装は安定形 softplus で有限値を返す")

edge("gnll_var_zero", "gaussian_nll", {"reduction": "sum", "full": False, "eps": 1e-6},
     {"x": [1.0, 2.0], "t": [0.5, 2.0], "var": [0.0, 0.0]},
     lambda t: F.gaussian_nll_loss(t["x"], t["t"], t["var"], reduction="sum"),
     ["x", "t", "var"])
edge("gnll_var_inf", "gaussian_nll", {"reduction": "sum", "full": False, "eps": 1e-6},
     {"x": [1.0], "t": [0.5], "var": [INF]},
     lambda t: F.gaussian_nll_loss(t["x"], t["t"], t["var"], reduction="sum"),
     ["x", "t", "var"])
edge("gnll_x_eq_t", "gaussian_nll", {"reduction": "sum", "full": True, "eps": 1e-6},
     {"x": [0.7, -1.2], "t": [0.7, -1.2], "var": [1.0, 0.25]},
     lambda t: F.gaussian_nll_loss(t["x"], t["t"], t["var"], full=True, reduction="sum"),
     ["x", "t", "var"])

fixture = {
    "torch_version": torch.__version__,
    "python_version": platform.python_version(),
    "seed": SEED,
    "cases": cases,
    "edge_cases": edge_cases,
}
json.dump(fixture, sys.stdout, indent=1, allow_nan=False)
print()
