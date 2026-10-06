#!/usr/bin/env python3
"""MultiMargin・MultiLabelMargin・MultiLabelSoftMargin・sigmoid focal loss の PyTorch 参照値を生成する。

イシュー #2653 の `tests/margin_focal_loss_parity.rs` が参照する固定フィクスチャ
`margin_focal_loss_reference.json` の生成スクリプト。CI は Python／PyTorch に依存せず
コミット済み JSON のみを読む（`README.md` 参照）。

- 3 つのマージン系損失は `torch.nn.functional` の実装をそのまま呼ぶ。
- `torch.nn.functional` に focal loss は存在しないため、`torchvision.ops.sigmoid_focal_loss`
  と同じ式（下記 `sigmoid_focal`）を素の torch 2.14.0 の autograd で評価する
  （torchvision は導入しない。出自を「torch 2.14.0 のみ」に保つため）。
- 上流勾配 1 の `loss.backward()` で `input` の勾配を取る（`target`・`weight` は非追跡）。
  入力・出力・勾配・shape・パラメータをすべて JSON に保存し、Rust 側で入力を再生成しない。
  dtype は float32。`cases` は NaN／inf を含まず、マージン系は `z == 0` ちょうどを含まない
  （境界は `edge_cases` で扱う）ことを生成時に assert する。
- `edge_cases` は NaN／inf を JSON で表せないため
  `{"class": nan|pos_inf|neg_inf|finite, "value": ...}` で保存する。
  `diverges` は決定記録に数学的理由を書ける差分にのみ付ける。
"""

import json
import math
import platform
import sys

import torch
import torch.nn.functional as F

EXPECTED_TORCH_PREFIX = "2.14.0"
assert torch.__version__.startswith(EXPECTED_TORCH_PREFIX), torch.__version__

SEED = 2653
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


def ilst(t):
    return [int(v) for v in t.flatten().tolist()]


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


REDUCTIONS = ("mean", "sum")
cases = []
edge_cases = []


def sigmoid_focal(x, t, alpha, gamma, reduction):
    """`torchvision.ops.sigmoid_focal_loss` と同じ式（alpha < 0 は重み付けなし）。"""
    p = torch.sigmoid(x)
    ce = F.binary_cross_entropy_with_logits(x, t, reduction="none")
    p_t = p * t + (1 - p) * (1 - t)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * t + (1 - alpha) * (1 - t)
        loss = alpha_t * loss
    if reduction == "mean":
        return loss.mean()
    return loss.sum()


def run_case(op, x, target, params, weight=None, kind="f32"):
    """`x` を requires_grad にして損失と勾配を得る。"""
    xt = x.clone().float().requires_grad_(True)
    r = params["reduction"]
    if op == "multi_margin":
        loss = F.multi_margin_loss(xt, target, p=params["p"], margin=params["margin"],
                                   weight=weight, reduction=r)
    elif op == "multilabel_margin":
        loss = F.multilabel_margin_loss(xt, target, reduction=r)
    elif op == "multilabel_soft_margin":
        loss = F.multilabel_soft_margin_loss(xt, target, weight=weight, reduction=r)
    elif op == "sigmoid_focal":
        loss = sigmoid_focal(xt, target, params["alpha"] if params["alpha"] is not None else -1.0,
                             params["gamma"], r)
    else:
        raise AssertionError(op)
    loss.backward()
    return xt, loss


def case(name, op, shape, params, x, target, weight=None):
    xt, loss = run_case(op, x, target, params, weight)
    out = {
        "name": name, "op": op, "shape": shape, "params": params,
        "loss": float(loss.detach()),
        "x": lst(x),
        "target": ilst(target) if target.dtype in (torch.int64, torch.int32) else lst(target),
        "target_shape": list(target.shape),
        "weight": lst(weight) if weight is not None else None,
        "grad_x": lst(xt.grad),
    }
    assert torch.isfinite(x).all() and math.isfinite(out["loss"])
    assert torch.isfinite(xt.grad).all(), name
    cases.append(out)


# --- multi_margin -----------------------------------------------------------
def check_no_boundary_mm(x2, target, margin):
    for i in range(x2.shape[0]):
        y = int(target[i])
        z = margin - x2[i, y] + x2[i]
        z = torch.cat([z[:y], z[y + 1:]])
        assert not (z == 0).any()


for shape in ([5], [4, 3], [6, 5]):
    for p in (1, 2):
        for margin in (1.0, 0.5):
            for use_w in (False, True):
                for r in REDUCTIONS:
                    x = sample(shape)
                    c = shape[-1]
                    if len(shape) == 1:
                        target = torch.randint(0, c, ()).long()
                        check_no_boundary_mm(x.reshape(1, c), target.reshape(1), margin)
                    else:
                        target = torch.randint(0, c, (shape[0],)).long()
                        check_no_boundary_mm(x, target, margin)
                    w = (torch.rand(c) * 2.0 + 0.1) if use_w else None
                    case(f"mm_{name_of(shape)}_p{p}_m{margin}_{'w' if use_w else 'nw'}_{r}",
                         "multi_margin", shape,
                         {"reduction": r, "p": p, "margin": margin, "has_weight": use_w,
                          "alpha": None, "gamma": 0.0},
                         x, target, w)

# --- multilabel_margin ------------------------------------------------------
def make_ml_targets(n, c, ks):
    rows = []
    for i in range(n):
        k = ks[i % len(ks)]
        perm = torch.randperm(c)[:k].tolist()
        row = perm + [-1] * (c - k)
        # `-1` 以降にゴミ値（無視される）を入れる行を混ぜる。
        if 0 < k < c - 1 and i % 2 == 1:
            row[k + 1] = int(torch.randint(0, c, ()).item())
        rows.append(row)
    return torch.tensor(rows, dtype=torch.long)


for shape in ([5], [4, 5], [3, 6]):
    for r in REDUCTIONS:
        x = sample(shape)
        if len(shape) == 1:
            target = make_ml_targets(1, shape[0], [2])[0]
        else:
            n, c = shape
            # 個数 0（全 -1）・C（終端なし）・中間を散らす
            target = make_ml_targets(n, c, [0, c, 2, 1, 3])
        case(f"mlm_{name_of(shape)}_{r}", "multilabel_margin", shape,
             {"reduction": r, "p": 1, "margin": 1.0, "has_weight": False,
              "alpha": None, "gamma": 0.0},
             x, target)

# 全クラス target・全 -1 行・ゴミ値行を確実に含む決め打ちケース
for r in REDUCTIONS:
    x = sample([4, 5])
    target = torch.tensor([[-1, 3, 3, 3, 3],
                           [0, 1, 2, 3, 4],
                           [2, 0, -1, 4, 1],
                           [4, -1, 0, -1, 2]], dtype=torch.long)
    # 行 0 は先頭が -1（空集合）。行 1 は全クラス。行 2/3 は -1 以降にゴミ値。
    case(f"mlm_fixed_4x5_{r}", "multilabel_margin", [4, 5],
         {"reduction": r, "p": 1, "margin": 1.0, "has_weight": False,
          "alpha": None, "gamma": 0.0},
         x, target)

# --- multilabel_soft_margin -------------------------------------------------
for shape in ([5], [4, 3], [6, 5]):
    for use_w in (False, True):
        for soft in (False, True):
            for r in REDUCTIONS:
                x = sample(shape, -4.0, 4.0)
                c = shape[-1]
                if soft:
                    target = torch.rand(numel(shape)).reshape(shape)
                else:
                    target = torch.randint(0, 2, (numel(shape),)).reshape(shape).float()
                w = (torch.rand(c) * 2.0 + 0.1) if use_w else None
                case(f"mlsm_{name_of(shape)}_{'w' if use_w else 'nw'}_{'soft' if soft else 'hard'}_{r}",
                     "multilabel_soft_margin", shape,
                     {"reduction": r, "p": 1, "margin": 1.0, "has_weight": use_w,
                      "alpha": None, "gamma": 0.0},
                     x, target, w)

# --- sigmoid_focal ----------------------------------------------------------
ALPHA_GAMMA = [(0.25, 2.0), (None, 2.0), (0.5, 0.0), (None, 0.0), (0.75, 1.0),
               (0.25, 0.5), (0.25, 3.5)]
for shape in ([6], [4, 3], [2, 3, 4]):
    for alpha, gamma in ALPHA_GAMMA:
        for soft in (False, True):
            for r in REDUCTIONS:
                x = sample(shape, -4.0, 4.0)
                if soft:
                    target = torch.rand(numel(shape)).reshape(shape)
                else:
                    target = torch.randint(0, 2, (numel(shape),)).reshape(shape).float()
                case(f"focal_{name_of(shape)}_a{alpha}_g{gamma}_{'soft' if soft else 'hard'}_{r}",
                     "sigmoid_focal", shape,
                     {"reduction": r, "p": 1, "margin": 1.0, "has_weight": False,
                      "alpha": alpha, "gamma": gamma},
                     x, target)


# --- edge_cases -------------------------------------------------------------
def edge(name, op, shape, params, x, target, weight=None, diverges=None):
    xs = torch.tensor(x, dtype=torch.float32).reshape(shape)
    if op in ("multi_margin", "multilabel_margin"):
        tg = torch.tensor(target, dtype=torch.long)
        if op == "multi_margin":
            tg = tg.reshape(()) if len(shape) == 1 else tg.reshape(shape[0])
        else:
            tg = tg.reshape(shape)
        tj = [int(v) for v in target]
    else:
        tg = torch.tensor(target, dtype=torch.float32).reshape(shape)
        tj = [klass(float(v)) for v in target]
    w = torch.tensor(weight, dtype=torch.float32) if weight is not None else None
    full = {"reduction": "sum", "p": 1, "margin": 1.0, "has_weight": w is not None,
            "alpha": None, "gamma": 0.0}
    full.update(params)
    xt, loss = run_case(op, xs, tg, full, w)
    e = {"name": name, "op": op, "shape": shape, "params": full,
         "loss": klass(float(loss.detach())),
         "x": [klass(float(a)) for a in x],
         "target": tj,
         "weight": list(weight) if weight is not None else None,
         "grad_x": klst(xt.grad)}
    if diverges:
        e["diverges"] = diverges
    edge_cases.append(e)


# multi_margin: z == 0 ちょうど（p = 1, 2）
for p in (1, 2):
    edge(f"mm_boundary_p{p}", "multi_margin", [3], {"p": p}, [2.0, 1.0, 0.5], [0])
# multi_margin: NaN／inf
edge("mm_nan_inf", "multi_margin", [4, 3], {"p": 1},
     [NAN, 0.5, 0.2, 0.1, INF, 0.3, 0.4, -INF, 0.0, 0.2, 0.3, NAN], [0, 1, 2, 1])
edge("mm_nan_inf_p2", "multi_margin", [3, 3], {"p": 2},
     [NAN, 0.5, 0.2, 0.1, INF, 0.3, 0.4, -INF, 0.0], [0, 1, 2])
# multilabel_margin: 境界・重複・全 -1・全クラス・-1 以降のゴミ・NaN／inf
edge("mlm_boundary", "multilabel_margin", [4], {}, [2.0, 1.0, 0.5, 1.0], [0, -1, 0, 0])
edge("mlm_duplicate_target", "multilabel_margin", [4], {}, [0.2, 0.5, 0.1, 0.9], [0, 0, 1, -1])
edge("mlm_all_minus1_row", "multilabel_margin", [2, 3], {},
     [0.2, 0.5, 0.1, 0.9, 0.3, 0.7], [-1, 0, 1, 1, -1, 2])
edge("mlm_all_classes_row", "multilabel_margin", [1, 4], {}, [0.2, 0.5, 0.1, 0.9],
     [0, 1, 2, 3])
edge("mlm_garbage_after_terminator", "multilabel_margin", [1, 5], {},
     [0.2, 0.5, 0.1, 0.9, 0.3], [1, -1, 2, 4, 0])
edge("mlm_nan_inf", "multilabel_margin", [2, 4], {},
     [NAN, 0.5, 0.1, 0.9, INF, -INF, 0.3, 0.2], [0, 1, -1, 0, 2, -1, 0, 0])
# multilabel_soft_margin: ±inf（0·inf の NaN）・大振幅・NaN
edge("mlsm_inf", "multilabel_soft_margin", [3], {}, [INF, -INF, 0.0], [0.0, 1.0, 0.5])
edge("mlsm_inf_match", "multilabel_soft_margin", [2], {}, [INF, -INF], [1.0, 0.0])
edge("mlsm_big", "multilabel_soft_margin", [4], {}, [100.0, -100.0, 100.0, -100.0],
     [0.0, 1.0, 1.0, 0.0])
edge("mlsm_nan", "multilabel_soft_margin", [2], {}, [NAN, 1.0], [0.5, 0.5])
# focal: 飽和域。γ < 1 では PyTorch の f32 評価が `1 - p_t == 0`（pow の逆伝播が 0·inf）となり
# 勾配が NaN になる。本実装は `q = 1 - p_t` を桁落ちしない形で f64 評価するため有限値（真の勾配は 0）を
# 返す。この差分にのみ `diverges` を付ける（決定記録 §5）。
SAT_DIVERGES = (
    "PyTorch は f32 で 1 - p_t を直接評価するため飽和域（|x| >= 約 17）で 1 - p_t が 0 に潰れ、"
    "pow(·, γ) の逆伝播（γ < 1 では ∞·0）が NaN になる。本実装は q = 1 - p_t を "
    "t·σ(-x) + (1-t)·σ(x) の桁落ちしない形で f64 評価し、真の勾配 0 に近い有限値を返す"
)
for gamma in (0.0, 0.5, 1.0, 2.0):
    edge(f"focal_saturated_g{gamma}", "sigmoid_focal", [6], {"alpha": 0.25, "gamma": gamma},
         [40.0, -40.0, 40.0, -40.0, 20.0, -20.0], [0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
         diverges=SAT_DIVERGES if gamma == 0.5 else None)
# ±inf は 1 要素ずつ個別に保存する（要素間で NaN が混ざり差分の所在が不明になるのを避ける）。
edge("focal_inf_pos_t1", "sigmoid_focal", [1], {"alpha": 0.25, "gamma": 2.0}, [INF], [1.0])
edge("focal_inf_neg_t0", "sigmoid_focal", [1], {"alpha": 0.25, "gamma": 2.0}, [-INF], [0.0])
edge("focal_inf_pos_t0", "sigmoid_focal", [1], {"alpha": 0.25, "gamma": 2.0}, [INF], [0.0])
edge("focal_inf_neg_t1", "sigmoid_focal", [1], {"alpha": 0.25, "gamma": 2.0}, [-INF], [1.0])
edge("focal_nan", "sigmoid_focal", [2], {"alpha": 0.25, "gamma": 2.0}, [NAN, 0.5], [1.0, 0.0])

fixture = {
    "torch_version": torch.__version__,
    "python_version": platform.python_version(),
    "seed": SEED,
    "cases": cases,
    "edge_cases": edge_cases,
}
json.dump(fixture, sys.stdout, indent=1, allow_nan=False)
print()
