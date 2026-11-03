#!/usr/bin/env python3
"""cosine_similarity・pairwise_distance の PyTorch 参照値（forward 出力と入力勾配）を生成する。

イシュー #3821 の `tests/similarity_distance_pytorch_parity.rs` が参照する固定フィクスチャ
`similarity_distance_reference.json` の生成スクリプト。CI は Python／PyTorch に依存せず
コミット済み JSON のみを読む（`README.md` 参照）。

損失は `(out * g).sum()`（`g` は固定シードの上流勾配）。入力・`g`・出力・勾配・shape・
パラメータをすべて JSON に保存し、Rust 側で入力を再生成しない。dtype は float32。
`cases` は NaN／inf／kink（差が 0 の要素）を含まない。`clamp_cases` は
`0 < ‖x‖ < eps`（クランプ内）と `‖x‖ == 0` を含み、PyTorch の straight-through 勾配を
そのまま保存する（設計記録 `docs/autodiff-similarity-distance-decision.md` §5）。
"""

import json
import platform
import sys

import torch
import torch.nn.functional as F

EXPECTED_TORCH_PREFIX = "2.14.0"
assert torch.__version__.startswith(EXPECTED_TORCH_PREFIX), torch.__version__

torch.manual_seed(3821)


def lst(t):
    return [float(v) for v in t.flatten().tolist()]


def run(fn, x1, x2, g_seed):
    x1 = x1.clone().requires_grad_(True)
    x2 = x2.clone().requires_grad_(True)
    out = fn(x1, x2)
    g = torch.empty_like(out).uniform_(-1.0, 1.0)
    (out * g).sum().backward()
    return out.detach(), g, x1.grad, x2.grad


def case(name, op, x1, x2, params, fn):
    out, g, gx1, gx2 = run(fn, x1, x2, 0)
    for t in (x1, x2, out, gx1, gx2):
        assert torch.isfinite(t).all(), name
    return {
        "name": name,
        "op": op,
        "params": params,
        "shape": list(x1.shape),
        "out_shape": list(out.shape),
        "x1": lst(x1),
        "x2": lst(x2),
        "g": lst(g),
        "out": lst(out),
        "grad_x1": lst(gx1),
        "grad_x2": lst(gx2),
    }


cases = []
for shape, dims in [([5], [0]), ([3, 4], [0, 1]), ([2, 3, 4], [0, 1, 2])]:
    for dim in dims:
        x1 = torch.randn(shape)
        x2 = torch.randn(shape)
        cases.append(
            case(
                f"cosine_{'x'.join(map(str, shape))}_dim{dim}",
                "cosine_similarity",
                x1,
                x2,
                {"dim": dim, "eps": 1e-8},
                lambda a, b, dim=dim: F.cosine_similarity(a, b, dim=dim, eps=1e-8),
            )
        )

for shape in [[4], [3, 5], [2, 3, 4]]:
    for p in [1.0, 2.0, 3.0]:
        for keepdim in [False, True]:
            x1 = torch.randn(shape)
            x2 = torch.randn(shape)
            cases.append(
                case(
                    f"pairwise_{'x'.join(map(str, shape))}_p{p}_kd{int(keepdim)}",
                    "pairwise_distance",
                    x1,
                    x2,
                    {"p": p, "eps": 1e-6, "keepdim": keepdim},
                    lambda a, b, p=p, keepdim=keepdim: F.pairwise_distance(
                        a, b, p=p, eps=1e-6, keepdim=keepdim
                    ),
                )
            )

# eps = 0（pairwise は eps を足さない素の距離）
x1 = torch.randn(3, 4)
x2 = torch.randn(3, 4)
cases.append(
    case(
        "pairwise_3x4_p2_eps0",
        "pairwise_distance",
        x1,
        x2,
        {"p": 2.0, "eps": 0.0, "keepdim": False},
        lambda a, b: F.pairwise_distance(a, b, p=2.0, eps=0.0),
    )
)

# クランプ内: 行 0 は 0 < ‖x1‖ < eps、行 1 は x1 がゼロベクトル、行 2 は通常。eps = 1.0。
x1 = torch.tensor(
    [[0.3, 0.4, 0.0], [0.0, 0.0, 0.0], [1.5, -0.5, 2.0]], dtype=torch.float32
)
x2 = torch.tensor(
    [[1.0, 2.0, -1.0], [0.5, -1.5, 2.0], [-1.0, 0.7, 0.2]], dtype=torch.float32
)
clamp_cases = [
    case(
        "cosine_clamp_eps1",
        "cosine_similarity",
        x1,
        x2,
        {"dim": 1, "eps": 1.0},
        lambda a, b: F.cosine_similarity(a, b, dim=1, eps=1.0),
    ),
    case(
        "cosine_zero_norm_default_eps",
        "cosine_similarity",
        x1,
        x2,
        {"dim": 1, "eps": 1e-8},
        lambda a, b: F.cosine_similarity(a, b, dim=1, eps=1e-8),
    ),
]

doc = {
    "torch_version": torch.__version__,
    "python_version": platform.python_version(),
    "seed": 3821,
    "cases": cases,
    "clamp_cases": clamp_cases,
}
json.dump(doc, sys.stdout, indent=None, separators=(",", ":"))
sys.stdout.write("\n")
