#!/usr/bin/env python3
"""結合 4 演算（Concatenate／Add／Multiply／Average）の PyTorch 参照値を生成する。

イシュー #2666（親 #2663）の `tests/merge_ops_parity.rs` が参照する固定フィクスチャ
`merge_ops_reference.json` の生成スクリプト。CI は Python／PyTorch に依存せず、
コミット済み JSON のみを読む（`README.md` 参照）。

f32 はすべて u32 ビットパターン配列で保存する。入力・上流勾配 `g`・出力・入力勾配を
すべて保存し、Rust 側で入力を再生成しない。損失は `(y * g).sum()`。dtype は float32。
参照式: Concatenate は `torch.cat`、Add は `x1 + x2 + ...`、Multiply は `x1 * x2 * ...`、
Average は `(x1 + ... + xn) / n` の左畳み込み（`torch.stack(..).mean(0)` は使わない）。
`uses` は結合の各引数がどの葉テンソルかの添字で、同一テンソルの重複指定を表せる。
"""

import json
import platform
import sys

import torch

assert torch.__version__.startswith("2.14.0"), torch.__version__

SEED = 2666


def bits_of(t):
    flat = t.detach().contiguous().reshape(-1)
    if flat.numel() == 0:
        return []
    return [v & 0xFFFFFFFF for v in flat.view(torch.int32).tolist()]


def numel(shape):
    n = 1
    for s in shape:
        n *= s
    return n


def merge(op, dim, views):
    if op == "concatenate":
        return torch.cat(views, dim=dim)
    if op == "add":
        acc = views[0]
        for v in views[1:]:
            acc = acc + v
        return acc
    if op == "multiply":
        acc = views[0]
        for v in views[1:]:
            acc = acc * v
        return acc
    if op == "average":
        acc = views[0]
        for v in views[1:]:
            acc = acc + v
        return acc / float(len(views))
    raise AssertionError(op)


def record(name, op, dim, shapes, gen, uses=None, pre_transpose=None):
    leaves = [
        torch.randn(tuple(s), generator=gen, dtype=torch.float32).requires_grad_(True)
        for s in shapes
    ]
    if uses is None:
        uses = list(range(len(leaves)))
    views = [leaves[u] for u in uses]
    if pre_transpose is not None:
        views[0] = leaves[uses[0]].transpose(pre_transpose[0], pre_transpose[1])
    y = merge(op, dim, views)
    g = torch.randn(list(y.shape), generator=gen, dtype=torch.float32)
    loss = (y * g).sum()
    loss.backward()
    return {
        "name": name,
        "op": op,
        "dim": dim,
        "uses": uses,
        "pre_transpose": pre_transpose,
        "inputs": [{"shape": list(l.shape), "x_bits": bits_of(l)} for l in leaves],
        "out": {"shape": list(y.shape), "bits": bits_of(y), "g_bits": bits_of(g)},
        "grads": [bits_of(l.grad) if l.grad is not None else [0] * numel(l.shape) for l in leaves],
    }


def try_raises(op, dim, shapes):
    xs = [torch.zeros(tuple(s), dtype=torch.float32) for s in shapes]
    try:
        y = merge(op, dim, xs)
        return False, list(y.shape)
    except Exception:
        return True, None


def sn(shape):
    return "x".join(map(str, shape)) if len(shape) > 0 else "scalar"


def main():
    gen = torch.Generator().manual_seed(SEED)
    cases = []

    elementwise = ["add", "multiply", "average"]
    same_shapes = [(), (5,), (2, 3), (1, 4), (2, 1, 3), (2, 2, 3, 2)]
    for op in elementwise:
        for shape in same_shapes:
            for n in (2, 3, 7):
                cases.append(record(f"{op}_n{n}_{sn(shape)}", op, 0, [shape] * n, gen))
        # 多数件（MAX_FUSED_CHAIN_LEN を超える畳み込み連鎖）
        cases.append(record(f"{op}_n12_2x3", op, 0, [(2, 3)] * 12, gen))
        # 非 contiguous 入力（先頭入力を transpose した view）
        cases.append(record(f"{op}_pre_transpose_2x3", op, 0, [(3, 2), (2, 3), (2, 3)], gen,
                            pre_transpose=(0, 1)))
        # 同一テンソルの重複
        cases.append(record(f"{op}_dup_x_x_2x3", op, 0, [(2, 3)], gen, uses=[0, 0]))
        cases.append(record(f"{op}_dup_x_y_x_2x3", op, 0, [(2, 3), (2, 3)], gen, uses=[0, 1, 0]))

    # concatenate: 各 dim・不揃い軸長・軸長 1・軸長 0・rank 1〜4
    for shape_list, dim in [
        ([(2,), (3,)], 0),
        ([(2,), (3,), (1,)], 0),
        ([(2, 3), (4, 3)], 0),
        ([(2, 3), (2, 5)], 1),
        ([(2, 3), (2, 1), (2, 4)], 1),
        ([(1, 3), (1, 3)], 0),
        ([(2, 3, 4), (2, 5, 4)], 1),
        ([(2, 3, 4), (2, 3, 1), (2, 3, 2)], 2),
        ([(2, 3, 4), (1, 3, 4)], 0),
        ([(2, 2, 3, 2), (2, 2, 3, 3)], 3),
        ([(2, 2, 3, 2), (2, 4, 3, 2)], 1),
        ([(2, 3)] * 9, 1),
        ([(0, 3), (2, 3)], 0),
    ]:
        nm = "_".join(sn(s) for s in shape_list[:4])
        cases.append(record(f"concatenate_dim{dim}_{nm}_n{len(shape_list)}", "concatenate", dim, shape_list, gen))
    cases.append(record("concatenate_pre_transpose", "concatenate", 1, [(3, 2), (2, 4)], gen,
                        pre_transpose=(0, 1)))
    cases.append(record("concatenate_dup_x_x", "concatenate", 0, [(2, 3)], gen, uses=[0, 0]))
    cases.append(record("concatenate_dup_x_y_x", "concatenate", 1, [(2, 3), (2, 2)], gen, uses=[0, 1, 0]))

    # torch が受理するが本実装が拒否するもの、両方が拒否するものを記録
    error_specs = [
        ("add_broadcastable", "add", 0, [(2, 3), (1, 3)]),
        ("multiply_broadcastable", "multiply", 0, [(2, 3), (3,)]),
        ("average_broadcastable", "average", 0, [(2, 3), (2, 1)]),
        ("add_mismatch", "add", 0, [(2, 3), (2, 4)]),
        ("multiply_mismatch", "multiply", 0, [(2, 3), (3, 2)]),
        ("average_mismatch", "average", 0, [(2, 3), (4, 3)]),
        ("add_single_input", "add", 0, [(2, 3)]),
        ("multiply_single_input", "multiply", 0, [(2, 3)]),
        ("average_single_input", "average", 0, [(2, 3)]),
        ("concatenate_single_input", "concatenate", 0, [(2, 3)]),
        ("concatenate_dim_oor", "concatenate", 2, [(2, 3), (2, 3)]),
        ("concatenate_other_axis_mismatch", "concatenate", 0, [(2, 3), (2, 4)]),
        ("concatenate_rank_mismatch", "concatenate", 0, [(2, 3), (2, 3, 1)]),
    ]
    error_cases = []
    for name, op, dim, shapes in error_specs:
        raises, shape = try_raises(op, dim, shapes)
        error_cases.append({
            "name": name,
            "op": op,
            "dim": dim,
            "input_shapes": [list(s) for s in shapes],
            "torch_raises": raises,
            "torch_out_shape": shape,
        })

    doc = {
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "seed": SEED,
        "cases": cases,
        "error_cases": error_cases,
    }
    json.dump(doc, sys.stdout, separators=(",", ":"))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
