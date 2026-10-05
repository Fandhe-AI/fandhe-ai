#!/usr/bin/env python3
"""unbind／movedim／swapaxes／tensor_split／meshgrid／rot90 の PyTorch 参照値を生成する。

イシュー #2639（親 #2625）の `tests/shape_view_parity.rs` が参照する固定
フィクスチャ `shape_view_reference.json` の生成スクリプト。CI は Python／
PyTorch に依存せず、コミット済み JSON のみを読む（`README.md` 参照）。

f32 はすべて u32 ビットパターン配列で保存する（JSON は NaN／inf を運べない
ため。NaN の payload も保存する）。入力・上流勾配 `g`・各出力・入力勾配を
すべて保存し、Rust 側で入力を再生成しない。損失は `Σ_k (y_k * g_k).sum()`。
dtype は float32。
"""

import json
import platform
import struct
import sys

import torch

EXPECTED_TORCH_PREFIX = "2.14.0"
assert torch.__version__.startswith(EXPECTED_TORCH_PREFIX), torch.__version__

SEED = 2639

# 非有限値の入力に差し込むビットパターン（NaN payload 付き・±inf・-0.0）。
SPECIAL_BITS = [0x7FC00001, 0x7F800000, 0xFF800000, 0x80000000, 0xFFC00002, 0x00000000]


def bits_of(t):
    """f32 テンソルを u32 ビットパターン配列へ（NaN payload を含め厳密）。"""
    flat = t.detach().contiguous().reshape(-1)
    if flat.numel() == 0:
        return []
    return [v & 0xFFFFFFFF for v in flat.view(torch.int32).tolist()]


def from_bits(bits, shape):
    if len(bits) == 0:
        return torch.zeros(tuple(shape), dtype=torch.float32)
    buf = bytearray(struct.pack("<%dI" % len(bits), *bits))
    return torch.frombuffer(buf, dtype=torch.float32).clone().reshape(shape)


def rnd(shape, gen):
    return torch.randn(tuple(shape), generator=gen, dtype=torch.float32)


def numel(shape):
    n = 1
    for s in shape:
        n *= s
    return n


def apply(op, p, xs):
    """演算を実行し出力テンソルのリストを返す。"""
    if op == "unbind":
        return list(torch.unbind(xs[0], p["dim"]))
    if op == "movedim":
        return [torch.movedim(xs[0], p["source"], p["destination"])]
    if op == "swapaxes":
        return [torch.swapaxes(xs[0], p["a"], p["b"])]
    if op == "tensor_split":
        return list(torch.tensor_split(xs[0], p["sections"], p["dim"]))
    if op == "tensor_split_indices":
        return list(torch.tensor_split(xs[0], p["indices"], p["dim"]))
    if op == "meshgrid":
        return list(torch.meshgrid(*xs, indexing=p["indexing"]))
    if op == "rot90":
        return [torch.rot90(xs[0], p["k"], p["dims"])]
    raise AssertionError(op)


def record(name, op, params, shapes, gen, pre_transpose=None, nonfinite=False):
    leaves = []
    for shape in shapes:
        x0 = rnd(shape, gen)
        if nonfinite:
            n = numel(shape)
            bits = bits_of(x0)
            for j in range(0, n, 2):
                bits[j] = SPECIAL_BITS[(j // 2) % len(SPECIAL_BITS)]
            x0 = from_bits(bits, shape)
        leaves.append(x0.clone().requires_grad_(True))
    views = list(leaves)
    if pre_transpose is not None:
        views[0] = leaves[0].transpose(pre_transpose[0], pre_transpose[1])
    outs = apply(op, params, views)
    loss = torch.zeros((), dtype=torch.float32)
    out_recs = []
    for y in outs:
        g = torch.randn(list(y.shape), generator=gen, dtype=torch.float32)
        loss = loss + (y * g).sum()
        out_recs.append(
            {"shape": list(y.shape), "bits": bits_of(y), "g_bits": bits_of(g)}
        )
    if len(outs) > 0:
        loss.backward()
    grads = []
    for leaf in leaves:
        grads.append(
            bits_of(leaf.grad) if leaf.grad is not None else [0] * numel(leaf.shape)
        )
    return {
        "name": name,
        "op": op,
        "params": params,
        "pre_transpose": pre_transpose,
        "inputs": [{"shape": list(l.shape), "x_bits": bits_of(l)} for l in leaves],
        "outs": out_recs,
        "grads": grads,
    }


def try_raises(op, params, shapes):
    xs = [torch.zeros(tuple(s), dtype=torch.float32) for s in shapes]
    try:
        outs = apply(op, params, xs)
        return False, [list(o.shape) for o in outs]
    except Exception:
        return True, None


def main():
    gen = torch.Generator().manual_seed(SEED)
    cases = []

    def shape_name(shape):
        return "x".join(map(str, shape)) if len(shape) > 0 else "scalar"

    # unbind
    for shape in [(3,), (2, 3), (2, 3, 4), (1, 3), (3, 1), (0, 3), (2, 0, 3)]:
        for dim in range(len(shape)):
            cases.append(record(f"unbind_{shape_name(shape)}_d{dim}", "unbind", {"dim": dim}, [shape], gen))
    for dim in range(3):
        cases.append(record(f"unbind_t02_342_d{dim}", "unbind", {"dim": dim}, [(3, 2, 4)], gen, pre_transpose=(0, 2)))
    for dim in range(3):
        cases.append(record(f"unbind_nonfinite_d{dim}", "unbind", {"dim": dim}, [(2, 3, 4)], gen, nonfinite=True))

    # movedim
    md = [
        ((3, 4), [0], [1]),
        ((2, 3, 4, 5), [0], [2]),
        ((2, 3, 4, 5), [3], [0]),
        ((2, 3, 4, 5), [1], [1]),
        ((2, 3, 4, 5), [0, 1], [2, 3]),
        ((2, 3, 4, 5), [0, 1], [1, 0]),
        ((2, 3, 4, 5), [2, 0], [0, 2]),
        ((2, 3, 4, 5), [0, 1, 2, 3], [3, 2, 1, 0]),
        ((2, 3, 4, 5), [3, 1], [0, 1]),
        ((2, 3, 4), [], []),
    ]
    for i, (shape, s, d) in enumerate(md):
        cases.append(record(f"movedim_{i}", "movedim", {"source": s, "destination": d}, [shape], gen))
    cases.append(record("movedim_t01", "movedim", {"source": [0], "destination": [2]}, [(3, 2, 4)], gen, pre_transpose=(0, 1)))
    cases.append(record("movedim_nonfinite", "movedim", {"source": [2, 0], "destination": [0, 1]}, [(2, 3, 4)], gen, nonfinite=True))

    # swapaxes
    for i, (shape, a, b) in enumerate([
        ((2, 3, 4), 0, 2), ((2, 3, 4), 1, 1), ((2, 3, 4), 2, 0), ((2, 3, 4), 0, 1),
        ((3, 4), 0, 1), ((5,), 0, 0), ((2, 0, 3), 0, 1),
    ]):
        cases.append(record(f"swapaxes_{i}", "swapaxes", {"a": a, "b": b}, [shape], gen))
    cases.append(record("swapaxes_nonfinite", "swapaxes", {"a": 0, "b": 1}, [(2, 3)], gen, nonfinite=True))

    # tensor_split（分割数）
    for i, (shape, sec, dim) in enumerate([
        ((7,), 3, 0), ((6,), 3, 0), ((2,), 5, 0), ((0,), 3, 0), ((7,), 1, 0),
        ((2, 7, 3), 4, 1), ((4, 3), 2, 0), ((5, 4), 4, 1), ((3, 2), 7, 0),
    ]):
        cases.append(record(f"tensor_split_{i}", "tensor_split", {"sections": sec, "dim": dim}, [shape], gen))
    cases.append(record("tensor_split_t01", "tensor_split", {"sections": 3, "dim": 1}, [(4, 7)], gen, pre_transpose=(0, 1)))
    cases.append(record("tensor_split_nonfinite", "tensor_split", {"sections": 3, "dim": 0}, [(7, 2)], gen, nonfinite=True))

    # tensor_split（境界添字列）
    idx_cases = [
        ((10,), [], 0), ((10,), [3], 0), ((10,), [2, 5, 7], 0), ((10,), [5, 3], 0),
        ((10,), [0, 10], 0), ((10,), [12], 0), ((10,), [3, 3], 0), ((10,), [2, 100], 0),
        ((10,), [10], 0), ((10,), [0], 0), ((10,), [100, 2], 0), ((10,), [7, 3, 9], 0),
        ((0,), [1, 2], 0), ((2, 8, 3), [1, 4, 6], 1), ((4, 3), [1, 3], 0),
        ((3, 6), [4, 2], 1),
    ]
    for i, (shape, idx, dim) in enumerate(idx_cases):
        cases.append(record(f"tensor_split_indices_{i}", "tensor_split_indices", {"indices": idx, "dim": dim}, [shape], gen))
    cases.append(record("tensor_split_indices_nonfinite", "tensor_split_indices", {"indices": [2, 5], "dim": 0}, [(7, 2)], gen, nonfinite=True))

    # meshgrid
    mg_shapes = [
        [(3,)], [(3,), (2,)], [(3,), (2,), (4,)], [(), (3,)], [(1,), (3,)],
        [(0,), (2,)], [(2,), (3,), (0,)], [(2,), ()], [()], [(2,), (3,), (2,), (2,)],
    ]
    for indexing in ["ij", "xy"]:
        for i, shapes in enumerate(mg_shapes):
            cases.append(record(f"meshgrid_{indexing}_{i}", "meshgrid", {"indexing": indexing}, shapes, gen))
        cases.append(record(f"meshgrid_{indexing}_nonfinite", "meshgrid", {"indexing": indexing}, [(3,), (2,)], gen, nonfinite=True))

    # rot90
    for shape in [(3, 4), (3, 3), (0, 3), (2, 3, 4)]:
        dims_list = [[0, 1], [1, 0]]
        if len(shape) == 3:
            dims_list += [[0, 2], [2, 1]]
        for dims in dims_list:
            for k in [-5, -2, -1, 0, 1, 2, 3, 4, 5]:
                cases.append(record(f"rot90_{shape_name(shape)}_d{dims[0]}{dims[1]}_k{k}", "rot90", {"k": k, "dims": dims}, [shape], gen))
    for k in [1, 2, 3]:
        cases.append(record(f"rot90_nonfinite_k{k}", "rot90", {"k": k, "dims": [0, 1]}, [(3, 4)], gen, nonfinite=True))
    for k in [1, 3]:
        cases.append(record(f"rot90_t01_k{k}", "rot90", {"k": k, "dims": [0, 1]}, [(4, 3)], gen, pre_transpose=(0, 1)))

    # error_cases
    err_specs = [
        ("unbind_rank0", "unbind", {"dim": 0}, [()]),
        ("unbind_axis_oor", "unbind", {"dim": 5}, [(2, 3)]),
        ("movedim_len_mismatch", "movedim", {"source": [0, 1], "destination": [0]}, [(2, 3, 4)]),
        ("movedim_dup_source", "movedim", {"source": [0, 0], "destination": [1, 2]}, [(2, 3, 4)]),
        ("movedim_dup_dest", "movedim", {"source": [0, 1], "destination": [2, 2]}, [(2, 3, 4)]),
        ("movedim_src_oor", "movedim", {"source": [3], "destination": [0]}, [(2, 3, 4)]),
        ("movedim_dst_oor", "movedim", {"source": [0], "destination": [3]}, [(2, 3, 4)]),
        ("swapaxes_oor", "swapaxes", {"a": 0, "b": 3}, [(2, 3, 4)]),
        ("swapaxes_rank0", "swapaxes", {"a": 0, "b": 0}, [()]),
        ("tensor_split_sections0", "tensor_split", {"sections": 0, "dim": 0}, [(4,)]),
        ("tensor_split_rank0", "tensor_split", {"sections": 2, "dim": 0}, [()]),
        ("tensor_split_axis_oor", "tensor_split", {"sections": 2, "dim": 3}, [(4,)]),
        ("tensor_split_indices_rank0", "tensor_split_indices", {"indices": [1], "dim": 0}, [()]),
        ("tensor_split_indices_axis_oor", "tensor_split_indices", {"indices": [1], "dim": 2}, [(4,)]),
        ("meshgrid_rank2_input", "meshgrid", {"indexing": "ij"}, [(2, 3), (2,)]),
        ("rot90_dims_same", "rot90", {"k": 1, "dims": [0, 0]}, [(3, 4)]),
        ("rot90_dims_oor", "rot90", {"k": 1, "dims": [0, 2]}, [(3, 4)]),
        ("rot90_rank1", "rot90", {"k": 1, "dims": [0, 1]}, [(4,)]),
    ]
    error_cases = []
    for name, op, params, shapes in err_specs:
        raises, out_shapes = try_raises(op, params, shapes)
        error_cases.append({
            "name": name, "op": op, "params": params,
            "input_shapes": [list(s) for s in shapes],
            "torch_raises": raises, "out_shapes": out_shapes,
        })
    # 入力リストが空の meshgrid は torch.meshgrid() が引数なし呼び出しになる。
    try:
        torch.meshgrid(indexing="ij")
        empty_raises = False
    except Exception:
        empty_raises = True
    error_cases.append({
        "name": "meshgrid_empty", "op": "meshgrid", "params": {"indexing": "ij"},
        "input_shapes": [], "torch_raises": empty_raises, "out_shapes": None,
    })

    json.dump({
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "seed": SEED,
        "cases": cases,
        "error_cases": error_cases,
    }, sys.stdout, separators=(",", ":"))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
