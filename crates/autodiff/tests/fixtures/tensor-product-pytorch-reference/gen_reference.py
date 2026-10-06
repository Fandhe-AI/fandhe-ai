#!/usr/bin/env python3
"""kron／tensordot／cdist／cross の PyTorch 参照値を生成する。

イシュー #2640（親 #2625）の `tests/tensor_product_parity.rs` が参照する固定
フィクスチャ `tensor_product_reference.json` の生成スクリプト。CI は Python／
PyTorch に依存せず、コミット済み JSON のみを読む（`README.md` 参照）。

f32 はすべて u32 ビットパターン配列で保存する（JSON は NaN／inf を運べないため）。
入力・上流勾配 `g`・出力・入力勾配をすべて保存し、Rust 側で入力を再生成しない。
損失は `(y * g).sum()`。dtype は float32。
"""

import json
import platform
import struct
import sys

import torch

assert torch.__version__.startswith("2.14.0"), torch.__version__

SEED = 2640
SPECIAL_BITS = [0x7FC00001, 0x7F800000, 0xFF800000, 0x80000000, 0xFFC00002, 0x00000000]


def bits_of(t):
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
    if op == "kron":
        return torch.kron(xs[0], xs[1])
    if op == "tensordot":
        return torch.tensordot(xs[0], xs[1], dims=p["n"])
    if op == "tensordot_axes":
        return torch.tensordot(xs[0], xs[1], dims=(p["dims_a"], p["dims_b"]))
    if op == "cdist":
        return torch.cdist(xs[0], xs[1], p=float(p["p"]), compute_mode="donot_use_mm_for_euclid_dist")
    if op == "cross":
        return torch.linalg.cross(xs[0], xs[1], dim=p["dim"])
    raise AssertionError(op)


def record(name, op, params, shapes, gen, pre_transpose=None, nonfinite=False, make=None):
    leaves = []
    for k, shape in enumerate(shapes):
        x0 = rnd(shape, gen)
        if make is not None:
            x0 = make(k, x0)
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
    y = apply(op, params, views)
    g = torch.randn(list(y.shape), generator=gen, dtype=torch.float32)
    loss = (y * g).sum()
    loss.backward()
    grads = [bits_of(l.grad) if l.grad is not None else [0] * numel(l.shape) for l in leaves]
    return {
        "name": name,
        "op": op,
        "params": params,
        "pre_transpose": pre_transpose,
        "inputs": [{"shape": list(l.shape), "x_bits": bits_of(l)} for l in leaves],
        "out": {"shape": list(y.shape), "bits": bits_of(y), "g_bits": bits_of(g)},
        "grads": grads,
    }


def try_raises(op, params, shapes):
    xs = [torch.zeros(tuple(s), dtype=torch.float32) for s in shapes]
    try:
        y = apply(op, params, xs)
        return False, list(y.shape)
    except Exception:
        return True, None


def sn(shape):
    return "x".join(map(str, shape)) if len(shape) > 0 else "scalar"


def pname(p):
    return str(p).replace(".", "_")


def main():
    gen = torch.Generator().manual_seed(SEED)
    cases = []

    # kron
    for i, (sa, sb) in enumerate([
        ((2,), (3,)), ((2, 3), (3, 2)), ((2, 2, 2), (2, 3, 2)), ((2,), (2, 3)),
        ((2, 3), (4,)), ((2, 2), (3, 2, 2)), ((), (3,)), ((3,), ()), ((), ()),
        ((0, 2), (2, 2)), ((1, 3), (3, 1)), ((2, 1), (1, 2)), ((2, 0), (1, 3)),
    ]):
        cases.append(record(f"kron_{i}_{sn(sa)}_{sn(sb)}", "kron", {}, [sa, sb], gen))
    cases.append(record("kron_nonfinite", "kron", {}, [(2, 3), (2, 2)], gen, nonfinite=True))

    # tensordot（dims=n）
    for i, (sa, sb, n) in enumerate([
        ((2, 3), (3, 4), 1), ((2, 3, 4), (3, 4, 5), 2), ((2, 3), (4, 5), 0),
        ((2, 3), (2, 3), 2), ((3,), (3,), 1), ((2, 3, 4), (4, 5), 1), ((2, 3), (3,), 1),
        ((), (3,), 0), ((), (), 0), ((2, 0), (0, 3), 1), ((0, 3), (3, 2), 1),
        ((2, 3), (3, 0), 1), ((2, 3, 4), (3, 4), 2), ((5,), (2,), 0),
    ]):
        cases.append(record(f"tensordot_{i}_{sn(sa)}_{sn(sb)}_n{n}", "tensordot", {"n": n}, [sa, sb], gen))
    cases.append(record("tensordot_t01", "tensordot", {"n": 1}, [(3, 2), (3, 4)], gen, pre_transpose=(0, 1)))
    cases.append(record("tensordot_nonfinite", "tensordot", {"n": 1}, [(2, 3), (3, 2)], gen, nonfinite=True))

    # tensordot（軸リスト）
    for i, (sa, sb, da, db) in enumerate([
        ((2, 3, 4), (4, 3), [2, 1], [0, 1]),
        ((2, 3, 4), (3, 4, 5), [1, 2], [0, 1]),
        ((2, 3, 4), (4, 3), [1, 2], [1, 0]),
        ((2, 3, 4, 5), (5, 2, 4), [0, 2], [1, 2]),
        ((3, 4), (4, 3), [1, 0], [0, 1]),
        ((2, 3, 4), (3, 6, 4), [1, 2], [0, 2]),
        ((3, 4), (5, 4), [1], [1]),
        ((3, 4), (3, 5), [0], [0]),
        ((2, 3), (4, 5), [], []),
    ]):
        cases.append(record(f"tensordot_axes_{i}", "tensordot_axes", {"dims_a": da, "dims_b": db}, [sa, sb], gen))

    # cdist
    for p in [0.5, 1.0, 1.5, 2.0, 3.0]:
        cases.append(record(f"cdist_p{pname(p)}_4x3_5x3", "cdist", {"p": p}, [(4, 3), (5, 3)], gen))
    for p in [1.0, 2.0, 3.0]:
        cases.append(record(f"cdist_p{pname(p)}_batch", "cdist", {"p": p}, [(2, 3, 4), (2, 5, 4)], gen))
        cases.append(record(f"cdist_p{pname(p)}_batch_bcast", "cdist", {"p": p}, [(2, 3, 4), (5, 4)], gen))
        cases.append(record(f"cdist_p{pname(p)}_batch_bcast2", "cdist", {"p": p}, [(1, 3, 4), (2, 5, 4)], gen))
        cases.append(record(f"cdist_p{pname(p)}_1x2_1x2", "cdist", {"p": p}, [(1, 2), (1, 2)], gen))
        cases.append(record(f"cdist_p{pname(p)}_P1", "cdist", {"p": p}, [(1, 3), (4, 3)], gen))
        cases.append(record(f"cdist_p{pname(p)}_R1", "cdist", {"p": p}, [(4, 3), (1, 3)], gen))
        cases.append(record(f"cdist_p{pname(p)}_M1", "cdist", {"p": p}, [(3, 1), (4, 1)], gen))
        cases.append(record(f"cdist_p{pname(p)}_Pzero", "cdist", {"p": p}, [(0, 3), (4, 3)], gen))
        cases.append(record(f"cdist_p{pname(p)}_Rzero", "cdist", {"p": p}, [(4, 3), (0, 3)], gen))
        cases.append(record(f"cdist_p{pname(p)}_Mzero", "cdist", {"p": p}, [(3, 0), (4, 0)], gen))
        # 同一点（x1 == x2。対角の距離は 0）。両入力へ同じ値を入れる。
        base = rnd((3, 4), gen)
        cases.append(record(f"cdist_p{pname(p)}_identical", "cdist", {"p": p}, [(3, 4), (3, 4)], gen,
                            make=lambda k, x, base=base: base.clone()))
    cases.append(record("cdist_t01", "cdist", {"p": 2.0}, [(4, 3), (5, 4)], gen, pre_transpose=(0, 1)))
    cases.append(record("cdist_nonfinite", "cdist", {"p": 2.0}, [(3, 4), (2, 4)], gen, nonfinite=True))
    cases.append(record("cdist_nonfinite_p1", "cdist", {"p": 1.0}, [(3, 4), (2, 4)], gen, nonfinite=True))

    # cross
    for i, (shape, dim) in enumerate([
        ((3,), 0), ((4, 3), 1), ((3, 4), 0), ((2, 3, 4), 1), ((2, 4, 3), 2),
        ((3, 2, 4), 0), ((2, 3, 4), 1), ((0, 3), 1), ((3, 0), 0), ((1, 3), 1),
    ]):
        cases.append(record(f"cross_{i}_{sn(shape)}_d{dim}", "cross", {"dim": dim}, [shape, shape], gen))
    cases.append(record("cross_parallel", "cross", {"dim": 0}, [(3,), (3,)], gen,
                        make=lambda k, x: torch.tensor([1.0, 2.0, 3.0]) * (1.0 + k)))
    cases.append(record("cross_t01", "cross", {"dim": 1}, [(3, 5), (5, 3)], gen, pre_transpose=(0, 1)))
    cases.append(record("cross_nonfinite", "cross", {"dim": 1}, [(2, 3), (2, 3)], gen, nonfinite=True))

    # error_cases
    err_specs = [
        ("cross_axis_len4", "cross", {"dim": 0}, [(4,), (4,)]),
        ("cross_axis_len2", "cross", {"dim": 1}, [(3, 2), (3, 2)]),
        ("cross_axis_oor", "cross", {"dim": 2}, [(3, 3), (3, 3)]),
        ("cross_shape_mismatch", "cross", {"dim": 1}, [(2, 3), (3, 3)]),
        ("cross_broadcastable", "cross", {"dim": 1}, [(2, 3), (1, 3)]),
        ("cross_rank0", "cross", {"dim": 0}, [(), ()]),
        ("cdist_m_mismatch", "cdist", {"p": 2.0}, [(3, 4), (3, 5)]),
        ("cdist_rank1", "cdist", {"p": 2.0}, [(4,), (3, 4)]),
        ("cdist_p_zero", "cdist", {"p": 0.0}, [(3, 4), (3, 4)]),
        ("cdist_p_inf", "cdist", {"p": "inf"}, [(3, 4), (3, 4)]),
        ("cdist_p_neg", "cdist", {"p": -1.0}, [(3, 4), (3, 4)]),
        ("cdist_batch_incompatible", "cdist", {"p": 2.0}, [(2, 3, 4), (3, 5, 4)]),
        ("tensordot_dup_axes", "tensordot_axes", {"dims_a": [0, 0], "dims_b": [0, 1]}, [(3, 3), (3, 3)]),
        ("tensordot_len_mismatch", "tensordot_axes", {"dims_a": [0], "dims_b": [0, 1]}, [(3, 3), (3, 3)]),
        ("tensordot_axis_len_mismatch", "tensordot_axes", {"dims_a": [0], "dims_b": [0]}, [(3, 4), (5, 4)]),
        ("tensordot_size1_vs_n", "tensordot_axes", {"dims_a": [0], "dims_b": [0]}, [(1, 4), (3, 4)]),
        ("tensordot_n_exceeds_rank", "tensordot", {"n": 3}, [(2, 3), (3, 4)]),
        ("tensordot_axis_oor", "tensordot_axes", {"dims_a": [5], "dims_b": [0]}, [(3, 4), (3, 4)]),
    ]
    error_cases = []
    for name, op, params, shapes in err_specs:
        raises, out_shape = try_raises(op, params, shapes)
        error_cases.append({
            "name": name, "op": op, "params": params,
            "input_shapes": [list(s) for s in shapes],
            "torch_raises": raises, "out_shape": out_shape,
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
