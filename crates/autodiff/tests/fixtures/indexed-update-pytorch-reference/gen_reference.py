#!/usr/bin/env python3
"""scatter_reduce／index_add／index_copy／masked_scatter の PyTorch 参照値を生成する。

イシュー #2641（親 #2625）の `tests/indexed_update_parity.rs` が参照する固定
フィクスチャ `indexed_update_reference.json` の生成スクリプト。CI は Python／
PyTorch に依存せず、コミット済み JSON のみを読む（`README.md` 参照）。

f32 はすべて u32 ビットパターン配列で保存する（JSON は NaN／inf を運べない
ため）。入力・src・上流勾配 `g`・出力・勾配をすべて保存し、Rust 側で入力を
再生成しない。損失は `(out * g).sum()`。dtype は float32。
"""

import json
import platform
import struct
import sys

import torch

EXPECTED_TORCH_PREFIX = "2.14.0"
assert torch.__version__.startswith(EXPECTED_TORCH_PREFIX), torch.__version__

SEED = 2641
NAN = float("nan")
INF = float("inf")
MODES = ["sum", "prod", "mean", "amax", "amin"]


def to_bits(values):
    return [struct.unpack("<I", struct.pack("<f", float(v)))[0] for v in values]


def flat_bits(t):
    return to_bits(t.detach().contiguous().reshape(-1).tolist())


def make(shape, kind, gen):
    if isinstance(kind, list):
        return torch.tensor(kind, dtype=torch.float32).reshape(shape)
    if kind == "randn":
        return torch.randn(*shape, generator=gen, dtype=torch.float32)
    if kind == "tied":
        return (torch.randn(*shape, generator=gen, dtype=torch.float32) * 2).round() / 2
    raise AssertionError(kind)


def grad_pair(fn, x0, s0, g_gen):
    """forward 値と (grad_x, grad_src)。backward が例外なら grad は None。"""
    x = x0.clone().requires_grad_(True)
    s = s0.clone().requires_grad_(True)
    y = fn(x, s)
    g = torch.randn(*y.shape, generator=g_gen, dtype=torch.float32)
    raised = ""
    gx = gs = None
    try:
        (y * g).sum().backward()
        gx = x.grad
        gs = s.grad
    except Exception as e:  # noqa: BLE001 - 実測として記録する
        raised = f"{type(e).__name__}: {e}"
    return y, g, gx, gs, raised


def record(name, op, x0, s0, g_gen, fn, **meta):
    y, g, gx, gs, raised = grad_pair(fn, x0, s0, g_gen)
    rec = {
        "name": name,
        "op": op,
        "x_shape": list(x0.shape),
        "x_bits": flat_bits(x0),
        "src_shape": list(s0.shape),
        "src_bits": flat_bits(s0),
        "g_bits": flat_bits(g),
        "out_bits": flat_bits(y),
        "grad_x_bits": flat_bits(gx) if gx is not None else None,
        "grad_src_bits": flat_bits(gs) if gs is not None else None,
        "grad_raises": raised,
    }
    rec.update(meta)
    return rec


# scatter_reduce: (name, x_shape, dim, idx_shape, idx(flat), x_kind, src_kind)
SR_CASES = [
    ("r1_dup", [5], 0, [4], [0, 2, 0, 2], "randn", "randn"),
    ("r1_untouched", [6], 0, [3], [1, 1, 4], "randn", "randn"),
    ("r1_all_same", [4], 0, [5], [2, 2, 2, 2, 2], "randn", "randn"),
    ("r2_dim0", [3, 4], 0, [2, 4], [0, 1, 2, 0, 0, 2, 2, 1], "randn", "randn"),
    ("r2_dim1", [3, 4], 1, [3, 2], [0, 3, 1, 1, 2, 0], "randn", "randn"),
    ("r2_small_idx", [3, 4], 1, [2, 2], [0, 0, 3, 1], "randn", "randn"),
    ("r3_dim1", [2, 3, 2], 1, [2, 2, 2], [0, 1, 2, 2, 1, 1, 0, 2], "randn", "randn"),
    ("r3_dim2", [2, 2, 3], 2, [2, 2, 3], [0, 1, 2, 2, 2, 0, 1, 1, 1, 0, 0, 2], "randn", "randn"),
    ("empty_index", [4], 0, [0], [], "randn", "randn"),
    ("ties", [6], 0, [6], [0, 0, 1, 1, 2, 2], "tied", "tied"),
    ("tie_vs_self", [3], 0, [4], [0, 0, 1, 1], [2.0, 1.0, 0.5], [2.0, 1.0, 1.0, 3.0]),
    ("prod_no_zero", [4], 0, [5], [0, 0, 1, 2, 2], [1.5, 2.0, -1.0, 0.5], [2.0, -3.0, 1.5, 2.5, -0.5]),
    ("prod_one_zero", [4], 0, [5], [0, 0, 1, 2, 2], [1.5, 2.0, -1.0, 0.5], [2.0, 0.0, 1.5, 2.5, -0.5]),
    ("prod_two_zeros", [4], 0, [5], [0, 0, 0, 2, 2], [1.5, 2.0, -1.0, 0.5], [2.0, 0.0, 0.0, 2.5, -0.5]),
    ("prod_self_zero", [3], 0, [4], [0, 0, 1, 1], [0.0, 2.0, 1.0], [2.0, 3.0, 0.0, 4.0]),
    ("signed_zero", [3], 0, [4], [0, 0, 1, 1], [0.0, -0.0, 1.0], [-0.0, 0.0, -0.0, -0.0]),
]

# include_self=False で入力が偶然結果と一致する群（勾配分配の差分を独立に実測する）
SELFEQ_CASES = [
    ("selfeq_amax", [3], 0, [3], [0, 0, 1], [3.0, 1.0, 5.0], [3.0, 1.0, 2.0]),
    ("selfeq_amin", [3], 0, [3], [0, 0, 1], [1.0, 1.0, 5.0], [1.0, 3.0, 2.0]),
]

NONFINITE = [
    ("nan_src", [4], 0, [4], [0, 0, 1, 1], [1.0, 2.0, 3.0, 4.0], [NAN, 1.0, 2.0, 3.0]),
    ("nan_self", [4], 0, [4], [0, 0, 1, 1], [NAN, 2.0, 3.0, 4.0], [1.0, 1.0, 2.0, 3.0]),
    ("posinf_src", [4], 0, [4], [0, 0, 1, 1], [1.0, 2.0, 3.0, 4.0], [INF, 1.0, 2.0, 3.0]),
    ("neginf_src", [4], 0, [4], [0, 0, 1, 1], [1.0, 2.0, 3.0, 4.0], [-INF, 1.0, 2.0, 3.0]),
    ("inf_minus_inf", [4], 0, [4], [0, 0, 1, 1], [1.0, 2.0, 3.0, 4.0], [INF, -INF, INF, 1.0]),
    ("all_neginf_nonself", [3], 0, [3], [0, 0, 1], [1.0, 2.0, 3.0], [-INF, -INF, -INF]),
    ("all_posinf_nonself", [3], 0, [3], [0, 0, 1], [1.0, 2.0, 3.0], [INF, INF, INF]),
]


def sr_fn(dim, idx, mode, inc):
    def f(x, s):
        return x.scatter_reduce(dim, idx, s, reduce=mode, include_self=inc)
    return f


def gen_scatter_reduce(gen, cases, group, mode_filter=None):
    out = []
    for name, xs, dim, ishape, iflat, xk, sk in cases:
        x0 = make(xs, xk, gen)
        s0 = make(ishape, sk, gen)
        idx = torch.tensor(iflat, dtype=torch.int64).reshape(ishape)
        for mode in MODES:
            if mode_filter and mode not in mode_filter(name):
                continue
            for inc in (True, False):
                rec = record(
                    f"sr_{name}_{mode}_{'self' if inc else 'noself'}", "scatter_reduce",
                    x0, s0, gen, sr_fn(dim, idx, mode, inc),
                    dim=dim, index_shape=ishape, index=iflat, reduce=mode,
                    include_self=inc, group=group)
                # include_self=False の amax/amin で、触れられた位置の入力値が結果と
                # 偶然一致する場合、PyTorch は入力も分配数に数える（実測）。本実装は
                # 数えないため勾配分配が異なる。この一致が起きる行は独立の群に分ける。
                if group == "finite" and mode in ("amax", "amin") and not inc:
                    touched = torch.zeros_like(x0).scatter_add(
                        dim, idx, torch.ones_like(s0)) > 0
                    out_t = x0.scatter_reduce(dim, idx, s0, reduce=mode, include_self=False)
                    if bool(((out_t == x0) & touched).any()):
                        rec["group"] = "selfeq"
                out.append(rec)
    return out


def gen_index_ops(gen):
    out = []
    # (name, op, x_shape, dim, index, source_shape)
    cases = [
        ("add_dup_dim0", "index_add", [5, 3], 0, [2, 0, 2, 4], [4, 3]),
        ("add_dim1", "index_add", [3, 5], 1, [4, 1, 1], [3, 3]),
        ("add_rank1", "index_add", [6], 0, [5, 5, 0], [3]),
        ("add_empty", "index_add", [4], 0, [], [0]),
        ("copy_dim0", "index_copy", [5, 3], 0, [3, 0, 4], [3, 3]),
        ("copy_dim1", "index_copy", [3, 5], 1, [4, 0, 2], [3, 3]),
        ("copy_rank1", "index_copy", [6], 0, [5, 1], [2]),
        ("copy_empty", "index_copy", [4], 0, [], [0]),
    ]
    for name, op, xs, dim, iflat, ss in cases:
        x0 = make(xs, "randn", gen)
        s0 = make(ss, "randn", gen)
        idx = torch.tensor(iflat, dtype=torch.int64)
        if op == "index_add":
            fn = lambda x, s, d=dim, i=idx: x.index_add(d, i, s)  # noqa: E731
        else:
            fn = lambda x, s, d=dim, i=idx: x.index_copy(d, i, s)  # noqa: E731
        out.append(record(name, op, x0, s0, gen, fn, dim=dim, index=iflat,
                          index_shape=[len(iflat)], group="finite"))
    return out


def gen_masked_scatter(gen):
    out = []
    # (name, x_shape, mask_shape, mask(flat 0/1), source_shape)
    cases = [
        ("ms_excess", [2, 3], [2, 3], [1, 0, 1, 0, 1, 0], [5]),
        ("ms_all_false", [2, 3], [2, 3], [0] * 6, [4]),
        ("ms_all_true", [2, 2], [2, 2], [1] * 4, [2, 2]),
        ("ms_mask_small", [2, 3], [3], [1, 0, 1], [6]),
        ("ms_mask_col", [3, 2], [3, 1], [1, 0, 1], [6]),
        ("ms_source_2d", [2, 3], [2, 3], [0, 1, 1, 1, 0, 0], [2, 2]),
        ("ms_mask_large", [3], [2, 3], [1, 0, 1, 0, 1, 1], [8]),
        ("ms_rank1", [5], [5], [0, 1, 0, 1, 1], [3]),
    ]
    for name, xs, ms, mflat, ss in cases:
        x0 = make(xs, "randn", gen)
        s0 = make(ss, "randn", gen)
        mask = torch.tensor(mflat, dtype=torch.bool).reshape(ms)
        out.append(record(name, "masked_scatter", x0, s0, gen,
                          lambda x, s, m=mask: torch.masked_scatter(x, m, s),
                          mask_shape=ms, mask=mflat, group="finite"))
    return out


def raises(name, op, thunk, **meta):
    try:
        r = thunk()
        return {"name": name, "op": op, "torch_raises": False, "message": "",
                "out_shape": list(r.shape), **meta}
    except Exception as e:  # noqa: BLE001
        return {"name": name, "op": op, "torch_raises": True,
                "message": f"{type(e).__name__}: {e}", "out_shape": [], **meta}


def gen_errors():
    def z(*s):
        return torch.zeros(*s)

    def i(v, *s):
        t = torch.tensor(v, dtype=torch.int64)
        return t.reshape(*s) if s else t

    out = []
    out.append(raises("sr_idx_oob", "scatter_reduce", lambda: z(4).scatter_reduce(0, i([4], 1), z(1), "sum"), kind="idx_oob"))
    out.append(raises("sr_idx_neg", "scatter_reduce", lambda: z(4).scatter_reduce(0, i([-1], 1), z(1), "sum"), kind="idx_neg"))
    out.append(raises("sr_dim_oob", "scatter_reduce", lambda: z(4).scatter_reduce(1, i([0], 1), z(1), "sum"), kind="dim_oob"))
    out.append(raises("sr_idx_gt_src", "scatter_reduce", lambda: z(4).scatter_reduce(0, i([0, 1], 2), z(1), "sum"), kind="index_larger_than_src"))
    out.append(raises("sr_idx_lt_src", "scatter_reduce", lambda: z(4).scatter_reduce(0, i([0], 1), z(2), "sum"), kind="index_smaller_than_src"))
    out.append(raises("sr_idx_gt_input", "scatter_reduce", lambda: z(2, 3).scatter_reduce(0, i([0] * 12, 4, 3), z(4, 3), "sum"), kind="index_larger_than_input_dim"))
    out.append(raises("sr_rank0", "scatter_reduce", lambda: torch.tensor(1.0).scatter_reduce(0, i([0]), z(1), "sum"), kind="rank0"))
    out.append(raises("ia_idx_oob", "index_add", lambda: z(4).index_add(0, i([4], 1), z(1)), kind="idx_oob"))
    out.append(raises("ia_idx_neg", "index_add", lambda: z(4).index_add(0, i([-1], 1), z(1)), kind="idx_neg"))
    out.append(raises("ia_len_mismatch", "index_add", lambda: z(4).index_add(0, i([0, 1], 2), z(3)), kind="len_mismatch"))
    out.append(raises("ia_other_dim_mismatch", "index_add", lambda: z(3, 4).index_add(0, i([0], 1), z(1, 5)), kind="other_dim_mismatch"))
    out.append(raises("ia_dim_oob", "index_add", lambda: z(4).index_add(1, i([0], 1), z(1)), kind="dim_oob"))
    out.append(raises("ic_idx_oob", "index_copy", lambda: z(4).index_copy(0, i([4], 1), z(1)), kind="idx_oob"))
    out.append(raises("ic_len_mismatch", "index_copy", lambda: z(4).index_copy(0, i([0, 1], 2), z(3)), kind="len_mismatch"))
    out.append(raises("ic_other_dim_mismatch", "index_copy", lambda: z(3, 4).index_copy(0, i([0], 1), z(1, 5)), kind="other_dim_mismatch"))
    out.append(raises("ms_source_short", "masked_scatter", lambda: torch.masked_scatter(z(4), torch.tensor([True] * 4), z(3)), kind="source_short"))
    out.append(raises("ms_mask_not_broadcastable", "masked_scatter", lambda: torch.masked_scatter(z(3), torch.tensor([True, False]), z(3)), kind="mask_not_broadcastable"))
    return out


def main():
    gen = torch.Generator().manual_seed(SEED)
    cases = []
    cases += gen_scatter_reduce(gen, SR_CASES, "finite")
    cases += gen_scatter_reduce(
        gen, SELFEQ_CASES, "selfeq",
        lambda n: ["amax"] if n.endswith("amax") else ["amin"])
    cases += gen_scatter_reduce(gen, NONFINITE, "nonfinite")
    cases += gen_index_ops(gen)
    cases += gen_masked_scatter(gen)
    doc = {
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "seed": SEED,
        "cases": cases,
        "error_cases": gen_errors(),
    }
    json.dump(doc, sys.stdout, ensure_ascii=False, separators=(",", ":"))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
