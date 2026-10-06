#!/usr/bin/env python3
"""Subset／ConcatDataset／random_split の PyTorch 参照値を生成する。

イシュー #2661（親 #2660）の `crates/facade/tests/data_dataset_compose.rs` が参照する
固定フィクスチャ `dataset_compose_reference.json` の生成スクリプト。CI は Python／
PyTorch に依存せず、コミット済み JSON のみを読む（`README.md` 参照）。

- f32 は u32、割合（f64）は u64 のビットパターンで保存する。
- 整数表（長さ列・cumulative_sizes）は値そのまま。
- randperm の数列そのものは本リポの RNG と一致しないため保存しない（分割の構造のみ検証）。
"""

import json
import math
import platform
import struct
import sys
import warnings

import torch
from torch.utils.data import ConcatDataset, Dataset, Subset, random_split

EXPECTED_TORCH_PREFIX = "2.14.0"
assert torch.__version__.startswith(EXPECTED_TORCH_PREFIX), torch.__version__

warnings.simplefilter("ignore")
SEED = 2661
GEN = torch.Generator().manual_seed(SEED)


def f32_bits(t):
    flat = t.detach().contiguous().reshape(-1)
    if flat.numel() == 0:
        return []
    return [v & 0xFFFFFFFF for v in flat.view(torch.int32).tolist()]


def f64_bits(x):
    return struct.unpack("<Q", struct.pack("<d", x))[0]


class Rows(Dataset):
    """先頭軸をサンプル軸とするテンソル 1 本（TensorDataset の単一テンソル版）。"""

    def __init__(self, t):
        self.t = t

    def __len__(self):
        return self.t.shape[0]

    def __getitem__(self, i):
        return self.t[i]


class Pair(Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def make(n, d, offset=0.0):
    return torch.randn(n, d, generator=GEN) + offset


def stack_rows(ds, idxs):
    if not idxs:
        return None
    return torch.stack([ds[i] for i in idxs])


def tensor_rec(t):
    return {"shape": list(t.shape), "bits": f32_bits(t)}


def try_split(n, lengths):
    ds = Rows(torch.zeros(n, 1))
    try:
        parts = random_split(ds, lengths, generator=torch.Generator().manual_seed(SEED))
    except Exception as e:  # noqa: BLE001
        return {"torch_raises": True, "error": type(e).__name__}
    idx = [list(p.indices) for p in parts]
    flat = [i for sub in idx for i in sub]
    assert sorted(flat) == list(range(n))
    return {"torch_raises": False, "lengths": [len(s) for s in idx]}


def main():
    out = {
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "seed": SEED,
    }

    # 割合指定: 解決後の長さ列（余り配分・丸め境界・範囲外）
    frac_inputs = [
        (10, [0.3, 0.3, 0.4]),
        (7, [0.5, 0.5]),
        (11, [0.2, 0.3, 0.5]),
        (3, [0.1] * 10),
        (0, [0.5, 0.5]),
        (1, [0.5, 0.5]),
        (5, [1.0]),
        (5, [0.0, 1.0]),
        (100, [0.7, 0.15, 0.15]),
        (9, [1 / 3, 1 / 3, 1 / 3]),
        (10, [0.6, 0.5]),
        (10, [0.5, 0.5000001]),
        (10, [-0.1, 1.1]),
        (10, [0.4, 0.4]),
        (10, [float("nan")]),
        (10, []),
        (1000, [0.1] * 10),
    ]
    out["fraction_cases"] = []
    for n, fr in frac_inputs:
        rec = {"n": n, "fractions_bits": [f64_bits(f) for f in fr]}
        # 割合として解釈される入力のみ（長さ列として解釈される場合は別経路）
        rec.update(try_split(n, fr))
        out["fraction_cases"].append(rec)

    # 整数長指定（合計不一致・0 長・割合と誤解釈されない形）
    out["int_split_cases"] = []
    for n, ln in [
        (10, [3, 7]),
        (10, [10]),
        (10, [0, 10]),
        (10, [3, 3]),
        (10, [4, 7]),
        (0, [0]),
        (6, [2, 2, 2]),
        (5, [2, 3, 0]),
    ]:
        rec = {"n": n, "lengths": ln}
        rec.update(try_split(n, ln))
        out["int_split_cases"].append(rec)

    # Subset
    out["subset_cases"] = []
    base = make(8, 3)
    inner = Subset(Rows(base), [7, 2, 2, 5, 0])
    cases = [
        ("basic", Rows(base), [3, 1, 4, 1, 5]),
        ("duplicates", Rows(base), [2, 2, 2]),
        ("reverse_all", Rows(base), list(range(7, -1, -1))),
        ("empty", Rows(base), []),
        ("nested", inner, [4, 0, 2, 2]),
    ]
    for name, ds, idx in cases:
        sub = Subset(ds, idx)
        sel = list(range(len(sub)))
        rows = stack_rows(sub, sel)
        if name == "nested":
            root = Subset(Rows(base), [7, 2, 2, 5, 0])
            sub = Subset(root, idx)
            rows = stack_rows(sub, list(range(len(sub))))
        rec = {
            "name": name,
            "base": tensor_rec(base),
            "outer_indices": idx,
            "inner_indices": [7, 2, 2, 5, 0] if name == "nested" else None,
            "len": len(sub),
            "rows": tensor_rec(rows) if rows is not None else {"shape": [0, 3], "bits": []},
        }
        out["subset_cases"].append(rec)

    # ConcatDataset
    out["concat_cases"] = []
    parts = [make(3, 2), torch.zeros(0, 2), make(4, 2, 10.0), make(1, 2, 20.0)]
    cat = ConcatDataset([Rows(p) for p in parts])
    n = len(cat)
    perms = {
        "identity": list(range(n)),
        "reverse": list(range(n - 1, -1, -1)),
        "interleaved": [7, 0, 3, 6, 2, 3, 4, 1],
        "boundaries": [2, 3, 6, 7, 0],
    }
    out["concat_cases"].append(
        {
            "name": "plain",
            "parts": [tensor_rec(p) for p in parts],
            "cumulative_sizes": cat.cumulative_sizes,
            "queries": [
                {"name": k, "indices": v, "rows": tensor_rec(stack_rows(cat, v))}
                for k, v in perms.items()
            ],
        }
    )
    xs = [make(2, 2), make(3, 2, 5.0)]
    ys = [torch.tensor([1, 2], dtype=torch.int32), torch.tensor([7, 8, 9], dtype=torch.int32)]
    pcat = ConcatDataset([Pair(x, y) for x, y in zip(xs, ys)])
    idx = [4, 0, 2, 1, 3]
    items = [pcat[i] for i in idx]
    out["concat_cases"].append(
        {
            "name": "tuple_f32_i32",
            "xs": [tensor_rec(x) for x in xs],
            "ys": [[int(v) for v in y.tolist()] for y in ys],
            "cumulative_sizes": pcat.cumulative_sizes,
            "indices": idx,
            "x_rows": tensor_rec(torch.stack([a for a, _ in items])),
            "y_rows": [int(b) for _, b in items],
        }
    )
    # Subset 同士の連結
    s1 = Subset(Rows(base), [1, 3])
    s2 = Subset(Rows(base), [6, 0, 5])
    scat = ConcatDataset([s1, s2])
    idx = [4, 0, 3, 1, 2]
    out["concat_cases"].append(
        {
            "name": "subset_of_subsets",
            "base": tensor_rec(base),
            "subset_indices": [[1, 3], [6, 0, 5]],
            "cumulative_sizes": scat.cumulative_sizes,
            "indices": idx,
            "rows": tensor_rec(stack_rows(scat, idx)),
        }
    )

    # 例外（torch の成否）
    errs = []
    try:
        ConcatDataset([])
        errs.append({"name": "empty_concat", "torch_raises": False})
    except Exception as e:  # noqa: BLE001
        errs.append({"name": "empty_concat", "torch_raises": True, "error": type(e).__name__})
    sub = Subset(Rows(make(3, 1)), [0, 5])
    try:
        sub[1]
        errs.append({"name": "subset_oob_at_access", "torch_raises": False})
    except Exception as e:  # noqa: BLE001
        errs.append({"name": "subset_oob_at_access", "torch_raises": True, "error": type(e).__name__})
    c2 = ConcatDataset([Rows(make(2, 1))])
    try:
        c2[2]
        errs.append({"name": "concat_oob", "torch_raises": False})
    except Exception as e:  # noqa: BLE001
        errs.append({"name": "concat_oob", "torch_raises": True, "error": type(e).__name__})
    out["error_cases"] = errs

    # 分割の構造: 同シードの randperm を累積 offset で切った連続区間
    g = torch.Generator().manual_seed(SEED)
    parts = random_split(Rows(torch.zeros(10, 1)), [3, 3, 4], generator=g)
    perm = torch.randperm(10, generator=torch.Generator().manual_seed(SEED)).tolist()
    assert [i for p in parts for i in p.indices] == perm
    out["split_structure_verified"] = True

    json.dump(out, sys.stdout, separators=(",", ":"))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
