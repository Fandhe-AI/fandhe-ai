#!/usr/bin/env python3
"""IterableDataset／BatchSampler の PyTorch 参照値を生成する。

イシュー #2662（親 #2660）の `crates/facade/tests/data_iterable_batch_sampler.rs` が参照する
固定フィクスチャ `iterable_batch_sampler_reference.json` の生成スクリプト。CI は Python／
PyTorch に依存せず、コミット済み JSON のみを読む（`README.md` 参照）。

- f32 は u32 のビットパターンで保存する。
- 添字バッチ列・バッチ数などの整数表は値そのまま。
- 乱数 sampler（RandomSampler）経由のケースは含めない（randperm の数列は本リポの RNG と一致しない）。
"""

import json
import platform
import sys
import warnings

import torch
from torch.utils.data import BatchSampler, DataLoader, IterableDataset

EXPECTED_TORCH_PREFIX = "2.14.0"
assert torch.__version__.startswith(EXPECTED_TORCH_PREFIX), torch.__version__

warnings.simplefilter("ignore")
SEED = 2662
GEN = torch.Generator().manual_seed(SEED)
D = 2


def f32_bits(t):
    flat = t.detach().contiguous().reshape(-1)
    if flat.numel() == 0:
        return []
    return [v & 0xFFFFFFFF for v in flat.view(torch.int32).tolist()]


def tensor_rec(t):
    return {"shape": list(t.shape), "bits": f32_bits(t)}


class Stream(IterableDataset):
    """行を順に yield する iterable-style データセット（`keep` で長さ不定にできる）。"""

    def __init__(self, rows, keep=None):
        self.rows = rows
        self.keep = keep

    def __iter__(self):
        for r in self.rows:
            if self.keep is None or self.keep(r):
                yield r


class TupleStream(IterableDataset):
    def __init__(self, xs, ys):
        self.xs, self.ys = xs, ys

    def __iter__(self):
        for x, y in zip(self.xs, self.ys):
            yield x, y


class Exploding(IterableDataset):
    def __init__(self, rows, after):
        self.rows, self.after = rows, after

    def __iter__(self):
        for i, r in enumerate(self.rows):
            if i == self.after:
                raise RuntimeError("stream failure")
            yield r


def make_rows(n):
    return torch.randn(n, D, generator=GEN)


def exc_name(fn):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        return type(e).__name__
    return None


def main():
    out = {
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "seed": SEED,
    }

    # --- BatchSampler: 添字列 × batch_size × drop_last ---
    sources = {}
    for n in [0, 1, 5, 6, 7, 10]:
        sources[f"range_{n}"] = list(range(n))
    sources["reversed_5"] = [4, 3, 2, 1, 0]
    sources["duplicates_7"] = [2, 2, 0, 5, 5, 1, 9]
    sources["sparse_5"] = [0, 3, 6, 9, 12]
    bs_cases = []
    for name, order in sources.items():
        n = len(order)
        sizes = sorted({1, 2, 3, n + 1} | ({n} if n > 0 else set()))
        for k in sizes:
            for drop_last in (False, True):
                bsamp = BatchSampler(order, batch_size=k, drop_last=drop_last)
                batches = [list(b) for b in bsamp]
                bs_cases.append(
                    {
                        "name": f"{name}_k{k}_{'drop' if drop_last else 'keep'}",
                        "order": order,
                        "batch_size": k,
                        "drop_last": drop_last,
                        "batches": batches,
                        "len": len(bsamp),
                    }
                )
    out["batch_sampler_cases"] = bs_cases

    # --- IterableDataset + DataLoader(batch_size=k, drop_last=d) ---
    it_cases = []
    for n in [0, 5, 6, 7]:
        rows = make_rows(n)
        for k in [1, 2, 3, 10]:
            for drop_last in (False, True):
                dl = DataLoader(Stream(list(rows)), batch_size=k, drop_last=drop_last)
                batches = [tensor_rec(b) for b in dl]
                it_cases.append(
                    {
                        "name": f"rows_{n}_k{k}_{'drop' if drop_last else 'keep'}",
                        "rows": tensor_rec(rows) if n > 0 else {"shape": [0, D], "bits": []},
                        "batch_size": k,
                        "drop_last": drop_last,
                        "batches": batches,
                    }
                )
    # フィルタ付きジェネレータ（長さが事前に分からない例）: 出力された行そのものを記録する
    base = make_rows(12)
    keep = lambda r: float(r[0]) > 0.0  # noqa: E731
    kept = [r for r in base if keep(r)]
    assert 0 < len(kept) < 12
    for k, drop_last in [(3, False), (3, True), (4, False)]:
        dl = DataLoader(Stream(list(base), keep=keep), batch_size=k, drop_last=drop_last)
        it_cases.append(
            {
                "name": f"filtered_k{k}_{'drop' if drop_last else 'keep'}",
                "rows": tensor_rec(torch.stack(kept)),
                "batch_size": k,
                "drop_last": drop_last,
                "batches": [tensor_rec(b) for b in dl],
            }
        )
    out["iterable_cases"] = it_cases

    # --- タプル（f32 行 + 整数ラベル） ---
    xs = [r for r in make_rows(7)]
    ys = [torch.tensor(i * 3 - 4, dtype=torch.int64) for i in range(7)]
    tup = []
    for k, drop_last in [(3, False), (3, True), (2, False)]:
        dl = DataLoader(TupleStream(xs, ys), batch_size=k, drop_last=drop_last)
        bl = []
        for bx, by in dl:
            bl.append({"x": tensor_rec(bx), "y": [int(v) for v in by.tolist()]})
        tup.append(
            {
                "name": f"tuple_k{k}_{'drop' if drop_last else 'keep'}",
                "xs": tensor_rec(torch.stack(xs)),
                "ys": [int(y) for y in ys],
                "batch_size": k,
                "drop_last": drop_last,
                "batches": bl,
            }
        )
    out["tuple_cases"] = tup

    # --- 例外（torch の成否） ---
    errs = []
    it5 = Stream(list(make_rows(5)))

    def rec(name, fn):
        e = exc_name(fn)
        errs.append({"name": name, "torch_raises": e is not None, "error": e})

    rec("batch_sampler_zero_batch_size", lambda: BatchSampler(range(5), 0, False))
    rec("iterable_shuffle_true", lambda: DataLoader(it5, batch_size=2, shuffle=True))
    rec("iterable_sampler", lambda: DataLoader(it5, batch_size=2, sampler=[0, 1]))
    rec(
        "iterable_batch_sampler",
        lambda: DataLoader(it5, batch_sampler=BatchSampler(range(5), 2, False)),
    )
    rec("iterable_len", lambda: len(DataLoader(it5, batch_size=2)))
    mismatched = [torch.zeros(2), torch.zeros(3), torch.zeros(2)]
    rec(
        "iterable_shape_mismatch",
        lambda: list(DataLoader(Stream(mismatched), batch_size=2)),
    )
    # ストリーム途中の例外: 失敗前に完成したバッチ数も記録する
    got = []
    try:
        for b in DataLoader(Exploding(list(make_rows(6)), after=3), batch_size=2):
            got.append(b)
        errs.append({"name": "iterable_mid_stream_error", "torch_raises": False, "error": None})
    except Exception as e:  # noqa: BLE001
        errs.append(
            {
                "name": "iterable_mid_stream_error",
                "torch_raises": True,
                "error": type(e).__name__,
                "batches_before_error": len(got),
            }
        )
    out["error_cases"] = errs

    json.dump(out, sys.stdout, separators=(",", ":"))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
