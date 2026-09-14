#!/usr/bin/env python3
"""イシュー #1590: `gemm_transposed_perf`（§3.2 の正式補助 A/B。5 プロセス
起動）のログを集計し、形状ごとの speedup 5 run 中央値表（markdown）を
出力する。

`run_ignored_tests.sh` が生成する
`docs/perf/logs/cuda-gemm-vjp-transposed-entry-1590/aux/
gemm_transposed_perf_run{1..5}.log` を入力とする。

python3 標準ライブラリのみ（他の集計スクリプト群と同方針。追加依存
なしで表記トリック迂回を遮断する必要はここでは薄いが、一貫性のため）。

行フォーマット（`crates/backend-cuda/tests/gemm_transposed_perf.rs` の
出力。`docs/perf/logs/lowlayer-diagnosis-2026-09-12/dgx/
gemm_transposed_perf.log` で実測確認済み）:

    test nt_transposed_entry_vs_contiguous_across_shapes ... NT m=64 k=784 n=256 before_median_s=0.000439 after_median_s=0.000079 speedup=5.563x
    NT m=64 k=256 n=10 before_median_s=0.000038 after_median_s=0.000036 speedup=1.057x
    ...

1 つ目の形状行は `test ... ...` と同一行に続き、以降の形状行は単独行に
なる。`re.search`（行頭アンカーなし）でどちらの形も拾う。
"""
from __future__ import annotations

import argparse
import io
import re
import statistics
import sys
from pathlib import Path

SHAPE_RE = re.compile(
    r"(?P<pattern>NT|TN) m=(?P<m>\d+) k=(?P<k>\d+) n=(?P<n>\d+) "
    r"before_median_s=(?P<before>[0-9.]+) after_median_s=(?P<after>[0-9.]+) "
    r"speedup=(?P<speedup>[0-9.]+)x"
)


def parse_log(text: str) -> list[dict]:
    """1 プロセス起動分のログ本文から形状行をすべて抽出する。"""
    rows = []
    for line in text.splitlines():
        m = SHAPE_RE.search(line)
        if not m:
            continue
        rows.append(
            {
                "pattern": m.group("pattern"),
                "m": int(m.group("m")),
                "k": int(m.group("k")),
                "n": int(m.group("n")),
                "before_s": float(m.group("before")),
                "after_s": float(m.group("after")),
                "speedup": float(m.group("speedup")),
            }
        )
    return rows


def aggregate(run_texts: list[str]) -> list[dict]:
    """複数起動分のログをまとめ、形状（pattern, m, k, n）ごとに
    before/after/speedup の中央値を計算する。順序は最初に出現した順を
    保つ（`dict` のキー挿入順に依存。Python 3.7+ の仕様どおり）。"""
    by_shape: dict[tuple, dict] = {}
    for text in run_texts:
        for row in parse_log(text):
            key = (row["pattern"], row["m"], row["k"], row["n"])
            entry = by_shape.setdefault(
                key, {"before": [], "after": [], "speedup": []}
            )
            entry["before"].append(row["before_s"])
            entry["after"].append(row["after_s"])
            entry["speedup"].append(row["speedup"])
    out = []
    for (pattern, m, k, n), entry in by_shape.items():
        out.append(
            {
                "pattern": pattern,
                "m": m,
                "k": k,
                "n": n,
                "n_runs": len(entry["speedup"]),
                "before_median_s": statistics.median(entry["before"]),
                "after_median_s": statistics.median(entry["after"]),
                "speedup_median": statistics.median(entry["speedup"]),
            }
        )
    return out


def render_markdown(rows: list[dict]) -> str:
    lines = [
        "| パターン | m | k | n | before 中央値 (s) | after 中央値 (s) | 倍率中央値 | n_runs |",
        "|----------|---|---|---|-------------------|-------------------|------------|--------|",
    ]
    for row in rows:
        lines.append(
            "| {pattern} | {m} | {k} | {n} | {before_median_s:.6f} | "
            "{after_median_s:.6f} | {speedup_median:.3f}x | {n_runs} |".format(**row)
        )
    return "\n".join(lines) + "\n"


def _self_test() -> int:
    # 実ログの断片を模したフィクスチャ（1 つ目の形状行が `test ...` と
    # 同一行に続く実フォーマットを含む。5 run 分・NT 1 形状のみで検証）。
    fixture_runs = [
        (
            "running 2 tests\n"
            "test nt_transposed_entry_vs_contiguous_across_shapes ... "
            "NT m=64 k=784 n=256 before_median_s=0.000439 after_median_s=0.000079 speedup=5.5{i}3x\n"
            "ok\n"
        ).format(i=i)
        for i in range(5)
    ]
    rows = aggregate(fixture_runs)
    assert len(rows) == 1, rows
    row = rows[0]
    assert row["pattern"] == "NT"
    assert row["m"] == 64 and row["k"] == 784 and row["n"] == 256
    assert row["n_runs"] == 5
    expected_speedups = [5.5 + 0.001 * i + 0.03 for i in range(5)]
    # 上の speedup 文字列は "5.5{i}3x" -> 5.503, 5.513, 5.523, 5.533, 5.543
    expected_speedups = [5.503, 5.513, 5.523, 5.533, 5.543]
    assert abs(row["speedup_median"] - statistics.median(expected_speedups)) < 1e-9
    md = render_markdown(rows)
    assert "NT" in md and "5.523x" in md
    print("self-test OK")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "logs", nargs="*", type=Path, help="gemm_transposed_perf_run*.log のパス"
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    if args.self_test:
        return _self_test()

    if not args.logs:
        parser.error("少なくとも 1 つのログファイルを指定するか --self-test を使うこと")

    run_texts = []
    for path in args.logs:
        run_texts.append(path.read_text(encoding="utf-8"))
    rows = aggregate(run_texts)
    if not rows:
        print("warning: 形状行が 1 件も抽出できなかった", file=sys.stderr)
    sys.stdout.write(render_markdown(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
