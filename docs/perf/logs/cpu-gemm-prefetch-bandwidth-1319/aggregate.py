#!/usr/bin/env python3
"""イシュー #1319 の実測ログ集計スクリプト（Python3 標準ライブラリのみ）。

`resid-<machine>[-bigpin]-run{1..5}.txt`（`microkernel_residency_diag`
の出力）から `mode=<name> threads=<single|multi> median_gflops=<f64>`
行を正規表現で読み、事前宣言ゲート（`gemm_prefetch_bandwidth_diag_tests.rs`
モジュール doc・`cpu-gemm-prefetch-decision.md`）が定義する代表値
`R_dram = streamed_dram の中央値 GFLOP/s / l1_resident の中央値 GFLOP/s`
を「l1_resident の 5 run 中央値」と「streamed_dram の 5 run 中央値」の
比として算出する（run ごとの比の中央値ではない。中央値を取る演算と
除算は可換ではないため両者は一般に一致しない。codex-review 指摘対応・
イシュー #1319）。5 run 中 3 run 以上で `R_dram <= 0.95` を満たすかの
判定（事前宣言ゲートの併記条件）には、別途 run ごとの比（同一 run 内の
streamed_dram / l1_resident）を用いる。
サンプル数が 5 でなければ非ゼロ終了する（fail-closed。実測値の
欠落・捏造を機械的に防ぐ）。

使い方: `python3 aggregate.py`（本ディレクトリで実行する前提。
`resid-*-run*.txt`／`bw-*-run*.txt` を自動検出する）。
"""

from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

LINE_RE = re.compile(r"^mode=(\S+)\s+threads=(single|multi)\s+median_(?:gflops|gbps)=([0-9.]+)\s*$")


def parse_file(path: Path) -> dict[tuple[str, str], float]:
    """1 run のログから (mode, threads) -> 値 の辞書を作る。"""
    out: dict[tuple[str, str], float] = {}
    for line in path.read_text().splitlines():
        m = LINE_RE.match(line.strip())
        if m:
            mode, threads, value = m.group(1), m.group(2), float(m.group(3))
            out[(mode, threads)] = value
    return out


def load_runs(pattern: str, expected: int = 5) -> list[dict[tuple[str, str], float]]:
    """`pattern`（glob）に一致する run ログをソートして読み込む。
    件数が `expected` と一致しなければ fail-closed で終了する。
    """
    paths = sorted(Path(".").glob(pattern))
    if len(paths) != expected:
        print(
            f"FATAL: {pattern} が {expected} 件でない（実際 {len(paths)} 件: {paths}）",
            file=sys.stderr,
        )
        sys.exit(1)
    return [parse_file(p) for p in paths]


def l1_dram_values(
    runs: list[dict[tuple[str, str], float]], threads: str
) -> tuple[list[float], list[float]]:
    """(l1_resident 値の列, streamed_dram 値の列) を run 順に返す。"""
    l1_vals: list[float] = []
    dram_vals: list[float] = []
    for r in runs:
        l1 = r.get(("l1_resident", threads))
        dram = r.get(("streamed_dram", threads))
        if l1 is None or dram is None:
            print(f"FATAL: l1_resident/streamed_dram (threads={threads}) が欠落", file=sys.stderr)
            sys.exit(1)
        l1_vals.append(l1)
        dram_vals.append(dram)
    return l1_vals, dram_vals


def r_dram_per_run_series(l1_vals: list[float], dram_vals: list[float]) -> list[float]:
    """run ごとの比（同一 run 内の streamed_dram / l1_resident）。
    事前宣言ゲートの「5 run 中 3 run 以上で `R_dram <= 0.95`」判定にのみ
    使う補助指標（代表値ではない。代表値は
    `r_dram_declared`〈中央値の比〉を参照）。
    """
    return [d / l for l, d in zip(l1_vals, dram_vals)]


def r_dram_declared(l1_vals: list[float], dram_vals: list[float]) -> float:
    """事前宣言どおりの代表値
    `R_dram = streamed_dram の中央値 GFLOP/s / l1_resident の中央値 GFLOP/s`。
    run ごとの比の中央値（`r_dram_per_run_series` の中央値）とは一般に
    一致しない（中央値と除算は非可換）。codex-review 指摘対応・
    イシュー #1319。
    """
    return statistics.median(dram_vals) / statistics.median(l1_vals)


def report_residency(label: str, pattern: str) -> None:
    runs = load_runs(pattern)
    print(f"### {label}（{pattern}。5 run）")
    print()
    print(
        "| threads | R_dram 各 run（参考: run 内比） | "
        "R_dram 代表値（中央値の比。事前宣言式） | <=0.95 の run 数 |"
    )
    print("|---|---|---|---|")
    for threads in ("single", "multi"):
        l1_vals, dram_vals = l1_dram_values(runs, threads)
        per_run = r_dram_per_run_series(l1_vals, dram_vals)
        declared = r_dram_declared(l1_vals, dram_vals)
        under_95 = sum(1 for v in per_run if v <= 0.95)
        series_str = ", ".join(f"{v:.4f}" for v in per_run)
        print(f"| {threads} | {series_str} | {declared:.4f} | {under_95}/5 |")
    print()


def report_bandwidth(label: str, pattern: str) -> None:
    runs = load_runs(pattern)
    print(f"### {label}（{pattern}。5 run。achievable_read GB/s）")
    print()
    print("| threads | 各 run | 中央値 |")
    print("|---|---|---|")
    for threads in ("single", "multi"):
        vals = [r[("achievable_read", threads)] for r in runs]
        median = statistics.median(vals)
        vals_str = ", ".join(f"{v:.4f}" for v in vals)
        print(f"| {threads} | {vals_str} | {median:.4f} |")
    print()


def main() -> None:
    print("## G-b（主判定）: R_dram = streamed_dram / l1_resident")
    print()
    report_residency("Apple M4 Max（共有負荷下）", "resid-m4max-run*.txt")
    report_residency("DGX Spark GB10（無 pin。異種コアスケジューリング混入）", "resid-dgx-run*.txt")
    report_residency(
        "DGX Spark GB10（big core pin: taskset -c 5-9,15-19）",
        "resid-dgx-bigpin-run*.txt",
    )
    print("## G-a 文脈: 実測到達帯域（achievable_bandwidth_diag）")
    print()
    report_bandwidth("Apple M4 Max", "bw-m4max-run*.txt")
    report_bandwidth("DGX Spark GB10（無 pin）", "bw-dgx-run*.txt")


if __name__ == "__main__":
    main()
