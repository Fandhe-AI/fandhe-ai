#!/usr/bin/env python3
"""イシュー #1336 実機実測フェーズ: `host_view_staging_readout_ab_1336`
の 5 プロセス起動生ログ（`ab-run1.log`〜`ab-run5.log`）から CSV 行
（`phase,n,bytes_mib,cold_ms,min_ms,q1_ms,median_ms,q3_ms,max_ms`）を
抽出し、`phase,n` ごとの 5 run 中央値（median_ms の中央値・min/max）を
`aggregate.md` へ書き出す。

Python3 標準ライブラリのみ（`.claude/rules/deps-policy.md` の依存方針・
既存 A/B 集計スクリプト群〈例: `logs/cuda-tiled-pipeline-128x64-1344/`〉
と同じ方針）。
"""

from __future__ import annotations

import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN_LOGS = [HERE / f"ab-run{i}.log" for i in range(1, 6)]
HEADER = "phase,n,bytes_mib,cold_ms,min_ms,q1_ms,median_ms,q3_ms,max_ms"
PHASES = ("before", "after_pageable", "after_pinned")


def extract_rows(log_path: Path) -> list[list[str]]:
    rows = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        for phase in PHASES:
            if line.startswith(phase + ","):
                rows.append(line.split(","))
                break
    return rows


def main() -> None:
    # (phase, n) -> list of median_ms across the 5 runs
    by_key: dict[tuple[str, str], list[float]] = {}
    bytes_mib_by_n: dict[str, str] = {}

    for log_path in RUN_LOGS:
        if not log_path.exists():
            raise SystemExit(f"missing run log: {log_path}")
        for row in extract_rows(log_path):
            phase, n, bytes_mib, _cold, _min, _q1, median_ms, _q3, _max = row
            by_key.setdefault((phase, n), []).append(float(median_ms))
            bytes_mib_by_n[n] = bytes_mib

    lines = []
    lines.append("# host_view_staging_readout_ab_1336 集計（5 run 中央値）")
    lines.append("")
    lines.append(
        "各セルは `median_ms`（1 run あたり 20/20 warmup+計測の中央値）を"
        " 5 プロセス起動ぶん集めた系列の中央値。`before` は `download()`"
        "（毎回新規確保）、`after_pageable` は本番既定"
        "（`HOST_STAGING_KIND=Pageable`）、`after_pinned` はキャッシュ経由"
        " `Pinned`（`new_with_host_staging_kind`）。"
    )
    lines.append("")
    lines.append(
        "| N | bytes(MiB) | before median_ms (5run) | after_pageable"
        " median_ms (5run) | after_pinned median_ms (5run) |"
        " pageable/before | pinned/before |"
    )
    lines.append("|---|---|---|---|---|---|---|")

    ns = sorted({n for (_p, n) in by_key}, key=int)
    for n in ns:
        before_vals = by_key.get(("before", n), [])
        pageable_vals = by_key.get(("after_pageable", n), [])
        pinned_vals = by_key.get(("after_pinned", n), [])
        before_med = statistics.median(before_vals)
        pageable_med = statistics.median(pageable_vals)
        pinned_med = statistics.median(pinned_vals)
        pageable_ratio = pageable_med / before_med if before_med else float("nan")
        pinned_ratio = pinned_med / before_med if before_med else float("nan")
        lines.append(
            f"| {n} | {bytes_mib_by_n[n]} | {before_med:.4f}"
            f" (n={len(before_vals)}) | {pageable_med:.4f}"
            f" (n={len(pageable_vals)}) | {pinned_med:.4f}"
            f" (n={len(pinned_vals)}) | {pageable_ratio:.3f}x |"
            f" {pinned_ratio:.3f}x |"
        )

    lines.append("")
    lines.append(
        "比率列は `median / before_median` で、1 未満は before に対する"
        "高速化（例: 0.10x は約 10 倍高速）を表す。"
    )

    out_path = HERE / "aggregate.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
