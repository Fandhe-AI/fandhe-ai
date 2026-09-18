#!/usr/bin/env python3
"""イシュー #1973 Layer B（`gemm_reuse_phase_diag_select`／`_classic`）の 5 run 集計。

役割: `layerB-run{1..5}.log`（`crates/backend-cuda/src/gemm_reuse_phase_diag_tests.rs`
の標準出力を含む cargo test ログ）から、N × 変種（Select／Classic）× 7 フェーズ
（`h2d_a`／`h2d_b`／`alloc_c`／`launch_issue`／`kernel_wait`／`d2h`／`host_copy`）の
各 run 内中央値（`median=` 値。各 run は 20 warmup + 20 trial の中央値）を抽出し、
5 run 中央値と 5 run の生値を Markdown 表として `aggregate_layer_b.md` 相当を
標準出力へ書き出す。あわせて `host_copy` を除く 6 フェーズの 5 run 中央値の和
（`docs/perf/cuda-gemm-reuse-phase-breakdown.md` §5 の「Layer A `matmul` 相当」区間）を
セルごとに出力し、Layer A との突合に使う。

呼び出し: `python3 aggregate_layer_b.py > aggregate_layer_b.md`（本ディレクトリで実行。
python3 標準ライブラリのみ・追加依存なし）。

完全性検査（fail-closed）: 各 run ログに N=1024／2048／4096 × Select／Classic の
6 セルが 1 回ずつ現れ、各セルに 7 フェーズが揃っていない場合は表を出力せず
非ゼロ終了する（`aggregate.py` の `LogIntegrityError` と同じ方針。壊れた run を
黙って除外しない）。

注意: N=4096 の `d2h` は run 内中央値そのものが二峰性（約 25 ms／約 620 ms）を
示すため、5 run 中央値は代表値として意味を持たない。本スクリプトは生値 5 個を
必ず併記し、判定側（doc §12）が二峰性として扱えるようにする。
"""

from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

SIZES = (1024, 2048, 4096)
VARIANTS = ("Select", "Classic")
PHASES = (
    "h2d_a",
    "h2d_b",
    "alloc_c",
    "launch_issue",
    "kernel_wait",
    "d2h",
    "host_copy",
)
RUNS = 5
HEADER_RE = re.compile(r"N=(\d+) kernel=(Select|Classic) \(median over")
PHASE_RE = re.compile(r"^\s*(\w+): median=([0-9.]+) ms")


class LogIntegrityError(Exception):
    """run ログの欠落・重複・フェーズ不足を表す（fail-closed）。"""


def parse_run(path: Path) -> dict[tuple[int, str], dict[str, float]]:
    cells: dict[tuple[int, str], dict[str, float]] = {}
    current: tuple[int, str] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        m = HEADER_RE.search(line)
        if m:
            key = (int(m.group(1)), m.group(2))
            if key in cells:
                raise LogIntegrityError(f"{path.name}: duplicate cell {key}")
            cells[key] = {}
            current = key
            continue
        pm = PHASE_RE.match(line)
        if pm and current is not None and pm.group(1) in PHASES:
            if pm.group(1) in cells[current]:
                raise LogIntegrityError(f"{path.name}: duplicate phase {pm.group(1)} in {current}")
            cells[current][pm.group(1)] = float(pm.group(2))
    for n in SIZES:
        for v in VARIANTS:
            key = (n, v)
            if key not in cells:
                raise LogIntegrityError(f"{path.name}: missing cell {key}")
            missing = [p for p in PHASES if p not in cells[key]]
            if missing:
                raise LogIntegrityError(f"{path.name}: cell {key} missing phases {missing}")
    return cells


def main() -> int:
    here = Path(__file__).resolve().parent
    runs = []
    for i in range(1, RUNS + 1):
        p = here / f"layerB-run{i}.log"
        if not p.is_file():
            raise LogIntegrityError(f"missing {p.name}")
        runs.append(parse_run(p))

    out: list[str] = []
    out.append("# イシュー #1973 Layer B 集計（自動生成。`aggregate_layer_b.py`）")
    out.append("")
    out.append(
        "各セルは run 内中央値（20 trial）の 5 run 中央値（ms）。括弧内は run1〜run5 の生値。"
        "`Σ(matmul 相当)` は `host_copy` を除く 6 フェーズの 5 run 中央値の和"
        "（doc §5 の Layer A `matmul` 相当区間）。"
    )
    out.append("")
    out.append("| N | kernel | " + " | ".join(PHASES) + " | Σ(matmul 相当) |")
    out.append("| --- | --- | " + " | ".join("---" for _ in PHASES) + " | --- |")
    for n in SIZES:
        for v in VARIANTS:
            row = [str(n), v]
            total = 0.0
            for ph in PHASES:
                vals = [r[(n, v)][ph] for r in runs]
                med = statistics.median(vals)
                if ph != "host_copy":
                    total += med
                raw = ", ".join(f"{x:.4f}" for x in vals)
                row.append(f"{med:.4f}（{raw}）")
            row.append(f"{total:.4f}")
            out.append("| " + " | ".join(row) + " |")
    out.append("")
    out.append(
        "注: N=4096 の `d2h` は run 内中央値自体が二峰性（約 25 ms／約 620 ms）のため "
        "5 run 中央値・Σ(matmul 相当) は代表値として扱わない（生値を参照）。"
    )
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except LogIntegrityError as e:
        print(f"LogIntegrityError: {e}", file=sys.stderr)
        sys.exit(2)
