#!/usr/bin/env python3
"""Layer B（gemm_reuse_phase_diag_cpu、イシュー #1301）5 run 集計。

`layerB-<node>-<arm>-run{1..5}.log`（イシュー #1292 の
`run_layerB_m4max.sh`／`run_layerB_dgx.sh` と同一フォーマット。各 run は
テスト自身が 20 trial の中央値を出す `median=... ms` 行を N=512/1024/2048
ごとに `alloc_c`／`kernel`／`ops_gemm` 等の区間名で出力する）を読み、5 run
分の `median=` 値からさらに中央値を取って on/off 比を Markdown 表にする
（coding-rust.md「ベンチは 5 回計測の中央値」に従い、プロセス起動 5 回の
中央値を最終値とする。標準ライブラリのみで完結させる。#1301）。

使い方:
  python3 aggregate_layer_b.py --node dgx --off-glob 'layerB-dgx-off-run*.log' \\
      --on-glob 'layerB-dgx-on-run*.log' > aggregate-dgx.md
"""
from __future__ import annotations

import argparse
import glob
import re
import statistics
import sys

LINE_RE = re.compile(
    r"^\s*(?P<phase>[A-Za-z_]+)(?:\s*\([^)]*\))?:\s*median=(?P<median>[0-9.]+)\s*ms"
)
SIZE_RE = re.compile(r"^\s*N=(?P<n>\d+)\s*\(median over")

PHASES = ["alloc_c", "kernel", "tensor_wrap", "ops_gemm", "tape_matmul", "to_tensor", "host_copy", "checksum"]


def parse_log(path: str) -> dict[int, dict[str, float]]:
    """1 run のログを { N: { phase: median_ms } } へパースする。"""
    out: dict[int, dict[str, float]] = {}
    cur_n: int | None = None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m_size = SIZE_RE.match(line)
            if m_size:
                cur_n = int(m_size.group("n"))
                out.setdefault(cur_n, {})
                continue
            m = LINE_RE.match(line)
            if m and cur_n is not None:
                phase = m.group("phase")
                if phase in PHASES:
                    out[cur_n][phase] = float(m.group("median"))
    return out


def aggregate(paths: list[str]) -> dict[int, dict[str, list[float]]]:
    """複数 run 分の parse_log 結果を { N: { phase: [values] } } へ束ねる。"""
    agg: dict[int, dict[str, list[float]]] = {}
    for p in paths:
        parsed = parse_log(p)
        for n, phases in parsed.items():
            agg.setdefault(n, {})
            for phase, val in phases.items():
                agg[n].setdefault(phase, []).append(val)
    return agg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", required=True)
    ap.add_argument("--off-glob", required=True)
    ap.add_argument("--on-glob", required=True)
    args = ap.parse_args()

    off_paths = sorted(glob.glob(args.off_glob))
    on_paths = sorted(glob.glob(args.on_glob))
    if not off_paths or not on_paths:
        print(f"ERROR: off={len(off_paths)} on={len(on_paths)} 件（0 件は不可）", file=sys.stderr)
        return 1

    off_agg = aggregate(off_paths)
    on_agg = aggregate(on_paths)

    print(f"# Layer B 集計（node={args.node}。イシュー #1301）\n")
    print(f"off runs: {off_paths}")
    print(f"on runs: {on_paths}\n")

    sizes = sorted(set(off_agg.keys()) | set(on_agg.keys()))
    for n in sizes:
        print(f"## N={n}\n")
        print("| phase | off median (of 5 run medians, ms) | off n | on median (ms) | on n | on/off 比 |")
        print("|---|---|---|---|---|---|")
        for phase in PHASES:
            off_vals = off_agg.get(n, {}).get(phase, [])
            on_vals = on_agg.get(n, {}).get(phase, [])
            off_med = statistics.median(off_vals) if off_vals else float("nan")
            on_med = statistics.median(on_vals) if on_vals else float("nan")
            ratio = (on_med / off_med) if off_vals and on_vals and off_med != 0 else float("nan")
            print(
                f"| {phase} | {off_med:.4f} | {len(off_vals)} | {on_med:.4f} | {len(on_vals)} | {ratio:.4f} |"
            )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
