#!/usr/bin/env python3
"""イシュー #1253: 排他環境での phase1-only 3 run から spread 分布表を再生成する。

`phase1_run{1,2,3}.log`（`gemm_transpose_route_ab_bench --phase1-only` の
標準出力）に含まれる `phase1_round_stats` 行（size 別のラウンド別 min/max・
spread・median）と `uptime_before_run{N}.txt`（実行直前 load average）を
key=value 形式で単純パースし、run × size の表（A: spread/ゲート判定、
B: ゲート成立回数、C: スパイク位置）を Markdown で標準出力へ書く。

Python3 標準ライブラリのみ（前例:
`docs/perf/logs/cpu-gemm-ic-dynamic-ab-1367/aggregate.py` と同方針）。
本スクリプト自体は `STABILITY_SPREAD_GATE` 等の閾値を再定義せず、
ログ行に既に埋め込まれた `gate=`／`within_gate=` の値をそのまま転記する
（閾値の単一真実源は `crates/bench-harness/src/ab.rs`。本スクリプトは
数値を変更・再判定しない）。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIZES = [256, 512, 1024, 2048, 4096]
RUNS = [1, 2, 3]

ROUND_STATS_RE = re.compile(
    r"^phase1_round_stats size=(?P<size>\d+) rounds=(?P<rounds>\d+) "
    r"spread=(?P<spread>[0-9.eE+-]+) gate=(?P<gate>[0-9.eE+-]+) "
    r"within_gate=(?P<within_gate>[Tt]rue|[Ff]alse) median_secs=(?P<median_secs>[0-9.eE+-]+) "
    r"min_secs=(?P<min_secs>[0-9.eE+-]+) min_round_idx=(?P<min_round_idx>\d+) "
    r"max_secs=(?P<max_secs>[0-9.eE+-]+) max_round_idx=(?P<max_round_idx>\d+) "
    r"round_medians_secs=(?P<round_medians_secs>[0-9.eE+,-]+)$"
)

LOAD_RE = re.compile(r"load averages?:\s*([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)")


def parse_run_log(path: Path) -> dict[int, dict]:
    """1 run の phase1_run{N}.log から size -> phase1_round_stats dict を作る。"""
    out: dict[int, dict] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        m = ROUND_STATS_RE.match(line.strip())
        if not m:
            continue
        d = m.groupdict()
        size = int(d["size"])
        round_medians = [float(x) for x in d["round_medians_secs"].split(",")]
        median = float(d["median_secs"])
        gate = float(d["gate"])
        # 診断用（ゲート判定には使わない）: 単発スパイクか広範な散らばりかの補助指標。
        deviating = sum(
            1 for v in round_medians if abs(v - median) > gate * median
        )
        out[size] = {
            "spread": float(d["spread"]),
            "gate": gate,
            "within_gate": d["within_gate"].lower() == "true",
            "median_secs": median,
            "min_secs": float(d["min_secs"]),
            "min_round_idx": int(d["min_round_idx"]),
            "max_secs": float(d["max_secs"]),
            "max_round_idx": int(d["max_round_idx"]),
            "round_medians_secs": round_medians,
            "deviating_rounds": deviating,
        }
    return out


def parse_load_before(path: Path) -> str:
    if not path.exists():
        return "N/A"
    text = path.read_text()
    m = LOAD_RE.search(text)
    if not m:
        return "N/A"
    return f"{m.group(1)} {m.group(2)} {m.group(3)}"


def main() -> None:
    runs_data: dict[int, dict[int, dict]] = {}
    loads_before: dict[int, str] = {}
    for n in RUNS:
        runs_data[n] = parse_run_log(HERE / f"phase1_run{n}.log")
        loads_before[n] = parse_load_before(HERE / f"uptime_before_run{n}.txt")

    lines: list[str] = []
    lines.append("### 表 A: run × size の spread・ゲート判定\n")
    lines.append("| size | " + " | ".join(f"run{n} spread" for n in RUNS) + " | gate |")
    lines.append("|---|" + "---|" * (len(RUNS) + 1))
    for size in SIZES:
        row = [str(size)]
        gate_val = None
        for n in RUNS:
            d = runs_data.get(n, {}).get(size)
            if d is None:
                row.append("N/A")
                continue
            gate_val = d["gate"]
            mark = "OK" if d["within_gate"] else "NG"
            row.append(f"{d['spread']:.4e} ({mark})")
        row.append(f"{gate_val:.4e}" if gate_val is not None else "N/A")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("### 表 B: run ごとのゲート成立状況・サイズごとの成立回数\n")
    lines.append("| run | 実行直前 load average (1/5/15) | all_within_gate | gate 成立サイズ数 (/5) |")
    lines.append("|---|---|---|---|")
    for n in RUNS:
        d = runs_data.get(n, {})
        ok_count = sum(1 for size in SIZES if d.get(size, {}).get("within_gate"))
        measured = sum(1 for size in SIZES if size in d)
        all_ok = ok_count == len(SIZES) and measured == len(SIZES)
        lines.append(
            f"| run{n} | {loads_before.get(n, 'N/A')} | {all_ok} | {ok_count}/{measured if measured else 5} |"
        )

    lines.append("")
    lines.append("| size | gate 成立回数 (/3) |")
    lines.append("|---|---|")
    for size in SIZES:
        cnt = sum(1 for n in RUNS if runs_data.get(n, {}).get(size, {}).get("within_gate"))
        lines.append(f"| {size} | {cnt}/3 |")

    lines.append("")
    lines.append(
        "### 表 C: スパイク位置（max_round_idx。秒基準・0 始まり）と診断用「乖離ラウンド数」\n"
    )
    lines.append(
        "診断用列は `|median_i - median| > gate * median` を満たすラウンド数。"
        "単発スパイク（1 ラウンドのみ乖離）か広範な散らばりかを示す**補助指標であり、"
        "ゲート判定には使わない**（ゲート判定は表 A の spread 列が正）。\n"
    )
    header_c = "| size | " + " | ".join(
        f"run{n} max_round_idx (乖離ラウンド数)" for n in RUNS
    ) + " |"
    lines.append(header_c)
    lines.append("|---|" + "---|" * len(RUNS))
    for size in SIZES:
        row = [str(size)]
        for n in RUNS:
            d = runs_data.get(n, {}).get(size)
            if d is None:
                row.append("N/A")
                continue
            row.append(f"{d['max_round_idx']} ({d['deviating_rounds']})")
        lines.append("| " + " | ".join(row) + " |")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
