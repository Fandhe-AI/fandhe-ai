#!/usr/bin/env python3
"""イシュー #1255: 負荷環境での phase 1 spread 分布集計。

`1255-orchestrate.sh` が生成した `1255-phase1_run{n}.log`／
`1255-phase1_run{n}_monitor.log`／`1255-uptime_{before,after}_run{n}.txt`・
`1255-DONE` を読み、以下を Markdown として stdout へ出力する:

- 表 A: run × size の spread／within_gate／落ち込みラウンド（`max_round_idx`。
  `phase1_round_stats` は秒基準のため「最大」＝最も遅い＝最も TFLOPS が
  低いラウンド）・実行中 load1 中央値・load_class
- 表 B: サイズ別ゲート成立回数（分母は load_class=high かつ run_valid の
  run 数）
- 表 C: スパイク位置分布（`max_round_idx` の出現頻度）
- 参考表 R: #1187 試行 4（`docs/perf/logs/metal-gemm-transpose-route-ab-1187/
  route_ab_run4.log`）の `size=… spread=… round_tflops=[…]` 行を変換して
  併記する。TFLOPS は「最小＝秒最大」のため落ち込みラウンドは
  `argmin(round_tflops)` で求める。「非排他（開始時 load 1.38・実行中
  7〜11 へ上昇）」ラベルを付け、排他側としては提示しない
  （`docs/perf/metal-gemm-transpose-tiled.md` §5.4 参照）。
- 排他側: `1255-DONE`／`../metal-gemm-transpose-route-ab-1242/DONE_TIMEOUT_
  ATTEMPT{1,2}`（#1253。本ディレクトリ直下）を検出し
  「排他環境 valid_runs=0（§5.6 参照）」を固定文言で出力する。

閾値の再定義・再判定はしない。`gate=`／`within_gate=` はログの値を転記
するのみ（イシュー #1255 計画 §3.3・§3.4）。

`--self-test` で合成ログに対する自己検証を行う
（`1261-aggregate.py::self_test` と同型の位置づけ）。
"""
from __future__ import annotations

import os
import re
import sys
import tempfile
from typing import Optional

# LOGDIR は環境変数または第 1 引数で指定する。未指定時はこのスクリプトの
# 配置元ディレクトリを既定とする（orchestrate.sh と同じ入出力契約。
# #1253 orchestrate.sh 五度目の是正と同型）。
SELF_DIR = os.path.dirname(os.path.abspath(__file__))

ROUND_STATS_RE = re.compile(
    r"^phase1_round_stats size=(?P<size>\d+) rounds=(?P<rounds>\d+) "
    r"spread=(?P<spread>[0-9.eE+-]+) gate=(?P<gate>[0-9.eE+-]+) "
    r"within_gate=(?P<within_gate>true|false) "
    r"median_secs=(?P<median_secs>[0-9.eE+-]+) "
    r"min_secs=(?P<min_secs>[0-9.eE+-]+) min_round_idx=(?P<min_round_idx>\d+) "
    r"max_secs=(?P<max_secs>[0-9.eE+-]+) max_round_idx=(?P<max_round_idx>\d+) "
    r"round_medians_secs=(?P<round_medians_secs>.*)$"
)
RUN_CLASS_RE = re.compile(
    r"^RUN_CLASSIFICATION run=(?P<run>\d+) run_valid=(?P<run_valid>\d+) "
    r"load_class=(?P<load_class>\S+) median_load1=(?P<median_load1>\S+)$"
)
REF_LINE_RE = re.compile(
    r"^size=(?P<size>\d+) spread=(?P<spread>[0-9.]+) \((?P<verdict>[^)]*)\) "
    r"round_tflops=\[(?P<tflops>[^\]]*)\]$"
)


def logdir() -> str:
    if len(sys.argv) > 1 and sys.argv[1] != "--self-test":
        return os.path.abspath(sys.argv[1])
    env = os.environ.get("LOGDIR")
    if env:
        return os.path.abspath(env)
    return SELF_DIR


def parse_round_stats(log_path: str) -> list[dict]:
    if not os.path.isfile(log_path):
        return []
    rows = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            m = ROUND_STATS_RE.match(line.strip())
            if m:
                rows.append(m.groupdict())
    return rows


def parse_run_classification(monitor_log_path: str) -> Optional[dict]:
    if not os.path.isfile(monitor_log_path):
        return None
    last = None
    with open(monitor_log_path, encoding="utf-8") as f:
        for line in f:
            m = RUN_CLASS_RE.match(line.strip())
            if m:
                last = m.groupdict()
    return last


RUN_LOG_RE = re.compile(r"^1255-phase1_run(\d+)\.log$")


def discover_run_numbers(ld: str) -> list[int]:
    """`1255-phase1_run{n}.log` の実ファイル一覧から run 番号を検出する。

    `MAX_RUNS`（既定 5 だが `1255-orchestrate.sh` の環境変数で変更可能）を
    決め打ちせず、実ログの有無だけを根拠にすることで、MAX_RUNS を変更して
    実行した場合でも run6 以降が集計から黙って除外される事態を防ぐ
    （codex-review 指摘。`1255-DONE` の `runs_executed` とも整合する）。
    """
    if not os.path.isdir(ld):
        return []
    found = []
    for name in os.listdir(ld):
        m = RUN_LOG_RE.match(name)
        if m:
            found.append(int(m.group(1)))
    return sorted(found)


def build_table_a(ld: str, max_runs: Optional[int] = None) -> tuple[list[dict], int]:
    rows = []
    runs_found = 0
    run_numbers = discover_run_numbers(ld)
    if max_runs is not None:
        # 明示指定時は互換のため上限として尊重するが、既定（None）では
        # 実ログ検出数をそのまま使う（上記 docstring 参照）。
        run_numbers = [n for n in run_numbers if n <= max_runs]
    for n in run_numbers:
        log_path = os.path.join(ld, f"1255-phase1_run{n}.log")
        monitor_path = os.path.join(ld, f"1255-phase1_run{n}_monitor.log")
        if not os.path.isfile(log_path):
            continue
        runs_found += 1
        stats = parse_round_stats(log_path)
        cls = parse_run_classification(monitor_path) or {
            "run_valid": "0",
            "load_class": "undetermined",
            "median_load1": "NA",
        }
        for s in stats:
            rows.append(
                {
                    "run": n,
                    "size": int(s["size"]),
                    "spread": float(s["spread"]),
                    "gate": float(s["gate"]),
                    "within_gate": s["within_gate"],
                    "max_round_idx": int(s["max_round_idx"]),
                    "min_round_idx": int(s["min_round_idx"]),
                    "load_class": cls["load_class"],
                    "median_load1": cls["median_load1"],
                    "run_valid": cls["run_valid"],
                }
            )
    return rows, runs_found


def render_table_a(rows: list[dict]) -> str:
    if not rows:
        return "（run ログが見つからないため表 A は空）\n"
    out = [
        "| run | size | spread | gate | within_gate | 落ち込みラウンド(max_round_idx) | load_class | median_load1 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: (r["run"], r["size"])):
        out.append(
            f"| {r['run']} | {r['size']} | {r['spread']:.4e} | {r['gate']:.4e} | "
            f"{r['within_gate']} | {r['max_round_idx']} | {r['load_class']} | {r['median_load1']} |"
        )
    return "\n".join(out) + "\n"


def render_table_b(rows: list[dict]) -> str:
    high_valid_runs = {r["run"] for r in rows if r["load_class"] == "high" and r["run_valid"] == "1"}
    sizes = sorted({r["size"] for r in rows})
    out = [
        f"分母（load_class=high かつ run_valid=1 の run 数）: {len(high_valid_runs)}",
        "",
        "| size | within_gate 成立数 | 分母 |",
        "|---|---|---|",
    ]
    for size in sizes:
        gate_ok = sum(
            1
            for r in rows
            if r["size"] == size
            and r["run"] in high_valid_runs
            and r["within_gate"] == "true"
        )
        out.append(f"| {size} | {gate_ok} | {len(high_valid_runs)} |")
    return "\n".join(out) + "\n"


def render_table_c(rows: list[dict]) -> str:
    high_valid_rows = [r for r in rows if r["load_class"] == "high" and r["run_valid"] == "1"]
    if not high_valid_rows:
        return "（load_class=high の有効 run が無いため表 C は空）\n"
    counts: dict[int, int] = {}
    for r in high_valid_rows:
        counts[r["max_round_idx"]] = counts.get(r["max_round_idx"], 0) + 1
    out = ["| max_round_idx（0 始まり） | 出現回数 |", "|---|---|"]
    for idx in sorted(counts):
        out.append(f"| {idx} | {counts[idx]} |")
    return "\n".join(out) + "\n"


def render_reference_table(ref_path: str) -> str:
    if not os.path.isfile(ref_path):
        return "（参考ログが見つからないため参考表 R は空）\n"
    out = [
        "非排他（開始時 load 1.38・実行中 7〜11 へ上昇。#1187 試行 4。"
        "`docs/perf/metal-gemm-transpose-tiled.md` §5.4 参照）:",
        "",
        "| size | spread | verdict | 落ち込みラウンド(argmin(round_tflops)) |",
        "|---|---|---|---|",
    ]
    with open(ref_path, encoding="utf-8") as f:
        for line in f:
            m = REF_LINE_RE.match(line.strip())
            if not m:
                continue
            tflops = [float(x) for x in m.group("tflops").split(",") if x.strip()]
            if not tflops:
                continue
            argmin_idx = min(range(len(tflops)), key=lambda i: tflops[i])
            out.append(
                f"| {m.group('size')} | {m.group('spread')} | {m.group('verdict')} | {argmin_idx} |"
            )
    return "\n".join(out) + "\n"


def render_exclusive_side(ld: str) -> str:
    parent_dir = os.path.dirname(ld) if os.path.basename(ld) else ld
    # 1255-* は 1242 ディレクトリ直下に置くため、排他側マーカー
    # （#1253 由来の DONE_TIMEOUT_ATTEMPT{1,2}）は同じディレクトリにある。
    markers = [
        os.path.join(ld, "DONE_TIMEOUT_ATTEMPT1"),
        os.path.join(ld, "DONE_TIMEOUT_ATTEMPT2"),
    ]
    found = [m for m in markers if os.path.isfile(m)]
    lines = ["排他環境 valid_runs=0（#1253。`docs/perf/metal-gemm-transpose-tiled.md` §5.6 参照）"]
    if found:
        lines.append(f"検出したマーカー: {', '.join(os.path.basename(m) for m in found)}")
    else:
        lines.append("（本ディレクトリに #1253 の DONE_TIMEOUT_ATTEMPT{1,2} マーカーは見つからなかった。"
                      "§5.6 の記述を正とする）")
    del parent_dir
    return "\n".join(lines) + "\n"


def render_done_summary(ld: str) -> str:
    done_path = os.path.join(ld, "1255-DONE")
    if not os.path.isfile(done_path):
        return "（1255-DONE が見つからない。実行未完了または未実行）\n"
    with open(done_path, encoding="utf-8") as f:
        return f.read().strip() + "\n"


def main() -> None:
    ld = logdir()
    print("# イシュー #1255 集計結果\n")
    print("## 1255-DONE\n")
    print(render_done_summary(ld))
    rows, runs_found = build_table_a(ld)
    print(f"\n（run ログ検出数: {runs_found}）\n")
    print("## 表 A: run × size\n")
    print(render_table_a(rows))
    print("## 表 B: サイズ別ゲート成立回数（分母: load_class=high かつ run_valid の run 数）\n")
    print(render_table_b(rows))
    print("## 表 C: スパイク位置分布（load_class=high かつ run_valid の run のみ）\n")
    print(render_table_c(rows))
    print("## 参考表 R: #1187 試行 4（非排他）\n")
    ref_path = os.path.join(
        os.path.dirname(ld), "metal-gemm-transpose-route-ab-1187", "route_ab_run4.log"
    )
    print(render_reference_table(ref_path))
    print("## 排他側\n")
    print(render_exclusive_side(ld))


# ---------------------------------------------------------------------
# self-test: 合成ログで表 A〜C の生成と分類の転記を検証する
# （実測値の正しさではなく、パーサ・集計ロジック自体の自己検証。
# `1261-aggregate.py --self-test` と同じ位置づけ）。
# ---------------------------------------------------------------------
def self_test() -> None:
    with tempfile.TemporaryDirectory() as td:
        # run1: high・within_gate 成立
        with open(os.path.join(td, "1255-phase1_run1.log"), "w", encoding="utf-8") as f:
            f.write(
                "mode=phase1_only\n"
                "phase1_round_stats size=256 rounds=10 spread=1.0000e-02 gate=5.0000e-02 "
                "within_gate=true median_secs=1.000000e-03 min_secs=9.900000e-04 min_round_idx=2 "
                "max_secs=1.010000e-03 max_round_idx=7 round_medians_secs=1e-3,1e-3\n"
                "verdict=not_evaluated (--phase1-only)\n"
            )
        with open(os.path.join(td, "1255-phase1_run1_monitor.log"), "w", encoding="utf-8") as f:
            f.write(
                "2026-09-09T00:00:00+0900 load1=3.5 load_error=0\n"
                "2026-09-09T00:00:30+0900 load1=4.1 load_error=0\n"
                "RUN_CLASSIFICATION run=1 run_valid=1 load_class=high median_load1=3.8\n"
            )
        # run2: mid・within_gate 不成立
        with open(os.path.join(td, "1255-phase1_run2.log"), "w", encoding="utf-8") as f:
            f.write(
                "mode=phase1_only\n"
                "phase1_round_stats size=256 rounds=10 spread=1.0000e-01 gate=5.0000e-02 "
                "within_gate=false median_secs=1.000000e-03 min_secs=9.000000e-04 min_round_idx=0 "
                "max_secs=1.100000e-03 max_round_idx=3 round_medians_secs=1e-3,1e-3\n"
                "verdict=not_evaluated (--phase1-only)\n"
            )
        with open(os.path.join(td, "1255-phase1_run2_monitor.log"), "w", encoding="utf-8") as f:
            f.write(
                "2026-09-09T00:05:00+0900 load1=2.5 load_error=0\n"
                "RUN_CLASSIFICATION run=2 run_valid=1 load_class=mid median_load1=2.5\n"
            )
        rows, runs_found = build_table_a(td)
        assert runs_found == 2, runs_found
        assert len(rows) == 2, rows
        table_a = render_table_a(rows)
        assert "high" in table_a and "mid" in table_a
        table_b = render_table_b(rows)
        assert "分母（load_class=high かつ run_valid=1 の run 数）: 1" in table_b
        assert "| 256 | 1 | 1 |" in table_b
        table_c = render_table_c(rows)
        assert "| 7 | 1 |" in table_c
        exclusive = render_exclusive_side(td)
        assert "valid_runs=0" in exclusive
        print("self-test: ok")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        self_test()
    else:
        main()
