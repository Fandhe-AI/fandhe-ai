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

PR #1459 codex-review 指摘の是正（イシュー #1253）: `orchestrate.sh` は
run 実行中の排他条件違反（BREACH。load average 逸脱・他 GPU/build 系プロ
セスの残存）を検出した run を `valid_runs` からは除外するが、
`phase1_run${n}.log`（生の計測出力）自体は削除せず残す設計になってい
る。従来版はこの監視結果（`phase1_run${n}_monitor.log`）を一切参照せず
ログ内の `within_gate` のみで OK 表示・成立回数へ加算していたため、排他
条件不成立の計測が有効な計測と区別されずに集計表へ混入していた。本版は
`phase1_run${n}_monitor.log` に `BREACH` 行が 1 行でも存在する run を
「排他条件不成立」として全表から除外し、除外した run 数を明示する
（`valid_run_ids`／`excluded_run_ids`。表 B の分母・表下の gate 成立回数
の分母も除外後の run 数へ追従する）。

PR #1459 codex-review 再指摘の是正（イシュー #1253・P2）: 上記の
`parse_breach` は監視ログが存在しない・空の場合に `False`（BREACH では
ない）を返すため、その run は `valid_run_ids` へ算入されてしまう。監視
記録が存在しない run は「排他条件が成立していた」ことを一度も確認できて
おらず、計測ログすら無い状態でも分母が増えてしまうのは誤り。本版は
`parse_breach` を 3 値（"ok"／"breach"／"missing"）を返す
`parse_breach_status` へ置き換え、`missing`（監視ログ不在・空）の run も
`valid_run_ids` から除外したうえで、`breach` と区別して「監視記録なし・
判定不能」として明示する（除外理由を混同させない）。
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


def parse_breach_status(path: Path) -> str:
    """1 run の phase1_run{N}_monitor.log の監視結果を 3 値で判定する。

    `orchestrate.sh` は監視中に排他条件違反（load average 逸脱・他
    GPU/build 系プロセスの残存）を検出すると `... BREACH` 行を書き込み、
    その run を `valid_runs` から除外する（run 自体は最後まで実行し
    `phase1_run{N}.log` は残す）。本関数はその監視結果を読み、集計側で
    同じ run を排他条件不成立として除外するために使う。

    戻り値:
    - "breach": BREACH 行が 1 行でも存在する（排他条件違反を検出済み）。
    - "missing": ファイルが存在しない、または存在するが空（1 行も監視
      記録がない）。監視が一度も行われなかった、または監視ループが 1 度
      も反復せず run が終了した等が該当し、排他条件が成立していたことを
      一度も確認できていない「判定不能」の状態。
      PR #1459 codex-review 再指摘の是正（イシュー #1253・P2）: 従来版は
      この状態を「非 BREACH」（=有効）として `valid_run_ids` へ算入して
      いたため、計測・監視の記録が一切ない run でも分母に含まれてしまっ
      ていた。本版は "breach" と区別しつつも同じく除外対象とする。
    - "ok": 監視記録が存在し、BREACH 行を含まない（排他条件成立を確認
      できた）。
    """
    if not path.exists():
        return "missing"
    text = path.read_text()
    if not text.strip():
        return "missing"
    if "BREACH" in text:
        return "breach"
    return "ok"


def main() -> None:
    runs_data: dict[int, dict[int, dict]] = {}
    loads_before: dict[int, str] = {}
    breach_status: dict[int, str] = {}
    for n in RUNS:
        runs_data[n] = parse_run_log(HERE / f"phase1_run{n}.log")
        loads_before[n] = parse_load_before(HERE / f"uptime_before_run{n}.txt")
        breach_status[n] = parse_breach_status(HERE / f"phase1_run{n}_monitor.log")

    # BREACH が記録された run、および監視記録が存在しない／空の run（排他
    # 条件成立を一度も確認できていない「判定不能」）はいずれも「排他条件
    # 不成立」として全表（A/B/C）から除外する。除外した run 番号・理由は
    # 表出力の直前に明示する（PR #1459 codex-review 再指摘の是正。
    # イシュー #1253・P2。監視記録なしの run を有効な計測と同列に集計し
    # ない）。
    valid_run_ids = [n for n in RUNS if breach_status[n] == "ok"]
    excluded_breach_ids = [n for n in RUNS if breach_status[n] == "breach"]
    excluded_missing_ids = [n for n in RUNS if breach_status[n] == "missing"]
    excluded_run_ids = sorted(excluded_breach_ids + excluded_missing_ids)

    lines: list[str] = []
    if excluded_breach_ids:
        excluded_label = "、".join(f"run{n}" for n in excluded_breach_ids)
        lines.append(
            f"**注意**: {excluded_label} は実行中に排他条件違反（BREACH。"
            "`phase1_run${n}_monitor.log` 参照）を検出したため、以下の表から"
            "除外した（排他条件不成立の計測を有効な計測と同列に集計しないため）。\n"
        )
    if excluded_missing_ids:
        excluded_label = "、".join(f"run{n}" for n in excluded_missing_ids)
        lines.append(
            f"**注意**: {excluded_label} は `phase1_run${{n}}_monitor.log` が"
            "存在しない、または空（監視記録が一度もない）ため、排他条件成立を"
            "確認できない「判定不能」として以下の表から除外した"
            "（BREACH とは区別する。計測・監視の記録が揃い正常完了を確認できた"
            "run のみ有効とする）。\n"
        )

    lines.append("### 表 A: run × size の spread・ゲート判定\n")
    lines.append(
        "| size | "
        + " | ".join(f"run{n} spread" for n in valid_run_ids)
        + " | gate |"
    )
    lines.append("|---|" + "---|" * (len(valid_run_ids) + 1))
    for size in SIZES:
        row = [str(size)]
        gate_val = None
        for n in valid_run_ids:
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
    for n in valid_run_ids:
        d = runs_data.get(n, {})
        ok_count = sum(1 for size in SIZES if d.get(size, {}).get("within_gate"))
        measured = sum(1 for size in SIZES if size in d)
        all_ok = ok_count == len(SIZES) and measured == len(SIZES)
        lines.append(
            f"| run{n} | {loads_before.get(n, 'N/A')} | {all_ok} | {ok_count}/{measured if measured else 5} |"
        )
    for n in excluded_breach_ids:
        lines.append(f"| run{n} | {loads_before.get(n, 'N/A')} | EXCLUDED（BREACH） | - |")
    for n in excluded_missing_ids:
        lines.append(f"| run{n} | {loads_before.get(n, 'N/A')} | EXCLUDED（監視記録なし・判定不能） | - |")

    lines.append("")
    lines.append(f"| size | gate 成立回数 (/{len(valid_run_ids)}) |")
    lines.append("|---|---|")
    for size in SIZES:
        cnt = sum(
            1 for n in valid_run_ids if runs_data.get(n, {}).get(size, {}).get("within_gate")
        )
        lines.append(f"| {size} | {cnt}/{len(valid_run_ids)} |")

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
        f"run{n} max_round_idx (乖離ラウンド数)" for n in valid_run_ids
    ) + " |"
    lines.append(header_c)
    lines.append("|---|" + "---|" * len(valid_run_ids))
    for size in SIZES:
        row = [str(size)]
        for n in valid_run_ids:
            d = runs_data.get(n, {}).get(size)
            if d is None:
                row.append("N/A")
                continue
            row.append(f"{d['max_round_idx']} ({d['deviating_rounds']})")
        lines.append("| " + " | ".join(row) + " |")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
