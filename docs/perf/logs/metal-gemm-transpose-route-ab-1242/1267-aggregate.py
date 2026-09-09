#!/usr/bin/env python3
"""イシュー #1267: `1267-orchestrate.sh` が生成する本計測ログ
（`1267-attempt<N>-run.log`）・監視ログ（`1267-attempt<N>-monitor.log`）・
外側ゲートログ（`1267-gate.log`）・完了マーカー（`1267-DONE_*`）から、
phase 1 表・phase 2 30 セル表・`verdict` 確定値を Markdown へ転記する。

Python3 標準ライブラリのみ（依存追加なし。deps-policy.md 対象外）。

方針（既存 `aggregate.py`／`1261-aggregate.py` と同一）: 判定閾値
（`STABILITY_SPREAD_GATE`＝0.05・B/A>=1.0）はここで再定義・再判定せず、
ログ中の `gate=`／`within_gate=`／`verdict=`／`*_cell` 行を**転記のみ**
する。BREACH（実行中の排他条件逸脱）のある run は「排他条件未確認」と
して verdict 表から除外し理由を明示する（fail-closed。共有負荷下の値を
排他環境の結果として提示しない）。

使い方:
    python3 1267-aggregate.py [LOGDIR] > 1267-aggregate.md
    python3 1267-aggregate.py --self-test
"""

from __future__ import annotations

import os
import re
import sys

SELF_DIR = os.path.dirname(os.path.abspath(__file__))

CELL_RE = re.compile(
    r"^shape=\((?P<m>\d+),(?P<n>\d+),(?P<k>\d+)\) pattern=(?P<pattern>\S+) "
    r"cfg=(?P<cfg>\S+) resolved_matches_requested=(?P<resolved>true|false) "
    r"a_median_tflops=(?P<a>[-0-9.eE+]+) b_median_tflops=(?P<b>[-0-9.eE+]+) "
    r"b_over_a_tflops=(?P<ratio>[-0-9.eE+]+) spread_a=(?P<spread_a>[-0-9.eE+]+) "
    r"spread_b=(?P<spread_b>[-0-9.eE+]+) "
)
SKIPPED_RE = re.compile(r"^skipped_cell shape=\((\d+),(\d+),(\d+)\) pattern=(\S+)$")
GATE_EXCEEDED_RE = re.compile(
    r"^gate_exceeded_cell shape=\((\d+),(\d+),(\d+)\) pattern=(\S+) "
    r"spread_a=([-0-9.eE+]+) spread_b=([-0-9.eE+]+)$"
)
RESOLUTION_MISMATCHED_RE = re.compile(
    r"^resolution_mismatched_cell shape=\((\d+),(\d+),(\d+)\) pattern=(\S+)"
)
BELOW_THRESHOLD_RE = re.compile(
    r"^below_threshold_cell shape=\((\d+),(\d+),(\d+)\) pattern=(\S+) "
    r"b_over_a_tflops=([-0-9.eE+]+)$"
)
PHASE1_ROUND_STATS_RE = re.compile(
    r"^phase1_round_stats size=(?P<size>\d+) rounds=(?P<rounds>\d+) "
    r"spread=(?P<spread>[-0-9.eE+]+) gate=(?P<gate>[-0-9.eE+]+) "
    r"within_gate=(?P<within_gate>true|false)"
)
PHASE1_SUMMARY_RE = re.compile(
    r"^phase1_summary sizes_measured=(?P<measured>\d+) "
    r"sizes_gate_exceeded=(?P<exceeded>\S+) all_within_gate=(?P<all_ok>true|false)"
)
CELLS_SUMMARY_RE = re.compile(
    r"^cells_measured=(?P<measured>\d+) cells_skipped=(?P<skipped>\d+) "
    r"cells_below_threshold=(?P<below>\d+) cells_gate_exceeded=(?P<gate>\d+) "
    r"cells_resolution_mismatched=(?P<mismatch>\d+)"
)
MIN_RATIO_RE = re.compile(
    r"^min_b_over_a_tflops=([-0-9.eE+]+) at shape=\((\d+),(\d+),(\d+)\) pattern=(\S+)"
)
VERDICT_RE = re.compile(r"^verdict=(\S+)")
ENV_GUARD_RESULT_RE = re.compile(r"^env_guard_result=(\S+) attempts=(\d+)")
ENV_GUARD_BLOCK_RE = re.compile(r"^== env_guard\((\w+)\) ==$")


def parse_run_log(text: str) -> dict:
    """本計測ログ（`1267-attempt<N>-run.log`）を解析する純関数。
    キーが存在しない場合は空リスト／None のまま返す（呼び出し側が
    `--phase1-only` 相当の途中終了・`--guard-only` 等の他モードで生成
    されたログを渡しても壊れない）。"""
    lines = text.splitlines()
    env_guard_results: list[tuple[str, str, int]] = []  # (label, result, attempts)
    cur_label = None
    for line in lines:
        m = ENV_GUARD_BLOCK_RE.match(line)
        if m:
            cur_label = m.group(1)
            continue
        m = ENV_GUARD_RESULT_RE.match(line)
        if m:
            label = cur_label or "unknown"
            env_guard_results.append((label, m.group(1), int(m.group(2))))
            continue

    phase1_rows = []
    for line in lines:
        m = PHASE1_ROUND_STATS_RE.match(line)
        if m:
            phase1_rows.append(
                {
                    "size": int(m.group("size")),
                    "rounds": int(m.group("rounds")),
                    "spread": float(m.group("spread")),
                    "gate": float(m.group("gate")),
                    "within_gate": m.group("within_gate") == "true",
                }
            )

    phase1_summary = None
    for line in lines:
        m = PHASE1_SUMMARY_RE.match(line)
        if m:
            phase1_summary = {
                "measured": int(m.group("measured")),
                "exceeded": m.group("exceeded"),
                "all_ok": m.group("all_ok") == "true",
            }

    cells = []
    for line in lines:
        m = CELL_RE.match(line)
        if m:
            cells.append(
                {
                    "shape": (int(m.group("m")), int(m.group("n")), int(m.group("k"))),
                    "pattern": m.group("pattern"),
                    "cfg": m.group("cfg"),
                    "resolved": m.group("resolved") == "true",
                    "a_tflops": float(m.group("a")),
                    "b_tflops": float(m.group("b")),
                    "ratio": float(m.group("ratio")),
                    "spread_a": float(m.group("spread_a")),
                    "spread_b": float(m.group("spread_b")),
                }
            )

    skipped = [SKIPPED_RE.match(l).groups() for l in lines if SKIPPED_RE.match(l)]
    gate_exceeded = [
        GATE_EXCEEDED_RE.match(l).groups() for l in lines if GATE_EXCEEDED_RE.match(l)
    ]
    resolution_mismatched = [
        RESOLUTION_MISMATCHED_RE.match(l).groups()
        for l in lines
        if RESOLUTION_MISMATCHED_RE.match(l)
    ]
    below_threshold = [
        BELOW_THRESHOLD_RE.match(l).groups() for l in lines if BELOW_THRESHOLD_RE.match(l)
    ]

    cells_summary = None
    for line in lines:
        m = CELLS_SUMMARY_RE.match(line)
        if m:
            cells_summary = {
                "measured": int(m.group("measured")),
                "skipped": int(m.group("skipped")),
                "below": int(m.group("below")),
                "gate": int(m.group("gate")),
                "mismatch": int(m.group("mismatch")),
            }

    min_ratio = None
    for line in lines:
        m = MIN_RATIO_RE.match(line)
        if m:
            min_ratio = {
                "ratio": float(m.group(1)),
                "shape": (int(m.group(2)), int(m.group(3)), int(m.group(4))),
                "pattern": m.group(5),
            }

    verdict = None
    verdict_line = None
    for line in lines:
        m = VERDICT_RE.match(line)
        if m:
            verdict = m.group(1)
            verdict_line = line

    return {
        "env_guard_results": env_guard_results,
        "phase1_rows": phase1_rows,
        "phase1_summary": phase1_summary,
        "cells": cells,
        "skipped": skipped,
        "gate_exceeded": gate_exceeded,
        "resolution_mismatched": resolution_mismatched,
        "below_threshold": below_threshold,
        "cells_summary": cells_summary,
        "min_ratio": min_ratio,
        "verdict": verdict,
        "verdict_line": verdict_line,
    }


# 簡略版 monitor.log（`<timestamp> load1=<値>` のみ。ok/BREACH/
# UNDETERMINED 分類なし）を検出する正規表現。#1267 の実測（`1267-
# attempt{1,2}-monitor.log`）は外側ゲート／pgrep ベースの BREACH
# 分類器を経由せずこの形式で記録された（`docs/perf/metal-gemm-
# transpose-tiled.md` §5.9「実行方法」節参照）。
SIMPLE_LOAD_LINE_RE = re.compile(r"^\S+\s+load1=(?P<load1>[-0-9.]+|NA)\s*$")

# 外側ゲート（`gate_common.sh`／`wait_gate.sh`／`1267-orchestrate.sh`）の
# 事前宣言判定閾値と同一の値（load average(1 分) < 2.0）。簡略版
# monitor.log しか無い attempt（#1267 の実測）に対して、ここで load1
# の実測値から breach／undetermined を再判定するために使う。緩める
# 変更は不可（事前宣言パラメータ。同スクリプト冒頭コメント参照）。
MONITOR_LOAD_BREACH_THRESHOLD = 2.0


def parse_monitor_log(text: str) -> dict:
    """`1267-attempt<N>-monitor.log` を解析し BREACH/UNDETERMINED 件数・
    総サンプル数を返す純関数。ファイルが空（外側ゲート不通過で本計測が
    起動しなかった等）なら total=0 を返す。

    2 つの形式を扱う（PR #1462 codex-review 指摘: 従来は
    `1267-orchestrate.sh` が生成する形式のみを対象とし、それ以外の
    形式は暗黙に「breach=0（逸脱なし）」と誤読される余地があった）:

    (1) `1267-orchestrate.sh` が生成する形式（各行末尾が
        ` ok`／` BREACH`／` UNDETERMINED`）——従来どおりそのまま集計する。
    (2) #1267 の実測で実際に使われた簡略版（`SIMPLE_LOAD_LINE_RE`）。
        こちらは分類済みの行が 1 つもない場合にのみフォールバックで
        適用し、`MONITOR_LOAD_BREACH_THRESHOLD` 以上の load1 を breach・
        `NA`／解析不能な行を undetermined として集計し直す（実測値を
        黙って「breach=0」扱いにしない）。
    """
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return {"total": 0, "breach": 0, "undetermined": 0, "ok": 0, "format": "empty"}

    classified = sum(
        1
        for l in lines
        if l.endswith(" BREACH") or l.endswith(" UNDETERMINED") or l.endswith(" ok")
    )
    if classified > 0:
        breach = sum(1 for l in lines if l.endswith(" BREACH"))
        undetermined = sum(1 for l in lines if l.endswith(" UNDETERMINED"))
        ok = sum(1 for l in lines if l.endswith(" ok"))
        return {
            "total": len(lines),
            "breach": breach,
            "undetermined": undetermined,
            "ok": ok,
            "format": "orchestrate",
        }

    breach = 0
    undetermined = 0
    ok = 0
    matched = 0
    for l in lines:
        m = SIMPLE_LOAD_LINE_RE.match(l)
        if not m:
            undetermined += 1
            continue
        matched += 1
        raw = m.group("load1")
        if raw == "NA":
            undetermined += 1
            continue
        try:
            load1 = float(raw)
        except ValueError:
            undetermined += 1
            continue
        if load1 >= MONITOR_LOAD_BREACH_THRESHOLD:
            breach += 1
        else:
            ok += 1
    fmt = "simple_load" if matched > 0 else "unrecognized"
    return {
        "total": len(lines),
        "breach": breach,
        "undetermined": undetermined,
        "ok": ok,
        "format": fmt,
    }


def parse_gate_log(text: str) -> dict:
    """`1267-gate.log` から gate_result／gate_elapsed_secs／attempt を
    抽出する純関数（`wait_gate.sh`／本オーケストレータが書く要約行を
    対象。存在しなければ None のまま返す）。"""
    out: dict = {}
    for line in text.splitlines():
        if line.startswith("gate_start_unix="):
            for tok in line.split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    out[f"start_{k}"] = v
        if line.startswith("gate_end_unix="):
            for tok in line.split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    out[k] = v
    return out


def find_gate_result(logdir: str, attempt: int) -> tuple[str | None, str | None]:
    """外側ゲートの実行証跡（`gate.log`）を探し `gate_result` を返す。
    見つからなければ `(None, None)`（PR #1462 codex-review 指摘: 証跡
    がない attempt を証拠なく「通過（PASSED）」と表示しない——#1267 の
    実測は `1267-orchestrate.sh` を経由せずバイナリを直接起動したため
    `gate.log` 自体が存在しない。`docs/perf/metal-gemm-transpose-
    tiled.md` §5.9「実行方法」節参照）。

    探索先は 2 通り: 是正後（本 PR）の attempt 接尾辞付きパス
    `1267-attempt<N>-gate.log` を優先し、見つからなければ是正前の
    非接尾辞パス `1267-gate.log`（旧 `1267-orchestrate.sh` が生成し
    うる形式。複数 attempt 分が上書きされている可能性があるため
    後方互換の fallback に留める）を試す。"""
    candidates = [
        os.path.join(logdir, f"1267-attempt{attempt}-gate.log"),
        os.path.join(logdir, "1267-gate.log"),
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path) as f:
                info = parse_gate_log(f.read())
            return info.get("gate_result"), path
    return None, None


def discover_attempts(logdir: str) -> list[int]:
    attempts = set()
    for name in os.listdir(logdir):
        m = re.match(r"^1267-attempt(\d+)-run\.log$", name)
        if m:
            attempts.add(int(m.group(1)))
        m = re.match(r"^1267-DONE_(?:TIMEOUT_|GATE_LAUNCH_ERROR_)?ATTEMPT(\d+)$", name)
        if m:
            attempts.add(int(m.group(1)))
    return sorted(attempts)


def render_attempt(logdir: str, attempt: int) -> str:
    prefix = os.path.join(logdir, f"1267-attempt{attempt}-")
    out = [f"### attempt {attempt}\n"]

    timeout_marker = os.path.join(logdir, f"1267-DONE_TIMEOUT_ATTEMPT{attempt}")
    gate_error_marker = os.path.join(logdir, f"1267-DONE_GATE_LAUNCH_ERROR_ATTEMPT{attempt}")
    done_marker = os.path.join(logdir, f"1267-DONE_ATTEMPT{attempt}")

    if os.path.exists(timeout_marker):
        with open(timeout_marker) as f:
            content = f.read().strip()
        out.append(f"- 外側ゲート: **不通過（TIMEOUT）** — `{content}`\n")
        out.append(
            "- 本計測（phase 1 → phase 2）は未起動（fail-closed。"
            "計画 §3.4 と同じ「排他環境未確保」扱い）\n"
        )
        return "".join(out)

    if os.path.exists(gate_error_marker):
        with open(gate_error_marker) as f:
            content = f.read().strip()
        out.append(f"- 外側ゲート: **起動エラー** — `{content}`\n")
        out.append("- 本計測は未起動\n")
        return "".join(out)

    gate_result, gate_log_path = find_gate_result(logdir, attempt)
    if gate_result == "PASSED":
        out.append(f"- 外側ゲート: **通過（PASSED）** — `{os.path.basename(gate_log_path)}`\n")
    elif gate_result is not None:
        out.append(
            f"- 外側ゲート: 記録あり・結果は `{gate_result}`"
            f"（`{os.path.basename(gate_log_path)}`。PASSED 以外の値の場合は"
            "取り扱いを個別確認すること）\n"
        )
    else:
        out.append(
            "- 外側ゲート: **記録なし**（対応する `gate.log` が見つからない。"
            "`1267-orchestrate.sh` を経由せずバイナリを直接起動した可能性が"
            "あり、その場合は外側ゲートを経由していない——証拠がないため"
            "「通過（PASSED）」とは表示しない。`docs/perf/metal-gemm-"
            "transpose-tiled.md` §5.9「実行方法」節参照）\n"
        )

    run_log_path = prefix + "run.log"
    monitor_log_path = prefix + "monitor.log"
    if not os.path.exists(run_log_path):
        out.append("- 本計測ログが見つからない（異常終了の可能性）\n")
        return "".join(out)

    with open(run_log_path) as f:
        run = parse_run_log(f.read())
    monitor_log_exists = os.path.exists(monitor_log_path)
    monitor = {"total": 0, "breach": 0, "undetermined": 0, "ok": 0, "format": "missing"}
    if monitor_log_exists:
        with open(monitor_log_path) as f:
            monitor = parse_monitor_log(f.read())

    valid = "valid=?"
    if os.path.exists(done_marker):
        with open(done_marker) as f:
            valid = f.read().strip()
    out.append(f"- 完了記録: `{valid}`\n")
    out.append(
        f"- 実行中監視: total={monitor['total']} ok={monitor['ok']} "
        f"breach={monitor['breach']} undetermined={monitor['undetermined']} "
        f"(format={monitor.get('format', 'unknown')})\n"
    )

    # PR #1462 codex-review 指摘: 監視ログが欠落／空、または完了記録が
    # `valid=0` の場合でも従来はここが False のままになり verdict が
    # 確定扱いされる余地があった。いずれも「排他条件を確認できない」
    # 状態として明示的に除外理由へ含める（fail-closed）。
    monitor_missing = (not monitor_log_exists) or monitor["total"] == 0
    valid_is_zero = bool(re.search(r"(?:^|\s)valid=0(?:\s|$)", valid))
    excluded = (
        monitor["breach"] > 0
        or monitor["undetermined"] > 0
        or monitor_missing
        or valid_is_zero
    )
    if excluded:
        reasons = []
        if monitor["breach"] > 0:
            reasons.append(f"breach={monitor['breach']}")
        if monitor["undetermined"] > 0:
            reasons.append(f"undetermined={monitor['undetermined']}")
        if monitor_missing:
            reasons.append("実行中監視ログが欠落／空")
        if valid_is_zero:
            reasons.append("完了記録が valid=0")
        out.append(
            "- **排他条件を確認できない（" + "・".join(reasons) + "）ため、"
            "本 attempt は verdict 確定の対象から除外する**"
            "（fail-closed。共有負荷下の値・監視できていない値を排他"
            "環境の結果として提示しない）。\n"
        )

    out.append("\n#### env_guard 結果\n\n")
    out.append("| フェーズ | 結果 | attempts |\n|---|---|---|\n")
    for label, result, attempts_used in run["env_guard_results"]:
        out.append(f"| {label} | {result} | {attempts_used} |\n")

    if run["phase1_rows"]:
        out.append("\n#### phase 1（安定性セルフチェック）\n\n")
        out.append("| size | spread | gate | within_gate |\n|---|---|---|---|\n")
        for row in run["phase1_rows"]:
            out.append(
                f"| {row['size']} | {row['spread']:.4e} | {row['gate']:.4e} "
                f"| {row['within_gate']} |\n"
            )
        if run["phase1_summary"]:
            s = run["phase1_summary"]
            out.append(
                f"\nphase1_summary: sizes_measured={s['measured']} "
                f"sizes_gate_exceeded={s['exceeded']} all_within_gate={s['all_ok']}\n"
            )

    if run["cells"]:
        out.append("\n#### phase 2（30 セル A/B）\n\n")
        out.append(
            "| shape | pattern | cfg | resolved | a TFLOPS | b TFLOPS | B/A | "
            "spread_a | spread_b |\n"
            "|---|---|---|---|---|---|---|---|---|\n"
        )
        for c in run["cells"]:
            m, n, k = c["shape"]
            out.append(
                f"| ({m},{n},{k}) | {c['pattern']} | {c['cfg']} | {c['resolved']} | "
                f"{c['a_tflops']:.4f} | {c['b_tflops']:.4f} | {c['ratio']:.4f} | "
                f"{c['spread_a']:.4f} | {c['spread_b']:.4f} |\n"
            )

    if run["skipped"]:
        out.append("\nskipped cells:\n")
        for m, n, k, label in run["skipped"]:
            out.append(f"- shape=({m},{n},{k}) pattern={label}\n")
    if run["gate_exceeded"]:
        out.append("\ngate_exceeded cells:\n")
        for m, n, k, label, sa, sb in run["gate_exceeded"]:
            out.append(f"- shape=({m},{n},{k}) pattern={label} spread_a={sa} spread_b={sb}\n")
    if run["resolution_mismatched"]:
        out.append("\nresolution_mismatched cells:\n")
        for m, n, k, label in run["resolution_mismatched"]:
            out.append(f"- shape=({m},{n},{k}) pattern={label}\n")
    if run["below_threshold"]:
        out.append("\nbelow_threshold cells:\n")
        for m, n, k, label, ratio in run["below_threshold"]:
            out.append(f"- shape=({m},{n},{k}) pattern={label} b_over_a_tflops={ratio}\n")

    if run["cells_summary"]:
        s = run["cells_summary"]
        out.append(
            f"\ncells_measured={s['measured']} cells_skipped={s['skipped']} "
            f"cells_below_threshold={s['below']} cells_gate_exceeded={s['gate']} "
            f"cells_resolution_mismatched={s['mismatch']}\n"
        )
    if run["min_ratio"]:
        r = run["min_ratio"]
        m, n, k = r["shape"]
        out.append(f"min_b_over_a_tflops={r['ratio']:.4f} at shape=({m},{n},{k}) pattern={r['pattern']}\n")

    if run["verdict"]:
        raw_verdict = run["verdict"]
        effective_verdict = raw_verdict
        if excluded and raw_verdict in ("route_ok", "route_ng"):
            effective_verdict = "undetermined (排他条件逸脱のため確定判定から除外)"
        out.append(f"\n**verdict（生ログ）**: `{run['verdict_line']}`\n")
        out.append(f"**verdict（本 attempt の確定扱い）**: `{effective_verdict}`\n")

    return "".join(out)


def render(logdir: str) -> str:
    out = ["# イシュー #1267 集計（`1267-aggregate.py` 生成）\n"]
    attempts = discover_attempts(logdir)
    if not attempts:
        out.append("\n記録された attempt が見つからない。\n")
        return "".join(out)
    for attempt in attempts:
        out.append("\n")
        out.append(render_attempt(logdir, attempt))
    return "".join(out)


def _self_test() -> None:
    """合成ログで表生成ロジックを検証する（実機なしで CI 実行可能）。"""
    sample_run_log = """\
== env_guard(phase1) ==
env_guard_result=pass attempts=1
mode=phase1_only
phase1_round_stats size=256 rounds=10 spread=1.0000e-2 gate=5.0000e-2 within_gate=true median_secs=1.0e-3 min_secs=9.0e-4 min_round_idx=0 max_secs=1.1e-3 max_round_idx=3 round_medians_secs=1e-3,1e-3
phase1_summary sizes_measured=5 sizes_gate_exceeded=none all_within_gate=true
== env_guard(phase2) ==
env_guard_result=pass attempts=1
--- フェーズ 2: 転置タイル variant ルーティング A/B（A=classic strided / B=strided tiled）---
shape=(512,512,512) pattern=NT cfg=64x64x16_wm2wn2 resolved_matches_requested=true a_median_tflops=10.0000 b_median_tflops=11.0000 b_over_a_tflops=1.1000 spread_a=0.0100 spread_b=0.0100 a_round_tflops=[10.0] b_round_tflops=[11.0]
cells_measured=1 cells_skipped=0 cells_below_threshold=0 cells_gate_exceeded=0 cells_resolution_mismatched=0
min_b_over_a_tflops=1.1000 at shape=(512,512,512) pattern=NT
verdict=route_ok (全形状 × NT/TN/TT で B/A(TFLOPS) >= 1.0 かつ全セル spread が gate 内。結線可)
"""
    parsed = parse_run_log(sample_run_log)
    assert parsed["env_guard_results"] == [("phase1", "pass", 1), ("phase2", "pass", 1)]
    assert len(parsed["phase1_rows"]) == 1
    assert parsed["phase1_rows"][0]["within_gate"] is True
    assert parsed["phase1_summary"]["all_ok"] is True
    assert len(parsed["cells"]) == 1
    assert parsed["cells"][0]["ratio"] == 1.1
    assert parsed["cells_summary"]["measured"] == 1
    assert parsed["min_ratio"]["ratio"] == 1.1
    assert parsed["verdict"] == "route_ok"

    monitor_ok = "2026-01-01T00:00:00+0900 load1=1.0 load_error=0 other_count=0 other_procs=[] vanished=[] enum_error=0 ok\n"
    m = parse_monitor_log(monitor_ok)
    assert m == {"total": 1, "breach": 0, "undetermined": 0, "ok": 1, "format": "orchestrate"}

    monitor_breach = monitor_ok + (
        "2026-01-01T00:01:00+0900 load1=5.0 load_error=0 other_count=1 "
        "other_procs=[cargo:1234,] vanished=[] enum_error=0 BREACH\n"
    )
    m2 = parse_monitor_log(monitor_breach)
    assert m2 == {"total": 2, "breach": 1, "undetermined": 0, "ok": 1, "format": "orchestrate"}

    monitor_undetermined = "2026-01-01T00:02:00+0900 load1=NA load_error=1 other_count=0 other_procs=[] vanished=[] enum_error=0 UNDETERMINED\n"
    m3 = parse_monitor_log(monitor_undetermined)
    assert m3 == {"total": 1, "breach": 0, "undetermined": 1, "ok": 0, "format": "orchestrate"}

    # PR #1462 codex-review 指摘への回帰テスト: #1267 の実測で実際に
    # 使われた簡略版 monitor.log（`load1=` のみ・ok/BREACH/UNDETERMINED
    # 分類なし）を、黙って「breach=0」扱いにせず load1 の実測値から
    # 再判定できることを確認する。
    monitor_simple_ok = "2026-01-01T00:00:00+0900 load1=1.20\n2026-01-01T00:00:30+0900 load1=1.83\n"
    m4 = parse_monitor_log(monitor_simple_ok)
    assert m4 == {"total": 2, "breach": 0, "undetermined": 0, "ok": 2, "format": "simple_load"}

    monitor_simple_breach = monitor_simple_ok + "2026-01-01T00:01:00+0900 load1=9.27\n"
    m5 = parse_monitor_log(monitor_simple_breach)
    assert m5 == {"total": 3, "breach": 1, "undetermined": 0, "ok": 2, "format": "simple_load"}

    monitor_simple_na = "2026-01-01T00:00:00+0900 load1=NA\n"
    m6 = parse_monitor_log(monitor_simple_na)
    assert m6 == {"total": 1, "breach": 0, "undetermined": 1, "ok": 0, "format": "simple_load"}

    m7 = parse_monitor_log("")
    assert m7 == {"total": 0, "breach": 0, "undetermined": 0, "ok": 0, "format": "empty"}

    sample_undetermined_log = sample_run_log.replace(
        "verdict=route_ok (全形状 × NT/TN/TT で B/A(TFLOPS) >= 1.0 かつ全セル spread が gate 内。結線可)",
        "verdict=undetermined (フェーズ 2 のセルに skip がある)",
    )
    sample_undetermined_log = sample_undetermined_log.replace(
        "cells_measured=1 cells_skipped=0",
        "cells_measured=0 cells_skipped=1",
    )
    sample_undetermined_log += "skipped_cell shape=(1024,1024,1024) pattern=TN\n"
    parsed2 = parse_run_log(sample_undetermined_log)
    assert parsed2["verdict"] == "undetermined"
    assert parsed2["skipped"] == [("1024", "1024", "1024", "TN")]

    print("self-test OK", file=sys.stderr)


def main() -> None:
    if "--self-test" in sys.argv:
        _self_test()
        return
    logdir = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else SELF_DIR
    print(render(logdir))


if __name__ == "__main__":
    main()
