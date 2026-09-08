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

PR #1459 codex-review 三度目・Cursor Bugbot 指摘の是正（イシュー #1253）:
`valid_run_ids` は監視ログ（`phase1_run{n}_monitor.log`）の BREACH 有無
のみで決まり、`orchestrate.sh` 側が確認している run 自体の終了コード
（`uptime_after_run{n}.txt` の `rc=`）・全サイズ計測完了（`phase1_run{n}.log`
に 5 サイズすべての `phase1_round_stats` 行が揃っているか）を一切参照して
いなかった。監視ループで一度 `ok`（BREACH なし）を記録した後に、cargo の
ビルド失敗・実行時クラッシュ・計測の途中中断が起きても、この不整合を
検出できず有効件数の分母・成立回数へ混入する（`orchestrate.sh` 側は
`RC -eq 0 && breach -eq 0` を要求しており両者は不整合だった）。本版は
`parse_exit_code`（`uptime_after_run{n}.txt` の `rc=` を読む）を追加し、
`classify_run` で breach／missing／rc 非 0・不明／サイズ欠落／ok の優先順位
で 1 run を分類する。`valid_run_ids` は「ok」（監視記録あり・BREACH なし・
rc=0・5 サイズ全計測済み）の run のみとし、それ以外は理由別に除外して
表の直前に明示する。

PR #1459 codex-review 五度目の指摘の是正（イシュー #1253・P2）: `orchestrate.sh`
は環境変数 `LOGDIR` で出力先ディレクトリを本ディレクトリ以外へ変更した
再試行に対応済みだが、本スクリプトは常にスクリプト配置元（`HERE`）から
ログを読んでいたため、別ディレクトリに正常な計測を保存しても古い結果
または「監視記録なし」が表示されていた。本版は集計対象ディレクトリを
第 1 引数 → 環境変数 `LOGDIR` → `HERE`（既定）の優先順位で解決し
（`resolve_logdir`）、`orchestrate.sh` と同じ `LOGDIR` 契約で入出力を揃える。
使用例: `LOGDIR=/path/to/out python3 aggregate.py` または
`python3 aggregate.py /path/to/out`（`orchestrate.sh` を同じ `LOGDIR` で
実行した出力先を指定する）。指定ディレクトリが存在しない場合は
`sys.exit(2)` で fail-closed に終了し、集計対象ディレクトリを出力冒頭に
明示する。
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def resolve_logdir(argv: list[str]) -> Path:
    """集計対象ディレクトリを第 1 引数 → 環境変数 `LOGDIR` → `HERE` の順で
    解決する（`orchestrate.sh` の `LOGDIR="${LOGDIR:-$SELF_DIR}"` と同じ既定
    値・同じ環境変数名。PR #1459 codex-review 五度目の指摘の是正）。

    存在しないディレクトリを黙って `HERE` へフォールバックすると、指定先
    のログが無いことに気づかずに古い結果を集計してしまうため、fail-closed
    に終了する。
    """
    if len(argv) > 2:
        print(
            f"使い方: {argv[0]} [LOGDIR]（または環境変数 LOGDIR）",
            file=sys.stderr,
        )
        sys.exit(2)
    if len(argv) == 2:
        raw = argv[1]
    else:
        raw = os.environ.get("LOGDIR") or str(HERE)
    logdir = Path(raw).expanduser().resolve()
    if not logdir.is_dir():
        print(
            f"エラー: 集計対象ディレクトリ '{logdir}' が存在しない"
            "（orchestrate.sh と同じ LOGDIR を指定すること）",
            file=sys.stderr,
        )
        sys.exit(2)
    return logdir
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


EXIT_CODE_RE = re.compile(r"rc=(-?\d+)")


def parse_exit_code(path: Path) -> int | None:
    """1 run の uptime_after_run{N}.txt から post-check 時点のベンチ終了コード
    （`orchestrate.sh` が書く「=== run${n} post-check ... rc=${RC} ===」行）
    を読む。

    PR #1459 codex-review 三度目・Cursor Bugbot 指摘の是正（イシュー
    #1253）: 監視ログ（`phase1_run{N}_monitor.log`）が BREACH を含まなく
    ても、ベンチ自体が非 0 終了（`cargo run` の required-features エラー・
    クラッシュ等）した run は排他条件不成立の run と同様に不完全な計測で
    あり、有効な計測と区別する必要がある。`orchestrate.sh` 側は
    `RC -eq 0 && breach -eq 0` を要求しており、集計側もこれと整合させる。

    戻り値: パースできた終了コード（int）。ファイル不在・`rc=` 未検出
    （判定不能）の場合は None。
    """
    if not path.exists():
        return None
    text = path.read_text()
    m = EXIT_CODE_RE.search(text)
    if not m:
        return None
    return int(m.group(1))


def classify_run(
    breach: str, exit_code: int | None, measured_sizes: int, total_sizes: int
) -> str:
    """1 run の有効性を「breach」「missing」「rc_unknown」「rc_nonzero」
    「incomplete」「ok」の 6 値へ分類する（優先順位はこの列挙順）。

    PR #1459 codex-review 三度目・Cursor Bugbot 指摘の是正（イシュー
    #1253）: 従来は `breach_status`（BREACH 行の有無）のみで「有効」を
    判定しており、監視ループで一度 `ok` を記録した後に run 自体が失敗・
    中断したケースを検出できなかった。本関数は監視結果に加えて
    `orchestrate.sh` の終了コード記録（`uptime_after_run{N}.txt` の
    `rc=`）・`phase1_run{N}.log` に 5 サイズ全ての `phase1_round_stats`
    行が揃っているか（途中中断の検出）も合わせて判定する。

    - "breach": 監視ログに BREACH 行あり（排他条件違反を検出済み）。
    - "missing": 監視ログ不在・空（排他条件成立を一度も確認できていない）。
    - "rc_unknown": `uptime_after_run{N}.txt` が不在、または `rc=` を
      読み取れない（終了コードが判定不能）。
    - "rc_nonzero": ベンチの終了コードが非 0（ビルド失敗・クラッシュ等）。
    - "incomplete": 終了コード 0 だが `phase1_run{N}.log` に 5 サイズ全て
      の `phase1_round_stats` 行が揃っていない（途中中断）。
    - "ok": 監視記録あり・BREACH なし・rc=0・5 サイズ全計測済み（有効）。
    """
    if breach == "breach":
        return "breach"
    if breach == "missing":
        return "missing"
    if exit_code is None:
        return "rc_unknown"
    if exit_code != 0:
        return "rc_nonzero"
    if measured_sizes != total_sizes:
        return "incomplete"
    return "ok"


def main() -> None:
    # 集計対象は orchestrate.sh と同じ LOGDIR 契約で解決する（第 1 引数 →
    # 環境変数 LOGDIR → HERE。PR #1459 codex-review 五度目の指摘の是正）。
    logdir = resolve_logdir(sys.argv)
    runs_data: dict[int, dict[int, dict]] = {}
    loads_before: dict[int, str] = {}
    breach_status: dict[int, str] = {}
    exit_codes: dict[int, int | None] = {}
    run_class: dict[int, str] = {}
    for n in RUNS:
        runs_data[n] = parse_run_log(logdir / f"phase1_run{n}.log")
        loads_before[n] = parse_load_before(logdir / f"uptime_before_run{n}.txt")
        breach_status[n] = parse_breach_status(logdir / f"phase1_run{n}_monitor.log")
        exit_codes[n] = parse_exit_code(logdir / f"uptime_after_run{n}.txt")
        run_class[n] = classify_run(
            breach_status[n], exit_codes[n], len(runs_data[n]), len(SIZES)
        )

    # BREACH・監視記録なし・終了コード判定不能／非 0・サイズ欠落（途中中断）
    # のいずれかに該当する run は「排他条件不成立、または計測が不完全」と
    # して全表（A/B/C）から除外する。除外した run 番号・理由は表出力の
    # 直前に明示する（PR #1459 codex-review 三度目・Cursor Bugbot 指摘の
    # 是正。イシュー #1253。`classify_run` の分類を参照）。
    valid_run_ids = [n for n in RUNS if run_class[n] == "ok"]
    excluded_breach_ids = [n for n in RUNS if run_class[n] == "breach"]
    excluded_missing_ids = [n for n in RUNS if run_class[n] == "missing"]
    excluded_rc_unknown_ids = [n for n in RUNS if run_class[n] == "rc_unknown"]
    excluded_rc_nonzero_ids = [n for n in RUNS if run_class[n] == "rc_nonzero"]
    excluded_incomplete_ids = [n for n in RUNS if run_class[n] == "incomplete"]
    excluded_run_ids = sorted(
        excluded_breach_ids
        + excluded_missing_ids
        + excluded_rc_unknown_ids
        + excluded_rc_nonzero_ids
        + excluded_incomplete_ids
    )

    lines: list[str] = []
    # 表示はリポジトリルート相対（`HERE` の 4 階層上）にし、絶対パス
    # （内部ホスト名・ユーザー名を含みうる）を集計出力へ残さない。
    # リポジトリ外を指定した場合のみ絶対パスのまま表示する。
    repo_root = HERE.parents[3]
    try:
        logdir_label = str(logdir.relative_to(repo_root))
    except ValueError:
        logdir_label = str(logdir)
    lines.append(f"集計対象ディレクトリ（LOGDIR）: `{logdir_label}`\n")
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
    if excluded_rc_unknown_ids:
        excluded_label = "、".join(f"run{n}" for n in excluded_rc_unknown_ids)
        lines.append(
            f"**注意**: {excluded_label} は `uptime_after_run${{n}}.txt` が"
            "存在しない、または `rc=` を読み取れないため、ベンチの終了コードを"
            "確認できない「判定不能」として以下の表から除外した"
            "（PR #1459 codex-review 三度目・Cursor Bugbot 指摘の是正）。\n"
        )
    if excluded_rc_nonzero_ids:
        excluded_label = "、".join(f"run{n}" for n in excluded_rc_nonzero_ids)
        lines.append(
            f"**注意**: {excluded_label} はベンチの終了コードが非 0"
            "（`uptime_after_run${n}.txt` の `rc=` 参照。ビルド失敗・実行時"
            "クラッシュ等）だったため、監視ログに BREACH がなくても以下の表"
            "から除外した（`orchestrate.sh` の `RC -eq 0 && breach -eq 0` と"
            "整合させるため。PR #1459 codex-review 三度目・Cursor Bugbot 指摘"
            "の是正）。\n"
        )
    if excluded_incomplete_ids:
        excluded_label = "、".join(f"run{n}" for n in excluded_incomplete_ids)
        lines.append(
            f"**注意**: {excluded_label} は終了コード 0 だが "
            f"`phase1_run${{n}}.log` に {len(SIZES)} サイズ全ての "
            "`phase1_round_stats` 行が揃っておらず、計測が途中で中断した"
            "とみなし以下の表から除外した"
            "（PR #1459 codex-review 三度目・Cursor Bugbot 指摘の是正）。\n"
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
    for n in excluded_rc_unknown_ids:
        lines.append(f"| run{n} | {loads_before.get(n, 'N/A')} | EXCLUDED（終了コード判定不能） | - |")
    for n in excluded_rc_nonzero_ids:
        lines.append(
            f"| run{n} | {loads_before.get(n, 'N/A')} | "
            f"EXCLUDED（rc={exit_codes.get(n)}） | - |"
        )
    for n in excluded_incomplete_ids:
        lines.append(f"| run{n} | {loads_before.get(n, 'N/A')} | EXCLUDED（計測途中中断） | - |")

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
