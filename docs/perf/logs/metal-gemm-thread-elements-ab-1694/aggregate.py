#!/usr/bin/env python3
"""thread_elements() 方式 BlockMMA 候補（イシュー #1693）の M4 Max A/B
（イシュー #1694）ログを集計する。

`crates/backend-metal/src/gemm_te_diag_tests.rs::
te_kernel_gpu_ab_vs_production_select` が各プロセス起動
（`kernel_gpu_te_ab_run<N>.log`）へ出力する以下の行を解析する:

    N=<n> pair=te_vs_production_select production_select_resolved=<cfg>
    N=<n> pair=te_vs_production_select mode=base kernel_gpu_median_ms=<x> q1=<y> q3=<z>
    N=<n> pair=te_vs_production_select mode=head kernel_gpu_median_ms=<x> q1=<y> q3=<z>
    N=<n> head_over_base_kernel_gpu=<r> base_checksum=<f64:e> head_checksum=<f64:e> bit_identical=<bool>

イシュー #1694 issue コメントの事前登録判定規則（本ファイル冒頭コメント
と同一の内容。事後緩和禁止）:

1. 前提ゲート（R0〜R3）は `orchestrate.sh gate` が実行する（本スクリプト
   はそのログを追加入力として受け取り検証する。R0/R1 FAIL は REJECT 確定・
   性能 A/B 非実施。R2/R3 FAIL は機構契約の不成立として記録し、性能 A/B は
   実施しても総合判定は undetermined）。`--gate-dir` を渡さない場合・
   ゲートログから前提充足を確認できない場合は、N ごとの表は参考情報として
   出力するが総合判定は必ず undetermined とする（codex-review・Cursor
   Bugbot 指摘。イシュー #1694 PR レビュー是正）。
2. 正しさ（REQ-2）: `gemm_te_diag_tests.rs` 内の `assert_parity` が
   fail-closed に検証済み（本スクリプトは追加の正しさ判定を行わない）。
3. checksum 完全一致: 1 run・1 N でも `bit_identical=false` ならその N は
   比の値によらず REJECT。
4. 性能指標: `head_over_base_kernel_gpu` の N ごと 5 run 中央値。
5. N ごとの判定:
   - 5 run 中央値 <= 1.00 かつ 5/5 run 符号一貫（全 run <= 1.00）
     -> ADOPT-as-opt-in-candidate
   - 5 run 中央値 > 1.00 かつ 5/5 run 符号一貫 -> REJECT
   - 符号が run 間で反転 -> undetermined
6. 総合判定: 全 N が ADOPT なら候補前進を推奨。1 N でも REJECT なら
   無条件前進は推奨しない。undetermined を含めば総合も undetermined。
   いずれの場合も本番結線は行わない。
7. 負荷: record_only（本スクリプトは判定に使わず記録のみ）。
8. フォールバック: `resolved_cfg != cfg` は診断テスト側の assert で
   run 自体が abort するため、本スクリプトの対象ログには現れない
   （abort した run はログが不完全になり、本スクリプトの完全性検査
   （期待 N 集合の充足）で undetermined として検出される）。
9. ちょうど 5 run が正式判定の対象（`MIN_FORMAL_RUNS`）。

python3 標準ライブラリのみ（許容依存外の追加なし）。

使い方:
    python3 aggregate.py --self-test
    python3 aggregate.py --gate-dir ../metal-gemm-thread-elements-1693 \
        kernel_gpu_te_ab_run1.log kernel_gpu_te_ab_run2.log \
        kernel_gpu_te_ab_run3.log kernel_gpu_te_ab_run4.log \
        kernel_gpu_te_ab_run5.log > aggregate.md

`--gate-dir` を省略した場合・指定先に R0〜R3 ゲートログ（`orchestrate.sh
gate` の出力）が見つからない場合は、前提ゲート未確認として総合判定を
必ず undetermined にする（N ごとの表は診断のため出力するが、それを
理由に ADOPT／REJECT を確定させない）。
"""

from __future__ import annotations

import argparse
import os
import re
import statistics
import sys
import tempfile
from dataclasses import dataclass, field

MIN_FORMAL_RUNS = 5
EXPECTED_SIZES = (512, 1024, 2048, 4096)

# `N=<n> head_over_base_kernel_gpu=<r> base_checksum=<be> head_checksum=<he>
# bit_identical=<bool>` 行（行頭アンカーなし。`test <name> ...` プレフィクス
# を伴う `cargo test --nocapture` 出力にも対応するため）。
_RESULT_RE = re.compile(
    r"N=(?P<n>\d+)\s+head_over_base_kernel_gpu=(?P<ratio>[0-9.eE+-]+)\s+"
    r"base_checksum=(?P<base_checksum>\S+)\s+"
    r"head_checksum=(?P<head_checksum>\S+)\s+"
    r"bit_identical=(?P<bit_identical>true|false)"
)


@dataclass
class SizeResult:
    ratio: float
    bit_identical: bool
    base_checksum: str
    head_checksum: str


@dataclass
class RunData:
    """1 run（1 ログファイル）から抽出した N ごとの結果。

    `by_size` は N ごとに観測した `SizeResult` の全件を保持する（辞書での
    上書きにせず list を使う）。同一 N の行が複数回出力された場合、後発の
    行が `bit_identical=true` でも先行の `bit_identical=false` を隠して
    ADOPT 誤判定を招かないようにするため（codex-review 指摘。イシュー
    #1694 PR レビュー是正）。正常系（1 run に 1 N=1 行）では重複は生じず
    `check_completeness` が非重複を検査する。
    """

    by_size: dict[int, list[SizeResult]] = field(default_factory=dict)


def parse_log(text: str) -> RunData:
    run = RunData()
    for m in _RESULT_RE.finditer(text):
        n = int(m.group("n"))
        run.by_size.setdefault(n, []).append(
            SizeResult(
                ratio=float(m.group("ratio")),
                bit_identical=(m.group("bit_identical") == "true"),
                base_checksum=m.group("base_checksum"),
                head_checksum=m.group("head_checksum"),
            )
        )
    return run


def check_completeness(runs: list[RunData]) -> list[str]:
    """各 run がちょうど `EXPECTED_SIZES` の全 N を 1 回ずつ含むかを検査
    する（欠落・重複は run 自体の中断・多重実行を示唆する）。同一 N の
    結果行が 2 回以上現れる場合も重複として検出する（辞書上書きによる
    `bit_identical=false` の隠蔽を防ぐため。regression テスト
    `_self_test` ケース 7 参照）。
    """
    violations: list[str] = []
    for i, run in enumerate(runs, start=1):
        missing = [n for n in EXPECTED_SIZES if n not in run.by_size]
        if missing:
            violations.append(f"run{i}: N={missing} の結果行が見つからない")
        extra = [n for n in run.by_size if n not in EXPECTED_SIZES]
        if extra:
            violations.append(f"run{i}: 想定外の N={extra} の結果行がある")
        duplicated = [n for n, results in run.by_size.items() if len(results) > 1]
        if duplicated:
            violations.append(
                f"run{i}: N={sorted(duplicated)} の結果行が複数回出力されて"
                "いる（同一 run の多重実行・ログ結合ミスの可能性。"
                "手動で正しいログへ差し替える）"
            )
    return violations


def _single_result(run: RunData, n: int) -> SizeResult:
    """`check_completeness` で非重複・非欠落が確認済みの前提で、N の唯一の
    `SizeResult` を返す（呼び出し前に完全性検査を通すこと）。
    """
    results = run.by_size[n]
    assert len(results) == 1, f"N={n} の結果行が {len(results)} 件（重複未検出）"
    return results[0]


def check_bit_identical(runs: list[RunData]) -> dict[int, bool]:
    """N ごとに全 run で `bit_identical=true` かを判定する（規則 3）。"""
    result: dict[int, bool] = {}
    for n in EXPECTED_SIZES:
        result[n] = all(
            n in run.by_size and _single_result(run, n).bit_identical
            for run in runs
        )
    return result


def judge_size(ratios: list[float]) -> str:
    """規則 5: N ごとの判定（ADOPT-as-opt-in-candidate / REJECT /
    undetermined）。"""
    if not ratios:
        return "undetermined"
    median = statistics.median(ratios)
    all_le_one = all(r <= 1.00 for r in ratios)
    all_gt_one = all(r > 1.00 for r in ratios)
    if median <= 1.00 and all_le_one:
        return "ADOPT-as-opt-in-candidate"
    if median > 1.00 and all_gt_one:
        return "REJECT"
    return "undetermined"


# `orchestrate.sh gate` が `../metal-gemm-thread-elements-1693/` へ残す
# R0〜R3 ゲートログのファイル名 -> テスト短縮名（モジュールパス省略形）。
# ファイル単位ではなくテスト名で R0/R1（正しさ不成立で REJECT 確定）と
# R2/R3（機構契約。FAIL でも性能 A/B は参考値扱いで総合 undetermined）を
# 区別する。`orchestrate.sh` の R1 ステップは `parity_run.log` へ R2
# （`te_rejects_non_staged_candidate`）の出力を追記するため、ファイル
# 単位の判定では R1/R2 を区別できない点に注意（codex-review 指摘。
# イシュー #1694 PR レビュー是正）。
R0_R1_GATE_TESTS: dict[str, list[str]] = {
    "probe_run.log": ["te_layout_probe_matches_model"],
    "parity_run.log": [
        "te_square_shapes_all_patterns",
        "te_tall_wide_and_k_tail_shapes_all_patterns",
        "te_ragged_shapes_nn",
    ],
    "all_staged_candidates_run.log": [
        "all_staged_candidates_match_te_cpu_reference_512_nn",
    ],
}
R2_R3_GATE_TESTS: dict[str, list[str]] = {
    "parity_run.log": ["te_rejects_non_staged_candidate"],
    "bit_match_run.log": ["te_bit_match_with_production_dispatch_auto"],
}

# `cargo test ... --nocapture` が出力する `test <name> ... ok`／
# `test <name> ... FAILED` 行（行頭アンカーなし。モジュールパス付きの
# 完全修飾名で出力されるため呼び出し側で短縮名との一致を判定する）。
_TEST_RESULT_LINE_RE = re.compile(
    r"test\s+(?P<name>\S+)\s+\.\.\.\s+(?P<result>ok|FAILED)"
)


@dataclass
class GateEvidence:
    """R0〜R3 ゲートログから読み取った前提充足状況（事前登録判定規則 1）。

    `r0_r1_status`／`r2_r3_status` はいずれも "pass"（全テスト ok 確認済
    み）／"fail"（1 件以上 FAILED 確認）／"unknown"（ログ欠落・該当行なし
    で確認不能）の 3 値。
    """

    r0_r1_status: str
    r2_r3_status: str
    notes: list[str] = field(default_factory=list)


def _test_result_in_text(text: str, short_name: str) -> bool | None:
    """`text` 中の `test <name> ... ok/FAILED` 行から `short_name`
    （完全修飾名の末尾一致、またはモジュールパスなしの完全一致）に該当
    するテストの結果を返す。同名のテストが複数回出力された場合は 1 件
    でも FAILED があれば False（保守的判定）。該当行が 1 件も無ければ
    None（未確認。呼び出し側は "unknown" として扱う）。
    """
    results: list[bool] = []
    for m in _TEST_RESULT_LINE_RE.finditer(text):
        full_name = m.group("name")
        if full_name == short_name or full_name.endswith("::" + short_name):
            results.append(m.group("result") == "ok")
    if not results:
        return None
    return all(results)


def _evaluate_gate_tests(
    gate_dir: str, tests_by_file: dict[str, list[str]]
) -> tuple[str, list[str]]:
    """`tests_by_file`（ファイル名 -> テスト短縮名リスト）の全テストが
    `gate_dir` 配下のログ内で `ok` と確認できるかを判定する。
    戻り値は ("pass" | "fail" | "unknown", notes)。
    """
    notes: list[str] = []
    any_fail = False
    any_unknown = False
    for filename, test_names in tests_by_file.items():
        file_path = os.path.join(gate_dir, filename)
        if not os.path.isfile(file_path):
            notes.append(f"{filename} が見つからない（{gate_dir}）")
            any_unknown = True
            continue
        with open(file_path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        for name in test_names:
            result = _test_result_in_text(text, name)
            if result is None:
                notes.append(f"{filename} 内に {name} の結果行が見つからない")
                any_unknown = True
            elif result is False:
                notes.append(f"{filename} 内の {name} が FAILED")
                any_fail = True
    if any_fail:
        return "fail", notes
    if any_unknown:
        return "unknown", notes
    return "pass", notes


def evaluate_gate_evidence(gate_dir: str | None) -> GateEvidence:
    """`--gate-dir` の内容から R0/R1・R2/R3 の充足状況を判定する（規則
    1）。`gate_dir` が None・非存在の場合は両方 "unknown" とし、
    `render_report` 側で総合判定を undetermined へフォールバックさせる
    （codex-review 指摘: 前提ゲート結果を確認しないまま ADOPT を出す
    ことを防ぐ。イシュー #1694 PR レビュー是正）。
    """
    if gate_dir is None:
        return GateEvidence(
            r0_r1_status="unknown",
            r2_r3_status="unknown",
            notes=["--gate-dir が指定されていない（前提ゲート結果を確認できない）"],
        )
    if not os.path.isdir(gate_dir):
        return GateEvidence(
            r0_r1_status="unknown",
            r2_r3_status="unknown",
            notes=[f"--gate-dir で指定されたディレクトリが見つからない: {gate_dir}"],
        )
    r0_r1_status, r0_r1_notes = _evaluate_gate_tests(gate_dir, R0_R1_GATE_TESTS)
    r2_r3_status, r2_r3_notes = _evaluate_gate_tests(gate_dir, R2_R3_GATE_TESTS)
    return GateEvidence(
        r0_r1_status=r0_r1_status,
        r2_r3_status=r2_r3_status,
        notes=r0_r1_notes + r2_r3_notes,
    )


def overall_verdict(size_verdicts: dict[int, str]) -> str:
    """規則 6: 総合判定。"""
    values = set(size_verdicts.values())
    if "undetermined" in values:
        return "undetermined"
    if values == {"ADOPT-as-opt-in-candidate"}:
        return "ADOPT-as-opt-in-candidate（本番結線は別途ユーザー承認が必要）"
    if "REJECT" in values:
        return "REJECT（1 形状以上で後退。無条件前進は推奨しない）"
    return "undetermined"


def render_report(runs: list[RunData], gate_dir: str | None = None) -> str:
    lines: list[str] = []
    lines.append("# thread_elements() 方式 BlockMMA 候補 A/B 集計（イシュー #1694）")
    lines.append("")

    gate = evaluate_gate_evidence(gate_dir)
    lines.append("## 前提ゲート（R0〜R3）確認状況")
    lines.append("")
    lines.append(f"- R0/R1（正しさ）: **{gate.r0_r1_status}**")
    lines.append(f"- R2/R3（機構契約）: **{gate.r2_r3_status}**")
    if gate.notes:
        for note in gate.notes:
            lines.append(f"  - {note}")
    lines.append("")

    completeness_violations = check_completeness(runs)
    if len(runs) != MIN_FORMAL_RUNS:
        completeness_violations.insert(
            0,
            f"run 数が {len(runs)}（正式判定は {MIN_FORMAL_RUNS} run のみ対象。"
            "揃わない場合は追加起動せず undetermined を記録する）",
        )

    if completeness_violations:
        lines.append("## 完全性検査 FAIL（undetermined 確定）")
        for v in completeness_violations:
            lines.append(f"- {v}")
        lines.append("")
        lines.append("総合判定: **undetermined**")
        return "\n".join(lines) + "\n"

    bit_identical_by_size = check_bit_identical(runs)

    lines.append("## N ごとの結果（5 run 中央値）")
    lines.append("")
    lines.append("| N | run ごとの比 | 中央値 | checksum 完全一致 | 判定 |")
    lines.append("|---|---|---|---|---|")

    size_verdicts: dict[int, str] = {}
    for n in EXPECTED_SIZES:
        ratios = [_single_result(run, n).ratio for run in runs]
        bit_ok = bit_identical_by_size[n]
        if not bit_ok:
            # checksum 不一致は規則 3 により median／符号判定に優先して
            # REJECT 確定とする（表示用ラベルには理由を残す）。
            verdict = "REJECT（checksum 不一致。規則 3）"
            size_verdicts[n] = "REJECT"
        else:
            verdict = judge_size(ratios)
            size_verdicts[n] = verdict
        ratios_str = ", ".join(f"{r:.4f}" for r in ratios)
        lines.append(
            f"| {n} | {ratios_str} | {statistics.median(ratios):.4f} | "
            f"{'yes' if bit_ok else 'NO'} | {verdict} |"
        )
    lines.append("")

    # 規則 1: 前提ゲート（R0〜R3）の充足状況を総合判定へ反映する。
    # R0/R1 が fail/unknown、または R2/R3 が pass 以外（fail/unknown）
    # の場合は、N ごとの表がどう出ようと総合判定を REJECT／undetermined
    # へ強制する（表自体は診断のため常に出力する。codex-review 指摘:
    # 集計が前提ゲート成否を反映せず README 上 undetermined とすべき
    # 場面で ADOPT を出しうる点の是正。イシュー #1694 PR レビュー是正）。
    if gate.r0_r1_status == "fail":
        verdict_str = "REJECT（R0/R1 ゲート不成立につき正しさ不成立。§7.1 規則 1）"
    elif gate.r0_r1_status == "unknown":
        verdict_str = (
            "undetermined（R0/R1 ゲート結果を確認できない。"
            "--gate-dir で前提ゲートログを渡すこと）"
        )
    elif gate.r2_r3_status != "pass":
        verdict_str = (
            "undetermined（R2/R3 が pass と確認できないため、"
            "性能値は参考値扱い。§7.1 規則 1）"
        )
    else:
        verdict_str = overall_verdict(size_verdicts)

    lines.append(f"総合判定: **{verdict_str}**")
    lines.append("")
    lines.append(
        "（本番結線〈`tile::select`／`dispatch_auto` 既定化〉はいずれの判定でも"
        "本イシューの対象外。別イシューでユーザー承認を要する）"
    )
    return "\n".join(lines) + "\n"


def _self_test() -> None:
    def make_log(ratios_by_size: dict[int, float], bit_identical: bool = True) -> str:
        parts = []
        for n, r in ratios_by_size.items():
            parts.append(
                f"N={n} pair=te_vs_production_select production_select_resolved=TileConfig{{}}"
            )
            parts.append(
                f"N={n} pair=te_vs_production_select mode=base kernel_gpu_median_ms=1.0 q1=0.9 q3=1.1"
            )
            parts.append(
                f"N={n} pair=te_vs_production_select mode=head kernel_gpu_median_ms={r:.4f} q1=0.9 q3=1.1"
            )
            parts.append(
                f"N={n} head_over_base_kernel_gpu={r:.6f} base_checksum=1.0e0 "
                f"head_checksum=1.0e0 bit_identical={'true' if bit_identical else 'false'}"
            )
        return "\n".join(parts)

    def write_gate_dir(gate_dir: str, r0_r1_ok: bool = True, r2_r3_ok: bool = True) -> None:
        """R0〜R3 ゲートログの疑似ファイルを `gate_dir` へ書き出す
        （`R0_R1_GATE_TESTS`／`R2_R3_GATE_TESTS` が要求する全テスト名を
        `ok` または `FAILED` として出力する）。
        """
        r0_r1_result = "ok" if r0_r1_ok else "FAILED"
        r2_r3_result = "ok" if r2_r3_ok else "FAILED"
        for filename, names in R0_R1_GATE_TESTS.items():
            with open(os.path.join(gate_dir, filename), "a", encoding="utf-8") as f:
                for name in names:
                    f.write(f"test {name} ... {r0_r1_result}\n")
        for filename, names in R2_R3_GATE_TESTS.items():
            with open(os.path.join(gate_dir, filename), "a", encoding="utf-8") as f:
                for name in names:
                    f.write(f"test {name} ... {r2_r3_result}\n")

    # ケース 1: 全 N が ADOPT（比 <= 1.00・5/5 run 一貫）・ゲート全 pass
    adopt_runs = [
        parse_log(make_log({512: 0.9, 1024: 0.85, 2048: 0.8, 4096: 0.75}))
        for _ in range(5)
    ]
    with tempfile.TemporaryDirectory() as gate_ok:
        write_gate_dir(gate_ok)
        report = render_report(adopt_runs, gate_dir=gate_ok)
    assert "ADOPT-as-opt-in-candidate（本番結線" in report, report
    assert "R0/R1（正しさ）: **pass**" in report, report
    assert "R2/R3（機構契約）: **pass**" in report, report

    # ケース 2: N=512 が REJECT（比 > 1.00・5/5 run 一貫）・他は ADOPT
    mixed_runs = [
        parse_log(make_log({512: 1.2, 1024: 0.85, 2048: 0.8, 4096: 0.75}))
        for _ in range(5)
    ]
    with tempfile.TemporaryDirectory() as gate_ok:
        write_gate_dir(gate_ok)
        report = render_report(mixed_runs, gate_dir=gate_ok)
    assert "REJECT（1 形状以上で後退" in report, report

    # ケース 3: checksum 不一致（bit_identical=false）は REJECT 確定
    checksum_fail_runs = [
        parse_log(
            make_log({512: 0.9, 1024: 0.85, 2048: 0.8, 4096: 0.75}, bit_identical=False)
        )
        for _ in range(5)
    ]
    with tempfile.TemporaryDirectory() as gate_ok:
        write_gate_dir(gate_ok)
        report = render_report(checksum_fail_runs, gate_dir=gate_ok)
    assert "REJECT（checksum 不一致。規則 3）" in report, report
    assert "REJECT（1 形状以上で後退" in report, report

    # ケース 4: 符号が run 間で反転 -> undetermined
    flip_runs = [
        parse_log(make_log({512: 0.9, 1024: 0.85, 2048: 0.8, 4096: 0.75}))
        for _ in range(4)
    ] + [parse_log(make_log({512: 1.1, 1024: 0.85, 2048: 0.8, 4096: 0.75}))]
    with tempfile.TemporaryDirectory() as gate_ok:
        write_gate_dir(gate_ok)
        report = render_report(flip_runs, gate_dir=gate_ok)
    assert "undetermined" in report.splitlines()[-2] or "undetermined" in report, report

    # ケース 5: run 数不足 -> undetermined（完全性検査。ゲート結果に関わらず）
    with tempfile.TemporaryDirectory() as gate_ok:
        write_gate_dir(gate_ok)
        report = render_report(adopt_runs[:3], gate_dir=gate_ok)
    assert "総合判定: **undetermined**" in report, report

    # ケース 6: N の欠落（不完全な run）-> undetermined
    incomplete = parse_log(make_log({512: 0.9, 1024: 0.85, 2048: 0.8}))
    with tempfile.TemporaryDirectory() as gate_ok:
        write_gate_dir(gate_ok)
        report = render_report([incomplete] + adopt_runs[:4], gate_dir=gate_ok)
    assert "総合判定: **undetermined**" in report, report

    # ケース 7（codex-review／Cursor Bugbot 指摘の是正確認）: 同一 N の
    # 結果行が重複出力された run は、後発行が bit_identical=true でも
    # 先行の bit_identical=false を隠さず undetermined（完全性検査 FAIL）
    # として検出する。
    duplicated_text = make_log({512: 0.9, 1024: 0.85, 2048: 0.8}, bit_identical=False) + "\n" + make_log(
        {4096: 0.75}
    )
    # N=512 の行をもう一度（bit_identical=true で）追記し、辞書上書きなら
    # false が隠れてしまう状況を再現する。
    duplicated_text += "\n" + make_log({512: 0.9})
    duplicated_run = parse_log(duplicated_text)
    with tempfile.TemporaryDirectory() as gate_ok:
        write_gate_dir(gate_ok)
        report = render_report([duplicated_run] + adopt_runs[:4], gate_dir=gate_ok)
    assert "総合判定: **undetermined**" in report, report
    assert "複数回出力" in report, report

    # ケース 8（codex-review 指摘の是正確認）: `--gate-dir` 省略時は N
    # ごとの表が ADOPT を示していても総合判定は undetermined へ強制する。
    report_no_gate = render_report(adopt_runs, gate_dir=None)
    assert "総合判定: **undetermined" in report_no_gate, report_no_gate
    assert "R0/R1（正しさ）: **unknown**" in report_no_gate, report_no_gate

    # ケース 9（codex-review 指摘の是正確認）: R0/R1 が FAIL の場合は
    # N ごとの比が ADOPT でも総合判定を REJECT へ強制する。
    with tempfile.TemporaryDirectory() as gate_bad:
        write_gate_dir(gate_bad, r0_r1_ok=False)
        report = render_report(adopt_runs, gate_dir=gate_bad)
    assert "総合判定: **REJECT（R0/R1 ゲート不成立" in report, report

    # ケース 10（codex-review 指摘の是正確認）: R2/R3 が FAIL の場合は
    # N ごとの比が ADOPT でも総合判定を undetermined へ強制する（規則 1:
    # 性能値は参考値扱い）。
    with tempfile.TemporaryDirectory() as gate_r2r3_bad:
        write_gate_dir(gate_r2r3_bad, r2_r3_ok=False)
        report = render_report(adopt_runs, gate_dir=gate_r2r3_bad)
    assert "総合判定: **undetermined（R2/R3" in report, report

    print("self-test: OK", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="*", help="kernel_gpu_te_ab_run<N>.log のパス")
    parser.add_argument(
        "--gate-dir",
        default=None,
        help=(
            "`orchestrate.sh gate` が残した R0〜R3 ログのディレクトリ"
            "（既定は `../metal-gemm-thread-elements-1693`。省略・不在の"
            "場合は前提ゲート未確認として総合判定を undetermined へ"
            "フォールバックする）"
        ),
    )
    parser.add_argument(
        "--self-test", action="store_true", help="埋め込みの疑似ログで判定ロジックを自己検証する"
    )
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return

    if not args.logs:
        parser.error("ログファイルを 1 つ以上指定する（または --self-test）")

    runs: list[RunData] = []
    for path in args.logs:
        with open(path, encoding="utf-8") as f:
            runs.append(parse_log(f.read()))

    print(render_report(runs, gate_dir=args.gate_dir), end="")


if __name__ == "__main__":
    main()
