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

1. 前提ゲート（R0〜R3）は本スクリプトの対象外（`orchestrate.sh gate` が
   別途実行・判定する。R0/R1 FAIL は REJECT 確定・性能 A/B 非実施）。
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
    python3 aggregate.py kernel_gpu_te_ab_run1.log kernel_gpu_te_ab_run2.log \
        kernel_gpu_te_ab_run3.log kernel_gpu_te_ab_run4.log \
        kernel_gpu_te_ab_run5.log > aggregate.md
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
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
    """1 run（1 ログファイル）から抽出した N ごとの結果。"""

    by_size: dict[int, SizeResult] = field(default_factory=dict)


def parse_log(text: str) -> RunData:
    run = RunData()
    for m in _RESULT_RE.finditer(text):
        n = int(m.group("n"))
        run.by_size[n] = SizeResult(
            ratio=float(m.group("ratio")),
            bit_identical=(m.group("bit_identical") == "true"),
            base_checksum=m.group("base_checksum"),
            head_checksum=m.group("head_checksum"),
        )
    return run


def check_completeness(runs: list[RunData]) -> list[str]:
    """各 run がちょうど `EXPECTED_SIZES` の全 N を 1 回ずつ含むかを検査
    する（欠落・重複は run 自体の中断・多重実行を示唆する）。
    """
    violations: list[str] = []
    for i, run in enumerate(runs, start=1):
        missing = [n for n in EXPECTED_SIZES if n not in run.by_size]
        if missing:
            violations.append(f"run{i}: N={missing} の結果行が見つからない")
        extra = [n for n in run.by_size if n not in EXPECTED_SIZES]
        if extra:
            violations.append(f"run{i}: 想定外の N={extra} の結果行がある")
    return violations


def check_bit_identical(runs: list[RunData]) -> dict[int, bool]:
    """N ごとに全 run で `bit_identical=true` かを判定する（規則 3）。"""
    result: dict[int, bool] = {}
    for n in EXPECTED_SIZES:
        result[n] = all(
            run.by_size.get(n) is not None and run.by_size[n].bit_identical
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


def render_report(runs: list[RunData]) -> str:
    lines: list[str] = []
    lines.append("# thread_elements() 方式 BlockMMA 候補 A/B 集計（イシュー #1694）")
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
        ratios = [run.by_size[n].ratio for run in runs]
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
    lines.append(f"総合判定: **{overall_verdict(size_verdicts)}**")
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

    # ケース 1: 全 N が ADOPT（比 <= 1.00・5/5 run 一貫）
    adopt_runs = [
        parse_log(make_log({512: 0.9, 1024: 0.85, 2048: 0.8, 4096: 0.75}))
        for _ in range(5)
    ]
    report = render_report(adopt_runs)
    assert "ADOPT-as-opt-in-candidate（本番結線" in report, report

    # ケース 2: N=512 が REJECT（比 > 1.00・5/5 run 一貫）・他は ADOPT
    mixed_runs = [
        parse_log(make_log({512: 1.2, 1024: 0.85, 2048: 0.8, 4096: 0.75}))
        for _ in range(5)
    ]
    report = render_report(mixed_runs)
    assert "REJECT（1 形状以上で後退" in report, report

    # ケース 3: checksum 不一致（bit_identical=false）は REJECT 確定
    checksum_fail_runs = [
        parse_log(
            make_log({512: 0.9, 1024: 0.85, 2048: 0.8, 4096: 0.75}, bit_identical=False)
        )
        for _ in range(5)
    ]
    report = render_report(checksum_fail_runs)
    assert "REJECT（checksum 不一致。規則 3）" in report, report
    assert "REJECT（1 形状以上で後退" in report, report

    # ケース 4: 符号が run 間で反転 -> undetermined
    flip_runs = [
        parse_log(make_log({512: 0.9, 1024: 0.85, 2048: 0.8, 4096: 0.75}))
        for _ in range(4)
    ] + [parse_log(make_log({512: 1.1, 1024: 0.85, 2048: 0.8, 4096: 0.75}))]
    report = render_report(flip_runs)
    assert "undetermined" in report.splitlines()[-2] or "undetermined" in report, report

    # ケース 5: run 数不足 -> undetermined（完全性検査）
    report = render_report(adopt_runs[:3])
    assert "総合判定: **undetermined**" in report, report

    # ケース 6: N の欠落（不完全な run）-> undetermined
    incomplete = parse_log(make_log({512: 0.9, 1024: 0.85, 2048: 0.8}))
    report = render_report([incomplete] + adopt_runs[:4])
    assert "総合判定: **undetermined**" in report, report

    print("self-test: OK", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="*", help="kernel_gpu_te_ab_run<N>.log のパス")
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

    print(render_report(runs), end="")


if __name__ == "__main__":
    main()
