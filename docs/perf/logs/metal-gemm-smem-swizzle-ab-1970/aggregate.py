#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""協調ロードの threadgroup メモリ格納位置 XOR swizzle 軸（イシュー
#1970）の性能 A/B（`orchestrate.sh <run番号>` が生成する
`kernel_gpu_run<N>.log`。5 run 分）を集計し、issue コメントで固定した
事前登録判定規則（§6）を機械適用して `aggregate.md` を生成する。

python3 標準ライブラリのみを使う（`docs/perf/logs/
metal-gemm-thread-elements-ab-1694/aggregate.py` と同じ方針・
`--self-test` を持つ他の aggregate.py と同型の設計）。

# 判定規則（issue #1970 コメントで固定・事後緩和しない）

1. 前提ゲート: `orchestrate.sh gate`（bit 一致 12 テスト）が全 pass で
   あること（本スクリプトは gate ログを直接検査しないが、README の
   実行手順が gate を run1〜5 より前に要求する）。
2. bit 同一（run 内）: 各 run・各 N・各 arm の trial 0 出力（`checksum`／
   `bit_identical` 行）が base と一致すること。1 セルでも
   `bit_identical=false` なら当該 arm は比の値によらず全 N で REJECT。
3. 指標: 同一 run 内の `head_over_base_kernel_gpu`（arm 中央値 / base
   中央値）。N・arm ごとに 5 run の中央値を採用する。
4. N 別判定（arm ごと）: 5 run 中央値 `<= 1.00` かつ 5/5 run が `< 1.00`
   で符号一貫 -> `ADOPT-as-opt-in-candidate`／中央値 `> 1.00` かつ 5/5
   run が `> 1.00` -> `REJECT`／それ以外（符号反転）-> `undetermined`。
5. XOR の帰属（参考区分・判定は緩めない）: `L0-P0-S1`／`L0-P0-S2` は
   base 比に加えて対照 `L0-P0-S0` 比も併記する。
6. 総合: いずれの判定でも本番既定（`tile::COOP_LOAD_CONFIG`・
   `tile::SMEM_SWIZZLE`・`tile::select*`・`MetalGemm::new`）は変更しない。
7. 負荷: record_only（専有ゲートなし）。5 run 揃わない場合は追加起動せず、
   揃った run 数と undetermined を記録する。
8. 未実測時: `verdict=undetermined（計測未実施）` で出荷する。
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
from pathlib import Path

SIZES = [512, 1024, 2048, 4096]
ARMS = ["L0-P4-S0", "L0-P0-S0", "L0-P4-S1", "L0-P0-S1", "L0-P8-S1", "L0-P4-S2", "L0-P0-S2"]
BASE_ARM = "L0-P4-S0"
CONTRAST_ARM = "L0-P0-S0"
REQUIRED_RUNS = 5

# ファイル名から run 番号（1〜5）を厳密に抽出する。`glob` による広いパターン
# 一致（例: 余分な退避ログ `kernel_gpu_run1.log.bak` や run 番号が範囲外の
# ファイル）を排除するため、`kernel_gpu_run<N>.log` の完全一致のみを受理する
# （規則 1 の是正: 5 run が揃ったことを run 番号で機械確認するため）。
RUN_FILE_RE = re.compile(r"^kernel_gpu_run(?P<run>[1-5])\.log$")

RATIO_RE = re.compile(
    r"^N=(?P<n>\d+) arm=(?P<arm>\S+) head_over_base_kernel_gpu=(?P<ratio>[0-9.eE+-]+)$"
)
BIT_RE = re.compile(
    r"^N=(?P<n>\d+) arm=(?P<arm>\S+) checksum=(?P<checksum>[0-9.eE+-]+)"
    r"(?: bit_identical=(?P<bit_identical>true|false))?$"
)


def discover_runs(log_dir: Path) -> dict[int, Path]:
    """`log_dir` から run 番号（1〜5）-> ファイルパスの対応を作る。

    `kernel_gpu_run<N>.log`（N は 1〜5 の 1 桁）に厳密一致するファイルのみを
    対象とし、退避ログ（`.bak` 等）や範囲外の run 番号を機械的に除外する
    （規則 1 の是正: 「揃った run 数」を run 番号の集合で判定するため、
    重複や余分なファイルを事前に排除する）。
    """
    runs: dict[int, Path] = {}
    for p in sorted(log_dir.iterdir()):
        if not p.is_file():
            continue
        m = RUN_FILE_RE.match(p.name)
        if m:
            runs[int(m.group("run"))] = p
    return runs


def parse_log(path: Path) -> tuple[dict[tuple[int, str], float], dict[tuple[int, str], bool]]:
    """1 run 分のログから (N, arm) -> ratio・(N, arm) -> bit_identical を抽出する。

    base arm（`L0-P4-S0`）自身の trial 0 行には `bit_identical` が付かない
    （比較元のため常に一致とみなす）。パース対象外の行は無視する（fail-open
    ではなく単に無関係な出力行のため。ratio・bit_identical の欠落自体は
    呼び出し元が件数検査で fail-closed に扱う）。
    """
    ratios: dict[tuple[int, str], float] = {}
    bits: dict[tuple[int, str], bool] = {}
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        m = RATIO_RE.match(line)
        if m:
            key = (int(m.group("n")), m.group("arm"))
            ratios[key] = float(m.group("ratio"))
            continue
        m = BIT_RE.match(line)
        if m:
            key = (int(m.group("n")), m.group("arm"))
            bi = m.group("bit_identical")
            if bi is not None:
                bits[key] = bi == "true"
            elif m.group("arm") == BASE_ARM:
                bits[key] = True
    return ratios, bits


def judge(n_runs_available: int, median_ratio: float, per_run_ratios: list[float]) -> str:
    """規則 1・4・7 を 1 セル（N, arm）へ適用する。

    5 run（`REQUIRED_RUNS`）が揃い、かつそのセルの ratio が 5 件すべて
    出揃っていなければ確定判定（ADOPT-as-opt-in-candidate／REJECT）を出さず
    undetermined とする（規則 1 の是正: 1 run のみの結果で ADOPT を出さない）。
    """
    if n_runs_available < REQUIRED_RUNS:
        return f"undetermined（run 数不足: {n_runs_available}/{REQUIRED_RUNS}）"
    if len(per_run_ratios) < REQUIRED_RUNS:
        return f"undetermined（当該セルの計測が {len(per_run_ratios)}/{REQUIRED_RUNS} 件のみ）"
    if median_ratio <= 1.00 and all(r < 1.00 for r in per_run_ratios):
        return "ADOPT-as-opt-in-candidate"
    if median_ratio > 1.00 and all(r > 1.00 for r in per_run_ratios):
        return "REJECT"
    return "undetermined"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-dir",
        default=str(Path(__file__).resolve().parent),
        help="kernel_gpu_run<N>.log を探すディレクトリ（既定: 本スクリプトと同じディレクトリ）",
    )
    parser.add_argument("--self-test", action="store_true", help="ユニットテストのみ実行して終了する")
    args = parser.parse_args()

    if args.self_test:
        return run_self_test()

    log_dir = Path(args.log_dir)
    runs = discover_runs(log_dir)
    if not runs:
        print(
            "verdict=undetermined（計測未実施）: kernel_gpu_run<N>.log（N=1〜5）が"
            " 見つかりません。 orchestrate.sh gate -> orchestrate.sh 1..5 の順に"
            "実行してください。"
        )
        return 0

    n_runs_available = len(runs)
    run_numbers = sorted(runs.keys())

    # run 番号でインデックスした ratio・bit を保持する（規則 3・5 の同一 run
    # 内比較を run 識別子で機械的に保証するため。欠落は None ではなく
    # キー自体を作らないことで表現し、後段の集合演算で扱う）。
    per_run_ratios: dict[tuple[int, str], dict[int, float]] = {
        (n, a): {} for n in SIZES for a in ARMS
    }
    per_run_bits: dict[tuple[int, str], dict[int, bool]] = {
        (n, a): {} for n in SIZES for a in ARMS
    }

    for run_no in run_numbers:
        ratios, bits = parse_log(runs[run_no])
        for key, v in ratios.items():
            if key in per_run_ratios:
                per_run_ratios[key][run_no] = v
        for key, v in bits.items():
            if key in per_run_bits:
                per_run_bits[key][run_no] = v

    # 規則 3: bit 不一致は同じ arm の全サイズへ反映する。まず arm ごとに
    # 「いずれかの (N, run) で bit_identical=false が明示されたか」を先に
    # 収集し、判定本体（次のループ）より前に確定させる。証跡ゼロ（値が
    # 存在しないセル）は不一致として扱わない（規則 2 の是正）。
    arm_bit_mismatch: dict[str, bool] = {a: False for a in ARMS}
    for n in SIZES:
        for arm in ARMS:
            if arm == BASE_ARM:
                continue
            bits_by_run = per_run_bits.get((n, arm), {})
            if bits_by_run and not all(bits_by_run.values()):
                arm_bit_mismatch[arm] = True

    lines: list[str] = []
    lines.append("# XOR swizzle A/B 集計（イシュー #1970）\n")
    lines.append(
        f"対象 run: {n_runs_available} 件（run 番号 "
        f"{', '.join(str(r) for r in run_numbers)}。`kernel_gpu_run<N>.log` "
        "完全一致のみを走査対象とし、退避ログ・run 番号範囲外のファイルは除外済み）\n"
    )
    lines.append("")
    lines.append("| N | arm | bit_identical（全 run） | run 数 | 5run 比 | 中央値 | 判定 |")
    lines.append("|---|-----|------------------------|--------|---------|--------|------|")

    any_bit_mismatch = any(arm_bit_mismatch.values())
    for n in SIZES:
        for arm in ARMS:
            if arm == BASE_ARM:
                continue
            key = (n, arm)
            bits_by_run = per_run_bits.get(key, {})
            ratios_by_run = per_run_ratios.get(key, {})
            ratios = [ratios_by_run[r] for r in sorted(ratios_by_run)]

            # 規則 2: run ごとに ratio と bit 証跡を対応付ける。証跡が 1 件も
            # 無いセルは「不一致」ではなく「証跡なし」として区別する。
            if not bits_by_run:
                bit_display = "N/A"
            elif all(bits_by_run.values()):
                bit_display = f"true（{len(bits_by_run)}/{n_runs_available}）"
            else:
                bit_display = f"false 含む（{len(bits_by_run)}/{n_runs_available}）"

            if arm_bit_mismatch[arm]:
                # 規則 3: 当該 arm のいずれかの N で bit 不一致が確定したため、
                # このセル自身の bit_identical 値に関わらず全 N を REJECT とする。
                verdict = "REJECT（bit 不一致・同一 arm の他 N で検出）"
            else:
                median = statistics.median(ratios) if ratios else float("nan")
                verdict = judge(n_runs_available, median, ratios)

            median_str = f"{statistics.median(ratios):.6f}" if ratios else "-"
            ratios_str = ", ".join(f"{r:.6f}" for r in ratios) if ratios else "-"
            lines.append(
                f"| {n} | {arm} | {bit_display} | {len(ratios)} | {ratios_str} | {median_str} | {verdict} |"
            )

    lines.append("")
    lines.append("## XOR の帰属（規則 5。参考区分・判定は緩めない）\n")
    lines.append("| N | arm | 対 base 中央値 | 対 L0-P0-S0 中央値 |")
    lines.append("|---|-----|----------------|---------------------|")
    for n in SIZES:
        for arm in ["L0-P0-S1", "L0-P0-S2"]:
            base_ratios_by_run = per_run_ratios.get((n, arm), {})
            contrast_ratios_by_run = per_run_ratios.get((n, CONTRAST_ARM), {})
            base_ratios = [base_ratios_by_run[r] for r in sorted(base_ratios_by_run)]
            base_median = f"{statistics.median(base_ratios):.6f}" if base_ratios else "-"
            # 規則 3/5: 同一 run の head／対照の測定同士だけを run 番号で
            # 対応付けて比を取る（run 識別子を保持せず配列を zip すると、
            # 欠落箇所が異なる場合に別 run 同士を割ってしまうため）。
            common_runs = sorted(set(base_ratios_by_run) & set(contrast_ratios_by_run))
            per_run_vs_contrast = [
                base_ratios_by_run[r] / contrast_ratios_by_run[r]
                for r in common_runs
                if contrast_ratios_by_run[r] != 0
            ]
            contrast_median = (
                f"{statistics.median(per_run_vs_contrast):.6f}" if per_run_vs_contrast else "-"
            )
            lines.append(f"| {n} | {arm} | {base_median} | {contrast_median} |")

    lines.append("")
    lines.append("## 総合（規則 6・7・8）\n")
    lines.append(
        "いずれの判定でも `tile::COOP_LOAD_CONFIG`・`tile::SMEM_SWIZZLE`・"
        "`tile::select*`・`MetalGemm::new` の本番既定は変更しない。"
    )
    if any_bit_mismatch:
        mismatched_arms = ", ".join(a for a in ARMS if arm_bit_mismatch.get(a))
        lines.append(
            f"\n**bit_identical=false が検出された arm（{mismatched_arms}）は"
            "正しさ不成立として全 N を REJECT に確定し、以降の性能判定は"
            "参考値扱いとする。**"
        )
    if n_runs_available < REQUIRED_RUNS:
        lines.append(
            f"\n記録: run 数が {n_runs_available} / {REQUIRED_RUNS} のみ（規則 7。"
            f"{REQUIRED_RUNS} run 揃わない場合も追加起動せず undetermined として"
            "記録する）。"
        )

    out_path = log_dir / "aggregate.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


def run_self_test() -> int:
    """python3 標準ライブラリのみのユニットテスト（実機ログなしで実行可能）。"""
    import shutil
    import tempfile

    # judge() の分岐（5 run 完全性チェック込み）。
    assert judge(5, 0.9, [0.8, 0.85, 0.9, 0.95, 0.99]) == "ADOPT-as-opt-in-candidate"
    assert judge(5, 1.1, [1.05, 1.1, 1.1, 1.15, 1.2]) == "REJECT"
    assert judge(5, 0.99, [0.9, 1.01, 0.9, 0.9, 0.9]) == "undetermined"
    assert judge(5, float("nan"), []) == "undetermined（当該セルの計測が 0/5 件のみ）"
    # 規則 1 の是正: 1 run のみでは（当該セルが全 run で改善方向でも）ADOPT を
    # 出さず undetermined とする。
    assert judge(1, 0.8, [0.8]).startswith("undetermined（run 数不足")
    # 5 run 揃っていても当該セル自身の計測が 5 件未満なら undetermined とする。
    assert judge(5, 0.8, [0.8, 0.8, 0.8]).startswith("undetermined（当該セルの計測が")

    # parse_log の行パーサ。
    sample = (
        "N=1024 arm=L0-P4-S0 checksum=1.234560\n"
        "N=1024 arm=L0-P0-S1 checksum=1.234560 bit_identical=true\n"
        "N=1024 arm=L0-P0-S2 checksum=9.999990 bit_identical=false\n"
        "N=1024 arm=L0-P4-S0 smem_swizzle=Off coop_load=... resolved_tile=... "
        "kernel_gpu_median_ms=1.0 q1=0.9 q3=1.1\n"
        "N=1024 arm=L0-P0-S1 head_over_base_kernel_gpu=0.987654\n"
        "N=2048 arm=L0-P8-S1 head_over_base_kernel_gpu=1.234567\n"
    )
    tmp = Path("/tmp") / "smem_swizzle_aggregate_self_test.log"
    tmp.write_text(sample, encoding="utf-8")
    try:
        ratios, bits = parse_log(tmp)
        assert ratios[(1024, "L0-P0-S1")] == 0.987654
        assert ratios[(2048, "L0-P8-S1")] == 1.234567
        assert bits[(1024, "L0-P4-S0")] is True
        assert bits[(1024, "L0-P0-S1")] is True
        assert bits[(1024, "L0-P0-S2")] is False
    finally:
        tmp.unlink(missing_ok=True)

    # discover_runs: `kernel_gpu_run<N>.log`（N=1〜5）の完全一致のみを拾い、
    # 退避ログ・run 番号範囲外のファイルを除外する（規則 1 の是正）。
    work = Path(tempfile.mkdtemp(prefix="smem_swizzle_discover_"))
    try:
        (work / "kernel_gpu_run1.log").write_text("", encoding="utf-8")
        (work / "kernel_gpu_run3.log").write_text("", encoding="utf-8")
        (work / "kernel_gpu_run1.log.bak").write_text("stale", encoding="utf-8")
        (work / "kernel_gpu_run9.log").write_text("", encoding="utf-8")
        (work / "kernel_gpu_run1_old.log").write_text("", encoding="utf-8")
        runs = discover_runs(work)
        assert set(runs.keys()) == {1, 3}
        assert runs[1] == work / "kernel_gpu_run1.log"
    finally:
        shutil.rmtree(work)

    # main() の結合テスト: (1) 1 run のみでは ADOPT を出さない、(2) ある arm の
    # 1 セルが bit_identical=false なら同じ arm の全 N を REJECT にする、
    # (3) 対照比は同一 run の測定同士でのみ計算し欠落 run を跨いで割らない
    # ことを検証する。
    work2 = Path(tempfile.mkdtemp(prefix="smem_swizzle_main_it_"))
    try:
        # run1〜5: L0-P0-S1 は全 run で改善方向（5/5 run < 1.00・bit 一致）。
        # ただし N=2048 でのみ run3 に bit_identical=false を混入させ、
        # 同一 arm の N=512/1024/4096 も REJECT へ波及することを確認する。
        for run_no in range(1, 6):
            body_lines = []
            for n in SIZES:
                bit_val = "true"
                if n == 2048 and run_no == 3:
                    bit_val = "false"
                body_lines.append(
                    f"N={n} arm=L0-P0-S1 checksum=1.0 bit_identical={bit_val}"
                )
                body_lines.append(f"N={n} arm=L0-P0-S1 head_over_base_kernel_gpu=0.9")
                # 対照 arm（L0-P0-S0）は run4 のみ欠落させ、run 識別子を
                # 保持した対応付けでしか正しく計算できない状況を作る。
                if not (n == 1024 and run_no == 4):
                    body_lines.append(
                        f"N={n} arm=L0-P0-S0 head_over_base_kernel_gpu=0.95"
                    )
            (work2 / f"kernel_gpu_run{run_no}.log").write_text(
                "\n".join(body_lines) + "\n", encoding="utf-8"
            )
        sys.argv = ["aggregate.py", "--log-dir", str(work2)]
        rc = main()
        assert rc == 0
        md = (work2 / "aggregate.md").read_text(encoding="utf-8")
        # (2) N=2048 の run3 bit 不一致が同じ arm の全 N（512/1024/4096 含む）
        # を REJECT にしていること。
        for n in SIZES:
            row = next(line for line in md.splitlines() if line.startswith(f"| {n} | L0-P0-S1 |"))
            assert "REJECT（bit 不一致" in row, row
        # (3) 対照比: run4 は L0-P0-S0 が欠落しているため run 識別子で
        # 対応付けた 4 run 分のみで中央値を計算する（run 識別子を保持せず
        # 配列を zip すると欠落箇所の後ろの run 同士がずれて誤った値になる）。
        # 欠落を除いた 4 件の比はすべて 0.9/0.95 で一致するため中央値も同値。
        xor_section = md.split("## XOR の帰属", 1)[1]
        contrast_row = next(
            line for line in xor_section.splitlines() if line.startswith("| 1024 | L0-P0-S1 |")
        )
        expected_contrast = f"{0.9 / 0.95:.6f}"
        assert expected_contrast in contrast_row, (contrast_row, expected_contrast)
    finally:
        shutil.rmtree(work2)

    # (1) 1 run のみの単純ケースで ADOPT が出ないことも別途確認する。
    work3 = Path(tempfile.mkdtemp(prefix="smem_swizzle_main_onerun_"))
    try:
        (work3 / "kernel_gpu_run1.log").write_text(
            "N=512 arm=L0-P0-S1 checksum=1.0 bit_identical=true\n"
            "N=512 arm=L0-P0-S1 head_over_base_kernel_gpu=0.5\n",
            encoding="utf-8",
        )
        sys.argv = ["aggregate.py", "--log-dir", str(work3)]
        rc = main()
        assert rc == 0
        md = (work3 / "aggregate.md").read_text(encoding="utf-8")
        row = next(line for line in md.splitlines() if line.startswith("| 512 | L0-P0-S1 |"))
        assert "ADOPT" not in row, row
        assert "undetermined" in row, row
    finally:
        shutil.rmtree(work3)

    print("self-test OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
