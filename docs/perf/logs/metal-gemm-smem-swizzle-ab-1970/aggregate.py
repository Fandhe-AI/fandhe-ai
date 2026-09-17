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
import glob
import re
import statistics
import sys
from pathlib import Path

SIZES = [512, 1024, 2048, 4096]
ARMS = ["L0-P4-S0", "L0-P0-S0", "L0-P4-S1", "L0-P0-S1", "L0-P8-S1", "L0-P4-S2", "L0-P0-S2"]
BASE_ARM = "L0-P4-S0"
CONTRAST_ARM = "L0-P0-S0"

RATIO_RE = re.compile(
    r"^N=(?P<n>\d+) arm=(?P<arm>\S+) head_over_base_kernel_gpu=(?P<ratio>[0-9.eE+-]+)$"
)
BIT_RE = re.compile(
    r"^N=(?P<n>\d+) arm=(?P<arm>\S+) checksum=(?P<checksum>[0-9.eE+-]+)"
    r"(?: bit_identical=(?P<bit_identical>true|false))?$"
)


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


def judge(median_ratio: float, per_run_ratios: list[float]) -> str:
    """規則 4 を 1 セル（N, arm）へ適用する。"""
    if not per_run_ratios:
        return "undetermined（計測未実施）"
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
    run_paths = sorted(glob.glob(str(log_dir / "kernel_gpu_run*.log")))
    if not run_paths:
        print(
            "verdict=undetermined（計測未実施）: kernel_gpu_run*.log が見つかりません。"
            " orchestrate.sh gate -> orchestrate.sh 1..5 の順に実行してください。"
        )
        return 0

    per_run_ratios: dict[tuple[int, str], list[float]] = {(n, a): [] for n in SIZES for a in ARMS}
    per_run_bits: dict[tuple[int, str], list[bool]] = {(n, a): [] for n in SIZES for a in ARMS}

    for p in run_paths:
        ratios, bits = parse_log(Path(p))
        for key, v in ratios.items():
            if key in per_run_ratios:
                per_run_ratios[key].append(v)
        for key, v in bits.items():
            if key in per_run_bits:
                per_run_bits[key].append(v)

    lines: list[str] = []
    lines.append("# XOR swizzle A/B 集計（イシュー #1970）\n")
    lines.append(f"対象 run: {len(run_paths)} 件（{', '.join(Path(p).name for p in run_paths)}）\n")
    lines.append("")
    lines.append("| N | arm | bit_identical（全 run） | run 数 | 5run 比 | 中央値 | 判定 |")
    lines.append("|---|-----|------------------------|--------|---------|--------|------|")

    any_bit_mismatch = False
    for n in SIZES:
        for arm in ARMS:
            if arm == BASE_ARM:
                continue
            key = (n, arm)
            bits = per_run_bits.get(key, [])
            bit_ok = len(bits) > 0 and all(bits)
            if bits and not bit_ok:
                any_bit_mismatch = True
            ratios = per_run_ratios.get(key, [])
            if not bit_ok:
                verdict = "REJECT（bit 不一致）"
            else:
                median = statistics.median(ratios) if ratios else float("nan")
                verdict = judge(median, ratios) if ratios else "undetermined（計測未実施）"
            median_str = f"{statistics.median(ratios):.6f}" if ratios else "-"
            ratios_str = ", ".join(f"{r:.6f}" for r in ratios) if ratios else "-"
            lines.append(
                f"| {n} | {arm} | {bit_ok if bits else 'N/A'} | {len(ratios)} | {ratios_str} | {median_str} | {verdict} |"
            )

    lines.append("")
    lines.append("## XOR の帰属（規則 5。参考区分・判定は緩めない）\n")
    lines.append("| N | arm | 対 base 中央値 | 対 L0-P0-S0 中央値 |")
    lines.append("|---|-----|----------------|---------------------|")
    for n in SIZES:
        for arm in ["L0-P0-S1", "L0-P0-S2"]:
            base_ratios = per_run_ratios.get((n, arm), [])
            contrast_ratios = per_run_ratios.get((n, CONTRAST_ARM), [])
            base_median = f"{statistics.median(base_ratios):.6f}" if base_ratios else "-"
            if base_ratios and contrast_ratios and len(base_ratios) == len(contrast_ratios):
                # 対 L0-P0-S0 比 = (arm/base の中央値) / (L0-P0-S0/base の中央値)
                per_run_vs_contrast = [
                    a / c for a, c in zip(base_ratios, contrast_ratios) if c != 0
                ]
                contrast_median = (
                    f"{statistics.median(per_run_vs_contrast):.6f}" if per_run_vs_contrast else "-"
                )
            else:
                contrast_median = "-"
            lines.append(f"| {n} | {arm} | {base_median} | {contrast_median} |")

    lines.append("")
    lines.append("## 総合（規則 6・7・8）\n")
    lines.append(
        "いずれの判定でも `tile::COOP_LOAD_CONFIG`・`tile::SMEM_SWIZZLE`・"
        "`tile::select*`・`MetalGemm::new` の本番既定は変更しない。"
    )
    if any_bit_mismatch:
        lines.append(
            "\n**bit_identical=false のセルが検出された。正しさ不成立として"
            "REJECT を確定し、以降の性能判定は参考値扱いとする。**"
        )
    if len(run_paths) < 5:
        lines.append(
            f"\n記録: run 数が {len(run_paths)} / 5 のみ（規則 7。5 run 揃わない場合も"
            "追加起動せず undetermined として記録する）。"
        )

    out_path = log_dir / "aggregate.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


def run_self_test() -> int:
    """python3 標準ライブラリのみのユニットテスト（実機ログなしで実行可能）。"""
    # judge() の 3 分岐。
    assert judge(0.9, [0.8, 0.85, 0.9, 0.95, 0.99]) == "ADOPT-as-opt-in-candidate"
    assert judge(1.1, [1.05, 1.1, 1.1, 1.15, 1.2]) == "REJECT"
    assert judge(0.99, [0.9, 1.01, 0.9, 0.9, 0.9]) == "undetermined"
    assert judge(float("nan"), []) == "undetermined（計測未実施）"

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

    print("self-test OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
