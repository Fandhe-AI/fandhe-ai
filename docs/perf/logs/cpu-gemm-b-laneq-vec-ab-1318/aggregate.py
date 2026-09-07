#!/usr/bin/env python3
"""B 側 laneq ベクトル転置版（`GemmDriverVariant::RowPanelBLaneqVec`。イシュー
#1317・#1318）5 回独立プロセス実行ログの集計スクリプト。

`gemm_blis_variant_ab_1024_2048`／`gemm_blis_variant_ab_4096`
（`crates/backend-cpu/src/gemm_blis/mod.rs`）が出力する
`variant={variant} size={dim} median_gflops={v}` 行を集計し、
候補ごと・形状ごとの 5 run 中央値・対 `RowPanel` 比・ペアワイズ勝ち run 数
（同一プロセス内 round-robin 計測。#1318 計画 §2 Tier 1 条件 3「ノイズガード」）
を算出する。

`docs/perf/logs/cpu-gemm-ic-dynamic-ab-1367/aggregate.py`（#1367）の系譜を
踏襲しつつ、EXPECTED_VARIANTS を本 issue の 5 候補
（`RowPanel`・`SharedB`・`SharedBPcOuter`・`IcDynamic`・`RowPanelBLaneqVec`。
`all_gemm_driver_variants()` aarch64 版〈#1317 で追加〉と一致）へ拡張し、
5 サンプル未満・候補欠落での集計を fail-closed で拒否する
（実測値の捏造・不完全データでの判定確定を防ぐ）。Python3 標準ライブラリのみ。

ペアワイズ勝ち run 数の算出は「同一プロセス内比較」というノイズガードの契約
（#1318 計画 §2 Tier 1 条件 3）を厳密に満たすため、値を (variant, size) 単位で
フラットに集約せず、**ファイル（= 1 プロセス実行）単位でレコードを保持し、
同一ファイル内の値同士のみを対応付けて**勝敗を数える（同一サイズについて
異なるファイルが異なる候補集合を欠落させていても、値の対応がファイルを跨いで
ズレたまま「5 件ずつ」の見かけ上のサンプル数一致検査を通過し、実際には
異なるプロセスの値同士を比較してしまう事故を防ぐ。codex-review 指摘・
イシュー #1318 PR #1433）。

使い方: python3 aggregate.py "<機種名>" <ログファイル...>
"""
import os
import re
import statistics
import sys
from collections import defaultdict

LINE_RE = re.compile(
    r"variant=(?P<variant>\S+) size=(?P<size>\d+) median_gflops=(?P<gflops>[\d.]+)"
)

# 本 A/B（イシュー #1318）で対象とした固定候補集合（aarch64 版
# `all_gemm_driver_variants()` と一致。#1317 で `RowPanelBLaneqVec` を追加）。
EXPECTED_VARIANTS = ("RowPanel", "SharedB", "SharedBPcOuter", "IcDynamic", "RowPanelBLaneqVec")
BASELINE_VARIANT = "RowPanel"
EXPECTED_SAMPLES = 5


def parse(path):
    # (variant, size) -> [gflops, ...]（1 ファイル = 1 run 分の出力。
    # 正常なファイルは各 (variant, size) につき厳密に 1 件のみ持つ）。
    out = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = LINE_RE.search(line)
            if m:
                variant = m.group("variant")
                size = int(m.group("size"))
                gflops = float(m.group("gflops"))
                out[(variant, size)].append(gflops)
    return out


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <machine_name> <log files...>", file=sys.stderr)
        sys.exit(2)

    machine = sys.argv[1]
    paths = sys.argv[2:]

    # fail-closed: 同一ファイル（symlink・相対/絶対パス表記違いを
    # `os.path.realpath` で正規化してから比較）が複数回渡されていないか
    # 検証する（同じ run のログを水増しして「5 run median」として受理される
    # 事故を防ぐ。#1315 の check_distinct_paths と同方針）。
    seen = {}
    duplicates = []
    for path in paths:
        real = os.path.realpath(path)
        if real in seen:
            duplicates.append((path, seen[real]))
        else:
            seen[real] = path
    if duplicates:
        print(
            f"ERROR: duplicate input file path(s) detected (same run passed "
            f"more than once): {duplicates}",
            file=sys.stderr,
        )
        sys.exit(1)

    # per_file[path] = {(variant, size): gflops}（1 ファイル = 1 プロセス実行分）。
    # 後段の勝敗判定を「同一ファイル（= 同一プロセス）内の値同士」に限定する
    # ため、(variant, size) へフラットに集約する前にファイル単位で保持する。
    per_file = {}
    for path in paths:
        d = parse(path)
        record = {}
        for k, v in d.items():
            # fail-closed: 同一ファイル内で同じ (variant, size) が複数回
            # 出力された場合（プロセス側の異常出力等）を検出する。
            if len(v) != 1:
                print(
                    f"ERROR: {path} contains {len(v)} lines for "
                    f"{k} (expected exactly 1 per run file)",
                    file=sys.stderr,
                )
                sys.exit(1)
            record[k] = v[0]
        per_file[path] = record

    all_records = [rec for rec in per_file.values() if rec]
    if not all_records:
        print(
            f"ERROR: no `variant=... size=... median_gflops=...` lines found "
            f"in input files: {paths}",
            file=sys.stderr,
        )
        sys.exit(1)

    sizes = sorted({size for rec in all_records for (_, size) in rec})

    # fail-closed: 各ファイルが、そのファイルが言及する size について
    # EXPECTED_VARIANTS を過不足なく含むことを検証する（同一プロセス内
    # 比較の前提。ファイルによって欠落する候補が異なると、後段の
    # (variant, size) 単位フラット集約でサンプル数だけは一致しつつ実際には
    # 異なるプロセスの値同士が対応してしまうため、ファイル単位で先に
    # 完全性を確認する）。
    incomplete = []
    for path, rec in per_file.items():
        sizes_in_file = sorted({size for (_, size) in rec})
        for size in sizes_in_file:
            present = {variant for (variant, s) in rec if s == size}
            missing_variants = set(EXPECTED_VARIANTS) - present
            extra_variants = present - set(EXPECTED_VARIANTS)
            if missing_variants or extra_variants:
                incomplete.append((path, size, sorted(missing_variants), sorted(extra_variants)))

    if incomplete:
        print(f"=== {machine} ===", file=sys.stderr)
        for path, size, missing_variants, extra_variants in incomplete:
            print(
                f"ERROR: {path} size={size} does not contain exactly "
                f"EXPECTED_VARIANTS (missing={missing_variants}, "
                f"extra={extra_variants})",
                file=sys.stderr,
            )
        print(
            "集計を中止しました（ファイル内候補欠落。同一プロセス内比較という"
            "ノイズガードの前提を満たさないファイルが含まれるため中央値・"
            "比較比・勝ち run 数は算出しません）。",
            file=sys.stderr,
        )
        sys.exit(1)

    # ここまでの検証により、各ファイルは自身が言及する各 size について
    # EXPECTED_VARIANTS を過不足なく 1 件ずつ持つことが保証されている。
    # よって size ごとに「その size を含むファイル」を入力順に走査すれば、
    # 同じ添字位置の値は必ず同一ファイル（= 同一プロセス実行）由来になり、
    # ペアワイズ比較の対応関係が保たれる。
    data = defaultdict(list)
    for size in sizes:
        files_for_size = [rec for rec in per_file.values() if any(s == size for (_, s) in rec)]
        for rec in files_for_size:
            for variant in EXPECTED_VARIANTS:
                data[(variant, size)].append(rec[(variant, size)])

    # fail-closed: 各 (variant, size) のサンプル数が EXPECTED_SAMPLES と
    # 一致することを確認する（5 回独立プロセス計測という契約。#1367 と同方針）。
    missing = []
    for size in sizes:
        for variant in EXPECTED_VARIANTS:
            vals = data.get((variant, size), [])
            if len(vals) != EXPECTED_SAMPLES:
                missing.append((variant, size, len(vals), vals))

    if missing:
        print(f"=== {machine} ===", file=sys.stderr)
        for variant, size, n, vals in missing:
            print(
                f"ERROR: {variant} size={size} has {n} samples (expected "
                f"{EXPECTED_SAMPLES}): {vals}",
                file=sys.stderr,
            )
        print(
            "集計を中止しました（サンプル数不一致。5 回計測中央値の前提を満たさない"
            "ため中央値・比較比は算出しません）。",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"## {machine}\n")
    for size in sizes:
        base_key = (BASELINE_VARIANT, size)
        base_series = data[base_key]
        base_median = statistics.median(base_series)
        print(f"### size={size}\n")
        print("| variant | 5 run median GFLOP/s | 対 RowPanel 比 | RowPanel に勝った run 数 |")
        print("|---|---|---|---|")
        for variant in EXPECTED_VARIANTS:
            series = data[(variant, size)]
            med = statistics.median(series)
            ratio = med / base_median if base_median > 0 else float("nan")
            if variant == BASELINE_VARIANT:
                print(f"| {variant} | {med:.3f} | {ratio:.4f} | — |")
                continue
            # `series[i]` と `base_series[i]` は上記のファイル単位検証により
            # 同一ファイル（= 同一プロセス実行）内の値であることが保証されて
            # いるため、この zip はノイズガード契約どおり同一プロセス内比較
            # になる。
            wins = sum(1 for a, b in zip(series, base_series) if a > b)
            print(f"| {variant} | {med:.3f} | {ratio:.4f} | {wins}/{len(series)} |")
        print()


if __name__ == "__main__":
    main()
