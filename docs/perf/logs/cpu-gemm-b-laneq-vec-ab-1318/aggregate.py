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
    # (variant, size) -> [gflops, ...]（1 ファイル = 1 run 分の出力）。
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

    data = defaultdict(list)
    for path in paths:
        d = parse(path)
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
            data[k].extend(v)

    if not data:
        print(
            f"ERROR: no `variant=... size=... median_gflops=...` lines found "
            f"in input files: {paths}",
            file=sys.stderr,
        )
        sys.exit(1)

    sizes = sorted({size for (_, size) in data})

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
            wins = sum(1 for a, b in zip(series, base_series) if a > b)
            print(f"| {variant} | {med:.3f} | {ratio:.4f} | {wins}/{len(series)} |")
        print()


if __name__ == "__main__":
    main()
