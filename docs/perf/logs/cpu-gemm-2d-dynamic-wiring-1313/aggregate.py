#!/usr/bin/env python3
"""`TwoDDynamic`（`jobs_per_worker` 別）vs `RowPanel` の 5 回独立プロセス
実行ログ（イシュー #1312）の集計スクリプト。

`gemm_blis_two_d_dynamic_ab_1024_2048`／`gemm_blis_two_d_dynamic_ab_4096`
（`crates/backend-cpu/src/gemm_blis/mod.rs`）が出力する
`variant={variant} jobs_per_worker={jpw} num_threads={t} size={dim}
median_gflops={v}` 行を集計し、(variant, jobs_per_worker, num_threads, size)
ごとの 5 run 中央値・対 `RowPanel` 比・ペアワイズ勝ち run 数を算出する。

`docs/perf/logs/cpu-gemm-b-laneq-vec-ab-1318/aggregate.py`（#1318）の
系譜を踏襲する: ファイル単位でレコードを保持し、同一ファイル（= 同一
プロセス実行）内の値同士のみを対応付けて勝敗を数える（ノイズガードの
契約。#1318 計画 §2 Tier 1 条件 3）。`num_threads` はログファイル名では
なく出力行自体から読み取る（イシュー #1312 計画 §3.1(c)）。
5 サンプル未満・候補欠落・重複ファイルは fail-closed で拒否する。
Python3 標準ライブラリのみ。

使い方: python3 aggregate.py "<機種名>" <ログファイル...>
"""
import os
import re
import statistics
import sys
from collections import defaultdict

LINE_RE = re.compile(
    r"variant=(?P<variant>\S+) jobs_per_worker=(?P<jpw>\d+) "
    r"num_threads=(?P<threads>\d+) size=(?P<size>\d+) "
    r"median_gflops=(?P<gflops>[\d.]+)"
)

# 本 A/B（イシュー #1312）で対象とした固定候補集合（計画 §3.1(c)）:
# RowPanel（jobs_per_worker はログ上 0 固定）・TwoDDynamic(jpw=2)・
# TwoDDynamic(jpw=4)。
EXPECTED_CANDIDATES = (("RowPanel", 0), ("TwoDDynamic", 2), ("TwoDDynamic", 4))
BASELINE_CANDIDATE = ("RowPanel", 0)
EXPECTED_SAMPLES = 5


def parse(path):
    # (variant, jpw, threads, size) -> [gflops, ...]
    out = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = LINE_RE.search(line)
            if m:
                key = (
                    m.group("variant"),
                    int(m.group("jpw")),
                    int(m.group("threads")),
                    int(m.group("size")),
                )
                out[key].append(float(m.group("gflops")))
    return out


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <machine_name> <log files...>", file=sys.stderr)
        sys.exit(2)

    machine = sys.argv[1]
    paths = sys.argv[2:]

    # fail-closed: 同一ファイルの重複渡しを検出する（symlink・相対/絶対
    # パス表記違いを realpath で正規化してから比較。#1315/#1318 と同方針）。
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

    # per_file[path] = {(variant, jpw, threads, size): gflops}
    per_file = {}
    for path in paths:
        d = parse(path)
        record = {}
        for k, v in d.items():
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
            f"ERROR: no `variant=... jobs_per_worker=... num_threads=... "
            f"size=... median_gflops=...` lines found in input files: {paths}",
            file=sys.stderr,
        )
        sys.exit(1)

    # (threads, size) の組ごとに集計する（THREADS スイープを跨いで
    # サンプルを混ぜない）。
    thread_size_pairs = sorted({(t, s) for rec in all_records for (_, _, t, s) in rec})

    # fail-closed: 各ファイルが、そのファイルが言及する (threads, size) に
    # ついて EXPECTED_CANDIDATES を過不足なく含むことを検証する。
    incomplete = []
    for path, rec in per_file.items():
        ts_in_file = sorted({(t, s) for (_, _, t, s) in rec})
        for (t, s) in ts_in_file:
            present = {(variant, jpw) for (variant, jpw, tt, ss) in rec if tt == t and ss == s}
            missing = set(EXPECTED_CANDIDATES) - present
            extra = present - set(EXPECTED_CANDIDATES)
            if missing or extra:
                incomplete.append((path, t, s, sorted(missing), sorted(extra)))

    if incomplete:
        print(f"=== {machine} ===", file=sys.stderr)
        for path, t, s, missing, extra in incomplete:
            print(
                f"ERROR: {path} threads={t} size={s} does not contain exactly "
                f"EXPECTED_CANDIDATES (missing={missing}, extra={extra})",
                file=sys.stderr,
            )
        print(
            "集計を中止しました（ファイル内候補欠落。同一プロセス内比較という"
            "ノイズガードの前提を満たさないファイルが含まれるため中央値・"
            "比較比・勝ち run 数は算出しません）。",
            file=sys.stderr,
        )
        sys.exit(1)

    # 各ファイルはその言及する (threads, size) について EXPECTED_CANDIDATES
    # を過不足なく 1 件ずつ持つことが保証されているため、(threads, size) ごと
    # にそれを含むファイルを入力順に走査すれば同じ添字位置の値は必ず同一
    # ファイル（= 同一プロセス実行）由来になる。
    data = defaultdict(list)
    for (t, s) in thread_size_pairs:
        files_for_ts = [
            rec for rec in per_file.values() if any(tt == t and ss == s for (_, _, tt, ss) in rec)
        ]
        for rec in files_for_ts:
            for (variant, jpw) in EXPECTED_CANDIDATES:
                data[(variant, jpw, t, s)].append(rec[(variant, jpw, t, s)])

    # fail-closed: サンプル数が EXPECTED_SAMPLES と一致することを確認する。
    missing_samples = []
    for (t, s) in thread_size_pairs:
        for (variant, jpw) in EXPECTED_CANDIDATES:
            vals = data.get((variant, jpw, t, s), [])
            if len(vals) != EXPECTED_SAMPLES:
                missing_samples.append((variant, jpw, t, s, len(vals), vals))

    if missing_samples:
        print(f"=== {machine} ===", file=sys.stderr)
        for variant, jpw, t, s, n, vals in missing_samples:
            print(
                f"ERROR: {variant}(jpw={jpw}) threads={t} size={s} has {n} "
                f"samples (expected {EXPECTED_SAMPLES}): {vals}",
                file=sys.stderr,
            )
        print(
            "集計を中止しました（サンプル数不一致。5 回計測中央値の前提を満たさない"
            "ため中央値・比較比は算出しません）。",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"## {machine}\n")
    for (t, s) in thread_size_pairs:
        base_series = data[(BASELINE_CANDIDATE[0], BASELINE_CANDIDATE[1], t, s)]
        base_median = statistics.median(base_series)
        print(f"### threads={t} size={s}\n")
        print("| candidate | jobs_per_worker | 5 run median GFLOP/s | 対 RowPanel 比 | RowPanel に勝った run 数 |")
        print("|---|---|---|---|---|")
        for (variant, jpw) in EXPECTED_CANDIDATES:
            series = data[(variant, jpw, t, s)]
            med = statistics.median(series)
            ratio = med / base_median if base_median > 0 else float("nan")
            if (variant, jpw) == BASELINE_CANDIDATE:
                print(f"| {variant} | {jpw} | {med:.3f} | {ratio:.4f} | — |")
                continue
            wins = sum(1 for a, b in zip(series, base_series) if a > b)
            print(f"| {variant} | {jpw} | {med:.3f} | {ratio:.4f} | {wins}/{len(series)} |")
        print()

    # DGX 非単調性（AC-2）向け: threads 軸を跨いだ比を候補別に出力する
    # （size=1024 限定。#1305 §17.3 の T=10 対 T=8 と同型）。
    sizes_present = sorted({s for (_, s) in thread_size_pairs})
    threads_present = sorted({t for (t, _) in thread_size_pairs})
    if 1024 in sizes_present and {8, 10} <= set(threads_present):
        print("### 非単調性クロスチェック（size=1024。AC-2）\n")
        print("| candidate | jobs_per_worker | T=10 median | T=8 median | T10/T8 比 |")
        print("|---|---|---|---|---|")
        for (variant, jpw) in EXPECTED_CANDIDATES:
            s10 = data.get((variant, jpw, 10, 1024))
            s8 = data.get((variant, jpw, 8, 1024))
            if not s10 or not s8:
                continue
            m10 = statistics.median(s10)
            m8 = statistics.median(s8)
            ratio = m10 / m8 if m8 > 0 else float("nan")
            print(f"| {variant} | {jpw} | {m10:.3f} | {m8:.3f} | {ratio:.4f} |")
        print()


if __name__ == "__main__":
    main()
