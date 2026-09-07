#!/usr/bin/env python3
"""KC 再スイープ（イシュー #1315）5 回独立プロセス実行ログの集計スクリプト。

`gemm_blis_kc_sweep_ab_1024_2048`／`gemm_blis_kc_sweep_ab_4096`
（`crates/backend-cpu/src/gemm_blis/mod.rs`）が出力する
`variant=RowPanel kc={kc} size={dim} median_gflops={v}` 行を集計し、
KC ごと・形状ごとの 5 run 中央値・対 KC=256 比を算出する。

#1367（`cpu-gemm-ic-dynamic-variant.md` 引き継ぎ）の aggregate.py 系譜を
踏襲し、5 サンプル未満での集計を fail-closed で拒否する（実測値の
捏造・不完全データでの判定確定を防ぐ）。Python3 標準ライブラリのみ。

さらに #1430 codex-review 指摘（PR #1430 スレッド
https://github.com/Fandhe-AI/fandhe-ai/pull/1430#discussion_r3952477456）を
受け、以下を集計前に検証する（空ログ・候補/形状の丸ごと欠落を
`exit 0` で通さない）:
  - 入力行が 1 件も見つからない場合は拒否する（空ログ・/dev/null 対策）。
  - 期待する KC 候補（EXPECTED_KCS。本スイープの固定候補集合）が
    検出された各 size で全て揃っているか検証する。
  - 基準 KC（BASELINE_KC=256）が検出された全 size に存在するか検証する
    （欠落時に黙ってその size をスキップしない）。

使い方: python3 aggregate.py "<機種名>" <ログファイル...>
"""
import re
import statistics
import sys
from collections import defaultdict

LINE_RE = re.compile(
    r"variant=(?P<variant>\S+) kc=(?P<kc>\d+) size=(?P<size>\d+) "
    r"median_gflops=(?P<gflops>[\d.]+)"
)

# 本スイープ（イシュー #1315）で計測対象とした固定 KC 候補集合。
# `docs/perf/logs/cpu-gemm-kc-sweep-1315/kc-*-run*.txt` は全てこの 5 候補を
# 含む前提（`aggregate.md` の実測表と対応）。
EXPECTED_KCS = (128, 192, 256, 384, 512)
BASELINE_KC = 256


def parse_files(paths):
    # (kc, size) -> [gflops, ...]（5 run 分。プロセスごとに 1 出力行）
    samples = defaultdict(list)
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                m = LINE_RE.search(line)
                if m:
                    kc = int(m.group("kc"))
                    size = int(m.group("size"))
                    gflops = float(m.group("gflops"))
                    samples[(kc, size)].append(gflops)
    return samples


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <machine_name> <log files...>", file=sys.stderr)
        sys.exit(2)

    machine = sys.argv[1]
    paths = sys.argv[2:]
    samples = parse_files(paths)

    # fail-closed: 入力行が 1 件もなければ拒否する（空ログ・/dev/null 等の
    # 与えられたログファイル集合が丸ごと欠落した入力を、見出しのみ出力し
    # exit 0 で通さない）。
    if not samples:
        print(
            "ERROR: no `variant=... kc=... size=... median_gflops=...` "
            f"lines found in input files: {paths}",
            file=sys.stderr,
        )
        sys.exit(1)

    # fail-closed: 5 サンプル未満の (kc, size) があれば拒否する
    incomplete = {k: len(v) for k, v in samples.items() if len(v) < 5}
    if incomplete:
        print(f"ERROR: sample count < 5 for: {incomplete}", file=sys.stderr)
        sys.exit(1)

    sizes = sorted({size for (_, size) in samples})
    kcs = sorted({kc for (kc, _) in samples})

    # fail-closed: 検出された各 size で期待 KC 候補（EXPECTED_KCS）が
    # 全て揃っているか検証する。候補が丸ごと欠落した入力（一部ログファイル
    # の取り違え・欠落）を黙ってスキップしない。
    missing_candidates = {
        size: sorted(set(EXPECTED_KCS) - {kc for (kc, s) in samples if s == size})
        for size in sizes
    }
    missing_candidates = {k: v for k, v in missing_candidates.items() if v}
    if missing_candidates:
        print(
            f"ERROR: missing expected KC candidates {EXPECTED_KCS} for sizes: "
            f"{missing_candidates}",
            file=sys.stderr,
        )
        sys.exit(1)

    # fail-closed: 基準 KC（BASELINE_KC=256）が検出された全 size に
    # 存在するか検証する（欠落時に黙ってその size を集計から除外しない。
    # 上の missing_candidates 検査で通常は捕捉されるが、EXPECTED_KCS 外の
    # KC のみで構成された入力等の想定外ケースにも独立に fail-closed で
    # 備える）。
    missing_baseline = [size for size in sizes if (BASELINE_KC, size) not in samples]
    if missing_baseline:
        print(
            f"ERROR: baseline KC={BASELINE_KC} missing for sizes: {missing_baseline}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"## {machine}\n")
    for size in sizes:
        base_key = (256, size)
        base_median = statistics.median(samples[base_key])
        base_series = samples[base_key]
        print(f"### size={size}\n")
        # 「対 KC=256 比」は 5 run 中央値同士の比（採否ゲートの判定基準）。
        # 「KC=256 に勝った run 数」は run 単位（同一プロセス起動内での
        # インターリーブ計測。§run_candidates_interleaved のドキュメント
        # 参照）でのペアワイズ比較で、run 間の符号反転（計測ノイズ）の
        # 有無を可視化する診断列であり、採否判定そのものには使わない
        # （計画 §3-4「5 run の対 KC=256 比の符号が run 間で反転する場合は
        # 判定不可とする」の裏付け資料）。
        print("| KC | 5 run median GFLOP/s | 対 KC=256 比 | KC=256 に勝った run 数 |")
        print("|---|---|---|---|")
        for kc in kcs:
            key = (kc, size)
            if key not in samples:
                continue
            med = statistics.median(samples[key])
            ratio = med / base_median if base_median > 0 else float("nan")
            if kc == 256:
                print(f"| {kc} | {med:.3f} | {ratio:.4f} | — |")
                continue
            series = samples[key]
            wins = sum(1 for a, b in zip(series, base_series) if a > b)
            print(f"| {kc} | {med:.3f} | {ratio:.4f} | {wins}/{len(series)} |")
        print()


if __name__ == "__main__":
    main()
