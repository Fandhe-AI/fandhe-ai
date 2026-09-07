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

さらに #1430 codex-review 2 巡目指摘（PR #1430 のレビュースレッド。
「サンプル数を数える前に独立した run を検証する」）を受け、以下も検証する
（入力ファイルの識別を捨てて行数だけを数えると、同一 run のログファイルの
パスを 5 回渡しても「5 run median」として受理されてしまい、5 回独立プロセス
計測という契約が崩れるため）:
  - 同一の入力ファイルパス（`os.path.realpath` 正規化後）が引数に複数回
    渡された場合は拒否する（同一 run の水増しを遮断）。
  - 各ログファイル（1 run 分）が、そのファイル内で検出された各 (kc, size) を
    ちょうど 1 回ずつ含むか検証する（同一 run 内の重複行・0 回〈欠落〉を
    拒否し、`samples[(kc, size)]` の要素数が「file 数」＝「独立 run 数」と
    一致することを保証する）。

使い方: python3 aggregate.py "<機種名>" <ログファイル...>
"""
import os
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


def check_distinct_paths(paths):
    # fail-closed: 同一ファイルが複数回渡されていないか検証する（symlink・
    # 相対/絶対パス表記違いを `os.path.realpath` で正規化してから比較する）。
    # これを怠ると、同じ run1 のログを 5 回渡しても「5 run median」として
    # 受理されてしまい、5 回独立プロセス計測という契約（本ファイル冒頭の
    # docstring・計画 §3-4）が崩れる。
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
            "ERROR: duplicate input file path(s) detected (same run passed "
            f"more than once): {duplicates}",
            file=sys.stderr,
        )
        sys.exit(1)


def parse_files(paths):
    # (kc, size) -> [gflops, ...]（file 数＝独立 run 数分。プロセスごとに
    # 1 出力行。file 単位の重複・欠落は per_file_keys で検証する）
    samples = defaultdict(list)
    for path in paths:
        # このファイル（1 run 分）内で検出済みの (kc, size) 集合。同一 run
        # 内で同じ候補×形状の行が複数回出力された場合（プロセス側の異常出力・
        # ログの結合ミス等）を検出するため、行ごとに逐次照合する。
        per_file_keys = set()
        with open(path, encoding="utf-8") as f:
            for line in f:
                m = LINE_RE.search(line)
                if m:
                    kc = int(m.group("kc"))
                    size = int(m.group("size"))
                    gflops = float(m.group("gflops"))
                    key = (kc, size)
                    if key in per_file_keys:
                        print(
                            f"ERROR: duplicate line for kc={kc} size={size} "
                            f"within a single run file: {path}",
                            file=sys.stderr,
                        )
                        sys.exit(1)
                    per_file_keys.add(key)
                    samples[key].append(gflops)
    return samples


def check_run_identity(paths, samples):
    # fail-closed: 各 (kc, size) のサンプル数が入力ファイル数（＝独立 run 数）
    # と一致するか検証する。parse_files の同一 run 内重複拒否と組み合わせる
    # ことで、「samples[(kc, size)] の要素数 == 独立プロセス起動回数」が
    # 保証され、後段の `statistics.median`／run 単位のペアワイズ比較
    # （zip(series, base_series)）が異なる run 同士を取り違えて比較する
    # 余地を無くす。
    expected_runs = len(paths)
    mismatched = {
        k: len(v) for k, v in samples.items() if len(v) != expected_runs
    }
    if mismatched:
        print(
            f"ERROR: sample count does not match input file count "
            f"({expected_runs} files) for: {mismatched}",
            file=sys.stderr,
        )
        sys.exit(1)


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <machine_name> <log files...>", file=sys.stderr)
        sys.exit(2)

    machine = sys.argv[1]
    paths = sys.argv[2:]

    # fail-closed: run 識別（同一ファイルの水増し）を集計より先に検証する
    # （PR #1430 codex-review 2 巡目指摘: 「サンプル数を数える前に独立した
    # run を検証する」）。
    check_distinct_paths(paths)

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

    # fail-closed: 各 (kc, size) のサンプル数が入力ファイル数（独立 run 数）と
    # 一致するか検証する（run 重複・欠落の検出。PR #1430 codex-review 2 巡目
    # 指摘）。次の「5 サンプル未満」検査より先に行い、run 数がそもそも 5 に
    # 満たない・一致しない場合に原因を明示する。
    check_run_identity(paths, samples)

    # fail-closed: 5 サンプル未満の (kc, size) があれば拒否する（本スイープの
    # 契約は 5 回独立プロセス計測。上の check_run_identity により、この時点で
    # samples の各要素数は入力ファイル数と一致していることが保証されている
    # ため、本検査は「5 run 未満のログ集合が渡された」場合を検出する）
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
