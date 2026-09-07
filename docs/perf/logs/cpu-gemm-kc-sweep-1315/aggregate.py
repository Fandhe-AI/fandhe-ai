#!/usr/bin/env python3
"""KC 再スイープ（イシュー #1315）5 回独立プロセス実行ログの集計スクリプト。

`gemm_blis_kc_sweep_ab_1024_2048`／`gemm_blis_kc_sweep_ab_4096`
（`crates/backend-cpu/src/gemm_blis/mod.rs`）が出力する
`variant=RowPanel kc={kc} size={dim} median_gflops={v}` 行を集計し、
KC ごと・形状ごとの 5 run 中央値・対 KC=256 比を算出する。

#1367（`cpu-gemm-ic-dynamic-variant.md` 引き継ぎ）の aggregate.py 系譜を
踏襲し、5 サンプル未満での集計を fail-closed で拒否する（実測値の
捏造・不完全データでの判定確定を防ぐ）。Python3 標準ライブラリのみ。

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

    # fail-closed: 5 サンプル未満の (kc, size) があれば拒否する
    incomplete = {k: len(v) for k, v in samples.items() if len(v) < 5}
    if incomplete:
        print(f"ERROR: sample count < 5 for: {incomplete}", file=sys.stderr)
        sys.exit(1)

    sizes = sorted({size for (_, size) in samples})
    kcs = sorted({kc for (kc, _) in samples})

    print(f"## {machine}\n")
    for size in sizes:
        base_key = (256, size)
        if base_key not in samples:
            continue
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
