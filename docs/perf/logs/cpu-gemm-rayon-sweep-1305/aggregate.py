#!/usr/bin/env python3
"""イシュー #1305: RAYON_NUM_THREADS スイープログの集計スクリプト。

python3 標準ライブラリのみで完結する（#1367 の aggregate.py と同方針）。

入力ログは run_sweep.sh が生成する形式:
    == RAYON_NUM_THREADS=<T> run=<i> taskset=<mask|none> ==
    {JSON Lines...}
    == RAYON_NUM_THREADS=<T> run=<i> rc=<code> ==

fail-closed 検証（#1367 と同方針）:
  - run ヘッダの T と JSON 行の impl_threads が self_gemm_blis_parallel /
    gemm について一致しない場合は集計を中止する（rayon プールへの
    RAYON_NUM_THREADS 反映漏れの検出）。matrixmultiply は単一スレッド実装の
    ため impl_threads は常に 1 で T に追従しない仕様であり検証対象外。
  - 各 (impl, size, T) の組み合わせのサンプル数が期待値（既定 5）と
    一致しない場合は集計を中止する。

出力: 標準出力へ表形式の集計・§2.1 事前宣言基準に基づく再現判定を出す。
"""
import re
import statistics
import sys
import json

HEADER_RE = re.compile(r"^== RAYON_NUM_THREADS=(\d+) run=(\d+) taskset=(\S+) ==$")
RC_RE = re.compile(r"^== RAYON_NUM_THREADS=(\d+) run=(\d+) rc=(-?\d+) ==$")

TARGET_IMPLS = ["self_gemm_blis_parallel", "gemm"]


def parse_log(path):
    """1 ログファイルをパースし [(T, taskset, size, impl, impl_threads, tflops_median), ...] を返す。"""
    records = []
    cur_t = None
    cur_taskset = None
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            m = HEADER_RE.match(line)
            if m:
                cur_t = int(m.group(1))
                cur_taskset = m.group(3)
                continue
            m = RC_RE.match(line)
            if m:
                cur_t = None
                continue
            if line.startswith("{"):
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if cur_t is None:
                    print(f"WARN: {path}: JSON 行がヘッダ外に出現: {line[:80]}", file=sys.stderr)
                    continue
                records.append(
                    (cur_t, cur_taskset, obj["size"], obj["impl"], obj["impl_threads"], obj["tflops_median"])
                )
    return records


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <machine-label> <logfile> [expected_samples]", file=sys.stderr)
        sys.exit(2)
    machine = sys.argv[1]
    path = sys.argv[2]
    expected_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    records = parse_log(path)
    if not records:
        print(f"ERROR: {path} からレコードを 1 件も取得できなかった", file=sys.stderr)
        sys.exit(1)

    # fail-closed 検証 1: impl_threads と T の整合性（self/gemm のみ）
    mismatches = []
    for t, taskset, size, impl, impl_threads, tflops in records:
        if impl in TARGET_IMPLS and impl_threads != t:
            mismatches.append((t, taskset, size, impl, impl_threads))
    if mismatches:
        print(f"=== {machine}: impl_threads 不一致 ===", file=sys.stderr)
        for m in mismatches:
            print(f"  T={m[0]} taskset={m[1]} size={m[2]} impl={m[3]} impl_threads={m[4]}", file=sys.stderr)
        print("集計を中止しました（RAYON_NUM_THREADS がプールへ反映されていない疑い）。", file=sys.stderr)
        sys.exit(1)

    # グルーピング: (taskset, size, T, impl) -> [tflops_median, ...]
    grouped = {}
    for t, taskset, size, impl, impl_threads, tflops in records:
        if impl not in TARGET_IMPLS:
            continue
        grouped.setdefault((taskset, size, t, impl), []).append(tflops)

    # fail-closed 検証 2: サンプル数
    bad = []
    for k, vals in grouped.items():
        if len(vals) != expected_samples:
            bad.append((k, len(vals), vals))
    if bad:
        print(f"=== {machine}: サンプル数不一致（期待 {expected_samples}）===", file=sys.stderr)
        for (taskset, size, t, impl), n, vals in bad:
            print(f"  taskset={taskset} size={size} T={t} impl={impl}: n={n} vals={vals}", file=sys.stderr)
        print("集計を中止しました。", file=sys.stderr)
        sys.exit(1)

    medians = {k: statistics.median(v) for k, v in grouped.items()}

    print(f"=== {machine}: 中央値表（n={expected_samples}）===")
    tasksets = sorted({k[0] for k in medians})
    sizes = sorted({k[1] for k in medians})
    threads = sorted({k[2] for k in medians})
    for taskset in tasksets:
        for size in sizes:
            print(f"-- taskset={taskset} size={size} --")
            base = medians.get((taskset, size, 8, "self_gemm_blis_parallel"))
            for t in threads:
                row = []
                for impl in TARGET_IMPLS:
                    v = medians.get((taskset, size, t, impl))
                    if v is not None:
                        row.append(f"{impl}={v:.4f}")
                ratio_str = ""
                if base is not None:
                    v_self = medians.get((taskset, size, t, "self_gemm_blis_parallel"))
                    if v_self is not None:
                        ratio_str = f" self/T8_ratio={v_self / base:.4f}"
                print(f"  T={t:3d} " + " ".join(row) + ratio_str)
    print()

    # §2.1 再現判定（taskset=none のみ対象。T_big は呼び出し側が別途指定して比較する）
    # ここでは taskset=none の全 T について T=8 比だけを機械的に出力するに留め、
    # 「再現/非再現/判定不能」の最終確定は §2.1 の run 単位符号一貫性条件を
    # 人間 (ドキュメント執筆時) が run ごとの生値と突合して判定する。
    print("=== run 単位の T8 比（符号一貫性確認用。taskset=none）===")
    for size in sizes:
        base_vals = grouped.get(("none", size, 8, "self_gemm_blis_parallel"))
        if not base_vals:
            continue
        for t in threads:
            if t == 8:
                continue
            vals = grouped.get(("none", size, t, "self_gemm_blis_parallel"))
            if not vals:
                continue
            ratios = [v / b for v, b in zip(vals, base_vals)]
            below_1 = sum(1 for r in ratios if r < 1.0)
            print(f"  size={size} T={t}: ratios={['%.4f' % r for r in ratios]} below_1_count={below_1}/{len(ratios)}")


if __name__ == "__main__":
    main()
