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
  - run ヘッダ（== RAYON_NUM_THREADS=<T> run=<i> taskset=<mask> ==）の
    run id が (T, taskset) の組ごとに 1..N の連番かつ重複がないことを
    検証する（同一 run の重複実行・run id の欠落を検出する。run id を
    見ず単純にサンプル数のみで判定すると、重複と欠落が互いを相殺して
    期待件数を満たしてしまう検出漏れを防ぐ）。
  - 各 JSON レコードには、直前に出現した run ヘッダの run id を保持する
    （#1429 codex-review 追加指摘）。これにより (a) ある run 内での JSON
    行の重複・別 run での対応行の欠落が互いを相殺してサンプル数検査を
    通過してしまう事態を、グループ内の run id 集合が {1..expected_samples}
    と完全一致することを直接検証することで検出する。(b) 末尾の run 単位
    符号一貫性比較（taskset=none, T8 比）は、ファイル中の出現順に基づく
    位置的な zip ではなく run id で明示的に対応付ける（run ブロックの並び
    順が条件間で異なるログでも、異なる run 同士を比較しないようにする）。
  - 各 (impl, size, T, taskset) の組み合わせのサンプル数が期待値
    （既定 5）と一致しない場合は集計を中止する。判定対象の組み合わせは
    「ログ中に出現した (T, taskset) の全組」×「ログ中に出現した size の
    全種類」×「TARGET_IMPLS」の直積として事前に列挙する（JSON 行から
    実際に得られたキーだけを走査すると、ある T・size・impl の組み合わせで
    サンプルが 1 件も出力されなかった場合にそのキー自体が grouped に
    存在せず検知できないため。#1429 codex-review 指摘）。

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
    """1 ログファイルをパースし records・headers を返す。

    records: [(T, taskset, run_id, size, impl, impl_threads, tflops_median), ...]
        run_id は直前に出現した run ヘッダの run id（#1429 codex-review 指摘対応。
        重複・欠落の相殺検出とサンプル数検査・run 単位符号一貫性比較の run id 対応
        付けに使う）。
    headers: [(T, taskset, run_id), ...]（出現順そのまま。重複・欠落検証用）
    """
    records = []
    headers = []
    cur_t = None
    cur_taskset = None
    cur_run_id = None
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            m = HEADER_RE.match(line)
            if m:
                cur_t = int(m.group(1))
                cur_taskset = m.group(3)
                cur_run_id = int(m.group(2))
                headers.append((cur_t, cur_taskset, cur_run_id))
                continue
            m = RC_RE.match(line)
            if m:
                cur_t = None
                cur_run_id = None
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
                    (
                        cur_t,
                        cur_taskset,
                        cur_run_id,
                        obj["size"],
                        obj["impl"],
                        obj["impl_threads"],
                        obj["tflops_median"],
                    )
                )
    return records, headers


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <machine-label> <logfile> [expected_samples]", file=sys.stderr)
        sys.exit(2)
    machine = sys.argv[1]
    path = sys.argv[2]
    expected_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    records, headers = parse_log(path)
    if not records:
        print(f"ERROR: {path} からレコードを 1 件も取得できなかった", file=sys.stderr)
        sys.exit(1)
    if not headers:
        print(f"ERROR: {path} から run ヘッダを 1 件も取得できなかった", file=sys.stderr)
        sys.exit(1)

    # fail-closed 検証 1: impl_threads と T の整合性（self/gemm のみ）
    mismatches = []
    for t, taskset, run_id, size, impl, impl_threads, tflops in records:
        if impl in TARGET_IMPLS and impl_threads != t:
            mismatches.append((t, taskset, size, impl, impl_threads))
    if mismatches:
        print(f"=== {machine}: impl_threads 不一致 ===", file=sys.stderr)
        for m in mismatches:
            print(f"  T={m[0]} taskset={m[1]} size={m[2]} impl={m[3]} impl_threads={m[4]}", file=sys.stderr)
        print("集計を中止しました（RAYON_NUM_THREADS がプールへ反映されていない疑い）。", file=sys.stderr)
        sys.exit(1)

    # fail-closed 検証 2: run id の重複・欠落（(T, taskset) 単位で 1..N の連番か）
    run_ids_by_condition = {}
    for t, taskset, run_id in headers:
        run_ids_by_condition.setdefault((t, taskset), []).append(run_id)
    run_id_errors = []
    for (t, taskset), ids in run_ids_by_condition.items():
        n = len(ids)
        if len(set(ids)) != n:
            run_id_errors.append((t, taskset, "重複", ids))
        elif sorted(ids) != list(range(1, n + 1)):
            run_id_errors.append((t, taskset, "非連番（欠落の疑い）", ids))
    if run_id_errors:
        print(f"=== {machine}: run id 異常 ===", file=sys.stderr)
        for t, taskset, kind, ids in run_id_errors:
            print(f"  T={t} taskset={taskset}: {kind} run_ids={sorted(ids)}", file=sys.stderr)
        print("集計を中止しました（同一 run の重複実行または run id 欠落の疑い）。", file=sys.stderr)
        sys.exit(1)

    # グルーピング: (taskset, size, T, impl) -> [(run_id, tflops_median), ...]
    grouped = {}
    for t, taskset, run_id, size, impl, impl_threads, tflops in records:
        if impl not in TARGET_IMPLS:
            continue
        grouped.setdefault((taskset, size, t, impl), []).append((run_id, tflops))

    # fail-closed 検証 3: サンプル数（全欠落条件を含む網羅的チェック）。
    # 判定対象は「ログに出現した (T, taskset) の全組」×「ログに出現した
    # size の全種類（impl 不問。matrixmultiply も含めて SIZES 引数の実際の
    # 値を復元する）」×「TARGET_IMPLS」の直積として事前に列挙する。
    # grouped.items() のみを走査すると、ある組み合わせでサンプルが
    # 1 件も出力されなかった場合にそのキー自体が存在せず検知できない。
    conditions = sorted({(t, taskset) for t, taskset, _ in headers})
    all_sizes = sorted({r[3] for r in records})
    expected_keys = [
        (taskset, size, t, impl)
        for (t, taskset) in conditions
        for size in all_sizes
        for impl in TARGET_IMPLS
    ]
    bad = []
    for k in expected_keys:
        n = len(grouped.get(k, []))
        if n != expected_samples:
            bad.append((k, n, [v for _, v in grouped.get(k, [])]))
    if bad:
        print(f"=== {machine}: サンプル数不一致（期待 {expected_samples}）===", file=sys.stderr)
        for (taskset, size, t, impl), n, vals in bad:
            print(f"  taskset={taskset} size={size} T={t} impl={impl}: n={n} vals={vals}", file=sys.stderr)
        print("集計を中止しました。", file=sys.stderr)
        sys.exit(1)

    # fail-closed 検証 4: グループ内 run id 集合の完全一致（#1429 codex-review
    # 追加指摘）。検証 3（件数一致）だけでは、ある run の JSON 行が重複し別の
    # run の対応行が欠落しても件数は期待値のまま変わらず検出できない。ここでは
    # 各グループの run id 集合が {1..expected_samples} と完全一致することを
    # 直接検証し、重複・欠落の相殺を検出する。
    run_id_bad = []
    for k in expected_keys:
        ids = sorted(run_id for run_id, _ in grouped.get(k, []))
        if ids != list(range(1, expected_samples + 1)):
            run_id_bad.append((k, ids))
    if run_id_bad:
        print(f"=== {machine}: グループ内 run id 不一致（期待 1..{expected_samples}）===", file=sys.stderr)
        for (taskset, size, t, impl), ids in run_id_bad:
            print(f"  taskset={taskset} size={size} T={t} impl={impl}: run_ids={ids}", file=sys.stderr)
        print("集計を中止しました（run の重複・欠落が相殺されサンプル数検査を通過した疑い）。", file=sys.stderr)
        sys.exit(1)

    medians = {k: statistics.median(v for _, v in vs) for k, vs in grouped.items()}

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
        base_entries = grouped.get(("none", size, 8, "self_gemm_blis_parallel"))
        if not base_entries:
            continue
        # run id -> tflops の辞書化。位置的な zip ではなく run id で明示的に
        # 対応付けることで、run ブロックの並び順が条件間で異なるログでも
        # 異なる run 同士を比較しない（#1429 codex-review 指摘対応）。
        base_by_run = dict(base_entries)
        for t in threads:
            if t == 8:
                continue
            entries = grouped.get(("none", size, t, "self_gemm_blis_parallel"))
            if not entries:
                continue
            vals_by_run = dict(entries)
            common_run_ids = sorted(set(base_by_run) & set(vals_by_run))
            if set(base_by_run) != set(vals_by_run):
                print(
                    f"WARN: size={size} T={t}: base run_ids={sorted(base_by_run)} と "
                    f"vals run_ids={sorted(vals_by_run)} が不一致（共通分のみ比較）",
                    file=sys.stderr,
                )
            ratios = [vals_by_run[rid] / base_by_run[rid] for rid in common_run_ids]
            below_1 = sum(1 for r in ratios if r < 1.0)
            print(f"  size={size} T={t}: ratios={['%.4f' % r for r in ratios]} below_1_count={below_1}/{len(ratios)}")


if __name__ == "__main__":
    main()
