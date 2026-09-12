#!/usr/bin/env python3
"""イシュー #1583 マイクロベンチ A/B 集計（python3 標準ライブラリのみ）。

before_run{1..5}.log / after_run{1..5}.log を読み、
bench[<case>][<numel>].median_s= 行から各 run の値を集め、
5 run の中央値同士の比 (after/before) を計算する。
grad[<case>][<numel>].fold_bits= 行は全 run・before/after で完全一致することを検査する。

集計前に、想定される全セル（18 セル）× 全 run（既定 5 run）× before/after の
bench・grad 記録がすべて揃っていることを検証する（イシュー #1583 PR #1674
codex-review 指摘）。従来実装は 1 本目のログから期待セル集合を決め、grad は
run 間で値集合へ単純合体していたため、セル欠落や一部 run の勾配欠落があっても
値の集合が単一要素になり `grad bit-fold all match: True` と誤表示され得た。
本実装は各 run・各 before/after で bench・grad とも同一のキー集合を報告して
いることと、grad の報告回数が n_runs と一致することを明示的に検査し、
欠落があれば集計を打ち切って fail-closed に報告する。
"""
import re
import sys
import statistics


def parse(path):
    bench = {}
    grad = {}
    with open(path) as f:
        for line in f:
            m = re.match(r"bench\[([^\]]+)\]\[(\d+)\]\.median_s=([0-9.eE+-]+)", line)
            if m:
                key = (m.group(1), int(m.group(2)))
                bench[key] = float(m.group(3))
                continue
            m = re.match(r"grad\[([^\]]+)\]\[(\d+)\]\.fold_bits=(0x[0-9a-f]+)", line)
            if m:
                key = (m.group(1), int(m.group(2)))
                grad[key] = m.group(3)
    return bench, grad


def main(logdir, n_runs=5):
    before_samples = {}
    after_samples = {}
    # grad は run ごとの値をすべて保持する（欠落 run・欠落セルを件数で検知するため
    # 値集合〈set〉だけに畳み込まない。旧実装は values の union のみを見ており、
    # 一部 run が欠落していても報告された値がたまたま一致すれば見分けが付かなかった）。
    before_grad = {}
    after_grad = {}

    expected_keys = None
    missing = []

    for run in range(1, n_runs + 1):
        before_path = f"{logdir}/before_run{run}.log"
        after_path = f"{logdir}/after_run{run}.log"
        b, bg = parse(before_path)
        a, ag = parse(after_path)

        if expected_keys is None:
            # 期待セル集合は「1 本目の before ログ」だけでなく、1 本目の
            # before/after 双方の bench キー和集合から決める。以降の全 run・
            # before/after がこの集合を過不足なく満たすことを検査する。
            expected_keys = sorted(set(b.keys()) | set(a.keys()))
            if not expected_keys:
                print(f"ERROR: {before_path} / {after_path} から bench セルを 1 件も読み取れませんでした")
                sys.exit(1)

        for label, path, keys_present in (
            ("before bench", before_path, set(b.keys())),
            ("after bench", after_path, set(a.keys())),
            ("before grad", before_path, set(bg.keys())),
            ("after grad", after_path, set(ag.keys())),
        ):
            lacking = set(expected_keys) - keys_present
            if lacking:
                missing.append((run, label, path, sorted(lacking)))

        for k in expected_keys:
            if k in b:
                before_samples.setdefault(k, []).append(b[k])
            if k in a:
                after_samples.setdefault(k, []).append(a[k])
            if k in bg:
                before_grad.setdefault(k, []).append(bg[k])
            if k in ag:
                after_grad.setdefault(k, []).append(ag[k])

    if missing:
        print(f"ERROR: 集計前検証に失敗しました（期待セル数={len(expected_keys)}・期待 run 数={n_runs}）")
        for run, label, path, lacking in missing:
            print(f"  run{run} {label} ({path}): 欠落セル {lacking}")
        print()
        print("全セル・全 run の計測記録が揃っていないため集計を中止します。")
        sys.exit(1)

    print(f"{'case':30s} {'numel':>9s} {'before_med_s':>14s} {'after_med_s':>14s} {'ratio':>8s} {'grad_ok':>8s}")
    all_ok = True
    for k in expected_keys:
        bench_samples_b = before_samples[k]
        bench_samples_a = after_samples[k]
        # bench 側も n_runs 件そろっていることを明示的に検査する（上の missing
        # 検査は bench 行の有無〈キー単位〉のみを見ているため、同一 run 内で
        # 同じキーが重複記録される異常なログでも件数不一致として拾えるようにする）。
        if len(bench_samples_b) != n_runs or len(bench_samples_a) != n_runs:
            print(f"ERROR: {k} の bench サンプル数が n_runs={n_runs} と一致しません "
                  f"(before={len(bench_samples_b)}, after={len(bench_samples_a)})")
            sys.exit(1)

        bmed = statistics.median(bench_samples_b)
        amed = statistics.median(bench_samples_a)
        ratio = amed / bmed if bmed > 0 else float("nan")

        bvals = before_grad[k]
        avals = after_grad[k]
        # grad_ok は「値が一意」であることに加え、報告件数が n_runs と一致する
        # ことも要求する（欠落 run があれば件数不一致で検知し MISMATCH とする）。
        grad_ok = (
            len(bvals) == n_runs
            and len(avals) == n_runs
            and len(set(bvals)) == 1
            and len(set(avals)) == 1
            and set(bvals) == set(avals)
        )
        if not grad_ok:
            all_ok = False
        flag = "OK" if grad_ok else "MISMATCH"
        over = " <=1.00" if ratio <= 1.00 else " >1.00"
        print(f"{k[0]:30s} {k[1]:9d} {bmed:14.9f} {amed:14.9f} {ratio:8.4f} {flag:>8s}{over}")

    print()
    print("grad bit-fold all match:", all_ok)


if __name__ == "__main__":
    logdir = sys.argv[1] if len(sys.argv) > 1 else "."
    n_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    main(logdir, n_runs)
