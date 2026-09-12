#!/usr/bin/env python3
"""イシュー #1583 マイクロベンチ A/B 集計（python3 標準ライブラリのみ）。

before_run{1..5}.log / after_run{1..5}.log を読み、
bench[<case>][<numel>].median_s= 行から各 run の値を集め、
5 run の中央値同士の比 (after/before) を計算する。
grad[<case>][<numel>].fold_bits= 行は全 run・before/after で完全一致することを検査する。
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
    keys = None
    before_samples = {}
    after_samples = {}
    before_grad = {}
    after_grad = {}
    for run in range(1, n_runs + 1):
        b, bg = parse(f"{logdir}/before_run{run}.log")
        a, ag = parse(f"{logdir}/after_run{run}.log")
        if keys is None:
            keys = sorted(b.keys())
        for k in keys:
            before_samples.setdefault(k, []).append(b[k])
            after_samples.setdefault(k, []).append(a[k])
        for k, v in bg.items():
            before_grad.setdefault(k, set()).add(v)
        for k, v in ag.items():
            after_grad.setdefault(k, set()).add(v)

    print(f"{'case':30s} {'numel':>9s} {'before_med_s':>14s} {'after_med_s':>14s} {'ratio':>8s} {'grad_ok':>8s}")
    all_ok = True
    for k in keys:
        bmed = statistics.median(before_samples[k])
        amed = statistics.median(after_samples[k])
        ratio = amed / bmed if bmed > 0 else float("nan")
        bset = before_grad.get(k, set())
        aset = after_grad.get(k, set())
        grad_ok = len(bset) == 1 and len(aset) == 1 and bset == aset
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
