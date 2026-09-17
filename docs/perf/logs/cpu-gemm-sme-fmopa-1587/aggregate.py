#!/usr/bin/env python3
"""イシュー #1978（#1587 事前登録規則 R4）: SME vs NEON 16 格子点 × 5 run の集計。

各 run・各格子点で `ratio = SME median_gflops / NEON median_gflops` を取り、
格子点ごとに 5 run 中央値と「5/5 run で SME>=NEON（ratio>=1.0）」を判定する。
R4: しきい値 (min_mn, min_k) 以上の全格子点が 5/5 run で SME>=NEON となる
極小の組（パレート極小）を採用候補として列挙する。python3 標準ライブラリのみ。
使い方: aggregate.py [--self-test] <dir> > aggregate_r4.md
"""
import re
import statistics
import sys
from pathlib import Path

LINE = re.compile(r"(?:^|\s)variant=(NEON|SME)\([A-Za-z]+\) size=\((\d+),(\d+),(\d+)\) median_gflops=([0-9.]+)$")
MN = [64, 128, 256, 512]
KS = [32, 64, 128, 256]


def parse(text):
    out = {}
    for line in text.splitlines():
        m = LINE.search(line.strip())
        if m:
            out[(m.group(1), int(m.group(2)), int(m.group(4)))] = float(m.group(5))
    return out


def candidates(ok):
    """ok[(mn,k)] -> bool。全格子点が ok となる (min_mn,min_k) のパレート極小集合。"""
    feas = [(a, b) for a in MN for b in KS
            if all(ok[(mn, k)] for mn in MN for k in KS if mn >= a and k >= b)]
    return [(a, b) for (a, b) in feas
            if not any((c <= a and d <= b and (c, d) != (a, b)) for (c, d) in feas)]


def self_test():
    ok = {(mn, k): (mn >= 256 and k >= 64) for mn in MN for k in KS}
    assert candidates(ok) == [(256, 64)], candidates(ok)
    ok = {(mn, k): False for mn in MN for k in KS}
    assert candidates(ok) == []
    r = parse("variant=NEON(TwoDDynamic) size=(64,64,32) median_gflops=1.500\n"
              "variant=SME(TwoDDynamicSme) size=(64,64,32) median_gflops=3.000\n")
    assert r[("SME", 64, 32)] / r[("NEON", 64, 32)] == 2.0
    print("self-test ok")


def main():
    if "--self-test" in sys.argv:
        self_test()
        return 0
    d = Path(sys.argv[1])
    runs = []
    for i in range(1, 6):
        f = d / f"sme_r4_grid_run{i}.log"
        if not f.exists():
            print(f"error: {f.name} が無い（5 run 必須）", file=sys.stderr)
            return 1
        r = parse(f.read_text(encoding="utf-8"))
        if len(r) != 32:
            print(f"error: {f.name} の行数が 32 でない（{len(r)}）", file=sys.stderr)
            return 1
        runs.append(r)
    print("# SME vs NEON R4 格子（16 点 × 5 run。ratio = SME/NEON の median_gflops 比）\n")
    print("| min(m,n) | k | run ごとの比 | 中央値 | 5/5 run で SME>=NEON |")
    print("|---:|---:|---|---:|---|")
    ok = {}
    for mn in MN:
        for k in KS:
            ratios = [r[("SME", mn, k)] / r[("NEON", mn, k)] for r in runs]
            ok[(mn, k)] = all(x >= 1.0 for x in ratios)
            print(f"| {mn} | {k} | {', '.join(f'{x:.3f}' for x in ratios)} | "
                  f"{statistics.median(ratios):.3f} | {'yes' if ok[(mn, k)] else 'no'} |")
    c = candidates(ok)
    print("\n## R4 採用候補（しきい値以上の全格子点が 5/5 run で SME>=NEON となる極小の組）\n")
    if c:
        for a, b in c:
            print(f"- `min(m,n) >= {a}` かつ `k >= {b}`")
    else:
        print("- なし（格子内に条件を満たす組が存在しない）")
    print("\n現行定数 `SME_MIN_M/N=256`・`SME_MIN_K=64` は本集計では変更しない（本番切替は #1979 のユーザー承認事項）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
