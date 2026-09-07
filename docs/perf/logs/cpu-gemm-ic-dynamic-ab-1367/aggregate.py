import re, statistics, glob, sys

def parse(path):
    out = {}
    with open(path) as f:
        for line in f:
            m = re.match(r"variant=(\S+) size=(\d+) median_gflops=([\d.]+)", line)
            if m:
                variant, size, val = m.group(1), int(m.group(2)), float(m.group(3))
                out.setdefault((variant, size), []).append(val)
    return out

def aggregate(prefix, sizes_files):
    data = {}
    for path in sizes_files:
        d = parse(path)
        for k, v in d.items():
            data.setdefault(k, []).extend(v)
    return data

import sys
machine = sys.argv[1]
files = sys.argv[2:]
data = {}
for path in files:
    d = parse(path)
    for k, v in d.items():
        data.setdefault(k, []).extend(v)

variants = ["RowPanel", "SharedB", "SharedBPcOuter", "IcDynamic"]
sizes = [1024, 2048, 4096]
EXPECTED_SAMPLES = 5

# 事前検証: 全 (variant, size) 組み合わせのサンプル数が EXPECTED_SAMPLES と
# 一致することを確認する。採否判断は AGENTS.md の「性能予算」（5 回計測中央値）
# を前提としており、件数不一致（ログ渡し忘れ等）のまま中央値・比較比を算出
# すると採否判断の根拠が誤る。不一致があれば非ゼロ終了し、集計を行わない
# （fail-closed。イシュー #1367 PR #1412 codex-review 指摘）。
missing = []
for size in sizes:
    for variant in variants:
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

print(f"=== {machine} ===")
medians = {}
for size in sizes:
    for variant in variants:
        vals = data.get((variant, size), [])
        med = statistics.median(vals)
        medians[(variant, size)] = med
        print(f"{variant:16s} size={size:5d} n={len(vals)} vals={vals} median={med:.3f}")
print()
for size in sizes:
    rp = medians[("RowPanel", size)]
    ic = medians[("IcDynamic", size)]
    ratio = ic / rp
    print(f"size={size}: IcDynamic/RowPanel = {ic:.3f}/{rp:.3f} = {ratio:.4f}")
