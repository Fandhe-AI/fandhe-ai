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
print(f"=== {machine} ===")
medians = {}
for size in sizes:
    for variant in variants:
        vals = data.get((variant, size), [])
        if len(vals) != 5:
            print(f"WARNING: {variant} size={size} has {len(vals)} samples: {vals}")
        med = statistics.median(vals)
        medians[(variant, size)] = med
        print(f"{variant:16s} size={size:5d} n={len(vals)} vals={vals} median={med:.3f}")
print()
for size in sizes:
    rp = medians[("RowPanel", size)]
    ic = medians[("IcDynamic", size)]
    ratio = ic / rp
    print(f"size={size}: IcDynamic/RowPanel = {ic:.3f}/{rp:.3f} = {ratio:.4f}")
