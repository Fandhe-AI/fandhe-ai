import re, statistics

files = {
    "global": "docs/perf/logs/cpu-gemm-small-shape-thread-cap-1575/phase0_global.log",
    "dedicated:2": "docs/perf/logs/cpu-gemm-small-shape-thread-cap-1575/phase0_dedicated_2.log",
    "dedicated:4": "docs/perf/logs/cpu-gemm-small-shape-thread-cap-1575/phase0_dedicated_4.log",
    "dedicated:6": "docs/perf/logs/cpu-gemm-small-shape-thread-cap-1575/phase0_dedicated_6.log",
    "dedicated:8": "docs/perf/logs/cpu-gemm-small-shape-thread-cap-1575/phase0_dedicated_8.log",
    "rayon_env_4": "docs/perf/logs/cpu-gemm-small-shape-thread-cap-1575/phase0_rayon_env_4.log",
}

data = {}
checksums = {}

for arm, path in files.items():
    data[arm] = {}
    with open(path) as f:
        for line in f:
            m = re.search(r'shape=(\S+) m=(\d+) n=(\d+) k=(\d+) median_secs=([\d.]+) checksum=(0x[0-9a-f]+)', line)
            if not m:
                continue
            shape, mm, nn, kk, median, checksum = m.groups()
            data[arm].setdefault(shape, []).append(float(median))
            checksums.setdefault(shape, set()).add(checksum)

print("=== checksum consistency ===")
for shape, cs in checksums.items():
    status = "OK" if len(cs) == 1 else "MISMATCH"
    print(f"{shape}: {status} ({cs})")

print()
print("=== medians (5-run median of per-process median) and ratio vs global ===")
shapes = list(next(iter(data.values())).keys())
for shape in shapes:
    global_med = statistics.median(data["global"][shape])
    print(f"-- {shape} (global median={global_med*1e6:.2f} us) --")
    for arm in files:
        vals = data[arm][shape]
        med = statistics.median(vals)
        ratio = med / global_med
        print(f"   {arm:14s} median={med*1e6:8.2f} us  ratio_vs_global={ratio:.4f}")
