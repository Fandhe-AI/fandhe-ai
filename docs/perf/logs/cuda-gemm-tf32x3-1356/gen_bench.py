import statistics
from collections import defaultdict

def load_bench_csv(path):
    rows = []
    with open(path, newline="") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            if line.startswith("route,") or line.startswith("device compute"):
                continue
            cols = line.rstrip("\n").split(",")
            if len(cols) != 9:
                continue
            route, m, n, k, kernel, median_ms, q1_ms, q3_ms, tflops = cols
            rows.append(dict(route=route, shape=f"{m}x{n}x{k}", kernel=kernel,
                median_ms=float(median_ms), tflops=float(tflops)))
    return rows

def aggregate_bench_runs(paths):
    grouped = defaultdict(list)
    for p in paths:
        for r in load_bench_csv(p):
            grouped[(r["route"], r["shape"])].append(r)
    out = {}
    for key, entries in grouped.items():
        out[key] = dict(
            kernel=entries[0]["kernel"],
            median_ms=statistics.median(e["median_ms"] for e in entries),
            tflops=statistics.median(e["tflops"] for e in entries),
            n_runs=len(entries),
        )
    return out

agg = aggregate_bench_runs([f"bench-run{i}.csv" for i in range(1, 6)])

SHAPES = ["512x512x512", "1024x1024x1024", "2048x2048x2048", "4096x4096x4096", "256x256x4096"]
ROUTES = ["f32_simt", "wmma_tf32", "mma_tf32", "mma_tf32x3"]

print("| 形状 | " + " | ".join(f"{r} (TFLOPS)" for r in ROUTES) + " | tf32x3/f32_simt | tf32x3/mma_tf32 |")
print("|---" * (len(ROUTES) + 3) + "|")
for shape in SHAPES:
    vals = {}
    for route in ROUTES:
        key = (route, shape)
        vals[route] = agg[key]["tflops"] if key in agg else None
    def f(x):
        return "n/a" if x is None else f"{x:.3f}"
    ratio1 = vals["mma_tf32x3"] / vals["f32_simt"] if vals["mma_tf32x3"] and vals["f32_simt"] else None
    ratio2 = vals["mma_tf32x3"] / vals["mma_tf32"] if vals["mma_tf32x3"] and vals["mma_tf32"] else None
    row = [shape] + [f(vals[r]) for r in ROUTES] + [f(ratio1), f(ratio2)]
    print("| " + " | ".join(row) + " |")
