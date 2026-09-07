import json, math

with open("agg_f32fma.json") as f:
    A = json.load(f)
with open("agg_f64.json") as f:
    B = json.load(f)

SHAPES_ORDER = [
    "1x1x1 (sub-K-tile, TF32 suite)",
    "32x32x32 (block tile)",
    "64x64x64 (block tile x2)",
    "128x128x128 (block tile x4)",
    "256x256x256 (K sweep base)",
    "512x512x512 (block tile x16)",
    "17x23x19 (non-multiple edge, TF32 suite)",
    "17x19x23 (non-multiple edge, f16 suite)",
    "33x31x65 (non-multiple edge)",
    "100x100x100 (non-multiple edge)",
    "130x70x90 (non-multiple edge)",
    "64x96x128 (non-square)",
    "256x256x512 (K sweep)",
    "256x256x1024 (K sweep)",
    "256x256x4096 (K sweep, PoC-v2-5 stress)",
]
ROUTES = ["f32_simt", "mma_tf32", "mma_tf32x3"]

# 一様に f64（診断行）の max_abs_diff を r(s) の指標として使う（f32fma 側は
# fail=0 セルで max_fail_abs_diff=0 になり得り比が定義不能になるため。
# f64 diagnostic max_abs_diff は丸め誤差そのものを表すため常に非ゼロ）。
def metric_series(route, shape):
    key1 = f"{route}|{shape}|1"
    if key1 not in B:
        return None
    vals = {}
    for s in ["0.1", "1", "10", "100"]:
        k = f"{route}|{shape}|{s}"
        if k not in B:
            return None
        vals[s] = B[k]["max_abs"]
    return vals

results = {}
for route in ROUTES:
    rows = []
    for shape in SHAPES_ORDER:
        vals = metric_series(route, shape)
        if vals is None:
            continue
        base = vals["1"]
        if base is None or base == 0.0:
            rows.append((shape, None, None, None))
            continue
        r01 = vals["0.1"] / (0.01 * base) if vals["0.1"] is not None else None
        r10 = vals["10"] / (100 * base) if vals["10"] is not None else None
        r100 = vals["100"] / (10000 * base) if vals["100"] is not None else None
        rows.append((shape, r01, r10, r100))
    results[route] = rows

def logr(x):
    if x is None or x <= 0:
        return None
    return math.log10(x)

for route in ROUTES:
    print(f"== {route} ==")
    all_r = []
    for shape, r01, r10, r100 in results[route]:
        def f(x):
            return "n/a" if x is None else f"{x:.3f}"
        print(f"{shape} | r(0.1)={f(r01)} r(10)={f(r10)} r(100)={f(r100)}")
        for r in (r01, r10, r100):
            if r is not None:
                all_r.append(r)
    logs = [abs(logr(r)) for r in all_r if logr(r) is not None]
    if logs:
        print(f"  n={len(all_r)} min={min(all_r):.3f} median={sorted(all_r)[len(all_r)//2]:.3f} max={max(all_r):.3f} max|log10 r|={max(logs):.3f}")
    print()

with open("s2_results.json", "w") as f:
    json.dump(results, f)
