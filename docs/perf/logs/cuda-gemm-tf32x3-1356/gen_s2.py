import json, math, statistics

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

# --- 主指標（#995 §8/§9 の判定基準そのもの: 対 f32fma の max_fail_abs_diff）---
# r(s) = max_fail_abs_diff(s) / (s^2 * max_fail_abs_diff(1))。
# max_fail_abs_diff(1) が 0（= assert_parity 側の fail が 0 セルで比が定義不能）の
# 形状に限り、対 f64 の max_abs_diff を代替指標として使う（f64 診断行の
# max_abs_diff は丸め誤差そのものを表すため常に非ゼロ）。
# 比が計算可能な形状まで一律 f64 へ置き換えない（#995 §8/§9 の契約を維持する）。
def primary_metric_series(route, shape):
    key1 = f"{route}|{shape}|1"
    if key1 not in A:
        return None, None
    base = A[key1]["max_fail_abs"]
    if base and base > 0.0:
        source = "f32fma"
        vals = {}
        for s in ["0.1", "1", "10", "100"]:
            k = f"{route}|{shape}|{s}"
            if k not in A:
                return None, None
            vals[s] = A[k]["max_fail_abs"]
        return source, vals
    # fail=0（比が定義不能）: f64 診断行の max_abs_diff で代替する。
    key1b = f"{route}|{shape}|1"
    if key1b not in B:
        return None, None
    source = "f64(fallback)"
    vals = {}
    for s in ["0.1", "1", "10", "100"]:
        k = f"{route}|{shape}|{s}"
        if k not in B:
            return None, None
        vals[s] = B[k]["max_abs"]
    return source, vals

results = {}
sources = {}
for route in ROUTES:
    rows = []
    row_sources = []
    for shape in SHAPES_ORDER:
        source, vals = primary_metric_series(route, shape)
        if vals is None:
            continue
        base = vals["1"]
        if base is None or base == 0.0:
            rows.append((shape, None, None, None))
            row_sources.append(source)
            continue
        # f32fma 主指標では特定スケールで fail=0（max_fail_abs=0）になり
        # 比自体が定義不能な個別セルが生じうる（例: 256x256x256 の s=0.1）。
        # base(1)=0 の形状除外と同じ理由で、そのセルは n/a（比較対象外）とする
        # （0.000 という比を実測値として集計に混ぜない）。
        def ratio(v, scale2):
            if v is None:
                return None
            if source == "f32fma" and v == 0.0:
                return None
            return v / (scale2 * base)
        r01 = ratio(vals["0.1"], 0.01)
        r10 = ratio(vals["10"], 100)
        r100 = ratio(vals["100"], 10000)
        rows.append((shape, r01, r10, r100))
        row_sources.append(source)
    results[route] = rows
    sources[route] = row_sources

def logr(x):
    if x is None or x <= 0:
        return None
    return math.log10(x)

print("# 主指標（判定に使用。#995 §8/§9 と同一契約: 対 f32fma の max_fail_abs_diff。")
print("# 比が定義不能〈fail=0〉の形状のみ f64 診断行 max_abs_diff で代替）")
print()
for route in ROUTES:
    print(f"== {route} ==")
    all_r = []
    for (shape, r01, r10, r100), source in zip(results[route], sources[route]):
        def f(x):
            return "n/a" if x is None else f"{x:.3f}"
        print(f"{shape} [{source}] | r(0.1)={f(r01)} r(10)={f(r10)} r(100)={f(r100)}")
        for r in (r01, r10, r100):
            if r is not None:
                all_r.append(r)
    logs = [abs(logr(r)) for r in all_r if logr(r) is not None]
    if logs:
        print(f"  n={len(all_r)} min={min(all_r):.3f} median={statistics.median(all_r):.3f} max={max(all_r):.3f} max|log10 r|={max(logs):.3f}")
    print()

with open("s2_results.json", "w") as f:
    json.dump(results, f)

# --- 診断: 一様に f64（診断行）の max_abs_diff を使った場合の参考集計 ---
# 判定には使わない（f32fma 側の max_fail_abs_diff が比較可能な形状まで
# f64 指標へ置き換えてしまうと #995 §8/§9 の契約と不整合になるため、
# あくまで補助的な参考値として区別して記録する）。
def f64_diag_series(route, shape):
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

diag_results = {}
for route in ROUTES:
    rows = []
    for shape in SHAPES_ORDER:
        vals = f64_diag_series(route, shape)
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
    diag_results[route] = rows

print("# 診断（参考値。判定には使わない）: 一様に f64 診断行 max_abs_diff を使った場合")
print()
for route in ROUTES:
    print(f"== {route} (diagnostic, f64 uniform) ==")
    all_r = []
    for shape, r01, r10, r100 in diag_results[route]:
        def f(x):
            return "n/a" if x is None else f"{x:.3f}"
        print(f"{shape} | r(0.1)={f(r01)} r(10)={f(r10)} r(100)={f(r100)}")
        for r in (r01, r10, r100):
            if r is not None:
                all_r.append(r)
    logs = [abs(logr(r)) for r in all_r if logr(r) is not None]
    if logs:
        print(f"  n={len(all_r)} min={min(all_r):.3f} median={statistics.median(all_r):.3f} max={max(all_r):.3f} max|log10 r|={max(logs):.3f}")
    print()

with open("s2_results_diag_f64.json", "w") as f:
    json.dump(diag_results, f)
