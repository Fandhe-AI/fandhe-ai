import math, json
from collections import defaultdict

def parse_file(path):
    rows = []
    route = None
    with open(path) as f:
        for line in f:
            if line.startswith("## f32 SIMT"):
                route = "f32_simt"; continue
            if line.startswith("## TF32 mma.sync"):
                route = "mma_tf32"; continue
            if line.startswith("## 3xTF32"):
                route = "mma_tf32x3"; continue
            if not line.startswith("| ") or line.startswith("| scale |") or line.startswith("|---"):
                continue
            cols = [c.strip() for c in line.strip("|\n").split("|")]
            if len(cols) != 17:
                continue
            scale, shape, seed, ref_label = cols[0], cols[1], cols[2], cols[3]
            failtotal = cols[4]
            if failtotal.startswith("("):
                continue
            fail, total = map(int, failtotal.split("/"))
            def pf(x):
                x = x.strip()
                return None if x in ("n/a", "NaN") else float(x)
            rows.append(dict(route=route, scale=scale, shape=shape, ref=ref_label,
                fail=fail, total=total, max_abs=pf(cols[5]),
                max_rel=pf(cols[7]), max_fail_abs=pf(cols[12])))
    return rows

def dedupe_256(rows):
    return [r for r in rows if r["shape"] != "256x256x256 (block tile x8)"]

def agg_max(values):
    finite = [v for v in values if v is not None]
    if not finite:
        return None
    if any(math.isinf(v) for v in finite):
        return float("inf")
    return max(finite)

rows = dedupe_256(parse_file("probe-mma-run1.md"))

def build_table(ref):
    groups = defaultdict(list)
    for r in rows:
        if r["ref"] != ref: continue
        groups[(r["route"], r["shape"], r["scale"])].append(r)
    out = {}
    for key, entries in groups.items():
        fail = sum(e["fail"] for e in entries)
        total = sum(e["total"] for e in entries)
        max_abs = agg_max(e["max_abs"] for e in entries)
        max_rel = agg_max(e["max_rel"] for e in entries)
        max_fail_abs = agg_max(e["max_fail_abs"] for e in entries)
        out[key] = dict(fail=fail, total=total, max_abs=max_abs, max_rel=max_rel, max_fail_abs=max_fail_abs, n=len(entries))
    return out

t_f32fma = build_table("f32fma")
t_f64 = build_table("f64")

with open("agg_f32fma.json","w") as f:
    json.dump({f"{k[0]}|{k[1]}|{k[2]}": v for k,v in t_f32fma.items()}, f)
with open("agg_f64.json","w") as f:
    json.dump({f"{k[0]}|{k[1]}|{k[2]}": v for k,v in t_f64.items()}, f)

print("f32fma groups:", len(t_f32fma))
print("f64 groups:", len(t_f64))
print(set(k[0] for k in t_f32fma))
print(sorted(set(k[1] for k in t_f32fma)))
