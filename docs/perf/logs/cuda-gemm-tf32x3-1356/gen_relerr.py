import json

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

for route in ROUTES:
    print(f"== {route} ==")
    ratios = []
    for shape in SHAPES_ORDER:
        key1 = f"{route}|{shape}|1"
        if key1 not in B:
            continue
        base = B[key1]["max_rel"]
        if not base:
            continue
        vals = {}
        for s in ["0.1", "10", "100"]:
            k = f"{route}|{shape}|{s}"
            vals[s] = B[k]["max_rel"] / base if B[k]["max_rel"] is not None else None
        print(f"{shape} | s=0.1:{vals['0.1']:.3f} s=10:{vals['10']:.3f} s=100:{vals['100']:.3f}")
        ratios += [v for v in vals.values() if v is not None]
    print(f"  n={len(ratios)} min={min(ratios):.3f} median={sorted(ratios)[len(ratios)//2]:.3f} max={max(ratios):.3f}")
    print()
