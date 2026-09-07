import json

with open("agg_f64.json") as f:
    B = json.load(f)

SHAPES_ORDER = [
    "32x32x32 (block tile)",
    "64x64x64 (block tile x2)",
    "128x128x128 (block tile x4)",
    "256x256x256 (K sweep base)",
    "512x512x512 (block tile x16)",
    "100x100x100 (non-multiple edge)",
    "64x96x128 (non-square)",
    "256x256x512 (K sweep)",
    "256x256x1024 (K sweep)",
    "256x256x4096 (K sweep, PoC-v2-5 stress)",
]
# mma系が到達可能な10形状（整列対応形状）のみで比較する（f32_simtは全15形状で値を持つが
# mma_tf32/mma_tf32x3が存在しない5形状は比較不能なため除外）。

print("| 形状 | scale | f32_simt max_abs(f64) | mma_tf32 max_abs(f64) | mma_tf32x3 max_abs(f64) | tf32x3/f32_simt | mma_tf32/tf32x3 |")
print("|---|---|---|---|---|---|---|")
ratios_vs_simt = []
ratios_vs_mma = []
for shape in SHAPES_ORDER:
    for s in ["0.1", "1", "10", "100"]:
        simt = B[f"f32_simt|{shape}|{s}"]["max_abs"]
        mma = B[f"mma_tf32|{shape}|{s}"]["max_abs"]
        x3 = B[f"mma_tf32x3|{shape}|{s}"]["max_abs"]
        r1 = x3 / simt if simt else None
        r2 = mma / x3 if x3 else None
        if r1 is not None:
            ratios_vs_simt.append(r1)
        if r2 is not None:
            ratios_vs_mma.append(r2)
        print(f"| {shape} | {s} | {simt:.3e} | {mma:.3e} | {x3:.3e} | {r1:.3f} | {r2:.3f} |")

print()
print(f"tf32x3/f32_simt: n={len(ratios_vs_simt)} max={max(ratios_vs_simt):.3f} (P2 needs <=2.0) all<=2.0: {all(r<=2.0 for r in ratios_vs_simt)}")
print(f"mma_tf32/tf32x3 (改善倍率): n={len(ratios_vs_mma)} min={min(ratios_vs_mma):.3f} (P2 needs >=10 for '1桁改善') all>=10: {all(r>=10 for r in ratios_vs_mma)}")
