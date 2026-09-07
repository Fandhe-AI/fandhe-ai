import json

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
SCALES = ["0.1", "1", "10", "100"]
ROUTES = ["f32_simt", "mma_tf32", "mma_tf32x3"]

def fmt(v):
    if v is None:
        return "n/a"
    if v == float("inf"):
        return "inf"
    return f"{v:.3e}"

def gen(ref_label, jsonfile, outfile):
    with open(jsonfile) as f:
        data = json.load(f)
    with open(outfile, "w") as out:
        for route in ROUTES:
            out.write(f"### {ref_label} 対 {route}\n\n")
            out.write("| 形状 | scale | fail/total（5 シード） | max_abs_diff | max_rel_err | max_fail_abs_diff |\n")
            out.write("|---|---|---|---|---|---|\n")
            for shape in SHAPES_ORDER:
                for scale in SCALES:
                    key = f"{route}|{shape}|{scale}"
                    if key not in data:
                        out.write(f"| {shape} | {scale} | (skipped: alignment n%4/k%4) | - | - | - |\n")
                        continue
                    d = data[key]
                    out.write(f"| {shape} | {scale} | {d['fail']}/{d['total']} | {fmt(d['max_abs'])} | {fmt(d['max_rel'])} | {fmt(d['max_fail_abs'])} |\n")
            out.write("\n")

gen("f32fma（REQ-2 判定行）", "agg_f32fma.json", "table_f32fma.md")
gen("f64（診断行）", "agg_f64.json", "table_f64.md")
print("done")
