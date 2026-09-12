import json, glob, statistics, re, sys

outdir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/phase1-1575-out"

cells = [
    ("train", "fresh", ""),
    ("train", "reuse", ""),
    ("infer", "fresh", ""),
    ("infer", "reuse", ""),
    ("gemm", "fresh", "512"),
    ("gemm", "reuse", "512"),
    ("gemm", "fresh", "1024"),
    ("gemm", "reuse", "1024"),
    ("gemm", "fresh", "2048"),
    ("gemm", "reuse", "2048"),
]

judged = {("train", "fresh", ""), ("train", "reuse", ""), ("infer", "fresh", ""), ("infer", "reuse", "")}

print(f"{'cell':30s} {'before_med_s':>14s} {'after_med_s':>14s} {'ratio(after/before)':>20s} {'checksum_match':>15s} {'judged':>8s}")
all_ok = True
for task, mode, size in cells:
    before_meds, after_meds = [], []
    before_cs, after_cs = set(), set()
    for arm, meds, css in (("before", before_meds, before_cs), ("after", after_meds, after_cs)):
        for i in range(1, 6):
            path = f"{outdir}/{task}_{mode}_{size}_{arm}_run{i}.jsonl"
            with open(path) as f:
                lines = [l for l in f if l.strip()]
            rec = json.loads(lines[-1])
            meds.append(rec["median_s"])
            css.add(round(rec["checksum"], 6))
    before_med = statistics.median(before_meds)
    after_med = statistics.median(after_meds)
    ratio = after_med / before_med
    checksum_match = (before_cs | after_cs)
    match_ok = len(checksum_match) == 1
    if not match_ok:
        all_ok = False
    label = f"{task}:{mode}" + (f":{size}" if size else "")
    is_judged = (task, mode, size) in judged
    print(f"{label:30s} {before_med:14.9f} {after_med:14.9f} {ratio:20.4f} {'OK' if match_ok else 'MISMATCH':>15s} {'yes' if is_judged else 'no':>8s}")

print()
print(f"checksum全体一致: {'OK' if all_ok else 'MISMATCH'}")
