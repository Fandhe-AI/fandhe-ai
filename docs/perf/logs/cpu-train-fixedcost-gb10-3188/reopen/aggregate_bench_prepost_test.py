#!/usr/bin/env python3
"""aggregate_bench_prepost.py の自己テスト（`python3 -I aggregate_bench_prepost_test.py`。合成 fixture のみ）。"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aggregate_bench_prepost as a  # noqa: E402

EXP = a.EXPECTED_PHASES
PHASES = list(EXP["train_reuse_phases"])


def write(d, drop=None, bad_phase_set=False, missing_all=False):
    for l in a.LABELS:
        for r in range(1, a.ROUNDS + 1):
            rd = os.path.join(d, l, f"round{r}")
            os.makedirs(rd)
            for c in a.CELLS:
                if drop == (l, r, c):
                    continue
                with open(os.path.join(rd, c + ".jsonl"), "w") as f:
                    base = 100e-6 * (2 if l == "pre" else 1) * (1 + 0.01 * r)
                    if c.endswith("_phases"):
                        phs = list(EXP[c])
                        if missing_all and c == "train_reuse_phases":
                            phs = [p for p in phs if p != "device_update"]
                        phs = phs if not (bad_phase_set and l == "head" and r == 3) else ["x"]
                        for p in phs:
                            f.write(json.dumps({"median_s": base, "checksum": 1.5, "phase": p}) + "\n")
                    else:
                        f.write(json.dumps({"median_s": base, "checksum": 1.5}) + "\n")
    with open(os.path.join(d, "load_gate_status.txt"), "w") as f:
        for r in range(1, 6):
            f.write(f"round{r} gate=pass\n")


def expect_err(fn):
    try:
        fn()
    except a.AggError:
        return
    raise AssertionError("AggError が出ること")


with tempfile.TemporaryDirectory() as t:
    ok = os.path.join(t, "ok")
    write(ok)
    text = a.aggregate(*a.load_all(ok))
    assert "0.500" in text and "一致" in text and "reuse/fresh" in text, text
    miss = os.path.join(t, "miss")
    write(miss, drop=("head", 2, "infer_fresh_total"))
    expect_err(lambda: a.load_all(miss))
    bad = os.path.join(t, "bad")
    write(bad, bad_phase_set=True)
    expect_err(lambda: a.load_all(bad))
    miss_all = os.path.join(t, "miss_all")
    write(miss_all, missing_all=True)
    expect_err(lambda: a.load_all(miss_all))
print("aggregate_bench_prepost self-test: OK")
