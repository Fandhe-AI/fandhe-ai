#!/usr/bin/env python3
"""aggregate_h4.py の自己テスト（`python3 -I aggregate_h4_test.py`。合成 fixture のみ使い実測値は含まない）。"""
import json
import os
import sys
import tempfile

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
import aggregate_h4 as a  # noqa: E402


def write_series(d, cells, gate="pass", n=80, runs=5, drop=None, ck_override=None):
    os.makedirs(d, exist_ok=True)
    for r in range(1, runs + 1):
        with open(os.path.join(d, f"run{r}.jsonl"), "w") as f:
            for arm, phases in a.EXPECTED.items():
                for ph in phases:
                    if drop == (arm, ph):
                        continue
                    ck = "aa" if (arm, ph) in a.CHECKSUM_INSITU else (
                        "bb" if (arm, ph) in a.CHECKSUM_STANDALONE else None)
                    if ck_override and (arm, ph) == ck_override[0] and r == 2:
                        ck = ck_override[1]
                    v = cells[(arm, ph)]
                    f.write(json.dumps({"arm": arm, "phase": ph, "median_s": v, "q1_s": v, "q3_s": v,
                                        "n": n, "checksum_bits": ck, "minflt_total": 3, "core_moves": 1}) + "\n")
    with open(os.path.join(d, a.GATE_FILE), "w") as f:
        for r in range(1, 6):
            f.write(f"run{r} gate={gate}\n")


def cells(T, P, R, K, S, Kc, Zh, Zx):
    return {("insitu_direct", "device_update"): T, ("insitu_probe", "prologue_probe"): P,
            ("insitu_probe", "after_probe"): R, ("standalone", "sgd_kernel"): K,
            ("standalone", "stage"): S, ("standalone", "sgd_kernel_cold"): Kc,
            ("standalone", "zip_hot"): Zh, ("standalone", "zip_xcore"): Zx}


def expect_err(fn):
    try:
        fn()
    except a.AggError:
        return
    raise AssertionError("AggError が出ること")


def main():
    with tempfile.TemporaryDirectory() as t:
        # prologue 支持: T=170, P=120, R=50 -> H4'=170-(28+1)=141, P/H4'=85%
        base = cells(170e-6, 120e-6, 50e-6, 28e-6, 1e-6, 60e-6, 10e-6, 12e-6)
        d = os.path.join(t, "m")
        write_series(d, base)
        runs, un = a.load_series(d)
        assert un == []
        text = a.aggregate(runs)
        assert "H4a(prologue)=P" in text and "-> 支持" in text, text
        jl, res = a.judge(a.values(runs))
        assert res["verdict_a"] == "支持" and res["verdict_b"] == "未確定", res
        # カーネル超過支持 + cold 整合: T=170, P=5, R=165, K=28, S=1 -> H4b=136; x=R-S=164 vs Kc=160
        k2 = cells(170e-6, 5e-6, 165e-6, 28e-6, 1e-6, 160e-6, 10e-6, 12e-6)
        d2 = os.path.join(t, "k")
        write_series(d2, k2)
        jl, res = a.judge(a.values(a.load_series(d2)[0]))
        assert res["verdict_b"] == "支持" and res["state"].startswith("cold"), res
        # 加算性不成立 -> 両方未確定
        k3 = cells(170e-6, 5e-6, 20e-6, 28e-6, 1e-6, 160e-6, 10e-6, 12e-6)
        d3 = os.path.join(t, "n")
        write_series(d3, k3)
        jl, res = a.judge(a.values(a.load_series(d3)[0]))
        assert not res["additive"] and res["verdict_a"] == res["verdict_b"] == "未確定", res
        # single 系列（規則 5）
        ds = os.path.join(t, "s")
        write_series(ds, cells(60e-6, 3e-6, 55e-6, 28e-6, 1e-6, 60e-6, 10e-6, 12e-6))
        out = a.aggregate(a.load_series(d)[0], a.load_series(ds)[0])
        assert "マルチコア由来の状態を支持" in out, out
        out = a.aggregate(a.load_series(d)[0], a.load_series(ds)[0], (), ("run1",))
        assert "判定なし" in out
        # fail-closed 各種
        for name, kw in {
            "drop": dict(drop=("standalone", "stage")),
            "n": dict(n=79),
            "ck": dict(ck_override=(("standalone", "zip_hot"), "cc")),
            "gate": dict(gate="bogus"),
            "runs": dict(runs=4),
        }.items():
            dd = os.path.join(t, "bad_" + name)
            write_series(dd, base, **kw)
            expect_err(lambda dd=dd: a.load_series(dd))
    print("aggregate_h4 self-test: OK")


if __name__ == "__main__":
    main()
