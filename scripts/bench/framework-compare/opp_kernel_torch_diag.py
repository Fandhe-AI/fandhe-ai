#!/usr/bin/env python3
"""Mac mini で PyTorch（CPU／MPS）の `torch.mm` 純カーネル時間と Python 側 checksum を測る診断（イシュー #3179）。

## 位置づけ

`docs/perf/logs/lowlayer-diagnosis-2026-09-12/bench_py.py`（framework-compare と同一プロトコルの
Python 版）の計測窓には「matmul のディスパッチ + `.cpu().numpy()` + `checksum()`
（`np.cumsum(astype(float64))`）」が全部入っている。本スクリプトは窓を区間に分け、相手の
純カーネル時間と checksum の ns／要素を別々に測る。`parity_torch_truth.py` と同じ診断専用ツールで、
**bench_py.py・相手側ベンチ・判定規則（checksum_contract.py・compare_gemm_gate.py）は変えない**。
bench_py.py は `--bench-py` で受け取り、RULE.txt に固定した sha256 と一致したものだけを
importlib で読み込んで `fill_vec`／`checksum`／`quantile`／`WARMUP`／`ITERS` を再利用する
（定数・関数を複製しない。測る checksum が本番の関数そのものになる）。

## 呼び出し元

- Mac mini の venv-torch 上で `docs/perf/logs/opp-kernel-macmini-3179/run-macmini.sh` が `measure` を腕×run で起動し、
  最後に `aggregate` を呼ぶ。
- CI で動くのは torch／numpy 不要の部分だけ（`opp_kernel_torch_diag_test.py`）。
- GB10 の cuBLAS（#3185）は `--device cuda` を足せば同じ区間定義で拡張できる（本イシューでは実装しない）。

## サブコマンド

  measure   --bench-py PATH --device cpu|mps --shapes square|mlp|all --arm base|threads1|threads4|threads10|numpy --run K --out F.jsonl
  aggregate --runs-dir DIR --reference-py-jsonl F --out-json F --out-md F [--runs 5]
  --self-test

判定・閾値は設けない（record_only）。規則の正は docs/perf/logs/opp-kernel-macmini-3179/RULE.txt。
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

ISSUE = 3179
# RULE.txt 項 3: 0.11.0 計測で使った bench_py.py の sha256（全桁）。
BENCH_PY_SHA256 = "9fd2c08f37d66b5bb41ececab8a5b3daaa36b55609770502ed3b35ae2ce2409b"
SQUARE_SIZES = (256, 512, 1024, 2048, 4096)
# (shape_id, m, n, k, trans)。#3177 RULE 項 2 と同じ MLP 6 形状。
MLP_SHAPES = (
    ("l1_fwd", 64, 256, 784, "NN"),
    ("l1_dw", 784, 256, 64, "TN"),
    ("l1_dx", 64, 784, 256, "NT"),
    ("l2_fwd", 64, 10, 256, "NN"),
    ("l2_dw", 256, 10, 64, "TN"),
    ("l2_dx", 64, 256, 10, "NT"),
)
# MLP 形状の入力用シード（RULE 項 4。bench_py の SEED_* とは混ぜない）。
MLP_SEED_A, MLP_SEED_B = 0x3179A, 0x3179B
ARMS = ("base", "threads1", "threads4", "threads10", "numpy")
ARM_THREADS = {"threads1": 1, "threads4": 4, "threads10": 10}
# スレッド軸・numpy 腕は N=2048 と N=512 の CPU だけ（RULE 項 6・7）。
THREAD_AXIS_SIZES = (512, 2048)
PHASES = ("mm_pure", "mm_pure_batch10", "readout", "checksum", "checksum_hot", "window")
REQUIRED_KEYS = (
    "issue", "arm", "device", "shape_id", "m", "n", "k", "trans", "phase", "median_s",
    "q1_s", "q3_s", "warmup", "iters", "run", "torch_threads", "veclib_max_threads",
    "torch_version", "numpy_version", "bench_py_sha256", "checksum",
)
NUM_RUNS = 5


def shapes_for(kind):
    sq = [(f"sq{n}", n, n, n, "NN") for n in SQUARE_SIZES]
    if kind == "square":
        return sq
    if kind == "mlp":
        return list(MLP_SHAPES)
    if kind == "all":
        return sq + list(MLP_SHAPES)
    raise ValueError(f"unknown shapes: {kind}")


def ns_per_element(median_s, m, n):
    return median_s / (m * n) * 1e9


def gflops(m, n, k, median_s):
    return 2.0 * m * n * k / median_s / 1e9


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_bench_py(path):
    """bench_py.py を RULE の sha256 と照合してから読み込む（不一致は停止）。"""
    p = Path(path).resolve()
    if not p.is_file():
        raise SystemExit(f"bench_py.py が見つからない: {p}")
    got = sha256_file(p)
    if got != BENCH_PY_SHA256:
        raise SystemExit(f"bench_py.py の sha256 が RULE と不一致: {got}")
    spec = importlib.util.spec_from_file_location("bench_py_pinned", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def arm_key(r):
    """VECLIB_MAXIMUM_THREADS を持つ行は腕名に付けて別腕として扱う（base と衝突させない）。"""
    return r["arm"] if not r["veclib_max_threads"] else f"{r['arm']}+vl{r['veclib_max_threads']}"


def validate_row(row):
    """JSONL 1 行の厳格検証。欠損キー・非有限値・不正 phase は例外。"""
    for k in REQUIRED_KEYS:
        if k not in row:
            raise ValueError(f"キー欠損: {k}")
    if row["issue"] != ISSUE:
        raise ValueError("issue 不一致")
    if row["phase"] not in PHASES:
        raise ValueError(f"不明な phase: {row['phase']}")
    if row["arm"] not in ARMS:
        raise ValueError(f"不明な arm: {row['arm']}")
    for k in ("median_s", "q1_s", "q3_s", "checksum"):
        v = row[k]
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
            raise ValueError(f"非有限値: {k}={v!r}")
    if row["median_s"] <= 0:
        raise ValueError("median_s が正でない")
    if not isinstance(row["run"], int) or row["run"] < 1:
        raise ValueError("run が不正")
    return row


def sample(fn, bp, sync=None):
    """warmup + iters 回の秒数列から (median, q1, q3) を返す。"""
    for _ in range(bp.WARMUP):
        fn()
    secs = []
    for _ in range(bp.ITERS):
        if sync:
            sync()
        t = time.perf_counter()
        fn()
        if sync:
            sync()
        secs.append(time.perf_counter() - t)
    return bp.quantile(secs, 0.5), bp.quantile(secs, 0.25), bp.quantile(secs, 0.75)


def cmd_measure(args):
    bp = load_bench_py(args.bench_py)
    import numpy as np
    import torch

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"出力が既にある（上書きしない）: {out}")
    if args.arm in ARM_THREADS:
        torch.set_num_threads(ARM_THREADS[args.arm])
    if args.arm == "numpy":
        cfg = str(np.show_config(mode="dicts")) if hasattr(np, "show_config") else ""
        if args.device != "cpu" or "accelerate" not in cfg.lower():
            print("numpy 腕は Accelerate 版 numpy のときだけ測る: unavailable", file=sys.stderr)
            raise SystemExit(3)
    dev = args.device
    sync = torch.mps.synchronize if dev == "mps" else None
    sha = sha256_file(Path(args.bench_py).resolve())
    veclib = os.environ.get("VECLIB_MAXIMUM_THREADS", "")
    rows = []
    for sid, m, n, k, trans in shapes_for(args.shapes):
        if (args.arm != "base" or veclib) and (dev != "cpu" or not (sid.startswith("sq") and int(sid[2:]) in THREAD_AXIS_SIZES)):
            continue
        if sid.startswith("sq"):
            a_np = bp.fill_vec(bp.SEED_A, n * n).reshape(n, n)
            b_np = bp.fill_vec(bp.SEED_B, n * n).reshape(n, n)
        else:
            ash = (k, m) if trans[0] == "T" else (m, k)
            bsh = (n, k) if trans[1] == "T" else (k, n)
            a_np = bp.fill_vec(MLP_SEED_A, ash[0] * ash[1]).reshape(ash)
            b_np = bp.fill_vec(MLP_SEED_B, bsh[0] * bsh[1]).reshape(bsh)

        # 転置は .t() のビューで渡す（autograd の backward と同じ形。NN の連続コピーにしない）。
        if args.arm == "numpy":
            a, b = a_np, b_np
            if trans[0] == "T":
                a = a.T
            if trans[1] == "T":
                b = b.T
            mm = lambda: a @ b
            to_host = lambda c: c
        else:
            a = torch.from_numpy(a_np).to(dev)
            b = torch.from_numpy(b_np).to(dev)
            if trans[0] == "T":
                a = a.t()
            if trans[1] == "T":
                b = b.t()
            mm = lambda: torch.mm(a, b)
            to_host = lambda c: c.cpu().numpy()

        def batch10():
            for _ in range(10):
                mm()

        meas = {}
        meas["mm_pure"] = sample(mm, bp, sync)
        if args.arm != "numpy":
            meas["mm_pure_batch10"] = tuple(x / 10 for x in sample(batch10, bp, sync))
        c0 = mm()
        if sync:
            sync()
        meas["readout"] = sample(lambda: to_host(c0), bp)
        host = to_host(c0)
        # checksum: 毎回新しく実体化した配列に対する単体時間（読み出しは計時外）。
        for _ in range(bp.WARMUP):
            bp.checksum(to_host(mm()))
        secs = []
        for _ in range(bp.ITERS):
            h = to_host(mm())
            t = time.perf_counter()
            bp.checksum(h)
            secs.append(time.perf_counter() - t)
        meas["checksum"] = (bp.quantile(secs, 0.5), bp.quantile(secs, 0.25), bp.quantile(secs, 0.75))
        meas["checksum_hot"] = sample(lambda: bp.checksum(host), bp)
        meas["window"] = sample(lambda: bp.checksum(to_host(mm())), bp, sync)
        cks = bp.checksum(host)
        for ph, (med, q1, q3) in meas.items():
            rows.append(validate_row({
                "issue": ISSUE, "arm": args.arm, "device": dev, "shape_id": sid,
                "m": m, "n": n, "k": k, "trans": trans, "phase": ph,
                "median_s": med, "q1_s": q1, "q3_s": q3, "warmup": bp.WARMUP, "iters": bp.ITERS,
                "run": args.run, "torch_threads": torch.get_num_threads(), "veclib_max_threads": veclib,
                "torch_version": torch.__version__, "numpy_version": np.__version__,
                "bench_py_sha256": sha, "checksum": cks,
            }))
    with open(out, "x") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows -> {out}")


def load_runs(runs_dir, num_runs=NUM_RUNS):
    """runs_dir/run{K}/*.jsonl を全読みして検証する。run の欠け・重複は例外。"""
    rows = []
    for f in sorted(Path(runs_dir).glob("run*/*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                rows.append(validate_row(json.loads(line)))
    groups = {}
    for r in rows:
        key = (arm_key(r), r["device"], r["shape_id"], r["phase"])
        groups.setdefault(key, []).append(r["run"])
    if not groups:
        raise ValueError("行が 1 つも無い")
    want = list(range(1, num_runs + 1))
    for key, runs in groups.items():
        if sorted(runs) != want:
            raise ValueError(f"run 不完備または重複: {key} runs={sorted(runs)}")
    return rows


def aggregate_rows(rows, reference, checksums_match):
    """5 run の中央値・Σ区間/window の検算・系列 B との比・checksum 一致を返す（不一致は例外）。

    reference: {(device, size): {"median_s": .., "checksum": ..}}（device は cpu／mps）。
    checksums_match: checksum_contract.checksums_match（定数を複製しない）。
    """
    by = {}
    for r in rows:
        by.setdefault((arm_key(r), r["device"], r["shape_id"], r["phase"]), {})[r["run"]] = r
    med = {}
    for key, runs in by.items():
        first = next(iter(runs.values()))
        med[key] = {
            "median_s": statistics.median(x["median_s"] for x in runs.values()),
            "per_run": {k: runs[k]["median_s"] for k in sorted(runs)},
            "m": first["m"], "n": first["n"], "k": first["k"],
        }
    out = []
    for (arm, dev, sid, ph), v in sorted(med.items()):
        e = {"arm": arm, "device": dev, "shape_id": sid, "phase": ph, "median_s": v["median_s"],
             "per_run": v["per_run"]}
        if ph == "checksum":
            e["ns_per_element"] = ns_per_element(v["median_s"], v["m"], v["n"])
        if ph in ("mm_pure", "mm_pure_batch10", "window"):
            e["gflops"] = gflops(v["m"], v["n"], v["k"], v["median_s"])
        if ph == "window":
            parts = [med.get((arm, dev, sid, p)) for p in ("mm_pure", "readout", "checksum")]
            if all(parts):
                e["sum_over_window"] = sum(p["median_s"] for p in parts) / v["median_s"]
            if sid.startswith("sq") and arm == "base":  # arm_key は VECLIB 指定腕を含まない
                ref = reference.get((dev, int(sid[2:])))
                if ref is None:
                    raise ValueError(f"系列 B に基準行が無い: {dev} {sid}")
                e["ratio_to_series_b"] = v["median_s"] / ref["median_s"]
                for r in rows:
                    if (arm_key(r), r["device"], r["shape_id"]) == (arm, dev, sid) and not checksums_match(
                        r["checksum"], ref["checksum"]
                    ):
                        raise ValueError(
                            f"checksum が系列 B と不一致: {dev} {sid} {r['checksum']} vs {ref['checksum']}"
                        )
        out.append(e)
    return out


def render_md(entries):
    lines = ["# #3179 集計（5 run 中央値。run 別も併記）", "",
             "| arm | device | shape | phase | median_s | ns/要素 | GFLOPS | Σ区間/window | window/系列B | run 別 |",
             "|---|---|---|---|---|---|---|---|---|---|"]

    def f(x, p):
        return "-" if x is None else format(x, p)

    for e in entries:
        runs = ", ".join(f"r{k}={v:.3e}" for k, v in e["per_run"].items())
        lines.append(
            f"| {e['arm']} | {e['device']} | {e['shape_id']} | {e['phase']} | {e['median_s']:.4e} | "
            f"{f(e.get('ns_per_element'), '.3f')} | {f(e.get('gflops'), '.1f')} | "
            f"{f(e.get('sum_over_window'), '.3f')} | {f(e.get('ratio_to_series_b'), '.3f')} | {runs} |"
        )
    return "\n".join(lines) + "\n"


def load_reference(path):
    """系列 B の pytorch gemm 行を {(device, size): row} に読む（metal は mps として扱う）。"""
    ref = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("framework") == "pytorch" and r.get("task") == "gemm":
            dev = "mps" if r["device"] == "metal" else r["device"]
            ref[(dev, int(r["size"]))] = {"median_s": float(r["median_s"]), "checksum": float(r["checksum"])}
    return ref


def load_checksums_match():
    p = Path(__file__).with_name("checksum_contract.py")
    spec = importlib.util.spec_from_file_location("checksum_contract", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.checksums_match


def cmd_aggregate(args):
    rows = load_runs(args.runs_dir, args.runs)
    entries = aggregate_rows(rows, load_reference(args.reference_py_jsonl), load_checksums_match())
    for p in (args.out_json, args.out_md):
        if Path(p).exists():
            raise SystemExit(f"出力が既にある（上書きしない）: {p}")
    Path(args.out_json).write_text(json.dumps(entries, indent=1) + "\n")
    Path(args.out_md).write_text(render_md(entries))
    print(f"aggregate ok: {len(entries)} entries")


def self_test():
    assert len(shapes_for("all")) == 11 and len(MLP_SHAPES) == 6
    assert abs(ns_per_element(3.5e-9 * 65536, 256, 256) - 3.5) < 1e-9
    assert abs(gflops(2, 2, 2, 16e-9) - 1.0) < 1e-9
    base = dict(issue=ISSUE, arm="base", device="cpu", shape_id="sq256", m=256, n=256, k=256, trans="NN",
                phase="mm_pure", median_s=1e-3, q1_s=1e-3, q3_s=1e-3, warmup=20, iters=20, run=1,
                torch_threads=10, veclib_max_threads="", torch_version="x", numpy_version="y",
                bench_py_sha256="z", checksum=1.0)
    validate_row(dict(base))
    for bad in ({"median_s": float("nan")}, {"phase": "nope"}):
        try:
            validate_row({**base, **bad})
        except ValueError:
            continue
        raise AssertionError("不正行を拒否していない")
    print("self-test ok")


def main(argv=None):
    ap = argparse.ArgumentParser(description="PyTorch 純カーネル時間と checksum の診断（#3179）")
    ap.add_argument("--self-test", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    m = sub.add_parser("measure")
    m.add_argument("--bench-py", required=True)
    m.add_argument("--device", choices=("cpu", "mps"), required=True)
    m.add_argument("--shapes", choices=("square", "mlp", "all"), default="all")
    m.add_argument("--arm", choices=ARMS, default="base")
    m.add_argument("--run", type=int, required=True)
    m.add_argument("--out", required=True)
    a = sub.add_parser("aggregate")
    a.add_argument("--runs-dir", required=True)
    a.add_argument("--reference-py-jsonl", required=True)
    a.add_argument("--out-json", required=True)
    a.add_argument("--out-md", required=True)
    a.add_argument("--runs", type=int, default=NUM_RUNS)
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    if args.cmd == "measure":
        return cmd_measure(args)
    if args.cmd == "aggregate":
        return cmd_aggregate(args)
    ap.error("サブコマンドか --self-test が必要")


if __name__ == "__main__":
    main()
