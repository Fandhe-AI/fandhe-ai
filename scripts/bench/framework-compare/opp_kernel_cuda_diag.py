#!/usr/bin/env python3
"""GB10 で PyTorch（cuBLAS）の `torch.mm` 純カーネル時間（FP32・TF32）と Python 側 checksum を測る診断（イシュー #3185）。

## 位置づけ

兄弟 #3179（Mac mini 版 `opp_kernel_torch_diag.py`）と同じ診断専用ツール。`bench_py.py` の計測窓
（matmul + `.cpu().numpy()` + `checksum()`）を区間に分け、相手（PyTorch の cuBLAS SGEMM）の純カーネル時間と
checksum の ns／要素を別々に測る。fandhe 側は HEAD の既存 `#[ignore]` 診断テスト
（`gemm_reuse_phase_diag_select`・`gemm_small_launch_cost_diag`。Rust のコードは足さない）の出力ログを
`parse-fandhe` で読み、時計（wall／CUDA event）ごとに比を並べる。
**bench_py.py・相手側ベンチ・判定規則（checksum_contract.py・compare_gemm_gate.py）・#3179 のツール／規則は変えない**。
#3179 のモジュールは importlib で読み込んで `load_bench_py`／`sample`／`ns_per_element`／`gflops`／
`load_checksums_match`／`sha256_file` を再利用する（定数・関数を複製しない。#3179 側は拡張もしない）。

## 呼び出し元

- GB10 の `~/work/.venv-bench` 上で `docs/perf/logs/opp-kernel-gb10-3185/run-gb10.sh` が各サブコマンドを起動する。
- CI で動くのは torch／numpy 不要の部分だけ（`opp_kernel_cuda_diag_test.py`）。

## サブコマンド

  measure         --bench-py PATH --arm fp32|tf32|cpu --run K --out F.jsonl
  cpu-blas-probe  --out-dir DIR
  parse-fandhe    --log F --kind layerb|diag2109 --run K --out F.jsonl
  aggregate       --runs-dir DIR --reference-py-jsonl F --out-json F --out-md F [--runs 5]
  --self-test

判定・閾値は設けない（record_only）。規則の正は docs/perf/logs/opp-kernel-gb10-3185/RULE.txt。
"""
import argparse
import importlib.util
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path


def _load_base():
    p = Path(__file__).with_name("opp_kernel_torch_diag.py")
    spec = importlib.util.spec_from_file_location("opp_kernel_torch_diag_base", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


D = _load_base()

ISSUE = 3185
# RULE 項 2: 0.11.0 計測と同じ torch。違えば計測を止める（fail-closed）。
TORCH_VERSION = "2.14.0+cu130"
SIZES = D.SQUARE_SIZES
ARMS = ("fp32", "tf32", "cpu")
ARM_DEVICE = {"fp32": "cuda", "tf32": "cuda", "cpu": "cpu"}
EVENT_PHASES = ("mm_event", "mm_event_batch10")
PHASES = ("mm_pure", "mm_event", "mm_event_batch10", "readout", "checksum", "checksum_hot", "window")
REQUIRED_KEYS = (
    "issue", "arm", "device", "shape_id", "n", "phase", "median_s", "q1_s", "q3_s", "warmup", "iters",
    "run", "allow_tf32", "fp32_matmul_precision", "fp32_precision", "torch_version", "cuda_version",
    "numpy_version", "bench_py_sha256", "checksum", "gpu_name",
)
NUM_RUNS = 5
# Issue 本文の推定値（逆算。範囲 2.4〜3.0）。実測との差を記録するための参照値で、判定には使わない。
CHECKSUM_NS_ESTIMATE = 2.7
# 公称ピーク（RULE 項 15。FP32 約 27〜31 TFLOPS・TF32 は第三者実測 53.3）。到達率を並べるだけ。
PEAK_FP32_TFLOPS = 31.0
PEAK_TF32_TFLOPS = 53.3


def validate_row(row):
    """JSONL 1 行の厳格検証。欠損キー・非有限値・不正な腕／phase は例外。"""
    for k in REQUIRED_KEYS:
        if k not in row:
            raise ValueError(f"キー欠損: {k}")
    if row["issue"] != ISSUE:
        raise ValueError("issue 不一致")
    if row["arm"] not in ARMS:
        raise ValueError(f"不明な arm: {row['arm']}")
    if row["device"] != ARM_DEVICE[row["arm"]]:
        raise ValueError(f"arm と device が不整合: {row['arm']} {row['device']}")
    if row["phase"] not in PHASES:
        raise ValueError(f"不明な phase: {row['phase']}")
    if row["arm"] == "cpu" and row["phase"] in EVENT_PHASES:
        raise ValueError("cpu 腕は event 区間を持たない")
    if row["arm"] != "cpu" and row["allow_tf32"] != (row["arm"] == "tf32"):
        raise ValueError("allow_tf32 の記録が腕と食い違う")
    for k in ("median_s", "q1_s", "q3_s", "checksum"):
        v = row[k]
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
            raise ValueError(f"非有限値: {k}={v!r}")
    if row["median_s"] <= 0:
        raise ValueError("median_s が正でない")
    if not isinstance(row["run"], int) or isinstance(row["run"], bool) or row["run"] < 1:
        raise ValueError("run が不正")
    if not isinstance(row["n"], int) or row["n"] not in SIZES:
        raise ValueError("n が不正")
    return row


def check_tf32_readback(arm, allow_tf32):
    """読み戻した allow_tf32 が腕と食い違えば例外（RULE 項 12）。cpu 腕は検査しない。"""
    if arm == "cpu":
        return
    if bool(allow_tf32) != (arm == "tf32"):
        raise RuntimeError(f"allow_tf32 の読み戻し {allow_tf32!r} が腕 {arm} と食い違う")


def event_sample(torch, mm, bp, reps=1):
    """CUDA event 対で `reps` 回の mm を挟んだ時間（秒）の (median, q1, q3) を `reps` で割って返す。"""
    for _ in range(bp.WARMUP):
        mm()
        torch.cuda.synchronize()
    # event 対はループ外で作り、両方を一度 record・同期して遅延初期化を計測前に済ませて再利用する
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    e.record()
    torch.cuda.synchronize()
    secs = []
    for _ in range(bp.ITERS):
        torch.cuda.synchronize()
        s.record()
        for _ in range(reps):
            mm()
        e.record()
        torch.cuda.synchronize()
        secs.append(s.elapsed_time(e) * 1e-3 / reps)
    return bp.quantile(secs, 0.5), bp.quantile(secs, 0.25), bp.quantile(secs, 0.75)


def cmd_measure(args):
    bp = D.load_bench_py(args.bench_py)
    import numpy as np
    import torch

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"出力が既にある（上書きしない）: {out}")
    if torch.__version__ != TORCH_VERSION:
        raise SystemExit(f"torch の版が RULE 項 2 と不一致: {torch.__version__}")
    dev = ARM_DEVICE[args.arm]
    if dev == "cuda" and not torch.cuda.is_available():
        print("CUDA が使えない: unavailable", file=sys.stderr)
        raise SystemExit(3)
    allow = False
    precision = fp32_prec = ""
    gpu = ""
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = args.arm == "tf32"
        allow = bool(torch.backends.cuda.matmul.allow_tf32)
        check_tf32_readback(args.arm, allow)
        gpu = torch.cuda.get_device_name(0)
    precision = str(torch.get_float32_matmul_precision())
    fp32_prec = str(getattr(torch.backends.cuda.matmul, "fp32_precision", ""))
    sync = torch.cuda.synchronize if dev == "cuda" else None
    sha = D.sha256_file(Path(args.bench_py).resolve())
    rows = []
    for n in SIZES:
        a_np = bp.fill_vec(bp.SEED_A, n * n).reshape(n, n)
        b_np = bp.fill_vec(bp.SEED_B, n * n).reshape(n, n)
        a = torch.from_numpy(a_np).to(dev)
        b = torch.from_numpy(b_np).to(dev)
        mm = lambda: torch.mm(a, b)  # noqa: E731
        to_host = lambda c: c.cpu().numpy()  # noqa: E731
        meas = {"mm_pure": D.sample(mm, bp, sync)}
        if dev == "cuda":
            meas["mm_event"] = event_sample(torch, mm, bp)
            meas["mm_event_batch10"] = event_sample(torch, mm, bp, reps=10)
        c0 = mm()
        if sync:
            sync()
        meas["readout"] = D.sample(lambda: to_host(c0), bp)
        host = to_host(c0)
        for _ in range(bp.WARMUP):
            bp.checksum(to_host(mm()))
        secs = []
        for _ in range(bp.ITERS):
            h = to_host(mm())
            t = time.perf_counter()
            bp.checksum(h)
            secs.append(time.perf_counter() - t)
        meas["checksum"] = (bp.quantile(secs, 0.5), bp.quantile(secs, 0.25), bp.quantile(secs, 0.75))
        meas["checksum_hot"] = D.sample(lambda: bp.checksum(host), bp)
        # window は bench_py.py の計測区間と同じにそろえる（追加の synchronize を挟まない）。
        meas["window"] = D.sample(lambda: bp.checksum(to_host(mm())), bp)
        cks = bp.checksum(host)
        for ph, (med, q1, q3) in meas.items():
            rows.append(validate_row({
                "issue": ISSUE, "arm": args.arm, "device": dev, "shape_id": f"sq{n}", "n": n, "phase": ph,
                "median_s": med, "q1_s": q1, "q3_s": q3, "warmup": bp.WARMUP, "iters": bp.ITERS,
                "run": args.run, "allow_tf32": allow, "fp32_matmul_precision": precision,
                "fp32_precision": fp32_prec, "torch_version": torch.__version__,
                "cuda_version": str(torch.version.cuda), "numpy_version": np.__version__,
                "bench_py_sha256": sha, "checksum": cks, "gpu_name": gpu,
            }))
    with open(out, "x") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows -> {out}")


# ---- CPU 側 BLAS 経路の確認（RULE 項 13） ----

BLAS_KEYWORDS = ("nvpl", "openblas", "mkl", "armpl", "arm_compute", "dnnl")
# 子プロセスで実行する固定コード（外部入力を組み込まない）。
_PROBE_CODE = r"""
import json, torch
a = torch.ones(1024, 1024); b = torch.ones(1024, 1024)
c = torch.mm(a, b)
maps = open('/proc/self/maps').read()
print(json.dumps({'config': torch.__config__.show(),
                  'mkldnn_available': bool(torch.backends.mkldnn.is_available()),
                  'maps': maps, 'c00': float(c[0, 0])}))
"""


def blas_libs_from_maps(maps_text):
    """/proc/self/maps から BLAS 系共有ライブラリの **ファイル名部分だけ**を重複なしで返す（パスは落とす）。"""
    names = set()
    for line in maps_text.splitlines():
        parts = line.split(None, 5)
        if len(parts) < 6:
            continue
        base = parts[5].rsplit("/", 1)[-1]
        if ".so" in base and any(k in base.lower() for k in BLAS_KEYWORDS):
            names.add(base)
    return sorted(names)


def judge_blas(stderr_text, libs):
    """実行イベント行（`*_verbose,...,exec,...`）だけを根拠に「確認できた／確認できなかった」を返す（RULE 項 13）。

    共有ライブラリのロード（libs）と verbose の初期化情報（`,info,` 行）は matmul を呼んだ証拠ではないため、
    参考情報（libs・info_lines）として残すだけで判定には使わない。出力が無いことから経路を推測しない。
    """
    verbose = [ln for ln in stderr_text.splitlines() if re.match(r"^(onednn|dnnl)_verbose", ln, re.I)]
    evidence = [ln for ln in verbose if "exec" in ln.lower().split(",")]
    info_lines = [ln for ln in verbose if ln not in evidence]
    verdict = "確認できた" if evidence else "確認できなかった"
    return {"verdict": verdict, "evidence_lines": evidence[:5], "info_lines": info_lines[:5], "libs": list(libs)}


def cmd_cpu_blas_probe(args):
    out_dir = Path(args.out_dir)
    if out_dir.exists():
        raise SystemExit(f"出力先が既にある（上書きしない）: {out_dir}")
    out_dir.mkdir(parents=True)
    env = {**os.environ, "DNNL_VERBOSE": "1", "ONEDNN_VERBOSE": "1"}
    p = subprocess.run([sys.executable, "-c", _PROBE_CODE], env=env, capture_output=True, text=True, timeout=600)
    (out_dir / "cpu_blas_stderr.txt").write_text(p.stderr)
    if p.returncode != 0:
        raise SystemExit(f"probe が失敗: rc={p.returncode}")
    info = json.loads(p.stdout.strip().splitlines()[-1])
    libs = blas_libs_from_maps(info["maps"])
    verdict = judge_blas(p.stderr + "\n" + p.stdout.replace(info["maps"], ""), libs)
    report = {"mkldnn_available": info["mkldnn_available"], "libs": libs, "judge": verdict}
    (out_dir / "cpu_blas_config.txt").write_text(info["config"])
    (out_dir / "cpu_blas_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(f"cpu-blas-probe: {verdict['verdict']} libs={libs}")


# ---- fandhe 側ログのパーサー（RULE 項 7） ----

TEST_OK = re.compile(r"^test result: ok\.")
# --test-threads=1 --nocapture では最初のヘッダが libtest 接頭辞（"test <path> ... "）と同一行に出る
# （docs/perf/logs/cuda-gemm-reuse-phase-1182/layerB-run1.log:186）。接頭辞だけを限定的に許容する。
LAYERB_HEAD = re.compile(r"^(?:test \S+ \.\.\. )?  N=(\d+) kernel=(\w+) \(median over")
LAYERB_LINE = re.compile(
    r"^    (launch_issue|kernel_wait): median=([0-9.]+) ms  q1=([0-9.]+) ms  q3=([0-9.]+) ms\s*$")
DIAG_LINE = re.compile(
    r"^DIAG2109 n=(\d+) layer=(\S+) phase=(\S+) median_us=([0-9.]+) q1_us=([0-9.]+) q3_us=([0-9.]+)\s*$")
LAYERB_SIZES = (1024, 2048, 4096)
DIAG_SIZES = (128, 256, 512)
# (layer, phase, clock)。L2S kernel_wait は wall、D 層は CUDA event 時計。
DIAG_WANTED = {("L2S", "kernel_wait"): "wall", ("D", "dev_kernel_seg"): "event", ("D", "dev_kernel_b2b"): "event"}


def _frow(kind, n, layer, phase, clock, med, q1, q3, run):
    return {"issue": ISSUE, "source": "fandhe", "kind": kind, "n": n, "layer": layer, "phase": phase,
            "clock": clock, "median_s": med, "q1_s": q1, "q3_s": q3, "run": run}


def parse_layerb(text, run):
    """Layer B ログから kernel=Select ブロックの launch_issue／kernel_wait を読む（重複・欠落・test ok 不在は例外）。"""
    lines = text.splitlines()
    if not any(TEST_OK.match(ln) for ln in lines):
        raise ValueError("test result: ok の行が無い（無効 run）")
    rows, seen, cur = [], set(), None
    for ln in lines:
        h = LAYERB_HEAD.match(ln)
        if h:
            cur = (int(h.group(1)), h.group(2))
            continue
        m = LAYERB_LINE.match(ln)
        if m and cur and cur[1] == "Select":
            n = cur[0]
            key = (n, m.group(1))
            if key in seen:
                raise ValueError(f"重複: {key}")
            seen.add(key)
            rows.append(_frow("layerb", n, "B", m.group(1), "wall",
                              float(m.group(2)) * 1e-3, float(m.group(3)) * 1e-3, float(m.group(4)) * 1e-3, run))
    want = {(n, p) for n in LAYERB_SIZES for p in ("launch_issue", "kernel_wait")}
    if seen != want:
        raise ValueError(f"Layer B の区間が過不足: 欠落={sorted(want - seen)} 想定外={sorted(seen - want)}")
    return rows


def parse_diag2109(text, run):
    """DIAG2109 ログから必要な (n, layer, phase) を読む（重複・欠落・test ok 不在は例外）。"""
    lines = text.splitlines()
    if not any(TEST_OK.match(ln) for ln in lines):
        raise ValueError("test result: ok の行が無い（無効 run）")
    rows, seen = [], set()
    for ln in lines:
        m = DIAG_LINE.match(ln)
        if not m:
            continue
        n, layer, phase = int(m.group(1)), m.group(2), m.group(3)
        if (layer, phase) not in DIAG_WANTED or n not in DIAG_SIZES:
            continue
        if (n, layer, phase) in seen:
            raise ValueError(f"重複: {(n, layer, phase)}")
        seen.add((n, layer, phase))
        rows.append(_frow("diag2109", n, layer, phase, DIAG_WANTED[(layer, phase)],
                          float(m.group(4)) * 1e-6, float(m.group(5)) * 1e-6, float(m.group(6)) * 1e-6, run))
    want = {(n, ly, ph) for n in DIAG_SIZES for (ly, ph) in DIAG_WANTED}
    if seen != want:
        raise ValueError(f"DIAG2109 の区間が過不足: 欠落={sorted(want - seen)} 想定外={sorted(seen - want)}")
    return rows


def cmd_parse_fandhe(args):
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"出力が既にある（上書きしない）: {out}")
    text = Path(args.log).read_text()
    rows = (parse_layerb if args.kind == "layerb" else parse_diag2109)(text, args.run)
    with open(out, "x") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows -> {out}")


# ---- 集計 ----

def load_py_runs(runs_dir, num_runs=NUM_RUNS):
    """runs_dir/run{K}/*.jsonl を全読みして検証し、run の欠け・重複を例外にする。"""
    rows = []
    for f in sorted(Path(runs_dir).glob("run*/*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                rows.append(validate_row(json.loads(line)))
    if not rows:
        raise ValueError("行が 1 つも無い")
    groups = {}
    for r in rows:
        groups.setdefault((r["arm"], r["n"], r["phase"]), []).append(r["run"])
    want = list(range(1, num_runs + 1))
    for key, runs in groups.items():
        if sorted(runs) != want:
            raise ValueError(f"run 不完備または重複: {key} runs={sorted(runs)}")
    return rows


def load_fandhe_runs(runs_dir, num_runs=NUM_RUNS):
    rows = []
    for f in sorted((Path(runs_dir) / "fandhe_rows").glob("*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                for k in ("n", "layer", "phase", "clock", "run"):
                    if k not in r:
                        raise ValueError(f"fandhe 行のキー欠損: {k}")
                for k in ("median_s", "q1_s", "q3_s"):
                    if not isinstance(r.get(k), (int, float)) or not math.isfinite(r[k]) or r[k] <= 0:
                        raise ValueError(f"fandhe 行の値が不正: {k}")
                rows.append(r)
    groups = {}
    for r in rows:
        groups.setdefault((r["n"], r["layer"], r["phase"]), []).append(r["run"])
    want = list(range(1, num_runs + 1))
    for key, runs in groups.items():
        if sorted(runs) != want:
            raise ValueError(f"fandhe の run 不完備または重複: {key} runs={sorted(runs)}")
    return rows


def expected_py_groups():
    want = set()
    for arm in ARMS:
        for n in SIZES:
            for ph in PHASES:
                if arm == "cpu" and ph in EVENT_PHASES:
                    continue
                want.add((arm, n, ph))
    return want


def expected_fandhe_groups():
    want = {(n, "B", p) for n in LAYERB_SIZES for p in ("launch_issue", "kernel_wait")}
    want |= {(n, ly, ph) for n in DIAG_SIZES for (ly, ph) in DIAG_WANTED}
    return want


def check_complete(py_rows, fandhe_rows):
    """必須グループの欠落・想定外を検出する（腕ファイル丸ごと・fandhe 側の欠落を含む）。"""
    have = {(r["arm"], r["n"], r["phase"]) for r in py_rows}
    missing, extra = sorted(expected_py_groups() - have), sorted(have - expected_py_groups())
    if missing:
        raise ValueError(f"必須グループが欠落: {len(missing)} 件 例={missing[:3]}")
    if extra:
        raise ValueError(f"想定外のグループ: {len(extra)} 件 例={extra[:3]}")
    fh = {(r["n"], r["layer"], r["phase"]) for r in fandhe_rows}
    fm, fe = sorted(expected_fandhe_groups() - fh), sorted(fh - expected_fandhe_groups())
    if fm:
        raise ValueError(f"fandhe 側の必須区間が欠落: {len(fm)} 件 例={fm[:3]}")
    if fe:
        raise ValueError(f"fandhe 側の想定外の区間: {len(fe)} 件 例={fe[:3]}")


def load_reference(path):
    """系列の pytorch gemm 行から {(device, size): checksum} を読む。同キーで checksum が食い違えば例外。"""
    ref = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("framework") == "pytorch" and r.get("task") == "gemm":
            key, ck = (r["device"], int(r["size"])), float(r["checksum"])
            if key in ref and ref[key] != ck:
                raise ValueError(f"参照系列の checksum が同キーで食い違う: {key}")
            ref[key] = ck
    return ref


def aggregate_rows(py_rows, fandhe_rows, reference, checksums_match):
    """5 run 中央値・換算値・比の表を返す。fp32／cpu 腕の checksum 不一致は例外、tf32 は記録のみ。"""
    by = {}
    for r in py_rows:
        by.setdefault((r["arm"], r["n"], r["phase"]), {})[r["run"]] = r
    # fp32／cpu 腕の checksum を参照系列と照合（fail-closed）。tf32 は照合しない（一致しないのが正常）。
    for r in py_rows:
        if r["arm"] == "tf32":
            continue
        ref = reference.get((r["device"], r["n"]))
        if ref is None:
            raise ValueError(f"参照系列に基準行が無い: {r['device']} N={r['n']}")
        if not checksums_match(r["checksum"], ref):
            raise ValueError(f"checksum が参照系列と不一致: {r['arm']} N={r['n']} {r['checksum']} vs {ref}")
    py = {}
    for (arm, n, ph), runs in sorted(by.items()):
        med = statistics.median(x["median_s"] for x in runs.values())
        e = {"median_s": med, "per_run": {k: runs[k]["median_s"] for k in sorted(runs)}}
        if ph in ("checksum", "checksum_hot", "readout"):
            e["ns_per_element"] = D.ns_per_element(med, n, n)
        if ph in ("mm_pure", "mm_event", "mm_event_batch10", "window"):
            e["tflops"] = D.gflops(n, n, n, med) / 1e3
        py[(arm, n, ph)] = e
    for (arm, n, ph), e in py.items():
        if ph == "window":
            parts = [py[(arm, n, p)]["median_s"] for p in ("mm_pure", "readout", "checksum")]
            e["sum_over_window"] = sum(parts) / e["median_s"]
        if ph == "checksum":
            e["vs_estimate_abs_ns"] = e["ns_per_element"] - CHECKSUM_NS_ESTIMATE
            e["vs_estimate_ratio"] = e["ns_per_element"] / CHECKSUM_NS_ESTIMATE
    # fandhe 側（run 別の wall は launch_issue+kernel_wait を run ごとに足してから中央値）。
    fb = {}
    for r in fandhe_rows:
        fb.setdefault((r["n"], r["layer"], r["phase"]), {})[r["run"]] = r["median_s"]
    fandhe = {}
    for n in LAYERB_SIZES:
        runs = sorted(fb[(n, "B", "kernel_wait")])
        per = {k: fb[(n, "B", "launch_issue")][k] + fb[(n, "B", "kernel_wait")][k] for k in runs}
        fandhe[(n, "wall")] = {"median_s": statistics.median(per.values()), "per_run": per,
                               "source": "Layer B launch_issue+kernel_wait"}
    for n in DIAG_SIZES:
        # L2S は kernel_wait（同期のみ）しか持たず、mm_pure（投入〜同期完了）と境界が違うため
        # wall 比の分子にしない（未計測扱い。codex P2）。参考値だけ別キーで残す。
        per = fb[(n, "L2S", "kernel_wait")]
        fandhe[(n, "kernel_wait_only")] = {"median_s": statistics.median(per.values()),
                                           "per_run": dict(sorted(per.items())), "source": "L2S kernel_wait"}
        for ph in ("dev_kernel_seg", "dev_kernel_b2b"):
            per = fb[(n, "D", ph)]
            fandhe[(n, ph)] = {"median_s": statistics.median(per.values()), "per_run": dict(sorted(per.items())),
                               "source": f"D {ph}"}
    ratios = []
    for n in SIZES:
        pure = py[("fp32", n, "mm_pure")]["median_s"]
        ev = py[("fp32", n, "mm_event")]["median_s"]
        t32 = py[("tf32", n, "mm_pure")]["median_s"]
        t32e = py[("tf32", n, "mm_event")]["median_s"]
        fw = fandhe.get((n, "wall"))
        seg, b2b = fandhe.get((n, "dev_kernel_seg")), fandhe.get((n, "dev_kernel_b2b"))
        c_fp, c_tf = py[("fp32", n, "window")], py[("tf32", n, "window")]
        ck_fp = next(r["checksum"] for r in py_rows if r["arm"] == "fp32" and r["n"] == n)
        ck_tf = next(r["checksum"] for r in py_rows if r["arm"] == "tf32" and r["n"] == n)
        ratios.append({
            "n": n,
            "fp32_mm_pure_s": pure, "fp32_mm_event_s": ev,
            "fp32_tflops": py[("fp32", n, "mm_pure")]["tflops"],
            "fp32_peak_frac": py[("fp32", n, "mm_pure")]["tflops"] / PEAK_FP32_TFLOPS,
            "tf32_tflops": py[("tf32", n, "mm_pure")]["tflops"],
            "tf32_peak_frac": py[("tf32", n, "mm_pure")]["tflops"] / PEAK_TF32_TFLOPS,
            "fandhe_wall_s": fw["median_s"] if fw else None,
            "wall_ratio_fandhe_over_cublas_fp32": fw["median_s"] / pure if fw else None,
            "fandhe_dev_kernel_seg_s": seg["median_s"] if seg else None,
            "fandhe_dev_kernel_b2b_s": b2b["median_s"] if b2b else None,
            "event_ratio_seg_over_cublas_fp32": seg["median_s"] / ev if seg else None,
            "event_ratio_b2b_over_cublas_fp32": b2b["median_s"] / ev if b2b else None,
            "tf32_over_fp32_wall": t32 / pure, "tf32_over_fp32_event": t32e / ev,
            "tf32_checksum_minus_fp32": ck_tf - ck_fp, "tf32_checksum_bit_equal_fp32": ck_tf == ck_fp,
            "window_ratio_tf32_over_fp32": c_tf["median_s"] / c_fp["median_s"],
        })
    return {"py": {f"{a}|{n}|{p}": v for (a, n, p), v in py.items()},
            "fandhe": {f"{n}|{k}": v for (n, k), v in fandhe.items()}, "ratios": ratios}


def render_md(agg):
    f = lambda x, p: "該当計測なし" if x is None else format(x, p)  # noqa: E731
    L = ["# #3185 集計（5 run 中央値。run 別は JSON に併記）", "",
         "## 比（時計ごとに列を分ける。wall 対 wall、event 対 event）", "",
         "| N | cuBLAS fp32 mm_pure [ms] | fp32 mm_event [ms] | fp32 TFLOPS(対31) | tf32 TFLOPS(対53.3) | "
         "fandhe wall [ms] | wall比 fandhe/cuBLAS | fandhe dev_seg [ms] | event比 seg | event比 b2b | "
         "tf32/fp32 wall | tf32/fp32 event | tf32−fp32 checksum | bit一致 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in agg["ratios"]:
        ms = lambda s: None if s is None else s * 1e3  # noqa: E731
        L.append(
            f"| {r['n']} | {r['fp32_mm_pure_s']*1e3:.4f} | {r['fp32_mm_event_s']*1e3:.4f} | "
            f"{r['fp32_tflops']:.2f}({r['fp32_peak_frac']:.2f}) | {r['tf32_tflops']:.2f}({r['tf32_peak_frac']:.2f}) | "
            f"{f(ms(r['fandhe_wall_s']), '.4f')} | {f(r['wall_ratio_fandhe_over_cublas_fp32'], '.3f')} | "
            f"{f(ms(r['fandhe_dev_kernel_seg_s']), '.4f')} | {f(r['event_ratio_seg_over_cublas_fp32'], '.3f')} | "
            f"{f(r['event_ratio_b2b_over_cublas_fp32'], '.3f')} | {r['tf32_over_fp32_wall']:.3f} | "
            f"{r['tf32_over_fp32_event']:.3f} | {r['tf32_checksum_minus_fp32']:.6g} | "
            f"{r['tf32_checksum_bit_equal_fp32']} |")
    L += ["", "注: fandhe の wall は N>=1024 が Layer B（warmup20/計測20）、N<=512 は投入を含む同境界の計測がないため wall 比を未計測（-）とする。"
          "event 時計は N<=512 のみ。入力は bench_py と別シード。", "",
          "## 区間（arm|N|phase）", "",
          "| arm | N | phase | median [s] | ns/要素 | TFLOPS | Σ区間/window | 推定2.7との差[ns] | run 別 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k, v in agg["py"].items():
        arm, n, ph = k.split("|")
        runs = ", ".join(f"r{r}={x:.3e}" for r, x in v["per_run"].items())
        L.append(f"| {arm} | {n} | {ph} | {v['median_s']:.4e} | {f(v.get('ns_per_element'), '.3f')} | "
                 f"{f(v.get('tflops'), '.2f')} | {f(v.get('sum_over_window'), '.3f')} | "
                 f"{f(v.get('vs_estimate_abs_ns'), '+.3f')} | {runs} |")
    L += ["", "## fandhe 側（N|区間）", "", "| key | median [s] | 出典 | run 別 |", "|---|---|---|---|"]
    for k, v in agg["fandhe"].items():
        runs = ", ".join(f"r{r}={x:.3e}" for r, x in v["per_run"].items())
        L.append(f"| {k} | {v['median_s']:.4e} | {v['source']} | {runs} |")
    return "\n".join(L) + "\n"


def cmd_aggregate(args):
    py_rows = load_py_runs(args.runs_dir, args.runs)
    fandhe_rows = load_fandhe_runs(args.runs_dir, args.runs)
    check_complete(py_rows, fandhe_rows)
    agg = aggregate_rows(py_rows, fandhe_rows, load_reference(args.reference_py_jsonl), D.load_checksums_match())
    for p in (args.out_json, args.out_md):
        if Path(p).exists():
            raise SystemExit(f"出力が既にある（上書きしない）: {p}")
    Path(args.out_json).write_text(json.dumps(agg, indent=1) + "\n")
    Path(args.out_md).write_text(render_md(agg))
    print(f"aggregate ok: {len(agg['py'])} py entries, {len(agg['fandhe'])} fandhe entries")


def self_test():
    assert D.ISSUE == 3179 and ISSUE == 3185
    assert abs(D.ns_per_element(2.7e-9 * 65536, 256, 256) - 2.7) < 1e-9
    base = dict(issue=ISSUE, arm="fp32", device="cuda", shape_id="sq256", n=256, phase="mm_pure",
                median_s=1e-3, q1_s=1e-3, q3_s=1e-3, warmup=20, iters=20, run=1, allow_tf32=False,
                fp32_matmul_precision="highest", fp32_precision="", torch_version="x", cuda_version="13.0",
                numpy_version="y", bench_py_sha256="z", checksum=1.0, gpu_name="g")
    validate_row(dict(base))
    for bad in ({"median_s": float("nan")}, {"phase": "nope"}, {"allow_tf32": True}, {"device": "cpu"}):
        try:
            validate_row({**base, **bad})
        except ValueError:
            continue
        raise AssertionError("不正行を拒否していない")
    hd = "test gemm_reuse_phase_diag_tests::gemm_reuse_phase_diag_select ...   N=1024 kernel=Select (median over 20 trials, 20 warmup):"
    hm = LAYERB_HEAD.match(hd)
    assert hm and hm.groups() == ("1024", "Select"), "libtest 接頭辞付きヘッダを認識しない"
    assert LAYERB_HEAD.match("  N=2048 kernel=Select (median over 20 trials, 20 warmup):")
    assert not LAYERB_HEAD.match("xtest foo ...   N=1024 kernel=Select (median over")
    txt = hd + "\n    launch_issue: median=0.0029 ms  q1=0.0029 ms  q3=0.0030 ms\n" \
        "    kernel_wait: median=1.0 ms  q1=1.0 ms  q3=1.0 ms\n"
    for n in (2048, 4096):
        txt += f"  N={n} kernel=Select (median over 20 trials, 20 warmup):\n" \
            "    launch_issue: median=0.1 ms  q1=0.1 ms  q3=0.1 ms\n    kernel_wait: median=1.0 ms  q1=1.0 ms  q3=1.0 ms\n"
    assert len(parse_layerb(txt + "test result: ok. 1 passed\n", 1)) == 6
    assert blas_libs_from_maps("7f 0 0 0 0 /usr/lib/libopenblas.so.0\n") == ["libopenblas.so.0"]
    print("self-test ok")


def main(argv=None):
    ap = argparse.ArgumentParser(description="GB10 の cuBLAS 純カーネル時間と checksum の診断（#3185）")
    ap.add_argument("--self-test", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    m = sub.add_parser("measure")
    m.add_argument("--bench-py", required=True)
    m.add_argument("--arm", choices=ARMS, required=True)
    m.add_argument("--run", type=int, required=True)
    m.add_argument("--out", required=True)
    c = sub.add_parser("cpu-blas-probe")
    c.add_argument("--out-dir", required=True)
    p = sub.add_parser("parse-fandhe")
    p.add_argument("--log", required=True)
    p.add_argument("--kind", choices=("layerb", "diag2109"), required=True)
    p.add_argument("--run", type=int, required=True)
    p.add_argument("--out", required=True)
    a = sub.add_parser("aggregate")
    a.add_argument("--runs-dir", required=True)
    a.add_argument("--reference-py-jsonl", required=True)
    a.add_argument("--out-json", required=True)
    a.add_argument("--out-md", required=True)
    a.add_argument("--runs", type=int, default=NUM_RUNS)
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    handlers = {"measure": cmd_measure, "cpu-blas-probe": cmd_cpu_blas_probe,
                "parse-fandhe": cmd_parse_fandhe, "aggregate": cmd_aggregate}
    if args.cmd in handlers:
        return handlers[args.cmd](args)
    ap.error("サブコマンドか --self-test が必要")


if __name__ == "__main__":
    main()
