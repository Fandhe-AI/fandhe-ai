#!/usr/bin/env python3
"""Mac mini の PyTorch CPU の MLP 経路（train／infer）と GEMM N=256 の窓を区間に分けて測る診断（イシュー #4148）。

## 位置づけ

`docs/perf/logs/lowlayer-diagnosis-2026-09-12/bench_py.py` の PyTorch CPU 窓には、Accelerate（BLAS）の
純カーネル時間のほかに torch の dispatch／autograd／Python 固定費と非 GEMM の要素演算
（relu・bias 加算・MSE・SGD 更新・readout・checksum）が全部入っている。本スクリプトは窓を区間に分け、
それぞれを証拠タグ（実測／推定／未測定）付きで記録する。`opp_kernel_torch_diag.py`（#3179）と同じ
診断専用ツールで、**bench_py.py・相手側ベンチ・判定規則・fandhe 本体は変えない**。判定・閾値は設けない（record_only）。
bench_py.py は `--bench-py` で受け取り、RULE.txt に固定した sha256 と一致したものだけを importlib で読み、
`fill_vec`／`checksum`／`quantile`／`WARMUP`／`ITERS`／`mlp_init`／`Torch` を再利用する（定数・関数を複製しない）。

## 呼び出し元

- Mac mini の venv-torch 上で `docs/perf/logs/diag-0140-torch-cpu-mlp-macmini-4148/run_round.sh` が
  env → measure（train・infer・gemm256・shapes）→ profile を順に起動する。
- 結果は `docs/perf/logs/diag-0140-torch-cpu-mlp-macmini-4148/aggregate_4148.py` が厳格パースして区間表にする。
- CI で動くのは torch／numpy 不要の部分だけ（`torch_cpu_mlp_diag_test.py`・`--self-test`）。

## サブコマンド

  env     --bench-py PATH --out F.json
  measure --bench-py PATH --section train|infer|gemm256|shapes --run K --out F.jsonl
  profile --bench-py PATH --section train|infer|gemm256 --run K --steps N --out F.json
  --self-test

## 計時腕と profiler 腕の分離

`torch.profiler` は計測を摂動する。`measure`（計時腕）は profiler を一切有効にしない。`profile` は別腕で、
op 在庫・step あたりの op 数・`aten::mm` が子 op を持つか（BLAS 呼び出しの直下か）だけを記録し、
その時間値は「profiler 下・割合の参考」であって計時腕の値と混ぜない。

## 安全面

Accelerate は固定パス・固定シンボル名（`cblas_sgemm`）だけを ctypes で呼ぶ。パス・シンボルは環境変数や引数から組み立てない。
`otool` も固定の実行名と固定の引数だけで起動する（外部入力をシェルに渡さない）。
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ISSUE = 4148
# RULE.txt の bench_py_sha256 と同じ値（#3179 と同一。0.11.0 以降の計測で使った bench_py.py）。
BENCH_PY_SHA256 = "9fd2c08f37d66b5bb41ececab8a5b3daaa36b55609770502ed3b35ae2ce2409b"
TORCH_VERSION_PREFIX = "2.14.0"
# Accelerate の固定パスと固定シンボル。外部入力から組み立てない。
ACCELERATE_PATH = "/System/Library/Frameworks/Accelerate.framework/Accelerate"
ACCELERATE_SYMBOL = "cblas_sgemm"
OTOOL_PATH = "/usr/bin/otool"

# (shape_id, m, n, k, trans)。#3177／#4146 RULE と同じ MLP 6 形状（l1_dx は PyTorch が計算しないため窓の和には入れない）。
MLP_SHAPES = (
    ("l1_fwd", 64, 256, 784, "NN"),
    ("l1_dw", 784, 256, 64, "TN"),
    ("l1_dx", 64, 784, 256, "NT"),
    ("l2_fwd", 64, 10, 256, "NN"),
    ("l2_dw", 256, 10, 64, "TN"),
    ("l2_dx", 64, 256, 10, "NT"),
)
TRAIN_GEMM_FWD = ("l1_fwd", "l2_fwd")
TRAIN_GEMM_BWD = ("l2_dw", "l2_dx", "l1_dw")
INFER_GEMM = ("l1_fwd", "l2_fwd")
MLP_SEED_A, MLP_SEED_B = 0x4148A, 0x4148B
GEMM_N = 256
FLOOR_REPS = 100

# section -> 必須 item 集合（aggregate_4148.py と同一。テストで突き合わせる）。
SECTION_ITEMS = {
    "train": ("window", "fwd", "grad", "update", "loss_item", "ew_fwd", "dispatch_floor_add", "dispatch_floor_mm"),
    "infer": ("window", "fwd_compute", "readout", "checksum", "ew_fwd", "dispatch_floor_add", "dispatch_floor_mm"),
    "gemm256": ("window", "mm_pure", "mm_pure_batch10", "readout", "checksum", "accelerate_batch10", "dispatch_floor_mm"),
    "shapes": tuple(f"{p}_{s[0]}" for s in MLP_SHAPES for p in ("mm_pure", "mm_batch10", "accel_batch10")),
}
PROFILE_SECTIONS = ("train", "infer", "gemm256")
TAGS = ("実測", "推定", "未測定")
REQUIRED_KEYS = (
    "issue", "section", "item", "status", "tag", "median_s", "q1_s", "q3_s", "warmup", "iters", "run",
    "torch_threads", "torch_version", "numpy_version", "bench_py_sha256", "diag_script_sha256",
)
ENV_KEYS = ("issue", "torch_version", "numpy_version", "torch_threads", "torch_config", "otool_libtorch_cpu",
            "accelerate_linked", "accelerate_direct_available", "bench_py_sha256", "diag_script_sha256")
PROFILE_KEYS = ("issue", "section", "run", "steps", "torch_version", "n_ops_aten_total", "n_ops_aten_toplevel",
                "mm_has_child_ops", "ops", "note", "bench_py_sha256", "diag_script_sha256")


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def self_sha256():
    return sha256_file(os.path.abspath(__file__))


def quantile(secs, p):
    """bench_py.quantile と同じ線形補間（torch 不在の self-test 用。実測では bp.quantile を使う）。"""
    s = sorted(secs)
    idx = p * (len(s) - 1)
    lo, hi = int(math.floor(idx)), int(math.ceil(idx))
    return s[lo] * (1 - (idx - lo)) + s[hi] * (idx - lo)


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


def _finite_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def validate_row(row):
    """計時腕 JSONL 1 行の厳格検証。欠損・不明 section／item・非有限・unavailable の値混入は例外。"""
    for k in REQUIRED_KEYS:
        if k not in row:
            raise ValueError(f"キー欠損: {k}")
    if row["issue"] != ISSUE:
        raise ValueError("issue 不一致")
    if row["section"] not in SECTION_ITEMS:
        raise ValueError(f"不明な section: {row['section']}")
    if row["item"] not in SECTION_ITEMS[row["section"]]:
        raise ValueError(f"不明な item: {row['section']}/{row['item']}")
    if row["tag"] not in TAGS:
        raise ValueError(f"不明な tag: {row['tag']}")
    if not isinstance(row["run"], int) or isinstance(row["run"], bool) or row["run"] < 1:
        raise ValueError("run が不正")
    if row["status"] == "ok":
        for k in ("median_s", "q1_s", "q3_s"):
            if not _finite_num(row[k]):
                raise ValueError(f"非有限値: {k}={row[k]!r}")
        if row["median_s"] <= 0:
            raise ValueError("median_s が正でない")
        if row["tag"] == "未測定":
            raise ValueError("status=ok に tag=未測定は付けられない")
    elif row["status"] == "unavailable":
        # 到達不能な量は推定値で埋めない: 値は null、tag は 未測定。
        if any(row[k] is not None for k in ("median_s", "q1_s", "q3_s")) or row["tag"] != "未測定":
            raise ValueError("unavailable の行は値 null・tag=未測定でなければならない")
        if not isinstance(row.get("reason"), str) or not row["reason"]:
            raise ValueError("unavailable の行は reason が必要")
    else:
        raise ValueError(f"不明な status: {row['status']}")
    return row


def validate_env(d):
    for k in ENV_KEYS:
        if k not in d:
            raise ValueError(f"env キー欠損: {k}")
    if d["issue"] != ISSUE:
        raise ValueError("issue 不一致")
    if not isinstance(d["torch_version"], str) or not d["torch_version"].startswith(TORCH_VERSION_PREFIX):
        raise ValueError("torch の版が RULE（2.14.0）と違う")
    return d


def validate_profile(d):
    for k in PROFILE_KEYS:
        if k not in d:
            raise ValueError(f"profile キー欠損: {k}")
    if d["issue"] != ISSUE or d["section"] not in PROFILE_SECTIONS:
        raise ValueError("profile の issue／section が不正")
    for k in ("n_ops_aten_total", "n_ops_aten_toplevel", "steps"):
        if not isinstance(d[k], int) or isinstance(d[k], bool) or d[k] < 1:
            raise ValueError(f"profile の {k} が不正")
    if d["n_ops_aten_toplevel"] > d["n_ops_aten_total"]:
        raise ValueError("toplevel が total を超える")
    return d


def stats_of(secs):
    return statistics.median(secs), quantile(secs, 0.25), quantile(secs, 0.75)


def sample(fn, warmup, iters, prep=None):
    """warmup + iters 回の秒数列から (median, q1, q3)。prep があれば計時外で毎回呼び、その戻り値を fn に渡す。"""
    for _ in range(warmup):
        a = prep() if prep else None
        fn(a) if prep else fn()
    secs = []
    for _ in range(iters):
        a = prep() if prep else None
        t = time.perf_counter()
        fn(a) if prep else fn()
        secs.append(time.perf_counter() - t)
    return stats_of(secs)


def per_call(stats, n):
    return tuple(x / n for x in stats)


def make_row(ctx, section, item, stats, tag, warmup, iters, run, reason=None):
    base = {
        "issue": ISSUE, "section": section, "item": item, "warmup": warmup, "iters": iters, "run": run,
        "torch_threads": ctx["threads"], "torch_version": ctx["torch_version"], "numpy_version": ctx["numpy_version"],
        "bench_py_sha256": ctx["bench_sha"], "diag_script_sha256": ctx["self_sha"],
    }
    if stats is None:
        base.update(status="unavailable", tag="未測定", median_s=None, q1_s=None, q3_s=None, reason=reason or "unavailable")
    else:
        base.update(status="ok", tag=tag, median_s=stats[0], q1_s=stats[1], q3_s=stats[2])
    return validate_row(base)


# ---- Accelerate cblas_sgemm の直呼び（固定パス・固定シンボル。ロード不可・不在は None） ----------
def load_accelerate():
    import ctypes
    try:
        lib = ctypes.CDLL(ACCELERATE_PATH)
        fn = getattr(lib, ACCELERATE_SYMBOL)
    except (OSError, AttributeError):
        return None
    fn.restype = None
    fn.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                   ctypes.c_float, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
                   ctypes.c_float, ctypes.c_void_p, ctypes.c_int]
    return fn


def accel_gemm_fn(fn, a_np, b_np, m, n, k, trans):
    """行優先 C(m×n)=op(A)·op(B)。a_np／b_np は格納形状のままの連続 float32。戻り値は (呼び出し関数, C)。"""
    import numpy as np
    c = np.zeros((m, n), dtype=np.float32)
    ta = 112 if trans[0] == "T" else 111
    tb = 112 if trans[1] == "T" else 111
    lda, ldb = a_np.shape[1], b_np.shape[1]
    pa, pb, pc = a_np.ctypes.data, b_np.ctypes.data, c.ctypes.data

    def call():
        fn(101, ta, tb, m, n, k, 1.0, pa, lda, pb, ldb, 0.0, pc, n)
    return call, c


def accel_batch10(fn, a_np, b_np, m, n, k, trans, ref, warmup, iters):
    """Accelerate 直呼びの 10 回償却値。torch.mm の結果 ref と一致しなければ None（unavailable）。"""
    import numpy as np
    call, c = accel_gemm_fn(fn, a_np, b_np, m, n, k, trans)
    call()
    ok = np.all((np.abs(c - ref) <= 1e-5) | (np.abs(c - ref) <= 1e-3 * np.abs(ref)))

    def batch():
        for _ in range(10):
            call()
    if not ok:
        return None
    return per_call(sample(batch, warmup, iters), 10)


def shape_inputs(bp, np, torch, sid, m, n, k, trans):
    ash = (k, m) if trans[0] == "T" else (m, k)
    bsh = (n, k) if trans[1] == "T" else (k, n)
    a_np = np.ascontiguousarray(bp.fill_vec(MLP_SEED_A, ash[0] * ash[1]).reshape(ash))
    b_np = np.ascontiguousarray(bp.fill_vec(MLP_SEED_B, bsh[0] * bsh[1]).reshape(bsh))
    a, b = torch.from_numpy(a_np), torch.from_numpy(b_np)
    if trans[0] == "T":
        a = a.t()
    if trans[1] == "T":
        b = b.t()
    return a_np, b_np, a, b


def make_ctx(bp, torch, np, bench_py_path):
    return {"threads": torch.get_num_threads(), "torch_version": torch.__version__, "numpy_version": np.__version__,
            "bench_sha": sha256_file(Path(bench_py_path).resolve()), "self_sha": self_sha256()}


def floors(torch, reps=FLOOR_REPS):
    """計算量ゼロの op の 1 回あたり時間（カーネルを含まない dispatch の床）。"""
    one = torch.ones(1)
    m1 = torch.ones(1, 1)
    add = lambda: [one + one for _ in range(reps)]
    mm = lambda: [torch.mm(m1, m1) for _ in range(reps)]
    return {"dispatch_floor_add": per_call(sample(add, 20, 20), reps), "dispatch_floor_mm": per_call(sample(mm, 20, 20), reps)}


# ---- セクション別の計測クロージャ（measure と profile が共有） -------------------------------------
def build_train(bp, torch, np):
    be = bp.Torch("cpu")
    w1, b1, w2, b2, x, y = bp.mlp_init()
    be.train_setup(w1, b1, w2, b2)
    X, Y = be.upload(x), be.upload(y)
    pw1, pb1, pw2, pb2 = be.p
    t = torch

    def fwd():
        h = t.relu(X @ pw1 + pb1)
        pred = h @ pw2 + pb2
        return ((pred - Y) ** 2).mean()

    def grad_prep():
        return fwd()

    def grad(loss):
        return t.autograd.grad(loss, be.p)

    gs = grad(fwd())
    loss0 = fwd()
    z1, z2 = X @ pw1, t.relu(X @ pw1 + pb1) @ pw2

    def update():
        with t.no_grad():
            for p, g in zip(be.p, gs):
                p.sub_(g * bp.LR)

    def ew_fwd():
        h = t.relu(z1 + pb1)
        pred = z2 + pb2
        return ((pred - Y) ** 2).mean()

    return {"window": lambda: be.train_step(X, Y), "fwd": fwd, "grad": (grad_prep, grad), "update": update,
            "loss_item": lambda: float(loss0.item()), "ew_fwd": ew_fwd}


def build_infer(bp, torch, np):
    be = bp.Torch("cpu")
    w1, b1, w2, b2, x, y = bp.mlp_init()
    be.train_setup(w1, b1, w2, b2)
    X = be.upload(x)
    pw1, pb1, pw2, pb2 = be.p
    t = torch

    def fwd_t():
        with t.no_grad():
            return t.relu(X @ pw1 + pb1) @ pw2 + pb2
    out_t = fwd_t()
    with t.no_grad():
        z1, z2 = X @ pw1, t.relu(X @ pw1 + pb1) @ pw2
    host = out_t.cpu().numpy()

    def ew():
        with t.no_grad():
            t.relu(z1 + pb1)
            z2 + pb2

    def window():
        return bp.checksum(be.forward_to_host(X))
    return {"window": window, "fwd_compute": fwd_t, "readout": lambda: out_t.cpu().numpy(),
            "checksum": lambda: bp.checksum(host), "ew_fwd": ew}


def batch10_out(torch, a, b):
    """torch.mm(a, b, out=o) を 10 回呼ぶ関数。出力は 1 つを再利用する（Accelerate 直呼びと同じ条件。10 個保持しない）。

    この値は dispatch（Python→ATen の呼び出し費用）を含み、出力確保は含まない。純カーネル時間そのものではなく
    「dispatch 込みの torch.mm 時間 / 回」であり、RULE §4・§5 でもそう扱う（過小評価でなく下限側の内訳として読む）。
    """
    o = torch.empty(a.shape[0], b.shape[1], dtype=a.dtype)

    def run():
        for _ in range(10):
            torch.mm(a, b, out=o)
    return run


def build_gemm(bp, torch, np):
    n = GEMM_N
    a_np = np.ascontiguousarray(bp.fill_vec(bp.SEED_A, n * n).reshape(n, n))
    b_np = np.ascontiguousarray(bp.fill_vec(bp.SEED_B, n * n).reshape(n, n))
    a, b = torch.from_numpy(a_np), torch.from_numpy(b_np)
    mm = lambda: torch.mm(a, b)
    c0 = mm()
    host = c0.cpu().numpy()
    return {"a_np": a_np, "b_np": b_np, "ref": host, "mm": mm, "mm_batch10": batch10_out(torch, a, b),
            "window": lambda: bp.checksum((a @ b).cpu().numpy()),
            "readout": lambda: c0.cpu().numpy(), "checksum": lambda: bp.checksum(host)}


def cmd_measure(args):
    bp = load_bench_py(args.bench_py)
    import numpy as np
    import torch
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"出力が既にある（上書きしない）: {out}")
    ctx = make_ctx(bp, torch, np, args.bench_py)
    W, I = bp.WARMUP, bp.ITERS
    TW, TI = bp.TRAIN_WARMUP, bp.TRAIN_STEPS - bp.TRAIN_WARMUP
    rows = []
    sec = args.section

    def add(item, stats, tag="実測", warmup=W, iters=I, reason=None):
        rows.append(make_row(ctx, sec, item, stats, tag, warmup, iters, args.run, reason))

    fl = floors(torch)
    if sec == "train":
        c = build_train(bp, torch, np)
        add("window", sample(c["window"], TW, TI), warmup=TW, iters=TI)
        add("fwd", sample(c["fwd"], TW, TI), warmup=TW, iters=TI)
        prep, g = c["grad"]
        add("grad", sample(g, TW, TI, prep=prep), warmup=TW, iters=TI)
        add("update", sample(c["update"], TW, TI), warmup=TW, iters=TI)
        add("loss_item", sample(c["loss_item"], TW, TI), warmup=TW, iters=TI)
        add("ew_fwd", sample(c["ew_fwd"], TW, TI), warmup=TW, iters=TI)
    elif sec == "infer":
        c = build_infer(bp, torch, np)
        for item in ("window", "fwd_compute", "readout", "checksum", "ew_fwd"):
            add(item, sample(c[item], W, I))
    elif sec == "gemm256":
        c = build_gemm(bp, torch, np)
        add("window", sample(c["window"], W, I))
        add("mm_pure", sample(c["mm"], W, I))
        add("mm_pure_batch10", per_call(sample(c["mm_batch10"], W, I), 10))
        add("readout", sample(c["readout"], W, I))
        add("checksum", sample(c["checksum"], W, I))
        fn = load_accelerate()
        if fn is None:
            add("accelerate_batch10", None, reason="Accelerate／cblas_sgemm をロードできない")
        else:
            st = accel_batch10(fn, c["a_np"], c["b_np"], GEMM_N, GEMM_N, GEMM_N, "NN", c["ref"], W, I)
            if st is None:
                add("accelerate_batch10", None, reason="cblas_sgemm の結果が torch.mm と一致しない")
            else:
                add("accelerate_batch10", st)
    elif sec == "shapes":
        fn = load_accelerate()
        for sid, m, n, k, trans in MLP_SHAPES:
            a_np, b_np, a, b = shape_inputs(bp, np, torch, sid, m, n, k, trans)
            mm = lambda: torch.mm(a, b)
            ref = mm().numpy()
            add(f"mm_pure_{sid}", sample(mm, W, I))
            add(f"mm_batch10_{sid}", per_call(sample(batch10_out(torch, a, b), W, I), 10))
            if fn is None:
                add(f"accel_batch10_{sid}", None, reason="Accelerate／cblas_sgemm をロードできない")
            else:
                st = accel_batch10(fn, a_np, b_np, m, n, k, trans, ref, W, I)
                if st is None:
                    add(f"accel_batch10_{sid}", None, reason="cblas_sgemm の結果が torch.mm と一致しない")
                else:
                    add(f"accel_batch10_{sid}", st)
    else:
        raise SystemExit(f"不明な section: {sec}")
    # dispatch_floor_add は SECTION_ITEMS（train／infer）にだけ定義がある。gemm256 に足すと validate_row が落ちる。
    if sec in ("train", "infer"):
        add("dispatch_floor_add", fl["dispatch_floor_add"], tag="実測", warmup=20, iters=20)
    if sec in ("train", "infer", "gemm256"):
        add("dispatch_floor_mm", fl["dispatch_floor_mm"], tag="実測", warmup=20, iters=20)
    got = {r["item"] for r in rows}
    if got != set(SECTION_ITEMS[sec]):
        raise SystemExit(f"item 集合が不完全: 不足={set(SECTION_ITEMS[sec]) - got} 余分={got - set(SECTION_ITEMS[sec])}")
    with open(out, "x") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows -> {out}")


def cmd_env(args):
    bp = load_bench_py(args.bench_py)
    import numpy as np
    import torch
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"出力が既にある（上書きしない）: {out}")
    otool_out = "unavailable"
    accelerate_linked = None
    lib = Path(torch.__file__).resolve().parent / "lib" / "libtorch_cpu.dylib"
    if lib.is_file() and Path(OTOOL_PATH).is_file():
        # 固定の実行ファイルと固定の引数 -L、対象は torch 自身のパスだけ（シェルを使わない）。
        r = subprocess.run([OTOOL_PATH, "-L", str(lib)], capture_output=True, text=True, timeout=60, check=False)
        if r.returncode == 0:
            otool_out = r.stdout
            accelerate_linked = "Accelerate" in otool_out
    d = {"issue": ISSUE, "torch_version": torch.__version__, "numpy_version": np.__version__,
         "torch_threads": torch.get_num_threads(), "torch_config": torch.__config__.show(),
         "otool_libtorch_cpu": otool_out, "accelerate_linked": accelerate_linked,
         "accelerate_direct_available": load_accelerate() is not None,
         "bench_py_sha256": sha256_file(Path(args.bench_py).resolve()), "diag_script_sha256": self_sha256()}
    validate_env(d)
    with open(out, "x") as f:
        json.dump(d, f, ensure_ascii=False, indent=1)
    print(f"wrote env -> {out}")


def cmd_profile(args):
    bp = load_bench_py(args.bench_py)
    import numpy as np
    import torch
    from torch.profiler import ProfilerActivity, profile
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"出力が既にある（上書きしない）: {out}")
    sec = args.section
    if sec == "train":
        step = build_train(bp, torch, np)["window"]
    elif sec == "infer":
        step = build_infer(bp, torch, np)["window"]
    elif sec == "gemm256":
        step = build_gemm(bp, torch, np)["window"]
    else:
        raise SystemExit(f"profile に使えない section: {sec}")
    for _ in range(bp.WARMUP):
        step()
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        for _ in range(args.steps):
            step()
    evs = [e for e in prof.events() if e.name.startswith("aten::")]

    def has_aten_ancestor(e):
        p = e.cpu_parent
        while p is not None:
            if p.name.startswith("aten::"):
                return True
            p = p.cpu_parent
        return False
    top = [e for e in evs if not has_aten_ancestor(e)]
    mm_has_child = any(e.name == "aten::mm" and len(e.cpu_children) > 0 for e in evs)
    agg = {}
    for e in evs:
        a = agg.setdefault(e.name, [0, 0.0, 0.0])
        a[0] += 1
        a[1] += e.self_cpu_time_total / 1e6
        a[2] += e.cpu_time_total / 1e6
    ops = [{"name": k, "count": v[0], "self_cpu_s": v[1], "cpu_total_s": v[2]} for k, v in sorted(agg.items())]
    d = {"issue": ISSUE, "section": sec, "run": args.run, "steps": args.steps, "torch_version": torch.__version__,
         "n_ops_aten_total": max(1, len(evs) // args.steps), "n_ops_aten_toplevel": max(1, len(top) // args.steps),
         "mm_has_child_ops": mm_has_child, "ops": ops,
         "note": "profiler 下の値。時間は計時腕の値と混ぜない（op 数と構成比だけを内訳計算に使う）",
         "bench_py_sha256": sha256_file(Path(args.bench_py).resolve()), "diag_script_sha256": self_sha256()}
    validate_profile(d)
    with open(out, "x") as f:
        json.dump(d, f, ensure_ascii=False, indent=1)
    print(f"wrote profile -> {out}")


def self_test():
    """torch／numpy 不要の検査（検証関数・統計・item 集合）。"""
    ctx = {"threads": 4, "torch_version": "2.14.0", "numpy_version": "2", "bench_sha": "s", "self_sha": "d"}
    r = make_row(ctx, "gemm256", "window", (1e-3, 9e-4, 1.1e-3), "実測", 20, 20, 1)
    assert r["status"] == "ok"
    u = make_row(ctx, "gemm256", "accelerate_batch10", None, "実測", 20, 20, 1, reason="x")
    assert u["median_s"] is None and u["tag"] == "未測定"
    for bad in (dict(r, median_s=float("nan")), dict(r, item="nope"), dict(r, section="nope"), dict(r, run=0),
                dict(u, median_s=1.0), dict(u, reason="")):
        try:
            validate_row(bad)
        except ValueError:
            continue
        raise AssertionError(f"検証をすり抜けた: {bad}")
    missing = {k: v for k, v in r.items() if k != "tag"}
    try:
        validate_row(missing)
        raise AssertionError("欠損キーが通った")
    except ValueError:
        pass
    assert stats_of([1.0, 2.0, 3.0, 4.0, 5.0]) == (3.0, 2.0, 4.0)
    assert len(SECTION_ITEMS["shapes"]) == 18
    print("self-test OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--self-test", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    e = sub.add_parser("env")
    e.add_argument("--bench-py", required=True)
    e.add_argument("--out", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--bench-py", required=True)
    m.add_argument("--section", required=True, choices=sorted(SECTION_ITEMS))
    m.add_argument("--run", type=int, required=True)
    m.add_argument("--out", required=True)
    p = sub.add_parser("profile")
    p.add_argument("--bench-py", required=True)
    p.add_argument("--section", required=True, choices=PROFILE_SECTIONS)
    p.add_argument("--run", type=int, required=True)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.self_test:
        self_test()
    elif args.cmd == "env":
        cmd_env(args)
    elif args.cmd == "measure":
        cmd_measure(args)
    elif args.cmd == "profile":
        cmd_profile(args)
    else:
        ap.print_help()
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
