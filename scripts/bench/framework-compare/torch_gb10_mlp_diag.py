#!/usr/bin/env python3
"""GB10 の PyTorch CPU・CUDA の MLP 窓を区間に分けて測る診断（イシュー #4149）。

## 位置づけ

`docs/perf/logs/lowlayer-diagnosis-2026-09-12/bench_py.py` の PyTorch 窓には、BLAS／cuBLAS の純カーネル時間の
ほかに torch の dispatch／autograd／Python 固定費、CUDA では launch・同期・D2H が全部入っている。
本スクリプトは GB10 上の窓を区間に分け、各値に証拠タグ（実測／推定／未測定）を付けて記録する。
`torch_cpu_mlp_diag.py`（#4148・Mac mini 向け）の GB10 派生で、**bench_py.py・相手側ベンチ・判定規則・fandhe 本体は変えない**。
判定・閾値は設けない（record_only）。fandhe 側との比較・分類は #4168・#4150 の担当でここでは行わない。

## 再利用（複製しない）

- `torch_cpu_mlp_diag.py`（#4148）の純粋なヘルパー（stats_of・sample・per_call・floors・shape_inputs・batch10_out・
  composite_close_np・accel_batch10・build_train・build_infer・MLP_SHAPES）を importlib で読む。
  それらは ISSUE を持たないため 4149 でも流用できる。ISSUE 依存の行・env・profile の検証だけ本ファイルで再定義する。
- `opp_kernel_cuda_diag.py`（#3179）の `event_sample`（CUDA event 対での計時）。
- 両ヘルパーの sha256 は env.json に記録し、RULE の PARAMS と突き合わせる（集計側）。
- bench_py.py は `--bench-py` で受け取り、RULE に固定した sha256 と一致したものだけ読む。

## 呼び出し元

- GB10 の torch 2.14.0 入り python 上で `docs/perf/logs/diag-0140-torch-gb10-mlp-4149/run_round.sh` が
  env → CPU measure → CUDA measure → profile を順に起動する。
- 結果は同ディレクトリの `aggregate_4149.py` が厳格パースして区間表にする。
- CI で動くのは torch／numpy 不要の部分だけ（`torch_gb10_mlp_diag_test.py`・`--self-test`）。

## サブコマンド

  env     --bench-py PATH --out F.json
  measure --bench-py PATH --section SECTION --run K --out F.jsonl
  profile --bench-py PATH --section cpu_train|cpu_infer|cuda_infer --run K --steps N --out F.json
  --self-test

SECTION は cpu_train／cpu_infer／cpu_shapes／cuda_infer／cuda_fixed／cuda_shapes。

## 計時腕と profiler 腕の分離

`measure`（計時腕）は profiler・LD_PRELOAD を一切有効にしない。`profile` は別腕で、op 数・kernel 名と回数・CUDA API
呼び出しの種別と回数（cudaGraphLaunch の有無を含む）だけを記録し、時間値は混ぜない。CUPTI が使えず CUDA 活動が
取れない場合は unavailable＋理由として記録し、推定値で埋めない。cuBLAS／sgemm_ の呼び出し回数は profiler ではなく
`probe/` の観測専用 shim（計時外）で決定的に数える。

## 安全面

NVPL は固定 soname（`libnvpl_blas_lp64_gomp.so.0`）・固定シンボル（`cblas_sgemm`）だけを ctypes で呼ぶ。
soname・シンボルは環境変数や引数から組み立てない。ロードは `import torch` 後に `RTLD_NOLOAD` で既ロード分だけを引く。
外部コマンドは使わない（nvidia-smi 等は run_round.sh 側が固定引数で起動する）。
"""
import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

ISSUE = 4149
TORCH_VERSION_PREFIX = "2.14.0"
# 固定 soname と固定シンボル。外部入力から組み立てない（A03）。
NVPL_SONAME = "libnvpl_blas_lp64_gomp.so.0"
NVPL_SYMBOL = "cblas_sgemm"
HERE = Path(__file__).resolve().parent

CPU_SECTIONS = ("cpu_train", "cpu_infer", "cpu_shapes")
CUDA_SECTIONS = ("cuda_infer", "cuda_fixed", "cuda_shapes")
SHAPE_IDS = ("l1_fwd", "l1_dw", "l1_dx", "l2_fwd", "l2_dw", "l2_dx")
TRAIN_GEMM_FWD = ("l1_fwd", "l2_fwd")
TRAIN_GEMM_BWD = ("l2_dw", "l2_dx", "l1_dw")
INFER_GEMM = ("l1_fwd", "l2_fwd")

# section -> 必須 item 集合（aggregate_4149.py と同一。テストで突き合わせる）。
SECTION_ITEMS = {
    "cpu_train": ("window", "fwd", "grad", "update", "loss_item", "ew_fwd", "dispatch_floor_add", "dispatch_floor_mm"),
    "cpu_infer": ("window", "fwd_compute", "readout", "checksum", "ew_fwd", "dispatch_floor_add", "dispatch_floor_mm"),
    "cpu_shapes": tuple(f"{p}_{s}" for s in SHAPE_IDS for p in ("mm_pure", "mm_batch10", "direct_batch10")),
    "cuda_infer": ("window", "issue", "readout_sync", "checksum", "gpu_span_event", "graph_replay_ref"),
    "cuda_fixed": ("sync_idle", "tiny_roundtrip", "launch_floor_add", "event_record_pair"),
    "cuda_shapes": tuple(f"mm_event_{s}" for s in SHAPE_IDS),
}
PROFILE_SECTIONS = ("cpu_train", "cpu_infer", "cuda_infer")
TAGS = ("実測", "推定", "未測定")
REQUIRED_KEYS = (
    "issue", "section", "item", "status", "tag", "median_s", "q1_s", "q3_s", "warmup", "iters", "run",
    "torch_threads", "torch_version", "numpy_version", "bench_py_sha256", "diag_script_sha256",
)
ENV_KEYS = ("issue", "torch_version", "numpy_version", "torch_threads", "thread_env", "torch_config",
            "mkldnn_available", "nvpl_direct_available", "cuda_device_name", "cuda_version",
            "fp32_matmul_precision", "allow_tf32", "is_current_stream_capturing",
            "bench_py_sha256", "diag_script_sha256", "helper_cpu_diag_sha256", "helper_cuda_diag_sha256")
PROFILE_KEYS = ("issue", "section", "run", "steps", "torch_version", "n_ops_aten_total", "n_ops_aten_toplevel",
                "mm_has_child_ops", "ops", "cuda", "cuda_unavailable_reason", "note",
                "bench_py_sha256", "diag_script_sha256")
THREAD_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NVPL_NUM_THREADS")
# profile 腕で数える CUDA ランタイム API（固定集合）。
CUDA_API_NAMES = ("cudaLaunchKernel", "cudaStreamSynchronize", "cudaMemcpyAsync", "cudaGraphLaunch",
                  "cudaDeviceSynchronize", "cudaEventRecord", "cuLaunchKernel", "cuLaunchKernelEx")


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cpu_diag():
    """#4148 の純粋ヘルパー群。ISSUE 非依存の関数だけを使う。"""
    return _load_module("torch_cpu_mlp_diag_helper", HERE / "torch_cpu_mlp_diag.py")


def cuda_diag():
    return _load_module("opp_kernel_cuda_diag_helper", HERE / "opp_kernel_cuda_diag.py")


def self_sha256():
    import hashlib
    return hashlib.sha256(Path(os.path.abspath(__file__)).read_bytes()).hexdigest()


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
    if d["section"] == "cuda_infer":
        # CUDA 活動は取れたか、取れない理由があるかのどちらか（沈黙の欠落を許さない）。
        if d["cuda"] is None and not (isinstance(d["cuda_unavailable_reason"], str) and d["cuda_unavailable_reason"]):
            raise ValueError("cuda_infer の profile は cuda か cuda_unavailable_reason が必要")
        if d["cuda"] is not None and (not isinstance(d["cuda"].get("graph_launch"), bool) or "kernels" not in d["cuda"]):
            raise ValueError("cuda の構造が不正")
    return d


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


# ---- NVPL cblas_sgemm の直呼び（固定 soname・固定シンボル。既ロード分のみ。不在は None） --------------
def load_nvpl_direct():
    """import torch 後に RTLD_NOLOAD で既ロードの NVPL を引く。新規ロードはしない（torch が使っていなければ None）。"""
    import ctypes
    try:
        lib = ctypes.CDLL(NVPL_SONAME, mode=os.RTLD_NOW | os.RTLD_NOLOAD)
        fn = getattr(lib, NVPL_SYMBOL)
    except (OSError, AttributeError):
        return None
    fn.restype = None
    fn.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                   ctypes.c_float, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
                   ctypes.c_float, ctypes.c_void_p, ctypes.c_int]
    return fn


def make_ctx(bp, torch, np, bench_py_path):
    h = cpu_diag()
    return {"threads": torch.get_num_threads(), "torch_version": torch.__version__, "numpy_version": np.__version__,
            "bench_sha": h.sha256_file(Path(bench_py_path).resolve()), "self_sha": self_sha256()}


# ---- CUDA 区間のクロージャ -------------------------------------------------------------------------
def build_cuda_infer(bp, torch, np):
    be = bp.Torch("cuda")
    w1, b1, w2, b2, x, _y = bp.mlp_init()
    be.train_setup(w1, b1, w2, b2)
    X = be.upload(x)
    pw1, pb1, pw2, pb2 = be.p
    t = torch

    def fwd():
        with t.no_grad():
            return t.relu(X @ pw1 + pb1) @ pw2 + pb2
    out = fwd()
    t.cuda.synchronize()
    host = out.cpu().numpy()
    return {"be": be, "X": X, "fwd": fwd, "out": out, "host": host,
            "window": lambda: bp.checksum(be.forward_to_host(X)),
            "readout_sync": lambda: out.cpu().numpy(),
            "checksum": lambda: bp.checksum(host)}


def issue_sample(torch, fn, bp):
    """forward を enqueue するまでの CPU 時間（同期なし）。計時外で毎回同期してキューを空にする。"""
    import time
    for _ in range(bp.WARMUP):
        torch.cuda.synchronize()
        fn()
    secs = []
    for _ in range(bp.ITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        secs.append(time.perf_counter() - t0)
    torch.cuda.synchronize()
    h = cpu_diag()
    return h.stats_of(secs)


def graph_replay_sample(torch, c, bp):
    """forward を CUDAGraph に capture し replay＋D2H を測る参考腕。失敗は (None, 理由)。"""
    import time
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                c["fwd"]()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_out = c["fwd"]()
        torch.cuda.synchronize()
        h = cpu_diag()

        def step():
            g.replay()
            return static_out.cpu().numpy()
        return h.sample(step, bp.WARMUP, bp.ITERS), None
    except Exception as e:  # capture の失敗は参考腕の unavailable にとどめる（窓の値には影響しない）
        return None, f"CUDAGraph capture／replay 失敗: {type(e).__name__}"


def cuda_fixed_stats(torch, bp):
    h = cpu_diag()
    reps = h.FLOOR_REPS
    x = torch.ones(1, device="cuda")
    torch.cuda.synchronize()
    out = {}
    out["sync_idle"] = h.sample(lambda: torch.cuda.synchronize(), bp.WARMUP, bp.ITERS, prep=None)

    def tiny():
        (x + x).cpu()
    out["tiny_roundtrip"] = h.sample(tiny, bp.WARMUP, bp.ITERS)
    # enqueue だけを計る。毎回、計時外で同期してキューを空にする。
    out["launch_floor_add"] = h.per_call(
        h.sample(lambda _a: [x + x for _ in range(reps)], bp.WARMUP, bp.ITERS, prep=torch.cuda.synchronize), reps)
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    e.record()
    torch.cuda.synchronize()

    def rec():
        s.record()
        e.record()
    out["event_record_pair"] = h.sample(lambda _a: rec(), bp.WARMUP, bp.ITERS, prep=torch.cuda.synchronize)
    return out


def cmd_measure(args):
    h = cpu_diag()
    bp = h.load_bench_py(args.bench_py)
    import numpy as np
    import torch
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"出力が既にある（上書きしない）: {out}")
    ctx = make_ctx(bp, torch, np, args.bench_py)
    W, I = bp.WARMUP, bp.ITERS
    TW, TI = bp.TRAIN_WARMUP, bp.TRAIN_STEPS - bp.TRAIN_WARMUP
    sec = args.section
    if sec not in SECTION_ITEMS:
        raise SystemExit(f"不明な section: {sec}")
    rows = []

    def add(item, stats, tag="実測", warmup=W, iters=I, reason=None):
        rows.append(make_row(ctx, sec, item, stats, tag, warmup, iters, args.run, reason))

    if sec == "cpu_train":
        c = h.build_train(bp, torch, np)
        fl = h.floors(torch)
        add("window", h.sample(c["window"], TW, TI), warmup=TW, iters=TI)
        add("fwd", h.sample(c["fwd"], TW, TI), warmup=TW, iters=TI)
        prep, g = c["grad"]
        add("grad", h.sample(g, TW, TI, prep=prep), warmup=TW, iters=TI)
        add("update", h.sample(c["update"], TW, TI), warmup=TW, iters=TI)
        add("loss_item", h.sample(c["loss_item"], TW, TI), warmup=TW, iters=TI)
        add("ew_fwd", h.sample(c["ew_fwd"], TW, TI), warmup=TW, iters=TI)
        add("dispatch_floor_add", fl["dispatch_floor_add"], warmup=20, iters=20)
        add("dispatch_floor_mm", fl["dispatch_floor_mm"], warmup=20, iters=20)
    elif sec == "cpu_infer":
        c = h.build_infer(bp, torch, np)
        fl = h.floors(torch)
        for item in ("window", "fwd_compute", "readout", "checksum", "ew_fwd"):
            add(item, h.sample(c[item], W, I))
        add("dispatch_floor_add", fl["dispatch_floor_add"], warmup=20, iters=20)
        add("dispatch_floor_mm", fl["dispatch_floor_mm"], warmup=20, iters=20)
    elif sec == "cpu_shapes":
        fn = load_nvpl_direct()
        for sid, m, n, k, trans in h.MLP_SHAPES:
            a_np, b_np, a, b = h.shape_inputs(bp, np, torch, sid, m, n, k, trans)
            ref = torch.mm(a, b).numpy()
            add(f"mm_pure_{sid}", h.sample(lambda: torch.mm(a, b), W, I))
            add(f"mm_batch10_{sid}", h.per_call(h.sample(h.batch10_out(torch, a, b), W, I), 10))
            if fn is None:
                add(f"direct_batch10_{sid}", None, reason="NVPL cblas_sgemm が既ロードでない（torch が NVPL を使っていない）")
            else:
                st = h.accel_batch10(fn, a_np, b_np, m, n, k, trans, ref, W, I)
                if st is None:
                    add(f"direct_batch10_{sid}", None, reason="cblas_sgemm の結果が torch.mm と一致しない")
                else:
                    add(f"direct_batch10_{sid}", st)
    elif sec == "cuda_infer":
        E = cuda_diag()
        c = build_cuda_infer(bp, torch, np)
        add("window", h.sample(c["window"], W, I))
        add("issue", issue_sample(torch, c["fwd"], bp))
        add("readout_sync", h.sample(c["readout_sync"], W, I))
        add("checksum", h.sample(c["checksum"], W, I))
        add("gpu_span_event", E.event_sample(torch, c["fwd"], bp))
        st, why = graph_replay_sample(torch, c, bp)
        add("graph_replay_ref", st, reason=why)
    elif sec == "cuda_fixed":
        for item, st in cuda_fixed_stats(torch, bp).items():
            add(item, st)
    elif sec == "cuda_shapes":
        E = cuda_diag()
        for sid, m, n, k, trans in h.MLP_SHAPES:
            _a, _b, a, b = h.shape_inputs(bp, np, torch, sid, m, n, k, trans)
            a, b = a.cuda(), b.cuda()
            add(f"mm_event_{sid}", E.event_sample(torch, lambda: torch.mm(a, b), bp))
    got = {r["item"] for r in rows}
    if got != set(SECTION_ITEMS[sec]):
        raise SystemExit(f"item 集合が不完全: 不足={set(SECTION_ITEMS[sec]) - got} 余分={got - set(SECTION_ITEMS[sec])}")
    with open(out, "x") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows -> {out}")


def cmd_env(args):
    h = cpu_diag()
    bp = h.load_bench_py(args.bench_py)  # noqa: F841  sha 照合のため
    import numpy as np
    import torch
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"出力が既にある（上書きしない）: {out}")
    if not torch.cuda.is_available():
        raise SystemExit("torch の CUDA が利用できない")
    m = torch.backends.cuda.matmul
    d = {"issue": ISSUE, "torch_version": torch.__version__, "numpy_version": np.__version__,
         "torch_threads": torch.get_num_threads(),
         "thread_env": {k: os.environ.get(k) for k in THREAD_ENV_VARS},
         "torch_config": torch.__config__.show(),
         "mkldnn_available": bool(torch.backends.mkldnn.is_available()),
         "nvpl_direct_available": load_nvpl_direct() is not None,
         "cuda_device_name": torch.cuda.get_device_name(0), "cuda_version": torch.version.cuda,
         "fp32_matmul_precision": torch.get_float32_matmul_precision(),
         "allow_tf32": bool(m.allow_tf32),
         "is_current_stream_capturing": bool(torch.cuda.is_current_stream_capturing()),
         "bench_py_sha256": h.sha256_file(Path(args.bench_py).resolve()), "diag_script_sha256": self_sha256(),
         "helper_cpu_diag_sha256": h.sha256_file(HERE / "torch_cpu_mlp_diag.py"),
         "helper_cuda_diag_sha256": h.sha256_file(HERE / "opp_kernel_cuda_diag.py")}
    validate_env(d)
    # bench_py は精度設定を変えない。既定（highest／TF32 無効）以外は想定外として停止する（#3179 RULE の読み戻しの作法）。
    if d["fp32_matmul_precision"] != "highest" or d["allow_tf32"]:
        raise SystemExit(f"fp32 matmul 精度が既定と違う: {d['fp32_matmul_precision']} allow_tf32={d['allow_tf32']}")
    with open(out, "x") as f:
        json.dump(d, f, ensure_ascii=False, indent=1)
    print(f"wrote env -> {out}")


def _cuda_profile(prof):
    """Kineto の key_averages から kernel 名・回数と CUDA API 回数を取る。取れなければ (None, 理由)。"""
    from torch.autograd import DeviceType
    kernels, api = {}, {}
    for e in prof.key_averages():
        if e.device_type == DeviceType.CUDA:
            kernels[e.key] = kernels.get(e.key, 0) + e.count
        elif e.key in CUDA_API_NAMES:
            api[e.key] = api.get(e.key, 0) + e.count
    if not kernels and not api:
        return None, "CUPTI から CUDA 活動が得られなかった（kernel も API 呼び出しも 0 件）"
    return {"kernels": kernels, "api": api}, None


def cmd_profile(args):
    h = cpu_diag()
    bp = h.load_bench_py(args.bench_py)
    import numpy as np
    import torch
    from torch.profiler import ProfilerActivity, profile
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"出力が既にある（上書きしない）: {out}")
    sec = args.section
    if sec == "cpu_train":
        step = h.build_train(bp, torch, np)["window"]
    elif sec == "cpu_infer":
        step = h.build_infer(bp, torch, np)["window"]
    elif sec == "cuda_infer":
        step = build_cuda_infer(bp, torch, np)["window"]
    else:
        raise SystemExit(f"profile に使えない section: {sec}")
    for _ in range(bp.WARMUP):
        step()
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if sec == "cuda_infer" else [])
    with profile(activities=acts) as prof:
        for _ in range(args.steps):
            step()
        if sec == "cuda_infer":
            torch.cuda.synchronize()
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
    cuda, why = (None, None)
    if sec == "cuda_infer":
        try:
            cuda, why = _cuda_profile(prof)
        except Exception as ex:  # CUPTI／Kineto が GB10 で使えない場合は理由を残す（推定値で埋めない）
            cuda, why = None, f"CUDA profile 取得失敗: {type(ex).__name__}"
        if cuda is not None:
            # 1 step あたりの回数へ正規化し、cudaGraphLaunch の有無を事実として残す。
            cuda = {"kernels": {k: v / args.steps for k, v in cuda["kernels"].items()},
                    "api": {k: v / args.steps for k, v in cuda["api"].items()},
                    "graph_launch": cuda["api"].get("cudaGraphLaunch", 0) > 0}
    d = {"issue": ISSUE, "section": sec, "run": args.run, "steps": args.steps, "torch_version": torch.__version__,
         "n_ops_aten_total": max(1, len(evs) // args.steps), "n_ops_aten_toplevel": max(1, len(top) // args.steps),
         "mm_has_child_ops": mm_has_child, "ops": ops, "cuda": cuda, "cuda_unavailable_reason": why,
         "note": "profiler 下の値。時間は計時腕の値と混ぜない（op 数・kernel／API の回数だけを内訳に使う）",
         "bench_py_sha256": h.sha256_file(Path(args.bench_py).resolve()), "diag_script_sha256": self_sha256()}
    validate_profile(d)
    with open(out, "x") as f:
        json.dump(d, f, ensure_ascii=False, indent=1)
    print(f"wrote profile -> {out}")


def self_test():
    """torch／numpy 不要の検査（検証関数・item 集合）。"""
    ctx = {"threads": 4, "torch_version": "2.14.0", "numpy_version": "2", "bench_sha": "s", "self_sha": "d"}
    r = make_row(ctx, "cuda_fixed", "sync_idle", (1e-5, 9e-6, 1.1e-5), "実測", 20, 20, 1)
    assert r["status"] == "ok"
    u = make_row(ctx, "cuda_infer", "graph_replay_ref", None, "実測", 20, 20, 1, reason="x")
    assert u["median_s"] is None and u["tag"] == "未測定"
    for bad in (dict(r, median_s=float("nan")), dict(r, item="nope"), dict(r, section="nope"), dict(r, run=0),
                dict(u, median_s=1.0), dict(u, reason=""), dict(r, issue=4148)):
        try:
            validate_row(bad)
        except ValueError:
            continue
        raise AssertionError(f"検証をすり抜けた: {bad}")
    try:
        validate_row({k: v for k, v in r.items() if k != "tag"})
        raise AssertionError("欠損キーが通った")
    except ValueError:
        pass
    assert len(SECTION_ITEMS["cpu_shapes"]) == 18 and len(SECTION_ITEMS["cuda_shapes"]) == 6
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
