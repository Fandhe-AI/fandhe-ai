#!/usr/bin/env python3
"""PyTorch(cuBLAS) の MLP 推論 2 形状を CUDA event で 1 launch ずつ挟んで測る（#4147）。

役割: G-CUDA-INF の診断 (3) の相手側「純カーネル」(clock=cuda_event)。fandhe 側は
  crates/backend-cuda/examples/cuda_mlp_event_bench_4147.rs が同じ境界（event 対で 1 launch、
  H2D・出力確保・event 生成は区間外、warmup/計測は bench_py.py の WARMUP/ITERS）で測る。
呼び出し元: run_round.sh（GB10）。出力 JSONL を to_values_4147.py が読む。
既存の opp_kernel_cuda_diag.py（RULE 固定済み）は変更せず、importlib で読み込んで event_sample を再利用する
（正方 GEMM のみで MLP 形状を持たないため、本スクリプトで形状だけ足す。複製しない）。
bench_py.py の sha256 を照合してから読み込む（不一致は停止）。torch は 2.14.0（+cu130 など）を要求する。
"""
import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

ISSUE = 4147
SHAPES = (("l1_fwd", 64, 256, 784), ("l2_fwd", 64, 10, 256))
SEED_A, SEED_B = 0x4147A, 0x4147B
REQUIRED_KEYS = ("issue", "device", "shape_id", "phase", "median_s", "q1_s", "q3_s", "run",
                 "torch_version", "cuda_version", "bench_py_sha256", "gpu_name")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def validate_row(r):
    for k in REQUIRED_KEYS:
        if k not in r:
            raise ValueError(f"missing key {k}")
    for k in ("median_s", "q1_s", "q3_s"):
        v = r[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
            raise ValueError(f"{k} must be finite > 0: {v!r}")
    return r


def cmd_measure(args):
    here = Path(__file__).resolve().parent
    # 既存ツールは --fc-dir（framework-compare）から読む。変更しない。
    fc = Path(args.fc_dir).resolve()
    base = load_module(fc / "opp_kernel_torch_diag.py", "opp_kernel_torch_diag_base")
    cuda = load_module(fc / "opp_kernel_cuda_diag.py", "opp_kernel_cuda_diag_base")
    bp = base.load_bench_py(args.bench_py)
    import torch

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"出力が既にある（上書きしない）: {out}")
    if not torch.__version__.startswith("2.14.0"):
        raise SystemExit(f"torch の版が RULE と不一致: {torch.__version__}")
    if not torch.cuda.is_available():
        print("CUDA が使えない: unavailable", file=sys.stderr)
        raise SystemExit(3)
    # 既定精度（fp32・TF32 なし）を明示する。fandhe 側の f32 SIMT 経路と同じ土俵（RULE 項 4）。
    torch.backends.cuda.matmul.allow_tf32 = False
    sha = base.sha256_file(Path(args.bench_py).resolve())
    gpu = torch.cuda.get_device_name(0)
    rows = []
    for sid, m, n, k in SHAPES:
        a = torch.from_numpy(bp.fill_vec(SEED_A, m * k).reshape(m, k)).to("cuda")
        b = torch.from_numpy(bp.fill_vec(SEED_B, k * n).reshape(k, n)).to("cuda")
        # 出力は形状ごとに事前確保し out= で渡す（呼び出しごとの確保を event 区間に入れない。fandhe 側と同じ境界）
        c = torch.empty((m, n), dtype=a.dtype, device="cuda")
        mm = lambda: torch.mm(a, b, out=c)  # noqa: E731
        med, q1, q3 = cuda.event_sample(torch, mm, bp, reps=1)
        rows.append(validate_row({
            "issue": ISSUE, "device": "cuda", "shape_id": sid, "phase": "mm_event", "median_s": med,
            "q1_s": q1, "q3_s": q3, "run": args.run, "torch_version": torch.__version__,
            "cuda_version": str(torch.version.cuda), "bench_py_sha256": sha, "gpu_name": gpu,
        }))
    with open(out, "x") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows -> {out}")


def self_test():
    ok = {"issue": 1, "device": "cuda", "shape_id": "l1_fwd", "phase": "mm_event", "median_s": 1e-5,
          "q1_s": 1e-5, "q3_s": 2e-5, "run": 1, "torch_version": "x", "cuda_version": "x",
          "bench_py_sha256": "x", "gpu_name": "x"}
    validate_row(dict(ok))
    for bad in ({"median_s": float("nan")}, {"median_s": 0}, {"median_s": True}):
        try:
            validate_row({**ok, **bad})
        except ValueError:
            continue
        raise AssertionError(f"should fail: {bad}")
    bad = dict(ok)
    del bad["gpu_name"]
    try:
        validate_row(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("missing key must fail")
    assert [s[0] for s in SHAPES] == ["l1_fwd", "l2_fwd"]
    print("self-test ok")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("measure")
    p.add_argument("--bench-py", required=True)
    p.add_argument("--fc-dir", required=True, help="scripts/bench/framework-compare の絶対パス")
    p.add_argument("--run", type=int, required=True)
    p.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.self_test:
        self_test()
    elif args.cmd == "measure":
        cmd_measure(args)
    else:
        ap.print_help()
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
