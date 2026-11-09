#!/usr/bin/env python3
"""Mac mini の PyTorch MPS の MLP train 1 step を区間に分けて測る診断（イシュー #4154）。

## 位置づけ

M-MTL-TRN（Mac mini・train Metal・相手は PyTorch MPS）の窓は、`docs/perf/logs/lowlayer-diagnosis-2026-09-12/bench_py.py`
の `Torch.train_step` を 1 step ずつ計った中央値である。この窓には「host が op ごとに MPS を呼んで GPU へ仕事を出す時間」
「GPU の実行と完了待ち」「`loss.item()` の同期と Python 側の固定費」が全部入っている。本スクリプトは窓を区間に分ける。
fandhe 側の内訳（`docs/compat-resident-train-step-decision.md` §17.6・#3612）と同じ定義で比べるため、時計が揃う wall の
3 区間（発行 I・待ち S・残差 R）を主軸とし、MPS イベントの GPU span は補助（別の時計）として分離して記録する。
**bench_py.py・相手側ベンチ・判定規則・fandhe 本体は変えない**。bench_py.py は `--bench-py` で受け取り、RULE.txt に固定した
sha256 と一致したものだけを importlib で読み、`mlp_init`／`Torch`／`LR`／`TRAIN_STEPS`／`TRAIN_WARMUP` を再利用する。

## 呼び出し元

- Mac mini の venv-torch 上で `docs/perf/logs/diag-0140-m-mtl-trn-4154/run_round.sh` が env → measure（split・event・floor）を起動する。
- 結果は `docs/perf/logs/diag-0140-m-mtl-trn-4154/aggregate_4154.py` が厳格パースして判定する。
- CI で動くのは torch／numpy 不要の部分だけ（`torch_mps_train_diag_test.py`・`--self-test`）。実機（MPS）経路は Linux で検証できない。

## サブコマンド

  env     --bench-py PATH --out F.json
  measure --bench-py PATH --arm split|event|floor --run K --out F.jsonl
  --self-test

## 腕の分離（摂動を混ぜない）

- `split`: step 本体（forward・`autograd.grad`・SGD 更新）を出し終えた時刻 t1 と、`torch.mps.synchronize()` が返った時刻 t2 を
  wall で取る。I = t1 - t0、S = t2 - t1。`loss.item()` は計時の外（窓の `.item()` は同期後なので読み出しだけ）。
- `event`: step の前後で `torch.mps.Event(enable_timing=True)` を record し GPU の span を取る。**イベントの record がコマンド
  バッファの区切りを変えうる**ため、この腕の値は split・bench_py の窓と混ぜない。時計は `mps_event`（共有イベントを
  host が観測した時刻の差と推定。GPU タイムスタンプとは別の時計。fandhe の gpu_busy〈gpu_timestamp〉と揃わない）。
- `floor`: 何も積まない同期と、1 要素 add + `.item()` の往復（同期の固定費の下限）。

## 安全面

外部コマンドは起動しない。パス・シンボルは引数や環境変数から組み立てない。出力ファイルが既にあれば上書きせず停止する。
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path

ISSUE = 4154
# RULE.txt の bench_py_sha256 と同じ値（#3179・#4148 と同一。0.11.0 以降の計測で使った bench_py.py）。
BENCH_PY_SHA256 = "9fd2c08f37d66b5bb41ececab8a5b3daaa36b55609770502ed3b35ae2ce2409b"
TORCH_VERSION_PREFIX = "2.14.0"
FLOOR_REPS = 100
FLOOR_WARMUP = 20

# 腕 -> item -> 時計。aggregate_4154.py の ARM_ITEMS と同一（テストで突き合わせる）。
ARM_ITEMS = {
    "split": {"issue_wall": "wall", "sync_wall": "wall", "step_wall": "wall"},
    "event": {"gpu_span": "mps_event", "event_step_wall": "wall"},
    "floor": {"floor_empty_sync": "wall", "floor_roundtrip": "wall"},
}
TAGS = ("実測", "推定", "未測定")
REQUIRED_KEYS = (
    "issue", "arm", "item", "clock", "status", "tag", "median_s", "q1_s", "q3_s", "warmup", "iters", "run",
    "torch_version", "bench_py_sha256", "diag_script_sha256", "checksum",
)
ENV_KEYS = ("issue", "torch_version", "numpy_version", "mps_available", "mps_event_available", "macos_version",
            "bench_py_sha256", "diag_script_sha256")


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def self_sha256():
    return sha256_file(os.path.abspath(__file__))


def quantile(secs, p):
    """bench_py.quantile と同じ線形補間（torch 不在の self-test 用）。"""
    s = sorted(secs)
    idx = p * (len(s) - 1)
    lo, hi = int(math.floor(idx)), int(math.ceil(idx))
    return s[lo] * (1 - (idx - lo)) + s[hi] * (idx - lo)


def stats_of(secs):
    return statistics.median(secs), quantile(secs, 0.25), quantile(secs, 0.75)


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
    """計時腕 JSONL 1 行の厳格検証。欠損・不明な腕／item・時計不一致・非有限・unavailable の値混入は例外。"""
    for k in REQUIRED_KEYS:
        if k not in row:
            raise ValueError(f"キー欠損: {k}")
    if row["issue"] != ISSUE:
        raise ValueError("issue 不一致")
    if row["arm"] not in ARM_ITEMS:
        raise ValueError(f"不明な arm: {row['arm']}")
    if row["item"] not in ARM_ITEMS[row["arm"]]:
        raise ValueError(f"不明な item: {row['arm']}/{row['item']}")
    if row["clock"] != ARM_ITEMS[row["arm"]][row["item"]]:
        raise ValueError(f"時計タグが定義と違う: {row['item']}={row['clock']}")
    if row["tag"] not in TAGS:
        raise ValueError(f"不明な tag: {row['tag']}")
    if not isinstance(row["run"], int) or isinstance(row["run"], bool) or row["run"] < 1:
        raise ValueError("run が不正")
    if row["checksum"] is not None and not _finite_num(row["checksum"]):
        raise ValueError("checksum が非有限")
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
    if d["mps_available"] is not True:
        raise ValueError("MPS が使えない")
    return d


def make_row(ctx, arm, item, stats, tag, warmup, iters, run, checksum=None, reason=None):
    base = {
        "issue": ISSUE, "arm": arm, "item": item, "clock": ARM_ITEMS[arm][item], "warmup": warmup, "iters": iters,
        "run": run, "torch_version": ctx["torch_version"], "bench_py_sha256": ctx["bench_sha"],
        "diag_script_sha256": ctx["self_sha"], "checksum": checksum,
    }
    if stats is None:
        base.update(status="unavailable", tag="未測定", median_s=None, q1_s=None, q3_s=None, reason=reason or "unavailable")
    else:
        base.update(status="ok", tag=tag, median_s=stats[0], q1_s=stats[1], q3_s=stats[2])
    return validate_row(base)


def make_ctx(torch, bench_py_path):
    return {"torch_version": torch.__version__, "bench_sha": sha256_file(Path(bench_py_path).resolve()),
            "self_sha": self_sha256()}


# ---- step 本体（bench_py.Torch.train_step の forward・grad・SGD 更新までを同一手順で再現） -------------
# train_step は戻り値の loss.item() で同期するため、区間を分けるには本体を切り出す必要がある。
# 手順が bench_py と食い違うと区間の意味が崩れるので、aggregate_4154.py が TW の checksum と TS の最終 loss の
# 一致を確度の条件に使う（MPS の決定性は保証されないため hard 条件にはしない）。
def step_body(bp, be, X, Y):
    t = be.t
    w1, b1, w2, b2 = be.p
    h = t.relu(X @ w1 + b1)
    pred = h @ w2 + b2
    loss = ((pred - Y) ** 2).mean()
    gs = t.autograd.grad(loss, be.p)
    with t.no_grad():
        for p, g in zip(be.p, gs):
            p.sub_(g * bp.LR)
    return loss


def setup(bp):
    import torch
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS が使えない（Mac mini の venv-torch で実行すること）")
    be = bp.Torch("metal")
    w1, b1, w2, b2, x, y = bp.mlp_init()
    be.train_setup(w1, b1, w2, b2)
    return torch, be, be.upload(x), be.upload(y)


def run_split(bp, args):
    torch, be, X, Y = setup(bp)
    ctx = make_ctx(torch, args.bench_py)
    iss, syn, tot, last = [], [], [], None
    for s in range(bp.TRAIN_STEPS):
        t0 = time.perf_counter()
        loss = step_body(bp, be, X, Y)
        t1 = time.perf_counter()
        torch.mps.synchronize()
        t2 = time.perf_counter()
        last = float(loss.item())  # 計時の外（同期後なので読み出しだけ）
        if s >= bp.TRAIN_WARMUP:
            iss.append(t1 - t0)
            syn.append(t2 - t1)
            tot.append(t2 - t0)
    if not math.isfinite(last):
        raise SystemExit("loss が非有限")
    w, n = bp.TRAIN_WARMUP, bp.TRAIN_STEPS - bp.TRAIN_WARMUP
    return [make_row(ctx, "split", name, stats_of(v), "実測", w, n, args.run, checksum=last)
            for name, v in (("issue_wall", iss), ("sync_wall", syn), ("step_wall", tot))]


def run_event(bp, args):
    torch, be, X, Y = setup(bp)
    ctx = make_ctx(torch, args.bench_py)
    w, n = bp.TRAIN_WARMUP, bp.TRAIN_STEPS - bp.TRAIN_WARMUP
    try:
        Event = torch.mps.Event
        spans, walls, last = [], [], None
        for s in range(bp.TRAIN_STEPS):
            e0, e1 = Event(enable_timing=True), Event(enable_timing=True)
            t0 = time.perf_counter()
            e0.record()
            loss = step_body(bp, be, X, Y)
            e1.record()
            last = float(loss.item())  # 同期。elapsed_time は完了後に読む
            dt = time.perf_counter() - t0
            ms = e0.elapsed_time(e1)
            if s >= bp.TRAIN_WARMUP:
                spans.append(float(ms) / 1000.0)
                walls.append(dt)
        if not all(math.isfinite(v) and v > 0 for v in spans):
            raise RuntimeError("elapsed_time が非有限または非正")
    except Exception as e:  # noqa: BLE001 - 到達不能は推定値で埋めず unavailable として記録する
        why = f"torch.mps.Event の計測に失敗: {type(e).__name__}"
        return [make_row(ctx, "event", it, None, "実測", w, n, args.run, reason=why) for it in ARM_ITEMS["event"]]
    return [make_row(ctx, "event", "gpu_span", stats_of(spans), "実測", w, n, args.run, checksum=last),
            make_row(ctx, "event", "event_step_wall", stats_of(walls), "実測", w, n, args.run, checksum=last)]


def run_floor(bp, args):
    torch, be, X, Y = setup(bp)
    ctx = make_ctx(torch, args.bench_py)
    one = torch.ones(1, device="mps")

    def sample(fn):
        for _ in range(FLOOR_WARMUP):
            fn()
        secs = []
        for _ in range(FLOOR_REPS):
            t = time.perf_counter()
            fn()
            secs.append(time.perf_counter() - t)
        return stats_of(secs)

    torch.mps.synchronize()
    empty = sample(torch.mps.synchronize)
    trip = sample(lambda: float((one + one).item()))
    return [make_row(ctx, "floor", "floor_empty_sync", empty, "実測", FLOOR_WARMUP, FLOOR_REPS, args.run),
            make_row(ctx, "floor", "floor_roundtrip", trip, "実測", FLOOR_WARMUP, FLOOR_REPS, args.run)]


def cmd_measure(args):
    bp = load_bench_py(args.bench_py)
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"出力が既にある（上書きしない）: {out}")
    rows = {"split": run_split, "event": run_event, "floor": run_floor}[args.arm](bp, args)
    got = {r["item"] for r in rows}
    if got != set(ARM_ITEMS[args.arm]):
        raise SystemExit(f"item 集合が不完全: {got}")
    with open(out, "x") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows -> {out}")


def cmd_env(args):
    load_bench_py(args.bench_py)
    import numpy as np
    import torch
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"出力が既にある（上書きしない）: {out}")
    d = {"issue": ISSUE, "torch_version": torch.__version__, "numpy_version": np.__version__,
         "mps_available": bool(torch.backends.mps.is_available()),
         "mps_event_available": hasattr(getattr(torch, "mps", None), "Event"),
         "macos_version": platform.mac_ver()[0],
         "bench_py_sha256": sha256_file(Path(args.bench_py).resolve()), "diag_script_sha256": self_sha256()}
    validate_env(d)
    with open(out, "x") as f:
        json.dump(d, f, ensure_ascii=False, indent=1)
    print(f"wrote env -> {out}")


def self_test():
    """torch／numpy 不要の検査（検証関数・統計・腕と item の定義）。"""
    ctx = {"torch_version": "2.14.0", "bench_sha": "s", "self_sha": "d"}
    r = make_row(ctx, "split", "issue_wall", (1e-3, 9e-4, 1.1e-3), "実測", 20, 80, 1, checksum=0.5)
    assert r["status"] == "ok" and r["clock"] == "wall"
    g = make_row(ctx, "event", "gpu_span", (1e-3, 9e-4, 1.1e-3), "実測", 20, 80, 1)
    assert g["clock"] == "mps_event"
    u = make_row(ctx, "event", "gpu_span", None, "実測", 20, 80, 1, reason="x")
    assert u["median_s"] is None and u["tag"] == "未測定"
    for bad in (dict(r, median_s=float("nan")), dict(r, item="nope"), dict(r, arm="nope"), dict(r, run=0),
                dict(r, clock="mps_event"), dict(u, median_s=1.0), dict(u, reason=""), dict(r, checksum=float("inf"))):
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
    assert stats_of([1.0, 2.0, 3.0, 4.0, 5.0]) == (3.0, 2.0, 4.0)
    assert {a: len(i) for a, i in ARM_ITEMS.items()} == {"split": 3, "event": 2, "floor": 2}
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
    m.add_argument("--arm", required=True, choices=sorted(ARM_ITEMS))
    m.add_argument("--run", type=int, required=True)
    m.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.self_test:
        self_test()
    elif args.cmd == "env":
        cmd_env(args)
    elif args.cmd == "measure":
        cmd_measure(args)
    else:
        ap.print_help()
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
