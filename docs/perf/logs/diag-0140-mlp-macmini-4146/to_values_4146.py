#!/usr/bin/env python3
"""ハーネス出力 → round{i}/values.tsv の変換（#4146・標準ライブラリのみ・厳格パース）。

役割: M-CPU-TRN・M-CPU-INF の診断 (3) 用に、round ディレクトリの raw/ にあるハーネス生出力から
aggregate_diag.py が読む values.tsv（cell/quantity/value/clock、単位 ms）を作る。
呼び出し元: 計測した人が各 round の後に手で実行する（README「変換手順」）。集計は aggregate_diag.py のみ。

入力（round 直下の raw/）:
  rust.jsonl        bench-fandhe の --out（device=cpu・size=64・framework=fandhe-ai の train/infer × fresh/reuse の 4 行）
  py.jsonl          bench_py.py の --out（framework=pytorch・device=cpu・size=64 の train/infer の 2 行）
  jobgrid.txt       cpu_jobgrid_bench <arm> threads の標準出力（key=value 形式）
  opp_kernel.jsonl  opp_kernel_torch_diag measure --shapes mlp --arm base --device cpu の出力（phase=mm_pure の行を使う）
純カーネルの境界: GEMM のみ。学習は l1_fwd・l2_fwd・l2_dw・l2_dx・l1_dw の 5 形状（l1_dx は常駐チェーンも
PyTorch も実行しないので含めない）、推論は l1_fwd・l2_fwd。時計タグは両側 wall。
欠損・重複・非有限・形状の過不足は例外で止まる（推定値で埋めない）。
"""
import json
import math
import os
import sys
import tempfile

CELLS = {
    "M-CPU-TRN": ("train", ("l1_fwd", "l2_fwd", "l2_dw", "l2_dx", "l1_dw")),
    "M-CPU-INF": ("infer", ("l1_fwd", "l2_fwd")),
}
ALL_SHAPES = ("l1_fwd", "l1_dw", "l1_dx", "l2_fwd", "l2_dw", "l2_dx")
CLOCK = "wall"


class ConvError(Exception):
    pass


def fin(x, what):
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ConvError(f"{what}: not a number: {x!r}")
    x = float(x)
    if not math.isfinite(x) or x <= 0:
        raise ConvError(f"{what}: must be finite and > 0: {x!r}")
    return x


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                raise ConvError(f"{path}:{n}: bad json: {e}")
            if not isinstance(r, dict):
                raise ConvError(f"{path}:{n}: not an object")
            rows.append(r)
    return rows


def window(rows, framework, task, mode, where):
    hit = [r for r in rows if r.get("framework") == framework and r.get("task") == task
           and r.get("device") == "cpu" and r.get("size") == 64 and r.get("mode") == mode]
    if len(hit) != 1:
        raise ConvError(f"{where}: expected exactly 1 row for {framework}/{task}/{mode}, got {len(hit)}")
    return fin(hit[0].get("median_s"), f"{where}:{framework}/{task}/{mode}") * 1000.0


def jobgrid(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip() or "median_secs=" not in line:
                continue
            kv = dict(t.split("=", 1) for t in line.split() if "=" in t)
            if kv.get("series") != "threads":
                raise ConvError(f"{path}:{n}: series must be threads")
            sh = kv.get("shape")
            if sh in out:
                raise ConvError(f"{path}:{n}: duplicate shape {sh}")
            out[sh] = fin(float(kv["median_secs"]), f"{path}:{n}") * 1000.0
    return out


def opp_kernel(path):
    out = {}
    for r in read_jsonl(path):
        if r.get("phase") != "mm_pure":
            continue
        if r.get("arm") != "base" or r.get("device") != "cpu":
            raise ConvError(f"{path}: arm/device must be base/cpu")
        sh = r.get("shape_id")
        if sh in out:
            raise ConvError(f"{path}: duplicate shape {sh}")
        out[sh] = fin(r.get("median_s"), f"{path}:{sh}") * 1000.0
    return out


def pure_sum(per_shape, shapes, where):
    miss = [s for s in shapes if s not in per_shape]
    if miss:
        raise ConvError(f"{where}: missing shapes {miss}")
    unknown = [s for s in per_shape if s not in ALL_SHAPES and not s.startswith("ref_")]
    if unknown:
        raise ConvError(f"{where}: unknown shapes {unknown}")
    return sum(per_shape[s] for s in shapes)


def convert(rdir):
    raw = os.path.join(rdir, "raw")
    rust = read_jsonl(os.path.join(raw, "rust.jsonl"))
    py = read_jsonl(os.path.join(raw, "py.jsonl"))
    jg = jobgrid(os.path.join(raw, "jobgrid.txt"))
    ok = opp_kernel(os.path.join(raw, "opp_kernel.jsonl"))
    lines = ["cell\tquantity\tvalue\tclock"]
    for cid, (task, shapes) in CELLS.items():
        vals = [
            ("fandhe_window", window(rust, "fandhe-ai", task, "reuse", "rust.jsonl"), "-"),
            ("fandhe_fresh_window", window(rust, "fandhe-ai", task, "fresh", "rust.jsonl"), "-"),
            ("opp_window", window(py, "pytorch", task, "fresh", "py.jsonl"), "-"),
            ("fandhe_pure_kernel", pure_sum(jg, shapes, "jobgrid.txt"), CLOCK),
            ("opp_pure_kernel", pure_sum(ok, shapes, "opp_kernel.jsonl"), CLOCK),
        ]
        for q, v, c in vals:
            lines.append(f"{cid}\t{q}\t{v:.9g}\t{c}")
    return "\n".join(lines) + "\n"


def self_test():
    def w(p, s):
        with open(p, "w", encoding="utf-8") as f:
            f.write(s)

    def build(d, drop=None, dup=None, bad=None):
        os.makedirs(os.path.join(d, "raw"))
        rows = []
        for t, ms in (("train", (0.5, 0.7)), ("infer", (0.2, 0.25))):
            for mode, v in zip(("reuse", "fresh"), ms):
                rows.append({"framework": "fandhe-ai", "task": t, "device": "cpu", "size": 64,
                             "mode": mode, "median_s": v / 1000})
        if drop == "rust":
            rows.pop()
        w(os.path.join(d, "raw/rust.jsonl"), "".join(json.dumps(r) + "\n" for r in rows))
        pr = [{"framework": "pytorch", "task": t, "device": "cpu", "size": 64, "mode": "fresh", "median_s": v / 1000}
              for t, v in (("train", 0.12), ("infer", 0.031))]
        if bad == "nan":
            pr[0]["median_s"] = float("nan")
        w(os.path.join(d, "raw/py.jsonl"), "".join(json.dumps(r) + "\n" for r in pr))
        jl = ""
        for s in ALL_SHAPES:
            if s == drop:
                continue
            jl += (f"arm=x series=threads shape={s} trans=nn m=1 n=1 k=1 rayon_threads=4 "
                   f"median_secs=0.000010000 gflops=1 checksum=0x0\n")
        if dup:
            jl += jl.splitlines()[0] + "\n"
        w(os.path.join(d, "raw/jobgrid.txt"), jl)
        ol = "".join(json.dumps({"arm": "base", "device": "cpu", "shape_id": s, "phase": "mm_pure",
                                 "median_s": 5e-6}) + "\n" for s in ALL_SHAPES)
        w(os.path.join(d, "raw/opp_kernel.jsonl"), ol)

    def expect_err(**kw):
        with tempfile.TemporaryDirectory() as t:
            build(t, **kw)
            try:
                convert(t)
            except ConvError:
                return
            raise AssertionError(f"should fail: {kw}")

    with tempfile.TemporaryDirectory() as t:
        build(t)
        tsv = convert(t).splitlines()
        assert len(tsv) == 11, tsv
        d = {tuple(ln.split("\t")[:2]): ln.split("\t") for ln in tsv[1:]}
        assert abs(float(d[("M-CPU-TRN", "fandhe_pure_kernel")][2]) - 5 * 0.01) < 1e-9
        assert abs(float(d[("M-CPU-INF", "fandhe_pure_kernel")][2]) - 2 * 0.01) < 1e-9
        assert abs(float(d[("M-CPU-TRN", "opp_pure_kernel")][2]) - 5 * 0.005) < 1e-9
        assert abs(float(d[("M-CPU-TRN", "fandhe_window")][2]) - 0.5) < 1e-9
        assert abs(float(d[("M-CPU-INF", "opp_window")][2]) - 0.031) < 1e-9
        assert d[("M-CPU-TRN", "fandhe_pure_kernel")][3] == "wall" and d[("M-CPU-INF", "opp_window")][3] == "-"
    expect_err(drop="rust")
    expect_err(drop="l2_dw")
    expect_err(dup=True)
    expect_err(bad="nan")
    print("self-test ok")


def main(argv):
    try:
        if "--self-test" in argv:
            self_test()
        elif len(argv) == 3 and argv[1] == "--round-dir":
            rdir = argv[2]
            text = convert(rdir)
            out = os.path.join(rdir, "values.tsv")
            if os.path.exists(out):
                raise ConvError(f"{out} already exists (no overwrite)")
            with open(out, "x", encoding="utf-8") as f:
                f.write(text)
            print(f"wrote {out}")
        else:
            print(__doc__)
            return 2
    except (ConvError, OSError, KeyError, ValueError) as e:
        print(f"CONVERT-ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
