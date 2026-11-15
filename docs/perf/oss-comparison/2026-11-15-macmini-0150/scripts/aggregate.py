#!/usr/bin/env python3
"""エンジン単体の対照（CPU GEMM 対 gemm crate／matrixmultiply、Metal GEMM 対 MLX／PyTorch MPS）の
Mac mini 5 ラウンド計測を集計する（スコアボード 0.14.0 の「エンジン単体の対照」カード用）。

入力: <out>/round{1..N}/ の各ファイル（orchestrate.sh が書く）
  gate.txt              1 行目が "PASS ..." または "TIMEOUT ..."
  oss.jsonl / oss.exit  oss-gemm-compare の JSON Lines と終了コード
  metal-transfer.txt    crates/backend-metal/examples/gemm_bench の標準出力（dynamic_tile_auto_tflops を使う）
  metal-prepared.txt    crates/backend-metal/examples/gemm_f32_prepared_bench の標準出力
  mlx.txt / mps.txt     scripts/bench/gemm_bench_mlx_f32.py・gemm_bench_torch_mps_f32.py の標準出力
出力: 標準出力へ、セルごとの run 間中央値（GFLOP/s）と、同一ラウンド内の比の run 間中央値。

集計規則は RULE.txt で計測前に固定する。欠損・重複・解釈できない行は例外で止める（寛容に読まない）。
--self-test は過去データ（M4 Max の 2026-09-03 ossx5・2026-08-23 の Metal 1 回）を 1 ラウンド形式に読み替え、
docs/perf/oss-gemm-comparison-baseline.md の値を再現できるかを確かめる。
"""
import json
import os
import re
import statistics
import sys

SIZES = [512, 1024, 2048, 4096]
OSS_IMPLS = ["self_gemm_blis_parallel", "gemm", "matrixmultiply"]


class AggError(Exception):
    pass


def read(path):
    if not os.path.isfile(path):
        raise AggError(f"missing file: {path}")
    with open(path, encoding="utf-8") as f:
        return f.read()


def parse_oss(text):
    out = {}
    for ln, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        r = json.loads(line)
        k = (r["impl"], int(r["size"]))
        if k in out:
            raise AggError(f"oss duplicate {k} line {ln}")
        if r["impl"] not in OSS_IMPLS:
            raise AggError(f"oss unknown impl {r['impl']}")
        if r["warmup"] != 20 or r["iters"] != 20:
            raise AggError(f"oss warmup/iters not 20/20 at {k}")
        out[k] = (float(r["tflops_median"]), bool(r["output_match"]), r.get("lib_version"), r.get("impl_threads"))
    for i in OSS_IMPLS:
        for s in SIZES:
            if (i, s) not in out:
                raise AggError(f"oss missing {(i, s)}")
    return out


def parse_kv_lines(text, first_key, keys):
    """'size=N <first_key>=x ...' の行だけを読む。同じ size が 2 回出たら止める。"""
    out = {}
    for line in text.splitlines():
        m = re.match(r"^size=(\d+) " + re.escape(first_key) + r"=", line)
        if not m:
            continue
        s = int(m.group(1))
        kv = dict(re.findall(r"(\w+)=([0-9.]+)(?=\s|$)", line))
        if s in out:
            raise AggError(f"duplicate size={s} for {first_key}")
        out[s] = {k: float(kv[k]) for k in keys}
    for s in SIZES:
        if s not in out:
            raise AggError(f"missing size={s} for {first_key}")
    return out


def header(text, key):
    for line in text.splitlines():
        m = re.match(r"^" + re.escape(key) + r"=(\S+)", line)
        if m:
            return m.group(1)
    raise AggError(f"missing header {key}=")


def load_round(d):
    g = read(os.path.join(d, "gate.txt")).splitlines()
    gate = g[0].split()[0] if g else ""
    if gate not in ("PASS", "TIMEOUT"):
        raise AggError(f"bad gate.txt in {d}")
    r = {"gate": gate}
    if gate != "PASS":
        return r
    r["oss_exit"] = read(os.path.join(d, "oss.exit")).strip()
    r["oss"] = parse_oss(read(os.path.join(d, "oss.jsonl")))
    r["tr"] = parse_kv_lines(read(os.path.join(d, "metal-transfer.txt")), "naive_tflops", ["dynamic_tile_auto_tflops"])
    r["pr"] = parse_kv_lines(read(os.path.join(d, "metal-prepared.txt")), "metal_f32_simdgroup_tiled_tflops",
                             ["metal_f32_simdgroup_tiled_tflops"])
    mlx = read(os.path.join(d, "mlx.txt"))
    r["mlx"] = parse_kv_lines(mlx, "mlx_f32_device_resident_tflops",
                              ["mlx_f32_device_resident_tflops", "mlx_f32_transfer_included_tflops"])
    mps = read(os.path.join(d, "mps.txt"))
    r["mps"] = parse_kv_lines(mps, "pytorch_mps_f32_tflops", ["pytorch_mps_f32_tflops"])
    r["ver"] = {"mlx": header(mlx, "mlx"), "torch": header(mps, "torch")}
    return r


def med(xs):
    return statistics.median(xs)


def summarize(rounds, expect_n):
    ok = [r for r in rounds if r["gate"] == "PASS"]
    lines = [f"rounds total={len(rounds)} gate_pass={len(ok)} status={'formal' if len(ok) == expect_n else 'partial'}"]
    if not ok:
        raise AggError("no round passed the gate")
    vers = sorted({json.dumps(r["ver"], sort_keys=True) for r in ok})
    lines.append("versions " + " | ".join(vers))
    lines.append("oss_exit " + ",".join(r["oss_exit"] for r in ok))
    # CPU
    lines.append("# CPU GEMM GFLOP/s（run 間中央値）と self÷相手（同一ラウンド内の比の run 間中央値）")
    for s in SIZES:
        v = {i: med([r["oss"][(i, s)][0] * 1000 for r in ok]) for i in OSS_IMPLS}
        rg = med([r["oss"][("self_gemm_blis_parallel", s)][0] / r["oss"][("gemm", s)][0] for r in ok])
        rm = med([r["oss"][("self_gemm_blis_parallel", s)][0] / r["oss"][("matrixmultiply", s)][0] for r in ok])
        mt = {i: sum(1 for r in ok if r["oss"][(i, s)][1]) for i in OSS_IMPLS}
        lines.append(
            f"cpu size={s} self={v['self_gemm_blis_parallel']:.0f} gemm={v['gemm']:.0f} mm={v['matrixmultiply']:.0f} "
            f"self/gemm={rg:.3f} self/mm={rm:.3f} "
            f"med_ratio self/gemm={v['self_gemm_blis_parallel'] / v['gemm']:.3f} match_rounds gemm={mt['gemm']}/{len(ok)} mm={mt['matrixmultiply']}/{len(ok)}")
    # Metal
    lines.append("# Metal GEMM TFLOPS（run 間中央値）と比（同一ラウンド内の比の run 間中央値）")
    for s in SIZES:
        tr = med([r["tr"][s]["dynamic_tile_auto_tflops"] for r in ok])
        pr = med([r["pr"][s]["metal_f32_simdgroup_tiled_tflops"] for r in ok])
        md = med([r["mlx"][s]["mlx_f32_device_resident_tflops"] for r in ok])
        mt = med([r["mlx"][s]["mlx_f32_transfer_included_tflops"] for r in ok])
        mp = med([r["mps"][s]["pytorch_mps_f32_tflops"] for r in ok])
        q1 = med([r["tr"][s]["dynamic_tile_auto_tflops"] / r["mlx"][s]["mlx_f32_transfer_included_tflops"] for r in ok])
        q2 = med([r["tr"][s]["dynamic_tile_auto_tflops"] / r["mps"][s]["pytorch_mps_f32_tflops"] for r in ok])
        q3 = med([r["pr"][s]["metal_f32_simdgroup_tiled_tflops"] / r["mlx"][s]["mlx_f32_device_resident_tflops"] for r in ok])
        lines.append(
            f"metal size={s} self_transfer={tr:.2f} self_prepared={pr:.2f} mlx_device={md:.2f} mlx_transfer={mt:.2f} "
            f"mps={mp:.2f} transfer/mlx_transfer={q1:.2f} transfer/mps={q2:.2f} prepared/mlx_device={q3:.2f}")
    return lines


def self_test(repo):
    # 過去データ: CPU は 5 run（2026-09-03 ossx5）、Metal は 1 回（2026-08-23）。各 run を 1 ラウンドに読み替える。
    oss_dir = os.path.join(repo, "docs/perf/oss-comparison/2026-09-03/cpu-retune-1041/ossx5")
    met_dir = os.path.join(repo, "docs/perf/oss-comparison/2026-08-23")
    met = {
        "tr": parse_kv_lines(read(os.path.join(met_dir, "metal-self-f32-transfer.txt")), "naive_tflops", ["dynamic_tile_auto_tflops"]),
        "pr": parse_kv_lines(read(os.path.join(met_dir, "metal-self-f32-prepared.txt")), "metal_f32_simdgroup_tiled_tflops",
                             ["metal_f32_simdgroup_tiled_tflops"]),
        "mlx": parse_kv_lines(read(os.path.join(met_dir, "mlx-f32.txt")), "mlx_f32_device_resident_tflops",
                              ["mlx_f32_device_resident_tflops", "mlx_f32_transfer_included_tflops"]),
        "mps": parse_kv_lines(read(os.path.join(met_dir, "torch-mps-f32.txt")), "pytorch_mps_f32_tflops", ["pytorch_mps_f32_tflops"]),
        "ver": {"mlx": header(read(os.path.join(met_dir, "mlx-f32.txt")), "mlx"),
                "torch": header(read(os.path.join(met_dir, "torch-mps-f32.txt")), "torch")},
    }
    rounds = []
    for i in range(1, 6):
        r = {"gate": "PASS", "oss_exit": read(os.path.join(oss_dir, f"oss-run{i}.exit")).strip(),
             "oss": parse_oss(read(os.path.join(oss_dir, f"oss-run{i}.jsonl")))}
        r.update(met)
        rounds.append(r)
    out = summarize(rounds, 5)
    text = "\n".join(out)
    # baseline.md §1.1（CPU・5 run 中央値）と §Metal f32 表の値
    want = ["cpu size=512 self=433 gemm=519", "cpu size=1024 self=786 gemm=837", "cpu size=2048 self=874 gemm=978",
            "cpu size=4096 self=1041 gemm=1070",
            "metal size=4096 self_transfer=4.16 self_prepared=7.23 mlx_device=12.78 mlx_transfer=9.64 mps=13.05 "
            "transfer/mlx_transfer=0.43 transfer/mps=0.32 prepared/mlx_device=0.57"]
    bad = [w for w in want if w not in text]
    # 欠損で止まることも確かめる
    try:
        parse_oss('{"impl":"gemm","size":512,"warmup":20,"iters":20,"tflops_median":1,"output_match":true}')
        bad.append("parse_oss accepted missing cells")
    except AggError:
        pass
    print(text)
    if bad:
        print("SELF-TEST FAIL: " + " / ".join(bad), file=sys.stderr)
        return 1
    print("self-test ok")
    return 0


def main(argv):
    if len(argv) == 3 and argv[1] == "--self-test":
        return self_test(argv[2])
    if len(argv) != 3:
        print("usage: aggregate.py <out-dir> <expected-rounds> | --self-test <repo-root>", file=sys.stderr)
        return 2
    n = int(argv[2])
    rounds = [load_round(os.path.join(argv[1], f"round{i}")) for i in range(1, n + 1)]
    print("\n".join(summarize(rounds, n)))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (AggError, KeyError, ValueError, json.JSONDecodeError) as e:
        print(f"AGGREGATE-ERROR: {e}", file=sys.stderr)
        sys.exit(1)
