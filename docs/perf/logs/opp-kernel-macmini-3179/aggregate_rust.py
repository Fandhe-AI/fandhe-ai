#!/usr/bin/env python3
"""イシュー #3179: Mac mini の Rust 側逐次総和（checksum 区間）の ns／要素を集計する。

run-macmini.sh が出す rust/n{N}_r{R}.jsonl（bench-fandhe gemm reuse --phases。task=gemm_phases）から
phase=checksum の median_s を読み、ns／要素 = median_s / N^2 * 1e9 を出す。#3182 の F2 との突き合わせは
結論 doc 側で行う。opp_kernel_torch_diag.py（CI の sha 照合対象）は変えないため独立スクリプトにしている。
欠損・重複・余分な行・非数値は 0 埋めせず AggError（exit 1）で拒否する（fail-closed）。標準ライブラリのみ。
使い方（python3 -I）: aggregate_rust.py <OUT> --out-json X --out-md Y（出力が既にあれば上書きしない）
                    aggregate_rust.py --self-test
"""
import argparse
import json
import math
import statistics
import sys
import tempfile
from pathlib import Path

SIZES = (256, 512, 1024, 2048, 4096)
RUNS = (1, 2, 3, 4, 5)
PHASES = ("matmul", "to_tensor", "host_copy", "checksum", "iter_total")


class AggError(Exception):
    pass


def _checksum_median(path, n):
    if not path.is_file():
        raise AggError(f"欠損: {path.name}")
    found = None
    seen = set()
    for ln, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError as e:
            raise AggError(f"{path.name}:{ln} JSON 不正: {e}")
        if not isinstance(r, dict):
            raise AggError(f"{path.name}:{ln} JSON オブジェクトではない")
        # 読み飛ばさず全行を検証する（別タスク・未知 phase・別 size／device／mode の混入を拒否）
        if r.get("task") != "gemm_phases" or r.get("device") != "cpu" or r.get("size") != n or r.get("mode") != "reuse":
            raise AggError(f"{path.name}:{ln} 想定外の行: task={r.get('task')!r} device={r.get('device')!r} size={r.get('size')!r} mode={r.get('mode')!r}")
        phase = r.get("phase")
        if phase not in PHASES:
            raise AggError(f"{path.name}:{ln} 未知の phase: {phase!r}")
        if phase in seen:
            raise AggError(f"{path.name}:{ln} phase={phase} の行が重複")
        seen.add(phase)
        if phase != "checksum":
            continue
        v = r.get("median_s")
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
            raise AggError(f"{path.name}:{ln} median_s が不正: {v!r}")
        found = float(v)
    if seen != set(PHASES):
        raise AggError(f"{path.name}: phase 欠損: {sorted(set(PHASES) - seen)}")
    if found is None:
        raise AggError(f"{path.name}: checksum 行が無い")
    return found


def aggregate(out_dir):
    rust = Path(out_dir) / "rust"
    expected = {f"n{n}_r{r}.jsonl" for n in SIZES for r in RUNS}
    present = {p.name for p in rust.glob("*.jsonl")} if rust.is_dir() else set()
    if present - expected:
        raise AggError(f"余分なファイル: {sorted(present - expected)}")
    table = {}
    for n in SIZES:
        per_run = [_checksum_median(rust / f"n{n}_r{r}.jsonl", n) / (n * n) * 1e9 for r in RUNS]
        table[n] = {"ns_per_elem_runs": per_run, "ns_per_elem_median": statistics.median(per_run)}
    return table


def to_md(table):
    lines = ["# Rust 逐次総和（checksum 区間）の ns／要素（Mac mini・CPU・reuse --phases）", "",
             "| N | run1 | run2 | run3 | run4 | run5 | 中央値 |", "|---|---|---|---|---|---|---|"]
    for n in SIZES:
        t = table[n]
        lines.append(f"| {n} | " + " | ".join(f"{v:.3f}" for v in t["ns_per_elem_runs"]) + f" | {t['ns_per_elem_median']:.3f} |")
    return "\n".join(lines) + "\n"


def _write_new(path, text):
    with open(path, "x") as f:  # 既存なら FileExistsError（上書きしない）
        f.write(text)


def self_test():
    def row(n, v, phase="checksum"):
        return json.dumps({"task": "gemm_phases", "phase": phase, "device": "cpu", "mode": "reuse", "size": n, "median_s": v}) + "\n"

    def full(n, ck=None):
        return "".join(row(n, (5e-10 * n * n if ck is None else ck) if ph == "checksum" else 1e-9 * n * n, ph) for ph in PHASES)

    def build(d, mutate=None):
        (d / "rust").mkdir()
        for n in SIZES:
            for r in RUNS:
                (d / "rust" / f"n{n}_r{r}.jsonl").write_text(full(n))
        if mutate:
            mutate(d)

    with tempfile.TemporaryDirectory() as t:
        d = Path(t) / "ok"
        d.mkdir()
        build(d)
        tb = aggregate(d)
        assert all(abs(tb[n]["ns_per_elem_median"] - 0.5) < 1e-9 for n in SIZES)
        bad = {
            "missing": lambda d: (d / "rust" / "n512_r3.jsonl").unlink(),
            "dup": lambda d: (d / "rust" / "n256_r1.jsonl").write_text(full(256) + row(256, 1e-5)),
            "othertask": lambda d: (d / "rust" / "n256_r1.jsonl").write_text(full(256) + json.dumps({"task": "gemm", "phase": "checksum", "device": "cpu", "mode": "reuse", "size": 256, "median_s": 1e-5}) + "\n"),
            "unknownphase": lambda d: (d / "rust" / "n256_r1.jsonl").write_text(full(256) + row(256, 1e-5, "bogus")),
            "wrongsize": lambda d: (d / "rust" / "n256_r1.jsonl").write_text(full(256) + row(512, 1e-5, "matmul")),
            "missingphase": lambda d: (d / "rust" / "n256_r1.jsonl").write_text(row(256, 1e-5)),
            "extra": lambda d: (d / "rust" / "n999_r1.jsonl").write_text(row(999, 1e-5)),
            "nonnum": lambda d: (d / "rust" / "n256_r1.jsonl").write_text(full(256, "x")),
        }
        for k, m in bad.items():
            d2 = Path(t) / k
            d2.mkdir()
            build(d2, m)
            try:
                aggregate(d2)
            except AggError:
                continue
            raise SystemExit(f"self-test 失敗: {k} が拒否されない")
    print("self-test OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", nargs="?")
    ap.add_argument("--out-json")
    ap.add_argument("--out-md")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not (a.out_dir and a.out_json and a.out_md):
        ap.error("out_dir・--out-json・--out-md が必要")
    try:
        table = aggregate(a.out_dir)
    except AggError as e:
        print(f"AggError: {e}", file=sys.stderr)
        sys.exit(1)
    _write_new(a.out_json, json.dumps({str(n): table[n] for n in SIZES}, indent=2) + "\n")
    _write_new(a.out_md, to_md(table))


if __name__ == "__main__":
    main()
