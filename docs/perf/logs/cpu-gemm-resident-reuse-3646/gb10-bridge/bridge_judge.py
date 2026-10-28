#!/usr/bin/env python3
"""イシュー #3646: RULE.v2.txt「ブリッジ v2」の判定器（標準ライブラリのみ。fail-closed）。

役割: bench-fandhe の gemm（fresh 通常行・reuse 通常行・reuse --phases の iter_total）と、
診断 gb10 系列の resident_public／fresh の iter_total から、N ごとに
  G_bench = reuse.iter_total - fresh（5 run 中央値同士）
  G_diag  = resident_public.iter_total - fresh.iter_total（診断 5 run 中央値同士）
  bridge=pass  <=>  G_bench > 0 かつ G_diag > 0 かつ 0.5 <= G_bench / G_diag <= 2.0
を判定し、`<N> bridge=pass|fail` を gb10/bridge_status.txt へ、算出過程を bridge_judge.md へ書く。
呼び出し元: 人手（実測後に `python3 bridge_judge.py <cpu-gemm-resident-reuse-3646 ディレクトリ>`）。
呼び出し先: gb10/bridge_status.txt を aggregate.py が読む（aggregate.py は pass/fail だけを読み閾値は持たない）。
寛容パースはしない: ファイル・行・キーの欠落、run 数の不足、非有限値、重複はすべて例外（exit 1）。
"""
import json
import math
import os
import statistics
import sys

SIZES = (256, 512, 1024, 2048, 4096)
RUNS = (1, 2, 3, 4, 5)
MODES = ("fresh", "reuse", "phases")
LOW, HIGH = 0.5, 2.0  # RULE.v2 の閾値（事後に変えない）
MIN_FILE = "bridge_status.txt"


class JudgeError(Exception):
    pass


def read_rows(path):
    rows = []
    try:
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError as e:
                    raise JudgeError(f"{path}:{i}: JSON 不正 ({e})")
    except OSError as e:
        raise JudgeError(f"{path}: 読み込み不可 ({e})")
    return rows


def finite_pos(x, where):
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise JudgeError(f"{where}: 数値でない {x!r}")
    if not math.isfinite(v) or v <= 0.0:
        raise JudgeError(f"{where}: 有限の正値でない {v}")
    return v


def bench_cell(bridge_dir, mode, n, r):
    path = os.path.join(bridge_dir, f"gemm_{mode}_n{n}_r{r}.jsonl")
    rows = read_rows(path)
    if mode in ("fresh", "reuse"):
        if len(rows) != 1:
            raise JudgeError(f"{path}: 行数 {len(rows)} != 1")
        row = rows[0]
        if row.get("task") != "gemm" or row.get("size") != n or row.get("mode") != mode or row.get("device") != "cpu":
            raise JudgeError(f"{path}: task/size/mode/device 不一致")
        return {"median_s": finite_pos(row.get("median_s"), path)}
    got = {}
    for row in rows:
        if row.get("task") != "gemm_phases" or row.get("size") != n or row.get("mode") != "reuse":
            raise JudgeError(f"{path}: task/size/mode 不一致")
        ph = row.get("phase")
        if ph in got:
            raise JudgeError(f"{path}: phase 重複 {ph}")
        # sub-ns 区間の 0 は許容しない（RULE.v2 は iter_total のみ判定に使う。他は記録）。
        v = float(row.get("median_s"))
        if not math.isfinite(v) or v < 0.0:
            raise JudgeError(f"{path}: phase {ph} の median_s が不正")
        got[ph] = v
    for ph in ("matmul", "to_tensor", "host_copy", "checksum", "iter_total"):
        if ph not in got:
            raise JudgeError(f"{path}: phase {ph} が無い")
    if got["iter_total"] <= 0.0:
        raise JudgeError(f"{path}: iter_total が正でない")
    return got


def diag_medians(gb10_dir):
    """診断 gb10 系列 5 run の (arm, n) -> run 間中央値 iter_total（秒）。"""
    per = {}
    for r in RUNS:
        path = os.path.join(gb10_dir, f"run{r}.jsonl")
        seen = set()
        for row in read_rows(path):
            if row.get("phase") != "iter_total" or row.get("arm") not in ("fresh", "resident_public"):
                continue
            key = (row["arm"], row["n"])
            if key in seen:
                raise JudgeError(f"{path}: セル重複 {key}")
            seen.add(key)
            per.setdefault(key, {})[r] = finite_pos(row.get("median_s"), f"{path} {key}")
    out = {}
    for arm in ("fresh", "resident_public"):
        for n in SIZES:
            runs = per.get((arm, n), {})
            if sorted(runs) != list(RUNS):
                raise JudgeError(f"診断 {arm}/n={n}: run が揃っていない ({sorted(runs)})")
            out[(arm, n)] = statistics.median(runs[r] for r in RUNS)
    return out


def check_rc(bridge_dir):
    path = os.path.join(bridge_dir, "rc.log")
    try:
        with open(path, encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
    except OSError as e:
        raise JudgeError(f"{path}: 読み込み不可 ({e})")
    want = len(MODES) * len(SIZES) * len(RUNS)
    if len(lines) != want:
        raise JudgeError(f"{path}: 行数 {len(lines)} != {want}")
    bad = [ln for ln in lines if not ln.endswith("rc=0")]
    if bad:
        raise JudgeError(f"{path}: rc != 0 の行 {bad[:3]}")


def main(argv):
    if len(argv) != 2:
        print("usage: bridge_judge.py <cpu-gemm-resident-reuse-3646 dir>", file=sys.stderr)
        return 2
    base = argv[1]
    bridge_dir = os.path.join(base, "gb10-bridge")
    gb10_dir = os.path.join(base, "gb10")
    check_rc(bridge_dir)
    diag = diag_medians(gb10_dir)
    status = {}
    lines = [
        "# ブリッジ v2 判定（RULE.v2.txt。#3646）",
        "",
        "bench-fandhe（registry `fandhe-ai =0.12.0`）の gemm CPU を N 別に各 5 run。値は秒の 5 run 中央値（run 間 min-max を併記）。",
        "`G_bench = reuse.iter_total - fresh`、`G_diag = resident_public - fresh`（診断 gb10 系列 5 run 中央値）。",
        f"pass の条件: `G_bench > 0` かつ `G_diag > 0` かつ `{LOW} <= G_bench / G_diag <= {HIGH}`。",
        "",
        "| N | fresh(bench) | reuse iter_total(bench) | reuse 通常行(bench) | G_bench | fresh(diag) | resident_public(diag) | G_diag | 比 | 判定 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    detail = []
    for n in SIZES:
        cells = {m: {r: bench_cell(bridge_dir, m, n, r) for r in RUNS} for m in MODES}
        fresh = [cells["fresh"][r]["median_s"] for r in RUNS]
        reuse = [cells["reuse"][r]["median_s"] for r in RUNS]
        phases = {ph: [cells["phases"][r][ph] for r in RUNS]
                  for ph in ("matmul", "to_tensor", "host_copy", "checksum", "iter_total")}
        f_med = statistics.median(fresh)
        it_med = statistics.median(phases["iter_total"])
        ru_med = statistics.median(reuse)
        g_bench = it_med - f_med
        g_diag = diag[("resident_public", n)] - diag[("fresh", n)]
        ratio = g_bench / g_diag if g_diag > 0 else float("nan")
        ok = g_bench > 0 and g_diag > 0 and LOW <= ratio <= HIGH
        status[n] = "pass" if ok else "fail"
        lines.append(
            f"| {n} | {f_med:.9f} ({min(fresh):.9f}-{max(fresh):.9f}) | {it_med:.9f} ({min(phases['iter_total']):.9f}-"
            f"{max(phases['iter_total']):.9f}) | {ru_med:.9f} | {g_bench:+.9f} | {diag[('fresh', n)]:.9f} | "
            f"{diag[('resident_public', n)]:.9f} | {g_diag:+.9f} | {ratio:.3f} | {status[n]} |"
        )
        mm = statistics.median(phases["matmul"])
        share = (mm / g_bench) if g_bench > 0 else float("nan")
        detail.append(
            f"| {n} | " + " | ".join(f"{statistics.median(phases[p]):.9f}" for p in ("matmul", "to_tensor", "host_copy", "checksum"))
            + f" | {share:.3f} |"
        )
    lines += [
        "",
        "## 記録のみ（判定外）: reuse `--phases` の区間 5 run 中央値（秒）",
        "",
        "| N | matmul | to_tensor | host_copy | checksum | matmul / G_bench |",
        "|---|---|---|---|---|---|",
    ] + detail
    lines += ["", "## run 別（run 番号付き。fresh／reuse 通常行／reuse iter_total）", "",
              "| N | run | fresh | reuse 通常行 | reuse iter_total |", "|---|---|---|---|---|"]
    for n in SIZES:
        for r in RUNS:
            lines.append(
                f"| {n} | {r} | {bench_cell(bridge_dir, 'fresh', n, r)['median_s']:.9f} | "
                f"{bench_cell(bridge_dir, 'reuse', n, r)['median_s']:.9f} | "
                f"{bench_cell(bridge_dir, 'phases', n, r)['iter_total']:.9f} |"
            )
    with open(os.path.join(gb10_dir, MIN_FILE), "w", encoding="utf-8") as f:
        f.writelines(f"{n} bridge={status[n]}\n" for n in SIZES)
    with open(os.path.join(bridge_dir, "bridge_judge.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(f"{n} bridge={status[n]}" for n in SIZES))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except JudgeError as e:
        print(f"bridge_judge: {e}", file=sys.stderr)
        sys.exit(1)
