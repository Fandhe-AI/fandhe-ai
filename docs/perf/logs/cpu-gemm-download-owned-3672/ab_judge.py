#!/usr/bin/env python3
"""イシュー #3672 の A/B 判定器（RULE.txt ＋ RULE.v2.txt を機械的に適用する。標準ライブラリのみ・fail-closed）。

役割: orchestrate.sh が機体ごとに残す before／after × 5 run の診断 JSONL と bench JSONL を読み、
RULE の ADOPT／REJECT／判定不能を出力する。閾値は RULE.txt の値をそのまま定数にしたもので、
ここで新しい基準を作らない。呼び出し元: session-scripts/session-3672.sh（機体ごとの最終段）と
回収後の手動実行（両機体をまたぐ最終判定 judge-3672.md）。

使い方:
  ab_judge.py --host gb10=<dir> [--host macmini=<dir>] [--out <md>]
  ab_judge.py --self-test
  <dir> は orchestrate.sh の --out ディレクトリ。構成は次のとおり。
    series_kind.txt / load_gate_status.txt / excl_status.txt（gb10 のみ）
    before/run{1..5}.jsonl  after/run{1..5}.jsonl            （診断。DIAG_JSON 行）
    before/bench/gemm_phases_n{N}_r{r}.jsonl  after/bench/…   （bench-fandhe --mode reuse --phases）

セルの解釈（RULE.v2.txt。RULE.txt の「上記以外の全セル」を厳しい側に固定）:
  - bench セル: gemm_phases の phase=iter_total × N 5 点
  - 診断セル: 全 8 腕の phase=iter_total × N 5 点
  - 処置セル: GB10 の bench N=4096 と GB10 の診断 resident_public N=4096。それ以外は全て非処置セル
  - iter_total 以外の区間は記録のみ（DL_before・X の計算にだけ使う）
  - ratio_r = after_r / before_r（同じ run 番号の対）。セルの ratio は 5 個の中央値

ADOPT は両機体が揃い（main 系列・全 run のゲート pass・GB10 は専有 pass）たときに限る。
片方が未実測・record_only なら ADOPT は出さず、REJECT 条件に当たる場合のみ REJECT、他は判定不能（未実測あり）。
fail-closed: run 数が 5 でない・行の欠落／重複・n_samples／iters の不一致・JSON 解析不能・ゲート記録の欠落は
例外にして終了コード 2。寛容なパースや暗黙のスキップはしない。
終了コード: 0=判定を出した（ADOPT／REJECT／判定不能のいずれも 0）、2=入力不備（fail-closed）、1=self-test 失敗。
"""
import argparse
import json
import math
import os
import re
import statistics
import sys
import tempfile

SIZES = [256, 512, 1024, 2048, 4096]
RUNS = [1, 2, 3, 4, 5]
DIAG_ARMS = [
    "fresh",
    "resident_public",
    "resident_reupload",
    "resident_split",
    "backend_drop",
    "gemm_device_hold",
    "gemm_device_drop_in",
    "gemm_device_download",
]
ARMS = ["before", "after"]
HOSTS = ["gb10", "macmini"]
# RULE.txt の閾値（変更しない）
ADOPT_TREAT_MEDIAN_MAX = 0.95
ADOPT_TREAT_ALL_RUNS_BELOW = 1.00
ADOPT_NONTREAT_MEDIAN_MAX = 1.00
ADOPT_RHO_REM_MIN = 0.5
TREAT_N = 4096


class JudgeError(Exception):
    """入力不備（fail-closed）。"""


def read_jsonl(path):
    if not os.path.isfile(path):
        raise JudgeError("ファイルが無い: %s" % path)
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, ln in enumerate(f, 1):
            if not ln.strip():
                raise JudgeError("空行: %s:%d" % (path, i))
            try:
                d = json.loads(ln)
            except ValueError as e:
                raise JudgeError("JSON 解析不能: %s:%d (%s)" % (path, i, e))
            if not isinstance(d, dict):
                raise JudgeError("オブジェクトでない行: %s:%d" % (path, i))
            rows.append(d)
    if not rows:
        raise JudgeError("行が 0 件: %s" % path)
    return rows


def pos_float(v, what):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
        raise JudgeError("正の有限数でない %s: %r" % (what, v))
    return float(v)


def load_diag_run(path):
    """{(arm, phase, n): (median_s, n_samples, checksum_bits)}。重複は例外。"""
    out = {}
    for d in read_jsonl(path):
        try:
            key = (d["arm"], d["phase"], d["n"])
            val = (d["median_s"], d["n_samples"], d.get("checksum_bits"))
        except KeyError as e:
            raise JudgeError("診断行にキーが無い: %s (%s)" % (path, e))
        if key in out:
            raise JudgeError("診断行が重複: %s %r" % (path, key))
        out[key] = val
    needed = [(a, "iter_total", n) for a in DIAG_ARMS for n in SIZES] + [("resident_split", "download", TREAT_N)]
    for k in needed:
        if k not in out:
            raise JudgeError("診断行が欠落: %s %r" % (path, k))
        # 判定に使う行だけ値を検査する（使わない区間行は Mac のタイマー分解能で 0.0 になりうるため検査しない）。
        pos_float(out[k][0], "median_s %r" % (k,))
    for a in DIAG_ARMS:
        for n in SIZES:
            if not isinstance(out[(a, "iter_total", n)][2], str) or not out[(a, "iter_total", n)][2]:
                raise JudgeError("iter_total の checksum_bits が無い: %s %s n=%d" % (path, a, n))
    return out


def load_bench_run(path, n):
    """phases 行から iter_total を取る: (median_s, iters, checksum)。"""
    rows = read_jsonl(path)
    seen = None
    for d in rows:
        for k, want in (("task", "gemm_phases"), ("device", "cpu"), ("mode", "reuse")):
            if d.get(k) != want:
                raise JudgeError("bench 行の %s が %r でない: %s" % (k, want, path))
        if d.get("size") != n:
            raise JudgeError("bench 行の size が %d でない: %s" % (n, path))
        if d.get("phase") == "iter_total":
            if seen is not None:
                raise JudgeError("iter_total 行が重複: %s" % path)
            if not isinstance(d.get("checksum"), (int, float)) or isinstance(d.get("checksum"), bool):
                raise JudgeError("bench の checksum が数値でない: %s" % path)
            if not isinstance(d.get("iters"), int):
                raise JudgeError("bench の iters が無い: %s" % path)
            seen = (pos_float(d.get("median_s"), "median_s"), d["iters"], d["checksum"])
    if seen is None:
        raise JudgeError("iter_total 行が無い: %s" % path)
    return seen


def read_kv_lines(path, pat):
    if not os.path.isfile(path):
        raise JudgeError("ファイルが無い: %s" % path)
    with open(path, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    for ln in lines:
        if not re.match(pat, ln):
            raise JudgeError("解析不能な行 %r: %s" % (ln, path))
    return lines


def load_host(root, host):
    """機体の全データを読む。ディレクトリ自体が無ければ None（未実測）。"""
    if root is None or not os.path.isdir(root):
        return None
    kind = read_kv_lines(os.path.join(root, "series_kind.txt"), r"^[a-z]+$")
    gates = read_kv_lines(os.path.join(root, "load_gate_status.txt"), r"^run[1-9][0-9]* gate=[a-z_]+$")
    if len(gates) != len(RUNS):
        raise JudgeError("load_gate_status の行数が %d でない: %s" % (len(RUNS), root))
    gate_by_run = {}
    for ln in gates:
        r, g = ln.split(" gate=")
        if r in gate_by_run:
            raise JudgeError("ゲート記録が重複: %s %s" % (root, r))
        gate_by_run[r] = g
    for r in RUNS:
        if "run%d" % r not in gate_by_run:
            raise JudgeError("ゲート記録が欠落: %s run%d" % (root, r))
    excl = None
    if host == "gb10":
        ex = read_kv_lines(os.path.join(root, "excl_status.txt"), r"^[a-z]+ excl=(pass|fail)$")
        d = dict(x.split(" excl=") for x in ex)
        if "main" not in d:
            raise JudgeError("excl_status に main が無い: %s" % root)
        excl = d["main"]
    data = {"diag": {}, "bench": {}}
    for arm in ARMS:
        for r in RUNS:
            data["diag"][(arm, r)] = load_diag_run(os.path.join(root, arm, "run%d.jsonl" % r))
            for n in SIZES:
                data["bench"][(arm, r, n)] = load_bench_run(
                    os.path.join(root, arm, "bench", "gemm_phases_n%d_r%d.jsonl" % (n, r)), n
                )
    # n_samples／iters は全 run・両腕で同一でなければならない（反復数の差は比較不能）
    for key in {(a, p, n) for (a, p, n) in data["diag"][("before", 1)]}:
        vals = {data["diag"][(arm, r)][key][1] for arm in ARMS for r in RUNS if key in data["diag"][(arm, r)]}
        if len(vals) != 1:
            raise JudgeError("n_samples が腕・run 間で不一致: %s %r %r" % (root, key, sorted(vals, key=str)))
    for n in SIZES:
        vals = {data["bench"][(arm, r, n)][1] for arm in ARMS for r in RUNS}
        if len(vals) != 1:
            raise JudgeError("bench の iters が腕・run 間で不一致: %s n=%d" % (root, n))
    reasons = []
    if kind != ["main"]:
        reasons.append("series_kind=%s（main 以外）" % ",".join(kind))
    bad = sorted(r for r, g in gate_by_run.items() if g != "pass")
    if bad:
        reasons.append("ゲート未通過の run: %s" % ",".join(bad))
    if excl is not None and excl != "pass":
        reasons.append("専有 excl=%s" % excl)
    return {"data": data, "record_only_reasons": reasons, "excl": excl}


def check_checksums(data):
    """checksum の全件一致を検査し、不一致の一覧を返す。"""
    mism = []
    for a in DIAG_ARMS:
        for n in SIZES:
            vals = {data["diag"][(arm, r)][(a, "iter_total", n)][2] for arm in ARMS for r in RUNS}
            if len(vals) != 1:
                mism.append("診断 %s n=%d: %s" % (a, n, sorted(vals)))
    for n in SIZES:
        vals = {data["bench"][(arm, r, n)][2] for arm in ARMS for r in RUNS}
        if len(vals) != 1:
            mism.append("bench n=%d: %s" % (n, sorted(vals)))
    return mism


def cell_stats(per_run_before, per_run_after):
    ratios = [a / b for a, b in zip(per_run_after, per_run_before)]
    return {
        "before_med": statistics.median(per_run_before),
        "after_med": statistics.median(per_run_after),
        "ratios": ratios,
        "ratio": statistics.median(ratios),
        "all_below_1": all(x < ADOPT_TREAT_ALL_RUNS_BELOW for x in ratios),
        "all_above_1": all(x > 1.00 for x in ratios),
    }


def host_cells(h):
    """{(group, label, n): stats}。group は bench／diag。"""
    d, b = h["data"]["diag"], h["data"]["bench"]
    cells = {}
    for n in SIZES:
        cells[("bench", "reuse iter_total", n)] = cell_stats(
            [b[("before", r, n)][0] for r in RUNS], [b[("after", r, n)][0] for r in RUNS]
        )
        for a in DIAG_ARMS:
            cells[("diag", a, n)] = cell_stats(
                [d[("before", r)][(a, "iter_total", n)][0] for r in RUNS],
                [d[("after", r)][(a, "iter_total", n)][0] for r in RUNS],
            )
    return cells


def gb10_rho(h):
    d = h["data"]["diag"]

    def x(arm):
        return statistics.median(
            d[(arm, r)][("resident_public", "iter_total", TREAT_N)][0]
            - d[(arm, r)][("gemm_device_hold", "iter_total", TREAT_N)][0]
            for r in RUNS
        )

    dl = statistics.median(d[("before", r)][("resident_split", "download", TREAT_N)][0] for r in RUNS)
    xb, xa = x("before"), x("after")
    bb = statistics.median(h["data"]["bench"][("before", r, TREAT_N)][0] for r in RUNS)
    ba = statistics.median(h["data"]["bench"][("after", r, TREAT_N)][0] for r in RUNS)
    rho_rem = (xb - xa) / dl
    rho_bench = (bb - ba) / dl
    return {"X_before": xb, "X_after": xa, "DL_before": dl, "rho_REM": rho_rem, "rho_BENCH": rho_bench,
            "rho_OFFSET": 1 - rho_bench, "bench_before": bb, "bench_after": ba}


def judge(hosts):
    """hosts: {name: loaded or None}。判定結果の辞書を返す。"""
    usable = {n: h for n, h in hosts.items() if h is not None and not h["record_only_reasons"]}
    notes = []
    for n in HOSTS:
        h = hosts.get(n)
        if h is None:
            notes.append("%s: 未実測（データ無し）" % n)
        elif h["record_only_reasons"]:
            notes.append("%s: record_only（%s）。判定に使わない" % (n, " / ".join(h["record_only_reasons"])))
    reject, adopt_blockers = [], []
    cells_by_host, rho = {}, None
    for n, h in usable.items():
        mism = check_checksums(h["data"])
        for m in mism:
            reject.append("%s checksum 不一致: %s" % (n, m))
        cells_by_host[n] = host_cells(h)
        for (grp, label, sz), s in cells_by_host[n].items():
            is_treat = n == "gb10" and sz == TREAT_N and ((grp, label) in (("bench", "reuse iter_total"), ("diag", "resident_public")))
            if s["all_above_1"]:
                reject.append("%s %s %s N=%d: 5/5 run で後退（ratio > 1.00）" % (n, grp, label, sz))
            if is_treat and s["ratio"] > 1.00:
                reject.append("%s 処置セル %s %s N=%d: 中央値 ratio %.4f > 1.00" % (n, grp, label, sz, s["ratio"]))
            if is_treat:
                if not (s["ratio"] <= ADOPT_TREAT_MEDIAN_MAX):
                    adopt_blockers.append("処置セル %s %s: 中央値 ratio %.4f > 0.95" % (grp, label, s["ratio"]))
                if not s["all_below_1"]:
                    adopt_blockers.append("処置セル %s %s: 5/5 run で ratio < 1.00 を満たさない" % (grp, label))
            elif s["ratio"] > ADOPT_NONTREAT_MEDIAN_MAX:
                adopt_blockers.append("非処置セル %s %s %s N=%d: 中央値 ratio %.4f > 1.00" % (n, grp, label, sz, s["ratio"]))
        if n == "gb10":
            rho = gb10_rho(h)
            if not (rho["rho_REM"] >= ADOPT_RHO_REM_MIN):
                adopt_blockers.append("rho_REM %.4f < 0.5" % rho["rho_REM"])
    missing = [n for n in HOSTS if n not in usable]
    if reject:
        verdict = "REJECT"
    elif missing:
        verdict = "判定不能（未実測あり）"
    elif adopt_blockers:
        verdict = "判定不能"
    else:
        verdict = "ADOPT"
    return {"verdict": verdict, "reject": reject, "adopt_blockers": adopt_blockers, "notes": notes,
            "cells": cells_by_host, "rho": rho, "missing": missing}


def fmt_s(v):
    return "%.6g" % v


def render(res, hosts):
    L = ["# #3672 A/B 判定（ab_judge.py の出力。手で書き換えない）", ""]
    L.append("**判定: %s**" % res["verdict"])
    L.append("")
    for n in res["notes"]:
        L.append("- %s" % n)
    if res["missing"]:
        L.append("- 判定に使えない機体（未実測）: %s" % ", ".join(res["missing"]))
    L.append("")
    if res["reject"]:
        L.append("## REJECT 条件に当たった項目")
        L += ["- %s" % x for x in res["reject"]]
        L.append("")
    if res["adopt_blockers"]:
        L.append("## ADOPT を妨げた項目（記録）")
        L += ["- %s" % x for x in res["adopt_blockers"]]
        L.append("")
    if res["rho"]:
        r = res["rho"]
        L += ["## GB10 N=4096 の rho（記録。rho_REM のみ ADOPT 条件）", "",
              "| 指標 | 値 |", "|---|---|"]
        for k in ("X_before", "X_after", "DL_before", "bench_before", "bench_after"):
            L.append("| %s (s) | %s |" % (k, fmt_s(r[k])))
        for k in ("rho_REM", "rho_BENCH", "rho_OFFSET"):
            L.append("| %s | %.4f |" % (k, r[k]))
        L.append("")
    for n, cells in res["cells"].items():
        L += ["## %s セル（after/before。中央値 ratio と run ごとの ratio）" % n, "",
              "| 群 | セル | N | 区分 | before 中央値(s) | after 中央値(s) | 中央値 ratio | run ごとの ratio |",
              "|---|---|---|---|---|---|---|---|"]
        for (grp, label, sz), s in sorted(cells.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
            treat = n == "gb10" and sz == TREAT_N and ((grp, label) in (("bench", "reuse iter_total"), ("diag", "resident_public")))
            L.append("| %s | %s | %d | %s | %s | %s | %.4f | %s |" % (
                grp, label, sz, "処置" if treat else "非処置", fmt_s(s["before_med"]), fmt_s(s["after_med"]),
                s["ratio"], " ".join("%.4f" % x for x in s["ratios"])))
        L.append("")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- self-test
def _w(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def make_host(root, host, factor, kind="main", gate="pass", excl="pass", cksum_after=None, drop=None, drop_bench=False):
    """合成 fixture。factor(arm, grp, label, n, r) は after に掛ける倍率を返す。"""
    _w(os.path.join(root, "series_kind.txt"), kind + "\n")
    _w(os.path.join(root, "load_gate_status.txt"), "".join("run%d gate=%s\n" % (r, gate) for r in RUNS))
    if host == "gb10":
        _w(os.path.join(root, "excl_status.txt"), "main excl=%s\n" % excl)
    for arm in ARMS:
        for r in RUNS:
            rows = []
            for a in DIAG_ARMS:
                for n in SIZES:
                    base = 0.001 * (n / 256.0) ** 2 * (1 + 0.01 * DIAG_ARMS.index(a))
                    if a == "gemm_device_hold":
                        base *= 0.5
                    v = base * (factor(arm, "diag", a, n, r) if arm == "after" else 1.0)
                    ck = "ck_%s_%d" % (a, n)
                    if arm == "after" and cksum_after and cksum_after == (a, n):
                        ck = "xx"
                    rows.append({"arm": a, "phase": "iter_total", "n": n, "median_s": v, "n_samples": 20, "checksum_bits": ck})
            dl = 0.01
            rows.append({"arm": "resident_split", "phase": "download", "n": TREAT_N, "median_s": dl, "n_samples": 20, "checksum_bits": None})
            # 使わない区間行の 0.0（タイマー分解能）は判定を止めない（Mac mini 実測で発生。判定器の修正点）
            rows.append({"arm": "fresh", "phase": "tape_build", "n": 256, "median_s": 0.0, "n_samples": 20, "checksum_bits": None})
            if drop == r and arm == "after":
                rows.pop(0)
            _w(os.path.join(root, arm, "run%d.jsonl" % r), "".join(json.dumps(x) + "\n" for x in rows))
            for n in SIZES:
                base = 0.0012 * (n / 256.0) ** 2
                v = base * (factor(arm, "bench", "reuse iter_total", n, r) if arm == "after" else 1.0)
                row = {"task": "gemm_phases", "device": "cpu", "mode": "reuse", "size": n, "phase": "iter_total",
                       "median_s": v, "iters": 20, "checksum": 1.5}
                if drop_bench and arm == "after" and r == 2 and n == 256:
                    continue
                _w(os.path.join(root, arm, "bench", "gemm_phases_n%d_r%d.jsonl" % (n, r)), json.dumps(row) + "\n")


def f_ok(arm, grp, label, n, r):
    # 処置セル（4096 の resident_public と bench）は大きく改善し、他は不変
    if n == TREAT_N and (label in ("resident_public", "reuse iter_total")):
        return 0.8
    if n == TREAT_N and label == "resident_split":
        return 0.9
    return 1.0


def _run(hostdirs):
    hosts = {n: load_host(hostdirs.get(n), n) for n in HOSTS}
    return judge(hosts)


def self_test():
    fails = []

    def expect(name, cond):
        if not cond:
            fails.append(name)

    with tempfile.TemporaryDirectory() as td:
        # ADOPT: 処置が大きく改善・他は不変。X の差が DL(0.01) の 0.5 以上になるよう f_ok を調整
        def ok_factor(arm, grp, label, n, r):
            return f_ok(arm, grp, label, n, r)

        # rho_REM = (X_before - X_after)/0.01。resident_public 4096 の base は 0.001*256*1.01 =~0.2586、0.8 倍で 0.0517 減 => rho=5.17>=0.5
        g, m = os.path.join(td, "ok_g"), os.path.join(td, "ok_m")
        make_host(g, "gb10", ok_factor)
        make_host(m, "macmini", ok_factor)
        r = _run({"gb10": g, "macmini": m})
        expect("ADOPT", r["verdict"] == "ADOPT")

        # 判定不能: 非処置セルが全 run でわずかに悪化せず、中央値だけ > 1（1 run だけ改善）
        def noisy(arm, grp, label, n, r):
            if grp == "diag" and label == "fresh" and n == 256:
                return 0.99 if r == 1 else 1.02
            return f_ok(arm, grp, label, n, r)

        g2 = os.path.join(td, "nz_g")
        make_host(g2, "gb10", noisy)
        r = _run({"gb10": g2, "macmini": m})
        expect("判定不能（未実測）にならない: macmini あり", r["verdict"] == "判定不能")

        # REJECT: 非処置セルが 5/5 後退
        def back(arm, grp, label, n, r):
            if grp == "diag" and label == "fresh" and n == 512:
                return 1.05
            return f_ok(arm, grp, label, n, r)

        g3 = os.path.join(td, "bk_g")
        make_host(g3, "gb10", back)
        r = _run({"gb10": g3, "macmini": m})
        expect("REJECT 5/5 後退", r["verdict"] == "REJECT")

        # REJECT: 処置セルの中央値 > 1.00
        def treat_bad(arm, grp, label, n, r):
            if n == TREAT_N and label == "reuse iter_total":
                return 1.0 + 0.01 * r if r != 5 else 0.9
            return f_ok(arm, grp, label, n, r)

        # 上は 5/5 後退ではない（run5 が 0.9）が中央値 1.03 > 1.00
        g4 = os.path.join(td, "tb_g")
        make_host(g4, "gb10", treat_bad)
        r = _run({"gb10": g4, "macmini": m})
        expect("REJECT 処置セル > 1.00", r["verdict"] == "REJECT" and any("処置セル" in x for x in r["reject"]))

        # REJECT: checksum 不一致
        g5 = os.path.join(td, "ck_g")
        make_host(g5, "gb10", f_ok, cksum_after=("resident_public", 4096))
        r = _run({"gb10": g5, "macmini": m})
        expect("REJECT checksum", r["verdict"] == "REJECT" and any("checksum" in x for x in r["reject"]))

        # 片機体欠落 → 判定不能（未実測あり）
        r = _run({"gb10": g, "macmini": None})
        expect("片機体欠落", r["verdict"] == "判定不能（未実測あり）")
        r = _run({"gb10": g, "macmini": os.path.join(td, "nonexistent")})
        expect("片機体ディレクトリ無し", r["verdict"] == "判定不能（未実測あり）")

        # record_only（ゲート unpassed）→ 判定に使わず未実測扱い
        g6 = os.path.join(td, "gate_g")
        make_host(g6, "gb10", f_ok, gate="unpassed")
        r = _run({"gb10": g6, "macmini": m})
        expect("ゲート unpassed は record_only", r["verdict"] == "判定不能（未実測あり）")
        g7 = os.path.join(td, "excl_g")
        make_host(g7, "gb10", f_ok, excl="fail")
        r = _run({"gb10": g7, "macmini": m})
        expect("excl=fail は record_only", r["verdict"] == "判定不能（未実測あり）")
        g8 = os.path.join(td, "kind_g")
        make_host(g8, "gb10", f_ok, kind="aux")
        r = _run({"gb10": g8, "macmini": m})
        expect("series_kind!=main は record_only", r["verdict"] == "判定不能（未実測あり）")

        # 欠落 → 例外（診断行・bench ファイル）
        g9 = os.path.join(td, "drop_g")
        make_host(g9, "gb10", f_ok, drop=3)
        try:
            _run({"gb10": g9, "macmini": m})
            expect("診断行の欠落は例外", False)
        except JudgeError:
            pass
        g10 = os.path.join(td, "dropb_g")
        make_host(g10, "gb10", f_ok, drop_bench=True)
        try:
            _run({"gb10": g10, "macmini": m})
            expect("bench ファイル欠落は例外", False)
        except JudgeError:
            pass

        # 壊れた JSON は例外
        g11 = os.path.join(td, "bad_g")
        make_host(g11, "gb10", f_ok)
        _w(os.path.join(g11, "before", "run1.jsonl"), "{not json\n")
        try:
            _run({"gb10": g11, "macmini": m})
            expect("JSON 不正は例外", False)
        except JudgeError:
            pass
        # ゲート記録欠落は例外
        g12 = os.path.join(td, "nogate_g")
        make_host(g12, "gb10", f_ok)
        os.remove(os.path.join(g12, "load_gate_status.txt"))
        try:
            _run({"gb10": g12, "macmini": m})
            expect("ゲート記録欠落は例外", False)
        except JudgeError:
            pass
        # 描画が例外にならない
        render(_run({"gb10": g, "macmini": m}), None)
    if fails:
        print("self-test FAIL: %s" % ", ".join(fails), file=sys.stderr)
        return 1
    print("self-test OK")
    return 0


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", action="append", default=[], help="gb10=<dir> / macmini=<dir>")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    dirs = {}
    for spec in a.host:
        if "=" not in spec:
            print("--host は name=dir 形式", file=sys.stderr)
            return 2
        name, d = spec.split("=", 1)
        if name not in HOSTS or name in dirs:
            print("host 名が不正または重複: %s" % name, file=sys.stderr)
            return 2
        dirs[name] = d
    if not dirs:
        print("--host が必要", file=sys.stderr)
        return 2
    try:
        hosts = {n: load_host(dirs.get(n), n) for n in HOSTS}
        res = judge(hosts)
    except JudgeError as e:
        print("入力不備（fail-closed）: %s" % e, file=sys.stderr)
        return 2
    text = render(res, hosts)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
