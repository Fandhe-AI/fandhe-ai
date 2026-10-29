#!/usr/bin/env python3
"""イシュー #3693 の帰属判定器（RULE.txt を機械的に適用する。標準ライブラリのみ・fail-closed）。

役割: orchestrate.sh が機体・系列ごとに残す before／after 木 x 5 run の診断 JSONL
（cpu_gemm_resident_matmul_split_diag.rs の DIAG_JSON 行）を読み、Mac mini の N=256 について
「主因の区間を特定した／特定できない／判定不能」を出力する。閾値（k >= 12・系列 3・係数 0.5）は
RULE.txt の値をそのまま定数にしたもので、ここで新しい基準を作らない。GB10 と N=512 は同じ手順で
計算するが「記録のみ」と明示する。呼び出し元: session-scripts/session-3693.sh（系列ごとの参考判定）と、
回収後の手動実行（系列を揃えた最終判定 judge-3693.md）。

使い方:
  split_judge.py --host macmini=<s1>,<s2>,... [--host gb10=<s1>,...] [--out <md>] [--rule <RULE.txt>]
  split_judge.py --self-test
  系列ディレクトリは起動順に並べる（RULE: ゲートを pass した最初の 3 系列を使う。値で選ばない）。構成:
    env_info.txt（rule_sha256・host_kind）/ series_kind.txt / first_arm.txt / load_gate_status.txt /
    excl_status.txt（gb10 のみ）/ before/run{1..5}.jsonl  after/run{1..5}.jsonl

fail-closed: run 数が 5 でない・行の欠落／重複・n_samples の不一致・JSON 解析不能・ゲート記録の欠落・
RULE の sha256 不一致・系列が 5 を超える場合は例外にして終了コード 2。
区間行の median_s は 0.0 を許容する（Mac のタイマー分解能）。iter_total は正の有限数を要求する。
終了コード: 0=判定を出した（どの結論でも 0）、2=入力不備、1=self-test 失敗。
"""
import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys
import tempfile

SIZES = [256, 512]
RUNS = [1, 2, 3, 4, 5]
TREES = ["before", "after"]
# 診断腕ごとの必須区間（iter_total に加える。欠落は入力不備）
PHASES = {
    "resident_public": ["matmul", "readout", "checksum", "result_drop"],
    "mirror_owned": ["gemm_device", "download_owned", "readout", "checksum", "result_drop"],
    "mirror_copy": ["gemm_device", "download", "out_drop", "readout", "checksum", "result_drop"],
    "alloc_probe_owned": ["alloc", "touch", "wrap", "result_drop"],
    "alloc_probe_copy": ["alloc", "touch", "copy", "src_drop", "result_drop"],
}
CHECKSUM_ARMS = ["resident_public", "mirror_owned", "mirror_copy"]
# RULE.txt の閾値（変更しない）
NEED_SERIES = 3
MAX_SERIES = 5
K_REPRODUCED = 12
MAIN_CAUSE_COEFF = 0.5
# 判定対象: macmini の N=256 のみ。ほかは記録のみ。
DECISIVE = ("macmini", 256)
# 帰属の区間群: (名前, owned 側の区間, copy 側の区間)
GROUPS = [
    ("S1", ["gemm_device"], ["gemm_device"]),
    ("S2", ["download_owned"], ["download", "out_drop"]),
    ("S3", ["readout"], ["readout"]),
    ("S4", ["checksum"], ["checksum"]),
]


class JudgeError(Exception):
    """入力不備（fail-closed）。"""


def read_lines(path, pat=None):
    if not os.path.isfile(path):
        raise JudgeError("ファイルが無い: %s" % path)
    with open(path, encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    if pat is not None:
        for ln in lines:
            if not re.match(pat, ln):
                raise JudgeError("解析不能な行 %r: %s" % (ln, path))
    return lines


def finite(v, what, positive):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise JudgeError("有限数でない %s: %r" % (what, v))
    if v < 0 or (positive and v == 0):
        raise JudgeError("%s が範囲外: %r" % (what, v))
    return float(v)


def load_run(path):
    """{(arm, phase, n): (median_s, n_samples, checksum_bits)}。重複・欠落は例外。"""
    if not os.path.isfile(path):
        raise JudgeError("ファイルが無い: %s" % path)
    out = {}
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
            try:
                key = (d["arm"], d["phase"], d["n"])
                val = (d["median_s"], d["n_samples"], d.get("checksum_bits"))
            except KeyError as e:
                raise JudgeError("診断行にキーが無い: %s:%d (%s)" % (path, i, e))
            if key in out:
                raise JudgeError("診断行が重複: %s %r" % (path, key))
            out[key] = val
    for arm, phases in PHASES.items():
        for n in SIZES:
            k = (arm, "iter_total", n)
            if k not in out:
                raise JudgeError("診断行が欠落: %s %r" % (path, k))
            finite(out[k][0], "median_s %r" % (k,), True)
            if not isinstance(out[k][2], str) or not out[k][2]:
                raise JudgeError("iter_total の checksum_bits が無い: %s %r" % (path, k))
            for ph in phases:
                kp = (arm, ph, n)
                if kp not in out:
                    raise JudgeError("区間行が欠落: %s %r" % (path, kp))
                finite(out[kp][0], "median_s %r" % (kp,), False)
    return out


def load_series(root, host, rule_sha):
    if not os.path.isdir(root):
        raise JudgeError("系列ディレクトリが無い: %s" % root)
    kind = read_lines(os.path.join(root, "series_kind.txt"), r"^[a-z]+$")
    first = read_lines(os.path.join(root, "first_arm.txt"), r"^(before|after)$")
    gates = read_lines(os.path.join(root, "load_gate_status.txt"), r"^run[1-9][0-9]* gate=[a-z_]+$")
    env = read_lines(os.path.join(root, "env_info.txt"))
    env_sha = [ln.split(": ", 1)[1] for ln in env if ln.startswith("rule_sha256: ")]
    if env_sha != [rule_sha]:
        raise JudgeError("RULE の sha256 が env_info と合わない: %s" % root)
    if ("host_kind: %s" % host) not in env:
        raise JudgeError("env_info の host_kind が %s でない: %s" % (host, root))
    if len(kind) != 1 or len(first) != 1:
        raise JudgeError("series_kind／first_arm が 1 行でない: %s" % root)
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
        ex = read_lines(os.path.join(root, "excl_status.txt"), r"^[a-z]+ excl=(pass|fail)$")
        d = dict(x.split(" excl=") for x in ex)
        if "main" not in d:
            raise JudgeError("excl_status に main が無い: %s" % root)
        excl = d["main"]
    runs = {(t, r): load_run(os.path.join(root, t, "run%d.jsonl" % r)) for t in TREES for r in RUNS}
    # 反復数（n_samples）は全 run・両木で同一でなければ比較不能
    for key in runs[("before", 1)]:
        vals = {runs[(t, r)][key][1] for t in TREES for r in RUNS if key in runs[(t, r)]}
        if len(vals) != 1:
            raise JudgeError("n_samples が木・run 間で不一致: %s %r" % (root, key))
    reasons = []
    if kind[0] != "main":
        reasons.append("series_kind=%s" % kind[0])
    bad = sorted(r for r, g in gate_by_run.items() if g != "pass")
    if bad:
        reasons.append("ゲート未通過の run: %s" % ",".join(bad))
    if excl is not None and excl != "pass":
        reasons.append("専有 excl=%s" % excl)
    return {"root": root, "runs": runs, "first": first[0], "reasons": reasons}


def med(row):
    return row[0]


def ratio_vals(series_list, num_fn, den_fn):
    """系列ごとの run 順の比（num/den）。分母 0 は None。戻り値は系列ごとのリストのリスト。"""
    out = []
    for s in series_list:
        rs = []
        for r in RUNS:
            nu, de = num_fn(s, r), den_fn(s, r)
            rs.append(None if not de or nu is None else nu / de)
        out.append(rs)
    return out


def cond_met(per_series):
    """k >= 12（> 1.00 の個数。None は数えない）かつ 3 系列それぞれの中央値 > 1.00。"""
    allv = [x for rs in per_series for x in rs]
    if any(x is None for x in allv):
        return False, None, None
    k = sum(1 for x in allv if x > 1.0)
    meds = [statistics.median(rs) for rs in per_series]
    return (k >= K_REPRODUCED and all(m > 1.0 for m in meds)), k, statistics.median(allv)


def phase_sum(rows, arm, phases, n):
    return sum(med(rows[(arm, p, n)]) for p in phases)


def attribution(series_list, tree, n):
    """tree の木の run ごとの d = owned - copy（15 個）の中央値。"""
    res = {}
    ds = {"D_iter": [], "residual": [], "result_drop": []}
    for g in GROUPS:
        ds[g[0]] = []
    for s in series_list:
        for r in RUNS:
            rows = s["runs"][(tree, r)]
            o_it = med(rows[("mirror_owned", "iter_total", n)])
            c_it = med(rows[("mirror_copy", "iter_total", n)])
            ds["D_iter"].append(o_it - c_it)
            o_res = o_it - phase_sum(rows, "mirror_owned", ["gemm_device", "download_owned", "readout", "checksum"], n)
            c_res = c_it - phase_sum(rows, "mirror_copy", ["gemm_device", "download", "out_drop", "readout", "checksum"], n)
            ds["residual"].append(o_res - c_res)
            ds["result_drop"].append(med(rows[("mirror_owned", "result_drop", n)]) - med(rows[("mirror_copy", "result_drop", n)]))
            for name, op, cp in GROUPS:
                ds[name].append(phase_sum(rows, "mirror_owned", op, n) - phase_sum(rows, "mirror_copy", cp, n))
    for k, v in ds.items():
        res[k] = statistics.median(v)
    return res


def judge_cell(host, n, selected, n_launched):
    decisive = (host, n) == DECISIVE
    res = {"host": host, "n": n, "decisive": decisive, "verdict": None, "why": "", "records": {}}
    if len(selected) < NEED_SERIES:
        res["verdict"] = "判定不能（系列不足）"
        res["why"] = "ゲートを pass した系列が %d（起動 %d）。3 未満" % (len(selected), n_launched)
        return res
    use = selected[:NEED_SERIES]
    sums = {s["runs"][(t, r)][(a, "iter_total", n)][2] for s in use for t in TREES for r in RUNS for a in CHECKSUM_ARMS}
    if len(sums) != 1:
        res["verdict"] = "判定不能（checksum）"
        res["why"] = "checksum_bits が全 run・両木・3 診断腕で一致しない: %s" % sorted(sums)
        return res
    rec = res["records"]

    def tree_ratio(arm, phase):
        def f(tree):
            return lambda s, r: med(s["runs"][(tree, r)][(arm, phase, n)])
        return ratio_vals(use, f("after"), f("before"))

    # 記録のみ: 木の間の ratio
    rec["public_phase_ratio"] = {}
    for ph in ["iter_total"] + PHASES["resident_public"]:
        v = [x for rs in tree_ratio("resident_public", ph) for x in rs if x is not None]
        rec["public_phase_ratio"][ph] = statistics.median(v) if v else None
    mirror_tree = {}
    for arm in ("mirror_owned", "mirror_copy"):
        ok, k, m = cond_met(tree_ratio(arm, "iter_total"))
        mirror_tree[arm] = {"k": k, "median": m, "outside_path": ok}
    rec["mirror_tree_ratio"] = mirror_tree
    rec["before_attribution"] = attribution(use, "before", n)
    probe = {}
    probe["iter_total"] = statistics.median(
        med(s["runs"][("after", r)][("alloc_probe_owned", "iter_total", n)])
        - med(s["runs"][("after", r)][("alloc_probe_copy", "iter_total", n)])
        for s in use for r in RUNS)
    rec["alloc_probe_diff_after"] = probe

    # 3. 健全性 A
    okA, kA, mA = cond_met(tree_ratio("resident_public", "iter_total"))
    rec["sanity_A"] = {"k": kA, "median": mA, "met": okA}
    if not okA:
        res["verdict"] = "判定不能（健全性 A: T2 が出ない）"
        res["why"] = "resident_public iter_total の ratio（after/before）が k>=12 かつ全系列の中央値>1.00 を満たさない（k=%s）" % kA
        return res
    # 4. 健全性 B
    pers = []
    for s in use:
        pers.append([med(s["runs"][("after", r)][("mirror_owned", "iter_total", n)])
                     / med(s["runs"][("after", r)][("mirror_copy", "iter_total", n)]) for r in RUNS])
    okB, kB, mB = cond_met(pers)
    att = attribution(use, "after", n)
    rec["sanity_B"] = {"k": kB, "median": mB, "met": okB}
    rec["after_attribution"] = att
    if not okB or att["D_iter"] <= 0:
        res["verdict"] = "判定不能（健全性 B: 写しが T2 を再現しない）"
        res["why"] = "mirror_owned/mirror_copy の ratio が条件を満たさない（k=%s）、または D_iter=%.3e <= 0" % (kB, att["D_iter"])
        return res
    # 5. 帰属
    main = [g[0] for g in GROUPS if att[g[0]] >= MAIN_CAUSE_COEFF * att["D_iter"]]
    res["main"] = main
    if len(main) == 1:
        res["verdict"] = "主因の区間を特定した: %s" % main[0]
    elif len(main) > 1:
        res["verdict"] = "主因の区間を特定した（複数併記。交互作用は断定しない）: %s" % ",".join(main)
    else:
        res["verdict"] = "特定できない"
        res["why"] = "D_s >= 0.5 x D_iter を満たす区間群が無い"
    return res


def judge_all(host_dirs, rule_path):
    with open(rule_path, "rb") as f:
        rule_sha = hashlib.sha256(f.read()).hexdigest()
    out = {"rule_sha": rule_sha, "hosts": {}, "cells": []}
    for host, dirs in host_dirs.items():
        if len(dirs) > MAX_SERIES:
            raise JudgeError("系列が %d を超える: %s" % (MAX_SERIES, host))
        real = [os.path.realpath(d) for d in dirs]
        if len(set(real)) != len(real):
            raise JudgeError("系列ディレクトリが重複している: %s" % host)
        loaded = [load_series(d, host, rule_sha) for d in dirs]
        selected = [s for s in loaded if not s["reasons"]][:NEED_SERIES]
        out["hosts"][host] = {"loaded": loaded, "selected": selected}
        for n in SIZES:
            out["cells"].append(judge_cell(host, n, selected, len(loaded)))
    if DECISIVE[0] not in host_dirs:
        out["cells"].insert(0, {"host": DECISIVE[0], "n": DECISIVE[1], "decisive": True, "verdict": "未実測",
                                "why": "%s のデータ無し" % DECISIVE[0], "records": {}})
    return out


def fmt(x, spec="%.4f"):
    return "-" if x is None else spec % x


def render(res):
    L = ["# #3693 帰属判定（split_judge.py の出力。手で書き換えない）", "", "RULE.txt sha256: `%s`" % res["rule_sha"], ""]
    L += ["## 結論", "", "| 機体 | N | 位置づけ | 結論 |", "|---|---|---|---|"]
    for c in res["cells"]:
        L.append("| %s | %d | %s | **%s** |" % (c["host"], c["n"], "判定対象" if c["decisive"] else "記録のみ", c["verdict"]))
    L.append("")
    for c in res["cells"]:
        if c["why"]:
            L.append("- %s N=%d: %s" % (c["host"], c["n"], c["why"]))
    L += ["", "## 系列の扱い", ""]
    for host, h in res["hosts"].items():
        for s in h["loaded"]:
            used = "判定に使用" if s in h["selected"] else ("record_only" if s["reasons"] else "余剰（判定に使わない）")
            L.append("- %s %s（先頭腕 %s）: %s%s" % (host, os.path.basename(s["root"].rstrip("/")), s["first"], used,
                                                 "（%s）" % " / ".join(s["reasons"]) if s["reasons"] else ""))
    for c in res["cells"]:
        r = c["records"]
        if not r:
            continue
        tag = "%s N=%d（%s）" % (c["host"], c["n"], "判定対象" if c["decisive"] else "記録のみ")
        L += ["", "## %s" % tag, ""]
        if "sanity_A" in r:
            L.append("- 健全性 A（resident_public iter_total の木の間 ratio）: k=%s 中央値=%s 成立=%s" % (
                r["sanity_A"]["k"], fmt(r["sanity_A"]["median"]), r["sanity_A"]["met"]))
        if "sanity_B" in r:
            L.append("- 健全性 B（after の木の owned/copy ratio）: k=%s 中央値=%s 成立=%s" % (
                r["sanity_B"]["k"], fmt(r["sanity_B"]["median"]), r["sanity_B"]["met"]))
        for key, title in (("after_attribution", "after の木の帰属（owned - copy の中央値。秒）"),
                           ("before_attribution", "before の木の同じ表（記録のみ）")):
            if key in r:
                a = r[key]
                L += ["", "### %s" % title, "", "| 項目 | 値 (s) | D_iter 比 |", "|---|---|---|"]
                for k in ["D_iter", "S1", "S2", "S3", "S4", "residual", "result_drop"]:
                    ratio = a[k] / a["D_iter"] if a["D_iter"] else None
                    L.append("| %s | %s | %s |" % (k, "%.3e" % a[k], fmt(ratio, "%.2f")))
        L += ["", "### resident_public の木の間 ratio（after/before の 15 個の中央値。記録のみ）", "",
              "| 区間 | 中央値 ratio |", "|---|---|"]
        for ph, v in r["public_phase_ratio"].items():
            L.append("| %s | %s |" % (ph, fmt(v)))
        L += ["", "### 写し 2 診断腕の木の間 ratio（記録のみ）", ""]
        for arm, v in r["mirror_tree_ratio"].items():
            L.append("- %s: k=%s 中央値=%s 木の差が経路の外にも出ている=%s" % (arm, v["k"], fmt(v["median"]), v["outside_path"]))
        L.append("- alloc_probe（after の木・owned - copy の iter_total 中央値）: %.3e s" % r["alloc_probe_diff_after"]["iter_total"])
    L.append("")
    return "\n".join(L)


# ---------------- self-test ----------------

def _w(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def make_series(root, host, rule_sha, fn, first="before", kind="main", gate="pass", excl="pass",
                ck=None, drop_run=False, dup=False, bad_sha=False):
    """fn(tree, arm, phase, n, run) -> median_s。"""
    _w(os.path.join(root, "env_info.txt"), "rule_sha256: %s\nhost_kind: %s\n" % ("0" * 64 if bad_sha else rule_sha, host))
    _w(os.path.join(root, "series_kind.txt"), kind + "\n")
    _w(os.path.join(root, "first_arm.txt"), first + "\n")
    _w(os.path.join(root, "load_gate_status.txt"), "".join("run%d gate=%s\n" % (r, gate) for r in RUNS))
    if host == "gb10":
        _w(os.path.join(root, "excl_status.txt"), "main excl=%s\n" % excl)
    for t in TREES:
        for r in RUNS:
            if drop_run and t == "after" and r == 5:
                continue
            rows = []
            for arm, phases in PHASES.items():
                for n in SIZES:
                    c = ck(t, arm, n, r) if ck else "c-%d" % n
                    rows.append({"arm": arm, "phase": "iter_total", "n": n, "median_s": fn(t, arm, "iter_total", n, r),
                                 "n_samples": 10, "checksum_bits": c})
                    for ph in phases:
                        rows.append({"arm": arm, "phase": ph, "n": n, "median_s": fn(t, arm, ph, n, r),
                                     "n_samples": 10, "checksum_bits": None})
            if dup:
                rows.append(rows[0])
            _w(os.path.join(root, t, "run%d.jsonl" % r), "".join(json.dumps(x) + "\n" for x in rows))


def self_test():
    ok = True

    def expect(name, cond):
        nonlocal ok
        print("%s: %s" % ("ok  " if cond else "FAIL", name))
        ok = ok and cond

    def model(owned_extra, public_extra=None):
        """after の木だけに追加時間（s）を足す。owned_extra は mirror_owned の区間、public_extra は
        resident_public の区間（既定は owned_extra を matmul/readout/checksum に写したもの）。
        iter_total は区間の和（残差 0）。"""
        base = {"gemm_device": 1.0e-4, "download_owned": 2.0e-5, "download": 2.0e-5, "out_drop": 1.0e-5,
                "readout": 1.0e-7, "checksum": 3.0e-5, "matmul": 1.2e-4, "result_drop": 1.0e-5,
                "alloc": 1e-5, "touch": 1e-5, "wrap": 1e-6, "copy": 1e-5, "src_drop": 1e-6}
        if public_extra is None:
            public_extra = {"matmul": owned_extra.get("gemm_device", 0.0) + owned_extra.get("download_owned", 0.0),
                            "readout": owned_extra.get("readout", 0.0),
                            "checksum": owned_extra.get("checksum", 0.0)}
        iter_parts = {
            "resident_public": ["matmul", "readout", "checksum"],
            "mirror_owned": ["gemm_device", "download_owned", "readout", "checksum"],
            "mirror_copy": ["gemm_device", "download", "out_drop", "readout", "checksum"],
            "alloc_probe_owned": ["alloc", "touch", "wrap"],
            "alloc_probe_copy": ["alloc", "touch", "copy", "src_drop"]}

        def fn(tree, arm, ph, n, r):
            def part(p):
                v = base[p]
                if tree == "after" and arm == "mirror_owned":
                    v += owned_extra.get(p, 0.0)
                if tree == "after" and arm == "resident_public":
                    v += public_extra.get(p, 0.0)
                return v
            if ph == "iter_total":
                return sum(part(p) for p in iter_parts[arm])
            return part(ph)
        return fn

    with tempfile.TemporaryDirectory() as tmp:
        rule = os.path.join(tmp, "RULE.txt")
        _w(rule, "dummy rule\n")
        sha = hashlib.sha256(b"dummy rule\n").hexdigest()

        def build(tag, fn, host="macmini", n_series=3, opts=None):
            opts = opts or {}
            dirs = []
            for i in range(1, n_series + 1):
                d = os.path.join(tmp, tag, "s%d" % i)
                make_series(d, host, sha, fn, first="before" if i % 2 else "after", **dict(opts.get(i, {})))
                dirs.append(d)
            return dirs

        def cell(res, host="macmini", n=256):
            return next(c for c in res["cells"] if c["host"] == host and c["n"] == n)

        def jd(dirs, host="macmini"):
            return judge_all({host: dirs}, rule)

        # 主因 1 つ（S1）。公開経路の matmul は gemm_device 追加分だけ後退させる
        res = jd(build("s1", model({"gemm_device": 2.0e-5, "download_owned": 1.0e-6})))
        c = cell(res)
        expect("主因 S1 を特定（%s）" % c["verdict"], c["verdict"].startswith("主因の区間を特定した: S1"))
        expect("render が文字列を返す", "S1" in render(res))
        expect("N=512 は記録のみ", cell(res, n=512)["decisive"] is False)

        # 主因 S2（gemm_device は同一）
        c = cell(jd(build("s2", model({"download_owned": 2.0e-5}))))
        expect("主因 S2 を特定（%s）" % c["verdict"], c["verdict"].startswith("主因の区間を特定した: S2"))

        # 複数併記（S4 が負で相殺し、S1 と S2 が D_iter の 0.5 以上ずつ）
        c = cell(jd(build("multi", model({"gemm_device": 3e-5, "download_owned": 3e-5, "checksum": -2e-5}))))
        expect("複数併記（%s）" % c["verdict"], "S1,S2" in c["verdict"])

        # 特定できない（S1..S4 が均等に 4 分割され各 0.25）
        c = cell(jd(build("none", model({"gemm_device": 1e-5, "download_owned": 1e-5, "readout": 1e-5, "checksum": 1e-5}))))
        expect("特定できない（%s）" % c["verdict"], c["verdict"] == "特定できない")

        # 健全性 A 不成立（公開経路が後退しない）
        c = cell(jd(build("sa", model({"gemm_device": 2e-5}, public_extra={}))))
        expect("健全性 A 不成立（%s）" % c["verdict"], c["verdict"].startswith("判定不能（健全性 A"))

        # 健全性 B 不成立（写しは差が無いが公開経路は後退）
        c = cell(jd(build("sb", model({}, public_extra={"matmul": 2e-5}))))
        expect("健全性 B 不成立（%s）" % c["verdict"], c["verdict"].startswith("判定不能（健全性 B"))

        # D_iter <= 0（owned の方が速い）かつ公開経路は後退
        c = cell(jd(build("neg", model({"gemm_device": -2e-5}, public_extra={"matmul": 2e-5}))))
        expect("D_iter <= 0 は判定不能（%s）" % c["verdict"], c["verdict"].startswith("判定不能（健全性 B"))

        # 系列不足
        c = cell(jd(build("short", model({"gemm_device": 2e-5}), n_series=2)))
        expect("系列不足", c["verdict"] == "判定不能（系列不足）")
        c = cell(jd(build("rec", model({"gemm_device": 2e-5}), n_series=4, opts={2: {"gate": "unpassed"}})))
        expect("record_only 1 本を除き 3 本で判定", c["verdict"].startswith("主因の区間を特定した"))

        # checksum 不一致
        c = cell(jd(build("ck", model({"gemm_device": 2e-5}),
                          opts={2: {"ck": lambda t, a, n, r: "x" if (t == "after" and a == "mirror_owned" and r == 2) else "c-%d" % n}})))
        expect("checksum 不一致", c["verdict"] == "判定不能（checksum）")

        # GB10 は記録のみ・macmini 無しなら未実測
        res = jd(build("gb", model({"gemm_device": 2e-5}), host="gb10"), host="gb10")
        expect("GB10 は記録のみ", cell(res, "gb10", 256)["decisive"] is False and res["cells"][0]["verdict"] == "未実測")

        # fail-closed
        def raises(dirs, host="macmini"):
            try:
                jd(dirs, host)
            except JudgeError:
                return True
            return False

        f0 = model({})
        expect("run 欠落は fail-closed", raises(build("miss", f0, opts={1: {"drop_run": True}})))
        expect("行重複は fail-closed", raises(build("dup", f0, opts={1: {"dup": True}})))
        expect("RULE sha 不一致は fail-closed", raises(build("sha", f0, opts={1: {"bad_sha": True}})))
        expect("系列ディレクトリ無しは fail-closed", raises([os.path.join(tmp, "nonexistent")]))
        same = build("samedir", f0, n_series=1)[0]
        expect("同一系列ディレクトリの重複は fail-closed", raises([same, same, same + "/../s1"]))
        expect("系列 6 本は fail-closed", raises(build("six", f0, n_series=6)))
    print("self-test: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", action="append", default=[], help="macmini=<dir>,<dir>,... / gb10=...")
    ap.add_argument("--out")
    ap.add_argument("--rule", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "RULE.txt"))
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    host_dirs = {}
    for h in a.host:
        m = re.match(r"^(gb10|macmini)=(.+)$", h)
        if not m or m.group(1) in host_dirs:
            print("--host の形式が不正または重複: %s" % h, file=sys.stderr)
            return 2
        host_dirs[m.group(1)] = m.group(2).split(",")
    if not host_dirs:
        print("--host が必要", file=sys.stderr)
        return 2
    try:
        text = render(judge_all(host_dirs, a.rule))
    except JudgeError as e:
        print("入力不備（fail-closed）: %s" % e, file=sys.stderr)
        return 2
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
