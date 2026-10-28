#!/usr/bin/env python3
"""イシュー #3679 の再現判定器（RULE.txt を機械的に適用する。標準ライブラリのみ・fail-closed）。

役割: orchestrate.sh が機体・系列ごとに残す before／after × 5 run の診断 JSONL を読み、
T1〜T3 の 3 セルについて「再現した／再現しない／判定不能」を出力する。閾値は RULE.txt の値を
そのまま定数にしたもので、ここで新しい基準を作らない。呼び出し元: session-scripts/session-3679.sh
（系列ごとの参考判定）と、回収後の手動実行（系列を揃えた最終判定 judge-3679.md）。

使い方:
  repro_judge.py --host gb10=<s1>,<s2>,... [--host macmini=<s1>,...] [--out <md>] [--rule <RULE.txt>]
  repro_judge.py --self-test
  系列ディレクトリは起動順に並べる（RULE: ゲートを pass した最初の 3 系列を使う。値で選ばない）。構成は次のとおり。
    env_info.txt（rule_sha256・host_kind）/ series_kind.txt / first_arm.txt / load_gate_status.txt / excl_status.txt（gb10 のみ）
    before/run{1..5}.jsonl  after/run{1..5}.jsonl   （診断。DIAG_JSON 行）

fail-closed: run 数が 5 でない・行の欠落／重複・n_samples の不一致・JSON 解析不能・ゲート記録の欠落・
RULE の sha256 不一致・系列が 5 を超える場合は例外にして終了コード 2。
判定に使わない区間行は median_s が 0.0 でも入力不備にしない（Mac のタイマー分解能。#3672 で事後に直した点）。
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

SIZES = [256, 512, 1024, 2048, 4096]
RUNS = [1, 2, 3, 4, 5]
DIAG_ARMS = [
    "fresh", "resident_public", "resident_reupload", "resident_split",
    "backend_drop", "gemm_device_hold", "gemm_device_drop_in", "gemm_device_download",
]
ARMS = ["before", "after"]
# 判定対象セル（RULE.txt。機体固定）: (名前, 機体, 診断の腕, N, 切り分け対象の区間)
CELLS = [
    ("T1", "gb10", "resident_split", 512, ["gemm_device", "download", "out_drop", "readout", "checksum"]),
    ("T2", "macmini", "resident_public", 256, ["matmul", "readout", "checksum"]),
    ("T3", "macmini", "gemm_device_download", 4096, []),
]
# RULE.txt の閾値（変更しない）
NEED_SERIES = 3
MAX_SERIES = 5
K_REPRODUCED = 12
K_NOT_REPRODUCED = 7


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


def pos_float(v, what):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
        raise JudgeError("正の有限数でない %s: %r" % (what, v))
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
    for a in DIAG_ARMS:
        for n in SIZES:
            k = (a, "iter_total", n)
            if k not in out:
                raise JudgeError("診断行が欠落: %s %r" % (path, k))
            pos_float(out[k][0], "median_s %r" % (k,))
            if not isinstance(out[k][2], str) or not out[k][2]:
                raise JudgeError("iter_total の checksum_bits が無い: %s %r" % (path, k))
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
    runs = {(arm, r): load_run(os.path.join(root, arm, "run%d.jsonl" % r)) for arm in ARMS for r in RUNS}
    # RULE.txt が記録を要求する区間行（T1／T2）の存在を検証する。欠落は入力不備。
    # 実在する median_s=0.0 は計時分解能による欠測として許容し、ratio 側で None（有効 run 数に数えない）にする。
    for c in CELLS:
        if c[1] != host:
            continue
        for ph in c[4]:
            for (arm_name, r), rows in runs.items():
                v = rows.get((c[2], ph, c[3]))
                if v is None:
                    raise JudgeError("区間行が欠落: %s %s/run%d %r" % (root, arm_name, r, (c[2], ph, c[3])))
                m = v[0]
                if isinstance(m, bool) or not isinstance(m, (int, float)) or not math.isfinite(m) or m < 0:
                    raise JudgeError("区間行の median_s が 0 以上の有限数でない: %s %r" % (root, (c[2], ph, c[3])))
    # 反復数（n_samples）は全 run・両腕で同一でなければ比較不能
    for key in runs[("before", 1)]:
        vals = {runs[(arm, r)][key][1] for arm in ARMS for r in RUNS if key in runs[(arm, r)]}
        if len(vals) != 1:
            raise JudgeError("n_samples が腕・run 間で不一致: %s %r" % (root, key))
    reasons = []
    if kind[0] != "main":
        reasons.append("series_kind=%s" % kind[0])
    bad = sorted(r for r, g in gate_by_run.items() if g != "pass")
    if bad:
        reasons.append("ゲート未通過の run: %s" % ",".join(bad))
    if excl is not None and excl != "pass":
        reasons.append("専有 excl=%s" % excl)
    return {"root": root, "runs": runs, "first": first[0], "reasons": reasons}


def ratios_of(series, arm, n, phase="iter_total"):
    """run 順の ratio_r（after / before）。分母が 0 の区間行は None。"""
    out = []
    for r in RUNS:
        b = series["runs"][("before", r)].get((arm, phase, n))
        a = series["runs"][("after", r)].get((arm, phase, n))
        if b is None or a is None or not b[0] or not a[0]:
            out.append(None)
        else:
            out.append(a[0] / b[0])
    return out


def judge_cell(cell, selected, n_launched):
    name, _host, arm, n, phases = cell
    res = {"name": name, "arm": arm, "n": n, "series": [], "k": None, "median": None, "verdict": None, "why": ""}
    if len(selected) < NEED_SERIES:
        res["verdict"] = "判定不能（系列不足）"
        res["why"] = "ゲートを pass した系列が %d（起動 %d）。3 未満" % (len(selected), n_launched)
        return res
    use = selected[:NEED_SERIES]
    sums = {s["runs"][(a, r)][(arm, "iter_total", n)][2] for s in use for a in ARMS for r in RUNS}
    if len(sums) != 1:
        res["verdict"] = "判定不能（checksum）"
        res["why"] = "checksum_bits が全 run で一致しない: %s" % sorted(sums)
        return res
    allr = []
    for s in use:
        rs = ratios_of(s, arm, n)
        allr += rs
        res["series"].append({"root": s["root"], "first": s["first"], "ratios": rs,
                              "median": statistics.median(rs), "all_above": all(x > 1.0 for x in rs)})
    k = sum(1 for x in allr if x > 1.0)
    med = statistics.median(allr)
    res["k"], res["median"] = k, med
    all_series_above = all(sr["median"] > 1.0 for sr in res["series"])
    if k >= K_REPRODUCED and all_series_above:
        res["verdict"] = "再現した"
    elif k <= K_NOT_REPRODUCED:
        res["verdict"] = "再現しない"
    else:
        res["verdict"] = "判定不能"
        res["why"] = "8 <= k <= 11、または k >= 12 で中央値 <= 1.00 の系列がある"
    if res["verdict"] == "再現した" and phases:
        res["phase_table"] = []
        for ph in phases:
            per = []
            for s in use:
                per += ratios_of(s, arm, n, ph)
            vals = [x for x in per if x is not None]
            res["phase_table"].append((ph, statistics.median(vals) if vals else None, len(vals)))
    return res


def noise_counts(series, host):
    """T1〜T3 以外の診断セルで 5/5 > 1.00・5/5 < 1.00 になった件数（記録のみ）。"""
    targets = {(c[2], c[3]) for c in CELLS if c[1] == host}
    up = down = total = 0
    for a in DIAG_ARMS:
        for n in SIZES:
            if (a, n) in targets:
                continue
            rs = ratios_of(series, a, n)
            total += 1
            up += all(x > 1.0 for x in rs)
            down += all(x < 1.0 for x in rs)
    return up, down, total


def judge_all(host_dirs, rule_path):
    with open(rule_path, "rb") as f:
        rule_sha = hashlib.sha256(f.read()).hexdigest()
    out = {"rule_sha": rule_sha, "hosts": {}}
    for host, dirs in host_dirs.items():
        if len(dirs) > MAX_SERIES:
            raise JudgeError("系列が %d を超える: %s" % (MAX_SERIES, host))
        # 同一ディレクトリの重複指定は別セッション 3 系列を満たさないため入力不備（realpath で正規化して検査）
        real = [os.path.realpath(d) for d in dirs]
        if len(set(real)) != len(real):
            raise JudgeError("系列ディレクトリが重複している: %s" % host)
        loaded = [load_series(d, host, rule_sha) for d in dirs]
        selected = [s for s in loaded if not s["reasons"]][:NEED_SERIES]
        out["hosts"][host] = {"loaded": loaded, "selected": selected,
                              "noise": [noise_counts(s, host) for s in selected]}
    out["cells"] = []
    for c in CELLS:
        h = out["hosts"].get(c[1])
        if h is None:
            out["cells"].append({"name": c[0], "arm": c[2], "n": c[3], "series": [], "k": None, "median": None,
                                 "verdict": "未実測", "why": "%s のデータ無し" % c[1]})
        else:
            out["cells"].append(judge_cell(c, h["selected"], len(h["loaded"])))
    return out


def render(res):
    L = ["# #3679 再現判定（repro_judge.py の出力。手で書き換えない）", "", "RULE.txt sha256: `%s`" % res["rule_sha"], ""]
    L += ["## 結論", "", "| セル | 機体 | 腕 | N | 結論 | k/15 | 15 個の中央値 ratio |", "|---|---|---|---|---|---|---|"]
    for c in res["cells"]:
        host = [x[1] for x in CELLS if x[0] == c["name"]][0]
        L.append("| %s | %s | %s | %d | **%s** | %s | %s |" % (
            c["name"], host, c["arm"], c["n"], c["verdict"],
            "-" if c["k"] is None else c["k"], "-" if c["median"] is None else "%.4f" % c["median"]))
    L.append("")
    for c in res["cells"]:
        if c["why"]:
            L.append("- %s: %s" % (c["name"], c["why"]))
    L += ["", "## 系列の扱い", ""]
    for host, h in res["hosts"].items():
        for s in h["loaded"]:
            used = "判定に使用" if s in h["selected"] else ("record_only" if s["reasons"] else "余剰（判定に使わない）")
            L.append("- %s %s（先頭腕 %s）: %s%s" % (host, os.path.basename(s["root"].rstrip("/")), s["first"], used,
                                                 "（%s）" % " / ".join(s["reasons"]) if s["reasons"] else ""))
    L += ["", "## セルごとの ratio_r（run1..5。系列順）", ""]
    for c in res["cells"]:
        for i, sr in enumerate(c["series"], 1):
            L.append("- %s 系列%d（先頭腕 %s）: %s / 中央値 %.4f / 5/5 > 1.00: %s" % (
                c["name"], i, sr["first"], " ".join("%.4f" % x for x in sr["ratios"]), sr["median"],
                "はい" if sr["all_above"] else "いいえ"))
    L += ["", "## ノイズの参考（T1〜T3 以外の診断セル。系列ごと。記録のみ）", ""]
    for host, h in res["hosts"].items():
        for i, (up, down, tot) in enumerate(h["noise"], 1):
            L.append("- %s 系列%d: 5/5 > 1.00 が %d 件・5/5 < 1.00 が %d 件（%d セル中）" % (host, i, up, down, tot))
    for c in res["cells"]:
        if c.get("phase_table"):
            L += ["", "## %s の区間行（再現したセルのみ。15 個の ratio の中央値。記録のみ）" % c["name"], "",
                  "| 区間 | 中央値 ratio | 有効 run 数 |", "|---|---|---|"]
            for ph, m, nv in c["phase_table"]:
                L.append("| %s | %s | %d |" % (ph, "n/a（計時分解能による欠測。有効 run 0）" if m is None else "%.4f" % m, nv))
    L.append("")
    return "\n".join(L)


# ---------------- self-test ----------------

def _w(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def make_series(root, host, rule_sha, fn, first="before", kind="main", gate="pass", excl="pass",
                cksum=None, drop_run=False, dup=False, bad_sha=False, zero_phase=False, no_phase=False):
    """fn(arm, diag_arm, n, run) -> after／before の median_s。"""
    _w(os.path.join(root, "env_info.txt"), "rule_sha256: %s\nhost_kind: %s\n" % ("0" * 64 if bad_sha else rule_sha, host))
    _w(os.path.join(root, "series_kind.txt"), kind + "\n")
    _w(os.path.join(root, "first_arm.txt"), first + "\n")
    _w(os.path.join(root, "load_gate_status.txt"), "".join("run%d gate=%s\n" % (r, gate) for r in RUNS))
    if host == "gb10":
        _w(os.path.join(root, "excl_status.txt"), "main excl=%s\n" % excl)
    for arm in ARMS:
        for r in RUNS:
            if drop_run and arm == "after" and r == 5:
                continue
            rows = []
            for a in DIAG_ARMS:
                for n in SIZES:
                    ck = cksum(arm, a, n, r) if cksum else "c-%s-%d" % (a, n)
                    rows.append({"arm": a, "phase": "iter_total", "n": n, "median_s": fn(arm, a, n, r),
                                 "n_samples": 10, "checksum_bits": ck})
            for c in CELLS:
                for ph in c[4]:
                    if no_phase and ph == c[4][0]:
                        continue
                    m = 0.0 if zero_phase else 1e-4 * (1.2 if arm == "after" else 1.0)
                    rows.append({"arm": c[2], "phase": ph, "n": c[3], "median_s": m,
                                 "n_samples": 10, "checksum_bits": None})
            if dup:
                rows.append(rows[0])
            _w(os.path.join(root, arm, "run%d.jsonl" % r), "".join(json.dumps(x) + "\n" for x in rows))


def self_test():
    ok = True

    def expect(name, cond):
        nonlocal ok
        print("%s: %s" % ("ok  " if cond else "FAIL", name))
        ok = ok and cond

    with tempfile.TemporaryDirectory() as tmp:
        rule = os.path.join(tmp, "RULE.txt")
        _w(rule, "dummy rule\n")
        sha = hashlib.sha256(b"dummy rule\n").hexdigest()

        def build(tag, fns, host="macmini", opts=None):
            opts = opts or {}
            dirs = []
            for i, fn in enumerate(fns, 1):
                d = os.path.join(tmp, tag, "s%d" % i)
                k = dict(opts.get(i, {}))
                make_series(d, host, sha, fn, first="before" if i % 2 else "after", **k)
                dirs.append(d)
            return dirs

        def slow(arm, a, n, r):  # after が常に 1.10 倍遅い（全セル再現）
            return 1.0 * (1.10 if arm == "after" else 1.0)

        def fast(arm, a, n, r):
            return 1.0 * (0.90 if arm == "after" else 1.0)

        def mid(arm, a, n, r):  # run ごとに after が速かったり遅かったり。k は 15 個中 8
            return 1.0 * (1.05 if (arm == "after" and r % 2 == 1) else 0.97 if arm == "after" else 1.0)

        def run_judge(dirs, host="macmini"):
            return judge_all({host: dirs}, rule)

        def verdicts(res):
            return {c["name"]: c["verdict"] for c in res["cells"]}

        res = run_judge(build("slow", [slow] * 3))
        v = verdicts(res)
        expect("再現した（T2・T3）", v["T2"] == "再現した" and v["T3"] == "再現した")
        expect("機体データ無しは未実測（T1）", v["T1"] == "未実測")
        expect("再現した T2 は区間表を持つ", bool(next(c for c in res["cells"] if c["name"] == "T2").get("phase_table")))
        expect("render が文字列を返す", "再現した" in render(res))

        res = run_judge(build("fast", [fast] * 3))
        expect("再現しない", verdicts(res)["T2"] == "再現しない")

        res = run_judge(build("mid", [mid] * 3))
        c = next(x for x in res["cells"] if x["name"] == "T2")
        expect("k=9 は判定不能（k=%s）" % c["k"], c["verdict"] == "判定不能" and c["k"] == 9)

        # k >= 12 でも 1 系列の中央値が 1.00 以下 → 判定不能
        def s_bad(arm, a, n, r):  # after が r<=2 だけ遅い（中央値 < 1）
            return 1.0 * (1.05 if (arm == "after" and r <= 2) else 0.97 if arm == "after" else 1.0)

        res = run_judge(build("k12", [slow, slow, s_bad]))
        c = next(x for x in res["cells"] if x["name"] == "T2")
        expect("k>=12 でも系列の中央値 <= 1 なら判定不能（k=%s）" % c["k"], c["k"] == 12 and c["verdict"] == "判定不能")

        def s_half(arm, a, n, r):  # 3 run 遅い・2 run 速い → 系列中央値 > 1、k=3
            return 1.0 * (1.05 if (arm == "after" and r <= 3) else 0.97 if arm == "after" else 1.0)

        res = run_judge(build("k12b", [slow, slow, s_half]))
        c = next(x for x in res["cells"] if x["name"] == "T2")
        expect("k=13 かつ全系列の中央値 > 1 は再現した（k=%s）" % c["k"], c["k"] == 13 and c["verdict"] == "再現した")

        # 系列不足: 2 系列のみ／record_only 混在
        res = run_judge(build("short", [slow] * 2))
        expect("系列不足", verdicts(res)["T2"] == "判定不能（系列不足）")
        res = run_judge(build("rec", [slow] * 4, opts={2: {"gate": "unpassed"}}))
        expect("record_only 1 本を除き 3 本で判定", verdicts(res)["T2"] == "再現した" and len(res["hosts"]["macmini"]["selected"]) == 3)
        res = run_judge(build("rec2", [slow] * 3, opts={3: {"gate": "unpassed"}}))
        expect("record_only で 2 本なら系列不足", verdicts(res)["T2"] == "判定不能（系列不足）")
        res = run_judge(build("excl", [slow] * 3, host="gb10", opts={1: {"excl": "fail"}}), host="gb10")
        expect("GB10 の専有 fail は record_only", verdicts(res)["T1"] == "判定不能（系列不足）")

        # checksum 不一致
        res = run_judge(build("ck", [slow] * 3, opts={2: {"cksum": lambda arm, a, n, r: "x" if (arm == "after" and r == 2) else "c-%s-%d" % (a, n)}}))
        expect("checksum 不一致", verdicts(res)["T2"] == "判定不能（checksum）")

        # 区間行 median_s=0.0 でも入力不備にしない
        res = run_judge(build("zero", [slow] * 3, host="gb10", opts={i: {"zero_phase": True} for i in (1, 2, 3)}), host="gb10")
        expect("区間行 0.0 は許容（T1 再現）", verdicts(res)["T1"] == "再現した")

        # fail-closed
        def raises(dirs, host="macmini"):
            try:
                run_judge(dirs, host)
            except JudgeError:
                return True
            return False

        expect("run 欠落は fail-closed", raises(build("miss", [slow] * 3, opts={1: {"drop_run": True}})))
        expect("行重複は fail-closed", raises(build("dup", [slow] * 3, opts={1: {"dup": True}})))
        expect("RULE sha 不一致は fail-closed", raises(build("sha", [slow] * 3, opts={1: {"bad_sha": True}})))
        expect("系列ディレクトリ無しは fail-closed", raises([os.path.join(tmp, "nonexistent")]))
        expect("区間行の欠落は fail-closed", raises(build("nophase", [slow] * 3, opts={2: {"no_phase": True}})))
        same = build("samedir", [slow], host="macmini")[0]
        expect("同一系列ディレクトリの重複は fail-closed", raises([same, same, same + "/../s1"]))
        expect("系列 6 本は fail-closed", raises(build("six", [slow] * 6)))
    print("self-test: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", action="append", default=[], help="gb10=<dir>,<dir>,... / macmini=...")
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
