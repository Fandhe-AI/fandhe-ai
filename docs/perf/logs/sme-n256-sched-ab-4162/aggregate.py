#!/usr/bin/env python3
"""イシュー #4162: N=256 の SME スレッド数・分配の事前登録 A/B の厳密集計。

役割: RULE.txt（同ディレクトリ・計測前に固定）の判定を、実測ログから機械的に出力する。
`docs/perf/logs/sme-default-on-ab-4156/aggregate.py` の流儀（厳格パース・欠損や重複は例外に写し
undetermined にする・FAIL を最優先）を継承した 7 腕版。標準ライブラリのみ。`python3 -I` で起動すること。

使い方:
    python3 -I aggregate.py <収録ディレクトリ>   # 例: macmini（ab/・gates.txt・load_gate.log を含む）
    python3 -I aggregate.py --self-test
    python3 -I aggregate.py --check-template      # RULE.template.txt の必須項目検査
    python3 -I aggregate.py --fingerprint         # 本スクリプト・patch・orchestrate の sha256 を出力

入力（収録ディレクトリ直下）:
    ab/results-<arm>-<task>.jsonl   arm は ARMS、task は gemm / train。各行 bench-fandhe の 1 実行（round 順）
    ab/results-torch-<task>.jsonl   PyTorch（記録のみ。無くてもよい）
    gates.txt                       行形式 `<GATE> <arm|all> <ok|fail|undetermined>`（P-TREE 等。下記 GATE_KEYS）
    load_gate.log                   行形式 `round<r> gate=pass gate_ok=1 ...`（5/5 round が gate_ok=1）

腕の向き: 比 = 候補 / base（1.00 以下なら候補が速い）。aa は base と同一バイナリの A/A（ノイズ幅の記録のみ）。
"""
import glob
import hashlib
import json
import math
import os
import re
import statistics
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROUNDS = 5
ARMS = ("base", "aa", "t4", "t6", "jpw1", "jpw4", "t4-jpw1")
CANDIDATES = ("t4", "t6", "jpw1", "jpw4", "t4-jpw1")
# 変更パラメータ数（tie-break 用）。jpw だけの腕は新しいスレッドを立てない。
PARAM_COUNT = {"t4": 1, "t6": 1, "jpw1": 1, "jpw4": 1, "t4-jpw1": 2}
JPW_ONLY = ("jpw1", "jpw4")
JUDGE = (("gemm", 256, "fresh"), ("gemm", 256, "reuse"))
GUARD = (("gemm", 512, "fresh"), ("gemm", 512, "reuse"), ("train", 64, "fresh"), ("train", 64, "reuse"))
CELLS = JUDGE + GUARD
TASKS = ("gemm", "train")
TIE_EPS = 0.005
# 前提ゲート。all は全腕共通、arm は腕ごと。
GATES_ALL = ("P-TREE", "P-R0", "P-LOCK", "P-CELLS-RUNAB", "P-COLLECT")
GATES_ARM = ("P-FIRE", "P-RT", "P-BIT")


class AggError(Exception):
    pass


def finite_num(v):
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return False
    try:
        return math.isfinite(v)
    except (OverflowError, ValueError, TypeError):
        return False


def read_lines(path):
    if not os.path.isfile(path):
        raise AggError("ログ欠損: " + os.path.basename(path))
    with open(path, encoding="utf-8") as f:
        return f.read().splitlines()


def load_arm(ab, arm, problems):
    """arm の全セルの行（round 順）を返す。欠損・不正は problems へ収集し例外で打ち切らない。"""
    out = {}
    for task in TASKS:
        path = os.path.join(ab, "results-%s-%s.jsonl" % (arm, task))
        try:
            lines = read_lines(path)
        except AggError as e:
            problems.append("arm=%s: %s" % (arm, e))
            continue
        for i, line in enumerate(lines, 1):
            if not line.strip():
                continue
            where = "%s:%d" % (os.path.basename(path), i)
            try:
                o = json.loads(line)
            except (ValueError, RecursionError):
                problems.append(where + " JSON 不正")
                continue
            if not isinstance(o, dict) or o.get("task") != task or o.get("device") != "cpu":
                problems.append(where + " task/device が不正")
                continue
            size, mode = o.get("size"), o.get("mode", "fresh")
            if isinstance(size, bool) or not isinstance(size, int) or not isinstance(mode, str):
                problems.append(where + " size/mode の型が不正")
                continue
            if (task, size, mode) not in CELLS:
                problems.append(where + " 未知のセル: %s %s %s" % (task, size, mode))
                continue
            if not finite_num(o.get("median_s")) or o["median_s"] <= 0 or not finite_num(o.get("checksum")):
                problems.append(where + " median_s/checksum が不正・非有限")
                continue
            out.setdefault((task, size, mode), []).append(o)
    for cell in CELLS:
        n = len(out.get(cell, []))
        if n != ROUNDS:
            problems.append("P-CELLS 不成立: arm=%s cell=%s は %d 行（%d 行ちょうどを要求）" % (arm, cell, n, ROUNDS))
    return out


def read_gates(d, problems):
    """gates.txt を厳格に読む。重複・未知のキー・列挙外は problems。戻り値は (gate, arm) -> result。"""
    got = {}
    try:
        lines = read_lines(os.path.join(d, "gates.txt"))
    except AggError as e:
        problems.append(str(e))
        return got
    for i, line in enumerate(lines, 1):
        if not line.strip():
            continue
        p = line.split()
        if len(p) != 3 or p[2] not in ("ok", "fail", "undetermined"):
            problems.append("gates.txt:%d 形式不正" % i)
            continue
        gate, arm, res = p
        if (gate in GATES_ALL and arm == "all") or (gate in GATES_ARM and arm in ARMS):
            if (gate, arm) in got:
                problems.append("gates.txt:%d 重複 %s %s" % (i, gate, arm))
            got[(gate, arm)] = res
        else:
            problems.append("gates.txt:%d 未知のゲート/腕 %s %s" % (i, gate, arm))
    for g in GATES_ALL:
        if (g, "all") not in got:
            problems.append("gates.txt: %s all が欠落" % g)
    for g in GATES_ARM:
        for a in ARMS:
            if (g, a) not in got:
                problems.append("gates.txt: %s %s が欠落" % (g, a))
    return got


def read_load_gate(d, problems):
    try:
        lines = read_lines(os.path.join(d, "load_gate.log"))
    except AggError as e:
        problems.append(str(e))
        return
    seen = {}
    for line in lines:
        m = re.match(r"^round(\d+) gate=(\S+) gate_ok=([01])\b", line)
        if m:
            r = int(m.group(1))
            if r in seen:
                problems.append("load_gate.log: round%d が重複" % r)
            seen[r] = m.group(3)
    for r in range(1, ROUNDS + 1):
        if seen.get(r) != "1":
            problems.append("P-GATE 不成立: round%d が gate_ok=1 でない" % r)


def load_torch(d):
    """PyTorch の gemm256 / gemm512 fresh の median_s 列（round 順）。記録のみ。欠損は None。"""
    res = {}
    p = os.path.join(d, "ab", "results-torch-gemm.jsonl")
    if not os.path.isfile(p):
        return None
    try:
        for line in open(p, encoding="utf-8"):
            if not line.strip():
                continue
            o = json.loads(line)
            if isinstance(o, dict) and o.get("task") == "gemm" and finite_num(o.get("median_s")) and o["median_s"] > 0:
                res.setdefault(o.get("size"), []).append(o["median_s"])
    except (ValueError, OSError):
        return None
    for s in (256, 512):
        if len(res.get(s, [])) != ROUNDS:
            return None
    return res


def ratios(arm_cells, base_cells, cell):
    return [arm_cells[cell][i]["median_s"] / base_cells[cell][i]["median_s"] for i in range(ROUNDS)]


def gmean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def judge(d):
    """判定を dict で返す。例外は投げず、異常は problems → undetermined に写す。"""
    problems = []
    res = {"problems": problems, "verdict": None, "selected": None, "table": {}, "notes": []}
    ab = os.path.join(d, "ab")
    cells = {a: load_arm(ab, a, problems) for a in ARMS}
    gates = read_gates(d, problems)
    read_load_gate(d, problems)

    # checksum 完全一致（読めた行だけで確認。FAIL は常に最優先）
    ck_fail = []
    for cell in CELLS:
        vals = set()
        for a in ARMS:
            for o in cells[a].get(cell, []):
                vals.add(o["checksum"])
        if len(vals) > 1:
            ck_fail.append(cell)
    res["checksum_fail"] = ck_fail
    # 構造ゲートの fail（P-BIT・P-FIRE・P-RT）は腕単位
    gate_fail = sorted("%s/%s" % (g, a) for (g, a), r in gates.items() if r == "fail" and g in ("P-BIT",))
    res["gate_fail"] = gate_fail
    if ck_fail or gate_fail:
        res["verdict"] = "FAIL"
        return res
    gate_bad = sorted("%s/%s=%s" % (g, a, r) for (g, a), r in gates.items() if r != "ok")
    if gate_bad:
        problems.append("前提ゲート不成立: " + ", ".join(gate_bad))
    if problems:
        res["verdict"] = "undetermined"
        return res

    base = cells["base"]
    # 候補ごとの比
    regress = {}
    elig = {}
    for a in CANDIDATES:
        per = {c: ratios(cells[a], base, c) for c in CELLS}
        med = {c: statistics.median(per[c]) for c in CELLS}
        judge_ok = all(med[c] <= 1.00 and all(r <= 1.00 for r in per[c]) for c in JUDGE)
        judge_reg = any(all(r > 1.00 for r in per[c]) for c in JUDGE)
        guard_reg = any(all(r > 1.00 for r in per[c]) for c in GUARD)
        elig[a] = judge_ok and not guard_reg
        regress[a] = judge_reg or guard_reg
        res["table"][a] = {"per": per, "med": med, "eligible": elig[a], "regress": regress[a],
                           "geo": gmean([med[c] for c in JUDGE])}
    aa = {c: ratios(cells["aa"], base, c) for c in CELLS}
    res["aa"] = aa
    eligible = [a for a in CANDIDATES if elig[a]]
    if eligible:
        # 幾何平均が最小。差 TIE_EPS 未満なら変更パラメータの少ない腕、なお並べば jpw だけの腕。
        best = min(res["table"][a]["geo"] for a in eligible)
        near = [a for a in eligible if res["table"][a]["geo"] - best < TIE_EPS]
        near.sort(key=lambda a: (PARAM_COUNT[a], 0 if a in JPW_ONLY else 1, res["table"][a]["geo"], a))
        res["selected"] = near[0]
        res["verdict"] = "ADOPT(%s)" % near[0]
    elif all(regress[a] for a in CANDIDATES):
        res["verdict"] = "REJECT"
    else:
        res["verdict"] = "undetermined"
        res["notes"].append("適格な候補はなく、全候補が後退でもない（判定セルまたはガードが 1.00 超だが 5/5 ではない等）")
    # 記録のみ: NB-RANGE5 / NB-SEP（選択腕。#4142 は所有者承認待ちのため判定に使わない）
    if res["selected"]:
        a = res["selected"]
        nb = {}
        for c in JUDGE:
            xs = [cells[a][c][i]["median_s"] for i in range(ROUNDS)]
            bs = [base[c][i]["median_s"] for i in range(ROUNDS)]
            r = res["table"][a]["per"][c]
            nb[c] = {"range": (min(r), max(r)), "sep": all(x < 1.0 for x in r) and max(xs) < min(bs)}
        res["nb"] = nb
    torch = load_torch(d)
    if torch is not None:
        tr = {}
        for a in ("base",) + ((res["selected"],) if res["selected"] else ()):
            for s in (256, 512):
                key = ("gemm", s, "fresh")
                tr[(a, s)] = [torch[s][i] / cells[a][key][i]["median_s"] for i in range(ROUNDS)]
        res["torch"] = tr
    else:
        res["notes"].append("PyTorch 比: 未計測または欠損（記録のみ）")
    return res


def fmt(xs):
    return " / ".join("%.3f" % x for x in xs)


def render(res, name):
    L = ["# #4162 N=256 SME スケジューリング A/B 集計（%s）" % name, "",
         "**判定: %s**（機械出力。手で直さない）" % res["verdict"], ""]
    if res["checksum_fail"]:
        L.append("- checksum 不一致セル: %s" % res["checksum_fail"])
    if res["gate_fail"]:
        L.append("- 構造ゲート fail: %s" % res["gate_fail"])
    for p in res["problems"]:
        L.append("- 問題: " + p)
    for n in res["notes"]:
        L.append("- 注記: " + n)
    if res["table"]:
        L += ["", "## 比（候補 / base。5 round の値と中央値）", "",
              "| 腕 | セル | 5 round | 中央値 | 適格 | 後退 |", "|---|---|---|---|---|---|"]
        for a in CANDIDATES:
            t = res["table"][a]
            for c in CELLS:
                L.append("| %s | %s%d-%s | %s | %.3f | %s | %s |" % (
                    a, c[0], c[1], c[2], fmt(t["per"][c]), t["med"][c],
                    "yes" if t["eligible"] else "no", "yes" if t["regress"] else "no"))
        L += ["", "選択規則: 判定 2 セル中央値比の幾何平均が最小。差 %.3f 未満なら変更パラメータの少ない腕、なお並べば jpw だけの腕。" % TIE_EPS]
        for a in CANDIDATES:
            L.append("- %s: 幾何平均 %.4f" % (a, res["table"][a]["geo"]))
    if res.get("aa"):
        L += ["", "## A/A（記録のみ。判定に使わない）", ""]
        for c in CELLS:
            L.append("- %s%d-%s: %s（範囲 %.3f〜%.3f）" % (c[0], c[1], c[2], fmt(res["aa"][c]), min(res["aa"][c]), max(res["aa"][c])))
    if res.get("nb"):
        L += ["", "## NB-RANGE5 / NB-SEP（記録のみ。#4142 は所有者承認待ちで未承認）", ""]
        for c, v in res["nb"].items():
            L.append("- %s%d-%s: range %.3f〜%.3f, NB-SEP=%s" % (c[0], c[1], c[2], v["range"][0], v["range"][1], v["sep"]))
    if res.get("torch"):
        L += ["", "## PyTorch 比（torch median / fandhe median。記録のみ）", ""]
        for (a, s), xs in sorted(res["torch"].items()):
            L.append("- %s gemm%d fresh: %s（中央値 %.3f）" % (a, s, fmt(xs), statistics.median(xs)))
    L += ["", "多重比較: 候補 5 つを同時に見るため、1 候補の「適格」は偶然を含みうる（限界として記録）。", ""]
    return "\n".join(L)


# ---------------------------------------------------------------- self-test
def _write_arm(ab, arm, per_cell_times, ck=1.5):
    for task in TASKS:
        with open(os.path.join(ab, "results-%s-%s.jsonl" % (arm, task)), "w") as f:
            for r in range(ROUNDS):
                for cell in CELLS:
                    if cell[0] != task:
                        continue
                    t = per_cell_times(cell, r)
                    f.write(json.dumps({"task": task, "size": cell[1], "mode": cell[2], "device": "cpu",
                                        "median_s": t, "checksum": ck}) + "\n")


def _coef(v, r):
    # 係数は数値（全 round 共通）または 5 要素のリスト（round 別）
    return v[r] if isinstance(v, list) else v


ONE_SLOW = [1.0, 1.0, 1.1, 1.0, 1.0]  # 3 round 目だけ遅い（後退は 5/5 ではない）


def _make(d, speed, ck_override=None, gate_override=None, drop=None, dup_gate=False, bad_load=False):
    """speed: arm -> (cell -> 係数)。base の時間は 1.0e-3 固定。"""
    ab = os.path.join(d, "ab")
    os.makedirs(ab, exist_ok=True)
    for a in ARMS:
        sp = speed.get(a, {})
        _write_arm(ab, a, lambda cell, r, sp=sp: 1.0e-3 * _coef(sp.get(cell, sp.get("*", 1.0)), r),
                   ck=(ck_override or {}).get(a, 1.5))
    with open(os.path.join(d, "gates.txt"), "w") as f:
        for g in GATES_ALL:
            f.write("%s all ok\n" % g)
        for g in GATES_ARM:
            for a in ARMS:
                res = (gate_override or {}).get((g, a), "ok")
                f.write("%s %s %s\n" % (g, a, res))
        if dup_gate:
            f.write("P-TREE all ok\n")
    with open(os.path.join(d, "load_gate.log"), "w") as f:
        for r in range(1, ROUNDS + 1):
            ok = 0 if (bad_load and r == 3) else 1
            f.write("round%d gate=pass gate_ok=%d threshold=3.0\n" % (r, ok))
    if drop:
        os.remove(os.path.join(ab, drop))


def self_test():
    fails = []

    def check(name, got, want):
        if got != want:
            fails.append("%s: got %r want %r" % (name, got, want))

    def run(**kw):
        with tempfile.TemporaryDirectory() as d:
            speed = kw.pop("speed", {})
            _make(d, speed, **kw)
            return judge(d)

    # ADOPT: t4 が判定 2 セルで改善・ガードは 1.00
    r = run(speed={"t4": {("gemm", 256, "fresh"): 0.9, ("gemm", 256, "reuse"): 0.92}})
    check("adopt", r["verdict"], "ADOPT(t4)")
    # 選択: 幾何平均が最小のものを選ぶ
    r = run(speed={"t4": {("gemm", 256, "fresh"): 0.9, ("gemm", 256, "reuse"): 0.9},
                   "jpw1": {("gemm", 256, "fresh"): 0.8, ("gemm", 256, "reuse"): 0.8}})
    check("select-min", r["verdict"], "ADOPT(jpw1)")
    # tie-break: 差 0.005 未満ならパラメータの少ない腕（t4-jpw1 より t6）
    r = run(speed={"t4-jpw1": {("gemm", 256, "fresh"): 0.900, ("gemm", 256, "reuse"): 0.900},
                   "t6": {("gemm", 256, "fresh"): 0.903, ("gemm", 256, "reuse"): 0.903}})
    check("tie-params", r["verdict"], "ADOPT(t6)")
    # tie-break: 同数なら jpw だけの腕
    r = run(speed={"t4": {("gemm", 256, "fresh"): 0.900, ("gemm", 256, "reuse"): 0.900},
                   "jpw4": {("gemm", 256, "fresh"): 0.902, ("gemm", 256, "reuse"): 0.902}})
    check("tie-jpw-only", r["verdict"], "ADOPT(jpw4)")
    # 判定セルが 1 round でも 1.00 超なら不適格 -> 他候補も 5/5 後退でなければ undetermined
    sp = {a: {("gemm", 256, "fresh"): ONE_SLOW} for a in CANDIDATES}
    check("one-round-regress", run(speed=sp)["verdict"], "undetermined")
    # REJECT: 全候補が判定セルで 5/5 後退
    sp = {a: {("gemm", 256, "fresh"): 1.1} for a in CANDIDATES}
    check("reject", run(speed=sp)["verdict"], "REJECT")
    # ガード後退（512 が 5/5 悪化）の候補は不適格。他候補が 5/5 後退でなければ undetermined
    sp = {a: {("gemm", 256, "fresh"): ONE_SLOW} for a in CANDIDATES if a != "t4"}
    sp["t4"] = {("gemm", 256, "fresh"): 0.9, ("gemm", 256, "reuse"): 0.9, ("gemm", 512, "fresh"): 1.05}
    check("guard-blocks", run(speed=sp)["verdict"], "undetermined")
    # FAIL: checksum 不一致
    check("fail-checksum", run(ck_override={"t6": 1.6})["verdict"], "FAIL")
    # FAIL: P-BIT 不一致（他の問題より先）
    check("fail-bit", run(gate_override={("P-BIT", "t4"): "fail"}, drop="results-t6-gemm.jsonl")["verdict"], "FAIL")
    # undetermined: 前提ゲート不成立
    check("gate-undet", run(gate_override={("P-FIRE", "t4"): "undetermined"})["verdict"], "undetermined")
    check("fire-fail-undet", run(gate_override={("P-FIRE", "jpw1"): "fail"})["verdict"], "undetermined")
    # undetermined: load ゲート不成立
    check("load-undet", run(bad_load=True)["verdict"], "undetermined")
    # 欠損・重複は例外にせず undetermined（problems に残る）
    r = run(drop="results-t4-train.jsonl")
    check("missing-undet", (r["verdict"], any("t4" in p for p in r["problems"])), ("undetermined", True))
    check("dup-gate", run(dup_gate=True)["verdict"], "undetermined")
    # 行数過剰（重複行）
    with tempfile.TemporaryDirectory() as d:
        _make(d, {})
        p = os.path.join(d, "ab", "results-t6-gemm.jsonl")
        line = open(p).readline()
        open(p, "a").write(line)
        check("dup-row", judge(d)["verdict"], "undetermined")
    # 非有限値
    with tempfile.TemporaryDirectory() as d:
        _make(d, {})
        p = os.path.join(d, "ab", "results-base-gemm.jsonl")
        s = open(p).read().replace('"median_s": 0.001', '"median_s": NaN', 1)
        open(p, "w").write(s)
        check("nan-row", judge(d)["verdict"], "undetermined")
    # render が落ちない
    with tempfile.TemporaryDirectory() as d:
        _make(d, {"t4": {"*": 0.9}})
        render(judge(d), "self-test")
    if fails:
        print("SELF-TEST FAIL")
        for x in fails:
            print(" -", x)
        return 1
    print("SELF-TEST OK")
    return 0


REQUIRED_TEMPLATE_KEYS = ("fixed-at-utc:", "tree_sha:", "tree_fingerprint:", "aggregate.py sha256:",
                          "on-arm.patch sha256:", "orchestrate_macmini.sh sha256:", "P-TREE", "P-R0", "P-FIRE",
                          "P-RT", "P-BIT", "P-CELLS", "P-LOCK", "P-GATE", "P-DIAG", "ADOPT", "REJECT", "undetermined")


def check_template():
    p = os.path.join(HERE, "RULE.template.txt")
    s = open(p, encoding="utf-8").read()
    miss = [k for k in REQUIRED_TEMPLATE_KEYS if k not in s]
    for a in CANDIDATES:
        if "arm-%s.patch sha256:" % a not in s:
            miss.append("arm-%s.patch sha256:" % a)
    if miss:
        print("TEMPLATE NG: 欠落 %s" % miss)
        return 1
    print("TEMPLATE OK")
    return 0


def fingerprint():
    names = ["aggregate.py", "orchestrate_macmini.sh", "on-arm.patch"] + ["arm-%s.patch" % a for a in CANDIDATES]
    for n in names:
        p = os.path.join(HERE, n)
        if os.path.isfile(p):
            print("%s sha256: %s" % (n, hashlib.sha256(open(p, "rb").read()).hexdigest()))
    return 0


def main(argv):
    if len(argv) == 2 and argv[1] == "--self-test":
        return self_test()
    if len(argv) == 2 and argv[1] == "--check-template":
        return check_template()
    if len(argv) == 2 and argv[1] == "--fingerprint":
        return fingerprint()
    if len(argv) != 2 or argv[1].startswith("-"):
        print(__doc__)
        return 2
    d = argv[1]
    if not os.path.isdir(d):
        print("収録ディレクトリが無い: %s" % d)
        return 2
    res = judge(d)
    print(render(res, os.path.basename(os.path.normpath(d))))
    return 0 if res["verdict"].startswith(("ADOPT", "REJECT")) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
