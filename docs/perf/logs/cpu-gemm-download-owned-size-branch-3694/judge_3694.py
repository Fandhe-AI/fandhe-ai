#!/usr/bin/env python3
"""イシュー #3694 の A/B 判定器（RULE.txt「3 系列の合成」を機械的に適用する。標準ライブラリのみ・fail-closed）。

役割: #3681 の path_judge.py（1 対の系列に対する経路上セル判定。編集しない・importlib で読む）を、
GB10 と Mac mini の系列 k=1..3 の対ごとに呼び、RULE.txt の合成規則で 1 つの結論にする。
  ADOPT: 3 対すべて ADOPT ／ REJECT: 1 対でも REJECT ／ それ以外: 判定不能。
  いずれかの機体でゲートを通った系列が 3 未満で REJECT に当たらない場合は「判定不能（系列不足）」。
  未実測のまま ADOPT は出さない。
「ゲートを通った系列」は ab_judge の record_only_reasons が空の系列（値は見ない）。各機体で起動順に最初の 3 系列を使う。
T2（macmini・診断 resident_public・N=256）は経路上セルとして path_judge が判定対象にする。本判定器は対ごとの
中央値 ratio と 5 run の各値を名指しで出力へ載せる。小 N（64・128）の記録セルは判定に使わず別表にする
（欠落していても判定は止めず、その旨を表に書く）。
呼び出し元: session-scripts/session-3694.sh（--single。1 系列分の参考表示）と、回収後の手動実行（judge-3694.md）。

使い方:
  python3 -I judge_3694.py --host gb10=<s1>,<s2>,<s3>[,<s4>,<s5>] --host macmini=<s1>,... [--out <md>]
  python3 -I judge_3694.py --single <gb10|macmini>=<dir> [--out <md>]
  python3 -I judge_3694.py --self-test
  各 <dir> は orchestrate.sh の --out ディレクトリ。--host の系列は起動順（s1, s2, ...）で並べる。

fail-closed（終了コード 2）: path_judge.py・#3681 の RULE.txt・ab_judge.py の sha256 が固定値と不一致／
#3681 の RULE.txt のセル行と判定器の定数が不一致／env_info.txt の rule_sha256 が本ディレクトリの RULE.txt と不一致／
env_info.txt の rule_3681_sha256 が欠落・不一致／arms の rev-stamp が不正（before は a7aac2f0…+harness-overlay(<12 桁>)・
両腕 dirty=0）／系列の入力検査の失敗（run 欠落・行の重複・n_samples 不一致・JSON 解析不能・ゲート記録の欠落）。
終了コード: 0=判定を出した（ADOPT／REJECT／判定不能のいずれも 0）、2=入力不備、1=self-test 失敗。
値は RULE の固定定数だけを使い、外部入力から組み立てない（importlib で読むのは sha256 照合済みのリポ内ファイルだけ）。
"""
import argparse
import hashlib
import importlib.util
import json
import os
import re
import statistics
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.dirname(HERE)
PATH_JUDGE_PATH = os.path.join(LOGS, "cpu-gemm-download-owned-path-3681", "path_judge.py")
RULE_3681_PATH = os.path.join(LOGS, "cpu-gemm-download-owned-path-3681", "RULE.txt")
RULE_PATH = os.path.join(HERE, "RULE.txt")
# RULE.txt（#3694）が名指しする流用元の sha256（ドリフト検知）
PATH_JUDGE_SHA256 = "1afa378d9dfd84ec278b0ffb2c684ea518f604c4fd7f8dfeedea9337ba4ad535"
RULE_3681_SHA256 = "48b3da16e68c53cf111a14e0fb431d3cb149f5816705e43c04339002dcf7614e"
AB_JUDGE_SHA256 = "748deb3bf7bf3cb55244aaf4961031db9b18fdbea9917ff7b867133491252af4"
BASE_SHA = "a7aac2f0227add243845418e0897ac0675c5b622"
BEFORE_STAMP_RE = re.compile(r"^before_rev_stamp: %s\+harness-overlay\([0-9a-f]{12}\) dirty=0$" % BASE_SHA)
AFTER_STAMP_RE = re.compile(r"^after_rev_stamp: [0-9a-f]{40}(\+[A-Za-z0-9._()-]+)? dirty=0$")
NEEDED_SERIES = 3
MAX_SERIES = 5
SMALL_SIZES = [64, 128]
HOSTS = ["gb10", "macmini"]
T2 = ("diag", "resident_public", 256)


class InputError(Exception):
    """入力不備（fail-closed。終了コード 2）。"""


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def verify_pinned(path, want, what):
    """sha256 で固定した同リポ内ファイルだけを許す（不在・不一致は入力不備）。"""
    if not os.path.isfile(path):
        raise InputError("%s が無い" % what)
    if sha256_file(path) != want:
        raise InputError("%s の sha256 が固定値と不一致" % what)


def load_path_judge():
    verify_pinned(PATH_JUDGE_PATH, PATH_JUDGE_SHA256, "path_judge.py")
    verify_pinned(RULE_3681_PATH, RULE_3681_SHA256, "#3681 の RULE.txt")
    verify_pinned(os.path.join(LOGS, "cpu-gemm-download-owned-3672", "ab_judge.py"), AB_JUDGE_SHA256, "ab_judge.py")
    spec = importlib.util.spec_from_file_location("path_judge_3681", PATH_JUDGE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.check_rule_consistency(RULE_3681_PATH)
    return mod


def read_env(root):
    p = os.path.join(root, "env_info.txt")
    if not os.path.isfile(p):
        raise InputError("env_info.txt が無い: %s" % root)
    with open(p, encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]


def check_env_extra(root):
    """#3694 固有の同定検査（rule_3681_sha256 と arms の rev-stamp）。"""
    lines = read_env(root)
    got = [ln.split(": ", 1)[1].strip() for ln in lines if ln.startswith("rule_3681_sha256: ")]
    if len(got) != 1:
        raise InputError("env_info.txt の rule_3681_sha256 が欠落または重複: %s" % root)
    if got[0] != RULE_3681_SHA256:
        raise InputError("env_info.txt の rule_3681_sha256 が #3681 の RULE.txt と不一致: %s" % root)
    before = [ln for ln in lines if ln.startswith("before_rev_stamp: ")]
    after = [ln for ln in lines if ln.startswith("after_rev_stamp: ")]
    if len(before) != 1 or not BEFORE_STAMP_RE.match(before[0]):
        raise InputError("before の rev-stamp が不正（%s+harness-overlay(<12 桁>) dirty=0 が必要）: %s" % (BASE_SHA, root))
    if len(after) != 1 or not AFTER_STAMP_RE.match(after[0]):
        raise InputError("after の rev-stamp が不正（dirty=0 が必要）: %s" % root)
    first = [ln.split(": ", 1)[1].strip() for ln in lines if ln.startswith("first_arm: ")]
    date = [ln.split(": ", 1)[1].strip() for ln in lines if ln.startswith("date_utc: ")]
    return {"first_arm": first[0] if len(first) == 1 else "?", "date_utc": date[0] if len(date) == 1 else "?"}


# ------------------------------------------------------------ 小 N（記録のみ）
def _read_rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                rows.append(json.loads(ln))
    return rows


def small_cells(root):
    """{(group, label, n): (before_med, after_med, ratio)} と欠落の説明一覧。判定に使わない・失敗しても止めない。"""
    cells, missing = {}, []
    base = os.path.join(root, "small")
    if not os.path.isdir(base):
        return cells, ["small/ ディレクトリが無い"]
    per = {}
    for arm in ("before", "after"):
        for r in range(1, 6):
            try:
                for d in _read_rows(os.path.join(base, arm, "run%d.jsonl" % r)):
                    if d.get("phase") == "iter_total" and d.get("n") in SMALL_SIZES:
                        per.setdefault((("diag", d["arm"], d["n"]), arm), {})[r] = float(d["median_s"])
            except (OSError, ValueError, KeyError, TypeError) as e:
                missing.append("diag %s run%d: %s" % (arm, r, type(e).__name__))
            for n in SMALL_SIZES:
                try:
                    for d in _read_rows(os.path.join(base, arm, "bench", "gemm_phases_n%d_r%d.jsonl" % (n, r))):
                        if d.get("phase") == "iter_total":
                            per.setdefault((("bench", "reuse iter_total", n), arm), {})[r] = float(d["median_s"])
                except (OSError, ValueError, KeyError, TypeError) as e:
                    missing.append("bench %s n=%d run%d: %s" % (arm, n, r, type(e).__name__))
    keys = sorted({k for (k, _a) in per})
    for k in keys:
        b, a = per.get((k, "before"), {}), per.get((k, "after"), {})
        if len(b) != 5 or len(a) != 5:
            missing.append("セル %s %s N=%d: run 欠落（before %d・after %d）" % (k[0], k[1], k[2], len(b), len(a)))
            continue
        bm, am = statistics.median(b.values()), statistics.median(a.values())
        if bm > 0:
            cells[k] = (bm, am, am / bm)
    return cells, missing


# ------------------------------------------------------------ 判定
def load_series(pj, root, host):
    if not os.path.isdir(root):
        raise InputError("系列ディレクトリが無い: %s" % root)
    h = pj.load_host(root, host, RULE_PATH)
    if h is None:
        raise InputError("系列を読めない: %s" % root)
    extra = check_env_extra(root)
    h["dir"] = root
    h["first_arm"] = extra["first_arm"]
    h["date_utc"] = extra["date_utc"]
    return h


def compose(pj, series):
    """series={host: [h, ...]（起動順）}。RULE.txt の合成規則を適用して結果 dict を返す。"""
    passed, skipped = {}, {}
    for host in HOSTS:
        lst = series.get(host, [])
        passed[host] = [h for h in lst if not h["record_only_reasons"]][:NEEDED_SERIES]
        skipped[host] = [h for h in lst if h["record_only_reasons"]]
    npairs = min(len(passed["gb10"]), len(passed["macmini"]))
    pairs = []
    for k in range(npairs):
        res = pj.judge({"gb10": passed["gb10"][k], "macmini": passed["macmini"][k]})
        pairs.append({"k": k + 1, "gb10": passed["gb10"][k], "macmini": passed["macmini"][k], "res": res})
    short = [h for h in HOSTS if len(passed[h]) < NEEDED_SERIES]
    if any(p["res"]["verdict"] == "REJECT" for p in pairs):
        verdict = "REJECT"
    elif short:
        verdict = "判定不能（系列不足）"
    elif all(p["res"]["verdict"] == "ADOPT" for p in pairs):
        verdict = "ADOPT"
    else:
        verdict = "判定不能"
    return {"verdict": verdict, "pairs": pairs, "passed": passed, "skipped": skipped, "short": short}


def single_view(pj, series):
    """--single 用: 1 機体 1 系列を path_judge に単独で通した参考表示（最終判定ではない。もう一方の機体は未実測扱い）。"""
    host = next(h for h in HOSTS if series.get(h))
    h = series[host][0]
    stub = {"dir": "(未測定)", "first_arm": "-", "date_utc": "-", "record_only_reasons": []}
    res = pj.judge({host: h, **{x: None for x in HOSTS if x != host}})
    pair = {"k": 1, "gb10": stub, "macmini": stub, "res": res}
    pair[host] = h
    passed = {x: [] for x in HOSTS}
    skipped = {x: [] for x in HOSTS}
    (skipped if h["record_only_reasons"] else passed)[host].append(h)
    return {"verdict": "参考表示（1 系列。最終判定ではない）: " + res["verdict"], "pairs": [pair],
            "passed": passed, "skipped": skipped, "short": []}


def fnum(v):
    return "%.6g" % v


def render(out):
    L = ["# #3694 サイズ分岐 A/B 判定（judge_3694.py の出力。手で書き換えない）", ""]
    L.append("**判定: %s**" % out["verdict"])
    L.append("")
    L.append("- RULE.txt（#3694）sha256: `%s`" % sha256_file(RULE_PATH))
    L.append("- 流用元 path_judge.py sha256: `%s`" % PATH_JUDGE_SHA256)
    L.append("- 流用元 #3681 RULE.txt sha256: `%s`" % RULE_3681_SHA256)
    L.append("- 流用元 ab_judge.py sha256: `%s`" % AB_JUDGE_SHA256)
    L.append("- 合成: 3 対すべて ADOPT で ADOPT／1 対でも REJECT で REJECT／それ以外は判定不能／系列不足（REJECT なし）は判定不能（系列不足）")
    for h in HOSTS:
        used = out["passed"][h]
        L.append("- %s: ゲートを通った系列 %d 本（使用: %s）。先頭腕: %s。開始: %s" % (
            h, len(used), ", ".join(os.path.basename(x["dir"].rstrip("/")) for x in used) or "なし",
            ",".join(x["first_arm"] for x in used) or "-", ",".join(x["date_utc"] for x in used) or "-"))
        for x in out["skipped"][h]:
            L.append("  - 使わない系列 %s: %s" % (os.path.basename(x["dir"].rstrip("/")), " / ".join(x["record_only_reasons"])))
    if out["short"]:
        L.append("- 系列不足の機体: %s（必要 %d 本）" % (", ".join(out["short"]), NEEDED_SERIES))
    L.append("")
    L += ["## 対ごとの判定", "", "| 対 k | GB10 系列 | Mac mini 系列 | 判定 |", "|---|---|---|---|"]
    for p in out["pairs"]:
        L.append("| %d | %s | %s | %s |" % (p["k"], os.path.basename(p["gb10"]["dir"].rstrip("/")),
                                           os.path.basename(p["macmini"]["dir"].rstrip("/")), p["res"]["verdict"]))
    L.append("")
    L += ["## T2（macmini・診断 resident_public・N=256。経路上セル）", "",
          "| 対 k | before 中央値(s) | after 中央値(s) | 中央値 ratio | 5 run の ratio（run1..run5） |", "|---|---|---|---|---|"]
    for p in out["pairs"]:
        s = p["res"]["cells"].get("macmini", {}).get((T2[0], T2[1], T2[2]))
        if s is None:
            L.append("| %d | - | - | - | セル無し |" % p["k"])
        else:
            L.append("| %d | %s | %s | %.4f | %s |" % (p["k"], fnum(s["before_med"]), fnum(s["after_med"]), s["ratio"],
                                                      " ".join("%.4f" % x for x in s["ratios"])))
    L.append("")
    for p in out["pairs"]:
        res = p["res"]
        L.append("## 対 %d の詳細（path_judge の判定: %s）" % (p["k"], res["verdict"]))
        L.append("")
        for n in res["notes"]:
            L.append("- %s" % n)
        if res["reject"]:
            L += ["", "### REJECT 条件に当たった項目"] + ["- %s" % x for x in res["reject"]]
        if res["adopt_blockers"]:
            L += ["", "### ADOPT を妨げた項目（記録）"] + ["- %s" % x for x in res["adopt_blockers"]]
        if res["rho"]:
            r = res["rho"]
            L += ["", "### GB10 N=4096 の rho（記録。rho_REM のみ ADOPT 条件）", "", "| 指標 | 値 |", "|---|---|"]
            for key in ("X_before", "X_after", "DL_before", "bench_before", "bench_after"):
                L.append("| %s (s) | %s |" % (key, fnum(r[key])))
            for key in ("rho_REM", "rho_BENCH", "rho_OFFSET"):
                L.append("| %s | %.4f |" % (key, r[key]))
        names = {"treat": "処置", "path": "経路上の非処置", "out": "経路外（記録のみ）"}
        for host, cells in res["cells"].items():
            for cls in ("treat", "path", "out"):
                rows = sorted(((k, s) for k, s in cells.items() if pj_cell_class(host, k) == cls),
                              key=lambda kv: (kv[0][0], kv[0][1], kv[0][2]))
                if not rows:
                    continue
                L += ["", "### 対 %d・%s: %s セル（after/before。中央値 ratio と run ごとの ratio）" % (p["k"], host, names[cls]), "",
                      "| 群 | セル | N | before 中央値(s) | after 中央値(s) | 中央値 ratio | run ごとの ratio |",
                      "|---|---|---|---|---|---|---|"]
                for (grp, label, sz), s in rows:
                    L.append("| %s | %s | %d | %s | %s | %.4f | %s |" % (
                        grp, label, sz, fnum(s["before_med"]), fnum(s["after_med"]), s["ratio"],
                        " ".join("%.4f" % x for x in s["ratios"])))
        L.append("")
    L += ["## 小 N（N=64・128）の記録セル（判定に使わない。after/before の中央値の比）", ""]
    for h in HOSTS:
        for x in out["passed"][h]:
            cells, missing = small_cells(x["dir"])
            L.append("### %s %s" % (h, os.path.basename(x["dir"].rstrip("/"))))
            L.append("")
            if cells:
                L += ["| 群 | セル | N | before 中央値(s) | after 中央値(s) | 中央値の比 |", "|---|---|---|---|---|---|"]
                for (grp, label, n), (bm, am, ra) in sorted(cells.items()):
                    L.append("| %s | %s | %d | %s | %s | %.4f |" % (grp, label, n, fnum(bm), fnum(am), ra))
                L.append("")
            if missing:
                L.append("欠落: " + "；".join(missing[:12]) + ("（ほか %d 件）" % (len(missing) - 12) if len(missing) > 12 else ""))
                L.append("")
            if not cells and not missing:
                L += ["小 N のセルが無い", ""]
    return "\n".join(L) + "\n"


PJ = None


def pj_cell_class(host, key):
    grp, label, sz = key
    return PJ.cell_class(host, grp, label, sz)


# ------------------------------------------------------------ self-test
def self_test():
    fails = []

    def expect(name, cond):
        if not cond:
            fails.append(name)

    def expect_err(name, fn):
        try:
            fn()
            fails.append(name)
        except (InputError, PJ.JudgeError):
            pass

    ab = PJ.ab
    rsha = sha256_file(RULE_PATH)
    before_ok = "before_rev_stamp: %s+harness-overlay(0123456789ab) dirty=0" % BASE_SHA
    after_ok = "after_rev_stamp: " + "c" * 40 + " dirty=0"

    def write_env(root, sha=None, r3681=None, before=None, after=None, first="before", with_3681=True):
        lines = ["date_utc: 2026-01-01T00:00:00Z", "rule_sha256: %s" % (sha or rsha)]
        if with_3681:
            lines.append("rule_3681_sha256: %s" % (r3681 or RULE_3681_SHA256))
        lines += ["first_arm: %s" % first, before or before_ok, after or after_ok]
        with open(os.path.join(root, "env_info.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def f_base(arm, grp, label, n, r):
        if n == ab.TREAT_N and label in ("resident_public", "reuse iter_total"):
            return 0.8
        return 1.0

    def with_over(over):
        def f(arm, grp, label, n, r):
            v = over(arm, grp, label, n, r)
            return f_base(arm, grp, label, n, r) if v is None else v
        return f

    def f_reject(arm, grp, label, n, r):
        return 1.05 if (grp == "diag" and label == "resident_public" and n == 256) else None

    def f_weak(arm, grp, label, n, r):
        return 0.97 if (n == ab.TREAT_N and label == "reuse iter_total") else None

    counter = [0]

    def mk(td, host, f=f_base, small=True, **kw):
        env = {k: kw.pop(k) for k in ("sha", "r3681", "before", "after", "first", "with_3681") if k in kw}
        counter[0] += 1
        d = os.path.join(td, "%s-s%d" % (host, counter[0]))
        ab.make_host(d, host, f, **kw)
        write_env(d, **env)
        if small:
            for arm in ("before", "after"):
                os.makedirs(os.path.join(d, "small", arm, "bench"))
                for r in range(1, 6):
                    with open(os.path.join(d, "small", arm, "run%d.jsonl" % r), "w", encoding="utf-8") as fh:
                        for n in SMALL_SIZES:
                            fh.write(json.dumps({"arm": "resident_public", "phase": "iter_total", "n": n,
                                                 "median_s": 0.001 * (1.1 if arm == "after" else 1.0)}) + "\n")
                    for n in SMALL_SIZES:
                        with open(os.path.join(d, "small", arm, "bench", "gemm_phases_n%d_r%d.jsonl" % (n, r)), "w", encoding="utf-8") as fh:
                            fh.write(json.dumps({"phase": "iter_total", "median_s": 0.002}) + "\n")
        return d

    def run(td, spec):
        """spec={host: [(f, kw), ...]} から系列を作って合成する。"""
        series = {h: [load_series(PJ, mk(td, h, f, **kw), h) for (f, kw) in spec[h]] for h in HOSTS}
        return compose(PJ, series)

    ok = (f_base, {})
    rej = (with_over(f_reject), {})
    weak = (with_over(f_weak), {})
    unpassed = (f_base, {"gate": "unpassed"})

    with tempfile.TemporaryDirectory() as td:
        r = run(td, {"gb10": [ok] * 3, "macmini": [ok] * 3})
        expect("3 対 ADOPT は ADOPT", r["verdict"] == "ADOPT")
        text = render(r)
        expect("T2 の名指し表がある", "## T2（macmini・診断 resident_public・N=256" in text and "| 1.0000 | 1.0000 1.0000 1.0000 1.0000 1.0000 |" in text)
        expect("小 N 表がある", "小 N（N=64・128）の記録セル" in text and "1.1000" in text)

        r = run(td, {"gb10": [ok, ok, ok], "macmini": [ok, rej, ok]})
        expect("1 対 REJECT は REJECT", r["verdict"] == "REJECT")
        expect("REJECT は 2 対目", [p["res"]["verdict"] for p in r["pairs"]] == ["ADOPT", "REJECT", "ADOPT"])

        r = run(td, {"gb10": [ok, ok, weak], "macmini": [ok, ok, ok]})
        expect("混在（ADOPT・ADOPT・判定不能）は判定不能", r["verdict"] == "判定不能")

        r = run(td, {"gb10": [ok, ok], "macmini": [ok, ok, ok]})
        expect("系列不足は判定不能（系列不足）", r["verdict"] == "判定不能（系列不足）")

        r = run(td, {"gb10": [ok, rej], "macmini": [ok, ok, ok]})
        expect("系列不足でも REJECT が優先", r["verdict"] == "REJECT")

        r = run(td, {"gb10": [ok, ok, ok], "macmini": []})
        expect("片機体ゼロ系列は判定不能（系列不足）", r["verdict"] == "判定不能（系列不足）" and r["pairs"] == [])

        r = run(td, {"gb10": [ok, unpassed, ok, ok], "macmini": [ok, ok, ok]})
        names = [os.path.basename(h["dir"]) for h in r["passed"]["gb10"]]
        expect("ゲート不通過を飛ばして 4 本目を使う", r["verdict"] == "ADOPT" and len(names) == 3 and len(r["skipped"]["gb10"]) == 1)
        expect("使った系列に不通過が含まれない", all(not h["record_only_reasons"] for h in r["passed"]["gb10"]) and len(set(names)) == 3)

        r = run(td, {"gb10": [unpassed, unpassed, unpassed], "macmini": [ok, ok, ok]})
        expect("ゲート不通過だけなら ADOPT しない", r["verdict"] == "判定不能（系列不足）")

        # 小 N の欠落があっても判定は変わらない
        r1 = run(td, {"gb10": [(f_base, {"small": False})] * 3, "macmini": [ok] * 3})
        expect("小 N 欠落でも判定は同じ", r1["verdict"] == "ADOPT")
        t1 = render(r1)
        expect("小 N 欠落が表に書かれる", "small/ ディレクトリが無い" in t1)

        # fail-closed（それぞれ入力不備）
        expect_err("rule_sha256 不一致", lambda: load_series(PJ, mk(td, "gb10", sha="0" * 64), "gb10"))
        expect_err("rule_3681_sha256 の欠落", lambda: load_series(PJ, mk(td, "gb10", with_3681=False), "gb10"))
        expect_err("rule_3681_sha256 の不一致", lambda: load_series(PJ, mk(td, "gb10", r3681="1" * 64), "gb10"))
        expect_err("before の rev-stamp が a7aac2f0 でない", lambda: load_series(
            PJ, mk(td, "gb10", before="before_rev_stamp: " + "d" * 40 + "+harness-overlay(0123456789ab) dirty=0"), "gb10"))
        expect_err("before の dirty=1", lambda: load_series(
            PJ, mk(td, "gb10", before=before_ok.replace("dirty=0", "dirty=1")), "gb10"))
        expect_err("before に overlay 印が無い", lambda: load_series(
            PJ, mk(td, "gb10", before="before_rev_stamp: %s dirty=0" % BASE_SHA), "gb10"))
        expect_err("after の dirty=1", lambda: load_series(PJ, mk(td, "gb10", after=after_ok.replace("dirty=0", "dirty=1")), "gb10"))
        expect_err("系列ディレクトリ無し", lambda: load_series(PJ, os.path.join(td, "nonexistent"), "gb10"))

        # 固定 sha の照合（偽ファイルで不一致が入力不備になる）
        fake = os.path.join(td, "fake.py")
        with open(fake, "w", encoding="utf-8") as fh:
            fh.write("x = 1\n")
        expect_err("path_judge.py の sha 不一致", lambda: verify_pinned(fake, PATH_JUDGE_SHA256, "path_judge.py"))
        expect_err("#3681 RULE の sha 不一致", lambda: verify_pinned(fake, RULE_3681_SHA256, "#3681 の RULE.txt"))
        expect_err("ab_judge.py の sha 不一致", lambda: verify_pinned(fake, AB_JUDGE_SHA256, "ab_judge.py"))
        expect_err("固定ファイルの不在", lambda: verify_pinned(os.path.join(td, "none.py"), AB_JUDGE_SHA256, "ab_judge.py"))

        # 描画が例外を出さない
        render(run(td, {"gb10": [ok] * 3, "macmini": [rej, ok, ok]}))
    if fails:
        print("self-test FAIL: %s" % ", ".join(fails), file=sys.stderr)
        return 1
    print("self-test OK")
    return 0


def parse_hosts(specs):
    dirs = {}
    for spec in specs:
        if "=" not in spec:
            raise InputError("--host は name=dir[,dir...] 形式")
        name, ds = spec.split("=", 1)
        if name not in HOSTS or name in dirs:
            raise InputError("host 名が不正または重複: %s" % name)
        lst = [d for d in ds.split(",") if d]
        if not lst or len(lst) > MAX_SERIES:
            raise InputError("系列は 1〜%d 本（起動順）: %s" % (MAX_SERIES, name))
        dirs[name] = lst
    return dirs


def main(argv):
    global PJ
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", action="append", default=[], help="gb10=<s1>,<s2>,<s3>[,...] / macmini=...")
    ap.add_argument("--single", help="<gb10|macmini>=<dir>（1 系列分の参考表示）")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    try:
        PJ = load_path_judge()
        if a.self_test:
            return self_test()
        if a.single:
            dirs = parse_hosts([a.single])
            if len(dirs) != 1 or len(next(iter(dirs.values()))) != 1:
                raise InputError("--single は 1 機体 1 系列")
        else:
            dirs = parse_hosts(a.host)
            if not dirs:
                raise InputError("--host または --single が必要")
        series = {h: [load_series(PJ, d, h) for d in dirs.get(h, [])] for h in HOSTS}
        if a.single:
            out = single_view(PJ, series)
        else:
            out = compose(PJ, series)
    except (InputError, PJ.JudgeError if PJ else InputError) as e:
        print("入力不備（fail-closed）: %s" % e, file=sys.stderr)
        return 2
    text = render(out)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
