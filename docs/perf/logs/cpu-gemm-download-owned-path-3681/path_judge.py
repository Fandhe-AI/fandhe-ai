#!/usr/bin/env python3
"""イシュー #3681 の A/B 判定器（RULE.txt を機械的に適用する。標準ライブラリのみ・fail-closed）。

役割: #3672 の判定器 ab_judge.py のローダー・セル統計・rho 計算を流用し、#3681 の RULE.txt が定める
「非処置セルを matmul_resident の経路に絞った」判定を出す。#3672 の判定との差は 1 点だけ:
REJECT の「5/5 run 後退」と ADOPT の「非処置セル中央値 ratio <= 1.00」を経路上のセル（処置 2 + 非処置 28）にだけ
適用し、経路外セルは記録のみにする。checksum は全腕・全 N のまま（正しさの検査で経路と無関係）。
閾値は #3672 と同値（ab_judge.py の定数と等しいことを起動時に検査する。緩めない）。
呼び出し元: session-scripts/session-3681.sh（機体ごとの最終段の参考判定）と、回収後の手動実行（judge-3681.md）。

使い方:
  path_judge.py --host gb10=<dir> [--host macmini=<dir>] [--out <md>]
  path_judge.py --self-test
  <dir> は orchestrate.sh の --out ディレクトリ（構成は ab_judge.py の docstring と同じ。env_info.txt を追加で読む）。

fail-closed（終了コード 2）: ab_judge.py の sha256 が固定値と不一致（#3672 は計測後に 1 度修正された経緯があるため
ドリフトを検知する）／env_info.txt の rule_sha256 が本ディレクトリの RULE.txt と不一致／RULE.txt の
treat-cell・path-cell 行が本ファイルの定数と集合として不一致／ab_judge.py 側の入力検査の失敗
（run 欠落・行の重複・n_samples 不一致・JSON 解析不能・ゲート記録の欠落）。
終了コード: 0=判定を出した（ADOPT／REJECT／判定不能のいずれも 0）、2=入力不備、1=self-test 失敗。
注意: #3672 の回収データに対して本判定器を実行しない（新しい規則で旧データを判定した結果を作らないため）。
"""
import argparse
import hashlib
import importlib.util
import os
import re
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
AB_JUDGE_PATH = os.path.join(HERE, "..", "cpu-gemm-download-owned-3672", "ab_judge.py")
RULE_PATH = os.path.join(HERE, "RULE.txt")
# #3681 の RULE 固定時点で存在した ab_judge.py の sha256（流用元のドリフト検知）
AB_JUDGE_SHA256 = "748deb3bf7bf3cb55244aaf4961031db9b18fdbea9917ff7b867133491252af4"


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_ab():
    """sha256 で固定した同リポ内の ab_judge.py だけを読み込む（任意パスは読まない）。"""
    if not os.path.isfile(AB_JUDGE_PATH):
        raise SystemExit("入力不備（fail-closed）: ab_judge.py が無い")
    if sha256_file(AB_JUDGE_PATH) != AB_JUDGE_SHA256:
        print("入力不備（fail-closed）: ab_judge.py の sha256 が固定値と不一致", file=sys.stderr)
        raise SystemExit(2)
    spec = importlib.util.spec_from_file_location("ab_judge_3672", AB_JUDGE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ab = load_ab()
JudgeError = ab.JudgeError
SIZES, RUNS, DIAG_ARMS, HOSTS = ab.SIZES, ab.RUNS, ab.DIAG_ARMS, ab.HOSTS
TREAT_N = ab.TREAT_N

# RULE.txt の閾値（#3672 と同値。流用元と等しいことを検査する）
ADOPT_TREAT_MEDIAN_MAX = 0.95
ADOPT_TREAT_ALL_RUNS_BELOW = 1.00
ADOPT_NONTREAT_MEDIAN_MAX = 1.00
ADOPT_RHO_REM_MIN = 0.5
assert ADOPT_TREAT_MEDIAN_MAX == ab.ADOPT_TREAT_MEDIAN_MAX
assert ADOPT_TREAT_ALL_RUNS_BELOW == ab.ADOPT_TREAT_ALL_RUNS_BELOW
assert ADOPT_NONTREAT_MEDIAN_MAX == ab.ADOPT_NONTREAT_MEDIAN_MAX
assert ADOPT_RHO_REM_MIN == ab.ADOPT_RHO_REM_MIN

# bench セルのラベル（RULE.txt の機械可読行では空白を避けて reuse_iter_total）
BENCH_LABEL_RULE = "reuse_iter_total"
BENCH_LABEL_AB = "reuse iter_total"
# 経路上の診断腕（matmul_resident を窓内で呼ぶ。RULE.txt「経路の定義」）
PATH_DIAG_ARMS = ["resident_public", "resident_reupload"]
OUT_DIAG_ARMS = [a for a in DIAG_ARMS if a not in PATH_DIAG_ARMS]

TREAT_CELLS = {("gb10", "bench", BENCH_LABEL_RULE, TREAT_N), ("gb10", "diag", "resident_public", TREAT_N)}
PATH_CELLS = set()
for _h in HOSTS:
    for _n in SIZES:
        PATH_CELLS.add((_h, "bench", BENCH_LABEL_RULE, _n))
        for _a in PATH_DIAG_ARMS:
            PATH_CELLS.add((_h, "diag", _a, _n))
PATH_CELLS -= TREAT_CELLS
assert len(TREAT_CELLS) == 2 and len(PATH_CELLS) == 28


def rule_label(grp, label):
    return BENCH_LABEL_RULE if grp == "bench" else label


def cell_class(host, grp, label, n):
    key = (host, grp, rule_label(grp, label), n)
    if key in TREAT_CELLS:
        return "treat"
    if key in PATH_CELLS:
        return "path"
    return "out"


def parse_rule_cells(path):
    """RULE.txt の treat-cell・path-cell 行を (処置集合, 経路上非処置集合) として読む。"""
    treat, pth = set(), set()
    pat = re.compile(r"^\s+(treat-cell|path-cell): (gb10|macmini) (bench|diag) ([a-z_]+) (\d+)\s*$")
    with open(path, encoding="utf-8") as f:
        for ln in f:
            if "-cell:" not in ln or not ln.startswith("  "):
                continue
            m = pat.match(ln)
            if not m:
                raise JudgeError("RULE.txt のセル行が解析不能: %r" % ln)
            key = (m.group(2), m.group(3), m.group(4), int(m.group(5)))
            tgt = treat if m.group(1) == "treat-cell" else pth
            if key in tgt:
                raise JudgeError("RULE.txt のセル行が重複: %r" % ln)
            tgt.add(key)
    return treat, pth


def check_rule_consistency(rule_path):
    treat, pth = parse_rule_cells(rule_path)
    if treat != TREAT_CELLS or pth != PATH_CELLS:
        raise JudgeError("RULE.txt のセル列挙が判定器の定数と一致しない（処置 %d・経路上 %d）" % (len(treat), len(pth)))


def check_env_rule_sha(root, rule_path):
    """env_info.txt の rule_sha256 が本ディレクトリの RULE.txt と一致すること。"""
    p = os.path.join(root, "env_info.txt")
    if not os.path.isfile(p):
        raise JudgeError("env_info.txt が無い: %s" % root)
    got = None
    with open(p, encoding="utf-8") as f:
        for ln in f:
            if ln.startswith("rule_sha256: "):
                got = ln.split(": ", 1)[1].strip()
    if got is None:
        raise JudgeError("env_info.txt に rule_sha256 が無い: %s" % root)
    if got != sha256_file(rule_path):
        raise JudgeError("env_info.txt の rule_sha256 が RULE.txt と不一致: %s" % root)


def load_host(root, host, rule_path):
    if root is not None and os.path.isdir(root):
        check_env_rule_sha(root, rule_path)
    return ab.load_host(root, host)


def judge(hosts):
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
        for m in ab.check_checksums(h["data"]):
            reject.append("%s checksum 不一致: %s" % (n, m))
        cells_by_host[n] = ab.host_cells(h)
        for (grp, label, sz), s in cells_by_host[n].items():
            cls = cell_class(n, grp, label, sz)
            if cls == "out":
                continue
            where = "%s %s %s N=%d" % (n, grp, label, sz)
            if s["all_above_1"]:
                reject.append("%s 経路上セル %s: 5/5 run で後退（ratio > 1.00）" % (n, where))
            if cls == "treat":
                if s["ratio"] > 1.00:
                    reject.append("%s 処置セル: 中央値 ratio %.4f > 1.00" % (where, s["ratio"]))
                if not (s["ratio"] <= ADOPT_TREAT_MEDIAN_MAX):
                    adopt_blockers.append("処置セル %s: 中央値 ratio %.4f > 0.95" % (where, s["ratio"]))
                if not s["all_below_1"]:
                    adopt_blockers.append("処置セル %s: 5/5 run で ratio < 1.00 を満たさない" % where)
            elif s["ratio"] > ADOPT_NONTREAT_MEDIAN_MAX:
                adopt_blockers.append("経路上の非処置セル %s: 中央値 ratio %.4f > 1.00" % (where, s["ratio"]))
        if n == "gb10":
            rho = ab.gb10_rho(h)
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
    out_summary = {}
    for n, cells in cells_by_host.items():
        outs = [s for (g, l, z), s in cells.items() if cell_class(n, g, l, z) == "out"]
        out_summary[n] = {
            "cells": len(outs),
            "above": sum(1 for s in outs if s["all_above_1"]),
            "below": sum(1 for s in outs if s["all_below_1"]),
            "expected": len(outs) / 32.0,
        }
    return {"verdict": verdict, "reject": reject, "adopt_blockers": adopt_blockers, "notes": notes,
            "cells": cells_by_host, "rho": rho, "missing": missing, "out_summary": out_summary}


def fmt_s(v):
    return "%.6g" % v


def render(res, rule_path):
    L = ["# #3681 A/B 判定（path_judge.py の出力。手で書き換えない）", ""]
    L.append("**判定: %s**" % res["verdict"])
    L.append("")
    L.append("- RULE.txt sha256: `%s`" % sha256_file(rule_path))
    L.append("- 流用元 ab_judge.py sha256: `%s`" % AB_JUDGE_SHA256)
    for n in res["notes"]:
        L.append("- %s" % n)
    if res["missing"]:
        L.append("- 判定に使えない機体（未実測）: %s" % ", ".join(res["missing"]))
    L.append("")
    if res["reject"]:
        L += ["## REJECT 条件に当たった項目"] + ["- %s" % x for x in res["reject"]] + [""]
    if res["adopt_blockers"]:
        L += ["## ADOPT を妨げた項目（記録）"] + ["- %s" % x for x in res["adopt_blockers"]] + [""]
    if res["rho"]:
        r = res["rho"]
        L += ["## GB10 N=4096 の rho（記録。rho_REM のみ ADOPT 条件）", "", "| 指標 | 値 |", "|---|---|"]
        for k in ("X_before", "X_after", "DL_before", "bench_before", "bench_after"):
            L.append("| %s (s) | %s |" % (k, fmt_s(r[k])))
        for k in ("rho_REM", "rho_BENCH", "rho_OFFSET"):
            L.append("| %s | %.4f |" % (k, r[k]))
        L.append("")
    names = {"treat": "処置", "path": "経路上の非処置", "out": "経路外（記録のみ）"}
    for n, cells in res["cells"].items():
        for cls in ("treat", "path", "out"):
            rows = sorted(((k, s) for k, s in cells.items() if cell_class(n, *k) == cls),
                          key=lambda kv: (kv[0][0], kv[0][1], kv[0][2]))
            if not rows:
                continue
            L += ["## %s: %s セル（after/before。中央値 ratio と run ごとの ratio）" % (n, names[cls]), "",
                  "| 群 | セル | N | before 中央値(s) | after 中央値(s) | 中央値 ratio | run ごとの ratio |",
                  "|---|---|---|---|---|---|---|"]
            for (grp, label, sz), s in rows:
                L.append("| %s | %s | %d | %s | %s | %.4f | %s |" % (
                    grp, label, sz, fmt_s(s["before_med"]), fmt_s(s["after_med"]), s["ratio"],
                    " ".join("%.4f" % x for x in s["ratios"])))
            L.append("")
    if res["out_summary"]:
        L += ["## 経路外セルの件数（記録のみ。判定に使わない）", "",
              "| 機体 | セル数 | 5/5 run で ratio > 1.00 | 5/5 run で ratio < 1.00 | 差が無い場合の期待件数（セル数 / 32） |",
              "|---|---|---|---|---|"]
        for n, s in res["out_summary"].items():
            L.append("| %s | %d | %d | %d | %.2f |" % (n, s["cells"], s["above"], s["below"], s["expected"]))
        L.append("")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- self-test
def _write_env(root, sha):
    with open(os.path.join(root, "env_info.txt"), "w", encoding="utf-8") as f:
        f.write("date_utc: 2026-01-01T00:00:00Z\nrule_sha256: %s\n" % sha)


def self_test():
    fails = []

    def expect(name, cond):
        if not cond:
            fails.append(name)

    def expect_err(name, fn):
        try:
            fn()
            fails.append(name)
        except (JudgeError, SystemExit):
            pass

    rsha = sha256_file(RULE_PATH)
    check_rule_consistency(RULE_PATH)

    def f_base(arm, grp, label, n, r):
        if n == TREAT_N and label in ("resident_public", "reuse iter_total"):
            return 0.8
        return 1.0

    def with_over(over):
        def f(arm, grp, label, n, r):
            v = over(arm, grp, label, n, r)
            return f_base(arm, grp, label, n, r) if v is None else v
        return f

    def mk(td, name, host, f, **kw):
        d = os.path.join(td, name)
        ab.make_host(d, host, f, **kw)
        _write_env(d, rsha)
        return d

    def run(dirs):
        hosts = {n: load_host(dirs.get(n), n, RULE_PATH) for n in HOSTS}
        return judge(hosts)

    with tempfile.TemporaryDirectory() as td:
        g = mk(td, "g", "gb10", f_base)
        m = mk(td, "m", "macmini", f_base)
        expect("ADOPT", run({"gb10": g, "macmini": m})["verdict"] == "ADOPT")

        # 経路外セルが 5/5 後退しても REJECT にならず ADOPT（#3672 規則との唯一の差の正方向）
        def out_back(arm, grp, label, n, r):
            if grp == "diag" and ((label == "resident_split" and n == 512) or (label == "gemm_device_download" and n == 4096)):
                return 1.10
            return None
        g1 = mk(td, "g1", "gb10", with_over(out_back))
        m1 = mk(td, "m1", "macmini", with_over(out_back))
        r = run({"gb10": g1, "macmini": m1})
        expect("経路外 5/5 後退でも ADOPT", r["verdict"] == "ADOPT")
        expect("経路外の件数が記録される", r["out_summary"]["gb10"]["above"] >= 1)

        # 経路上の非処置セルが 5/5 後退 → REJECT（逆方向）
        def mac_pub256(arm, grp, label, n, r):
            return 1.05 if (grp == "diag" and label == "resident_public" and n == 256) else None
        m2 = mk(td, "m2", "macmini", with_over(mac_pub256))
        expect("経路上 macmini resident_public N=256 の 5/5 後退は REJECT",
               run({"gb10": g, "macmini": m2})["verdict"] == "REJECT")

        def gb_reup1024(arm, grp, label, n, r):
            return 1.05 if (grp == "diag" and label == "resident_reupload" and n == 1024) else None
        g2 = mk(td, "g2", "gb10", with_over(gb_reup1024))
        expect("経路上 gb10 resident_reupload N=1024 の 5/5 後退は REJECT",
               run({"gb10": g2, "macmini": m})["verdict"] == "REJECT")

        # 経路上の非処置セルが中央値 > 1.00 だが 5/5 ではない → 判定不能
        def noisy(arm, grp, label, n, r):
            if grp == "diag" and label == "resident_reupload" and n == 256:
                return 0.99 if r in (1, 2) else 1.02
            return None
        g3 = mk(td, "g3", "gb10", with_over(noisy))
        expect("経路上の中央値 > 1.00（5/5 でない）は判定不能",
               run({"gb10": g3, "macmini": m})["verdict"] == "判定不能")

        # 処置セルの中央値 > 1.00（5/5 ではない）→ REJECT
        def treat_bad(arm, grp, label, n, r):
            if n == TREAT_N and label == "reuse iter_total":
                return 1.0 + 0.01 * r if r != 5 else 0.9
            return None
        g4 = mk(td, "g4", "gb10", with_over(treat_bad))
        expect("処置セル > 1.00 は REJECT", run({"gb10": g4, "macmini": m})["verdict"] == "REJECT")

        # 処置セルが 0.95 を超える → ADOPT しない
        def treat_weak(arm, grp, label, n, r):
            return 0.97 if (n == TREAT_N and label == "reuse iter_total") else None
        g5 = mk(td, "g5", "gb10", with_over(treat_weak))
        expect("処置セル 0.97 は ADOPT しない", run({"gb10": g5, "macmini": m})["verdict"] == "判定不能")

        # 経路外の腕の checksum 不一致 → REJECT（checksum は絞らない）
        g6 = mk(td, "g6", "gb10", f_base, cksum_after=("fresh", 256))
        r = run({"gb10": g6, "macmini": m})
        expect("経路外の腕の checksum 不一致は REJECT", r["verdict"] == "REJECT" and any("checksum" in x for x in r["reject"]))

        # rho_REM < 0.5 → ADOPT しない（X の差を DL(0.01) の半分未満にする）
        def rho_low(arm, grp, label, n, r):
            if grp == "diag" and n == TREAT_N and label == "resident_public":
                return 0.99
            return None
        g7 = mk(td, "g7", "gb10", with_over(rho_low))
        r = run({"gb10": g7, "macmini": m})
        expect("rho_REM < 0.5 は ADOPT しない", r["verdict"] != "ADOPT" and any("rho_REM" in x for x in r["adopt_blockers"]))

        # 片機体欠落・record_only
        expect("片機体欠落", run({"gb10": g, "macmini": None})["verdict"] == "判定不能（未実測あり）")
        g8 = mk(td, "g8", "gb10", f_base, gate="unpassed")
        expect("ゲート unpassed は record_only", run({"gb10": g8, "macmini": m})["verdict"] == "判定不能（未実測あり）")
        g9 = mk(td, "g9", "gb10", f_base, excl="fail")
        expect("excl=fail は record_only", run({"gb10": g9, "macmini": m})["verdict"] == "判定不能（未実測あり）")
        # record_only の機体に経路上後退があっても REJECT に使わない
        g10 = mk(td, "g10", "gb10", with_over(gb_reup1024), gate="unpassed")
        expect("record_only は REJECT に使わない", run({"gb10": g10, "macmini": m})["verdict"] == "判定不能（未実測あり）")

        # fail-closed: RULE sha 不一致・run 欠落
        g11 = mk(td, "g11", "gb10", f_base)
        _write_env(g11, "0" * 64)
        expect_err("RULE sha 不一致は例外", lambda: run({"gb10": g11, "macmini": m}))
        g12 = mk(td, "g12", "gb10", f_base, drop=3)
        expect_err("診断行の欠落は例外", lambda: run({"gb10": g12, "macmini": m}))
        g13 = mk(td, "g13", "gb10", f_base)
        os.remove(os.path.join(g13, "env_info.txt"))
        expect_err("env_info 欠落は例外", lambda: run({"gb10": g13, "macmini": m}))

        # RULE.txt のセル行と定数の不一致は例外
        bad = os.path.join(td, "RULE_bad.txt")
        with open(RULE_PATH, encoding="utf-8") as f:
            txt = f.read()
        with open(bad, "w", encoding="utf-8") as f:
            f.write(txt.replace("  path-cell: macmini diag resident_public 256\n", "", 1))
        expect_err("path-cell 欠落は例外", lambda: check_rule_consistency(bad))
        with open(bad, "w", encoding="utf-8") as f:
            f.write(txt + "  path-cell: gb10 diag fresh 256\n")
        expect_err("path-cell 追加は例外", lambda: check_rule_consistency(bad))

        # 描画
        render(run({"gb10": g1, "macmini": m1}), RULE_PATH)
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
        check_rule_consistency(RULE_PATH)
        hosts = {n: load_host(dirs.get(n), n, RULE_PATH) for n in HOSTS}
        res = judge(hosts)
    except JudgeError as e:
        print("入力不備（fail-closed）: %s" % e, file=sys.stderr)
        return 2
    text = render(res, RULE_PATH)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
