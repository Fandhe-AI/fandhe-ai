#!/usr/bin/env python3
"""#4166（steel 候補 T0U-LU-FB の結線）の Mac mini 前後比較の判定。セル A/B だけの系列を判定する。

役割: #4160 の aggregate.py（sha256 で固定）を読み込み、セル A/B の部分（RULE.txt 項 9〜11 のうちセル A/B に
かかわる部分）の関数 load_ab・scoreboard をそのまま呼ぶ。判定式・閾値・NB-RANGE5／NB-SEP をここで書き直さない。
aggregate.py の main は純カーネル計測（kern）の出力を要求するので、セル A/B だけの本系列には使えない。

呼び出し: python3 -I -B judge_ab.py <系列ディレクトリ> [--aggregate <aggregate.py>]
         python3 -I -B judge_ab.py --checksum-only <系列ディレクトリ> [--aggregate <aggregate.py>]
  --checksum-only は全ゲート通過でない系列に使う（RULE.txt 項 6）。値を出さず `checksum=fail|not-fail|unknown` の 1 行だけを出す。
  aggregate.py の既定の場所: リポジトリでは本ディレクトリの隣の metal-steel-large-n-ab-macmini-4160/、Mac mini の展開（deploy.sh）では
  W/rule/ から見た W/tree/docs/perf/logs/metal-steel-large-n-ab-macmini-4160/。先に見つかった方を使う（どちらでも sha256 を照合する）。
  系列ディレクトリ: ab/（runner の results/raw から本系列のラベルの記録だけを写したもの）と
  status.txt（drive_series.sh が書く `ab rc=<N>` の行を含む状態ファイル）を持つ。
  結果の Markdown を標準出力へ出す。終了コード: 0 = 判定を出した（PASS／FAIL／UNDETERMINED のいずれでも）・
  2 = 入力が壊れている（判定を出さない）。
python3 -I -B judge_ab.py --self-test: #4160 の formal-v2-2 の記録で再現を確かめ、変異で各分岐を確かめる。

判定（RULE.txt 項 2 が #4160 の RULE.txt 項 9・11 を参照する形。ここでは写像だけを行う）:
  - FAIL: load_ab の status が fail（checksum 不一致）または anomaly（5/5 round で 2% を超えて遅いセルがある）。
  - checksum 不一致の FAIL はゲートの通過に依らない（checksum は正しさの検査。#4160 の RULE.txt 項 11「checksum 不一致は FAIL として停止」）。
  - UNDETERMINED: 全ゲート通過でない（status.txt の `ab rc=0` が無い・ゲートの行に gate=pass 以外がある・
    load_ab の status が undetermined／unavailable）。
  - PASS: 上のいずれでもない（load_ab の status が ok）。
"""
import glob
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import shutil
import sys
import tempfile

# #4160 の RULE.txt の aggregate_script_sha256 と同じ値（9eba21749・main のどちらでも同じ内容）。
AGGREGATE_SHA256 = "e8ae78af7aa956753e333c45c9719d00484c06aa5e0b778f78a468dc3bc06307"
HERE = os.path.dirname(os.path.abspath(__file__))
REF_DIR = "metal-steel-large-n-ab-macmini-4160"


def ref_dir(here):
    """#4160 の記録ディレクトリ。リポジトリの配置（隣のディレクトリ）と deploy.sh の配置（W/rule → W/tree）の順に探す。"""
    cands = [os.path.join(here, "..", REF_DIR),
             os.path.join(here, "..", "tree", "docs", "perf", "logs", REF_DIR)]
    for c in cands:
        if os.path.isfile(os.path.join(c, "aggregate.py")):
            return os.path.normpath(c)
    return os.path.normpath(cands[0])


DEFAULT_AGGREGATE = os.path.join(ref_dir(HERE), "aggregate.py")


class JudgeError(Exception):
    pass


def load_aggregate(path):
    with open(path, "rb") as f:
        data = f.read()
    got = hashlib.sha256(data).hexdigest()
    if got != AGGREGATE_SHA256:
        raise JudgeError("aggregate.py の sha256 が固定値と違う: %s" % got)
    sys.dont_write_bytecode = True  # 展開ツリーや記録ディレクトリに __pycache__ を作らない
    spec = importlib.util.spec_from_file_location("aggregate4160", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read(path):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError as e:
        raise JudgeError("読めない: %s (%s)" % (path, e))


def series_gates(series):
    """全ゲート通過の判定（RULE.txt 項 6。drive_series.sh の判定と同じ定義）。戻り値: (ok, 理由のリスト)。"""
    reasons = []
    st = read(os.path.join(series, "status.txt"))
    if len(re.findall(r"^ab rc=0 ", st, re.MULTILINE)) != 1:
        reasons.append("status.txt に `ab rc=0` の行がちょうど 1 つ無い")
    logs = glob.glob(os.path.join(series, "ab", "load-gate-4160-metal-*.log"))
    if len(logs) != 1:
        raise JudgeError("ゲートのログがちょうど 1 つ無い: %d" % len(logs))
    gate_lines = [ln for ln in read(logs[0]).splitlines() if "gate=" in ln]
    if not gate_lines:
        reasons.append("ゲートの行が無い")
    bad = [ln for ln in gate_lines if not re.search(r"(^|\s)gate=pass(\s|$)", ln)]
    if bad:
        reasons.append("gate=pass 以外の行が %d 件" % len(bad))
    return (not reasons), reasons


def judge(series, agg):
    abdir = os.path.join(series, "ab")
    try:
        ab = agg.load_ab(abdir)
    except agg.AggregateError as e:
        # 記録が揃わない系列（途中で止まったもの）。記録済みの行に確かな checksum 不一致があれば、速度表を要求せずに FAIL とする
        # （RULE.txt 項 2・6。checksum 不一致の FAIL はゲートの通過に依らない）。不一致がなければ判定を出さない。
        try:
            mm, label = partial_checksum_scan(abdir)
        except JudgeError:
            mm, label = None, None
        if not mm:
            raise JudgeError("load_ab が入力を拒否した: %s" % e)
        try:
            gates_ok, reasons = series_gates(series)
        except JudgeError as ge:
            gates_ok, reasons = False, [str(ge)]
        ab = {"status": "fail-partial", "cells": {}, "label": label,
              "notes": ["記録が揃っていない（%s）。記録済みの before／after の行に checksum 不一致がある" % e]}
        return "FAIL", ab, gates_ok, reasons, {n: None for n in agg.SCOPE}
    try:
        gates_ok, reasons = series_gates(series)
    except JudgeError as ge:
        # checksum 不一致の FAIL はゲートの通過に依らない（RULE.txt 項 2・6）。ゲート関連の入力が欠けても FAIL は出し、理由に残す。
        if ab["status"] != "fail":
            raise
        gates_ok, reasons = False, [str(ge)]
    if ab["status"] == "fail":
        verdict = "FAIL"
    elif not gates_ok or ab["status"] in ("undetermined", "unavailable"):
        verdict = "UNDETERMINED"
    elif ab["status"] == "anomaly":
        verdict = "FAIL"
    elif ab["status"] == "ok":
        verdict = "PASS"
    else:
        raise JudgeError("想定外の status: %r" % ab["status"])
    boards = {n: agg.scoreboard(ab, n) for n in agg.SCOPE}
    if not gates_ok:
        # scoreboard は load_ab の正式系列の条件しか見ない。全ゲート通過でない系列の NB-SEP は出さない（RULE.txt 項 2）。
        boards = {n: (None if b is None else dict(b, sep=None)) for n, b in boards.items()}
    return verdict, ab, gates_ok, reasons, boards


def _checksum_ok(vals):
    """load_ab と同じ checksum の条件: 全記録の checksum が有限の数値で、すべて等しい。"""
    return all(isinstance(c, (int, float)) and not isinstance(c, bool) and math.isfinite(c)
               and c == vals[0] for c in vals)


def partial_checksum_mismatch(abdir):
    return partial_checksum_scan(abdir)[0]


def partial_checksum_scan(abdir):
    """途中で止まった系列の部分記録から、すでに確認できる checksum 不一致を探す（RULE.txt 項 6）。

    load_ab は全セル 5 行と candle 5 行が揃わないと例外を出すので、その場合に限って使う。条件は load_ab の
    checksum の条件（セルごとに before／after の全記録の checksum が有限の数値で完全一致）を、記録済みの行だけに当てる。
    戻り値: (結果, ラベル)。結果は True = 不一致あり・False = 記録済みの行には不一致なし・None = ラベルを 1 つに決められない／
    before・after の行が無い。
    """
    labels = set()
    for arm in ("before", "after"):
        for p in glob.glob(os.path.join(abdir, "results-%s-*-metal-gemm.jsonl" % arm)):
            m = re.match(r"results-%s-(.+)-metal-gemm\.jsonl$" % arm, os.path.basename(p))
            if m:
                labels.add(m.group(1))
    if len(labels) != 1:
        return None, None
    label = labels.pop()
    cells, seen = {}, 0
    for arm in ("before", "after"):
        path = os.path.join(abdir, "results-%s-%s-metal-gemm.jsonl" % (arm, label))
        if not os.path.isfile(path):
            continue
        for line in read(path).splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue  # 書きかけの行は数えない（不一致の根拠にも一致の根拠にもしない）
            if not isinstance(r, dict) or r.get("task") != "gemm" or r.get("device") != "metal":
                continue
            cells.setdefault((r.get("size"), r.get("mode") or "fresh"), []).append(r.get("checksum"))
            seen += 1
    if not seen:
        return None, label
    return any(not _checksum_ok(v) for v in cells.values()), label


def checksum_only(series, agg):
    """全ゲート通過でない系列の checksum 不一致だけを確かめる（RULE.txt 項 6）。値は返さない。

    記録が揃っていれば load_ab の status で決める。揃っていなければ（load_ab が例外）、部分記録に不一致があれば fail、
    なければ unknown（揃っていないので not-fail とは言えない）。
    """
    abdir = os.path.join(series, "ab")
    try:
        ab = agg.load_ab(abdir)
    except (agg.AggregateError, JudgeError):
        try:
            return "fail" if partial_checksum_mismatch(abdir) else "unknown"
        except JudgeError:
            return "unknown"
    if ab["status"] == "fail":
        return "fail"
    return "not-fail" if ab["cells"] else "unknown"


def render(series, verdict, ab, gates_ok, reasons, boards, agg):
    out = []
    w = out.append
    w("# #4166 結線の前後比較（Mac mini・セル A/B）の判定")
    w("")
    w("- 系列: `%s`（runner のラベル `%s`）" % (os.path.basename(os.path.normpath(series)), ab.get("label", "?")))
    w("- 腕: before = 定数 false（未コミットの 1 行差分）、after = 定数 true（clean な 9eba21749）。比は after/before で、1 未満が after の速さ。")
    w("- 全ゲート通過: **%s**%s" % ("はい" if gates_ok else "いいえ", "" if gates_ok else "（" + "・".join(reasons) + "）"))
    w("- load_ab の状態: **%s**" % ab["status"])
    for n in ab.get("notes", []):
        w("  - %s" % n)
    w("- **判定: %s**" % verdict)
    w("")
    if ab["cells"]:
        w("| N | mode | before ms | after ms | 比 after/before（round 別） | 中央値比 | 2% 超の round | checksum |")
        w("|---|---|---|---|---|---|---|---|")
        for n in agg.SIZES:
            for mode in agg.MODES:
                c = ab["cells"][(n, mode)]
                w("| %d | %s | %.4f | %.4f | %s | %.3f | %d/5 | %s |" % (
                    n, mode, c["before_ms"], c["after_ms"], " ".join("%.3f" % r for r in c["ratios"]),
                    c["ratio_med"], c["over5"], "完全一致" if c["checksum_exact"] else "不一致"))
        w("")
    w("## スコアボード比（#4160 の RULE.txt 項 10。相手は candle だけ。G4096 は記録のみ）")
    w("")
    w("| N | 対象 | before の比（範囲） | after の比（範囲） | NB-SEP |")
    w("|---|---|---|---|---|")
    for n in agg.SCOPE:
        b = boards[n]
        role = "判定対象" if n in agg.G_SIZES else "記録のみ"
        if b is None:
            w("| %d | %s | unavailable | unavailable | 判定不能 |" % (n, role))
            continue
        sep = {True: "成立", False: "不成立", None: "判定不能"}[b["sep"]]
        w("| %d | %s | %.3f〜%.3f | %.3f〜%.3f | %s |" % (
            n, role, b["range_before"][0], b["range_before"][1], b["range_after"][0], b["range_after"][1], sep))
    w("")
    return "\n".join(out) + "\n"


def main(argv):
    if len(argv) >= 2 and argv[1] == "--self-test":
        return self_test()
    args = argv[1:]
    ck_only = bool(args) and args[0] == "--checksum-only"
    if ck_only:
        args = args[1:]
    agg_path = DEFAULT_AGGREGATE
    if len(args) == 3 and args[1] == "--aggregate":
        agg_path = args[2]
        args = args[:1]
    if len(args) != 1:
        print("usage: judge_ab.py [--checksum-only] <系列ディレクトリ> [--aggregate <aggregate.py>] | --self-test", file=sys.stderr)
        return 2
    try:
        agg = load_aggregate(agg_path)
        if ck_only:
            print("checksum=%s" % checksum_only(args[0], agg))
            return 0
        res = judge(args[0], agg)
    except JudgeError as e:
        print("error: %s" % e, file=sys.stderr)
        return 2
    sys.stdout.write(render(args[0], *res, agg))
    return 0


# ---------------------------------------------------------------- self-test
SAMPLE = os.path.join(ref_dir(HERE), "series-formal-v2-2", "ab")


def _copy_series(tmp, name, status="ab rc=0 2026-11-11T00:00:00Z\n"):
    d = os.path.join(tmp, name)
    shutil.copytree(SAMPLE, os.path.join(d, "ab"))
    with open(os.path.join(d, "status.txt"), "w", encoding="utf-8") as f:
        f.write("start x\n")
        f.write(status)
    return d


def _edit(path, fn):
    with open(path, encoding="utf-8") as f:
        s = f.read()
    with open(path, "w", encoding="utf-8") as f:
        f.write(fn(s))


def _scale_after(d, n, mode, k):
    p = glob.glob(os.path.join(d, "ab", "results-after-*-metal-gemm.jsonl"))[0]
    lines = []
    for ln in read(p).splitlines():
        if ln.strip():
            r = json.loads(ln)
            if r.get("size") == n and (r.get("mode") or "fresh") == mode:
                r["median_s"] = r["median_s"] * k
            ln = json.dumps(r)
        lines.append(ln)
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def self_test():
    fails = []

    def check(name, cond):
        print("%s %s" % ("ok  " if cond else "FAIL", name))
        if not cond:
            fails.append(name)

    if not os.path.isdir(SAMPLE):
        print("error: 見本の記録が無い: %s" % SAMPLE, file=sys.stderr)
        return 2
    agg = load_aggregate(DEFAULT_AGGREGATE)
    tmp = tempfile.mkdtemp(prefix="judge4166-")
    try:
        # 1. #4160 formal-v2-2 の再現（aggregate.md: 状態 ok・NB-SEP は 2048 だけ成立）。
        d = _copy_series(tmp, "base")
        v, ab, gok, _, b = judge(d, agg)
        check("見本は PASS", v == "PASS")
        check("見本の状態は ok", ab["status"] == "ok")
        check("見本は全ゲート通過", gok)
        check("NB-SEP 1024 不成立", b[1024]["sep"] is False)
        check("NB-SEP 2048 成立", b[2048]["sep"] is True)
        check("NB-SEP 4096 不成立", b[4096]["sep"] is False)
        check("N=2048 reuse の中央値比 0.933",
              abs(ab["cells"][(2048, "reuse")]["ratio_med"] - 0.933) < 5e-4)
        md = render(d, v, ab, gok, [], b, agg)
        check("表に 10 セル", md.count("| fresh |") + md.count("| reuse |") == 10)
        # 2. 5/5 round で 2% を超えて遅いセル → FAIL（anomaly）。
        d = _copy_series(tmp, "slow")
        _scale_after(d, 512, "fresh", 1.10)
        v, ab, _, _, _ = judge(d, agg)
        check("5/5 round 後退は FAIL", v == "FAIL" and ab["status"] == "anomaly")
        # 3. 4/5 round だけ遅い → PASS のまま（閾値は 5/5）。
        d = _copy_series(tmp, "slow4")
        p = glob.glob(os.path.join(d, "ab", "results-after-*-metal-gemm.jsonl"))[0]
        rows, seen = [], 0
        for ln in read(p).splitlines():
            if ln.strip():
                r = json.loads(ln)
                if r.get("size") == 256 and (r.get("mode") or "fresh") == "reuse":
                    seen += 1
                    if seen <= 4:
                        r["median_s"] *= 1.10
                ln = json.dumps(r)
            rows.append(ln)
        with open(p, "w", encoding="utf-8") as f:
            f.write("\n".join(rows) + "\n")
        v, ab, _, _, _ = judge(d, agg)
        check("4/5 round 後退は PASS", v == "PASS" and ab["cells"][(256, "reuse")]["over5"] == 4)
        # 4. checksum 不一致 → FAIL。
        d = _copy_series(tmp, "ck")
        p = glob.glob(os.path.join(d, "ab", "results-after-*-metal-gemm.jsonl"))[0]
        rows = []
        done = False
        for ln in read(p).splitlines():
            if ln.strip() and not done:
                r = json.loads(ln)
                r["checksum"] = (r.get("checksum") or 0) + 1.0
                ln = json.dumps(r)
                done = True
            rows.append(ln)
        with open(p, "w", encoding="utf-8") as f:
            f.write("\n".join(rows) + "\n")
        v, ab, _, _, _ = judge(d, agg)
        check("checksum 不一致は FAIL", v == "FAIL" and ab["status"] == "fail")
        os.remove(os.path.join(d, "status.txt"))
        try:
            v, _, gok, rs, _ = judge(d, agg)
        except JudgeError:
            v, gok, rs = None, None, []
        check("status.txt が無くても checksum 不一致は FAIL（理由に残す）", v == "FAIL" and gok is False and any("status.txt" in r for r in rs))
        dn = _copy_series(tmp, "nost")
        os.remove(os.path.join(dn, "status.txt"))
        try:
            judge(dn, agg)
            check("checksum が合う系列で status.txt が無ければ判定を出さない", False)
        except JudgeError:
            check("checksum が合う系列で status.txt が無ければ判定を出さない", True)
        # 5. ゲート: 各 round 内のセルの行（gate=invalid ... cell=）→ UNDETERMINED。
        d = _copy_series(tmp, "cellgate")
        g = glob.glob(os.path.join(d, "ab", "load-gate-4160-metal-*.log"))[0]
        _edit(g, lambda s: s + "round3 gate=invalid reason=foreign-after phase=after cell=candle-N4096-fresh at=x\n")
        v, _, gok, _, _ = judge(d, agg)
        check("セルの foreign-after は UNDETERMINED", v == "UNDETERMINED" and not gok)
        # 6. round 先頭のゲートが timeout → UNDETERMINED。
        d = _copy_series(tmp, "tmo")
        g = glob.glob(os.path.join(d, "ab", "load-gate-4160-metal-*.log"))[0]
        _edit(g, lambda s: s.replace("round2 gate=pass", "round2 gate=timeout", 1))
        v, _, _, _, _ = judge(d, agg)
        check("round 先頭の timeout は UNDETERMINED", v == "UNDETERMINED")
        # 6b. pass に似た別の状態（緩和閾値の系列の gate=pass-reference 等）は pass と数えない。
        d = _copy_series(tmp, "passref")
        g = glob.glob(os.path.join(d, "ab", "load-gate-4160-metal-*.log"))[0]
        _edit(g, lambda s: s + "round3 gate=pass-reference reason=x cell=after-N256-fresh at=x\n")
        v, _, gok, _, _ = judge(d, agg)
        check("gate=pass-reference は UNDETERMINED", v == "UNDETERMINED" and not gok)
        # 7. ab rc が 0 でない → UNDETERMINED。
        d = _copy_series(tmp, "rc", status="ab rc=1 x\n")
        v, _, gok, _, _ = judge(d, agg)
        check("ab rc=1 は UNDETERMINED", v == "UNDETERMINED" and not gok)
        check("全ゲート通過でない系列は NB-SEP 判定不能（見本では 2048 が成立）", judge(d, agg)[4][2048]["sep"] is None)
        check("全ゲート通過でない系列も範囲は残す", judge(d, agg)[4][2048]["range_after"] is not None)
        # 8. skipped ログに記録 → UNDETERMINED（load_ab が正式系列から外す）。
        d = _copy_series(tmp, "skip")
        s = glob.glob(os.path.join(d, "ab", "skipped-4160-metal-*.log"))[0]
        _edit(s, lambda x: x + "something failed\n")
        v, ab, _, _, _ = judge(d, agg)
        check("skipped に記録は UNDETERMINED", v == "UNDETERMINED" and ab["status"] == "undetermined")
        # 9. 判定不能な系列では NB-SEP を出さない。
        check("UNDETERMINED の系列は NB-SEP 判定不能", judge(d, agg)[4][2048]["sep"] is None)
        # 10. 遅いセルがあってもゲート不通過なら UNDETERMINED（後退と書かない）。
        d = _copy_series(tmp, "slowgate", status="ab rc=1 x\n")
        _scale_after(d, 512, "fresh", 1.10)
        v, _, _, _, _ = judge(d, agg)
        check("ゲート不通過の後退は UNDETERMINED", v == "UNDETERMINED")
        # 10b. checksum 不一致はゲートの通過に依らず FAIL（ゲート不通過でも）。--checksum-only は値を出さない。
        d = _copy_series(tmp, "ckgate", status="ab rc=1 x\n")
        p = glob.glob(os.path.join(d, "ab", "results-after-*-metal-gemm.jsonl"))[0]
        rows = read(p).splitlines()
        r = json.loads(rows[0]); r["checksum"] = (r.get("checksum") or 0) + 1.0; rows[0] = json.dumps(r)
        with open(p, "w", encoding="utf-8") as f:
            f.write("\n".join(rows) + "\n")
        g = glob.glob(os.path.join(d, "ab", "load-gate-4160-metal-*.log"))[0]
        _edit(g, lambda s: s.replace("round2 gate=pass", "round2 gate=timeout", 1))
        v, _, gok, _, _ = judge(d, agg)
        check("ゲート不通過でも checksum 不一致は FAIL", v == "FAIL" and not gok)
        check("--checksum-only: 不一致は fail", checksum_only(d, agg) == "fail")
        check("--checksum-only: 見本は not-fail", checksum_only(os.path.join(tmp, "base"), agg) == "not-fail")
        check("--checksum-only: ゲート不通過だけなら not-fail", checksum_only(os.path.join(tmp, "rc"), agg) == "not-fail")
        check("--checksum-only: 記録が無ければ unknown", checksum_only(os.path.join(tmp, "nope"), agg) == "unknown")
        os.makedirs(os.path.join(tmp, "short", "ab"))
        shutil.copy(glob.glob(os.path.join(SAMPLE, "load-gate-4160-metal-*.log"))[0], os.path.join(tmp, "short", "ab"))
        src = glob.glob(os.path.join(SAMPLE, "results-before-*-metal-gemm.jsonl"))[0]
        with open(os.path.join(tmp, "short", "ab", os.path.basename(src)), "w", encoding="utf-8") as f:
            f.write(read(src).splitlines()[0] + "\n")
        check("--checksum-only: 途中で止まった系列は unknown", checksum_only(os.path.join(tmp, "short"), agg) == "unknown")
        # 途中で止まった系列（candle が欠け load_ab が例外）でも、記録済みの checksum 不一致は fail。
        d2 = _copy_series(tmp, "partck")
        for c in glob.glob(os.path.join(d2, "ab", "results-candle-*-metal-gemm.jsonl")):
            os.remove(c)
        check("--checksum-only: candle が欠けて不一致なしなら unknown", checksum_only(d2, agg) == "unknown")
        p = glob.glob(os.path.join(d2, "ab", "results-before-*-metal-gemm.jsonl"))[0]
        rows = read(p).splitlines()
        r = json.loads(rows[-1]); r["checksum"] = (r.get("checksum") or 0) + 1.0; rows[-1] = json.dumps(r)
        with open(p, "w", encoding="utf-8") as f:
            f.write("\n".join(rows) + "\n")
        check("--checksum-only: 途中で止まった系列の記録済みの不一致は fail", checksum_only(d2, agg) == "fail")
        try:
            v, ab, _, _, b = judge(d2, agg)
        except JudgeError:
            v, ab, b = None, {"status": None, "cells": {}, "notes": []}, {n: None for n in agg.SCOPE}
        check("通常の判定: 途中で止まった系列の記録済みの不一致は FAIL", v == "FAIL" and ab["status"] == "fail-partial"
              and all(x is None for x in b.values()))
        md = render(d2, v, ab, True, [], b, agg)
        check("通常の判定: 速度表を出さずに FAIL を書く", "判定: FAIL" in md and "| fresh |" not in md)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = main(["judge_ab.py", d2])
        check("通常の判定の CLI: 途中で止まった系列の不一致は rc 0 で FAIL", rc == 0 and "判定: FAIL" in buf.getvalue())
        os.remove(glob.glob(os.path.join(d2, "ab", "load-gate-4160-metal-*.log"))[0])
        check("通常の判定: ゲートのログが無くても checksum 不一致は FAIL", judge(d2, agg)[0] == "FAIL")
        d4 = _copy_series(tmp, "partok")
        for c in glob.glob(os.path.join(d4, "ab", "results-candle-*-metal-gemm.jsonl")):
            os.remove(c)
        try:
            judge(d4, agg)
            check("通常の判定: 途中で止まり不一致が無ければ判定を出さない", False)
        except JudgeError:
            check("通常の判定: 途中で止まり不一致が無ければ判定を出さない", True)
        d3 = _copy_series(tmp, "partck2")
        p = glob.glob(os.path.join(d3, "ab", "results-after-*-metal-gemm.jsonl"))[0]
        rows = [ln for ln in read(p).splitlines() if ln.strip()]
        r = json.loads(rows[0]); r["checksum"] = None; rows[0] = json.dumps(r)
        with open(p, "w", encoding="utf-8") as f:
            f.write("\n".join(rows[:3]) + "\n{\"task\": \"gem")  # 途中で止まり、最後の行は書きかけ
        check("--checksum-only: 部分記録の checksum 欠損は fail（load_ab と同じ条件）", checksum_only(d3, agg) == "fail")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = main(["judge_ab.py", "--checksum-only", d])
        check("--checksum-only の CLI は 1 行だけ（値なし）", rc == 0 and buf.getvalue() == "checksum=fail\n")
        # 10c. aggregate.py の既定の場所: deploy.sh の配置（W/rule → W/tree/...）でも見つける。
        nw = os.path.join(tmp, "W")
        os.makedirs(os.path.join(nw, "rule"))
        os.makedirs(os.path.join(nw, "tree", "docs", "perf", "logs", REF_DIR))
        shutil.copy(DEFAULT_AGGREGATE, os.path.join(nw, "tree", "docs", "perf", "logs", REF_DIR, "aggregate.py"))
        check("deploy.sh の配置で aggregate.py を見つける",
              ref_dir(os.path.join(nw, "rule")) == os.path.join(nw, "tree", "docs", "perf", "logs", REF_DIR))
        check("リポジトリの配置を先に使う", ref_dir(HERE) == os.path.normpath(os.path.join(HERE, "..", REF_DIR)))
        # 11. aggregate.py の sha256 が違えば読み込まない。
        bad = os.path.join(tmp, "aggregate.py")
        shutil.copy(DEFAULT_AGGREGATE, bad)
        with open(bad, "a", encoding="utf-8") as f:
            f.write("\n# changed\n")
        try:
            load_aggregate(bad)
            check("sha256 不一致を拒否", False)
        except JudgeError:
            check("sha256 不一致を拒否", True)
        # 12. ラベルが 2 つ混ざった ab/ は入力不正（判定を出さない）。
        d = _copy_series(tmp, "two")
        src = glob.glob(os.path.join(d, "ab", "results-before-*-metal-gemm.jsonl"))[0]
        shutil.copy(src, src.replace("results-before-", "results-before-x"))
        try:
            judge(d, agg)
            check("ラベル混在を拒否", False)
        except JudgeError:
            check("ラベル混在を拒否", True)
        # 13. CLI: 正常系は rc 0、壊れた入力は rc 2。
        check("CLI 正常 rc 0", main(["judge_ab.py", os.path.join(tmp, "base")]) == 0)
        check("CLI 入力不正 rc 2", main(["judge_ab.py", os.path.join(tmp, "nope")]) == 2)
    finally:
        shutil.rmtree(tmp)
    check("見本の記録ディレクトリに __pycache__ を作らない",
          not os.path.exists(os.path.join(os.path.dirname(DEFAULT_AGGREGATE), "__pycache__", "aggregate4160.cpython-%d%d.pyc" % sys.version_info[:2])))
    if fails:
        print("self-test FAILED: %d" % len(fails))
        return 1
    print("self-test ok")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
