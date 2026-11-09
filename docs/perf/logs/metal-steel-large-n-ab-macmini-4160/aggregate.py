#!/usr/bin/env python3
"""イシュー #4160 の集計（標準ライブラリのみ。`python3 -I aggregate.py <dir> [opts]`）。

規則の正は同ディレクトリの RULE.txt（雛形 RULE.template.txt を計測前に埋めて単独コミットで固定したもの）。
入力（<dir>）は orchestrate_macmini.sh が出力した trace_run{1..5}.log・kern_run{1..5}.log・
candle_run{1..5}.jsonl・prereq_tile.log・prereq_steel.log・bitident.log・parity.log・load_gate.log・
rev-stamp.txt・env_info.txt、および gemm セル A/B の出力 <dir>/ab/
（run_ab_steel_large_n_gemm_metal.sh の results-{before,after,candle}-<label>-metal-gemm.jsonl・
load-gate-4160-metal-<label>.log・skipped-4160-metal-<label>.log。無ければセル A/B は unavailable）。
出力（標準出力）は aggregate.md 相当の Markdown。判定は RULE 項 7・10・11 の機械判定そのもの。

厳格パース方針（欠損は寛容に埋めず必ず例外 → exit 1）:
- RULE.txt に未記入欄（`<...>`）が残る・aggregate_script_sha256 が本スクリプトの sha256 と一致しない系列は拒否する
- run が 5 本揃わない・行が足りない・欠損フィールド・重複・非数値・非有限値は拒否する
- DEVICE／TRACE が run 間で一致しない、同じ腕の hash が run 間で一致しない系列は拒否する
- TRACE の到達（N>=1024 の sl が profile=steel_large_n・fallback=false、N<=512 が profile=table）が
  成立しない N は「前提不成立」として undetermined（推定で埋めない）
- load_gate.log に全 gate 行が重複なく揃わない・gate=pass 以外・FAILED を含む・series done が無い系列は
  「正式系列でない」（判定は undetermined）
- `--expect-rev` 指定時は rev-stamp.txt と不一致なら拒否する
`--self-test` は一時ディレクトリへ生成した fixture で上記を検証する。

総合は FAIL（セル A/B の checksum 不一致）> undetermined > ADOPT／REJECT の順
（欠損があるまま ADOPT と書かない安全側）。スコープ（N=1024／2048／4096 の 3 点）を事後に絞って
ADOPT を作ることはしない（RULE 項 11）。
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

RUNS = [1, 2, 3, 4, 5]
SIZES = [256, 512, 1024, 2048, 4096]
SCOPE = [1024, 2048, 4096]  # opt-in が効く実測 3 点
NOOP = [256, 512]  # ON/OFF で解決タイルも出力も同一であるべき点
G_SIZES = [1024, 2048]  # スコアボード比の判定対象（G4096 は記録のみ）
MODES = ["fresh", "reuse"]
BAND = 0.02  # RULE 項 7・11: 2% 以内は帯内
ARMS = ["cur", "sl", "ref", "cur_aa"]
EXPECT_DEVICE = ("10", "Apple M4")
EXPECT_TABLE_PREFIXES = ("Some(M4Base(", "Some(M4Base)")
EST_PEAK = (3.3, 3.8)  # M4 10 コア GPU の f32 FMA 上限の推定（出てくるたびに「推定」と書く）
PLACEHOLDER_KEYS = ("fixed-at-utc", "tree_sha", "tree_fingerprint", "aggregate_script_sha256")


class AggregateError(Exception):
    pass


def fail(msg):
    raise AggregateError(msg)


def read_text(path):
    if not os.path.isfile(path):
        fail("ファイルが無い: %s" % os.path.basename(path))
    with open(path, encoding="utf-8") as f:
        return f.read()


def num(s, what):
    try:
        v = float(s)
    except ValueError:
        fail("非数値 %s=%r" % (what, s))
    if not math.isfinite(v):
        fail("非有限値 %s=%r" % (what, s))
    return v


def self_sha256():
    with open(os.path.abspath(__file__), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def check_rule(path):
    """RULE.txt の記入欄が埋まっており、集計器が固定時のものと同一であることを検査する（RULE 項 12）。"""
    text = read_text(path)
    found = {}
    for line in text.splitlines():
        m = re.match(r"^\s*(%s)\s*[:=]\s*(.*?)\s*$" % "|".join(re.escape(k) for k in PLACEHOLDER_KEYS), line)
        if m:
            found[m.group(1)] = m.group(2)
    for k in PLACEHOLDER_KEYS:
        v = found.get(k)
        if not v or re.search(r"<[^<>]*>", v):
            fail("RULE.txt の記入欄 %s が未記入（RULE.template.txt のまま）" % k)
    if found["aggregate_script_sha256"] != self_sha256():
        fail("aggregate_script_sha256 が本スクリプトと一致しない（集計器が固定後に差し替えられた）")
    return found


DEVICE_RE = re.compile(
    r'^DEVICE gpu_cores=(\S+) soc="([^"]*)" verified_tile_table=(\S+) split_k_enabled=(true|false)$')
TRACE_RE = re.compile(
    r"^TRACE n=(\d+) arm=(cur|sl) cfg=(\S+) profile=(table|steel_large_n) route=(Classic|SplitK) "
    r"fallback=(true|false)$")
PARITY_RE = re.compile(r"^PARITY n=(\d+) arm=(cur|sl) ok=(true|false)$")
BITIDENT_RE = re.compile(r"^BITIDENT n=(\d+) pair=(sl_vs_ref|sl_vs_cur) ok=(true|false)$")
MEAS_RE = re.compile(
    r"^MEAS n=(\d+) arm=(cur|sl|ref|cur_aa) role=(judge|record_only) kernel_gpu_ms=(\S+) encode_ms=(\S+) "
    r"commit_wait_ms=(\S+) alloc_c_ms=(\S+) wall_sum_ms=(\S+) hash=([0-9a-f]{16})$")
TOKENS = ("DEVICE ", "TRACE ", "MEAS ", "PARITY ", "BITIDENT ")


def lines_of(text):
    """cargo のテスト名前置（`test x ... `）を剥がし、診断行だけを返す。"""
    out = []
    for raw in text.splitlines():
        line = raw.strip()
        if "... " in line:
            _, tail = line.split("... ", 1)
            if tail.startswith(TOKENS):
                line = tail
        if line.startswith(TOKENS):
            out.append(line)
    return out


def parse_trace(path):
    dev, traces = None, {}
    for line in lines_of(read_text(path)):
        m = DEVICE_RE.match(line)
        if m:
            if dev is not None:
                fail("DEVICE 行が重複: %s" % os.path.basename(path))
            dev = m.groups()
            continue
        m = TRACE_RE.match(line)
        if m:
            key = (int(m.group(1)), m.group(2))
            if key in traces:
                fail("TRACE 行が重複 %s: %s" % (key, os.path.basename(path)))
            traces[key] = m.groups()[2:]
    if dev is None:
        fail("DEVICE 行が無い: %s" % os.path.basename(path))
    if sorted(traces) != sorted((n, a) for n in SIZES for a in ("cur", "sl")):
        fail("TRACE 行が 5 サイズ x 2 腕そろわない: %s" % os.path.basename(path))
    return dev, traces


def parse_kern(path):
    times = {}
    for line in lines_of(read_text(path)):
        m = MEAS_RE.match(line)
        if m:
            key = (int(m.group(1)), m.group(2))
            if key in times:
                fail("MEAS 行が重複 %s: %s" % (key, os.path.basename(path)))
            vals = [num(m.group(i), "metric") for i in range(4, 9)]
            if vals[0] <= 0:
                fail("kernel_gpu が非正 %s" % (key,))
            rec = dict(zip(("kg", "enc", "cw", "al", "wall"), vals))
            rec["role"], rec["hash"] = m.group(3), m.group(9)
            times[key] = rec
    for n in SIZES:
        for a in ARMS:
            if (n, a) not in times:
                fail("MEAS 欠損 N=%d arm=%s: %s" % (n, a, os.path.basename(path)))
    return times


def parse_parity(path):
    text = read_text(path)
    res = {}
    for line in lines_of(text):
        m = PARITY_RE.match(line)
        if m:
            res[(int(m.group(1)), m.group(2))] = m.group(3) == "true"
    for n in SIZES:
        for a in ("cur", "sl"):
            if (n, a) not in res:
                fail("PARITY 欠損 N=%d arm=%s" % (n, a))
    return res, "test result: ok." in text


def parse_bitident(path):
    text = read_text(path)
    res = {}
    for line in lines_of(text):
        m = BITIDENT_RE.match(line)
        if m:
            key = int(m.group(1))
            if key in res:
                fail("BITIDENT が重複 N=%d" % key)
            expect = "sl_vs_ref" if key in SCOPE else "sl_vs_cur"
            if m.group(2) != expect:
                fail("BITIDENT の pair が不正 N=%d pair=%s（期待 %s）" % (key, m.group(2), expect))
            res[key] = m.group(3) == "true"
    for n in SIZES:
        if n not in res:
            fail("BITIDENT 欠損 N=%d" % n)
    return res, "test result: ok." in text


def check_test_ok(path, what):
    text = read_text(path)
    if "test result:" not in text:
        fail("%s に test result: 行が無い（前提ゲート未完了）" % os.path.basename(path))
    return "test result: ok." in text and "test result: FAILED" not in text, what


GATE_LABELS = (["prereq-tile", "prereq-steel", "prereq-bitident", "prereq-parity"]
               + ["run%d-%s" % (i, k) for i in RUNS for k in ("trace", "kern")]
               + ["run%d-candle-N%d" % (i, n) for i in RUNS for n in SIZES])


def parse_gate(path):
    """load_gate.log を厳格に読む（RULE 項 6）。全 gate 行が重複なく gate=pass・FAILED なし・series done。"""
    text = read_text(path)
    official, reasons, seen = True, [], {}
    for line in text.splitlines():
        if "FAILED" in line:
            official = False
            reasons.append(line.strip())
        m = re.match(r"^(\S+) gate=(\S+)", line)
        if m:
            label, status = m.group(1), m.group(2)
            if label in seen:
                official = False
                reasons.append("gate 行が重複: %s" % label)
            seen[label] = status
            if status != "pass":
                official = False
                reasons.append(line.strip())
    for label in GATE_LABELS:
        if label not in seen:
            official = False
            reasons.append("gate 行が欠損: %s" % label)
    for label in seen:
        if label not in GATE_LABELS:
            official = False
            reasons.append("未知の gate ラベル: %s" % label)
    if not re.search(r"^series done\s*$", text, re.MULTILINE):
        official = False
        reasons.append("series done が無い（系列が完了していない）")
    return official, reasons


def parse_candle_excl(path):
    out, version = {}, None
    for ln, line in enumerate(read_text(path).splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            fail("JSON でない行 %s:%d" % (os.path.basename(path), ln))
        if rec.get("task") != "gemm_transfer_excl":
            continue
        n, med = rec.get("size"), rec.get("median_s")
        if rec.get("device") != "metal" or n in out:
            fail("candle excl の device／重複が不正 N=%s: %s" % (n, os.path.basename(path)))
        if (not isinstance(med, (int, float)) or isinstance(med, bool)
                or not math.isfinite(med) or med <= 0):
            fail("median_s が不正 N=%s: %s" % (n, os.path.basename(path)))
        ptot, pfail = rec.get("parity_total"), rec.get("parity_fail_count")
        if (not isinstance(n, int) or isinstance(ptot, bool) or not isinstance(ptot, int)
                or isinstance(pfail, bool) or not isinstance(pfail, int) or pfail < 0 or ptot != n * n):
            fail("parity_total／parity_fail_count が不正 N=%s: %s" % (n, os.path.basename(path)))
        out[n] = float(med) if pfail == 0 else None  # parity 不成立は判定不能（None）
        version = rec.get("version")
    if sorted(out) != SIZES:
        fail("candle excl の N が揃っていない: %s" % os.path.basename(path))
    if not version:
        fail("candle の version が無い")
    return out, version


def median(xs):
    return statistics.median(xs)


def judge_n(kg_cur, kg_sl, flags):
    """RULE 項 7。kg_* は 5 run の run 内中央値（ms）。flags は undetermined／not_adoptable の理由。

    戻り値: (status, ratio, faster, slower, regression)。regression = 2% 超かつ 5/5 run で遅い。
    """
    mc, ms = median(kg_cur), median(kg_sl)
    ratio = ms / mc - 1.0
    faster = sum(1 for c, t in zip(kg_cur, kg_sl) if t < c)
    slower = sum(1 for c, t in zip(kg_cur, kg_sl) if t > c)
    regression = ms > mc * (1 + BAND) and slower == 5
    if flags["undetermined"]:
        return "undetermined", ratio, faster, slower, regression
    if flags["not_adoptable"]:
        return "NOT_ADOPTABLE", ratio, faster, slower, regression
    if ms < mc * (1 - BAND) and faster == 5:
        return "ADOPT_CANDIDATE", ratio, faster, slower, regression
    return "REJECT_CELL", ratio, faster, slower, regression


def load_ab(abdir):
    """gemm セル A/B（RULE 項 9）。status(ok|anomaly|fail|unavailable|undetermined)・cells・candle・notes。"""
    notes = []
    if not os.path.isdir(abdir):
        return {"status": "unavailable", "cells": {}, "candle": {}, "notes": ["ab/ が無い"]}
    labels = set()
    for p in glob.glob(os.path.join(abdir, "results-before-*-metal-gemm.jsonl")):
        labels.add(re.match(r"results-before-(.+)-metal-gemm\.jsonl$", os.path.basename(p)).group(1))
    if len(labels) != 1:
        fail("ab/ の label を 1 つに確定できない: %s" % sorted(labels))
    label = labels.pop()

    def rows(arm):
        path = os.path.join(abdir, "results-%s-%s-metal-gemm.jsonl" % (arm, label))
        res = {}
        for ln, line in enumerate(read_text(path).splitlines(), 1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:
                fail("JSON でない行 %s:%d" % (os.path.basename(path), ln))
            if r.get("task") != "gemm" or r.get("device") != "metal":
                continue
            med = r.get("median_s")
            if (not isinstance(med, (int, float)) or isinstance(med, bool)
                    or not math.isfinite(med) or med <= 0):
                fail("median_s が不正 %s:%d" % (os.path.basename(path), ln))
            key = (r.get("size"), r.get("mode") or "fresh")
            res.setdefault(key, []).append(r)
        return res

    before, after, candle = rows("before"), rows("after"), rows("candle")
    gtext = read_text(os.path.join(abdir, "load-gate-4160-metal-%s.log" % label))
    official = "series=official" in gtext
    gates = re.findall(r"^round(\d+) gate=(\S+)", gtext, re.MULTILINE)
    if sorted(int(g[0]) for g in gates) != RUNS or any(g[1] != "pass" for g in gates):
        official = False
        notes.append("ab の load1 ゲートが全 round pass でない")
    skip = os.path.join(abdir, "skipped-4160-metal-%s.log" % label)
    if os.path.isfile(skip) and read_text(skip).strip():
        official = False
        notes.append("skipped ログに記録あり")
    cells, status = {}, "ok"
    for n in SIZES:
        for mode in MODES:
            b, a = before.get((n, mode), []), after.get((n, mode), [])
            if len(b) != 5 or len(a) != 5:
                fail("ab のセル行数が 5 でない N=%d mode=%s (before=%d after=%d)" % (n, mode, len(b), len(a)))
            ratios = [x["median_s"] / y["median_s"] for x, y in zip(a, b)]
            cks = all(x.get("checksum") == y.get("checksum") and x.get("checksum") is not None
                      for x, y in zip(a, b))
            cells[(n, mode)] = {
                "ratios": ratios, "ratio_med": median(ratios), "checksum_exact": cks,
                "over5": sum(1 for r in ratios if r > 1 + BAND),
                "before_ms": median([x["median_s"] for x in b]) * 1e3,
                "after_ms": median([x["median_s"] for x in a]) * 1e3,
                "before_runs": [x["median_s"] * 1e3 for x in b],
                "after_runs": [x["median_s"] * 1e3 for x in a],
            }
            if not cks:
                status = "fail"
    if status != "fail":
        if not official:
            status = "undetermined"
        elif any(c["over5"] == 5 for c in cells.values()):
            status = "anomaly"
    cc = {}
    for n in SIZES:
        c = candle.get((n, "fresh"), [])
        if len(c) != 5:
            fail("ab の candle 行数が 5 でない N=%d (%d)" % (n, len(c)))
        pf = [x.get("parity_fail_count") for x in c]
        cc[n] = {"ms": [x["median_s"] * 1e3 for x in c],
                 "valid": all(isinstance(p, int) and not isinstance(p, bool) and p == 0 for p in pf)}
    return {"status": status, "cells": cells, "candle": cc, "notes": notes, "label": label}


def scoreboard(ab, n):
    """RULE 項 10。round 別の比 = candle の gemm 窓（fresh）÷ fandhe-ai の reuse 窓（同じ round）。

    NB-RANGE5 = 両腕の最小〜最大。NB-SEP = after の 5/5 round で比 > 1.0 かつ after の最小 > before の最大。
    戻り値: dict（unavailable なら None）。
    """
    if not ab["cells"] or not ab["candle"][n]["valid"]:
        return None
    cd = ab["candle"][n]["ms"]
    cell = ab["cells"][(n, "reuse")]
    rb = [c / y for c, y in zip(cd, cell["before_runs"])]
    ra = [c / y for c, y in zip(cd, cell["after_runs"])]
    sep = all(r > 1.0 for r in ra) and min(ra) > max(rb)
    return {"before": rb, "after": ra, "sep": sep,
            "range_before": (min(rb), max(rb)), "range_after": (min(ra), max(ra))}


def overall(statuses, ab, nreg, noop_ok):
    if ab["status"] == "fail":
        return "FAIL（セル A/B の checksum 不一致）"
    if any(s == "undetermined" for s in statuses) or not noop_ok:
        return "undetermined"
    if ab["status"] in ("unavailable", "undetermined"):
        return "undetermined（セル A/B が %s）" % ab["status"]
    gate = [s for n, s in zip(SCOPE, statuses)]
    gated = [s for n, s in zip(SCOPE, statuses) if n in G_SIZES]
    na = sum(1 for s in gate if s == "NOT_ADOPTABLE")
    if any(s == "ADOPT_CANDIDATE" for s in gated) and na == 0 and nreg == 0 and ab["status"] == "ok":
        return "ADOPT"
    return "REJECT"


def aggregate(d, expect_rev=None):
    rule = check_rule(os.path.join(d, "RULE.txt"))
    rev = read_text(os.path.join(d, "rev-stamp.txt")).strip()
    # 事前登録した tree_sha と実測リビジョン（.rev-stamp の 1 行目）が一致しない系列は拒否する（腕の取り違え防止）。
    rev_first = rev.splitlines()[0].strip() if rev else ""
    if rule["tree_sha"] != rev_first:
        fail("RULE.txt の tree_sha と rev-stamp が不一致: %r != %r" % (rule["tree_sha"], rev_first))
    if expect_rev is not None and rev != expect_rev:
        fail("rev-stamp 不一致: %r != %r" % (rev, expect_rev))
    tile_ok, _ = check_test_ok(os.path.join(d, "prereq_tile.log"), "tile 単体")
    steel_ok, _ = check_test_ok(os.path.join(d, "prereq_steel.log"), "既存 steel 診断")
    bitident, bit_run_ok = parse_bitident(os.path.join(d, "bitident.log"))
    parity, parity_run_ok = parse_parity(os.path.join(d, "parity.log"))
    official, gate_reasons = parse_gate(os.path.join(d, "load_gate.log"))
    env = read_text(os.path.join(d, "env_info.txt"))
    if "host_kind=macmini" not in env:
        fail("env_info.txt に host_kind=macmini が無い（ラベル誤り防止。RULE 項 2）")
    traces, kerns, candles = [], [], []
    for i in RUNS:
        traces.append(parse_trace(os.path.join(d, "trace_run%d.log" % i)))
        kerns.append(parse_kern(os.path.join(d, "kern_run%d.log" % i)))
        candles.append(parse_candle_excl(os.path.join(d, "candle_run%d.jsonl" % i)))
    for t in traces[1:]:
        if t != traces[0]:
            fail("DEVICE／TRACE が run 間で一致しない")
    for key in kerns[0]:
        for field in ("hash", "role"):
            if len({k[key][field] for k in kerns}) != 1:
                fail("MEAS の %s が run 間で一致しない %s" % (field, key))
    versions = {v for _, v in candles}
    if len(versions) != 1:
        fail("candle の版が run 間で一致しない")
    cver = versions.pop()
    dev, trace = traces[0]
    if tuple(dev[:2]) != EXPECT_DEVICE or not dev[2].startswith(EXPECT_TABLE_PREFIXES):
        official = False
        gate_reasons = list(gate_reasons) + [
            "DEVICE が Mac mini（M4Base）の期待値と不一致: gpu_cores=%s soc=\"%s\" verified_tile_table=%s" % tuple(dev[:3])]
    ab = load_ab(os.path.join(d, "ab"))

    out = []
    w = out.append
    w("# #4160 集計（N≥1024 限定 steel 候補 T0U-LU-FB と現行 M4Base 表の Mac mini A/B）\n")
    w("- rev: `%s`" % rev)
    w("- DEVICE: gpu_cores=%s soc=\"%s\" verified_tile_table=%s split_k_enabled=%s" % tuple(dev))
    w("- 正式系列（load1 ゲート全通過）: %s" % ("はい" if official else "いいえ（参考系列）"))
    for r in gate_reasons:
        w("  - %s" % r)
    w("- 前提ゲート: tile 単体 %s／既存 steel 診断 %s／bit 一致テスト %s／parity テスト %s" % tuple(
        "pass" if x else "**FAIL**" for x in (tile_ok, steel_ok, bit_run_ok, parity_run_ok)))
    w("- candle: version=%s・5 run\n" % cver)

    w("## 到達追跡（5 run 全一致。RULE 項 4）\n")
    w("| N | cur の解決 | sl の解決 | sl の profile | route | fallback | 到達 |")
    w("|---|---|---|---|---|---|---|")
    reach = {}
    for n in SIZES:
        c, s = trace[(n, "cur")], trace[(n, "sl")]
        if n in SCOPE:
            ok = s[1] == "steel_large_n" and s[0].startswith("C0:") and s[2] == "Classic" and s[3] == "false"
        else:
            ok = s[1] == "table" and s[0] == c[0] and s[3] == "false"
        reach[n] = ok
        w("| %d | %s | %s | %s | %s | %s | %s |" % (n, c[0], s[0], s[1], s[2], s[3], "成立" if ok else "**前提不成立**"))

    w("\n## N<=512 の no-op（RULE 項 8）\n")
    noop_ok = True
    for n in NOOP:
        same = trace[(n, "cur")][0] == trace[(n, "sl")][0] and bitident[n] and reach[n]
        noop_ok = noop_ok and same
        w("- N=%d: 解決タイル同一・bit 一致 = %s" % (n, "はい" if same else "**いいえ（欠陥。系列を無効にする）**"))

    w("\n## 純カーネル時間（kernel_gpu の run 内中央値 ms）と判定（RULE 項 7）\n")
    w("| N | 腕 | run1 | run2 | run3 | run4 | run5 | 中央値 | sl/cur | TFLOPS（推定上限比ではない実測値） |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    statuses, regs, ninfo = [], 0, {}
    for n in SIZES:
        kg = {a: [k[(n, a)]["kg"] for k in kerns] for a in ARMS}
        for a in ("cur", "sl", "ref", "cur_aa"):
            tf = 2.0 * n ** 3 / (median(kg[a]) * 1e-3) / 1e12
            rt = "%+.1f%%" % ((median(kg["sl"]) / median(kg["cur"]) - 1) * 100) if a == "sl" else (
                "基準" if a == "cur" else "記録のみ")
            w("| %d | %s | %s | %.4f | %s | %.3f |" % (
                n, a, " | ".join("%.4f" % v for v in kg[a]), median(kg[a]), rt, tf))
        if n not in SCOPE:
            continue
        pass_all = parity[(n, "cur")] and parity[(n, "sl")] and parity_run_ok
        flags = {
            "undetermined": (not official) or (not reach[n]) or not (tile_ok and steel_ok),
            "not_adoptable": (not bitident[n]) or (not pass_all) or (not bit_run_ok),
        }
        st, ratio, fa, sl_, reg = judge_n(kg["cur"], kg["sl"], flags)
        statuses.append(st)
        regs += 1 if reg else 0
        ninfo[n] = (st, ratio, fa, sl_, reg, bitident[n], pass_all)
    w("\n| N | 判定 | 速い run | 遅い run | 後退（2% 超かつ 5/5 遅い） | bit 一致（sl=ref） | parity |")
    w("|---|---|---|---|---|---|---|")
    for n in SCOPE:
        st, ratio, fa, sl_, reg, bit_ok, par_ok = ninfo[n]
        w("| %d | %s（%+.1f%%） | %d/5 | %d/5 | %s | %s | %s |" % (
            n, st, ratio * 100, fa, sl_, "はい" if reg else "いいえ", "一致" if bit_ok else "**不一致**",
            "pass" if par_ok else "**FAIL**"))
    w("\n- cur_aa（cur と同構成の 2 本目）との差は A/A 対照として記録する（判定に使わない）。")
    for n in SCOPE:
        da = median([k[(n, "cur_aa")]["kg"] for k in kerns]) / median([k[(n, "cur")]["kg"] for k in kerns]) - 1
        w("  - N=%d: cur_aa/cur = %+.1f%%" % (n, da * 100))

    w("\n## gemm Metal セル A/B（非後退ゲート。RULE 項 9）\n")
    w("- 状態: **%s**" % ab["status"])
    for n_ in ab["notes"]:
        w("  - %s" % n_)
    if ab["cells"]:
        w("\n| N | mode | before ms | after ms | 比 after/before（round 別） | 中央値比 | 2% 超の round | checksum |")
        w("|---|---|---|---|---|---|---|---|")
        for n in SIZES:
            for mode in MODES:
                c = ab["cells"][(n, mode)]
                w("| %d | %s | %.4f | %.4f | %s | %.3f | %d/5 | %s |" % (
                    n, mode, c["before_ms"], c["after_ms"], " ".join("%.3f" % r for r in c["ratios"]),
                    c["ratio_med"], c["over5"], "完全一致" if c["checksum_exact"] else "**不一致**"))

    w("\n## 総合判定（RULE 項 11）\n")
    verdict = overall(statuses, ab, regs, noop_ok)
    w("- **候補 T0U-LU-FB（N≥1024 限定）: %s**" % verdict)
    w("- Mac mini 単機の判定。ADOPT は「Phase 4（#4166）で所有者が結線を判断する価値がある」の意味に限る。")
    w("- スコープ（3 点）を事後に絞って ADOPT を作ることはしない。")

    w("\n## スコアボード比（RULE 項 10。相手は candle だけ。M-MTL-G1024／G2048。G4096 は記録のみ）\n")
    w("- 比 = candle の gemm 窓（fresh）÷ fandhe-ai の reuse 窓（同じ round）。1 超で fandhe-ai が速い。")
    w("- NB-RANGE5 = 両腕の最小〜最大。NB-SEP = after の 5/5 round で比 > 1.0 かつ after の最小 > before の最大。")
    w("- NB-SEP が成立しない場合、「1 位に変わった」とは書かない（改善傾向・確定せず、または変化なし／後退）。")
    w("- NB-RANGE5／NB-SEP は docs/perf/noise-band-decision.md（#4142。記録時点で所有者承認待ちの案）の定義を名前で参照する。\n")
    w("| N | 対象 | before の比（範囲） | after の比（範囲） | NB-SEP |")
    w("|---|---|---|---|---|")
    for n in (1024, 2048, 4096):
        sb = scoreboard(ab, n)
        tag = "判定対象" if n in G_SIZES else "記録のみ"
        if sb is None:
            w("| %d | %s | unavailable | unavailable | 判定不能 |" % (n, tag))
            continue
        w("| %d | %s | %.3f〜%.3f | %.3f〜%.3f | %s |" % (
            n, tag, *sb["range_before"], *sb["range_after"], "成立" if sb["sep"] else "不成立"))

    w("\n## candle excl（record_only。5 run 中央値 ms）と達成 TFLOPS\n")
    w("| N | candle excl ms | run 別 | cur kernel_gpu | sl kernel_gpu | sl TFLOPS（実測。推定上限 %.1f〜%.1f に対し推定 %s） |" % (
        EST_PEAK[0], EST_PEAK[1], "到達率"))
    w("|---|---|---|---|---|---|")
    for n in SIZES:
        raw = [c[0][n] for c in candles]
        kc = median([k[(n, "cur")]["kg"] for k in kerns])
        ks = median([k[(n, "sl")]["kg"] for k in kerns])
        tf = 2.0 * n ** 3 / (ks * 1e-3) / 1e12
        est = "推定 %.0f%%〜%.0f%%" % (tf / EST_PEAK[1] * 100, tf / EST_PEAK[0] * 100)
        if any(v is None for v in raw):
            w("| %d | 判定不能（candle parity 不成立の run あり） | - | %.4f | %.4f | %.3f（%s） |" % (n, kc, ks, tf, est))
            continue
        w("| %d | %.4f | %s | %.4f | %.4f | %.3f（%s） |" % (
            n, median(raw) * 1e3, " ".join("%.4f" % (v * 1e3) for v in raw), kc, ks, tf, est))
    return "\n".join(out) + "\n"


# ---------------- self-test ----------------

def _fixture(d, ratio=0.9, official=True, drop_run=None, nonfinite=False, hash_mismatch=False,
             reach_ok=True, bit_fail=False, ab_mode="ok", soc="Apple M4", host_kind=True,
             rule="ok", dup_trace=False, noop_break=False):
    """正常系 fixture を d に書く。ratio = sl の kernel_gpu / cur。"""
    def wr(name, text):
        with open(os.path.join(d, name), "w") as f:
            f.write(text)

    keys = {"fixed-at-utc": "2026-11-09T00:00:00Z", "tree_sha": "deadbeef", "tree_fingerprint": "b" * 64,
            "aggregate_script_sha256": self_sha256()}
    if rule == "placeholder":
        keys["tree_sha"] = "<git rev-parse HEAD>"
    if rule == "treemismatch":
        keys["tree_sha"] = "a" * 40
    if rule == "badsha":
        keys["aggregate_script_sha256"] = "0" * 64
    if rule != "none":
        wr("RULE.txt", "".join("%s: %s\n" % kv for kv in keys.items()))
    wr("rev-stamp.txt", "deadbeef\n")
    wr("env_info.txt", ("host_kind=macmini\n" if host_kind else "") + "Apple M4\n")
    wr("prereq_tile.log", "test result: ok. 3 passed; 0 failed\n")
    wr("prereq_steel.log", "test result: ok. 6 passed; 0 failed\n")
    wr("bitident.log", "".join(
        "BITIDENT n=%d pair=%s ok=%s\n" % (
            n, "sl_vs_ref" if n in SCOPE else "sl_vs_cur",
            "false" if (bit_fail and n == 1024) or (noop_break and n == 512) else "true")
        for n in SIZES) + "test result: ok. 1 passed\n")
    wr("parity.log", "".join("PARITY n=%d arm=%s ok=true\n" % (n, a)
                             for n in SIZES for a in ("cur", "sl")) + "test result: ok. 1 passed\n")
    g = "build-done x\n"
    for lab in GATE_LABELS:
        g += "%s gate=%s load1=0.5\n" % (lab, "pass" if official or lab != "run3-trace" else "timeout")
    wr("load_gate.log", g + "series done\n")
    for i in RUNS:
        if drop_run == i:
            continue
        t = 'DEVICE gpu_cores=10 soc="%s" verified_tile_table=Some(M4Base(VerifiedM4BaseGpuCoreCount(10))) split_k_enabled=true\n' % soc
        for n in SIZES:
            for a in ("cur", "sl"):
                scope_sl = a == "sl" and n in SCOPE
                prof = "steel_large_n" if scope_sl else "table"
                cfg = "C0:x" if scope_sl else "C1:x"
                if scope_sl and not reach_ok:
                    prof = "table"
                t += "TRACE n=%d arm=%s cfg=%s profile=%s route=Classic fallback=false\n" % (n, a, cfg, prof)
                if dup_trace and n == 256 and a == "cur":
                    t += "TRACE n=%d arm=%s cfg=%s profile=%s route=Classic fallback=false\n" % (n, a, cfg, prof)
        wr("trace_run%d.log" % i, t)
        k = ""
        for n in SIZES:
            for a in ARMS:
                h = 0x1111
                if a == "sl" and hash_mismatch and i == 2:
                    h = 0x2222
                base = (0.1 + 0.001 * i) * (n / 256.0)
                v = base * (ratio if a == "sl" else 1.0)
                vs = "nan" if (nonfinite and i == 1 and a == "cur" and n == 256) else "%.4f" % v
                k += ("MEAS n=%d arm=%s role=%s kernel_gpu_ms=%s encode_ms=0.01 commit_wait_ms=0.2 "
                      "alloc_c_ms=0.01 wall_sum_ms=%.4f hash=%016x\n" % (
                          n, a, "judge" if a in ("cur", "sl") else "record_only", vs, base * 1.5, h))
        wr("kern_run%d.log" % i, k)
        c = ""
        for n in SIZES:
            c += json.dumps({"framework": "candle", "version": "0.11.0", "task": "gemm_transfer_excl",
                             "device": "metal", "size": n, "median_s": 0.002 * (n / 256.0),
                             "parity_total": n * n, "parity_fail_count": 0}) + "\n"
        wr("candle_run%d.jsonl" % i, c)
    if ab_mode != "none":
        ab = os.path.join(d, "ab")
        os.makedirs(ab)
        for arm in ("before", "after", "candle"):
            lines = []
            for r in RUNS:
                for n in SIZES:
                    for mode in (MODES if arm != "candle" else ["fresh"]):
                        med = 0.001 * (n / 256.0) * (1.05 if (arm == "after" and ab_mode == "regress") else 1.0)
                        ck = 3.5 + (0.1 if (arm == "after" and ab_mode == "ckfail") else 0.0)
                        lines.append(json.dumps({"task": "gemm", "device": "metal", "size": n, "mode": mode,
                                                 "median_s": med, "checksum": ck, "parity_fail_count": 0}))
            with open(os.path.join(ab, "results-%s-4160-metal-gemm.jsonl" % arm), "w") as f:
                f.write("\n".join(lines) + "\n")
        with open(os.path.join(ab, "load-gate-4160-metal-4160.log"), "w") as f:
            f.write("threshold=3.0 rule_threshold=3.0 series=official\n")
            for r in RUNS:
                f.write("round%d gate=pass threshold=3.0 load1=0.5 waited_s=0\n" % r)
        with open(os.path.join(ab, "skipped-4160-metal-4160.log"), "w") as f:
            f.write("")


def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + name)
        ok = ok and cond

    def run(**kw):
        with tempfile.TemporaryDirectory() as d:
            _fixture(d, **kw)
            return aggregate(d, "deadbeef")

    def rejects(name, **kw):
        try:
            run(**kw)
        except AggregateError:
            check("fail-closed: " + name, True)
            return
        check("fail-closed: " + name + "（拒否されなかった）", False)

    T = "T0U-LU-FB（N≥1024 限定）: "
    check("正常系（10% 速い sl は ADOPT）", T + "ADOPT**" in run())
    check("帯内（+1%）は REJECT", T + "REJECT**" in run(ratio=1.01))
    md = run(ratio=1.2)
    check("2% 超かつ 5/5 遅いは REJECT（後退）", T + "REJECT**" in md and "| はい |" in md)
    check("正式系列でなければ undetermined", T + "undetermined**" in run(official=False))
    check("到達不成立（profile=table）は undetermined", T + "undetermined**" in run(reach_ok=False))
    check("bit 不一致は NOT_ADOPTABLE（ADOPT にならない）",
          "NOT_ADOPTABLE" in run(bit_fail=True) and T + "ADOPT**" not in run(bit_fail=True))
    check("N=512 の no-op 崩れは undetermined", T + "undetermined**" in run(noop_break=True))
    check("機体不一致（M4 Max）は参考系列で ADOPT にならない", T + "ADOPT**" not in run(soc="Apple M4 Max"))
    check("セル A/B の checksum 不一致は FAIL", T + "FAIL" in run(ab_mode="ckfail"))
    check("セル A/B が無ければ undetermined", T + "undetermined" in run(ab_mode="none"))
    check("セル A/B の 5/5 round 後退は ADOPT にならない", T + "REJECT" in run(ab_mode="regress"))
    check("NB-SEP は before=after のとき不成立", "| 不成立 |" in run())
    rejects("run 数不足", drop_run=4)
    rejects("非有限値", nonfinite=True)
    rejects("hash の run 間不一致", hash_mismatch=True)
    rejects("host_kind 欠落", host_kind=False)
    rejects("TRACE 行の重複", dup_trace=True)
    rejects("RULE.txt の記入欄が未記入", rule="placeholder")
    rejects("RULE.txt の集計器 sha256 不一致", rule="badsha")
    rejects("RULE.txt が無い", rule="none")
    rejects("RULE.txt の tree_sha と rev-stamp の不一致", rule="treemismatch")
    with tempfile.TemporaryDirectory() as d:
        _fixture(d)
        try:
            aggregate(d, "other")
            check("rev 不一致が拒否されなかった", False)
        except AggregateError:
            check("fail-closed: rev 不一致", True)
        os.remove(os.path.join(d, "parity.log"))
        try:
            aggregate(d, "deadbeef")
            check("parity 欠損が拒否されなかった", False)
        except AggregateError:
            check("fail-closed: parity.log 欠損", True)
    with tempfile.TemporaryDirectory() as d:
        _fixture(d)
        with open(os.path.join(d, "load_gate.log"), "w") as f:
            f.write("build-done x\nprereq-tile gate=pass load1=0.5\nseries done\n")
        md = aggregate(d, "deadbeef")
        check("一部のみの load_gate.log は正式系列でない", "いいえ（参考系列）" in md and T + "undetermined" in md)
    return 0 if ok else 1


def main(argv):
    if "--self-test" in argv:
        return self_test()
    args = [a for a in argv[1:] if not a.startswith("--")]
    expect = None
    if "--expect-rev" in argv:
        expect = argv[argv.index("--expect-rev") + 1]
        args = [a for a in args if a != expect]
    if len(args) != 1:
        print("usage: aggregate.py <dir> [--expect-rev SHA] | --self-test", file=sys.stderr)
        return 2
    try:
        sys.stdout.write(aggregate(args[0], expect))
    except AggregateError as e:
        print("AGGREGATE ERROR: %s" % e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
