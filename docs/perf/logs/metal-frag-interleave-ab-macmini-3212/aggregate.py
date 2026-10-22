#!/usr/bin/env python3
"""イシュー #3212 の集計（標準ライブラリのみ。`python3 -I aggregate.py <dir> [opts]`）。

規則の正は同ディレクトリの RULE.txt。入力（<dir>）は orchestrate_macmini.sh が出力した
trace_run{1..5}.log・kern_run{1..5}.log・candle_run{1..5}.jsonl・parity.log・prereq_bitident.log・
bitident_report.log・load_gate.log・rev-stamp.txt、および gemm セル A/B の出力 <dir>/ab/
（run_ab_frag_interleave_gemm_metal.sh の results-{before,after,candle}-<label>-metal-gemm.jsonl・
load-gate-3212-metal-<label>.log。無ければセル A/B は unavailable）。
出力（標準出力）は aggregate.md 相当の Markdown。判定は RULE 項 10・12 の機械判定そのもの。

厳格パース方針（欠損は寛容に埋めず必ず例外 → exit 1）:
- run が 5 本揃わない・行が足りない・欠損フィールド・非数値・非有限値は拒否する
- DEVICE／TRACE が run 間で一致しない、同じ腕の hash／解決構成が run 間で一致しない系列は拒否する
- load_gate.log に全 gate 行が重複なく揃わない・gate=pass 以外・FAILED を含む・series done が無い系列は
  「正式系列でない」（判定は undetermined）
- `--expect-rev` 指定時は rev-stamp.txt と不一致なら拒否する
`--self-test` は一時ディレクトリへ生成した fixture（正常系・欠損・run 数不足・非有限値・hash 不一致・
正式系列でない・active=false・cell A/B の checksum 不一致 等）で上記を検証する。

判定の実装上の読み: 総合は FAIL（セル A/B の checksum 不一致）> undetermined（必要なセルが 1 件でも
undetermined、またはセル A/B が判定不能）> ADOPT／REJECT の順（欠損があるまま ADOPT と書かない安全側）。
"""
import glob
import json
import math
import os
import re
import statistics
import sys
import tempfile

RUNS = [1, 2, 3, 4, 5]
SIZES = [256, 512, 1024, 2048, 4096]
MODES = ["fresh", "reuse"]
BAND = 0.02  # RULE 項 10・12: 2% 以内は帯内
AUX = ["c1_off", "c1_on", "c2_off", "c2_on", "c3_off", "c3_on"]
ARMS = ["cur", "fi"] + AUX
PAIRS = ["fi_vs_cur", "c1_on_vs_c1_off", "c2_on_vs_c2_off", "c3_on_vs_c3_off"]
EXPECT_DEVICE = ("10", "Apple M4")
EXPECT_TABLE_PREFIXES = ("Some(M4Base(", "Some(M4Base)")
# 3.3〜3.8 TFLOPS は M4 10 コア GPU の f32 FMA 上限の推定（RULE 項 14。導出は steel-candidates §10.9）
EST_PEAK = (3.3, 3.8)


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


DEVICE_RE = re.compile(
    r'^DEVICE gpu_cores=(\S+) soc="([^"]*)" verified_tile_table=(\S+) split_k_enabled=(true|false)$')
TRACE_RE = re.compile(
    r"^TRACE n=(\d+) should_split_k=(\S+) route=(\S+) cur=(\S+) resolved_cur=(\S+) "
    r"unroll_acc_eff=(true|false) classic=(true|false) active_on=(true|false)$")
PARITY_RE = re.compile(r"^PARITY n=(\d+) arm=(cur|fi) fallback=(true|false) ok=(true|false|skipped)$")
BITIDENT_RE = re.compile(
    r"^BITIDENT n=(\d+) pair=(fi_vs_cur|c[123]_on_vs_c[123]_off) fallback=(true|false) ok=(true|false|skipped)$")
ARM_RE = re.compile(
    r"^ARM n=(\d+) arm=(cur|fi|c[123]_(?:on|off)) role=(judge|record_only) gate=(on|off) requested=(\S+) "
    r"resolved=(\S+) fallback=(true|false) active=(true|false) checksum=(\S+) hash=([0-9a-f]{16})$")
MEAS_RE = re.compile(
    r"^MEAS n=(\d+) arm=(cur|fi|c[123]_(?:on|off)) kernel_gpu_median_ms=(\S+) q1=(\S+) q3=(\S+) "
    r"encode_median_ms=(\S+) commit_wait_median_ms=(\S+) alloc_c_median_ms=(\S+) wall_sum_median_ms=(\S+)$")
TOKENS = ("DEVICE ", "TRACE ", "ARM ", "MEAS ", "PARITY ", "BITIDENT ")


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
            n = int(m.group(1))
            if n in traces:
                fail("TRACE 行が重複 N=%d: %s" % (n, os.path.basename(path)))
            traces[n] = m.groups()
    if dev is None:
        fail("DEVICE 行が無い: %s" % os.path.basename(path))
    if sorted(traces) != SIZES:
        fail("TRACE 行が 5 サイズそろわない: %s" % os.path.basename(path))
    return dev, traces


def parse_kern(path):
    heads, times = {}, {}
    for line in lines_of(read_text(path)):
        m = ARM_RE.match(line)
        if m:
            key = (int(m.group(1)), m.group(2))
            if key in heads:
                fail("ARM 行が重複 %s: %s" % (key, os.path.basename(path)))
            num(m.group(9), "checksum")
            heads[key] = {"role": m.group(3), "gate": m.group(4), "requested": m.group(5),
                          "resolved": m.group(6), "fallback": m.group(7) == "true",
                          "active": m.group(8) == "true", "hash": m.group(10)}
            continue
        m = MEAS_RE.match(line)
        if m:
            key = (int(m.group(1)), m.group(2))
            if key in times:
                fail("MEAS 行が重複 %s: %s" % (key, os.path.basename(path)))
            vals = [num(m.group(i), "metric") for i in range(3, 10)]
            if vals[0] <= 0:
                fail("kernel_gpu が非正 %s" % (key,))
            times[key] = dict(zip(("kg", "q1", "q3", "enc", "cw", "al", "wall"), vals))
    for n in SIZES:
        for a in ARMS:
            if (n, a) not in heads or (n, a) not in times:
                fail("ARM／MEAS 欠損 N=%d arm=%s: %s" % (n, a, os.path.basename(path)))
    return heads, times


def parse_parity(path):
    text = read_text(path)
    res = {}
    for line in lines_of(text):
        m = PARITY_RE.match(line)
        if m:
            res[(int(m.group(1)), m.group(2))] = (m.group(3) == "true", m.group(4))
            if m.group(2) == "cur" and m.group(4) == "false":
                fail("cur 腕の parity が FAIL（系列を成立させない）")
    for n in SIZES:
        for a in ("cur", "fi"):
            if (n, a) not in res:
                fail("PARITY 欠損 N=%d arm=%s" % (n, a))
    if "test result: ok." not in text:
        fail("parity.log に test result: ok. が無い（前提ゲートの assert が落ちた可能性）")
    return res


def parse_bitident(path):
    text = read_text(path)
    res = {}
    for line in lines_of(text):
        m = BITIDENT_RE.match(line)
        if m:
            key = (int(m.group(1)), m.group(2))
            if key in res:
                fail("BITIDENT が重複 %s" % (key,))
            res[key] = (m.group(3) == "true", m.group(4))
    for n in SIZES:
        for p in PAIRS:
            if (n, p) not in res:
                fail("BITIDENT 欠損 N=%d pair=%s" % (n, p))
    if "test result: ok." not in text:
        fail("bitident_report.log に test result: ok. が無い")
    return res


def check_prereq(path):
    text = read_text(path)
    if "test result:" not in text:
        fail("prereq_bitident.log に test result: 行が無い（前提ゲート未完了）")
    ok = "test result: ok." in text and "test result: FAILED" not in text
    return ok, "既存の厳密診断（gemm_frag_interleave_diag_tests）: %s" % (
        "全件 pass" if ok else "失敗あり（ADOPT にしない）")


GATE_LABELS = (["prereq-bitident", "prereq-bitident-report", "parity"]
               + ["run%d-%s" % (i, k) for i in RUNS for k in ("trace", "kern")]
               + ["run%d-candle-N%d" % (i, n) for i in RUNS for n in SIZES])


def parse_gate(path):
    """load_gate.log を厳格に読む（RULE 項 9）。全 gate 行が重複なく gate=pass・FAILED なし・series done。"""
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


def judge_n(kg_cur, kg_fi, flags):
    """RULE 項 10。kg_* は 5 run の run 内中央値（ms）。flags は前提（undetermined／NOT_ADOPTABLE の理由）。

    戻り値: (status, ratio, faster, slower, regression)。regression = 2% 超かつ 5/5 run で遅い。
    """
    mc, mf = median(kg_cur), median(kg_fi)
    ratio = mf / mc - 1.0
    faster = sum(1 for c, t in zip(kg_cur, kg_fi) if t < c)
    slower = sum(1 for c, t in zip(kg_cur, kg_fi) if t > c)
    regression = mf > mc * (1 + BAND) and slower == 5
    if flags["undetermined"]:
        return "undetermined", ratio, faster, slower, regression
    if flags["not_adoptable"]:
        return "NOT_ADOPTABLE", ratio, faster, slower, regression
    if mf < mc * (1 - BAND) and faster == 5:
        return "ADOPT_CANDIDATE", ratio, faster, slower, regression
    return "REJECT_CELL", ratio, faster, slower, regression


def load_ab(abdir):
    """gemm セル A/B（RULE 項 11）。戻り値 dict: status(ok|anomaly|fail|unavailable|undetermined)・cells・notes。"""
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
    gate_path = os.path.join(abdir, "load-gate-3212-metal-%s.log" % label)
    gtext = read_text(gate_path)
    official = "series=official" in gtext
    gates = re.findall(r"^round(\d+) gate=(\S+)", gtext, re.MULTILINE)
    if sorted(int(g[0]) for g in gates) != RUNS or any(g[1] != "pass" for g in gates):
        official = False
        notes.append("ab の load1 ゲートが全 round pass でない")
    skip = os.path.join(abdir, "skipped-3212-metal-%s.log" % label)
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


def overall(statuses, ab, nreg):
    if ab["status"] == "fail":
        return "FAIL（セル A/B の checksum 不一致）"
    if any(s == "undetermined" for s in statuses):
        return "undetermined"
    if ab["status"] in ("unavailable", "undetermined"):
        return "undetermined（セル A/B が %s）" % ab["status"]
    adopt = sum(1 for s in statuses if s == "ADOPT_CANDIDATE")
    na = sum(1 for s in statuses if s == "NOT_ADOPTABLE")
    if adopt >= 1 and na == 0 and nreg == 0 and ab["status"] == "ok":
        return "ADOPT"
    return "REJECT"


def baseline_3181(d):
    """#3181 の candle excl（5 run 中央値 ms）。無ければ None（ドリフト比較は unavailable）。"""
    base = os.path.join(os.path.dirname(os.path.abspath(d)), "metal-tile-macmini-3181")
    out = {n: [] for n in SIZES}
    for i in RUNS:
        p = os.path.join(base, "candle_run%d.jsonl" % i)
        if not os.path.isfile(p):
            return None
        try:
            c, _ = parse_candle_excl(p)
        except AggregateError:
            return None
        for n in SIZES:
            if c[n] is None:
                return None
            out[n].append(c[n] * 1e3)
    return {n: median(v) for n, v in out.items()}


def aggregate(d, expect_rev=None, with_baseline=True):
    rev = read_text(os.path.join(d, "rev-stamp.txt")).strip()
    if expect_rev is not None and rev != expect_rev:
        fail("rev-stamp 不一致: %r != %r" % (rev, expect_rev))
    strict_ok, strict_note = check_prereq(os.path.join(d, "prereq_bitident.log"))
    bitident = parse_bitident(os.path.join(d, "bitident_report.log"))
    official, gate_reasons = parse_gate(os.path.join(d, "load_gate.log"))
    parity = parse_parity(os.path.join(d, "parity.log"))
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
    for key in kerns[0][0]:
        for field in ("hash", "fallback", "requested", "resolved", "active", "gate", "role"):
            if len({k[0][key][field] for k in kerns}) != 1:
                fail("ARM の %s が run 間で一致しない %s" % (field, key))
    versions = {v for _, v in candles}
    if len(versions) != 1:
        fail("candle の版が run 間で一致しない")
    cver = versions.pop()
    dev, trace = traces[0]
    heads0 = kerns[0][0]
    if tuple(dev[:2]) != EXPECT_DEVICE or not dev[2].startswith(EXPECT_TABLE_PREFIXES):
        official = False
        gate_reasons = list(gate_reasons) + [
            "DEVICE が Mac mini（M4Base）の期待値と不一致: gpu_cores=%s soc=\"%s\" verified_tile_table=%s" % tuple(dev[:3])]
    ab = load_ab(os.path.join(d, "ab"))

    out = []
    w = out.append
    w("# #3212 集計（simdgroup 交互配置候補 FI の Mac mini A/B）\n")
    w("- rev: `%s`" % rev)
    w("- DEVICE: gpu_cores=%s soc=\"%s\" verified_tile_table=%s split_k_enabled=%s" % tuple(dev))
    w("- 正式系列（load1 ゲート全通過）: %s" % ("はい" if official else "いいえ（参考系列）"))
    for r in gate_reasons:
        w("  - %s" % r)
    w("- %s" % strict_note)
    w("- candle: version=%s・5 run\n" % cver)

    w("## 到達追跡（5 run 全一致。RULE 項 5）\n")
    w("| N | should_split_k | route | cur（解決） | unroll_acc_eff | classic | active（ON） |")
    w("|---|---|---|---|---|---|---|")
    reach = {}
    for n in SIZES:
        t = trace[n]
        ok = t[1] == "none" and t[6] == "true" and t[7] == "true" and heads0[(n, "fi")]["active"]
        reach[n] = ok
        w("| %d | %s | %s | %s | %s | %s | %s |" % (n, t[1], t[2], t[4], t[5], t[6], t[7]))

    w("\n## 主腕の純カーネル時間（kernel_gpu の run 内中央値 ms。run 1〜5・5 run 中央値）\n")
    w("| N | 腕 | run1 | run2 | run3 | run4 | run5 | 中央値 | fi/cur | TFLOPS | wall_sum 中央値 |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    statuses, regs, ninfo = [], 0, {}
    for n in SIZES:
        kg = {a: [k[1][(n, a)]["kg"] for k in kerns] for a in ARMS}
        wall = {a: median([k[1][(n, a)]["wall"] for k in kerns]) for a in ARMS}
        hc, hf = heads0[(n, "cur")], heads0[(n, "fi")]
        bit_ok = hc["hash"] == hf["hash"] and bitident[(n, "fi_vs_cur")][1] == "true"
        pc, pf = parity[(n, "cur")], parity[(n, "fi")]
        fb = hc["fallback"] or hf["fallback"] or bitident[(n, "fi_vs_cur")][0] or pc[0] or pf[0]
        parity_ok = pc[1] == "true" and pf[1] == "true"
        flags = {
            "undetermined": (not official) or (not reach[n]),
            "not_adoptable": fb or (not bit_ok) or (not parity_ok),
        }
        st, ratio, fa, sl, reg = judge_n(kg["cur"], kg["fi"], flags)
        if st == "ADOPT_CANDIDATE" and not strict_ok:
            st = "undetermined"
        statuses.append(st)
        regs += 1 if reg else 0
        ninfo[n] = (st, ratio, fa, sl, reg, bit_ok, parity_ok, fb)
        for a in ("cur", "fi"):
            tf = 2.0 * n ** 3 / (median(kg[a]) * 1e-3) / 1e12
            rt = "%+.1f%%" % (ratio * 100) if a == "fi" else "基準"
            w("| %d | %s | %s | %.4f | %s | %.3f | %.4f |" % (
                n, a, " | ".join("%.4f" % v for v in kg[a]), median(kg[a]), rt, tf, wall[a]))
    w("\n## N ごとの判定（RULE 項 10）\n")
    w("| N | 判定 | 速い run | 遅い run | 後退（2% 超かつ 5/5 遅い） | bit 一致 | parity | fallback |")
    w("|---|---|---|---|---|---|---|---|")
    for n in SIZES:
        st, ratio, fa, sl, reg, bit_ok, parity_ok, fb = ninfo[n]
        w("| %d | %s（%+.1f%%） | %d/5 | %d/5 | %s | %s | %s | %s |" % (
            n, st, ratio * 100, fa, sl, "はい" if reg else "いいえ", "一致" if bit_ok else "**不一致**",
            "pass" if parity_ok else "**FAIL**", "あり" if fb else "なし"))

    w("\n## gemm Metal セル A/B（非後退ゲート。RULE 項 11）\n")
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

    w("\n## 候補 FI の総合判定（RULE 項 12）\n")
    verdict = overall(statuses, ab, regs)
    w("- **候補 FI: %s**" % verdict)
    w("- Mac mini 単機の判定。ゲートは機体に依存しない（M4 Max にも効く）ため、結線の前に M4 Max での確認が要る。")

    w("\n## candle 比（RULE 項 13）\n")
    w("- candle excl は `matmul + synchronize` のホスト壁時計。主比 = excl ÷ fandhe wall_sum、参考比 = excl ÷ fandhe kernel_gpu。")
    w("  どちらも純カーネル比そのものではない（#3181 §4）。1 未満は fandhe が遅い。\n")
    base = baseline_3181(d) if with_baseline else None
    w("| N | candle excl ms（5 run 中央値） | run 別 | 主比 cur | 主比 fi | 参考比 cur | 参考比 fi | #3181 の excl ms | ドリフト |")
    w("|---|---|---|---|---|---|---|---|---|")
    for n in SIZES:
        raw = [c[0][n] for c in candles]
        if any(v is None for v in raw):
            w("| %d | 判定不能（candle parity 不成立の run あり） | - | n/a | n/a | n/a | n/a | - | - |" % n)
            continue
        ex = median(raw) * 1e3
        wc = median([k[1][(n, "cur")]["wall"] for k in kerns])
        wf = median([k[1][(n, "fi")]["wall"] for k in kerns])
        kc = median([k[1][(n, "cur")]["kg"] for k in kerns])
        kf = median([k[1][(n, "fi")]["kg"] for k in kerns])
        if base is None:
            b_txt, dr = "unavailable", "-"
        else:
            b_txt = "%.4f" % base[n]
            drift = ex / base[n] - 1.0
            dr = "%+.1f%%%s" % (drift * 100, "（10% 超: ドリフト）" if abs(drift) > 0.10 else "")
        w("| %d | %.4f | %s | %.3f | %.3f | %.3f | %.3f | %s | %s |" % (
            n, ex, " ".join("%.4f" % (v * 1e3) for v in raw), ex / wc, ex / wf, ex / kc, ex / kf, b_txt, dr))

    if ab["cells"]:
        w("\n### セル比（candle gemm 窓 ÷ fandhe gemm 窓。同じ round の値の中央値。record_only）\n")
        w("| N | mode | candle ms | before | after | 比（candle/before） | 比（candle/after） |")
        w("|---|---|---|---|---|---|---|")
        for n in SIZES:
            cd = ab["candle"][n]
            for mode in MODES:
                c = ab["cells"][(n, mode)]
                if mode != "fresh" or not cd["valid"]:
                    w("| %d | %s | %s | %.4f | %.4f | %s | %s |" % (
                        n, mode, "-" if mode != "fresh" else "判定不能", c["before_ms"], c["after_ms"], "n/a", "n/a"))
                    continue
                rb = median([x / y for x, y in zip(cd["ms"], c["before_runs"])])
                ra = median([x / y for x, y in zip(cd["ms"], c["after_runs"])])
                w("| %d | %s | %.4f | %.4f | %.4f | %.3f | %.3f |" % (
                    n, mode, median(cd["ms"]), c["before_ms"], c["after_ms"], rb, ra))

    w("\n## 達成 TFLOPS と推定上限（RULE 項 14。上限は**推定**）\n")
    w("| N | cur TFLOPS | fi TFLOPS | 推定上限 %.1f〜%.1f に対する fi の到達率（推定） |" % EST_PEAK)
    w("|---|---|---|---|")
    for n in SIZES:
        tf = {a: 2.0 * n ** 3 / (median([k[1][(n, a)]["kg"] for k in kerns]) * 1e-3) / 1e12 for a in ("cur", "fi")}
        w("| %d | %.3f | %.3f | %.0f%%〜%.0f%% |" % (n, tf["cur"], tf["fi"],
                                                    tf["fi"] / EST_PEAK[1] * 100, tf["fi"] / EST_PEAK[0] * 100))

    w("\n## 補助腕（記録のみ。判定に使わない。kernel_gpu 5 run 中央値 ms）\n")
    w("| N | " + " | ".join(AUX) + " | ON/OFF 比 c1 | c2 | c3 | bit（ON=OFF） |")
    w("|---|" + "---|" * (len(AUX) + 4))
    for n in SIZES:
        v = {a: median([k[1][(n, a)]["kg"] for k in kerns]) for a in AUX}
        bits = ",".join(bitident[(n, p)][1] for p in PAIRS[1:])
        w("| %d | %s | %.3f | %.3f | %.3f | %s |" % (
            n, " | ".join("%.4f" % v[a] for a in AUX),
            v["c1_on"] / v["c1_off"], v["c2_on"] / v["c2_off"], v["c3_on"] / v["c3_off"], bits))

    w("\n## 副指標（5 run 中央値 ms: encode／commit_wait／alloc_c／wall_sum）\n")
    w("| N | 腕 | encode | commit_wait | alloc_c | wall_sum |")
    w("|---|---|---|---|---|---|")
    for n in SIZES:
        for a in ("cur", "fi"):
            m = [median([k[1][(n, a)][f] for k in kerns]) for f in ("enc", "cw", "al", "wall")]
            w("| %d | %s | %.4f | %.4f | %.4f | %.4f |" % (n, a, m[0], m[1], m[2], m[3]))
    return "\n".join(out) + "\n"


# ---------------- self-test ----------------

def _fixture(d, ratio=0.9, official=True, drop_run=None, nonfinite=False, hash_mismatch=False,
             active=True, bit_fail=False, ab_mode="ok", soc="Apple M4", host_kind=True):
    """正常系 fixture を d に書く。ratio = fi の kernel_gpu / cur。"""
    def wr(name, text):
        with open(os.path.join(d, name), "w") as f:
            f.write(text)

    wr("rev-stamp.txt", "deadbeef\n")
    wr("env_info.txt", ("host_kind=macmini\n" if host_kind else "") + "Apple M4\n")
    wr("prereq_bitident.log", "test result: ok. 7 passed; 0 failed\n")
    wr("bitident_report.log", "".join(
        "BITIDENT n=%d pair=%s fallback=false ok=%s\n" % (
            n, p, "false" if (bit_fail and p == "fi_vs_cur") else "true") for n in SIZES for p in PAIRS)
        + "test result: ok. 1 passed\n")
    wr("parity.log", "".join("PARITY n=%d arm=%s fallback=false ok=true\n" % (n, a)
                             for n in SIZES for a in ("cur", "fi")) + "test result: ok. 1 passed\n")
    g = "build-done x\n"
    for lab in GATE_LABELS:
        g += "%s gate=%s load1=0.5\n" % (lab, "pass" if official or lab != "run3-trace" else "timeout")
    wr("load_gate.log", g + "series done\n")
    for i in RUNS:
        if drop_run == i:
            continue
        t = 'DEVICE gpu_cores=10 soc="%s" verified_tile_table=Some(M4Base(VerifiedM4BaseGpuCoreCount(10))) split_k_enabled=true\n' % soc
        for n in SIZES:
            t += ("TRACE n=%d should_split_k=none route=Classic/C1 cur=C1 resolved_cur=C1 "
                  "unroll_acc_eff=false classic=true active_on=%s\n" % (n, "true" if active else "false"))
        wr("trace_run%d.log" % i, t)
        k = ""
        for n in SIZES:
            for a in ARMS:
                h = 0x1111
                if a == "fi":
                    h = 0x2222 if (hash_mismatch and i == 2) else 0x1111
                k += ("ARM n=%d arm=%s role=%s gate=%s requested=R resolved=R fallback=false active=%s "
                      "checksum=1.0 hash=%016x\n" % (
                          n, a, "judge" if a in ("cur", "fi") else "record_only",
                          "on" if a == "fi" or a.endswith("_on") else "off",
                          "true" if (active and (a == "fi" or a.endswith("_on"))) else "false", h))
                base = (0.1 + 0.001 * i) * (n / 256.0)
                v = base * (ratio if a == "fi" else 1.0)
                vs = "nan" if (nonfinite and i == 1 and a == "cur" and n == 256) else "%.4f" % v
                k += ("MEAS n=%d arm=%s kernel_gpu_median_ms=%s q1=0.1 q3=0.2 encode_median_ms=0.01 "
                      "commit_wait_median_ms=0.2 alloc_c_median_ms=0.01 wall_sum_median_ms=%.4f\n" % (
                          n, a, vs, base * 1.5))
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
            with open(os.path.join(ab, "results-%s-3212-metal-gemm.jsonl" % arm), "w") as f:
                f.write("\n".join(lines) + "\n")
        with open(os.path.join(ab, "load-gate-3212-metal-3212.log"), "w") as f:
            f.write("threshold=3.0 rule_threshold=3.0 series=official\n")
            for r in RUNS:
                f.write("round%d gate=pass threshold=3.0 load1=0.5 waited_s=0\n" % r)
        with open(os.path.join(ab, "skipped-3212-metal-3212.log"), "w") as f:
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
            return aggregate(d, "deadbeef", with_baseline=False)

    def rejects(name, **kw):
        try:
            run(**kw)
        except AggregateError:
            check("fail-closed: " + name, True)
            return
        check("fail-closed: " + name + "（拒否されなかった）", False)

    check("正常系（10% 速い FI は ADOPT）", "候補 FI: ADOPT**" in run())
    check("帯内（+1%）は REJECT", "候補 FI: REJECT**" in run(ratio=1.01))
    md = run(ratio=1.2)
    check("2% 超かつ 5/5 遅いは REJECT（後退）", "候補 FI: REJECT**" in md and "| はい |" in md)
    check("正式系列でなければ undetermined", "候補 FI: undetermined**" in run(official=False))
    check("active=false は undetermined", "候補 FI: undetermined**" in run(active=False))
    check("bit 不一致は NOT_ADOPTABLE（ADOPT にならない）",
          "NOT_ADOPTABLE" in run(bit_fail=True) and "候補 FI: ADOPT" not in run(bit_fail=True))
    check("機体不一致（M4 Max）は参考系列で ADOPT にならない",
          "候補 FI: ADOPT" not in run(soc="Apple M4 Max"))
    check("セル A/B の checksum 不一致は FAIL", "候補 FI: FAIL" in run(ab_mode="ckfail"))
    check("セル A/B が無ければ undetermined", "候補 FI: undetermined" in run(ab_mode="none"))
    check("セル A/B の 5/5 round 後退は ADOPT にならない", "候補 FI: REJECT" in run(ab_mode="regress"))
    rejects("run 数不足", drop_run=4)
    rejects("非有限値", nonfinite=True)
    rejects("hash の run 間不一致", hash_mismatch=True)
    rejects("host_kind 欠落", host_kind=False)
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
            f.write("build-done x\nprereq-bitident gate=pass load1=0.5\nseries done\n")
        md = aggregate(d, "deadbeef", with_baseline=False)
        check("一部のみの load_gate.log は正式系列でない", "いいえ（参考系列）" in md and "候補 FI: undetermined" in md)
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
