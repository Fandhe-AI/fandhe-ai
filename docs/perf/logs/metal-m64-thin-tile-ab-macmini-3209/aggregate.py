#!/usr/bin/env python3
"""イシュー #3209 の集計（標準ライブラリのみ。`python3 -I aggregate.py <dir> [opts]`）。

規則の正は同ディレクトリの RULE.txt。入力は orchestrate_macmini.sh が出力した
trace_run{1..5}.log・kern_run{1..5}.log・parity.log・prereq_bitident.log・load_gate.log・rev-stamp.txt。
出力（標準出力）は aggregate.md 相当の Markdown。判定は RULE 項 10〜12 の機械判定そのもの。

厳格パース方針（欠損は寛容に埋めず必ず例外 → exit 1）:
- run が 5 本揃わない・行が足りない・欠損フィールド・非数値・非有限値は拒否する
- DEVICE／TRACE が run 間で一致しない、同じ腕の hash が run 間で一致しない系列は拒否する
- load_gate.log に全 gate 行（prereq-bitident・parity・各 run の trace／kern）が重複なく揃わない・gate=pass 以外・FAILED を含む・series done が無い系列は「正式系列でない」（判定は undetermined）
- `--expect-rev` 指定時は rev-stamp.txt と不一致なら拒否する
- `--cell-ab {ok,anomaly,fail,unavailable}` は train/infer セル A/B（RULE 項 13）の結果。既定は
  unavailable。ok 以外ではゲート判定を ADOPT にしない（fail は FAIL、anomaly／unavailable は
  undetermined へ落とす）
`--self-test` は一時ディレクトリへ生成した fixture（正常系・欠損・run 数不足・非有限値・hash 不一致・
正式系列でない）で上記を検証する。

判定の実装上の読み（RULE 項 11 の優先順）: 候補の総合は「必要セルに undetermined が 1 件でもあれば
undetermined」を最優先とし、次に ADOPT（ADOPT_CANDIDATE 1 件以上かつ NOT_ADOPTABLE 0 件）、
それ以外を REJECT とする（欠損セルがあるまま ADOPT と書かないための安全側の読み）。
"""
import math
import os
import re
import statistics
import sys
import tempfile

RUNS = [1, 2, 3, 4, 5]
BAND = 0.02  # RULE 項 10: 2% 以内は帯内
NN_SHAPES = ["64x256x784", "64x10x256", "784x64x256", "256x64x10", "784x256x64", "256x10x64"]
STRIDED_CELLS = [("NT", "784x64x256"), ("TN", "784x256x64")]
CELLS = [("NN", s) for s in NN_SHAPES] + STRIDED_CELLS
CANDS = ["T1", "T2", "T3", "T4"]
ARMS = ["cur"] + CANDS + ["prod"]
# RULE 項 5 の机上の前提: ゲート ON で thin 写像が非 none になる形状と期待タイル
EXPECT_THIN = {"64x256x784": "T1", "784x64x256": "T3"}
GATE_PAIRS = [("64x256x784", "T1"), ("784x64x256", "T3")]


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
    r"^TRACE shape=(\d+x\d+x\d+) eff=(\d+x\d+x\d+) should_split_k=(\S+) route_off=(\S+) "
    r"thin_on=(\S+) cur=(\S+) resolved_cur=(\S+)$")
CAND_HEAD_RE = re.compile(
    r"^CAND orient=(NN|NT|TN) shape=(\d+x\d+x\d+) eff=(\d+x\d+x\d+) arm=(cur|T[1-4]|prod) "
    r"role=(judge|record_only) requested=(\S+) resolved=(\S+) fallback=(true|false) "
    r"checksum=(\S+) hash=([0-9a-f]{16}) bit_identical_vs_cur=(true|false|na)$")
CAND_TIME_RE = re.compile(
    r"^CAND orient=(NN|NT|TN) shape=(\d+x\d+x\d+) eff=(\d+x\d+x\d+) arm=(cur|T[1-4]|prod) "
    r"kernel_gpu_median_ms=(\S+) q1=(\S+) q3=(\S+) encode_median_ms=(\S+) "
    r"commit_wait_median_ms=(\S+) alloc_c_median_ms=(\S+)$")
PARITY_RE = re.compile(
    r"^PARITY orient=(NN|NT|TN) shape=(\d+x\d+x\d+) eff=(\d+x\d+x\d+) arm=(cur|T[1-4]) "
    r"fallback=(true|false) ok=(true|skipped)$")
TOKENS = ("DEVICE ", "TRACE ", "CAND ", "PARITY ")


def lines_of(text):
    """cargo のテスト名前置（`test x ... `）を剥がし、診断行だけを返す。"""
    out = []
    for raw in text.splitlines():
        line = raw.strip()
        if "... " in line:
            head, tail = line.split("... ", 1)
            if tail.startswith(TOKENS):
                line = tail
        if line.startswith(TOKENS):
            out.append(line)
    return out


def parse_trace(path):
    dev = None
    traces = {}
    for line in lines_of(read_text(path)):
        m = DEVICE_RE.match(line)
        if m:
            if dev is not None:
                fail("DEVICE 行が重複: %s" % os.path.basename(path))
            dev = m.groups()
            continue
        m = TRACE_RE.match(line)
        if m:
            if m.group(1) in traces:
                fail("TRACE 行が重複 %s: %s" % (m.group(1), os.path.basename(path)))
            traces[m.group(1)] = m.groups()
    if dev is None:
        fail("DEVICE 行が無い: %s" % os.path.basename(path))
    if sorted(traces) != sorted(NN_SHAPES):
        fail("TRACE 行が 6 形状そろわない: %s" % os.path.basename(path))
    return dev, traces


def parse_kern(path):
    heads, times = {}, {}
    for line in lines_of(read_text(path)):
        m = CAND_HEAD_RE.match(line)
        if m:
            key = (m.group(1), m.group(2), m.group(4))
            if key in heads:
                fail("CAND 見出しが重複 %s: %s" % (key, os.path.basename(path)))
            num(m.group(9), "checksum")
            heads[key] = {
                "fallback": m.group(8) == "true", "hash": m.group(10),
                "role": m.group(5), "requested": m.group(6), "resolved": m.group(7),
            }
            continue
        m = CAND_TIME_RE.match(line)
        if m:
            key = (m.group(1), m.group(2), m.group(4))
            if key in times:
                fail("CAND 計測行が重複 %s: %s" % (key, os.path.basename(path)))
            vals = [num(m.group(i), "metric") for i in range(5, 11)]
            if vals[0] < 0:
                fail("kernel_gpu が負 %s" % (key,))
            times[key] = {"kg": vals[0], "q1": vals[1], "q3": vals[2],
                          "enc": vals[3], "cw": vals[4], "al": vals[5]}
    for o, s in CELLS:
        for a in ARMS:
            if (o, s, a) not in heads or (o, s, a) not in times:
                fail("CAND 欠損 %s %s %s: %s" % (o, s, a, os.path.basename(path)))
    return heads, times


def parse_parity(path):
    text = read_text(path)
    res = {}
    for line in lines_of(text):
        m = PARITY_RE.match(line)
        if m:
            res[(m.group(1), m.group(2), m.group(4))] = (m.group(5) == "true", m.group(6))
    for o, s in CELLS:
        for a in ["cur"] + CANDS:
            if (o, s, a) not in res:
                fail("PARITY 欠損 %s %s %s" % (o, s, a))
    if "test result: ok." not in text:
        fail("parity.log に test result: ok. が無い（前提ゲートの assert が落ちた可能性）")
    return res


GATE_LABELS = (["prereq-bitident", "parity"]
               + ["run%d-%s" % (i, k) for i in RUNS for k in ("trace", "kern")])


def parse_gate(path):
    """load_gate.log を厳格に読む（RULE 項 9）。

    正式系列の条件: 全プロセス（prereq-bitident・parity・各 run の trace／kern）の gate 行が
    重複なく揃い、すべて gate=pass で、FAILED を含まず、`series done` で終わっていること。
    欠損・重複・未知ラベルは正式系列でない（official=False。判定は undetermined）。
    """
    text = read_text(path)
    official = True
    reasons = []
    seen = {}
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


def check_prereq_bitident(path):
    """前提ゲート（RULE 項 6(a)）のログの存在と完了を検証する。欠損・未完了は例外（採用判定を出さない）。"""
    text = read_text(path)
    if "test result: ok." not in text:
        fail("prereq_bitident.log に test result: ok. が無い（前提ゲート未完了）")
    if re.search(r"test result: FAILED|\d+ failed", text) and not re.search(r"\b0 failed", text):
        fail("prereq_bitident.log に失敗がある")


def median(xs):
    return statistics.median(xs)


def judge_cell(kg_cur, kg_t, fallback, bit_ok, parity):
    """RULE 項 10。kg_* は 5 run の run 内中央値（ms）。戻り値: (status, ratio, faster_runs, slower_runs)。"""
    if len(kg_cur) != 5 or len(kg_t) != 5:
        return "undetermined", None, 0, 0
    mc, mt = median(kg_cur), median(kg_t)
    ratio = mt / mc - 1.0 if mc > 0 else None
    faster = sum(1 for c, t in zip(kg_cur, kg_t) if t < c)
    slower = sum(1 for c, t in zip(kg_cur, kg_t) if t > c)
    if ratio is None:
        return "undetermined", None, faster, slower
    if parity is None:
        return "undetermined", ratio, faster, slower
    if fallback or (not bit_ok) or (not parity):
        return "NOT_ADOPTABLE", ratio, faster, slower
    if mt < mc * (1 - BAND) and faster == 5:
        return "ADOPT_CANDIDATE", ratio, faster, slower
    return "REJECT", ratio, faster, slower


def candidate_overall(statuses):
    if any(s == "undetermined" for s in statuses):
        return "undetermined"
    if any(s == "ADOPT_CANDIDATE" for s in statuses) and not any(s == "NOT_ADOPTABLE" for s in statuses):
        return "ADOPT"
    return "REJECT"


def gate_verdict(pair_info, official, reach_ok, cell_ab):
    """RULE 項 12・13。pair_info: [(status, ratio, faster, slower)]（NN の 2 写像対）。"""
    if cell_ab == "fail":
        return "FAIL（セル A/B の checksum 不一致）"
    if not official:
        return "undetermined（正式系列でない）"
    if not reach_ok:
        return "undetermined（規則の前提不成立: 到達追跡が机上の前提と食い違う）"
    if any(s in ("undetermined", "NOT_ADOPTABLE") for s, _, _, _ in pair_info):
        return "undetermined"
    slow = [(r is not None and r > BAND and sl == 5) for _, r, _, sl in pair_info]
    if any(slow):
        return "REJECT"
    in_band_or_adopt = all(s == "ADOPT_CANDIDATE" or (r is not None and abs(r) <= BAND)
                           for s, r, _, _ in pair_info)
    if in_band_or_adopt and any(s == "ADOPT_CANDIDATE" for s, _, _, _ in pair_info):
        if cell_ab != "ok":
            return "undetermined（セル A/B が %s のため ADOPT にしない）" % cell_ab
        return "ADOPT"
    return "undetermined"


def aggregate(d, expect_rev=None, cell_ab="unavailable"):
    rev = read_text(os.path.join(d, "rev-stamp.txt")).strip()
    if expect_rev is not None and rev != expect_rev:
        fail("rev-stamp 不一致: %r != %r" % (rev, expect_rev))
    check_prereq_bitident(os.path.join(d, "prereq_bitident.log"))
    official, gate_reasons = parse_gate(os.path.join(d, "load_gate.log"))
    parity = parse_parity(os.path.join(d, "parity.log"))
    traces, kerns = [], []
    for i in RUNS:
        traces.append(parse_trace(os.path.join(d, "trace_run%d.log" % i)))
        kerns.append(parse_kern(os.path.join(d, "kern_run%d.log" % i)))
    for t in traces[1:]:
        if t != traces[0]:
            fail("DEVICE／TRACE が run 間で一致しない")
    for key in kerns[0][0]:
        hs = {k[0][key]["hash"] for k in kerns}
        if len(hs) != 1:
            fail("hash が run 間で一致しない %s" % (key,))
        for field in ("fallback", "requested", "resolved"):
            if len({k[0][key][field] for k in kerns}) != 1:
                fail("%s が run 間で一致しない %s" % (field, key))
    dev, trace = traces[0]
    heads0 = kerns[0][0]

    # 到達前提（RULE 項 5）
    reach_ok = True
    for s in NN_SHAPES:
        thin = trace[s][4]
        want = EXPECT_THIN.get(s)
        if want is None:
            reach_ok = reach_ok and thin == "none"
        else:
            reach_ok = reach_ok and thin.startswith(want + ":")

    out = []
    out.append("# #3209 集計（m=64 細タイル候補 Mac mini A/B）\n")
    out.append("- rev: `%s`" % rev)
    out.append("- DEVICE: gpu_cores=%s soc=\"%s\" verified_tile_table=%s split_k_enabled=%s" % tuple(dev))
    out.append("- 正式系列（load1 ゲート全通過）: %s" % ("はい" if official else "いいえ（参考系列）"))
    for r in gate_reasons:
        out.append("  - %s" % r)
    out.append("- 到達前提（ゲート ON で変わるのは (64,256,784)→T1 と (784,64,256)→T3 の 2 件のみ）: %s\n"
               % ("成立" if reach_ok else "**不成立**"))

    out.append("## 到達追跡（5 run 全一致）\n")
    out.append("| 形状（eff） | should_split_k | route（ゲート OFF） | thin（ゲート ON） | cur |")
    out.append("|---|---|---|---|---|")
    for s in NN_SHAPES:
        t = trace[s]
        out.append("| %s（%s） | %s | %s | %s | %s |" % (s, t[1], t[2], t[3], t[4], t[5]))

    out.append("\n## 純カーネル時間（kernel_gpu の run 内中央値 ms。run 1〜5・5 run 中央値・cur 比）\n")
    out.append("| 向き | 形状 | 腕 | run1 | run2 | run3 | run4 | run5 | 中央値 | 比 | 判定 |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    cand_statuses = {c: [] for c in CANDS}
    cell_status = {}
    for o, s in CELLS:
        kg = {a: [k[1][(o, s, a)]["kg"] for k in kerns] for a in ARMS}
        for a in ARMS:
            ratio_txt, verdict = "-", "record_only" if a == "prod" else "基準"
            if a in CANDS:
                h = heads0[(o, s, a)]
                bit_ok = h["hash"] == heads0[(o, s, "cur")]["hash"]
                pr = parity[(o, s, a)]
                # parity は前提ゲートで assert 済み。skipped は fallback（pr[0]）として扱い、
                # 欠損は parse_parity が拒否するため、ここでの parity 引数は常に True。
                st, ratio, fa, sl = judge_cell(kg["cur"], kg[a], h["fallback"] or pr[0], bit_ok, True)
                if not official:
                    st = "undetermined"
                cand_statuses[a].append(st)
                cell_status[(o, s, a)] = (st, ratio, fa, sl)
                ratio_txt = "%+.1f%%" % (ratio * 100) if ratio is not None else "-"
                verdict = "%s（速い %d/5・遅い %d/5%s%s）" % (
                    st, fa, sl, "・fallback" if h["fallback"] else "", "" if bit_ok else "・bit 不一致")
            out.append("| %s | %s | %s | %s | %.4f | %s | %s |" % (
                o, s, a, " | ".join("%.4f" % v for v in kg[a]), median(kg[a]), ratio_txt, verdict))

    out.append("\n## 候補ごとの総合（RULE 項 11）\n")
    out.append("| 候補 | 総合 |")
    out.append("|---|---|")
    for c in CANDS:
        out.append("| %s | %s |" % (c, candidate_overall(cand_statuses[c])))

    out.append("\n## ゲート（M64_THIN_TILE_ENABLED の写像。RULE 項 12）\n")
    pair_info = [cell_status[("NN", s, c)] for s, c in GATE_PAIRS]
    verdict = gate_verdict(pair_info, official, reach_ok, cell_ab)
    for (s, c), info in zip(GATE_PAIRS, pair_info):
        out.append("- (%s, %s): %s、比 %s" % (s, c, info[0], "%+.1f%%" % (info[1] * 100) if info[1] is not None else "-"))
    out.append("- セル A/B（A/A 対照）の結果: %s" % cell_ab)
    out.append("- **ゲート判定: %s**" % verdict)
    out.append("- これは Mac mini 単機の判定。写像は機体非依存のため、結線の前に M4 Max での確認が要る。")

    out.append("\n## 副指標（5 run 中央値 ms: encode／commit_wait／alloc_c）\n")
    out.append("| 向き | 形状 | 腕 | encode | commit_wait | alloc_c |")
    out.append("|---|---|---|---|---|---|")
    for o, s in CELLS:
        for a in ARMS:
            e = median([k[1][(o, s, a)]["enc"] for k in kerns])
            c = median([k[1][(o, s, a)]["cw"] for k in kerns])
            al = median([k[1][(o, s, a)]["al"] for k in kerns])
            out.append("| %s | %s | %s | %.4f | %.4f | %.4f |" % (o, s, a, e, c, al))
    return "\n".join(out) + "\n"


# ---------------- self-test ----------------

def _fixture(d, kg_t=None, official=True, drop_run=None, nonfinite=False, hash_mismatch=False,
             thin_wrong=False):
    """正常系 fixture を d に書く。kg_t は候補の kernel_gpu 倍率（cur 比。既定 0.9）。"""
    kg_t = kg_t if kg_t is not None else 0.9
    with open(os.path.join(d, "rev-stamp.txt"), "w") as f:
        f.write("deadbeef\n")
    with open(os.path.join(d, "prereq_bitident.log"), "w") as f:
        f.write("test result: ok. 2 passed; 0 failed\n")
    with open(os.path.join(d, "load_gate.log"), "w") as f:
        f.write("build-done x\n")
        for lab in GATE_LABELS:
            f.write("%s gate=%s load1=0.5\n" % (lab, "pass" if official or lab != "run3-trace" else "timeout"))
        f.write("series done\n")
    with open(os.path.join(d, "parity.log"), "w") as f:
        for o, s in CELLS:
            for a in ["cur"] + CANDS:
                f.write("PARITY orient=%s shape=%s eff=%s arm=%s fallback=false ok=true\n" % (o, s, s, a))
        f.write("test result: ok. 1 passed\n")
    for i in RUNS:
        if drop_run == i:
            continue
        with open(os.path.join(d, "trace_run%d.log" % i), "w") as f:
            f.write('DEVICE gpu_cores=10 soc="Apple M4" verified_tile_table=Some(M4Base) split_k_enabled=true\n')
            for s in NN_SHAPES:
                thin = "none"
                if s in EXPECT_THIN and not thin_wrong:
                    thin = EXPECT_THIN[s] + ":bm16"
                f.write("TRACE shape=%s eff=%s should_split_k=none route_off=Classic/C1 thin_on=%s cur=C1 resolved_cur=C1\n"
                        % (s, s, thin))
        with open(os.path.join(d, "kern_run%d.log" % i), "w") as f:
            for o, s in CELLS:
                for a in ARMS:
                    h = "%016x" % 0x1111
                    if hash_mismatch and i == 2 and a == "T1" and s == "784x64x256" and o == "NN":
                        h = "%016x" % 0x2222
                    role = "record_only" if a == "prod" else "judge"
                    bit = "na" if a == "prod" else "true"
                    f.write("CAND orient=%s shape=%s eff=%s arm=%s role=%s requested=R resolved=R fallback=false "
                            "checksum=1.0 hash=%s bit_identical_vs_cur=%s\n" % (o, s, s, a, role, h, bit))
                    base = 0.1 + 0.001 * i
                    v = base * (kg_t if a in CANDS else 1.0)
                    vs = "nan" if (nonfinite and i == 1 and a == "cur" and s == "64x256x784" and o == "NN") else "%.4f" % v
                    f.write("CAND orient=%s shape=%s eff=%s arm=%s kernel_gpu_median_ms=%s q1=0.1 q3=0.2 "
                            "encode_median_ms=0.01 commit_wait_median_ms=0.2 alloc_c_median_ms=0.01\n" % (o, s, s, a, vs))


def self_test():
    ok = True

    def expect_fail(name, **kw):
        nonlocal ok
        with tempfile.TemporaryDirectory() as d:
            _fixture(d, **kw)
            try:
                aggregate(d, "deadbeef")
            except AggregateError:
                print("PASS fail-closed: %s" % name)
                return
            print("FAIL: %s が拒否されなかった" % name)
            ok = False

    with tempfile.TemporaryDirectory() as d:
        _fixture(d)
        md = aggregate(d, "deadbeef", cell_ab="ok")
        if "ゲート判定: ADOPT**" in md and "| T1 | ADOPT |" in md:
            print("PASS 正常系（10% 速い候補は ADOPT）")
        else:
            print("FAIL 正常系の判定が想定と異なる"); ok = False
        md2 = aggregate(d, "deadbeef")
        if "undetermined（セル A/B が unavailable" in md2:
            print("PASS セル A/B 不明ならゲートは ADOPT にならない")
        else:
            print("FAIL セル A/B 不明の扱い"); ok = False
    with tempfile.TemporaryDirectory() as d:
        _fixture(d, kg_t=1.01)
        md = aggregate(d, "deadbeef", cell_ab="ok")
        if "| T1 | REJECT |" in md:
            print("PASS 帯内（+1%）は REJECT")
        else:
            print("FAIL 帯内の判定"); ok = False
    with tempfile.TemporaryDirectory() as d:
        _fixture(d, kg_t=1.2)
        md = aggregate(d, "deadbeef", cell_ab="ok")
        if "**ゲート判定: REJECT**" in md:
            print("PASS 2% 超かつ 5/5 遅いゲートは REJECT")
        else:
            print("FAIL ゲート REJECT"); ok = False
    with tempfile.TemporaryDirectory() as d:
        _fixture(d, official=False)
        md = aggregate(d, "deadbeef", cell_ab="ok")
        if "正式系列（load1 ゲート全通過）: いいえ" in md and "| T1 | undetermined |" in md:
            print("PASS 正式系列でなければ undetermined")
        else:
            print("FAIL 正式系列判定"); ok = False
    with tempfile.TemporaryDirectory() as d:
        _fixture(d, thin_wrong=True)
        md = aggregate(d, "deadbeef", cell_ab="ok")
        if "規則の前提不成立" in md:
            print("PASS 到達前提の食い違いはゲート undetermined")
        else:
            print("FAIL 到達前提の判定"); ok = False
    with tempfile.TemporaryDirectory() as d:
        _fixture(d)
        try:
            aggregate(d, "other")
            print("FAIL rev 不一致が拒否されなかった"); ok = False
        except AggregateError:
            print("PASS fail-closed: rev 不一致")
    expect_fail("run 数不足", drop_run=4)
    expect_fail("非有限値", nonfinite=True)
    expect_fail("hash の run 間不一致", hash_mismatch=True)
    with tempfile.TemporaryDirectory() as d:
        _fixture(d)
        os.remove(os.path.join(d, "parity.log"))
        try:
            aggregate(d, "deadbeef")
            print("FAIL parity 欠損が拒否されなかった"); ok = False
        except AggregateError:
            print("PASS fail-closed: parity.log 欠損")
    # 空・一部のみの load_gate.log は正式系列にならない
    for name, body in (("空", ""), ("一部のみ", "build-done x\nprereq-bitident gate=pass load1=0.5\nseries done\n")):
        with tempfile.TemporaryDirectory() as d:
            _fixture(d)
            with open(os.path.join(d, "load_gate.log"), "w") as f:
                f.write(body)
            md = aggregate(d, "deadbeef", cell_ab="ok")
            if "正式系列（load1 ゲート全通過）: いいえ" in md and "ゲート判定: ADOPT" not in md:
                print("PASS load_gate.log %s は正式系列でない" % name)
            else:
                print("FAIL load_gate.log %s" % name); ok = False
    with tempfile.TemporaryDirectory() as d:
        _fixture(d)
        os.remove(os.path.join(d, "prereq_bitident.log"))
        try:
            aggregate(d, "deadbeef", cell_ab="ok")
            print("FAIL prereq_bitident.log 欠損が拒否されなかった"); ok = False
        except AggregateError:
            print("PASS fail-closed: prereq_bitident.log 欠損")
    with tempfile.TemporaryDirectory() as d:
        _fixture(d, official=False)
        md = aggregate(d, "deadbeef", cell_ab="fail")
        if "ゲート判定: FAIL" in md:
            print("PASS 参考系列でも cell_ab=fail は FAIL")
        else:
            print("FAIL cell_ab=fail の優先"); ok = False
    return 0 if ok else 1


def main(argv):
    if "--self-test" in argv:
        return self_test()
    args = [a for a in argv[1:] if not a.startswith("--")]
    expect = None
    cell_ab = "unavailable"
    if "--expect-rev" in argv:
        expect = argv[argv.index("--expect-rev") + 1]
        args = [a for a in args if a != expect]
    if "--cell-ab" in argv:
        cell_ab = argv[argv.index("--cell-ab") + 1]
        args = [a for a in args if a != cell_ab]
        if cell_ab not in ("ok", "anomaly", "fail", "unavailable"):
            print("--cell-ab は ok|anomaly|fail|unavailable", file=sys.stderr)
            return 2
    if len(args) != 1:
        print("usage: aggregate.py <dir> [--expect-rev SHA] [--cell-ab ok|anomaly|fail|unavailable] | --self-test",
              file=sys.stderr)
        return 2
    try:
        sys.stdout.write(aggregate(args[0], expect, cell_ab))
    except AggregateError as e:
        print("AGGREGATE ERROR: %s" % e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
