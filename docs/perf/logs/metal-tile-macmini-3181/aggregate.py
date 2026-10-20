#!/usr/bin/env python3
"""イシュー #3181 の集計（標準ライブラリのみ。`python3 -I aggregate.py <dir> [--expect-rev SHA]`）。

規則の正は同ディレクトリの RULE.txt。入力は orchestrate_macmini.sh が出力した
trace_run{1..5}.log・cand_run{1..5}.log・candle_run{1..5}.jsonl・parity.log・
load_gate.log・rev-stamp.txt。出力（標準出力）は aggregate.md 相当の Markdown。

厳格パース方針（欠損は寛容に埋めず必ず例外 → exit 1）:
- run が 5 本揃わない・行数が足りない・欠損フィールド・非数値は拒否する
- TRACE／DEVICE が run 間で一致しない、同じ候補の hash が run 間で一致しない系列は拒否する
- load_gate.log に gate=pass 以外・FAILED を含む系列は拒否する
- `--expect-rev` 指定時は rev-stamp.txt と不一致なら拒否する
`--self-test` は埋め込み fixture（正常系・異常系）で上記を検証する。
"""
import json
import math
import os
import re
import statistics
import sys
import tempfile

SIZES = [256, 512, 1024, 2048, 4096]
NUM_CAND = 11
RUNS = [1, 2, 3, 4, 5]
SAME_TILE_BAND = 0.02  # RULE 14(b): 2% 以内は同等とみなす


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
    if v != v or v in (float("inf"), float("-inf")):
        fail("非有限値 %s=%r" % (what, s))
    return v


def parse_bool(s, what):
    if s not in ("true", "false"):
        fail("真偽値でない %s=%r" % (what, s))
    return s == "true"


DEVICE_RE = re.compile(
    r'DEVICE gpu_cores=(\S+) soc="([^"]*)" verified_m4_max=(\S+) split_k_enabled=(\S+)$')
TRACE_RE = re.compile(
    r"TRACE N=(\d+) select=(\S+) select_for_device=(\S+) should_split_k=(\S+) "
    r"route=(\S+) executed_route=(\S+) resolved=(\S+)$")
CAND_ID_RE = re.compile(
    r"CAND N=(\d+) idx=(\d+) requested=(\S+) resolved=(\S+) fallback=(\S+) selected=(\S+) "
    r"checksum=(\S+) hash=([0-9a-f]{16}) bit_identical_vs_selected=(\S+)$")
CAND_T_RE = re.compile(
    r"CAND N=(\d+) idx=(\d+) kernel_gpu_median_ms=(\S+) q1=(\S+) q3=(\S+) encode_median_ms=(\S+) "
    r"commit_wait_median_ms=(\S+) alloc_c_median_ms=(\S+) wall_sum_median_ms=(\S+)$")
PARITY_RE = re.compile(r"PARITY N=(\d+) cand=(\d+) fallback=(true|false)(?: resolved=(\S+)| ok=true)$")
GATE_RE = re.compile(r"^(\S+) gate=(\S+) load1=(\S+) waited_s=(\d+) ")


def parse_trace(path):
    dev = None
    rows = {}
    for line in read_text(path).splitlines():
        m = DEVICE_RE.search(line)
        if m:
            if dev is not None:
                fail("DEVICE 行が重複: %s" % os.path.basename(path))
            dev = m.groups()
            continue
        m = TRACE_RE.search(line)
        if m:
            n = int(m.group(1))
            if n in rows:
                fail("TRACE N=%d が重複: %s" % (n, os.path.basename(path)))
            rows[n] = m.groups()[1:]
    if dev is None:
        fail("DEVICE 行が無い: %s" % os.path.basename(path))
    if sorted(rows) != SIZES:
        fail("TRACE の N が揃っていない: %s (%s)" % (os.path.basename(path), sorted(rows)))
    return dev, rows


def parse_cand(path):
    ident, timing = {}, {}
    for line in read_text(path).splitlines():
        m = CAND_ID_RE.search(line)
        if m:
            g = m.groups()
            key = (int(g[0]), int(g[1]))
            if key in ident:
                fail("CAND 識別行が重複 %s: %s" % (key, os.path.basename(path)))
            num(g[6], "checksum")
            ident[key] = {
                "requested": g[2], "resolved": g[3],
                "fallback": parse_bool(g[4], "fallback"),
                "selected": parse_bool(g[5], "selected"),
                "hash": g[7], "bit_identical": parse_bool(g[8], "bit_identical_vs_selected"),
            }
            continue
        m = CAND_T_RE.search(line)
        if m:
            g = m.groups()
            key = (int(g[0]), int(g[1]))
            if key in timing:
                fail("CAND 時間行が重複 %s: %s" % (key, os.path.basename(path)))
            names = ["kernel_gpu", "q1", "q3", "encode", "commit_wait", "alloc_c", "wall_sum"]
            timing[key] = {nm: num(v, nm) for nm, v in zip(names, g[2:])}
            if timing[key]["kernel_gpu"] <= 0 or timing[key]["wall_sum"] <= 0:
                fail("時間が 0 以下 %s: %s" % (key, os.path.basename(path)))
    want = [(n, i) for n in SIZES for i in range(NUM_CAND)]
    if sorted(ident) != want:
        fail("CAND 識別行が %d 件でない: %s" % (len(want), os.path.basename(path)))
    if sorted(timing) != want:
        fail("CAND 時間行が %d 件でない: %s" % (len(want), os.path.basename(path)))
    return ident, timing


def parse_parity(path):
    text = read_text(path)
    if not re.search(r"^test result: ok\. 1 passed; 0 failed", text, re.M):
        fail("parity.log に成功（test result: ok. 1 passed）が無い")
    seen = {}
    for line in text.splitlines():
        m = PARITY_RE.search(line)
        if m:
            key = (int(m.group(1)), int(m.group(2)))
            if key in seen:
                fail("PARITY が重複 %s" % (key,))
            seen[key] = m.group(3) == "true"
    want = [(n, i) for n in SIZES for i in range(NUM_CAND)]
    if sorted(seen) != want:
        fail("PARITY 行が %d 件でない" % len(want))
    return seen


def parse_candle(path):
    out = {}
    version = None
    for ln, line in enumerate(read_text(path).splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            fail("JSON でない行 %s:%d" % (os.path.basename(path), ln))
        if rec.get("task") != "gemm_transfer_excl":
            continue
        if rec.get("device") != "metal":
            fail("device が metal でない %s:%d" % (os.path.basename(path), ln))
        n = rec.get("size")
        med = rec.get("median_s")
        if n in out:
            fail("candle excl が重複 N=%s: %s" % (n, os.path.basename(path)))
        if (not isinstance(med, (int, float)) or isinstance(med, bool)
                or not math.isfinite(med) or med <= 0):
            fail("median_s が不正（非数値・非有限・非正）N=%s: %s" % (n, os.path.basename(path)))
        ptot = rec.get("parity_total")
        pfail = rec.get("parity_fail_count")
        if (not isinstance(n, int) or isinstance(n, bool)
                or not isinstance(ptot, int) or isinstance(ptot, bool)
                or not isinstance(pfail, int) or isinstance(pfail, bool)
                or pfail < 0 or ptot != n * n):
            fail("parity_total / parity_fail_count が欠損・不正 N=%s: %s" % (n, os.path.basename(path)))
        # parity 不成立のセルは比較データとして判定不能（None）。比率集計から除外する。
        out[n] = float(med) if pfail == 0 else None
        version = rec.get("version")
    if sorted(out) != SIZES:
        fail("candle excl の N が揃っていない: %s (%s)" % (os.path.basename(path), sorted(out)))
    if not version:
        fail("candle の version が無い")
    return out, version


def parse_gate(path):
    text = read_text(path)
    labels = {}
    for line in text.splitlines():
        if "FAILED" in line:
            fail("load_gate.log に FAILED: %s" % line.strip())
        m = GATE_RE.match(line)
        if m:
            if m.group(2) != "pass":
                fail("GATE が pass でない: %s" % line.strip())
            labels[m.group(1)] = float(m.group(3)) if re.fullmatch(r"[0-9.]+", m.group(3)) else fail(
                "load1 が数値でない: %s" % line.strip())
    want = []
    for i in RUNS:
        want += ["run%d-trace" % i, "run%d-cand" % i] + ["run%d-candle-N%d" % (i, n) for n in SIZES]
    missing = [w for w in want if w not in labels]
    if missing:
        fail("gate 記録が無い: %s" % missing[:3])
    if "series done" not in text:
        fail("load_gate.log に 'series done' が無い（系列が完走していない）")
    return len(want)


def idx_of(label):
    head = label.split(":", 1)[0]
    return int(head) if head.isdigit() else None


def build(d, expect_rev=None):
    rev = read_text(os.path.join(d, "rev-stamp.txt")).strip()
    if not rev:
        fail("rev-stamp.txt が空")
    if expect_rev is not None and rev.split()[0] != expect_rev:
        fail("rev-stamp 不一致: %s != %s" % (rev.split()[0], expect_rev))
    gates = parse_gate(os.path.join(d, "load_gate.log"))
    parity = parse_parity(os.path.join(d, "parity.log"))

    traces = [parse_trace(os.path.join(d, "trace_run%d.log" % i)) for i in RUNS]
    for i, t in enumerate(traces[1:], 2):
        if t != traces[0]:
            fail("TRACE／DEVICE が run1 と run%d で一致しない" % i)
    dev, trace = traces[0]

    cands = [parse_cand(os.path.join(d, "cand_run%d.log" % i)) for i in RUNS]
    for i, (ident, _) in enumerate(cands[1:], 2):
        if ident != cands[0][0]:
            fail("候補の識別情報（hash・fallback 等）が run1 と run%d で一致しない" % i)
    ident = cands[0][0]
    for (n, idx), row in ident.items():
        if row["fallback"] != (row["requested"] != row["resolved"]):
            fail("fallback と requested/resolved が矛盾 N=%d idx=%d" % (n, idx))
        if parity[(n, idx)] != row["fallback"]:
            fail("parity.log と cand の fallback 判定が不一致 N=%d idx=%d" % (n, idx))
    for n in SIZES:
        sel = [i for i in range(NUM_CAND) if ident[(n, i)]["selected"]]
        want = idx_of(trace[n][1])
        if want is None:
            if sel:
                fail("選択タイルが CANDIDATES 外なのに selected が立っている N=%d" % n)
        elif sel != [want]:
            fail("trace の select_for_device と cand の selected が不一致 N=%d" % n)

    candles = [parse_candle(os.path.join(d, "candle_run%d.jsonl" % i)) for i in RUNS]
    versions = {v for _, v in candles}
    if len(versions) != 1:
        fail("candle の版が run 間で一致しない: %s" % sorted(versions))
    candle_version = versions.pop()

    return {
        "rev": rev, "gates": gates, "dev": dev, "trace": trace, "ident": ident,
        "timing": [t for _, t in cands], "candle": [c for c, _ in candles],
        "candle_version": candle_version,
    }


def med(vals):
    return statistics.median(vals)


def render(r):
    ident, timing, trace = r["ident"], r["timing"], r["trace"]
    out = []
    w = out.append
    w("# イシュー #3181 集計（Mac mini・タイル選択の追跡と全候補計測）\n")
    gc, soc, ver, sk = r["dev"]
    w("- rev-stamp: `%s`" % r["rev"].split()[0])
    w("- DEVICE: gpu_cores=%s soc=\"%s\" verified_m4_max=%s split_k_enabled=%s" % (gc, soc, ver, sk))
    w("- 負荷ゲート: %d/%d プロセスが pass（load_gate.log）" % (r["gates"], r["gates"]))
    w("- candle: version=%s・5 run・TRACE と hash は 5 run 全一致\n" % r["candle_version"])

    w("## (i) 選択の追跡（AC1・AC3）\n")
    w("| N | select(N,N,N) | select_for_device | should_split_k | executed_route | resolved | AC3 |")
    w("|---|---|---|---|---|---|---|")
    verdicts = {}
    for n in SIZES:
        sel, seld, split, route, exe, res = trace[n]
        ok = (idx_of(seld) == 3 and idx_of(res) == 3 and exe.startswith("Classic/"))
        verdicts[n] = ok
        w("| %d | %s | %s | %s | %s | %s | %s |" % (n, sel, seld, split, exe, res, "支持" if ok else "棄却"))
    w("\nAC3（`CANDIDATES[3]` 選択の推測）: %s\n" % (
        "全 N で支持" if all(verdicts.values()) else
        "棄却（支持された N: %s）" % [n for n in SIZES if verdicts[n]]))

    w("## (ii) 候補 x N の kernel_gpu／壁時計合計（5 run 中央値。run 別は [r1..r5]）\n")
    kmed, wmed = {}, {}
    for n in SIZES:
        w("### N=%d\n" % n)
        w("| idx | requested | fallback | kernel_gpu ms (中央値) | run 別 kernel_gpu | wall_sum ms (中央値) | run 別 wall_sum | hash |")
        w("|---|---|---|---|---|---|---|---|")
        for i in range(NUM_CAND):
            k = [t[(n, i)]["kernel_gpu"] for t in timing]
            wl = [t[(n, i)]["wall_sum"] for t in timing]
            kmed[(n, i)], wmed[(n, i)] = med(k), med(wl)
            mark = " (selected)" if ident[(n, i)]["selected"] else ""
            w("| %d%s | %s | %s | %.4f | %s | %.4f | %s | %s |" % (
                i, mark, ident[(n, i)]["requested"], ident[(n, i)]["fallback"], kmed[(n, i)],
                "[" + ", ".join("r%d=%.4f" % (j + 1, v) for j, v in enumerate(k)) + "]",
                wmed[(n, i)],
                "[" + ", ".join("r%d=%.4f" % (j + 1, v) for j, v in enumerate(wl)) + "]",
                ident[(n, i)]["hash"]))
        w("")

    w("## (iii) candle excl と比（速度比 = candle / fandhe。1 未満は fandhe が遅い）\n")
    w("- candle excl は `matmul + Device::synchronize()` のホスト壁時計（A・B は転送済み）。fandhe の kernel_gpu は GPU")
    w("  タイムスタンプで境界が異なる。主比は壁時計どうし（alloc_c+encode+commit_wait）、参考比は kernel_gpu で、")
    w("  参考比は candle 側に encode・同期が余計に入るため fandhe に有利に偏る。どちらも純カーネル比そのものではない。\n")
    w("| N | candle excl ms (5 run 中央値) | run 別 | 選択タイル idx | 主比 (excl / fandhe wall_sum) | 参考比 (excl / fandhe kernel_gpu) | 全候補での最大 参考比 (idx) |")
    w("|---|---|---|---|---|---|---|")
    for n in SIZES:
        raw = [x[n] for x in r["candle"]]
        sel = idx_of(trace[n][1])
        valid = [i for i in range(NUM_CAND) if not ident[(n, i)]["fallback"]]
        best_i = min(valid, key=lambda i: kmed[(n, i)])
        if any(v is None for v in raw):
            w("| %d | 判定不能（candle parity 不成立の run あり） | - | %s | n/a | n/a | n/a (%d) |" % (
                n, sel, best_i))
            continue
        c = [v * 1e3 for v in raw]
        cm = med(c)
        if sel is None:
            main_r = ref_r = "n/a"
        else:
            main_r = "%.3f" % (cm / wmed[(n, sel)])
            ref_r = "%.3f" % (cm / kmed[(n, sel)])
        w("| %d | %.4f | %s | %s | %s | %s | %.3f (%d) |" % (
            n, cm, "[" + ", ".join("r%d=%.4f" % (j + 1, v) for j, v in enumerate(c)) + "]",
            sel, main_r, ref_r, cm / kmed[(n, best_i)], best_i))
    w("")

    w("## (iv) N ごとの最良候補と分類（RULE 14(b)）\n")
    w("| N | 最良候補 (fallback 除く kernel_gpu 最小) | 最良 ms | 選択タイル ms | 最良/選択 | 5 run すべて選択より速い | 分類 |")
    w("|---|---|---|---|---|---|---|")
    for n in SIZES:
        valid = [i for i in range(NUM_CAND) if not ident[(n, i)]["fallback"]]
        best = min(valid, key=lambda i: kmed[(n, i)])
        sel = idx_of(trace[n][1])
        if sel is None or ident[(n, sel)]["fallback"]:
            w("| %d | %d | %.4f | n/a | n/a | n/a | unclassified |" % (n, best, kmed[(n, best)]))
            continue
        ratio = kmed[(n, best)] / kmed[(n, sel)]
        all_faster = all(t[(n, best)]["kernel_gpu"] < t[(n, sel)]["kernel_gpu"] for t in timing)
        if best != sel and ratio < 1.0 - SAME_TILE_BAND and all_faster:
            cls = "Mini 専用表の候補あり（#3206 へ）"
        else:
            cls = "現行選択と帯内（専用表は不要）"
        w("| %d | %d | %.4f | %.4f | %.3f | %s | %s |" % (
            n, best, kmed[(n, best)], kmed[(n, sel)], ratio, all_faster, cls))
    w("")
    return "\n".join(out) + "\n"


# ---- self-test -------------------------------------------------------------

def _fx_write(d, fn, text):
    with open(os.path.join(d, fn), "w", encoding="utf-8") as f:
        f.write(text)


def _fx_make(d):
    _fx_write(d, "rev-stamp.txt", "abc123 fixture\n")
    gate = []
    for i in RUNS:
        for lab in ["run%d-trace" % i, "run%d-cand" % i] + ["run%d-candle-N%d" % (i, n) for n in SIZES]:
            gate.append("%s gate=pass load1=1.0 waited_s=0 at=x" % lab)
    gate.append("series done")
    _fx_write(d, "load_gate.log", "\n".join(gate) + "\n")
    par = []
    for n in SIZES:
        for i in range(NUM_CAND):
            par.append("PARITY N=%d cand=%d fallback=false ok=true" % (n, i) if i != 10 else
                       "PARITY N=%d cand=%d fallback=true resolved=3:x" % (n, i))
    par.append("test result: ok. 1 passed; 0 failed; 0 ignored")
    _fx_write(d, "parity.log", "\n".join(par) + "\n")
    for r in RUNS:
        tr = ['test x ... DEVICE gpu_cores=10 soc="Apple M4" verified_m4_max=false split_k_enabled=true']
        for n in SIZES:
            tr.append("TRACE N=%d select=3:a select_for_device=3:a should_split_k=none "
                      "route=Classic/3:a executed_route=Classic/3:a resolved=3:a" % n)
        _fx_write(d, "trace_run%d.log" % r, "\n".join(tr) + "\n")
        cd = []
        for n in SIZES:
            for i in range(NUM_CAND):
                fb = i == 10
                cd.append("CAND N=%d idx=%d requested=%d:r resolved=%s fallback=%s selected=%s "
                          "checksum=1.000000 hash=%016x bit_identical_vs_selected=%s" % (
                              n, i, i, "3:x" if fb else "%d:r" % i, "true" if fb else "false",
                              "true" if i == 3 else "false", i + n, "true" if i == 3 else "false"))
            for i in range(NUM_CAND):
                k = 1.0 + 0.1 * abs(i - 3) + 0.001 * r + (0.0 if i == 3 else 0.0)
                cd.append("CAND N=%d idx=%d kernel_gpu_median_ms=%.4f q1=1 q3=2 encode_median_ms=0.1 "
                          "commit_wait_median_ms=1.0 alloc_c_median_ms=0.1 wall_sum_median_ms=%.4f" % (
                              n, i, k, k + 0.3))
        _fx_write(d, "cand_run%d.log" % r, "\n".join(cd) + "\n")
        cj = []
        for n in SIZES:
            cj.append(json.dumps({"task": "gemm_transfer_incl", "device": "metal", "size": n,
                                  "median_s": 0.5, "version": "0.11.0"}))
            cj.append(json.dumps({"task": "gemm_transfer_excl", "device": "metal", "size": n,
                                  "median_s": 0.002, "version": "0.11.0",
                                  "parity_total": n * n, "parity_fail_count": 0}))
        _fx_write(d, "candle_run%d.jsonl" % r, "\n".join(cj) + "\n")


def self_test():
    def expect_fail(label, mutate):
        with tempfile.TemporaryDirectory() as d:
            _fx_make(d)
            mutate(d)
            try:
                render(build(d))
            except AggregateError:
                return
            raise SystemExit("self-test 失敗: 異常系 %s が拒否されなかった" % label)

    with tempfile.TemporaryDirectory() as d:
        _fx_make(d)
        text = render(build(d, expect_rev="abc123"))
        assert "AC3" in text and "支持" in text and "unclassified" not in text
        assert "現行選択と帯内" in text

    def rm(fn):
        return lambda d: os.remove(os.path.join(d, fn))

    def edit(fn, old, new):
        def f(d):
            p = os.path.join(d, fn)
            s = open(p, encoding="utf-8").read()
            assert old in s, (fn, old)
            open(p, "w", encoding="utf-8").write(s.replace(old, new, 1))
        return f

    expect_fail("run 欠損", rm("cand_run5.log"))
    expect_fail("candle run 欠損", rm("candle_run3.jsonl"))
    expect_fail("hash 不一致", edit("cand_run2.log", "hash=%016x" % (256 + 0), "hash=%016x" % 999))
    expect_fail("TRACE 不一致", edit("trace_run4.log", "resolved=3:a", "resolved=5:a"))
    expect_fail("GATE 非 pass", edit("load_gate.log", "run2-cand gate=pass", "run2-cand gate=timeout"))
    expect_fail("FAILED 混入", edit("load_gate.log", "series done", "run1 FAILED x\nseries done"))
    expect_fail("非数値", edit("cand_run1.log", "kernel_gpu_median_ms=1.3010", "kernel_gpu_median_ms=abc"))
    expect_fail("parity 失敗", edit("parity.log", "test result: ok. 1 passed", "test result: FAILED. 0 passed"))
    expect_fail("candle median_s 非有限", edit("candle_run1.jsonl", '"median_s": 0.002', '"median_s": 1e999'))
    expect_fail("candle median_s NaN", edit("candle_run1.jsonl", '"median_s": 0.002', '"median_s": NaN'))
    expect_fail("candle parity 欠損", edit("candle_run1.jsonl", ', "parity_total": 65536', ''))
    expect_fail("candle parity 総数不一致", edit("candle_run1.jsonl", '"parity_total": 65536', '"parity_total": 1'))
    with tempfile.TemporaryDirectory() as d:
        _fx_make(d)
        edit("candle_run1.jsonl", '"parity_fail_count": 0', '"parity_fail_count": 3')(d)
        t = render(build(d, expect_rev="abc123"))
        assert "判定不能" in t
    expect_fail("行不足", edit("trace_run1.log", "TRACE N=4096", "TRACE M=4096"))
    with tempfile.TemporaryDirectory() as d:
        _fx_make(d)
        try:
            build(d, expect_rev="zzz")
        except AggregateError:
            pass
        else:
            raise SystemExit("self-test 失敗: rev 不一致が拒否されなかった")
    print("self-test OK")


def main(argv):
    if "--self-test" in argv:
        self_test()
        return 0
    args = [a for a in argv[1:] if not a.startswith("--")]
    expect = None
    if "--expect-rev" in argv:
        expect = argv[argv.index("--expect-rev") + 1]
        args = [a for a in args if a != expect]
    if len(args) != 1:
        print("usage: aggregate.py <dir> [--expect-rev SHA] | --self-test", file=sys.stderr)
        return 2
    try:
        sys.stdout.write(render(build(args[0], expect)))
    except AggregateError as e:
        print("AGGREGATE ERROR: %s" % e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
