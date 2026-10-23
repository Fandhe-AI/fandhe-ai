#!/usr/bin/env python3
"""イシュー #3206 の集計（標準ライブラリのみ。`python3 -I aggregate.py <dir> [--expect-rev SHA]`）。

規則の正は同ディレクトリの RULE.txt。入力は orchestrate_macmini.sh が出力した
select_run{1..5}.log・bit_run{1..5}.log・ab_run{1..5}.log・candle_run{1..5}.jsonl・precheck.log・
bitgate.log・load_gate.log・rev-stamp.txt・candle_tree.txt。出力（標準出力）は aggregate.md 相当の Markdown。

厳格パース方針（欠損は寛容に埋めず必ず例外 → exit 1）:
- run が 5 本揃わない・行数が足りない・欠損フィールド・非数値・非有限は拒否する
- 各 run の 3 ログすべてに MINI_TABLE_ACTIVE と DEVICE（gpu_cores=10 soc="Apple M4" token=M4Base）を必須にし、
  `SKIP not-m4-base` を含むログは拒否する（偽 pass の防止）
- SELECT は期待 idx・fallback=false・ok=true で run 間一致、BITMATCH は bit_identical=true・parity_ok=true で
  hash が run 間・前提ゲートと一致しない系列は拒否する
- candle は run・N ごとに gemm_transfer_excl がちょうど 1 行、parity_fail_count は非負整数必須、版は
  candle_tree.txt と一致すること
- load_gate.log に gate=pass 以外・FAILED を含む、または `series done` が無い系列は「正式系列でない」と
  明示する（参考集計は出す）。`--expect-rev` が rev-stamp と不一致なら拒否する
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
RUNS = [1, 2, 3, 4, 5]
EXPECT_IDX = {256: 3, 512: 1, 1024: 1, 2048: 2, 4096: 2}
ARMS = ["before", "after"]
IMPROVE_BELOW = 0.98
REGRESS_ABOVE = 1.02
DEVICE_LINE = 'DEVICE gpu_cores=10 soc="Apple M4" token=M4Base'


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


def parse_bool(s, what):
    if s not in ("true", "false"):
        fail("真偽値でない %s=%r" % (what, s))
    return s == "true"


SELECT_RE = re.compile(r"SELECT N=(\d+) idx=(\d+) fallback=(\S+) ok=(\S+)$")
BIT_RE = re.compile(
    r"BITMATCH N=(\d+) bit_identical=(\S+) hash_after=([0-9a-f]{16}) hash_before=([0-9a-f]{16}) "
    r"parity_ok=(\S+)$")
AB_RE = re.compile(r"AB N=(\d+) arm=(before|after) kernel_gpu_median_ms=(\S+) wall_sum_median_ms=(\S+)$")
GATE_RE = re.compile(r"^(\S+) gate=(\S+) load1=(\S+) waited_s=(\d+) ")


def check_common(text, name):
    """3 ログ共通: 実機（M4Base）で動いた証拠。SKIP・非アクティブを拒否する。"""
    if "SKIP" in text:
        fail("SKIP を含むログ（Mini で実行されていない）: %s" % name)
    if "MINI_TABLE_ACTIVE" not in text:
        fail("MINI_TABLE_ACTIVE が無い: %s" % name)
    n_dev = sum(1 for ln in text.splitlines() if ln.endswith(DEVICE_LINE) or (" " + DEVICE_LINE) in ln)
    if n_dev != 1:
        fail("DEVICE 行が 1 件でない（%d）: %s" % (n_dev, name))
    if not re.search(r"^test result: ok\. 1 passed; 0 failed", text, re.M):
        fail("成功（test result: ok. 1 passed）が無い: %s" % name)


def parse_select(path):
    text = read_text(path)
    name = os.path.basename(path)
    check_common(text, name)
    rows = {}
    for line in text.splitlines():
        m = SELECT_RE.search(line)
        if m:
            n = int(m.group(1))
            if n in rows:
                fail("SELECT N=%d が重複: %s" % (n, name))
            idx = int(m.group(2))
            fb = parse_bool(m.group(3), "fallback")
            ok = parse_bool(m.group(4), "ok")
            if fb or not ok:
                fail("SELECT が fallback／ok=false N=%d: %s" % (n, name))
            if idx != EXPECT_IDX[n]:
                fail("SELECT の idx が期待と不一致 N=%d idx=%d: %s" % (n, idx, name))
            rows[n] = idx
    if sorted(rows) != SIZES:
        fail("SELECT の N が揃っていない: %s (%s)" % (name, sorted(rows)))
    return rows


def parse_bit(path):
    text = read_text(path)
    name = os.path.basename(path)
    check_common(text, name)
    rows = {}
    for line in text.splitlines():
        m = BIT_RE.search(line)
        if m:
            n = int(m.group(1))
            if n in rows:
                fail("BITMATCH N=%d が重複: %s" % (n, name))
            if not parse_bool(m.group(2), "bit_identical") or not parse_bool(m.group(5), "parity_ok"):
                fail("BITMATCH が bit_identical／parity_ok でない N=%d: %s" % (n, name))
            if m.group(3) != m.group(4):
                fail("hash_after と hash_before が不一致 N=%d: %s" % (n, name))
            rows[n] = m.group(3)
    if sorted(rows) != SIZES:
        fail("BITMATCH の N が揃っていない: %s (%s)" % (name, sorted(rows)))
    return rows


def parse_ab(path):
    text = read_text(path)
    name = os.path.basename(path)
    check_common(text, name)
    rows = {}
    for line in text.splitlines():
        m = AB_RE.search(line)
        if m:
            key = (int(m.group(1)), m.group(2))
            if key in rows:
                fail("AB %s が重複: %s" % (key, name))
            k, w = num(m.group(3), "kernel_gpu"), num(m.group(4), "wall_sum")
            if k <= 0 or w <= 0:
                fail("時間が 0 以下 %s: %s" % (key, name))
            rows[key] = {"kernel": k, "wall": w}
    want = [(n, a) for n in SIZES for a in ARMS]
    if sorted(rows) != sorted(want):
        fail("AB の N×腕が揃っていない（ちょうど 1 行ずつ）: %s" % name)
    return rows


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
        v = rec.get("version")
        if not v or (version is not None and v != version):
            fail("candle の version が欠損・不一致: %s" % os.path.basename(path))
        version = v
        # parity 不成立のセルは比較データとして判定不能（None）。比率集計から除外する。
        out[n] = float(med) if pfail == 0 else None
    if sorted(out) != SIZES:
        fail("candle excl の N が揃っていない: %s (%s)" % (os.path.basename(path), sorted(out)))
    return out, version


def parse_gate(path):
    """戻り値: (ラベル数, 正式系列でない理由のリスト)。必須ラベルの欠落は例外。"""
    text = read_text(path)
    labels = {}
    reasons = []
    for line in text.splitlines():
        if "FAILED" in line:
            reasons.append("FAILED 行: %s" % line.strip())
        m = GATE_RE.match(line)
        if m:
            if m.group(2) != "pass":
                reasons.append("gate=%s: %s" % (m.group(2), line.strip()))
            if not re.fullmatch(r"[0-9.]+", m.group(3)) and m.group(2) == "pass":
                fail("load1 が数値でない: %s" % line.strip())
            labels[m.group(1)] = m.group(2)
    want = ["precheck", "bitgate"]
    for i in RUNS:
        want += ["run%d-select" % i, "run%d-bit" % i, "run%d-ab" % i]
        want += ["run%d-candle-N%d" % (i, n) for n in SIZES]
    missing = [w for w in want if w not in labels]
    if missing:
        fail("gate 記録が無い: %s" % missing[:3])
    if "series done" not in text:
        reasons.append("'series done' が無い（系列が完走していない）")
    return len(want), reasons


def build(d, expect_rev=None):
    rev = read_text(os.path.join(d, "rev-stamp.txt")).strip()
    if not rev:
        fail("rev-stamp.txt が空")
    if expect_rev is not None and rev.split()[0] != expect_rev:
        fail("rev-stamp 不一致: %s != %s" % (rev.split()[0], expect_rev))
    ngate, reasons = parse_gate(os.path.join(d, "load_gate.log"))

    pre = read_text(os.path.join(d, "precheck.log"))
    if not re.search(r"^test result: ok\. [1-9][0-9]* passed; 0 failed", pre, re.M):
        fail("precheck.log に成功（tile:: 単体テスト）が無い")
    gate_bit = parse_bit(os.path.join(d, "bitgate.log"))

    sel = [parse_select(os.path.join(d, "select_run%d.log" % i)) for i in RUNS]
    bit = [parse_bit(os.path.join(d, "bit_run%d.log" % i)) for i in RUNS]
    for i in range(1, 5):
        if sel[i] != sel[0]:
            fail("SELECT が run1 と run%d で一致しない" % (i + 1))
    for i in range(5):
        if bit[i] != gate_bit:
            fail("BITMATCH の hash が前提ゲートと run%d で一致しない" % (i + 1))
    ab = [parse_ab(os.path.join(d, "ab_run%d.log" % i)) for i in RUNS]

    tree_ver = read_text(os.path.join(d, "candle_tree.txt")).strip()
    m = re.fullmatch(r"[^\w]*candle-core v(\S+)", tree_ver)
    if not m:
        fail("candle_tree.txt を解釈できない: %r" % tree_ver)
    candles = [parse_candle(os.path.join(d, "candle_run%d.jsonl" % i)) for i in RUNS]
    for _, v in candles:
        if v != m.group(1):
            fail("candle の version が candle_tree.txt と不一致: %s != %s" % (v, m.group(1)))
    return {
        "rev": rev, "ngate": ngate, "reasons": reasons, "select": sel[0], "hash": gate_bit,
        "ab": ab, "candle": [c for c, _ in candles], "candle_version": m.group(1),
    }


def med(vals):
    return statistics.median(vals)


def classify(per_run_ratio_before_after):
    """per_run: [(before, after), ...]。RULE 13 の分類（記述上のもの）。"""
    ratio = med([a for _, a in per_run_ratio_before_after]) / med([b for b, _ in per_run_ratio_before_after])
    faster = sum(1 for b, a in per_run_ratio_before_after if a < b)
    slower = sum(1 for b, a in per_run_ratio_before_after if a > b)
    if ratio < IMPROVE_BELOW and faster == len(per_run_ratio_before_after):
        return ratio, "改善"
    if ratio > REGRESS_ABOVE and slower >= 3:
        return ratio, "退行"
    return ratio, "帯内"


def render(r):
    ab = r["ab"]
    out = []
    w = out.append
    formal = not r["reasons"]
    w("# イシュー #3206 集計（Mac mini・タイル表の追加前後と candle 比）\n")
    w("- 系列の位置づけ: **%s**" % ("正式系列（5/5 run 完備・全ゲート pass・series done）" if formal
                                 else "正式系列ではない（参考集計。AC4 は未達）"))
    for rs in r["reasons"]:
        w("  - 理由: %s" % rs)
    w("- rev-stamp: `%s`" % r["rev"].split()[0])
    w("- DEVICE: gpu_cores=10 soc=\"Apple M4\" token=M4Base（全 run の select／bit／ab ログで確認）")
    w("- 負荷ゲート: %d プロセス分の記録（load_gate.log）" % r["ngate"])
    w("- candle: version=%s・5 run" % r["candle_version"])
    w("- 整合検査: SELECT は 5 run 全一致、BITMATCH の hash は前提ゲートと 5 run で全一致（bit 一致・parity ok）\n")

    w("## (i) 本番経路の選択（AC1）と bit 一致（AC3）\n")
    w("| N | 選択 idx | fallback | bit 一致 hash（FNV-1a 64bit。before=after） |")
    w("|---|---|---|---|")
    for n in SIZES:
        w("| %d | %d | false | %s |" % (n, r["select"][n], r["hash"][n]))
    w("")

    kmed, wmed = {}, {}
    w("## (ii) 腕別の時間（5 run 中央値。run 別は [r1..r5]）\n")
    w("| N | 腕 | kernel_gpu ms (中央値) | run 別 kernel_gpu | wall_sum ms (中央値) | run 別 wall_sum |")
    w("|---|---|---|---|---|---|")
    for n in SIZES:
        for a in ARMS:
            k = [x[(n, a)]["kernel"] for x in ab]
            wl = [x[(n, a)]["wall"] for x in ab]
            kmed[(n, a)], wmed[(n, a)] = med(k), med(wl)
            w("| %d | %s | %.4f | %s | %.4f | %s |" % (
                n, a, kmed[(n, a)],
                "[" + ", ".join("r%d=%.4f" % (j + 1, v) for j, v in enumerate(k)) + "]",
                wmed[(n, a)],
                "[" + ", ".join("r%d=%.4f" % (j + 1, v) for j, v in enumerate(wl)) + "]"))
    w("")

    w("## (iii) after/before と分類（RULE 13。改善: 0.98 未満かつ 5 run すべて after が速い／"
      "退行: 1.02 超かつ 3/5 run 以上で after が遅い）\n")
    w("| N | kernel_gpu after/before | 分類 | wall_sum after/before | 分類 |")
    w("|---|---|---|---|---|")
    regress = []
    for n in SIZES:
        kr, kc = classify([(x[(n, "before")]["kernel"], x[(n, "after")]["kernel"]) for x in ab])
        wr, wc = classify([(x[(n, "before")]["wall"], x[(n, "after")]["wall"]) for x in ab])
        if kc == "退行" or wc == "退行":
            regress.append(n)
        w("| %d | %.3f | %s | %.3f | %s |" % (n, kr, kc, wr, wc))
    w("\n退行: %s\n" % ("なし（表を維持）" if not regress else "N=%s（記録して所有者へ上げる。表は変えない）" % regress))

    w("## (iv) candle excl と比（速度比 = candle / fandhe。1 未満は fandhe が遅い）\n")
    w("- candle excl は `matmul + Device::synchronize()` のホスト壁時計（A・B は転送済み）。fandhe の kernel_gpu は GPU")
    w("  タイムスタンプで境界が異なる。主比は壁時計どうし（alloc_c+encode+commit_wait）、参考比は kernel_gpu で、")
    w("  参考比は candle 側に encode・同期が余計に入るため fandhe に有利に偏る。どちらも純カーネル比そのものではない。\n")
    w("| N | candle excl ms (5 run 中央値) | run 別 | 主比 before | 主比 after | 参考比 before | 参考比 after |")
    w("|---|---|---|---|---|---|---|")
    for n in SIZES:
        raw = [x[n] for x in r["candle"]]
        if any(v is None for v in raw):
            w("| %d | 判定不能（candle parity 不成立の run あり） | - | n/a | n/a | n/a | n/a |" % n)
            continue
        c = [v * 1e3 for v in raw]
        cm = med(c)
        w("| %d | %.4f | %s | %.3f | %.3f | %.3f | %.3f |" % (
            n, cm, "[" + ", ".join("r%d=%.4f" % (j + 1, v) for j, v in enumerate(c)) + "]",
            cm / wmed[(n, "before")], cm / wmed[(n, "after")],
            cm / kmed[(n, "before")], cm / kmed[(n, "after")]))
    w("")
    return "\n".join(out) + "\n"


# ---- self-test -------------------------------------------------------------

def _fx_write(d, fn, text):
    with open(os.path.join(d, fn), "w", encoding="utf-8") as f:
        f.write(text)


def _hdr(test):
    return "test x::%s ... %s\nMINI_TABLE_ACTIVE\n" % (test, DEVICE_LINE)


TAIL = "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 800 filtered out\n"


def _fx_bit():
    s = _hdr("bit")
    for n in SIZES:
        h = "%016x" % (n * 7919)
        s += "BITMATCH N=%d bit_identical=true hash_after=%s hash_before=%s parity_ok=true\n" % (n, h, h)
    return s + TAIL


def _fx_make(d):
    _fx_write(d, "rev-stamp.txt", "abc123 fixture\n")
    _fx_write(d, "candle_tree.txt", "└── candle-core v0.11.0\n")
    _fx_write(d, "precheck.log", "test result: ok. 12 passed; 0 failed; 0 ignored\n")
    _fx_write(d, "bitgate.log", _fx_bit())
    gate = ["precheck gate=pass load1=1.0 waited_s=0 at=x", "bitgate gate=pass load1=1.0 waited_s=0 at=x"]
    for i in RUNS:
        for lab in ["run%d-select" % i, "run%d-bit" % i, "run%d-ab" % i] + [
                "run%d-candle-N%d" % (i, n) for n in SIZES]:
            gate.append("%s gate=pass load1=1.0 waited_s=0 at=x" % lab)
    gate.append("series done")
    _fx_write(d, "load_gate.log", "\n".join(gate) + "\n")
    for r in RUNS:
        s = _hdr("select")
        for n in SIZES:
            s += "SELECT N=%d idx=%d fallback=false ok=true\n" % (n, EXPECT_IDX[n])
        _fx_write(d, "select_run%d.log" % r, s + TAIL)
        _fx_write(d, "bit_run%d.log" % r, _fx_bit())
        s = _hdr("ab")
        for n in SIZES:
            s += "AB N=%d arm=before kernel_gpu_median_ms=%.4f wall_sum_median_ms=%.4f\n" % (
                n, 1.0 + 0.001 * r, 1.3 + 0.001 * r)
            s += "AB N=%d arm=after kernel_gpu_median_ms=%.4f wall_sum_median_ms=%.4f\n" % (
                n, 0.8 + 0.001 * r, 1.1 + 0.001 * r)
        _fx_write(d, "ab_run%d.log" % r, s + TAIL)
        cj = []
        for n in SIZES:
            cj.append(json.dumps({"task": "gemm_transfer_incl", "device": "metal", "size": n,
                                  "median_s": 0.5, "version": "0.11.0"}))
            cj.append(json.dumps({"task": "gemm_transfer_excl", "device": "metal", "size": n,
                                  "median_s": 0.002, "version": "0.11.0",
                                  "parity_total": n * n, "parity_fail_count": 0}))
        _fx_write(d, "candle_run%d.jsonl" % r, "\n".join(cj) + "\n")


def self_test():
    def edit(fn, old, new):
        def f(d):
            p = os.path.join(d, fn)
            s = open(p, encoding="utf-8").read()
            assert old in s, (fn, old)
            open(p, "w", encoding="utf-8").write(s.replace(old, new, 1))
        return f

    def rm(fn):
        return lambda d: os.remove(os.path.join(d, fn))

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
        assert "正式系列（5/5" in text and "改善" in text and "退行: なし" in text, text
        assert "判定不能" not in text

    expect_fail("run 欠落", rm("ab_run5.log"))
    expect_fail("candle run 欠落", rm("candle_run3.jsonl"))
    expect_fail("MINI_TABLE_ACTIVE 欠落", edit("select_run2.log", "MINI_TABLE_ACTIVE", "X"))
    expect_fail("SKIP 混入", edit("ab_run1.log", "MINI_TABLE_ACTIVE", "SKIP not-m4-base"))
    expect_fail("DEVICE 不一致", edit("bit_run1.log", "token=M4Base", "token=Other"))
    expect_fail("hash run 間不一致", edit("bit_run3.log", "%016x" % (256 * 7919), "%016x" % 1))
    expect_fail("hash 腕間不一致", edit("bit_run3.log", "hash_before=%016x" % (256 * 7919),
                                       "hash_before=%016x" % 1))
    expect_fail("bit_identical=false", edit("bit_run1.log", "bit_identical=true", "bit_identical=false", ))
    expect_fail("期待 idx 不一致", edit("select_run1.log", "SELECT N=2048 idx=2", "SELECT N=2048 idx=3"))
    expect_fail("fallback", edit("select_run1.log", "N=512 idx=1 fallback=false", "N=512 idx=1 fallback=true"))
    expect_fail("AB 行欠落", edit("ab_run1.log", "AB N=4096 arm=after", "AB N=4096 arm=aftr"))
    expect_fail("AB 非数値", edit("ab_run1.log", "kernel_gpu_median_ms=1.0010", "kernel_gpu_median_ms=abc"))
    expect_fail("candle 行欠落", edit("candle_run1.jsonl", '"task": "gemm_transfer_excl", "device": "metal", '
                                      '"size": 4096', '"task": "other", "device": "metal", "size": 4096'))
    expect_fail("candle median_s 非有限", edit("candle_run1.jsonl", '"median_s": 0.002', '"median_s": 1e999'))
    expect_fail("candle parity 欠損", edit("candle_run1.jsonl", ', "parity_total": 65536', ''))
    expect_fail("candle 版不一致", edit("candle_tree.txt", "v0.11.0", "v0.12.0"))
    expect_fail("precheck 失敗", edit("precheck.log", "ok. 12 passed", "FAILED. 0 passed"))
    for label, old, new in [
            ("gate 非 pass", "run2-ab gate=pass", "run2-ab gate=timeout"),
            ("FAILED 混入", "series done", "run1 FAILED x\nseries done"),
            ("series done 欠落", "series done", "x")]:
        with tempfile.TemporaryDirectory() as d:
            _fx_make(d)
            edit("load_gate.log", old, new)(d)
            t = render(build(d))
            assert "正式系列ではない" in t and "正式系列（5/5" not in t, label
    with tempfile.TemporaryDirectory() as d:
        _fx_make(d)
        edit("candle_run1.jsonl", '"parity_fail_count": 0', '"parity_fail_count": 3')(d)
        assert "判定不能" in render(build(d))
    with tempfile.TemporaryDirectory() as d:
        _fx_make(d)
        try:
            build(d, expect_rev="zzz")
        except AggregateError:
            pass
        else:
            raise SystemExit("self-test 失敗: rev 不一致が拒否されなかった")
    # 分類の境界
    assert classify([(1.0, 0.9)] * 5)[1] == "改善"
    assert classify([(1.0, 0.9)] * 4 + [(1.0, 1.01)])[1] == "帯内"
    assert classify([(1.0, 1.1)] * 3 + [(1.0, 0.9)] * 2)[1] == "退行"
    assert classify([(1.0, 1.1)] * 2 + [(1.0, 0.9)] * 3)[1] == "帯内"
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
