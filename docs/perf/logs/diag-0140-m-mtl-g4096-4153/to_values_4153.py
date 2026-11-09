#!/usr/bin/env python3
"""#4153 の 1 round 分の生出力 raw/ から values.tsv・aux.tsv を作る変換（厳格パース）。

役割: bench-fandhe／bench-candle の JSON 行・xctrace（Metal System Trace）の XML export・
診断テストの出力を、aggregate_diag.py が読む values.tsv（セル M-MTL-G4096）と、
attrib_4153.py が読む補助 aux.tsv へ変換する。docs/perf/logs/diag-0140-template/README.md の
「ハーネス固有の出力から values.tsv への変換は各 issue の責務」に対応する。実測値は含まない。

入力（<round-dir>/raw/）:
  fandhe_reuse.jsonl  fandhe_fresh.jsonl  candle_gemm.jsonl  candle_split.jsonl   trace なしの窓
  fandhe_reuse_traced.jsonl  candle_split_traced.jsonl                               trace 下の窓（摂動検査用）
  fandhe_reuse.trace.xml  candle_split.trace.xml                                     xctrace export（AUX の xctrace=available のとき）
  route.txt  diag.txt                                                                診断テストの出力
欠損・重複・非有限・区間本数の不一致・DOCTYPE／ENTITY・解決できない ref はすべて ConvError で停止する。
xctrace の XML は非信頼データ（XXE・巨大入力を拒否）。標準ライブラリのみ。`python3 -I` で実行すること。

使い方:
  python3 -I to_values_4153.py --round-dir <round{i}> --rule <RULE.txt>
  python3 -I to_values_4153.py --self-test
"""
import json
import math
import os
import re
import statistics
import sys
import tempfile
import xml.etree.ElementTree as ET

CELL = "M-MTL-G4096"
SIZE = 4096
CLOCK = "gpu_trace"
XML_MAX_BYTES = 64 * 1024 * 1024
AUX_KEYS = ["xctrace", "xctrace_schema", "xctrace_start_col", "xctrace_dur_col", "xctrace_proc_col",
            "xctrace_dur_unit_ns", "xctrace_expected_extra", "gemm_interval_min_ms"]


class ConvError(Exception):
    pass


def finite(s, what):
    try:
        v = float(s)
    except (TypeError, ValueError):
        raise ConvError(f"{what}: not a number: {s!r}")
    if not math.isfinite(v):
        raise ConvError(f"{what}: not finite: {s!r}")
    return v


def read_text(path):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError as e:
        raise ConvError(f"cannot read {path}: {e}")


def parse_aux(rule_text):
    """RULE の `## AUX` 節（`## PARAMS` の手前まで）を key: value として読む。"""
    lines = rule_text.splitlines()
    try:
        a = next(i for i, l in enumerate(lines) if l.strip() == "## AUX")
    except StopIteration:
        raise ConvError("RULE: '## AUX' section required")
    kv = {}
    for l in lines[a + 1:]:
        s = l.strip()
        if s.startswith("## "):
            break
        if not s or s.startswith("（") or s.startswith("#"):
            continue
        if ":" not in s:
            raise ConvError(f"AUX: not key: value: {l!r}")
        k, v = s.split(":", 1)
        k, v = k.strip(), v.strip()
        if k in kv:
            raise ConvError(f"AUX: duplicate key {k!r}")
        kv[k] = v
    for k in AUX_KEYS:
        if k not in kv:
            raise ConvError(f"AUX: missing key {k!r}")
        if re.search(r"<[^<>]*>", kv[k]):
            raise ConvError(f"AUX: unreplaced placeholder in {k}")
    if kv["xctrace"] not in ("available", "unavailable"):
        raise ConvError("AUX: xctrace must be available|unavailable")
    return kv


def load_record(path, task, mode=None):
    """JSON 行から (task, device=metal, size=4096[, mode]) に合う 1 件だけを返す。0 件・2 件以上は停止。"""
    hits = []
    for ln, line in enumerate(read_text(path).splitlines(), 1):
        s = line.strip()
        if not s.startswith("{"):
            continue
        try:
            r = json.loads(s)
        except ValueError as e:
            raise ConvError(f"{path}:{ln}: bad JSON: {e}")
        if r.get("task") == task and r.get("device") == "metal" and r.get("size") == SIZE:
            if mode is None or r.get("mode") == mode:
                hits.append(r)
    if len(hits) != 1:
        raise ConvError(f"{path}: expected exactly 1 record for task={task} mode={mode}, got {len(hits)}")
    r = hits[0]
    ms = finite(r.get("median_s"), f"{path}: median_s") * 1e3
    if ms <= 0:
        raise ConvError(f"{path}: median must be > 0")
    for k in ("warmup", "iters"):
        if not isinstance(r.get(k), int) or r[k] < 0:
            raise ConvError(f"{path}: {k} must be a non-negative integer")
    return ms, r["warmup"], r["iters"]


def _elem_text(el):
    if el.get("fmt"):
        return el.get("fmt")
    t = "".join(el.itertext()).strip()
    return t


def gemm_intervals(xml_path, aux, proc_match):
    """xctrace export から対象プロセスの GEMM 区間 (start, dur_ms) を開始順で返す。"""
    try:
        with open(xml_path, "rb") as f:
            # 上限 + 1 バイトまでしか読まない（巨大ファイルを全量メモリへ載せる前に拒否する）
            data = f.read(XML_MAX_BYTES + 1)
    except OSError as e:
        raise ConvError(f"cannot read {xml_path}: {e}")
    if len(data) > XML_MAX_BYTES:
        raise ConvError(f"{xml_path}: too large")
    if b"<!DOCTYPE" in data.upper() or b"<!ENTITY" in data.upper():
        raise ConvError(f"{xml_path}: DOCTYPE/ENTITY is rejected")
    try:
        root = ET.fromstring(data)
    except ET.ParseError as e:
        raise ConvError(f"{xml_path}: XML parse error: {e}")
    ids = {}
    for el in root.iter():
        i = el.get("id")
        if i is not None:
            if i in ids:
                raise ConvError(f"{xml_path}: duplicate id {i!r}")
            ids[i] = el

    def resolve(el):
        ref = el.get("ref")
        if ref is None:
            return el
        if ref not in ids:
            raise ConvError(f"{xml_path}: unresolved ref {ref!r}")
        return ids[ref]

    schema = [s for s in root.iter("schema") if s.get("name") == aux["xctrace_schema"]]
    if len(schema) != 1:
        raise ConvError(f"{xml_path}: schema {aux['xctrace_schema']!r} expected once, got {len(schema)}")
    cols = []
    for c in schema[0].findall("col"):
        m = c.find("mnemonic")
        if m is None or not (m.text or "").strip():
            raise ConvError(f"{xml_path}: col without mnemonic")
        cols.append(m.text.strip())
    if len(set(cols)) != len(cols):
        raise ConvError(f"{xml_path}: duplicate col mnemonic")
    for need in ("xctrace_start_col", "xctrace_dur_col", "xctrace_proc_col"):
        if aux[need] not in cols:
            raise ConvError(f"{xml_path}: col {aux[need]!r} not in schema {cols}")
    unit = int(finite(aux["xctrace_dur_unit_ns"], "xctrace_dur_unit_ns"))
    min_ms = finite(aux["gemm_interval_min_ms"], "gemm_interval_min_ms")
    out = []
    for row in root.iter("row"):
        kids = list(row)
        if len(kids) != len(cols):
            raise ConvError(f"{xml_path}: row has {len(kids)} cells, schema has {len(cols)}")
        cell = dict(zip(cols, kids))
        proc = _elem_text(resolve(cell[aux["xctrace_proc_col"]]))
        if proc_match not in proc:
            continue
        start = finite((resolve(cell[aux["xctrace_start_col"]]).text or "").strip(), f"{xml_path}: start")
        # fmt は表示用（"6.2 ms" 等）のため、数値は要素本体のテキストを使う
        dur = finite((resolve(cell[aux["xctrace_dur_col"]]).text or "").strip(), f"{xml_path}: duration")
        dur_ms = dur * unit / 1e6
        if dur_ms >= min_ms:
            out.append((start, dur_ms))
    out.sort()
    return out


def pure_kernel(xml_path, aux, proc_match, warmup, iters, lead=0, trail=0):
    """開始順の区間列から測定反復 iters 本だけを切り出して中央値(ms)を返す。

    lead＝測定の前に同一プロセスで走る GEMM の本数、trail＝後ろに走る本数（いずれも除外する）。
    candle の gemm-transfer-split は転送込み(warmup+iters)→転送除外(warmup+iters)→checksum 検証 1 回の順で
    実行するため、転送除外の測定反復は lead=warmup+iters・trail=1 で切り出す。本数の総数が合わなければ INVALID。
    """
    iv = gemm_intervals(xml_path, aux, proc_match)
    expected = lead + warmup + iters + trail + int(finite(aux["xctrace_expected_extra"], "xctrace_expected_extra"))
    if len(iv) != expected:
        raise ConvError(f"{xml_path}: GEMM interval count {len(iv)} != expected {expected} (INVALID)")
    measured = [d for _, d in iv[lead + warmup:lead + warmup + iters]]
    if iters < 1 or len(measured) != iters:
        raise ConvError(f"{xml_path}: no measured intervals after warmup")
    return statistics.median(measured)


def parse_diag(route_text, diag_text):
    """route.txt の経路確認と diag.txt の N=4096 ブロックの kernel_gpu 中央値(ms)を返す。"""
    if not re.search(rf"^ROUTE_ACTUAL N={SIZE} route=Classic\b", route_text, re.M):
        raise ConvError(f"route.txt: ROUTE_ACTUAL N={SIZE} route=Classic not found (INVALID)")
    if not re.search(rf"^DIAG_BITEQ N={SIZE} equal=true\b", route_text, re.M):
        raise ConvError(f"route.txt: DIAG_BITEQ N={SIZE} equal=true not found (INVALID)")
    lines = diag_text.splitlines()
    starts = [i for i, l in enumerate(lines) if re.search(rf"\] N={SIZE} requested_tile", l)]
    if len(starts) != 1:
        raise ConvError(f"diag.txt: expected exactly 1 N={SIZE} block, got {len(starts)}")
    found = []
    for l in lines[starts[0] + 1:]:
        if re.match(r"\s+\[.*\] N=\d+ requested_tile", l):
            break
        m = re.match(r"\s+commit_wait\.kernel_gpu: median=([0-9.eE+-]+) ms", l)
        if m:
            found.append(finite(m.group(1), "kernel_gpu"))
    if len(found) != 1 or found[0] <= 0:
        raise ConvError("diag.txt: commit_wait.kernel_gpu expected once and > 0")
    return found[0]


def convert(round_dir, rule_text):
    aux = parse_aux(rule_text)
    raw = os.path.join(round_dir, "raw")
    p = lambda n: os.path.join(raw, n)
    fw, _, _ = load_record(p("fandhe_reuse.jsonl"), "gemm", "reuse")
    ffw, _, _ = load_record(p("fandhe_fresh.jsonl"), "gemm", "fresh")
    ow, _, _ = load_record(p("candle_gemm.jsonl"), "gemm")
    # bench-candle gemm-transfer-split の JSON task 名は gemm_transfer_incl／gemm_transfer_excl（転送除外側を使う）
    osp, o_warm, o_iters = load_record(p("candle_split.jsonl"), "gemm_transfer_excl")
    avail = aux["xctrace"] == "available"
    if avail:
        fw_t, f_warm, f_iters = load_record(p("fandhe_reuse_traced.jsonl"), "gemm", "reuse")
        osp_t, o_warm_t, o_iters_t = load_record(p("candle_split_traced.jsonl"), "gemm_transfer_excl")
        if (o_warm_t, o_iters_t) != (o_warm, o_iters):
            raise ConvError("candle_split(_traced): warmup/iters differ between traced and untraced runs")
    kgpu = parse_diag(read_text(p("route.txt")), read_text(p("diag.txt")))
    values = [("fandhe_window", fw, "-"), ("fandhe_fresh_window", ffw, "-"), ("opp_window", ow, "-")]
    aux_rows = [("fandhe_kernel_gpu_ts", kgpu), ("opp_transfer_excl", osp)]
    if avail:
        fk = pure_kernel(p("fandhe_reuse.trace.xml"), aux, "bench-fandhe", f_warm, f_iters)
        ok = pure_kernel(p("candle_split.trace.xml"), aux, "bench-candle", o_warm, o_iters,
                         lead=o_warm + o_iters, trail=1)
        values += [("fandhe_pure_kernel", fk, CLOCK), ("opp_pure_kernel", ok, CLOCK)]
        aux_rows += [("opp_transfer_excl_traced", osp_t), ("fandhe_window_traced", fw_t)]
    # xctrace unavailable（F2 のみ）では trace 下の窓・純カーネルを取らず、これらの行を出さない（attrib は unavailable 扱い）
    vt = "cell\tquantity\tvalue\tclock\n" + "".join(f"{CELL}\t{q}\t{v:.6f}\t{c}\n" for q, v, c in values)
    at = "quantity\tvalue\n" + "".join(f"{q}\t{v:.6f}\n" for q, v in aux_rows)
    return vt, at


def write_new(path, text):
    if os.path.exists(path):
        raise ConvError(f"{path} already exists (no overwrite)")
    with open(path, "x", encoding="utf-8") as f:
        f.write(text)


# ---------------------------------------------------------------- self-test（合成データのみ。数値は架空）
def _rec(task, mode, med_ms, warmup=2, iters=3):
    return json.dumps({"framework": "x", "version": "0", "task": task, "device": "metal", "size": SIZE,
                       "median_s": med_ms / 1e3, "q1_s": 0.0, "q3_s": 0.0, "checksum": 0.0,
                       "warmup": warmup, "iters": iters, "mode": mode}) + "\n"


def _xml(proc, durs_ms, extra_rows=(), doctype="", ref_bad=False):
    cols = ["start", "dur", "proc"]
    head = "".join(f"<col><mnemonic>{c}</mnemonic></col>" for c in cols)
    rows = []
    for i, d in enumerate(durs_ms):
        rows.append(f'<row><start-time id="s{i}">{1000 * (i + 1)}</start-time>'
                    f'<duration id="d{i}" fmt="{d} ms">{int(d * 1e6)}</duration>'
                    f'<process id="p{i}" fmt="{proc} (1)"/></row>')
    for i, (p, d) in enumerate(extra_rows):  # 別プロセス・短い区間
        rows.append(f'<row><start-time id="xs{i}">{50 + i}</start-time><duration id="xd{i}">{int(d * 1e6)}</duration>'
                    f'<process id="xp{i}" fmt="{p} (2)"/></row>')
    if ref_bad:
        rows.append('<row><start-time id="zz">1</start-time><duration ref="nope"/><process ref="p0"/></row>')
    return (f'<?xml version="1.0"?>{doctype}<trace-query-result><node><schema name="S">{head}</schema>'
            + "".join(rows) + "</node></trace-query-result>")


RULE_FIX = """RULE fixed-at-utc 2000-01-01T00:00:00Z
## AUX
xctrace: {xc}
xctrace_schema: S
xctrace_start_col: start
xctrace_dur_col: dur
xctrace_proc_col: proc
xctrace_dur_unit_ns: 1
xctrace_expected_extra: 0
gemm_interval_min_ms: 5.0
## PARAMS
"""

ROUTE = f"ROUTE_ACTUAL N={SIZE} route=Classic\nDIAG_BITEQ N={SIZE} equal=true\n"
DIAG = (f"  [label] N=1024 requested_tile=T\n    commit_wait.kernel_gpu: median=1.0000 ms  q1=0 ms  q3=0 ms\n"
        f"  [label] N={SIZE} requested_tile=T\n    commit_wait.kernel_gpu: median=60.0000 ms  q1=0 ms  q3=0 ms\n")


def _build(d, xc="available", f_durs=(60, 61, 62, 63, 64), o_durs=(100, 101, 102, 103, 104, 50, 51, 52, 53, 54, 77), diag=DIAG, route=ROUTE,
           f_doctype="", ref_bad=False, dup_fandhe=False):
    raw = os.path.join(d, "raw")
    os.makedirs(raw)
    w = lambda n, t: open(os.path.join(raw, n), "w", encoding="utf-8").write(t)
    w("fandhe_reuse.jsonl", _rec("gemm", "reuse", 64.0) * (2 if dup_fandhe else 1))
    w("fandhe_fresh.jsonl", _rec("gemm", "fresh", 70.0))
    w("candle_gemm.jsonl", _rec("gemm", "fresh", 56.0))
    w("candle_split.jsonl", _rec("gemm_transfer_incl", "fresh", 90.0) + _rec("gemm_transfer_excl", "fresh", 42.0))
    w("fandhe_reuse_traced.jsonl", "noise line\n" + _rec("gemm", "reuse", 65.0))
    w("candle_split_traced.jsonl", _rec("gemm_transfer_incl", "fresh", 91.0) + _rec("gemm_transfer_excl", "fresh", 43.0))
    extra = [("kernel_task", 9.0), ("bench-fandhe", 0.5)]
    w("fandhe_reuse.trace.xml", _xml("bench-fandhe", f_durs, extra, f_doctype, ref_bad))
    w("candle_split.trace.xml", _xml("bench-candle", o_durs))
    w("route.txt", route)
    w("diag.txt", diag)
    return d, RULE_FIX.format(xc=xc)


def _expect(fn, label, needle=None):
    try:
        fn()
    except ConvError as e:
        if needle and needle not in str(e):
            raise AssertionError(f"{label}: wrong error: {e}")
        return
    raise AssertionError(f"{label}: expected ConvError")


def self_test():
    with tempfile.TemporaryDirectory() as t:
        d, rule = _build(os.path.join(t, "ok"))
        vt, at = convert(d, rule)
        v = {l.split("\t")[1]: l.split("\t") for l in vt.splitlines()[1:]}
        assert v["fandhe_window"][2] == "64.000000" and v["opp_window"][2] == "56.000000"
        # fandhe: 5 区間（warmup 2 を捨てて 62,63,64 の中央値 63）。別プロセス・短区間は除外される
        assert v["fandhe_pure_kernel"][2:] == ["63.000000", CLOCK], v["fandhe_pure_kernel"]
        # candle: 転送込み 5 本・転送除外 5 本・checksum 1 本。転送除外の測定反復(warmup 2 を捨てた 52,53,54)の中央値 53
        assert v["opp_pure_kernel"][2] == "53.000000"
        assert "fandhe_kernel_gpu_ts\t60.000000" in at and "fandhe_window_traced\t65.000000" in at
        # フォールバック: xctrace unavailable なら純カーネル行を出さない
        d2, rule2 = _build(os.path.join(t, "fb"), xc="unavailable")
        vt2, _ = convert(d2, rule2)
        assert "pure_kernel" not in vt2
        _, at2 = convert(d2, rule2)
        assert "traced" not in at2 and "opp_transfer_excl\t42.000000" in at2
        # xctrace unavailable では trace 関連ファイルが無くても変換できる
        for n in ("fandhe_reuse_traced.jsonl", "candle_split_traced.jsonl", "fandhe_reuse.trace.xml", "candle_split.trace.xml"):
            os.remove(os.path.join(d2, "raw", n))
        convert(d2, rule2)
        # 旧 task 名（実出力に存在しない）は拒否される
        d2b, r2b = _build(os.path.join(t, "oldname"))
        open(os.path.join(d2b, "raw", "candle_split.jsonl"), "w").write(_rec("gemm-transfer-split", "fresh", 42.0))
        _expect(lambda: convert(d2b, r2b), "old task name", "exactly 1")
        # 区間本数が測定のみ(5)の場合は INVALID（incl/checksum を含む全本数を要求する）
        d2c, r2c = _build(os.path.join(t, "candle5"), o_durs=(50, 51, 52, 53, 54))
        _expect(lambda: convert(d2c, r2c), "candle count", "count")
        # XML サイズ上限は全量読み込み前に効く
        global XML_MAX_BYTES
        saved, XML_MAX_BYTES = XML_MAX_BYTES, 10
        try:
            d2d, r2d = _build(os.path.join(t, "big"))
            _expect(lambda: convert(d2d, r2d), "xml too large", "too large")
        finally:
            XML_MAX_BYTES = saved
        # 異常系（すべて停止すること）
        d3, r3 = _build(os.path.join(t, "cnt"), f_durs=(60, 61, 62, 63))
        _expect(lambda: convert(d3, r3), "interval count mismatch", "count")
        d4, r4 = _build(os.path.join(t, "dup"), dup_fandhe=True)
        _expect(lambda: convert(d4, r4), "duplicate record", "exactly 1")
        d5, r5 = _build(os.path.join(t, "ref"), ref_bad=True)
        _expect(lambda: convert(d5, r5), "unresolved ref", "unresolved")
        d6, r6 = _build(os.path.join(t, "doctype"), f_doctype='<!DOCTYPE a [<!ENTITY e "x">]>')
        _expect(lambda: convert(d6, r6), "doctype", "DOCTYPE")
        d7, r7 = _build(os.path.join(t, "route"), route="ROUTE_ACTUAL N=4096 route=Other\n")
        _expect(lambda: convert(d7, r7), "route not classic", "route")
        d8, r8 = _build(os.path.join(t, "diag"), diag="no block\n")
        _expect(lambda: convert(d8, r8), "missing diag block", "block")
        d9, r9 = _build(os.path.join(t, "ph"))
        _expect(lambda: convert(d9, r9.replace("xctrace_schema: S", "xctrace_schema: <schema>")), "placeholder", "placeholder")
        _expect(lambda: convert(d9, "RULE fixed-at-utc 2000-01-01T00:00:00Z\n## PARAMS\n"), "no AUX", "AUX")
        # 非有限値
        d10, r10 = _build(os.path.join(t, "nan"))
        open(os.path.join(d10, "raw", "candle_gemm.jsonl"), "w").write(
            '{"task":"gemm","device":"metal","size":4096,"median_s":NaN,"warmup":1,"iters":1}\n')
        _expect(lambda: convert(d10, r10), "nan", "finite")
        # 上書き禁止
        pth = os.path.join(t, "x.tsv")
        write_new(pth, "a")
        _expect(lambda: write_new(pth, "b"), "overwrite", "exists")
    print("self-test ok")


def main(argv):
    if "--self-test" in argv:
        self_test()
        return 0
    args = dict(zip(argv[0::2], argv[1::2]))
    if set(args) != {"--round-dir", "--rule"}:
        print(__doc__)
        return 2
    rd = args["--round-dir"]
    vt, at = convert(rd, read_text(args["--rule"]))
    write_new(os.path.join(rd, "values.tsv"), vt)
    write_new(os.path.join(rd, "aux.tsv"), at)
    print(f"wrote {rd}/values.tsv and aux.tsv")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except (ConvError, AssertionError) as e:
        print(f"CONVERT-ERROR: {e}", file=sys.stderr)
        sys.exit(1)
