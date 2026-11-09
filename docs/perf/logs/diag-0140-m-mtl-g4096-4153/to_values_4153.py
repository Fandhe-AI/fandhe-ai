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
欠損・重複・非有限・区間本数の不一致・区間の重なり・DOCTYPE／ENTITY・解決できない ref・未知の AUX キー・
AUX の xctrace と cell 行の formulas の食い違いはすべて ConvError で停止する。
xctrace=unavailable（F2 だけのフォールバック）では trace を読まないため、trace 専用の AUX キー（TRACE_KEYS）は
要求しない（`<…>` のまま・未記載でよい）。available では従来どおりすべて実値を要求する。
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
# trace を読む経路（xctrace=available）だけが使う AUX キー。attrib_4153.py の TRACE_KEYS と同じ集合に保つ。
TRACE_KEYS = ["xctrace_schema", "xctrace_start_col", "xctrace_dur_col", "xctrace_proc_col",
              "xctrace_time_unit_ns", "gemm_interval_min_ms"]
# RULE の AUX に書いてよいキーの全集合（未知のキーは止める。attrib_4153.py の AUX_ALLOWED と同じ集合に保つ）
AUX_ALLOWED = {"xctrace", *TRACE_KEYS, "perturbation_max", "cross_check_max", "kernel_share_threshold",
               "backend_metal_diff_vs_v0_14_0", "to_values_sha256", "attrib_sha256", "run_round_sha256"}
# AUX の xctrace と PARAMS の cell 行 formulas の対応（RULE 項 5 のフォールバック）。aggregate_diag.py は AUX を読まないため、
# F1 を測らない経路で formulas に F1 が残る（またはその逆）食い違いはここと attrib_4153.py で止める。
FORMULAS_BY_MODE = {"available": ["F1", "F2"], "unavailable": ["F2"]}


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


def cell_formulas(rule_text):
    """PARAMS の cell 行（本セル）の formulas を返す。0 件・2 件以上は停止。"""
    hits = re.findall(rf"^cell:\s*{CELL}\s*\|(.*)$", rule_text.split("## PARAMS", 1)[-1], re.M)
    if len(hits) != 1:
        raise ConvError(f"PARAMS: cell {CELL} expected once, got {len(hits)}")
    fs = [f.split("=", 1)[1].strip() for f in hits[0].split("|") if f.strip().startswith("formulas=")]
    if len(fs) != 1:
        raise ConvError(f"PARAMS: cell {CELL}: formulas expected once")
    return [x.strip() for x in fs[0].split(",")]


def parse_aux(rule_text):
    """RULE の `## AUX` 節（`## PARAMS` の手前まで）を key: value として読み、xctrace と formulas の対応も検査する。"""
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
        if k not in AUX_ALLOWED:
            raise ConvError(f"AUX: unknown key {k!r}")
        kv[k] = v
    if kv.get("xctrace") not in FORMULAS_BY_MODE:
        raise ConvError("AUX: xctrace must be available|unavailable")
    # unavailable では trace を読まないので trace 専用キーは要求しない（probe が schema を決められないときは <…> のまま）
    for k in (TRACE_KEYS if kv["xctrace"] == "available" else []):
        if k not in kv:
            raise ConvError(f"AUX: missing key {k!r}")
        if re.search(r"<[^<>]*>", kv[k]):
            raise ConvError(f"AUX: unreplaced placeholder in {k}")
    fs = cell_formulas(rule_text)
    if fs != FORMULAS_BY_MODE[kv["xctrace"]]:
        raise ConvError(f"AUX xctrace={kv['xctrace']} requires formulas={','.join(FORMULAS_BY_MODE[kv['xctrace']])}, "
                        f"PARAMS has formulas={','.join(fs)}")
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
    """xctrace export から対象プロセスの GEMM 区間 (start_ms, dur_ms) を開始順で返す。

    開始時刻と継続時間は同じ単位（AUX xctrace_time_unit_ns）として ms へ直す。同一プロセスの GEMM 区間が
    重なる（前の区間の終わりより次の開始が早い）場合は、表が 1 行＝1 実行区間になっていない（抽出の前提が崩れている）
    とみなして停止する。重ならないことは区間の並びの正しさを保証しない（余分な区間の位置は pure_kernel を参照）。
    """
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
    unit_f = finite(aux["xctrace_time_unit_ns"], "xctrace_time_unit_ns")
    if unit_f != int(unit_f) or unit_f < 1:
        raise ConvError("AUX: xctrace_time_unit_ns must be an integer >= 1")
    unit = int(unit_f)
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
        start = finite((resolve(cell[aux["xctrace_start_col"]]).text or "").strip(), f"{xml_path}: start") * unit / 1e6
        # fmt は表示用（"6.2 ms" 等）のため、数値は要素本体のテキストを使う
        dur = finite((resolve(cell[aux["xctrace_dur_col"]]).text or "").strip(), f"{xml_path}: duration")
        dur_ms = dur * unit / 1e6
        if dur_ms >= min_ms:
            out.append((start, dur_ms))
    out.sort()
    for (s0, d0), (s1, _) in zip(out, out[1:]):
        if s1 < s0 + d0:
            raise ConvError(f"{xml_path}: GEMM intervals overlap at start {s1} ms (not one row per GPU interval; INVALID)")
    return out


def pure_kernel(xml_path, aux, proc_match, warmup, iters, lead=0, trail=0):
    """開始順の区間列から測定反復 iters 本だけを切り出して中央値(ms)を返す。

    lead＝測定の前に同一プロセスで走る GEMM の本数、trail＝後ろに走る本数（いずれも除外する）。
    candle の gemm-transfer-split は転送込み(warmup+iters)→転送除外(warmup+iters)→checksum 検証 1 回の順で
    実行するため、転送除外の測定反復は lead=転送込みの warmup+iters・trail=1 で切り出す。
    本数は lead＋warmup＋iters＋trail と完全一致を要求する（余分な区間の許容枠は設けない）。継続時間だけでは
    余分な ≥ 閾値の区間が先頭・途中・末尾のどこにあるかを特定できず、許容すると測定スライスが黙ってずれるため、
    1 本でも多い／少ない round は INVALID とする（扱いを変えるなら RULE.v2 で抽出規則ごと定め直す）。
    件数の一致は並びの正しさを保証しない（GEMM の欠落と余分な区間が同数なら検出できない）。
    """
    iv = gemm_intervals(xml_path, aux, proc_match)
    expected = lead + warmup + iters + trail
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
        # 区間の切り出しには trace 下の run 自身の warmup/iters を使い、trace なしの run と同じ構成であることを確かめる
        # （摂動 = traced ÷ untraced が同じ反復構成の比較になる前提）
        fw_t, f_warm, f_iters = load_record(p("fandhe_reuse_traced.jsonl"), "gemm", "reuse")
        _, f_warm_u, f_iters_u = load_record(p("fandhe_reuse.jsonl"), "gemm", "reuse")
        if (f_warm, f_iters) != (f_warm_u, f_iters_u):
            raise ConvError("fandhe_reuse(_traced): warmup/iters differ between traced and untraced runs")
        osp_t, o_warm_t, o_iters_t = load_record(p("candle_split_traced.jsonl"), "gemm_transfer_excl")
        if (o_warm_t, o_iters_t) != (o_warm, o_iters):
            raise ConvError("candle_split(_traced): warmup/iters differ between traced and untraced runs")
        # 先行する転送込みの本数は転送込みレコード自身の warmup/iters から取る（転送除外と同じと仮定しない）
        _, i_warm_t, i_iters_t = load_record(p("candle_split_traced.jsonl"), "gemm_transfer_incl")
    kgpu = parse_diag(read_text(p("route.txt")), read_text(p("diag.txt")))
    values = [("fandhe_window", fw, "-"), ("fandhe_fresh_window", ffw, "-"), ("opp_window", ow, "-")]
    aux_rows = [("fandhe_kernel_gpu_ts", kgpu), ("opp_transfer_excl", osp)]
    if avail:
        fk = pure_kernel(p("fandhe_reuse.trace.xml"), aux, "bench-fandhe", f_warm, f_iters)
        ok = pure_kernel(p("candle_split.trace.xml"), aux, "bench-candle", o_warm_t, o_iters_t,
                         lead=i_warm_t + i_iters_t, trail=1)
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


def _xml(proc, durs_ms, extra_rows=(), doctype="", ref_bad=False, overlap=False):
    cols = ["start", "dur", "proc"]
    head = "".join(f"<col><mnemonic>{c}</mnemonic></col>" for c in cols)
    rows = []
    t = 10 ** 6  # ns（単位 1 ns）。区間は 1 ms の隙間を空けて重ならないよう並べる
    for i, d in enumerate(durs_ms):
        start = t - (int(d * 1e6) // 2 if overlap and i == 1 else 0)  # overlap: 2 本目を 1 本目の途中から始める
        rows.append(f'<row><start-time id="s{i}">{start}</start-time>'
                    f'<duration id="d{i}" fmt="{d} ms">{int(d * 1e6)}</duration>'
                    f'<process id="p{i}" fmt="{proc} (1)"/></row>')
        t += int(d * 1e6) + 10 ** 6
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
xctrace_time_unit_ns: 1
gemm_interval_min_ms: 5.0
## PARAMS
cell: M-MTL-G4096 | machine=macmini | scoreboard_ratio=0.88 | formulas={fs} | kernel_owner=constraint | opp=candle | prev_class=x
"""
# フォールバックで probe が schema を決められなかったときの自然な形（trace 専用キーは雛形の <…> のまま）
RULE_FB_PLACEHOLDER = """RULE fixed-at-utc 2000-01-01T00:00:00Z
## AUX
xctrace: unavailable
xctrace_schema: <表の schema 名>
xctrace_start_col: <開始時刻の列 mnemonic>
xctrace_dur_col: <継続時間の列 mnemonic>
xctrace_proc_col: <プロセス名の列 mnemonic>
xctrace_time_unit_ns: <開始時刻・継続時間 1 単位が何 ns か。整数>
gemm_interval_min_ms: 5.0
## PARAMS
cell: M-MTL-G4096 | machine=macmini | scoreboard_ratio=0.88 | formulas=F2 | kernel_owner=constraint | opp=candle | prev_class=x
"""

ROUTE = f"ROUTE_ACTUAL N={SIZE} route=Classic\nDIAG_BITEQ N={SIZE} equal=true\n"
DIAG = (f"  [label] N=1024 requested_tile=T\n    commit_wait.kernel_gpu: median=1.0000 ms  q1=0 ms  q3=0 ms\n"
        f"  [label] N={SIZE} requested_tile=T\n    commit_wait.kernel_gpu: median=60.0000 ms  q1=0 ms  q3=0 ms\n")


def _build(d, xc="available", f_durs=(60, 61, 62, 63, 64), o_durs=(100, 101, 102, 103, 104, 50, 51, 52, 53, 54, 77), diag=DIAG, route=ROUTE,
           f_doctype="", ref_bad=False, dup_fandhe=False, incl_wi=(2, 3), f_traced_wi=(2, 3), f_overlap=False):
    raw = os.path.join(d, "raw")
    os.makedirs(raw)
    w = lambda n, t: open(os.path.join(raw, n), "w", encoding="utf-8").write(t)
    w("fandhe_reuse.jsonl", _rec("gemm", "reuse", 64.0) * (2 if dup_fandhe else 1))
    w("fandhe_fresh.jsonl", _rec("gemm", "fresh", 70.0))
    w("candle_gemm.jsonl", _rec("gemm", "fresh", 56.0))
    w("candle_split.jsonl", _rec("gemm_transfer_incl", "fresh", 90.0) + _rec("gemm_transfer_excl", "fresh", 42.0))
    w("fandhe_reuse_traced.jsonl", "noise line\n" + _rec("gemm", "reuse", 65.0, *f_traced_wi))
    w("candle_split_traced.jsonl", _rec("gemm_transfer_incl", "fresh", 91.0, *incl_wi) + _rec("gemm_transfer_excl", "fresh", 43.0))
    extra = [("kernel_task", 9.0), ("bench-fandhe", 0.5)]
    w("fandhe_reuse.trace.xml", _xml("bench-fandhe", f_durs, extra, f_doctype, ref_bad, f_overlap))
    w("candle_split.trace.xml", _xml("bench-candle", o_durs))
    w("route.txt", route)
    w("diag.txt", diag)
    return d, RULE_FIX.format(xc=xc, fs=",".join(FORMULAS_BY_MODE[xc]))


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
        assert convert(d2, rule2)[0] == vt2
        # フォールバックでは trace 専用キーを要求しない: <…> のまま・未記載のどちらでも変換できる（指摘 2）
        assert convert(d2, RULE_FB_PLACEHOLDER)[0] == vt2
        rule_fb_absent = "\n".join(l for l in RULE_FB_PLACEHOLDER.splitlines()
                                   if l.split(":", 1)[0] not in TRACE_KEYS) + "\n"
        assert "xctrace_schema" not in rule_fb_absent
        assert convert(d2, rule_fb_absent)[0] == vt2
        # 通常経路（available）は従来どおり trace 専用キーの実値を要求する（<…> 残り・未記載は停止）
        d2a, r2a = _build(os.path.join(t, "avail-strict"))
        for k in TRACE_KEYS:
            ph = re.sub(rf"^{k}: .*$", f"{k}: <x>", r2a, flags=re.M)
            _expect(lambda: convert(d2a, ph), f"available placeholder {k}", "placeholder")
            gone = re.sub(rf"^{k}: .*\n", "", r2a, flags=re.M)
            _expect(lambda: convert(d2a, gone), f"available missing {k}", "missing")
        # AUX xctrace と formulas の食い違いは両方向とも停止する（指摘 1。aggregate_diag.py は AUX を読まないため）
        _expect(lambda: convert(d2a, r2a.replace("formulas=F1,F2", "formulas=F2")), "available+F2", "requires formulas=F1,F2")
        _expect(lambda: convert(d2, rule2.replace("formulas=F2", "formulas=F1,F2")), "unavailable+F1,F2", "requires formulas=F2")
        _expect(lambda: convert(d2, rule2.replace("xctrace: unavailable", "xctrace: maybe")), "bad xctrace", "available|unavailable")
        # 未知の AUX キー（旧 xctrace_expected_extra を含む）は黙って無視せず停止する
        _expect(lambda: convert(d2a, r2a.replace("## PARAMS", "xctrace_expected_extra: 1\n## PARAMS")), "stale key", "unknown key")
        # 余分な ≥ 5 ms の区間は位置（先頭・途中・末尾）によらず INVALID（指摘 3。位置を仮定して切り出さない）
        for lab, fd in (("head", (99, 60, 61, 62, 63, 64)), ("mid", (60, 61, 99, 62, 63, 64)), ("tail", (60, 61, 62, 63, 64, 99))):
            dx, rx = _build(os.path.join(t, f"extra-{lab}"), f_durs=fd)
            _expect(lambda: convert(dx, rx), f"extra interval {lab}", "count")
        dxc, rxc = _build(os.path.join(t, "extra-candle-mid"), o_durs=(100, 101, 102, 103, 104, 99, 50, 51, 52, 53, 54, 77))
        _expect(lambda: convert(dxc, rxc), "extra candle interval between incl and excl", "count")
        # 同一プロセスの GEMM 区間が重なる表は 1 行＝1 実行区間ではないので停止する
        dov, rov = _build(os.path.join(t, "overlap"), f_overlap=True)
        _expect(lambda: convert(dov, rov), "overlap", "overlap")
        # 時刻単位は 1 以上の整数
        _expect(lambda: convert(d2a, r2a.replace("xctrace_time_unit_ns: 1", "xctrace_time_unit_ns: 0.5")), "unit", "integer")
        # candle の先行本数は転送込みレコード自身の warmup/iters から取る（incl 1+2 本・excl 2+3 本・checksum 1 本）
        dli, rli = _build(os.path.join(t, "incl-diff"), incl_wi=(1, 2), o_durs=(100, 101, 102, 50, 51, 52, 53, 54, 77))
        vli, _ = convert(dli, rli)
        assert f"{CELL}\topp_pure_kernel\t53.000000\t{CLOCK}" in vli, vli
        # fandhe の trace 下と trace なしで反復構成が違えば停止する
        dfw, rfw = _build(os.path.join(t, "fandhe-wi"), f_traced_wi=(1, 4))
        _expect(lambda: convert(dfw, rfw), "fandhe traced warmup/iters", "differ")
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
        _expect(lambda: convert(d9, r9.split("cell:")[0]), "no cell line", "expected once")
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
