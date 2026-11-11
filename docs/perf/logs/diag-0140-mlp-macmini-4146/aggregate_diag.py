#!/usr/bin/env python3
"""1 位でないセルの診断 (3)（F1 カーネル置換・F2 fresh 窓の再現）を機械集計する（#4144）。

役割: docs/perf/loss-attribution-matrix.md §10.1 の算出式 F1・F2 を、RULE.txt の PARAMS で
測る前に固定した規則どおりに集計し、3 分類（自分たちの制約／ハードの上限／計測窓の定義）または
分類保留と確度を出す。Phase 2 の各診断 issue（#4146・#4147・#4154 ほか）が
`docs/perf/logs/diag-0140-<対象>-<issue番号>/` へ本ファイルをコピーして使う。実測値は含まない。

入力（<dir> 配下）:
  RULE.txt                 1 行目 `RULE fixed-at-utc …` と末尾の `## PARAMS` ブロック（RULE.template.txt が雛形）
  round{1..N}/gate.txt     1 語目が PASS／TIMEOUT／INVALID
  round{1..N}/values.tsv   ヘッダ `cell<TAB>quantity<TAB>value<TAB>clock`。PASS のラウンドだけ読む
出力: aggregate.md（人間用）と aggregate.tsv（機械可読。Phase 4 の分類付け直しが読む）。

厳格パース: 欠損・重複・未知・非有限・プレースホルダ残存・スクリプトハッシュ不一致はすべて
AggError で停止する（寛容に読まない）。RULE 固定後の本ファイル改変は
`aggregate_script_sha256` の不一致で検出される。標準ライブラリのみ。`python3 -I` で実行すること。

使い方:
  python3 -I aggregate_diag.py --dir <記録先> [--rule RULE.txt] [--out aggregate.md] [--tsv aggregate.tsv]
  python3 -I aggregate_diag.py --self-test
  python3 -I aggregate_diag.py --check-template [RULE.template.txt]
  python3 -I aggregate_diag.py --fingerprint <tree-dir> [--list-out <file>]
"""
import hashlib
import math
import os
import re
import statistics
import sys
import tempfile

QUANTITIES = ["fandhe_window", "fandhe_fresh_window", "opp_window", "fandhe_pure_kernel", "opp_pure_kernel"]
PARAM_KEYS = ["fixed-at-utc", "issue", "rounds", "tree_sha", "tree_fingerprint",
              "aggregate_script_sha256", "reach_threshold", "noise_band", "cell"]
SINGLE_KEYS = [k for k in PARAM_KEYS if k != "cell"]
CELL_FIELDS = ["machine", "scoreboard_ratio", "formulas", "kernel_owner", "opp", "prev_class"]
CELL_ID_RE = re.compile(r"^[A-Z]-[A-Z]{3,4}-[A-Z0-9]+$")
TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
DRIFT_LIMIT = 0.10
CLASS_WINDOW = "計測窓の定義"
CLASS_HW = "ハードの上限"
CLASS_OWN = "自分たちの制約"
CLASS_HOLD = "分類保留"
FP_ROOTS = ["crates", "scripts/bench", "Cargo.toml", "Cargo.lock", "rust-toolchain.toml"]
FP_SKIP = {"target", ".git"}
PLACEHOLDER_MARK = "【記入"


class AggError(Exception):
    pass


def self_sha256():
    with open(os.path.abspath(__file__), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def read(path):
    if not os.path.isfile(path):
        raise AggError(f"missing file: {path}")
    with open(path, encoding="utf-8") as f:
        return f.read()


def parse_float(s, what):
    try:
        v = float(s)
    except ValueError:
        raise AggError(f"{what}: not a number: {s!r}")
    if not math.isfinite(v):
        raise AggError(f"{what}: not finite: {s!r}")
    return v


def split_params(text):
    """RULE 全文から (1 行目, PARAMS 行のリスト) を返す。"""
    lines = text.splitlines()
    if not lines or not lines[0].startswith("RULE fixed-at-utc "):
        raise AggError("RULE: first line must be 'RULE fixed-at-utc <ts>'")
    idx = [i for i, l in enumerate(lines) if l.strip() == "## PARAMS"]
    if len(idx) != 1:
        raise AggError("RULE: exactly one '## PARAMS' heading required")
    body = [l for l in lines[idx[0] + 1:] if l.strip() and not l.lstrip().startswith("#")]
    return lines[0], body


def parse_kv(body):
    kv, cells = {}, []
    for l in body:
        if ":" not in l:
            raise AggError(f"PARAMS: not key: value: {l!r}")
        k, v = l.split(":", 1)
        k, v = k.strip(), v.strip()
        if k not in PARAM_KEYS:
            raise AggError(f"PARAMS: unknown key {k!r}")
        if k == "cell":
            cells.append(v)
        elif k in kv:
            raise AggError(f"PARAMS: duplicate key {k!r}")
        else:
            kv[k] = v
    return kv, cells


def parse_cell(v):
    parts = [p.strip() for p in v.split("|")]
    cid = parts[0]
    if not CELL_ID_RE.match(cid):
        raise AggError(f"PARAMS: bad cell id {cid!r}")
    f = {}
    for p in parts[1:]:
        if "=" not in p:
            raise AggError(f"PARAMS: cell {cid}: bad field {p!r}")
        a, b = p.split("=", 1)
        if a in f or a not in CELL_FIELDS:
            raise AggError(f"PARAMS: cell {cid}: duplicate/unknown field {a!r}")
        f[a] = b.strip()
    for a in CELL_FIELDS:
        if a not in f or not f[a]:
            raise AggError(f"PARAMS: cell {cid}: missing field {a}")
    if f["machine"] not in ("macmini", "gb10"):
        raise AggError(f"PARAMS: cell {cid}: machine must be macmini|gb10")
    f["scoreboard_ratio"] = parse_float(f["scoreboard_ratio"], f"cell {cid} scoreboard_ratio")
    if f["scoreboard_ratio"] <= 0:
        raise AggError(f"PARAMS: cell {cid}: scoreboard_ratio must be > 0")
    fs = [x.strip() for x in f["formulas"].split(",")]
    if not fs or any(x not in ("F1", "F2") for x in fs) or len(set(fs)) != len(fs):
        raise AggError(f"PARAMS: cell {cid}: formulas must be a subset of F1,F2")
    f["formulas"] = fs
    if f["kernel_owner"] not in ("hardware", "constraint"):
        raise AggError(f"PARAMS: cell {cid}: kernel_owner must be hardware|constraint")
    f["id"] = cid
    return f


def parse_rule(text):
    first, body = split_params(text)
    ts = first[len("RULE fixed-at-utc "):].strip()
    if not TS_RE.match(ts):
        raise AggError(f"RULE: bad timestamp {ts!r}")
    if PLACEHOLDER_MARK in text:
        raise AggError("RULE: unreplaced placeholder (【記入) remains")
    if any("<" in l or ">" in l for l in body):
        raise AggError("PARAMS: unreplaced placeholder (<…>) remains")
    kv, cells = parse_kv(body)
    for k in SINGLE_KEYS:
        if k not in kv:
            raise AggError(f"PARAMS: missing key {k!r}")
    if kv["fixed-at-utc"] != ts:
        raise AggError("PARAMS: fixed-at-utc differs from first line")
    if not re.match(r"^\d+$", kv["issue"]):
        raise AggError("PARAMS: issue must be an integer")
    if not re.match(r"^[1-9]\d*$", kv["rounds"]):
        raise AggError("PARAMS: rounds must be an integer >= 1")
    if not HEX40.match(kv["tree_sha"]):
        raise AggError("PARAMS: tree_sha must be 40 hex")
    if not HEX64.match(kv["tree_fingerprint"]):
        raise AggError("PARAMS: tree_fingerprint must be 64 hex")
    if not HEX64.match(kv["aggregate_script_sha256"]):
        raise AggError("PARAMS: aggregate_script_sha256 must be 64 hex")
    if kv["aggregate_script_sha256"] != self_sha256():
        raise AggError("PARAMS: aggregate_script_sha256 mismatch (script changed after RULE was fixed)")
    thr = parse_float(kv["reach_threshold"], "reach_threshold")
    if thr <= 0:
        raise AggError("PARAMS: reach_threshold must be > 0")
    if kv["noise_band"] != "NB-RANGE5":
        raise AggError("PARAMS: noise_band must be NB-RANGE5")
    if not cells:
        raise AggError("PARAMS: at least one cell required")
    parsed = [parse_cell(c) for c in cells]
    if len({c["id"] for c in parsed}) != len(parsed):
        raise AggError("PARAMS: duplicate cell id")
    return {"ts": ts, "issue": int(kv["issue"]), "rounds": int(kv["rounds"]), "tree_sha": kv["tree_sha"],
            "fp": kv["tree_fingerprint"], "script": kv["aggregate_script_sha256"], "thr": thr,
            "noise_band": kv["noise_band"], "cells": parsed}


def needed(cell):
    """宣言した式から要る量（観測比のため fandhe_window・opp_window は常に要る）。"""
    n = {"fandhe_window", "opp_window"}
    if "F1" in cell["formulas"]:
        n |= {"fandhe_pure_kernel", "opp_pure_kernel"}
    if "F2" in cell["formulas"]:
        n |= {"fandhe_fresh_window"}
    return n


def parse_gate(path):
    t = read(path).split()
    if not t or t[0] not in ("PASS", "TIMEOUT", "INVALID"):
        raise AggError(f"gate: first word must be PASS|TIMEOUT|INVALID: {path}")
    return t[0]


def parse_values(text, cells, where):
    lines = text.splitlines()
    if not lines or lines[0].split("\t") != ["cell", "quantity", "value", "clock"]:
        raise AggError(f"{where}: bad header")
    byid = {c["id"]: c for c in cells}
    out = {}
    for ln, line in enumerate(lines[1:], 2):
        if not line.strip():
            continue
        f = line.split("\t")
        if len(f) != 4:
            raise AggError(f"{where}:{ln}: need 4 tab-separated fields")
        cid, q, v, clock = f
        if cid not in byid:
            raise AggError(f"{where}:{ln}: unknown cell {cid!r}")
        if q not in QUANTITIES:
            raise AggError(f"{where}:{ln}: unknown quantity {q!r}")
        if q not in needed(byid[cid]):
            raise AggError(f"{where}:{ln}: quantity {q} not needed by declared formulas of {cid}")
        if (cid, q) in out:
            raise AggError(f"{where}:{ln}: duplicate ({cid},{q})")
        val = parse_float(v, f"{where}:{ln}")
        if val <= 0:
            raise AggError(f"{where}:{ln}: value must be > 0")
        if q.endswith("pure_kernel"):
            if clock in ("", "-"):
                raise AggError(f"{where}:{ln}: pure_kernel needs a clock tag")
        elif clock != "-":
            raise AggError(f"{where}:{ln}: clock must be '-' for {q}")
        out[(cid, q)] = (val, clock)
    for c in cells:
        for q in needed(c):
            if (c["id"], q) not in out:
                raise AggError(f"{where}: missing ({c['id']},{q})")
    return out


def round_metrics(cell, vals):
    """1 ラウンド分の F1・F2・観測比（相手の窓 ÷ fandhe-ai の窓）。時計が違う・窓超過なら F1 は None。"""
    cid = cell["id"]
    g = lambda q: vals[(cid, q)][0]
    fw, ow = g("fandhe_window"), g("opp_window")
    m = {"obs": ow / fw, "F1": None, "F2": None, "clock_ok": True, "over": []}
    if "F2" in cell["formulas"]:
        m["F2"] = ow / g("fandhe_fresh_window")
    if "F1" in cell["formulas"]:
        fk, ok = g("fandhe_pure_kernel"), g("opp_pure_kernel")
        # 時計不一致時に両側の計測境界を併記できるよう、純カーネル値と時計タグを保持する（RULE §4・matrix §10.1）
        m["bound"] = (fk, vals[(cid, "fandhe_pure_kernel")][1], ok, vals[(cid, "opp_pure_kernel")][1])
        m["win"] = (fw, ow)
        # 時計が違う純カーネル値は窓との大小を同じ尺で比べられない（別プロセスの値を混ぜた約 0.49 ms 級の
        # セルでは揺れだけで超過しうる）。不一致は先に分類保留として記録する（#4190 レビュー指摘）。
        # 時計が揃っていても純カーネル（単発 GEMM の和）は step 内の状態と一致せず窓を上回りうる
        # （#4146 の試走。所有者承認 2026-11-11）。窓超過のラウンドは例外にせず F1 を計算せず印を付け、
        # 時計不一致と同じ道筋（F1 比較不能・境界併記）で扱う。
        if vals[(cid, "fandhe_pure_kernel")][1] != vals[(cid, "opp_pure_kernel")][1]:
            m["clock_ok"] = False
        else:
            if fk > fw:
                m["over"].append("fandhe")
            if ok > ow:
                m["over"].append("opp")
            if not m["over"]:
                den = fw - fk + ok
                if den <= 0:
                    raise AggError(f"{cid}: F1 denominator <= 0")
                m["F1"] = ow / den
    return m


def side(vals, thr):
    """run 別値がすべて閾値の同じ側か（境界ちょうどはまたぐ扱い）。"""
    return min(vals) > thr or max(vals) < thr


def classify(cell, ms, thr, formal, rnos):
    """RULE §5 の優先順位つき規則。F2 → F1（owner で振り分け）→ 保留。rnos は ms の元のラウンド番号。"""
    f2 = [m["F2"] for m in ms] if "F2" in cell["formulas"] else None
    clock_ok = all(m["clock_ok"] for m in ms)
    # 時計不一致または窓超過のラウンドが 1 つでもあれば、そのセルの F1 は比較不能
    f1_ok = clock_ok and not any(m["over"] for m in ms)
    f1 = [m["F1"] for m in ms] if "F1" in cell["formulas"] and f1_ok else None
    med = statistics.median
    if f2 is not None and med(f2) >= thr:
        cls, used, why = CLASS_WINDOW, [f2], "F2 が届く（fresh 窓で相手に届く）"
    elif f1 is not None and med(f1) >= thr:
        cls = CLASS_HW if cell["kernel_owner"] == "hardware" else CLASS_OWN
        # 優先される F2 が「届かない」と全 run 安定して言えなければ確度に反映する（RULE §5・NB-RANGE5）
        used = [f1] + ([f2] if f2 is not None else [])
        why = "F1 が届く（相手の純カーネルに置換すると届く）"
    else:
        cls = CLASS_HOLD
        used = [v for v in (f2, f1) if v is not None]
        if "F1" in cell["formulas"] and not f1_ok:
            parts = []
            if not clock_ok:
                parts.append("F1 の時計不一致")
            if any(m["over"] for m in ms):
                ov = "／".join(f"round {i} {'・'.join(m['over'])}" for i, m in zip(rnos, ms) if m["over"])
                parts.append(f"F1 の窓超過（純カーネル > 窓。{ov}）")
            why = "、".join(parts) + "（比較不能。境界を併記）"
        else:
            why = "どれも届かない（残差は固定費など別区間。別診断へ）"
    high = formal and bool(used) and all(side(v, thr) for v in used) and \
        ("F1" not in cell["formulas"] or f1_ok)
    return cls, why, high, f1, f2


def fnum(x):
    return "-" if x is None else f"{x:.4f}"


def aggregate(dirp, rule_name="RULE.txt"):
    rule = parse_rule(read(os.path.join(dirp, rule_name)))
    thr, n = rule["thr"], rule["rounds"]
    gates, vals, rnos = [], [], []
    for i in range(1, n + 1):
        rd = os.path.join(dirp, f"round{i}")
        if not os.path.isdir(rd):
            raise AggError(f"missing round dir: {rd}")
        g = parse_gate(os.path.join(rd, "gate.txt"))
        gates.append(g)
        if g == "PASS":
            rnos.append(i)  # 元のラウンド番号（部分通過でも round{i}/ と表の run を対応させる）
            vals.append(parse_values(read(os.path.join(rd, "values.tsv")), rule["cells"], f"round{i}/values.tsv"))
    if not vals:
        raise AggError("no PASS round")
    formal = all(g == "PASS" for g in gates)
    status = "formal" if formal else "partial"
    md = [f"# 診断 (3) 集計 #{rule['issue']}" + ("" if formal else "（参考）"), "",
          f"- status: {status}（ゲート: {' '.join(gates)}）",
          f"- RULE: fixed-at-utc={rule['ts']} tree_sha={rule['tree_sha']} tree_fingerprint={rule['fp']}",
          f"- aggregate_script_sha256={rule['script']} reach_threshold={thr:.2f} noise_band={rule['noise_band']}", ""]
    tsv = ["cell\tF1\tF2\tclass\tconfidence\treason"]
    for c in rule["cells"]:
        ms = [round_metrics(c, v) for v in vals]
        cls, why, high, f1, f2 = classify(c, ms, thr, formal, rnos)
        md += [f"## {c['id']}（{c['machine']}・相手 {c['opp']}・kernel_owner={c['kernel_owner']}）", "",
               "| run | F1 | F2 | 観測比 |", "|---|---|---|---|"]
        for i, m in zip(rnos, ms):
            md.append(f"| {i} | {fnum(m['F1'])} | {fnum(m['F2'])} | {fnum(m['obs'])} |")
        mf1 = statistics.median(f1) if f1 else None
        mf2 = statistics.median(f2) if f2 else None
        mobs = statistics.median([m["obs"] for m in ms])
        for i, m in zip(rnos, ms):
            if not m["clock_ok"]:
                fk, fc, ok, oc = m["bound"]
                md.append(f"| {i} 時計不一致 | fandhe 純カーネル {fk:.4f} ms（{fc}） | 相手 純カーネル {ok:.4f} ms（{oc}） | |")
            elif m["over"]:
                fk, fc, ok, oc = m["bound"]
                fw, ow = m["win"]
                md.append(f"| {i} 窓超過（{'・'.join(m['over'])}） | fandhe 純カーネル {fk:.4f} ms（{fc}）／窓 {fw:.4f} ms | "
                          f"相手 純カーネル {ok:.4f} ms（{oc}）／窓 {ow:.4f} ms | |")
        md.append(f"| 中央値 | {fnum(mf1)} | {fnum(mf2)} | {fnum(mobs)} |")
        for name, v in (("F1", f1), ("F2", f2)):
            if v:
                cross = "またぐ" if not side(v, thr) else "またがない"
                md.append(f"| NB-RANGE5 {name} | {min(v):.4f}〜{max(v):.4f}（閾値を{cross}） | | |")
        md.append("")
        conf = "高" if high else "低"
        reason = ("" if high else "確度低: ") + why
        if not formal:
            reason += "（部分通過）"
        drift = abs(mobs / c["scoreboard_ratio"] - 1.0)
        if drift > DRIFT_LIMIT:
            reason += f"／ドリフト: 観測比 {mobs:.4f} がスコアボード {c['scoreboard_ratio']:.4f} から {drift * 100:.1f}% 乖離"
        diff = "同" if c["prev_class"] == cls else f"変更（{c['prev_class']} → {cls}）"
        md += [f"- 分類: {cls}／確度: {conf}／前分類との異同: {diff}", f"- 理由: {reason}", ""]
        tsv.append("\t".join([c["id"], fnum(mf1), fnum(mf2), cls, conf, reason]))
    return "\n".join(md) + "\n", "\n".join(tsv) + "\n"


def fingerprint(tree):
    """tree の指紋（FP_ROOTS の全ファイルの `<sha256>  <相対パス>` 一覧を C ロケール順に並べた全体の SHA-256）。"""
    tree = os.path.abspath(tree)
    if not os.path.isdir(tree):
        raise AggError(f"not a directory: {tree}")
    files = []

    def add(rel):
        full = os.path.join(tree, rel)
        if os.path.islink(full):
            h = hashlib.sha256(("symlink:" + os.readlink(full)).encode()).hexdigest()
        else:
            hh = hashlib.sha256()
            with open(full, "rb") as f:
                for blk in iter(lambda: f.read(1 << 20), b""):
                    hh.update(blk)
            h = hh.hexdigest()
        files.append((rel, h))

    def walk(rel):
        full = os.path.join(tree, rel)
        if os.path.islink(full) or os.path.isfile(full):
            add(rel)
        elif os.path.isdir(full):
            for e in os.scandir(full):
                if e.name in FP_SKIP:
                    continue
                walk(os.path.join(rel, e.name))

    for r in FP_ROOTS:
        if os.path.lexists(os.path.join(tree, r)):
            walk(r)
    files.sort(key=lambda t: t[0].encode())
    listing = "".join(f"{h}  {rel}\n" for rel, h in files)
    return hashlib.sha256(listing.encode()).hexdigest(), len(files), listing


def check_template(path):
    """雛形の PARAMS が必須キーを過不足なく持ち、値がすべてプレースホルダであることを検査する。"""
    text = read(path)
    lines = text.splitlines()
    if not lines or not re.match(r"^RULE fixed-at-utc <.+>$", lines[0]):
        raise AggError("template: first line must carry a <…> timestamp placeholder")
    _, body = split_params(text)
    kv, cells = parse_kv(body)
    if set(kv) != set(SINGLE_KEYS):
        raise AggError(f"template: PARAMS keys mismatch: missing={set(SINGLE_KEYS) - set(kv)} extra={set(kv) - set(SINGLE_KEYS)}")
    if not cells:
        raise AggError("template: at least one cell line required")
    for k, v in list(kv.items()) + [("cell", c) for c in cells]:
        if not re.search(r"<[^<>]+>", v):
            raise AggError(f"template: value of {k} is not a placeholder: {v!r}")
    for c in cells:
        for fld in CELL_FIELDS:
            if fld + "=" not in c:
                raise AggError(f"template: cell line lacks field {fld}")


# ---------------------------------------------------------------- self-test
# 合成データのみ。セル ID・数値は実測と無関係な架空値で、期待値は手計算。

def _expect_err(fn, label):
    try:
        fn()
    except AggError:
        return
    raise SystemExit(f"self-test FAIL: no AggError for {label}")


def _cell_line(cid, owner="hardware", forms="F1,F2", sr=1.0, machine="gb10"):
    return (f"cell: {cid} | machine={machine} | scoreboard_ratio={sr} | formulas={forms} | "
            f"kernel_owner={owner} | opp=synthetic | prev_class=分類保留")


def _rule_text(cells, rounds=5, script=None, ts="2000-01-01T00:00:00Z"):
    s = script or self_sha256()
    return "\n".join([f"RULE fixed-at-utc {ts}", "合成データ用 RULE（self-test）", "", "## PARAMS",
                      f"fixed-at-utc: {ts}", "issue: 1", f"rounds: {rounds}", "tree_sha: " + "a" * 40,
                      "tree_fingerprint: " + "b" * 64, f"aggregate_script_sha256: {s}",
                      "reach_threshold: 1.00", "noise_band: NB-RANGE5"] + cells) + "\n"


def _build(root, cells, per_round, gates, script=None):
    """per_round: ラウンドごとの {(cell,q): (value,clock)}。gates: ラウンドごとの語。"""
    with open(os.path.join(root, "RULE.txt"), "w", encoding="utf-8") as f:
        f.write(_rule_text(cells, rounds=len(gates), script=script))
    for i, (g, rows) in enumerate(zip(gates, per_round), 1):
        d = os.path.join(root, f"round{i}")
        os.makedirs(d)
        with open(os.path.join(d, "gate.txt"), "w") as f:
            f.write(f"{g} synthetic\n")
        with open(os.path.join(d, "values.tsv"), "w") as f:
            f.write("cell\tquantity\tvalue\tclock\n")
            for (c, q), (v, ck) in rows.items():
                f.write(f"{c}\t{q}\t{v}\t{ck}\n")


def _rows(cid, fw=100, fresh=None, ow=100, fk=None, ok=None, ck=("cuda_event", "cuda_event")):
    r = {(cid, "fandhe_window"): (fw, "-"), (cid, "opp_window"): (ow, "-")}
    if fresh is not None:
        r[(cid, "fandhe_fresh_window")] = (fresh, "-")
    if fk is not None:
        r[(cid, "fandhe_pure_kernel")] = (fk, ck[0])
        r[(cid, "opp_pure_kernel")] = (ok, ck[1])
    return r


def _run(cells, per_round, gates, script=None):
    with tempfile.TemporaryDirectory() as t:
        _build(t, cells, per_round, gates, script)
        return aggregate(t)


def self_test():
    # 7 経路（窓・ハード・制約・保留・時計不一致・ドリフト・確度）を 1 回の集計で確認
    cells = [
        _cell_line("X-TST-A1", forms="F1,F2", sr=0.84),
        _cell_line("X-TST-B1", owner="hardware", forms="F1", sr=0.9),
        _cell_line("X-TST-B2", owner="constraint", forms="F1", sr=0.9),
        _cell_line("X-TST-C1", forms="F1,F2", sr=0.6),
        _cell_line("X-TST-E1", forms="F1", sr=1.0),
        _cell_line("X-TST-F1", forms="F2", sr=0.8),
    ]
    base = {}
    base.update(_rows("X-TST-A1", fw=100, fresh=70, ow=84, fk=50, ok=40))        # F2=84/70=1.2
    base.update(_rows("X-TST-B1", fw=100, ow=90, fk=50, ok=30))                  # F1=90/(100-50+30)=1.125
    base.update(_rows("X-TST-B2", fw=100, ow=90, fk=50, ok=30))
    base.update(_rows("X-TST-C1", fw=100, fresh=95, ow=60, fk=50, ok=45))        # F2=F1=60/95<1
    base.update(_rows("X-TST-E1", fw=100, ow=90, fk=50, ok=30, ck=("gpu_timestamp", "wall")))
    base.update(_rows("X-TST-F1", fw=100, fresh=40, ow=50))                      # F2=1.25・観測0.5 vs 0.8
    md, tsv = _run(cells, [dict(base) for _ in range(5)], ["PASS"] * 5)
    rows = {l.split("\t")[0]: l.split("\t") for l in tsv.splitlines()[1:]}
    assert rows["X-TST-A1"][3:5] == [CLASS_WINDOW, "高"] and rows["X-TST-A1"][2] == "1.2000", rows["X-TST-A1"]
    assert rows["X-TST-B1"][3:5] == [CLASS_HW, "高"] and rows["X-TST-B1"][1] == "1.1250"
    assert rows["X-TST-B2"][3:5] == [CLASS_OWN, "高"]
    assert rows["X-TST-C1"][3:5] == [CLASS_HOLD, "高"], rows["X-TST-C1"]
    assert rows["X-TST-E1"][3] == CLASS_HOLD and rows["X-TST-E1"][4] == "低"
    assert "時計不一致" in rows["X-TST-E1"][5] and rows["X-TST-E1"][1] == "-"
    # 時計不一致かつ純カーネルが窓を超える入力は異常終了せず分類保留になる（同一時計の超過は従来どおり異常終了）
    ce = [_cell_line("X-TST-E2", forms="F1", sr=1.0)]
    re_ = _run(ce, [_rows("X-TST-E2", fw=100, ow=100, fk=120, ok=30, ck=("gpu_timestamp", "mps_event"))] * 5, ["PASS"] * 5)[1]
    assert CLASS_HOLD in re_ and "-" == re_.splitlines()[1].split("\t")[1], re_
    # 時計が揃った窓超過は例外にせず F1 比較不能として扱う（#4146 の試走。所有者承認 2026-11-11）。
    # round 3 だけ超過・他は通常（F1 = 100/(100-50+30) = 1.25）
    def over_rounds(over_row, forms="F1,F2", fresh=None):
        good = _rows("X-TST-W1", fw=100, fresh=fresh, ow=100, fk=50, ok=30)
        return [over_row if i == 2 else good for i in range(5)]
    cw = [_cell_line("X-TST-W1", forms="F1,F2", sr=1.0)]
    # (a) fandhe 側だけ 1 ラウンド超過・F2 = 100/80 = 1.25 が閾値以上 → F1 比較不能でも「計測窓の定義」
    ra = _run(cw, over_rounds(_rows("X-TST-W1", fw=100, fresh=80, ow=100, fk=120, ok=30), fresh=80), ["PASS"] * 5)
    la = ra[1].splitlines()[1].split("\t")
    assert la[1] == "-" and la[3] == CLASS_WINDOW, la
    assert "3 窓超過（fandhe）" in ra[0] and "120.0000 ms（cuda_event）／窓 100.0000 ms" in ra[0], ra[0]
    # (b) F2 = 100/120 < 閾値 → 「分類保留」・確度低・理由に窓超過と元のラウンド番号と側
    rb = _run(cw, over_rounds(_rows("X-TST-W1", fw=100, fresh=120, ow=100, fk=120, ok=30), fresh=120), ["PASS"] * 5)
    lb = rb[1].splitlines()[1].split("\t")
    assert lb[1] == "-" and lb[3:5] == [CLASS_HOLD, "低"], lb
    assert "F1 の窓超過（純カーネル > 窓。round 3 fandhe）" in lb[5] and "時計不一致" not in lb[5], lb
    # (c) opp 側の超過（F1 のみ宣言）
    cc = [_cell_line("X-TST-W2", forms="F1", sr=1.0)]
    good_c = _rows("X-TST-W2", fw=100, ow=100, fk=50, ok=30)
    pc = [_rows("X-TST-W2", fw=100, ow=100, fk=50, ok=130) if i == 0 else good_c for i in range(5)]
    rc = _run(cc, pc, ["PASS"] * 5)
    lc = rc[1].splitlines()[1].split("\t")
    assert lc[3:5] == [CLASS_HOLD, "低"] and "round 1 opp" in lc[5], lc
    assert "1 窓超過（opp）" in rc[0], rc[0]
    # 超過が両側・複数ラウンドでも元の番号で併記し、時計不一致のラウンドと共存できる
    pe = [_rows("X-TST-W2", fw=100, ow=100, fk=150, ok=130) if i == 3 else good_c for i in range(5)]
    pe[1] = _rows("X-TST-W2", fw=100, ow=100, fk=50, ok=30, ck=("wall", "gpu_timestamp"))
    le = _run(cc, pe, ["PASS"] * 5)[1].splitlines()[1].split("\t")
    assert "F1 の時計不一致" in le[5] and "round 4 fandhe・opp" in le[5], le
    # 部分通過（round2 が TIMEOUT）では超過ラウンドの元番号（round 3）を使う
    pp = [_rows("X-TST-W2", fw=100, ow=100, fk=150, ok=30) if i == 2 else good_c for i in range(5)]
    lp = _run(cc, pp, ["PASS", "TIMEOUT", "PASS", "PASS", "PASS"])[1].splitlines()[1].split("\t")
    assert "round 3 fandhe" in lp[5], lp
    assert ra == _run(cw, over_rounds(_rows("X-TST-W1", fw=100, fresh=80, ow=100, fk=120, ok=30), fresh=80), ["PASS"] * 5), "non-deterministic"
    # (d) 超過が無ければ従来どおり（F1 = 1.25 で「ハードの上限」・確度高）
    rd = _run([_cell_line("X-TST-W3", forms="F1")], [_rows("X-TST-W3", fw=100, ow=100, fk=50, ok=30)] * 5, ["PASS"] * 5)
    ld = rd[1].splitlines()[1].split("\t")
    assert ld[1] == "1.2500" and ld[3:5] == [CLASS_HW, "高"] and "窓超過" not in rd[0], ld
    # 時計不一致の run は両側の純カーネル値と時計タグを集計表に併記する（RULE §4）
    assert "50.0000 ms（gpu_timestamp）" in md and "30.0000 ms（wall）" in md, md
    assert "ドリフト" in rows["X-TST-F1"][5] and "ドリフト" not in rows["X-TST-A1"][5]
    assert md == _run(cells, [dict(base) for _ in range(5)], ["PASS"] * 5)[0], "non-deterministic"
    # 5 本が閾値をまたぐ（F2 = .98 1.02 1.0 1.04 .96、中央値 1.0 → 窓だが確度低）
    cd = [_cell_line("X-TST-D1", forms="F2", sr=1.0)]
    pr = [_rows("X-TST-D1", fw=100, fresh=100, ow=o) for o in (98, 102, 100, 104, 96)]
    r = _run(cd, pr, ["PASS"] * 5)[1].splitlines()[1].split("\t")
    assert r[3:5] == [CLASS_WINDOW, "低"] and r[5].startswith("確度低"), r
    # 最小が閾値ちょうど（境界）もまたぐ扱い
    pr = [_rows("X-TST-D1", fw=100, fresh=100, ow=o) for o in (100, 102, 103, 104, 105)]
    assert _run(cd, pr, ["PASS"] * 5)[1].splitlines()[1].split("\t")[4] == "低"
    # TIMEOUT 1 本は partial・確度低・表題「参考」
    c1 = [_cell_line("X-TST-A1", forms="F2", sr=1.2)]
    pr = [_rows("X-TST-A1", fw=100, fresh=70, ow=84) for _ in range(5)]
    md3, t3 = _run(c1, pr, ["PASS", "PASS", "TIMEOUT", "PASS", "PASS"])
    assert "status: partial" in md3 and "（参考）" in md3
    assert t3.splitlines()[1].split("\t")[4] == "低" and "部分通過" in t3
    # 部分通過でも表の run は元のラウンド番号（round3 が TIMEOUT なら 1,2,4,5）
    assert [l.split("|")[1].strip() for l in md3.splitlines() if l.startswith("| ") and l.split("|")[1].strip().isdigit()] == ["1", "2", "4", "5"], md3
    # F1 で分類しても F2 が閾値をまたぐなら確度低（F2 = .98 .99 .99 1.01 1.02、F1 は全 run 1.1）
    c5 = [_cell_line("X-TST-G1", forms="F1,F2", sr=1.0)]
    pr = [_rows("X-TST-G1", fw=100, fresh=100, ow=o, fk=50, ok=40 + (o - 100)) for o in (98, 99, 99, 101, 102)]
    r = _run(c5, pr, ["PASS"] * 5)[1].splitlines()[1].split("\t")
    assert r[3] == CLASS_HW and r[4] == "低", r

    # 拒否ケース
    c2 = [_cell_line("X-TST-A1", forms="F1,F2", sr=0.84)]
    ok_rows = lambda: [_rows("X-TST-A1", fw=100, fresh=70, ow=84, fk=50, ok=40) for _ in range(5)]

    def mut(fn):
        def go():
            pr = ok_rows()
            fn(pr)
            _run(c2, pr, ["PASS"] * 5)
        return go
    A = "X-TST-A1"
    _expect_err(mut(lambda p: p[0].pop((A, "fandhe_fresh_window"))), "missing quantity")
    _expect_err(mut(lambda p: p[1].update({("X-TST-ZZ", "opp_window"): (1, "-")})), "unknown cell")
    _expect_err(mut(lambda p: p[1].update({(A, "bogus"): (1, "-")})), "unknown quantity")
    _expect_err(mut(lambda p: p[1].update({(A, "opp_window"): (0, "-")})), "zero")
    _expect_err(mut(lambda p: p[1].update({(A, "opp_window"): (-1, "-")})), "negative")
    _expect_err(mut(lambda p: p[1].update({(A, "opp_window"): (float("inf"), "-")})), "inf")
    _expect_err(mut(lambda p: p[1].update({(A, "opp_window"): (float("nan"), "-")})), "nan")
    mut(lambda p: p[1].update({(A, "fandhe_pure_kernel"): (200, "cuda_event")}))()  # 窓超過は例外にならない（上の (a)〜(d)）
    _expect_err(mut(lambda p: p[1].update({(A, "opp_pure_kernel"): (0, "cuda_event")})), "zero kernel")
    _expect_err(mut(lambda p: p[1].update({(A, "opp_pure_kernel"): (40, "-")})), "kernel without clock")
    _expect_err(mut(lambda p: p[1].update({(A, "opp_window"): (84, "wall")})), "clock on window")
    c_f2 = [_cell_line(A, forms="F2", sr=1.2)]
    _expect_err(lambda: _run(c_f2, [_rows(A, fw=100, fresh=70, ow=84, fk=50, ok=40)] * 5, ["PASS"] * 5),
                "quantity not needed by declared formulas")
    with tempfile.TemporaryDirectory() as t:
        _build(t, c2, ok_rows(), ["PASS"] * 5)
        with open(os.path.join(t, "round2", "values.tsv"), "a") as f:
            f.write(f"{A}\topp_window\t84\t-\n")
        _expect_err(lambda: aggregate(t), "duplicate row")
    with tempfile.TemporaryDirectory() as t:
        _build(t, c2, ok_rows(), ["PASS"] * 5)
        os.rename(os.path.join(t, "round3"), os.path.join(t, "round3x"))
        _expect_err(lambda: aggregate(t), "missing round dir")
    _expect_err(lambda: _run(c2, ok_rows(), ["TIMEOUT"] * 5), "zero PASS")
    _expect_err(lambda: _run(c2, ok_rows(), ["PASS"] * 4 + ["MAYBE"]), "bad gate word")
    _expect_err(lambda: _run(c2, ok_rows(), ["PASS"] * 5, script="c" * 64), "script hash mismatch")
    _expect_err(lambda: _run(c2, ok_rows(), ["PASS"] * 5, script="c" * 63), "script hash digits")
    good = _rule_text(c2)
    for label, bad in [
        ("bad timestamp", good.replace("2000-01-01T00:00:00Z", "2000-01-01 00:00", 1)),
        ("sha digits", good.replace("a" * 40, "a" * 39)),
        ("placeholder", good.replace("issue: 1", "issue: <N>")),
        ("unknown key", good + "bogus: 1\n"),
        ("duplicate key", good + "issue: 2\n"),
        ("noise band", good.replace("NB-RANGE5", "NB-SEP")),
        ("no cell", good.replace(c2[0] + "\n", "")),
        ("bad cell id", good.replace("X-TST-A1", "xx")),
        ("bad formulas", good.replace("formulas=F1,F2", "formulas=F3")),
        ("bad owner", good.replace("kernel_owner=hardware", "kernel_owner=other")),
        ("ts mismatch", good.replace("fixed-at-utc: 2000-01-01T00:00:00Z", "fixed-at-utc: 2000-01-02T00:00:00Z")),
        ("marker", good + "【記入】\n"),
    ]:
        _expect_err(lambda b=bad: parse_rule(b), label)

    # 同梱雛形は --check-template を通り、非雛形は拒否される
    here = os.path.dirname(os.path.abspath(__file__))
    tpl = os.path.join(here, "RULE.template.txt")
    if os.path.exists(tpl):  # スクリプトだけをコピーした配置では雛形が無い（README 手順 1 は両方置くが依存はしない）
        check_template(tpl)
    with tempfile.TemporaryDirectory() as t:
        p = os.path.join(t, "bad.txt")
        with open(p, "w") as f:
            f.write(_rule_text(c2))
        _expect_err(lambda: check_template(p), "non-template accepted")

    # 指紋: 決定的・除外パス（target/・.git・対象外ルート）は不変・内容変更で変わる
    with tempfile.TemporaryDirectory() as t:
        os.makedirs(os.path.join(t, "crates", "x", "target"))
        os.makedirs(os.path.join(t, "scripts", "bench"))
        for rel, body in [("crates/x/a.rs", "a"), ("scripts/bench/b.py", "b"), ("Cargo.toml", "c")]:
            with open(os.path.join(t, rel), "w") as f:
                f.write(body)
        fp1 = fingerprint(t)
        with open(os.path.join(t, "crates", "x", "target", "junk"), "w") as f:
            f.write("junk")
        os.makedirs(os.path.join(t, ".git"))
        with open(os.path.join(t, "other.txt"), "w") as f:
            f.write("outside roots")
        assert fingerprint(t) == fp1 and fp1[1] == 3 and HEX64.match(fp1[0])
        with open(os.path.join(t, "Cargo.toml"), "w") as f:
            f.write("changed")
        assert fingerprint(t)[0] != fp1[0]
    print("self-test ok")


def main(argv):
    a = argv[1:]
    opt = lambda name, default=None: a[a.index(name) + 1] if name in a else default
    try:
        if "--self-test" in a:
            self_test()
        elif "--check-template" in a:
            i = a.index("--check-template")
            p = a[i + 1] if i + 1 < len(a) and not a[i + 1].startswith("--") else \
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "RULE.template.txt")
            check_template(p)
            print("template ok")
        elif "--fingerprint" in a:
            h, n, listing = fingerprint(opt("--fingerprint"))
            if "--list-out" in a:
                with open(opt("--list-out"), "w", encoding="utf-8") as f:
                    f.write(listing)
            print(f"fingerprint={h} files={n}")
        elif "--dir" in a:
            d = opt("--dir")
            md, tsv = aggregate(d, opt("--rule", "RULE.txt"))
            with open(os.path.join(d, opt("--out", "aggregate.md")), "w", encoding="utf-8") as f:
                f.write(md)
            with open(os.path.join(d, opt("--tsv", "aggregate.tsv")), "w", encoding="utf-8") as f:
                f.write(tsv)
            sys.stdout.write(md)
        else:
            print(__doc__)
            return 2
    except AggError as e:
        print(f"AGGREGATE-ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
