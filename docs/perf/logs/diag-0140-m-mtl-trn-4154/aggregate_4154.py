#!/usr/bin/env python3
"""M-MTL-TRN（Mac mini・train Metal・相手 PyTorch MPS）の窓を wall 3 区間に分けて比べ、A/B の要否を判定する（#4154）。

役割: scripts/bench/framework-compare/torch_mps_train_diag.py と fandhe 側ハーネスが書いた raw/ を、RULE.txt の項 8
（`### DECIDE-4154` ブロック）で測る前に固定した式どおりに集計する。区間は発行 I・待ち S・残差 R（すべて wall）。
RULE の `## PARAMS` は同梱の aggregate_diag.py（雛形 diag-0140-template のコピーに時計不一致時の分類保留の修正を加えたもの）の形式で、F1（純カーネル置換）の
集計は `--write-values` で作る values.tsv を aggregate_diag.py に渡して行う（時計が揃わなければ分類保留になる）。
実測値は含まない。

入力（<dir>/round{1..N}/）:
  gate.txt   1 語目が PASS／TIMEOUT／INVALID
  raw/a0.jsonl  bench-fandhe（framework=fandhe-ai・task=train・device=metal・size=64・mode=reuse）の 1 行
  raw/b0.jsonl・b1.jsonl  resident_chain_breakdown_3612_train_b0／b1 の JSONL（各 1 行。wait_max=1・command_buffers_max=1）
  raw/tw.jsonl  bench_py.py（framework=pytorch・task=train・device=metal・size=64）の 1 行
  raw/ts.jsonl・te.jsonl・tf.jsonl  torch_mps_train_diag.py measure の split／event／floor 腕
出力: aggregate_4154.md／aggregate_4154.tsv。

厳格パース: 欠損・重複・未知・非有限・0・負・プレースホルダ残存・sha 不一致・round 不足・PASS 0 本はすべて AggError（終了コード 1）。
標準ライブラリのみ。`python3 -I` で実行すること。

使い方:
  python3 -I aggregate_4154.py --dir <記録先> [--rule RULE.txt]
  python3 -I aggregate_4154.py --write-values <round ディレクトリ>   # raw/ から values.tsv（aggregate_diag.py 用）を作る
  python3 -I aggregate_4154.py --self-test
  python3 -I aggregate_4154.py --check-template [RULE.template.txt]
  python3 -I aggregate_4154.py --check-rule <RULE.txt>
"""
import hashlib
import importlib.util
import json
import math
import os
import re
import statistics
import sys
import tempfile

ISSUE = 4154
CELL = "M-MTL-TRN"
HERE = os.path.dirname(os.path.abspath(__file__))
DIAG_REL = "scripts/bench/framework-compare/torch_mps_train_diag.py"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
DECIDE_BEGIN, DECIDE_END = "### DECIDE-4154", "### END-DECIDE-4154"
DECIDE_KEYS = ["share_threshold", "bench_py_sha256", "torch_version", "diag_script_sha256", "aggregate_4154_sha256"]
# torch_mps_train_diag.py の ARM_ITEMS と同一（テストで突き合わせる）。
ARM_ITEMS = {
    "split": {"issue_wall": "wall", "sync_wall": "wall", "step_wall": "wall"},
    "event": {"gpu_span": "mps_event", "event_step_wall": "wall"},
    "floor": {"floor_empty_sync": "wall", "floor_roundtrip": "wall"},
}
ARM_FILE = {"split": "ts.jsonl", "event": "te.jsonl", "floor": "tf.jsonl"}
# checksum の一致は確度の条件に使うだけの記録（tolerance ではない）。REQ-2 統一複合判定と同じ値を写す。
CK_REL, CK_ABS = 1e-3, 1e-5
DRIFT_LIMIT = 0.10


class AggError(Exception):
    pass


def load_diag():
    p = os.path.join(HERE, "aggregate_diag.py")
    spec = importlib.util.spec_from_file_location("aggregate_diag_pinned", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


AD = load_diag()
AggError = AD.AggError  # 雛形側の例外と同一クラスにして main の捕捉に統一する


def sha_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def self_sha256():
    return sha_file(os.path.abspath(__file__))


def read(path):
    if not os.path.isfile(path):
        raise AggError(f"missing file: {path}")
    with open(path, encoding="utf-8") as f:
        return f.read()


def fin_pos(v, what):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
        raise AggError(f"{what}: must be finite and > 0: {v!r}")
    return float(v)


def read_jsonl(path):
    rows = []
    for n, line in enumerate(read(path).splitlines(), 1):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError as e:
            raise AggError(f"{path}:{n}: bad json: {e}")
        if not isinstance(r, dict):
            raise AggError(f"{path}:{n}: not an object")
        rows.append(r)
    return rows


# ---------------------------------------------------------------- RULE
def parse_decide(text):
    lines = text.splitlines()
    b = [i for i, l in enumerate(lines) if l.strip() == DECIDE_BEGIN]
    e = [i for i, l in enumerate(lines) if l.strip() == DECIDE_END]
    if len(b) != 1 or len(e) != 1 or e[0] < b[0]:
        raise AggError("RULE: exactly one DECIDE-4154 block required")
    pi = [i for i, l in enumerate(lines) if l.strip() == "## PARAMS"]
    if len(pi) != 1 or e[0] > pi[0]:
        raise AggError("RULE: DECIDE-4154 block must precede '## PARAMS'")
    kv = {}
    for l in lines[b[0] + 1:e[0]]:
        if not l.strip() or l.lstrip().startswith("#"):
            continue
        if ":" not in l:
            raise AggError(f"DECIDE: not key: value: {l!r}")
        k, v = l.split(":", 1)
        k, v = k.strip(), v.strip()
        if k not in DECIDE_KEYS or k in kv:
            raise AggError(f"DECIDE: unknown/duplicate key {k!r}")
        kv[k] = v
    return kv


def parse_rule_full(path, check_files=True):
    text = read(path)
    rule = AD.parse_rule(text)  # 雛形側の検査（プレースホルダ・sha・キー過不足）
    if len(rule["cells"]) != 1 or rule["cells"][0]["id"] != CELL or rule["issue"] != ISSUE:
        raise AggError(f"PARAMS: issue={ISSUE} と単一セル {CELL} が必要")
    kv = parse_decide(text)
    for k in DECIDE_KEYS:
        if k not in kv:
            raise AggError(f"DECIDE: missing key {k!r}")
        if "<" in kv[k] or ">" in kv[k]:
            raise AggError(f"DECIDE: unreplaced placeholder in {k}")
    thr = AD.parse_float(kv["share_threshold"], "share_threshold")
    if not 0 < thr < 1:
        raise AggError("DECIDE: share_threshold must be in (0,1)")
    for k in ("bench_py_sha256", "diag_script_sha256", "aggregate_4154_sha256"):
        if not HEX64.match(kv[k]):
            raise AggError(f"DECIDE: {k} must be 64 hex")
    if kv["aggregate_4154_sha256"] != self_sha256():
        raise AggError("DECIDE: aggregate_4154_sha256 mismatch (script changed after RULE was fixed)")
    if not re.match(r"^\d+\.\d+\.\d+", kv["torch_version"]):
        raise AggError("DECIDE: torch_version malformed")
    diag = os.path.join(HERE, "..", "..", "..", "..", DIAG_REL)
    if check_files and os.path.isfile(diag) and sha_file(diag) != kv["diag_script_sha256"]:
        raise AggError("DECIDE: diag_script_sha256 mismatch with torch_mps_train_diag.py")
    rule["decide"] = {"thr": thr, "bench": kv["bench_py_sha256"], "torch": kv["torch_version"],
                      "diag": kv["diag_script_sha256"]}
    return rule


def check_template(path):
    text = read(path)
    AD.check_template(path)
    kv = parse_decide(text)
    if set(kv) != set(DECIDE_KEYS):
        raise AggError("template: DECIDE keys mismatch")
    for k in ("bench_py_sha256", "diag_script_sha256", "aggregate_4154_sha256", "torch_version"):
        if not re.search(r"<[^<>]+>", kv[k]):
            raise AggError(f"template: DECIDE {k} is not a placeholder")


# ---------------------------------------------------------------- 入力
def one(rows, pred, what):
    hit = [r for r in rows if pred(r)]
    if len(hit) != 1:
        raise AggError(f"{what}: expected exactly 1 row, got {len(hit)}")
    return hit[0]


def load_diag_arm(path, arm, run, decide):
    rows = read_jsonl(path)
    items = ARM_ITEMS[arm]
    if len(rows) != len(items):
        raise AggError(f"{path}: expected {len(items)} rows, got {len(rows)}")
    out, cks = {}, set()
    for r in rows:
        for k in ("issue", "arm", "item", "clock", "status", "tag", "median_s", "run", "torch_version",
                  "bench_py_sha256", "diag_script_sha256", "checksum"):
            if k not in r:
                raise AggError(f"{path}: key missing: {k}")
        if r["issue"] != ISSUE or r["arm"] != arm or r["run"] != run:
            raise AggError(f"{path}: issue/arm/run mismatch")
        it = r["item"]
        if it not in items or it in out:
            raise AggError(f"{path}: unknown/duplicate item {it!r}")
        if r["clock"] != items[it]:
            raise AggError(f"{path}: clock mismatch for {it}")
        if r["bench_py_sha256"] != decide["bench"] or r["diag_script_sha256"] != decide["diag"]:
            raise AggError(f"{path}: provenance sha mismatch with RULE")
        if not str(r["torch_version"]).startswith(decide["torch"]):
            raise AggError(f"{path}: torch_version mismatch with RULE")
        if r["status"] == "ok":
            out[it] = fin_pos(r["median_s"], f"{path}:{it}") * 1e6  # us
        elif r["status"] == "unavailable":
            if r["median_s"] is not None or r["tag"] != "未測定":
                raise AggError(f"{path}: unavailable row must be null/未測定")
            if not isinstance(r.get("reason"), str) or not r["reason"]:
                raise AggError(f"{path}: unavailable row needs a reason")
            out[it] = None
        else:
            raise AggError(f"{path}: bad status")
        if r["checksum"] is not None:
            cks.add(fin_pos(abs(r["checksum"]) or 1e-30, f"{path}:checksum") * (1 if r["checksum"] >= 0 else -1))
    if len(cks) > 1:
        raise AggError(f"{path}: checksum differs between rows")
    if set(out) != set(items):
        raise AggError(f"{path}: item set incomplete")
    if arm != "event" and any(v is None for v in out.values()):
        raise AggError(f"{path}: unavailable not allowed in arm {arm}")
    # event 腕の unavailable は「プローブで Event API が無い・使えない」場合だけで、両 item が揃って unavailable になる
    # （torch_mps_train_diag.py の run_event）。本計測中の失敗は腕ごと失敗し行を書かない。片方だけの unavailable は止める。
    if arm == "event" and len({v is None for v in out.values()}) > 1:
        raise AggError(f"{path}: event arm must be all ok or all unavailable")
    return out, (next(iter(cks)) if cks else None)


def load_round(rdir, run, rule):
    raw = os.path.join(rdir, "raw")
    d = rule["decide"]
    a0 = one(read_jsonl(os.path.join(raw, "a0.jsonl")),
             lambda r: r.get("framework") == "fandhe-ai" and r.get("task") == "train" and r.get("device") == "metal"
             and r.get("size") == 64 and r.get("mode") == "reuse", "a0.jsonl")
    tw = one(read_jsonl(os.path.join(raw, "tw.jsonl")),
             lambda r: r.get("framework") == "pytorch" and r.get("task") == "train" and r.get("device") == "metal"
             and r.get("size") == 64, "tw.jsonl")
    b = {}
    for arm in ("b0", "b1"):
        r = one(read_jsonl(os.path.join(raw, f"{arm}.jsonl")), lambda r, arm=arm: r.get("arm") == arm and r.get("task") == "train",
                f"{arm}.jsonl")
        if r.get("wait_max") != 1 or r.get("command_buffers_max") != 1:
            raise AggError(f"{arm}.jsonl: wait_max/command_buffers_max must be 1 (hard condition)")
        b[arm] = r
    k = lambda r, key, w: fin_pos(r.get(key), w)
    b1 = b["b1"]
    ts, ts_ck = load_diag_arm(os.path.join(raw, "ts.jsonl"), "split", run, d)
    te, _ = load_diag_arm(os.path.join(raw, "te.jsonl"), "event", run, d)
    tf, _ = load_diag_arm(os.path.join(raw, "tf.jsonl"), "floor", run, d)
    return {
        "W_f": k(a0, "median_s", "a0") * 1e6, "W_t": k(tw, "median_s", "tw") * 1e6,
        "I_f": k(b1, "prep_us", "b1.prep") + k(b1, "encode_us", "b1.encode"),
        "S_f": k(b1, "commit_wait_us", "b1.commit_wait"), "gpu_busy": k(b1, "gpu_busy_us", "b1.gpu_busy"),
        "host_fixed": k(b1, "host_fixed_us", "b1.host_fixed"), "B0_total": k(b["b0"], "total_us", "b0.total"),
        "B1_total": k(b1, "total_us", "b1.total"),
        "I_t": ts["issue_wall"], "S_t": ts["sync_wall"], "G_t": te["gpu_span"],
        "F_empty": tf["floor_empty_sync"], "F_trip": tf["floor_roundtrip"],
        "ck_tw": fin_pos(abs(tw.get("checksum")) or 1e-30, "tw.checksum") * (1 if tw.get("checksum", 0) >= 0 else -1),
        "ck_ts": ts_ck,
    }


# ---------------------------------------------------------------- 判定
def ck_match(a, b):
    if a is None or b is None:
        return False
    diff = abs(a - b)
    return diff < CK_ABS or diff / max(abs(a), abs(b), 1e-12) < CK_REL


def derive(m):
    r = dict(m)
    r["R_f"] = m["W_f"] - m["I_f"] - m["S_f"]
    r["R_t"] = m["W_t"] - m["I_t"] - m["S_t"]
    r["obs"] = m["W_t"] / m["W_f"]
    r["delta"] = m["W_f"] - m["W_t"]
    r["dI"], r["dS"], r["dR"] = m["I_f"] - m["I_t"], m["S_f"] - m["S_t"], r["R_f"] - r["R_t"]
    return r


def decide(rs, thr, all_pass):
    """RULE 項 8 の優先順位つき判定。rs は derive 済みの PASS ラウンド。戻り値 (code, reason, high)。"""
    obs = [r["obs"] for r in rs]
    lo, hi = min(obs), max(obs)
    cks = all(ck_match(r["ck_tw"], r["ck_ts"]) for r in rs)
    high = all_pass and cks
    if lo <= 1.0 <= hi:
        return "N", "ノイズ帯 NB-RANGE5 が 1.0 を含む（A/B を立てない）", high
    if lo > 1.0:
        return "W", "帯の全体が 1.0 より上（このセッションでは 1 位。A/B を立てない）", high
    hit = []
    for code, key in (("A-S", "dS"), ("A-I", "dI"), ("A-R", "dR")):
        if all(r["delta"] > 0 and r[key] >= thr * r["delta"] for r in rs):
            hit.append(code)
    if len(hit) == 1:
        return hit[0], {"A-S": "待ち側", "A-I": "発行側", "A-R": "窓の外側（facade・bench の固定費）"}[hit[0]] \
            + f"の差が全ラウンドで差の {thr:.0%} 以上（A/B 候補）", high
    if len(hit) > 1:
        return "H", f"複数区間 {','.join(hit)} が同時に半分以上（符号相殺）。分類保留・記録のみ", high
    return "H", "どの区間も全ラウンドで閾値に届かない。分類保留・記録のみ", high


def fmt(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def aggregate(dirp, rule_name="RULE.txt"):
    rule = parse_rule_full(os.path.join(dirp, rule_name))
    n = rule["rounds"]
    gates, rs, nos = [], [], []
    for i in range(1, n + 1):
        rd = os.path.join(dirp, f"round{i}")
        if not os.path.isdir(rd):
            raise AggError(f"missing round dir: {rd}")
        g = AD.parse_gate(os.path.join(rd, "gate.txt"))
        gates.append(g)
        if g == "PASS":
            nos.append(i)
            rs.append(derive(load_round(rd, i, rule)))
    if not rs:
        raise AggError("no PASS round")
    all_pass = all(g == "PASS" for g in gates)
    thr = rule["decide"]["thr"]
    code, why, high = decide(rs, thr, all_pass)
    med = lambda k: statistics.median(r[k] for r in rs)
    mobs = med("obs")
    sb = rule["cells"][0]["scoreboard_ratio"]
    reason = ("" if high else "確度低: ") + why + ("" if all_pass else "（部分通過）")
    if not all(ck_match(r["ck_tw"], r["ck_ts"]) for r in rs):
        reason += "／TW の checksum と TS の最終 loss が不一致の round あり"
    drift = abs(mobs / sb - 1.0)
    if drift > DRIFT_LIMIT:
        reason += f"／ドリフト: 観測比 {mobs:.4f} がスコアボード {sb:.4f} から {drift * 100:.1f}% 乖離"
    md = [f"# {CELL} wall 3 区間の比較 #{ISSUE}" + ("" if all_pass else "（参考）"), "",
          f"- ゲート: {' '.join(gates)}（PASS {len(rs)}/{n}）",
          f"- RULE: fixed-at-utc={rule['ts']} tree_sha={rule['tree_sha']} share_threshold={thr:.2f}", "",
          "単位 µs/step（各腕の中央値）。fandhe 側 I=prep+encode・S=commit_wait（#3612 B1）、R=窓−I−S。",
          "", "| run | W_f | W_t | 観測比 | I_f | I_t | S_f | S_t | R_f | R_t | ΔI | ΔS | ΔR | Δ |", "|" + "---|" * 14]
    for no, r in zip(nos, rs):
        md.append(f"| {no} | {fmt(r['W_f'])} | {fmt(r['W_t'])} | {r['obs']:.4f} | {fmt(r['I_f'])} | {fmt(r['I_t'])} | "
                  f"{fmt(r['S_f'])} | {fmt(r['S_t'])} | {fmt(r['R_f'])} | {fmt(r['R_t'])} | {fmt(r['dI'])} | "
                  f"{fmt(r['dS'])} | {fmt(r['dR'])} | {fmt(r['delta'])} |")
    obs = [r["obs"] for r in rs]
    md += ["", f"- 観測比の中央値 {mobs:.4f}／NB-RANGE5 {min(obs):.4f}〜{max(obs):.4f}",
           "", "補助（推定タグ。時計が違うため区間と足さない・確定に使わない）:", "",
           "| 量 | 中央値 µs | 時計 |", "|---|---|---|"]
    gt = [r["G_t"] for r in rs if r["G_t"] is not None]
    md += [f"| G_t（PyTorch MPS の GPU span） | {fmt(statistics.median(gt) if gt else None)} | mps_event |",
           f"| gpu_busy（fandhe B1） | {fmt(med('gpu_busy'))} | gpu_timestamp |",
           f"| F_empty（空の synchronize） | {fmt(med('F_empty'))} | wall |",
           f"| F_roundtrip（1 要素 add + item） | {fmt(med('F_trip'))} | wall |",
           f"| host_fixed（fandhe B1） | {fmt(med('host_fixed'))} | wall |",
           f"| B0 total／B1 total（観測負荷の記録） | {fmt(med('B0_total'))}／{fmt(med('B1_total'))} | wall |", "",
           f"- 判定: {code}／確度: {'高' if high else '低'}", f"- 理由: {reason}", ""]
    if code in ("A-S", "A-I", "A-R"):
        md.append("- 提案: A/B の追加起票を所有者に提案する（起票は承認後）。GPU 実行と同期固定費のどちらが主かは時計が揃わず確定しない。")
        md.append("")
    tsv = ["cell\tcode\tconfidence\tobs_median\treason", f"{CELL}\t{code}\t{'高' if high else '低'}\t{mobs:.4f}\t{reason}"]
    return "\n".join(md) + "\n", "\n".join(tsv) + "\n"


def write_values(rdir):
    """round の raw/ から aggregate_diag.py 用 values.tsv を作る。G_t が unavailable なら F1 は作れず停止する。"""
    out = os.path.join(rdir, "values.tsv")
    if os.path.exists(out):
        raise AggError(f"values.tsv exists (no overwrite): {out}")
    # provenance（sha・torch 版）は --dir 側の load_round が RULE と照合する。ここは値の写しだけを行う。
    raw = os.path.join(rdir, "raw")
    rows = {}
    for arm, f in (("split", "ts.jsonl"), ("event", "te.jsonl")):
        rs = read_jsonl(os.path.join(raw, f))
        rows[arm] = {r["item"]: r for r in rs}
    gt = rows["event"].get("gpu_span")
    if gt is None or gt.get("status") != "ok":
        raise AggError("G_t (gpu_span) unavailable: F1 の values.tsv は作れない（RULE を改訂して式を外す）")
    a0 = one(read_jsonl(os.path.join(raw, "a0.jsonl")), lambda r: r.get("framework") == "fandhe-ai" and r.get("task") == "train"
             and r.get("device") == "metal" and r.get("mode") == "reuse", "a0.jsonl")
    tw = one(read_jsonl(os.path.join(raw, "tw.jsonl")), lambda r: r.get("framework") == "pytorch" and r.get("task") == "train", "tw.jsonl")
    b1 = one(read_jsonl(os.path.join(raw, "b1.jsonl")), lambda r: r.get("arm") == "b1", "b1.jsonl")
    ms = lambda v, w: fin_pos(v, w)
    lines = ["cell\tquantity\tvalue\tclock",
             f"{CELL}\tfandhe_window\t{ms(a0['median_s'], 'a0') * 1000:.6f}\t-",
             f"{CELL}\topp_window\t{ms(tw['median_s'], 'tw') * 1000:.6f}\t-",
             f"{CELL}\tfandhe_pure_kernel\t{fin_pos(b1['gpu_busy_us'], 'gpu_busy') / 1000:.6f}\tgpu_timestamp",
             f"{CELL}\topp_pure_kernel\t{fin_pos(gt['median_s'], 'gpu_span') * 1000:.6f}\tmps_event"]
    with open(out, "x", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------- self-test
def _d(obs_w_t, dI, dS, dR, ok=True):
    """W_f=500・指定の区間差を持つ derive 済み round（合成値）。"""
    W_f = 500.0
    m = {"W_f": W_f, "W_t": obs_w_t, "I_f": 100.0 + dI, "I_t": 100.0, "S_f": 200.0 + dS, "S_t": 200.0,
         "gpu_busy": 1.0, "host_fixed": 1.0, "B0_total": 1.0, "B1_total": 1.0, "G_t": None, "F_empty": 1.0, "F_trip": 1.0,
         "ck_tw": 0.5, "ck_ts": 0.5 if ok else 0.9}
    r = derive(m)
    # dR は R_f−R_t。W_t から決まる値なので、合成では I_t/S_t を固定して整合する W_t を選ぶ前提（下で検査）
    return r


def _rs(wts, shares):
    out = []
    for w, (a, b, c) in zip(wts, shares):
        delta = 500.0 - w
        out.append(_d(w, a * delta, b * delta, c * delta))
    return out


def self_test():
    thr = 0.5
    # 恒等式 ΔI+ΔS+ΔR=Δ（合成: R_f−R_t は W と I,S から決まる）
    r = _d(480.0, 10.0, 6.0, 0.0)
    assert abs(r["dI"] + r["dS"] + r["dR"] - r["delta"]) < 1e-9
    # N: 帯が 1.0 を含む／境界ちょうど
    w = [500.0 * x for x in (0.98, 1.0, 1.01, 1.0, 0.99)]
    assert decide(_rs(w, [(0, 1, 0)] * 5), thr, True)[0] == "N"
    w = [500.0 * x for x in (0.9, 0.95, 1.0, 0.92, 0.91)]
    assert decide([_d(v, 0, 0, 0) for v in w], thr, True)[0] == "N"
    # W: 全体が 1.0 より上
    w = [500.0 * x for x in (1.02, 1.05, 1.03, 1.04, 1.06)]
    assert decide([_d(v, 0, 0, 0) for v in w], thr, True)[0] == "W"
    # 帯が全部 1.0 未満: A-S／A-I／A-R／H
    w = [480.0, 485.0, 490.0, 482.0, 488.0]
    # _rs は dI,dS,dR を直接与えるが derive の dR は W から決まるため、直接 r を上書きして分岐だけを検査する
    def mk(shares):
        rs = []
        for wv, (a, b, c) in zip(w, shares):
            rr = _d(wv, 0, 0, 0)
            d = rr["delta"]
            rr["dI"], rr["dS"], rr["dR"] = a * d, b * d, c * d
            rs.append(rr)
        return rs
    assert decide(mk([(0.1, 0.8, 0.1)] * 5), thr, True)[0] == "A-S"
    assert decide(mk([(0.8, 0.1, 0.1)] * 5), thr, True)[0] == "A-I"
    assert decide(mk([(0.1, 0.1, 0.8)] * 5), thr, True)[0] == "A-R"
    assert decide(mk([(0.3, 0.4, 0.3)] * 5), thr, True)[0] == "H"
    assert decide(mk([(0.5, 0.5, 0.0)] * 5), thr, True)[0] == "H"  # 境界ちょうど（=閾値）は両方当たり -> 複数で H
    assert decide(mk([(0.6, 0.6, -0.2)] * 5), thr, True)[0] == "H"  # 符号相殺
    sh = [(0.1, 0.8, 0.1)] * 4 + [(0.1, 0.4, 0.5)]
    assert decide(mk(sh), thr, True)[0] != "A-S"  # 1 round でも届かなければ A-S にならない
    # 確度
    assert decide(mk([(0.1, 0.8, 0.1)] * 5), thr, True)[2] is True
    assert decide(mk([(0.1, 0.8, 0.1)] * 5), thr, False)[2] is False
    bad = mk([(0.1, 0.8, 0.1)] * 5)
    bad[0]["ck_ts"] = 0.9
    assert decide(bad, thr, True)[2] is False
    # 入力の厳格検査
    for v in (0, -1, float("nan"), float("inf"), True, "1"):
        try:
            fin_pos(v, "x")
        except AggError:
            continue
        raise AssertionError(f"fin_pos をすり抜けた: {v!r}")
    # DECIDE ブロックの構文
    good = f"x\n{DECIDE_BEGIN}\nshare_threshold: 0.50\n{DECIDE_END}\n## PARAMS\n"
    assert parse_decide(good) == {"share_threshold": "0.50"}
    for t in (good.replace(DECIDE_END, ""), good + good, good.replace("share_threshold", "bogus"),
              f"{DECIDE_BEGIN}\n{DECIDE_END}\n"):
        try:
            parse_decide(t)
        except AggError:
            continue
        raise AssertionError("DECIDE の不正を通した")
    # 厳格ローダ: 合成 round を作って読み、1 つ壊して止まることを確認
    with tempfile.TemporaryDirectory() as t:
        _synth_round(t)
        rule = {"decide": {"bench": "b" * 64, "diag": "d" * 64, "torch": "2.14.0", "thr": 0.5}}
        m = load_round(t, 1, rule)
        assert m["I_f"] == 130.0 and m["S_f"] == 250.0 and m["I_t"] == 100.0
        with open(os.path.join(t, "raw", "b1.jsonl"), "w") as f:
            f.write(json.dumps(dict(_b("b1"), wait_max=7)) + "\n")
        try:
            load_round(t, 1, rule)
            raise AssertionError("wait_max!=1 を通した")
        except AggError:
            pass
    # event 腕: 両 item unavailable（理由つき）は読めて G_t=None、片方だけ・理由なしは止まる。G_t なしの values.tsv は止まる。
    una = lambda it, reason="torch.mps.Event が無い": dict(_drow("event", it, None), status="unavailable", tag="未測定",
                                                       q1_s=None, q3_s=None, checksum=None, reason=reason)
    for rows, ok in (([una("gpu_span"), una("event_step_wall")], True),
                     ([una("gpu_span"), _drow("event", "event_step_wall", 4e-4)], False),
                     ([_drow("event", "gpu_span", 2e-4), una("event_step_wall")], False),
                     ([una("gpu_span", ""), una("event_step_wall", "")], False),
                     ([dict(una("gpu_span"), reason=None), una("event_step_wall")], False)):
        with tempfile.TemporaryDirectory() as t:
            _synth_round(t)
            with open(os.path.join(t, "raw", "te.jsonl"), "w") as f:
                f.write("".join(json.dumps(r) + "\n" for r in rows))
            try:
                m = load_round(t, 1, rule)
            except AggError:
                assert not ok, f"正しい event 行を止めた: {rows}"
                continue
            assert ok, f"不正な event 行を通した: {rows}"
            assert m["G_t"] is None
            try:
                write_values(t)
                raise AssertionError("G_t unavailable の values.tsv を作った")
            except AggError:
                pass
            assert not os.path.exists(os.path.join(t, "values.tsv"))
    print("self-test OK")


def _b(arm):
    return {"task": "train", "arm": arm, "n": 80, "prep_us": 100.0, "encode_us": 30.0, "commit_wait_us": 250.0,
            "gpu_busy_us": 200.0, "host_fixed_us": 50.0, "readout_us": 5.0, "total_us": 400.0, "wait_max": 1,
            "command_buffers_max": 1, "encode_calls_max": 3, "loss_last": 0.5}


def _drow(arm, item, v, run=1):
    return {"issue": ISSUE, "arm": arm, "item": item, "clock": ARM_ITEMS[arm][item], "status": "ok", "tag": "実測",
            "median_s": v, "q1_s": v, "q3_s": v, "warmup": 20, "iters": 80, "run": run, "torch_version": "2.14.0",
            "bench_py_sha256": "b" * 64, "diag_script_sha256": "d" * 64, "checksum": 0.5 if arm != "floor" else None}


def _synth_round(t):
    os.makedirs(os.path.join(t, "raw"))
    w = lambda name, rows: open(os.path.join(t, "raw", name), "w").write("".join(json.dumps(r) + "\n" for r in rows))
    w("a0.jsonl", [{"framework": "fandhe-ai", "task": "train", "device": "metal", "size": 64, "mode": "reuse", "median_s": 5e-4}])
    w("tw.jsonl", [{"framework": "pytorch", "task": "train", "device": "metal", "size": 64, "median_s": 4.9e-4, "checksum": 0.5}])
    w("b0.jsonl", [_b("b0")])
    w("b1.jsonl", [_b("b1")])
    w("ts.jsonl", [_drow("split", "issue_wall", 1e-4), _drow("split", "sync_wall", 3e-4), _drow("split", "step_wall", 4e-4)])
    w("te.jsonl", [_drow("event", "gpu_span", 2e-4), _drow("event", "event_step_wall", 4e-4)])
    w("tf.jsonl", [_drow("floor", "floor_empty_sync", 1e-5), _drow("floor", "floor_roundtrip", 2e-5)])


def main(argv):
    def opt(name, default=None):
        i = argv.index(name) + 1 if name in argv else 0
        return argv[i] if 0 < i < len(argv) and not argv[i].startswith("--") else default
    try:
        if "--self-test" in argv:
            self_test()
        elif "--check-template" in argv:
            check_template(opt("--check-template") or os.path.join(HERE, "RULE.template.txt"))
            print("template OK")
        elif "--check-rule" in argv:
            p = opt("--check-rule")
            if not p:
                raise AggError("--check-rule needs a RULE.txt path")
            parse_rule_full(p)
            print("rule OK")
        elif "--write-values" in argv:
            write_values(opt("--write-values"))
            print("values.tsv written")
        elif "--dir" in argv:
            d = opt("--dir")
            md, tsv = aggregate(d, opt("--rule", "RULE.txt"))
            with open(os.path.join(d, "aggregate_4154.md"), "w", encoding="utf-8") as f:
                f.write(md)
            with open(os.path.join(d, "aggregate_4154.tsv"), "w", encoding="utf-8") as f:
                f.write(tsv)
            sys.stdout.write(md)
        else:
            print(__doc__)
            return 2
    except AggError as e:
        print(f"AGGREGATE ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
