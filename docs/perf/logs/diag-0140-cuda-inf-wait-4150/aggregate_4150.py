#!/usr/bin/env python3
"""#4150: G-CUDA-INF の kernel_wait（導出値）を GPU 実行と待ちに分ける診断の集計（stdlib のみ）。

役割: run_round.sh が作る round{i}/{gate.txt, raw/chain.txt} と固定済み RULE.txt を厳格にパースし、
  内訳（実測／推定タグ付き）・分類・確度を aggregate.md / aggregate.tsv に出す。record_only（合否判定ではない）。
  欠損・重複・未知キー・腕不足・反復数不一致・RULE のプレースホルダ残存・sha 不一致は AggError で止まる。
  DEVICE4150 ヘッダは厳密に解析し、round を round{i} の i と、warmup・iters を RULE と照合する（重複は拒否）。
  TIMEOUT／INVALID を 1 件でも含む系列は status=partial・表題「参考」とし、確度を出さない（RULE 項 3）。
  派生元: diag-0140-torch-gb10-mlp-4149/aggregate_4149.py（RULE 解析・指紋・テンプレート検査の作法）。

使い方（python3 -I で実行する）:
  aggregate_4150.py --self-test
  aggregate_4150.py --check-template [RULE.template.txt]
  aggregate_4150.py --check-rule RULE.txt
  aggregate_4150.py --fingerprint <tree-dir>
  aggregate_4150.py --dir <記録先> [--rule RULE.txt] [--out aggregate.md] [--tsv aggregate.tsv] [--opp-tsv opp.tsv]

--opp-tsv は #4149 の相手側実測（item<TAB>median_us 行。ヘッダ行 'item\\tmedian_us'）。無ければ相手欄は
unavailable と出し、推定値で埋めない。
"""

import hashlib
import math
import os
import re
import statistics
import sys
import tempfile

ISSUE = 4150
PARAM_KEYS = ["fixed-at-utc", "issue", "rounds", "tree_sha", "tree_fingerprint", "example_sha256",
              "aggregate_script_sha256", "warmup", "iters"]
HASH_KEYS = ("tree_fingerprint", "example_sha256", "aggregate_script_sha256")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
FP_ROOTS = ["crates", "scripts/bench", "Cargo.toml", "Cargo.lock", "rust-toolchain.toml"]
FP_SKIP = {"target", ".git"}
PLACEHOLDER_MARK = "【記入"
# example の出力キー（cuda_infer_chain_event_diag_4150.rs の format_line 呼び出しと同一。テストで突き合わせる）。
ARM_KEYS = {
    "P": ("h2d_upload_us", "l1_issue_us", "l2_issue_us", "download_us", "download_again_us",
          "kernel_wait_derived_us", "total_us"),
    "E": ("up_gpu_us", "l1_span_us", "l2_span_us", "chain_gpu_us", "download_us", "l1_issue_us", "l2_issue_us"),
    "I": ("l1_iso_us", "l2_iso_us", "l1_host_us", "l2_host_us"),
    "F": ("sync_idle_us", "launch_floor_1x1_us", "tiny_roundtrip_us", "event_record_pair_us", "event_pair_gpu_us"),
}
# 分類規則（RULE 項 5 と同一。事後に動かさない）。
PERTURBATION_LIMIT = 0.10
SHARE_EXEC_THRESHOLD = 0.5
# #4149 の cuda_infer／cuda_fixed／cuda_shapes の item 名。
OPP_ITEMS = ("issue", "readout_sync", "gpu_span_event", "mm_event_l1_fwd", "mm_event_l2_fwd",
             "sync_idle", "tiny_roundtrip", "launch_floor_add", "event_record_pair")
POSINT = r"(?:0|[1-9]\d*)"
LINE_RE = re.compile(r"^CHAIN4150 arm=([PEIF]) iter=(" + POSINT + r")((?: [a-z0-9_]+=-?\d+\.\d{3})+)$")
# example の DEVICE4150 行（println! の書式と同一。接頭辞だけでは受理しない）。
DEVICE_RE = re.compile(r"^DEVICE4150 compute_capability=(\d+)\.(\d+) round=(" + POSINT + r") warmup=(" + POSINT
                       + r") iters=(" + POSINT + r")$")
CHECKSUM_RE = re.compile(r"^CHECKSUM4150 value=(\d+)$")
GATE_TOKENS = ("PASS", "TIMEOUT", "INVALID")
ROUND_DIR_RE = re.compile(r"^round(\d+)$")


class AggError(Exception):
    pass


def self_sha256():
    with open(os.path.abspath(__file__), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def read(path):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError as e:
        raise AggError(f"cannot read {path}: {e}")


def split_params(text):
    marker = "\n## PARAMS\n"
    i = text.find(marker)
    if i < 0:
        raise AggError("RULE: '## PARAMS' block not found")
    return text[:i], text[i + len(marker):]


def parse_kv(body):
    kv = {}
    for ln in body.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        k, sep, v = ln.partition(":")
        if not sep:
            raise AggError(f"PARAMS: unparsable line: {ln!r}")
        k, v = k.strip(), v.strip()
        if k in kv:
            raise AggError(f"PARAMS: duplicate key {k}")
        kv[k] = v
    return kv


def parse_rule(path):
    text = read(path)
    if PLACEHOLDER_MARK in text:
        raise AggError("RULE: placeholder (【記入…) remains")
    head, body = split_params(text)
    m = re.match(r"^RULE fixed-at-utc (\S+)", head)
    if not m or not TS_RE.match(m.group(1)):
        raise AggError("RULE: first line must be 'RULE fixed-at-utc <UTC timestamp>'")
    kv = parse_kv(body)
    if set(kv) != set(PARAM_KEYS):
        raise AggError(f"PARAMS keys mismatch: missing={set(PARAM_KEYS) - set(kv)} extra={set(kv) - set(PARAM_KEYS)}")
    if not TS_RE.match(kv["fixed-at-utc"]) or kv["fixed-at-utc"] != m.group(1):
        raise AggError("PARAMS: fixed-at-utc invalid or differs from first line")
    if kv["issue"] != str(ISSUE):
        raise AggError("PARAMS: issue must be 4150")
    for k in ("rounds", "warmup", "iters"):
        if not kv[k].isdigit() or int(kv[k]) < 1:
            raise AggError(f"PARAMS: {k} must be a positive integer")
    if not HEX40.match(kv["tree_sha"]):
        raise AggError("PARAMS: tree_sha must be 40 hex")
    for k in HASH_KEYS:
        if not HEX64.match(kv[k]):
            raise AggError(f"PARAMS: {k} must be 64 hex")
    if kv["aggregate_script_sha256"] != self_sha256():
        raise AggError("PARAMS: aggregate_script_sha256 mismatch (script changed after RULE was fixed)")
    return kv


def parse_device(ln, round_no, warmup, iters):
    """DEVICE4150 行を厳密に解析し、round・warmup・iters を期待値（run_dir の i と RULE）と照合する。"""
    m = DEVICE_RE.match(ln)
    if not m:
        raise AggError(f"chain: malformed DEVICE4150 header: {ln!r}")
    got = {"round": int(m.group(3)), "warmup": int(m.group(4)), "iters": int(m.group(5))}
    want = {"round": round_no, "warmup": warmup, "iters": iters}
    for k in ("round", "warmup", "iters"):
        if got[k] != want[k]:
            raise AggError(f"chain: DEVICE4150 {k}={got[k]} does not match expected {want[k]}")


def parse_chain(text, round_no, warmup, iters):
    """chain.txt（example の標準出力）を厳格にパースし {arm: {key: [値…]}} を返す。

    round_no は round{i} の i、warmup・iters は RULE の値。DEVICE4150 ヘッダはちょうど 1 行で先頭に置かれ、
    done4150 は最終行（重複・後続行は拒否）。いずれも fail-closed（AggError）。
    """
    seen = {a: {} for a in ARM_KEYS}
    checksum = None
    device = done = False
    for ln in text.splitlines():
        if done:
            raise AggError(f"chain: line after done4150: {ln!r}")
        if ln.startswith("DEVICE4150"):
            if device:
                raise AggError("chain: duplicate DEVICE4150 header")
            parse_device(ln, round_no, warmup, iters)
            device = True
            continue
        if not device and not ln.startswith("ERROR4150"):
            raise AggError(f"chain: record before DEVICE4150 header: {ln!r}")
        if ln == "done4150":
            done = True
            continue
        if ln.startswith("CHECKSUM4150"):
            if checksum is not None:
                raise AggError("chain: duplicate CHECKSUM4150")
            cm = CHECKSUM_RE.match(ln)
            if not cm:
                raise AggError(f"chain: malformed CHECKSUM4150: {ln!r}")
            checksum = cm.group(1)
            continue
        if ln.startswith("ERROR4150"):
            raise AggError(f"chain: example reported error: {ln}")
        m = LINE_RE.match(ln)
        if not m:
            raise AggError(f"chain: unparsable line: {ln!r}")
        arm, it = m.group(1), int(m.group(2))
        kv = {}
        for tok in m.group(3).split():
            k, v = tok.split("=", 1)
            if k in kv:
                raise AggError(f"chain: duplicate key {k} in {ln!r}")
            val = float(v)
            if not math.isfinite(val):
                raise AggError(f"chain: non-finite value in {ln!r}")
            kv[k] = val
        if tuple(kv) != ARM_KEYS[arm]:
            raise AggError(f"chain: arm {arm} keys mismatch: {tuple(kv)}")
        if it in seen[arm].setdefault("_iters", {}):
            raise AggError(f"chain: duplicate arm={arm} iter={it}")
        seen[arm]["_iters"][it] = kv
    if not (device and done and checksum is not None):
        raise AggError("chain: DEVICE4150／CHECKSUM4150／done4150 のいずれかが欠損")
    out = {}
    for arm, d in seen.items():
        its = d.get("_iters", {})
        if sorted(its) != list(range(iters)):
            raise AggError(f"chain: arm {arm} iterations incomplete ({len(its)} of {iters})")
        out[arm] = {k: [its[i][k] for i in range(iters)] for k in ARM_KEYS[arm]}
    return out, checksum


def med(xs):
    return statistics.median(xs)


def derive(chain):
    """round 1 件分の腕別中央値と導出量（RULE 項 5）。各量に証拠タグを付ける。"""
    m = {a: {k: med(v) for k, v in kv.items()} for a, kv in chain.items()}
    P, E, I, F = m["P"], m["E"], m["I"], m["F"]
    kwd = P["kernel_wait_derived_us"]
    if kwd <= 0:
        raise AggError("kernel_wait_derived の中央値が 0 以下（share が定義できない）")
    # event_perturbation の分母。除算より前に検査する（ZeroDivisionError ではなく AggError で止める）。
    if P["download_us"] <= 0:
        raise AggError("P 腕 download の中央値が 0 以下")
    e_sum = I["l1_iso_us"] + I["l2_iso_us"]
    host_issue = P["l1_issue_us"] + P["l2_issue_us"]
    gpu_tail = max(0.0, E["chain_gpu_us"] - host_issue)
    d = {
        "kernel_wait_derived": (kwd, "実測"),
        "E_sum": (e_sum, "実測"),
        "chain_gpu": (E["chain_gpu_us"], "実測"),
        "gpu_gap": (E["chain_gpu_us"] - e_sum, "実測"),
        "host_issue_sum": (host_issue, "実測"),
        "gpu_tail": (gpu_tail, "推定"),
        "wait_resid": (kwd - gpu_tail, "推定"),
        "share_exec": (e_sum / kwd, "推定"),
        "share_wait": (1.0 - e_sum / kwd, "推定"),
        "event_perturbation": ((E["download_us"] - P["download_us"]) / P["download_us"], "実測"),
        "l1_iso": (I["l1_iso_us"], "実測"),
        "l2_iso": (I["l2_iso_us"], "実測"),
        "l1_span": (E["l1_span_us"], "実測"),
        "l2_span": (E["l2_span_us"], "実測"),
        "up_gpu": (E["up_gpu_us"], "実測"),
        "download": (P["download_us"], "実測"),
        "download_again": (P["download_again_us"], "実測"),
        "h2d_upload": (P["h2d_upload_us"], "実測"),
        "l1_issue": (P["l1_issue_us"], "実測"),
        "l2_issue": (P["l2_issue_us"], "実測"),
        "total": (P["total_us"], "実測"),
        "sync_idle": (F["sync_idle_us"], "実測"),
        "tiny_roundtrip": (F["tiny_roundtrip_us"], "実測"),
        "launch_floor_1x1": (F["launch_floor_1x1_us"], "実測"),
        "event_record_pair": (F["event_record_pair_us"], "実測"),
    }
    return d


def summarize(per_round):
    rows = {}
    for k in per_round[0]:
        vals = [r[k][0] for r in per_round]
        rows[k] = {"tag": per_round[0][k][1], "median": med(vals), "min": min(vals), "max": max(vals), "vals": vals}
    return rows


def classify(rows):
    """RULE 項 5 の優先順（摂動 → 支配側 → 確度）。事後に動かさない。"""
    pert = rows["event_perturbation"]["median"]
    share = rows["share_exec"]
    side = "GPU 実行が支配" if share["median"] >= SHARE_EXEC_THRESHOLD else "待ち（起動遅延・間隙・同期復帰）が支配"
    reasons = []
    conf = "高"
    if pert > PERTURBATION_LIMIT:
        conf = "低"
        reasons.append(f"event 摂動 {pert:.3f} > {PERTURBATION_LIMIT}")
    sides = {v >= SHARE_EXEC_THRESHOLD for v in share["vals"]}
    if len(sides) != 1:
        conf = "低"
        reasons.append("round 間で 0.5 の側が割れた")
    if share["min"] < SHARE_EXEC_THRESHOLD <= share["max"]:
        conf = "低"
        reasons.append("NB-RANGE5 帯が 0.5 をまたぐ")
    return side, conf, reasons


def load_opp(path):
    if path is None:
        return None
    return load_opp_text(read(path))


def load_opp_text(text):
    lines = text.splitlines()
    if not lines or lines[0] != "item\tmedian_us":
        raise AggError("opp-tsv: header must be 'item\\tmedian_us'")
    out = {}
    for ln in lines[1:]:
        if not ln:
            continue
        parts = ln.split("\t")
        if len(parts) != 2 or parts[0] not in OPP_ITEMS or parts[0] in out:
            raise AggError(f"opp-tsv: bad or duplicate row: {ln!r}")
        try:
            v = float(parts[1])
        except ValueError:
            raise AggError(f"opp-tsv: not a number: {ln!r}")
        if not math.isfinite(v):
            raise AggError(f"opp-tsv: non-finite: {ln!r}")
        out[parts[0]] = v
    return out


def fnum(x, nd=3):
    return f"{x:.{nd}f}"


# 対応表（fandhe 側の量, #4149 側の item）。非対称は README／md に併記する。
PAIRS = (("host_issue_sum", "issue"), ("download", "readout_sync"), ("chain_gpu", "gpu_span_event"),
         ("l1_iso", "mm_event_l1_fwd"), ("l2_iso", "mm_event_l2_fwd"), ("sync_idle", "sync_idle"),
         ("tiny_roundtrip", "tiny_roundtrip"), ("launch_floor_1x1", "launch_floor_add"),
         ("event_record_pair", "event_record_pair"))


def series_status(statuses):
    """全 round が PASS なら complete、TIMEOUT／INVALID を 1 件でも含めば partial（RULE 項 3）。"""
    return "complete" if statuses and all(s == "PASS" for s in statuses) else "partial"


def render(rule, statuses, rows, opp):
    side, conf, reasons = classify(rows)
    status = series_status(statuses)
    n_ok = len(rows["share_exec"]["vals"])
    if status == "complete":
        title = f"# #4150 集計（record_only の内訳。tree_sha={rule['tree_sha']}）"
        cls = f"- 分類: **{side}**（確度: {conf}" + (f"。理由: {'、'.join(reasons)}" if reasons else "") + "）"
    else:
        # RULE 項 3: 不完全な系列は「参考」。確度（と確度の理由）は書かない。
        title = f"# 参考: #4150 集計（status=partial・正式系列ではない。tree_sha={rule['tree_sha']}）"
        cls = f"- 分類（参考）: {side}（status=partial のため確度は出さない）"
    md = [title, "",
          f"- status: {status}",
          f"- round 状態: {' '.join(statuses)}（PASS {n_ok} 件を使用）",
          cls,
          "- 値の単位は µs。round 別中央値の中央値と NB-RANGE5（最小〜最大）。タグ: 実測／推定",
          "- 区間: E_sum（l1_iso＋l2_iso）は into 経路の融合カーネル幅で出力のゼロ初期化を含まない。"
          "chain_gpu・l1_issue／l2_issue は層ごとの出力確保＋ゼロ初期化（#3189 の呼び出し列）を含むため、"
          "その分は gpu_gap・share_wait 側に入る（RULE 項 4・5）", "",
          "| 量 | タグ | 中央値 | 最小 | 最大 |", "|---|---|---|---|---|"]
    tsv = ["item\ttag\tmedian\tmin\tmax", f"status\t{status}\t\t\t"]
    for k, r in rows.items():
        nd = 4 if k.startswith("share") or k == "event_perturbation" else 3
        md.append(f"| {k} | {r['tag']} | {fnum(r['median'], nd)} | {fnum(r['min'], nd)} | {fnum(r['max'], nd)} |")
        tsv.append(f"{k}\t{r['tag']}\t{fnum(r['median'], nd)}\t{fnum(r['min'], nd)}\t{fnum(r['max'], nd)}")
    md += ["", "## 相手側（#4149 PyTorch CUDA）との対応", ""]
    if opp is None:
        md.append("相手側の実測は未入力（--opp-tsv なし）: **unavailable**。推定値では埋めない。")
        tsv.append("opp\tunavailable\t\t\t")
    else:
        md += ["| fandhe の量 | 相手の item | fandhe | 相手 | 差（実測の差） |", "|---|---|---|---|---|"]
        for mine, theirs in PAIRS:
            if theirs in opp:
                a = rows[mine]["median"]
                md.append(f"| {mine} | {theirs} | {fnum(a)} | {fnum(opp[theirs])} | {fnum(a - opp[theirs])} |")
            else:
                md.append(f"| {mine} | {theirs} | {fnum(rows[mine]['median'])} | unavailable | unavailable |")
        md += ["", "非対称: fandhe の窓は H2D を含み PyTorch 側は X が device 上で H2D を含まない。"
               "fandhe の l1_iso／l2_iso は bias＋ReLU 融合込みの同じカーネル（into 経路）、PyTorch の mm_event は純 GEMM。"
               "fandhe の chain_gpu は層出力のゼロ初期化を含む（PyTorch の caching allocator は層ごとのゼロ初期化をしない）。"
               "launch_floor_1x1（1x1 linear の単発発行）と launch_floor_add（1 要素 add 100 回の 1 回あたり）は"
               "区間定義が異なる参考の並びで、差を内訳として読まない。"]
    return "\n".join(md) + "\n", "\n".join(tsv) + "\n"


def parse_gate(text, i):
    """gate.txt はちょうど 1 行（run_round.sh の gate() が書く）。先頭トークンが PASS／TIMEOUT／INVALID。"""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) != 1:
        raise AggError(f"round{i}/gate.txt: expected exactly one line, got {len(lines)}")
    tok = lines[0].split()[0]
    if tok not in GATE_TOKENS:
        raise AggError(f"round{i}/gate.txt: unparsable")
    return tok


def run_dir(d, rule_path, out_md, out_tsv, opp_path=None):
    rule = parse_rule(rule_path)
    n, iters = int(rule["rounds"]), int(rule["iters"])
    warmup = int(rule["warmup"])
    # RULE 項 3: run の追加・除外は禁止。round{k} ディレクトリの集合が round1..round{rounds} と一致すること。
    try:
        found = {int(m.group(1)) for e in os.listdir(d) for m in [ROUND_DIR_RE.match(e)] if m}
    except OSError as e:
        raise AggError(f"cannot list {d}: {e}")
    if found != set(range(1, n + 1)):
        raise AggError(f"round dirs mismatch: expected round1..round{n}, found {sorted(found)}")
    statuses, per_round, sums = [], [], set()
    for i in range(1, n + 1):
        rd = os.path.join(d, f"round{i}")
        gate = parse_gate(read(os.path.join(rd, "gate.txt")), i)
        statuses.append(gate)
        if gate == "PASS":
            chain, cks = parse_chain(read(os.path.join(rd, "raw", "chain.txt")), i, warmup, iters)
            sums.add(cks)
            per_round.append(derive(chain))
    if not per_round:
        raise AggError("no PASS round")
    if len(sums) != 1:
        raise AggError(f"checksum が round 間で不一致: {sorted(sums)}")
    rows = summarize(per_round)
    md, tsv = render(rule, statuses, rows, load_opp(opp_path))
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)
    with open(out_tsv, "w", encoding="utf-8") as f:
        f.write(tsv)


def fingerprint(tree):
    """tree の指紋。aggregate_diag.py（diag-0140-template）／aggregate_4149.py と同一ロジック。"""
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
    """雛形の PARAMS が必須キーを過不足なく持ち、値がすべてプレースホルダ（<…>）であることを検査する。"""
    text = read(path)
    lines = text.splitlines()
    if not lines or not re.match(r"^RULE fixed-at-utc <.+>$", lines[0]):
        raise AggError("template: first line must carry a <…> timestamp placeholder")
    _, body = split_params(text)
    kv = parse_kv(body)
    if set(kv) != set(PARAM_KEYS):
        raise AggError(f"template: PARAMS keys mismatch: missing={set(PARAM_KEYS) - set(kv)} extra={set(kv) - set(PARAM_KEYS)}")
    if kv["issue"] != str(ISSUE):
        raise AggError("template: issue must be 4150")
    for k, v in kv.items():
        if k == "issue":
            continue
        if not re.search(r"<[^<>]+>", v):
            raise AggError(f"template: value of {k} is not a placeholder: {v!r}")
    if PLACEHOLDER_MARK not in text:
        raise AggError("template: no 【記入 marker found")


def check_rule(path):
    parse_rule(path)


# ---------------------------------------------------------------- self-test
def _synth_chain(iters, e_sum_each=25.0, kwd=100.0, p_dl=100.0, e_dl=100.0, drop_key=None, dup=False, arms=None,
                 round_no=1, warmup=1):
    lines = ["DEVICE4150 compute_capability=12.1 round=%d warmup=%d iters=%d" % (round_no, warmup, iters)]
    vals = {
        "P": dict(h2d_upload_us=10, l1_issue_us=20, l2_issue_us=20, download_us=p_dl, download_again_us=p_dl - kwd,
                  kernel_wait_derived_us=kwd, total_us=150),
        "E": dict(up_gpu_us=10, l1_span_us=30, l2_span_us=20, chain_gpu_us=60, download_us=e_dl, l1_issue_us=20,
                  l2_issue_us=20),
        "I": dict(l1_iso_us=e_sum_each, l2_iso_us=e_sum_each, l1_host_us=20, l2_host_us=20),
        "F": dict(sync_idle_us=5, launch_floor_1x1_us=3, tiny_roundtrip_us=30, event_record_pair_us=1, event_pair_gpu_us=2),
    }
    for arm in (arms or "PEIF"):
        for i in range(iters):
            kv = dict(vals[arm])
            if drop_key and arm == "P":
                del kv[drop_key]
            lines.append(f"CHAIN4150 arm={arm} iter={i}" + "".join(f" {k}={v:.3f}" for k, v in kv.items()))
    if dup:
        lines.append(lines[1])
    lines += ["CHECKSUM4150 value=1", "done4150"]
    return "\n".join(lines) + "\n"


def _expect_error(fn, label):
    try:
        fn()
    except AggError:
        return
    raise AssertionError(f"{label}: AggError expected")


def self_test():
    n = 5
    ok, _ = parse_chain(_synth_chain(n), 1, 1, n)
    d = derive(ok)
    assert d["share_exec"][0] == 0.5 and d["kernel_wait_derived"][0] == 100.0
    assert d["gpu_gap"][0] == 60 - 50 and d["gpu_tail"][0] == 20 and d["wait_resid"][0] == 80
    # 境界: share_exec ちょうど 0.5 は GPU 実行側、摂動ちょうど 0.10 は確度を落とさない。
    rows = summarize([derive(parse_chain(_synth_chain(n, e_dl=110.0), 1, 1, n)[0]) for _ in range(3)])
    assert rows["event_perturbation"]["median"] == 0.1
    side, conf, why = classify(rows)
    assert side.startswith("GPU 実行") and conf == "高" and not why, (side, conf, why)
    rows = summarize([derive(parse_chain(_synth_chain(n, e_dl=111.0), 1, 1, n)[0]) for _ in range(3)])
    assert classify(rows)[1] == "低"
    # 待ち支配（share < 0.5）。
    rows = summarize([derive(parse_chain(_synth_chain(n, e_sum_each=10.0), 1, 1, n)[0]) for _ in range(3)])
    assert classify(rows)[0].startswith("待ち") and classify(rows)[1] == "高"
    # round 間で側が割れる → 低。
    mixed = [derive(parse_chain(_synth_chain(n, e_sum_each=x), 1, 1, n)[0]) for x in (10.0, 30.0, 30.0)]
    side, conf, why = classify(summarize(mixed))
    assert conf == "低" and any("割れた" in w or "またぐ" in w for w in why)
    # 異常系。
    _expect_error(lambda: parse_chain(_synth_chain(n, drop_key="total_us"), 1, 1, n), "missing key")
    _expect_error(lambda: parse_chain(_synth_chain(n, dup=True), 1, 1, n), "duplicate iter")
    _expect_error(lambda: parse_chain(_synth_chain(n, arms="PEI"), 1, 1, n), "missing arm")
    _expect_error(lambda: parse_chain(_synth_chain(n), 1, 1, n + 1), "iters mismatch")
    _expect_error(lambda: parse_chain(_synth_chain(n) + "garbage\n", 1, 1, n), "unparsable")
    _expect_error(lambda: parse_chain(_synth_chain(n).replace("done4150\n", ""), 1, 1, n), "no done")
    # DEVICE4150 ヘッダの厳密解析（指摘: 接頭辞だけで受理していた）。
    _expect_error(lambda: parse_chain(_synth_chain(n, round_no=2), 1, 1, n), "header round mismatch")
    _expect_error(lambda: parse_chain(_synth_chain(n, warmup=50), 1, 1, n), "header warmup mismatch")
    hdr_iters = _synth_chain(n).replace(f"iters={n}\n", f"iters={n + 1}\n", 1)
    _expect_error(lambda: parse_chain(hdr_iters, 1, 1, n), "header iters mismatch")
    syn = _synth_chain(n)
    first = syn.splitlines()[0]
    _expect_error(lambda: parse_chain(syn.replace(first + "\n", first + "\n" + first + "\n", 1), 1, 1, n),
                  "duplicate header")
    _expect_error(lambda: parse_chain(syn.replace(first, "DEVICE4150 garbage", 1), 1, 1, n), "malformed header")
    _expect_error(lambda: parse_chain(syn.replace(first, first + " extra=1", 1), 1, 1, n), "header trailing token")
    _expect_error(lambda: parse_chain(syn.replace(first, "DEVICE4150x" + first[len("DEVICE4150"):], 1), 1, 1, n),
                  "header prefix only")
    _expect_error(lambda: parse_chain(syn.replace(first + "\n", "", 1), 1, 1, n), "no header")
    _expect_error(lambda: parse_chain(syn.replace(first + "\n", "", 1).replace("done4150\n", first + "\ndone4150\n"),
                                      1, 1, n), "header not first")
    # CHECKSUM4150・done4150 の厳密化。
    _expect_error(lambda: parse_chain(syn.replace("value=1", "value=abc"), 1, 1, n), "checksum not digits")
    _expect_error(lambda: parse_chain(syn + "done4150\n", 1, 1, n), "duplicate done")
    _expect_error(lambda: parse_chain(syn + "CHECKSUM4150 value=1\n", 1, 1, n), "line after done")
    _expect_error(lambda: parse_chain(syn.replace("iter=0 ", "iter=00 ", 1), 1, 1, n), "leading zero iter")
    assert parse_chain(_synth_chain(n, round_no=3, warmup=50), 3, 50, n)[1] == "1"
    # gate.txt の厳密化。
    assert parse_gate("PASS load1=0.1\n", 1) == "PASS"
    _expect_error(lambda: parse_gate("PASS x\nTIMEOUT y\n", 1), "gate two lines")
    _expect_error(lambda: parse_gate("PASSED x\n", 1), "gate bad token")
    _expect_error(lambda: parse_gate("", 1), "gate empty")
    # series_status と partial の描画（指摘: 状態によらず同じ表題・確度を出していた）。
    assert series_status(["PASS"] * 5) == "complete"
    assert series_status(["PASS", "TIMEOUT", "TIMEOUT", "TIMEOUT", "TIMEOUT"]) == "partial"
    assert series_status(["PASS", "PASS", "INVALID", "PASS", "PASS"]) == "partial"
    one = summarize([derive(parse_chain(_synth_chain(n), 1, 1, n)[0])])
    rule_stub = {"tree_sha": "a" * 40}
    md_p, tsv_p = render(rule_stub, ["PASS", "TIMEOUT", "TIMEOUT", "TIMEOUT", "TIMEOUT"], one, None)
    assert md_p.startswith("# 参考: ") and "確度:" not in md_p and "確度: 高" not in md_p and "理由:" not in md_p
    assert "status=partial のため確度は出さない" in md_p and "status\tpartial\t" in tsv_p
    assert "- status: partial" in md_p
    md_c, tsv_c = render(rule_stub, ["PASS"], one, None)
    assert md_c.startswith("# #4150 集計") and "確度: 高" in md_c and "status\tcomplete\t" in tsv_c
    # 区間の注記（I 腕は into 経路でゼロ初期化を含まず、chain_gpu は含む）を complete・partial の両方に出す。
    assert all("出力のゼロ初期化を含まない" in x and "gpu_gap・share_wait 側に入る" in x for x in (md_c, md_p))
    _expect_error(lambda: load_opp_text("item\tmedian_us\nissue\tabc\n"), "opp not a number")
    _expect_error(lambda: derive(parse_chain(_synth_chain(n, kwd=0.0), 1, 1, n)[0]), "kwd zero")
    # example 出力キーとの突き合わせ（Rust 側の format_line 呼び出しと同一であること）。
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "crates", "backend-cuda",
                       "examples", "cuda_infer_chain_event_diag_4150.rs")
    if os.path.exists(src):
        text = read(src)
        for arm, keys in ARM_KEYS.items():
            for k in keys:
                assert f'"{k}"' in text, f"example に出力キー {k}（arm {arm}）がない"
    # RULE／テンプレート／end-to-end。
    with tempfile.TemporaryDirectory() as t:
        rule = os.path.join(t, "RULE.txt")
        body = (f"RULE fixed-at-utc 2026-11-09T00:00:00Z\nx\n\n## PARAMS\nfixed-at-utc: 2026-11-09T00:00:00Z\nissue: 4150\n"
                f"rounds: 2\ntree_sha: {'a' * 40}\ntree_fingerprint: {'b' * 64}\nexample_sha256: {'c' * 64}\n"
                f"aggregate_script_sha256: {self_sha256()}\nwarmup: 1\niters: {n}\n")
        open(rule, "w").write(body)
        for i in (1, 2):
            os.makedirs(os.path.join(t, f"round{i}", "raw"))
            open(os.path.join(t, f"round{i}", "gate.txt"), "w").write("PASS x\n")
            open(os.path.join(t, f"round{i}", "raw", "chain.txt"), "w").write(_synth_chain(n, round_no=i))
        run_dir(t, rule, os.path.join(t, "a.md"), os.path.join(t, "a.tsv"))
        assert "unavailable" in read(os.path.join(t, "a.md"))
        assert "status\tcomplete\t" in read(os.path.join(t, "a.tsv")) and "確度: " in read(os.path.join(t, "a.md"))
        # round2 の chain に round1 のヘッダ（取り違え）→ 拒否。
        open(os.path.join(t, "round2", "raw", "chain.txt"), "w").write(_synth_chain(n, round_no=1))
        _expect_error(lambda: run_dir(t, rule, os.path.join(t, "x.md"), os.path.join(t, "x.tsv")), "e2e round mismatch")
        # warmup が RULE（1）と不一致 → 拒否。
        open(os.path.join(t, "round2", "raw", "chain.txt"), "w").write(_synth_chain(n, round_no=2, warmup=50))
        _expect_error(lambda: run_dir(t, rule, os.path.join(t, "x.md"), os.path.join(t, "x.tsv")), "e2e warmup mismatch")
        open(os.path.join(t, "round2", "raw", "chain.txt"), "w").write(_synth_chain(n, round_no=2))
        # PASS が一部だけ（round2 が TIMEOUT）→ partial・参考・確度なし。
        open(os.path.join(t, "round2", "gate.txt"), "w").write("TIMEOUT load1=2.0 x\n")
        run_dir(t, rule, os.path.join(t, "p.md"), os.path.join(t, "p.tsv"))
        pmd = read(os.path.join(t, "p.md"))
        assert pmd.startswith("# 参考: ") and "確度: " not in pmd and "status\tpartial\t" in read(os.path.join(t, "p.tsv"))
        open(os.path.join(t, "round2", "gate.txt"), "w").write("PASS x\n")
        # 余分な round ディレクトリ（run の追加）→ 拒否。
        os.makedirs(os.path.join(t, "round3"))
        _expect_error(lambda: run_dir(t, rule, os.path.join(t, "x.md"), os.path.join(t, "x.tsv")), "extra round dir")
        os.rmdir(os.path.join(t, "round3"))
        opp = os.path.join(t, "opp.tsv")
        open(opp, "w").write("item\tmedian_us\nissue\t10.0\n")
        run_dir(t, rule, os.path.join(t, "b.md"), os.path.join(t, "b.tsv"), opp)
        assert "| host_issue_sum | issue |" in read(os.path.join(t, "b.md"))
        open(opp, "w").write("item\tmedian_us\nissue\t10.0\nissue\t11.0\n")
        _expect_error(lambda: load_opp(opp), "dup opp")
        # 欠損 round／checksum 不一致／プレースホルダ残存／sha 不一致。
        open(os.path.join(t, "round2", "raw", "chain.txt"), "w").write(
            _synth_chain(n, round_no=2).replace("value=1", "value=2"))
        _expect_error(lambda: run_dir(t, rule, os.path.join(t, "c.md"), os.path.join(t, "c.tsv")), "checksum")
        os.remove(os.path.join(t, "round2", "gate.txt"))
        _expect_error(lambda: run_dir(t, rule, os.path.join(t, "c.md"), os.path.join(t, "c.tsv")), "missing gate")
        ph = os.path.join(t, "ph.txt")
        open(ph, "w").write(body + "# 【記入: x】\n")
        _expect_error(lambda: parse_rule(ph), "placeholder")
        bad = os.path.join(t, "bad.txt")
        open(bad, "w").write(body.replace(self_sha256(), "d" * 64))
        _expect_error(lambda: parse_rule(bad), "sha mismatch")
        # 指紋は内容変更で変わる。
        os.makedirs(os.path.join(t, "tree", "crates"))
        open(os.path.join(t, "tree", "crates", "f"), "w").write("1")
        fp1 = fingerprint(os.path.join(t, "tree"))[0]
        open(os.path.join(t, "tree", "crates", "f"), "w").write("2")
        assert fp1 != fingerprint(os.path.join(t, "tree"))[0]
    print("self-test OK")


def main(argv):
    def opt(name, default=None):
        i = argv.index(name) + 1 if name in argv else 0
        return argv[i] if 0 < i < len(argv) and not argv[i].startswith("--") else default
    try:
        if "--self-test" in argv:
            self_test()
        elif "--check-template" in argv:
            check_template(opt("--check-template") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "RULE.template.txt"))
            print("template OK")
        elif "--check-rule" in argv:
            p = opt("--check-rule")
            if not p:
                raise AggError("--check-rule needs a RULE.txt path")
            check_rule(p)
            print("rule OK")
        elif "--fingerprint" in argv:
            p = opt("--fingerprint")
            if not p:
                raise AggError("--fingerprint needs a tree dir")
            fp, n, _ = fingerprint(p)
            print(f"{fp}  files={n}")
        elif "--dir" in argv:
            d = opt("--dir")
            if not d:
                raise AggError("--dir needs a directory")
            run_dir(d, opt("--rule", os.path.join(d, "RULE.txt")), opt("--out", os.path.join(d, "aggregate.md")),
                    opt("--tsv", os.path.join(d, "aggregate.tsv")), opt("--opp-tsv"))
            print("aggregate written")
        else:
            print(__doc__)
            return 2
    except AggError as e:
        print(f"AGGREGATE ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
