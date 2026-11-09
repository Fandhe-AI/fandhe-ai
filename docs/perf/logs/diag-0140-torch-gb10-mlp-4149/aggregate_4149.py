#!/usr/bin/env python3
"""GB10 の PyTorch CPU・CUDA の MLP 窓の区間内訳を機械集計する（#4149）。

役割: scripts/bench/framework-compare/torch_gb10_mlp_diag.py が書いた raw/ を、RULE.txt の PARAMS で
測る前に固定した規則どおりに区間別の表（中央値・run 別の最小〜最大・窓に対する比・証拠タグ）にする。
判定・分類は行わない（record_only）。fandhe 側との比較・分類は #4168・#4150 で別に扱う。実測値は含まない。
派生元: docs/perf/logs/diag-0140-torch-cpu-mlp-macmini-4148/aggregate_4148.py（構造と厳格パースの契約を継ぐ）。

入力（<dir> 配下）:
  RULE.txt                      1 行目 `RULE fixed-at-utc …` と末尾の `## PARAMS` ブロック
  round{1..N}/gate.txt          1 語目が PASS／TIMEOUT／INVALID
  round{1..N}/raw/env.json・measure_{cpu_train,cpu_infer,cpu_shapes,cuda_infer,cuda_fixed,cuda_shapes}.jsonl
                      ・profile_{cpu_train,cpu_infer,cuda_infer}.json
出力: aggregate.md（人間用）と aggregate.tsv（機械可読）。

厳格パース: 欠損・重複・未知・非有限・プレースホルダ残存・sha 不一致・round 不足・PASS 0 本はすべて AggError で停止する。
RULE 固定後の本ファイル改変は aggregate_script_sha256 の不一致で検出される。標準ライブラリのみ。`python3 -I` で実行すること。
profiler 腕の時間値は計時腕と混ぜない（op 数・kernel／API 回数だけを使う）。到達不能（unavailable）な量は推定値で埋めず空欄にする。

使い方:
  python3 -I aggregate_4149.py --dir <記録先> [--rule RULE.txt] [--out aggregate.md] [--tsv aggregate.tsv]
  python3 -I aggregate_4149.py --self-test
  python3 -I aggregate_4149.py --check-template [RULE.template.txt]
  python3 -I aggregate_4149.py --check-rule <RULE.txt>   # 固定済み RULE の検証（雛形用の --check-template とは別）
  python3 -I aggregate_4149.py --fingerprint <tree-dir>
"""
import hashlib
import json
import math
import os
import re
import statistics
import sys
import tempfile

ISSUE = 4149
PARAM_KEYS = ["fixed-at-utc", "issue", "rounds", "tree_sha", "tree_fingerprint", "bench_py_sha256",
              "torch_version", "diag_script_sha256", "helper_cpu_diag_sha256", "helper_cuda_diag_sha256",
              "aggregate_script_sha256", "probe_shim_sha256", "profile_steps"]
HASH_KEYS = ("tree_fingerprint", "bench_py_sha256", "diag_script_sha256", "helper_cpu_diag_sha256",
             "helper_cuda_diag_sha256", "aggregate_script_sha256", "probe_shim_sha256")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
FP_ROOTS = ["crates", "scripts/bench", "Cargo.toml", "Cargo.lock", "rust-toolchain.toml"]
FP_SKIP = {"target", ".git"}
PLACEHOLDER_MARK = "【記入"
# torch_gb10_mlp_diag.py の REQUIRED_KEYS・TAGS と同一（計測側 validate_row と同等の検証を集計側にも課す）。
ROW_REQUIRED_KEYS = ("issue", "section", "item", "status", "tag", "median_s", "q1_s", "q3_s", "warmup", "iters", "run",
                     "torch_threads", "torch_version", "numpy_version", "bench_py_sha256", "diag_script_sha256")
ROW_TAGS = ("実測", "推定", "未測定")
# torch_gb10_mlp_diag.py の SECTION_ITEMS と同一（テストで突き合わせる）。
SHAPE_IDS = ("l1_fwd", "l1_dw", "l1_dx", "l2_fwd", "l2_dw", "l2_dx")
SECTION_ITEMS = {
    "cpu_train": ("window", "fwd", "grad", "update", "loss_item", "ew_fwd", "dispatch_floor_add", "dispatch_floor_mm"),
    "cpu_infer": ("window", "fwd_compute", "readout", "checksum", "ew_fwd", "dispatch_floor_add", "dispatch_floor_mm"),
    "cpu_shapes": tuple(f"{p}_{s}" for s in SHAPE_IDS for p in ("mm_pure", "mm_batch10", "direct_batch10")),
    "cuda_infer": ("window", "issue", "readout_sync", "checksum", "gpu_span_event", "graph_replay_ref"),
    "cuda_fixed": ("sync_idle", "tiny_roundtrip", "launch_floor_add", "event_record_pair"),
    "cuda_shapes": tuple(f"mm_event_{s}" for s in SHAPE_IDS),
}
PROFILE_SECTIONS = ("cpu_train", "cpu_infer", "cuda_infer")
TRAIN_FWD, TRAIN_BWD, INFER_G = ("l1_fwd", "l2_fwd"), ("l2_dw", "l2_dx", "l1_dw"), ("l1_fwd", "l2_fwd")


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
        raise AggError("PARAMS: issue must be 4149")
    if not kv["rounds"].isdigit() or int(kv["rounds"]) < 1:
        raise AggError("PARAMS: rounds must be a positive integer")
    if not kv["profile_steps"].isdigit() or int(kv["profile_steps"]) < 1:
        raise AggError("PARAMS: profile_steps must be a positive integer")
    if not HEX40.match(kv["tree_sha"]):
        raise AggError("PARAMS: tree_sha must be 40 hex")
    for k in HASH_KEYS:
        if not HEX64.match(kv[k]):
            raise AggError(f"PARAMS: {k} must be 64 hex")
    if not re.match(r"^\d+\.\d+\.\d+$", kv["torch_version"]):
        raise AggError("PARAMS: torch_version must be x.y.z")
    if kv["aggregate_script_sha256"] != self_sha256():
        raise AggError("PARAMS: aggregate_script_sha256 mismatch (script changed after RULE was fixed)")
    return kv


def finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def load_rows(path, section, run, rule):
    """1 section の JSONL を厳格に読み {item: median_s|None} を返す。"""
    vals = {}
    for ln in read(path).splitlines():
        if not ln.strip():
            continue
        try:
            r = json.loads(ln)
        except ValueError as e:
            raise AggError(f"{path}: invalid JSON: {e}")
        if not isinstance(r, dict):
            raise AggError(f"{path}: row is not an object")
        missing = [k for k in ROW_REQUIRED_KEYS if k not in r]
        if missing:
            raise AggError(f"{path}: missing keys {missing} in row {r.get('item')!r}")
        if r.get("issue") != ISSUE or r.get("section") != section or r.get("run") != run:
            raise AggError(f"{path}: issue/section/run mismatch: {r.get('item')}")
        if r["tag"] not in ROW_TAGS:
            raise AggError(f"{path}: unknown tag {r['tag']!r} for {r.get('item')}")
        if r.get("bench_py_sha256") != rule["bench_py_sha256"]:
            raise AggError(f"{path}: bench_py_sha256 differs from RULE")
        if r.get("diag_script_sha256") != rule["diag_script_sha256"]:
            raise AggError(f"{path}: diag_script_sha256 differs from RULE")
        if not str(r.get("torch_version", "")).startswith(rule["torch_version"]):
            raise AggError(f"{path}: torch_version differs from RULE")
        item = r.get("item")
        if item not in SECTION_ITEMS[section]:
            raise AggError(f"{path}: unknown item {item!r}")
        if item in vals:
            raise AggError(f"{path}: duplicate item {item}")
        if r.get("status") == "ok":
            for k in ("median_s", "q1_s", "q3_s"):
                if not finite(r.get(k)):
                    raise AggError(f"{path}: non-finite {k} for {item}")
            if r["median_s"] <= 0:
                raise AggError(f"{path}: non-finite/non-positive median for {item}")
            if r["tag"] == "未測定":
                raise AggError(f"{path}: status=ok with tag=未測定 for {item}")
            vals[item] = r["median_s"]
        elif r.get("status") == "unavailable":
            if any(r.get(k) is not None for k in ("median_s", "q1_s", "q3_s")):
                raise AggError(f"{path}: unavailable row carries a value: {item}")
            if r["tag"] != "未測定":
                raise AggError(f"{path}: unavailable row must have tag=未測定: {item}")
            if not isinstance(r.get("reason"), str) or not r["reason"]:
                raise AggError(f"{path}: unavailable row needs reason: {item}")
            vals[item] = None
        else:
            raise AggError(f"{path}: unknown status for {item}")
    if set(vals) != set(SECTION_ITEMS[section]):
        raise AggError(f"{path}: item set incomplete: missing={set(SECTION_ITEMS[section]) - set(vals)}")
    return vals


def _check_provenance(path, d, rule):
    """profile／env 用: 版（torch_version）と両ハッシュを RULE と照合する（load_rows と同じ契約。RULE §7）。"""
    for k in ("bench_py_sha256", "diag_script_sha256"):
        if d.get(k) != rule[k]:
            raise AggError(f"{path}: {k} differs from RULE")
    if not str(d.get("torch_version", "")).startswith(rule["torch_version"]):
        raise AggError(f"{path}: torch_version differs from RULE")


def load_profile(path, section, run, rule):
    try:
        d = json.loads(read(path))
    except ValueError as e:
        raise AggError(f"{path}: invalid JSON: {e}")
    if d.get("issue") != ISSUE or d.get("section") != section or d.get("run") != run:
        raise AggError(f"{path}: issue/section/run mismatch")
    _check_provenance(path, d, rule)
    if d.get("steps") != int(rule["profile_steps"]):
        raise AggError(f"{path}: steps differs from RULE profile_steps")
    for k in ("n_ops_aten_total", "n_ops_aten_toplevel"):
        if not isinstance(d.get(k), int) or isinstance(d.get(k), bool) or d[k] < 1:
            raise AggError(f"{path}: invalid {k}")
    if section == "cuda_infer":
        # CUDA 活動は取れたか、取れない理由があるかのどちらか。沈黙の欠落は集計を止める。
        c = d.get("cuda")
        if c is None:
            if not isinstance(d.get("cuda_unavailable_reason"), str) or not d["cuda_unavailable_reason"]:
                raise AggError(f"{path}: cuda_infer profile needs cuda or cuda_unavailable_reason")
        elif not isinstance(c, dict) or not isinstance(c.get("graph_launch"), bool) \
                or not isinstance(c.get("kernels"), dict) or not isinstance(c.get("api"), dict):
            raise AggError(f"{path}: invalid cuda structure")
    return d


def load_round(rdir, run, rule):
    raw = os.path.join(rdir, "raw")
    env = json.loads(read(os.path.join(raw, "env.json")))
    if env.get("issue") != ISSUE:
        raise AggError(f"{raw}/env.json: issue mismatch")
    _check_provenance(f"{raw}/env.json", env, rule)
    # 再利用するヘルパーの改変も検出する（RULE に固定した sha256 と照合）。
    for k in ("helper_cpu_diag_sha256", "helper_cuda_diag_sha256"):
        if env.get(k) != rule[k]:
            raise AggError(f"{raw}/env.json: {k} differs from RULE")
    out = {"env": env, "vals": {}, "prof": {}}
    for s in SECTION_ITEMS:
        out["vals"][s] = load_rows(os.path.join(raw, f"measure_{s}.jsonl"), s, run, rule)
    for s in PROFILE_SECTIONS:
        out["prof"][s] = load_profile(os.path.join(raw, f"profile_{s}.json"), s, run, rule)
    return out


def _sum(vals, section, items, prefix):
    """prefix_item の和。1 つでも None なら None（推定値で埋めない）。"""
    tot = 0.0
    for i in items:
        v = vals[section][f"{prefix}_{i}"]
        if v is None:
            return None
        tot += v
    return tot


def _sub(a, *bs):
    if a is None or any(b is None for b in bs):
        return None
    return a - sum(bs)


def derive(rd):
    """1 round の区間量。戻り値は [(section, quantity, value_s|None, tag)]。"""
    v, prof = rd["vals"], rd["prof"]
    out = []

    def add(s, q, val, tag):
        out.append((s, q, val, tag))

    # cpu_train
    t = v["cpu_train"]
    gall = TRAIN_FWD + TRAIN_BWD
    gk = _sum(v, "cpu_shapes", gall, "mm_batch10")
    gf = _sum(v, "cpu_shapes", TRAIN_FWD, "mm_batch10")
    gb = _sum(v, "cpu_shapes", TRAIN_BWD, "mm_batch10")
    add("cpu_train", "window", t["window"], "実測")
    add("cpu_train", "gemm_kernel_batch10", gk, "実測")
    add("cpu_train", "gemm_mm_pure_sum", _sum(v, "cpu_shapes", gall, "mm_pure"), "実測")
    add("cpu_train", "ew_fwd", t["ew_fwd"], "実測")
    add("cpu_train", "update", t["update"], "実測")
    add("cpu_train", "loss_item", t["loss_item"], "実測")
    add("cpu_train", "residual", _sub(t["window"], gk, t["ew_fwd"], t["update"], t["loss_item"]), "推定")
    add("cpu_train", "fwd_nongemm", _sub(t["fwd"], gf), "推定")
    add("cpu_train", "bwd_nongemm", _sub(t["grad"], gb), "推定")
    add("cpu_train", "parts_minus_window", t["fwd"] + t["grad"] + t["update"] + t["loss_item"] - t["window"], "実測")
    add("cpu_train", "dispatch_floor_est", prof["cpu_train"]["n_ops_aten_toplevel"] * t["dispatch_floor_add"], "推定")
    add("cpu_train", "torch_vs_direct_gap", _sub(gk, _sum(v, "cpu_shapes", gall, "direct_batch10")), "実測")
    # cpu_infer
    i = v["cpu_infer"]
    gk = _sum(v, "cpu_shapes", INFER_G, "mm_batch10")
    add("cpu_infer", "window", i["window"], "実測")
    add("cpu_infer", "gemm_kernel_batch10", gk, "実測")
    add("cpu_infer", "gemm_mm_pure_sum", _sum(v, "cpu_shapes", INFER_G, "mm_pure"), "実測")
    add("cpu_infer", "ew_fwd", i["ew_fwd"], "実測")
    add("cpu_infer", "readout", i["readout"], "実測")
    add("cpu_infer", "checksum", i["checksum"], "実測")
    add("cpu_infer", "residual", _sub(i["window"], gk, i["ew_fwd"], i["readout"], i["checksum"]), "推定")
    add("cpu_infer", "dispatch_floor_est", prof["cpu_infer"]["n_ops_aten_toplevel"] * i["dispatch_floor_add"], "推定")
    add("cpu_infer", "torch_vs_direct_gap", _sub(gk, _sum(v, "cpu_shapes", INFER_G, "direct_batch10")), "実測")
    # cuda_infer（窓は H2D を含まない: X は device 上。fandhe 側は窓に H2D を含む非対称）
    c = v["cuda_infer"]
    cf = v["cuda_fixed"]
    add("cuda_infer", "window", c["window"], "実測")
    add("cuda_infer", "issue", c["issue"], "実測")
    add("cuda_infer", "readout_sync", c["readout_sync"], "実測")
    add("cuda_infer", "checksum", c["checksum"], "実測")
    add("cuda_infer", "gpu_span_event", c["gpu_span_event"], "実測")
    add("cuda_infer", "gemm_event_sum", _sum(v, "cuda_shapes", INFER_G, "mm_event"), "実測")
    add("cuda_infer", "residual", _sub(c["window"], c["issue"], c["readout_sync"], c["checksum"]), "推定")
    add("cuda_infer", "issue_minus_gpu_span", _sub(c["issue"], c["gpu_span_event"]), "実測")
    add("cuda_infer", "readout_sync_minus_tiny_roundtrip", _sub(c["readout_sync"], cf["tiny_roundtrip"]), "実測")
    add("cuda_infer", "graph_replay_ref", c["graph_replay_ref"], "実測")
    # cuda_fixed／cuda_shapes（窓なし）
    for it in SECTION_ITEMS["cuda_fixed"]:
        add("cuda_fixed", it, cf[it], "実測")
    for it in SECTION_ITEMS["cuda_shapes"]:
        add("cuda_shapes", it, v["cuda_shapes"][it], "実測")
    return out


def summarize(per_round):
    """quantity ごとに run 間の中央値・最小・最大。1 run でも None なら unavailable。窓のない section は比を出さない。"""
    keys = [(s, q, tag) for s, q, _, tag in per_round[0]]
    rows = []
    for idx, (s, q, tag) in enumerate(keys):
        vals = [pr[idx][2] for pr in per_round]
        if any(x is None for x in vals):
            rows.append((s, q, "未測定", None, None, None, None, None, None))
            continue
        wins = [next((x[2] for x in pr if x[0] == s and x[1] == "window"), None) for pr in per_round]
        if any(w is None for w in wins):
            rows.append((s, q, tag, statistics.median(vals), min(vals), max(vals), None, None, None))
            continue
        sh = [a / w for a, w in zip(vals, wins)]
        rows.append((s, q, tag, statistics.median(vals), min(vals), max(vals),
                     statistics.median(sh), min(sh), max(sh)))
    return rows


def fnum(x, scale=1e3, nd=5):
    return "" if x is None else f"{x * scale:.{nd}f}"


def _prof_cuda_lines(profs):
    """cuda_infer profile の kernel 名・API 回数・Graph の有無（run 間で union。1 step あたり）。"""
    lines = []
    unav = [p["cuda_unavailable_reason"] for p in profs if p.get("cuda") is None]
    if unav:
        lines.append(f"- CUDA 活動が取れなかった round あり（unavailable）: {sorted(set(unav))}")
    ok = [p["cuda"] for p in profs if p.get("cuda") is not None]
    if ok:
        lines.append(f"- cudaGraphLaunch の有無（いずれかの round で観測）: {'あり' if any(c['graph_launch'] for c in ok) else 'なし（eager）'}")
        api, kern = {}, {}
        for c in ok:
            for k, n in c["api"].items():
                api.setdefault(k, []).append(n)
            for k, n in c["kernels"].items():
                kern.setdefault(k, []).append(n)
        lines.append("- CUDA API 回数 / step（run 間の中央値）: " +
                     (", ".join(f"{k}={statistics.median(n):.2f}" for k, n in sorted(api.items())) or "なし"))
        lines += ["", "| kernel 名 | 回数 / step（run 間の中央値） |", "|---|---|"]
        for k, n in sorted(kern.items(), key=lambda kv: -statistics.median(kv[1])):
            lines.append(f"| `{k}` | {statistics.median(n):.2f} |")
    return lines


def render(rule, statuses, rows, profs):
    formal = all(st == "PASS" for st in statuses)
    title = "区間内訳（正式）" if formal else "区間内訳（参考: ゲート不通過 round あり。確度を書かない）"
    md = [f"# {title} #4149", "",
          f"- RULE: fixed-at-utc={rule['fixed-at-utc']} tree_sha={rule['tree_sha']} tree_fingerprint={rule['tree_fingerprint']}",
          f"- round 状態: {' '.join(f'r{n + 1}={s}' for n, s in enumerate(statuses))}（PASS のみ集計）",
          "- 時間は ms。中央値・最小〜最大は run 間。比は run ごとの 量÷窓 の中央値（窓のない section は空欄）。空欄は unavailable（推定値で埋めない）。",
          "- 推定: 窓から実測区間を引いた残差。CPU は autograd・Python 解釈・メモリ確保・dispatch が混ざる。CUDA は enqueue と並行して進む GPU 実行待ちが混ざる。",
          "- CUDA の窓は X が device 上にあり H2D を含まない（fandhe 側は窓に H2D を含む。並べて読むときの非対称）。",
          "- profiler 腕の値は時間として使っていない（op 数・kernel／API 回数のみ）。", ""]
    tsv = ["section\tquantity\ttag\tmedian_ms\tmin_ms\tmax_ms\tshare_median\tshare_min\tshare_max"]
    cur = None
    for idx, (s, q, tag, med, lo, hi, sm, sl, sh) in enumerate(rows):
        if s != cur:
            cur = s
            md += [f"## {s}", "", "| 量 | タグ | 中央値 ms | 最小〜最大 ms | 窓に対する比（中央値） | 比の最小〜最大 |", "|---|---|---|---|---|---|"]
        rng = "" if lo is None else f"{fnum(lo)}〜{fnum(hi)}"
        srg = "" if sl is None else f"{sl:.3f}〜{sh:.3f}"
        md.append(f"| {q} | {tag} | {fnum(med)} | {rng} | {'' if sm is None else f'{sm:.3f}'} | {srg} |")
        tsv.append("\t".join([s, q, tag, fnum(med), fnum(lo), fnum(hi), "" if sm is None else f"{sm:.4f}",
                              "" if sl is None else f"{sl:.4f}", "" if sh is None else f"{sh:.4f}"]))
        if idx + 1 == len(rows) or rows[idx + 1][0] != s:
            md.append("")
    md += ["## profiler 腕（op 数・kernel／API 回数のみ。時間は使わない）", "",
           "| section | step あたり aten op（全体） | 同（最上位） | aten::mm が子 op を持つ |", "|---|---|---|---|"]
    for s in PROFILE_SECTIONS:
        ps = profs[s]
        tot = sorted({p["n_ops_aten_total"] for p in ps})
        top = sorted({p["n_ops_aten_toplevel"] for p in ps})
        md.append(f"| {s} | {'/'.join(map(str, tot))} | {'/'.join(map(str, top))} | {'はい' if any(p['mm_has_child_ops'] for p in ps) else 'いいえ'} |")
    md += ["", "### cuda_infer の CUDA 活動", ""] + _prof_cuda_lines(profs["cuda_infer"])
    return "\n".join(md) + "\n", "\n".join(tsv) + "\n"


def run_dir(d, rule_path, out_md, out_tsv):
    rule = parse_rule(rule_path)
    n = int(rule["rounds"])
    statuses, loaded = [], []
    for i in range(1, n + 1):
        rd = os.path.join(d, f"round{i}")
        gate = read(os.path.join(rd, "gate.txt")).split()
        if not gate or gate[0] not in ("PASS", "TIMEOUT", "INVALID"):
            raise AggError(f"round{i}/gate.txt: unparsable")
        statuses.append(gate[0])
        if gate[0] == "PASS":
            loaded.append(load_round(rd, i, rule))
    if not loaded:
        raise AggError("no PASS round")
    rows = summarize([derive(r) for r in loaded])
    profs = {s: [r["prof"][s] for r in loaded] for s in PROFILE_SECTIONS}
    md, tsv = render(rule, statuses, rows, profs)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)
    with open(out_tsv, "w", encoding="utf-8") as f:
        f.write(tsv)


def fingerprint(tree):
    """tree の指紋。aggregate_diag.py（diag-0140-template）の fingerprint と同一ロジック（テストで一致を確認）。"""
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
        raise AggError("template: issue must be 4149")
    for k, v in kv.items():
        if k == "issue":
            continue
        if not re.search(r"<[^<>]+>", v):
            raise AggError(f"template: value of {k} is not a placeholder: {v!r}")
    if PLACEHOLDER_MARK not in text:
        raise AggError("template: no 【記入 marker found")


def check_rule(path):
    """固定済み RULE.txt を parse_rule で検証する（プレースホルダ残存・sha 不一致・キー過不足は AggError）。"""
    parse_rule(path)




def _synth_round(root, i, rule, ok=True, missing=False):
    rd = os.path.join(root, f"round{i}")
    raw = os.path.join(rd, "raw")
    os.makedirs(raw)
    with open(os.path.join(rd, "gate.txt"), "w") as f:
        f.write("PASS load1=0.5\n" if ok else "TIMEOUT load1=9\n")
    if not ok:
        return
    base = {"issue": ISSUE, "run": i, "bench_py_sha256": rule["bench_py_sha256"],
            "diag_script_sha256": rule["diag_script_sha256"], "torch_version": rule["torch_version"]}
    env = {k: base[k] for k in ("issue", "bench_py_sha256", "diag_script_sha256", "torch_version")}
    env["helper_cpu_diag_sha256"] = rule["helper_cpu_diag_sha256"]
    env["helper_cuda_diag_sha256"] = rule["helper_cuda_diag_sha256"]
    json.dump(env, open(os.path.join(raw, "env.json"), "w"))
    wins = {"cpu_train": 5e-3, "cpu_infer": 2e-3, "cuda_infer": 4e-5}
    for s, items in SECTION_ITEMS.items():
        with open(os.path.join(raw, f"measure_{s}.jsonl"), "w") as f:
            for k, it in enumerate(items):
                if missing and s == "cpu_train" and it == "update":
                    continue
                unavail = it.startswith("direct") and it.endswith("l1_dx")
                med = None if unavail else 1e-5 * (k + 1) * (1 + 0.01 * i)
                r = dict(base, section=s, item=it, status="unavailable" if unavail else "ok", median_s=med,
                         q1_s=None if unavail else med * 0.9, q3_s=None if unavail else med * 1.1,
                         tag="未測定" if unavail else "実測", warmup=1, iters=1, torch_threads=1, numpy_version="0",
                         **({"reason": "synthetic"} if unavail else {}))
                if it == "window" and s in wins:
                    r["median_s"], r["q1_s"], r["q3_s"] = wins[s], wins[s] * 0.9, wins[s] * 1.1
                f.write(json.dumps(r) + "\n")
    for s in PROFILE_SECTIONS:
        p = dict(base, section=s, steps=int(rule["profile_steps"]), n_ops_aten_total=30, n_ops_aten_toplevel=20,
                 mm_has_child_ops=True, cuda=None, cuda_unavailable_reason=None)
        if s == "cuda_infer":
            p["cuda"] = {"kernels": {"sm80_xmma_gemm_f32": 2.0}, "api": {"cudaLaunchKernel": 6.0, "cudaMemcpyAsync": 1.0},
                         "graph_launch": False}
        json.dump(p, open(os.path.join(raw, f"profile_{s}.json"), "w"))


def self_test():
    with tempfile.TemporaryDirectory() as root:
        rule_text = (
            "RULE fixed-at-utc 2026-11-09T00:00:00Z\n（合成）\n\n## PARAMS\n"
            "fixed-at-utc: 2026-11-09T00:00:00Z\nissue: 4149\nrounds: 3\n" + "tree_sha: " + "a" * 40 + "\n"
            "tree_fingerprint: " + "b" * 64 + "\nbench_py_sha256: " + "c" * 64 + "\ntorch_version: 2.14.0\n"
            "diag_script_sha256: " + "d" * 64 + "\nhelper_cpu_diag_sha256: " + "1" * 64 + "\nhelper_cuda_diag_sha256: " + "2" * 64
            + "\naggregate_script_sha256: " + self_sha256() + "\nprobe_shim_sha256: " + "3" * 64 + "\nprofile_steps: 50\n")
        rp = os.path.join(root, "RULE.txt")
        open(rp, "w").write(rule_text)
        rule = parse_rule(rp)
        d = os.path.join(root, "ok")
        os.makedirs(d)
        for i in (1, 2, 3):
            _synth_round(d, i, rule)
        run_dir(d, rp, os.path.join(d, "a.md"), os.path.join(d, "a.tsv"))
        tsv = open(os.path.join(d, "a.tsv")).read()
        assert "cpu_train\twindow\t実測\t5.00000" in tsv, tsv
        assert "cuda_infer\tissue_minus_gpu_span\t実測\t" in tsv
        assert "cuda_fixed\tsync_idle\t実測\t" in tsv
        # l1_dx の direct が unavailable でも、窓に使う形状の差は計算できる
        assert "cpu_train\ttorch_vs_direct_gap\t実測\t" in tsv
        assert "Graph" in open(os.path.join(d, "a.md")).read()
        # 固定済み RULE は --check-rule で受理される（--check-template は拒否される）
        check_rule(rp)
        try:
            check_template(rp)
            raise SystemExit("check_template accepted a fixed RULE")
        except AggError:
            pass
        # 行の不正は集計側でも停止する
        mp = os.path.join(d, "round1", "raw", "measure_cpu_train.jsonl")
        orig = open(mp).read()
        first = json.loads(orig.splitlines()[0])
        tails = "\n".join(orig.splitlines()[1:]) + "\n"
        for mut in ({"tag": None}, {"tag": "未測定"}, {"q1_s": float("nan")}, {"status": "unavailable", "median_s": None}):
            row = dict(first)
            if mut.get("tag") is None and "tag" in mut:
                del row["tag"]
            else:
                row.update(mut)
            open(mp, "w").write(json.dumps(row) + "\n" + tails)
            try:
                load_rows(mp, "cpu_train", 1, rule)
                raise SystemExit(f"load_rows accepted invalid row: {mut}")
            except AggError:
                pass
        open(mp, "w").write(orig)
        # ゲート不通過 round を含む場合は「参考」
        d2 = os.path.join(root, "partial")
        os.makedirs(d2)
        _synth_round(d2, 1, rule)
        _synth_round(d2, 2, rule, ok=False)
        _synth_round(d2, 3, rule)
        run_dir(d2, rp, os.path.join(d2, "a.md"), os.path.join(d2, "a.tsv"))
        assert "参考" in open(os.path.join(d2, "a.md")).read()
        # 欠損 item は例外
        d3 = os.path.join(root, "miss")
        os.makedirs(d3)
        for i in (1, 2, 3):
            _synth_round(d3, i, rule, missing=(i == 2))
        try:
            run_dir(d3, rp, os.path.join(d3, "a.md"), os.path.join(d3, "a.tsv"))
        except AggError:
            pass
        else:
            raise AssertionError("missing item not rejected")
        # profile／env の版・ハッシュ不一致は例外（ヘルパー sha を含む）
        for tag, fname, key in (("prof", "profile_cpu_train.json", "diag_script_sha256"), ("env", "env.json", "diag_script_sha256"),
                                ("envh", "env.json", "helper_cuda_diag_sha256")):
            d5 = os.path.join(root, "prov_" + tag)
            os.makedirs(d5)
            for i in (1, 2, 3):
                _synth_round(d5, i, rule)
            fp = os.path.join(d5, "round2", "raw", fname)
            obj = json.load(open(fp))
            obj[key] = "f" * 64
            json.dump(obj, open(fp, "w"))
            try:
                run_dir(d5, rp, os.path.join(d5, "a.md"), os.path.join(d5, "a.tsv"))
            except AggError:
                pass
            else:
                raise AssertionError(f"{fname}/{key} hash mismatch not rejected")
        # cuda_infer の profile が cuda も理由も持たない場合は例外
        d6 = os.path.join(root, "nocuda")
        os.makedirs(d6)
        for i in (1, 2, 3):
            _synth_round(d6, i, rule)
        fp = os.path.join(d6, "round1", "raw", "profile_cuda_infer.json")
        obj = json.load(open(fp))
        obj["cuda"] = None
        json.dump(obj, open(fp, "w"))
        try:
            run_dir(d6, rp, os.path.join(d6, "a.md"), os.path.join(d6, "a.tsv"))
            raise AssertionError("silent missing CUDA activity not rejected")
        except AggError:
            pass
        # PASS 0 本は例外
        d4 = os.path.join(root, "none")
        os.makedirs(d4)
        for i in (1, 2, 3):
            _synth_round(d4, i, rule, ok=False)
        try:
            run_dir(d4, rp, os.path.join(d4, "a.md"), os.path.join(d4, "a.tsv"))
            raise AssertionError("PASS 0 not rejected")
        except AggError:
            pass
        # sha 不一致は例外
        bad = os.path.join(root, "BAD.txt")
        open(bad, "w").write(rule_text.replace(self_sha256(), "e" * 64))
        try:
            parse_rule(bad)
            raise AssertionError("script sha mismatch not rejected")
        except AggError:
            pass
        # プレースホルダ残存は例外
        ph = os.path.join(root, "PH.txt")
        open(ph, "w").write(rule_text + "# 【記入: x】\n")
        try:
            parse_rule(ph)
            raise AssertionError("placeholder not rejected")
        except AggError:
            pass
        # 指紋
        t = os.path.join(root, "tree")
        os.makedirs(os.path.join(t, "crates"))
        open(os.path.join(t, "crates", "a.rs"), "w").write("x")
        fp1 = fingerprint(t)
        assert fingerprint(t) == fp1 and HEX64.match(fp1[0]) and fp1[1] == 1
    print("self-test OK")


def main(argv):
    def opt(name, default=None):
        i = argv.index(name) + 1 if name in argv else 0
        return argv[i] if 0 < i < len(argv) and not argv[i].startswith("--") else default
    try:
        if "--self-test" in argv:
            self_test()
        elif "--check-template" in argv:
            p = opt("--check-template") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "RULE.template.txt")
            check_template(p)
            print("template OK")
        elif "--check-rule" in argv:
            p = opt("--check-rule")
            if not p:
                raise AggError("--check-rule needs a RULE.txt path")
            check_rule(p)
            print("rule OK")
        elif "--fingerprint" in argv:
            fp, n, listing = fingerprint(opt("--fingerprint"))
            print(f"{fp}  files={n}")
        elif "--dir" in argv:
            d = opt("--dir")
            run_dir(d, opt("--rule", os.path.join(d, "RULE.txt")), opt("--out", os.path.join(d, "aggregate.md")),
                    opt("--tsv", os.path.join(d, "aggregate.tsv")))
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
