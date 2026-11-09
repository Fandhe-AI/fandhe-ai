#!/usr/bin/env python3
"""NR=12 タイルの端数が無駄になっていないかの診断（#4152）の機械集計。

役割: crates/backend-cpu/examples/cpu_gemm_nr_tail_4152.rs が書いた raw/ を、RULE.txt の PARAMS で
測る前に固定した判定式どおりに、ファミリ別の w（端数の無駄が占める割合）・g（正方 512 との差のうち端数で
説明できる割合）・r10_12 と分類にする。分類は RULE の閾値だけで決まり、集計側で閾値を変えない。
実測値は含まない（PR-1 は道具立てのみ）。

入力（<dir> 配下）:
  RULE.txt                 1 行目 `RULE fixed-at-utc …` と末尾の `## PARAMS` ブロック
  round{1..N}/gate.txt     1 語目が PASS／TIMEOUT／INVALID
  round{1..N}/raw/t1.txt・native.txt   example の 1 行 1 計測（PASS の round のみ読む）
出力: aggregate.md（人間用）と aggregate.tsv（機械可読）。

厳格パース: 欠損・重複・未知の family／n・非有限／0／負の median_secs・nr／isa 不一致・round ディレクトリ欠落・
PASS 0 本・round 間の bits_hash 不一致・プレースホルダ残存・sha256 不一致はすべて AggError で停止する。
RULE 固定後の本ファイル改変は aggregate_script_sha256 の不一致で検出される。標準ライブラリのみ。`python3 -I` で実行すること。

使い方:
  python3 -I aggregate_4152.py --dir <記録先> [--rule RULE.txt] [--out aggregate.md] [--tsv aggregate.tsv]
  python3 -I aggregate_4152.py --self-test
  python3 -I aggregate_4152.py --check-template [RULE.template.txt]
  python3 -I aggregate_4152.py --check-rule <RULE.txt>   # 固定済み RULE の検証（--check-template は雛形専用）
  python3 -I aggregate_4152.py --check-tree <tree-dir> --rule <RULE.txt>   # example_sha256・tree_fingerprint の照合（計測前必須）
  python3 -I aggregate_4152.py --fingerprint <tree-dir>
"""
import hashlib
import math
import os
import re
import statistics
import sys
import tempfile

ISSUE = 4152
# 測る前に決まらない値（SHA・日時・ハッシュ）だけがプレースホルダ。他は数値を直接書く。
PLACEHOLDER_KEYS = ["fixed-at-utc", "tree_sha", "tree_fingerprint", "example_sha256", "aggregate_script_sha256"]
FIXED_KEYS = ["issue", "rounds", "nr_expected", "isa_expected", "w_small", "w_large", "g_main", "sanity_lo",
              "sanity_hi", "mm177_l2_fwd_gflops", "mm177_l2_dw_gflops"]
PARAM_KEYS = PLACEHOLDER_KEYS + FIXED_KEYS
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
FP_ROOTS = ["crates", "scripts/bench", "Cargo.toml", "Cargo.lock", "rust-toolchain.toml"]
FP_SKIP = {"target", ".git"}
PLACEHOLDER_MARK = "【記入"
# example の FAMILIES／REF_SQUARE と同一（テストで合成 fixture が通ることで突き合わせる）。
FAMILY_SPEC = {  # family: (trans, m, k, ns)
    "l2_fwd": ("nn", 64, 256, (8, 10, 11, 12, 13, 24)),
    "l2_dw": ("tn", 256, 64, (8, 10, 11, 12, 13, 24)),
    "l1_fwd": ("nn", 64, 784, (256, 264, 276)),
    "ref_square_512": ("nn", 512, 512, (512,)),
}
JUDGED = ("l2_fwd", "l2_dw")  # 判定対象。n=10・12・24 を使う（NR=12 の端数 2 レーン）
N_LO, N_HI, N_NEXT = 10, 12, 24
SERIES = ("t1", "native")
ROW_RE = re.compile(
    r"^arm=(?P<arm>[A-Za-z0-9_-]{1,32}) series=(?P<series>t1|native) family=(?P<family>[a-z0-9_]+) "
    r"trans=(?P<trans>nn|tn) m=(?P<m>\d+) n=(?P<n>\d+) k=(?P<k>\d+) isa=(?P<isa>[A-Za-z0-9]+) "
    r"mr=(?P<mr>\d+) nr=(?P<nr>\d+) rayon_threads=(?P<threads>\d+) median_secs=(?P<secs>\S+) "
    r"gflops=(?P<gflops>\S+) bits_hash=(?P<hash>0x[0-9a-f]{16})$")


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


def num(kv, k):
    try:
        v = float(kv[k])
    except ValueError:
        raise AggError(f"PARAMS: {k} must be a number: {kv[k]!r}")
    if not math.isfinite(v) or v <= 0:
        raise AggError(f"PARAMS: {k} must be positive finite")
    return v


def parse_rule(path):
    """固定済み RULE.txt を検証して数値化した dict を返す。"""
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
        raise AggError(f"PARAMS: issue must be {ISSUE}")
    if not kv["rounds"].isdigit() or int(kv["rounds"]) < 1:
        raise AggError("PARAMS: rounds must be a positive integer")
    if not HEX40.match(kv["tree_sha"]):
        raise AggError("PARAMS: tree_sha must be 40 hex")
    for k in ("tree_fingerprint", "example_sha256", "aggregate_script_sha256"):
        if not HEX64.match(kv[k]):
            raise AggError(f"PARAMS: {k} must be 64 hex")
    if not kv["nr_expected"].isdigit() or int(kv["nr_expected"]) < 1:
        raise AggError("PARAMS: nr_expected must be a positive integer")
    if not re.match(r"^[A-Za-z0-9]+$", kv["isa_expected"]):
        raise AggError("PARAMS: isa_expected invalid")
    rule = {"rounds": int(kv["rounds"]), "nr": int(kv["nr_expected"]), "isa": kv["isa_expected"]}
    for k in ("w_small", "w_large", "g_main", "sanity_lo", "sanity_hi", "mm177_l2_fwd_gflops", "mm177_l2_dw_gflops"):
        rule[k] = num(kv, k)
    if not (rule["w_small"] < rule["w_large"]) or not (rule["sanity_lo"] < rule["sanity_hi"]):
        raise AggError("PARAMS: threshold ordering invalid")
    if kv["aggregate_script_sha256"] != self_sha256():
        raise AggError("PARAMS: aggregate_script_sha256 mismatch (script changed after RULE was fixed)")
    rule["kv"] = kv
    return rule


def parse_raw(path, rule, series, arm):
    """1 系列の raw を厳格に読み {(family, n): (secs, gflops, hash)} を返す。

    series は読むファイルの期待系列（t1／native）。全行の series 一致を要求し、t1 は rayon_threads=1 を要求する
    （native.txt を t1.txt に誤コピーしても受理しない）。arm は期待ラウンド（run_round.sh が記録する r{i}）で、
    全行の arm 一致を要求する（別ラウンドの raw を誤コピーしても独立 5 回計測として受理しない）。"""
    rows = {}
    for ln in read(path).splitlines():
        if not ln.strip():
            continue
        mt = ROW_RE.match(ln)
        if not mt:
            raise AggError(f"{path}: unparsable line: {ln!r}")
        d = mt.groupdict()
        fam = d["family"]
        if fam not in FAMILY_SPEC:
            raise AggError(f"{path}: unknown family {fam}")
        trans, fm, fk, ns = FAMILY_SPEC[fam]
        n, m, k = int(d["n"]), int(d["m"]), int(d["k"])
        if n not in ns or d["trans"] != trans or m != fm or k != fk:
            raise AggError(f"{path}: unexpected shape in line: {ln!r}")
        if int(d["nr"]) != rule["nr"] or d["isa"] != rule["isa"]:
            raise AggError(f"{path}: isa/nr mismatch (got {d['isa']}/{d['nr']}, RULE {rule['isa']}/{rule['nr']})")
        if d["arm"] != arm:
            raise AggError(f"{path}: arm mismatch (line has {d['arm']}, round expects {arm})")
        if d["series"] != series:
            raise AggError(f"{path}: series mismatch (line has {d['series']}, file expects {series})")
        if series == "t1" and int(d["threads"]) != 1:
            raise AggError(f"{path}: t1 series with rayon_threads != 1")
        try:
            secs, gf = float(d["secs"]), float(d["gflops"])
        except ValueError:
            raise AggError(f"{path}: non-numeric value in line: {ln!r}")
        if not (math.isfinite(secs) and secs > 0 and math.isfinite(gf) and gf > 0):
            raise AggError(f"{path}: non-finite or non-positive value in line: {ln!r}")
        key = (fam, n)
        if key in rows:
            raise AggError(f"{path}: duplicate row {key}")
        rows[key] = (secs, gf, d["hash"])
    expected = {(f, n) for f, (_, _, _, ns) in FAMILY_SPEC.items() for n in ns}
    if set(rows) != expected:
        raise AggError(f"{path}: row set mismatch: missing={sorted(expected - set(rows))} extra={sorted(set(rows) - expected)}")
    return rows


def judged_round(rows, fam, rule):
    """1 round・1 ファミリの導出値。P<=0 は p_ok=False。"""
    t10, t12, t24 = (rows[(fam, n)][0] for n in (N_LO, N_HI, N_NEXT))
    _, m, k, _ = FAMILY_SPEC[fam]
    p = t24 - t12
    t_ideal = t12 - p * (N_HI - N_LO) / N_HI
    waste = t10 - t_ideal
    gsq = rows[("ref_square_512", 512)][1]
    t_peak = 2.0 * m * N_LO * k / (gsq * 1e9)
    denom = t10 - t_peak
    return {"p": p, "p_ok": p > 0, "waste": waste, "w": waste / t10, "r": t10 / t12,
            "g": (waste / denom) if denom > 0 else None, "gflops10": rows[(fam, N_LO)][1]}


def classify(vals, rule):
    """RULE 項の順どおりに分類する（閾値は PARAMS の値）。"""
    if any(not v["p_ok"] for v in vals):
        return "判定不能（限界費用 P が 0 以下の round あり）"
    ws = [v["w"] for v in vals]
    gs = [v["g"] for v in vals]
    if all(g is not None for g in gs) and min(gs) >= rule["g_main"]:
        return "主因"
    if min(ws) >= rule["w_small"]:
        return "寄与あり（大）" if min(ws) >= rule["w_large"] else "寄与あり"
    if max(ws) < rule["w_small"]:
        return "無視できる"
    return "判定不能（揺れの中）"


def fmt(x, nd=4):
    return "未定義" if x is None else f"{x:.{nd}f}"


def load(d, rule):
    """round ごとの {series: rows} を返す（PASS のみ）と全 round の gate 一覧。"""
    gates, per_round = [], []
    for i in range(1, rule["rounds"] + 1):
        rd = os.path.join(d, f"round{i}")
        if not os.path.isdir(rd):
            raise AggError(f"round directory missing: {rd}")
        g = read(os.path.join(rd, "gate.txt")).strip()
        word = g.split()[0] if g else ""
        if word not in ("PASS", "TIMEOUT", "INVALID"):
            raise AggError(f"{rd}/gate.txt: unknown gate word {word!r}")
        gates.append((i, word))
        if word == "PASS":
            per_round.append((i, {s: parse_raw(os.path.join(rd, "raw", f"{s}.txt"), rule, s, f"r{i}") for s in SERIES}))
    if not per_round:
        raise AggError("no PASS round")
    # 同じキーの bits_hash は round 間・系列間で一致しなければならない（異なる n 同士は比べない）。
    ref = {}
    for i, by_s in per_round:
        for s, rows in by_s.items():
            for key, (_, _, h) in rows.items():
                if ref.setdefault(key, (h, i, s))[0] != h:
                    raise AggError(f"bits_hash mismatch for {key}: round{i}/{s} {h} vs round{ref[key][1]}/{ref[key][2]} {ref[key][0]}")
    return gates, per_round


def run_dir(d, rule_path, out_md, out_tsv):
    rule = parse_rule(rule_path)
    gates, per_round = load(d, rule)
    partial = len(per_round) != rule["rounds"]
    tsv = ["section\tfamily\tn\tmetric\tvalues_by_round\tmedian\tmin\tmax"]
    md = [f"# NR={rule['nr']} 端数の診断 集計（#{ISSUE}）" + ("（参考）" if partial else ""), "",
          f"- status: {'partial（参考。確度は書かない）' if partial else 'complete'}  gates: "
          + " ".join(f"round{i}={w}" for i, w in gates),
          f"- tree_sha: {rule['kv']['tree_sha']}  isa: {rule['isa']}  nr: {rule['nr']}  系列: t1 が判定、native は record_only", ""]
    md += ["## 判定（t1・n=10 の端数）", "",
           "| family | w 中央値 [最小〜最大] | g 中央値 [最小〜最大] | r10_12 中央値 [最小〜最大] | P 中央値 [s] | 分類 |"
           + ("" if partial else " 確度 |"), "|---|---|---|---|---|---|" + ("" if partial else "---|")]
    notes = []
    for fam in JUDGED:
        vals = [judged_round({**by_s["t1"]}, fam, rule) for _, by_s in per_round]
        cls = classify(vals, rule)

        def series_of(key):
            return [v[key] for v in vals]

        def stat(key):
            xs = series_of(key)
            if any(x is None for x in xs):
                return None, None, None
            return statistics.median(xs), min(xs), max(xs)

        row = {k: stat(k) for k in ("w", "g", "r", "p")}
        for k, name in (("w", "w"), ("g", "g"), ("r", "r10_12"), ("p", "P")):
            tsv.append(f"judged\t{fam}\t{N_LO}\t{name}\t" + ",".join(fmt(x, 9) for x in series_of(k)) + "\t"
                       + "\t".join(fmt(x, 9) for x in row[k]))
        tsv.append(f"judged\t{fam}\t{N_LO}\tclass\t\t{cls}\t\t")
        ref = rule["mm177_l2_fwd_gflops" if fam == "l2_fwd" else "mm177_l2_dw_gflops"]
        gmed = statistics.median(series_of("gflops10"))
        sane = rule["sanity_lo"] <= gmed / ref <= rule["sanity_hi"]
        if not sane:
            notes.append(f"{fam}: n=10 の t1 GFLOPS 中央値 {gmed:.2f} が MM-177 §3 の {ref} の "
                         f"{rule['sanity_lo']}〜{rule['sanity_hi']} 倍の外（環境差の可能性" + ("" if partial else "。確度は低") + "）")
        conf = "低" if not sane else "通常"
        md.append(f"| {fam} | " + " | ".join(
            f"{fmt(row[k][0])} [{fmt(row[k][1])}〜{fmt(row[k][2])}]" for k in ("w", "g", "r")) +
            f" | {fmt(row['p'][0], 9)} | {cls} |" + ("" if partial else f" {conf} |"))
    md += ["", "判定式と閾値は RULE.txt の PARAMS が正（w_small=%s, w_large=%s, g_main=%s）。" % (
        rule["w_small"], rule["w_large"], rule["g_main"])]
    if partial:
        md += ["", "注記: PASS でない round があるため status=partial。"]
    for n_ in notes:
        md += ["", "注記: " + n_]
    for s in SERIES:
        md += ["", f"## record_only: {s} 系列の n 別 GFLOPS（round 中央値 [最小〜最大]）", "", "| family | n | GFLOPS | median_secs [s] |", "|---|---|---|---|"]
        for fam, (_, _, _, ns) in FAMILY_SPEC.items():
            for n in ns:
                gf = [by_s[s][(fam, n)][1] for _, by_s in per_round]
                sc = [by_s[s][(fam, n)][0] for _, by_s in per_round]
                md.append(f"| {fam} | {n} | {statistics.median(gf):.3f} [{min(gf):.3f}〜{max(gf):.3f}] | {statistics.median(sc):.9f} |")
                tsv.append(f"record_only_{s}\t{fam}\t{n}\tgflops\t" + ",".join(f"{x:.3f}" for x in gf)
                           + f"\t{statistics.median(gf):.3f}\t{min(gf):.3f}\t{max(gf):.3f}")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    with open(out_tsv, "w", encoding="utf-8") as f:
        f.write("\n".join(tsv) + "\n")


def fingerprint(tree):
    """tree の指紋。diag-0140-template の aggregate_diag.py の fingerprint と同一ロジック。"""
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


EXAMPLE_REL = os.path.join("crates", "backend-cpu", "examples", "cpu_gemm_nr_tail_4152.rs")


def check_tree(tree, rule_path):
    """計測対象ツリーが RULE の example_sha256・tree_fingerprint と一致することを検証する（RULE 項7。計測前の必須検査）。"""
    rule = parse_rule(rule_path)
    full = os.path.join(os.path.abspath(tree), EXAMPLE_REL)
    try:
        with open(full, "rb") as f:
            got = hashlib.sha256(f.read()).hexdigest()
    except OSError as e:
        raise AggError(f"cannot read example source {full}: {e}")
    if got != rule["kv"]["example_sha256"]:
        raise AggError(f"example_sha256 mismatch (tree {got}, RULE {rule['kv']['example_sha256']})")
    fp, _, _ = fingerprint(tree)
    if fp != rule["kv"]["tree_fingerprint"]:
        raise AggError(f"tree_fingerprint mismatch (tree {fp}, RULE {rule['kv']['tree_fingerprint']})")


def check_template(path):
    """雛形の PARAMS が必須キーを過不足なく持ち、プレースホルダ対象だけが <…> であることを検査する。"""
    text = read(path)
    lines = text.splitlines()
    if not lines or not re.match(r"^RULE fixed-at-utc <.+>$", lines[0]):
        raise AggError("template: first line must carry a <…> timestamp placeholder")
    _, body = split_params(text)
    kv = parse_kv(body)
    if set(kv) != set(PARAM_KEYS):
        raise AggError(f"template: PARAMS keys mismatch: missing={set(PARAM_KEYS) - set(kv)} extra={set(kv) - set(PARAM_KEYS)}")
    for k, v in kv.items():
        has_ph = bool(re.search(r"<[^<>]+>", v))
        if k in PLACEHOLDER_KEYS and not has_ph:
            raise AggError(f"template: value of {k} is not a placeholder: {v!r}")
        if k in FIXED_KEYS and has_ph:
            raise AggError(f"template: value of {k} must be fixed in advance, not a placeholder: {v!r}")
    if kv["issue"] != str(ISSUE):
        raise AggError(f"template: issue must be {ISSUE}")
    if PLACEHOLDER_MARK not in text:
        raise AggError("template: no 【記入 marker found")


# ---------------------------------------------------------------- self-test
def _row(series, fam, n, secs, gf, nr=12, isa="Neon", h="0x0123456789abcdef", threads=1, arm="t"):
    trans, m, k, _ = FAMILY_SPEC[fam]
    return (f"arm={arm} series={series} family={fam} trans={trans} m={m} n={n} k={k} isa={isa} mr=8 nr={nr} "
            f"rayon_threads={threads} median_secs={secs:.9f} gflops={gf:.3f} bits_hash={h}")


def _synth(root, rule_path, rounds=5, t10=lambda r: 10e-6, t12=lambda r: 10e-6, t24=lambda r: 20e-6, mutate=None,
           gate=lambda i: "PASS", sq_gf=117.5):
    """合成 fixture。t10/t12/t24(r) が l2_fwd・l2_dw 共通の時間 [s]。"""
    for i in range(1, rounds + 1):
        rd = os.path.join(root, f"round{i}")
        os.makedirs(os.path.join(rd, "raw"))
        with open(os.path.join(rd, "gate.txt"), "w") as f:
            f.write(f"{gate(i)} load1=0.5\n")
        if gate(i) != "PASS":
            continue
        for s in SERIES:
            lines = []
            for fam, (_, m, k, ns) in FAMILY_SPEC.items():
                for n in ns:
                    if fam == "ref_square_512":
                        secs = 2.0 * 512 ** 3 / (sq_gf * 1e9)
                    else:
                        base = {10: t10(i), 12: t12(i), 24: t24(i)}.get(n, 12e-6 * n / 12 + 1e-6)
                        secs = base
                    gf = 2.0 * m * n * k / secs / 1e9
                    h = "0x%016x" % (hash((fam, n)) & 0xFFFFFFFFFFFFFFFF)
                    lines.append(_row(s, fam, n, secs, gf, h=h, arm=f"r{i}"))
            if mutate:
                lines = mutate(i, s, lines)
            with open(os.path.join(rd, "raw", f"{s}.txt"), "w") as f:
                f.write("\n".join(lines) + "\n")


def _write_rule(root, **over):
    kv = {"issue": "4152", "rounds": "5", "tree_sha": "a" * 40, "tree_fingerprint": "b" * 64, "example_sha256": "c" * 64,
          "aggregate_script_sha256": self_sha256(), "nr_expected": "12", "isa_expected": "Neon", "w_small": "0.03",
          "w_large": "0.10", "g_main": "0.50", "sanity_lo": "0.5", "sanity_hi": "2.0",
          "mm177_l2_fwd_gflops": "32.5", "mm177_l2_dw_gflops": "28.8"}
    kv.update(over)
    p = os.path.join(root, "RULE.txt")
    with open(p, "w") as f:
        f.write("RULE fixed-at-utc 2026-11-10T00:00:00Z\n本文\n\n## PARAMS\nfixed-at-utc: 2026-11-10T00:00:00Z\n"
                + "".join(f"{k}: {v}\n" for k, v in kv.items()))
    return p


def _expect_error(label, fn, needle):
    try:
        fn()
    except AggError as e:
        if needle not in str(e):
            raise AssertionError(f"{label}: wrong error: {e}")
        return
    raise AssertionError(f"{label}: AggError expected")


def self_test():
    def cls_for(**kw):
        with tempfile.TemporaryDirectory() as d:
            rp = _write_rule(d)
            _synth(d, rp, **kw)
            run_dir(d, rp, os.path.join(d, "o.md"), os.path.join(d, "o.tsv"))
            tsv = read(os.path.join(d, "o.tsv"))
            return [ln.split("\t")[5] for ln in tsv.splitlines() if "\tclass\t" in ln]

    # P=10us/12 レーン相当。t12=10, t24=20 -> t_ideal(10)=10-10*2/12=8.333us
    # 無視できる: t10 = t_ideal (waste 0)
    assert cls_for(t10=lambda r: 8.333e-6) == ["無視できる", "無視できる"]
    # 寄与あり: waste 約 5%（t10=8.77us -> w=0.05）。g は小さい（分母が大きい）
    assert cls_for(t10=lambda r: 8.77e-6) == ["寄与あり", "寄与あり"]
    # 寄与あり（大）: w>=0.10（t10=9.6us -> w≈0.132）。g は t_peak 次第で主因にならない小さな値にする
    assert cls_for(t10=lambda r: 9.6e-6, sq_gf=1.0) == ["寄与あり（大）"] * 2
    # 主因: g>=0.5。sq_gf を大きくして t_peak を 0 近くにし、waste が t10-t_peak の半分以上になるようにする
    assert cls_for(t10=lambda r: 20e-6, sq_gf=1e6) == ["主因", "主因"]
    # 帯をまたぐ: round により w が 0.03 の上下
    assert cls_for(t10=lambda r: 8.333e-6 if r % 2 else 8.77e-6) == ["判定不能（揺れの中）"] * 2
    # P<=0
    assert cls_for(t24=lambda r: 10e-6) == ["判定不能（限界費用 P が 0 以下の round あり）"] * 2

    def run_err(label, needle, rule_kw=None, **kw):
        with tempfile.TemporaryDirectory() as d:
            rp = _write_rule(d, **(rule_kw or {}))
            _synth(d, rp, **kw)
            _expect_error(label, lambda: run_dir(d, rp, os.path.join(d, "o.md"), os.path.join(d, "o.tsv")), needle)

    def mut_hash(i, s, lines):
        return [ln.replace("0x", "0xf", 1)[:-1] if (i == 3 and "family=l2_fwd" in ln and " n=10 " in ln) else ln for ln in lines]
    run_err("hash mismatch", "bits_hash mismatch", mutate=mut_hash)
    run_err("nr=16 (x86)", "isa/nr mismatch", mutate=lambda i, s, ls: [ln.replace("nr=12", "nr=16") for ln in ls])
    run_err("duplicate row", "duplicate row", mutate=lambda i, s, ls: ls + [ls[0]] if i == 2 else ls)
    run_err("missing row", "row set mismatch", mutate=lambda i, s, ls: ls[1:] if i == 2 else ls)
    run_err("unknown family", "unknown family", mutate=lambda i, s, ls: ls[:-1] + [ls[-1].replace("ref_square_512", "bogus")])
    run_err("zero secs", "non-finite or non-positive", mutate=lambda i, s, ls: [ls[0].replace("median_secs=", "median_secs=-", 1)] + ls[1:])
    run_err("t1 threads", "rayon_threads != 1",
            mutate=lambda i, s, ls: [ln.replace("rayon_threads=1", "rayon_threads=4") for ln in ls] if s == "t1" else ls)
    run_err("native copied as t1", "series mismatch",
            mutate=lambda i, s, ls: [ln.replace("series=native", "series=t1") for ln in ls] if s == "native" else ls)
    run_err("series=native in t1.txt", "series mismatch",
            mutate=lambda i, s, ls: [ln.replace("series=t1", "series=native") for ln in ls] if s == "t1" else ls)
    run_err("raw copied from another round", "arm mismatch",
            mutate=lambda i, s, ls: [ln.replace("arm=r2 ", "arm=r1 ") for ln in ls] if i == 2 and s == "t1" else ls)
    run_err("missing round", "round directory missing", rounds=4)
    run_err("no PASS", "no PASS round", gate=lambda i: "TIMEOUT")
    run_err("bad rule sha", "aggregate_script_sha256 mismatch", rule_kw={"aggregate_script_sha256": "d" * 64})
    with tempfile.TemporaryDirectory() as d:  # プレースホルダ残存
        p = os.path.join(d, "RULE.txt")
        with open(p, "w") as f:
            f.write("RULE fixed-at-utc <YYYY-MM-DDThh:mm:ssZ>\n【記入: x】\n\n## PARAMS\nissue: 4152\n")
        _expect_error("placeholder", lambda: parse_rule(p), "placeholder")
    # partial（TIMEOUT 1 本）は通り、status に出る
    with tempfile.TemporaryDirectory() as d:
        rp = _write_rule(d)
        _synth(d, rp, gate=lambda i: "TIMEOUT" if i == 2 else "PASS")
        run_dir(d, rp, os.path.join(d, "o.md"), os.path.join(d, "o.tsv"))
        md_ = read(os.path.join(d, "o.md"))
        assert "partial" in md_ and "（参考）" in md_ and "確度" not in md_.replace("確度は書かない", "")
    with tempfile.TemporaryDirectory() as d:  # complete は確度欄を出す
        rp = _write_rule(d)
        _synth(d, rp)
        run_dir(d, rp, os.path.join(d, "o.md"), os.path.join(d, "o.tsv"))
        assert "| 確度 |" in read(os.path.join(d, "o.md"))
    with tempfile.TemporaryDirectory() as d:  # check-tree: example／指紋の照合
        tree = os.path.join(d, "tree")
        os.makedirs(os.path.join(tree, "crates", "backend-cpu", "examples"))
        ex = os.path.join(tree, EXAMPLE_REL)
        with open(ex, "w") as f:
            f.write("fn main(){}\n")
        fp, _, _ = fingerprint(tree)
        exh = hashlib.sha256(open(ex, "rb").read()).hexdigest()
        rp = _write_rule(d, example_sha256=exh, tree_fingerprint=fp)
        check_tree(tree, rp)
        rp2 = _write_rule(d, example_sha256="e" * 64, tree_fingerprint=fp)
        _expect_error("example mismatch", lambda: check_tree(tree, rp2), "example_sha256 mismatch")
        with open(ex, "w") as f:
            f.write("fn main(){1;}\n")
        _expect_error("tree changed", lambda: check_tree(tree, rp), "example_sha256 mismatch")
    # 健全性範囲外の注記（n=10 の GFLOPS が MM-177 の値から外れる）
    with tempfile.TemporaryDirectory() as d:
        rp = _write_rule(d, mm177_l2_fwd_gflops="1000")
        _synth(d, rp)
        run_dir(d, rp, os.path.join(d, "o.md"), os.path.join(d, "o.tsv"))
        assert "環境差の可能性" in read(os.path.join(d, "o.md"))
    # 雛形検査
    tpl = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RULE.template.txt")
    if os.path.exists(tpl):
        check_template(tpl)
    with tempfile.TemporaryDirectory() as d:  # 固定済み RULE は雛形検査で拒否される
        rp = _write_rule(d)
        _expect_error("fixed rule as template", lambda: check_template(rp), "first line")
        parse_rule(rp)
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
            parse_rule(p)
            print("rule OK")
        elif "--check-tree" in argv:
            t = opt("--check-tree")
            r = opt("--rule")
            if not t or not r:
                raise AggError("--check-tree <tree-dir> --rule <RULE.txt> が必要")
            check_tree(t, r)
            print("tree OK")
        elif "--fingerprint" in argv:
            fp, n, _ = fingerprint(opt("--fingerprint"))
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
