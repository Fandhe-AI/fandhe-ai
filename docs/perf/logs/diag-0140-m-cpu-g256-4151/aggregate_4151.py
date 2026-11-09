#!/usr/bin/env python3
"""M-CPU-G256（Mac mini・CPU gemm N=256）の固定費診断 (#4151) を機械集計する。

役割: example `cpu_gemm_n256_fixed_cost_4151` の出力（4 腕 × 系列）と、ラウンドごとの窓の値
（values.tsv）を、RULE.txt の PARAMS で測る前に固定した規則どおりに集計し、区間の内訳と
分類（自分たちの制約／ハードの上限／計測窓の定義／分類保留）・確度を出す。
`docs/perf/logs/diag-0140-template/aggregate_diag.py` は F1／F2 の 5 量しか扱わないため、本 issue 用に別に持つ
（#4148・#4150 と同じ理由）。実測値は含まない。

入力（<dir> 配下）:
  RULE.txt                      1 行目 `RULE fixed-at-utc …` と末尾の `## PARAMS`（RULE.template.txt が雛形）
  round{1..N}/gate.txt          1 語目が PASS／TIMEOUT／INVALID。PASS のラウンドだけ読む
  round{1..N}/values.tsv        ヘッダ `quantity<TAB>value`。W_off W_on Wf_off Wf_on O K_opp（ms）と
                                c_sme_kc0_ns c_sme_intercept_ns（ns。sme_microcall_fixed_cost_probe の値）
  round{1..N}/raw/example-{off-base,on-base,off-timed,on-timed}.txt   example の標準出力
出力: aggregate.md（人間用）と aggregate.tsv（機械可読）。

厳格パース: 欠損・重複・未知・非有限・0／負・プレースホルダ残存・スクリプトハッシュ不一致・bits_hash 不一致・
区間和の超過・マイクロカーネル呼び出し回数の下限割れはすべて AggError で停止する（寛容に読まない）。
RULE 固定後の本ファイル改変は `aggregate_script_sha256` の不一致で検出される。標準ライブラリのみ。
`python3 -I` で実行すること。

使い方:
  python3 -I aggregate_4151.py --dir <記録先> [--rule RULE.txt] [--out aggregate.md] [--tsv aggregate.tsv]
  python3 -I aggregate_4151.py --self-test
  python3 -I aggregate_4151.py --check-template [RULE.template.txt]
  python3 -I aggregate_4151.py --fingerprint <tree-dir> [--list-out <file>]
"""
import hashlib
import math
import os
import re
import statistics
import sys
import tempfile

ARMS = ["off-base", "on-base", "off-timed", "on-timed"]
SERIES = ["cold_first_call", "hot", "gap", "serial_t1", "forkjoin_floor_hot", "forkjoin_floor_gap"]
TIMED_SERIES = ["hot", "gap"]
REQUIRED_QUANTITIES = ["W_off", "W_on", "Wf_off", "Wf_on", "O", "K_opp"]
# probe 由来の上限参照用の量。OFF 腕の分類に不要で、取得不能なら values.tsv に書かない（欠損時は SME 上限欄のみ unavailable）
OPTIONAL_QUANTITIES = ["c_sme_kc0_ns", "c_sme_intercept_ns"]
QUANTITIES = REQUIRED_QUANTITIES + OPTIONAL_QUANTITIES
D_KEYS = ["d_regions", "d_mr", "d_nr", "d_jobs", "d_p_used", "d_n_calls", "d_call_ns", "d_pre_ns", "d_grid_ns",
          "d_split_ns", "d_fork_wake_min_ns", "d_fork_wake_max_ns", "d_stage_in_ns", "d_alloc_ns", "d_pack_b_ns",
          "d_pack_a_ns", "d_kernel_ns", "d_stage_out_ns", "d_kernel_crit_ns", "d_join_ns", "d_region_wall_ns",
          "d_kernel_ideal_ns", "d_imbalance_ns", "d_parts_ideal_ns"]
MEASURE_CALLS = 500  # RULE 項4 が固定する測定回数
ROW_BASE_KEYS = ["arm", "series", "m", "n", "k", "rayon_threads", "calls", "median_ns", "mean_ns", "bits_hash"]
ENV_KEYS = ["arm", "rayon_threads", "sme_os_flag", "sme_svl_bytes", "sme_kernel_enabled", "sme_production_enabled",
            "thread_limit_effective",
            "thread_limit_env_override", "small_shape_cap_eligible", "timer_tick_ns", "timer_pair_ns", "gap_us"]
PARAM_KEYS = ["fixed-at-utc", "issue", "rounds", "tree_sha", "tree_fingerprint", "aggregate_script_sha256",
              "reach_threshold", "noise_band", "cell", "scoreboard_ratio", "prev_class", "kernel_owner", "gap_us",
              "kc", "perturbation_max", "sum_tolerance", "dim"]
TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
CELL_RE = re.compile(r"^[A-Z]-[A-Z]{3,4}-[A-Z0-9]+$")
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
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError as e:
        raise AggError(f"cannot read {path}: {e}")


def pf(s, what):
    try:
        v = float(s)
    except ValueError:
        raise AggError(f"{what}: not a number: {s!r}")
    if not math.isfinite(v):
        raise AggError(f"{what}: not finite: {s!r}")
    return v


def ppos(s, what):
    v = pf(s, what)
    if v <= 0:
        raise AggError(f"{what}: must be > 0: {s!r}")
    return v


# ---------------------------------------------------------------- RULE
def split_params(text):
    lines = text.splitlines()
    if not lines or not lines[0].startswith("RULE fixed-at-utc "):
        raise AggError("RULE: first line must start with 'RULE fixed-at-utc '")
    idx = [i for i, l in enumerate(lines) if l.strip() == "## PARAMS"]
    if len(idx) != 1:
        raise AggError("RULE: exactly one '## PARAMS' required")
    return lines[0], [l for l in lines[idx[0] + 1:] if l.strip()]


def parse_kv(body):
    kv = {}
    for l in body:
        if ":" not in l:
            raise AggError(f"PARAMS: bad line {l!r}")
        k, v = l.split(":", 1)
        k, v = k.strip(), v.strip()
        if k not in PARAM_KEYS:
            raise AggError(f"PARAMS: unknown key {k!r}")
        if k in kv:
            raise AggError(f"PARAMS: duplicate key {k!r}")
        kv[k] = v
    return kv


def parse_rule(text):
    first, body = split_params(text)
    ts = first[len("RULE fixed-at-utc "):].strip()
    if not TS_RE.match(ts):
        raise AggError(f"RULE: bad timestamp {ts!r}")
    if PLACEHOLDER_MARK in text:
        raise AggError("RULE: unreplaced placeholder (【記入) remains")
    if any("<" in l or ">" in l for l in body):
        raise AggError("PARAMS: unreplaced placeholder (<…>) remains")
    kv = parse_kv(body)
    for k in PARAM_KEYS:
        if k not in kv:
            raise AggError(f"PARAMS: missing key {k!r}")
    if kv["fixed-at-utc"] != ts:
        raise AggError("PARAMS: fixed-at-utc differs from first line")
    if kv["issue"] != "4151":
        raise AggError("PARAMS: issue must be 4151")
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
    if kv["noise_band"] != "NB-RANGE5":
        raise AggError("PARAMS: noise_band must be NB-RANGE5")
    if not CELL_RE.match(kv["cell"]):
        raise AggError("PARAMS: bad cell id")
    if kv["kernel_owner"] not in ("hardware", "constraint"):
        raise AggError("PARAMS: kernel_owner must be hardware|constraint")
    for k in ("reach_threshold", "scoreboard_ratio", "perturbation_max", "sum_tolerance"):
        ppos(kv[k], k)
    for k in ("gap_us", "kc", "dim"):
        if not re.match(r"^[1-9]\d*$", kv[k]):
            raise AggError(f"PARAMS: {k} must be a positive integer")
    return {"ts": ts, "rounds": int(kv["rounds"]), "tree_sha": kv["tree_sha"], "fp": kv["tree_fingerprint"],
            "thr": float(kv["reach_threshold"]), "cell": kv["cell"], "ratio": float(kv["scoreboard_ratio"]),
            "prev_class": kv["prev_class"], "owner": kv["kernel_owner"], "gap_us": int(kv["gap_us"]),
            "kc": int(kv["kc"]), "pert_max": float(kv["perturbation_max"]),
            "sum_tol": float(kv["sum_tolerance"]), "dim": int(kv["dim"])}


# ---------------------------------------------------------------- raw parse
def parse_tokens(line, head):
    toks = line.split()
    if toks[0] != head:
        raise AggError(f"unexpected line head {toks[0]!r}")
    kv = {}
    for t in toks[1:]:
        if "=" not in t:
            raise AggError(f"bad token {t!r} in {head}")
        k, v = t.split("=", 1)
        if k in kv:
            raise AggError(f"duplicate key {k!r} in {head}")
        kv[k] = v
    return kv


def parse_example(text, arm, rule, where):
    env = None
    rows = {}
    end = False
    for line in text.splitlines():
        if not line.strip():
            continue
        head = line.split()[0]
        if head == "ENV4151":
            if env is not None:
                raise AggError(f"{where}: duplicate ENV4151")
            env = parse_tokens(line, "ENV4151")
        elif head == "ROW4151":
            kv = parse_tokens(line, "ROW4151")
            s = kv.get("series")
            if s not in SERIES:
                raise AggError(f"{where}: unknown series {s!r}")
            if s in rows:
                raise AggError(f"{where}: duplicate series {s}")
            rows[s] = kv
        elif head == "END4151":
            end = True
        else:
            raise AggError(f"{where}: unknown line {head!r}")
    if env is None or not end:
        raise AggError(f"{where}: ENV4151/END4151 missing")
    if set(env) != set(ENV_KEYS):
        raise AggError(f"{where}: ENV keys mismatch: {sorted(set(env) ^ set(ENV_KEYS))}")
    if env["arm"] != arm:
        raise AggError(f"{where}: arm label {env['arm']!r} != {arm!r}")
    if int(env["gap_us"]) != rule["gap_us"]:
        raise AggError(f"{where}: gap_us differs from RULE")
    want_sme = "true" if arm.startswith("on-") else "false"
    # arm 整合はビルドの本番有効フラグ（SME_PRODUCTION_ENABLED）で検証する。ON 腕はさらに
    # ハード能力 sme_kernel_enabled=true も必須（false だと try_new が None で NEON にフォールバックし、
    # ON 腕として SME を測っていないため。RULE 項 3 の停止条件と一致させる。#4192 レビュー指摘）
    if env["sme_production_enabled"] != want_sme:
        raise AggError(f"{where}: arm {arm} requires sme_production_enabled={want_sme}（ON/OFF ビルドの取り違え防止）")
    if env["sme_kernel_enabled"] not in ("true", "false"):
        raise AggError(f"{where}: sme_kernel_enabled must be true/false")
    if arm.startswith("on-") and env["sme_kernel_enabled"] != "true":
        raise AggError(f"{where}: arm {arm} requires sme_kernel_enabled=true（SME 非対応ホストでは NEON を測ってしまう）")
    if env["thread_limit_env_override"] != "false":
        raise AggError(f"{where}: RAYON_NUM_THREADS must not be set")
    if sorted(rows) != sorted(SERIES):
        raise AggError(f"{where}: missing series {sorted(set(SERIES) - set(rows))}")
    timed = arm.endswith("-timed")
    out = {}
    for s, kv in rows.items():
        want = list(ROW_BASE_KEYS) + (D_KEYS if (timed and s in TIMED_SERIES) else [])
        if set(kv) != set(want):
            raise AggError(f"{where}: series {s} keys mismatch: {sorted(set(kv) ^ set(want))}")
        if kv["arm"] != arm:
            raise AggError(f"{where}: row arm mismatch")
        for k in ("m", "n", "k"):
            if int(kv[k]) != rule["dim"]:
                raise AggError(f"{where}: {k} != dim")
        if int(kv["rayon_threads"]) != int(env["rayon_threads"]):
            raise AggError(f"{where}: rayon_threads differs between ENV and ROW")
        if not re.match(r"^0x[0-9a-f]{16}$", kv["bits_hash"]):
            raise AggError(f"{where}: bad bits_hash")
        # RULE 項4 の事前登録回数: cold_first_call は 1 回、それ以外の系列は測定 500 回
        want_calls = 1 if s == "cold_first_call" else MEASURE_CALLS
        if int(kv["calls"]) != want_calls:
            raise AggError(f"{where}: series {s} calls {kv['calls']} != {want_calls}（RULE 項4）")
        row = {"bits_hash": kv["bits_hash"], "median": ppos(kv["median_ns"], f"{where}:{s}:median_ns"),
               "mean": ppos(kv["mean_ns"], f"{where}:{s}:mean_ns"), "calls": int(kv["calls"])}
        if timed and s in TIMED_SERIES:
            d = {k: pf(kv[k], f"{where}:{s}:{k}") for k in D_KEYS}
            for k, v in d.items():
                if v < 0:
                    raise AggError(f"{where}:{s}:{k} negative")
            check_timed(d, row["calls"], rule, f"{where}:{s}")
            row["d"] = d
        out[s] = row
    return {"env": env, "rows": out}


def check_timed(d, calls, rule, where):
    if int(d["d_regions"]) != calls:
        raise AggError(f"{where}: d_regions != calls")
    p = d["d_p_used"]
    if p < 1:
        raise AggError(f"{where}: d_p_used < 1")
    lo = math.ceil(rule["dim"] / d["d_mr"]) * math.ceil(rule["dim"] / d["d_nr"]) * math.ceil(rule["dim"] / rule["kc"])
    if d["d_n_calls"] < lo:
        raise AggError(f"{where}: d_n_calls {d['d_n_calls']} below hand-computed lower bound {lo}")
    # region ごとに計算済みの値（patch 側で region 内割り算→中央値）を使う。中央値同士の比は使わない
    if d["d_parts_ideal_ns"] > d["d_region_wall_ns"] * (1 + rule["sum_tol"]):
        raise AggError(f"{where}: interval sum exceeds region_wall beyond tolerance")
    if d["d_kernel_crit_ns"] * (1 + 1e-9) < d["d_kernel_ideal_ns"]:
        raise AggError(f"{where}: kernel_crit below kernel_ideal")


def parse_values(text, where):
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines or lines[0].split("\t") != ["quantity", "value"]:
        raise AggError(f"{where}: header must be 'quantity<TAB>value'")
    vals = {}
    for l in lines[1:]:
        p = l.split("\t")
        if len(p) != 2:
            raise AggError(f"{where}: bad line {l!r}")
        if p[0] not in QUANTITIES:
            raise AggError(f"{where}: unknown quantity {p[0]!r}")
        if p[0] in vals:
            raise AggError(f"{where}: duplicate quantity {p[0]!r}")
        vals[p[0]] = ppos(p[1], f"{where}:{p[0]}")
    miss = [q for q in REQUIRED_QUANTITIES if q not in vals]
    if miss:
        raise AggError(f"{where}: missing quantities {miss}")
    return vals


# ---------------------------------------------------------------- compute
def ms(ns):
    return ns / 1e6


def round_metrics(rule, ex, vals):
    out = {}
    for a, w, wf in (("off", vals["W_off"], vals["Wf_off"]), ("on", vals["W_on"], vals["Wf_on"])):
        g = ms(ex[a + "-base"]["rows"]["hot"]["median"])
        d = ex[a + "-timed"]["rows"]["hot"]["d"]
        ki = ms(d["d_kernel_ideal_ns"])
        kc = ms(d["d_kernel_crit_ns"])
        fdrv = g - ki
        out[a] = {
            "G": g, "kernel_ideal": ki, "kernel_crit": kc, "imbalance": ms(d["d_imbalance_ns"]), "F_drv": fdrv,
            "perturbation": (ex[a + "-timed"]["rows"]["hot"]["median"] - ex[a + "-base"]["rows"]["hot"]["median"])
            / ex[a + "-base"]["rows"]["hot"]["median"],
            "F2": vals["O"] / wf,
            "F1": denom(vals["O"], w - g + vals["K_opp"], "W-G+K_opp"),
            "F1_drv": denom(vals["O"], w - fdrv, "W-F_drv"),
            "obs": vals["O"] / w,
        }
        if a == "on":
            nc, pu = d["d_n_calls"], d["d_p_used"]
            for key, q in (("sme_fixed_ub_kc0", "c_sme_kc0_ns"), ("sme_fixed_ub_icpt", "c_sme_intercept_ns")):
                if q in vals:
                    out[a][key] = ms(nc * vals[q] / pu)
    return out


def denom(num, den, what):
    if den <= 0:
        raise AggError(f"non-positive denominator ({what}): {den}")
    return num / den


def side(vals, thr):
    if min(vals) > thr:
        return "above"
    if max(vals) < thr:
        return "below"
    return "straddle"


def classify(rule, per_round, formal):
    thr = rule["thr"]
    med = {k: statistics.median(r["off"][k] for r in per_round) for k in ("F1", "F2", "F1_drv")}

    def vs(k):
        return [r["off"][k] for r in per_round]

    if med["F2"] >= thr:
        cls, used, reason = CLASS_WINDOW, ["F2"], "fresh 窓の再現で届く"
    elif med["F1"] >= thr:
        used = ["F2", "F1", "F1_drv"]  # F2 が閾値未満という判定にも依存する（RULE 項 6）
        if med["F1_drv"] >= thr:
            cls = CLASS_OWN
            reason = "GEMM 駆動部の固定費（区間別の数字は表）。「ハードの上限」は成り立たない"
        else:
            cls = CLASS_HW if rule["owner"] == "hardware" else CLASS_OWN
            reason = "駆動部の固定費を除いても届かず、マイクロカーネルの速度で決まる"
    else:
        cls, used, reason = CLASS_HOLD, ["F2", "F1"], "GEMM の外の区間が主因。別の診断へ"
    sides = {k: side(vs(k), thr) for k in used}
    pert_ok = all(abs(r[a]["perturbation"]) <= rule["pert_max"] for r in per_round for a in ("off", "on"))
    high = formal and pert_ok and all(s != "straddle" for s in sides.values())
    if not high:
        why = []
        if not formal:
            why.append("ゲート不通過のラウンドあり")
        if not pert_ok:
            why.append("計時の摂動が上限超過")
        if any(s == "straddle" for s in sides.values()):
            why.append("ラウンド別の値が閾値をまたぐ")
        reason = "確度低（" + "・".join(why) + "）: " + reason
    return cls, reason, high, med, sides


def aggregate(dirp, rule_name="RULE.txt"):
    rule = parse_rule(read(os.path.join(dirp, rule_name)))
    per_round, bits, rows_by_round, statuses = [], set(), [], []
    for i in range(1, rule["rounds"] + 1):
        rd = os.path.join(dirp, f"round{i}")
        if not os.path.isdir(rd):
            raise AggError(f"missing round directory: round{i}")
        g = read(os.path.join(rd, "gate.txt")).split()
        if not g or g[0] not in ("PASS", "TIMEOUT", "INVALID"):
            raise AggError(f"round{i}: bad gate.txt")
        if g[0] == "INVALID":
            raise AggError(f"round{i}: gate INVALID")
        statuses.append(g[0])
        if g[0] != "PASS":
            continue
        ex = {}
        for a in ARMS:
            ex[a] = parse_example(read(os.path.join(rd, "raw", f"example-{a}.txt")), a, rule, f"round{i}/{a}")
            for r in ex[a]["rows"].values():
                bits.add(r["bits_hash"])
        vals = parse_values(read(os.path.join(rd, "values.tsv")), f"round{i}/values.tsv")
        per_round.append(round_metrics(rule, ex, vals))
        rows_by_round.append(ex)
    if not per_round:
        raise AggError("no PASS round")
    if len(bits) != 1:
        raise AggError(f"bits_hash mismatch across arms/series/rounds: {sorted(bits)}")
    formal = all(s == "PASS" for s in statuses)
    cls, reason, high, med, sides = classify(rule, per_round, formal)
    obs = statistics.median(r["off"]["obs"] for r in per_round)
    drift = abs(obs - rule["ratio"]) / rule["ratio"]
    return {"rule": rule, "per_round": per_round, "rows": rows_by_round, "statuses": statuses, "cls": cls,
            "reason": reason, "high": high, "med": med, "sides": sides, "obs": obs, "drift": drift,
            "drifted": drift > DRIFT_LIMIT, "formal": formal}


def fnum(x):
    return f"{x:.4f}"


def render(res):
    r = res["rule"]
    pr = res["per_round"]
    md = [f"# 集計結果（#4151・{r['cell']}）", "",
          f"- status: {'formal' if res['formal'] else 'partial（参考）'}・ラウンド {res['statuses']}",
          f"- tree_sha: {r['tree_sha']}", "",
          f"## 判定: {res['cls']}（確度{'高' if res['high'] else '低'}）", "", f"- 理由: {res['reason']}",
          f"- 現分類（prev_class）: {r['prev_class']}",
          f"- 閾値 {r['thr']}・判定に使った式の側: {res['sides']}",
          f"- 観測比 O÷W_OFF の中央値 {fnum(res['obs'])}（スコアボード {r['ratio']}・乖離 {res['drift']:.1%}"
          f"{'・10% 超: ドリフトを記録' if res['drifted'] else ''}）", "",
          "## 式の値（OFF 腕＝0.14.0 の本番構成。中央値〔ラウンド別〕）", ""]
    for k in ("F2", "F1", "F1_drv"):
        md.append(f"- {k}: {fnum(res['med'][k])} 〔{', '.join(fnum(x['off'][k]) for x in pr)}〕")
    md += ["", "## 区間の内訳（ms。timed 腕・hot 系列・ラウンド中央値。時計は wall）", "",
           "| 量 | OFF | ON |", "|---|---|---|"]
    names = [("G（base の hot 中央値）", "G"), ("kernel_ideal = Σkernel÷P_used", "kernel_ideal"),
             ("kernel_crit", "kernel_crit"), ("imbalance", "imbalance"), ("F_drv = G − kernel_ideal", "F_drv"),
             ("perturbation（timed÷base−1）", "perturbation")]
    for label, k in names:
        md.append(f"| {label} | {fnum(statistics.median(x['off'][k] for x in pr))} | "
                  f"{fnum(statistics.median(x['on'][k] for x in pr))} |")
    md += ["", "区間の raw 内訳（ns・hot／gap 系列・ラウンド中央値）:", "",
           "| 区間 | off hot | off gap | on hot | on gap |", "|---|---|---|---|---|"]
    for k in D_KEYS[1:]:
        cells = []
        for a in ("off-timed", "on-timed"):
            for s in TIMED_SERIES:
                cells.append(f"{statistics.median(e[a]['rows'][s]['d'][k] for e in res['rows']):.0f}")
        md.append(f"| {k} | " + " | ".join(cells) + " |")
    md += ["", "系列の中央値（ms・ラウンド中央値）:", "", "| 系列 | off-base | on-base |", "|---|---|---|"]
    for s in SERIES:
        c = [fnum(ms(statistics.median(e[a]["rows"][s]["median"] for e in res["rows"]))) for a in ("off-base", "on-base")]
        md.append(f"| {s}{'（記録のみ）' if s == 'cold_first_call' else ''} | {c[0]} | {c[1]} |")
    def sme_ub(key):
        # 全ラウンドに値があるときだけ中央値を出す。欠損ラウンドがあれば推定で埋めず unavailable
        if all(key in x["on"] for x in pr):
            return fnum(statistics.median(x["on"][key] for x in pr))
        return "unavailable"

    md += ["", "## SME ON の場合（記録のみ。分類は動かさない）", "",
           f"- F2 {fnum(statistics.median(x['on']['F2'] for x in pr))}・F1 {fnum(statistics.median(x['on']['F1'] for x in pr))}"
           f"・F1_drv {fnum(statistics.median(x['on']['F1_drv'] for x in pr))}",
           f"- smstart／smstop を含む 1 呼び出し固定費の上限 × 呼び出し回数 ÷ P_used（ms・上限であり実測ではない）: "
           f"kc=0 直接計時 {sme_ub('sme_fixed_ub_kc0')}・"
           f"切片 {sme_ub('sme_fixed_ub_icpt')}（kernel の内側。F_drv には足さない）", ""]
    tsv = ["key\tvalue", f"class\t{res['cls']}", f"confidence\t{'high' if res['high'] else 'low'}",
           f"status\t{'formal' if res['formal'] else 'partial'}", f"obs_ratio\t{res['obs']:.6f}",
           f"drift\t{res['drift']:.6f}"]
    for k in ("F2", "F1", "F1_drv"):
        tsv.append(f"{k}_median\t{res['med'][k]:.6f}")
    for a in ("off", "on"):
        for k in ("G", "kernel_ideal", "kernel_crit", "imbalance", "F_drv", "perturbation"):
            tsv.append(f"{a}_{k}\t{statistics.median(x[a][k] for x in pr):.6f}")
    return "\n".join(md), "\n".join(tsv) + "\n"


# ---------------------------------------------------------------- fingerprint / template
def fingerprint(tree):
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
    text = read(path)
    lines = text.splitlines()
    if not lines or not re.match(r"^RULE fixed-at-utc <.+>$", lines[0]):
        raise AggError("template: first line must carry a <…> timestamp placeholder")
    _, body = split_params(text)
    kv = parse_kv(body)
    if set(kv) != set(PARAM_KEYS):
        raise AggError(f"template: PARAMS keys mismatch: {sorted(set(kv) ^ set(PARAM_KEYS))}")
    for k, v in kv.items():
        if not (v.startswith("<") and v.endswith(">")):
            raise AggError(f"template: PARAMS value of {k!r} must be a <…> placeholder")
    if PLACEHOLDER_MARK not in text:
        raise AggError("template: no 【記入 mark")
    try:
        parse_rule(text)
    except AggError:
        return
    raise AggError("template: parse_rule must reject the template")


# ---------------------------------------------------------------- self-test
def _expect(fn, label, frag=None):
    try:
        fn()
    except AggError as e:
        if frag and frag not in str(e):
            raise SystemExit(f"self-test FAIL ({label}): wrong error: {e}")
        return
    raise SystemExit(f"self-test FAIL ({label}): AggError expected")


def _rule_text(script, rounds=2, thr="1.00"):
    return "\n".join([
        "RULE fixed-at-utc 2000-01-01T00:00:00Z", "本文", "", "## PARAMS",
        "fixed-at-utc: 2000-01-01T00:00:00Z", "issue: 4151", f"rounds: {rounds}", "tree_sha: " + "a" * 40,
        "tree_fingerprint: " + "b" * 64, "aggregate_script_sha256: " + script, f"reach_threshold: {thr}",
        "noise_band: NB-RANGE5", "cell: M-CPU-G256", "scoreboard_ratio: 0.69", "prev_class: ハードの上限",
        "kernel_owner: hardware", "gap_us: 50", "kc: 256", "perturbation_max: 0.10", "sum_tolerance: 0.15",
        "dim: 256", ""])


def _example(arm, hot_ns=200000, kernel_total=1_000_000, bits="0x" + "1" * 16, mutate=None):
    timed = arm.endswith("-timed")
    on = arm.startswith("on-")
    env = {"arm": arm, "rayon_threads": 10, "sme_os_flag": "true", "sme_svl_bytes": 64,
           "sme_kernel_enabled": "true",  # ハード能力（M4 は arm 非依存で true）
           "sme_production_enabled": "true" if on else "false", "thread_limit_effective": 10,
           "thread_limit_env_override": "false", "small_shape_cap_eligible": "false", "timer_tick_ns": 41,
           "timer_pair_ns": "80.00", "gap_us": 50}
    lines = ["ENV4151 " + " ".join(f"{k}={v}" for k, v in env.items())]
    for s in SERIES:
        kv = {"arm": arm, "series": s, "m": 256, "n": 256, "k": 256, "rayon_threads": 10,
              "calls": 1 if s == "cold_first_call" else 500, "median_ns": hot_ns, "mean_ns": hot_ns + 1,
              "bits_hash": bits}
        if timed and s in TIMED_SERIES:
            kv.update({"d_regions": 500, "d_mr": 16 if on else 8, "d_nr": 16 if on else 12, "d_jobs": 20,
                       "d_p_used": 10, "d_n_calls": 800, "d_call_ns": hot_ns, "d_pre_ns": 4000, "d_grid_ns": 900,
                       "d_split_ns": 3000, "d_fork_wake_min_ns": 12000, "d_fork_wake_max_ns": 18000,
                       "d_stage_in_ns": 20000, "d_alloc_ns": 40000, "d_pack_b_ns": 60000, "d_pack_a_ns": 80000,
                       "d_kernel_ns": kernel_total, "d_stage_out_ns": 30000,
                       "d_kernel_crit_ns": kernel_total // 10 + 5000, "d_join_ns": 16000,
                       "d_region_wall_ns": hot_ns - 5000,
                       "d_kernel_ideal_ns": kernel_total // 10,
                       "d_imbalance_ns": 5000,
                       "d_parts_ideal_ns": (20000 + 40000 + 60000 + 80000 + kernel_total + 30000) // 10})
        if mutate:
            mutate(arm, s, kv)
        lines.append("ROW4151 " + " ".join(f"{k}={v}" for k, v in kv.items()))
    lines.append("END4151 arm=" + arm + " forkjoin_jobs=20")
    return "\n".join(lines) + "\n"


def _vals(W=0.238, Wf=0.238, O=0.164, K=0.024):
    q = {"W_off": W, "W_on": W, "Wf_off": Wf, "Wf_on": Wf, "O": O, "K_opp": K, "c_sme_kc0_ns": 60.0,
         "c_sme_intercept_ns": 50.0}
    return "quantity\tvalue\n" + "".join(f"{k}\t{v}\n" for k, v in q.items())


def _build(root, rounds=2, script=None, gates=None, vals=None, ex=None, rule_thr="1.00"):
    script = script or self_sha256()
    with open(os.path.join(root, "RULE.txt"), "w", encoding="utf-8") as f:
        f.write(_rule_text(script, rounds, rule_thr))
    for i in range(1, rounds + 1):
        rd = os.path.join(root, f"round{i}")
        os.makedirs(os.path.join(rd, "raw"))
        with open(os.path.join(rd, "gate.txt"), "w") as f:
            f.write((gates or {}).get(i, "PASS load1=0.5"))
        with open(os.path.join(rd, "values.tsv"), "w") as f:
            f.write(vals or _vals())
        for a in ARMS:
            with open(os.path.join(rd, "raw", f"example-{a}.txt"), "w") as f:
                f.write((ex or (lambda arm: _example(arm)))(a))


def _run(**kw):
    with tempfile.TemporaryDirectory() as d:
        _build(d, **kw)
        return aggregate(d)


def self_test():
    # 駆動部の固定費が大きく、除けば届く → 自分たちの制約（F1_drv>=1）。
    # O=0.164, W=0.238, G=0.2, K_opp=0.024: F1=0.164/(0.238-0.2+0.024)=2.65>=1。kernel_ideal=0.1 → F_drv=0.1。
    # F1_drv=0.164/(0.238-0.1)=1.19>=1 → 自分たちの制約。
    res = _run(vals=_vals(Wf=0.5))
    assert res["cls"] == CLASS_OWN, res["cls"]
    # 駆動部を除いても届かない → ハードの上限（kernel_ideal を 0.195 ms 相当に）。
    res = _run(vals=_vals(Wf=0.5), ex=lambda a: _example(a, kernel_total=1_950_000))
    assert res["cls"] == CLASS_HW, res["cls"]
    # F2 が閾値以上 → 計測窓の定義
    res = _run(vals=_vals(Wf=0.1, O=0.164))
    assert res["cls"] in (CLASS_OWN, CLASS_WINDOW)
    res = _run(vals=_vals(Wf=0.15, O=0.164))
    assert res["cls"] == CLASS_WINDOW, res["cls"]
    # どれも届かない → 保留
    res = _run(vals=_vals(Wf=0.5, O=0.05, K=0.024))
    assert res["cls"] == CLASS_HOLD, res["cls"]
    # TIMEOUT のラウンドは参考（確度低・partial）
    res = _run(gates={2: "TIMEOUT load1=9"}, vals=_vals(Wf=0.5))
    assert (not res["formal"]) and (not res["high"])
    # ドリフト
    assert _run(vals=_vals(Wf=0.5))["drifted"] is False
    assert _run(vals=_vals(Wf=0.5, O=0.3))["drifted"] is True
    # 停止条件
    _expect(lambda: _run(script="0" * 64), "script hash", "mismatch")
    _expect(lambda: _run(gates={1: "INVALID x", 2: "INVALID x"}), "gate invalid", "INVALID")
    _expect(lambda: _run(gates={1: "TIMEOUT", 2: "TIMEOUT"}), "no pass", "no PASS")
    _expect(lambda: _run(vals=_vals().replace("K_opp\t0.024", "K_opp\t0")), "zero value", "> 0")
    _expect(lambda: _run(vals=_vals().replace("K_opp\t0.024", "K_opp\tnan")), "nan", "finite")
    _expect(lambda: _run(vals=_vals() + "W_off\t1\n"), "duplicate quantity", "duplicate")
    _expect(lambda: _run(vals=_vals() + "bogus\t1\n"), "unknown quantity", "unknown")
    _expect(lambda: _run(vals=_vals().replace("O\t0.164\n", "")), "missing quantity", "missing")
    _expect(lambda: _run(ex=lambda a: _example(a, bits="0x" + "2" * 16) if a == "on-base" else _example(a)),
            "bits mismatch", "bits_hash mismatch")

    def drop_series(arm, s, kv):
        pass

    def bad_regions(arm, s, kv):
        if "d_regions" in kv:
            kv["d_regions"] = 499

    _expect(lambda: _run(ex=lambda a: _example(a, mutate=bad_regions)), "regions", "d_regions")

    def low_calls(arm, s, kv):
        if "d_n_calls" in kv:
            kv["d_n_calls"] = 3

    _expect(lambda: _run(ex=lambda a: _example(a, mutate=low_calls)), "n_calls", "lower bound")

    def big_sum(arm, s, kv):
        if "d_parts_ideal_ns" in kv:
            kv["d_parts_ideal_ns"] = 10**9

    _expect(lambda: _run(ex=lambda a: _example(a, mutate=big_sum)), "sum", "interval sum")

    def extra_key(arm, s, kv):
        kv["zzz"] = 1

    _expect(lambda: _run(ex=lambda a: _example(a, mutate=extra_key)), "extra key", "keys mismatch")

    _expect(lambda: _run(ex=lambda a: _example(a).replace("sme_production_enabled=true", "sme_production_enabled=false")),
            "sme not enabled", "sme_production_enabled")
    _expect(lambda: _run(ex=lambda a: _example(a).replace("sme_production_enabled=false", "sme_production_enabled=true")),
            "off arm built with sme", "sme_production_enabled")

    def short_calls(arm, s, kv):
        if s != "cold_first_call" and "d_regions" in kv:
            kv["calls"] = 1
            kv["d_regions"] = 1

    _expect(lambda: _run(ex=lambda a: _example(a, mutate=short_calls)), "short calls", "RULE 項4")
    with tempfile.TemporaryDirectory() as d:
        _build(d)
        os.rename(os.path.join(d, "round2"), os.path.join(d, "round9"))
        _expect(lambda: aggregate(d), "missing round", "missing round")
    # 摂動超過 → 確度低
    res = _run(vals=_vals(Wf=0.5), ex=lambda a: _example(a, hot_ns=300000) if a.endswith("timed") else _example(a))
    assert not res["high"]
    md, tsv = render(_run(vals=_vals(Wf=0.5)))
    assert "判定" in md and tsv.startswith("key\tvalue")
    # 指紋は決定的
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "crates"))
        with open(os.path.join(d, "crates", "x"), "w") as f:
            f.write("x")
        assert fingerprint(d)[0] == fingerprint(d)[0]
    print("self-test OK")


def main(argv):
    try:
        if "--self-test" in argv:
            self_test()
            return 0
        if "--check-template" in argv:
            i = argv.index("--check-template")
            check_template(argv[i + 1] if i + 1 < len(argv) else "RULE.template.txt")
            print("template OK")
            return 0
        if "--fingerprint" in argv:
            i = argv.index("--fingerprint")
            fp, n, listing = fingerprint(argv[i + 1])
            if "--list-out" in argv:
                with open(argv[argv.index("--list-out") + 1], "w", encoding="utf-8") as f:
                    f.write(listing)
            print(fp)
            print(f"files={n}", file=sys.stderr)
            return 0
        opts = {"--dir": None, "--rule": "RULE.txt", "--out": "aggregate.md", "--tsv": "aggregate.tsv"}
        for k in opts:
            if k in argv:
                opts[k] = argv[argv.index(k) + 1]
        if not opts["--dir"]:
            raise AggError("--dir required")
        res = aggregate(opts["--dir"], opts["--rule"])
        md, tsv = render(res)
        with open(os.path.join(opts["--dir"], opts["--out"]), "w", encoding="utf-8") as f:
            f.write(md)
        with open(os.path.join(opts["--dir"], opts["--tsv"]), "w", encoding="utf-8") as f:
            f.write(tsv)
        print(md)
        return 0
    except AggError as e:
        print(f"AGGREGATE ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
