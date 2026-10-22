#!/usr/bin/env python3
"""イシュー #3183: Mac mini の infer Metal 二峰性診断の集計器（python3 標準ライブラリのみ）。

判定規則は同ディレクトリの RULE.txt（実測前に固定）が正。orchestrate.sh が残した
run{1..5}.jsonl（1 反復 1 行）と load_gate_status.txt を読み、モード検出・帰属・
系列 A/B の窓判定を Markdown で出力する。

fail-closed（exit != 0）: run の欠落、n != 500、非有限値・負値、JSON 不正、腕の欠落、
iter の欠番、ゲート状態の欠落・不合格、checksum の反復間／run 間／腕間（R・D・Don）不一致。

第 2 ラウンドの規則は RULE2.txt（実測前に固定）。`--rule 2 <dir>` が第 2 ラウンド、
`--rule 2 --r1 <dir>` が第 1 ラウンドへの事後適用（探索・判定に使わない）。

使い方: python3 -I aggregate.py <dir> / python3 -I aggregate.py --rule 2 [--r1] <dir> / python3 -I aggregate.py --self-test
"""
import json
import math
import os
import random
import statistics
import sys
import tempfile

RUNS = (1, 2, 3, 4, 5)
ARMS = ("R", "D", "Don", "G")
N = 500
GAP = 0.10          # 隣接値の相対ギャップ閾値（RULE.txt）
SIDE = 0.05         # 分割の両側に残す最小割合
SWITCH_ALT = 10     # プロセス内の交互出現とみなす切り替え回数
STICKY = 0.95       # プロセス単位の固着とみなす割合
ATTR_RATIO = 0.5    # 帰属の比の閾値
WINDOW = 20
BENCH_B = 0.388e-3  # 系列 B（bench 観測 5 run 中央値）
BENCH_A = 0.539e-3  # 系列 A
BENCH_LOW = 0.385e-3
SPANS = {
    "R": ("predict_resident", "host_copy", "checksum", "iter_total"),
    "D": ("provider_select", "tape_new", "snapshot", "chain", "host_copy", "checksum", "iter_total"),
    "Don": ("provider_select", "tape_new", "snapshot", "chain", "host_copy", "checksum", "iter_total"),
    "G": ("upload", "encode", "commit_wait", "download", "wall", "iter_total", "gpu_busy"),
}


class Invalid(Exception):
    """系列を無効にする入力不備（fail-closed）。"""


def finite_nonneg(v, what):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
        raise Invalid(f"{what}: 非有限または負の値 {v!r}")
    return float(v)


def load_run(path, n, arms=ARMS):
    rows = {a: [] for a in arms}
    try:
        with open(path, encoding="utf-8") as f:
            lines = [ln for ln in f.read().splitlines() if ln.strip()]
    except OSError as e:
        raise Invalid(f"run{n}: 読み込み失敗 {e}")
    for ln in lines:
        try:
            r = json.loads(ln)
        except ValueError as e:
            raise Invalid(f"run{n}: JSON 不正 {e}")
        arm = r.get("arm") if isinstance(r, dict) else None
        if arm not in rows:
            raise Invalid(f"run{n}: 未知の arm {arm!r}")
        rows[arm].append(r)
    out = {}
    for arm in arms:
        rs = rows[arm]
        if len(rs) != N:
            raise Invalid(f"run{n} arm {arm}: n={len(rs)} (期待 {N})")
        sums = set()
        series = []
        for i, r in enumerate(rs):
            if r.get("iter") != i:
                raise Invalid(f"run{n} arm {arm}: iter 欠番/順序不正 (位置 {i}, iter={r.get('iter')!r})")
            t = finite_nonneg(r.get("t_since_start_s"), f"run{n} {arm}[{i}] t_since_start_s")
            sp = r.get("spans")
            if not isinstance(sp, dict):
                raise Invalid(f"run{n} arm {arm}[{i}]: spans 欠落")
            spans = {}
            for k in SPANS[arm]:
                if k not in sp:
                    raise Invalid(f"run{n} arm {arm}[{i}]: span {k} 欠落")
                if k == "gpu_busy" and sp[k] is None:
                    spans[k] = None
                    continue
                spans[k] = finite_nonneg(sp[k], f"run{n} {arm}[{i}] {k}")
            if spans["iter_total"] <= 0:
                raise Invalid(f"run{n} arm {arm}[{i}]: iter_total <= 0")
            cs = r.get("checksum_bits")
            if not isinstance(cs, str) or not cs:
                raise Invalid(f"run{n} arm {arm}[{i}]: checksum_bits 欠落")
            sums.add(cs)
            ct = r.get("counters")
            if not isinstance(ct, dict):
                raise Invalid(f"run{n} arm {arm}[{i}]: counters 欠落")
            counters = {k: finite_nonneg(v, f"run{n} {arm}[{i}] counter {k}") for k, v in ct.items()}
            if arm == "G":
                spans["commit_wait-gpu_busy"] = (
                    None if spans["gpu_busy"] is None else spans["commit_wait"] - spans["gpu_busy"]
                )
            series.append({"t": t, "spans": spans, "counters": counters})
        if len(sums) != 1:
            raise Invalid(f"run{n} arm {arm}: 反復間で checksum 不一致 {sorted(sums)}")
        out[arm] = {"series": series, "checksum": next(iter(sums))}
    return out


def read_gate(d):
    path = os.path.join(d, "load_gate_status.txt")
    try:
        with open(path, encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
    except OSError as e:
        raise Invalid(f"ゲート状態の読み込み失敗: {e}")
    status = {}
    for ln in lines:
        parts = ln.split()
        if len(parts) < 2 or not parts[0].startswith("run") or not parts[1].startswith("gate="):
            raise Invalid(f"ゲート状態の書式不正: {ln!r}")
        status[parts[0]] = parts[1][len("gate="):]
    for n in RUNS:
        s = status.get(f"run{n}")
        if s is None:
            raise Invalid(f"ゲート状態の欠落: run{n}")
        if s != "pass":
            raise Invalid(f"run{n} のゲートが pass でない: {s}")
    return status


def median(v):
    return statistics.median(v)


def percentile(sorted_v, p):
    k = (len(sorted_v) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    return sorted_v[lo] + (sorted_v[hi] - sorted_v[lo]) * (k - lo)


def detect_modes(values):
    """RULE.txt のモード検出。戻り値: (labels, clusters)。label 0 が低モード、最大が高モード。"""
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    xs = [values[i] for i in order]
    cuts = []
    for p in range(n - 1):
        left, right = p + 1, n - p - 1
        if left < SIDE * n or right < SIDE * n:
            continue
        if xs[p] > 0 and xs[p + 1] / xs[p] - 1.0 >= GAP:
            cuts.append(p)
    labels = [0] * n
    clusters = []
    start = 0
    for c, p in enumerate(cuts + [n - 1]):
        members = order[start:p + 1]
        for i in members:
            labels[i] = c
        vals = [values[i] for i in members]
        clusters.append({"count": len(members), "share": len(members) / n, "median": median(vals)})
        start = p + 1
    return labels, clusters


def time_structure(labels):
    switches = sum(1 for a, b in zip(labels, labels[1:]) if a != b)
    runs, cur = [], 1
    for a, b in zip(labels, labels[1:]):
        if a == b:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    return switches, median(runs)


def analyze_arm(arm, per_run):
    """腕 1 本分（5 run）の解析結果。"""
    res = []
    for n in RUNS:
        series = per_run[n][arm]["series"]
        totals = [s["spans"]["iter_total"] for s in series]
        labels, clusters = detect_modes(totals)
        q = sorted(totals)
        entry = {
            "run": n,
            "clusters": clusters,
            "multimodal": len(clusters) >= 2,
            "q3_q1": percentile(q, 0.75) / percentile(q, 0.25),
            "median": median(totals),
            "labels": labels,
            "totals": totals,
        }
        entry["switches"], entry["run_len_median"] = time_structure(labels)
        if entry["multimodal"]:
            hi, lo = len(clusters) - 1, 0
            entry["high_share"] = clusters[hi]["share"]
            entry["low_share"] = clusters[lo]["share"]
            entry["attr"], entry["excluded"], entry["counter_diff"] = attribute(series, labels, hi, lo)
        res.append(entry)
    return res


def attribute(series, labels, hi, lo):
    """高モードと低モードで区間ごとの中央値差を iter_total の差で割った比。"""
    hs = [s for s, lb in zip(series, labels) if lb == hi]
    ls = [s for s, lb in zip(series, labels) if lb == lo]
    attr, excluded = {}, 0
    for k in series[0]["spans"]:
        if k == "iter_total":
            continue
        # GPU 計測不能（None）の反復は区間側だけでなく分母 Δ_total の標本からも外す（RULE.txt:
        # 「帰属判定から外す」）。区間ごとに有効反復へ対応する iter_total で高低差を取る。
        hv = [s for s in hs if s["spans"][k] is not None]
        lv = [s for s in ls if s["spans"][k] is not None]
        if k in ("gpu_busy", "commit_wait-gpu_busy"):
            excluded = max(excluded, len(hs) + len(ls) - len(hv) - len(lv))
        if not hv or not lv:
            attr[k] = None
            continue
        d_total = median([s["spans"]["iter_total"] for s in hv]) - median([s["spans"]["iter_total"] for s in lv])
        if d_total <= 0:
            attr[k] = None
            continue
        attr[k] = (median([s["spans"][k] for s in hv]) - median([s["spans"][k] for s in lv])) / d_total
    cdiff = {}
    for k in series[0]["counters"]:
        hm = statistics.mean(s["counters"][k] for s in hs)
        lm = statistics.mean(s["counters"][k] for s in ls)
        if abs(hm - lm) > 1e-9:
            cdiff[k] = (lm, hm)
    return attr, excluded, cdiff


def attributed_spans(entries):
    multi = [e for e in entries if e["multimodal"]]
    if not multi:
        return [], 0
    spans = set()
    for e in multi:
        spans.update(e["attr"].keys())
    # 腕 G の wall は iter_total の複製（比が常に 1.0）で、RULE.txt も帰属対象区間に含めない。
    # 表の分布表示には残すが、帰属候補からは除外する（AC2 の「特定できなかった」を潰さないため）。
    spans.discard("wall")
    out = []
    for k in sorted(spans):
        cnt = sum(1 for e in multi if e["attr"].get(k) is not None and e["attr"][k] >= ATTR_RATIO)
        if cnt * 2 > len(multi):
            out.append((k, cnt))
    return out, len(multi)


def sticky_state(entries):
    """プロセス単位の固着判定。戻り値: (固着 True/False/None=判定不能, 説明)。

    各反復を共通の基準（参照低モード・参照高モードの中点）で分類して低/高の割合を求める。
    単峰 run でも広い分布なら両側に割合が分かれ、95% 条件を満たさない。参照モードは
    多峰 run があればそのクラスタ中央値、無ければ run 中央値の最小・最大（相対差 10% 以上）。
    対応する 2 モードが定義できない場合は判定不能（None）。
    """
    multi = [e for e in entries if e["multimodal"]]
    if multi:
        ref_lo = median([e["clusters"][0]["median"] for e in multi])
        ref_hi = median([e["clusters"][-1]["median"] for e in multi])
    else:
        meds = [e["median"] for e in entries]
        ref_lo, ref_hi = min(meds), max(meds)
    if ref_lo <= 0 or ref_hi / ref_lo - 1.0 < GAP:
        return None, ["モードが定義できない（参照モード間差 10% 未満）"]
    mid = (ref_lo + ref_hi) / 2
    desc, lows, highs = [], 0, 0
    for e in entries:
        vals = e["totals"]
        lo_share = sum(1 for v in vals if v < mid) / len(vals)
        hi_share = 1.0 - lo_share
        if lo_share >= STICKY:
            lows += 1
        if hi_share >= STICKY:
            highs += 1
        desc.append(f"run{e['run']}: 低 {lo_share:.1%} / 高 {hi_share:.1%}")
    return (lows > 0 and highs > 0), desc


def window_judgement(per_run, r_entries):
    meds_by_run = []
    for n in RUNS:
        t = [s["spans"]["iter_total"] for s in per_run[n]["R"]["series"]]
        meds_by_run.append([median(t[i:i + WINDOW]) for i in range(len(t) - WINDOW + 1)])
    pooled = sorted(m for ms_ in meds_by_run for m in ms_)
    p5, p95 = percentile(pooled, 0.05), percentile(pooled, 0.95)
    multi = [e for e in r_entries if e["multimodal"]]
    cond_a = bool(multi)
    cond_b = p5 <= BENCH_B and BENCH_A <= p95
    cond_c = False
    low_med = median([e["median"] for e in r_entries])
    frac_lo = frac_hi = None
    if multi:
        low_med = median([e["clusters"][0]["median"] for e in multi])
        high_med = median([e["clusters"][-1]["median"] for e in multi])
        mid = (low_med + high_med) / 2
        lo_n = sum(1 for m in pooled if m < mid)
        frac_lo, frac_hi = lo_n / len(pooled), 1 - lo_n / len(pooled)
        cond_c = frac_lo >= 0.10 and frac_hi >= 0.10
    rng = random.Random(3183)
    boot = sorted(median([rng.choice(ms_) for ms_ in meds_by_run]) for _ in range(10000))
    ok = sum([cond_a, cond_b, cond_c])
    verdict = "説明できる" if ok == 3 else ("部分的" if ok > 0 else "説明できない")
    return {
        "p5": p5, "p95": p95, "cond": (cond_a, cond_b, cond_c), "verdict": verdict,
        "frac": (frac_lo, frac_hi), "low_med": low_med,
        # RULE.txt は「低モード中央値で割った比」と定める。分母は実測の low_med（BENCH_LOW は事前の想定値で
        # 併記のみ。#3183 第 2 ラウンドで分母の取り違えを訂正。比は記録のみで判定に使わない）。
        "ratio_b": BENCH_B / low_med, "ratio_a": BENCH_A / low_med,
        "ratio_b_const": BENCH_B / BENCH_LOW, "ratio_a_const": BENCH_A / BENCH_LOW,
        "boot": (percentile(boot, 0.05), percentile(boot, 0.95)),
    }


def ms(x):
    return f"{x * 1e3:.3f} ms"


def render(per_run, gate):
    out = ["# 集計結果（イシュー #3183。RULE.txt の規則どおり）", ""]
    out.append("ゲート状態: " + ", ".join(f"{k}={v}" for k, v in sorted(gate.items())))
    out.append("")
    all_entries = {}
    for arm in ARMS:
        entries = analyze_arm(arm, per_run)
        all_entries[arm] = entries
        out += [f"## 腕 {arm}（{'wall' if arm == 'G' else 'iter_total'}）", ""]
        out += ["| run | 中央値 | Q3/Q1 | クラスタ数 | クラスタ（割合 / 中央値） | 切替回数 | モード連続長の中央値 |",
                "|---|---|---|---|---|---|---|"]
        for e in entries:
            cl = " ; ".join(f"{c['share']:.1%} / {ms(c['median'])}" for c in e["clusters"])
            out.append(f"| {e['run']} | {ms(e['median'])} | {e['q3_q1']:.2f} | {len(e['clusters'])} | {cl} | "
                       f"{e['switches']} | {e['run_len_median']:.1f} |")
        multi = [e for e in entries if e["multimodal"]]
        out += ["", f"- 多峰ありの run: {len(multi)} / 5"]
        alt = [e["run"] for e in entries if e["switches"] >= SWITCH_ALT and e["multimodal"]]
        out.append(f"- プロセス内の交互出現（切替 {SWITCH_ALT} 回以上）: {alt if alt else 'なし'}")
        sticky, desc = sticky_state(entries)
        out.append(f"- プロセス単位の固着: {'判定不能' if sticky is None else ('あり' if sticky else 'なし')}（" + "; ".join(desc) + "）")
        spans, nm = attributed_spans(entries)
        if nm == 0:
            out.append("- 帰属: 多峰の run がないため判定対象なし")
        elif spans:
            out.append(f"- 帰属（比 {ATTR_RATIO} 以上が多峰 run の過半数）: " +
                       ", ".join(f"{k}（{c}/{nm} run）" for k, c in spans))
        else:
            out.append("- 帰属: **特定できなかった**（該当する区間なし）")
        if nm:
            ex = max(e["excluded"] for e in multi)
            out.append(f"- gpu_busy 判定不能で除外した反復の最大数（run 内）: {ex}")
            out += ["", "| 区間 | " + " | ".join(f"run{e['run']}" for e in multi) + " |",
                    "|---|" + "---|" * len(multi)]
            for k in multi[0]["attr"]:
                cells = ["n/a" if e["attr"].get(k) is None else f"{e['attr'][k]:.2f}" for e in multi]
                out.append(f"| {k} | " + " | ".join(cells) + " |")
            cd = {}
            for e in multi:
                for k, (lo, hi) in e["counter_diff"].items():
                    cd.setdefault(k, []).append(f"run{e['run']}: 低 {lo:.3f} / 高 {hi:.3f}")
            out.append("")
            out.append("- カウンタの高低差: " +
                       ("; ".join(f"{k} ({', '.join(v)})" for k, v in cd.items()) if cd else "なし"))
        out.append("")
    w = window_judgement(per_run, all_entries["R"])
    out += ["## 系列 A/B の差の説明（腕 R・連続 20 反復の窓）", "",
            f"- 窓中央値の 5〜95%: {ms(w['p5'])} 〜 {ms(w['p95'])}（bench 観測 B={ms(BENCH_B)} / A={ms(BENCH_A)}）",
            f"- 低モード中央値: {ms(w['low_med'])}。bench 比（B/低・A/低、分母は実測の低モード中央値）: "
            f"{w['ratio_b']:.2f} / {w['ratio_a']:.2f}"
            f"（RULE 記載の定数 {ms(BENCH_LOW)} で割った値: {w['ratio_b_const']:.2f} / {w['ratio_a_const']:.2f}）"]
    if w["frac"][0] is not None:
        out.append(f"- 窓中央値が低側／高側に入る割合: {w['frac'][0]:.1%} / {w['frac'][1]:.1%}")
    ca, cb, cc = w["cond"]
    out += [f"- 条件 (a) R で多峰あり: {ca} / (b) 5〜95% が A・B を両方含む: {cb} / (c) 低側・高側が各 10% 以上: {cc}",
            f"- **判定: {w['verdict']}**",
            f"- 参考（記録のみ）: 5 run 中央値の中央値のブートストラップ 5〜95%: "
            f"{ms(w['boot'][0])} 〜 {ms(w['boot'][1])}",
            ""]
    return "\n".join(out)


def aggregate(d):
    gate = read_gate(d)
    per_run = {n: load_run(os.path.join(d, f"run{n}.jsonl"), n) for n in RUNS}
    ref = {}
    for arm in ("R", "D", "Don"):
        sums = {per_run[n][arm]["checksum"] for n in RUNS}
        if len(sums) != 1:
            raise Invalid(f"腕 {arm}: run 間で checksum 不一致 {sorted(sums)}")
        ref[arm] = next(iter(sums))
    if len(set(ref.values())) != 1:
        raise Invalid(f"R・D・Don の checksum が腕間で不一致 {ref}")
    gs = {per_run[n]["G"]["checksum"] for n in RUNS}
    if len(gs) != 1:
        raise Invalid(f"腕 G: run 間で checksum 不一致 {sorted(gs)}")
    return render(per_run, gate)


# --------------------------------------------------------------------------- RULE2（第 2 ラウンド）
# 判定規則は同ディレクトリの RULE2.txt（実測前に固定）が正。RULE.txt の集計（上記）とは別系統で、
# 同じ入力の読み込み・fail-closed の流儀を共有する（`--rule 2 <dir>`、事後適用は `--rule 2 --r1 <dir>`）。

TAIL_MULT = 1.25     # 裾 = 腕×run の中央値のこの倍率を超える反復
TAIL_MIN = 10        # 対象 run: 裾が 10 反復（2%）以上
GPU_FLAT = 0.05      # 「GPU 実行時間は不変」: |Δ gpu_busy| / 本体中央値 < 5%
LOWPERF = 1.5        # 低性能 run: 指標が 5 run 中央値の 1.5 倍以上
STARTUP = 1.10       # 立ち上がり上振れ: W / S >= 1.10
STARTUP_WIN = 20     # bench の計測窓（計測 iter 0〜19）
VERSION_RATIO = 1.10
BAND = 0.10          # 水準の再現: ±10%
OVER_MS = 0.4e-3     # 記録のみ: 0.4 ms を超える反復の割合
R2_CONDS_HEAD = ("R", "D", "Don", "G")
R2_CONDS_V0 = ("R", "D", "Don")
R2_G_SPANS = ("upload", "encode", "commit_wait", "gpu_busy", "commit_wait-gpu_busy", "download")


def span_names(arm):
    """裾の帰属で対象にする区間（iter_total と、G で iter_total の複製である wall は除く）。"""
    return R2_G_SPANS if arm == "G" else tuple(k for k in SPANS[arm] if k != "iter_total")


def read_meta(d):
    """run_meta.txt（記録のみ・書式は fail-closed）。書式: run=N order=<条件列> since_build_s=<int> load1=<f> load5=<f>。"""
    path = os.path.join(d, "run_meta.txt")
    try:
        with open(path, encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
    except OSError as e:
        raise Invalid(f"run_meta.txt の読み込み失敗: {e}")
    meta = {}
    for ln in lines:
        kv = {}
        for tok in ln.split():
            if "=" not in tok:
                raise Invalid(f"run_meta.txt の書式不正: {ln!r}")
            k, v = tok.split("=", 1)
            kv[k] = v
        try:
            n = int(kv["run"])
            meta[n] = {
                "order": kv["order"], "since_build_s": int(kv["since_build_s"]),
                "load1": finite_nonneg(float(kv["load1"]), f"run{n} load1"),
                "load5": finite_nonneg(float(kv["load5"]), f"run{n} load5"),
            }
        except (KeyError, ValueError) as e:
            raise Invalid(f"run_meta.txt の値不正: {ln!r} ({e})")
    for n in RUNS:
        if n not in meta:
            raise Invalid(f"run_meta.txt の欠落: run{n}")
    return meta


def read_bench_line(path, what):
    """bench-fandhe の infer metal 64 reuse の 1 行（1 プロセス 1 行）。fail-closed。"""
    try:
        with open(path, encoding="utf-8") as f:
            lines = [ln for ln in f.read().splitlines() if ln.strip()]
    except OSError as e:
        raise Invalid(f"{what}: 読み込み失敗 {e}")
    if len(lines) != 1:
        raise Invalid(f"{what}: 行数 {len(lines)} (期待 1)")
    try:
        r = json.loads(lines[0])
    except ValueError as e:
        raise Invalid(f"{what}: JSON 不正 {e}")
    if not isinstance(r, dict) or r.get("task") != "infer" or r.get("device") != "metal" \
            or r.get("mode") != "reuse" or r.get("size") != 64:
        raise Invalid(f"{what}: infer metal 64 reuse の行ではない")
    med = finite_nonneg(r.get("median_s"), f"{what} median_s")
    if med <= 0:
        raise Invalid(f"{what}: median_s <= 0")
    cs = r.get("checksum")
    if isinstance(cs, bool) or not isinstance(cs, (int, float)) or not math.isfinite(cs):
        raise Invalid(f"{what}: checksum 欠落/非有限")
    return {"median": med, "checksum": float(cs)}


def r2_load(d, posthoc_r1=False):
    """第 2 ラウンドの入力一式（posthoc_r1 は第 1 ラウンドのディレクトリへの事後適用）。"""
    gate = read_gate(d)
    head_dir = d if posthoc_r1 else os.path.join(d, "head")
    head = {n: load_run(os.path.join(head_dir, f"run{n}.jsonl"), n, R2_CONDS_HEAD) for n in RUNS}
    v0, v0_reason = None, None
    if posthoc_r1:
        v0_reason = "第 1 ラウンドに存在しない条件"
    else:
        nr = os.path.join(d, "v0110", "NOT_RUN.txt")
        if os.path.exists(nr):
            with open(nr, encoding="utf-8") as f:
                v0_reason = f.read().strip() or "理由の記載なし"
        else:
            v0 = {n: load_run(os.path.join(d, "v0110", f"run{n}.jsonl"), n, R2_CONDS_V0) for n in RUNS}
    bc, bct = {}, None
    bdir = os.path.join(d, "bench_control")
    for n in RUNS:
        name = f"run{n}.jsonl" if posthoc_r1 else f"run{n}_Bc.jsonl"
        bc[n] = read_bench_line(os.path.join(bdir, name), f"Bc run{n}")
    if not posthoc_r1:
        bct = {n: read_bench_line(os.path.join(bdir, f"run{n}_BcT.jsonl"), f"BcT run{n}") for n in RUNS}
    meta = None if posthoc_r1 else read_meta(d)
    # checksum 規則（fail-closed）: HEAD の R・D・Don は全 run で一致、G は腕内で一致。
    ref = {head[n][a]["checksum"] for n in RUNS for a in ("R", "D", "Don")}
    if len(ref) != 1:
        raise Invalid(f"HEAD の R・D・Don の checksum が不一致 {sorted(ref)}")
    if len({head[n]["G"]["checksum"] for n in RUNS}) != 1:
        raise Invalid("HEAD の G の checksum が run 間で不一致")
    v0_valid, v0_note = False, None
    if v0 is not None:
        sums = {v0[n][a]["checksum"] for n in RUNS for a in R2_CONDS_V0}
        if len(sums) != 1:
            raise Invalid(f"v0110 の R・D・Don の checksum が腕間・run 間で不一致 {sorted(sums)}")
        v0_valid = sums == ref
        v0_note = "HEAD と一致" if v0_valid else "HEAD と不一致（出力差は判定不能として性能値の比較から外す）"
    bsums = {x["checksum"] for x in bc.values()} | ({x["checksum"] for x in bct.values()} if bct else set())
    if len(bsums) != 1:
        raise Invalid(f"bench 対照の checksum が不一致 {sorted(bsums)}")
    return {"gate": gate, "head": head, "v0": v0, "v0_reason": v0_reason, "v0_valid": v0_valid,
            "v0_note": v0_note, "bc": bc, "bct": bct, "meta": meta, "posthoc": posthoc_r1}


def tail_analysis(series, arm):
    """腕×run の裾の帰属。対象外（裾 10 反復未満）なら eligible=False。"""
    totals = [s["spans"]["iter_total"] for s in series]
    med = median(totals)
    tail = [s for s, t in zip(series, totals) if t > TAIL_MULT * med]
    body = [s for s, t in zip(series, totals) if t <= TAIL_MULT * med]
    res = {"median": med, "tail_n": len(tail), "eligible": len(tail) >= TAIL_MIN and len(body) > 0,
           "over_ms": sum(1 for t in totals if t > OVER_MS) / len(totals), "ratios": {}, "delta": {},
           "excluded": 0, "gpu_flat": None}
    if not res["eligible"]:
        return res
    for k in span_names(arm):
        tv = [s for s in tail if s["spans"][k] is not None]
        bv = [s for s in body if s["spans"][k] is not None]
        if k in ("gpu_busy", "commit_wait-gpu_busy"):
            res["excluded"] = max(res["excluded"], len(tail) + len(body) - len(tv) - len(bv))
        if not tv or not bv:
            res["ratios"][k] = None
            continue
        # gpu_busy が null の反復は区間側だけでなく分母 Δ_total の標本からも外す（RULE.txt と同じ扱い）
        d_tot = median([s["spans"]["iter_total"] for s in tv]) - median([s["spans"]["iter_total"] for s in bv])
        d_p = median([s["spans"][k] for s in tv]) - median([s["spans"][k] for s in bv])
        res["delta"][k] = d_p
        res["ratios"][k] = d_p / d_tot if d_tot > 0 else None
    if arm == "G":
        bv = [s["spans"]["gpu_busy"] for s in body if s["spans"]["gpu_busy"] is not None]
        if bv and "gpu_busy" in res["delta"] and median(bv) > 0:
            res["gpu_flat"] = abs(res["delta"]["gpu_busy"]) / median(bv) < GPU_FLAT
    return res


def tail_attributed(per_run_res, arm):
    """対象 run の過半数で比 >= 0.5 になった区間と、G の「GPU 実行時間は不変」判定。"""
    elig = [r for r in per_run_res if r["eligible"]]
    out = []
    for k in span_names(arm):
        cnt = sum(1 for r in elig if r["ratios"].get(k) is not None and r["ratios"][k] >= ATTR_RATIO)
        if elig and cnt * 2 > len(elig):
            out.append((k, cnt))
    flat = None
    if arm == "G" and elig:
        yes = sum(1 for r in elig if r["gpu_flat"])
        flat = (yes * 2 > len(elig), yes, len(elig))
    return out, len(elig), flat


def startup_window(series):
    t = [s["spans"]["iter_total"] for s in series]
    return median(t[:STARTUP_WIN]), median(t[STARTUP_WIN:])


def run_median(series, k):
    v = [s["spans"][k] for s in series if s["spans"][k] is not None]
    return median(v) if v else None


def r2_judge(data):
    """AC3 の機械判定と低性能 run の抽出。"""
    head, v0, bc, bct = data["head"], data["v0"], data["bc"], data["bct"]
    use_v0 = v0 is not None and data["v0_valid"]
    gpu = {n: run_median(head[n]["G"]["series"], "gpu_busy") for n in RUNS}
    cpu = {n: run_median(head[n]["R"]["series"], "checksum") for n in RUNS}
    valid_gpu = [x for x in gpu.values() if x is not None]
    if not valid_gpu:
        raise Invalid("G の gpu_busy が全 run で判定不能")
    gm, cm = median(valid_gpu), median(list(cpu.values()))
    low_gpu = [n for n in RUNS if gpu[n] is not None and gpu[n] >= LOWPERF * gm]
    low_cpu = [n for n in RUNS if cpu[n] >= LOWPERF * cm]
    sys_low = [n for n in RUNS if n in low_gpu and n in low_cpu]
    v0_cpu, v0_low = {}, None
    if v0 is not None:
        v0_cpu = {n: run_median(v0[n]["R"]["series"], "checksum") for n in RUNS}
        v0m = median(list(v0_cpu.values()))
        v0_low = [n for n in RUNS if v0_cpu[n] >= LOWPERF * v0m]
    h_w = median([startup_window(head[n]["R"]["series"])[0] for n in RUNS])
    h_s = median([startup_window(head[n]["R"]["series"])[1] for n in RUNS])
    v_w = v_s = None
    if use_v0:
        v_w = median([startup_window(v0[n]["R"]["series"])[0] for n in RUNS])
        v_s = median([startup_window(v0[n]["R"]["series"])[1] for n in RUNS])
    bc_m = median([bc[n]["median"] for n in RUNS])
    bct_m = median([bct[n]["median"] for n in RUNS]) if bct else None
    ver = None if v_s is None else (v_s / h_s >= VERSION_RATIO or v_s / h_s <= 1 / VERSION_RATIO)
    if v_w is not None:
        ref_w, ref_name = v_w, "v0110-R の W"
    else:
        ref_w, ref_name = h_w, "HEAD-R の W（v0110 が未実施または無効のため代用）"
    hsp = bc_m / ref_w >= VERSION_RATIO
    ordr = None if bct_m is None else bct_m / bc_m >= VERSION_RATIO
    cands = [("Bc", bc_m)]
    if bct_m is not None:
        cands.append(("BcT", bct_m))
    cands.append(("HEAD-R の W", h_w))
    if v_w is not None:
        cands.append(("v0110-R の W", v_w))

    def in_band(x, c):
        return c * (1 - BAND) <= x <= c * (1 + BAND)

    b_hit = [name for name, x in cands if in_band(x, BENCH_B)]
    a_hit = [name for name, x in cands if in_band(x, BENCH_A)]
    for n in sys_low:
        r_med = median([s["spans"]["iter_total"] for s in head[n]["R"]["series"]])
        if in_band(r_med, BENCH_A):
            a_hit.append(f"低性能 run{n} の R 中央値")
    reproduced = int(bool(b_hit)) + int(bool(a_hit))
    verdict = {2: "説明できる", 1: "部分的", 0: "説明できない"}[reproduced]
    return {"gpu": gpu, "cpu": cpu, "gm": gm, "cm": cm, "low_gpu": low_gpu, "low_cpu": low_cpu, "sys_low": sys_low,
            "v0_cpu": v0_cpu, "v0_low": v0_low, "h_w": h_w, "h_s": h_s, "v_w": v_w, "v_s": v_s, "bc_m": bc_m,
            "bct_m": bct_m, "ver": ver, "hsp": hsp, "ord": ordr, "ref_w": ref_w, "ref_name": ref_name,
            "b_hit": b_hit, "a_hit": a_hit, "verdict": verdict, "use_v0": use_v0}


def yn(x):
    return "判定不能（条件未実施）" if x is None else ("あり" if x else "なし")


def r2_render(data):
    head, v0 = data["head"], data["v0"]
    title = ("# 第 1 ラウンドへの RULE2 の事後適用（事後・探索。判定に使わない。イシュー #3183）" if data["posthoc"]
             else "# 集計結果（イシュー #3183 第 2 ラウンド。RULE2.txt の規則どおり）")
    out = [title, ""]
    out.append("ゲート状態: " + ", ".join(f"{k}={v}" for k, v in sorted(data["gate"].items())))
    out.append("")
    if data["meta"]:
        out += ["## run の記録（記録のみ）", "", "| run | 条件の起動順 | ビルド終了からの経過 (s) | load1 | load5 |",
                "|---|---|---|---|---|"]
        for n in RUNS:
            m = data["meta"][n]
            out.append(f"| {n} | {m['order']} | {m['since_build_s']} | {m['load1']:.2f} | {m['load5']:.2f} |")
        out.append("")
    conds = [("HEAD", a, head) for a in R2_CONDS_HEAD]
    if v0 is not None:
        conds += [("v0110", a, v0) for a in R2_CONDS_V0]
    out += ["## 裾の帰属（裾 = 腕×run の中央値の 1.25 倍超。対象 run = 裾 10 反復以上）", ""]
    if v0 is None:
        out += [f"- v0110 条件は未実施: {data['v0_reason']}", ""]
    else:
        out += [f"- v0110 の checksum: {data['v0_note']}", ""]
    for tag, arm, src in conds:
        res = [tail_analysis(src[n][arm]["series"], arm) for n in RUNS]
        out += [f"### {tag}-{arm}", "",
                "| run | 中央値 | 裾の反復数 | 対象 | 0.4 ms 超の割合（記録のみ） | 5% 規則の多峰（記録のみ） |",
                "|---|---|---|---|---|---|"]
        for n, r in zip(RUNS, res):
            totals = [s["spans"]["iter_total"] for s in src[n][arm]["series"]]
            nclu = len(detect_modes(totals)[1])
            out.append(f"| {n} | {ms(r['median'])} | {r['tail_n']} | {'○' if r['eligible'] else '-'} | "
                       f"{r['over_ms']:.1%} | {'あり' if nclu >= 2 else 'なし'} |")
        attr, ne, flat = tail_attributed(res, arm)
        out.append("")
        if ne == 0:
            out.append("- 帰属: 対象 run なし（裾が 10 反復以上ある run がない）")
        else:
            out.append(f"- 帰属（比 {ATTR_RATIO} 以上が対象 run {ne} の過半数）: " +
                       (", ".join(f"{k}（{c}/{ne} run）" for k, c in attr) if attr else "**特定できなかった**"))
            if flat is not None:
                out.append("- G の GPU 実行時間は不変（|Δ gpu_busy| / 本体中央値 < 5% が対象 run の過半数）: "
                           f"{'成立' if flat[0] else '不成立'}（{flat[1]}/{flat[2]} run）")
            if arm == "G":
                ex = max(r["excluded"] for r in res if r["eligible"])
                out.append(f"- gpu_busy 判定不能で除外した反復の最大数（run 内）: {ex}")
            elig = [(n, r) for n, r in zip(RUNS, res) if r["eligible"]]
            out += ["", "| 区間 | " + " | ".join(f"run{n}（比 / Δ µs）" for n, _ in elig) + " |",
                    "|---|" + "---|" * len(elig)]
            for k in span_names(arm):
                cells = []
                for _, r in elig:
                    ra, de = r["ratios"].get(k), r["delta"].get(k)
                    cells.append("n/a" if ra is None else f"{ra:.2f} / {de * 1e6:+.1f}")
                out.append(f"| {k} | " + " | ".join(cells) + " |")
        out.append("")
    j = r2_judge(data)
    out += ["## run 全体の低性能状態（指標が 5 run 中央値の 1.5 倍以上の run）", "",
            "| run | G gpu_busy 中央値 (µs) | HEAD-R checksum 中央値 (µs) | v0110-R checksum 中央値 (µs) |",
            "|---|---|---|---|"]
    for n in RUNS:
        g = "n/a" if j["gpu"][n] is None else f"{j['gpu'][n] * 1e6:.2f}"
        v = "未実施" if not j["v0_cpu"] else f"{j['v0_cpu'][n] * 1e6:.2f}"
        out.append(f"| {n} | {g} | {j['cpu'][n] * 1e6:.2f} | {v} |")
    out += ["", f"- GPU 低性能の run: {j['low_gpu'] or 'なし'} / CPU 低性能の run: {j['low_cpu'] or 'なし'}",
            f"- **システム全体の低性能状態（GPU・CPU 両方）の run: {j['sys_low'] or 'なし'}**",
            f"- v0110-R の CPU 低性能 run（記録のみ）: {'未実施' if j['v0_low'] is None else (j['v0_low'] or 'なし')}", ""]
    out += ["## 立ち上がり窓（bench と同じ計測 iter 0〜19 の中央値 W と iter 20〜499 の中央値 S）", "",
            "| 条件 | " + " | ".join(f"run{n} W/S (ms)" for n in RUNS) + " | 上振れ（W/S >= 1.10）の件数 |",
            "|---|" + "---|" * (len(RUNS) + 1)]
    tot_up = 0
    for tag, arm, src in conds:
        ws = [startup_window(src[n][arm]["series"]) for n in RUNS]
        up = sum(1 for w, s_ in ws if w / s_ >= STARTUP)
        tot_up += up
        out.append(f"| {tag}-{arm} | " + " | ".join(f"{w * 1e3:.3f}/{s_ * 1e3:.3f}={w / s_:.2f}" for w, s_ in ws)
                   + f" | {up} |")
    out += ["", f"- 立ち上がり上振れの腕×run: {tot_up} / {len(conds) * len(RUNS)}", ""]
    if v0 is None:
        v0_line = "- v0110-R: 未実施（" + str(data["v0_reason"]) + "）"
    elif j["use_v0"]:
        v0_line = f"- v0110-R: W = {ms(j['v_w'])} / S = {ms(j['v_s'])}"
    else:
        v0_line = f"- v0110-R: checksum 不一致のため性能値の比較から除外（{data['v0_note']}）"
    out += ["## 系列 A／B の説明（run 単位の値の 5 run 中央値）", "",
            f"- HEAD-R: W = {ms(j['h_w'])} / S = {ms(j['h_s'])}", v0_line,
            f"- Bc（bench の infer metal 64 reuse 単独）: {ms(j['bc_m'])}",
            "- BcT（train の直後の infer）: " + ("未実施" if j["bct_m"] is None else ms(j["bct_m"])),
            f"- (v) 版差（v0110-R の S ÷ HEAD-R の S が 1.10 以上または 1/1.10 以下）: {yn(j['ver'])}"
            + ("" if j["v_s"] is None else f"（比 {j['v_s'] / j['h_s']:.3f}）"),
            f"- (h) bench 固有の差（Bc ÷ {j['ref_name']} が 1.10 以上）: {yn(j['hsp'])}（比 {j['bc_m'] / j['ref_w']:.3f}）",
            f"- (o) 実行順の効果（BcT ÷ Bc が 1.10 以上）: {yn(j['ord'])}"
            + ("" if j["bct_m"] is None else f"（比 {j['bct_m'] / j['bc_m']:.3f}）"),
            f"- B 水準 {ms(BENCH_B)} ±10% の再現: " + (", ".join(j["b_hit"]) if j["b_hit"] else "なし"),
            f"- A 水準 {ms(BENCH_A)} ±10% の再現: " + (", ".join(j["a_hit"]) if j["a_hit"] else "なし"),
            f"- **判定: {j['verdict']}**", ""]
    return "\n".join(out)


def r2_aggregate(d, posthoc_r1=False):
    return r2_render(r2_load(d, posthoc_r1))


# --------------------------------------------------------------------------- self-test

def synth_total(kind, i, rng, base):
    if kind == "uni":
        return base * (1 + rng.uniform(-0.02, 0.02))
    if kind == "bi_alt":
        return base * (1.6 if (i // 7) % 2 else 1.0) * (1 + rng.uniform(-0.02, 0.02))
    if kind == "tri":
        return base * (1.0, 1.3, 1.9)[(i // 11) % 3] * (1 + rng.uniform(-0.02, 0.02))
    raise ValueError(kind)


def write_synth(d, kinds, *, gate="pass", n=N, corrupt=None, drop_arm=None):
    """kinds: run 番号 -> 'uni'|'bi_alt'|'tri'|'sticky_hi'。高モードで主区間が増える合成データ。"""
    rng = random.Random(1)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "load_gate_status.txt"), "w") as f:
        for r in RUNS:
            if gate == "missing" and r == 3:
                continue
            f.write(f"run{r} gate={'GATE-TIMEOUT' if gate == 'timeout' and r == 2 else 'pass'}\n")
    for r in RUNS:
        kind = kinds[r]
        lines = []
        for arm in ARMS:
            if arm == drop_arm:
                continue
            count = n - 1 if (arm == "R" and r == 1 and corrupt == "n") else n
            for i in range(count):
                k = "uni" if kind == "sticky_hi" else kind
                base = 0.8e-3 if kind == "sticky_hi" else 0.4e-3
                total = synth_total(k, i, rng, base)
                high = total > base * 1.15 and k != "uni"
                if arm == "G":
                    gpu = 0.1e-3 * (1.8 if high else 1.0)
                    sp = {"upload": 20e-6, "encode": 30e-6, "commit_wait": total * 0.6, "download": 20e-6,
                          "wall": total, "iter_total": total, "gpu_busy": gpu}
                    ct = {"encode": 2, "command_buffers": 1, "wait": 2}
                else:
                    main = total * (0.7 if high else 0.5)
                    sp = {kk: total * 0.05 for kk in SPANS[arm]}
                    sp["predict_resident" if arm == "R" else "chain"] = main
                    sp["iter_total"] = total
                    ct = {"encode": 2, "command_buffers": 1, "wait": 1}
                row = {"arm": arm, "iter": i, "t_since_start_s": i * 1e-3, "spans": sp, "counters": ct,
                       "checksum_bits": "deadbeef" if arm != "G" else "cafe"}
                if corrupt == "nan" and arm == "D" and r == 2 and i == 5:
                    row["spans"]["chain"] = float("nan")
                if corrupt == "neg" and arm == "D" and r == 2 and i == 5:
                    row["spans"]["chain"] = -1.0
                if corrupt == "cs" and arm == "Don" and r == 4 and i == 9:
                    row["checksum_bits"] = "other"
                if corrupt == "cs_arm" and arm == "Don":
                    row["checksum_bits"] = "other"
                lines.append(json.dumps(row))
        if corrupt == "json" and r == 5:
            lines.append("{not json")
        with open(os.path.join(d, f"run{r}.jsonl"), "w") as f:
            f.write("\n".join(lines) + "\n")


def self_test():
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + msg)
        ok = ok and cond

    rng = random.Random(7)
    check(len(detect_modes([1.0 + rng.uniform(0, 0.02) for _ in range(500)])[1]) == 1, "単峰は 1 クラスタ")
    check(len(detect_modes([1.0] * 250 + [1.6] * 250)[1]) == 2, "二峰は 2 クラスタ")
    check(len(detect_modes([1.0] * 200 + [1.3] * 200 + [1.9] * 100)[1]) == 3, "三峰は 3 クラスタ")
    check(len(detect_modes([1.0] * 490 + [2.0] * 10)[1]) == 1, "片側 5% 未満の外れ値は分割しない")
    lb, _ = detect_modes([1.0, 1.0, 1.7, 1.7] * 125)
    check(time_structure(lb)[0] == 249, "切り替え回数が数えられる")

    def expect_invalid(label, **kw):
        with tempfile.TemporaryDirectory() as d:
            write_synth(d, {r: "uni" for r in RUNS}, **kw)
            try:
                aggregate(d)
            except Invalid:
                check(True, f"拒否: {label}")
            else:
                check(False, f"拒否されるべき: {label}")

    expect_invalid("n != 500", corrupt="n")
    expect_invalid("NaN", corrupt="nan")
    expect_invalid("負値", corrupt="neg")
    expect_invalid("反復間 checksum 不一致", corrupt="cs")
    expect_invalid("腕間 checksum 不一致", corrupt="cs_arm")
    expect_invalid("JSON 不正", corrupt="json")
    expect_invalid("腕 G の欠落", drop_arm="G")
    expect_invalid("ゲート状態の欠落", gate="missing")
    expect_invalid("GATE-TIMEOUT", gate="timeout")
    with tempfile.TemporaryDirectory() as d:
        write_synth(d, {r: "uni" for r in RUNS})
        os.remove(os.path.join(d, "run4.jsonl"))
        try:
            aggregate(d)
            check(False, "run 欠落は拒否")
        except Invalid:
            check(True, "拒否: run の欠落")

    with tempfile.TemporaryDirectory() as d:
        write_synth(d, {r: "uni" for r in RUNS})
        md = aggregate(d)
        check("多峰の run がないため" in md, "単峰: 帰属対象なし")
        check("説明できない" in md, "単峰: 系列 A/B は説明できない")
    with tempfile.TemporaryDirectory() as d:
        write_synth(d, {r: "bi_alt" for r in RUNS})
        md = aggregate(d)
        sec_d = md.split("## 腕 D（")[1].split("## 腕 Don")[0]
        check("帰属（" in sec_d and "chain" in sec_d.split("帰属（")[1].split("\n")[0], "二峰交互: D の帰属が chain")
        check("交互出現（切替 10 回以上）: [1, 2, 3, 4, 5]" in md, "二峰交互: 交互出現を検出")
        check("固着: なし" in md, "二峰交互: 固着なし")
    with tempfile.TemporaryDirectory() as d:
        write_synth(d, {r: "tri" for r in RUNS})
        md = aggregate(d)
        check("| 3 |" in md, "三峰を報告")
    with tempfile.TemporaryDirectory() as d:
        write_synth(d, {1: "uni", 2: "uni", 3: "sticky_hi", 4: "uni", 5: "sticky_hi"})
        md = aggregate(d)
        check("固着: あり" in md, "run 単位の固着を検出")
    # wall は iter_total の複製なので、実区間の比がすべて閾値未満なら帰属なしになる
    ent = [{"multimodal": True, "attr": {"upload": 0.25, "encode": 0.25, "commit_wait": 0.25,
                                          "download": 0.25, "wall": 1.0, "gpu_busy": 0.25}}] * 3
    check(attributed_spans(ent)[0] == [], "G: wall 比 1.0 でも実区間が全て 0.25 なら帰属なし")
    ent = [{"multimodal": True, "attr": {"encode": 0.9, "wall": 1.0, "gpu_busy": 0.25}}] * 3
    check([k for k, _ in attributed_spans(ent)[0]] == ["encode"], "G: 実区間 encode のみ帰属")
    # 広い単峰 run（隣接ギャップ 10% 未満）は中央値だけで固着扱いにしない
    def _ent(run, vals, multi):
        return {"run": run, "totals": vals, "median": median(vals), "multimodal": multi,
                "clusters": [{"median": 1.0}, {"median": 2.0}] if multi else []}
    wide = [1.0 + 1.0 * i / 499 for i in range(500)]
    st, _ = sticky_state([_ent(1, [1.0] * 500, True), _ent(2, [2.0] * 500, True), _ent(3, wide, False)])
    check(st is True, "固着: 両端 run があれば広い単峰 run は無関係に判定")
    st, _ = sticky_state([_ent(1, [1.0] * 500, True), _ent(2, wide, False)])
    check(st is False, "固着: 広い単峰 run は 95% を満たさない")
    st, _ = sticky_state([_ent(r, [1.0] * 500, False) for r in RUNS])
    check(st is None, "固着: モード未定義は判定不能")

    def _s(t, g):
        return {"spans": {"iter_total": t, "gpu_busy": g}, "counters": {}}
    ser = [_s(1.0, 0.5)] * 5 + [_s(2.0, 1.5)] * 5 + [_s(9.0, None)] * 5
    at, ex, _ = attribute(ser, [0] * 5 + [1] * 5 + [1] * 5, 1, 0)
    check(abs(at["gpu_busy"] - 1.0) < 1e-9 and ex == 5, "gpu_busy: null 反復を分母からも除外")
    # ---- RULE2（第 2 ラウンド）
    def r2_rows(arm, run, *, tail=False, startup=False, low=False, cs="beef"):
        rows = []
        for i in range(N):
            is_tail = tail and i % 20 == 5          # 25 反復 = 5%
            mult = 1.3 if (startup and i < STARTUP_WIN) else 1.0
            lowm = 2.0 if low else 1.0
            if arm == "G":
                cw = (250e-6 + (150e-6 if is_tail else 0.0)) * mult * lowm
                gb = 90e-6 * lowm
                sp = {"upload": 20e-6, "encode": 5e-6, "commit_wait": cw, "download": 10e-6, "gpu_busy": gb}
                sp["wall"] = sp["iter_total"] = sp["upload"] + sp["encode"] + cw + sp["download"]
            else:
                chk = 2e-6 * lowm
                pr = (290e-6 + (200e-6 if is_tail else 0.0)) * mult * lowm
                sp = {k: 1e-6 for k in SPANS[arm]}
                sp["checksum"] = chk
                sp["predict_resident" if arm == "R" else "chain"] = pr
                sp["iter_total"] = pr + 3 * 1e-6 + chk
            rows.append(json.dumps({"arm": arm, "iter": i, "t_since_start_s": i * 1e-3, "spans": sp,
                                    "counters": {"encode": 2}, "checksum_bits": cs if arm != "G" else "g1"}))
        return "\n".join(rows) + "\n"

    def r2_bench_line(m):
        return json.dumps({"task": "infer", "device": "metal", "size": 64, "mode": "reuse",
                           "median_s": m, "checksum": 13.97}) + "\n"

    def write_r2(d, *, tail_runs=(), startup=False, low_run=None, v0="ok", bc=0.40e-3, bct=0.45e-3,
                 drop=None, posthoc=False, v0_slow=1.0):
        os.makedirs(d, exist_ok=True)
        hd = d if posthoc else os.path.join(d, "head")
        bd = os.path.join(d, "bench_control")
        os.makedirs(hd, exist_ok=True)
        os.makedirs(bd, exist_ok=True)
        with open(os.path.join(d, "load_gate_status.txt"), "w") as f:
            f.write("".join(f"run{r} gate=pass\n" for r in RUNS))
        if not posthoc:
            os.makedirs(os.path.join(d, "v0110"), exist_ok=True)
        for r in RUNS:
            with open(os.path.join(hd, f"run{r}.jsonl"), "w") as f:
                for arm in R2_CONDS_HEAD:
                    f.write(r2_rows(arm, r, tail=r in tail_runs, startup=startup, low=(r == low_run)))
            if not posthoc and v0 != "none":
                with open(os.path.join(d, "v0110", f"run{r}.jsonl"), "w") as f:
                    for arm in R2_CONDS_V0:
                        txt = r2_rows(arm, r, cs="beef" if v0 == "ok" else "dead")
                        if v0_slow != 1.0:
                            scaled = []
                            for ln in txt.splitlines():
                                o = json.loads(ln)
                                o["spans"] = {k: v * v0_slow for k, v in o["spans"].items()}
                                scaled.append(json.dumps(o))
                            txt = "\n".join(scaled) + "\n"
                        f.write(txt)
            if posthoc:
                with open(os.path.join(bd, f"run{r}.jsonl"), "w") as f:
                    f.write(r2_bench_line(bc))
            else:
                if not (drop == "bc" and r == 3):
                    with open(os.path.join(bd, f"run{r}_Bc.jsonl"), "w") as f:
                        f.write(r2_bench_line(bc))
                with open(os.path.join(bd, f"run{r}_BcT.jsonl"), "w") as f:
                    f.write(r2_bench_line(bct))
        if not posthoc:
            if v0 == "none":
                with open(os.path.join(d, "v0110", "NOT_RUN.txt"), "w") as f:
                    f.write("ビルド不能（テスト用）\n")
            with open(os.path.join(d, "run_meta.txt"), "w") as f:
                for r in RUNS:
                    f.write(f"run={r} order=R,D,Don,G since_build_s={r * 10} load1=1.20 load5=1.10\n")

    with tempfile.TemporaryDirectory() as d:
        write_r2(d, tail_runs=(2, 3, 4), v0="ok")
        md = r2_aggregate(d)
        sec = md.split("### HEAD-G")[1].split("###")[0]
        check("commit_wait（3/3 run）" in sec and "commit_wait-gpu_busy（3/3 run）" in sec, "r2: 裾は G の commit_wait に帰属")
        check(not any(x.startswith("gpu_busy（") for x in sec.split("帰属（")[1].split("\n")[0].split(": ", 1)[1].split(", ")),
              "r2: gpu_busy は帰属に入らない")
        check("GPU 実行時間は不変" in sec and "成立（3/3 run）" in sec, "r2: G の GPU 実行時間は不変が成立")
        check("**システム全体の低性能状態（GPU・CPU 両方）の run: なし**" in md, "r2: 低性能 run なし")
        check("立ち上がり上振れの腕×run: 0 /" in md, "r2: 立ち上がり上振れなし")
        check("なし（比 1.000）" in md, "r2: 同一データなら版差なし")
    with tempfile.TemporaryDirectory() as d:
        write_r2(d, startup=True, low_run=2, v0="ok")
        md = r2_aggregate(d)
        check("立ち上がり上振れの腕×run: 20 / 35" in md, "r2: 立ち上がり窓を検出")
        check("**システム全体の低性能状態（GPU・CPU 両方）の run: [2]**" in md, "r2: 低性能 run を検出")
    with tempfile.TemporaryDirectory() as d:
        write_r2(d, v0="none")
        md = r2_aggregate(d)
        check("v0110 条件は未実施: ビルド不能（テスト用）" in md and "判定不能（条件未実施）" in md, "r2: v0110 未実施を明記")
        check("HEAD-R の W（v0110 が未実施または無効のため代用）" in md, "r2: (h) は HEAD-R の W で代用")
    with tempfile.TemporaryDirectory() as d:
        write_r2(d, v0="mismatch")
        md = r2_aggregate(d)
        check("HEAD と不一致" in md and "比較から除外" in md, "r2: v0110 の checksum 不一致は比較から除外")
    with tempfile.TemporaryDirectory() as d:
        write_r2(d, v0="ok", v0_slow=1.5, bc=0.39e-3, bct=0.54e-3)
        md = r2_aggregate(d)
        check("(v) 版差（v0110-R の S ÷ HEAD-R の S が 1.10 以上または 1/1.10 以下）: あり" in md, "r2: 版差あり")
        check("(o) 実行順の効果（BcT ÷ Bc が 1.10 以上）: あり" in md, "r2: 実行順の効果あり")
        check("B 水準 0.388 ms ±10% の再現: Bc" in md and "A 水準 0.539 ms ±10% の再現: BcT" in md, "r2: A/B 水準の再現を列挙")
        check("**判定: 説明できる**" in md, "r2: 両水準を再現なら説明できる")
    with tempfile.TemporaryDirectory() as d:
        write_r2(d, v0="ok", bc=0.30e-3, bct=0.31e-3)
        md = r2_aggregate(d)
        check("**判定: 説明できない**" in md, "r2: どちらも再現しなければ説明できない")
    with tempfile.TemporaryDirectory() as d:
        write_r2(d, v0="ok", drop="bc")
        try:
            r2_aggregate(d)
            check(False, "r2: Bc の欠落は拒否されるべき")
        except Invalid:
            check(True, "r2: 拒否 Bc の欠落")
    with tempfile.TemporaryDirectory() as d:
        write_r2(d, v0="ok")
        os.remove(os.path.join(d, "run_meta.txt"))
        try:
            r2_aggregate(d)
            check(False, "r2: run_meta 欠落は拒否されるべき")
        except Invalid:
            check(True, "r2: 拒否 run_meta の欠落")
    with tempfile.TemporaryDirectory() as d:
        write_r2(d, posthoc=True, tail_runs=(2, 3, 4))
        md = r2_aggregate(d, posthoc_r1=True)
        check("事後・探索。判定に使わない" in md and "第 1 ラウンドに存在しない条件" in md, "r2: 事後適用の表示")
    # RULE 1 の bench 比の分母は実測の低モード中央値（定数 0.385 ms ではない）
    with tempfile.TemporaryDirectory() as d:
        write_synth(d, {r: "bi_alt" for r in RUNS})
        md = aggregate(d)
        wj = [ln for ln in md.splitlines() if "bench 比" in ln][0]
        lowm = float(wj.split("低モード中央値: ")[1].split(" ms")[0]) * 1e-3
        b_ratio = float(wj.split("分母は実測の低モード中央値）: ")[1].split(" /")[0])
        check(abs(b_ratio - BENCH_B / lowm) < 0.01 and abs(lowm - BENCH_LOW) > 1e-5,
              "RULE1: bench 比の分母は実測の低モード中央値")
    print("self-test: " + ("OK" if ok else "NG"))
    return 0 if ok else 1


def main(argv):
    if len(argv) == 2 and argv[1] == "--self-test":
        return self_test()
    if len(argv) in (4, 5) and argv[1] == "--rule" and argv[2] == "2" and (len(argv) == 4 or argv[3] == "--r1"):
        try:
            print(r2_aggregate(argv[-1], posthoc_r1=(len(argv) == 5)))
        except Invalid as e:
            print(f"INVALID: {e}", file=sys.stderr)
            return 1
        return 0
    if len(argv) != 2:
        print("usage: aggregate.py <dir> | --rule 2 [--r1] <dir> | --self-test", file=sys.stderr)
        return 2
    try:
        print(aggregate(argv[1]))
    except Invalid as e:
        print(f"INVALID: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
