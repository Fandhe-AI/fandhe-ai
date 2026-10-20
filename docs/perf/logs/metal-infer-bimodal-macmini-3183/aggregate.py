#!/usr/bin/env python3
"""イシュー #3183: Mac mini の infer Metal 二峰性診断の集計器（python3 標準ライブラリのみ）。

判定規則は同ディレクトリの RULE.txt（実測前に固定）が正。orchestrate.sh が残した
run{1..5}.jsonl（1 反復 1 行）と load_gate_status.txt を読み、モード検出・帰属・
系列 A/B の窓判定を Markdown で出力する。

fail-closed（exit != 0）: run の欠落、n != 500、非有限値・負値、JSON 不正、腕の欠落、
iter の欠番、ゲート状態の欠落・不合格、checksum の反復間／run 間／腕間（R・D・Don）不一致。

使い方: python3 -I aggregate.py <dir> / python3 -I aggregate.py --self-test
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


def load_run(path, n):
    rows = {a: [] for a in ARMS}
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
    for arm in ARMS:
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
    d_total = median([s["spans"]["iter_total"] for s in hs]) - median([s["spans"]["iter_total"] for s in ls])
    attr, excluded = {}, 0
    for k in series[0]["spans"]:
        if k == "iter_total":
            continue
        hv = [s["spans"][k] for s in hs if s["spans"][k] is not None]
        lv = [s["spans"][k] for s in ls if s["spans"][k] is not None]
        if k in ("gpu_busy", "commit_wait-gpu_busy"):
            excluded = max(excluded, len(hs) + len(ls) - len(hv) - len(lv))
        if not hv or not lv or d_total <= 0:
            attr[k] = None
            continue
        attr[k] = (median(hv) - median(lv)) / d_total
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
    """プロセス単位の固着判定。単峰 run は参照モード（多峰 run があればその中央値、無ければ run 中央値）に近い側で分類。"""
    multi = [e for e in entries if e["multimodal"]]
    if multi:
        ref_lo = median([e["clusters"][0]["median"] for e in multi])
        ref_hi = median([e["clusters"][-1]["median"] for e in multi])
    else:
        meds = [e["median"] for e in entries]
        ref_lo, ref_hi = min(meds), max(meds)
        if ref_hi / ref_lo - 1.0 < GAP:
            return False, ["全 run が単峰かつ run 間差 10% 未満"]
    desc, lows, highs = [], 0, 0
    for e in entries:
        if e["multimodal"]:
            lo_share, hi_share = e["low_share"], e["high_share"]
        else:
            near_lo = abs(e["median"] - ref_lo) <= abs(e["median"] - ref_hi)
            lo_share, hi_share = (1.0, 0.0) if near_lo else (0.0, 1.0)
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
        "ratio_b": BENCH_B / BENCH_LOW, "ratio_a": BENCH_A / BENCH_LOW,
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
        out.append(f"- プロセス単位の固着: {'あり' if sticky else 'なし'}（" + "; ".join(desc) + "）")
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
            f"- 低モード中央値: {ms(w['low_med'])}。bench 比（B/低・A/低）: {w['ratio_b']:.2f} / {w['ratio_a']:.2f}"]
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
    print("self-test: " + ("OK" if ok else "NG"))
    return 0 if ok else 1


def main(argv):
    if len(argv) == 2 and argv[1] == "--self-test":
        return self_test()
    if len(argv) != 2:
        print("usage: aggregate.py <dir> | --self-test", file=sys.stderr)
        return 2
    try:
        print(aggregate(argv[1]))
    except Invalid as e:
        print(f"INVALID: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
