#!/usr/bin/env python3
"""イシュー #3186 の集計器。判定規則は同ディレクトリの RULE.txt（実測前に固定）が正。

使い方:
  python3 aggregate.py <dir>     # <dir>/aggregate.md を生成（exit 0）
  python3 aggregate.py --self-test

fail-closed: ファイル欠落・run 不足・checksum 不一致は例外（exit != 0）。
欠損を寛容にパースしない（集計を黙って通さない）。
"""
import json
import math
import os
import statistics
import sys

TRIALS = 20
SIZES = [512, 1024, 2048, 4096]
RUNS = 5
ARMS = [
    "drop_pretouched", "keep_pretouched", "drop_untouched", "keep_untouched",
    "drop_reused", "drop_pinned", "keep_pinned",
    "drop_pretouched_split", "keep_pretouched_split",
]
FRONT = ["h2d_a", "h2d_b", "alloc_c", "launch_issue"]
KNOWN = {1024: -1855.597736, 2048: -6016.774008, 4096: -25768.747284}
CHECKSUM_TOL = 1e-5
SUPPORT, REJECT = 0.80, 0.50
HIGH_MODE_FACTOR = 5.0
UNTOUCHED_ARMS = ["drop_untouched", "keep_untouched"]
NONTOUCHED_D2H_ARMS = ["drop_pretouched_split", "keep_pretouched_split", "drop_reused"]


class Invalid(Exception):
    pass


def load_jsonl_one(path):
    if not os.path.isfile(path):
        raise Invalid(f"欠損: {path}")
    with open(path) as f:
        lines = [l for l in f.read().splitlines() if l.strip()]
    if len(lines) != 1:
        raise Invalid(f"1 行であるべき: {path} ({len(lines)} 行)")
    return json.loads(lines[0])


def check_trials(row, label):
    """全区間の trials_us が TRIALS 要素の有限数値配列であることを検証する（欠損・短縮を拒否）。"""
    for name, ph in row.get("phases", {}).items():
        t = ph.get("trials_us") if isinstance(ph, dict) else None
        if not isinstance(t, list) or len(t) != TRIALS:
            raise Invalid(f"trials_us が {TRIALS} 要素でない: {label} {name}")
        if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
                   for x in t):
            raise Invalid(f"trials_us に非有限・非数値: {label} {name}")
        # 判定の分母・分子に使う run 内 median も有限・非負の数値であることを要求する
        st = ph.get("stat")
        nonneg_number(st.get("median_us") if isinstance(st, dict) else None,
                      f"stat.median_us {label} {name}")


def nonneg_number(x, label):
    """有限かつ非負の数値（bool 不可）であることを検証する。NaN・inf・負値・欠損は例外。"""
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0:
        raise Invalid(f"有限・非負の数値でない: {label} = {x!r}")
    return x


def finite_number(x, label):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise Invalid(f"checksum が有限の数値でない: {label} = {x!r}")
    return x


def load_dir(d):
    """{(run, n, arm): row}, {(run, n, mode): layerA row}, gate dict を返す。"""
    b, a = {}, {}
    for k in range(1, RUNS + 1):
        for n in SIZES:
            for arm in ARMS:
                row = load_jsonl_one(os.path.join(d, f"run{k}", f"n{n}_{arm}.jsonl"))
                if row.get("n") != n or row.get("arm") != arm:
                    raise Invalid(f"n/arm 不一致: run{k} n{n} {arm}")
                check_trials(row, f"run{k} n{n} {arm}")
                b[(k, n, arm)] = row
            for mode in ("fresh", "reuse"):
                a[(k, n, mode)] = load_jsonl_one(
                    os.path.join(d, f"run{k}", f"layerA_{mode}_n{n}.jsonl"))
    gate = {}
    p = os.path.join(d, "load_gate_status.txt")
    if os.path.isfile(p):
        for line in open(p).read().splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1].startswith("gate="):
                gate[parts[0]] = parts[1][5:]
    return b, a, gate


def gate_ok(gate):
    return all(gate.get(f"run{k}") == "pass" for k in range(1, RUNS + 1))


def check_checksums(b, a):
    for n in SIZES:
        bits = {b[(k, n, arm)]["checksum_bits"] for k in range(1, RUNS + 1) for arm in ARMS}
        if len(bits) != 1:
            raise Invalid(f"N={n}: Layer B checksum_bits が不一致 {sorted(bits)}")
        for k in range(1, RUNS + 1):
            for arm in ARMS:
                finite_number(b[(k, n, arm)].get("checksum"), f"Layer B run{k} n{n} {arm}")
        ref = b[(1, n, ARMS[0])]["checksum"]
        if n in KNOWN and abs(ref - KNOWN[n]) > CHECKSUM_TOL:
            raise Invalid(f"N={n}: 既知 checksum と不一致 {ref} vs {KNOWN[n]}")
        for k in range(1, RUNS + 1):
            for mode in ("fresh", "reuse"):
                nonneg_number(a[(k, n, mode)].get("median_s"), f"Layer A {mode} run{k} n{n} median_s")
                c = finite_number(a[(k, n, mode)].get("checksum"), f"Layer A {mode} run{k} n{n}")
                if abs(c - ref) > CHECKSUM_TOL:
                    raise Invalid(f"N={n}: Layer A {mode} run{k} checksum {c} vs {ref}")


def med_phase(row, name):
    ph = row["phases"].get(name)
    if ph is None:
        raise Invalid(f"区間欠落: {name}")
    return ph["stat"]["median_us"]


def readback_total(row):
    """D2H を含む readback 区間の合計（run 内 median の和）。"""
    ph = row["phases"]
    if "readback_total" in ph:
        return med_phase(row, "readback_total")
    if "dest_alloc" in ph:
        return sum(med_phase(row, p) for p in ("dest_alloc", "pretouch_fill", "d2h_issue", "d2h_sync"))
    return med_phase(row, "d2h_issue") + med_phase(row, "d2h_sync")


def d2h_only(row):
    ph = row["phases"]
    if "readback_total" in ph:
        return med_phase(row, "readback_total")  # untouched 腕のみ呼ぶ
    return med_phase(row, "d2h_issue") + med_phase(row, "d2h_sync")


def across_runs(fn):
    vals = [fn(k) for k in range(1, RUNS + 1)]
    return statistics.median(vals), min(vals), max(vals), vals


def verdict(share, ok):
    if share is None:
        return "判定しない"
    if share >= SUPPORT:
        v = "支持"
    elif share < REJECT:
        v = "棄却"
    else:
        v = "未確定"
    return v if ok else f"{v}（参考扱い: 専有ゲート未通過／不明）"


def judge_w(b, a, n, gate_pass):
    # RULE.txt: median(keep) - median(drop)（各腕の run 間中央値を取ってから差を取る）
    gap_ret = (across_runs(lambda k: readback_total(b[(k, n, "keep_pretouched")]))[0]
               - across_runs(lambda k: readback_total(b[(k, n, "drop_pretouched")]))[0])
    a_fresh = across_runs(lambda k: a[(k, n, "fresh")]["median_s"] * 1e6)[0]
    a_reuse = across_runs(lambda k: a[(k, n, "reuse")]["median_s"] * 1e6)[0]
    gap_a = a_reuse - a_fresh
    if gap_a <= 0:
        return dict(n=n, gap_ret=gap_ret, gap_a=gap_a, share=None,
                    verdict="非再現（reuse−fresh が 0 以下。判定しない）",
                    a_fresh=a_fresh, a_reuse=a_reuse)
    share = gap_ret / gap_a
    v = verdict(share, gate_pass)
    if n == 512:
        v += "［record_only］"
    return dict(n=n, gap_ret=gap_ret, gap_a=gap_a, share=share, verdict=v,
                a_fresh=a_fresh, a_reuse=a_reuse)


def judge_f(b, a, gate_pass, n=4096):
    reused = lambda k: b[(k, n, "drop_reused")]
    # RULE.txt: median(drop_pretouched) - median(drop_reused)（中央値の差）
    page = (across_runs(lambda k: readback_total(b[(k, n, "drop_pretouched")]))[0]
            - across_runs(lambda k: readback_total(reused(k)))[0])
    host = across_runs(lambda k: med_phase(reused(k), "host_read"))[0]
    layerb_sum = across_runs(lambda k: sum(med_phase(reused(k), p) for p in FRONT)
                             + readback_total(reused(k)))[0]
    a_fresh = across_runs(lambda k: a[(k, n, "fresh")]["median_s"] * 1e6)[0]
    residual = a_fresh - layerb_sum - host
    d2h_reused = across_runs(lambda k: readback_total(reused(k)))[0]
    if residual <= 0:
        return dict(page=page, residual=residual, share=None, a_fresh=a_fresh, host=host,
                    layerb_sum=layerb_sum, d2h_reused=d2h_reused,
                    verdict="非再現（residual が 0 以下）")
    share = page / residual
    v = verdict(share, gate_pass)
    v = v.replace("支持", "帰属できた").replace("棄却", "帰属できなかった（主因は別）")
    return dict(page=page, residual=residual, share=share, a_fresh=a_fresh, host=host,
                layerb_sum=layerb_sum, d2h_reused=d2h_reused, verdict=v)


def high_mode_counts(b, n=4096):
    base = across_runs(lambda k: d2h_only(b[(k, n, "drop_reused")]))[0]
    thr = HIGH_MODE_FACTOR * base
    out = {}
    for arm in UNTOUCHED_ARMS + NONTOUCHED_D2H_ARMS:
        per_run = []
        for k in range(1, RUNS + 1):
            row = b[(k, n, arm)]
            ph = row["phases"]
            if "readback_total" in ph:
                trials = ph["readback_total"]["trials_us"]
            else:
                ia, isy = ph["d2h_issue"]["trials_us"], ph["d2h_sync"]["trials_us"]
                trials = [x + y for x, y in zip(ia, isy)]
            per_run.append(sum(1 for t in trials if t > thr))
        out[arm] = per_run
    untouched_hi = any(sum(out[a]) > 0 for a in UNTOUCHED_ARMS)
    other_hi = any(sum(out[a]) > 0 for a in NONTOUCHED_D2H_ARMS)
    if untouched_hi and not other_hi:
        v = "未タッチ宛先が条件"
    elif not untouched_hi and not other_hi:
        v = "再現せず"
    else:
        v = "未確定（混在）"
    return thr, out, v


def fault_summary(b, n):
    rows = []
    for arm in ARMS:
        mf = [sum(x for x in b[(k, n, arm)].get("minflt_trials", []) if x is not None)
              for k in range(1, RUNS + 1)]
        rows.append((arm, statistics.median(mf)))
    return rows


def render(b, a, gate):
    gp = gate_ok(gate)
    L = ["# #3186 集計（RULE.txt に従う）", ""]
    L.append(f"専有ゲート: {'全 run pass' if gp else '参考扱い（未通過・欠落・不明あり）'} {gate}")
    L += ["", "## 判定 W（窓の構成＝tape 蓄積の寄与）", "",
          "| N | Layer A fresh (us) | Layer A reuse (us) | gap_A | gap_retention | retention_share | 判定 |",
          "|---|---|---|---|---|---|---|"]
    for n in SIZES:
        w = judge_w(b, a, n, gp)
        sh = "-" if w["share"] is None else f"{w['share']:.3f}"
        L.append(f"| {n} | {w['a_fresh']:.1f} | {w['a_reuse']:.1f} | {w['gap_a']:.1f} | "
                 f"{w['gap_ret']:.1f} | {sh} | {w['verdict']} |")
    f = judge_f(b, a, gp)
    L += ["", "## 判定 F（N=4096 fresh の未説明分）", "",
          f"- Layer A fresh: {f['a_fresh']:.1f} us / Layer B Σ(drop_reused): {f['layerb_sum']:.1f} us / "
          f"host checksum: {f['host']:.1f} us",
          f"- residual_fresh_4096 = {f['residual']:.1f} us / page_cost_4096 = {f['page']:.1f} us",
          f"- share_F = {'-' if f['share'] is None else format(f['share'], '.3f')} -> {f['verdict']}",
          f"- 記録のみ: drop_reused の D2H 区間 {f['d2h_reused']:.1f} us（帯域下限の推定約 1100 us との差 "
          f"{f['d2h_reused'] - 1100:.1f} us）"]
    thr, hm, hv = high_mode_counts(b)
    L += ["", "## 二峰性（N=4096）", "", f"- 高モード閾値: {thr:.1f} us（5 x drop_reused の D2H 中央値）",
          f"- 判定: {hv}{'' if gp else '（参考扱い: 専有ゲート未通過／不明）'}", "", "| 腕 | run1..5 の高モード試行数 |", "|---|---|"]
    for arm, cnt in hm.items():
        L.append(f"| {arm} | {cnt} |")
    L += ["", "## minflt 合計（計測 20 試行・run 間中央値。付記のみ）", ""]
    for n in SIZES:
        L.append(f"- N={n}: " + ", ".join(f"{arm}={v:.0f}" for arm, v in fault_summary(b, n)))
    L += ["", "## 区間表（run 間中央値 [min-max]・us）", "",
          "| N | 腕 | readback 区間 | min | max |", "|---|---|---|---|---|"]
    for n in SIZES:
        for arm in ARMS:
            med, lo, hi, _ = across_runs(lambda k: readback_total(b[(k, n, arm)]))
            L.append(f"| {n} | {arm} | {med:.1f} | {lo:.1f} | {hi:.1f} |")
    L += ["", "## run 別（run 番号付き。readback 区間 us）", ""]
    for n in SIZES:
        for arm in ARMS:
            vals = ", ".join(f"r{k}={readback_total(b[(k, n, arm)]):.1f}" for k in range(1, RUNS + 1))
            L.append(f"- N={n} {arm}: {vals}")
    return "\n".join(L) + "\n"


def main_dir(d):
    b, a, gate = load_dir(d)
    check_checksums(b, a)
    text = render(b, a, gate)
    with open(os.path.join(d, "aggregate.md"), "w") as f:
        f.write(text)
    print(text)


# ---- self-test ----
def _row(n, arm, rb=1000.0, host=100.0, cs=-1855.597736, trials=None, extra=None):
    def ph(m, t=None):
        return {"stat": {"median_us": m}, "trials_us": t or [m] * 20}
    phases = {p: ph(10.0) for p in FRONT}
    if "split" in arm:
        phases.update({"dest_alloc": ph(5.0), "pretouch_fill": ph(rb - 25), "d2h_issue": ph(10.0),
                       "d2h_sync": ph(10.0, trials)})
    elif arm == "drop_reused":
        phases.update({"d2h_issue": ph(10.0), "d2h_sync": ph(rb - 10, trials)})
    else:
        phases["readback_total"] = ph(rb, trials)
    phases["host_read"] = ph(host)
    return {"n": n, "arm": arm, "checksum": cs, "checksum_bits": f"{cs.hex()}",
            "phases": phases, "minflt_trials": [1] * 20}


def _synthetic(keep_extra=0.0, a_fresh=1000.0, a_reuse=3000.0, bad_checksum=False):
    b, a = {}, {}
    for k in range(1, RUNS + 1):
        for n in SIZES:
            cs = KNOWN.get(n, -100.0)
            for arm in ARMS:
                rb = 1000.0
                if arm == "keep_pretouched":
                    rb += keep_extra
                if arm == "drop_pretouched":
                    rb = 1500.0
                if arm == "drop_reused":
                    rb = 500.0
                c = cs + (1.0 if bad_checksum and k == 3 and arm == "keep_pinned" else 0.0)
                b[(k, n, arm)] = _row(n, arm, rb=rb, cs=c)
            for mode, v in (("fresh", a_fresh), ("reuse", a_reuse)):
                a[(k, n, mode)] = {"median_s": v * 1e-6, "checksum": cs}
    return b, a


def self_test():
    # 判定境界
    assert verdict(0.80, True) == "支持"
    assert verdict(0.7999, True) == "未確定"
    assert verdict(0.50, True) == "未確定"
    assert verdict(0.4999, True) == "棄却"
    assert "参考扱い" in verdict(0.9, False)
    # W: gap_A = 2000, gap_ret = keep_extra - 500 (keep 1000+extra vs drop 1500)
    b, a = _synthetic(keep_extra=2100.0)  # gap_ret = 1600 -> 0.8
    check_checksums(b, a)
    w = judge_w(b, a, 1024, True)
    assert abs(w["share"] - 0.8) < 1e-9 and w["verdict"] == "支持", w
    b, a = _synthetic(keep_extra=1500.0)  # gap_ret = 1000 -> 0.5
    assert judge_w(b, a, 1024, True)["verdict"] == "未確定"
    b, a = _synthetic(keep_extra=1400.0)  # 900 -> 0.45
    assert judge_w(b, a, 1024, True)["verdict"] == "棄却"
    assert "record_only" in judge_w(b, a, 512, True)["verdict"]
    b, a = _synthetic(a_fresh=3000.0, a_reuse=3000.0)
    assert judge_w(b, a, 1024, True)["verdict"].startswith("非再現")
    # F: page = 1500-500 = 1000; layerb = 4*10 + 500 = 540; host=100
    b, a = _synthetic(a_fresh=1740.0)  # residual = 1740-540-100 = 1100 -> 0.909
    f = judge_f(b, a, True)
    assert abs(f["residual"] - 1100.0) < 1e-9 and "帰属できた" in f["verdict"], f
    b, a = _synthetic(a_fresh=500.0)
    assert judge_f(b, a, True)["verdict"].startswith("非再現")
    # checksum 不一致は例外
    b, a = _synthetic(bad_checksum=True)
    try:
        check_checksums(b, a)
        raise AssertionError("checksum 不一致を検出できていない")
    except Invalid:
        pass
    # NaN checksum / trials 欠損・短縮は例外
    b, a = _synthetic()
    a[(2, 1024, "reuse")]["checksum"] = float("nan")
    try:
        check_checksums(b, a)
        raise AssertionError("NaN checksum を検出できていない")
    except Invalid:
        pass
    for bad in ([], [1.0] * 19):
        try:
            r = _row(4096, "drop_untouched")
            r["phases"]["readback_total"]["trials_us"] = bad
            check_trials(r, "t")
            raise AssertionError("trials 長不足を検出できていない")
        except Invalid:
            pass
    # median_s / stat.median_us の NaN・負値は例外
    for bad in (float("nan"), float("inf"), -1.0, None):
        b, a = _synthetic()
        a[(1, 1024, "fresh")]["median_s"] = bad
        try:
            check_checksums(b, a)
            raise AssertionError("median_s 不正を検出できていない")
        except Invalid:
            pass
        r = _row(4096, "drop_untouched")
        r["phases"]["readback_total"]["stat"]["median_us"] = bad
        try:
            check_trials(r, "t")
            raise AssertionError("stat.median_us 不正を検出できていない")
        except Invalid:
            pass
    # 欠損は例外
    try:
        load_jsonl_one("/nonexistent/path.jsonl")
        raise AssertionError("欠損を検出できていない")
    except Invalid:
        pass
    # 二峰性: untouched にのみ高モード
    b, a = _synthetic()
    hi = [500.0] * 19 + [20000.0]
    for k in range(1, RUNS + 1):
        b[(k, 4096, "drop_untouched")] = _row(4096, "drop_untouched", rb=500.0, cs=KNOWN[4096], trials=hi)
    assert high_mode_counts(b)[2] == "未タッチ宛先が条件"
    b[(2, 4096, "drop_reused")] = _row(4096, "drop_reused", rb=500.0, cs=KNOWN[4096], trials=hi)
    assert high_mode_counts(b)[2].startswith("未確定")
    b, a = _synthetic()
    assert high_mode_counts(b)[2] == "再現せず"
    # ゲート判定
    assert gate_ok({f"run{k}": "pass" for k in range(1, 6)})
    assert not gate_ok({"run1": "pass"})
    # render が通る
    b, a = _synthetic(keep_extra=2100.0)
    assert "判定 W" in render(b, a, {f"run{k}": "pass" for k in range(1, 6)})
    assert "二峰性（N=4096）" in render(b, a, {}) and "参考扱い" in render(b, a, {}).split("## 二峰性")[1]
    print("self-test OK")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--self-test":
        self_test()
    elif len(sys.argv) == 2:
        try:
            main_dir(sys.argv[1])
        except Invalid as e:
            print(f"系列は無効: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(__doc__)
        sys.exit(2)
