#!/usr/bin/env python3
"""イシュー #3671: 診断側の独立 5 run（minflt 補助のみ 1 run）の JSONL を RULE.txt／RULE.v2.txt の規則で集計する。

役割: orchestrate.sh が残す run{1..5}.jsonl・load_gate_status.txt・series_kind.txt・env_info.txt を検証し、
(arm, n, phase) ごとの 5 run 中央値・min-max と run 番号付きの iter_total 表・G_X を出力する。
判定（支持／非支持）は bench_judge.py が bench 側の値と合わせて行う（本スクリプトは閾値を持たない）。
標準ライブラリのみ。`python3 -I aggregate.py <dir>` で実行する。寛容なパースはしない:
欠落セル・run 欠落・不正 JSON・n_samples 不一致・非有限／非正の median_s・checksum の bit 不一致・
ゲート状態／系列種別／env_info の欠落・ビルド失敗の記録（空の jsonl）はすべて例外にして exit 1。

使い方: aggregate.py <dir>        （dir は series_kind が main／solo／minflt の系列）
        aggregate.py --self-test
正式系列の条件: series_kind が main または solo で、全 run が gate=pass。満たさない系列は「参考記録」と出力する
（専有 excl=pass は bench_judge.py が excl_status.txt で別途確認する）。
"""
import json
import math
import os
import statistics
import sys
import tempfile

SIZES = [256, 512, 1024, 2048, 4096]
ARMS_MAIN = {
    "fresh": ["tape_build", "var_upload", "matmul", "readout", "checksum", "iter_total", "tape_drop"],
    "resident_public": ["matmul", "readout", "checksum", "iter_total"],
    "rp_verify_touch": ["matmul", "readout", "checksum", "iter_total", "gap"],
    "rp_verify_gap": ["matmul", "readout", "checksum", "iter_total", "gap"],
    "rp_hold_footprint": ["matmul", "readout", "checksum", "iter_total"],
    "rp_ref_compute": ["matmul", "readout", "checksum", "iter_total"],
}
ARMS_SOLO = {"rp_solo": ["matmul", "readout", "checksum", "iter_total"]}
KINDS = {"main": ARMS_MAIN, "minflt": ARMS_MAIN, "solo": ARMS_SOLO}
SERIES_LABEL = {"main": "diag", "minflt": "diag", "solo": "solo"}
# 計測反復 x ラウンド（テスト側 iters_for x ROUNDS。main は 4 ラウンド、solo は 1 ラウンド。N=4096 だけ 10 反復）
ITERS = {n: (10 if n >= 4096 else 20) for n in SIZES}
EXPECTED_N = {
    "main": {n: ITERS[n] * 4 for n in SIZES},
    "minflt": {n: ITERS[n] * 4 for n in SIZES},
    "solo": dict(ITERS),
}
GATE_FILE = "load_gate_status.txt"
KIND_FILE = "series_kind.txt"
ENV_FILE = "env_info.txt"


class AggError(Exception):
    pass


def run_count(kind):
    return 1 if kind == "minflt" else 5


def load_kind(d):
    path = os.path.join(d, KIND_FILE)
    try:
        with open(path, encoding="utf-8") as f:
            kind = f.read().strip()
    except OSError as e:
        raise AggError(f"{path}: 読み込み不可 ({e})")
    if kind not in KINDS:
        raise AggError(f"{path}: 不正な系列種別 {kind!r}")
    return kind


def load_gate(d, nruns):
    """run 名 -> gate 状態。欠落・不正・重複は fail-closed。"""
    path = os.path.join(d, GATE_FILE)
    try:
        with open(path, encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
    except OSError as e:
        raise AggError(f"{path}: 読み込み不可 ({e})")
    seen = {}
    for ln in lines:
        parts = ln.split()
        if len(parts) != 2 or not parts[1].startswith("gate="):
            raise AggError(f"{path}: 不正な行 {ln!r}")
        status = parts[1][5:]
        if status not in ("pass", "unpassed", "record_only") or parts[0] in seen:
            raise AggError(f"{path}: 不正または重複した行 {ln!r}")
        seen[parts[0]] = status
    if set(seen) != {f"run{n}" for n in range(1, nruns + 1)}:
        raise AggError(f"{path}: run1..run{nruns} の全行が必要 ({sorted(seen)})")
    return seen


def load_env(d):
    """env_info.txt から key->value。RULE の sha256 と rev_stamp が欠けていれば fail-closed。"""
    path = os.path.join(d, ENV_FILE)
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except OSError as e:
        raise AggError(f"{path}: 読み込み不可 ({e})")
    env = {}
    for ln in text.splitlines():
        if ": " in ln and not ln.startswith(" "):
            k, v = ln.split(": ", 1)
            env[k.strip()] = v.strip()
    for k in ("rev_stamp_head", "rev_stamp_dirty", "rule_sha256", "rule_v2_sha256", "series_kind"):
        if not env.get(k):
            raise AggError(f"{path}: {k} の記録が無い（ドリフトガード不能）")
    return env


def load_run(path, kind):
    arms = KINDS[kind]
    cells = {}
    try:
        with open(path, encoding="utf-8") as f:
            lines = [ln for ln in f.read().splitlines() if ln.strip()]
    except OSError as e:
        raise AggError(f"{path}: 読み込み不可 ({e})")
    if not lines:
        raise AggError(f"{path}: 空（実行失敗またはビルド失敗の記録）")
    for i, line in enumerate(lines, 1):
        try:
            rec = json.loads(line)
            key = (rec["arm"], rec["n"], rec["phase"])
            rec["median_s"] = float(rec["median_s"])
            ns = rec["n_samples"]
            issue = rec["issue"]
            series = rec["series"]
        except (ValueError, KeyError, TypeError) as e:
            raise AggError(f"{path}:{i}: 不正な行 ({e})")
        if issue != 3671 or series != SERIES_LABEL[kind]:
            raise AggError(f"{path}:{i}: issue/series が想定外 ({issue!r}, {series!r})")
        if key[0] not in arms or key[2] not in arms[key[0]] or key[1] not in SIZES:
            raise AggError(f"{path}:{i}: 想定外のセル {key}")
        if not math.isfinite(rec["median_s"]) or rec["median_s"] <= 0.0:
            raise AggError(f"{path}:{i}: median_s が有限の正値でない {key}")
        if isinstance(ns, bool) or ns != EXPECTED_N[kind][key[1]]:
            raise AggError(f"{path}:{i}: n_samples={ns!r} が {EXPECTED_N[kind][key[1]]} でない {key}")
        if key in cells:
            raise AggError(f"{path}:{i}: セル重複 {key}")
        if key[2] == "iter_total" and not isinstance(rec.get("checksum_bits"), str):
            raise AggError(f"{path}:{i}: iter_total に checksum_bits が無い {key}")
        cells[key] = rec
    for arm, phases in arms.items():
        for n in SIZES:
            for ph in phases:
                if (arm, n, ph) not in cells:
                    raise AggError(f"{path}: セル欠落 {arm}/{n}/{ph}")
    return cells


def check_checksums(runs, kind):
    """N ごとに全腕・全 run の iter_total checksum_bits が一致すること（RULE.txt checksum 規則）。"""
    for n in SIZES:
        ref = None
        for ri, cells in enumerate(runs, 1):
            for arm in KINDS[kind]:
                b = cells[(arm, n, "iter_total")]["checksum_bits"]
                if ref is None:
                    ref = b
                elif b != ref:
                    raise AggError(f"checksum bit 不一致: n={n} run{ri} {arm} {b} != {ref}")


def summarize(runs):
    """(arm, n, phase) -> (5 run 中央値, min, max)。run ごとの median_s を採る。"""
    out = {}
    for key in runs[0]:
        vals = [r[key]["median_s"] for r in runs]
        out[key] = (statistics.median(vals), min(vals), max(vals))
    return out


def load_series(d):
    """検証済みの系列を返す: dict(kind, env, gate, runs, summary, official, why)。"""
    kind = load_kind(d)
    nruns = run_count(kind)
    gate = load_gate(d, nruns)
    env = load_env(d)
    if env["series_kind"] != kind:
        raise AggError(f"{d}: series_kind.txt と env_info.txt が不一致")
    runs = [load_run(os.path.join(d, f"run{n}.jsonl"), kind) for n in range(1, nruns + 1)]
    check_checksums(runs, kind)
    why = []
    if kind == "minflt":
        why.append("補助系列（判定外）")
    if any(v != "pass" for v in gate.values()):
        why.append("負荷ゲート未通過または record_only の run を含む: " + ",".join(f"{k}={v}" for k, v in sorted(gate.items())))
    if env["rev_stamp_dirty"] != "dirty=0":
        why.append(f"計測ツリーが dirty（{env['rev_stamp_dirty']}）")
    return {"kind": kind, "env": env, "gate": gate, "runs": runs, "summary": summarize(runs),
            "official": not why, "why": why}


def fmt(x):
    return f"{x:.9f}"


def report(s):
    kind, summ, runs = s["kind"], s["summary"], s["runs"]
    arms = KINDS[kind]
    print(f"series_kind: {kind}  runs: {len(runs)}  rev_stamp: {s['env']['rev_stamp_head']} {s['env']['rev_stamp_dirty']}")
    print(f"rule_sha256: {s['env']['rule_sha256']}  rule_v2_sha256: {s['env']['rule_v2_sha256']}")
    print("OFFICIAL:", "yes" if s["official"] else "no（参考記録: " + "; ".join(s["why"]) + "）")
    print("\n## iter_total（秒。5 run 中央値 [min-max]）")
    print("| N | " + " | ".join(arms) + " |")
    print("|---|" + "---|" * len(arms))
    for n in SIZES:
        row = []
        for a in arms:
            m, lo, hi = summ[(a, n, "iter_total")]
            row.append(f"{fmt(m)} [{fmt(lo)}-{fmt(hi)}]")
        print(f"| {n} | " + " | ".join(row) + " |")
    print("\n## run 別 iter_total（run 番号付き）")
    print("| N | arm | " + " | ".join(f"run{i}" for i in range(1, len(runs) + 1)) + " |")
    print("|---|---|" + "---|" * len(runs))
    for n in SIZES:
        for a in arms:
            print(f"| {n} | {a} | " + " | ".join(fmt(r[(a, n, 'iter_total')]['median_s']) for r in runs) + " |")
    if kind in ("main", "minflt"):
        print("\n## G_X = X.iter_total - fresh.iter_total（同じ診断プロセスの fresh。5 run 中央値同士）")
        print("| N | " + " | ".join(a for a in arms if a != "fresh") + " |")
        print("|---|" + "---|" * (len(arms) - 1))
        for n in SIZES:
            f = summ[("fresh", n, "iter_total")][0]
            print(f"| {n} | " + " | ".join(f"{summ[(a, n, 'iter_total')][0] - f:+.9f}" for a in arms if a != "fresh") + " |")
        print("\n## 窓の外 gap 区間（verify 腕。空き時間が揃っているかの記録）")
        print("| N | rp_verify_touch.gap | rp_verify_gap.gap |")
        print("|---|---|---|")
        for n in SIZES:
            print(f"| {n} | {fmt(summ[('rp_verify_touch', n, 'gap')][0])} | {fmt(summ[('rp_verify_gap', n, 'gap')][0])} |")


def _gen_fixture(d, kind="main", runs=None, mutate=None, gate="pass"):
    arms = KINDS[kind]
    nruns = run_count(kind) if runs is None else runs
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, KIND_FILE), "w") as f:
        f.write(kind + "\n")
    with open(os.path.join(d, GATE_FILE), "w") as f:
        for n in range(1, run_count(kind) + 1):
            f.write(f"run{n} gate={gate}\n")
    with open(os.path.join(d, ENV_FILE), "w") as f:
        f.write(f"rev_stamp_head: abc\nrev_stamp_dirty: dirty=0\nrule_sha256: x\nrule_v2_sha256: y\nseries_kind: {kind}\n")
    for r in range(1, nruns + 1):
        lines = []
        for arm, phases in arms.items():
            for n in SIZES:
                for ph in phases:
                    rec = {"issue": 3671, "series": SERIES_LABEL[kind], "arm": arm, "n": n, "phase": ph,
                           "median_s": 0.001 * (1 + r * 0.01), "n_samples": EXPECTED_N[kind][n],
                           "checksum_bits": "00000000deadbeef" if ph == "iter_total" else None}
                    if mutate:
                        rec = mutate(r, rec)
                    if rec is not None:
                        lines.append(json.dumps(rec))
        with open(os.path.join(d, f"run{r}.jsonl"), "w") as f:
            f.write("\n".join(lines) + "\n")


def self_test():
    ok = True

    def expect_fail(name, **kw):
        nonlocal ok
        with tempfile.TemporaryDirectory() as d:
            _gen_fixture(d, **kw)
            try:
                load_series(d)
            except AggError:
                return
            print(f"SELF-TEST FAIL {name}: 例外にならなかった")
            ok = False

    for kind in ("main", "solo", "minflt"):
        with tempfile.TemporaryDirectory() as d:
            _gen_fixture(d, kind=kind)
            try:
                s = load_series(d)
            except AggError as e:
                print(f"SELF-TEST FAIL pass-{kind}: {e}")
                ok = False
                continue
            if (kind == "minflt") == s["official"]:
                print(f"SELF-TEST FAIL official-{kind}")
                ok = False
    with tempfile.TemporaryDirectory() as d:
        _gen_fixture(d, gate="unpassed")
        if load_series(d)["official"]:
            print("SELF-TEST FAIL unpassed-gate-official")
            ok = False
    expect_fail("missing-run", runs=4)
    expect_fail("missing-cell", mutate=lambda r, rec: None if (rec["arm"], rec["n"], rec["phase"]) == ("rp_verify_gap", 512, "gap") else rec)
    expect_fail("checksum-mismatch", mutate=lambda r, rec: {**rec, "checksum_bits": "0000000000000001"} if (r == 3 and rec["arm"] == "rp_ref_compute" and rec["phase"] == "iter_total" and rec["n"] == 256) else rec)
    expect_fail("n-samples", mutate=lambda r, rec: {**rec, "n_samples": 79} if (rec["arm"] == "fresh" and rec["n"] == 256) else rec)
    expect_fail("nonfinite", mutate=lambda r, rec: {**rec, "median_s": float("nan")} if rec["n"] == 1024 else rec)
    expect_fail("nonpositive", mutate=lambda r, rec: {**rec, "median_s": 0.0} if rec["n"] == 1024 else rec)
    expect_fail("unknown-arm", mutate=lambda r, rec: {**rec, "arm": "bogus"} if (rec["arm"] == "fresh" and rec["n"] == 256 and rec["phase"] == "matmul") else rec)
    expect_fail("wrong-issue", mutate=lambda r, rec: {**rec, "issue": 3646})
    expect_fail("no-checksum", mutate=lambda r, rec: {**rec, "checksum_bits": None} if (rec["phase"] == "iter_total" and rec["n"] == 256) else rec)
    with tempfile.TemporaryDirectory() as d:
        _gen_fixture(d)
        with open(os.path.join(d, "run2.jsonl"), "w") as f:
            f.write("")
        try:
            load_series(d)
            print("SELF-TEST FAIL empty-run")
            ok = False
        except AggError:
            pass
    with tempfile.TemporaryDirectory() as d:
        _gen_fixture(d)
        os.remove(os.path.join(d, GATE_FILE))
        try:
            load_series(d)
            print("SELF-TEST FAIL gate-missing")
            ok = False
        except AggError:
            pass
    with tempfile.TemporaryDirectory() as d:
        _gen_fixture(d)
        os.remove(os.path.join(d, ENV_FILE))
        try:
            load_series(d)
            print("SELF-TEST FAIL env-missing")
            ok = False
        except AggError:
            pass
    print("self-test:", "ok" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--self-test":
        sys.exit(self_test())
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    try:
        report(load_series(sys.argv[1]))
    except AggError as e:
        print(f"集計エラー: {e}", file=sys.stderr)
        sys.exit(1)
