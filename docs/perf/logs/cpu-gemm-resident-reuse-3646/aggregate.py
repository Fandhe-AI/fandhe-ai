#!/usr/bin/env python3
"""イシュー #3646: 独立 5 run（補助系列 minflt のみ 1 run）の JSONL を RULE.txt の規則で集計する（標準ライブラリのみ）。

使い方: aggregate.py <dir>   （dir は run1.jsonl.. と load_gate_status.txt・series_kind.txt を持つ）
        aggregate.py --self-test
欠落セル・不正 JSON・n_samples 不一致・非有限/非正の median_s・checksum の bit 不一致・
ゲート状態／系列種別の欠落は fail-closed（exit 1）。
ブリッジ確認結果（bridge_status.txt）が未記録・不成立の N は帰属を出さず保留する。判定閾値は RULE.txt（実測前に固定）が正で、
本スクリプトは変更しない。判定は series_kind=main かつ gate=pass の系列だけで行う。
"""
import json
import math
import os
import statistics
import sys
import tempfile

RUNS = 5
SIZES = [256, 512, 1024, 2048, 4096]
EXPECTED = {
    "fresh": ["tape_build", "var_upload", "matmul", "readout", "checksum", "iter_total", "tape_drop"],
    "resident_public": ["matmul", "readout", "checksum", "iter_total"],
    "resident_reupload": ["matmul", "readout", "checksum", "iter_total"],
    "resident_split": ["gemm_device", "download", "out_drop", "readout", "checksum", "iter_total"],
    "gemm_device_hold": ["iter_total"],
    "gemm_device_drop_in": ["iter_total"],
    "gemm_device_download": ["iter_total"],
    "backend_drop": ["kernel_call", "iter_total"],
}
# 計測反復 x 4 ラウンド（テスト側 iters_for x ROUNDS。N=4096 だけ 10 反復）
EXPECTED_N = {n: (40 if n >= 4096 else 80) for n in SIZES}
GATE_FILE = "load_gate_status.txt"
KIND_FILE = "series_kind.txt"
BRIDGE_FILE = "bridge_status.txt"
KINDS = ("main", "minflt", "tunables", "taskset")
THRESHOLD = 0.5  # RULE.txt 帰属規則（事後に変えない）
TUNABLES_NEAR_ONE = 1.05  # RULE.txt 補助系列の解釈規則


class AggError(Exception):
    pass


def load_run(path):
    cells = {}
    try:
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    key = (rec["arm"], rec["n"], rec["phase"])
                    rec["median_s"] = float(rec["median_s"])
                    ns = rec["n_samples"]
                    mf = rec["minflt_enabled"]
                except (ValueError, KeyError, TypeError) as e:
                    raise AggError(f"{path}:{i}: 不正な行 ({e})")
                if key[0] not in EXPECTED or key[2] not in EXPECTED[key[0]] or key[1] not in SIZES:
                    raise AggError(f"{path}:{i}: 想定外のセル {key}")
                if not math.isfinite(rec["median_s"]) or rec["median_s"] <= 0.0:
                    raise AggError(f"{path}:{i}: median_s が有限の正値でない {key}")
                if isinstance(ns, bool) or ns != EXPECTED_N[key[1]]:
                    raise AggError(f"{path}:{i}: n_samples={ns!r} が {EXPECTED_N[key[1]]} でない {key}")
                if not isinstance(mf, bool):
                    raise AggError(f"{path}:{i}: minflt_enabled が bool でない {key}")
                if key in cells:
                    raise AggError(f"{path}:{i}: セル重複 {key}")
                if key[2] == "iter_total" and not isinstance(rec.get("checksum_bits"), str):
                    raise AggError(f"{path}:{i}: iter_total に checksum_bits が無い {key}")
                cells[key] = rec
    except OSError as e:
        raise AggError(f"{path}: 読み込み不可 ({e})")
    for arm, phases in EXPECTED.items():
        for n in SIZES:
            for ph in phases:
                if (arm, n, ph) not in cells:
                    raise AggError(f"{path}: セル欠落 {arm}/{n}/{ph}")
    return cells


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


def load_bridge(d):
    """N -> ブリッジ確認結果（pass / fail）。ファイル欠落は空 dict（帰属保留）、書式不正は fail-closed。

    RULE.txt「ブリッジ」: --phases の matmul 区間差が iter_total 差の 50% 以上かを N ごとに
    確認した結果を `<N> bridge=pass|fail` で 1 行ずつ記録する。未記録・fail の N は帰属を保留する。
    """
    path = os.path.join(d, BRIDGE_FILE)
    if not os.path.exists(path):
        return {}
    seen = {}
    try:
        with open(path, encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
    except OSError as e:
        raise AggError(f"{path}: 読み込み不可 ({e})")
    for ln in lines:
        parts = ln.split()
        if (len(parts) != 2 or not parts[0].isdigit() or int(parts[0]) not in SIZES
                or parts[1] not in ("bridge=pass", "bridge=fail") or int(parts[0]) in seen):
            raise AggError(f"{path}: 不正または重複した行 {ln!r}")
        seen[int(parts[0])] = parts[1][7:]
    return seen


def check_checksums(runs):
    for n in SIZES:
        ref = None
        for ri, cells in enumerate(runs, 1):
            for arm in EXPECTED:
                b = cells[(arm, n, "iter_total")]["checksum_bits"]
                if ref is None:
                    ref = b
                elif b != ref:
                    raise AggError(f"checksum bit 不一致: n={n} run{ri} {arm} {b} != {ref}")


def med(runs, arm, n, ph):
    return statistics.median(r[(arm, n, ph)]["median_s"] for r in runs)


def rng(runs, arm, n, ph):
    v = [r[(arm, n, ph)]["median_s"] for r in runs]
    return min(v), max(v)


def us(x):
    return f"{x * 1e6:.1f}"


def rho_line(name, num, g, support_text):
    r = num / g if g > 0 else float("nan")
    return f"- {name} = {r:.2f}（{support_text if r >= THRESHOLD else '未達'}）"


def minflt_section(runs):
    lines = ["## minflt / iter（補助・判定外）", "",
             "| N | arm | phase | minflt/iter |", "|---|---|---|---|"]
    for n in SIZES:
        for arm, phases in EXPECTED.items():
            for ph in phases:
                v = runs[0][(arm, n, ph)].get("minflt_delta_per_iter")
                ok = isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                lines.append(f"| {n} | {arm} | {ph} | {f'{v:.1f}' if ok else '取得不能'} |")
    lines.append("")
    return lines


def aggregate(d):
    kind = load_kind(d)
    nruns = 1 if kind == "minflt" else RUNS
    runs = [load_run(os.path.join(d, f"run{i}.jsonl")) for i in range(1, nruns + 1)]
    if nruns == 1 and os.path.exists(os.path.join(d, "run2.jsonl")):
        raise AggError("minflt 補助系列は run1 のみ（run2 以降が存在する）")
    flags = {rec["minflt_enabled"] for r in runs for rec in r.values()}
    if flags != {kind == "minflt"}:
        raise AggError(f"minflt_enabled が系列種別 {kind} と整合しない ({sorted(flags)})")
    gate = load_gate(d, nruns)
    unpassed = sorted(r for r, s in gate.items() if s == "unpassed")
    record_only = sorted(r for r, s in gate.items() if s == "record_only")
    if record_only and len(record_only) != nruns:
        raise AggError(f"record_only と他状態の run が混在 ({record_only})")
    check_checksums(runs)
    bridge = load_bridge(d)
    judge = kind == "main" and not unpassed and not record_only
    out = []
    w = out.append
    w(f"# 集計（#3646）: {os.path.basename(os.path.abspath(d))}（系列種別 {kind}）")
    w("")
    if unpassed:
        w(f"**注意: 負荷ゲート未通過の run あり（{', '.join(unpassed)}）。系列は参考扱い（判定しない）。**")
        w("")
    if not judge:
        w("**判定対象外の系列: 数値記録のみ（RULE.txt）。帰属判定は行わない。**")
        w("")
    w("checksum: 全 n・全腕・全 run で iter_total の checksum_bits が一致（hard 検査通過）。")
    w("")
    w(f"## 判定表（{nruns} run 中央値。単位 us）")
    w("")
    for n in SIZES:
        f_tot = med(runs, "fresh", n, "iter_total")
        r_tot = med(runs, "resident_public", n, "iter_total")
        ratio = r_tot / f_tot
        w(f"### N={n}")
        w("")
        w(f"- fresh.iter_total = {us(f_tot)}, resident_public.iter_total = {us(r_tot)}, ratio = {ratio:.3f}")
        sp = med(runs, "resident_split", n, "iter_total")
        sp_sum = sum(med(runs, "resident_split", n, ph) for ph in EXPECTED["resident_split"][:-1])
        w(f"- resident_split: 区間和 {us(sp_sum)} / iter_total {us(sp)}（残差 {us(sp - sp_sum)}。記録のみ）")
        if kind == "tunables":
            if ratio <= TUNABLES_NEAR_ONE:
                w(f"- tunables 解釈: ratio <= {TUNABLES_NEAR_ONE}。機構は glibc の確保とページ返却を支持（RULE.txt）。")
            else:
                w(f"- tunables 解釈: ratio > {TUNABLES_NEAR_ONE}。glibc 確保機構だけでは説明できない。")
        if not judge:
            w("")
            continue
        if ratio <= 1.0:
            w("- 判定: 非再現（ratio <= 1.00）。帰属は行わない。")
            w("")
            continue
        if bridge.get(n) != "pass":
            why = "未記録" if n not in bridge else "不成立"
            w(f"- 判定: 再現（ratio > 1.00）。ただしブリッジ確認が{why}（{BRIDGE_FILE}）。"
              "bench と診断の窓の一致が未確認のため帰属を保留する（RULE.txt）。")
            w("")
            continue
        g = r_tot - f_tot
        h_hold = med(runs, "gemm_device_hold", n, "iter_total")
        h_dl = med(runs, "gemm_device_download", n, "iter_total")
        h_drop = med(runs, "gemm_device_drop_in", n, "iter_total")
        k_call = med(runs, "backend_drop", n, "kernel_call")
        reup = med(runs, "resident_reupload", n, "iter_total")
        w(f"- 判定: 再現（ratio > 1.00）。G = resident_public - fresh = {us(g)}")
        w(f"- gemm_device_hold = {us(h_hold)}, +download = {us(h_dl)}, drop_in = {us(h_drop)}, "
          f"backend_drop.kernel_call = {us(k_call)}, resident_reupload = {us(reup)}")
        parts = (("rho_DL", h_dl - h_hold, "H1 支持（download の追加確保＋コピー）"),
                 ("rho_DROP", h_drop - h_hold, "H1b 支持（出力の窓内解放）"),
                 ("rho_K", h_hold - k_call, "gemm_device 内の確保かカーネルを支持"),
                 ("rho_IN", abs(reup - r_tot), "H2・H3 支持（入力の寿命・配置）"))
        for name, num, lab in parts:
            w(rho_line(name, num, g, lab))
        sup = [name for name, num, _ in parts if g > 0 and num / g >= THRESHOLD]
        w("- 帰属: " + ("支持 " + ", ".join(sup) if sup else "未確定（どの rho も 0.5 未達）"))
        w("")
    if kind == "minflt":
        out.extend(minflt_section(runs))
    w(f"## 区間表（{nruns} run 中央値 [min–max]、us）")
    w("")
    for n in SIZES:
        w(f"### N={n}")
        w("")
        w("| arm | phase | median [min–max] |")
        w("|---|---|---|")
        for arm, phases in EXPECTED.items():
            for ph in phases:
                lo, hi = rng(runs, arm, n, ph)
                w(f"| {arm} | {ph} | {us(med(runs, arm, n, ph))} [{us(lo)}–{us(hi)}] |")
        w("")
    w("## run 別 iter_total（fresh / resident_public、us）")
    w("")
    w("| N | arm | " + " | ".join(f"run{i}" for i in range(1, nruns + 1)) + " |")
    w("|---|---|" + "---|" * nruns)
    for n in SIZES:
        for arm in ("fresh", "resident_public"):
            w(f"| {n} | {arm} | " + " | ".join(us(r[(arm, n, 'iter_total')]['median_s']) for r in runs) + " |")
    w("")
    return "\n".join(out)


def _rec(arm, n, ph, med_s, minflt=False):
    return {"arm": arm, "n": n, "phase": ph, "median_s": med_s, "n_samples": EXPECTED_N[n],
            "minflt_enabled": minflt, "checksum_bits": "3ff0000000000000" if ph == "iter_total" else None,
            "minflt_delta_per_iter": None}


def _write_series(d, mutate=None, kind="main", gate=True, gate_status="pass", nruns=RUNS):
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, KIND_FILE), "w", encoding="utf-8") as f:
        f.write(kind + "\n")
    for r in range(1, nruns + 1):
        lines = []
        for arm, phases in EXPECTED.items():
            for n in SIZES:
                for ph in phases:
                    v = 1e-3 * (1.5 if arm == "resident_public" else 1.0)
                    lines.append(_rec(arm, n, ph, v, minflt=(kind == "minflt")))
        if mutate:
            lines = mutate(r, lines)
        with open(os.path.join(d, f"run{r}.jsonl"), "w", encoding="utf-8") as f:
            for rec in lines:
                f.write(json.dumps(rec) + "\n")
    if gate:
        with open(os.path.join(d, GATE_FILE), "w", encoding="utf-8") as f:
            for r in range(1, nruns + 1):
                f.write(f"run{r} gate={gate_status}\n")


def self_test():
    ok = True
    with tempfile.TemporaryDirectory() as t:
        good = os.path.join(t, "good")
        _write_series(good)
        with open(os.path.join(good, BRIDGE_FILE), "w", encoding="utf-8") as f:
            f.writelines(f"{n} bridge=pass\n" for n in SIZES)
        text = aggregate(good)
        assert "再現（ratio > 1.00）" in text and "hard 検査通過" in text and "rho_DL" in text
        nb = os.path.join(t, "nobridge")
        _write_series(nb)
        text = aggregate(nb)
        assert "帰属を保留" in text and "- 帰属" not in text and "rho_DL" not in text
        bf = os.path.join(t, "bridgefail")
        _write_series(bf)
        with open(os.path.join(bf, BRIDGE_FILE), "w", encoding="utf-8") as f:
            f.writelines(f"{n} bridge=fail\n" for n in SIZES)
        text = aggregate(bf)
        assert "ブリッジ確認が不成立" in text and "- 帰属" not in text
        ro = os.path.join(t, "ro")
        _write_series(ro, gate_status="record_only")
        text = aggregate(ro)
        assert "- 帰属" not in text and "判定対象外" in text
        tu = os.path.join(t, "tu")
        _write_series(tu, kind="tunables")
        assert "tunables 解釈" in aggregate(tu)
        mf = os.path.join(t, "mf")
        _write_series(mf, kind="minflt", nruns=1)
        text = aggregate(mf)
        assert "minflt / iter" in text and "取得不能" in text and "- 帰属" not in text

        def expect_fail(name, mutate=None, gate=True, drop_run=False, kind="main"):
            nonlocal ok
            d = os.path.join(t, name)
            _write_series(d, mutate, kind=kind, gate=gate)
            if drop_run:
                os.remove(os.path.join(d, "run3.jsonl"))
            try:
                aggregate(d)
            except AggError:
                return
            print(f"self-test 失敗: {name} が fail-closed にならない", file=sys.stderr)
            ok = False

        def bad_bits(r, lines):
            if r == 2:
                for rec in lines:
                    if rec["arm"] == "resident_public" and rec["n"] == 512 and rec["phase"] == "iter_total":
                        rec["checksum_bits"] = "3ff0000000000001"
            return lines

        def bad_n(r, lines):
            lines[0]["n_samples"] = 79
            return lines

        def bad_n4096(r, lines):
            for rec in lines:
                if rec["n"] == 4096:
                    rec["n_samples"] = 80
                    break
            return lines

        def bad_median(r, lines):
            lines[0]["median_s"] = float("nan") if r == 1 else 0.0
            return lines

        expect_fail("bits", bad_bits)
        expect_fail("n79", bad_n)
        expect_fail("n4096", bad_n4096)
        expect_fail("median", bad_median)
        expect_fail("cell", lambda r, lines: lines[1:] if r == 4 else lines)
        expect_fail("nogate", gate=False)
        expect_fail("norun", drop_run=True)
        expect_fail("kind_mismatch", kind="minflt")
    if not ok:
        sys.exit(1)
    print("self-test OK")


def main(argv):
    if len(argv) == 2 and argv[1] == "--self-test":
        self_test()
        return 0
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    try:
        print(aggregate(argv[1]))
    except AggError as e:
        print(f"集計失敗（fail-closed）: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
