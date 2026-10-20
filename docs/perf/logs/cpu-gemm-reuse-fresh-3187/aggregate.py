#!/usr/bin/env python3
"""イシュー #3187: 独立 5 run の JSONL を RULE.txt の規則で集計する（標準ライブラリのみ）。

使い方: aggregate.py <dir>   （dir は run1.jsonl..run5.jsonl と load_gate_status.txt を持つ）
        aggregate.py --self-test
欠落セル・不正 JSON・n_samples != 80・非有限/非正の median_s・checksum の bit 不一致・
ゲート状態の欠落は fail-closed（exit 1）。判定閾値は RULE.txt（実測前に固定）が正で、
本スクリプトは変更しない。
"""
import json
import math
import os
import statistics
import sys
import tempfile

RUNS = 5
SIZES = [256, 512, 1024, 2048]
FRESH_PHASES = ["tape_build", "var_upload", "matmul", "host_copy", "checksum", "iter_total"]
REUSE_PHASES = ["matmul", "host_copy", "checksum", "iter_total"]
EXPECTED = {
    "fresh": FRESH_PHASES + ["tape_drop"],
    "fresh_hold": FRESH_PHASES,
    "fresh_arena": FRESH_PHASES + ["tape_drop"],
    "fresh_legacy": FRESH_PHASES + ["tape_drop"],
    "reuse": REUSE_PHASES,
    "reuse_arena": REUSE_PHASES,
    "reuse_legacy": REUSE_PHASES,
    "backend_drop": ["kernel_call", "iter_total"],
    "backend_hold": ["kernel_call", "iter_total"],
}
EXPECTED_N = 80  # 計測 20 反復 x 4 ラウンド（テスト側 MEASURE_ITERS x ROUNDS）
GATE_FILE = "load_gate_status.txt"
ATTRIB_THRESHOLD = 0.5  # RULE.txt 帰属規則（事後に変えない）


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
                if isinstance(ns, bool) or ns != EXPECTED_N:
                    raise AggError(f"{path}:{i}: n_samples={ns!r} が {EXPECTED_N} でない {key}")
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


def load_gate(d):
    """(未通過 run 名の一覧, record_only の run 名の一覧) を返す（欠落・不正は fail-closed）。"""
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
    if set(seen) != {f"run{n}" for n in range(1, RUNS + 1)}:
        raise AggError(f"{path}: run1..run{RUNS} の全行が必要 ({sorted(seen)})")
    return (sorted(r for r, st in seen.items() if st == "unpassed"),
            sorted(r for r, st in seen.items() if st == "record_only"))


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


def arena_hits(runs, arm, n):
    """5 run 合算の arena (hit, miss, recycled)。統計欠落は None。"""
    tot = [0, 0, 0]
    for r in runs:
        st = r[(arm, n, "iter_total")].get("arena_stats")
        if not isinstance(st, dict):
            return None
        for i, k in enumerate(("hit", "miss", "recycled")):
            tot[i] += int(st[k])
    return tuple(tot)


def read_pagesize(d):
    """env_info.txt の `pagesize: N` を返す（欠落・不正・unknown は None＝取得不能）。"""
    try:
        with open(os.path.join(d, "env_info.txt"), encoding="utf-8") as f:
            for ln in f:
                if ln.startswith("pagesize:"):
                    v = ln.split(":", 1)[1].strip()
                    return int(v) if v.isdigit() and int(v) > 0 else None
    except OSError:
        pass
    return None


def minflt_section(runs, d):
    """補助系列の minflt/iter 表（判定外）。N^2*4/PAGESIZE と並べる。取得不能は明示する。"""
    ps = read_pagesize(d)
    lines = ["## ページ確保の機構（補助・判定外）: minflt / iter", ""]
    if ps is None:
        lines.append("pagesize: 取得不能（env_info.txt に有効な pagesize 行なし）。N^2*4/PAGESIZE は算出しない。")
    else:
        lines.append(f"pagesize = {ps} B。基準ページ数 = N^2*4/PAGESIZE（f32 の N x N 出力 1 枚分）。")
    lines += ["", "| N | arm | phase | minflt/iter (run 中央値) | N^2*4/PAGESIZE | 比 |", "|---|---|---|---|---|---|"]
    missing = 0
    for n in SIZES:
        pages = (n * n * 4 / ps) if ps else None
        for arm, phases in EXPECTED.items():
            for ph in phases:
                vals = []
                for r in runs:
                    v = r[(arm, n, ph)].get("minflt_delta_per_iter")
                    if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
                        vals.append(float(v))
                if not vals:
                    missing += 1
                    lines.append(f"| {n} | {arm} | {ph} | 取得不能 | {f'{pages:.1f}' if pages else '取得不能'} | - |")
                    continue
                m = statistics.median(vals)
                ratio = f"{m / pages:.2f}" if pages else "-"
                lines.append(f"| {n} | {arm} | {ph} | {m:.1f} | {f'{pages:.1f}' if pages else '取得不能'} | {ratio} |")
    if missing:
        lines += ["", f"注意: minflt を取得できなかったセルが {missing} 件（取得不能と表示）。"]
    lines.append("")
    return lines


def us(x):
    return f"{x * 1e6:.1f}"


def aggregate(d):
    runs = [load_run(os.path.join(d, f"run{i}.jsonl")) for i in range(1, RUNS + 1)]
    unpassed, record_only = load_gate(d)
    # record_only（M4 Max 疎通確認）は RULE.txt で判定対象外。全 run 一致のときだけ数値記録モード、
    # 混在は系列の性質が不明なため fail-closed。
    if record_only and len(record_only) != RUNS:
        raise AggError(f"record_only と他状態の run が混在 ({record_only})")
    # GEMM_REUSE_MINFLT=1 の補助系列は判定外（RULE.txt「ページ確保の機構」）。正式系列は全レコード OFF。
    flags = {rec["minflt_enabled"] for r in runs for rec in r.values()}
    if len(flags) > 1:
        raise AggError("minflt_enabled が true/false 混在")
    minflt_series = flags == {True}
    # 補助系列は正式判定へ入れず、記録専用処理（数値表のみ）へ分離する
    judge = not (record_only or minflt_series)
    check_checksums(runs)
    out = []
    w = out.append
    w(f"# 集計（#3187）: {os.path.basename(os.path.abspath(d))}")
    w("")
    if unpassed:
        w(f"**注意: 負荷ゲート未通過の run あり（{', '.join(unpassed)}）。系列は参考扱い。**")
        w("")
    if not judge:
        why = "GEMM_REUSE_MINFLT=1 の補助系列" if minflt_series else "record_only（疎通確認）系列"
        w(f"**{why}: 数値記録のみ。再現判定・帰属・因果・arena 判定は行わない（RULE.txt: 判定対象外）。**")
        w("")
    w("checksum: 全 n・全腕・全 run で iter_total の checksum_bits が一致（hard 検査通過）。")
    w("")
    w("## 判定表（5 run 中央値。単位 us）" if judge else "## 記録表（判定なし。5 run 中央値。単位 us）")
    w("")
    for n in SIZES:
        f_tot = med(runs, "fresh", n, "iter_total")
        r_tot = med(runs, "reuse", n, "iter_total")
        ratio = r_tot / f_tot
        w(f"### N={n}")
        w("")
        w(f"- fresh.iter_total = {us(f_tot)}, reuse.iter_total = {us(r_tot)}, ratio = {ratio:.3f}")
        # 残差 = iter_total - sum(区間)（RULE.txt）。区間外の時間が総時間差を説明していないか確認する
        res_f = f_tot - sum(med(runs, "fresh", n, ph) for ph in FRESH_PHASES[:-1])
        res_r = r_tot - sum(med(runs, "reuse", n, ph) for ph in REUSE_PHASES[:-1])
        w(f"- 残差 (iter_total - sum(区間)): fresh = {us(res_f)}, reuse = {us(res_r)}, "
          f"reuse - fresh = {us(res_r - res_f)}（総時間差 {us(r_tot - f_tot)}）")
        if not judge:
            w("")
            continue
        if ratio <= 1.0:
            w("- 判定: 非再現（ratio <= 1.00）。帰属は行わない。")
            w("")
            continue
        w("- 判定: 逆転が再現（ratio > 1.00）")
        d_mm = med(runs, "reuse", n, "matmul") - med(runs, "fresh", n, "matmul")
        d_ro = med(runs, "reuse", n, "host_copy") - med(runs, "fresh", n, "host_copy")
        d_ck = med(runs, "reuse", n, "checksum") - med(runs, "fresh", n, "checksum")
        g = d_mm + d_ro + d_ck
        f_only = med(runs, "fresh", n, "tape_build") + med(runs, "fresh", n, "var_upload")
        w(f"- D_matmul = {us(d_mm)}, D_readout = {us(d_ro)}, D_checksum = {us(d_ck)}, G = {us(g)}")
        w(f"- F_only (tape_build + var_upload) = {us(f_only)}")
        if g <= 0:
            w("- 帰属: 共有区間では説明できない（未確定）")
        else:
            names = {"matmul": ("出力確保（結果蓄積・ページ確保）", d_mm),
                     "host_copy": ("readout 方式", d_ro),
                     "checksum": ("読み出し側", d_ck)}
            sup = [f"{ph}→{lab}（{v / g:.0%}）" for ph, (lab, v) in names.items() if v / g >= ATTRIB_THRESHOLD]
            w("- 帰属: " + ("支持 " + "; ".join(sup) if sup else "未確定（どの区間も 50% 未満）"))
        if d_mm > 0:
            rho_fh = (med(runs, "fresh_hold", n, "matmul") - med(runs, "fresh", n, "matmul")) / d_mm
            rho_b = (med(runs, "backend_hold", n, "kernel_call") - med(runs, "backend_drop", n, "kernel_call")) / d_mm
            w(f"- rho_FH = {rho_fh:.2f}（{'結果蓄積が原因を支持' if rho_fh >= 0.5 else '未確定'}）")
            w(f"- rho_B = {rho_b:.2f}（{'出力バッファの確保パターンが原因を支持' if rho_b >= 0.5 else '未確定'}）")
        else:
            w("- rho_FH / rho_B: D_matmul <= 0 のため未確定")
        fa = arena_hits(runs, "fresh_arena", n)
        ra = arena_hits(runs, "reuse_arena", n)
        if fa is None or ra is None or fa[0] <= 0:
            w(f"- arena A/B: undetermined（fresh_arena hit={fa}, reuse_arena={ra}）")
        else:
            ra_tot = med(runs, "reuse_arena", n, "iter_total")
            gap = r_tot - f_tot
            fixed = abs(ra_tot - r_tot) >= ATTRIB_THRESHOLD * gap
            w(f"- arena A/B: fresh_arena hit={fa[0]}, reuse_arena hit={ra[0]}（予測: 0）, "
              f"reuse_arena.iter_total = {us(ra_tot)}")
            w("  - " + ("host arena で差の 50% 以上が動く" if fixed
                        else "既存の host arena (#2104) では reuse の逆転を是正できない"))
        w("")
    if minflt_series:
        out.extend(minflt_section(runs, d))
    w("## 区間表（5 run 中央値 [min–max]、us）")
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
    w("## run 別 iter_total（fresh / reuse、us）")
    w("")
    w("| N | arm | " + " | ".join(f"run{i}" for i in range(1, RUNS + 1)) + " |")
    w("|---|---|" + "---|" * RUNS)
    for n in SIZES:
        for arm in ("fresh", "reuse"):
            w(f"| {n} | {arm} | " + " | ".join(us(r[(arm, n, 'iter_total')]['median_s']) for r in runs) + " |")
    w("")
    return "\n".join(out)


def _rec(arm, n, ph, med_s, bits="3ff0000000000000", ns=EXPECTED_N, arena=None):
    return {"arm": arm, "n": n, "phase": ph, "median_s": med_s, "n_samples": ns, "minflt_enabled": False,
            "checksum_bits": bits if ph == "iter_total" else None, "arena_stats": arena}


def _write_series(d, mutate=None, gate=True, gate_status="pass"):
    os.makedirs(d, exist_ok=True)
    for r in range(1, RUNS + 1):
        lines = []
        for arm, phases in EXPECTED.items():
            for n in SIZES:
                for ph in phases:
                    v = 1e-3 * (1.5 if arm.startswith("reuse") else 1.0)
                    arena = None
                    if arm.endswith("_arena") and ph == "iter_total":
                        arena = {"hit": 10 if arm == "fresh_arena" else 0, "miss": 5, "recycled": 5, "rejected": 0}
                    lines.append(_rec(arm, n, ph, v, arena=arena))
        if mutate:
            lines = mutate(r, lines)
        with open(os.path.join(d, f"run{r}.jsonl"), "w", encoding="utf-8") as f:
            for rec in lines:
                f.write(json.dumps(rec) + "\n")
    if gate:
        with open(os.path.join(d, GATE_FILE), "w", encoding="utf-8") as f:
            for r in range(1, RUNS + 1):
                f.write(f"run{r} gate={gate_status}\n")


def self_test():
    ok = True
    with tempfile.TemporaryDirectory() as t:
        good = os.path.join(t, "good")
        _write_series(good)
        text = aggregate(good)
        assert "逆転が再現" in text and "hard 検査通過" in text and "残差" in text

        ro = os.path.join(t, "ro")
        _write_series(ro, gate_status="record_only")
        text = aggregate(ro)
        assert "判定: 逆転が再現" not in text and "- 帰属" not in text and "数値記録のみ" in text

        def set_minflt(val):
            def m(r, lines):
                for rec in lines:
                    rec["minflt_enabled"] = val(r)
                return lines
            return m

        mf = os.path.join(t, "mf")
        _write_series(mf, set_minflt(lambda r: True))
        text = aggregate(mf)
        assert "判定: 逆転が再現" not in text and "MINFLT" in text
        assert "minflt / iter" in text and "取得不能" in text

        def expect_fail(name, mutate=None, gate=True, drop_run=False):
            nonlocal ok
            d = os.path.join(t, name)
            _write_series(d, mutate, gate)
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
                    if rec["arm"] == "reuse" and rec["n"] == 512 and rec["phase"] == "iter_total":
                        rec["checksum_bits"] = "3ff0000000000001"
            return lines

        def bad_n(r, lines):
            lines[0]["n_samples"] = 79
            return lines

        def bad_median(r, lines):
            lines[0]["median_s"] = float("nan") if r == 1 else 0.0
            return lines

        def missing_cell(r, lines):
            return lines[1:] if r == 4 else lines

        expect_fail("bits", bad_bits)
        expect_fail("n79", bad_n)
        expect_fail("median", bad_median)
        expect_fail("cell", missing_cell)
        expect_fail("nogate", gate=False)
        expect_fail("minflt_mix", set_minflt(lambda r: r == 2))
        expect_fail("norun", drop_run=True)
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
