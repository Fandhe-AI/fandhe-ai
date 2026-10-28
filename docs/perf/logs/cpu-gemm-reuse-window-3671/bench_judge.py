#!/usr/bin/env python3
"""イシュー #3671: 診断側の腕と bench 側の腕を合わせ、RULE.txt／RULE.v2.txt の判定を出す（標準ライブラリのみ）。

役割: aggregate.py（診断系列の検証・集計。importlib で同ディレクトリから読み込む）と、bench-fandhe の
bench_base／bench_tunables／bench_headpatch の run を読み、N=256〜2048 ごとに次を出す。
  - 基準の再確認（RULE.v2）: bridge_v2 = G_bench>0 かつ G_diag>0 かつ 0.5<=G_bench/G_diag<=2.0。fail の N が「再現」
  - 診断側の腕 X（verify_touch／verify_gap／hold_footprint／ref_compute／solo）: G_bench>0 かつ G_X>0 かつ
    0.5<=G_bench/G_X<=2.0 なら支持
  - bench 側の腕 Y（tunables／headpatch）: G_Y>0 かつ G_diag>0 かつ 0.5<=G_Y/G_diag<=2.0 なら支持
  - 記録のみ: matmul 区間/G_bench・X.iter_total/bench reuse・E(N)・N=4096（対照・判定外）
閾値は RULE.v2（#3646 ブリッジ v2 と同じ 0.5〜2.0）が正で、本スクリプトは新しい閾値を作らない。

使い方: bench_judge.py --diag <dir> --bench-root <dir> [--solo <dir>] [--excl <excl_status.txt>] [--gate-log <file>]
        bench_judge.py --self-test
bench-root の構成: <root>/{base,tunables,headpatch}/gemm_{fresh,reuse,phases}_n<N>_r<1..5>.jsonl と各 arm の rc.log
（"<mode> n=<N> r=<R> rc=<rc>" 行を 75 件・全て rc=0）。gate-log は "bench_<arm>/n<N>/r<R> gate=pass" 行を要求する。
データ不備（欠落・parity fail・rc 非 0・checksum 不一致・不正 JSON）は exit 1。専有・ゲート・dirty が未確認の系列は
「参考記録」と出力し、判定を正式値として扱わない（exit 0）。
"""
import importlib.util
import json
import math
import os
import statistics
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("aggregate_3671", os.path.join(HERE, "aggregate.py"))
agg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(agg)

JUDGED = [256, 512, 1024, 2048]
CONTROL = 4096
LO, HI = 0.5, 2.0  # RULE.v2（#3646 ブリッジ v2）。新しい閾値は作らない
BENCH_ARMS = ["base", "tunables", "headpatch"]
MODES = ["fresh", "reuse", "phases"]
RUNS = 5
DIAG_X = ["rp_verify_touch", "rp_verify_gap", "rp_hold_footprint", "rp_ref_compute"]


class JudgeError(Exception):
    pass


def _num(v, what):
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise JudgeError(f"{what}: 数値でない ({v!r})")
    if not math.isfinite(x) or x <= 0.0:
        raise JudgeError(f"{what}: 有限の正値でない ({v!r})")
    return x


def _read_rows(path):
    try:
        with open(path, encoding="utf-8") as f:
            lines = [ln for ln in f.read().splitlines() if ln.strip()]
    except OSError as e:
        raise JudgeError(f"{path}: 読み込み不可 ({e})")
    if not lines:
        raise JudgeError(f"{path}: 空（実行失敗の記録）")
    rows = []
    for i, ln in enumerate(lines, 1):
        try:
            rows.append(json.loads(ln))
        except ValueError as e:
            raise JudgeError(f"{path}:{i}: 不正な JSON ({e})")
    return rows


def _check_parity(row, path, n):
    if row.get("size") != n:
        raise JudgeError(f"{path}: size={row.get('size')!r} が {n} でない")
    pf = row.get("parity_fail_count")
    if isinstance(pf, bool) or not isinstance(pf, int) or pf != 0:
        raise JudgeError(f"{path}: parity_fail_count={pf!r}（0 でない。系列無効）")
    _num(row.get("median_s"), f"{path} median_s")


def load_bench_arm(root, arm):
    """arm -> {n: {"fresh": [..5], "reuse": [..5], "phases": {phase: [..5]}}} と checksum 集合（N 別）。"""
    d = os.path.join(root, arm)
    rc_path = os.path.join(d, "rc.log")
    try:
        with open(rc_path, encoding="utf-8") as f:
            rc_lines = [ln.strip() for ln in f if ln.strip()]
    except OSError as e:
        raise JudgeError(f"{rc_path}: 読み込み不可 ({e})")
    want = len(agg.SIZES) * len(MODES) * RUNS
    if len(rc_lines) != want or any(not ln.endswith(" rc=0") for ln in rc_lines):
        raise JudgeError(f"{rc_path}: {want} 件の rc=0 が必要（実際 {len(rc_lines)} 件／非 0 を含む）")
    out, sums = {}, {}
    for n in agg.SIZES:
        cell = {"fresh": [], "reuse": [], "phases": {}}
        for r in range(1, RUNS + 1):
            for m in MODES:
                path = os.path.join(d, f"gemm_{m}_n{n}_r{r}.jsonl")
                rows = _read_rows(path)
                for row in rows:
                    _check_parity(row, path, n)
                    sums.setdefault(n, set()).add(str(row.get("checksum")))
                if m in ("fresh", "reuse"):
                    if len(rows) != 1 or rows[0].get("task") != "gemm" or rows[0].get("mode") != m:
                        raise JudgeError(f"{path}: gemm {m} の 1 行ではない")
                    cell[m].append(_num(rows[0]["median_s"], path))
                else:
                    seen = {}
                    for row in rows:
                        if row.get("task") != "gemm_phases" or "phase" not in row or row["phase"] in seen:
                            raise JudgeError(f"{path}: gemm_phases の行が不正または重複")
                        seen[row["phase"]] = _num(row["median_s"], path)
                    for need in ("iter_total", "matmul"):
                        if need not in seen:
                            raise JudgeError(f"{path}: phase={need} が無い")
                    for ph, v in seen.items():
                        cell["phases"].setdefault(ph, []).append(v)
        for ph, vs in cell["phases"].items():
            if len(vs) != RUNS:
                raise JudgeError(f"{arm} n={n}: phase={ph} が {len(vs)} run（{RUNS} 必要）")
        out[n] = cell
    return out, sums


def load_gate_log(path, arms):
    try:
        with open(path, encoding="utf-8") as f:
            seen = {}
            for ln in f:
                parts = ln.split()
                if len(parts) == 2 and parts[1].startswith("gate="):
                    seen[parts[0]] = parts[1][5:]
    except OSError as e:
        raise JudgeError(f"{path}: 読み込み不可 ({e})")
    bad = []
    for a in arms:
        for n in agg.SIZES:
            for r in range(1, RUNS + 1):
                label = f"bench_{a}/n{n}/r{r}"
                if seen.get(label) != "pass":
                    bad.append(label)
    return bad


def load_excl(path):
    try:
        with open(path, encoding="utf-8") as f:
            st = {}
            for ln in f:
                parts = ln.split()
                if len(parts) == 2 and parts[1].startswith("excl="):
                    st[parts[0]] = parts[1][5:]
    except OSError as e:
        raise JudgeError(f"{path}: 読み込み不可 ({e})")
    return st


def med(vs):
    return statistics.median(vs)


def ratio_ok(num, den):
    return num > 0 and den > 0 and LO <= num / den <= HI


def judge(diag, solo, bench, excl, gate_bad):
    """判定本体。戻り値は (行のリスト, 参考理由リスト, 結論行)。"""
    ds = diag["summary"]
    notes = list(diag["why"])
    if solo is not None:
        notes += [f"solo: {w}" for w in solo["why"]]
    if gate_bad:
        notes.append(f"bench の負荷ゲート未通過または未記録: {len(gate_bad)} 件（例 {gate_bad[0]}）")
    need = ["diag"] + (["solo"] if solo is not None else []) + list(bench)
    if excl is None:
        notes.append("専有（excl_status.txt）が未確認")
    else:
        for lab in need:
            if excl.get(lab) != "pass":
                notes.append(f"専有 excl={excl.get(lab)!r}（{lab}）")
    lines = []
    supported = {n: [] for n in JUDGED}
    reproduced = {}
    for n in agg.SIZES:
        base = bench["base"][n]
        fresh_b = med(base["fresh"])
        reuse_b = med(base["phases"]["iter_total"])
        g_bench = reuse_b - fresh_b
        fresh_d = ds[("fresh", n, "iter_total")][0]
        rp = ds[("resident_public", n, "iter_total")][0]
        g_diag = rp - fresh_d
        bridge = ratio_ok(g_bench, g_diag)
        judged = n in JUDGED
        if judged:
            reproduced[n] = not bridge
        tag = "判定対象" if judged else "対照（判定外）"
        r_bd = f"{g_bench / g_diag:.3f}" if g_diag != 0 else "n/a"
        e_n = (reuse_b / rp) / (fresh_b / fresh_d)
        lines.append(f"N={n} [{tag}] fresh_bench={fresh_b:.9f} reuse_bench={reuse_b:.9f} G_bench={g_bench:+.9f} "
                     f"fresh_diag={fresh_d:.9f} resident_public={rp:.9f} G_diag={g_diag:+.9f} "
                     f"G_bench/G_diag={r_bd} bridge_v2={'pass' if bridge else 'fail'} E={e_n:.3f} "
                     f"matmul/G_bench={med(base['phases']['matmul']) / g_bench if g_bench > 0 else float('nan'):.3f}")
        xs = {x: ds[(x, n, "iter_total")][0] for x in DIAG_X}
        if solo is not None:
            xs["rp_solo"] = solo["summary"][("rp_solo", n, "iter_total")][0]
        for x, v in xs.items():
            g_x = v - fresh_d
            ok = ratio_ok(g_bench, g_x)
            note = "（G は主診断系列の fresh を使用）" if x == "rp_solo" else ""
            lines.append(f"  X {x}: iter_total={v:.9f} G_X={g_x:+.9f} G_bench/G_X="
                         f"{(g_bench / g_x) if g_x != 0 else float('nan'):.3f} 支持={'yes' if ok else 'no'}"
                         f" iter_total/bench_reuse={v / reuse_b:.3f}{note}")
            if judged and reproduced[n] and ok:
                supported[n].append(x)
        for y in ("tunables", "headpatch"):
            if y not in bench:
                continue
            c = bench[y][n]
            g_y = med(c["phases"]["iter_total"]) - med(c["fresh"])
            ok = ratio_ok(g_y, g_diag)
            lines.append(f"  Y bench_{y}: reuse={med(c['phases']['iter_total']):.9f} fresh={med(c['fresh']):.9f} "
                         f"G_Y={g_y:+.9f} G_Y/G_diag={(g_y / g_diag) if g_diag != 0 else float('nan'):.3f} "
                         f"支持={'yes' if ok else 'no'}")
            if judged and reproduced[n] and ok:
                supported[n].append("bench_" + y)
    concl = []
    for n in JUDGED:
        if not reproduced[n]:
            concl.append(f"N={n}: 現象が再現しない（bridge_v2=pass）。帰属は行わない")
            continue
        s = supported[n]
        if not s:
            concl.append(f"N={n}: 特定できず（どの腕も支持されない。各比は上記に記録）")
            continue
        extra = ""
        if "rp_verify_touch" in s and "rp_verify_gap" not in s:
            extra = "（記録: キャッシュとページへの影響）"
        elif "rp_verify_touch" in s and "rp_verify_gap" in s:
            extra = "（記録: 単一スレッドの空き時間）"
        concl.append(f"N={n}: 支持された腕 = {', '.join(s)}{extra}（交互作用は断定しない）")
    return lines, notes, concl


def run(args):
    diag = agg.load_series(args["diag"])
    if diag["kind"] != "main":
        raise JudgeError("--diag は series_kind=main の系列")
    solo = None
    if args.get("solo"):
        solo = agg.load_series(args["solo"])
        if solo["kind"] != "solo":
            raise JudgeError("--solo は series_kind=solo の系列")
    bench, sums = {}, {}
    for a in BENCH_ARMS:
        if os.path.isdir(os.path.join(args["bench_root"], a)):
            bench[a], sums[a] = load_bench_arm(args["bench_root"], a)
    if "base" not in bench:
        raise JudgeError("bench_base の系列が無い（基準の再確認ができない）")
    for n in agg.SIZES:
        allsum = set()
        for a in bench:
            allsum |= sums[a].get(n, set())
        if len(allsum) != 1:
            raise JudgeError(f"bench の checksum が N={n} で一致しない: {sorted(allsum)}")
    gate_bad = load_gate_log(args["gate_log"], list(bench)) if args.get("gate_log") else ["(gate-log 未指定)"]
    excl = load_excl(args["excl"]) if args.get("excl") else None
    lines, notes, concl = judge(diag, solo, bench, excl, gate_bad)
    print("OFFICIAL:", "yes" if not notes else "no（参考記録: " + "; ".join(notes) + "）")
    for ln in lines:
        print(ln)
    print("## 結論（RULE.v2 の規則どおり。因果の断定ではない）")
    for c in concl:
        print(c)


def _mk_bench(root, arm, g, n_phase_matmul=0.5):
    d = os.path.join(root, arm)
    os.makedirs(d, exist_ok=True)
    rc = []
    for n in agg.SIZES:
        for r in range(1, RUNS + 1):
            for m in MODES:
                base = {"framework": "fandhe-ai", "size": n, "parity_fail_count": 0, "checksum": "1.0"}
                fresh = 0.001
                if m == "fresh":
                    rows = [{**base, "task": "gemm", "mode": "fresh", "median_s": fresh}]
                elif m == "reuse":
                    rows = [{**base, "task": "gemm", "mode": "reuse", "median_s": fresh + g}]
                else:
                    rows = [{**base, "task": "gemm_phases", "phase": "matmul", "median_s": (fresh + g) * n_phase_matmul},
                            {**base, "task": "gemm_phases", "phase": "iter_total", "median_s": fresh + g}]
                with open(os.path.join(d, f"gemm_{m}_n{n}_r{r}.jsonl"), "w") as f:
                    f.write("\n".join(json.dumps(x) for x in rows) + "\n")
                rc.append(f"{m} n={n} r={r} rc=0")
    with open(os.path.join(d, "rc.log"), "w") as f:
        f.write("\n".join(rc) + "\n")


def self_test():
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        diag_dir = os.path.join(tmp, "diag")

        def mut(r, rec):
            # fresh=0.001 / resident_public=0.0011（G_diag=+0.0001）/ verify_touch=0.0013（G=+0.0003）
            if rec["phase"] != "iter_total":
                return rec
            v = {"fresh": 0.001, "resident_public": 0.0011, "rp_verify_touch": 0.0013, "rp_verify_gap": 0.0011,
                 "rp_hold_footprint": 0.0011, "rp_ref_compute": 0.0011}[rec["arm"]]
            return {**rec, "median_s": v}

        agg._gen_fixture(diag_dir, mutate=mut)
        bench_root = os.path.join(tmp, "bench")
        _mk_bench(bench_root, "base", 0.0004)  # G_bench=0.0004: G_diag=0.0001 とは比 4.0（bridge fail）／G_touch=0.0003 とは 1.33
        _mk_bench(bench_root, "tunables", 0.0004)  # G_Y/G_diag=4.0 → 非支持
        _mk_bench(bench_root, "headpatch", 0.00015)  # G_Y/G_diag=1.5 → 支持
        diag = agg.load_series(diag_dir)
        bench, sums = {}, {}
        for a in BENCH_ARMS:
            bench[a], sums[a] = load_bench_arm(bench_root, a)
        lines, notes, concl = judge(diag, None, bench, {"diag": "pass", **{a: "pass" for a in bench}}, [])
        text = "\n".join(lines + concl)
        for n in JUDGED:
            row = [c for c in concl if c.startswith(f"N={n}:")][0]
            if "rp_verify_touch" not in row or "bench_headpatch" not in row or "rp_verify_gap" in row or "bench_tunables" in row:
                print("SELF-TEST FAIL judge-support:", row)
                ok = False
        if "キャッシュとページ" not in text or notes:
            print("SELF-TEST FAIL notes/cache:", notes)
            ok = False
        # 再現しない場合: G_bench を G_diag に近づけると帰属しない
        _mk_bench(bench_root, "base", 0.0001)
        bench["base"], _ = load_bench_arm(bench_root, "base")
        _, _, concl2 = judge(diag, None, bench, None, [])
        if not all("再現しない" in c for c in concl2):
            print("SELF-TEST FAIL not-reproduced:", concl2)
            ok = False
        _, notes2, _ = judge(diag, None, bench, None, ["x"])
        if len(notes2) < 2:
            print("SELF-TEST FAIL notes-excl-gate")
            ok = False
        # 異常系: parity fail・rc 非 0・欠落ファイル・checksum 不一致
        def must_fail(name, fn):
            nonlocal ok
            try:
                fn()
            except JudgeError:
                return
            print("SELF-TEST FAIL", name)
            ok = False

        p = os.path.join(bench_root, "base", "gemm_fresh_n256_r1.jsonl")
        good = open(p).read()
        open(p, "w").write(good.replace('"parity_fail_count": 0', '"parity_fail_count": 3'))
        must_fail("parity", lambda: load_bench_arm(bench_root, "base"))
        open(p, "w").write(good)
        os.remove(p)
        must_fail("missing", lambda: load_bench_arm(bench_root, "base"))
        open(p, "w").write(good)
        rc = os.path.join(bench_root, "base", "rc.log")
        rg = open(rc).read()
        open(rc, "w").write(rg.replace("rc=0", "rc=1", 1))
        must_fail("rc", lambda: load_bench_arm(bench_root, "base"))
        open(rc, "w").write(rg)
        open(p, "w").write(good.replace('"checksum": "1.0"', '"checksum": "2.0"'))
        b2, s2 = load_bench_arm(bench_root, "base")
        if len(s2[256]) != 2:
            print("SELF-TEST FAIL checksum-set")
            ok = False
    print("self-test:", "ok" if ok else "FAILED")
    return 0 if ok else 1


def main(argv):
    if argv == ["--self-test"]:
        return self_test()
    keys = {"--diag": "diag", "--bench-root": "bench_root", "--solo": "solo", "--excl": "excl", "--gate-log": "gate_log"}
    args = {}
    i = 0
    while i < len(argv):
        if argv[i] not in keys or i + 1 >= len(argv):
            print(__doc__, file=sys.stderr)
            return 2
        args[keys[argv[i]]] = argv[i + 1]
        i += 2
    if "diag" not in args or "bench_root" not in args:
        print(__doc__, file=sys.stderr)
        return 2
    try:
        run(args)
    except (JudgeError, agg.AggError) as e:
        print(f"判定エラー: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
