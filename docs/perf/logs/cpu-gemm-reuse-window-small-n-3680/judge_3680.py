#!/usr/bin/env python3
"""イシュー #3680: RULE.txt の段階 1（差が残っているか）と段階 2（診断腕の支持）を機械的に判定する（標準ライブラリのみ）。

役割: #3671 の aggregate.py（診断系列の検証・集計）と bench_judge.py のローダー（bench の run 検証）を
importlib で同ディレクトリ階層から読み込み、bench の 5 腕（base／vafter／tunboth／tunmmap／tuntrim）を検証して
N=256・512 ごとに次を出す。閾値は新しく作らない（ブリッジ v2 の 0.5〜2.0 と、その下限 0.5 の流用のみ）。
  段階 1: G_bench<=0 → (a) 逆転が消えた／bridge_v2 pass → (b) 差が残っていない／bridge_v2 fail → (c) 差が残っている
  段階 2（(c) の N だけ）: 主判定 G_Y−G_floor <= 0.5×(G_bench−G_floor)（G_floor=max(G_diag,0)）で腕 Y を支持。
          記録のみ: #3671 と同じ式（G_Y>0 かつ G_diag>0 かつ 0.5<=G_Y/G_diag<=2.0）
データ不備（欠落・parity fail・rc 非 0・checksum 不一致・不正 JSON）は exit 1。専有・ゲート・dirty・RULE sha が
未確認なら「参考記録」と出力し OFFICIAL: no（exit 0）。

使い方: judge_3680.py --diag <dir> --bench-root <dir> --excl <excl_status.txt> --gate-log <file> --env <env_info_3680.txt>
        judge_3680.py --self-test
bench-root の構成: <root>/{base,vafter,tunboth,tunmmap,tuntrim}/gemm_{fresh,reuse,phases}_n<N>_r<1..5>.jsonl と各腕の rc.log。
"""
import hashlib
import importlib.util
import os
import statistics
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
D3671 = os.path.join(os.path.dirname(HERE), "cpu-gemm-reuse-window-3671")


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(D3671, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


agg = _load("aggregate_3671", "aggregate.py")
bj = _load("bench_judge_3671", "bench_judge.py")
JudgeError = bj.JudgeError

JUDGED = [256, 512]
LO, HI = 0.5, 2.0  # ブリッジ v2（#3646）。新しい閾値は作らない
HALF = 0.5  # 段階 2 の主判定。ブリッジ v2 の下限をそのまま使う
ARMS = ["base", "vafter", "tunboth", "tunmmap", "tuntrim"]
Y_ARMS = ARMS[1:]


def med(vs):
    return statistics.median(vs)


def stage1(g_bench, g_diag):
    """(a)/(b)/(c)。bridge_v2 は #3671 の ratio_ok と同じ式。"""
    if g_bench <= 0:
        return "a"
    if bj.ratio_ok(g_bench, g_diag):
        return "b"
    return "c"


def support_main(g_y, g_bench, g_diag):
    floor = max(g_diag, 0.0)
    return (g_y - floor) <= HALF * (g_bench - floor)


def support_record(g_y, g_diag):
    return g_y > 0 and g_diag > 0 and LO <= g_y / g_diag <= HI


STAGE_TEXT = {"a": "(a) 逆転が消えた", "b": "(b) 差が残っていない（診断と同じ水準）", "c": "(c) 差が残っている"}


def conclude(n, branch, supported):
    if branch in ("a", "b"):
        return f"N={n}: {STAGE_TEXT[branch]}。帰属に進まない"
    if not supported:
        return f"N={n}: {STAGE_TEXT['c']}が、特定できず（どの腕も主判定で支持されない）"
    extra = ""
    s = set(supported)
    if "vafter" in s:
        extra = "（vafter 支持: 窓の外の検算の位置。窓の見直しは所有者判断で本イシューでは提案に留める）"
    knobs = s & {"tunmmap", "tuntrim"}
    if len(knobs) == 1 and "tunboth" in s:
        extra += f"（ノブは {next(iter(knobs))} のみ支持。因果は断定しない）"
    return f"N={n}: {STAGE_TEXT['c']}。支持された腕 = {', '.join(supported)}{extra}（交互作用は断定しない）"


def analyze(diag, bench):
    """行・結論を返す。diag は aggregate の系列、bench は {arm: {n: cell}}。"""
    ds = diag["summary"]
    lines, concl = [], []
    for n in agg.SIZES:
        fresh_d = ds[("fresh", n, "iter_total")][0]
        rp = ds[("resident_public", n, "iter_total")][0]
        g_diag = rp - fresh_d
        base = bench["base"][n]
        g_bench = med(base["phases"]["iter_total"]) - med(base["fresh"])
        branch = stage1(g_bench, g_diag)
        judged = n in JUDGED
        tag = "判定対象" if judged else "記録のみ（判定外）"
        ratio = f"{g_bench / g_diag:.3f}" if g_diag != 0 else "n/a"
        lines.append(f"N={n} [{tag}] G_bench={g_bench:+.9f} G_diag={g_diag:+.9f} G_bench/G_diag={ratio} "
                     f"段階1={STAGE_TEXT[branch]}")
        supported = []
        for y in Y_ARMS:
            c = bench[y][n]
            g_y = med(c["phases"]["iter_total"]) - med(c["fresh"])
            lo = min(c["phases"]["iter_total"])
            hi = max(c["phases"]["iter_total"])
            main_ok = support_main(g_y, g_bench, g_diag)
            rec_ok = support_record(g_y, g_diag)
            lines.append(f"  Y {y}: G_Y={g_y:+.9f} iter_total_median={med(c['phases']['iter_total']):.9f}"
                         f"[{lo:.9f}-{hi:.9f}] 主判定支持={'yes' if main_ok else 'no'} "
                         f"記録(#3671式)支持={'yes' if rec_ok else 'no'}")
            if judged and branch == "c" and main_ok:
                supported.append(y)
        if judged:
            concl.append(conclude(n, branch, supported))
    complete = all(c.startswith("N=") for c in concl)
    if complete and all(("(a)" in c or "(b)" in c) for c in concl):
        concl.append("段階 1 のみで完了（N=256・512 とも (a) か (b)）")
    return lines, concl


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_env_3680(path):
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except OSError as e:
        raise JudgeError(f"{path}: 読み込み不可 ({e})")
    env = {}
    for ln in text.splitlines():
        if ": " in ln and not ln.startswith(" "):
            k, v = ln.split(": ", 1)
            env[k.strip()] = v.strip()
    for k in ("rev_stamp_head", "rev_stamp_dirty", "rule_sha256", "bench_binary_sha256"):
        if not env.get(k):
            raise JudgeError(f"{path}: {k} の記録が無い（ドリフトガード不能）")
    return env


MODES = ("fresh", "reuse", "phases")


def load_gate_log_3680(path, arms):
    """gate ログにモードごとの通過記録（bench_<arm>/n<N>/r<R>/<mode> gate=pass）を要求し、欠落ラベルを返す。"""
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
            for r in range(1, bj.RUNS + 1):
                for m in MODES:
                    label = f"bench_{a}/n{n}/r{r}/{m}"
                    if seen.get(label) != "pass":
                        bad.append(label)
    return bad


def check_sums(sums):
    """N ごとに全腕・全 run の checksum が 1 種類（RULE.txt checksum 規則）。"""
    for n in agg.SIZES:
        allsum = set()
        for a in sums:
            allsum |= sums[a].get(n, set())
        if len(allsum) != 1:
            raise JudgeError(f"bench の checksum が N={n} で一致しない: {sorted(allsum)}")


def official_notes(diag, excl, gate_bad, env):
    notes = list(diag["why"])
    if gate_bad:
        notes.append(f"bench の負荷ゲート未通過または未記録: {len(gate_bad)} 件（例 {gate_bad[0]}）")
    if excl is None:
        notes.append("専有（excl_status.txt）が未確認")
    else:
        for lab in ("diag", "bench"):
            if excl.get(lab) != "pass":
                notes.append(f"専有 excl={excl.get(lab)!r}（{lab}）")
    if env is None:
        notes.append("env_info_3680.txt が未指定")
    else:
        actual = sha256_file(os.path.join(HERE, "RULE.txt"))
        if env["rule_sha256"] != actual:
            notes.append(f"rule_sha256 が RULE.txt の実ハッシュと不一致（記録 {env['rule_sha256']!r}）")
        diag_head = diag["env"].get("rev_stamp_head")
        if not diag_head or env["rev_stamp_head"] != diag_head:
            notes.append(
                f"診断系列と bench 系列の計測コミットが不一致（diag {diag_head!r} / bench {env['rev_stamp_head']!r}）"
            )
        if env["rev_stamp_dirty"] != "dirty=0":
            notes.append(f"計測ツリーが dirty（{env['rev_stamp_dirty']}）")
    return notes


def run(args):
    diag = agg.load_series(args["diag"])
    if diag["kind"] != "main":
        raise JudgeError("--diag は series_kind=main の系列")
    bench, sums = {}, {}
    for a in ARMS:
        if not os.path.isdir(os.path.join(args["bench_root"], a)):
            raise JudgeError(f"bench の腕 {a} が無い（RULE 必須。欠落は判定しない）")
        bench[a], sums[a] = bj.load_bench_arm(args["bench_root"], a)
    check_sums(sums)
    gate_bad = load_gate_log_3680(args["gate_log"], ARMS) if args.get("gate_log") else ["(gate-log 未指定)"]
    excl = bj.load_excl(args["excl"]) if args.get("excl") else None
    env = load_env_3680(args["env"]) if args.get("env") else None
    notes = official_notes(diag, excl, gate_bad, env)
    lines, concl = analyze(diag, bench)
    print("OFFICIAL:", "yes" if not notes else "no（参考記録: " + "; ".join(notes) + "）")
    for ln in lines:
        print(ln)
    print("## 結論（RULE.txt の規則どおり。因果の断定ではない）")
    for c in concl:
        print(c)


def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        if not cond:
            print("SELF-TEST FAIL", name)
            ok = False

    # 純関数: 段階 1
    check("a", stage1(-0.0001, 0.0001) == "a" and stage1(0.0, 0.0001) == "a")
    check("b", stage1(0.00015, 0.0001) == "b")
    check("c-ratio", stage1(0.0004, 0.0001) == "c")
    check("c-gdiag-neg", stage1(0.0004, -0.0001) == "c")
    # 段階 2 主判定（G_diag<0 でも判定できる）
    check("main-neg-floor", support_main(0.0001, 0.0004, -0.0002) and not support_main(0.0003, 0.0004, -0.0002))
    check("main-pos-floor", support_main(0.00025, 0.0004, 0.0001) and not support_main(0.0003, 0.0004, 0.0001))
    check("record-undefined-for-neg", not support_record(0.0001, -0.0001) and support_record(0.00015, 0.0001))
    # 結論文
    check("concl-none", "特定できず" in conclude(256, "c", []))
    check("concl-multi", "vafter, tunmmap" in conclude(256, "c", ["vafter", "tunmmap"]))
    check("concl-knob", "tunmmap のみ" in conclude(256, "c", ["tunboth", "tunmmap"]))
    check("concl-a", "逆転が消えた" in conclude(256, "a", []))

    # 合成データでの end-to-end
    with tempfile.TemporaryDirectory() as tmp:
        def make_diag(rp_val):
            d = os.path.join(tmp, f"diag{rp_val}")

            def mut(r, rec):
                if rec["phase"] != "iter_total":
                    return rec
                v = {"fresh": 0.001, "resident_public": rp_val, "rp_verify_touch": 0.0011, "rp_verify_gap": 0.0011,
                     "rp_hold_footprint": 0.0011, "rp_ref_compute": 0.0011}[rec["arm"]]
                return {**rec, "median_s": v}

            agg._gen_fixture(d, mutate=mut)
            return agg.load_series(d)

        def make_bench(gs):
            root = os.path.join(tmp, "b" + "_".join(str(g) for g in gs.values()))
            sums = {}
            bench = {}
            for a in ARMS:
                bj._mk_bench(root, a, gs[a])
                bench[a], sums[a] = bj.load_bench_arm(root, a)
            return root, bench, sums

        diag = make_diag(0.0011)  # G_diag=+0.0001
        # (c): base=+0.0004。vafter が超過分を消す（G=0.0001）、tunmmap は半分未満しか消えない
        gs = {"base": 0.0004, "vafter": 0.0001, "tunboth": 0.0004, "tunmmap": 0.0004, "tuntrim": 0.0004}
        _, bench, sums = make_bench(gs)
        check_sums(sums)
        _, concl = analyze(diag, bench)
        check("e2e-c-vafter", all("支持された腕 = vafter" in c for c in concl[:2]) and len(concl) == 2)
        # (a)
        gs_a = dict(gs, base=-0.0001)
        _, bench_a, _ = make_bench(gs_a)
        _, concl_a = analyze(diag, bench_a)
        check("e2e-a", all("逆転が消えた" in c for c in concl_a[:2]) and "段階 1 のみで完了" in concl_a[-1])
        # (b)
        gs_b = dict(gs, base=0.00015)
        _, bench_b, _ = make_bench(gs_b)
        _, concl_b = analyze(diag, bench_b)
        check("e2e-b", all("差が残っていない" in c for c in concl_b[:2]) and "段階 1 のみで完了" in concl_b[-1])
        # 支持なし
        gs_n = dict(gs, vafter=0.0004)
        _, bench_n, _ = make_bench(gs_n)
        _, concl_n = analyze(diag, bench_n)
        check("e2e-none", all("特定できず" in c for c in concl_n[:2]))
        # G_diag<0 の (c)
        diag_neg = make_diag(0.0009)
        _, concl_g = analyze(diag_neg, bench)
        check("e2e-gdiag-neg", all("支持された腕 = vafter" in c for c in concl_g[:2]))
        # 複数支持
        gs_m = dict(gs, tunmmap=0.0001)
        _, bench_m, _ = make_bench(gs_m)
        _, concl_m = analyze(diag, bench_m)
        check("e2e-multi", all("vafter, tunmmap" in c for c in concl_m[:2]))
        # データ不備: checksum 不一致・腕欠落は例外
        root_x, _, _ = make_bench(gs)
        p = os.path.join(root_x, "tuntrim", "gemm_fresh_n256_r1.jsonl")
        with open(p) as f:
            good = f.read()
        with open(p, "w") as f:
            f.write(good.replace('"checksum": "1.0"', '"checksum": "2.0"'))
        try:
            s2 = {a: bj.load_bench_arm(root_x, a)[1] for a in ARMS}
            check_sums(s2)
            check("checksum-mismatch", False)
        except JudgeError:
            pass
        try:
            run({"diag": "/nonexistent", "bench_root": root_x})
            check("bad-diag", False)
        except (JudgeError, agg.AggError):
            pass
        # 参考記録化
        check("notes-excl", len(official_notes(diag, None, ["x"], None)) >= 3)
        env_ok = {
            "rule_sha256": sha256_file(os.path.join(HERE, "RULE.txt")),
            "rev_stamp_dirty": "dirty=0",
            "rev_stamp_head": diag["env"]["rev_stamp_head"],
        }
        check("notes-clean", official_notes(diag, {"diag": "pass", "bench": "pass"}, [], env_ok) == diag["why"])
        env_bad = dict(env_ok, rule_sha256="0" * 64, rev_stamp_dirty="dirty=1")
        check("notes-env", len(official_notes(diag, {"diag": "pass", "bench": "pass"}, [], env_bad)) - len(diag["why"]) == 2)
        env_head = dict(env_ok, rev_stamp_head="different")
        check("notes-head", len(official_notes(diag, {"diag": "pass", "bench": "pass"}, [], env_head)) - len(diag["why"]) == 1)
        gl = os.path.join(tmp, "gate.log")
        with open(gl, "w") as f:
            for a in ARMS:
                for n in agg.SIZES:
                    for r in range(1, bj.RUNS + 1):
                        for m in MODES:
                            f.write(f"bench_{a}/n{n}/r{r}/{m} gate=pass\n")
        check("gate-modes-ok", load_gate_log_3680(gl, ARMS) == [])
        with open(gl) as f:
            lines = f.readlines()
        with open(gl, "w") as f:
            f.writelines(lines[:-1])
        check("gate-modes-missing", len(load_gate_log_3680(gl, ARMS)) == 1)
    print("self-test:", "ok" if ok else "FAILED")
    return 0 if ok else 1


def main(argv):
    if argv == ["--self-test"]:
        return self_test()
    keys = {"--diag": "diag", "--bench-root": "bench_root", "--excl": "excl", "--gate-log": "gate_log", "--env": "env"}
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
