#!/usr/bin/env python3
"""#3175 の計測ログ集計（判定規則は同ディレクトリ RULE.txt）。

役割: orchestrate_macmini.sh が出力した .log を厳密にパースし、スレッド掃引（SME/NEON の
GFLOPS・スケール比・頭打ち判定）、micro-call 固定費、E コア系列を aggregate.md 形式で出力する。
標準ライブラリのみ。`python3 -I aggregate.py [ログディレクトリ]`、`--self-test` で自己検査。
欠落・重複・不一致はすべて例外で止める（fail-closed。寛容パースはしない）。
"""
import os
import re
import statistics
import sys
import tempfile

RUNS = [1, 2, 3, 4, 5]
THREADS = [1, 2, 4, 10]
MAIN_SHAPES = [(256, 256, 128), (512, 512, 256), (1024, 1024, 512), (2048, 2048, 1024)]
REF_SHAPES = [(64, 256, 784), (784, 256, 64)]
ALL_SHAPES = MAIN_SHAPES + REF_SHAPES
VARIANTS = {"NEON(TwoDDynamic)": "NEON", "SME(TwoDDynamicSme)": "SME"}
KCS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]
THIRD_PARTY_GFLOPS = 2008.0
MICRO_FLOP_PER_KC = 2 * 16 * 16  # MR=NR=16 の 1 kc あたり flop


class ParseError(Exception):
    pass


def read_lines(path):
    if not os.path.isfile(path):
        raise ParseError(f"ログ欠落: {path}")
    with open(path, encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]
    if not lines:
        raise ParseError(f"空ログ: {path}")
    return lines


SWEEP_RE = re.compile(
    r"^variant=(\S+) num_threads=(\d+) size=\((\d+),(\d+),(\d+)\) median_gflops=([0-9]+\.[0-9]+)$"
)


def parse_sweep(path, expect_t):
    """{(kind, shape): gflops} を返す。num_threads 行・rayon 行・各 12 値を全検証する。"""
    nt = None
    got = {}
    for ln in read_lines(path):
        m = re.match(r"^num_threads=(\d+)$", ln)
        if m:
            if nt is not None:
                raise ParseError(f"num_threads 行が重複: {path}")
            nt = int(m.group(1))
            continue
        if re.match(r"^rayon_current_num_threads=\d+$", ln):
            continue
        m = SWEEP_RE.match(ln)
        if not m:
            raise ParseError(f"パース不能な行: {path}: {ln!r}")
        if m.group(1) not in VARIANTS:
            raise ParseError(f"未知の variant: {path}: {ln!r}")
        if int(m.group(2)) != expect_t:
            raise ParseError(f"num_threads 不一致: {path}: 行={m.group(2)} 期待={expect_t}")
        key = (VARIANTS[m.group(1)], (int(m.group(3)), int(m.group(4)), int(m.group(5))))
        if key in got:
            raise ParseError(f"重複行: {path}: {key}")
        got[key] = float(m.group(6))
    if nt != expect_t:
        raise ParseError(f"num_threads 行の不一致/欠落: {path}: {nt} != {expect_t}")
    for kind in ("SME", "NEON"):
        for shp in ALL_SHAPES:
            if (kind, shp) not in got:
                raise ParseError(f"値の欠落: {path}: {kind} {shp}")
    if len(got) != 2 * len(ALL_SHAPES):
        raise ParseError(f"想定外の行数: {path}")
    return got


PROBE_RE = re.compile(
    r"^probe=microcall kernel=(SME|NEON) kc=(\d+) reps=(\d+) median_ns_per_call=([0-9]+\.[0-9]+)$"
)


def parse_probe(path):
    got = {}
    for ln in read_lines(path):
        m = PROBE_RE.match(ln)
        if not m:
            raise ParseError(f"パース不能な行: {path}: {ln!r}")
        key = (m.group(1), int(m.group(2)))
        if key in got:
            raise ParseError(f"重複行: {path}: {key}")
        got[key] = float(m.group(4))
    for kern in ("SME", "NEON"):
        for kc in KCS:
            if (kern, kc) not in got:
                raise ParseError(f"値の欠落: {path}: {kern} kc={kc}")
    if len(got) != 2 * len(KCS):
        raise ParseError(f"想定外の行数: {path}")
    return got


GATE_RE = re.compile(
    r"^START series=(\w+) run=(\d+) T=(\d+) gate=(pass|unavailable) load1=([0-9.]+) waited_s=(\d+) at=\S+$"
)
END_RE = re.compile(
    r"^END series=(\w+) run=(\d+) T=(\d+) exit=(\d+) grep_exit=(\d+) end_load1=([0-9.]+)$"
)


def parse_gate(path):
    starts, ends = {}, {}
    for ln in read_lines(path):
        if ln.startswith("ecore_unavailable"):
            continue
        m = GATE_RE.match(ln)
        if m:
            key = (m.group(1), int(m.group(2)), int(m.group(3)))
            if key in starts:
                raise ParseError(f"START 重複: {ln!r}")
            starts[key] = m.group(4)
            continue
        m = END_RE.match(ln)
        if m:
            key = (m.group(1), int(m.group(2)), int(m.group(3)))
            if key in ends:
                raise ParseError(f"END 重複: {ln!r}")
            if m.group(4) != "0" or m.group(5) != "0":
                raise ParseError(f"プロセス失敗の記録: {ln!r}")
            ends[key] = True
            continue
        raise ParseError(f"パース不能なゲート行: {ln!r}")
    if set(starts) != set(ends):
        raise ParseError("START と END の集合が一致しない")
    return starts


def lsq_line(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        raise ParseError("最小二乗の分散が 0")
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - slope * mx, slope  # 切片, 傾き


def judge(s4):
    if s4 < 1.25:
        return "頭打ちあり"
    if s4 >= 2.0:
        return "頭打ちなし"
    return "部分的"


def aggregate(d):
    out = []
    w = out.append
    gate = parse_gate(os.path.join(d, "load_gate.log"))
    sweep = {}  # (T, run) -> dict
    for t in THREADS:
        for r in RUNS:
            if ("thread", r, t) not in gate:
                raise ParseError(f"ゲート記録の欠落: thread run={r} T={t}")
            sweep[(t, r)] = parse_sweep(os.path.join(d, f"thread_t{t}_run{r}.log"), t)
    probe = {}
    for r in RUNS:
        if ("microcall", r, 1) not in gate:
            raise ParseError(f"ゲート記録の欠落: microcall run={r}")
        probe[r] = parse_probe(os.path.join(d, f"microcall_run{r}.log"))
    formal_keys = [("thread", r, t) for t in THREADS for r in RUNS] + [("microcall", r, 1) for r in RUNS]
    passed = sum(1 for k in formal_keys if gate[k] == "pass")
    formal = passed == len(formal_keys)
    w("# 集計結果（#3175）")
    w("")
    w(f"- 負荷ゲート通過: {passed}/{len(formal_keys)} プロセス（系列 A+B）-> "
      + ("**正式系列**" if formal else "**参考系列**（全通過ではない）"))
    w("")

    med = {}  # (kind, T, shape) -> median
    for kind in ("SME", "NEON"):
        w(f"## {kind} の GFLOPS（run1..5 / 中央値）")
        w("")
        w("| 形状 | T | run1 | run2 | run3 | run4 | run5 | 中央値 |")
        w("|---|---|---|---|---|---|---|---|")
        for shp in ALL_SHAPES:
            for t in THREADS:
                vals = [sweep[(t, r)][(kind, shp)] for r in RUNS]
                m = statistics.median(vals)
                med[(kind, t, shp)] = m
                tag = "" if shp in MAIN_SHAPES else "(参考)"
                w(f"| {shp}{tag} | {t} | " + " | ".join(f"{v:.1f}" for v in vals) + f" | {m:.1f} |")
        w("")

    w("## スケール比と頭打ち判定（主 4 形状）")
    w("")
    w("| 形状 | S_sme(2) | S_sme(4) | S_sme(10) | G(10)/G(4) | 判定 | S_neon(4) | S_neon(10) | max_T G / 2008 |")
    w("|---|---|---|---|---|---|---|---|---|")
    verdicts = {}
    for shp in MAIN_SHAPES:
        g = {t: med[("SME", t, shp)] for t in THREADS}
        n = {t: med[("NEON", t, shp)] for t in THREADS}
        s = {t: g[t] / g[1] for t in THREADS}
        v = judge(s[4])
        verdicts[shp] = v
        w(f"| {shp} | {s[2]:.2f} | {s[4]:.2f} | {s[10]:.2f} | {g[10] / g[4]:.2f} | {v} | "
          f"{n[4] / n[1]:.2f} | {n[10] / n[1]:.2f} | {max(g.values()) / THIRD_PARTY_GFLOPS:.3f} |")
    w("")
    head = MAIN_SHAPES[-1]
    consistent = len(set(verdicts.values())) == 1
    w(f"- 見出し（{head}）: **{verdicts[head]}**。他 3 形状との一貫性: "
      + ("一致" if consistent else "不一致（形状ごとの表を参照）"))
    w(f"- 第三者実測 {THIRD_PARTY_GFLOPS:.0f} GFLOPS はリポジトリ外の値で合格ラインではない")
    w("")

    w("## SME/NEON 比（中央値）")
    w("")
    w("| 形状 | " + " | ".join(f"T={t}" for t in THREADS) + " |")
    w("|---|" + "---|" * len(THREADS))
    for shp in ALL_SHAPES:
        w(f"| {shp} | " + " | ".join(f"{med[('SME', t, shp)] / med[('NEON', t, shp)]:.2f}" for t in THREADS) + " |")
    w("")

    w("## micro-call 固定費（単スレッド・ns/call の run 中央値）")
    w("")
    pm = {}
    for kern in ("SME", "NEON"):
        for kc in KCS:
            pm[(kern, kc)] = statistics.median([probe[r][(kern, kc)] for r in RUNS])
    w("| kc | SME | NEON |")
    w("|---|---|---|")
    for kc in KCS:
        w(f"| {kc} | {pm[('SME', kc)]:.1f} | {pm[('NEON', kc)]:.1f} |")
    w("")
    fit = {}
    for kern in ("SME", "NEON"):
        xs = [float(kc) for kc in KCS if kc >= 1]
        ys = [pm[(kern, kc)] for kc in KCS if kc >= 1]
        c, s = lsq_line(xs, ys)
        fit[kern] = (c, s)
        w(f"- {kern}: kc=0 直接計時 {pm[(kern, 0)]:.1f} ns / kc>=1 の最小二乗 切片 c={c:.1f} ns・傾き s={s:.3f} ns/kc"
          f"（定常 {MICRO_FLOP_PER_KC / s:.1f} GFLOPS）")
    w("- 固定費は smstart/smstop 単体ではなく、SVL・TPIDR2_EL0 検査・長さ検査・C 16 行の ZA0 出し入れを含む「1 組の固定費の上限」")
    w("")
    c0 = pm[("SME", 0)]
    w("### (64,256,784) での固定費の占有（micro-call 256 組 = 4x16x4）")
    w("")
    w("| T | SME GFLOPS | 実測時間 us | 256 組 x kc=0 固定費 us（T で等分と仮定） | 占有 %（kc=0 基準） | 占有 %（切片 c 基準） |")
    w("|---|---|---|---|---|---|")
    m_, n_, k_ = 64, 256, 784
    for t in THREADS:
        g = med[("SME", t, (m_, n_, k_))]
        t_ns = 2.0 * m_ * n_ * k_ / g  # flop / (GFLOP/s) = ns
        fixed = 256 * c0 / t
        fixed_c = 256 * fit["SME"][0] / t
        w(f"| {t} | {g:.1f} | {t_ns / 1e3:.1f} | {fixed / 1e3:.1f} | {100.0 * fixed / t_ns:.1f} | "
          f"{100.0 * fixed_c / t_ns:.1f} |")
    w("")

    ep = os.path.join(d, "ecore_t1_run1.log")
    if os.path.isfile(ep):
        ecore = {}
        for r in RUNS:
            if ("ecore", r, 1) not in gate:
                raise ParseError(f"ゲート記録の欠落: ecore run={r}")
            ecore[r] = parse_sweep(os.path.join(d, f"ecore_t1_run{r}.log"), 1)
        w("## E コア側系列（taskpolicy -b・T=1）と P 側（T=1）の比較")
        w("")
        w("| 形状 | NEON P | NEON E(bg) | E/P | SME P | SME E(bg) | E/P |")
        w("|---|---|---|---|---|---|---|")
        ratios = []
        for shp in ALL_SHAPES:
            en = statistics.median([ecore[r][("NEON", shp)] for r in RUNS])
            es = statistics.median([ecore[r][("SME", shp)] for r in RUNS])
            pn, ps = med[("NEON", 1, shp)], med[("SME", 1, shp)]
            if shp in MAIN_SHAPES:
                ratios.append(en / pn)
            w(f"| {shp} | {pn:.1f} | {en:.1f} | {en / pn:.2f} | {ps:.1f} | {es:.1f} | {es / ps:.2f} |")
        mr = statistics.median(ratios)
        sep = not (0.8 <= mr <= 1.25)
        w("")
        w(f"- 主 4 形状の NEON E/P 中央値比 = {mr:.2f} -> " + ("P と E を分けられた" if sep else "**分けられなかった**（0.8〜1.25）"))
        w("")
    else:
        w("## E コア側系列")
        w("")
        w("- ecore_t1_run*.log なし（taskpolicy 無し等）。P と E は分けられなかったと記録する")
        w("")
    return "\n".join(out) + "\n"


def make_synthetic(d, mutate=None):
    """自己検査用の合成ログ一式を作る。"""
    gl = []
    for series, ts in [("thread", THREADS), ("microcall", [1]), ("ecore", [1])]:
        for r in RUNS:
            for t in ts:
                gl.append(f"START series={series} run={r} T={t} gate=pass load1=1.00 waited_s=0 at=2026-01-01T00:00:00Z")
                gl.append(f"END series={series} run={r} T={t} exit=0 grep_exit=0 end_load1=1.10")

    def sweep_text(t, scale):
        lines = [f"num_threads={t}", f"rayon_current_num_threads={t}"]
        for shp in ALL_SHAPES:
            for lab, base in (("NEON(TwoDDynamic)", 100.0), ("SME(TwoDDynamicSme)", 1000.0)):
                g = base * (1.0 if lab.startswith("NEON") else min(t, 1.1)) * scale
                lines.append(f"variant={lab} num_threads={t} size=({shp[0]},{shp[1]},{shp[2]}) median_gflops={g:.3f}")
        return "\n".join(lines) + "\n"

    for t in THREADS:
        for r in RUNS:
            with open(os.path.join(d, f"thread_t{t}_run{r}.log"), "w") as f:
                f.write(sweep_text(t, 1.0 + 0.01 * r))
    for r in RUNS:
        with open(os.path.join(d, f"ecore_t1_run{r}.log"), "w") as f:
            f.write(sweep_text(1, 0.4))
        lines = []
        for kern, fixed, slope in (("SME", 40.0, 1.2), ("NEON", 5.0, 1.5)):
            for kc in KCS:
                ns = fixed if kc == 0 else fixed + 50.0 + slope * kc
                lines.append(f"probe=microcall kernel={kern} kc={kc} reps=1000 median_ns_per_call={ns:.3f}")
        with open(os.path.join(d, f"microcall_run{r}.log"), "w") as f:
            f.write("\n".join(lines) + "\n")
    with open(os.path.join(d, "load_gate.log"), "w") as f:
        f.write("\n".join(gl) + "\n")
    if mutate:
        mutate(d)


def self_test():
    def expect_error(name, mutate):
        with tempfile.TemporaryDirectory() as d:
            make_synthetic(d, mutate)
            try:
                aggregate(d)
            except ParseError:
                return
            raise AssertionError(f"例外にならなかった: {name}")

    def drop_line(path_name):
        def f(d):
            p = os.path.join(d, path_name)
            lines = open(p).read().splitlines()
            del lines[-1]
            open(p, "w").write("\n".join(lines) + "\n")
        return f

    def dup_line(d):
        p = os.path.join(d, "thread_t2_run1.log")
        lines = open(p).read().splitlines()
        lines.append(lines[-1])
        open(p, "w").write("\n".join(lines) + "\n")

    def bad_nt(d):
        p = os.path.join(d, "thread_t4_run3.log")
        s = open(p).read().replace("num_threads=4", "num_threads=2")
        open(p, "w").write(s)

    def drop_file(d):
        os.remove(os.path.join(d, "thread_t10_run5.log"))

    with tempfile.TemporaryDirectory() as d:
        make_synthetic(d)
        md = aggregate(d)
        assert "正式系列" in md and "20/20" not in md
        assert "頭打ちあり" in md  # S(4)=1.1 < 1.25
        assert "分けられた" in md  # E/P = 0.4
        assert "| 0 | 40.0 |" in md  # kc=0 の直接計時の中央値
    expect_error("行欠落", drop_line("thread_t1_run1.log"))
    expect_error("プローブ行欠落", drop_line("microcall_run2.log"))
    expect_error("重複行", dup_line)
    expect_error("num_threads 不一致", bad_nt)
    expect_error("run 欠落", drop_file)
    print("self-test OK")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        self_test()
        return
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    sys.stdout.write(aggregate(d))


if __name__ == "__main__":
    main()
