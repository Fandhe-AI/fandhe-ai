#!/usr/bin/env python3
"""イシュー #3177 の実測ログ（macmini／gb10）を fail-closed で集計し Markdown を標準出力へ出す。

判定規則は同ディレクトリ RULE.txt。採否は書かない（ratio と勝ち run 数を記録するだけ）。
欠損・数値でない値・checksum 不一致・rayon_threads の食い違いは例外にし exit 非 0 とする
（寛容パースで欠損を握りつぶさない。前例: sme-macmini-3176/aggregate.py）。
標準ライブラリのみ。`python3 -I aggregate.py [--dir DIR] [--machines a,b]`・`--self-test`。
"""
import os
import re
import statistics
import sys
import tempfile

RUNS = [1, 2, 3, 4, 5]
SHAPES = {  # label: (trans, m, n, k)
    "l1_fwd": ("nn", 64, 256, 784),
    "l1_dw": ("tn", 784, 256, 64),
    "l1_dx": ("nt", 64, 784, 256),
    "l2_fwd": ("nn", 64, 10, 256),
    "l2_dw": ("tn", 256, 10, 64),
    "l2_dx": ("nt", 64, 256, 10),
}
BENCH_SHAPES = dict(SHAPES, ref_square_512=("nn", 512, 512, 512))
ARMS = ["base", "jpw1", "cap4", "cap8"]
COUNT_ARMS = ["native", "T1", "T10", "T20"]
# 再現性を要求する整数フィールド（workers_used は観測値のため除外し run ごとに記録のみ）
STABLE = ["rayon_threads", "eff_threads", "mr", "nr", "row_bands", "col_bands", "mc_job", "nc_job",
          "regions", "jobs", "a_pack_tiles", "a_pack_words", "a_ideal_words", "b_pack_tiles",
          "b_pack_words", "b_ideal_words", "alloc_job", "alloc_split", "c_stage_words"]
# diagnosis.md §2 の再計算値: (jobs, row_bands, col_bands, a_repack_x, b_repack_x)
EXPECT = {
    10: {"l1_fwd": (22, 2, 11, 11.0, 2.0), "l1_dw": (20, 10, 2, 2.0, 10.0), "l1_dx": (22, 1, 22, 22.0, 1.0),
         "l2_fwd": (8, 8, 1, 1.0, 8.0), "l2_dw": (32, 32, 1, 1.0, 32.0), "l2_dx": (22, 2, 11, 11.0, 2.0)},
    20: {"l1_fwd": (44, 4, 11, 11.0, 4.0), "l1_dw": (40, 10, 4, 4.0, 10.0), "l1_dx": (44, 2, 22, 22.0, 2.0),
         "l2_fwd": (8, 8, 1, 1.0, 8.0), "l2_dw": (32, 32, 1, 1.0, 32.0), "l2_dx": (44, 4, 11, 11.0, 4.0)},
}
GATE_STARTS = 5 + 5 + 5 + 20


class AggError(Exception):
    pass


def kv_line(line, path):
    d = dict(re.findall(r"(\w+)=(\S+)", line))
    if not d:
        raise AggError(f"{path}: key=value を含まない行: {line!r}")
    return d


def to_int(d, k, path):
    if k not in d or not re.fullmatch(r"\d+", d[k]):
        raise AggError(f"{path}: {k} が欠損または整数でない: {d.get(k)!r}")
    return int(d[k])


def to_float(d, k, path):
    try:
        v = float(d[k])
    except (KeyError, ValueError):
        raise AggError(f"{path}: {k} が欠損または数値でない: {d.get(k)!r}")
    if not (v == v) or v in (float("inf"), float("-inf")) or v <= 0:
        raise AggError(f"{path}: {k} が正の有限値でない: {v}")
    return v


def read_lines(path, prefix_re):
    if not os.path.isfile(path):
        raise AggError(f"欠損: {path}")
    out = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if not ln or ln.startswith("#"):
                continue
            if not re.match(prefix_re, ln):
                raise AggError(f"{path}: 想定外の行: {ln!r}")
            out.append(kv_line(ln, path))
    return out


def parse_count(root):
    """arm -> shape -> run -> dict。run 間で STABLE が全一致でなければ止める。"""
    data = {a: {s: {} for s in SHAPES} for a in COUNT_ARMS}
    for r in RUNS:
        p = os.path.join(root, "count", f"run{r}.log")
        rows = read_lines(p, r"probe=jobgrid ")
        seen = set()
        for d in rows:
            arm, shape = d.get("arm"), d.get("shape")
            if arm not in data or shape not in SHAPES:
                raise AggError(f"{p}: 未知の arm/shape: {arm}/{shape}")
            if (arm, shape) in seen:
                raise AggError(f"{p}: 重複: {arm}/{shape}")
            seen.add((arm, shape))
            rec = {k: to_int(d, k, p) for k in STABLE + ["workers_used"]}
            for k in ("a_repack_x", "b_repack_x"):
                rec[k] = to_float(d, k, p)
            data[arm][shape][r] = rec
        if len(seen) != len(COUNT_ARMS) * len(SHAPES):
            raise AggError(f"{p}: 腕×形状が揃っていない（{len(seen)} 行）")
    for arm in COUNT_ARMS:
        for shape in SHAPES:
            base = data[arm][shape][1]
            for r in RUNS[1:]:
                for k in STABLE:
                    if data[arm][shape][r][k] != base[k]:
                        raise AggError(f"計数が run 間で不一致: {arm}/{shape}/{k} run1={base[k]} run{r}={data[arm][shape][r][k]}")
    return data


def parse_bench(root, sub, series, arm_of, shapes, want_threads=None):
    """sub/run{N}[-arm].log -> {arm: {shape: {run: (median, checksum, rayon_threads)}}}"""
    out = {}
    for arm in arm_of:
        out[arm] = {s: {} for s in shapes}
        for r in RUNS:
            name = f"run{r}-{arm}.log" if sub == "ab" else f"run{r}.log"
            p = os.path.join(root, sub, name)
            rows = read_lines(p, r"arm=")
            seen = set()
            for d in rows:
                if d.get("arm") != arm or d.get("series") != series:
                    raise AggError(f"{p}: arm/series が食い違い: {d.get('arm')}/{d.get('series')}")
                shape = d.get("shape")
                if shape not in shapes or shape in seen:
                    raise AggError(f"{p}: 未知または重複の shape: {shape}")
                seen.add(shape)
                rt = to_int(d, "rayon_threads", p)
                if want_threads is not None and rt != want_threads:
                    raise AggError(f"{p}: rayon_threads={rt} が期待値 {want_threads} と食い違い")
                cs = d.get("checksum", "")
                if not re.fullmatch(r"0x[0-9a-f]{16}", cs):
                    raise AggError(f"{p}: checksum が不正: {cs!r}")
                out[arm][shape][r] = (to_float(d, "median_secs", p), cs, rt)
            if seen != set(shapes):
                raise AggError(f"{p}: 形状が揃っていない: {sorted(set(shapes) - seen)}")
    return out


def gflops(shape_tbl, shape, med):
    _, m, n, k = shape_tbl[shape]
    return 2.0 * m * n * k / med / 1e9


def gate_status(root):
    p = os.path.join(root, "load_gate.log")
    if not os.path.isfile(p):
        raise AggError(f"欠損: {p}")
    passes = unavail = 0
    with open(p, encoding="utf-8") as f:
        for ln in f:
            if ln.startswith("START "):
                if " gate=pass " in ln:
                    passes += 1
                elif " gate=unavailable " in ln:
                    unavail += 1
    return passes, unavail


def require_files(root):
    for f in ("env_info.txt", ".rev-stamp", "uname.txt", "tree_verify.txt", "patch_sha256.txt", "DONE"):
        if not os.path.isfile(os.path.join(root, f)):
            raise AggError(f"欠損: {os.path.join(root, f)}")


def machine_report(root, name):
    """(markdown, ok)。ok=False は A/B の checksum 不一致など unavailable を含む。"""
    require_files(root)
    ok = True
    L = [f"## {name}", ""]
    passes, unavail = gate_status(root)
    formal = passes == GATE_STARTS and unavail == 0
    L += [f"- 負荷ゲート: pass {passes}/{GATE_STARTS}・unavailable {unavail} -> "
          f"**{'正式系列' if formal else '参考系列（全プロセスがゲートを通っていない）'}**",
          f"- .rev-stamp: `{open(os.path.join(root, '.rev-stamp')).read().strip()[:60]}`", ""]

    cnt = parse_count(root)
    native_threads = cnt["native"]["l1_fwd"][1]["rayon_threads"]
    L += [f"### 計数（5 run 全一致。native の rayon_threads={native_threads}）", "",
          "| 形状 | 腕 | job（行帯×列帯） | A 再 pack 倍率 | B 再 pack 倍率 | alloc_job | alloc_split | C staging 語 | workers_used（run1〜5） | 再計算との突合 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for shape in SHAPES:
        for arm in COUNT_ARMS:
            r1 = cnt[arm][shape][1]
            th = {"native": native_threads, "T1": 1, "T10": 10, "T20": 20}[arm]
            exp = EXPECT.get(th, {}).get(shape)
            if exp is None:
                verdict = "期待表なし"
            else:
                got = (r1["jobs"], r1["row_bands"], r1["col_bands"], round(r1["a_repack_x"], 1), round(r1["b_repack_x"], 1))
                verdict = "一致" if got == exp else f"**不一致**（期待 {exp}）"
            ws = "/".join(str(cnt[arm][shape][r]["workers_used"]) for r in RUNS)
            L.append(f"| {shape} | {arm} | {r1['jobs']}（{r1['row_bands']}×{r1['col_bands']}） | {r1['a_repack_x']:.1f} | "
                     f"{r1['b_repack_x']:.1f} | {r1['alloc_job']} | {r1['alloc_split']} | {r1['c_stage_words']} | {ws} | {verdict} |")
    L.append("")

    st = parse_bench(root, "st", "threads", ["base"], BENCH_SHAPES, want_threads=1)["base"]
    ref = parse_bench(root, "serial_ref", "serial_ref", ["base"],
                      {s: v for s, v in BENCH_SHAPES.items() if v[0] == "nn"})["base"]
    L += ["### 単スレッド実効 GFLOPS（RAYON_NUM_THREADS=1・5 プロセスの median_secs の中央値から算出）", "",
          "| 形状 | GFLOPS（中央値） | run1〜5 の GFLOPS | serial_ref の GFLOPS（参照） |", "|---|---|---|---|"]
    for shape in BENCH_SHAPES:
        meds = [st[shape][r][0] for r in RUNS]
        per = "/".join(f"{gflops(BENCH_SHAPES, shape, st[shape][r][0]):.1f}" for r in RUNS)
        rf = "-"
        if shape in ref:
            rf = f"{gflops(BENCH_SHAPES, shape, statistics.median([ref[shape][r][0] for r in RUNS])):.1f}"
        L.append(f"| {shape} | {gflops(BENCH_SHAPES, shape, statistics.median(meds)):.1f} | {per} | {rf} |")
    L.append("")

    ab = parse_bench(root, "ab", "threads", ARMS, BENCH_SHAPES)
    ab_threads = ab["base"]["l1_fwd"][1][2]
    L += [f"### A/B（既定プール・rayon_threads={ab_threads}・ratio = median(腕)/median(base)・採否は書かない）", "",
          "| 形状 | base median（s） | 腕 | 腕 median（s） | ratio | 腕が勝った run（5 中） | checksum |", "|---|---|---|---|---|---|---|"]
    for shape in BENCH_SHAPES:
        sums = {ab[a][shape][r][1] for a in ARMS for r in RUNS}
        csum_ok = len(sums) == 1
        if not csum_ok:
            ok = False
        bmed = statistics.median([ab["base"][shape][r][0] for r in RUNS])
        for arm in ARMS[1:]:
            if not csum_ok:
                L.append(f"| {shape} | - | {arm} | - | unavailable | - | **不一致**（{len(sums)} 種） |")
                continue
            amed = statistics.median([ab[arm][shape][r][0] for r in RUNS])
            wins = sum(1 for r in RUNS if ab[arm][shape][r][0] < ab["base"][shape][r][0])
            L.append(f"| {shape} | {bmed:.9f} | {arm} | {amed:.9f} | {amed / bmed:.3f} | {wins} | 一致 |")
    L.append("")
    return "\n".join(L), ok


def self_test():
    def mk(root, mutate=None):
        os.makedirs(os.path.join(root, "count"))
        for d in ("st", "serial_ref", "ab"):
            os.makedirs(os.path.join(root, d))
        for f, txt in (("env_info.txt", "x"), (".rev-stamp", "abc"), ("uname.txt", "x"),
                       ("tree_verify.txt", "x"), ("patch_sha256.txt", "x"), ("DONE", "x")):
            open(os.path.join(root, f), "w").write(txt + "\n")
        open(os.path.join(root, "load_gate.log"), "w").write(
            "".join("START series=s run=1 arm=base gate=pass load1=1.0 waited_s=0 at=t\n" for _ in range(GATE_STARTS)))
        for r in RUNS:
            with open(os.path.join(root, "count", f"run{r}.log"), "w") as f:
                for arm, th in (("native", 10), ("T1", 1), ("T10", 10), ("T20", 20)):
                    for shape, (_, m, n, k) in SHAPES.items():
                        e = EXPECT.get(th, {}).get(shape, (1, 1, 1, 1.0, 1.0))
                        vals = dict(rayon_threads=th, eff_threads=th, mr=8, nr=12, row_bands=e[1], col_bands=e[2], mc_job=8,
                                    nc_job=12, regions=1, jobs=e[0], workers_used=r, a_pack_tiles=1, a_pack_words=1,
                                    a_ideal_words=1, b_pack_tiles=1, b_pack_words=1, b_ideal_words=1, alloc_job=3, alloc_split=4,
                                    c_stage_words=2, a_repack_x=e[3], b_repack_x=e[4])
                        f.write(f"probe=jobgrid shape={shape} trans=nn m={m} n={n} k={k} arm={arm} "
                                + " ".join(f"{a}={b}" for a, b in vals.items()) + "\n")

        def bench(path, arm, series, shapes, threads):
            with open(path, "w") as f:
                f.write("# header\n")
                for shape, (t, m, n, k) in shapes.items():
                    f.write(f"arm={arm} series={series} shape={shape} trans={t} m={m} n={n} k={k} rayon_threads={threads} "
                            f"median_secs=0.001000000 gflops=1.0 checksum=0x{'ab' * 8}\n")
        for r in RUNS:
            bench(os.path.join(root, "st", f"run{r}.log"), "base", "threads", BENCH_SHAPES, 1)
            bench(os.path.join(root, "serial_ref", f"run{r}.log"), "base", "serial_ref",
                  {s: v for s, v in BENCH_SHAPES.items() if v[0] == "nn"}, 1)
            for a in ARMS:
                bench(os.path.join(root, "ab", f"run{r}-{a}.log"), a, "threads", BENCH_SHAPES, 10)
        if mutate:
            mutate(root)

    def expect_raise(label, mutate):
        with tempfile.TemporaryDirectory() as t:
            mk(t, mutate)
            try:
                machine_report(t, "fx")
            except AggError:
                return
            raise SystemExit(f"self-test 失敗: {label} が例外にならない")

    with tempfile.TemporaryDirectory() as t:
        mk(t)
        _, ok = machine_report(t, "fx")
        assert ok, "正常 fixture が ok でない"
    expect_raise("ログ欠損", lambda r: os.remove(os.path.join(r, "ab", "run3-cap4.log")))

    def bad_threads(r):
        p = os.path.join(r, "st", "run2.log")
        s = open(p).read().replace("rayon_threads=1 ", "rayon_threads=2 ", 1)
        open(p, "w").write(s)
    expect_raise("rayon_threads の食い違い", bad_threads)

    def bad_count(r):
        p = os.path.join(r, "count", "run4.log")
        s = open(p).read().replace("jobs=22 ", "jobs=23 ", 1)
        open(p, "w").write(s)
    expect_raise("計数の run 間不一致", bad_count)

    def nan_median(r):
        p = os.path.join(r, "ab", "run1-base.log")
        s = open(p).read().replace("median_secs=0.001000000", "median_secs=nan", 1)
        open(p, "w").write(s)
    expect_raise("数値でない median", nan_median)

    with tempfile.TemporaryDirectory() as t:
        def bad_cs(r):
            p = os.path.join(r, "ab", "run1-jpw1.log")
            s = open(p).read().replace("checksum=0x" + "ab" * 8, "checksum=0x" + "cd" * 8, 1)
            open(p, "w").write(s)
        mk(t, bad_cs)
        _, ok = machine_report(t, "fx")
        assert not ok, "checksum 不一致が検出されない"
    print("self-test ok")


def main(argv):
    if "--self-test" in argv:
        self_test()
        return 0
    here = os.path.dirname(os.path.abspath(__file__))
    root, machines = here, ["macmini", "gb10"]
    if "--dir" in argv:
        root = argv[argv.index("--dir") + 1]
    if "--machines" in argv:
        machines = argv[argv.index("--machines") + 1].split(",")
    parts = ["# #3177 集計（aggregate.py の出力。手で直さない）", ""]
    ok_all = True
    for m in machines:
        md, ok = machine_report(os.path.join(root, m), m)
        parts.append(md)
        ok_all = ok_all and ok
    sys.stdout.write("\n".join(parts) + "\n")
    return 0 if ok_all else 1


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except AggError as e:
        sys.stderr.write(f"集計失敗（fail-closed）: {e}\n")
        sys.exit(2)
