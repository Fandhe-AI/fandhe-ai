#!/usr/bin/env python3
"""イシュー #3177 の実測ログ（macmini／gb10）を fail-closed で集計し Markdown を標準出力へ出す。

判定規則は同ディレクトリ RULE.txt。採否は書かない（ratio と勝ち run 数を記録するだけ）。
欠損・数値でない値・checksum 不一致・rayon_threads の食い違いは例外にし exit 非 0 とする
（ただしゲート記録 load_gate.log が gate=unavailable と明記した (series, run, arm) だけはログ不在を許し、
数値を補完せず unavailable として出す＝参考系列。ゲート記録は期待 35 組と厳密照合する）
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


def check_dims(d, table, shape, path):
    """ログの trans・m・n・k が期待表と一致することを要求する（GFLOPS は表の寸法から算出するため）。"""
    trans, m, n, k = table[shape]
    got = (d.get("trans"), *(d.get(x) for x in ("m", "n", "k")))
    if got != (trans, str(m), str(n), str(k)):
        raise AggError(f"{path}: {shape} の寸法・転置が期待 {(trans, m, n, k)} と不一致: {got}")


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


def skip_unavailable(gate, key, path):
    """ゲートが unavailable と記録した組はログ不在が正（存在すれば矛盾として止める）。それ以外の欠損は read_lines が例外にする。"""
    if gate.get(key) == "unavailable":
        if os.path.exists(path):
            raise AggError(f"{path}: gate=unavailable なのにログが存在する")
        return True
    return False


def parse_count(root, gate):
    """arm -> shape -> run -> dict（unavailable の run は欠く）。利用可能 run 間で STABLE が全一致でなければ止める。"""
    data = {a: {s: {} for s in SHAPES} for a in COUNT_ARMS}
    for r in RUNS:
        p = os.path.join(root, "count", f"run{r}.log")
        if skip_unavailable(gate, ("count", r, "base"), p):
            continue
        rows = read_lines(p, r"probe=jobgrid ")
        seen = set()
        for d in rows:
            arm, shape = d.get("arm"), d.get("shape")
            if arm not in data or shape not in SHAPES:
                raise AggError(f"{p}: 未知の arm/shape: {arm}/{shape}")
            if (arm, shape) in seen:
                raise AggError(f"{p}: 重複: {arm}/{shape}")
            seen.add((arm, shape))
            check_dims(d, SHAPES, shape, p)
            rec = {k: to_int(d, k, p) for k in STABLE + ["workers_used"]}
            for k in ("a_repack_x", "b_repack_x"):
                rec[k] = to_float(d, k, p)
            data[arm][shape][r] = rec
        if len(seen) != len(COUNT_ARMS) * len(SHAPES):
            raise AggError(f"{p}: 腕×形状が揃っていない（{len(seen)} 行）")
    for arm in COUNT_ARMS:
        for shape in SHAPES:
            got = sorted(data[arm][shape])
            for r in got[1:]:
                base = data[arm][shape][got[0]]
                for k in STABLE:
                    if data[arm][shape][r][k] != base[k]:
                        raise AggError(f"計数が run 間で不一致: {arm}/{shape}/{k} run{got[0]}={base[k]} run{r}={data[arm][shape][r][k]}")
    return data


def parse_bench(root, sub, series, arm_of, shapes, gate, want_threads=None):
    """sub/run{N}[-arm].log -> {arm: {shape: {run: (median, checksum, rayon_threads)}}}"""
    out = {}
    for arm in arm_of:
        out[arm] = {s: {} for s in shapes}
        for r in RUNS:
            name = f"run{r}-{arm}.log" if sub == "ab" else f"run{r}.log"
            p = os.path.join(root, sub, name)
            if skip_unavailable(gate, (sub, r, arm), p):
                continue
            rows = read_lines(p, r"arm=")
            seen = set()
            for d in rows:
                if d.get("arm") != arm or d.get("series") != series:
                    raise AggError(f"{p}: arm/series が食い違い: {d.get('arm')}/{d.get('series')}")
                shape = d.get("shape")
                if shape not in shapes or shape in seen:
                    raise AggError(f"{p}: 未知または重複の shape: {shape}")
                seen.add(shape)
                check_dims(d, shapes, shape, p)
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


def expected_gate_keys():
    return ([("count", r, "base") for r in RUNS] + [("st", r, "base") for r in RUNS]
            + [("serial_ref", r, "base") for r in RUNS] + [("ab", r, a) for r in RUNS for a in ARMS])


def gate_status(root):
    """load_gate.log の START 行を期待 35 組 (series, run, arm) と厳密照合し {key: pass|unavailable} を返す。
    欠損・重複・未知の組・gate 値の不正は AggError（RULE.txt §8: 全プロセス通過のみ正式系列）。"""
    p = os.path.join(root, "load_gate.log")
    if not os.path.isfile(p):
        raise AggError(f"欠損: {p}")
    status = {}
    with open(p, encoding="utf-8") as f:
        for ln in f:
            if not ln.startswith("START "):
                continue
            d = dict(re.findall(r"(\w+)=(\S+)", ln))
            if not re.fullmatch(r"\d+", d.get("run", "")):
                raise AggError(f"{p}: START 行の run が不正: {ln!r}")
            key = (d.get("series"), int(d["run"]), d.get("arm"))
            if d.get("gate") not in ("pass", "unavailable"):
                raise AggError(f"{p}: START 行の gate が不正: {ln!r}")
            if key in status:
                raise AggError(f"{p}: ゲート記録が重複: {key}")
            status[key] = d["gate"]
    exp = set(expected_gate_keys())
    unknown = sorted(set(status) - exp, key=str)
    missing = sorted(exp - set(status), key=str)
    if unknown or missing:
        raise AggError(f"{p}: ゲート記録が期待 {len(exp)} 組と不一致: 欠損={missing} 未知={unknown}")
    return status


def require_files(root):
    for f in ("env_info.txt", ".rev-stamp", "uname.txt", "tree_verify.txt", "patch_sha256.txt", "DONE"):
        if not os.path.isfile(os.path.join(root, f)):
            raise AggError(f"欠損: {os.path.join(root, f)}")


def machine_report(root, name):
    """(markdown, ok)。ok=False は A/B の checksum 不一致など unavailable を含む。"""
    require_files(root)
    ok = True
    L = [f"## {name}", ""]
    gate = gate_status(root)
    passes = sum(1 for v in gate.values() if v == "pass")
    unavail = sum(1 for v in gate.values() if v == "unavailable")
    formal = passes == GATE_STARTS and unavail == 0
    L += [f"- 負荷ゲート: pass {passes}/{GATE_STARTS}・unavailable {unavail} -> "
          f"**{'正式系列' if formal else '参考系列（全プロセスがゲートを通っていない）'}**",
          f"- .rev-stamp: `{open(os.path.join(root, '.rev-stamp')).read().strip()[:60]}`", ""]

    cnt = parse_count(root, gate)
    cnt_runs = sorted(cnt["native"]["l1_fwd"])
    native_threads = cnt["native"]["l1_fwd"][cnt_runs[0]]["rayon_threads"] if cnt_runs else None
    L += [f"### 計数（利用可能 {len(cnt_runs)}/5 run 全一致。native の rayon_threads={native_threads if cnt_runs else 'unavailable'}）", "",
          "| 形状 | 腕 | job（行帯×列帯） | A 再 pack 倍率 | B 再 pack 倍率 | alloc_job | alloc_split | C staging 語 | workers_used（run1〜5） | 再計算との突合 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for shape in SHAPES:
        for arm in COUNT_ARMS:
            if not cnt_runs:
                L.append(f"| {shape} | {arm} | unavailable | - | - | - | - | - | - | unavailable |")
                continue
            r1 = cnt[arm][shape][cnt_runs[0]]
            th = {"native": native_threads, "T1": 1, "T10": 10, "T20": 20}[arm]
            exp = EXPECT.get(th, {}).get(shape)
            if exp is None:
                verdict = "期待表なし"
            else:
                got = (r1["jobs"], r1["row_bands"], r1["col_bands"], round(r1["a_repack_x"], 1), round(r1["b_repack_x"], 1))
                verdict = "一致" if got == exp else f"**不一致**（期待 {exp}）"
            ws = "/".join(str(cnt[arm][shape][r]["workers_used"]) if r in cnt[arm][shape] else "NA" for r in RUNS)
            L.append(f"| {shape} | {arm} | {r1['jobs']}（{r1['row_bands']}×{r1['col_bands']}） | {r1['a_repack_x']:.1f} | "
                     f"{r1['b_repack_x']:.1f} | {r1['alloc_job']} | {r1['alloc_split']} | {r1['c_stage_words']} | {ws} | {verdict} |")
    L.append("")

    st = parse_bench(root, "st", "threads", ["base"], BENCH_SHAPES, gate, want_threads=1)["base"]
    ref = parse_bench(root, "serial_ref", "serial_ref", ["base"],
                      {s: v for s, v in BENCH_SHAPES.items() if v[0] == "nn"}, gate)["base"]
    L += ["### 単スレッド実効 GFLOPS（RAYON_NUM_THREADS=1・利用可能 run の median_secs の中央値から算出。unavailable は補完しない）", "",
          "| 形状 | GFLOPS（中央値） | run1〜5 の GFLOPS | serial_ref の GFLOPS（参照） |", "|---|---|---|---|"]
    for shape in BENCH_SHAPES:
        meds = [st[shape][r][0] for r in RUNS if r in st[shape]]
        per = "/".join(f"{gflops(BENCH_SHAPES, shape, st[shape][r][0]):.1f}" if r in st[shape] else "unavailable" for r in RUNS)
        rf = "-"
        if shape in ref:
            rm = [ref[shape][r][0] for r in RUNS if r in ref[shape]]
            rf = f"{gflops(BENCH_SHAPES, shape, statistics.median(rm)):.1f}" if rm else "unavailable"
        med = f"{gflops(BENCH_SHAPES, shape, statistics.median(meds)):.1f}" if meds else "unavailable"
        L.append(f"| {shape} | {med} | {per} | {rf} |")
    L.append("")

    ab = parse_bench(root, "ab", "threads", ARMS, BENCH_SHAPES, gate)
    ab_all = {ab[a][s][r][2] for a in ARMS for s in BENCH_SHAPES for r in ab[a][s]}
    if len(ab_all) > 1 or (ab_all and native_threads is not None and ab_all != {native_threads}):
        raise AggError(f"{root}: A/B の rayon_threads={sorted(ab_all)} が native の計数値 {native_threads} と不一致")
    ab_threads = next(iter(ab_all)) if ab_all else "unavailable"
    L += [f"### A/B（既定プール・rayon_threads={ab_threads}・ratio = median(腕)/median(base)・採否は書かない。unavailable の run は補完せず、腕が勝った run は双方が利用可能な run のみで数える）", "",
          "| 形状 | base median（s） | 腕 | 腕 median（s） | ratio | 腕が勝った run（5 中） | checksum |", "|---|---|---|---|---|---|---|"]
    for shape in BENCH_SHAPES:
        sums = {ab[a][shape][r][1] for a in ARMS for r in ab[a][shape]}
        csum_ok = len(sums) <= 1
        if not csum_ok:
            ok = False
        bruns = [r for r in RUNS if r in ab["base"][shape]]
        bmed = statistics.median([ab["base"][shape][r][0] for r in bruns]) if bruns else None
        for arm in ARMS[1:]:
            aruns = [r for r in RUNS if r in ab[arm][shape]]
            if not csum_ok:
                L.append(f"| {shape} | - | {arm} | - | unavailable | - | **不一致**（{len(sums)} 種） |")
                continue
            if bmed is None or not aruns:
                bcell = "unavailable" if bmed is None else f"{bmed:.9f}"
                acell = "unavailable" if not aruns else "-"
                L.append(f"| {shape} | {bcell} | {arm} | {acell} | unavailable | - | - |")
                continue
            amed = statistics.median([ab[arm][shape][r][0] for r in aruns])
            both = [r for r in aruns if r in ab["base"][shape]]
            wins = sum(1 for r in both if ab[arm][shape][r][0] < ab["base"][shape][r][0])
            L.append(f"| {shape} | {bmed:.9f} | {arm} | {amed:.9f} | {amed / bmed:.3f} | {wins}（{len(both)} 中） | 一致 |")
    L.append("")
    return "\n".join(L), ok


def self_test():
    def write_gate(root, unavail=(), drop=(), extra=()):
        with open(os.path.join(root, "load_gate.log"), "w") as f:
            for k in list(expected_gate_keys()) + list(extra):
                if k in drop:
                    continue
                g = "unavailable" if k in unavail else "pass"
                f.write(f"START series={k[0]} run={k[1]} arm={k[2]} gate={g} load1=1.0 waited_s=0 at=t\n")

    def mk(root, mutate=None):
        os.makedirs(os.path.join(root, "count"))
        for d in ("st", "serial_ref", "ab"):
            os.makedirs(os.path.join(root, d))
        for f, txt in (("env_info.txt", "x"), (".rev-stamp", "abc"), ("uname.txt", "x"),
                       ("tree_verify.txt", "x"), ("patch_sha256.txt", "x"), ("DONE", "x")):
            open(os.path.join(root, f), "w").write(txt + "\n")
        write_gate(root)
        for r in RUNS:
            with open(os.path.join(root, "count", f"run{r}.log"), "w") as f:
                for arm, th in (("native", 10), ("T1", 1), ("T10", 10), ("T20", 20)):
                    for shape, (tr, m, n, k) in SHAPES.items():
                        e = EXPECT.get(th, {}).get(shape, (1, 1, 1, 1.0, 1.0))
                        vals = dict(rayon_threads=th, eff_threads=th, mr=8, nr=12, row_bands=e[1], col_bands=e[2], mc_job=8,
                                    nc_job=12, regions=1, jobs=e[0], workers_used=r, a_pack_tiles=1, a_pack_words=1,
                                    a_ideal_words=1, b_pack_tiles=1, b_pack_words=1, b_ideal_words=1, alloc_job=3, alloc_split=4,
                                    c_stage_words=2, a_repack_x=e[3], b_repack_x=e[4])
                        f.write(f"probe=jobgrid shape={shape} trans={tr} m={m} n={n} k={k} arm={arm} "
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
    expect_raise("START 欠落と重複（行数 35 のまま）",
                 lambda r: write_gate(r, drop=[("st", 1, "base")], extra=[("st", 2, "base")]))
    expect_raise("未知の組", lambda r: write_gate(r, drop=[("st", 1, "base")], extra=[("zz", 1, "base")]))
    expect_raise("unavailable なのにログが存在", lambda r: write_gate(r, unavail=[("st", 1, "base")]))

    def unavail_ok(r):
        keys = [("st", 2, "base"), ("count", 3, "base"), ("ab", 1, "cap4"), ("serial_ref", 5, "base")]
        write_gate(r, unavail=keys)
        os.remove(os.path.join(r, "st", "run2.log"))
        os.remove(os.path.join(r, "count", "run3.log"))
        os.remove(os.path.join(r, "ab", "run1-cap4.log"))
        os.remove(os.path.join(r, "serial_ref", "run5.log"))
    with tempfile.TemporaryDirectory() as t:
        mk(t, unavail_ok)
        md, ok = machine_report(t, "fx")
        assert ok and "参考系列" in md and "unavailable" in md, "参考系列が出力されない"
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

    def bad_ab_threads(r):
        p = os.path.join(r, "ab", "run3-cap4.log")
        s = open(p).read().replace("rayon_threads=10 ", "rayon_threads=20 ")
        open(p, "w").write(s)
    expect_raise("A/B の rayon_threads が native と不一致", bad_ab_threads)

    def bad_dims(r):
        p = os.path.join(r, "ab", "run2-jpw1.log")
        s = open(p).read().replace(" m=64 ", " m=65 ", 1)
        open(p, "w").write(s)
    expect_raise("bench ログの寸法不一致", bad_dims)

    def bad_count_dims(r):
        p = os.path.join(r, "count", "run2.log")
        s = open(p).read().replace(" trans=nn ", " trans=tn ", 1)
        open(p, "w").write(s)
    expect_raise("count ログの転置不一致", bad_count_dims)

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
