#!/usr/bin/env python3
"""イシュー #3176: Mac mini の SME ON／OFF A/B の厳密集計（R1 到達セル・R2 checksum・R4 格子・長方形・R5 PyTorch 比）。

役割: orchestrate_macmini.sh が収録した `macmini/` 配下のログと、run_ab_sme_macmini_cpu.sh の
JSONL（`macmini/r1r2/`）を読み、RULE.txt の評価順 FAIL > REJECT > ADOPT 候補 > undetermined を
機械的に当てはめた結果と各表を Markdown で標準出力へ出す。本ツールは RULE の適用器であり、
規則そのものは RULE.txt が正。本番定数（SME_PRODUCTION_ENABLED）は切り替えない。

入力の厳格性: 重複行・非有限値・未知のキー・点数の過不足・m≠n の R4 行・欠損は捨てず、
前提ゲート不成立（undetermined）へ写像する。python3 標準ライブラリのみ。

使い方:
  python3 -I aggregate.py <macmini-dir> [<label>]   # 既定 label は 3176-r1
  python3 -I aggregate.py --self-test
"""
import glob
import json
import math
import os
import re
import statistics
import sys
import tempfile

ROUNDS = 5
GEMM_SIZES = (512, 1024, 2048)
MODES = ("fresh", "reuse")
CELLS = [("gemm", s, m) for s in GEMM_SIZES for m in MODES] + [
    (t, 64, m) for t in ("train", "infer") for m in MODES
]
# 到達セル（m=n=k>=512 の gemm 6 + train size=64 の 2。RULE.txt §2）。infer は m=64 で非到達
REACHED = [c for c in CELLS if c[0] == "gemm" or c[0] == "train"]
R4_MN = (64, 128, 256, 512)
R4_K = (32, 64, 128, 256)
RECT_SHAPES = ((64, 256, 784), (784, 256, 64), (784, 64, 256), (64, 10, 256))
# 現行しきい値（sme_shape_eligible: m,n>=256 かつ k>=64）。参照のみで本番値は変えない
CUR_MIN_MN, CUR_MIN_K = 256, 64
LINE = re.compile(
    r"^variant=(NEON\(TwoDDynamic\)|SME\(TwoDDynamicSme\)) size=\((\d+),(\d+),(\d+)\) median_gflops=(\S+)$"
)


class AggError(Exception):
    pass


def read_lines(path):
    with open(path) as f:
        return [ln.rstrip("\n") for ln in f]


def parse_variant_log(path):
    """1 run のログ -> {(m,n,k): {"NEON": g, "SME": g}}。重複・非有限は AggError。"""
    pts = {}
    for ln in read_lines(path):
        if ln.startswith("num_threads="):
            continue
        mt = LINE.match(ln)
        if not mt:
            raise AggError("%s: 未知の行: %r" % (path, ln))
        kind = "SME" if mt.group(1).startswith("SME") else "NEON"
        shape = (int(mt.group(2)), int(mt.group(3)), int(mt.group(4)))
        g = float(mt.group(5))
        if not math.isfinite(g) or g <= 0:
            raise AggError("%s: 非有限の GFLOPS: %r" % (path, ln))
        d = pts.setdefault(shape, {})
        if kind in d:
            raise AggError("%s: 重複行 %s %s" % (path, kind, shape))
        d[kind] = g
    return pts


def load_series(d, prefix, expected):
    """5 run 分を読み、各 run が expected 形状集合をちょうど 1 回ずつ（NEON・SME 両方）含むか検証する。"""
    runs = []
    for r in range(1, ROUNDS + 1):
        p = os.path.join(d, "%s_run%d.log" % (prefix, r))
        if not os.path.isfile(p):
            raise AggError("欠損: %s" % p)
        pts = parse_variant_log(p)
        if set(pts) != set(expected) or any(set(v) != {"NEON", "SME"} for v in pts.values()):
            raise AggError("%s: 形状集合が期待と一致しない（%d 点）" % (p, len(pts)))
        runs.append(pts)
    return runs


def r4_analysis(runs):
    """格子点ごとの SME/NEON 比・5/5 で SME>=NEON か・パレート極小・K 別の参考領域。"""
    table = {}
    for mn in R4_MN:
        for k in R4_K:
            ratios = [r[(mn, mn, k)]["SME"] / r[(mn, mn, k)]["NEON"] for r in runs]
            table[(mn, k)] = dict(median=statistics.median(ratios), all_ge=all(x >= 1.0 for x in ratios))
    ok = {key for key, v in table.items() if v["all_ge"]}
    minimal = sorted(
        key for key in ok
        if not any(o != key and o[0] <= key[0] and o[1] <= key[1] for o in ok)
    )
    region = {}
    for kk in (64, 128, 256):
        pts = [(mn, k) for mn in (256, 512) for k in R4_K if k >= kk]
        region[kk] = all(p in ok for p in pts)
    current_ok = all((mn, k) in ok for mn in R4_MN if mn >= CUR_MIN_MN for k in R4_K if k >= CUR_MIN_K)
    return table, minimal, region, current_ok


def rect_analysis(runs):
    out = []
    for sh in RECT_SHAPES:
        ratios = [r[sh]["SME"] / r[sh]["NEON"] for r in runs]
        elig = sh[0] >= CUR_MIN_MN and sh[1] >= CUR_MIN_MN and sh[2] >= CUR_MIN_K
        out.append((sh, statistics.median(ratios), sum(1 for x in ratios if x >= 1.0), elig))
    return out


def read_jsonl(path):
    if not os.path.isfile(path):
        raise AggError("ファイルがありません: %s" % path)
    rows = []
    with open(path) as f:
        for i, ln in enumerate(f, 1):
            ln = ln.strip()
            if ln:
                try:
                    rows.append(json.loads(ln))
                except ValueError as e:
                    raise AggError("%s:%d JSON 不正: %s" % (path, i, e))
    return rows


def fnum(v, what):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise AggError("非数値の %s: %r" % (what, v))
    return float(v)


def cells_of(rows, torch=False):
    out = {}
    for r in rows:
        key = (r.get("task"), r.get("size"), "fresh" if torch else r.get("mode", "fresh"))
        out.setdefault(key, []).append(r)
    return out


def series(cells, key, field, what):
    rows = cells.get(key)
    if rows is None or len(rows) != ROUNDS:
        raise AggError("%s: セル %s が %s 件（%d 件を要求）" % (what, key, "欠損" if rows is None else len(rows), ROUNDS))
    return [fnum(r.get(field), "%s %s.%s" % (what, key, field)) for r in rows]


def ab_analysis(d, label):
    before, after = {}, {}
    for task in ("gemm", "train", "infer"):
        before[task] = cells_of(read_jsonl(os.path.join(d, "results-before-%s-cpu-%s.jsonl" % (label, task))))
        after[task] = cells_of(read_jsonl(os.path.join(d, "results-after-%s-cpu-%s.jsonl" % (label, task))))
    hits = glob.glob(os.path.join(d, "results-torch-3176-*-%s-cpu.jsonl" % label))
    if len(hits) != 1:
        raise AggError("torch の JSONL が 1 件ではない: %r" % hits)
    torch = cells_of(read_jsonl(hits[0]), torch=True)
    res = []
    for key in CELLS:
        task, size, _ = key
        b = series(before[task], key, "median_s", "before")
        a = series(after[task], key, "median_s", "after")
        bc = series(before[task], key, "checksum", "before")
        ac = series(after[task], key, "checksum", "after")
        if any(x <= 0 for x in a + b):
            raise AggError("median_s が正でない: %s" % (key,))
        t = series(torch, (task, size, "fresh"), "median_s", "torch")
        rr = [x / y for x, y in zip(a, b)]
        res.append(dict(
            key=key, ratio=statistics.median(a) / statistics.median(b), rounds=rr,
            n_le=sum(1 for x in rr if x <= 1.0), n_gt=sum(1 for x in rr if x > 1.0),
            ck_ok=len(set(bc) | set(ac)) == 1,
            t_before=statistics.median([x / y for x, y in zip(t, b)]),
            t_after=statistics.median([x / y for x, y in zip(t, a)]),
        ))
    return res


def gate_series(d):
    """load_gate.log の round 行から正式系列（5/5 gate_ok=1）かを返す。"""
    paths = glob.glob(os.path.join(d, "r1r2", "load_gate-3176-*.log"))
    if len(paths) != 1:
        return False
    rounds = [ln for ln in read_lines(paths[0]) if ln.startswith("round")]
    return len(rounds) == ROUNDS and all("gate_ok=1" in ln for ln in rounds)


def preconditions(d, label):
    """前提ゲートを 1 か所で評価する。戻り値: (理由リスト, 計算済みデータ)。"""
    why, data = [], {}
    try:
        tv = read_lines(os.path.join(d, "tree_verify.txt"))
        if tv != ["./crates/backend-cpu/src/gemm_blis/mod.rs"]:
            why.append("P-TREE: 差分ファイルが mod.rs の 1 件だけではない")
        if not any("SME_PRODUCTION_ENABLED: bool = true;" in ln for ln in read_lines(os.path.join(d, "gate_constant.txt"))):
            why.append("P-TREE: after の定数が true でない")
    except (OSError, AggError):
        why.append("P-TREE: 記録なし")
    try:
        for arm in ("before", "after"):
            if not any("kernel_enabled: true" in ln for ln in read_lines(os.path.join(d, "sme_probe_%s.log" % arm))):
                why.append("P-R0: %s で kernel_enabled が true でない" % arm)
    except OSError:
        why.append("P-R0: 記録なし")
    r4_shapes = [(mn, mn, k) for mn in R4_MN for k in R4_K] + [(mn, mn, k) for mn in R4_MN for k in R4_K]
    try:
        data["r4"] = r4_analysis(load_series(d, "r4", set(r4_shapes)))
    except (OSError, AggError) as e:
        why.append("P-R4-LOG: %s" % e)
    try:
        data["rect"] = rect_analysis(load_series(d, "rect", set(RECT_SHAPES)))
    except (OSError, AggError) as e:
        why.append("P-RECT-LOG: %s" % e)
    try:
        data["ab"] = ab_analysis(os.path.join(d, "r1r2"), label)
    except (OSError, AggError) as e:
        why.append("P-CELLS: %s" % e)
    data["official"] = gate_series(d)
    if not data["official"]:
        why.append("系列: 5/5 round が gate_ok=1 ではない（record_only）")
    return why, data


def verdict(why, data):
    ab = data.get("ab")
    if ab and any(not r["ck_ok"] for r in ab):
        return "FAIL（checksum 不一致。ADOPT／REJECT に数えない）"
    if why:
        return "undetermined（" + "；".join(why) + "）"
    reached = [r for r in data["ab"] if r["key"] in REACHED]
    if any(r["n_gt"] == ROUNDS for r in reached):
        return "REJECT（到達セルに 5/5 一貫の後退あり）"
    nonreached = [r for r in data["ab"] if r["key"] not in REACHED]
    if any(r["n_gt"] == ROUNDS for r in nonreached):
        return "REJECT（非到達セルに 5/5 一貫の後退あり）"
    if data["r4"][3] and all(r["ratio"] <= 1.0 and r["n_le"] == ROUNDS for r in reached):
        return "ADOPT 候補（本番切替は #3230、しきい値は #3231。本イシューでは切り替えない）"
    return "undetermined（ADOPT 条件が成立せず REJECT 条件にも当たらない）"


def render(d, label):
    why, data = preconditions(d, label)
    L = ["# 集計（イシュー #3176）", "", "総合判定: **%s**" % verdict(why, data), ""]
    if "ab" in data:
        L += ["## R1／R2／R5（10 セル）", "",
              "| セル | 到達 | after/before 中央値比 | round 別 | <=1.00 の round | checksum | torch÷fandhe(before) | torch÷fandhe(after) |",
              "|---|---|---|---|---|---|---|---|"]
        for r in data["ab"]:
            t, s, m = r["key"]
            L.append("| %s %d %s | %s | %.4f | %s | %d/5 | %s | %.3f | %.3f |" % (
                t, s, m, "到達" if r["key"] in REACHED else "非到達", r["ratio"],
                " ".join("%.3f" % x for x in r["rounds"]), r["n_le"],
                "一致" if r["ck_ok"] else "**不一致**", r["t_before"], r["t_after"]))
        reached = [r for r in data["ab"] if r["key"] in REACHED]
        ac1 = all(r["ratio"] <= 1.0 and r["n_le"] == ROUNDS for r in reached) and all(r["ck_ok"] for r in data["ab"])
        L += ["", "AC1 直接判定（到達セルの中央値 <=1.00 かつ 5/5、checksum 一致）: %s" % ("成立" if ac1 else "不成立"), ""]
    if "r4" in data:
        table, minimal, region, cur = data["r4"]
        L += ["## R4 格子（SME/NEON 中央値比。5/5 で >=1.0 の点に *）", "", "| min(m,n) \\ k | " + " | ".join(map(str, R4_K)) + " |",
              "|---|" + "---|" * len(R4_K)]
        for mn in R4_MN:
            L.append("| %d | " % mn + " | ".join("%.3f%s" % (table[(mn, k)]["median"], "*" if table[(mn, k)]["all_ge"] else "") for k in R4_K) + " |")
        L += ["", "パレート極小の採用候補 (min(m,n), k): %s" % (minimal or "なし"),
              "参考: K>=Kmin かつ min(m,n)∈{256,512} の全点が SME>=NEON: " + ", ".join("Kmin=%d: %s" % (k, "成立" if v else "不成立") for k, v in sorted(region.items())),
              "現行しきい値領域（min>=256 かつ k>=64）が全点成立: %s" % ("成立" if cur else "不成立"), ""]
    if "rect" in data:
        L += ["## 長方形 4 形状（記録のみ。判定に使わない）", "", "| 形状 (m,n,k) | SME/NEON 中央値比 | 5 run 中 >=1.0 | 現行 sme_shape_eligible |", "|---|---|---|---|"]
        for sh, med, n, el in data["rect"]:
            L.append("| %s | %.3f | %d/5 | %s |" % (sh, med, n, "可" if el else "否"))
        L.append("")
    L += ["## 前提ゲートの不成立理由", ""] + (["- " + w for w in why] or ["なし"]) + [""]
    return "\n".join(L)


# ---- self-test（合成入力）----
def _w(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _variant_log(path, shapes, sme_scale):
    with open(path, "w") as f:
        f.write("num_threads=10\n")
        for sh in shapes:
            f.write("variant=NEON(TwoDDynamic) size=(%d,%d,%d) median_gflops=100.0\n" % sh)
            f.write("variant=SME(TwoDDynamicSme) size=(%d,%d,%d) median_gflops=%.1f\n" % (sh + (100.0 * sme_scale,)))


def _fixture(d, after_scale=0.9, ck_after=1.5, sme_scale=1.5, probe_true=True, gate_ok=True):
    os.makedirs(os.path.join(d, "r1r2"))
    with open(os.path.join(d, "tree_verify.txt"), "w") as f:
        f.write("./crates/backend-cpu/src/gemm_blis/mod.rs\n")
    with open(os.path.join(d, "gate_constant.txt"), "w") as f:
        f.write("1:const SME_PRODUCTION_ENABLED: bool = true;\n")
    for arm in ("before", "after"):
        with open(os.path.join(d, "sme_probe_%s.log" % arm), "w") as f:
            f.write("sme_report=Report { kernel_enabled: %s }\n" % ("true" if probe_true else "false"))
    r4 = [(mn, mn, k) for mn in R4_MN for k in R4_K]
    for r in range(1, ROUNDS + 1):
        _variant_log(os.path.join(d, "r4_run%d.log" % r), r4, sme_scale)
        _variant_log(os.path.join(d, "rect_run%d.log" % r), RECT_SHAPES, sme_scale)
    for arm, sc in (("before", 1.0), ("after", after_scale)):
        for task in ("gemm", "train", "infer"):
            rows = []
            for rnd in range(ROUNDS):
                for t, s, m in CELLS:
                    if t == task:
                        ck = ck_after if (arm == "after" and t == "train" and rnd == 0) else 1.5
                        rows.append(dict(task=t, size=s, mode=m, device="cpu", median_s=0.01 * sc * (1 + rnd * 0.01), checksum=ck))
            _w(os.path.join(d, "r1r2", "results-%s-L-cpu-%s.jsonl" % (arm, task)), rows)
    _w(os.path.join(d, "r1r2", "results-torch-3176-sme-L-cpu.jsonl"),
       [dict(task=t, size=s, device="cpu", median_s=0.02, checksum=1.0) for _ in range(ROUNDS) for t, s, m in CELLS if m == "fresh"])
    with open(os.path.join(d, "r1r2", "load_gate-3176-sme-cpu-L.log"), "w") as f:
        f.write("threshold=3.0 rule_threshold=3.0 series=official\n")
        for i in range(1, ROUNDS + 1):
            f.write("round%d gate=%s gate_ok=%d\n" % (i, "pass" if gate_ok else "timeout", 1 if gate_ok else 0))


def self_test():
    cases = [
        (dict(), "ADOPT 候補"),
        (dict(after_scale=1.2), "REJECT"),
        (dict(ck_after=2.5), "FAIL"),
        (dict(probe_true=False), "undetermined"),
        (dict(gate_ok=False), "undetermined"),
        (dict(sme_scale=0.5), "undetermined"),
    ]
    for kw, want in cases:
        with tempfile.TemporaryDirectory() as d:
            _fixture(d, **kw)
            text = render(d, "L")
            assert want in text.split("\n")[2], (kw, text.split("\n")[2])
    with tempfile.TemporaryDirectory() as d:  # 異常入力（重複行）は捨てず undetermined
        _fixture(d)
        with open(os.path.join(d, "r4_run1.log"), "a") as f:
            f.write("variant=SME(TwoDDynamicSme) size=(64,64,32) median_gflops=1.0\n")
        assert "undetermined" in render(d, "L").split("\n")[2]
    with tempfile.TemporaryDirectory() as d:  # 非有限値
        _fixture(d)
        with open(os.path.join(d, "rect_run2.log"), "a") as f:
            f.write("variant=SME(TwoDDynamicSme) size=(1,2,3) median_gflops=nan\n")
        assert "undetermined" in render(d, "L").split("\n")[2]
    print("self-test ok")


def main(argv):
    if argv[1:] == ["--self-test"]:
        self_test()
        return 0
    if len(argv) not in (2, 3):
        print(__doc__, file=sys.stderr)
        return 2
    print(render(argv[1], argv[2] if len(argv) == 3 else "3176-r1"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
