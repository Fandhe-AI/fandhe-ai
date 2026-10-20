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


def checked_cells(rows, expected, what, torch=False):
    """行を device／parity で検証し、セル集合が expected と完全一致（余分・欠損なし）か確かめて返す（RULE P-CELLS）。"""
    for r in rows:
        if r.get("device") != "cpu":
            raise AggError("%s: device が cpu ではない行: %r" % (what, r.get("device")))
        pf = r.get("parity_fail_count")
        # PyTorch は記録専用（RULE §7）。Accelerate を FMA 参照と比べる parity は非 0 になりうるので判定に使わない
        if not torch and pf is not None and (isinstance(pf, bool) or not isinstance(pf, int) or pf != 0):
            raise AggError("%s: parity_fail_count が 0 でない: %r" % (what, pf))
    cells = cells_of(rows, torch=torch)
    if set(cells) != set(expected):
        raise AggError("%s: セル集合が期待と完全一致しない（余分: %s／欠損: %s）" % (
            what, sorted(map(str, set(cells) - set(expected))), sorted(map(str, set(expected) - set(cells)))))
    for key, rs in cells.items():
        if len(rs) != ROUNDS:
            raise AggError("%s: セル %s が %d 件（%d 件を要求）" % (what, key, len(rs), ROUNDS))
    return cells


def ab_analysis(d, label):
    before, after = {}, {}
    for task in ("gemm", "train", "infer"):
        exp = [k for k in CELLS if k[0] == task]
        before[task] = checked_cells(read_jsonl(os.path.join(d, "results-before-%s-cpu-%s.jsonl" % (label, task))), exp, "before/" + task)
        after[task] = checked_cells(read_jsonl(os.path.join(d, "results-after-%s-cpu-%s.jsonl" % (label, task))), exp, "after/" + task)
    # PyTorch は記録専用。欠損・不正は A/B 判定（特に checksum FAIL）から切り離して注記のみにする（RULE §6・§7）
    torch, torch_note = None, None
    try:
        hits = glob.glob(os.path.join(d, "results-torch-3176-*-%s-cpu.jsonl" % label))
        if len(hits) != 1:
            raise AggError("torch の JSONL が 1 件ではない: %r" % hits)
        texp = sorted({(t, s, "fresh") for t, s, _ in CELLS})
        torch = checked_cells(read_jsonl(hits[0]), texp, "torch", torch=True)
    except (OSError, AggError) as e:
        torch_note = str(e)
    res = []
    for key in CELLS:
        task, size, _ = key
        b = series(before[task], key, "median_s", "before")
        a = series(after[task], key, "median_s", "after")
        bc = series(before[task], key, "checksum", "before")
        ac = series(after[task], key, "checksum", "after")
        if any(x <= 0 for x in a + b):
            raise AggError("median_s が正でない: %s" % (key,))
        t = None
        if torch is not None:
            try:
                t = series(torch, (task, size, "fresh"), "median_s", "torch")
                if any(x <= 0 for x in t):
                    t = None
            except AggError as e:
                torch_note = torch_note or str(e)
        rr = [x / y for x, y in zip(a, b)]
        res.append(dict(
            key=key, ratio=statistics.median(a) / statistics.median(b), rounds=rr,
            n_le=sum(1 for x in rr if x <= 1.0), n_gt=sum(1 for x in rr if x > 1.0),
            ck_ok=len(set(bc) | set(ac)) == 1,
            t_before=statistics.median([x / y for x, y in zip(t, b)]) if t else None,
            t_after=statistics.median([x / y for x, y in zip(t, a)]) if t else None,
        ))
    return res, torch_note


def checksum_mismatch(d, label):
    """before／after の checksum 不一致を、他セルの欠損・median_s 不正と独立に判定する（RULE §6: FAIL 最優先）。
    読めない行・ファイル・非数値は黙って飛ばし、読めた値だけで同一セル内に 2 値以上あれば True。"""
    seen = {}
    for task in ("gemm", "train", "infer"):
        for arm in ("before", "after"):
            path = os.path.join(d, "results-%s-%s-cpu-%s.jsonl" % (arm, label, task))
            try:
                rows = read_jsonl(path)
            except (OSError, AggError):
                continue
            for r in rows:
                v = r.get("checksum")
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                    continue
                seen.setdefault((r.get("task"), r.get("size"), r.get("mode", "fresh")), set()).add(float(v))
    return any(len(vs) > 1 for vs in seen.values())


def gate_series(d):
    """load_gate.log の round 行から正式系列（5/5 gate_ok=1）かを返す。"""
    paths = glob.glob(os.path.join(d, "r1r2", "load_gate-3176-*.log"))
    if len(paths) != 1:
        return False
    rounds = [ln for ln in read_lines(paths[0]) if ln.startswith("round")]
    return len(rounds) == ROUNDS and all("gate_ok=1" in ln for ln in rounds)


SHA_LINE = re.compile(r"^on-arm\.patch sha256: ([0-9a-f]{64})$")
LOAD1 = re.compile(r"^[0-9]+(\.[0-9]+)?$")


def rule_patch_sha():
    """同ディレクトリ RULE.txt の on-arm.patch sha256 固定値（取得できなければ AggError）。"""
    here = os.path.dirname(os.path.abspath(__file__))
    for ln in read_lines(os.path.join(here, "RULE.txt")):
        m = SHA_LINE.match(ln)
        if m:
            return m.group(1)
    raise AggError("RULE.txt から patch の sha256 を取得できない")


def kv(line):
    return dict(p.split("=", 1) for p in line.split() if "=" in p)


def orchestration_checks(d):
    """macmini/load_gate.log を検証する（RULE §4 P-R4-EXEC・P-RUNAB・P-COLLECT、§5 R4／rect の負荷ゲート、
    全 run の開始・終了記録）。欠損・重複・非ゼロ終了・ゲート不通過は理由リストへ返す。"""
    why = []
    try:
        lines = read_lines(os.path.join(d, "load_gate.log"))
    except OSError:
        return ["P-R4-EXEC: load_gate.log なし（R4・rect の負荷ゲート／開始・終了記録を検証できない）"]
    starts, ends, singles = {}, {}, {}
    for ln in lines:
        f = kv(ln)
        if ln.startswith("START "):
            starts.setdefault((f.get("series"), f.get("run")), []).append(f)
        elif ln.startswith("END "):
            ends.setdefault((f.get("series"), f.get("run")), []).append(f)
        elif ln.startswith("run_ab exit="):
            singles.setdefault("run_ab", []).append(f.get("exit"))
        elif ln.startswith("collect exit="):
            singles.setdefault("collect", []).append(f.get("exit"))
    for series in ("r4", "rect"):
        for r in range(1, ROUNDS + 1):
            key = (series, str(r))
            ss, es = starts.get(key, []), ends.get(key, [])
            if len(ss) != 1 or len(es) != 1:
                why.append("%s run%d: 開始 %d 件・終了 %d 件の記録（各 1 件を要求）" % (series, r, len(ss), len(es)))
                continue
            l1 = ss[0].get("load1", "")
            if ss[0].get("gate") != "pass" or not LOAD1.match(l1) or float(l1) >= 3.0:
                why.append("%s run%d: 負荷ゲート不通過（gate=%s load1=%s。record_only）" % (series, r, ss[0].get("gate"), l1))
            if es[0].get("exit") != "0" or es[0].get("grep_exit") != "0":
                why.append("%s run%d: 診断の終了コードが 0 でない（exit=%s grep_exit=%s）" % (series, r, es[0].get("exit"), es[0].get("grep_exit")))
    extra = sorted(k for k in set(starts) | set(ends) if k[0] not in ("r4", "rect") or k[1] not in [str(i) for i in range(1, ROUNDS + 1)])
    if extra:
        why.append("想定外の series／run の記録: %s" % extra)
    for name, label in (("run_ab", "P-RUNAB"), ("collect", "P-COLLECT")):
        v = singles.get(name, [])
        if v != ["0"]:
            why.append("%s: %s の終了コード記録が exit=0 の 1 件ではない（%r）" % (label, name, v))
    return why


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
    try:
        sha = read_lines(os.path.join(d, "patch_sha256.txt"))
        if sha != [rule_patch_sha()]:
            why.append("P-TREE: patch の sha256 が RULE 記載と一致しない")
    except (OSError, AggError) as e:
        why.append("P-TREE: patch sha256 を検証できない（%s）" % e)
    why.extend(orchestration_checks(d))
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
        data["ab"], data["torch_note"] = ab_analysis(os.path.join(d, "r1r2"), label)
    except (OSError, AggError) as e:
        why.append("P-CELLS: %s" % e)
    data["ck_fail"] = checksum_mismatch(os.path.join(d, "r1r2"), label)
    data["official"] = gate_series(d)
    if not data["official"]:
        why.append("系列: 5/5 round が gate_ok=1 ではない（record_only）")
    return why, data


def verdict(why, data):
    ab = data.get("ab")
    if data.get("ck_fail") or (ab and any(not r["ck_ok"] for r in ab)):
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
            L.append("| %s %d %s | %s | %.4f | %s | %d/5 | %s | %s | %s |" % (
                t, s, m, "到達" if r["key"] in REACHED else "非到達", r["ratio"],
                " ".join("%.3f" % x for x in r["rounds"]), r["n_le"],
                "一致" if r["ck_ok"] else "**不一致**",
                "%.3f" % r["t_before"] if r["t_before"] is not None else "n/a",
                "%.3f" % r["t_after"] if r["t_after"] is not None else "n/a"))
        if data.get("torch_note"):
            L += ["", "注記（PyTorch は記録専用で判定に影響しない）: " + data["torch_note"]]
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


def _fixture(d, after_scale=0.9, ck_after=1.5, sme_scale=1.5, probe_true=True, gate_ok=True, r4_gate_ok=True, run_ab_exit=0):
    os.makedirs(os.path.join(d, "r1r2"))
    with open(os.path.join(d, "tree_verify.txt"), "w") as f:
        f.write("./crates/backend-cpu/src/gemm_blis/mod.rs\n")
    with open(os.path.join(d, "gate_constant.txt"), "w") as f:
        f.write("1:const SME_PRODUCTION_ENABLED: bool = true;\n")
    for arm in ("before", "after"):
        with open(os.path.join(d, "sme_probe_%s.log" % arm), "w") as f:
            f.write("sme_report=Report { kernel_enabled: %s }\n" % ("true" if probe_true else "false"))
    with open(os.path.join(d, "patch_sha256.txt"), "w") as f:
        f.write(rule_patch_sha() + "\n")
    with open(os.path.join(d, "load_gate.log"), "w") as f:
        for series in ("r4", "rect"):
            for r in range(1, ROUNDS + 1):
                f.write("START series=%s run=%d gate=%s load1=%s waited_s=0 at=x\n" % (series, r, "pass" if r4_gate_ok else "unavailable", "0.5" if r4_gate_ok else "9.0"))
                f.write("END series=%s run=%d exit=0 grep_exit=0 end_load1=0.5\n" % (series, r))
        f.write("run_ab exit=%d\n" % run_ab_exit)
        f.write("collect exit=0\n")
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


def _drop_line(path, prefix):
    keep = [ln for ln in read_lines(path) if not ln.startswith(prefix)]
    with open(path, "w") as f:
        f.write("\n".join(keep) + "\n")


def _edit_jsonl(d, fn):
    p = os.path.join(d, "r1r2", "results-after-L-cpu-gemm.jsonl")
    rows = read_jsonl(p)
    fn(rows[0])
    _w(p, rows)


def _append_jsonl(d, row):
    p = os.path.join(d, "r1r2", "results-after-L-cpu-gemm.jsonl")
    _w(p, read_jsonl(p) + [row])


def self_test():
    cases = [
        (dict(), "ADOPT 候補"),
        (dict(after_scale=1.2), "REJECT"),
        (dict(ck_after=2.5), "FAIL"),
        (dict(probe_true=False), "undetermined"),
        (dict(gate_ok=False), "undetermined"),
        (dict(sme_scale=0.5), "undetermined"),
        (dict(r4_gate_ok=False), "undetermined"),
        (dict(run_ab_exit=1), "undetermined"),
    ]
    for kw, want in cases:
        with tempfile.TemporaryDirectory() as d:
            _fixture(d, **kw)
            text = render(d, "L")
            assert want in text.split("\n")[2], (kw, text.split("\n")[2])
    with tempfile.TemporaryDirectory() as d:  # torch 欠損 + checksum 不一致でも FAIL を失わない
        _fixture(d, ck_after=2.5)
        os.remove(os.path.join(d, "r1r2", "results-torch-3176-sme-L-cpu.jsonl"))
        assert "FAIL" in render(d, "L").split("\n")[2]
    with tempfile.TemporaryDirectory() as d:  # 別タスクの記録欠損・median_s 不正で ab_analysis が例外でも確認済み checksum 不一致は FAIL
        _fixture(d)
        _edit_jsonl(d, lambda r: r.update(checksum=9.5))
        os.remove(os.path.join(d, "r1r2", "results-after-L-cpu-infer.jsonl"))
        assert "FAIL" in render(d, "L").split("\n")[2]
    with tempfile.TemporaryDirectory() as d:
        _fixture(d)
        _edit_jsonl(d, lambda r: r.update(checksum=9.5, median_s=-1))
        assert "FAIL" in render(d, "L").split("\n")[2]
    with tempfile.TemporaryDirectory() as d:  # torch の parity_fail_count 非 0 は記録のみで ADOPT を妨げない
        _fixture(d)
        p = os.path.join(d, "r1r2", "results-torch-3176-sme-L-cpu.jsonl")
        rows = read_jsonl(p)
        for r in rows:
            r["parity_fail_count"] = 3
        _w(p, rows)
        assert "ADOPT 候補" in render(d, "L").split("\n")[2]
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
    for name, mut in (
        ("欠損 END", lambda d: _drop_line(os.path.join(d, "load_gate.log"), "END series=rect run=3")),
        ("patch sha 不一致", lambda d: open(os.path.join(d, "patch_sha256.txt"), "w").write("0" * 64 + "\n")),
        ("patch sha 欠損", lambda d: os.remove(os.path.join(d, "patch_sha256.txt"))),
        ("device 不一致", lambda d: _edit_jsonl(d, lambda r: r.update(device="gpu"))),
        ("parity 失敗", lambda d: _edit_jsonl(d, lambda r: r.update(parity_fail_count=2))),
        ("余分セル", lambda d: _append_jsonl(d, dict(task="gemm", size=4096, mode="fresh", device="cpu", median_s=0.01, checksum=1.5))),
    ):
        with tempfile.TemporaryDirectory() as d:
            _fixture(d)
            mut(d)
            assert "undetermined" in render(d, "L").split("\n")[2], name
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
