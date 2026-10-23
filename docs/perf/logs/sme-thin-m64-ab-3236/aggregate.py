#!/usr/bin/env python3
"""イシュー #3236: m=64 専用 SME 薄型カーネル（`SME_THIN_M64_ENABLED`）の ON/OFF A/B（Mac mini）の厳密集計。

役割: RULE.txt（同ディレクトリ・計測前に固定）の判定を、実測ログから機械的に出力する。
`docs/perf/logs/sme-default-on-ab-3232/aggregate.py` の派生で、framework-compare の 4 セル（train／infer size=64 x fresh/reuse）に
GEMM 単体（cpu_jobgrid_bench）と K 格子（cpu_sme_thin_m64_kgrid）を加え、ADOPT-k（k 下限つき）を判定する。
標準ライブラリのみ。`python3 -I` で起動すること。結果の Markdown は手で直さない。

使い方:
    python3 -I aggregate.py <subdir>     # subdir は本ファイルのあるディレクトリからの相対（例: macmini）
    python3 -I aggregate.py --self-test

腕の向き: before = main（OFF）、after = on-arm.patch（ON）。比 = after / before（1.00 以下なら ON が速い）。
異常入力（重複・欠損・非有限値・型不正・ログ欠損）は捨てずに problems へ収集し undetermined にする（寛容なパースはしない）。
checksum が完全一致しない場合は FAIL（性能判定へ進まない。RULE §3）。
評価順: FAIL > 前提不成立 → undetermined > REJECT / ADOPT-k > ADOPT > undetermined。
"""
import glob
import json
import math
import os
import re
import statistics
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROUNDS = 5
MODES = ("fresh", "reuse")
GEMM_REACH = ("l1_fwd", "l1_dx", "l2_dx")
GEMM_GUARD = ("l2_fwd", "l1_dw", "l2_dw", "ref_square_512")
GEMM_SHAPES = GEMM_REACH + GEMM_GUARD
FC_CELLS = [(t, 64, m) for t in ("train", "infer") for m in MODES]
KS = (10, 16, 32, 64, 128, 256, 784)
# k < 64 の到達 GEMM セル（ADOPT-k で後退を許す対象。現状 l2_dx の k=10 のみ）
SMALL_K_REACH = ("l2_dx",)
GATE_FILE = "crates/backend-cpu/src/gemm_blis/mod.rs"
GATE_CONST = "SME_THIN_M64_ENABLED"
# 系列 B（0.11.0 registry 版の値。本 run の値ではない）: fandhe の対 PyTorch 倍率（torch median / fandhe median）
# GEMM 単体の各 shape の (転置, m, n, k)。cpu_jobgrid_bench.rs の SHAPES・RULE §2 と一致させる（P-CELLS の契約）
JOBGRID_SPEC = {
    "l1_fwd": ("nn", 64, 256, 784), "l1_dw": ("tn", 784, 256, 64), "l1_dx": ("nt", 64, 784, 256),
    "l2_fwd": ("nn", 64, 10, 256), "l2_dw": ("tn", 256, 10, 64), "l2_dx": ("nt", 64, 256, 10),
    "ref_square_512": ("nn", 512, 512, 512),
}
KGRID_M, KGRID_N = 64, 256
SERIES_B = {("train", 64): 0.20, ("infer", 64): 0.25}


class AggError(Exception):
    pass


def rule_sha():
    for line in open(os.path.join(HERE, "RULE.txt"), encoding="utf-8"):
        m = re.match(r"^on-arm\.patch sha256: ([0-9a-f]{64})$", line.rstrip("\n"))
        if m:
            return m.group(1)
    raise AggError("RULE.txt から patch の sha256 を取得できない")


def read_text(path):
    if not os.path.isfile(path):
        raise AggError("ログ欠損: " + os.path.basename(path))
    return open(path, encoding="utf-8").read()


def finite_num(v):
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return False
    try:
        return math.isfinite(v)
    except (OverflowError, ValueError, TypeError):
        return False


def find_one(pattern):
    hits = sorted(glob.glob(pattern))
    if len(hits) != 1:
        raise AggError("ファイルが 1 件でない（%d 件）: %s" % (len(hits), os.path.basename(pattern)))
    return hits[0]


# ---------------------------------------------------------------- framework-compare（JSONL）
def load_fc(ab, arm, problems, ck_all):
    out = {}
    for task in ("train", "infer"):
        try:
            path = find_one(os.path.join(ab, "results-%s-*-cpu-%s.jsonl" % (arm, task)))
            lines = read_text(path).splitlines()
        except AggError as e:
            problems.append("arm=%s: %s" % (arm, e))
            continue
        for i, line in enumerate(lines, 1):
            if not line.strip():
                continue
            where = "%s:%d" % (os.path.basename(path), i)
            try:
                o = json.loads(line)
            except ValueError:
                problems.append(where + " JSON 不正")
                continue
            if not isinstance(o, dict) or o.get("task") != task or o.get("device") != "cpu":
                problems.append(where + " 行の形・task・device が不正")
                continue
            size, mode = o.get("size"), o.get("mode", "fresh")
            if isinstance(size, bool) or not isinstance(size, int) or not isinstance(mode, str) or (task, size, mode) not in FC_CELLS:
                problems.append(where + " 未知のセル")
                continue
            if finite_num(o.get("checksum")):
                ck_all.setdefault((task, size, mode), []).append(o["checksum"])
            if not finite_num(o.get("median_s")) or o["median_s"] <= 0 or not finite_num(o.get("checksum")):
                problems.append(where + " median_s／checksum が不正")
                continue
            out.setdefault((task, size, mode), []).append(o)
    for cell in FC_CELLS:
        n = len(out.get(cell, []))
        if n != ROUNDS:
            problems.append("P-CELLS 不成立: arm=%s cell=%s は %d 行（%d 行ちょうどを要求）" % (arm, cell, n, ROUNDS))
    return out


# ---------------------------------------------------------------- GEMM 単体・K 格子（key=value 行）
def parse_kv(line):
    d = {}
    for tok in line.split():
        if "=" not in tok:
            raise ValueError("トークン不正: " + tok)
        k, v = tok.split("=", 1)
        if k in d:
            raise ValueError("キー重複: " + k)
        d[k] = v
    return d


def load_grid(gdir, kind, arm, shapes, problems, ck_all):
    """kind は 'jobgrid'／'kgrid'。round ごとに gemm/<kind>-<arm>-r<N>.txt（1 round 1 ファイル）を読む。"""
    out = {}
    for r in range(1, ROUNDS + 1):
        path = os.path.join(gdir, "%s-%s-r%d.txt" % (kind, arm, r))
        try:
            lines = read_text(path).splitlines()
        except AggError as e:
            problems.append("arm=%s: %s" % (arm, e))
            continue
        seen = set()
        for i, line in enumerate(lines, 1):
            if not line.strip():
                continue
            where = "%s:%d" % (os.path.basename(path), i)
            try:
                d = parse_kv(line)
                shape = d["shape"]
                med = float(d["median_secs"])
                ck = d["checksum"]
                row_arm = d["arm"]
                dims = tuple(int(d[x]) for x in ("m", "n", "k"))
                trans = d["trans"] if kind == "jobgrid" else None
            except (ValueError, KeyError) as e:
                problems.append(where + " 行が不正: %s" % e)
                continue
            if row_arm != arm:
                problems.append(where + " arm 不一致: 行=%s 期待=%s（別腕のログの誤収録）" % (row_arm, arm))
                continue
            if shape not in shapes:
                problems.append(where + " 未知の shape: " + shape)
                continue
            if kind == "jobgrid":
                want = JOBGRID_SPEC[shape]
                got = (trans,) + dims
            else:
                want = (None, KGRID_M, KGRID_N, int(shape[len("kgrid_k"):]))
                got = (None,) + dims
            if got != want:
                problems.append(where + " shape 契約違反: %s は 期待 %s・実際 %s" % (shape, want, got))
                continue
            if shape in seen:
                problems.append(where + " shape 重複: " + shape)
                continue
            seen.add(shape)
            if d.get("series", "threads") != "threads":
                continue
            if not re.fullmatch(r"0x[0-9a-f]{16}", ck):
                problems.append(where + " checksum 形式不正")
                continue
            ck_all.setdefault((kind, shape), []).append(ck)
            if not math.isfinite(med) or med <= 0:
                problems.append(where + " median_secs が不正")
                continue
            out.setdefault(shape, []).append(med)
        for s in shapes:
            if s not in seen:
                problems.append("P-CELLS 不成立: %s round %d arm=%s に %s がない" % (kind, r, arm, s))
    for s in shapes:
        if len(out.get(s, [])) != ROUNDS and not any("P-CELLS" in p and s in p for p in problems):
            problems.append("P-CELLS 不成立: %s arm=%s %s は %d 行" % (kind, arm, s, len(out.get(s, []))))
    return out


def load_torch(ab):
    """PyTorch は記録専用（RULE §5）。欠測・不足・不正は (None, 注記) とし本体判定へ波及させない。"""
    hits = sorted(glob.glob(os.path.join(ab, "results-torch-*-cpu.jsonl")))
    if not hits:
        return None, "PyTorch 未計測（記録のみのため判定には影響しない）"
    by = {}
    try:
        lines = open(hits[0], encoding="utf-8").read().splitlines()
    except (OSError, ValueError) as e:
        return None, "PyTorch ログを読めない（記録のみ）: %s" % e
    for line in lines:
        if not line.strip():
            continue
        try:
            o = json.loads(line)
        except ValueError:
            return None, "PyTorch 行が不正（記録のみ）"
        if not isinstance(o, dict) or o.get("task") not in ("train", "infer") or not finite_num(o.get("median_s")) \
                or o["median_s"] <= 0 or isinstance(o.get("size"), bool) or not isinstance(o.get("size"), int):
            return None, "PyTorch 行が不正（記録のみ）"
        by.setdefault((o["task"], o["size"]), []).append(o["median_s"])
    for key in (("train", 64), ("infer", 64)):
        if len(by.get(key, [])) != ROUNDS:
            return None, "PyTorch の %s は %d 行（相対比を省略。記録のみ）" % (key, len(by.get(key, [])))
    return by, None


def fmt(x):
    return "%.3f" % x


def ratio_stats(before, after):
    ratios = [after[i] / before[i] for i in range(ROUNDS)]
    return {"ratios": ratios, "median": statistics.median(ratios),
            "n_le": sum(1 for r in ratios if r <= 1.0), "n_gt": sum(1 for r in ratios if r > 1.0),
            "off_med": statistics.median(before), "on_med": statistics.median(after)}


def cell_ok(s):
    return s["median"] <= 1.0 and s["n_le"] == ROUNDS


def evaluate(base):
    ab, gdir = os.path.join(base, "ab"), os.path.join(base, "gemm")
    problems, ck_all = [], {}
    fc = {a: load_fc(ab, a, problems, ck_all) for a in ("before", "after")}
    jg = {a: load_grid(gdir, "jobgrid", a, GEMM_SHAPES, problems, ck_all) for a in ("before", "after")}
    kg = {a: load_grid(gdir, "kgrid", a, ["kgrid_k%d" % k for k in KS], problems, ck_all) for a in ("before", "after")}
    res = {"gate_fail": list(problems), "notes": [], "incomplete": bool(problems)}
    res["checksum_bad"] = [c for c, v in ck_all.items() if len(set(v)) > 1]
    gf = res["gate_fail"]
    if problems:
        gf.append("前提不成立: 計測ログの欠測・不足のため性能判定を行わない")

    def rd(name):
        try:
            return read_text(os.path.join(base, name))
        except AggError as e:
            gf.append("前提不成立: %s" % e)
            return None

    t = rd("tree_verify.txt")
    if t is not None and t.strip() != GATE_FILE:
        gf.append("P-TREE: 差分ファイルが 1 件（%s）ではない" % GATE_FILE)
    t = rd("patch_sha256.txt")
    if t is not None:
        try:
            if t.strip() != rule_sha():
                gf.append("P-TREE: patch の sha256 が RULE と不一致")
        except (AggError, OSError) as e:
            gf.append("前提不成立: %s" % e)
    t = rd("gate_constant.txt")
    if t is not None:
        consts = [re.sub(r"^\d+:", "", l) for l in t.splitlines()]
        if consts != ["const %s: bool = false;" % GATE_CONST, "const %s: bool = true;" % GATE_CONST]:
            gf.append("P-TREE: 定数行が before=false・after=true ではない")
    t = rd("sme_report.txt")
    if t is not None:
        for arm in ("before", "after"):
            m = re.search(r"^%s: .*kernel_enabled: (true|false)" % arm, t, re.M)
            if not m or m.group(1) != "true":
                gf.append("P-R0: %s の kernel_enabled が true と確認できない" % arm)
    lg = rd("load_gate.log")
    lg_lines = lg.splitlines() if lg is not None else []
    if lg is not None:
        for key in ("p_bit exit=0", "gemm exit=0", "run_ab exit=0", "collect exit=0"):
            if key not in lg_lines:
                gf.append("P-BIT／P-RUNAB／P-COLLECT: '%s' がない" % key)
    hw = rd("hwlock.log")
    if hw is not None and (not re.search(r"^acquired at=", hw, re.M) or not re.search(r"^released-pending at=", hw, re.M)):
        gf.append("P-LOCK: hwlock.log に取得・解放の記録がない")
    t = rd("rev-stamp.txt")
    if t is not None and len(t.split()) != 2:
        gf.append("rev-stamp が 2 行（main と on-arm）ではない")
    gate_logs = [("framework-compare", None), ("gemm", os.path.join(gdir, "load_gate.log"))]
    try:
        gate_logs[0] = ("framework-compare", find_one(os.path.join(ab, "load_gate-*-cpu-*.log")))
    except AggError as e:
        gf.append("前提不成立: %s" % e)
        gate_logs[0] = None
    for item in gate_logs:
        if item is None or item[1] is None:
            continue
        try:
            txt = read_text(item[1])
        except AggError as e:
            gf.append("前提不成立: %s" % e)
            continue
        ok = re.findall(r"^round ?(\d) gate=pass gate_ok=1 ", txt, re.M)
        if sorted(ok) != ["1", "2", "3", "4", "5"]:
            gf.append("P-GATE: %s の負荷ゲートが 5/5 round で gate_ok=1 ではない（通過 round=%s）" % (item[0], ",".join(ok)))
    torch, note = load_torch(ab)
    res["torch"] = None
    res["stats"], res["kstats"], res["fc_stats"] = {}, {}, {}
    if problems:
        res["consistent"], res["verdict"], res["K"] = [], ("FAIL" if res["checksum_bad"] else "undetermined"), None
        res["notes"].append("計測ログ欠測のためセルごとの比・相手比は算出しない")
        return res
    for s in GEMM_SHAPES:
        res["stats"][("gemm", s)] = ratio_stats(jg["before"][s], jg["after"][s])
    for c in FC_CELLS:
        res["fc_stats"][c] = ratio_stats([o["median_s"] for o in fc["before"][c]], [o["median_s"] for o in fc["after"][c]])
    for k in KS:
        res["kstats"][k] = ratio_stats(kg["before"]["kgrid_k%d" % k], kg["after"]["kgrid_k%d" % k])
    if torch is None:
        res["notes"].append(note)
    else:
        res["torch"] = {c: {a: statistics.median(torch[(c[0], 64)][i] / [o["median_s"] for o in fc[a][c]][i] for i in range(ROUNDS))
                            for a in ("before", "after")} for c in FC_CELLS}
    all_cells = {**res["stats"], **res["fc_stats"]}
    consistent = [c for c, s in all_cells.items() if s["n_gt"] == ROUNDS]
    res["consistent"] = consistent
    fc_ok = all(cell_ok(res["fc_stats"][c]) for c in FC_CELLS)
    gemm_reach_ok = all(cell_ok(res["stats"][("gemm", s)]) for s in GEMM_REACH)
    # K 下限: k >= K の全点が 5/5 で ON <= OFF になる最小の K（格子上）
    K = None
    for i, k in enumerate(KS):
        if all(res["kstats"][kk]["n_le"] == ROUNDS for kk in KS[i:]):
            K = k
            break
    res["K"] = K
    allowed = {("gemm", s) for s in SMALL_K_REACH}
    if res["checksum_bad"]:
        res["verdict"] = "FAIL"
    elif gf:
        res["verdict"] = "undetermined"
    elif consistent and not set(consistent) <= allowed:
        res["verdict"] = "REJECT"
    elif consistent:
        if fc_ok and K is not None and K > KS[0]:
            res["verdict"] = "ADOPT-k（k >= %d）" % K
        else:
            # RULE.txt §4: K 下限を格子上で決められない（K is None）場合は REJECT も ADOPT-k も主張せず undetermined
            res["verdict"] = "REJECT" if not fc_ok else "undetermined"
            if K is None and fc_ok:
                res["notes"].append("K 格子の全点で ON が後退し K 下限を決められない（undetermined）")
            if K is not None and K <= KS[0]:
                res["notes"].append("K 格子では全 k で ON <= OFF だが GEMM 単体 l2_dx は後退しており矛盾（undetermined）")
    elif fc_ok and gemm_reach_ok:
        res["verdict"] = "ADOPT"
    else:
        res["verdict"] = "undetermined"
    return res


def render(res):
    L = ["# #3236 厳密集計 (macmini)", "",
         "機械出力（`python3 -I aggregate.py` の出力。手で直さない）。腕: before = main（OFF）・after = on-arm（ON）。比 = ON / OFF（<= 1.00 なら ON が速い）。", ""]
    if res["incomplete"]:
        L += ["計測ログの欠測・不足のためセルごとの比・相手比は算出しない（前提ゲート・FAIL 確認のみ実施）。", ""]
    else:
        L += ["## セルごと（5 round 中央値）", "",
              "| セル | 到達区分 | OFF 中央値 | ON 中央値 | ON/OFF 中央値比 | round 別比 | <=1.00 | >1.00 |", "|---|---|---|---|---|---|---|---|"]
        rows = []
        for s in GEMM_SHAPES:
            rows.append(("gemm %s" % s, "到達" if s in GEMM_REACH else "ガード（非到達）", res["stats"][("gemm", s)]))
        for c in FC_CELLS:
            rows.append(("%s %d %s" % c, "到達", res["fc_stats"][c]))
        for k in KS:
            rows.append(("kgrid k=%d（記録用）" % k, "K 格子", res["kstats"][k]))
        for name, cls, s in rows:
            L.append("| %s | %s | %.6g | %.6g | %s | %s | %d | %d |" % (
                name, cls, s["off_med"], s["on_med"], fmt(s["median"]),
                " ".join("r%d=%s" % (i + 1, fmt(r)) for i, r in enumerate(s["ratios"])), s["n_le"], s["n_gt"]))
        L += ["", "## 相手（PyTorch）比: PyTorch median / fandhe median の 5 round 中央値（記録のみ・判定に使わない）", ""]
        if res["torch"] is None:
            L.append("PyTorch 相手比なし（未計測・欠測。注記参照）。GEMM 単体は同 round の相手実測なし。")
        else:
            L += ["| セル | OFF | ON | 系列 B（参考。本 run の値ではない） |", "|---|---|---|---|"]
            for c in FC_CELLS:
                t = res["torch"][c]
                L.append("| %s %d %s | %s | %s | %.2f |" % (c + (fmt(t["before"]), fmt(t["after"]), SERIES_B[(c[0], c[1])])))
            L.append("")
            L.append("GEMM 単体は同 round の相手実測なし。")
    L += ["", "## 前提ゲート", ""]
    L += ["- " + g for g in res["gate_fail"]] if res["gate_fail"] else ["- すべて成立"]
    if res["checksum_bad"]:
        L += ["", "## checksum 不一致（FAIL）", ""] + ["- %s" % (c,) for c in res["checksum_bad"]]
    L += ["", "## 後退と K 下限", "",
          "- 5/5 一貫の後退セル: %s" % (", ".join(str(c) for c in res["consistent"]) or "なし"),
          "- K 下限（k >= K の全点が 5/5 で ON <= OFF となる最小の K。格子上）: %s" % (res["K"] if res["K"] is not None else "なし")]
    for n in res["notes"]:
        L.append("- 注記: " + n)
    L += ["", "## 総合判定", "", "**macmini**: %s" % res["verdict"], ""]
    return "\n".join(L)


def run(subdir, out=sys.stdout):
    try:
        res = evaluate(os.path.join(HERE, subdir))
    except AggError as e:
        out.write("# #3236 厳密集計 (macmini)\n\n**undetermined**: %s\n" % e)
        return "undetermined"
    out.write(render(res))
    return res["verdict"]


# ---------------------------------------------------------------- self-test
def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w", encoding="utf-8").write(text)


def _fixture(root, on_ratio=0.9, bad_arm=False, bad_dim=False, ck_bad=None, drop=None, dup=False, gate_ok=True, regress=None, kmin=None, rm_tree=False, torch=True):
    """regress: 5/5 後退させる GEMM shape 名。kmin: K 格子で ON が遅くなる k の上限（k < kmin が後退）。"""
    ab, gd = os.path.join(root, "ab"), os.path.join(root, "gemm")
    for arm in ("before", "after"):
        for task in ("train", "infer"):
            lines = []
            for (_, size, mode) in [c for c in FC_CELLS if c[0] == task]:
                for r in range(ROUNDS):
                    t = 0.001 * (1 if task == "train" else 2) * (on_ratio if arm == "after" else 1.0)
                    ck = 1.5 + (0.1 if (ck_bad == "fc" and arm == "after" and task == "train" and mode == "fresh") else 0)
                    lines.append(json.dumps({"task": task, "device": "cpu", "size": size, "mode": mode, "median_s": t, "checksum": ck}))
            if drop == "fc" and arm == "after" and task == "train":
                lines.pop()
            if dup and arm == "after" and task == "train":
                lines.append(lines[0])
            _write(os.path.join(ab, "results-%s-3236-r1-cpu-%s.jsonl" % (arm, task)), "\n".join(lines) + "\n")
        for r in range(1, ROUNDS + 1):
            jl, kl = [], []
            for s in GEMM_SHAPES:
                ratio = on_ratio if s in GEMM_REACH else 1.0
                if regress == s:
                    ratio = 1.2
                ck = 0x1000 + (1 if (ck_bad == "gemm" and arm == "after" and s == "l1_fwd") else 0)
                if not (drop == "gemm" and arm == "after" and s == "l2_dw" and r == 1):
                    jl.append("arm=%s series=threads shape=%s trans=%s m=%d n=%d k=%d rayon_threads=10 median_secs=%.9f gflops=1.0 checksum=0x%016x"
                              % ((arm, s) + JOBGRID_SPEC[s] + ( 0.001 * (ratio if arm == "after" else 1.0), ck)))
            for k in KS:
                ratio = on_ratio
                if kmin is not None and k < kmin:
                    ratio = 1.2
                kl.append("arm=%s shape=kgrid_k%d m=64 n=256 k=%d rayon_threads=10 median_secs=%.9f gflops=1.0 checksum=0x%016x"
                          % (arm, k, k, 0.001 * (ratio if arm == "after" else 1.0), 0x2000 + k))
            if bad_arm and arm == "after" and r == 1:
                jl = [x.replace("arm=after", "arm=before") for x in jl]
            if bad_dim and arm == "after" and r == 1:
                jl = [x.replace("trans=nn m=64 n=256 k=784", "trans=nn m=64 n=256 k=785") for x in jl]
            _write(os.path.join(gd, "jobgrid-%s-r%d.txt" % (arm, r)), "\n".join(jl) + "\n")
            _write(os.path.join(gd, "kgrid-%s-r%d.txt" % (arm, r)), "\n".join(kl) + "\n")
    g = "".join("round%d gate=%s gate_ok=%d threshold=3.0 load1=0.5 waited_s=0 at=x\n" % (r, "pass" if gate_ok else "timeout", 1 if gate_ok else 0) for r in range(1, 6))
    _write(os.path.join(ab, "load_gate-3236-sme-cpu-3236-r1.log"), g)
    _write(os.path.join(gd, "load_gate.log"), g)
    if torch:
        tl = [json.dumps({"task": t, "device": "cpu", "size": 64, "median_s": 0.002}) for _ in range(ROUNDS) for t in ("train", "infer")]
        _write(os.path.join(ab, "results-torch-3236-sme-3236-r1-cpu.jsonl"), "\n".join(tl) + "\n")
    if not rm_tree:
        _write(os.path.join(root, "tree_verify.txt"), GATE_FILE + "\n")
    _write(os.path.join(root, "patch_sha256.txt"), rule_sha() + "\n")
    _write(os.path.join(root, "gate_constant.txt"), "3601:const %s: bool = false;\n3601:const %s: bool = true;\n" % (GATE_CONST, GATE_CONST))
    _write(os.path.join(root, "sme_report.txt"), "before: sme_report= kernel_enabled: true\nafter: sme_report= kernel_enabled: true\n")
    _write(os.path.join(root, "load_gate.log"), "p_bit exit=0\ngemm exit=0\nrun_ab exit=0\ncollect exit=0\n")
    _write(os.path.join(root, "hwlock.log"), "acquired at=x\nreleased-pending at=y\n")
    _write(os.path.join(root, "rev-stamp.txt"), "abc\nabc+on-arm.patch\n")


def _verdict(**kw):
    with tempfile.TemporaryDirectory() as td:
        _fixture(td, **kw)
        try:
            return evaluate(td)["verdict"]
        except AggError as e:
            return "undetermined(AggError: %s)" % e


def self_test():
    ok = True

    def check(name, got, want):
        nonlocal ok
        good = got.startswith(want) and (want != "ADOPT" or got == "ADOPT")
        ok = ok and good
        print("%s %s -> %s" % ("ok  " if good else "FAIL", name, got))

    check("正常（ON が速い）", _verdict(), "ADOPT")
    check("torch なしでも ADOPT", _verdict(torch=False), "ADOPT")
    check("到達 GEMM l1_fwd の 5/5 後退は REJECT", _verdict(regress="l1_fwd"), "REJECT")
    check("非到達 GEMM の 5/5 後退は REJECT", _verdict(regress="l2_fwd"), "REJECT")
    check("l2_dx のみ後退 + K=16 は ADOPT-k", _verdict(regress="l2_dx", kmin=16), "ADOPT-k（k >= 16）")
    check("l2_dx のみ後退 + K=64 は ADOPT-k", _verdict(regress="l2_dx", kmin=64), "ADOPT-k（k >= 64）")
    check("l2_dx のみ後退でも K 格子に後退なしは undetermined", _verdict(regress="l2_dx"), "undetermined")
    check("l2_dx のみ後退で K 下限を格子上で決められないと undetermined", _verdict(regress="l2_dx", kmin=1000), "undetermined")
    check("l2_dx 後退 + l1_fwd 後退は REJECT", _verdict(regress="l1_fwd", kmin=16), "REJECT")
    check("ON が遅い（1.05）", _verdict(on_ratio=1.05), "REJECT")
    check("train/infer 後退 + l2_dx 後退は REJECT", _verdict(on_ratio=1.05, regress="l2_dx", kmin=16), "REJECT")
    check("FC 欠損行は undetermined", _verdict(drop="fc"), "undetermined")
    check("GEMM 欠損行は undetermined", _verdict(drop="gemm"), "undetermined")
    check("FC 重複行は undetermined", _verdict(dup=True), "undetermined")
    check("FC checksum 不一致は FAIL", _verdict(ck_bad="fc"), "FAIL")
    check("GEMM checksum 不一致は FAIL", _verdict(ck_bad="gemm"), "FAIL")
    check("欠損 + checksum 不一致でも FAIL", _verdict(ck_bad="gemm", drop="fc"), "FAIL")
    check("checksum 不一致 + 後退でも FAIL", _verdict(ck_bad="fc", regress="l1_fwd"), "FAIL")
    check("ゲート不通過は undetermined", _verdict(gate_ok=False), "undetermined")
    check("ゲート不通過 + 後退でも undetermined（REJECT より先）", _verdict(gate_ok=False, regress="l1_fwd"), "undetermined")
    check("after 名に before 腕ログを誤収録は undetermined", _verdict(bad_arm=True), "undetermined")
    check("shape ラベルだけ一致し次元が違う行は undetermined", _verdict(bad_dim=True), "undetermined")
    check("tree 欠損は undetermined", _verdict(rm_tree=True), "undetermined")
    print("self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--self-test":
        sys.exit(self_test())
    if len(sys.argv) != 2 or not re.fullmatch(r"[A-Za-z0-9_-]+", sys.argv[1]):
        sys.exit("usage: aggregate.py <subdir> | --self-test（subdir は英数字・-・_ のみ）")
    v = run(sys.argv[1])
    sys.exit(0 if not v.startswith("undetermined") else 1)
