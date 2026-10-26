#!/usr/bin/env python3
"""イシュー #3615: #3236 の l2_dx（GEMM 単体 64x256x10 Nt）を反復数・round 数を増やして測り直した系列（RULE-reopen-1）の厳密集計。

役割: RULE.txt 末尾の RULE-reopen-1（計測前にコミット済み）の判定を、実測ログから機械的に出力する。
`aggregate.py`（#3236 本体。変更しない）の派生で、framework-compare は測らず、GEMM 単体の到達 3 セル
（l1_fwd・l1_dx・l2_dx）と K 格子 7 点を ROUNDS=11・WARMUP=200・ITERS=2000 で判定する。
判定の閾値は #3236 と同じ（RULE §4 の「5/5」を「全 round（N/N）」と読み替える）。
標準ライブラリのみ。`python3 -I` で起動すること。結果の Markdown は手で直さない。

使い方:
    python3 -I aggregate_reopen.py <subdir>     # 例: macmini-reopen
    python3 -I aggregate_reopen.py --self-test

腕の向き: before = main（OFF）、after = on-arm.patch（ON）。比 = after / before（1.00 以下なら ON が速い）。
異常入力（重複・欠損・非有限値・arm 不一致・shape 契約違反・warmup／iters 不一致・ログ欠損）は捨てずに
problems へ収集し undetermined にする（寛容なパースはしない）。checksum が一致しなければ FAIL（性能判定へ進まない）。
評価順: FAIL > 前提不成立 → undetermined > REJECT / ADOPT-k > ADOPT > undetermined。
"""
import math
import os
import re
import statistics
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROUNDS = 11
WARMUP = 200
ITERS = 2000
REACH = ("l1_fwd", "l1_dx", "l2_dx")
KS = (10, 16, 32, 64, 128, 256, 784)
SMALL_K_REACH = ("l2_dx",)
GATE_FILE = "crates/backend-cpu/src/gemm_blis/mod.rs"
GATE_CONST = "SME_THIN_M64_ENABLED"
# 到達 GEMM セルの (転置, m, n, k)。cpu_sme_thin_m64_reopen.rs の GEMM_SHAPES・RULE §2 と一致させる（P-CELLS の契約）
SPEC = {
    "l1_fwd": ("nn", 64, 256, 784),
    "l1_dx": ("nt", 64, 784, 256),
    "l2_dx": ("nt", 64, 256, 10),
}
KGRID_M, KGRID_N = 64, 256


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
    """kind は 'gemm'／'kgrid'。round ごとに <kind>-<arm>-r<N>.txt（1 round 1 ファイル）を読む。"""
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
                warm, iters = int(d["warmup"]), int(d["iters"])
                trans = d["trans"] if kind == "gemm" else None
            except (ValueError, KeyError) as e:
                problems.append(where + " 行が不正: %s" % e)
                continue
            if row_arm != arm:
                problems.append(where + " arm 不一致: 行=%s 期待=%s（別腕のログの誤収録）" % (row_arm, arm))
                continue
            if shape not in shapes:
                problems.append(where + " 未知の shape: " + shape)
                continue
            if kind == "gemm":
                want = SPEC[shape]
                got = (trans,) + dims
            else:
                want = (None, KGRID_M, KGRID_N, int(shape[len("kgrid_k"):]))
                got = (None,) + dims
            if got != want:
                problems.append(where + " shape 契約違反: %s は 期待 %s・実際 %s" % (shape, want, got))
                continue
            if (warm, iters) != (WARMUP, ITERS):
                problems.append(where + " warmup／iters が RULE 値（%d／%d）と不一致: %d／%d" % (WARMUP, ITERS, warm, iters))
                continue
            if shape in seen:
                problems.append(where + " shape 重複: " + shape)
                continue
            seen.add(shape)
            if kind == "gemm" and d.get("series") != "threads":
                problems.append(where + " series が threads ではない")
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


def fmt(x):
    return "%.3f" % x


def ratio_stats(before, after):
    ratios = [after[i] / before[i] for i in range(ROUNDS)]
    return {"ratios": ratios, "median": statistics.median(ratios),
            "n_le": sum(1 for r in ratios if r <= 1.0), "n_gt": sum(1 for r in ratios if r > 1.0),
            "off_med": statistics.median(before), "on_med": statistics.median(after)}


def cell_ok(s):
    return s["median"] <= 1.0 and s["n_le"] == ROUNDS


def cell_label(s):
    if cell_ok(s):
        return "ADOPT"
    if s["n_gt"] == ROUNDS:
        return "後退"
    return "undetermined"


def evaluate(base):
    gdir = os.path.join(base, "gemm")
    problems, ck_all = [], {}
    jg = {a: load_grid(gdir, "gemm", a, REACH, problems, ck_all) for a in ("before", "after")}
    kg = {a: load_grid(gdir, "kgrid", a, ["kgrid_k%d" % k for k in KS], problems, ck_all) for a in ("before", "after")}
    res = {"gate_fail": list(problems), "notes": [], "incomplete": bool(problems)}
    res["checksum_bad"] = [c for c, v in ck_all.items() if len(set(v)) > 1]
    gf = res["gate_fail"]
    if problems:
        gf.append("前提不成立: 計測ログの欠測・不正のため性能判定を行わない")

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
    if lg is not None:
        lines = lg.splitlines()
        for key in ("p_bit exit=0", "gemm exit=0", "collect exit=0"):
            if key not in lines:
                gf.append("P-BIT／P-RUNAB／P-COLLECT: '%s' がない" % key)
    hw = rd("hwlock.log")
    if hw is not None and (not re.search(r"^acquired at=", hw, re.M) or not re.search(r"^released-pending at=", hw, re.M)):
        gf.append("P-LOCK: hwlock.log に取得・解放の記録がない")
    t = rd("rev-stamp.txt")
    if t is not None and len(t.split()) != 2:
        gf.append("rev-stamp が 2 行（main と on-arm）ではない")
    try:
        txt = read_text(os.path.join(gdir, "load_gate.log"))
        ok = re.findall(r"^round(\d+) gate=pass gate_ok=1 ", txt, re.M)
        if sorted(ok, key=int) != [str(r) for r in range(1, ROUNDS + 1)]:
            gf.append("P-GATE: 負荷ゲートが %d/%d round で gate_ok=1 ではない（通過 round=%s）" % (ROUNDS, ROUNDS, ",".join(ok)))
    except AggError as e:
        gf.append("前提不成立: %s" % e)
    res["stats"], res["kstats"] = {}, {}
    if problems:
        res["consistent"], res["verdict"], res["K"] = [], ("FAIL" if res["checksum_bad"] else "undetermined"), None
        res["notes"].append("計測ログ欠測・不正のためセルごとの比は算出しない")
        return res
    for s in REACH:
        res["stats"][s] = ratio_stats(jg["before"][s], jg["after"][s])
    for k in KS:
        res["kstats"][k] = ratio_stats(kg["before"]["kgrid_k%d" % k], kg["after"]["kgrid_k%d" % k])
    consistent = [s for s, st in res["stats"].items() if st["n_gt"] == ROUNDS]
    res["consistent"] = consistent
    reach_ok = all(cell_ok(res["stats"][s]) for s in REACH)
    K = None
    for i, k in enumerate(KS):
        if all(res["kstats"][kk]["n_le"] == ROUNDS for kk in KS[i:]):
            K = k
            break
    res["K"] = K
    others_ok = all(cell_ok(res["stats"][s]) for s in REACH if s not in SMALL_K_REACH)
    if res["checksum_bad"]:
        res["verdict"] = "FAIL"
    elif gf:
        res["verdict"] = "undetermined"
    elif consistent and not set(consistent) <= set(SMALL_K_REACH):
        res["verdict"] = "REJECT"
    elif consistent:
        if others_ok and K is not None and K > KS[0]:
            res["verdict"] = "ADOPT-k（k >= %d）" % K
        else:
            # RULE-reopen-1: l2_dx が後退して ADOPT-k に当たらない場合は REJECT（undetermined 例外なし）
            res["verdict"] = "REJECT"
            res["notes"].append("l2_dx が N/N 後退し ADOPT-k の条件（K 一意かつ K > %d・他セル ADOPT）を満たさない（RULE-reopen-1 により REJECT）" % KS[0])
    elif reach_ok:
        res["verdict"] = "ADOPT"
    else:
        res["verdict"] = "undetermined"
    return res


def render(res):
    L = ["# #3615 厳密集計 (macmini-reopen・RULE-reopen-1)", "",
         "機械出力（`python3 -I aggregate_reopen.py` の出力。手で直さない）。腕: before = main（OFF）・after = on-arm（ON）。"
         "比 = ON / OFF（<= 1.00 なら ON が速い）。ROUNDS=%d・WARMUP=%d・ITERS=%d。" % (ROUNDS, WARMUP, ITERS), ""]
    if res["incomplete"]:
        L += ["計測ログの欠測・不正のためセルごとの比は算出しない（前提ゲート・FAIL 確認のみ実施）。", ""]
    else:
        L += ["## セルごと（%d round 中央値）" % ROUNDS, "",
              "| セル | 区分 | OFF 中央値 | ON 中央値 | ON/OFF 中央値比 | <=1.00 | >1.00 | セル判定 |", "|---|---|---|---|---|---|---|---|"]
        for s in REACH:
            st = res["stats"][s]
            L.append("| gemm %s | 到達 | %.6g | %.6g | %s | %d | %d | %s |" % (
                s, st["off_med"], st["on_med"], fmt(st["median"]), st["n_le"], st["n_gt"], cell_label(st)))
        for k in KS:
            st = res["kstats"][k]
            L.append("| kgrid k=%d | K 格子（記録用） | %.6g | %.6g | %s | %d | %d | - |" % (
                k, st["off_med"], st["on_med"], fmt(st["median"]), st["n_le"], st["n_gt"]))
        L += ["", "## round 別比（ON/OFF）", "", "| セル | " + " | ".join("r%d" % r for r in range(1, ROUNDS + 1)) + " |",
              "|---|" + "---|" * ROUNDS]
        for s in REACH:
            L.append("| gemm %s | %s |" % (s, " | ".join(fmt(r) for r in res["stats"][s]["ratios"])))
        for k in KS:
            L.append("| kgrid k=%d | %s |" % (k, " | ".join(fmt(r) for r in res["kstats"][k]["ratios"])))
    L += ["", "## 前提ゲート", ""]
    L += ["- " + g for g in res["gate_fail"]] if res["gate_fail"] else ["- すべて成立"]
    if res["checksum_bad"]:
        L += ["", "## checksum 不一致（FAIL）", ""] + ["- %s" % (c,) for c in res["checksum_bad"]]
    L += ["", "## 後退と K 下限", "",
          "- 全 round 一貫の後退セル: %s" % (", ".join(res["consistent"]) or "なし"),
          "- K 下限（k >= K の全点が N/N で ON <= OFF となる最小の K。格子上）: %s" % (res["K"] if res["K"] is not None else "なし")]
    for n in res["notes"]:
        L.append("- 注記: " + n)
    L += ["", "## 総合判定（GEMM 単体の到達 3 セルのみ）", "", "**macmini-reopen**: %s" % res["verdict"], ""]
    if res["verdict"] == "ADOPT" or res["verdict"].startswith("ADOPT-k"):
        L += ["参考判定: framework-compare は本系列で再計測していない。#3236 の framework-compare 4 セル（5/5 で ON が速い）と組み合わせた場合に限り",
              "「ADOPT 候補」。framework-compare は 3f122311f（facade 等の変更前）のツリーで測った値であり、反転 PR を作るなら framework-compare の再計測、",
              "docs/cpu-gemm-sme-unsafe-audit.md §13 の独立監査、所有者の確認が必要（RULE §5）。既定はこの issue では反転しない。", ""]
    return "\n".join(L)


def run(subdir, out=sys.stdout):
    try:
        res = evaluate(os.path.join(HERE, subdir))
    except AggError as e:
        out.write("# #3615 厳密集計 (macmini-reopen)\n\n**undetermined**: %s\n" % e)
        return "undetermined"
    out.write(render(res))
    return res["verdict"]


# ---------------------------------------------------------------- self-test
def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w", encoding="utf-8").write(text)


def _fixture(root, on_ratio=0.9, regress=None, kmin=None, ck_bad=False, drop=False, dup=False, iters_bad=False,
             gate_n=ROUNDS, bad_arm=False, bad_dim=False, rm_tree=False, flaky=None):
    """regress: 全 round 後退させる shape。kmin: k < kmin の K 格子点を後退させる。flaky: 1 round だけ後退させる shape。"""
    gd = os.path.join(root, "gemm")
    for arm in ("before", "after"):
        for r in range(1, ROUNDS + 1):
            jl, kl = [], []
            for s in REACH:
                ratio = on_ratio
                if regress == s:
                    ratio = 1.2
                if flaky == s and r == 1:
                    ratio = 1.2
                ck = 0x1000 + (1 if (ck_bad and arm == "after" and s == "l1_fwd") else 0)
                if drop and arm == "after" and s == "l1_dx" and r == 1:
                    continue
                it = ITERS + 1 if (iters_bad and arm == "after" and r == 1) else ITERS
                jl.append("arm=%s series=threads shape=%s trans=%s m=%d n=%d k=%d rayon_threads=10 warmup=%d iters=%d median_secs=%.9f gflops=1.0 checksum=0x%016x"
                          % ((arm, s) + SPEC[s] + (WARMUP, it, 0.001 * (ratio if arm == "after" else 1.0), ck)))
                if dup and arm == "after" and s == "l1_fwd" and r == 1:
                    jl.append(jl[-1])
            for k in KS:
                ratio = on_ratio
                if kmin is not None and k < kmin:
                    ratio = 1.2
                kl.append("arm=%s shape=kgrid_k%d m=64 n=256 k=%d rayon_threads=10 warmup=%d iters=%d median_secs=%.9f gflops=1.0 checksum=0x%016x"
                          % (arm, k, k, WARMUP, ITERS, 0.001 * (ratio if arm == "after" else 1.0), 0x2000 + k))
            if bad_arm and arm == "after" and r == 1:
                jl = [x.replace("arm=after", "arm=before") for x in jl]
            if bad_dim and arm == "after" and r == 1:
                jl = [x.replace("m=64 n=784 k=256", "m=64 n=784 k=257") for x in jl]
            _write(os.path.join(gd, "gemm-%s-r%d.txt" % (arm, r)), "\n".join(jl) + "\n")
            _write(os.path.join(gd, "kgrid-%s-r%d.txt" % (arm, r)), "\n".join(kl) + "\n")
    g = "".join("round%d gate=pass gate_ok=1 threshold=3.0 load1=0.5 waited_s=0 at=x\n" % r for r in range(1, gate_n + 1))
    _write(os.path.join(gd, "load_gate.log"), g)
    if not rm_tree:
        _write(os.path.join(root, "tree_verify.txt"), GATE_FILE + "\n")
    _write(os.path.join(root, "patch_sha256.txt"), rule_sha() + "\n")
    _write(os.path.join(root, "gate_constant.txt"), "3603:const %s: bool = false;\n3603:const %s: bool = true;\n" % (GATE_CONST, GATE_CONST))
    _write(os.path.join(root, "sme_report.txt"), "before: sme_report= kernel_enabled: true\nafter: sme_report= kernel_enabled: true\n")
    _write(os.path.join(root, "load_gate.log"), "p_bit exit=0\ngemm exit=0\ncollect exit=0\n")
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
        good = got == want if want in ("ADOPT", "REJECT", "FAIL") else got.startswith(want)
        ok = ok and good
        print("%s %s -> %s" % ("ok  " if good else "FAIL", name, got))

    check("正常（ON が速い）", _verdict(), "ADOPT")
    check("l2_dx が 10/11 のみ後退は undetermined", _verdict(flaky="l2_dx"), "undetermined")
    check("l2_dx が 11/11 後退 + K=16 は ADOPT-k", _verdict(regress="l2_dx", kmin=16), "ADOPT-k（k >= 16）")
    check("l2_dx が 11/11 後退で K 格子に後退なしは REJECT（RULE-reopen-1）", _verdict(regress="l2_dx"), "REJECT")
    check("l2_dx が 11/11 後退 + K=10（K > 10 を満たさない）は REJECT", _verdict(regress="l2_dx", kmin=10), "REJECT")
    check("l1_fwd の 11/11 後退は REJECT", _verdict(regress="l1_fwd"), "REJECT")
    check("l1_dx の 11/11 後退 + l2_dx 後退は REJECT", _verdict(regress="l1_dx", kmin=16), "REJECT")
    check("checksum 不一致は FAIL", _verdict(ck_bad=True), "FAIL")
    check("checksum 不一致 + 後退でも FAIL", _verdict(ck_bad=True, regress="l1_fwd"), "FAIL")
    check("欠損行は undetermined", _verdict(drop=True), "undetermined")
    check("重複行は undetermined", _verdict(dup=True), "undetermined")
    check("iters 不一致は undetermined", _verdict(iters_bad=True), "undetermined")
    check("ゲート 10/11 は undetermined", _verdict(gate_n=ROUNDS - 1), "undetermined")
    check("ゲート不通過 + 後退でも undetermined（REJECT より先）", _verdict(gate_n=ROUNDS - 1, regress="l1_fwd"), "undetermined")
    check("arm の誤収録は undetermined", _verdict(bad_arm=True), "undetermined")
    check("次元違反は undetermined", _verdict(bad_dim=True), "undetermined")
    check("tree 欠損は undetermined", _verdict(rm_tree=True), "undetermined")
    print("self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--self-test":
        sys.exit(self_test())
    if len(sys.argv) != 2 or not re.fullmatch(r"[A-Za-z0-9_-]+", sys.argv[1]):
        sys.exit("usage: aggregate_reopen.py <subdir> | --self-test（subdir は英数字・-・_ のみ）")
    v = run(sys.argv[1])
    sys.exit(0 if not v.startswith("undetermined") else 1)
