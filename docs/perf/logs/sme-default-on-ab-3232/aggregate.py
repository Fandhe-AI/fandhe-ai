#!/usr/bin/env python3
"""イシュー #3232: SME 既定 ON/OFF A/B（Mac mini・GB10）の厳密集計。

役割: RULE.txt（同ディレクトリ・計測前に固定）の判定を、実測ログから機械的に出力する。
`docs/perf/logs/sme-macmini-3176/aggregate.py` の R1/R2/R5 部分の派生で、R4 格子・長方形は扱わない。
標準ライブラリのみ。`python3 -I` で起動すること。結果の Markdown は手で直さない。

使い方:
    python3 -I aggregate.py <subdir> <host>   # subdir は本ファイルのあるディレクトリからの相対（例: macmini macmini）
    python3 -I aggregate.py --self-test

腕の向き: before = OFF、after = ON。比 = after / before（1.00 以下なら ON が速い）。
異常入力（重複・欠損・非有限値・未知のキー・型不正・ログ欠損）は捨てずに AggError へ写し、undetermined にする。
checksum が数値として完全一致しない場合は FAIL（性能判定へ進まない）。
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
SIZES = (256, 512, 1024, 2048)
MODES = ("fresh", "reuse")
CELLS = [("gemm", s, m) for s in SIZES for m in MODES] + [("train", 64, m) for m in MODES] + [("infer", 64, m) for m in MODES]
KNOWN_TASKS = ("gemm", "train", "infer")
# 系列 B（0.11.0 registry 版の値。本 run の値ではない。gen_0110_macmini.out）: fandhe の対 PyTorch 倍率（torch median / fandhe median）
SERIES_B = {("gemm", 256): 1.04, ("gemm", 512): 1.04, ("gemm", 1024): 0.89, ("gemm", 2048): 0.69, ("train", 64): 0.20, ("infer", 64): 0.25}
MACMINI_NONREACH = {"infer"}
GATE_FILE = "./crates/backend-cpu/src/gemm_blis/mod.rs"
# 参考: #3231 の M4 Max bitdump（記録のみ）
M4MAX_BITDUMP_LINES = 6726847


class AggError(Exception):
    pass


def rule_sha():
    for line in open(os.path.join(HERE, "RULE.txt"), encoding="utf-8"):
        m = re.match(r"^off-arm\.patch sha256: ([0-9a-f]{64})$", line.rstrip("\n"))
        if m:
            return m.group(1)
    raise AggError("RULE.txt から patch の sha256 を取得できない")


def read_text(path):
    if not os.path.isfile(path):
        raise AggError("ログ欠損: " + os.path.relpath(path, HERE))
    return open(path, encoding="utf-8").read()


def finite_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def load_rows(path, task, allow_modes=True):
    rows = []
    for i, line in enumerate(read_text(path).splitlines(), 1):
        if not line.strip():
            continue
        where = "%s:%d" % (os.path.basename(path), i)
        try:
            o = json.loads(line)
        except ValueError:
            raise AggError(where + " JSON 不正")
        if not isinstance(o, dict):
            raise AggError(where + " 行がオブジェクトではない")
        if o.get("task") != task:
            raise AggError(where + " task が不一致: %r" % (o.get("task"),))
        if o.get("device") != "cpu":
            raise AggError(where + " device が cpu ではない")
        size, mode = o.get("size"), o.get("mode", "fresh")
        if isinstance(size, bool) or not isinstance(size, int) or not isinstance(mode, str):
            raise AggError(where + " size/mode の型が不正")
        if (task, size, mode) not in CELLS and task != "torch":
            raise AggError(where + " 未知のセル: %s size=%s mode=%s" % (task, size, mode))
        if not finite_num(o.get("median_s")) or o["median_s"] <= 0:
            raise AggError(where + " median_s が不正")
        if not finite_num(o.get("checksum")):
            raise AggError(where + " checksum が非有限・欠損")
        rows.append(o)
    return rows


def find_one(pattern):
    hits = sorted(glob.glob(pattern))
    if len(hits) != 1:
        raise AggError("ファイルが 1 件でない（%d 件）: %s" % (len(hits), os.path.relpath(pattern, HERE)))
    return hits[0]


def load_arm_cells(ab, arm):
    out = {}
    for task in KNOWN_TASKS:
        path = find_one(os.path.join(ab, "results-%s-*-cpu-%s.jsonl" % (arm, task)))
        for o in load_rows(path, task):
            out.setdefault((task, o["size"], o.get("mode", "fresh")), []).append(o)
    for cell in CELLS:
        n = len(out.get(cell, []))
        if n != ROUNDS:
            raise AggError("P-CELLS 不成立: arm=%s cell=%s は %d 行（%d 行ちょうどを要求）" % (arm, cell, n, ROUNDS))
    return out


def load_torch(ab):
    hits = sorted(glob.glob(os.path.join(ab, "results-torch-*-cpu.jsonl")))
    if not hits:
        return None
    rows = []
    for i, line in enumerate(read_text(hits[0]).splitlines(), 1):
        if not line.strip():
            continue
        o = json.loads(line)
        if not isinstance(o, dict) or not finite_num(o.get("median_s")) or o["median_s"] <= 0 \
                or isinstance(o.get("size"), bool) or not isinstance(o.get("size"), int):
            raise AggError("torch 行が不正: %s:%d" % (os.path.basename(hits[0]), i))
        rows.append(o)
    by = {}
    for o in rows:
        by.setdefault((o.get("task"), o["size"]), []).append(o["median_s"])
    for key in [("gemm", s) for s in SIZES] + [("train", 64), ("infer", 64)]:
        if len(by.get(key, [])) != ROUNDS:
            raise AggError("torch の %s は %d 行（%d 行ちょうどを要求）" % (key, len(by.get(key, [])), ROUNDS))
    return by


def fmt(x):
    return "%.3f" % x


def evaluate(base, host):
    """host のログ 1 式を評価して dict を返す。AggError は呼び出し側で undetermined へ写す。"""
    ab = os.path.join(base, "ab")
    res = {"host": host, "gate_fail": [], "notes": []}
    arm = {a: load_arm_cells(ab, a) for a in ("before", "after")}
    # checksum 完全一致（FAIL）
    ck_bad = []
    for cell in CELLS:
        vals = [o["checksum"] for a in arm for o in arm[a][cell]]
        if len(set(vals)) != 1:
            ck_bad.append(cell)
    res["checksum_bad"] = ck_bad
    # セルごとの比
    stats = {}
    for cell in CELLS:
        ratios = [arm["after"][cell][i]["median_s"] / arm["before"][cell][i]["median_s"] for i in range(ROUNDS)]
        stats[cell] = {
            "ratios": ratios,
            "median": statistics.median(ratios),
            "n_le": sum(1 for r in ratios if r <= 1.0),
            "n_gt": sum(1 for r in ratios if r > 1.0),
            "checksum": arm["after"][cell][0]["checksum"],
            "off_med": statistics.median(o["median_s"] for o in arm["before"][cell]),
            "on_med": statistics.median(o["median_s"] for o in arm["after"][cell]),
        }
    res["stats"] = stats
    # 前提ゲート
    gf = res["gate_fail"]
    if read_text(os.path.join(base, "tree_verify.txt")).strip() != GATE_FILE:
        gf.append("P-TREE: 差分ファイルが 1 件（%s）ではない" % GATE_FILE)
    if read_text(os.path.join(base, "patch_sha256.txt")).strip() != rule_sha():
        gf.append("P-TREE: patch の sha256 が RULE と不一致")
    consts = [re.sub(r"^\d+:", "", l) for l in read_text(os.path.join(base, "gate_constant.txt")).splitlines()]
    if consts != ["const SME_PRODUCTION_ENABLED: bool = false;", "const SME_PRODUCTION_ENABLED: bool = true;"]:
        gf.append("P-TREE: 定数行が OFF=false・ON=true ではない")
    expect_kernel = "true" if host == "macmini" else "false"
    rep = read_text(os.path.join(base, "sme_report.txt"))
    for t in ("off", "on"):
        m = re.search(r"^%s: .*kernel_enabled: (true|false)" % t, rep, re.M)
        if not m:
            gf.append("P-R0: %s の sme_report が読めない" % t)
        elif m.group(1) != expect_kernel:
            gf.append("P-R0: %s の kernel_enabled=%s（期待 %s）" % (t, m.group(1), expect_kernel))
    lg = read_text(os.path.join(base, "load_gate.log"))
    for key in ("run_ab exit=0", "collect exit=0"):
        if key not in lg.splitlines():
            gf.append("P-RUNAB/P-COLLECT: '%s' がない" % key)
    hw = read_text(os.path.join(base, "hwlock.log"))
    if not re.search(r"^acquired at=", hw, re.M) or not re.search(r"^released-pending at=", hw, re.M):
        gf.append("P-LOCK: hwlock.log に取得・解放の記録がない")
    if len(read_text(os.path.join(base, "rev-stamp.txt")).split()) != 2:
        gf.append("rev-stamp が 2 行（ON と OFF）ではない")
    inner = read_text(find_one(os.path.join(ab, "load_gate-*-cpu-*.log")))
    rounds_ok = re.findall(r"^round(\d) gate=pass gate_ok=1 ", inner, re.M)
    if sorted(rounds_ok) != ["1", "2", "3", "4", "5"]:
        gf.append("P-GATE: 内側ゲートが 5/5 round で gate_ok=1 ではない（通過 round=%s）" % ",".join(rounds_ok))
    if host == "gb10":
        outer = read_text(os.path.join(base, "load_gate_outer.log"))
        if not re.search(r" pass$", outer, re.M):
            gf.append("P-GATE: 外側ゲート（load1 < 1.0・GPU 0%）の通過記録がない")
        if "rt_on exit=0" not in lg.splitlines():
            gf.append("P-RT: ON ツリーの cargo test が全件 pass ではない（後退あり相当・要調査）")
        bd = re.search(r"^bitdump exit=(\d+) cmp_exit=(\d+)$", lg, re.M)
        summ = read_text(os.path.join(base, "bitdump", "summary.txt"))
        if not bd:
            gf.append("bitdump: 実行記録がない")
        else:
            res["bitdump_cmp_exit"] = int(bd.group(2))
            if bd.group(1) != "0":
                gf.append("bitdump: 行数が期待と一致しない・実行失敗")
        res["bitdump_summary"] = summ.strip()
    # PyTorch 相手比（記録のみ）
    torch = load_torch(ab)
    res["torch"] = None
    if torch is None:
        res["notes"].append("PyTorch 未計測（記録のみのため判定には影響しない）")
    else:
        tr = {}
        for cell in CELLS:
            key = (cell[0], cell[1])
            per = {}
            for a in ("before", "after"):
                per[a] = statistics.median(torch[key][i] / arm[a][cell][i]["median_s"] for i in range(ROUNDS))
            tr[cell] = per
        res["torch"] = tr
    # 分類
    if host == "macmini":
        res["reach"] = [c for c in CELLS if c[0] not in MACMINI_NONREACH]
        res["guard"] = [c for c in CELLS if c[0] in MACMINI_NONREACH]
    else:
        res["reach"] = []
        res["guard"] = list(CELLS)
    consistent = [c for c in CELLS if stats[c]["n_gt"] == ROUNDS]
    res["consistent_regress"] = consistent
    reach_ok = all(stats[c]["median"] <= 1.0 and stats[c]["n_le"] == ROUNDS for c in res["reach"])
    res["reach_ok"] = reach_ok
    # ホスト単位の判定
    if ck_bad:
        res["verdict"] = "FAIL"
    elif gf:
        res["verdict"] = "undetermined"
    elif consistent:
        res["verdict"] = "REJECT"
    elif host == "gb10":
        res["verdict"] = "非後退: pass" if res.get("bitdump_cmp_exit", 1) == 0 else "FAIL"
    else:
        res["verdict"] = "ADOPT 候補（Mac mini 側条件）" if reach_ok else "undetermined"
    if host == "gb10" and res["verdict"] == "FAIL" and not ck_bad:
        res["notes"].append("bitdump の cmp が不一致（bit 同一でない）")
    return res


def render(res, other):
    host = res["host"]
    L = ["# #3232 厳密集計 (%s)" % host, "",
         "機械出力（`python3 -I aggregate.py` の出力。手で直さない）。腕: before = OFF・after = ON。比 = ON / OFF（<= 1.00 なら ON が速い）。", ""]
    L += ["## セルごと（5 round 中央値）", "",
          "| セル | 到達区分 | OFF 中央値(s) | ON 中央値(s) | ON/OFF 中央値比 | round 別比 | <=1.00 | >1.00 | checksum |",
          "|---|---|---|---|---|---|---|---|---|"]
    for c in CELLS:
        s = res["stats"][c]
        cls = "到達" if c in res["reach"] else "ガード（非到達）"
        L.append("| %s %d %s | %s | %.6g | %.6g | %s | %s | %d | %d | %r |" % (
            c[0], c[1], c[2], cls, s["off_med"], s["on_med"], fmt(s["median"]),
            " ".join("r%d=%s" % (i + 1, fmt(r)) for i, r in enumerate(s["ratios"])), s["n_le"], s["n_gt"], s["checksum"]))
    L += ["", "## 相手（PyTorch）比: PyTorch median / fandhe median の 5 round 中央値（記録のみ・判定に使わない）", ""]
    if res["torch"] is None:
        L.append("PyTorch 未計測。")
    else:
        L += ["| セル | OFF | ON | 系列 B（参考。0.11.0 registry 版・本 run の値ではない） |", "|---|---|---|---|"]
        for c in CELLS:
            t = res["torch"][c]
            ref = SERIES_B.get((c[0], c[1]))
            L.append("| %s %d %s | %s | %s | %s |" % (c[0], c[1], c[2], fmt(t["before"]), fmt(t["after"]), "%.2f" % ref if (ref and host == "macmini") else "-"))
    L += ["", "## 前提ゲート", ""]
    L += ["- " + g for g in res["gate_fail"]] if res["gate_fail"] else ["- すべて成立"]
    if res.get("bitdump_summary"):
        L += ["", "## bitdump（ON/OFF の bit 同一。#3231 の M4 Max は %d 行・参考で判定しない）" % M4MAX_BITDUMP_LINES, "", "```", res["bitdump_summary"], "```"]
    if res["checksum_bad"]:
        L += ["", "## checksum 不一致（FAIL）", ""] + ["- %s" % (c,) for c in res["checksum_bad"]]
    L += ["", "## 後退", "",
          "- 5/5 一貫の後退セル: %s" % (", ".join("%s %d %s" % c for c in res["consistent_regress"]) or "なし"),
          "- 中央値 > 1.00 だが 5/5 一貫でないセル（後退の証拠なし・ノイズ帯）: %s" % (
              ", ".join("%s %d %s(%s)" % (c + (fmt(res["stats"][c]["median"]),)) for c in CELLS
                        if res["stats"][c]["median"] > 1.0 and res["stats"][c]["n_gt"] < ROUNDS) or "なし")]
    for n in res["notes"]:
        L.append("- 注記: " + n)
    verdict = res["verdict"]
    if host == "macmini" and verdict.startswith("ADOPT"):
        if other is None:
            verdict = "undetermined（GB10 未計測のため ADOPT 不可）"
        elif other["verdict"] != "非後退: pass":
            verdict = "undetermined（GB10 が「非後退: pass」でない: %s）" % other["verdict"]
        else:
            verdict = "ADOPT"
    L += ["", "## 総合判定", "", "**%s**: %s" % (host, verdict), ""]
    return "\n".join(L), verdict


def run(subdir, host, out=sys.stdout):
    base = os.path.join(HERE, subdir)
    try:
        res = evaluate(base, host)
    except AggError as e:
        out.write("# #3232 厳密集計 (%s)\n\n**undetermined**: %s\n" % (host, e))
        return "undetermined"
    other = None
    if host == "macmini":
        try:
            other = evaluate(os.path.join(HERE, "gb10"), "gb10")
        except AggError:
            other = None
    text, verdict = render(res, other)
    out.write(text)
    return verdict


# ---------------------------------------------------------------- self-test
def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w", encoding="utf-8").write(text)


def _fixture(root, host, on_ratio=0.9, ck_bad=False, drop_row=False, dup=False, gate_ok=True, bad_regress=False):
    ab = os.path.join(root, "ab")
    for arm in ("before", "after"):
        for task in KNOWN_TASKS:
            lines = []
            cells = [c for c in CELLS if c[0] == task]
            for (_, size, mode) in cells:
                for r in range(ROUNDS):
                    base_t = 0.001 * size
                    if arm == "after":
                        ratio = (1.1 if bad_regress and size == 512 else on_ratio)
                        t = base_t * ratio
                    else:
                        t = base_t
                    ck = 1.5 + (0.1 if (ck_bad and arm == "after" and size == 256 and mode == "fresh" and task == "gemm") else 0)
                    lines.append(json.dumps({"task": task, "device": "cpu", "size": size, "mode": mode, "median_s": t, "checksum": ck}))
            if drop_row and arm == "after" and task == "gemm":
                lines.pop()
            if dup and arm == "after" and task == "gemm":
                lines.append(lines[0])
            _write(os.path.join(ab, "results-%s-3232-r1-cpu-%s.jsonl" % (arm, task)), "\n".join(lines) + "\n")
    tl = []
    for r in range(ROUNDS):
        for key in [("gemm", s) for s in SIZES] + [("train", 64), ("infer", 64)]:
            tl.append(json.dumps({"task": key[0], "device": "cpu", "size": key[1], "median_s": 0.001 * key[1] * 1.2}))
    _write(os.path.join(ab, "results-torch-3232-sme-3232-r1-cpu.jsonl"), "\n".join(tl) + "\n")
    g = "".join("round%d gate=%s gate_ok=%d threshold=3.0 load1=0.5 waited_s=0 at=x\n" % (r, "pass" if gate_ok else "timeout", 1 if gate_ok else 0) for r in range(1, 6))
    _write(os.path.join(ab, "load_gate-3232-sme-cpu-3232-r1.log"), "threshold=3.0\n" + g)
    _write(os.path.join(root, "tree_verify.txt"), GATE_FILE + "\n")
    _write(os.path.join(root, "patch_sha256.txt"), rule_sha() + "\n")
    _write(os.path.join(root, "gate_constant.txt"), "3537:const SME_PRODUCTION_ENABLED: bool = false;\n3540:const SME_PRODUCTION_ENABLED: bool = true;\n")
    k = "true" if host == "macmini" else "false"
    _write(os.path.join(root, "sme_report.txt"), "off: sme_report= kernel_enabled: %s\non: sme_report= kernel_enabled: %s\n" % (k, k))
    lg = "run_ab exit=0\ncollect exit=0\n"
    if host == "gb10":
        lg += "rt_on exit=0\nbitdump exit=0 cmp_exit=0\n"
        _write(os.path.join(root, "load_gate_outer.log"), "start attempt=1 load1=0.1 gpu_util=0 pass\n")
        _write(os.path.join(root, "bitdump", "summary.txt"), "off_lines=6726847 on_lines=6726847 expected=6726847\n")
    _write(os.path.join(root, "load_gate.log"), lg)
    _write(os.path.join(root, "hwlock.log"), "acquired at=x\nreleased-pending at=y\n")
    _write(os.path.join(root, "rev-stamp.txt"), "abc\nabc+off-arm.patch\n")


def _verdict(host, **kw):
    global HERE
    saved = HERE
    with tempfile.TemporaryDirectory() as td:
        try:
            # RULE.txt は実物を使うため HERE を保持し、fixture だけ別ディレクトリへ置く
            _fixture(td, host, **kw)
            try:
                return evaluate(td, host)["verdict"]
            except AggError as e:
                return "undetermined(AggError: %s)" % e
        finally:
            HERE = saved


def self_test():
    ok = True

    def check(name, got, want):
        nonlocal ok
        good = got.startswith(want)
        ok = ok and good
        print("%s %s -> %s" % ("ok  " if good else "FAIL", name, got))

    check("macmini 正常（ON が速い）", _verdict("macmini"), "ADOPT")
    check("gb10 正常", _verdict("gb10"), "非後退: pass")
    check("欠損行", _verdict("macmini", drop_row=True), "undetermined(AggError")
    check("重複行", _verdict("macmini", dup=True), "undetermined(AggError")
    check("checksum 不一致", _verdict("macmini", ck_bad=True), "FAIL")
    check("5/5 一貫の後退", _verdict("macmini", bad_regress=True), "REJECT")
    check("ゲート不通過", _verdict("macmini", gate_ok=False), "undetermined")
    check("ON が遅い（1.05）", _verdict("macmini", on_ratio=1.05), "REJECT")
    # 総合: ADOPT は GB10 に依存する
    with tempfile.TemporaryDirectory() as td:
        _fixture(td, "macmini")
        res = evaluate(td, "macmini")
        _, v = render(res, None)
        check("GB10 未計測では ADOPT にしない", v, "undetermined")
        with tempfile.TemporaryDirectory() as td2:
            _fixture(td2, "gb10")
            _, v = render(res, evaluate(td2, "gb10"))
            check("GB10 pass なら ADOPT", v, "ADOPT")
    print("self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--self-test":
        sys.exit(self_test())
    if len(sys.argv) != 3 or sys.argv[2] not in ("macmini", "gb10"):
        sys.exit("usage: aggregate.py <subdir> <macmini|gb10> | --self-test")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", sys.argv[1]):
        sys.exit("subdir は英数字・-・_ のみ")
    v = run(sys.argv[1], sys.argv[2])
    sys.exit(0 if not v.startswith("undetermined") else 1)
