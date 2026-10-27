#!/usr/bin/env python3
"""イシュー #3613: GB10 の 3 腕 A/B（before / prime / after）を RULE.txt §4 の分類表で機械的に集計する。

役割: gb10/ 配下の収録ログ（A/B の JSONL・前提ゲートのログ）を厳格に読み、2 判定セルそれぞれへ
RULE.txt の帰属分類を適用し、aggregate-gb10.md を出力する（手で直さない）。#3232 の aggregate.py
（同ディレクトリの兄弟 ../sme-default-on-ab-3232/aggregate.py）の厳密集計方針を継承する:
欠損・型不正・件数不一致は例外にせず「判定不能（前提不成立）」へ倒す。標準ライブラリのみ。

使い方:
  python3 -I aggregate.py gb10      # このスクリプトのあるディレクトリの gb10/ を読み aggregate-gb10.md を書く
  python3 -I aggregate.py --self-test
"""
import json
import math
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LABEL = "3613-r1"
ROUNDS = 5
THRESHOLD = 1.00
ARMS = ("before", "prime", "after")
# 組: 名前 -> (分母の腕, 分子の腕)。比 = 分子 / 分母（> 1.00 = 後退）
PAIRS = {"pair0": ("before", "after"), "pair1": ("before", "prime"), "pair2": ("prime", "after")}
JUDGE_CELLS = (("gemm", "fresh", 256), ("gemm", "reuse", 2048))
ALL_CELLS = (
    [("gemm", m, n) for n in (256, 512, 1024, 2048) for m in ("fresh", "reuse")]
    + [("train", m, 64) for m in ("fresh", "reuse")]
    + [("infer", m, 64) for m in ("fresh", "reuse")]
)

# 分類名（RULE.txt §4 と一字一句一致させる）
C_PRECOND = "undetermined（前提不成立）"
C_FAIL = "FAIL"
C_NOREPRO = "非再現（ノイズ帯・判定不能）"
C_SENS = "方法感度（判定不能）"
C_SELECT = "SME 経路の選択ロジック（try_new 呼び出しの有無に伴うコード配置・実行コスト）に帰属"
C_UNSEP = "再現したが未分離（ノイズ帯）"


def classify(ratios, checksum_ok, preconditions_ok):
    """ratios: {pair名: [round1..5 の比]}。RULE.txt §4 を上から順に評価して分類名を返す。"""
    if not preconditions_ok:
        return C_PRECOND
    if not checksum_ok:
        return C_FAIL

    def all_regress(p):
        r = ratios.get(p)
        return r is not None and len(r) == ROUNDS and all(x > THRESHOLD for x in r)

    if not all_regress("pair0"):
        return C_NOREPRO
    if all_regress("pair1"):
        return C_SENS
    if all_regress("pair2"):
        return C_SELECT
    return C_UNSEP


def read_jsonl(path):
    """厳格読み込み。失敗は (None, 理由)。"""
    rows = []
    try:
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                o = json.loads(line)
                if not isinstance(o, dict):
                    return None, "%s:%d 行がオブジェクトではない" % (path, i)
                rows.append(o)
    except (OSError, ValueError) as e:
        return None, "%s: %s" % (path, e)
    return rows, None


def cell_rows(rows, mode, size):
    out = []
    for o in rows:
        sz, md = o.get("size"), o.get("mode", "fresh")
        if isinstance(sz, bool) or not isinstance(sz, int) or not isinstance(md, str):
            return None
        if o.get("device") == "cpu" and md == mode and sz == size:
            out.append(o)
    return out


def load_cells(abdir, label):
    """{(task,mode,size): {arm: [row...]}} と問題リストを返す。"""
    data, problems = {}, []
    files = {}
    for arm in ARMS:
        for task in ("gemm", "train", "infer"):
            p = os.path.join(abdir, "results-%s-%s-cpu-%s.jsonl" % (arm, label, task))
            rows, err = read_jsonl(p)
            if rows is None:
                problems.append("読み込み不能: " + err)
                rows = []
            files[(arm, task)] = rows
    # 想定セルに属さない行（別 device・想定外の size/mode）は集計の分母から黙って落とさず不備にする
    # （RULE.txt §3 P-CELLS「ちょうど」）。cell_rows が None のファイルは下のセル別検査で報告される。
    for (arm, task), rows in files.items():
        want = {(m, n) for t, m, n in ALL_CELLS if t == task}
        stray = 0
        for o in rows:
            sz, md = o.get("size"), o.get("mode", "fresh")
            if o.get("device") != "cpu" or isinstance(sz, bool) or not isinstance(sz, int) or (md, sz) not in want:
                stray += 1
        if stray:
            problems.append("P-CELLS 不成立: 想定セル外の行 %d 件 arm=%s task=%s" % (stray, arm, task))
    for task, mode, size in ALL_CELLS:
        per_arm = {}
        for arm in ARMS:
            rs = cell_rows(files[(arm, task)], mode, size)
            if rs is None or len(rs) != ROUNDS:
                problems.append("P-CELLS 不成立: %s %s %d arm=%s 行数=%s" % (task, mode, size, arm, "NA" if rs is None else len(rs)))
                rs = rs or []
            per_arm[arm] = rs
        data[(task, mode, size)] = per_arm
    return data, problems


def finite_pos(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0


def cell_analysis(per_arm):
    """セルごとに medians・checksum 一致・組別 round 比を返す（不正は ok=False と理由）。"""
    res = {"ok": True, "reason": "", "median": {}, "checksum_ok": True, "ratios": {}}
    for arm in ARMS:
        rs = per_arm.get(arm, [])
        if len(rs) != ROUNDS or not all(finite_pos(r.get("median_s")) for r in rs):
            res["ok"], res["reason"] = False, "median_s が不正または件数不足 (%s)" % arm
            return res
        res["median"][arm] = statistics.median(r["median_s"] for r in rs)
    cks = []
    for arm in ARMS:
        for r in per_arm[arm]:
            c = r.get("checksum")
            if isinstance(c, bool) or not isinstance(c, (int, float)) or not math.isfinite(c):
                # 配列・オブジェクト・欠損・非有限・bool は構造不正。集合へ入れず（unhashable で
                # TypeError になるため）、不一致の FAIL とも区別して判定不能（前提不成立）へ倒す。
                res["ok"], res["reason"] = False, "checksum が有限の数値ではない (%s)" % arm
                res["checksum_ok"] = False
                return res
            cks.append(c)
    if len(set(cks)) != 1:
        res["checksum_ok"] = False
    for name, (den, num) in PAIRS.items():
        res["ratios"][name] = [per_arm[num][i]["median_s"] / per_arm[den][i]["median_s"] for i in range(ROUNDS)]
    return res


def gate_checks(root):
    """前提ゲート。{名前: (ok, 理由)}。ファイル欠損は未実測として不成立。"""
    g = {}

    def read(rel):
        try:
            with open(os.path.join(root, rel), encoding="utf-8") as f:
                return f.read()
        except OSError:
            return None

    t = read("tree_verify.txt")
    rcs = re.findall(r"^rc=(\d+)\s*$", t or "", re.M)
    g["P-TREE"] = (rcs == ["0"], "tree_verify.txt に rc=0 が 1 行だけある記録が無い（未実測または不成立）")
    s = read("sme_report.txt")
    lines = [l for l in (s or "").splitlines() if l.strip()]
    r0_arms = [l.split(":", 1)[0].strip() for l in lines]
    g["P-R0"] = (r0_arms == list(ARMS) and all("kernel_enabled: false" in l for l in lines), "before・prime・after の 3 腕それぞれに kernel_enabled: false の記録が無い")
    bs = read("bitdump/summary.txt")
    cmp_ = read("bitdump/cmp.txt")
    ok = False
    if bs and cmp_ is not None:
        shas = re.findall(r"_sha256=([0-9a-f]{64})", bs)
        nl = re.findall(r"_lines=(\d+)", bs)
        ok = len(set(shas)) == 1 and len(shas) == 3 and all(x == "6726847" for x in nl) and cmp_.strip() == ""
    lo = read("load_gate.log")
    # 実行の終了コード。欠損・複数行・非ゼロはいずれも不成立（fail-closed）。
    # 出力行が揃っていても、その後の実行失敗（コンパイルエラー・非ゼロ終了）を成立と見なさない。
    bd = re.findall(r"^bitdump exit=(\d+) cmp_exit=(\d+)\s*$", lo or "", re.M)
    bit_rc_ok = len(bd) == 1 and bd[0] == ("0", "0")
    g["P-BIT"] = (ok and bit_rc_ok, "3 腕の bitdump が 6,726,847 行で sha 一致・cmp 差分 0 かつ load_gate.log の bitdump exit=0 cmp_exit=0 となる記録が無い")
    rt = read("rt_before.log")
    rt_rc = re.findall(r"^rt_before exit=(\d+)\s*$", lo or "", re.M)
    g["P-RT"] = (bool(rt) and "FAILED" not in rt and "test result: ok" in rt and rt_rc == ["0"],
                 "before の cargo test が全件 pass かつ load_gate.log の rt_before exit=0 となる記録が無い")
    lk = read("hwlock.log")
    g["P-LOCK"] = (bool(lk) and "acquired" in lk and "released" in lk, "hwlock.log に取得・解放の記録が無い")
    # load_gate-* は 1 ファイルだけを期待する（複数あれば末尾の 1 つを無作為に選ぶ代わりに不成立）
    abd = os.path.join(root, "ab")
    names = sorted(n for n in (os.listdir(abd) if os.path.isdir(abd) else []) if n.startswith("load_gate-"))
    lg = read(os.path.join("ab", names[0])) if len(names) == 1 else None
    n_ok = len(re.findall(r"gate_ok=1\b", lg or ""))
    n_rounds = len(re.findall(r"^round\d+ ", lg or "", re.M))
    outer = read("load_gate_outer.log")
    g["P-GATE"] = (n_ok == ROUNDS and n_rounds == ROUNDS and bool(outer) and " pass" in outer,
                   "内側 8.0 の 5/5 round 通過（round 行ちょうど 5 件・全件 gate_ok=1・load_gate-* は 1 ファイル）と外側ゲート pass の記録が無い"
                   "（gate_ok=1 は %d 件・round 行 %d 件・load_gate-* %d ファイル）" % (n_ok, n_rounds, len(names)))
    g["P-RUNAB"] = (re.findall(r"^run_ab exit=(\d+)\s*$", lo or "", re.M) == ["0"], "run_ab exit=0 が 1 行だけある記録が無い")
    g["P-COLLECT"] = (re.findall(r"^collect exit=(\d+)\s*$", lo or "", re.M) == ["0"], "collect exit=0 が 1 行だけある記録が無い")
    return g


def fmt_ratios(r):
    return " / ".join("%.3f" % x for x in r)


def render(root, label):
    abdir = os.path.join(root, "ab")
    data, problems = load_cells(abdir, label)
    gates = gate_checks(root)
    out = ["# #3613 3 腕 A/B 集計（GB10・機械出力。手で直さない）", ""]
    out.append("ラベル: `%s` / 規則: `RULE.txt` §4（閾値 %.2f・%d round・5/5 一貫）" % (label, THRESHOLD, ROUNDS))
    out += ["", "## 前提ゲート", ""]
    for k, (ok, why) in gates.items():
        out.append("- %s: %s" % (k, "成立" if ok else "不成立 — " + why))
    if problems:
        out.append("- P-CELLS ほか入力不備:")
        out += ["  - " + p for p in problems[:30]]
    # 12 セルすべてを先に解析し、1 つでも不正（非有限・欠損・checksum 構造不正）なら P-CELLS 不成立として
    # 両判定セルを前提不成立にする（併記セルの不正を判定セルの分類へ持ち込まない。RULE.txt §3・§4-1）。
    cells = {}
    for key in ALL_CELLS:
        cells[key] = cell_analysis(data[key]) if not problems else {"ok": False, "reason": "入力不備", "median": {}, "checksum_ok": True, "ratios": {}}
    bad_cells = ["%s %d %s（%s）" % (k[0], k[2], k[1], c["reason"]) for k, c in cells.items() if not c["ok"]]
    if bad_cells and not problems:
        out.append("- P-CELLS 不成立（セル検証）:")
        out += ["  - " + b for b in bad_cells]
    pre_ok = not problems and not bad_cells and all(ok for ok, _ in gates.values())
    out += ["", "## 判定セルの帰属分類", ""]
    for task, mode, size in JUDGE_CELLS:
        a = cells[(task, mode, size)]
        cls = classify(a["ratios"], a["checksum_ok"], pre_ok and a["ok"])
        out.append("- **%s %d %s**: %s" % (task, size, mode, cls))
    out += ["", "## セル別の組比（後者/前者の round 別比と中央値。> 1.00 = 後退）", "",
            "| セル | 判定 | 組 | round 別比 | 中央値比 | 5/5 後退 |", "|---|---|---|---|---|---|"]
    for key in ALL_CELLS:
        a = cells[key]
        tag = "判定" if key in JUDGE_CELLS else "併記"
        name = "%s %d %s" % (key[0], key[2], key[1])
        if not a["ok"] or not a["ratios"]:
            out.append("| %s | %s | - | 未実測または不正（%s） | - | - |" % (name, tag, a["reason"]))
            continue
        for p, (den, num) in PAIRS.items():
            r = a["ratios"][p]
            out.append("| %s | %s | %s (%s/%s) | %s | %.3f | %s |" % (
                name, tag, p, num, den, fmt_ratios(r), statistics.median(r), "はい" if all(x > THRESHOLD for x in r) else "いいえ"))
    # 「全セル一致」は 12 セルすべてが検証を通り checksum が一致した場合に限る。構造不正・未比較のセルが
    # あるときは一致と書かず、不一致と判定不能を区別して出す。
    mismatch = [k for k, c in cells.items() if c["ok"] and not c["checksum_ok"]]
    if not problems and not bad_cells and not mismatch:
        ck_line = "全セル一致"
    elif mismatch:
        ck_line = "不一致あり（%d セル）" % len(mismatch)
    else:
        ck_line = "未確認（不正・未比較のセルあり。判定不能）"
    out += ["", "checksum 3 腕完全一致: " + ck_line]
    out.append("")
    out.append("注: 分類は帰属の候補であり判定（ADOPT／REJECT）ではない。既定 ON へ戻すかは所有者の判断。")
    return "\n".join(out) + "\n"


def self_test():
    def R(v):
        return [v] * ROUNDS

    ok = [1.1] * 5
    flat = [1.0, 0.99, 1.01, 1.0, 1.0]
    cases = [
        (dict(pair0=ok, pair1=flat, pair2=ok), True, True, C_SELECT),
        (dict(pair0=ok, pair1=ok, pair2=ok), True, True, C_SENS),
        (dict(pair0=flat, pair1=flat, pair2=ok), True, True, C_NOREPRO),
        (dict(pair0=ok, pair1=flat, pair2=flat), True, True, C_UNSEP),
        (dict(pair0=ok, pair1=flat, pair2=ok), False, True, C_FAIL),
        (dict(pair0=ok, pair1=flat, pair2=ok), True, False, C_PRECOND),
        (dict(pair0=[1.1, 1.1, 1.1, 1.1, 1.0], pair1=flat, pair2=ok), True, True, C_NOREPRO),
        (dict(pair0=R(1.0), pair1=R(1.0), pair2=R(1.0)), True, True, C_NOREPRO),
    ]
    for ratios, ck, pre, want in cases:
        got = classify(ratios, ck, pre)
        assert got == want, (ratios, ck, pre, got, want)
    # 前提不成立が FAIL より優先される
    assert classify(dict(pair0=ok, pair1=ok, pair2=ok), False, False) == C_PRECOND
    # cell_analysis: 不正な median_s は ok=False
    bad = {a: [{"median_s": float("nan"), "checksum": 1.0}] * ROUNDS for a in ARMS}
    assert cell_analysis(bad)["ok"] is False
    good = {a: [{"median_s": 1.0, "checksum": 1.0}] * ROUNDS for a in ARMS}
    ga = cell_analysis(good)
    assert ga["ok"] and ga["checksum_ok"] and ga["ratios"]["pair0"] == R(1.0)
    diff = {a: [{"median_s": 1.0, "checksum": 1.0 if a != "prime" else 2.0}] * ROUNDS for a in ARMS}
    assert cell_analysis(diff)["checksum_ok"] is False
    # 構造不正な checksum（配列・オブジェクト・欠損・文字列）は TypeError にせず ok=False（判定不能）へ倒す
    for badck in ([1.0, 2.0], {"a": 1}, None, "1.0", True, float("nan")):
        inv = {a: [{"median_s": 1.0, "checksum": badck}] * ROUNDS for a in ARMS}
        ia = cell_analysis(inv)
        assert ia["ok"] is False and ia["checksum_ok"] is False, badck
        assert classify(ia["ratios"], ia["checksum_ok"], True and ia["ok"]) == C_PRECOND, badck
    nock = {a: [{"median_s": 1.0}] * ROUNDS for a in ARMS}
    assert cell_analysis(nock)["ok"] is False
    # 前提ゲートの終了コード検証（P-RT・P-BIT）。出力が揃っていても exit 欠損・非ゼロは不成立
    import tempfile

    def gate_fixture(load_gate):
        d = tempfile.mkdtemp()
        os.makedirs(os.path.join(d, "bitdump"))
        files = {
            "rt_before.log": "test result: ok. 10 passed\n",
            "bitdump/summary.txt": "expected=6726847\n" + "".join(
                "%s_lines=6726847\n%s_sha256=%s\n" % (t, t, "a" * 64) for t in ("before", "prime", "after")),
            "bitdump/cmp.txt": "",
        }
        if load_gate is not None:
            files["load_gate.log"] = load_gate
        for rel, body in files.items():
            with open(os.path.join(d, rel), "w", encoding="utf-8") as f:
                f.write(body)
        return d

    good_lg = "rt_before exit=0\nrun_ab exit=0\nbitdump exit=0 cmp_exit=0\ncollect exit=0\n"
    g = gate_checks(gate_fixture(good_lg))
    assert g["P-RT"][0] and g["P-BIT"][0], g
    for lg in (None, "", "rt_before exit=101\nbitdump exit=0 cmp_exit=0\n", "bitdump exit=1 cmp_exit=0\nrt_before exit=0\n",
               "bitdump exit=0 cmp_exit=1\nrt_before exit=0\n", "rt_before exit=0\n", "bitdump exit=0 cmp_exit=0\n",
               "rt_before exit=0\nrt_before exit=1\nbitdump exit=0 cmp_exit=0\n"):
        g = gate_checks(gate_fixture(lg))
        want_rt = lg is not None and re.findall(r"^rt_before exit=(\d+)$", lg, re.M) == ["0"]
        want_bit = lg is not None and re.findall(r"^bitdump exit=(\d+) cmp_exit=(\d+)$", lg, re.M) == [("0", "0")]
        assert g["P-RT"][0] == want_rt and g["P-BIT"][0] == want_bit, (lg, g)
        assert not (g["P-RT"][0] and g["P-BIT"][0]), lg
    # render の全セル前提: 併記セル（train）の NaN で両判定セルが undetermined になり、checksum 行は一致と書かない
    def write_root(break_train=False, bad_checksum=False, extra_row=False):
        d = tempfile.mkdtemp()
        os.makedirs(os.path.join(d, "ab"))
        for arm in ARMS:
            for task in ("gemm", "train", "infer"):
                rows = []
                for t, m, n in ALL_CELLS:
                    if t != task:
                        continue
                    for _ in range(ROUNDS):
                        row = {"device": "cpu", "size": n, "mode": m, "median_s": 1.0, "checksum": 1.0}
                        if break_train and task == "train" and m == "reuse" and arm == "after":
                            row["median_s"] = float("nan")
                        if bad_checksum and task == "infer" and m == "fresh" and arm == "prime":
                            row["checksum"] = [1.0]
                        rows.append(row)
                if extra_row and task == "gemm" and arm == "before":
                    rows.append({"device": "cpu", "size": 4096, "mode": "fresh", "median_s": 1.0, "checksum": 1.0})
                with open(os.path.join(d, "ab", "results-%s-%s-cpu-%s.jsonl" % (arm, LABEL, task)), "w", encoding="utf-8") as f:
                    for r in rows:
                        f.write(json.dumps(r) + "\n")
        return d

    # 前提ゲートは全成立の fixture を使う。成立しないと分類が C_PRECOND になり検証にならない
    def full_gate_root(**kw):
        d = write_root(**kw)
        os.makedirs(os.path.join(d, "bitdump"))
        files = {
            "tree_verify.txt": "prime changed_lines=4\nafter changed_lines=2\nrc=0\n",
            "sme_report.txt": "".join("%s: sme_report=x kernel_enabled: false\n" % a for a in ARMS),
            "bitdump/summary.txt": "".join("%s_lines=6726847\n%s_sha256=%s\n" % (t, t, "a" * 64) for t in ARMS),
            "bitdump/cmp.txt": "",
            "rt_before.log": "test result: ok. 1 passed\n",
            "hwlock.log": "acquired at=x\nreleased at=y\n",
            "load_gate_outer.log": "start attempt=1 load1=0.1 gpu_util=0 pass\n",
            "load_gate.log": good_lg,
            "ab/load_gate-3613-x-cpu-%s.log" % LABEL: "".join("round%d gate=pass gate_ok=1 threshold=3.0\n" % i for i in range(1, ROUNDS + 1)),
        }
        for rel, body in files.items():
            with open(os.path.join(d, rel), "w", encoding="utf-8") as f:
                f.write(body)
        return d

    base = render(full_gate_root(), LABEL)
    assert "全セル一致" in base and "undetermined" not in base, base
    assert all(ok for ok, _ in gate_checks(full_gate_root()).values())
    brk = render(full_gate_root(break_train=True), LABEL)
    assert brk.count(C_PRECOND) == len(JUDGE_CELLS) and "全セル一致" not in brk, brk
    bck = render(full_gate_root(bad_checksum=True), LABEL)
    assert bck.count(C_PRECOND) == len(JUDGE_CELLS) and "全セル一致" not in bck and "判定不能" in bck, bck
    ext = render(full_gate_root(extra_row=True), LABEL)
    assert ext.count(C_PRECOND) == len(JUDGE_CELLS) and "全セル一致" not in ext, ext
    # 前提ゲートの厳格化: 複数 rc・round 行不足・load_gate-* 複数・腕名欠落は不成立
    def gate_with(rel, body):
        d = full_gate_root()
        with open(os.path.join(d, rel), "w", encoding="utf-8") as f:
            f.write(body)
        return gate_checks(d)

    assert not gate_with("tree_verify.txt", "rc=1\nrc=0\n")["P-TREE"][0]
    assert not gate_with("sme_report.txt", "before: kernel_enabled: false\n" * 3)["P-R0"][0]
    assert not gate_with("load_gate.log", good_lg + "run_ab exit=1\n")["P-RUNAB"][0]
    assert not gate_with("ab/load_gate-3613-x-cpu-%s.log" % LABEL, "round1 gate=pass gate_ok=1\n" * 4)["P-GATE"][0]
    assert not gate_with("ab/load_gate-3613-y-cpu-%s.log" % LABEL, "round1 gate=pass gate_ok=1\n")["P-GATE"][0]
    print("self-test ok")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--self-test":
        self_test()
    elif len(sys.argv) == 2 and sys.argv[1] == "gb10":
        root = os.path.join(HERE, "gb10")
        text = render(root, LABEL)
        with open(os.path.join(HERE, "aggregate-gb10.md"), "w", encoding="utf-8") as f:
            f.write(text)
        print(text)
    else:
        sys.exit("usage: aggregate.py gb10 | --self-test")
