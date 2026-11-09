#!/usr/bin/env python3
"""イシュー #4156: SME 既定 ON の新しい事前登録 A/B（Mac mini・GB10）の厳密集計。

役割: RULE.txt（同ディレクトリ・計測前に固定）の判定を、実測ログから機械的に出力する。
`docs/perf/logs/sme-default-on-ab-3232/aggregate.py` の派生だが、GB10 の扱いは固定せず RULE §9 に転記された
所有者決定（GB10-MODE・NOISE-BAND）を読んで分岐する。決定が未転記なら P-OWNER 不成立で undetermined（推定・代筆しない）。
標準ライブラリのみ。`python3 -I` で起動すること。結果の Markdown は手で直さない。

使い方:
    python3 -I aggregate.py <subdir> <host>   # subdir は本ファイルのあるディレクトリからの相対（例: gb10 gb10 を先に、次に macmini macmini）
    python3 -I aggregate.py --self-test

腕の向き: before = OFF（main・クリーン）、after = ON（on-arm.patch 適用）。比 = after / before（1.00 以下なら ON が速い）。
異常入力（重複・欠損・非有限値・未知のキー・型不正・ログ欠損）は捨てずに AggError へ写し、undetermined にする。
checksum または bitdump が ON/OFF で一致しない場合は FAIL（性能判定へ進まない）。
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
MACMINI_NONREACH = {"infer"}
# P-RT で ON 腕から除外するテスト名（計測前に固定。定数 true では作りの上で必ず落ちるドリフト検出）
RT_SKIP_TEST = "sme_production_disabled_after_gb10_reject"
GATE_FILE = "crates/backend-cpu/src/gemm_blis/mod.rs"
# 参考: #3231 の M4 Max bitdump（記録のみ）
M4MAX_BITDUMP_LINES = 6726847


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
        raise AggError("ログ欠損: " + os.path.relpath(path, HERE))
    return open(path, encoding="utf-8").read()


def finite_num(v):
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return False
    try:
        return math.isfinite(v)
    except (OverflowError, ValueError, TypeError):
        # JSON の巨大整数（10**400 等）は float へ変換できず OverflowError になる。不正値として扱い集計を止めない
        return False


def load_rows(path, task, bad, ck_all):
    """正常行だけを返し、不正行は理由を bad へ収集する（例外で全行を捨てない）。

    正常行の checksum を保持して不一致 FAIL の確認を続けるため（RULE §4: FAIL は常に最優先）。
    ログ欠損（ファイルなし）のみ AggError。"""
    rows = []
    for i, line in enumerate(read_text(path).splitlines(), 1):
        if not line.strip():
            continue
        where = "%s:%d" % (os.path.basename(path), i)
        try:
            o = json.loads(line)
        except ValueError:
            bad.append(where + " JSON 不正")
            continue
        if not isinstance(o, dict):
            bad.append(where + " 行がオブジェクトではない")
            continue
        if o.get("task") != task:
            bad.append(where + " task が不一致: %r" % (o.get("task"),))
            continue
        if o.get("device") != "cpu":
            bad.append(where + " device が cpu ではない")
            continue
        size, mode = o.get("size"), o.get("mode", "fresh")
        if isinstance(size, bool) or not isinstance(size, int) or not isinstance(mode, str):
            bad.append(where + " size/mode の型が不正")
            continue
        if (task, size, mode) not in CELLS:
            bad.append(where + " 未知のセル: %s size=%s mode=%s" % (task, size, mode))
            continue
        ck_ok = finite_num(o.get("checksum"))
        if ck_ok:
            ck_all.setdefault((task, size, mode), []).append(o["checksum"])
        if not finite_num(o.get("median_s")) or o["median_s"] <= 0:
            bad.append(where + " median_s が不正")
            continue
        if not ck_ok:
            bad.append(where + " checksum が非有限・欠損")
            continue
        rows.append(o)
    return rows


def find_one(pattern):
    hits = sorted(glob.glob(pattern))
    if len(hits) != 1:
        raise AggError("ファイルが 1 件でない（%d 件）: %s" % (len(hits), os.path.relpath(pattern, HERE)))
    return hits[0]


def load_arm_cells(ab, arm, problems, ck_all):
    """読めた行だけを返し、欠測・件数不足・不正行は problems へ収集する（例外で打ち切らない）。

    checksum 不一致（FAIL）の確認を欠測で妨げないため（RULE §4: FAIL は常に最優先）。"""
    out = {}
    for task in KNOWN_TASKS:
        try:
            path = find_one(os.path.join(ab, "results-%s-*-cpu-%s.jsonl" % (arm, task)))
            rows = load_rows(path, task, problems, ck_all)
        except AggError as e:
            problems.append("arm=%s: %s" % (arm, e))
            continue
        for o in rows:
            out.setdefault((task, o["size"], o.get("mode", "fresh")), []).append(o)
    for cell in CELLS:
        n = len(out.get(cell, []))
        if n != ROUNDS:
            problems.append("P-CELLS 不成立: arm=%s cell=%s は %d 行（%d 行ちょうどを要求）" % (arm, cell, n, ROUNDS))
    return out


def load_torch(ab):
    """PyTorch は記録専用（RULE §6）。欠測・空・不足・不正は (None, 注記) とし、本体判定へ波及させない。"""
    hits = sorted(glob.glob(os.path.join(ab, "results-torch-*-cpu.jsonl")))
    if not hits:
        return None, "PyTorch 未計測（記録のみのため判定には影響しない）"
    rows = []
    try:
        with open(hits[0], encoding="utf-8") as f:
            lines = f.read().splitlines()
    except (OSError, ValueError) as e:
        return None, "PyTorch ログを読めない（記録のみ・判定に影響しない）: %s" % e
    for i, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            o = json.loads(line)
        except ValueError:
            return None, "PyTorch 行が不正（%s:%d・記録のみ・判定に影響しない）" % (os.path.basename(hits[0]), i)
        if not isinstance(o, dict) or not isinstance(o.get("task"), str) or o["task"] not in KNOWN_TASKS \
                or not finite_num(o.get("median_s")) or o["median_s"] <= 0 \
                or isinstance(o.get("size"), bool) or not isinstance(o.get("size"), int):
            return None, "PyTorch 行が不正（%s:%d・記録のみ・判定に影響しない）" % (os.path.basename(hits[0]), i)
        rows.append(o)
    by = {}
    for o in rows:
        by.setdefault((o.get("task"), o["size"]), []).append(o["median_s"])
    for key in [("gemm", s) for s in SIZES] + [("train", 64), ("infer", 64)]:
        if len(by.get(key, [])) != ROUNDS:
            return None, "PyTorch の %s は %d 行（%d 行未満・過多のため相手比を省略。記録のみ・判定に影響しない）" % (
                key, len(by.get(key, [])), ROUNDS)
    return by, None


def fmt(x):
    return "%.3f" % x


# ---------------------------------------------------------------- P-OWNER（RULE §9 の所有者決定の機械読み取り）
OWNER_URL_RE = re.compile(r"^https://github\.com/Fandhe-AI/fandhe-ai/(issues|pull)/[0-9]+#issuecomment-[0-9]+$")
GB10_MODES = ("GB10-RECORD", "GB10-NBSEP", "GB10-GATE-ONLY-STRUCTURAL")
OWNER_ENUMS = (
    ("GB10-MODE", GB10_MODES),
    ("NOISE-BAND", ("adopted", "not-adopted")),
    ("UNDETERMINED-HANDLING", ("auto-reremeasure", "return-to-owner")),
)
TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def rule_bytes_sha():
    import hashlib
    with open(os.path.join(HERE, "RULE.txt"), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def read_owner():
    """RULE.txt の所有者決定欄を読み (値の dict, 不成立理由の list) を返す。

    各キーはちょうど 1 回だけ行頭に現れ、列挙値と回答コメント URL の形式に完全一致しなければ不成立（fail-closed）。
    所有者の決定を推定・補完しない（`<未決>` は列挙外として不成立になる）。"""
    text = read_text(os.path.join(HERE, "RULE.txt"))
    vals, probs = {}, []
    for key, allowed in OWNER_ENUMS:
        hits = re.findall(r"^%s: *(.*?) *$" % re.escape(key), text, re.M)
        if len(hits) != 1:
            probs.append("P-OWNER 不成立: %s が %d 件（1 件ちょうどを要求）" % (key, len(hits)))
            continue
        if hits[0] not in allowed:
            probs.append("P-OWNER 不成立: %s が未決または列挙外 (%s)" % (key, hits[0] or "空"))
            continue
        vals[key] = hits[0]
        uh = re.findall(r"^%s-URL: *(.*?) *$" % re.escape(key), text, re.M)
        if len(uh) != 1 or not OWNER_URL_RE.match(uh[0]):
            probs.append("P-OWNER 不成立: %s-URL（所有者回答コメント）が欠落または形式不正" % key)
    return vals, probs


def nb_sep(on, off):
    """NB-SEP（noise-band-decision.md）。ON 腕の 5 round すべてで比 > 1.0 かつ ON 腕の最小 > OFF 腕の最大（厳密不等号）。
    on は ON 腕の round 別の値、off は OFF 腕の round 別の値（後退判定では median_s、1 位化では PyTorch 比）。"""
    return min(on) > max(off)


def evaluate(base, host):
    """host のログ 1 式を評価して dict を返す。AggError は呼び出し側で undetermined へ写す。"""
    ab = os.path.join(base, "ab")
    res = {"host": host, "gate_fail": [], "notes": []}
    problems = []
    ck_all = {}
    arm = {a: load_arm_cells(ab, a, problems, ck_all) for a in ("before", "after")}
    ck_bad = []
    for cell in CELLS:
        if len(set(ck_all.get(cell, []))) > 1:
            ck_bad.append(cell)
    res["checksum_bad"] = ck_bad
    res["incomplete"] = bool(problems)
    res["gate_fail"].extend(problems)
    stats = {}
    for cell in ([] if problems else CELLS):
        ratios = [arm["after"][cell][i]["median_s"] / arm["before"][cell][i]["median_s"] for i in range(ROUNDS)]
        offt = [arm["before"][cell][i]["median_s"] for i in range(ROUNDS)]
        ont = [arm["after"][cell][i]["median_s"] for i in range(ROUNDS)]
        stats[cell] = {
            "ratios": ratios,
            "median": statistics.median(ratios),
            "n_le": sum(1 for r in ratios if r <= 1.0),
            "n_gt": sum(1 for r in ratios if r > 1.0),
            "checksum": arm["after"][cell][0]["checksum"],
            "off_med": statistics.median(offt),
            "on_med": statistics.median(ont),
            "sep": all(r > 1.0 for r in ratios) and nb_sep(ont, offt),
        }
    res["stats"] = stats
    gf = res["gate_fail"]
    if problems:
        gf.append("前提不成立: 計測ログの欠測・不足のため性能判定（比・後退・到達判定）を行わない")

    def rd(path):
        try:
            return read_text(path)
        except AggError as e:
            gf.append("前提不成立: %s" % e)
            return None

    # P-OWNER（RULE §9）と prereg_append.txt（追記後 RULE の sha256 と準備時刻）
    try:
        owner, oprobs = read_owner()
    except (AggError, OSError) as e:
        owner, oprobs = {}, ["P-OWNER 不成立: %s" % e]
    res["owner"] = owner
    gf.extend(oprobs)
    pre = rd(os.path.join(base, "prereg_append.txt"))
    prepared = None
    if pre is not None:
        m = re.search(r"^rule_sha256=([0-9a-f]{64})$", pre, re.M)
        if not m or m.group(1) != rule_bytes_sha():
            gf.append("P-OWNER: prereg_append.txt の rule_sha256 が現在の RULE.txt と不一致")
        m = re.search(r"^prepared_at=(\S+)$", pre, re.M)
        if m and TS_RE.match(m.group(1)):
            prepared = m.group(1)
        else:
            gf.append("P-OWNER: prereg_append.txt の prepared_at が読めない")
    t = rd(os.path.join(base, "tree_verify.txt"))
    if t is not None and t.strip() != GATE_FILE:
        gf.append("P-TREE: 差分ファイルが 1 件（%s）ではない" % GATE_FILE)
    t = rd(os.path.join(base, "patch_sha256.txt"))
    if t is not None:
        try:
            if t.strip() != rule_sha():
                gf.append("P-TREE: patch の sha256 が RULE と不一致")
        except (AggError, OSError) as e:
            gf.append("前提不成立: %s" % e)
    t = rd(os.path.join(base, "gate_constant.txt"))
    if t is not None:
        consts = [re.sub(r"^\d+:", "", l) for l in t.splitlines()]
        if consts != ["const SME_PRODUCTION_ENABLED: bool = false;", "const SME_PRODUCTION_ENABLED: bool = true;"]:
            gf.append("P-TREE: 定数行が OFF=false・ON=true ではない")
    expect_kernel = "true" if host == "macmini" else "false"
    rep = rd(os.path.join(base, "sme_report.txt"))
    if rep is not None:
        for t in ("off", "on"):
            m = re.search(r"^%s: .*kernel_enabled: (true|false)" % t, rep, re.M)
            if not m:
                gf.append("P-R0: %s の sme_report が読めない" % t)
            elif m.group(1) != expect_kernel:
                gf.append("P-R0: %s の kernel_enabled=%s（期待 %s）" % (t, m.group(1), expect_kernel))
    lg = rd(os.path.join(base, "load_gate.log"))
    lg_lines = lg.splitlines() if lg is not None else []
    if lg is not None:
        for key in ("run_ab exit=0", "collect exit=0"):
            if key not in lg_lines:
                gf.append("P-RUNAB/P-COLLECT: '%s' がない" % key)
        # P-RT は両腕。ON は計測前に固定した 1 件だけを除外する
        if "rt_on exit=0 skip=%s" % RT_SKIP_TEST not in lg_lines:
            gf.append("P-RT: ON ツリーの cargo test（%s を除外）が全件 pass ではない（後退あり相当・要調査）" % RT_SKIP_TEST)
        if "rt_off exit=0" not in lg_lines:
            gf.append("P-RT: OFF ツリーの cargo test が全件 pass ではない")
    hw = rd(os.path.join(base, "hwlock.log"))
    if hw is not None:
        macq = re.search(r"^acquired at=(\S+)$", hw, re.M)
        if not macq or not re.search(r"^released-pending at=", hw, re.M):
            gf.append("P-LOCK: hwlock.log に取得・解放の記録がない")
        elif prepared is not None and not (TS_RE.match(macq.group(1)) and prepared < macq.group(1)):
            gf.append("P-OWNER: prereg_append.txt の準備時刻がロック取得時刻より前ではない（事前登録の順序不成立）")
    t = rd(os.path.join(base, "rev-stamp.txt"))
    if t is not None:
        parts = t.split()
        if len(parts) != 2 or parts[1] != parts[0] + "+on-arm.patch":
            gf.append("rev-stamp が 2 行（OFF の SHA と <SHA>+on-arm.patch）ではない")
    try:
        inner = rd(find_one(os.path.join(ab, "load_gate-*-cpu-*.log")))
    except AggError as e:
        gf.append("前提不成立: %s" % e)
        inner = None
    if inner is not None:
        rounds_ok = re.findall(r"^round(\d) gate=pass gate_ok=1 ", inner, re.M)
        if sorted(rounds_ok) != ["1", "2", "3", "4", "5"]:
            gf.append("P-GATE: 内側ゲートが 5/5 round で gate_ok=1 ではない（通過 round=%s）" % ",".join(rounds_ok))
    if host == "gb10":
        outer = rd(os.path.join(base, "load_gate_outer.log"))
        if outer is not None and not re.search(r"compute_apps=0 pass$", outer, re.M):
            gf.append("P-GATE: 外側ゲート（load1 < 1.0・GPU 0%・compute プロセス 0）の通過記録がない")
    # P-BIT（両機体）
    res["bitdump_mismatch"] = False
    bd = re.search(r"^bitdump exit=(\d+) cmp_exit=(\d+)$", lg or "", re.M)
    summ = rd(os.path.join(base, "bitdump", "summary.txt"))
    if not bd:
        if lg is not None:
            gf.append("P-BIT: 実行記録がない")
    else:
        res["bitdump_cmp_exit"] = int(bd.group(2))
        if bd.group(1) != "0":
            gf.append("P-BIT: 行数が期待と一致しない・実行失敗（比較不能）")
        elif bd.group(2) == "1":
            res["bitdump_mismatch"] = True
        elif bd.group(2) != "0":
            gf.append("P-BIT: cmp が実行エラー（exit=%s・比較不能）" % bd.group(2))
    if summ is not None:
        res["bitdump_summary"] = summ.strip()
        so = re.search(r"^off_sha256=([0-9a-f]{64})$", summ, re.M)
        sn = re.search(r"^on_sha256=([0-9a-f]{64})$", summ, re.M)
        if not so or not sn:
            gf.append("P-BIT: summary の sha256 が読めない")
        elif so.group(1) != sn.group(1):
            res["bitdump_mismatch"] = True
    # PyTorch 相手比（記録のみ。torch median_s / fandhe median_s の round 別比）
    torch, torch_note = load_torch(ab)
    res["torch"] = None
    if problems:
        res["notes"].append("計測ログ欠測のため PyTorch 相手比を省略")
    elif torch is None:
        res["notes"].append(torch_note)
    else:
        tr = {}
        for cell in CELLS:
            key = (cell[0], cell[1])
            per = {}
            for a in ("before", "after"):
                per[a] = [torch[key][i] / arm[a][cell][i]["median_s"] for i in range(ROUNDS)]
            tr[cell] = per
        res["torch"] = tr
    if host == "macmini":
        res["reach"] = [c for c in CELLS if c[0] not in MACMINI_NONREACH]
    else:
        res["reach"] = []
    res["gate_unmet"] = bool(gf)
    res["gb10_modes"] = None
    if problems:
        res["consistent_regress"] = []
        res["sep_regress"] = []
        res["reach_ok"] = False
        res["verdict"] = "FAIL" if (ck_bad or res["bitdump_mismatch"]) else "undetermined"
        if res["bitdump_mismatch"]:
            res["notes"].append("bitdump が ON/OFF で不一致（bit 同一でない）")
        return res
    consistent = [c for c in CELLS if stats[c]["n_gt"] == ROUNDS]
    sep = [c for c in CELLS if stats[c]["sep"]]
    res["consistent_regress"] = consistent
    res["sep_regress"] = sep
    res["reach_ok"] = all(stats[c]["median"] <= 1.0 and stats[c]["n_le"] == ROUNDS for c in res["reach"])
    if host == "gb10":
        # 3 方式の判定を全て併記する。拘束力があるのは P-OWNER で選ばれた 1 方式だけ
        res["gb10_modes"] = {
            "GB10-RECORD": "pass",
            "GB10-NBSEP": "REJECT" if sep else "pass",
            "GB10-GATE-ONLY-STRUCTURAL": "undetermined" if sep else "pass",
        }
    if ck_bad or res["bitdump_mismatch"]:
        res["verdict"] = "FAIL"
    elif gf:
        res["verdict"] = "undetermined"
    elif host == "macmini":
        if consistent:
            res["verdict"] = "REJECT"
        else:
            res["verdict"] = "ADOPT 候補（Mac mini 側条件）" if res["reach_ok"] else "undetermined"
    else:
        res["verdict"] = res["gb10_modes"][owner["GB10-MODE"]]
    if res["bitdump_mismatch"]:
        res["notes"].append("bitdump が ON/OFF で不一致（bit 同一でない）")
    return res


def combine(mac, gb):
    """総合判定。評価順 FAIL > undetermined（ゲート不成立）> REJECT > ADOPT > undetermined。"""
    hs = [h for h in (mac, gb) if h is not None]
    if any(h["verdict"] == "FAIL" for h in hs):
        return "FAIL"
    if any(h["gate_unmet"] for h in hs):
        return "undetermined（前提ゲート不成立）"
    if any(h["verdict"] == "REJECT" for h in hs):
        return "REJECT"
    if mac is None or gb is None:
        return "undetermined（%s 未計測のため ADOPT 不可）" % ("Mac mini" if mac is None else "GB10")
    if mac["verdict"].startswith("ADOPT") and gb["verdict"] == "pass":
        return "ADOPT"
    return "undetermined"


def render(res, other):
    host = res["host"]
    L = ["# #4156 厳密集計 (%s)" % host, "",
         "機械出力（`python3 -I aggregate.py` の出力。手で直さない）。腕: before = OFF・after = ON。比 = ON / OFF（<= 1.00 なら ON が速い）。", ""]
    owner = res.get("owner") or {}
    L += ["## 所有者決定（RULE §9）", "",
          "- GB10-MODE: %s" % owner.get("GB10-MODE", "未決"),
          "- NOISE-BAND: %s" % owner.get("NOISE-BAND", "未決"),
          "- UNDETERMINED-HANDLING: %s" % owner.get("UNDETERMINED-HANDLING", "未決"), ""]
    if res.get("incomplete"):
        L += ["計測ログの欠測・不足のためセルごとの比・相手比は算出しない（前提ゲート・FAIL 確認のみ実施）。", ""]
    else:
        L += ["## セルごと（5 round 中央値）", "",
              "| セル | 到達区分 | OFF 中央値(s) | ON 中央値(s) | ON/OFF 中央値比 | round 別比 | <=1.00 | >1.00 | 分離した後退 | checksum |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for c in CELLS:
            s = res["stats"][c]
            cls = "到達" if c in res["reach"] else "ガード（非到達）"
            L.append("| %s %d %s | %s | %.6g | %.6g | %s | %s | %d | %d | %s | %r |" % (
                c[0], c[1], c[2], cls, s["off_med"], s["on_med"], fmt(s["median"]),
                " ".join("r%d=%s" % (i + 1, fmt(r)) for i, r in enumerate(s["ratios"])), s["n_le"], s["n_gt"],
                "あり" if s["sep"] else "なし", s["checksum"]))
        if res.get("gb10_modes"):
            L += ["", "## GB10 の 3 方式の判定（拘束力があるのは P-OWNER で選ばれた 1 方式だけ）", "",
                  "| 方式 | 判定 | 拘束 |", "|---|---|---|"]
            for k in GB10_MODES:
                L.append("| %s | %s | %s |" % (k, res["gb10_modes"][k], "**拘束**" if owner.get("GB10-MODE") == k else "参考"))
        L += ["", "## 1 位化の記録（PyTorch 比 = torch median / fandhe median。判定には使わない）", ""]
        if res["torch"] is None:
            L.append("PyTorch 相手比なし（未計測・欠測。注記参照）。6 フレームワークでの 1 位化は Phase 6 で確定する。")
        else:
            nb = owner.get("NOISE-BAND") == "adopted"
            if nb:
                L += ["| セル | OFF 中央値比 | ON 中央値比 | ON の NB-RANGE5（最小〜最大） | NB-SEP | 記録 |", "|---|---|---|---|---|---|"]
            else:
                L += ["| セル | OFF round 別比 | ON round 別比 | ON 中央値比 |", "|---|---|---|---|"]
            for c in CELLS:
                if c[0] != "gemm" or c[1] not in (512, 1024, 2048):
                    continue
                t = res["torch"][c]
                m_off, m_on = statistics.median(t["before"]), statistics.median(t["after"])
                if nb:
                    s_ok = all(r > 1.0 for r in t["after"]) and nb_sep(t["after"], t["before"])
                    L.append("| %s %d %s | %s | %s | %s〜%s | %s | %s |" % (
                        c[0], c[1], c[2], fmt(m_off), fmt(m_on), fmt(min(t["after"])), fmt(max(t["after"])),
                        "成立" if s_ok else "不成立", "1 位（確定）" if s_ok else "改善傾向・確定せず"))
                else:
                    L.append("| %s %d %s | %s | %s | %s |" % (
                        c[0], c[1], c[2], " ".join(fmt(r) for r in t["before"]), " ".join(fmt(r) for r in t["after"]), fmt(m_on)))
            L += ["", "（PyTorch のみとの比較。6 フレームワークでの 1 位化は Phase 6 で確定する。）"]
    L += ["", "## 前提ゲート", ""]
    L += ["- " + g for g in res["gate_fail"]] if res["gate_fail"] else ["- すべて成立"]
    if res.get("bitdump_summary"):
        L += ["", "## bitdump（ON/OFF の bit 同一。#3231 の M4 Max は %d 行・参考で判定しない）" % M4MAX_BITDUMP_LINES, "", "```", res["bitdump_summary"], "```"]
    if res["checksum_bad"]:
        L += ["", "## checksum 不一致（FAIL）", ""] + ["- %s" % (c,) for c in res["checksum_bad"]]
    L += ["", "## 後退", "",
          "- 5/5 一貫の後退セル: %s" % (", ".join("%s %d %s" % c for c in res["consistent_regress"]) or "なし"),
          "- 分離した後退セル（5 round すべて ON/OFF > 1.00 かつ ON の最小 > OFF の最大）: %s" % (", ".join("%s %d %s" % c for c in res["sep_regress"]) or "なし")]
    for n in res["notes"]:
        L.append("- 注記: " + n)
    verdict = res["verdict"]
    if host == "macmini":
        verdict = combine(res, other)
    L += ["", "## 総合判定", "", "**%s**: %s" % (host, verdict), ""]
    return "\n".join(L), verdict


def run(subdir, host, out=sys.stdout):
    base = os.path.join(HERE, subdir)
    try:
        res = evaluate(base, host)
    except AggError as e:
        out.write("# #4156 厳密集計 (%s)\n\n**undetermined**: %s\n" % (host, e))
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
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


URL = "https://github.com/Fandhe-AI/fandhe-ai/issues/4143#issuecomment-1"


def _mk_rule(d, mode="GB10-GATE-ONLY-STRUCTURAL", nb="adopted", und="return-to-owner", ok=True):
    import hashlib
    body = "on-arm.patch sha256: %s\n" % ("ab" * 32)
    if ok:
        body += "GB10-MODE: %s\nGB10-MODE-URL: %s\nNOISE-BAND: %s\nNOISE-BAND-URL: %s\nUNDETERMINED-HANDLING: %s\nUNDETERMINED-HANDLING-URL: %s\n" % (
            mode, URL, nb, URL, und, URL)
    else:
        body += "GB10-MODE: <未決>\nNOISE-BAND: <未決>\nUNDETERMINED-HANDLING: <未決>\n"
    _write(os.path.join(d, "RULE.txt"), body)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _fixture(root, host, rule_sha_hex, on_ratio=0.9, ck_bad=False, drop_row=False, dup=False, gate_ok=True, regress=None,
             torch_mode="full", rm_tree=False, bd_mismatch=False, bd_fail=False, bd_cmp=None, bad_row=False,
             rt_on_fail=False, rt_off_fail=False, bad_r0=False, prereg_late=False, prereg_sha_bad=False, outer_fail=False):
    ab = os.path.join(root, "ab")
    for arm in ("before", "after"):
        for task in KNOWN_TASKS:
            lines = []
            for (_, size, mode) in [c for c in CELLS if c[0] == task]:
                for r in range(ROUNDS):
                    base_t = 0.001 * size
                    t = base_t
                    if regress and size == 512 and task == "gemm":
                        if regress == "sep":
                            t = base_t * (1.1 if arm == "after" else 1.0)
                        else:  # fake: 毎 round で ON > OFF だが round 間の揺れが大きく分離しない
                            off = base_t * (1.0 + 0.3 * r)
                            t = off * (1.02 if arm == "after" else 1.0)
                    elif arm == "after":
                        t = base_t * on_ratio
                    ck = 1.5 + (0.1 if (ck_bad and arm == "after" and size == 256 and mode == "fresh" and task == "gemm") else 0)
                    lines.append(json.dumps({"task": task, "device": "cpu", "size": size, "mode": mode, "median_s": t, "checksum": ck}))
            if drop_row and arm == "after" and task == "gemm":
                lines.pop()
            if dup and arm == "after" and task == "gemm":
                lines.append(lines[0])
            if bad_row and arm == "after" and task == "gemm":
                lines.append(json.dumps({"task": "gemm", "device": "cpu", "size": 256, "mode": "fresh", "median_s": -1, "checksum": 1.5}))
            _write(os.path.join(ab, "results-%s-4156-r1-cpu-%s.jsonl" % (arm, task)), "\n".join(lines) + "\n")
    tl = []
    for r in range(ROUNDS):
        for key in [("gemm", s) for s in SIZES] + [("train", 64), ("infer", 64)]:
            tl.append(json.dumps({"task": key[0], "device": "cpu", "size": key[1], "median_s": 0.001 * key[1] * 1.2}))
    tp = os.path.join(ab, "results-torch-4156-sme-4156-r1-cpu.jsonl")
    if torch_mode == "full":
        _write(tp, "\n".join(tl) + "\n")
    elif torch_mode == "partial":
        _write(tp, "\n".join(tl[:-3]) + "\n")
    elif torch_mode == "garbage":
        _write(tp, "not json\n")
    g = "".join("round%d gate=%s gate_ok=%d threshold=3.0 load1=0.5 waited_s=0 at=x\n" % (r, "pass" if gate_ok else "timeout", 1 if gate_ok else 0) for r in range(1, 6))
    _write(os.path.join(ab, "load_gate-4156-sme-cpu-4156-r1.log"), "threshold=3.0\n" + g)
    _write(os.path.join(root, "tree_verify.txt"), GATE_FILE + "\n")
    _write(os.path.join(root, "patch_sha256.txt"), rule_sha() + "\n")
    _write(os.path.join(root, "gate_constant.txt"), "3537:const SME_PRODUCTION_ENABLED: bool = false;\n3540:const SME_PRODUCTION_ENABLED: bool = true;\n")
    k = "true" if host == "macmini" else "false"
    if bad_r0:
        k = "false" if host == "macmini" else "true"
    _write(os.path.join(root, "sme_report.txt"), "off: sme_report= kernel_enabled: %s\non: sme_report= kernel_enabled: %s\n" % (k, k))
    lg = "run_ab exit=0\ncollect exit=0\n"
    lg += "rt_on exit=%d skip=%s\n" % (1 if rt_on_fail else 0, RT_SKIP_TEST)
    lg += "rt_off exit=%d\n" % (1 if rt_off_fail else 0)
    lg += "bitdump exit=%d cmp_exit=%d\n" % (1 if bd_fail else 0, bd_cmp if bd_cmp is not None else (1 if bd_mismatch else 0))
    sha_a = "a" * 64
    sha_b = "b" * 64 if bd_mismatch else sha_a
    _write(os.path.join(root, "bitdump", "summary.txt"),
           "off_lines=6726847 on_lines=6726847 expected=6726847\noff_sha256=%s\non_sha256=%s\n" % (sha_a, sha_b))
    if host == "gb10":
        _write(os.path.join(root, "load_gate_outer.log"),
               "start attempt=1 load1=0.1 gpu_util=0 compute_apps=%d %s\n" % (1 if outer_fail else 0, "wait" if outer_fail else "pass"))
    _write(os.path.join(root, "load_gate.log"), lg)
    _write(os.path.join(root, "hwlock.log"), "acquired at=2026-01-02T00:00:00Z\nreleased-pending at=2026-01-02T01:00:00Z\n")
    _write(os.path.join(root, "prereg_append.txt"), "rule_sha256=%s\nprepared_at=%s\n" % (
        "0" * 64 if prereg_sha_bad else rule_sha_hex, "2026-01-03T00:00:00Z" if prereg_late else "2026-01-01T00:00:00Z"))
    _write(os.path.join(root, "rev-stamp.txt"), "abc\nabc+on-arm.patch\n")
    if rm_tree:
        os.remove(os.path.join(root, "tree_verify.txt"))


def _res(host, mode="GB10-GATE-ONLY-STRUCTURAL", nb="adopted", owner_ok=True, **kw):
    """RULE を別ディレクトリへ合成して HERE を差し替え、1 機の評価結果を返す（実物の RULE.txt は読まない）。"""
    global HERE
    saved = HERE
    with tempfile.TemporaryDirectory() as hd, tempfile.TemporaryDirectory() as td:
        HERE = hd
        try:
            sha_hex = _mk_rule(hd, mode=mode, nb=nb, ok=owner_ok)
            _fixture(td, host, sha_hex, **kw)
            try:
                return evaluate(td, host)
            except AggError as e:
                return {"verdict": "undetermined(AggError: %s)" % e, "gate_unmet": True}
        finally:
            HERE = saved


def _v(host, **kw):
    return _res(host, **kw)["verdict"]


def self_test():
    ok = True

    def check(name, got, want):
        nonlocal ok
        good = got.startswith(want)
        ok = ok and good
        print("%s %s -> %s" % ("ok  " if good else "FAIL", name, got))

    # NB-SEP 境界（厳密不等号）
    check("nb_sep: 最小 == 最大 は不成立", str(nb_sep([2.0] * 5, [2.0] * 5)), "False")
    check("nb_sep: 最小が最大をちょうど超えない（等号）は不成立", str(nb_sep([2, 3, 4, 5, 6], [1, 1, 1, 1, 2])), "False")
    check("nb_sep: 分離すれば成立", str(nb_sep([2.1, 3, 4, 5, 6], [1, 1, 1, 1, 2])), "True")
    check("macmini 正常（ON が速い）", _v("macmini"), "ADOPT")
    check("gb10 正常", _v("gb10"), "pass")
    check("欠損行", _v("macmini", drop_row=True), "undetermined")
    check("重複行", _v("macmini", dup=True), "undetermined")
    check("不正行のみは undetermined", _v("macmini", bad_row=True), "undetermined")
    check("checksum 不一致", _v("macmini", ck_bad=True), "FAIL")
    check("欠測 + checksum 不一致でも FAIL", _v("macmini", ck_bad=True, drop_row=True), "FAIL")
    check("Mac mini 到達セル 5/5 後退（分離）は REJECT", _v("macmini", regress="sep"), "REJECT")
    check("Mac mini 偽の 5/5 後退（分離なし）も REJECT（事前登録の Mac mini 条件は 5/5 一貫）", _v("macmini", regress="fake"), "REJECT")
    check("ON が遅い（1.05）", _v("macmini", on_ratio=1.05), "REJECT")
    check("ON 比がちょうど 1.00 は後退ではない", _v("macmini", on_ratio=1.0), "ADOPT")
    check("ゲート不通過", _v("macmini", gate_ok=False), "undetermined")
    check("P-BIT 不一致（sha）は FAIL", _v("macmini", bd_mismatch=True), "FAIL")
    check("P-BIT 不一致 + ゲート不通過でも FAIL", _v("gb10", bd_mismatch=True, gate_ok=False), "FAIL")
    check("P-BIT 実行失敗は undetermined", _v("gb10", bd_fail=True), "undetermined")
    check("P-BIT cmp 実行エラーは undetermined", _v("gb10", bd_cmp=2), "undetermined")
    check("P-R0 外れ（macmini）は undetermined", _v("macmini", bad_r0=True), "undetermined")
    check("P-R0 外れ（gb10）は undetermined", _v("gb10", bad_r0=True), "undetermined")
    check("P-RT ON 失敗は undetermined", _v("macmini", rt_on_fail=True), "undetermined")
    check("P-RT OFF 失敗は undetermined", _v("gb10", rt_off_fail=True), "undetermined")
    check("外側ゲートの compute プロセスありは undetermined", _v("gb10", outer_fail=True), "undetermined")
    check("ログ欠損のみは undetermined", _v("macmini", rm_tree=True), "undetermined")
    check("P-OWNER 未決は undetermined（macmini）", _v("macmini", owner_ok=False), "undetermined")
    check("P-OWNER 未決は undetermined（gb10）", _v("gb10", owner_ok=False), "undetermined")
    check("P-OWNER 未決でも checksum 不一致 FAIL を保つ", _v("gb10", owner_ok=False, ck_bad=True), "FAIL")
    check("prereg の rule sha 不一致は undetermined", _v("macmini", prereg_sha_bad=True), "undetermined")
    check("prereg がロック取得より後は undetermined", _v("macmini", prereg_late=True), "undetermined")
    for tm in ("partial", "garbage", "none"):
        check("torch %s でも macmini は ADOPT 候補" % tm, _v("macmini", torch_mode=tm), "ADOPT")
    # GB10 3 方式 x {偽の 5/5 後退, 分離した後退}
    want = {
        ("GB10-RECORD", "fake"): "pass", ("GB10-RECORD", "sep"): "pass",
        ("GB10-NBSEP", "fake"): "pass", ("GB10-NBSEP", "sep"): "REJECT",
        ("GB10-GATE-ONLY-STRUCTURAL", "fake"): "pass", ("GB10-GATE-ONLY-STRUCTURAL", "sep"): "undetermined",
    }
    for (mode, kind), w in want.items():
        check("gb10 %s / %s" % (mode, kind), _v("gb10", mode=mode, regress=kind), w)
    # 総合
    def overall(mac_kw, gb_kw):
        return combine(_res("macmini", **mac_kw), _res("gb10", **gb_kw))
    check("総合: 両機 pass なら ADOPT", overall({}, {}), "ADOPT")
    check("総合: GB10 未計測では ADOPT にしない", combine(_res("macmini"), None), "undetermined")
    check("総合: GB10 NBSEP の分離後退は REJECT", overall({}, {"mode": "GB10-NBSEP", "regress": "sep"}), "REJECT")
    check("総合: GB10 RECORD は分離後退でも ADOPT", overall({}, {"mode": "GB10-RECORD", "regress": "sep"}), "ADOPT")
    check("総合: GB10 GATE-ONLY の分離後退は undetermined", overall({}, {"regress": "sep"}), "undetermined")
    check("総合: GB10 checksum 不一致は FAIL", overall({}, {"ck_bad": True}), "FAIL")
    check("総合: macmini 欠測でも GB10 FAIL を伝播", overall({"drop_row": True}, {"ck_bad": True}), "FAIL")
    check("総合: GB10 前提不成立 + macmini REJECT は undetermined", overall({"regress": "sep"}, {"gate_ok": False}), "undetermined")
    check("総合: macmini REJECT", overall({"regress": "sep"}, {}), "REJECT")
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
