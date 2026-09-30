#!/usr/bin/env python3
"""イシュー #2118: SME_MIN_K 候補（64／128／256）の M4 Max 再実測 + GB10 非後退の集計。

役割: 同ディレクトリの RULE.txt（事前登録）の判定規則を機械的に適用し
`aggregate.md` 用の Markdown を標準出力へ出す。規則を新たに決める場所ではない
（規則の正は RULE.txt。ここに書く定数・語彙はそれと同一で、変更は追記方式の
訂正のみ）。python3 標準ライブラリのみ。

判定の構造（RULE.txt §14・§15）:
  - 系列（M4 Max の候補 K・GB10 の候補 K）ごとに、ディスク上の記録を `load_m4_series`／
    `load_gb10_series` が 1 つの dict（series）へ読み込む。
  - 判定不能条件は `preconditions(series)` の 1 か所で全て評価し、不成立の理由（前提 ID 付き）の
    リストを返す。総合判定 `series_verdict` と AC1 欄 `ac1_verdict` は、最初にこれを呼び、
    空リストのときにだけ判定を返す。判定語彙（FAIL／REJECT／ADOPT 候補／後退あり／pass 等）を
    返す関数はこの 2 つ（と RT 部分判定 `rt_verdict`）に限り、self-test が AST で照合する。
  - 前提 ID と RULE.txt 条項の対応は `PRECONDITIONS`、条項ごとの self-test ケースは
    `CLAUSE_CASES`（条項 ID は `RULE_CLAUSE_IDS`）に持つ。

再利用（フォークしない）:
  - `../cpu-gemm-sme-fmopa-1587/aggregate.py` の `parse`／`candidates`（R4 の
    16 格子点のログ解析とパレート極小）。#1978 の M4 Max 実測と同じ解析にする。
  - `scripts/bench/framework-compare/compare_gemm_ab.py` の `load_rows`／
    `group_by_cell`／`evaluate_cell`／`per_run_ratios`（R1／R2 の round 比・
    中央値・checksum 完全一致）。`run_ab_sme_cpu.sh` が呼ぶ判定と同じ関数。

入力ディレクトリ配置（orchestrate_m4max.sh・gb10/orchestrate_gb10.sh の出力）:
  <base>/m4max/{sme_r4_grid_run{1..5}.log,load_gate_r4.log,r4_head.txt}
  <base>/m4max/r1r2/k{K}/results-{before,after}-2118-m4max-k{K}-cpu-{gemm,train,infer}.jsonl
  <base>/m4max/r1r2/k{K}/{env_info.txt,patch_sha256.txt,tree_diff.txt,tree_verify.txt,gate_constant.txt}
  <base>/gb10/k{K}/r1r2/results-{before,after}-2118-gb10-k{K}-cpu-*.jsonl
  <base>/gb10/k{K}/{sme_report.txt,rt_result.txt,load_gate_outer.log,patch_sha256.txt,tree_*.txt,gate_constant.txt}
系列ディレクトリが無い（または空の）系列は「未実測」と出力する（値の捏造・推定はしない）。

使い方: aggregate.py [--self-test] [<base_dir>] > aggregate.md
"""
import ast
import importlib.util
import json
import os
import re
import shutil
import statistics
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[3]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_R4 = _load("sme1587_aggregate", _HERE.parent / "cpu-gemm-sme-fmopa-1587" / "aggregate.py")
_CMP = _load("compare_gemm_ab", _ROOT / "scripts" / "bench" / "framework-compare" / "compare_gemm_ab.py")

KS_CANDIDATE = (64, 128, 256)
# RULE.txt §3: 候補 K の R4 格子は min(m,n) ∈ {256, 512} × k ∈ {32,64,128,256} のうち k >= K
R4_MN = (256, 512)
KNOWN_RT_FAIL = "gemm_blis::tests::sme_production_enabled_is_false_pending_measurement"
# RULE.txt ヘッダ「登録時点の main HEAD」。lib_trees.sh の SME2118_REGISTERED_BASE と同一（self-test で照合）
REGISTERED_BASE = "0b25525fa4026b951021a5d0da3a9d613b507502"
MOD_REL = "crates/backend-cpu/src/gemm_blis/mod.rs"

# RULE.txt §2 の到達セル表（出典: bench-fandhe/src/main.rs BATCH=64・D_IN=784・D_HIDDEN=256・
# D_OUT=10。L1 d_weight TN 784x256x64 が K=64 のときだけ SME に到達する）。
CELLS = (
    [("gemm", s, m) for s in (512, 1024, 2048) for m in ("fresh", "reuse")]
    + [("train", 64, m) for m in ("fresh", "reuse")]
    + [("infer", 64, m) for m in ("fresh", "reuse")]
)

# --- 判定語彙（RULE.txt §6・§8・§11）。これらを返してよいのは series_verdict／ac1_verdict のみ ---
UNDETERMINED = "undetermined"
V_FAIL = "FAIL"
V_REJECT = "REJECT"
V_ADOPT = "ADOPT 候補"
REFERENCE_ADOPT = "ADOPT 候補相当（参考・record_only）"
REFERENCE_REJECT = "REJECT（参考・record_only）"
G_REGRESSION = "後退あり"
G_SUSPECT = "後退あり相当（要調査）"
G_PASS = "pass"
G_REFERENCE_SUFFIX = "（参考）"
AC1_OK = "成立"
AC1_NG = "不成立"
AC1_UNDETERMINED = "判定不能"

# --- 前提（RULE.txt §15）: (ID, 対象機体, AC1 欄にも課すか, 出典条項, 内容) ---
# preconditions() はこの表の ID だけを理由の先頭に使う（self-test で照合）。
PRECONDITIONS = (
    ("P-TREE", ("m4max", "gb10"), True, "§1・§13・§15",
     "登録 sha 固定・指紋差分 mod.rs 1 件・定数 2 行・成功記録（patch_sha256／tree_diff／gate_constant／tree_verify）"),
    ("P-RUNAB", ("m4max", "gb10"), True, "§13", "run_ab_sme_cpu.sh の終了コード 0 の記録"),
    ("P-COLLECT", ("m4max", "gb10"), True, "§14", "成果物収録の終了コード 0 の記録"),
    ("P-CELLS", ("m4max", "gb10"), True, "§8・§14",
     "全 10 セルが判定可能（JSONL・round 欠損・重複行・値不正・checksum 欠損／非数値なし）"),
    ("P-R4-BASE", ("m4max",), False, "§13・§15", "R4 の登録 sha 固定の記録（r4_head.txt）"),
    ("P-R4-EXEC", ("m4max",), False, "§13", "R4 の 5 run 全て exit=0 grep_exit=0 と series done"),
    ("P-R4-LOG", ("m4max",), False, "§14", "R4 の 5 run 全てが 32 点をちょうど 1 回ずつ含む"),
    ("P-R0", ("gb10",), False, "§8", "R0 成立（両腕 kernel_enabled: false）"),
    ("P-RT", ("gb10",), False, "§9・§15", "RT の判定記録（rt_verdict 行）"),
)
_PRECOND_BY_ID = {p[0]: p for p in PRECONDITIONS}

# RULE.txt の判定不能系の条項（self-test の条項 → ケース対応表のキー）。RULE.txt に条項を足したら
# ここと CLAUSE_CASES の両方へ足す（RULE_KEYWORD_LINES の行数照合が追加漏れを検出する）。
RULE_CLAUSE_IDS = (
    "§1/P-TREE:tree_diff が 2 件",
    "§1/P-TREE:gate_constant の K 不一致",
    "§1/P-TREE:tree_verify 記録なし",
    "§13/P-TREE:登録 sha 以外を基準にした",
    "§13/P-RUNAB:run_ab 非ゼロ",
    "§13/P-RUNAB:run_ab 記録なし",
    "§14/P-COLLECT:collect 非ゼロ",
    "§14/P-COLLECT:collect 記録なし",
    "§14/P-CELLS:JSONL 欠損",
    "§14/P-CELLS:round 欠損",
    "§14/P-CELLS:重複行",
    "§14/P-CELLS:checksum 欠損",
    "§13/P-R4-BASE:r4_head が登録 sha でない",
    "§13/P-R4-EXEC:R4 run 非ゼロ",
    "§13/P-R4-EXEC:series done なし",
    "§13/P-R4-EXEC:R4 実行記録なし",
    "§14/P-R4-LOG:R4 重複行",
    "§14/P-R4-LOG:R4 run ログ欠損",
    "§8/P-R0:R0 不成立",
    "§8/P-R0:R0 記録なし",
    "§9/P-RT:rt_verdict 記録なし",
)

# RULE.txt の各節で判定不能系の語を含む行数（self-test が RULE.txt から数えて照合する）。
# 条項の追加・変更でずれたら、RULE_CLAUSE_IDS・CLAUSE_CASES・PRECONDITIONS を見直してから更新する。
RULE_KEYWORD_RE = re.compile(r"判定不能|undetermined|停止|中止|欠損|非ゼロ|前提|記録なし|記録がない")
RULE_KEYWORD_LINES = {"1": 1, "5": 1, "6": 1, "7": 1, "8": 2, "10": 1, "13": 3, "14": 4, "15": 5}


def reached(k, task):
    """RULE.txt §2 の表。gemm は全候補で到達・train は K<=64 のみ・infer は全候補で非到達。"""
    if task == "gemm":
        return True
    if task == "train":
        return k <= 64
    return False


def r4_holds(ok, k):
    """RULE.txt §3: 格子 {256,512} x {k' in 32..256, k' >= K} の全点が 5/5 run で SME>=NEON。"""
    pts = [(mn, kk) for mn in R4_MN for kk in _R4.KS if kk >= k]
    return all(ok[p] for p in pts), pts


def classify_cell(is_reached, ratios):
    """RULE.txt §4。ratios は round ごとの after/before 比（5 件）。

    到達セル: 中央値<=1.00 かつ 5/5 round<=1.00 で adopt。全 round>1.00 なら reject（5/5 一貫の後退）。
    非到達セル: 全 round>1.00 のときだけ reject 材料。それ以外は noise（値を併記するのみ）。
    """
    med = statistics.median(ratios)
    all_le = all(r <= 1.00 for r in ratios)
    all_gt = all(r > 1.00 for r in ratios)
    return {
        "median": med,
        "all_le_1": all_le,
        "all_gt_1": all_gt,
        "adopt": bool(is_reached and med <= 1.00 and all_le),
        "reject": bool(all_gt),
    }


CHECKSUM_FAIL_REASON = "checksum が複合判定を外れる"


def is_checksum_fail(c):
    """R2 の checksum 不一致（RULE.txt §5）。丸め後の完全一致でない ok セル、または複合判定外れ。
    checksum の欠損・非数値は「判定不能」であり FAIL に数えない（記録の欠損と不一致を混同しない）。"""
    if c["status"] == "ok":
        return not c["exact"]
    return (c.get("reason") or "").startswith(CHECKSUM_FAIL_REASON)


def validate_cells(cells):
    """入力の完全性（構造）検査。preconditions が最初に呼ぶ（RULE.txt §14）。

    cells は CELLS と同一のキー集合（task, size, mode）を重複なく持つこと。重複・欠落・未知キーは
    「記録が読めない」のではなく集計側の入力構造の破綻なので ValueError（fail-closed。値を推定しない）。
    """
    keys = [(c["task"], c["size"], c["mode"]) for c in cells]
    dup = sorted({k for k in keys if keys.count(k) > 1})
    unknown = sorted(set(keys) - set(CELLS))
    missing = sorted(set(CELLS) - set(keys))
    if dup or unknown or missing or len(keys) != len(CELLS):
        raise ValueError(f"セル集合が RULE.txt §2 の 10 セルと一致しない: dup={dup} unknown={unknown} missing={missing}")


def _cell_name(c):
    return f"{c['task']} {c['size']}/{c['mode']}"


def preconditions(series, scope="overall"):
    """RULE.txt §15 の前提を全て評価し、不成立の理由（先頭は PRECONDITIONS の ID）のリストを返す。

    空リストのときだけ判定してよい。scope="ac1" は §6 の AC1 直接判定欄用で、R4 を使わないため
    AC1 欄に課さない前提（PRECONDITIONS の 3 列目が False）を除く。セル集合の構造破綻は ValueError。
    series の各記録は True（成立）・False（不成立）・None（記録なし）で、True 以外は全て不成立とする。
    """
    validate_cells(series["cells"])
    machine = series["machine"]
    reasons = []

    def need(pid, ok, detail):
        mach, for_ac1 = _PRECOND_BY_ID[pid][1], _PRECOND_BY_ID[pid][2]
        if machine not in mach or (scope == "ac1" and not for_ac1):
            return
        if ok is not True:
            state = "記録なし" if ok is None else "不成立"
            reasons.append(f"{pid}: {detail}（{state}）")

    need("P-TREE", series["tree"], "ツリーの記録が登録 sha・mod.rs 1 件・候補 K の定数と一致")
    need("P-RUNAB", series["run_ab"], "run_ab_sme_cpu.sh の終了コード 0")
    need("P-COLLECT", series["collect"], "成果物収録の終了コード 0")
    bad = [c for c in series["cells"] if not (c["status"] == "ok" or is_checksum_fail(c))]
    need("P-CELLS", not bad, "全 10 セル判定可能。判定不能セル: "
         + ("、".join(f"{_cell_name(c)}={c['reason']}" for c in bad) or "なし"))
    if machine == "m4max":
        r4 = series["r4"]
        need("P-R4-BASE", r4["base"], "R4 の r4_head.txt が登録 sha")
        need("P-R4-EXEC", r4["exec"], "R4 5 run の exit=0 grep_exit=0 と series done")
        need("P-R4-LOG", r4["valid_runs"] == 5 and r4["grid"] is not None,
             f"R4 の 32 点ちょうど 1 回ずつの run が 5 本（有効 {r4['valid_runs']} 本）")
    else:
        need("P-R0", series["r0"], "R0（両腕 kernel_enabled: false）")
        need("P-RT", True if series["rt"] in ("pass", "regression-suspect") else None, "RT の rt_verdict 記録")
    return reasons


def series_verdict(series):
    """系列の総合判定（RULE.txt §6・§8・§11・§14・§15）。(判定, 理由リスト) を返す。

    最初に preconditions を評価し、不成立が 1 つでもあれば undetermined と理由を返す（この順を崩さない。
    判定を返す経路はこの関数だけ）。前提成立後の判定:
      M4 Max: FAIL（R2 checksum 不一致）> REJECT（到達セル 5/5 後退）> ADOPT 候補 > undetermined。
              §11 の record_only 系列は REJECT／ADOPT とも参考表示（REFERENCE_*）。
      GB10: 後退あり（5/5 後退セル・checksum 不一致）> 後退あり相当（RT の既知 FAIL 以外）> pass。
            §8 の外側専有ゲート不通過は「（参考）」を付ける。
    """
    reasons = preconditions(series)
    if reasons:
        return UNDETERMINED, reasons
    cells = series["cells"]
    official = series["official"]
    if series["machine"] == "m4max":
        if any(is_checksum_fail(c) for c in cells):
            return V_FAIL, []
        if any(c["reached"] and c["reject"] for c in cells):
            return (V_REJECT if official else REFERENCE_REJECT), []
        if (
            r4_holds(series["r4"]["grid"], series["k"])[0]
            and all(c["adopt"] for c in cells if c["reached"])
            and not any(c["reject"] for c in cells if not c["reached"])
        ):
            return (V_ADOPT if official else REFERENCE_ADOPT), []
        return UNDETERMINED, ["§6: 前提は全て成立したが FAIL・REJECT・ADOPT 候補のいずれの条件も満たさない"]
    if any(is_checksum_fail(c) or (c["status"] == "ok" and c["reject"]) for c in cells):
        v = G_REGRESSION
    elif series["rt"] == "regression-suspect":
        v = G_SUSPECT
    else:
        v = G_PASS
    return (v if official else v + G_REFERENCE_SUFFIX), []


def ac1_verdict(series):
    """AC1 の直接判定欄（RULE.txt §6。総合判定とは独立）。(欄の値, 理由リスト) を返す。

    R4 を使わない欄のため R4 系の前提は課さないが、それ以外の前提（P-TREE・P-RUNAB・P-COLLECT・P-CELLS）が
    不成立なら判定不能。成立 = 到達セルが 5 round 中央値<=1.00 かつ 5/5<=1.00、かつ全セル checksum 完全一致。
    """
    reasons = preconditions(series, scope="ac1")
    if reasons:
        return AC1_UNDETERMINED, reasons
    cells = series["cells"]
    if any(c["status"] != "ok" for c in cells):
        return AC1_NG, []
    ok = all(c["adopt"] for c in cells if c["reached"]) and all(c["exact"] for c in cells)
    return (AC1_OK if ok else AC1_NG), []


def rt_verdict(fail_names):
    """RULE.txt §9 の RT 部分判定（gb10/orchestrate_gb10.sh の rt_verdict と同じ規則。総合判定の入力）。
    既知 FAIL 1 件のみ（または FAIL なし）は pass。それ以外が 1 件でもあれば要調査。"""
    others = [n for n in fail_names if n != KNOWN_RT_FAIL]
    return "pass" if not others else "regression-suspect"


def evaluate_series(dirpath, label):
    """dirpath の JSONL（before/after × gemm/train/infer）を CELLS の順に判定した dict 列を返す。"""
    cells = []
    per_task = {}
    for task in ("gemm", "train", "infer"):
        paths = [os.path.join(dirpath, f"results-{arm}-{label}-cpu-{task}.jsonl") for arm in ("before", "after")]
        if not all(os.path.exists(p) for p in paths):
            per_task[task] = None
            continue
        size_set = _CMP._size_set_for("cpu", "full", task)
        b, _ = _CMP.load_rows(paths[0], device="cpu", task=task, size_set=size_set)
        a, _ = _CMP.load_rows(paths[1], device="cpu", task=task, size_set=size_set)
        gb, ga = _CMP.group_by_cell(b), _CMP.group_by_cell(a)
        # 未知セル（RULE.txt §2 の 10 セル外）が混入していれば fail-closed（読み飛ばして判定しない）
        expected = {(s, m) for t, s, m in CELLS if t == task}
        for g in (gb, ga):
            extra = sorted(set(g) - expected, key=repr)
            if extra:
                raise ValueError(f"{task}: RULE.txt §2 に無い未知セルが JSONL にある: {extra}")
        per_task[task] = (gb, ga)
    for task, size, mode in CELLS:
        rec = {"task": task, "size": size, "mode": mode, "reached": None, "status": "missing",
               "reason": "JSONL 欠損", "exact": False, "adopt": False, "reject": False,
               "ratios": None, "median": None}
        pt = per_task.get(task)
        if pt is not None:
            bc, ac = pt
            br, ar = bc.get((size, mode), []), ac.get((size, mode), [])
            ev = _CMP.evaluate_cell(br, ar, 1.00)
            rec["status"] = ev["status"]
            rec["reason"] = ev.get("reason")
            if ev["status"] == "ok":
                ratios = _CMP.per_run_ratios(br, ar)
                rec["ratios"] = ratios
                rec["exact"] = bool(ev["checksum_exact_match"])
                rec["median"] = ev["ratio"]
                # 到達判定は候補 K 依存のため呼び出し側で classify する（ここでは比の記録まで）
        cells.append(rec)
    return cells


def apply_k(cells, k):
    """候補 K に応じて到達分類と classify_cell を適用した dict 列を返す（元は変更しない）。
    k=None は GB10（SME 非対応のため全セル非到達。RULE.txt §8）。"""
    out = []
    for c in cells:
        d = dict(c)
        d["reached"] = False if k is None else reached(k, c["task"])
        if c["status"] == "ok" and c["ratios"] is not None:
            d.update(classify_cell(d["reached"], c["ratios"]))
            d["median"] = c["median"]
        out.append(d)
    return out


# --- 記録の読み込み（True=成立・False=不成立・None=記録なし）---

def load_gate_series(path):
    """load-gate ログから正式系列（official かつ全 round pass）かを返す。ログ不在は None。"""
    if not os.path.exists(path):
        return None
    text = Path(path).read_text(encoding="utf-8")
    official = "series=official" in text
    gates = re.findall(r"^round\d+ gate=(\S+)", text, re.M)
    return official and len(gates) == 5 and all(g == "pass" for g in gates)


def r4_runs_ok(path):
    """R4 系列の実行成否（RULE.txt §13）。load_gate_r4.log に run1..5 が全て `exit=0 grep_exit=0` で、
    かつ `series done` 行がある場合のみ True。ログ不在は None、異常終了・欠損・途中終了は False。"""
    if not os.path.exists(path):
        return None
    text = Path(path).read_text(encoding="utf-8")
    for i in range(1, 6):
        if not re.search(rf"^run{i} exit=0 grep_exit=0(\s|$)", text, re.M):
            return False
    if len(re.findall(r"^run\d+ exit=", text, re.M)) != 5:
        return False
    return bool(re.search(r"^series done\s*$", text, re.M))


def _gate_log_official(path, prefix):
    """load-gate ログ（`<prefix>N gate=pass|...` 行）が 5/5 通過か。ログ不在は None。"""
    if not os.path.exists(path):
        return None
    gs = re.findall(rf"^{prefix}\d+ gate=(\S+)", Path(path).read_text(encoding="utf-8"), re.M)
    return len(gs) == 5 and all(g == "pass" for g in gs)


def outer_gate_official(path):
    """GB10 外側専有ゲート（RULE.txt §8）。`start attempt=N ... pass` 行があれば正式。fail(reference)・ログ不在は参考。"""
    if not os.path.exists(path):
        return None
    text = Path(path).read_text(encoding="utf-8")
    return bool(re.search(r"^start attempt=\d+ .* pass$", text, re.M))


def head_record_ok(path):
    """`head=<sha>` 行が登録 sha（REGISTERED_BASE）か（RULE.txt §13）。ファイル・行なしは None。"""
    if not os.path.exists(path):
        return None
    m = re.search(r"^head=(\S+)\s*$", Path(path).read_text(encoding="utf-8"), re.M)
    return None if m is None else m.group(1) == REGISTERED_BASE


def tree_record_ok(dirpath, k):
    """P-TREE（RULE.txt §1・§13・§15）: lib_trees.sh の sme2118_prepare_trees が残す 4 記録の照合。

    patch_sha256.txt の head が登録 sha・tree_verify.txt が `trees_ok head=<登録 sha> K=<k>`・
    tree_diff.txt が mod.rs の 1 行のみ・gate_constant.txt が before=(false, 64)／after=(true, K)。
    いずれかのファイルが無ければ None、内容不一致は False。値を推定しない（記録の照合のみ）。
    """
    d = Path(dirpath)
    files = [d / n for n in ("patch_sha256.txt", "tree_verify.txt", "tree_diff.txt", "gate_constant.txt")]
    if not all(f.exists() for f in files):
        return None
    head = head_record_ok(str(files[0]))
    verify = files[1].read_text(encoding="utf-8").splitlines() == [f"trees_ok head={REGISTERED_BASE} K={k}"]
    diff = files[2].read_text(encoding="utf-8").splitlines() == [MOD_REL]
    const = files[3].read_text(encoding="utf-8").splitlines() == [
        "before: const SME_PRODUCTION_ENABLED: bool = false; / const SME_MIN_K: usize = 64;",
        f"after:  const SME_PRODUCTION_ENABLED: bool = true; / const SME_MIN_K: usize = {k};",
    ]
    return bool(head and verify and diff and const)


def _exit_record(text, pattern):
    """終了コード記録 1 件の評価。0 なら True・非ゼロなら False・記録なしは None。
    同じ記録が複数行ある場合は、どれを正とするか決められないため不成立（False）とする（fail-closed）。"""
    ms = re.findall(pattern, text, re.M)
    if len(ms) != 1:
        return None if not ms else False
    return int(ms[0]) == 0


def exit_records_m4max(env_info_path):
    """M4 Max: env_info.txt の `run_ab_exit=N`・`collect_exit=N`（RULE.txt §13・§14）を (run_ab, collect) で返す。"""
    if not os.path.exists(env_info_path):
        return None, None
    text = Path(env_info_path).read_text(encoding="utf-8")
    return _exit_record(text, r"\brun_ab_exit=(\d+)"), _exit_record(text, r"\bcollect_exit=(\d+)")


def exit_records_gb10(rt_result_path):
    """GB10: rt_result.txt の `run_ab rc=N`・`collect rc=N` を (run_ab, collect) で返す。意味は M4 Max と同じ。"""
    if not os.path.exists(rt_result_path):
        return None, None
    text = Path(rt_result_path).read_text(encoding="utf-8")
    return _exit_record(text, r"^run_ab rc=(\d+)"), _exit_record(text, r"^collect rc=(\d+)")


def parse_r4_strict(text):
    """R4 ログ 1 run 分を厳密に解析する。16 格子点 x {NEON, SME} = 32 点を重複・欠落・未知なく
    ちょうど 1 回ずつ含む場合のみ dict を返し、それ以外（重複行・欠落・未知の格子点）は None。
    1587 の `parse` は重複行を後勝ちで潰し件数だけ 32 に見えることがあるため、行数まで照合する。"""
    hits = [_R4.LINE.search(line.strip()) for line in text.splitlines()]
    keys = [(m.group(1), int(m.group(2)), int(m.group(4))) for m in hits if m]
    expected = {(v, mn, kk) for v in ("NEON", "SME") for mn in _R4.MN for kk in _R4.KS}
    if len(keys) != len(expected) or set(keys) != expected:
        return None
    return _R4.parse(text)


def load_r4(base):
    """M4 Max の R4 系列（候補共通の 1 系列。RULE.txt §3）を読む。ログの有無にかかわらず全記録を評価する。"""
    m4 = Path(base) / "m4max"
    runs = []
    for i in range(1, 6):
        f = m4 / f"sme_r4_grid_run{i}.log"
        if f.exists():
            r = parse_r4_strict(f.read_text(encoding="utf-8"))
            if r is not None:
                runs.append(r)
    grid, ratios = None, None
    if len(runs) == 5:
        grid, ratios = {}, {}
        for mn in _R4.MN:
            for kk in _R4.KS:
                rs = [r[("SME", mn, kk)] / r[("NEON", mn, kk)] for r in runs]
                ratios[(mn, kk)] = rs
                grid[(mn, kk)] = all(x >= 1.0 for x in rs)
    log = str(m4 / "load_gate_r4.log")
    return {"valid_runs": len(runs), "grid": grid, "ratios": ratios, "exec": r4_runs_ok(log),
            "gate": _gate_log_official(log, "run"), "base": head_record_ok(str(m4 / "r4_head.txt"))}


def _series_dir_present(d):
    return d.exists() and any(d.iterdir())


def load_m4_series(base, k, r4):
    """M4 Max の候補 K の系列を読む。系列ディレクトリが無い（空）なら None（未実測）。"""
    label = f"2118-m4max-k{k}"
    d = Path(base) / "m4max" / "r1r2" / f"k{k}"
    if not _series_dir_present(d):
        return None
    run_ab, collect = exit_records_m4max(str(d / "env_info.txt"))
    r1_gate = load_gate_series(str(d / f"load-gate-1978-cpu-{label}.log"))
    return {
        "machine": "m4max", "k": k, "cells": apply_k(evaluate_series(str(d), label), k),
        "tree": tree_record_ok(d, k), "run_ab": run_ab, "collect": collect, "r4": r4,
        "r1_gate": r1_gate,
        # RULE.txt §11: R4・R1 の両負荷ゲートを通過した系列だけを正式とする（不通過は record_only）
        "official": bool(r4["gate"]) and r1_gate is True,
    }


def load_gb10_series(base, k):
    """GB10 の候補 K の系列を読む。系列ディレクトリが無い（空）なら None（未実測）。
    R0 不成立で r1r2 を作らず中止した系列も、ここで読んで preconditions に判定不能の理由を出させる。"""
    label = f"2118-gb10-k{k}"
    d = Path(base) / "gb10" / f"k{k}"
    if not _series_dir_present(d):
        return None
    rep = d / "sme_report.txt"
    r0 = None if not rep.exists() else rep.read_text(encoding="utf-8").count("kernel_enabled: false") >= 2
    rt = None
    rtf = d / "rt_result.txt"
    if rtf.exists():
        m = re.search(r"^rt_verdict=(\S+)", rtf.read_text(encoding="utf-8"), re.M)
        if m:
            rt = "pass" if m.group(1) == "pass" else "regression-suspect"
    run_ab, collect = exit_records_gb10(str(rtf))
    return {
        "machine": "gb10", "k": k, "cells": apply_k(evaluate_series(str(d / "r1r2"), label), None),
        "tree": tree_record_ok(d, k), "run_ab": run_ab, "collect": collect, "r0": r0, "rt": rt,
        "outer_gate": outer_gate_official(str(d / "load_gate_outer.log")),
        # RULE.txt §8: 外側専有ゲート不通過（またはログなし）の系列は「参考」
        "official": outer_gate_official(str(d / "load_gate_outer.log")) is True,
    }


# --- 描画 ---

def _fmt_ratios(c):
    if not c.get("ratios"):
        return "-"
    return ", ".join(f"{x:.4f}" for x in c["ratios"])


def _ids(reasons):
    return "・".join(r.split(":", 1)[0] for r in reasons)


def render_cells(cells):
    lines = ["| セル | 分類 | 5 round の比 | 中央値 | checksum | 判定 |", "|---|---|---|---:|---|---|"]
    for c in cells:
        cls = "到達" if c["reached"] else "非到達"
        if c["status"] != "ok":
            lines.append(f"| {_cell_name(c)} | {cls} | - | - | - | 判定不能: {c['reason']} |")
            continue
        if c["reached"]:
            j = "ADOPT 候補条件成立" if c["adopt"] else ("5/5 後退" if c["reject"] else "ADOPT 候補条件不成立")
        else:
            j = "REJECT 材料（5/5 後退）" if c["reject"] else "後退の証拠なし"
        lines.append(f"| {_cell_name(c)} | {cls} | {_fmt_ratios(c)} | {c['median']:.4f} | "
                     f"{'完全一致' if c['exact'] else '不一致'} | {j} |")
    return "\n".join(lines)


def render_m4max(base):
    out = ["# SME_MIN_K 候補の M4 Max 再実測（イシュー #2118）\n"]
    r4 = load_r4(base)
    if r4["grid"] is not None:
        out += ["## R4 格子（ratio = SME/NEON。5/5 run で >=1.0 が SME>=NEON）\n",
                "| min(m,n) | k | 5 run の比 | 中央値 | 5/5 SME>=NEON |", "|---:|---:|---|---:|---|"]
        for mn in _R4.MN:
            for kk in _R4.KS:
                rs = r4["ratios"][(mn, kk)]
                out.append(f"| {mn} | {kk} | {', '.join(f'{x:.3f}' for x in rs)} | "
                           f"{statistics.median(rs):.3f} | {'yes' if r4['grid'][(mn, kk)] else 'no'} |")
        cand = _R4.candidates(r4["grid"])
        out.append("\n参考（パレート極小の採用候補）: " + (
            "、".join(f"`min(m,n) >= {a}` かつ `k >= {b}`" for a, b in cand) if cand else "なし"))
    else:
        out.append("## R4 格子\n\n未実測または不完全（`sme_r4_grid_run{1..5}.log` が揃っていない、"
                   "または 32 点を重複・欠落なく含まない run がある）。")
    out.append("\nR4 実行記録: " + {True: "5 run 全て exit=0 と series done", False: "異常終了・欠損・途中終了",
                                    None: "記録なし"}[r4["exec"]]
               + "／登録 sha: " + {True: "一致", False: "不一致", None: "記録なし"}[r4["base"]]
               + "／負荷ゲート: " + {True: "5/5 通過（正式）", False: "不通過を含む（record_only）",
                                    None: "ログなし（record_only 扱い）"}[r4["gate"]])
    out.append("\nR4 系の前提（P-R4-BASE・P-R4-EXEC・P-R4-LOG）は全候補の総合判定の前提（RULE.txt §15）。"
               "不成立なら全候補が undetermined になる。\n")
    out.append("## 候補別判定\n")
    summary = ["| K | R4 | R1 到達セル AC1 欄 | 総合判定 | 系列 |", "|---:|---|---|---|---|"]
    detail = []
    verdicts = {}
    for k in KS_CANDIDATE:
        s = load_m4_series(base, k, r4)
        if s is None:
            summary.append(f"| {k} | - | 未実測 | 未確定（実測未実施） | - |")
            continue
        v, reasons = series_verdict(s)
        a, a_reasons = ac1_verdict(s)
        verdicts[k] = v
        r4_pre = [r for r in preconditions(s) if r.startswith("P-R4-")]
        r4col = "判定不能" if r4_pre else ("成立" if r4_holds(r4["grid"], k)[0] else "不成立")
        series = {True: "正式（R1 load ゲート 5/5 通過）", False: "record_only（参考）",
                  None: "ゲートログなし（record_only 扱い）"}[s["r1_gate"]]
        if s["r1_gate"] is True and not r4["gate"]:
            series = "record_only（R4 負荷ゲート不通過・未確認）"
        vcol = f"{v}（{_ids(reasons)}）" if v == UNDETERMINED and reasons and reasons[0].startswith("P-") else v
        acol = f"{a}（{_ids(a_reasons)}）" if a_reasons else a
        summary.append(f"| {k} | {r4col} | {acol} | {vcol} | {series} |")
        pts = r4_holds(r4["grid"], k)[1] if r4["grid"] is not None else []
        why = "".join(f"\n- {r}" for r in reasons)
        detail.append(f"### K={k}\n\n格子点: {pts}\n\n判定理由:{why or ' なし'}\n\n{render_cells(s['cells'])}\n")
    out += summary + [""] + detail
    adopts = [k for k in KS_CANDIDATE if verdicts.get(k) == V_ADOPT]
    out.append("推奨候補（ADOPT 候補になった最小の K）: " + (str(adopts[0]) if adopts else "なし（または未実測）"))
    out.append("採否と定数切替は #2119 のユーザー承認事項。本集計は SME_PRODUCTION_ENABLED を切り替えない。")
    return "\n".join(out)


def render_gb10(base):
    out = ["\n# GB10 非後退再確認（候補ごと。語彙は RULE-gb10.txt を継承）\n"]
    rows = ["| K | R0 | RT | 5/5<=1.00 セル数（参考） | 総合判定 | 系列 |", "|---:|---|---|---:|---|---|"]
    detail = []
    for k in KS_CANDIDATE:
        s = load_gb10_series(base, k)
        if s is None:
            rows.append(f"| {k} | - | - | - | 未確定（実測未実施） | - |")
            continue
        v, reasons = series_verdict(s)
        if v == UNDETERMINED:
            v = f"{v}（{_ids(reasons)}）"
        series = {True: "正式（外側専有ゲート通過）", False: "参考（外側専有ゲート不通過）",
                  None: "参考（外側ゲートログなし）"}[s["outer_gate"]]
        n = sum(1 for c in s["cells"] if c.get("all_le_1"))
        r0 = {True: "成立", False: "不成立", None: "記録なし"}[s["r0"]]
        rows.append(f"| {k} | {r0} | {s['rt'] or '記録なし'} | {n} | {v} | {series} |")
        why = "".join(f"\n- {r}" for r in reasons)
        detail.append(f"### K={k}\n\n判定理由:{why or ' なし'}\n\n{render_cells(s['cells'])}\n")
    return "\n".join(out + rows + [""] + detail)


# --- self-test 用の模擬記録 ---

def _rec(task, size, mode, median, checksum=1.5, device="cpu"):
    return {"framework": "fandhe-ai", "version": "0.9.0", "task": task, "device": device, "size": size,
            "median_s": median, "q1_s": median, "q3_s": median, "checksum": checksum,
            "warmup": 5, "iters": 15, "mode": mode, "parity_fail_count": 0} if task == "gemm" else {
        "framework": "fandhe-ai", "version": "0.9.0", "task": task, "device": device, "size": size,
        "median_s": median, "q1_s": median, "q3_s": median, "checksum": checksum,
        "warmup": 5, "iters": 15, "mode": mode}


def _write_series(dirpath, label, ratios_by_cell, checksum_after=None):
    """ratios_by_cell[(task,size,mode)] = 5 round の after/before 比。before は常に 1.0。"""
    os.makedirs(dirpath, exist_ok=True)
    for task in ("gemm", "train", "infer"):
        for arm in ("before", "after"):
            lines = []
            for rnd in range(5):
                for (t, s, m) in CELLS:
                    if t != task:
                        continue
                    r = ratios_by_cell[(t, s, m)][rnd]
                    med = 1.0 if arm == "before" else r
                    cs = 1.5
                    if arm == "after" and checksum_after and (t, s, m) in checksum_after:
                        cs = checksum_after[(t, s, m)]
                    lines.append(json.dumps(_rec(t, s, m, med, checksum=cs)))
            Path(dirpath, f"results-{arm}-{label}-cpu-{task}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _uniform(gemm=0.7, train=0.9, infer=1.0):
    d = {}
    for (t, s, m) in CELLS:
        v = {"gemm": gemm, "train": train, "infer": infer}[t]
        d[(t, s, m)] = [v] * 5
    return d


def _write_tree_records(d, k):
    """sme2118_prepare_trees が成功時に残す 4 記録（P-TREE）を書く。"""
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    (d / "patch_sha256.txt").write_text(f"head={REGISTERED_BASE}\ncurrent_head=deadbeef\npatch_sha256=00\n")
    (d / "tree_verify.txt").write_text(f"trees_ok head={REGISTERED_BASE} K={k}\n")
    (d / "tree_diff.txt").write_text(MOD_REL + "\n")
    (d / "gate_constant.txt").write_text(
        "before: const SME_PRODUCTION_ENABLED: bool = false; / const SME_MIN_K: usize = 64;\n"
        f"after:  const SME_PRODUCTION_ENABLED: bool = true; / const SME_MIN_K: usize = {k};\n")


def _r4_text(sme=2.0):
    return "".join(f"variant={v}(X) size=({mn},{mn},{kk}) median_gflops={sme if v == 'SME' else 1.0}\n"
                   for v in ("NEON", "SME") for mn in _R4.MN for kk in _R4.KS)


def _write_m4_fixture(base, k, ratios, checksum_after=None):
    """前提が全て成立する M4 Max の記録一式（R4 共通 + 候補 K の R1/R2）を書く。"""
    m4 = Path(base) / "m4max"
    m4.mkdir(parents=True, exist_ok=True)
    for i in range(1, 6):
        (m4 / f"sme_r4_grid_run{i}.log").write_text(_r4_text())
    (m4 / "load_gate_r4.log").write_text(
        "".join(f"run{i} gate=pass load1=1\nrun{i} exit=0 grep_exit=0 end_load1=1\n" for i in range(1, 6))
        + "series done\n")
    (m4 / "r4_head.txt").write_text(f"head={REGISTERED_BASE}\ncurrent_head=deadbeef\n")
    label = f"2118-m4max-k{k}"
    d = m4 / "r1r2" / f"k{k}"
    _write_series(str(d), label, ratios, checksum_after)
    _write_tree_records(d, k)
    (d / "env_info.txt").write_text(f"label={label} K={k} run_ab_exit=0 collect_exit=0\nhostname=masked\n")
    (d / f"load-gate-1978-cpu-{label}.log").write_text(
        "threshold=8.0 rule_threshold=8.0 series=official\n"
        + "".join(f"round{i} gate=pass threshold=8.0 load1=1 waited_s=0\n" for i in range(1, 6)))


def _write_gb10_fixture(base, k, ratios, rt="pass", checksum_after=None):
    """前提が全て成立する GB10 の記録一式を書く。"""
    d = Path(base) / "gb10" / f"k{k}"
    _write_tree_records(d, k)
    (d / "sme_report.txt").write_text("before: sme_report=kernel_enabled: false\nafter: sme_report=kernel_enabled: false\n")
    (d / "rt_result.txt").write_text(
        f"RT rc=101\nrt_verdict={rt}\nrun_ab rc=0\ncollect rc=0\n")
    (d / "load_gate_outer.log").write_text("start attempt=1 load1=0.1 gpu_util=0 pass\n")
    _write_series(str(d / "r1r2"), f"2118-gb10-k{k}", ratios, checksum_after)


def _sub(path, old, new):
    p = Path(path)
    t = p.read_text(encoding="utf-8")
    if old not in t:
        raise AssertionError(f"模擬記録の書き換え対象が無い: {path}: {old!r}")
    p.write_text(t.replace(old, new, 1), encoding="utf-8")


def _series_dir(base, machine, k):
    return Path(base) / ("m4max/r1r2" if machine == "m4max" else "gb10") / f"k{k}"


def _jsonl(base, machine, k, arm, task):
    d = _series_dir(base, machine, k)
    if machine == "gb10":
        d = d / "r1r2"
    return d / f"results-{arm}-2118-{machine}-k{k}-cpu-{task}.jsonl"


def _exit_file(base, machine, k):
    return _series_dir(base, machine, k) / ("env_info.txt" if machine == "m4max" else "rt_result.txt")


def _drop_line(path, pred):
    p = Path(path)
    lines = p.read_text(encoding="utf-8").splitlines(True)
    keep = [ln for ln in lines if not pred(ln)]
    if len(keep) == len(lines):
        raise AssertionError(f"削除対象の行が無い: {path}")
    p.write_text("".join(keep), encoding="utf-8")


# 条項 → 破り方（ディスク上の記録を 1 条項分だけ書き換える）。infer セルはどの基準系列でも
# 判定の引き金にならないため、セル系の破り方は infer／train の JSONL に限る（引き金セルは残す）。
def _m_tree_diff2(b, m, k):
    Path(_series_dir(b, m, k), "tree_diff.txt").write_text(MOD_REL + "\nonly: crates/x.rs.orig\n")


def _m_tree_const(b, m, k):
    _sub(_series_dir(b, m, k) / "gate_constant.txt", f"SME_MIN_K: usize = {k};", "SME_MIN_K: usize = 32;")


def _m_tree_verify(b, m, k):
    (_series_dir(b, m, k) / "tree_verify.txt").unlink()


def _m_tree_base(b, m, k):
    _sub(_series_dir(b, m, k) / "patch_sha256.txt", f"head={REGISTERED_BASE}", "head=" + "1" * 40)


def _m_runab_nz(b, m, k):
    _sub(_exit_file(b, m, k), "run_ab_exit=0" if m == "m4max" else "run_ab rc=0",
         "run_ab_exit=2" if m == "m4max" else "run_ab rc=2")


def _m_runab_none(b, m, k):
    if m == "m4max":
        _sub(_exit_file(b, m, k), "run_ab_exit=0 ", "")
    else:
        _drop_line(_exit_file(b, m, k), lambda ln: ln.startswith("run_ab rc="))


def _m_collect_nz(b, m, k):
    _sub(_exit_file(b, m, k), "collect_exit=0" if m == "m4max" else "collect rc=0",
         "collect_exit=1" if m == "m4max" else "collect rc=1")


def _m_collect_none(b, m, k):
    if m == "m4max":
        _sub(_exit_file(b, m, k), " collect_exit=0", "")
    else:
        _drop_line(_exit_file(b, m, k), lambda ln: ln.startswith("collect rc="))


def _m_jsonl_missing(b, m, k):
    _jsonl(b, m, k, "after", "infer").unlink()


def _m_round_missing(b, m, k):
    f = _jsonl(b, m, k, "after", "infer")
    f.write_text("".join(f.read_text().splitlines(True)[:-1]))


def _m_dup_row(b, m, k):
    f = _jsonl(b, m, k, "before", "train")
    lines = f.read_text().splitlines(True)
    f.write_text("".join(lines) + lines[0])


def _m_checksum_missing(b, m, k):
    f = _jsonl(b, m, k, "after", "infer")
    lines = f.read_text().splitlines(True)
    rec = json.loads(lines[0])
    rec["checksum"] = None
    f.write_text(json.dumps(rec) + "\n" + "".join(lines[1:]))


def _m_r4_base(b, m, k):
    _sub(Path(b) / "m4max" / "r4_head.txt", f"head={REGISTERED_BASE}", "head=" + "2" * 40)


def _m_r4_nz(b, m, k):
    _sub(Path(b) / "m4max" / "load_gate_r4.log", "run5 exit=0", "run5 exit=101")


def _m_r4_no_done(b, m, k):
    _drop_line(Path(b) / "m4max" / "load_gate_r4.log", lambda ln: ln.startswith("series done"))


def _m_r4_no_record(b, m, k):
    (Path(b) / "m4max" / "load_gate_r4.log").unlink()


def _m_r4_dup(b, m, k):
    f = Path(b) / "m4max" / "sme_r4_grid_run3.log"
    f.write_text(f.read_text() + f.read_text().splitlines(True)[0])


def _m_r4_run_missing(b, m, k):
    (Path(b) / "m4max" / "sme_r4_grid_run4.log").unlink()


def _m_r0_fail(b, m, k):
    _sub(_series_dir(b, m, k) / "sme_report.txt", "after: sme_report=kernel_enabled: false",
         "after: sme_report=kernel_enabled: true")


def _m_r0_none(b, m, k):
    (_series_dir(b, m, k) / "sme_report.txt").unlink()


def _m_rt_none(b, m, k):
    _drop_line(_series_dir(b, m, k) / "rt_result.txt", lambda ln: ln.startswith("rt_verdict="))


BOTH = ("m4max", "gb10")
# 条項 ID → (破る前提 ID, 対象機体, 破り方)。self-test が RULE_CLAUSE_IDS・PRECONDITIONS との網羅を照合する。
CLAUSE_CASES = {
    "§1/P-TREE:tree_diff が 2 件": ("P-TREE", BOTH, _m_tree_diff2),
    "§1/P-TREE:gate_constant の K 不一致": ("P-TREE", BOTH, _m_tree_const),
    "§1/P-TREE:tree_verify 記録なし": ("P-TREE", BOTH, _m_tree_verify),
    "§13/P-TREE:登録 sha 以外を基準にした": ("P-TREE", BOTH, _m_tree_base),
    "§13/P-RUNAB:run_ab 非ゼロ": ("P-RUNAB", BOTH, _m_runab_nz),
    "§13/P-RUNAB:run_ab 記録なし": ("P-RUNAB", BOTH, _m_runab_none),
    "§14/P-COLLECT:collect 非ゼロ": ("P-COLLECT", BOTH, _m_collect_nz),
    "§14/P-COLLECT:collect 記録なし": ("P-COLLECT", BOTH, _m_collect_none),
    "§14/P-CELLS:JSONL 欠損": ("P-CELLS", BOTH, _m_jsonl_missing),
    "§14/P-CELLS:round 欠損": ("P-CELLS", BOTH, _m_round_missing),
    "§14/P-CELLS:重複行": ("P-CELLS", BOTH, _m_dup_row),
    "§14/P-CELLS:checksum 欠損": ("P-CELLS", BOTH, _m_checksum_missing),
    "§13/P-R4-BASE:r4_head が登録 sha でない": ("P-R4-BASE", ("m4max",), _m_r4_base),
    "§13/P-R4-EXEC:R4 run 非ゼロ": ("P-R4-EXEC", ("m4max",), _m_r4_nz),
    "§13/P-R4-EXEC:series done なし": ("P-R4-EXEC", ("m4max",), _m_r4_no_done),
    "§13/P-R4-EXEC:R4 実行記録なし": ("P-R4-EXEC", ("m4max",), _m_r4_no_record),
    "§14/P-R4-LOG:R4 重複行": ("P-R4-LOG", ("m4max",), _m_r4_dup),
    "§14/P-R4-LOG:R4 run ログ欠損": ("P-R4-LOG", ("m4max",), _m_r4_run_missing),
    "§8/P-R0:R0 不成立": ("P-R0", ("gb10",), _m_r0_fail),
    "§8/P-R0:R0 記録なし": ("P-R0", ("gb10",), _m_r0_none),
    "§9/P-RT:rt_verdict 記録なし": ("P-RT", ("gb10",), _m_rt_none),
}


def _baselines():
    """前提が全て成立し、判定が FAIL／REJECT／ADOPT（GB10 は 後退あり／後退あり相当／pass）になる基準系列。
    (名前, 機体, K, 書き込み関数, 期待する総合判定, 期待する AC1 欄)"""
    rej = _uniform()
    rej[("gemm", 512, "fresh")] = [1.03] * 5
    g_reg = _uniform(gemm=1.0, train=1.0, infer=1.0)
    g_reg[("gemm", 1024, "reuse")] = [1.02, 1.03, 1.01, 1.05, 1.02]
    g_ok = _uniform(gemm=1.0, train=1.0, infer=1.0)
    fail_cs = {("gemm", 1024, "reuse"): 1.5000001}
    return [
        ("m4-REJECT", "m4max", 256, lambda b: _write_m4_fixture(b, 256, rej), V_REJECT, AC1_NG),
        ("m4-FAIL", "m4max", 128, lambda b: _write_m4_fixture(b, 128, _uniform(), fail_cs), V_FAIL, AC1_NG),
        ("m4-ADOPT", "m4max", 128, lambda b: _write_m4_fixture(b, 128, _uniform()), V_ADOPT, AC1_OK),
        ("gb10-後退あり", "gb10", 64, lambda b: _write_gb10_fixture(b, 64, g_reg), G_REGRESSION, None),
        ("gb10-後退あり相当", "gb10", 128, lambda b: _write_gb10_fixture(b, 128, g_ok, rt="regression-suspect"),
         G_SUSPECT, None),
        ("gb10-pass", "gb10", 256, lambda b: _write_gb10_fixture(b, 256, g_ok), G_PASS, None),
    ]


def _load_fixture_series(base, machine, k):
    return load_m4_series(base, k, load_r4(base)) if machine == "m4max" else load_gb10_series(base, k)


def _mem_series(cells, k=128, machine="m4max", **over):
    """前提が全て成立するメモリ上の系列（単体テスト用）。over で個別の記録を上書きする。"""
    s = {"machine": machine, "k": k, "cells": cells, "tree": True, "run_ab": True, "collect": True,
         "official": True, "r4": {"base": True, "exec": True, "valid_runs": 5,
                                  "grid": {(mn, kk): True for mn in _R4.MN for kk in _R4.KS}},
         "r0": True, "rt": "pass"}
    s.update(over)
    return s


def _check_single_gate():
    """判定語彙を返す関数が series_verdict／ac1_verdict（と RT 部分判定 rt_verdict）に限られ、前者 2 つが
    最初の文で preconditions を呼ぶことを AST で照合する（判定関数を直接呼べる経路が残っていないことの確認）。"""
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    names = {"V_FAIL", "V_REJECT", "V_ADOPT", "REFERENCE_ADOPT", "REFERENCE_REJECT", "G_REGRESSION",
             "G_SUSPECT", "G_PASS", "G_REFERENCE_SUFFIX", "AC1_OK", "AC1_NG"}
    literals = {V_FAIL, V_REJECT, V_ADOPT, REFERENCE_ADOPT, REFERENCE_REJECT, G_REGRESSION, G_SUSPECT}
    gated = {"series_verdict", "ac1_verdict"}
    for fn in [n for n in tree.body if isinstance(n, ast.FunctionDef)]:
        compare_ids = {id(x) for c in ast.walk(fn) if isinstance(c, ast.Compare) for x in ast.walk(c)}
        uses = [x for x in ast.walk(fn) if id(x) not in compare_ids and (
            (isinstance(x, ast.Name) and x.id in names)
            or (isinstance(x, ast.Constant) and isinstance(x.value, str) and x.value in literals))]
        if fn.name in gated:
            body = [s for s in fn.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
            first = body[0]
            assert (isinstance(first, ast.Assign) and isinstance(first.value, ast.Call)
                    and getattr(first.value.func, "id", None) == "preconditions"), fn.name
        elif fn.name == "_check_single_gate":
            continue
        else:
            assert not uses or fn.name == "_baselines", f"{fn.name} が判定語彙を前提ゲートの外で扱う"
    assert "candidate_verdict" not in globals() and "gb10_verdict" not in globals()


def _check_rule_drift():
    """RULE.txt の節ごとの判定不能系の行数が RULE_KEYWORD_LINES と一致すること（条項追加の検出）と、
    条項 ID・前提 ID・ケースの網羅。"""
    counts, sec = {}, None
    for line in (_HERE / "RULE.txt").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^## (\d+)\.", line)
        if m:
            sec = m.group(1)
            continue
        if sec and RULE_KEYWORD_RE.search(line):
            counts[sec] = counts.get(sec, 0) + 1
    assert counts == RULE_KEYWORD_LINES, f"RULE.txt の判定不能系の行が変わった（条項表を見直す）: {counts}"
    assert set(CLAUSE_CASES) == set(RULE_CLAUSE_IDS) and len(RULE_CLAUSE_IDS) == len(set(RULE_CLAUSE_IDS))
    pids = {p[0] for p in PRECONDITIONS}
    for cid, (pid, machines, _) in CLAUSE_CASES.items():
        assert pid in pids and f"/{pid}:" in cid, cid
        assert set(machines) <= set(_PRECOND_BY_ID[pid][1]), cid
    for pid, machines, *_ in PRECONDITIONS:
        for mach in machines:
            assert any(p == pid and mach in ms for p, ms, _ in CLAUSE_CASES.values()), (pid, mach)
    lib = (_HERE / "lib_trees.sh").read_text(encoding="utf-8")
    assert f'SME2118_REGISTERED_BASE="{REGISTERED_BASE}"' in lib
    assert f'SME2118_MOD_REL="{MOD_REL}"' in lib


def _run_clause_table(td):
    """条項 → ケースの表駆動検査。各基準系列（FAIL／REJECT／ADOPT・後退あり／後退あり相当／pass）で、
    1 条項だけを破ると総合判定が undetermined になり、理由の前提 ID がその 1 つだけであることを確かめる。"""
    for bname, machine, k, write, expect, expect_ac1 in _baselines():
        base = Path(td) / "tbl" / bname / "base"
        write(base)
        s = _load_fixture_series(str(base), machine, k)
        assert preconditions(s) == [], (bname, preconditions(s))
        assert series_verdict(s) == (expect, []), (bname, series_verdict(s))
        if machine == "m4max":
            assert ac1_verdict(s) == (expect_ac1, []), (bname, ac1_verdict(s))
        for cid, (pid, machines, mutate) in CLAUSE_CASES.items():
            if machine not in machines:
                continue
            case = Path(td) / "tbl" / bname / f"case{RULE_CLAUSE_IDS.index(cid)}"
            shutil.copytree(base, case)
            mutate(str(case), machine, k)
            sm = _load_fixture_series(str(case), machine, k)
            v, reasons = series_verdict(sm)
            assert v == UNDETERMINED, f"[{cid}] {bname}: {v}"
            assert [r.split(":", 1)[0] for r in reasons] == [pid], f"[{cid}] {bname}: {reasons}"
            if machine == "m4max":
                a, a_reasons = ac1_verdict(sm)
                if _PRECOND_BY_ID[pid][2]:
                    assert a == AC1_UNDETERMINED and _ids(a_reasons) == pid, f"[{cid}] AC1 {bname}: {a}"
                else:
                    assert (a, a_reasons) == (expect_ac1, []), f"[{cid}] AC1 は R4 と独立 {bname}: {a}"
            shutil.rmtree(case)


def self_test():
    _check_rule_drift()
    _check_single_gate()
    # (f) K による到達分類: train は K=64 のみ到達・infer は全候補で非到達・gemm は全候補で到達
    assert [reached(k, "train") for k in KS_CANDIDATE] == [True, False, False]
    assert not any(reached(k, "infer") for k in KS_CANDIDATE)
    assert all(reached(k, "gemm") for k in KS_CANDIDATE)
    # classify: (b) 中央値 <=1 だが 1 round >1（train reuse の事例）→ adopt でない
    c = classify_cell(True, [1.0244, 0.9, 0.95, 0.98, 1.0248])
    assert c["median"] <= 1.0 and not c["adopt"] and not c["reject"]
    assert classify_cell(True, [0.9] * 5)["adopt"]
    # (c) 非到達セル 5/5 >1 → reject 材料・(d) 到達セル 5/5 >1 → reject
    assert classify_cell(False, [1.02] * 5)["reject"]
    assert classify_cell(True, [1.02] * 5)["reject"] and not classify_cell(True, [1.02] * 5)["adopt"]
    assert not classify_cell(False, [1.02, 1.02, 0.99, 1.02, 1.02])["reject"]
    # (g) R4 の候補別格子: (256,64) の 1 run が <1.0 なら K=64 は不成立・K=128 は成立
    ok = {(mn, kk): (mn >= 256 and kk >= 64) for mn in _R4.MN for kk in _R4.KS}
    ok[(256, 64)] = False
    assert not r4_holds(ok, 64)[0]
    assert r4_holds(ok, 128)[0]
    assert r4_holds({(mn, kk): True for mn in _R4.MN for kk in _R4.KS}, 64)[0]

    with tempfile.TemporaryDirectory() as td:
        # 条項 → ケースの表（1 条項 1 ケース × 基準系列）
        _run_clause_table(td)

        # (a) 全条件を満たす候補（K=128。train は非到達で 1.0 超のノイズがあっても reject 材料にならない）
        rb = _uniform()
        rb[("train", 64, "reuse")] = [1.02, 1.01, 0.99, 1.02, 1.03]  # 非到達ではノイズ
        _write_series(os.path.join(td, "a"), "L", rb)
        cells = apply_k(evaluate_series(os.path.join(td, "a"), "L"), 128)
        assert series_verdict(_mem_series(cells)) == (V_ADOPT, [])
        # §11: 負荷ゲート不通過（record_only）系列は ADOPT 条件を満たしても "ADOPT 候補" にしない
        assert series_verdict(_mem_series(cells, official=False))[0] == REFERENCE_ADOPT
        assert ac1_verdict(_mem_series(cells))[0] == AC1_OK
        # 同じデータを K=64 で見ると train reuse（到達）が ADOPT 候補条件不成立 → undetermined（(b)）
        cells64 = apply_k(evaluate_series(os.path.join(td, "a"), "L"), 64)
        v, why = series_verdict(_mem_series(cells64, k=64))
        assert v == UNDETERMINED and why and why[0].startswith("§6"), (v, why)
        assert ac1_verdict(_mem_series(cells64, k=64))[0] == AC1_NG
        # R4 不成立（前提は成立・格子が不成立）なら ADOPT にならない
        bad_grid = {(mn, kk): False for mn in _R4.MN for kk in _R4.KS}
        r4_bad = {"base": True, "exec": True, "valid_runs": 5, "grid": bad_grid}
        assert series_verdict(_mem_series(cells, r4=r4_bad))[0] == UNDETERMINED
        # §11: 不通過系列の 5/5 後退は正式 REJECT ではなく参考扱い
        rc = [dict(x, reject=True, adopt=False) if x["reached"] else dict(x) for x in cells]
        assert series_verdict(_mem_series(rc))[0] == V_REJECT
        assert series_verdict(_mem_series(rc, official=False))[0] == REFERENCE_REJECT
        # §15: R4 の実行記録なし・異常終了では、到達セルの 5/5 後退があっても REJECT を出さない（PRRT_kwDOTuUCJc6nj1Wj）
        for r4x in ({"exec": None}, {"exec": False}, {"valid_runs": 4, "grid": None}, {"base": False}):
            r4m = {"base": True, "exec": True, "valid_runs": 5, "grid": {p: True for p in bad_grid}}
            r4m.update(r4x)
            assert series_verdict(_mem_series(rc, r4=r4m))[0] == UNDETERMINED, r4x
            assert ac1_verdict(_mem_series(rc, r4=r4m))[0] == AC1_NG  # AC1 欄は R4 と独立
        # (c) 非到達セル（infer）が 5/5 >1.00 → 総合は ADOPT にならない
        rcx = _uniform()
        rcx[("infer", 64, "fresh")] = [1.01] * 5
        _write_series(os.path.join(td, "c"), "L", rcx)
        cellsc = apply_k(evaluate_series(os.path.join(td, "c"), "L"), 128)
        assert series_verdict(_mem_series(cellsc))[0] == UNDETERMINED
        assert any(x["reject"] and not x["reached"] for x in cellsc)
        # (h) GB10 語彙: RT の既知 FAIL は許容し pass・それ以外の FAIL は後退あり相当
        assert rt_verdict([]) == "pass"
        assert rt_verdict([KNOWN_RT_FAIL]) == "pass"
        assert rt_verdict([KNOWN_RT_FAIL, "gemm_blis::tests::other"]) == "regression-suspect"
        g = _uniform(gemm=1.0, train=1.0, infer=1.0)
        g[("gemm", 512, "fresh")] = [2.007, 0.99, 1.0, 0.98, 1.0]  # before 外れ値相当。5/5<=1 不成立でも後退ではない
        _write_series(os.path.join(td, "g"), "L", g)
        cellsg = apply_k(evaluate_series(os.path.join(td, "g"), "L"), None)
        assert series_verdict(_mem_series(cellsg, machine="gb10"))[0] == G_PASS
        assert series_verdict(_mem_series(cellsg, machine="gb10", rt="regression-suspect"))[0] == G_SUSPECT
        assert series_verdict(_mem_series(cellsg, machine="gb10", official=False))[0] == G_PASS + G_REFERENCE_SUFFIX
        # 構造の破綻（重複・欠落・未知セル）は例外（fail-closed。§14）。総合判定・AC1 欄のいずれでも
        bad_sets = {
            "dup": cells + [dict(cells[0])],
            "missing": cells[:-1],
            "unknown": [dict(cells[0], size=4096)] + cells[1:],
            "dup_replace": [dict(cells[0])] + [dict(cells[0])] + cells[2:],
        }
        for name, bad in bad_sets.items():
            for fn in (lambda c: series_verdict(_mem_series(c)),
                       lambda c: series_verdict(_mem_series(c, machine="gb10")),
                       lambda c: ac1_verdict(_mem_series(c))):
                try:
                    fn(bad)
                except ValueError:
                    pass
                else:
                    raise AssertionError(f"構造破綻 {name} が例外にならない")
        # 記録パーサ: 終了コード（0=成立・非ゼロ=不成立・記録なし=None・重複記録=不成立）
        p = os.path.join(td, "rec.txt")
        Path(p).write_text("label=x run_ab_exit=0 collect_exit=0\n")
        assert exit_records_m4max(p) == (True, True)
        Path(p).write_text("label=x run_ab_exit=1 collect_exit=0\n")
        assert exit_records_m4max(p) == (False, True)
        Path(p).write_text("label=x run_ab_exit=0\n")
        assert exit_records_m4max(p) == (True, None)
        Path(p).write_text("label=x run_ab_exit=0 run_ab_exit=0 collect_exit=0\n")
        assert exit_records_m4max(p) == (False, True)
        assert exit_records_m4max(os.path.join(td, "nonexistent")) == (None, None)
        Path(p).write_text("RT rc=101\nrun_ab rc=0\ncollect rc=0\n")
        assert exit_records_gb10(p) == (True, True)
        Path(p).write_text("RT rc=101\nrun_ab rc=2\n")
        assert exit_records_gb10(p) == (False, None)
        # R4 実行記録・ログの厳密解析
        _r4 = os.path.join(td, "r4gate.log")
        _good = "".join(f"run{i} gate=pass\nrun{i} exit=0 grep_exit=0 end_load1=1\n" for i in range(1, 6))
        Path(_r4).write_text(_good + "series done\n")
        assert r4_runs_ok(_r4) is True
        Path(_r4).write_text(_good.replace("run5 exit=0", "run5 exit=1") + "series done\n")
        assert r4_runs_ok(_r4) is False
        assert r4_runs_ok(os.path.join(td, "none.log")) is None
        r4txt = _r4_text(1.0)
        assert parse_r4_strict(r4txt) is not None and len(parse_r4_strict(r4txt)) == 32
        first = r4txt.splitlines(True)[0]
        assert parse_r4_strict(r4txt + first) is None  # 重複（後勝ちで 32 件に見える）
        assert parse_r4_strict("".join(r4txt.splitlines(True)[1:])) is None  # 欠落
        assert parse_r4_strict("".join(r4txt.splitlines(True)[1:])
                               + "variant=SME(X) size=(999,999,32) median_gflops=1.0\n") is None  # 未知
        # load ゲートログ・外側専有ゲートの解釈（§8・§11 の修飾。前提ではない）
        Path(p).write_text("threshold=8.0 rule_threshold=8.0 series=official\n" +
                           "".join(f"round{i} gate=pass threshold=8.0 load1=1 waited_s=0\n" for i in range(1, 6)))
        assert load_gate_series(p) is True
        Path(p).write_text("threshold=8.0 rule_threshold=8.0 series=official\nround1 gate=timeout\n")
        assert load_gate_series(p) is False
        assert load_gate_series(os.path.join(td, "none.log")) is None
        Path(p).write_text("start attempt=1 load1=0.1 gpu_util=0 wait\nstart attempt=2 load1=0.1 gpu_util=0 pass\n")
        assert outer_gate_official(p) is True
        Path(p).write_text("start attempt=20 load1=3 gpu_util=NA fail(reference)\n")
        assert outer_gate_official(p) is False
        Path(p).write_text("run1 gate=pass\n" * 4 + "run5 gate=timeout\n")
        assert _gate_log_official(p, "run") is False

        # 描画: 未実測ディレクトリは「未実測」
        e = os.path.join(td, "empty")
        os.makedirs(e)
        txt = render_m4max(e) + render_gb10(e)
        assert "未実測" in txt and "未確定（実測未実施）" in txt
        # 描画: R4 のログが揃わない系列は、R1 の 5/5 後退があっても REJECT を出さず undetermined（§15）
        rj = os.path.join(td, "render")
        rej = _uniform()
        rej[("gemm", 512, "fresh")] = [1.03] * 5
        _write_m4_fixture(rj, 256, rej)
        assert "| 256 | 成立 | 不成立 | REJECT |" in render_m4max(rj), render_m4max(rj)
        _m_r4_no_done(rj, "m4max", 256)
        t2 = render_m4max(rj)
        assert "| 256 | 判定不能 | 不成立 | undetermined（P-R4-EXEC） |" in t2, t2
        # 描画: GB10 の R0 不成立で中止（sme_report.txt とツリー記録のみ・r1r2 なし）は undetermined と理由を表示
        gdir = Path(td) / "render" / "gb10" / "k128"
        _write_tree_records(gdir, 128)
        (gdir / "sme_report.txt").write_text("before: sme_report=kernel_enabled: true\nafter: sme_report=kernel_enabled: true\n")
        t3 = render_gb10(rj)
        assert "| 128 | 不成立 | 記録なし | 0 | undetermined（P-RUNAB・P-COLLECT・P-CELLS・P-R0・P-RT） |" in t3, t3
    print("self-test ok")


def main(argv):
    if "--self-test" in argv:
        self_test()
        return 0
    args = [a for a in argv[1:] if not a.startswith("--")]
    base = args[0] if args else str(_HERE)
    print(render_m4max(base))
    print(render_gb10(base))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
