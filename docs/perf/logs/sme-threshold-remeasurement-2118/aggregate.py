#!/usr/bin/env python3
"""イシュー #2118: SME_MIN_K 候補（64／128／256）の M4 Max 再実測 + GB10 非後退の集計。

役割: 同ディレクトリの RULE.txt（事前登録）の判定規則を機械的に適用し
`aggregate.md` 用の Markdown を標準出力へ出す。規則を新たに決める場所ではない
（規則の正は RULE.txt。ここに書く定数・語彙はそれと同一で、変更は追記方式の
訂正のみ）。python3 標準ライブラリのみ。

再利用（フォークしない）:
  - `../cpu-gemm-sme-fmopa-1587/aggregate.py` の `parse`／`candidates`（R4 の
    16 格子点のログ解析とパレート極小）。#1978 の M4 Max 実測と同じ解析にする。
  - `scripts/bench/framework-compare/compare_gemm_ab.py` の `load_rows`／
    `group_by_cell`／`evaluate_cell`／`per_run_ratios`（R1／R2 の round 比・
    中央値・checksum 完全一致）。`run_ab_sme_cpu.sh` が呼ぶ判定と同じ関数。

入力ディレクトリ配置（orchestrate_m4max.sh・gb10/orchestrate_gb10.sh の出力）:
  <base>/m4max/sme_r4_grid_run{1..5}.log
  <base>/m4max/r1r2/k{K}/results-{before,after}-2118-m4max-k{K}-cpu-{gemm,train,infer}.jsonl
  <base>/gb10/k{K}/r1r2/results-{before,after}-2118-gb10-k{K}-cpu-*.jsonl
  <base>/gb10/k{K}/{sme_report.txt,rt_result.txt}
未実測の系列は「未実測」と出力する（値の捏造・推定はしない）。

使い方: aggregate.py [--self-test] [<base_dir>] > aggregate.md
"""
import importlib.util
import json
import os
import re
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

# RULE.txt §2 の到達セル表（出典: bench-fandhe/src/main.rs BATCH=64・D_IN=784・D_HIDDEN=256・
# D_OUT=10。L1 d_weight TN 784x256x64 が K=64 のときだけ SME に到達する）。
CELLS = (
    [("gemm", s, m) for s in (512, 1024, 2048) for m in ("fresh", "reuse")]
    + [("train", 64, m) for m in ("fresh", "reuse")]
    + [("infer", 64, m) for m in ("fresh", "reuse")]
)


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


REFERENCE_ADOPT = "ADOPT 候補相当（参考・record_only）"
REFERENCE_REJECT = "REJECT（参考・record_only）"


def candidate_verdict(r4_ok, cells, official=True, run_ok=True):
    """RULE.txt §6・§11。cells は {"reached", "adopt", "reject", "status", "exact", "reason"} の dict 列。

    FAIL（R2: checksum 不一致。停止扱いで ADOPT にも REJECT にも数えない）> REJECT > ADOPT 候補 > undetermined。
    official は M4 Max の負荷ゲート（R4 格子 5/5 通過 かつ R1 系列 5/5 通過）の結果。§11 により
    通過しない系列は record_only であり、ADOPT 条件を満たしても "ADOPT 候補" は返さず参考判定
    （REFERENCE_ADOPT）に落とす（推奨候補の選定対象外にする）。REJECT も同様で、不通過系列の
    5/5 後退は正式な REJECT ではなく参考（REFERENCE_REJECT）として返し、系列の正式性を判定欄に反映する。
    run_ok は run_ab_sme_cpu.sh の終了コード 0 の確認結果。非ゼロ・記録なしは計測失敗であり、
    部分的な JSONL から ADOPT／REJECT を出さず undetermined（判定不能）に倒す（fail-closed）。
    """
    if not run_ok:
        return "undetermined"
    for c in cells:
        if c["status"] == "ok" and not c["exact"]:
            return "FAIL"
        if c["status"] != "ok" and "checksum" in (c.get("reason") or ""):
            return "FAIL"
    if any(c["reached"] and c["reject"] for c in cells if c["status"] == "ok"):
        return "REJECT" if official else REFERENCE_REJECT
    if (
        r4_ok
        and len(cells) == len(CELLS)
        and all(c["status"] == "ok" for c in cells)
        and all(c["adopt"] for c in cells if c["reached"])
        and not any(c["reject"] for c in cells if not c["reached"])
    ):
        return "ADOPT 候補" if official else REFERENCE_ADOPT
    return "undetermined"


def ac1_ok(cells):
    """AC1 の直接判定欄（RULE.txt §6）。到達セルが 5 round 中央値<=1.00 かつ 5/5<=1.00、かつ全セル checksum 完全一致。"""
    if len(cells) != len(CELLS) or any(c["status"] != "ok" for c in cells):
        return False
    return all(c["adopt"] for c in cells if c["reached"]) and all(c["exact"] for c in cells)


def rt_verdict(fail_names):
    """RULE.txt §9。既知 FAIL 1 件のみ（または FAIL なし）は pass。それ以外が 1 件でもあれば要調査。"""
    others = [n for n in fail_names if n != KNOWN_RT_FAIL]
    return "pass" if not others else "regression-suspect"


def gb10_verdict(r0_ok, rt, cells, run_ok=True):
    """RULE.txt §8〜§9（語彙は 1587/gb10/RULE-gb10.txt を継承）。

    5/5 一貫の後退セルまたは checksum 不一致 = 後退あり。RT の既知 FAIL 以外 = 後退あり相当（要調査）。
    R0 不成立・計測失敗（run_ab_sme_cpu.sh の終了コード非ゼロ／記録なしを含む）・round 欠損 = undetermined。
    それ以外 = pass。「全セル 5/5<=1.00」は採らない。
    """
    if not run_ok:
        return "undetermined"
    if any(c["status"] == "ok" and (c["reject"] or not c["exact"]) for c in cells):
        return "後退あり"
    if any(c["status"] != "ok" and "checksum" in (c.get("reason") or "") for c in cells):
        return "後退あり"
    if rt == "regression-suspect":
        return "後退あり相当（要調査）"
    if not r0_ok or len(cells) != len(CELLS) or any(c["status"] != "ok" for c in cells):
        return "undetermined"
    return "pass"


def evaluate_series(dirpath, label):
    """dirpath の JSONL（before/after × gemm/train/infer）を CELLS の順に判定した dict 列と補足を返す。"""
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
        per_task[task] = (_CMP.group_by_cell(b), _CMP.group_by_cell(a))
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
    """候補 K に応じて到達分類と classify_cell を適用した dict 列を返す（元は変更しない）。"""
    out = []
    for c in cells:
        d = dict(c)
        d["reached"] = reached(k, c["task"])
        if c["status"] == "ok" and c["ratios"] is not None:
            d.update(classify_cell(d["reached"], c["ratios"]))
            d["median"] = c["median"]
        out.append(d)
    return out


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
    かつ `series done` 行がある場合のみ True。ログ不在は None、異常終了・欠損・途中終了は False。
    32 点をログから解析できても、非ゼロ終了した系列は R4 成立として扱わない（fail-closed）。"""
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


def _fmt_ratios(c):
    if not c.get("ratios"):
        return "-"
    return ", ".join(f"{x:.4f}" for x in c["ratios"])


def render_cells(cells):
    lines = ["| セル | 分類 | 5 round の比 | 中央値 | checksum | 判定 |", "|---|---|---|---:|---|---|"]
    for c in cells:
        name = f"{c['task']} {c['size']}/{c['mode']}"
        cls = "到達" if c["reached"] else "非到達"
        if c["status"] != "ok":
            lines.append(f"| {name} | {cls} | - | - | - | 判定不能: {c['reason']} |")
            continue
        if c["reached"]:
            j = "ADOPT 候補" if c["adopt"] else ("REJECT（5/5 後退）" if c["reject"] else "ADOPT 候補条件不成立")
        else:
            j = "REJECT 材料（5/5 後退）" if c["reject"] else "後退の証拠なし"
        lines.append(f"| {name} | {cls} | {_fmt_ratios(c)} | {c['median']:.4f} | "
                     f"{'完全一致' if c['exact'] else '不一致'} | {j} |")
    return "\n".join(lines)


def render_m4max(base):
    out = ["# SME_MIN_K 候補の M4 Max 再実測（イシュー #2118）\n"]
    m4 = Path(base) / "m4max"
    ok = None
    r4_gate = None
    r4_run = None
    runs = []
    for i in range(1, 6):
        f = m4 / f"sme_r4_grid_run{i}.log"
        if f.exists():
            r = _R4.parse(f.read_text(encoding="utf-8"))
            if len(r) == 32:
                runs.append(r)
    if len(runs) == 5:
        ok = {}
        out += ["## R4 格子（ratio = SME/NEON。5/5 run で >=1.0 が SME>=NEON）\n",
                "| min(m,n) | k | 5 run の比 | 中央値 | 5/5 SME>=NEON |", "|---:|---:|---|---:|---|"]
        for mn in _R4.MN:
            for kk in _R4.KS:
                rs = [r[("SME", mn, kk)] / r[("NEON", mn, kk)] for r in runs]
                ok[(mn, kk)] = all(x >= 1.0 for x in rs)
                out.append(f"| {mn} | {kk} | {', '.join(f'{x:.3f}' for x in rs)} | "
                           f"{statistics.median(rs):.3f} | {'yes' if ok[(mn, kk)] else 'no'} |")
        cand = _R4.candidates(ok)
        out.append("\n参考（パレート極小の採用候補）: " + (
            "、".join(f"`min(m,n) >= {a}` かつ `k >= {b}`" for a, b in cand) if cand else "なし"))
        r4_gate = _gate_log_official(str(m4 / "load_gate_r4.log"), "run")
        r4_run = r4_runs_ok(str(m4 / "load_gate_r4.log"))
        if r4_run is not True:
            out.append("\n**R4 計測の実行成否: 5 run 全ての exit=0 と `series done` を確認できない"
                       "（異常終了・記録なし）。RULE.txt §13 により R4 は判定不能（下表の格子は参考表示のみ）。**\n")
        out.append("R4 負荷ゲート: " + {True: "5/5 通過（正式）", False: "不通過を含む（record_only）",
                                       None: "ログなし（record_only 扱い）"}[r4_gate])
    else:
        out.append("## R4 格子\n\n未実測（`sme_r4_grid_run{1..5}.log` が揃っていない）。")
    out.append("\n## 候補別判定\n")
    summary = ["| K | R4 | R1 到達セル AC1 欄 | 総合判定 | 系列 |", "|---:|---|---|---|---|"]
    detail = []
    verdicts = {}
    for k in KS_CANDIDATE:
        label = f"2118-m4max-k{k}"
        d = m4 / "r1r2" / f"k{k}"
        if not d.exists():
            summary.append(f"| {k} | {'-' if ok is None else '判定可'} | 未実測 | 未確定（実測未実施） | - |")
            continue
        cells = apply_k(evaluate_series(str(d), label), k)
        r4c, pts = (False, []) if ok is None else r4_holds(ok, k)
        r4_exec_ok = r4_run is True
        if not r4_exec_ok:
            r4c = False  # RULE.txt §13: R4 実行成否が確認できない系列は R4 成立にしない
        gate = load_gate_series(str(d / f"load-gate-1978-cpu-{label}.log"))
        # RULE.txt §11: R4・R1 の両負荷ゲートを通過した系列だけを正式とする
        official = bool(r4_gate) and gate is True
        run_ok = run_ab_ok_m4max(str(d / "env_info.txt")) is True
        v = candidate_verdict(r4c, cells, official, run_ok and r4_exec_ok)
        verdicts[k] = v
        series = {True: "正式（R1 load ゲート 5/5 通過）", False: "record_only（参考）", None: "ゲートログなし（record_only 扱い）"}[gate]
        if gate is True and not r4_gate:
            series = "record_only（R4 負荷ゲート不通過・未確認）"
        if not r4_exec_ok:
            series += "／R4 が異常終了または終了記録なし（判定不能）"
        if not run_ok:
            series += "／run_ab_sme_cpu.sh 非ゼロ終了または終了コード記録なし（判定不能）"
        summary.append(f"| {k} | {'成立' if r4c else '不成立/未計測'} | {'成立' if ac1_ok(cells) else '不成立'} | {v} | {series} |")
        detail.append(f"### K={k}\n\n格子点: {pts}\n\n{render_cells(cells)}\n")
    out += summary + [""] + detail
    adopts = [k for k in KS_CANDIDATE if verdicts.get(k) == "ADOPT 候補"]
    out.append("推奨候補（ADOPT 候補になった最小の K）: " + (str(adopts[0]) if adopts else "なし（または未実測）"))
    out.append("採否と定数切替は #2119 のユーザー承認事項。本集計は SME_PRODUCTION_ENABLED を切り替えない。")
    return "\n".join(out)


def run_ab_ok_m4max(env_info_path):
    """M4 Max: env_info.txt の `run_ab_exit=N` が 0 か。ファイル・記録なしは None（判定不能扱い）。"""
    if not os.path.exists(env_info_path):
        return None
    m = re.search(r"run_ab_exit=(\d+)", Path(env_info_path).read_text(encoding="utf-8"))
    return None if m is None else int(m.group(1)) == 0


def run_ab_ok_gb10(rt_result_path):
    """GB10: rt_result.txt の `run_ab rc=N` が 0 か。ファイル・記録なしは None（判定不能扱い）。"""
    if not os.path.exists(rt_result_path):
        return None
    m = re.search(r"^run_ab rc=(\d+)", Path(rt_result_path).read_text(encoding="utf-8"), re.M)
    return None if m is None else int(m.group(1)) == 0


def _parse_rt(path):
    if not os.path.exists(path):
        return None
    text = Path(path).read_text(encoding="utf-8")
    m = re.search(r"rt_verdict=(\S+)", text)
    return m.group(1) if m else None


def render_gb10(base):
    out = ["\n# GB10 非後退再確認（候補ごと。語彙は RULE-gb10.txt を継承）\n"]
    rows = ["| K | R0 | RT | 5/5<=1.00 セル数（参考） | 総合判定 | 系列 |", "|---:|---|---|---:|---|---|"]
    detail = []
    for k in KS_CANDIDATE:
        label = f"2118-gb10-k{k}"
        d = Path(base) / "gb10" / f"k{k}"
        if not (d / "r1r2").exists():
            rows.append(f"| {k} | - | - | - | 未確定（実測未実施） | - |")
            continue
        rep = (d / "sme_report.txt").read_text(encoding="utf-8") if (d / "sme_report.txt").exists() else ""
        r0 = rep.count("kernel_enabled: false") >= 2
        rt = _parse_rt(str(d / "rt_result.txt")) or "regression-suspect"
        rt = "regression-suspect" if rt not in ("pass",) else "pass"
        cells = apply_k(evaluate_series(str(d / "r1r2"), label), 0)  # SME 非到達のため全セル非到達扱い
        for c in cells:
            c["reached"] = False
        run_ok = run_ab_ok_gb10(str(d / "rt_result.txt")) is True
        v = gb10_verdict(r0, rt, cells, run_ok)
        # RULE.txt §8: 外側専有ゲート不通過（またはログなし）の系列は「参考」と明記する
        gate = outer_gate_official(str(d / "load_gate_outer.log"))
        if gate is not True:
            v += "（参考）"
        series = {True: "正式（外側専有ゲート通過）", False: "参考（外側専有ゲート不通過）",
                  None: "参考（外側ゲートログなし）"}[gate]
        n = sum(1 for c in cells if c.get("all_le_1"))
        if not run_ok:
            series += "／run_ab_sme_cpu.sh 非ゼロ終了または終了コード記録なし"
        rows.append(f"| {k} | {'成立' if r0 else '不成立'} | {rt} | {n} | {v} | {series} |")
        detail.append(f"### K={k}\n\n{render_cells(cells)}\n")
    return "\n".join(out + rows + [""] + detail)


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


def self_test():
    # (f) K による到達分類: train は K=64 のみ到達・infer は全候補で非到達・gemm は全候補で到達
    assert [reached(k, "train") for k in KS_CANDIDATE] == [True, False, False]
    assert not any(reached(k, "infer") for k in KS_CANDIDATE)
    assert all(reached(k, "gemm") for k in KS_CANDIDATE)
    # classify: (b) 中央値 <=1 だが 1 round >1（train reuse の事例）→ adopt でない
    c = classify_cell(True, [1.0244, 0.9, 0.95, 0.98, 1.0248])
    assert c["median"] <= 1.0 and not c["adopt"] and not c["reject"]
    c = classify_cell(True, [0.9] * 5)
    assert c["adopt"]
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
        # (a) 全条件を満たす候補（K=128。train は非到達で 1.0 超のノイズがあっても reject 材料にならない）
        rb = _uniform()
        rb[("train", 64, "reuse")] = [1.02, 1.01, 0.99, 1.02, 1.03]  # 非到達ではノイズ
        _write_series(os.path.join(td, "a"), "L", rb)
        cells = apply_k(evaluate_series(os.path.join(td, "a"), "L"), 128)
        assert candidate_verdict(True, cells) == "ADOPT 候補", candidate_verdict(True, cells)
        # §11: 負荷ゲート不通過（record_only）系列は ADOPT 条件を満たしても "ADOPT 候補" にしない
        assert candidate_verdict(True, cells, official=False) == REFERENCE_ADOPT
        assert candidate_verdict(True, cells, official=False) != "ADOPT 候補"
        assert ac1_ok(cells)
        # 同じデータを K=64 で見ると train reuse（到達）が ADOPT 候補条件不成立 → undetermined（(b)）
        cells64 = apply_k(evaluate_series(os.path.join(td, "a"), "L"), 64)
        assert candidate_verdict(True, cells64) == "undetermined", candidate_verdict(True, cells64)
        assert not ac1_ok(cells64)
        # R4 不成立なら ADOPT にならない
        assert candidate_verdict(False, cells) == "undetermined"
        # §11: 不通過系列の 5/5 後退は正式 REJECT ではなく参考扱い
        rc = [dict(c) for c in cells]
        for c in rc:
            if c["reached"] and c["status"] == "ok":
                c["reject"] = True
        assert candidate_verdict(True, rc, official=True) == "REJECT"
        assert candidate_verdict(True, rc, official=False) == REFERENCE_REJECT
        assert candidate_verdict(True, rc, official=False, run_ok=False) == "undetermined"
        # §13: R4 の異常終了・欠損は判定不能
        _r4 = os.path.join(td, "r4gate.log")
        _good = "".join(f"run{i} gate=pass\nrun{i} exit=0 grep_exit=0 end_load1=1\n" for i in range(1, 6))
        Path(_r4).write_text(_good + "series done\n")
        assert r4_runs_ok(_r4) is True
        Path(_r4).write_text(_good)
        assert r4_runs_ok(_r4) is False
        Path(_r4).write_text(_good.replace("run5 exit=0", "run5 exit=1") + "series done\n")
        assert r4_runs_ok(_r4) is False
        assert r4_runs_ok(os.path.join(td, "none.log")) is None
        # run_ab_sme_cpu.sh の実行失敗は ADOPT／参考 ADOPT にも FAIL／REJECT にもせず判定不能
        assert candidate_verdict(True, cells, run_ok=False) == "undetermined"
        assert candidate_verdict(True, cells, official=False, run_ok=False) == "undetermined"
        assert gb10_verdict(True, "pass", cells, run_ok=False) == "undetermined"
        _p = os.path.join(td, "env_ok.txt")
        Path(_p).write_text("label=x run_ab_exit=0\n")
        assert run_ab_ok_m4max(_p) is True
        Path(_p).write_text("label=x run_ab_exit=1\n")
        assert run_ab_ok_m4max(_p) is False
        Path(_p).write_text("label=x\n")
        assert run_ab_ok_m4max(_p) is None
        assert run_ab_ok_m4max(os.path.join(td, "nonexistent")) is None
        Path(_p).write_text("RT rc=101\nrun_ab rc=0\n")
        assert run_ab_ok_gb10(_p) is True
        Path(_p).write_text("RT rc=101\nrun_ab rc=2\n")
        assert run_ab_ok_gb10(_p) is False
        # (c) 非到達セル（infer）が 5/5 >1.00 → 総合は ADOPT にならない
        rc = _uniform()
        rc[("infer", 64, "fresh")] = [1.01] * 5
        _write_series(os.path.join(td, "c"), "L", rc)
        cellsc = apply_k(evaluate_series(os.path.join(td, "c"), "L"), 128)
        assert candidate_verdict(True, cellsc) == "undetermined"
        assert any(x["reject"] and not x["reached"] for x in cellsc)
        # (d) 到達セル（gemm）が 5/5 >1.00 → REJECT
        rd = _uniform()
        rd[("gemm", 512, "fresh")] = [1.03] * 5
        _write_series(os.path.join(td, "d"), "L", rd)
        assert candidate_verdict(True, apply_k(evaluate_series(os.path.join(td, "d"), "L"), 256)) == "REJECT"
        # (e) checksum 不一致（丸めた集約値が複合判定内でも exact 不一致）→ FAIL
        _write_series(os.path.join(td, "e"), "L", _uniform(), checksum_after={("gemm", 1024, "reuse"): 1.5000001})
        cellse = apply_k(evaluate_series(os.path.join(td, "e"), "L"), 128)
        assert candidate_verdict(True, cellse) == "FAIL"
        assert not ac1_ok(cellse)
        # 欠損 JSONL は undetermined（値を捏造しない）
        os.makedirs(os.path.join(td, "m"))
        assert candidate_verdict(True, apply_k(evaluate_series(os.path.join(td, "m"), "L"), 128)) == "undetermined"
        # (h) GB10 語彙: RT の既知 FAIL は許容し pass・それ以外の FAIL は後退あり相当
        assert rt_verdict([]) == "pass"
        assert rt_verdict([KNOWN_RT_FAIL]) == "pass"
        assert rt_verdict([KNOWN_RT_FAIL, "gemm_blis::tests::other"]) == "regression-suspect"
        g = _uniform(gemm=1.0, train=1.0, infer=1.0)
        g[("gemm", 512, "fresh")] = [2.007, 0.99, 1.0, 0.98, 1.0]  # before 外れ値相当。5/5<=1 不成立でも後退ではない
        _write_series(os.path.join(td, "g"), "L", g)
        cellsg = apply_k(evaluate_series(os.path.join(td, "g"), "L"), 0)
        for x in cellsg:
            x["reached"] = False
        assert gb10_verdict(True, "pass", cellsg) == "pass"
        assert gb10_verdict(True, "regression-suspect", cellsg) == "後退あり相当（要調査）"
        assert gb10_verdict(False, "pass", cellsg) == "undetermined"
        g2 = _uniform(gemm=1.0)
        g2[("gemm", 1024, "reuse")] = [1.02, 1.03, 1.01, 1.05, 1.02]
        _write_series(os.path.join(td, "g2"), "L", g2)
        cg2 = apply_k(evaluate_series(os.path.join(td, "g2"), "L"), 0)
        assert gb10_verdict(True, "pass", cg2) == "後退あり"
        # load ゲートログの解釈
        p = os.path.join(td, "gate.log")
        Path(p).write_text("threshold=8.0 rule_threshold=8.0 series=official\n" +
                           "".join(f"round{i} gate=pass threshold=8.0 load1=1 waited_s=0\n" for i in range(1, 6)))
        assert load_gate_series(p) is True
        Path(p).write_text("threshold=8.0 rule_threshold=8.0 series=official\nround1 gate=timeout\n")
        assert load_gate_series(p) is False
        assert load_gate_series(os.path.join(td, "none.log")) is None
        # 外側専有ゲート（GB10）の解釈
        Path(p).write_text("start attempt=1 load1=0.1 gpu_util=0 wait\nstart attempt=2 load1=0.1 gpu_util=0 pass\n")
        assert outer_gate_official(p) is True
        Path(p).write_text("start attempt=20 load1=3 gpu_util=NA fail(reference)\n")
        assert outer_gate_official(p) is False
        assert outer_gate_official(os.path.join(td, "none.log")) is None
        Path(p).write_text("run1 gate=pass\n" * 4 + "run5 gate=timeout\n")
        assert _gate_log_official(p, "run") is False
        # 未実測ディレクトリの描画が例外なく「未実測」を出す
        txt = render_m4max(td) + render_gb10(td)
        assert "未実測" in txt and "未確定（実測未実施）" in txt
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
