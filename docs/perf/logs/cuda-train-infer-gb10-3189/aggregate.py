#!/usr/bin/env python3
"""イシュー #3189: GB10 の CUDA train／infer の内訳・同期回数・固定費を RULE.txt の規則で集計する。

標準ライブラリのみ。使い方（`python3 -I` で実行）:
  aggregate.py <dir> > aggregate.md
  aggregate.py --self-test

入力（<dir> 配下。orchestrate_gb10.sh の出力）:
  armA/run{1..5}.jsonl            bench-fandhe --phases（cuda 正式・cpu 参考）
  infer-diag/run{1..5}.facade.jsonl／.backend.jsonl   infer 上位・下位の分解診断（n=80）
  fixed-cost/run{1..5}.log        DIAG2109 行（N=128 の E／L2／L2S）
  census/<window>-K<k>-r<n>.{json,cuda_api_sum.csv,cuda_gpu_kern_sum.csv,cuda_gpu_mem_time_sum.csv}
  load_gate.log                   負荷ゲート記録

欠損・非数値・n≠80・checksum 不一致・run の欠落・ゲート状態の欠落・ホスト名／絶対パスの残存は
例外（exit 1）で拒否する（0 埋めしない）。判定閾値は RULE.txt が正で、本スクリプトは変更しない。
census が無い場合（nsys 不在）は同期回数の節を「未実測」と出す（コード読解の値を実測と書かない）。
"""
import csv
import io
import json
import math
import re
import statistics
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

RUNS = (1, 2, 3, 4, 5)
EXPECTED_N = 80
# 腕 A の cell（task, mode）ごとの必須フェーズ（bench-fandhe 固定版。cuda／cpu 共通の最小集合。RULE.txt 規則 7）
REQUIRED_PHASES = {
    ("train_phases", "fresh"): frozenset({"forward", "backward", "step_total"}),
    ("train_phases", "reuse"): frozenset({"forward_resident", "backward", "device_update", "step_total"}),
    ("infer_phases", "fresh"): frozenset({"predict", "iter_total"}),
    ("infer_phases", "reuse"): frozenset({"predict_resident", "iter_total"}),
}
U64_MAX = 2**64 - 1
TOTAL = {"train_phases": "step_total", "infer_phases": "iter_total"}
WINDOWS = ("fwd", "fwd_bwd", "full", "infer_reuse")
KS = (20, 120)
FIDELITY_LO, FIDELITY_HI = 0.8, 1.2
SYNC_NAMES = ("cuStreamSynchronize", "cuCtxSynchronize", "cuEventSynchronize")
CATEGORIES = {
    "sync": re.compile(r"^(cuStreamSynchronize|cuCtxSynchronize|cuEventSynchronize)"),
    "launch": re.compile(r"^cuLaunchKernel"),
    "htod": re.compile(r"^cuMemcpyHtoD"),
    "dtoh": re.compile(r"^cuMemcpyDtoH"),
    "alloc": re.compile(r"^cuMemAlloc"),
    "free": re.compile(r"^cuMemFree"),
    "wait_event": re.compile(r"^cuStreamWaitEvent"),
}
LEAK_RE = re.compile(r"(/home/[A-Za-z0-9_.-]+|/Users/[A-Za-z0-9_.-]+|spark-dbd9|local\.fandhe)")


class AggError(Exception):
    pass


def finite(v, what, signed=False):
    """所要時間は非負の有限数。signed=True（paired/residual＝反復ごとの public − Σ decomposed の符号付き差分）は
    負も許し、数値・有限のみを検証する（許容誤差・判定閾値は緩めない）。"""
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or (v < 0 and not signed):
        raise AggError(f"{what} が数値でない/負/非有限: {v!r}")
    return float(v)


def med(xs):
    return statistics.median(xs)


def rng(xs):
    return f"{min(xs)*1e6:.1f}–{max(xs)*1e6:.1f}"


# ---- 漏えい検査 ----
def check_leaks(d: Path):
    for f in sorted(d.rglob("*")):
        if not f.is_file() or f.suffix in (".nsys-rep", ".sqlite"):
            continue
        try:
            text = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        m = LEAK_RE.search(text)
        if m:
            raise AggError(f"{f.relative_to(d)} にホスト名/絶対パスの残存（{m.group(1)}）")


# ---- 負荷ゲート ----
GATE_RE = re.compile(r"^(run[1-5](?:-infer|-fixedcost)?) gate=(\S+) ")


def check_gate(d: Path):
    f = d / "load_gate.log"
    if not f.exists():
        raise AggError("load_gate.log が無い（負荷ゲート記録欠損）")
    status, failed = {}, []
    for line in f.read_text(encoding="utf-8").splitlines():
        m = GATE_RE.match(line)
        if m:
            if m.group(1) in status:
                raise AggError(f"load_gate.log に {m.group(1)} の gate 記録が重複")
            status[m.group(1)] = m.group(2)
        elif " FAILED " in line or line.startswith("FAILED"):
            failed.append(line)
    expected = [f"run{i}{s}" for i in RUNS for s in ("", "-infer", "-fixedcost")]
    missing = [k for k in expected if k not in status]
    if missing:
        raise AggError(f"load_gate.log に gate 記録がない: {missing}")
    bad = {k: status[k] for k in expected if status[k] != "pass"}
    if failed:
        # RULE.txt 規則7: 実行失敗の痕跡は参考系列へ降格せず例外（exit≠0）で拒否する
        raise AggError(f"load_gate.log に実行失敗の痕跡（{len(failed)} 件の FAILED）: {failed[:3]}")
    if bad:
        return False, f"負荷ゲート未通過（{bad}）"
    return True, "負荷ゲート 15/15 pass（run 5 × armA／infer／fixedcost）"


# ---- 腕 A ----
def load_arm_a(d: Path):
    vals = defaultdict(dict)
    order = {}
    sets = {}
    for i in RUNS:
        f = d / "armA" / f"run{i}.jsonl"
        if not f.exists():
            raise AggError(f"5 run 揃っていない（不足: armA/{f.name}）")
        err = d / "armA" / f"run{i}.err"
        if err.exists() and err.read_text(encoding="utf-8").strip():
            raise AggError(f"armA/run{i}.err が空でない（失敗の痕跡）")
        seen = set()
        for ln, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError as e:
                raise AggError(f"armA/{f.name}:{ln} が不正 JSON: {e}")
            if r.get("task") not in TOTAL:
                continue
            key = (r["device"], r["task"], r["mode"], r["phase"])
            if key in seen:
                raise AggError(f"armA/{f.name}:{ln} に重複行 {key}")
            seen.add(key)
            vals[key][i] = finite(r.get("median_s"), f"armA/{f.name}:{ln} median_s")
            order[key] = r["phase_index"]
        cells = defaultdict(set)
        for k in seen:
            cells[k[:3]].add(k[3])
        for dev in ("cuda", "cpu"):
            for task in TOTAL:
                for mode in ("fresh", "reuse"):
                    c = (dev, task, mode)
                    if c not in cells:
                        raise AggError(f"armA/{f.name} に欠落セル {c}")
                    miss = REQUIRED_PHASES[(task, mode)] - cells[c]
                    if miss:
                        raise AggError(f"armA/{f.name} の {c} に必須フェーズ {sorted(miss)} がない")
                    if TOTAL[task] not in cells[c]:
                        raise AggError(f"armA/{f.name} の {c} に合計フェーズ {TOTAL[task]} がない")
        sets[i] = {c: frozenset(p) for c, p in cells.items()}
    for i in RUNS[1:]:
        if sets[i] != sets[1]:
            raise AggError(f"run{i} と run1 でフェーズ集合が不一致")
    for k, per in vals.items():
        if sorted(per) != list(RUNS):
            raise AggError(f"{k} が 5 run 揃っていない（{sorted(per)}）")
    return vals, order


def render_arm_a(vals, order, formal):
    out = ["## 腕 A: train／infer `--phases`（registry `fandhe-ai =0.11.0`）\n"]
    out.append(f"- 系列区分: {'**正式系列（cuda）**' if formal else '**参考系列（負荷ゲート未通過。結論は参考）**'}。cpu は参考\n")
    meds = {}
    for dev in ("cuda", "cpu"):
        for task in TOTAL:
            for mode in ("fresh", "reuse"):
                cell = (dev, task, mode)
                keys = sorted((k for k in vals if k[:3] == cell), key=lambda k: order[k])
                total = med(vals[cell + (TOTAL[task],)].values())
                out.append(f"### {task} / {dev} / {mode}\n")
                out.append("| phase | run1..5 中央値 (µs) | min–max (µs) | 合計比 |")
                out.append("|---|---:|---|---:|")
                for k in keys:
                    runs = [vals[k][i] for i in RUNS]
                    m = med(runs)
                    meds[k] = m
                    out.append(f"| {k[3]} | {m*1e6:.1f} | {rng(runs)} | {m/total*100:.1f}% |")
                out.append("")
    return out, meds


# ---- infer 分解 ----
def load_jsonl_lines(f: Path, what):
    if not f.exists():
        raise AggError(f"{what}: {f.name} が無い")
    rows = []
    for ln, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except ValueError as e:
                raise AggError(f"{what}: {f.name}:{ln} が不正 JSON: {e}")
    return rows


def load_infer(d: Path):
    fac = defaultdict(dict)  # (arm, phase) -> {run: median}
    bak = defaultdict(dict)
    fsum, bsum = {}, {}
    for i in RUNS:
        for part, store, sums in (("facade", fac, fsum), ("backend", bak, bsum)):
            rows = load_jsonl_lines(d / "infer-diag" / f"run{i}.{part}.jsonl", "infer-diag")
            err = d / "infer-diag" / f"run{i}.{part}.err"
            if err.exists() and err.read_text(encoding="utf-8").strip():
                raise AggError(f"infer-diag/run{i}.{part}.err が空でない")
            for r in rows:
                if r.get("backend") != "cuda":
                    raise AggError(f"infer-diag/run{i}.{part}: backend が cuda でない（{r.get('backend')!r}）")
                if r.get("phase") == "checksum":
                    cs = r.get("checksum")
                    if "checksum" not in r or isinstance(cs, bool) or not isinstance(cs, int) or not 0 <= cs <= U64_MAX:
                        raise AggError(f"infer-diag/run{i}.{part}: checksum が u64 整数でない/欠落: {cs!r}")
                    sums[i] = cs
                    continue
                if r.get("n") != EXPECTED_N:
                    raise AggError(f"infer-diag/run{i}.{part}: n={r.get('n')!r} ≠ {EXPECTED_N}")
                key = (r.get("arm", "backend"), r["phase"])
                if i in store[key]:
                    raise AggError(f"infer-diag/run{i}.{part} に重複 {key}")
                store[key][i] = finite(r.get("median_s"), f"infer-diag/run{i}.{part} {key} median_s",
                                          signed=(key[0] == "paired"))
            if i not in sums:
                raise AggError(f"infer-diag/run{i}.{part} に checksum 行がない")
    for name, s in (("facade", fsum), ("backend", bsum)):
        if len(set(s.values())) != 1:
            raise AggError(f"infer-diag {name} の checksum が 5 run で不一致: {s}")
    for store, names in ((fac, "facade"), (bak, "backend")):
        for k, per in store.items():
            if sorted(per) != list(RUNS):
                raise AggError(f"infer-diag {names} {k} が 5 run 揃っていない")
    need_f = {("public", "predict_resident"), ("decomposed", "tape"), ("decomposed", "snapshot"),
              ("decomposed", "steps"), ("decomposed", "chain"), ("decomposed", "readout"), ("paired", "residual")}
    need_b = {("backend", p) for p in ("h2d_upload", "l1_issue", "l2_issue", "download",
                                       "download_again", "kernel_wait_derived", "total")}
    if set(fac) != need_f:
        raise AggError(f"infer-diag facade の区間集合が不一致（差分 {sorted(set(fac) ^ need_f)}）")
    if set(bak) != need_b:
        raise AggError(f"infer-diag backend の区間集合が不一致（差分 {sorted(set(bak) ^ need_b)}）")
    return fac, bak


def render_infer(fac, bak, meds):
    out = ["## 腕 B-2: infer reuse の内部内訳\n"]
    fm = {k: med(v.values()) for k, v in fac.items()}
    bm = {k[1]: med(v.values()) for k, v in bak.items()}
    out.append("### 上位（facade。decomposed ＋ paired/residual。5 run 中央値）\n")
    out.append("| arm | phase | 中央値 (µs) | min–max (µs) |")
    out.append("|---|---|---:|---|")
    for k in sorted(fm):
        out.append(f"| {k[0]} | {k[1]} | {fm[k]*1e6:.1f} | {rng(list(fac[k].values()))} |")
    out.append("\n### 下位（backend-cuda。5 run 中央値）\n")
    out.append("| phase | 中央値 (µs) | min–max (µs) |")
    out.append("|---|---:|---|")
    for k in sorted(bm):
        out.append(f"| {k} | {bm[k]*1e6:.1f} | {rng(list(bak[('backend', k)].values()))} |")
    chain = fm[("decomposed", "chain")]
    fid = bm["total"] / chain if chain > 0 else float("nan")
    ok = FIDELITY_LO <= fid <= FIDELITY_HI
    out.append(
        f"\n- fidelity = backend total / facade chain = {fid:.3f}（許容 {FIDELITY_LO}〜{FIDELITY_HI}）→ "
        f"{'区間は正式扱い' if ok else '**範囲外: 区間は参考扱い**'}"
    )
    upper = fm[("decomposed", "tape")] + fm[("decomposed", "snapshot")] + fm[("decomposed", "steps")] + fm[("paired", "residual")]
    out.append(f"- 上位の固定費 = tape＋snapshot＋steps＋paired/residual = {upper*1e6:.1f} µs")
    pub = fm[("public", "predict_resident")]
    out.append(f"- public predict_resident = {pub*1e6:.1f} µs（decomposed 合計 {sum(fm[('decomposed', p)] for p in ('tape','snapshot','steps','chain','readout'))*1e6:.1f} µs ＋ residual {fm[('paired','residual')]*1e6:.1f} µs）")
    ref = meds.get(("cuda", "infer_phases", "reuse", "iter_total"))
    if ref is not None:
        out.append(f"- 腕 A の infer reuse iter_total = {ref*1e6:.1f} µs（版差を含む。腕をまたぐ比較）")
    out.append("")
    return out, bm, fm, fid, ok


# ---- 固定費 ----
DIAG_RE = re.compile(r"^DIAG2109 n=(\d+) layer=(\S+) phase=(\S+) median_us=([0-9.]+) q1_us=([0-9.]+) q3_us=([0-9.]+)$")
FIXED_KEYS = (("E", "sync_idle"), ("E", "tiny_roundtrip"), ("E", "event_create_drop"), ("L2", "launch_issue"), ("L2S", "kernel_wait"))


def load_fixed(d: Path):
    res = defaultdict(dict)
    for i in RUNS:
        f = d / "fixed-cost" / f"run{i}.log"
        if not f.exists():
            raise AggError(f"fixed-cost/run{i}.log が無い")
        found = {}
        for line in f.read_text(encoding="utf-8").splitlines():
            m = DIAG_RE.match(line.strip())
            if m and int(m.group(1)) == 128 and (m.group(2), m.group(3)) in FIXED_KEYS:
                key = (m.group(2), m.group(3))
                if key in found:
                    raise AggError(f"fixed-cost/run{i}.log に重複 {key}")
                found[key] = float(m.group(4)) * 1e-6
        for k in FIXED_KEYS:
            if k not in found:
                raise AggError(f"fixed-cost/run{i}.log に N=128 の {k} 行がない")
            res[k][i] = found[k]
    return res


def render_fixed(fx):
    out = ["## 腕 B-3: launch／同期の固定費（N=128。#2109 の診断を計測器として借用）\n"]
    out.append("| layer | phase | 中央値 (µs) | min–max (µs) |")
    out.append("|---|---|---:|---|")
    m = {}
    for k in FIXED_KEYS:
        runs = [fx[k][i] for i in RUNS]
        m[k] = med(runs)
        out.append(f"| {k[0]} | {k[1]} | {m[k]*1e6:.2f} | {rng(runs)} |")
    out.append("\n- tiny_roundtrip は 1×1×1 GEMM の launch＋同期で、引数なしの空カーネルの代替（RULE 規則 5）\n")
    return out, m


# ---- census ----
def parse_api_csv(text: str):
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith('"Time (%)"') or l.startswith("Time (%)")), None)
    if start is None:
        raise AggError("nsys cuda_api_sum の CSV ヘッダが見つからない")
    rows = list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))
    cnt = {c: 0 for c in CATEGORIES}
    if not rows:
        raise AggError("nsys cuda_api_sum にデータ行が無い（ヘッダのみ。ゼロ回として扱わない）")
    for r in rows:
        name = (r.get("Name") or "").strip()
        n = r.get("Num Calls")
        if n is None or not re.fullmatch(r"[0-9]+", n.strip()):
            raise AggError(f"nsys cuda_api_sum の Num Calls が整数でない: {n!r}")
        for c, rx in CATEGORIES.items():
            if rx.match(name):
                cnt[c] += int(n)
    if cnt["launch"] == 0:
        raise AggError("nsys cuda_api_sum に cuLaunchKernel の記録が無い（必ず発生する記録の欠落）")
    return cnt


def load_census(d: Path):
    cdir = d / "census"
    if not cdir.exists() or not any(cdir.glob("*.cuda_api_sum.csv")):
        return None
    res = {}
    for w in WINDOWS:
        for r in RUNS:
            per_k = {}
            for k in KS:
                base = cdir / f"{w}-K{k}-r{r}"
                f = base.with_suffix("").parent / f"{base.name}.cuda_api_sum.csv"
                j = cdir / f"{base.name}.json"
                if not f.exists() or not j.exists():
                    raise AggError(f"census/{base.name} の csv/json が欠落")
                meta = json.loads(j.read_text(encoding="utf-8").replace("CENSUS_JSON ", "", 1))
                if meta.get("window") != w or meta.get("steps") != k:
                    raise AggError(f"census/{base.name}.json の窓/steps が不一致")
                per_k[k] = parse_api_csv(f.read_text(encoding="utf-8"))
            res[(w, r)] = {c: (per_k[120][c] - per_k[20][c]) / 100 for c in CATEGORIES}
    return res


def render_census(cs):
    out = ["## 腕 B-1: 1 step あたりの回数（nsys・K=20／120 差分法）\n"]
    if cs is None:
        out.append("**未実測**（census/ が無い: nsys 不在または実行失敗）。コード読解の約 9 回を実測とは書かない。\n")
        return out, None
    steps = {}
    out.append("| window | " + " | ".join(CATEGORIES) + " | 決定性 |")
    out.append("|---|" + "---:|" * len(CATEGORIES) + "---|")
    for w in WINDOWS:
        per_cat = {c: [cs[(w, r)][c] for r in RUNS] for c in CATEGORIES}
        det = all(len(set(v)) == 1 for v in per_cat.values())
        steps[w] = {c: med(v) for c, v in per_cat.items()}
        cells = []
        for c, v in per_cat.items():
            cells.append(f"{med(v):g}" if len(set(v)) == 1 else f"{med(v):g} ({min(v):g}–{max(v):g})")
        out.append(f"| {w} | " + " | ".join(cells) + f" | {'決定的' if det else '**非決定**'} |")
    out.append("\n### 区間への割り当て（RULE 規則 2）\n")
    out.append("| 区間 | sync | launch | htod | dtoh |")
    out.append("|---|---:|---:|---:|---:|")
    rows = {
        "forward": {c: steps["fwd"][c] for c in CATEGORIES},
        "backward": {c: steps["fwd_bwd"][c] - steps["fwd"][c] for c in CATEGORIES},
        "update＋tape_drop": {c: steps["full"][c] - steps["fwd_bwd"][c] for c in CATEGORIES},
        "full（1 step）": steps["full"],
        "infer_reuse": steps["infer_reuse"],
    }
    for name, v in rows.items():
        out.append(f"| {name} | {v['sync']:g} | {v['launch']:g} | {v['htod']:g} | {v['dtoh']:g} |")
    out.append("")
    return out, steps


# ---- 帰属 ----
def render_attrib(meds, fixed_m, steps, bm, fidelity_ok=True):
    out = ["## 帰属と見込み（record-only。原因の断定はしない）\n"]
    if steps is None:
        out.append("同期回数が未実測のため、同期回数に基づく見込みは出さない。\n")
        return out
    step = meds[("cuda", "train_phases", "reuse", "step_total")]
    inf = meds[("cuda", "infer_phases", "reuse", "iter_total")]
    n_full, n_inf = steps["full"]["sync"], steps["infer_reuse"]["sync"]
    cs = {
        "sync_idle": fixed_m[("E", "sync_idle")],
        "tiny_roundtrip": fixed_m[("E", "tiny_roundtrip")],
        "実カーネル同期(kernel_wait_derived)": bm["kernel_wait_derived"],
    }
    gap = step - inf
    if not fidelity_ok:
        # fidelity 範囲外の backend 区間は参考扱い: 実カーネル同期を c に使う見込みは出さない
        del cs["実カーネル同期(kernel_wait_derived)"]
        out.append("- **fidelity 範囲外のため kernel_wait_derived に基づく削減見込み・説明率は算出しない（参考扱い）**")
    out.append(f"- train reuse step_total = {step*1e6:.1f} µs、infer reuse iter_total = {inf*1e6:.1f} µs、差 = {gap*1e6:.1f} µs")
    out.append(f"- 同期回数: train full = {n_full:g}、infer = {n_inf:g}")
    out.append("\n| c の種別 | c (µs) | (n_full−1)×c (µs) | step_total に対する説明率 | 未説明残差 (µs) |")
    out.append("|---|---:|---:|---:|---:|")
    for name, c in cs.items():
        v = (n_full - 1) * c
        out.append(f"| {name} | {c*1e6:.2f} | {v*1e6:.1f} | {v/step*100:.1f}% | {(step-v)*1e6:.1f} |")
    launch_n = steps["full"]["launch"]
    li = fixed_m[("L2", "launch_issue")]
    v = max(launch_n - 1, 0) * li
    out.append(f"\n- launch 固定費の上限 (launch_count−1)×launch_issue = ({launch_n:g}−1)×{li*1e6:.2f} = {v*1e6:.1f} µs（step_total の {v/step*100:.1f}%）")
    if n_full != n_inf:
        per = gap / (n_full - n_inf)
        out.append(f"- 1 同期あたりの外挿 (step_total−iter_total)/(n_full−n_infer) = {per*1e6:.1f} µs（実測固定費と比べる。約 27 µs の再計算）")
    else:
        out.append("- 同期回数が同数のため 1 同期あたりの外挿は定義できない")
    out.append("")
    return out


def main_report(d: Path):
    check_leaks(d)
    gate_ok, gate_msg = check_gate(d)
    vals, order = load_arm_a(d)
    fac, bak = load_infer(d)
    fx = load_fixed(d)
    cs = load_census(d)
    out = [f"# #3189 GB10 集計（{'正式系列' if gate_ok else '参考系列'}）\n", f"- {gate_msg}\n"]
    sec, meds = render_arm_a(vals, order, gate_ok)
    out += sec
    sec, bm, fm, fid, ok = render_infer(fac, bak, meds)
    out += sec
    sec, fixed_m = render_fixed(fx)
    out += sec
    sec, steps = render_census(cs)
    out += sec
    out += render_attrib(meds, fixed_m, steps, bm, ok)
    return "\n".join(out)


# ---- self-test ----
def _w(p: Path, text: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def _fixture(d: Path, with_census=True):
    phases = {
        ("train_phases", "fresh"): ["forward", "backward", "step_total"],
        ("train_phases", "reuse"): ["forward_resident", "backward", "device_update", "step_total"],
        ("infer_phases", "fresh"): ["predict", "iter_total"],
        ("infer_phases", "reuse"): ["predict_resident", "iter_total"],
    }
    gate = []
    for i in RUNS:
        lines = []
        for dev in ("cuda", "cpu"):
            for (task, mode), ps in phases.items():
                for idx, p in enumerate(ps):
                    t = 3e-4 if p in ("step_total", "iter_total") else 1e-4
                    if task == "infer_phases" and p == "iter_total":
                        t = 1e-4
                    lines.append(json.dumps({"device": dev, "task": task, "mode": mode, "phase": p, "phase_index": idx, "median_s": t * (1 + 0.01 * i)}))
        _w(d / "armA" / f"run{i}.jsonl", "\n".join(lines) + "\n")
        _w(d / "armA" / f"run{i}.err", "")
        fl = []
        for arm, ph, v in [("public", "predict_resident", 1e-4), ("decomposed", "tape", 1e-6), ("decomposed", "snapshot", 1e-5),
                           ("decomposed", "steps", 1e-7), ("decomposed", "chain", 7e-5), ("decomposed", "readout", 1e-6), ("paired", "residual", 1e-6)]:
            fl.append(json.dumps({"backend": "cuda", "arm": arm, "phase": ph, "n": 80, "median_s": v}))
        fl.append(json.dumps({"backend": "cuda", "arm": "meta", "phase": "checksum", "checksum": 7}))
        _w(d / "infer-diag" / f"run{i}.facade.jsonl", "\n".join(fl) + "\n")
        _w(d / "infer-diag" / f"run{i}.facade.err", "")
        bl = []
        for ph, v in [("h2d_upload", 1e-5), ("l1_issue", 2e-5), ("l2_issue", 2e-5), ("download", 2e-5),
                      ("download_again", 5e-6), ("kernel_wait_derived", 1.5e-5), ("total", 7e-5)]:
            bl.append(json.dumps({"backend": "cuda", "phase": ph, "n": 80, "median_s": v}))
        bl.append(json.dumps({"backend": "cuda", "phase": "checksum", "checksum": 9}))
        _w(d / "infer-diag" / f"run{i}.backend.jsonl", "\n".join(bl) + "\n")
        _w(d / "infer-diag" / f"run{i}.backend.err", "")
        dl = [f"DIAG2109 n=128 layer={l} phase={p} median_us={v} q1_us={v} q3_us={v}" for (l, p), v in
              zip(FIXED_KEYS, (10.0, 30.0, 5.0, 8.0, 40.0))]
        _w(d / "fixed-cost" / f"run{i}.log", "\n".join(dl) + "\n")
        for s in ("", "-infer", "-fixedcost"):
            gate.append(f"run{i}{s} gate=pass load1=0.1 gpu_util=0 waited_s=0 at=x")
    _w(d / "load_gate.log", "\n".join(gate) + "\nseries done\n")
    if with_census:
        base_cnt = {"fwd": 3, "fwd_bwd": 6, "full": 9, "infer_reuse": 1}
        for w in WINDOWS:
            for r in RUNS:
                for k in KS:
                    n = 5 + base_cnt[w] * k
                    csvt = ('Generating report\n"Time (%)","Total Time (ns)","Num Calls","Avg (ns)","Name"\n'
                            f'"50","1","{n}","1","cuStreamSynchronize"\n'
                            f'"30","1","{(n + 2 * k)}","1","cuLaunchKernel"\n'
                            f'"20","1","{k}","1","cuMemcpyDtoH_v2"\n')
                    _w(d / "census" / f"{w}-K{k}-r{r}.cuda_api_sum.csv", csvt)
                    _w(d / "census" / f"{w}-K{k}-r{r}.json", json.dumps({"window": w, "steps": k}))


def self_test():
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        _fixture(d)
        rep = main_report(d)
        assert "正式系列" in rep and "| full（1 step） | 9 |" in rep, rep
        assert "backward | 3 |" in rep
        # 決定性: 1 run の値を変えると非決定が出る
        p = d / "census" / "full-K120-r3.cuda_api_sum.csv"
        p.write_text(p.read_text(encoding="utf-8").replace('"cuStreamSynchronize"', '"cuStreamSynchronize"').replace('"50","1","1085"', '"50","1","1086"'), encoding="utf-8")
        assert "非決定" in main_report(d)
    def expect_fail(label, mutate):
        with tempfile.TemporaryDirectory() as t:
            dd = Path(t)
            _fixture(dd)
            mutate(dd)
            try:
                main_report(dd)
            except AggError:
                return
            except Exception as e:  # noqa: BLE001
                raise SystemExit(f"self-test 失敗: {label} が AggError 以外（{e!r}）")
            raise SystemExit(f"self-test 失敗: {label} を検出できない")
    expect_fail("run 欠落", lambda dd: (dd / "armA" / "run4.jsonl").unlink())
    expect_fail("n≠80", lambda dd: _w(dd / "infer-diag" / "run2.backend.jsonl", (dd / "infer-diag" / "run2.backend.jsonl").read_text().replace('"n": 80', '"n": 79')))
    expect_fail("checksum 不一致", lambda dd: _w(dd / "infer-diag" / "run3.facade.jsonl", (dd / "infer-diag" / "run3.facade.jsonl").read_text().replace('"checksum": 7', '"checksum": 8')))
    expect_fail("非数値", lambda dd: _w(dd / "armA" / "run1.jsonl", re.sub(r'"median_s": [0-9.e-]+', '"median_s": "x"', (dd / "armA" / "run1.jsonl").read_text(), count=1)))
    expect_fail("ゲート欠落", lambda dd: _w(dd / "load_gate.log", "series done\n"))
    expect_fail("FAILED 痕跡", lambda dd: _w(dd / "load_gate.log", (dd / "load_gate.log").read_text() + "run1 FAILED infer facade\n"))
    expect_fail("err 非空", lambda dd: _w(dd / "armA" / "run2.err", "boom"))
    expect_fail("絶対パス残存", lambda dd: _w(dd / "env_info.txt", "/home/someone/x"))
    expect_fail("CPU 混入", lambda dd: _w(dd / "infer-diag" / "run1.facade.jsonl", (dd / "infer-diag" / "run1.facade.jsonl").read_text().replace('"cuda"', '"cpu"', 1)))
    expect_fail("必須フェーズ欠落", lambda dd: _w(dd / "armA" / "run1.jsonl", "\n".join(l for l in (dd / "armA" / "run1.jsonl").read_text().splitlines() if '"device_update"' not in l or '"cpu"' not in l) + "\n"))
    expect_fail("checksum が bool", lambda dd: [_w(dd / "infer-diag" / f"run{i}.facade.jsonl", (dd / "infer-diag" / f"run{i}.facade.jsonl").read_text().replace('"checksum": 7', '"checksum": true')) for i in RUNS])
    expect_fail("checksum が u64 超過", lambda dd: [_w(dd / "infer-diag" / f"run{i}.facade.jsonl", (dd / "infer-diag" / f"run{i}.facade.jsonl").read_text().replace('"checksum": 7', '"checksum": 18446744073709551616')) for i in RUNS])
    expect_fail("ヘッダのみ CSV", lambda dd: _w(dd / "census" / "fwd-K20-r1.cuda_api_sum.csv", '"Time (%)","Total Time (ns)","Num Calls","Avg (ns)","Name"\n'))
    expect_fail("固定費行欠落", lambda dd: _w(dd / "fixed-cost" / "run1.log", ""))
    # census 無しは「未実測」
    with tempfile.TemporaryDirectory() as t:
        dd = Path(t)
        _fixture(dd, with_census=False)
        assert "未実測" in main_report(dd)
    print("self-test OK")


def main(argv):
    if len(argv) == 2 and argv[1] == "--self-test":
        self_test()
        return 0
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    try:
        print(main_report(Path(argv[1])))
    except AggError as e:
        print(f"AGGREGATE ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
