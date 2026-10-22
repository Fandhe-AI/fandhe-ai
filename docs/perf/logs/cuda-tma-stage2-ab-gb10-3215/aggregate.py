#!/usr/bin/env python3
"""イシュー #3215 の収録ログから RULE.txt どおりに判定語と表を機械生成する（python3 標準ライブラリのみ）。

役割: orchestrate_gb10.sh が残したログ（gateA／gateB／kern_run{1..5}／cublas／cell）を厳密にパースし、
ゲート A〜C の判定・cuBLAS 比・gemm セル比・3 分類の材料表を aggregate.md に出す。判定ロジックは RULE.txt の
項 5〜11 の逐語実装で、事後に変えない。呼び出し: `python3 aggregate.py --dir <収録ディレクトリ> --out aggregate.md`。
同じログだけで再実行すれば同じ出力になる（決定的）。

厳密さの方針（codex が集計スクリプトの寛容パースを繰り返し指摘してきたため最初から固定）:
- 存在するファイルの欠損・重複・形式不正・非有限値・非正値は例外（exit 非 0）。
- ファイル自体が無い節は `unavailable` と明記して判定を undetermined にする（推定で埋めない）。
- run 番号付きの dict で保持し、run の完備（1..5 の全て）を検証する。
`--self-test` は合成データで ADOPT／REJECT／undetermined の各経路と拒否ケースを検査する。
"""
import argparse
import json
import math
import os
import re
import statistics
import sys
import tempfile

RUNS = (1, 2, 3, 4, 5)
SIZES = (256, 512, 1024, 2048, 4096)
WIRING_SIZES = (1024, 2048, 4096)  # RULE 項 7: 結線範囲（本番 128x64 と同じ N>=1024 かつ K>=1024）
ARMS = ("cur", "pipe128", "tma128_none", "tma128_strided")
CAND_ARMS = ("tma128_none", "tma128_strided")
ADOPT_RATIO = 1.05
GATE_A_CRITICAL = (
    "tiled_pipeline_tma_128x64_matches_cp_async_128x64_bit_exact",
    "tiled_pipeline_tma_128x64_matches_cp_async_with_nan_inf_inputs",
    "tiled_pipeline_tma_128x64_k_zero_determinism_prepared_and_pretransposed",
)
GATE_A_PREREQ = "tiled_pipeline_tma_prepared_matches_one_shot_launch_bit_exact"
GATE_B_BINARIES = (
    "parity_nonregression",
    "cpu_cuda_parity",
    "cpu_cuda_tiled_pipeline_parity",
    "gemm_transposed_parity",
    "transpose_parity",
    "gemm_tiled",
)
# 実行完備の期待値（fail-closed）。Stage 1 同型の収録ログ（GB10 実測）の `test result` 行の passed+failed（実行総数）。
# 一部のテストだけ走った・0 passed の空走りを ADOPT 判定へ通さないための下限固定（RULE 項 6 の必須ゲート）。
GATE_A_EXPECTED_PASSED = 9
GATE_B_EXPECTED_PASSED = {
    "parity_nonregression": 1,
    "cpu_cuda_parity": 2,
    "cpu_cuda_tiled_pipeline_parity": 18,
    "gemm_transposed_parity": 5,
    "transpose_parity": 4,
    "gemm_tiled": 6,
}
# 判定に必須の段（stage_rc.txt の rc が 0 であること。欠落・非 0 は ADOPT を出さない）。
# cuBLAS・セルは record_only のため含めない（RULE 項 9。セル失敗は判定に影響させない）。
REQUIRED_STAGES = (
    ("build", "gateA")
    + tuple(f"gateB_{b}" for b in GATE_B_BINARIES)
    + tuple(f"kern_run{r}" for r in RUNS)
)
# 既知の無関係 fail の除外（RULE 項 6）。Stage 1 の gateB_failed_names.txt は空のため空集合。
KNOWN_UNRELATED_FAILS = frozenset()
# #3185 の cuBLAS mm_event 中央値 [ms]（docs/perf/logs/opp-kernel-gb10-3185。RULE 項 10 の比較基準）。
REF_3185_MS = {
    "fp32": {256: 0.0170, 512: 0.0337, 1024: 0.1367, 2048: 0.9846, 4096: 7.9606},
    "tf32": {256: 0.0103, 512: 0.0164, 1024: 0.0659, 2048: 0.4253, 4096: 3.3167},
}
DRIFT = 0.10
# 0.11.0 スコアボード値（RULE 項 11。対最速相手の比）。
REF_0110_REUSE = {256: 0.85, 512: 0.42, 1024: 0.40, 2048: 0.48, 4096: 1.01}
REF_0110_BEST = {256: 0.88, 512: 0.92, 1024: 0.97, 2048: 0.99, 4096: 1.01}


class AggError(Exception):
    """収録ログの形式不正・欠損・非有限値（例外にして集計を止める）。"""


def finite_pos(x, what):
    if not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x) or x <= 0:
        raise AggError(f"{what}: 有限の正数でない: {x!r}")
    return float(x)


def read_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


# ---------- 純カーネル（kern_run） ----------
ARM_RE = re.compile(
    r"^ARM3215 n=(\d+) arm=(\w+) kern_event_ms=(\S+) kern_b2b_ms=(\S+) tflops=(\S+) tile=(\S+) samples=(\d+)$"
)
BIT_RE = re.compile(r"^BITIDENT3215 n=(\d+) arm=(\w+) fnv=(0x[0-9a-f]{16}) elems=(\d+)$")


def parse_kern(text, run):
    """1 run 分: {(n, arm): {event, b2b, tflops, tile}} と {(n, arm): fnv}。"""
    arms, bits = {}, {}
    done = False
    for line in text.splitlines():
        if line.startswith("ERROR3215"):
            raise AggError(f"kern_run{run}: {line}")
        if line == "done3215":
            done = True
        m = ARM_RE.match(line)
        if m:
            n, arm = int(m.group(1)), m.group(2)
            if n not in SIZES or arm not in ARMS:
                raise AggError(f"kern_run{run}: 未知の腕/N: {line}")
            if (n, arm) in arms:
                raise AggError(f"kern_run{run}: 重複: {line}")
            if int(m.group(7)) != 20:
                raise AggError(f"kern_run{run}: samples!=20: {line}")
            arms[(n, arm)] = {
                "event": finite_pos(float(m.group(3)), line),
                "b2b": finite_pos(float(m.group(4)), line),
                "tflops": finite_pos(float(m.group(5)), line),
                "tile": m.group(6),
            }
            continue
        m = BIT_RE.match(line)
        if m:
            n, arm = int(m.group(1)), m.group(2)
            if n not in SIZES or arm not in ARMS:
                raise AggError(f"kern_run{run}: 未知の腕/N: {line}")
            if (n, arm) in bits:
                raise AggError(f"kern_run{run}: 重複: {line}")
            if int(m.group(4)) != n * n:
                raise AggError(f"kern_run{run}: elems!=n*n: {line}")
            bits[(n, arm)] = m.group(3)
    if not done:
        raise AggError(f"kern_run{run}: 完了行 done3215 が無い（途中終了）")
    want = {(n, a) for n in SIZES for a in ARMS}
    if set(arms) != want or set(bits) != want:
        raise AggError(f"kern_run{run}: (n, arm) が不完全: arms={len(arms)} bits={len(bits)}")
    return arms, bits


def bit_identity(kern_bits):
    """RULE 項 4。戻り値: (ok, 不一致の説明リスト)。"""
    problems = []
    for n in SIZES:
        per_run = {a: {kern_bits[r][(n, a)] for r in RUNS} for a in ARMS}
        for a, hs in per_run.items():
            if len(hs) != 1:
                problems.append(f"N={n} {a}: run 間で hash が不一致")
        ref = next(iter(per_run["pipe128"]))
        for a in CAND_ARMS:
            if next(iter(per_run[a])) != ref:
                problems.append(f"N={n} {a} != pipe128")
        if n >= 1024 and next(iter(per_run["cur"])) != ref:
            problems.append(f"N={n} cur != pipe128（N>=1024 は同一 128x64 経路で一致必須）")
    return (not problems), problems


def gate_c(kern):
    """腕ごとの比と判定材料。kern: {run: {(n, arm): rec}}。"""
    med = {(n, a): statistics.median(kern[r][(n, a)]["event"] for r in RUNS) for n in SIZES for a in ARMS}
    out = {}
    for cand in CAND_ARMS:
        rows = {}
        for n in SIZES:
            ratio = med[(n, "cur")] / med[(n, cand)]
            run_ratios = [kern[r][(n, "cur")]["event"] / kern[r][(n, cand)]["event"] for r in RUNS]
            side = ratio >= 1.0
            opposite = sum(1 for x in run_ratios if (x >= 1.0) != side)
            rows[n] = {"ratio": ratio, "run_ratios": run_ratios, "opposite_runs": opposite}
        out[cand] = rows
    return med, out


def verdict_for(rows):
    ratios = {n: rows[n]["ratio"] for n in WIRING_SIZES}
    if any(r < 1.0 for r in ratios.values()):
        return "REJECT"
    if any(r >= ADOPT_RATIO for r in ratios.values()):
        return "ADOPT"
    return "undetermined"


# ---------- テストログ（ゲート A／B） ----------
TEST_RE = re.compile(r"^test (\S+) \.\.\. (ok|FAILED|ignored)\b")
TEST_RESULT_RE = re.compile(r"^test result: (ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored;")


def parse_test_log(text, label):
    """戻り値: ({名前: 状態}, [(判定語, passed, failed, ignored) の test result 行リスト])。--nocapture で結果が後続行に分かれる場合は
    `test <name> ...` の行末に出力が続き `ok` が次行に来るため、名前行のみで ok/FAILED が取れない場合は後続の単独 `ok`/`FAILED` 行を拾う。"""
    names, results = {}, []
    pending = None
    for line in text.splitlines():
        m = TEST_RE.match(line)
        if m:
            if m.group(1) in names:
                raise AggError(f"{label}: テスト名が重複: {m.group(1)}")
            names[m.group(1)] = m.group(2)
            pending = None
            continue
        m2 = re.match(r"^test (\S+) \.\.\. ", line)
        if m2:
            if m2.group(1) in names:
                raise AggError(f"{label}: テスト名が重複: {m2.group(1)}")
            pending = m2.group(1)
            continue
        if pending is not None and line.strip() in ("ok", "FAILED"):
            names[pending] = line.strip()
            pending = None
            continue
        mr = TEST_RESULT_RE.match(line)
        if mr:
            results.append((mr.group(1), int(mr.group(2)), int(mr.group(3)), int(mr.group(4))))
    if pending is not None:
        raise AggError(f"{label}: テスト {pending} の結果行が無い")
    return names, results


def check_complete(names, results, expected_passed, label):
    """実行完備の検査（fail-closed。違反は AggError で集計を止める）。
    test result 行がちょうど 1 つ・passed が期待数・個別結果の件数が summary と一致することを要求する。"""
    if len(results) != 1:
        raise AggError(f"{label}: test result 行が {len(results)} 個（ちょうど 1 個が必要。途中終了または重複）")
    status, passed, failed, ignored = results[0]
    if passed + failed != expected_passed or passed + failed == 0:
        raise AggError(f"{label}: passed+failed={passed + failed} が期待数 {expected_passed} と不一致（テスト欠落または 0 passed の空走り）")
    n_ok = sum(1 for s in names.values() if s == "ok")
    n_failed = sum(1 for s in names.values() if s == "FAILED")
    n_ign = sum(1 for s in names.values() if s == "ignored")
    if (n_ok, n_failed, n_ign) != (passed, failed, ignored):
        raise AggError(f"{label}: 個別結果 ok/FAILED/ignored={n_ok}/{n_failed}/{n_ign} が summary {passed}/{failed}/{ignored} と不一致")
    if (status == "ok") != (failed == 0):
        raise AggError(f"{label}: summary の判定語 {status} と failed={failed} が矛盾")


def gate_a(text):
    names, results = parse_test_log(text, "gateA")
    check_complete(names, results, GATE_A_EXPECTED_PASSED, "gateA")
    st = {}
    for n in GATE_A_CRITICAL + (GATE_A_PREREQ,):
        if n not in names:
            raise AggError(f"gateA: 必須テスト {n} がログに無い")
        st[n] = names[n]
    critical_fail = [n for n in GATE_A_CRITICAL if st[n] != "ok"]
    prereq_ok = st[GATE_A_PREREQ] == "ok"
    return st, critical_fail, prereq_ok


def gate_b(texts):
    fails = []
    for b in GATE_B_BINARIES:
        names, results = parse_test_log(texts[b], f"gateB_{b}")
        check_complete(names, results, GATE_B_EXPECTED_PASSED[b], f"gateB_{b}")
        fails += [f"{b}::{n}" for n, s in names.items() if s == "FAILED" and n not in KNOWN_UNRELATED_FAILS]
    return fails


# ---------- cuBLAS ----------
def load_cublas(base):
    """{arm: {n: {run: median_s}}}（phase=mm_event）。"""
    out = {"fp32": {n: {} for n in SIZES}, "tf32": {n: {} for n in SIZES}}
    for r in RUNS:
        for arm in out:
            p = os.path.join(base, "cublas", f"run{r}", f"{arm}.jsonl")
            if not os.path.isfile(p):
                raise FileNotFoundError(p)
            for i, line in enumerate(read_text(p).splitlines(), 1):
                rec = json.loads(line)
                if rec.get("arm") != arm or rec.get("run") != r:
                    raise AggError(f"{p}:{i}: arm/run が不一致")
                if rec.get("phase") != "mm_event":
                    continue
                n = rec["n"]
                if n not in SIZES:
                    raise AggError(f"{p}:{i}: 未知の n={n}")
                if r in out[arm][n]:
                    raise AggError(f"{p}:{i}: mm_event 重複 n={n}")
                out[arm][n][r] = finite_pos(rec["median_s"], f"{p}:{i}") * 1e3
    for arm in out:
        for n in SIZES:
            if set(out[arm][n]) != set(RUNS):
                raise AggError(f"cublas {arm} n={n}: mm_event が 5 run そろっていない")
    return out


# ---------- gemm セル ----------
FRAMEWORK_OF = {"bench-fandhe": "fandhe-ai", "bench-candle": "candle", "bench-burn": "burn"}


def expected_cell_sequence(rnd):
    seq = []
    for n in SIZES:
        if rnd % 2 == 1:
            seq += [("fandhe-ai", "reuse", n), ("fandhe-ai", "fresh", n), ("candle", "fresh", n), ("burn", "fresh", n)]
        else:
            seq += [("burn", "fresh", n), ("candle", "fresh", n), ("fandhe-ai", "fresh", n), ("fandhe-ai", "reuse", n)]
    return seq


def load_cell(path):
    """round ごとの {(framework, mode, n): median_s}。記録順から round を割り当て、起動順パターンを検証する。"""
    recs = [json.loads(l) for l in read_text(path).splitlines() if l.strip()]
    per = len(expected_cell_sequence(1))
    if len(recs) != per * len(RUNS):
        raise AggError(f"cell: 記録数が {per * len(RUNS)} でない: {len(recs)}")
    rounds = {}
    for idx, rec in enumerate(recs):
        rnd = idx // per + 1
        want = expected_cell_sequence(rnd)[idx % per]
        got = (rec.get("framework"), rec.get("mode"), rec.get("size"))
        if got != want or rec.get("task") != "gemm" or rec.get("device") != "cuda":
            raise AggError(f"cell: round{rnd} の記録順/内容が想定と不一致: got={got} want={want}")
        rounds.setdefault(rnd, {})[want] = finite_pos(rec["median_s"], f"cell record {idx}")
    return rounds


def cell_ratios(rounds):
    res = {}
    for n in SIZES:
        reuse, best = [], []
        for r in RUNS:
            d = rounds[r]
            opp = min(d[("candle", "fresh", n)], d[("burn", "fresh", n)])
            reuse.append(opp / d[("fandhe-ai", "reuse", n)])
            best.append(opp / min(d[("fandhe-ai", "reuse", n)], d[("fandhe-ai", "fresh", n)]))
        res[n] = {
            "reuse": statistics.median(reuse),
            "best": statistics.median(best),
            "reuse_runs": reuse,
            "cur_reuse_ms": statistics.median(rounds[r][("fandhe-ai", "reuse", n)] for r in RUNS) * 1e3,
            "opp_ms": statistics.median(min(rounds[r][("candle", "fresh", n)], rounds[r][("burn", "fresh", n)]) for r in RUNS) * 1e3,
        }
    return res


# ---------- 総合 ----------
def load_gate_status(base):
    p = os.path.join(base, "load_gate.log")
    lines = [l for l in read_text(p).splitlines() if l.strip()]
    bad = [l for l in lines if not l.startswith("GATE-PASS ")]
    return len(lines), bad


def stage_rcs(base):
    rc = {}
    for l in read_text(os.path.join(base, "stage_rc.txt")).splitlines():
        m = re.match(r"^(\S+) rc=(\S+)$", l)
        if not m:
            raise AggError(f"stage_rc.txt: 形式不正: {l}")
        rc[m.group(1)] = m.group(2)
    return rc


def fmt(x, d=4):
    return f"{x:.{d}f}"


def aggregate(base):
    md = ["# TMA Stage 2（候補 C1）GB10 A/B 集計（イシュー #3215）", "",
          "規則の正は `RULE.txt`。本表は `aggregate.py` が収録ログから機械生成した（手編集しない）。", ""]
    problems_undetermined = []

    rcs = stage_rcs(base)
    n_gate, bad_gate = load_gate_status(base)
    md += ["## 専有ゲート・段の終了コード", "", f"- load_gate.log: {n_gate} 行、GATE-PASS 以外 {len(bad_gate)} 行"]
    for l in bad_gate:
        md.append(f"  - `{l}`")
    if bad_gate:
        problems_undetermined.append("専有ゲート不通過あり")
    bad_rc = {k: v for k, v in rcs.items() if v != "0"}
    md.append(f"- 終了コード非 0 の段: {len(bad_rc)} 件" + ("" if not bad_rc else "（" + ", ".join(f"{k}={v}" for k, v in sorted(bad_rc.items())) + "）"))
    missing_rc = [k for k in REQUIRED_STAGES if k not in rcs]
    bad_required = sorted(k for k in REQUIRED_STAGES if k in rcs and rcs[k] != "0")
    if missing_rc or bad_required:
        md.append("- **必須段の終了コード異常**: " + "、".join([f"{k}=欠落" for k in missing_rc] + [f"{k}={rcs[k]}" for k in bad_required]))
        problems_undetermined.append("必須段（build/ゲート A・B/純カーネル）の終了コードが 0 でない・欠落")
    md.append("")

    # ゲート A
    ga_path = os.path.join(base, "gateA_cpu_cuda_tiled_pipeline_tma_parity.log")
    a_critical_fail, a_prereq_ok, a_unavail = [], True, False
    md += ["## ゲート A（bit 一致・必須）", ""]
    if os.path.isfile(ga_path):
        st, a_critical_fail, a_prereq_ok = gate_a(read_text(ga_path))
        for n, s in st.items():
            md.append(f"- `{n}`: **{s}**")
    else:
        a_unavail = True
        md.append("- unavailable（ログ無し）")
        problems_undetermined.append("ゲート A のログ無し")
    md.append("")

    # ゲート B
    md += ["## ゲート B（parity 非後退・必須）", ""]
    b_paths = {b: os.path.join(base, f"gateB_{b}.log") for b in GATE_B_BINARIES}
    b_fails = []
    if all(os.path.isfile(p) for p in b_paths.values()):
        b_fails = gate_b({b: read_text(p) for b, p in b_paths.items()})
        md.append(f"- 6 バイナリの FAILED（既知の無関係を除く）: {len(b_fails)} 件")
        for f in b_fails:
            md.append(f"  - `{f}`")
    else:
        md.append("- unavailable（ログ不足）")
        problems_undetermined.append("ゲート B のログ不足")
    md.append("")

    # 純カーネル
    kern, kern_bits = {}, {}
    missing = [r for r in RUNS if not os.path.isfile(os.path.join(base, f"kern_run{r}.log"))]
    md += ["## ゲート C（純カーネル・主判定）", ""]
    verdicts = {}
    bit_ok, bit_problems = True, []
    med = None
    cg = None
    if missing:
        md.append(f"- unavailable（kern_run {missing} が無い）")
        problems_undetermined.append("純カーネルの run 不足")
    else:
        for r in RUNS:
            kern[r], kern_bits[r] = parse_kern(read_text(os.path.join(base, f"kern_run{r}.log")), r)
        bit_ok, bit_problems = bit_identity(kern_bits)
        md.append(f"- BITIDENT（5 run・全 N）: {'一致' if bit_ok else '**不一致**'}")
        for p in bit_problems:
            md.append(f"  - {p}")
        med, cg = gate_c(kern)
        md += ["", "### 純カーネル時間 kern_event（5 run 中央値, ms）と TFLOPS", "",
               "| N | cur (tile) | pipe128 | tma128_none | tma128_strided | cur TFLOPS | tma128_none TFLOPS | tma128_strided TFLOPS |",
               "|---|---|---|---|---|---|---|---|"]
        for n in SIZES:
            tile = kern[1][(n, "cur")]["tile"]
            tf = {a: 2 * n**3 / (med[(n, a)] / 1e3) / 1e12 for a in ARMS}
            md.append(f"| {n} | {fmt(med[(n, 'cur')], 5)} ({tile}) | {fmt(med[(n, 'pipe128')], 5)} | {fmt(med[(n, 'tma128_none')], 5)} | "
                      f"{fmt(med[(n, 'tma128_strided')], 5)} | {fmt(tf['cur'], 3)} | {fmt(tf['tma128_none'], 3)} | {fmt(tf['tma128_strided'], 3)} |")
        b2b = {(n, a): statistics.median(kern[r][(n, a)]["b2b"] for r in RUNS) for n in SIZES for a in ARMS}
        md += ["", "### kern_b2b（連続 10 launch の 1 回あたり, ms。記録のみ）", "",
               "| N | " + " | ".join(ARMS) + " |", "|---|" + "---|" * len(ARMS)]
        for n in SIZES:
            md.append(f"| {n} | " + " | ".join(fmt(b2b[(n, a)], 5) for a in ARMS) + " |")
        for cand in CAND_ARMS:
            rows = cg[cand]
            md += ["", f"### 比 cur ÷ {cand}（>1 が改善。結線範囲は N=1024/2048/4096）", "",
                   "| N | 比（中央値） | run 別比 | 反対側の run | 結線範囲 |", "|---|---|---|---|---|"]
            for n in SIZES:
                r = rows[n]
                cons = "" if r["opposite_runs"] < 2 else "（5/5 一貫ではない）"
                md.append(f"| {n} | {fmt(r['ratio'])} | {' '.join(fmt(x, 3) for x in r['run_ratios'])} | {r['opposite_runs']}{cons} | "
                          f"{'対象' if n in WIRING_SIZES else '記録のみ'} |")
            verdicts[cand] = verdict_for(rows)
        md += ["", "### 帰属欄（記録のみ）", "",
               "| N | pipe128 ÷ cur | tma128_none ÷ pipe128 の逆数（pipe128/tma） | tma128_strided（同） |", "|---|---|---|---|"]
        for n in SIZES:
            md.append(f"| {n} | {fmt(med[(n, 'cur')] / med[(n, 'pipe128')])} | "
                      f"{fmt(med[(n, 'pipe128')] / med[(n, 'tma128_none')])} | {fmt(med[(n, 'pipe128')] / med[(n, 'tma128_strided')])} |")
        md.append(f"\n- N=512 の cur ÷ tma128_none = {fmt(cg['tma128_none'][512]['ratio'])}、cur ÷ tma128_strided = "
                  f"{fmt(cg['tma128_strided'][512]['ratio'])}（design doc §4「N>=512」仮説の材料。結線範囲外のため判定に使わない）")
    md.append("")

    # cuBLAS
    md += ["## cuBLAS 比（record_only・event 対 event）", ""]
    try:
        cub = load_cublas(base)
    except FileNotFoundError as e:
        cub = None
        md.append(f"- unavailable（{os.path.relpath(str(e), base)} が無い）")
    if cub is not None:
        cm = {arm: {n: statistics.median(cub[arm][n].values()) for n in SIZES} for arm in cub}
        md += ["| N | cuBLAS FP32 mm_event (ms) | #3185 値 | 乖離 | cuBLAS TF32 (ms) | #3185 値 | 乖離 | cuBLAS FP32 TFLOPS |", "|---|---|---|---|---|---|---|---|"]
        for n in SIZES:
            d32 = cm["fp32"][n] / REF_3185_MS["fp32"][n] - 1
            dtf = cm["tf32"][n] / REF_3185_MS["tf32"][n] - 1
            flag = lambda d: " **(>10% ドリフト)**" if abs(d) > DRIFT else ""
            md.append(f"| {n} | {fmt(cm['fp32'][n], 5)} | {REF_3185_MS['fp32'][n]} | {d32:+.1%}{flag(d32)} | {fmt(cm['tf32'][n], 5)} | "
                      f"{REF_3185_MS['tf32'][n]} | {dtf:+.1%}{flag(dtf)} | {fmt(2 * n**3 / (cm['fp32'][n] / 1e3) / 1e12, 3)} |")
        if med is not None:
            md += ["", "| N | " + " | ".join(f"{a} ÷ cuBLAS FP32" for a in ARMS) + " | " + " | ".join(f"{a} ÷ cuBLAS TF32" for a in CAND_ARMS) + " |",
                   "|---|" + "---|" * (len(ARMS) + len(CAND_ARMS))]
            for n in SIZES:
                md.append(f"| {n} | " + " | ".join(fmt(med[(n, a)] / cm["fp32"][n], 3) for a in ARMS) + " | "
                          + " | ".join(fmt(med[(n, a)] / cm["tf32"][n], 3) for a in CAND_ARMS) + " |")
    md.append("")

    # セル
    md += ["## gemm CUDA セル（現行のみ・record_only）", ""]
    cell_path = os.path.join(base, "cell", "results.jsonl")
    cr = None
    if os.path.isfile(cell_path) and os.path.isfile(os.path.join(base, "cell", "manifest.json")):
        cr = cell_ratios(load_cell(cell_path))
        md += ["| N | fandhe reuse (ms, 中央値) | 最速相手 fresh (ms, 中央値) | 比（reuse, 5 round 中央値） | 0.11.0 reuse | 比（best） | 0.11.0 best |",
               "|---|---|---|---|---|---|---|"]
        for n in SIZES:
            c = cr[n]
            md.append(f"| {n} | {fmt(c['cur_reuse_ms'], 4)} | {fmt(c['opp_ms'], 4)} | {fmt(c['reuse'], 3)} | {REF_0110_REUSE[n]} | "
                      f"{fmt(c['best'], 3)} | {REF_0110_BEST[n]} |")
        if med is not None:
            md += ["", "窓の固定費（gemm セル reuse 中央値 − cur の kern_event 中央値, ms。実測の差）:", "",
                   "| N | セル reuse | kern_event(cur) | 差 |", "|---|---|---|---|"]
            for n in SIZES:
                md.append(f"| {n} | {fmt(cr[n]['cur_reuse_ms'], 4)} | {fmt(med[(n, 'cur')], 4)} | {fmt(cr[n]['cur_reuse_ms'] - med[(n, 'cur')], 4)} |")
    else:
        md.append("- unavailable（cell/results.jsonl または manifest.json が無い）。推定値・他機の値で埋めない。")
    md.append("")

    # 総合
    md += ["## 総合判定（RULE 項 9 の語彙。ゲート D は結線 PR で実施）", ""]
    overall = {}
    for cand in CAND_ARMS:
        if b_fails:
            v = "undetermined（ゲート B 要調査）"
        elif a_unavail or missing or not a_prereq_ok and not a_critical_fail:
            v = "undetermined"
        elif a_critical_fail or not bit_ok:
            v = "REJECT"
        elif problems_undetermined:
            v = "undetermined"
        else:
            base_v = verdicts[cand]
            v = "ADOPT（結線候補。ゲート D は結線 PR で実施）" if base_v == "ADOPT" else base_v
        overall[cand] = v
        md.append(f"- `{cand}`: **{v}**")
    if all(v == "REJECT" for v in overall.values()):
        md.append("- 候補 C1 全体: **REJECT**（両腕 REJECT）")
    else:
        md.append("- 候補 C1 全体: " + (" / ".join(f"{c}={v}" for c, v in overall.items())))
    if problems_undetermined:
        md += ["", "undetermined の要因: " + "、".join(problems_undetermined)]

    # ADOPT 腕のセル推定
    for cand in CAND_ARMS:
        if overall[cand].startswith("ADOPT") and cr is not None and med is not None:
            md += ["", f"### 推定（測定値ではない）: `{cand}` を結線した場合のセル（cur のセル −（cur − 候補）の kern_event 差）", "",
                   "| N | 推定セル reuse (ms) | 推定比（reuse） |", "|---|---|---|"]
            for n in SIZES:
                if n in WIRING_SIZES:
                    est = cr[n]["cur_reuse_ms"] - (med[(n, "cur")] - med[(n, cand)])
                    md.append(f"| {n} | {fmt(est, 4)}（推定） | {fmt(cr[n]['opp_ms'] / est, 3)}（推定） |")
    return "\n".join(md) + "\n", overall


# ---------- self-test ----------
def _kern_log(run, cur_scale, cand_scale, bad_hash=False, nan=False):
    lines = ["DEVICE3215 compute_capability=12.1"]
    for n in SIZES:
        base = 2 * n**3 / 20e12 * 1e3
        for a in ARMS:
            t = base * (1.0 if a in ("cur", "pipe128") else cand_scale) * (cur_scale if a == "cur" else 1.0)
            ev = "nan" if (nan and a == "cur" and n == 256) else f"{t:.6f}"
            lines.append(f"ARM3215 n={n} arm={a} kern_event_ms={ev} kern_b2b_ms={t:.6f} tflops=20.0000 tile=Bm128Bn64 samples=20")
        for a in ARMS:
            h = "0x0123456789abcdef"
            if bad_hash and a == "tma128_strided" and n == 1024:
                h = "0x0123456789abcde0"
            lines.append(f"BITIDENT3215 n={n} arm={a} fnv={h} elems={n * n}")
    lines.append("done3215")
    return "\n".join(lines) + "\n"


def _write_fixture(d, cand_scale=0.9, bad_hash=False, nan=False, gate_a_fail=False, gate_b_fail=False, with_cell=True,
                   bad_rc=None, zero_pass_b=False):
    os.makedirs(os.path.join(d, "cell"), exist_ok=True)
    for r in RUNS:
        open(os.path.join(d, f"kern_run{r}.log"), "w").write(_kern_log(r, 1.0, cand_scale, bad_hash, nan))
        for arm in ("fp32", "tf32"):
            os.makedirs(os.path.join(d, "cublas", f"run{r}"), exist_ok=True)
            with open(os.path.join(d, "cublas", f"run{r}", f"{arm}.jsonl"), "w") as f:
                for n in SIZES:
                    f.write(json.dumps({"arm": arm, "run": r, "phase": "mm_event", "n": n, "median_s": REF_3185_MS[arm][n] / 1e3}) + "\n")
    names = list(GATE_A_CRITICAL) + [GATE_A_PREREQ] + [f"filler_a_{i}" for i in range(GATE_A_EXPECTED_PASSED - 4)]
    with open(os.path.join(d, "gateA_cpu_cuda_tiled_pipeline_tma_parity.log"), "w") as f:
        for i, nm in enumerate(names):
            f.write(f"test {nm} ... {'FAILED' if (gate_a_fail and i == 0) else 'ok'}\n")
        nf = 1 if gate_a_fail else 0
        f.write(f"test result: {'FAILED' if nf else 'ok'}. {len(names) - nf} passed; {nf} failed; 0 ignored; 0 measured; 0 filtered out; finished in 1s\n")
    for b in GATE_B_BINARIES:
        with open(os.path.join(d, f"gateB_{b}.log"), "w") as f:
            if zero_pass_b and b == "gemm_tiled":
                f.write("test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0s\n")
                continue
            total = GATE_B_EXPECTED_PASSED[b]
            fails = 1 if (gate_b_fail and b == "gemm_tiled") else 0
            for i in range(total):
                f.write(f"test some_test_{b}_{i} ... {'FAILED' if i < fails else 'ok'}\n")
            f.write(f"test result: {'FAILED' if fails else 'ok'}. {total - fails} passed; {fails} failed; 0 ignored; 0 measured; 0 filtered out; finished in 1s\n")
    open(os.path.join(d, "load_gate.log"), "w").write("GATE-PASS x load1=0.1 gpu_util=0\n")
    with open(os.path.join(d, "stage_rc.txt"), "w") as f:
        for k in REQUIRED_STAGES:
            if bad_rc is not None and k == bad_rc[0] and bad_rc[1] is None:
                continue  # 欠落
            f.write(f"{k} rc={0 if bad_rc is None or k != bad_rc[0] else bad_rc[1]}\n")
    if with_cell:
        with open(os.path.join(d, "cell", "results.jsonl"), "w") as f:
            for rnd in RUNS:
                for fw, mode, n in expected_cell_sequence(rnd):
                    t = 2 * n**3 / 15e12
                    t = t * (0.9 if fw == "fandhe-ai" else 1.0)
                    f.write(json.dumps({"framework": fw, "mode": mode, "size": n, "task": "gemm", "device": "cuda", "median_s": t}) + "\n")
        open(os.path.join(d, "cell", "manifest.json"), "w").write("{}\n")


def self_test():
    def run(**kw):
        with tempfile.TemporaryDirectory() as d:
            _write_fixture(d, **kw)
            return aggregate(d)

    md, ov = run(cand_scale=0.9)
    assert all(v.startswith("ADOPT") for v in ov.values()), ov
    assert "推定（測定値ではない）" in md
    _, ov = run(cand_scale=0.98)
    assert all(v == "undetermined" for v in ov.values()), ov
    _, ov = run(cand_scale=1.2)
    assert all(v == "REJECT" for v in ov.values()), ov
    _, ov = run(cand_scale=0.9, gate_a_fail=True)
    assert all(v == "REJECT" for v in ov.values()), ov
    _, ov = run(cand_scale=0.9, bad_hash=True)
    assert all(v == "REJECT" for v in ov.values()), ov
    _, ov = run(cand_scale=0.9, gate_b_fail=True)
    assert all(v == "undetermined（ゲート B 要調査）" for v in ov.values()), ov
    _, ov = run(cand_scale=0.9, with_cell=False)
    assert all(v.startswith("ADOPT") for v in ov.values()), ov  # セル unavailable は判定に影響しない（record_only）
    # 実行完備の fail-closed（codex #3531 P1）
    for kw in ({"bad_rc": ("gateA", 1)}, {"bad_rc": ("gateB_gemm_tiled", None)}, {"bad_rc": ("kern_run3", "GATE-FAIL")}):
        _, ov = run(cand_scale=0.9, **kw)
        assert all(v == "undetermined" for v in ov.values()), (kw, ov)
    _, ov = run(cand_scale=0.9, bad_rc=("cublas_run1_fp32", 1))  # record_only 段は判定に影響しない（必須段でないので fixture には出ない）
    # 拒否ケース
    for kw, label in (({"nan": True}, "NaN"), ({"zero_pass_b": True}, "0 passed のゲート B")):
        try:
            run(**kw)
        except AggError:
            pass
        else:
            raise AssertionError(f"{label} を拒否できていない")
    with tempfile.TemporaryDirectory() as d:  # 重複行
        _write_fixture(d)
        p = os.path.join(d, "kern_run3.log")
        txt = read_text(p)
        first = [l for l in txt.splitlines() if l.startswith("ARM3215")][0]
        open(p, "w").write(txt.replace("done3215", first + "\ndone3215"))
        try:
            aggregate(d)
        except AggError:
            pass
        else:
            raise AssertionError("重複行を拒否できていない")
    with tempfile.TemporaryDirectory() as d:  # 完了行欠落
        _write_fixture(d)
        p = os.path.join(d, "kern_run2.log")
        open(p, "w").write(read_text(p).replace("done3215\n", ""))
        try:
            aggregate(d)
        except AggError:
            pass
        else:
            raise AssertionError("完了行欠落を拒否できていない")
    with tempfile.TemporaryDirectory() as d:  # run 欠落 → undetermined
        _write_fixture(d)
        os.remove(os.path.join(d, "kern_run5.log"))
        _, ov = aggregate(d)
        assert all(v == "undetermined" for v in ov.values()), ov
    with tempfile.TemporaryDirectory() as d:  # セル順序パターン不一致
        _write_fixture(d)
        p = os.path.join(d, "cell", "results.jsonl")
        lines = read_text(p).splitlines()
        lines[0], lines[1] = lines[1], lines[0]
        open(p, "w").write("\n".join(lines) + "\n")
        try:
            aggregate(d)
        except AggError:
            pass
        else:
            raise AssertionError("セル起動順の不一致を拒否できていない")
    print("self-test OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not args.dir or not args.out:
        ap.error("--dir と --out が必要")
    md, _ = aggregate(args.dir)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
