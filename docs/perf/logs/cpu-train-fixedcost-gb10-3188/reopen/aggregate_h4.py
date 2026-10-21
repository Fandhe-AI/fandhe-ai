#!/usr/bin/env python3
"""イシュー #3188（再オープン分）: H4 分解診断の独立 5 run を RULE.txt §3 の規則で集計する（標準ライブラリのみ）。

使い方: aggregate_h4.py <multi_dir> [<single_dir>]
        （各 dir は run1.jsonl..run5.jsonl と load_gate_status.txt を持つ）
欠落セル・不正 JSON・n 不一致・run 数不一致・checksum 不一致は fail-closed（exit 1）。
判定閾値は RULE.txt（実測前に固定）が正で、本スクリプトは変更しない。
自己テストは aggregate_h4_test.py（python3 -I で実行）。
"""
import json
import math
import os
import statistics
import sys

RUNS = 5
EXPECTED = {
    "insitu_direct": ["device_update"],
    "insitu_probe": ["prologue_probe", "after_probe"],
    "standalone": ["stage", "sgd_kernel", "sgd_kernel_cold", "zip_hot", "zip_xcore"],
}
CHECKSUM_INSITU = [("insitu_direct", "device_update"), ("insitu_probe", "after_probe")]
CHECKSUM_STANDALONE = [
    ("standalone", p) for p in ("sgd_kernel", "sgd_kernel_cold", "zip_hot", "zip_xcore")
]
EXPECTED_N = 80  # 計測 20 反復 x 4 ラウンド（テスト側 ROUNDS x ITERS_PER_ROUND）
GATE_FILE = "load_gate_status.txt"
# RULE.txt §3.3（事後に変えない）
ADDITIVITY_TOL = 0.15
SUPPORT_SHARE = 0.5
STATE_TOL = 0.25
WRITER_RATIO = 0.5


class AggError(Exception):
    pass


def load_run(path):
    cells = {}
    try:
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    key = (rec["arm"], rec["phase"])
                    rec["median_s"] = float(rec["median_s"])
                    rec["q1_s"] = float(rec["q1_s"])
                    rec["q3_s"] = float(rec["q3_s"])
                    n = rec["n"]
                except (ValueError, KeyError, TypeError) as e:
                    raise AggError(f"{path}:{i}: 不正な行 ({e})")
                if key[1] not in EXPECTED.get(key[0], []):
                    raise AggError(f"{path}:{i}: 想定外のセル {key}")
                if not math.isfinite(rec["median_s"]) or rec["median_s"] <= 0.0:
                    raise AggError(f"{path}:{i}: median_s が有限の正値でない {key}")
                if isinstance(n, bool) or n != EXPECTED_N:
                    raise AggError(f"{path}:{i}: 計測回数 n={n!r} が {EXPECTED_N} でない {key}")
                if key in cells:
                    raise AggError(f"{path}:{i}: セル重複 {key}")
                cells[key] = rec
    except OSError as e:
        raise AggError(f"{path}: 読み込み不可 ({e})")
    for arm, phases in EXPECTED.items():
        for ph in phases:
            if (arm, ph) not in cells:
                raise AggError(f"{path}: セル欠落 {arm}/{ph}")
    return cells


def load_gate(d):
    """load_gate_status.txt を読み、未通過 run 名の一覧を返す（欠落・不正は fail-closed）。"""
    path = os.path.join(d, GATE_FILE)
    try:
        with open(path, encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
    except OSError as e:
        raise AggError(f"{path}: 読み込み不可 ({e})")
    seen = {}
    for ln in lines:
        parts = ln.split()
        if len(parts) != 2 or not parts[1].startswith("gate="):
            raise AggError(f"{path}: 不正な行 {ln!r}")
        status = parts[1][5:]
        if status not in ("pass", "unpassed") or parts[0] in seen:
            raise AggError(f"{path}: 不正または重複した行 {ln!r}")
        seen[parts[0]] = status
    if set(seen) != {f"run{n}" for n in range(1, RUNS + 1)}:
        raise AggError(f"{path}: run1..run{RUNS} の全行が必要 ({sorted(seen)})")
    return sorted(r for r, st in seen.items() if st == "unpassed")


def load_series(d):
    runs = [load_run(os.path.join(d, f"run{n}.jsonl")) for n in range(1, RUNS + 1)]
    check_checksums(runs)
    return runs, load_gate(d)


def check_checksums(runs):
    """run 間・腕間の checksum 不一致と欠落を fail-closed で検出する。"""
    if len(runs) != RUNS:
        raise AggError(f"run 数が {RUNS} でない ({len(runs)})")
    for group in (CHECKSUM_INSITU, CHECKSUM_STANDALONE):
        seen = set()
        for key in group:
            for r in runs:
                v = r[key].get("checksum_bits")
                if not v:
                    raise AggError(f"checksum_bits 欠落: {key}")
                seen.add(v)
        if len(seen) != 1:
            raise AggError(f"checksum が run 間または腕間で不一致: {[k[1] for k in group]} {sorted(seen)}")


def med(runs, arm, ph):
    vals = [r[(arm, ph)]["median_s"] for r in runs]
    return statistics.median(vals), min(vals), max(vals)


def us(x):
    return f"{x * 1e6:.1f}"


def values(runs):
    m = lambda a, p: med(runs, a, p)[0]
    return {
        "T": m("insitu_direct", "device_update"),
        "P": m("insitu_probe", "prologue_probe"),
        "R": m("insitu_probe", "after_probe"),
        "K": m("standalone", "sgd_kernel"),
        "S": m("standalone", "stage"),
        "Kc": m("standalone", "sgd_kernel_cold"),
        "Zh": m("standalone", "zip_hot"),
        "Zx": m("standalone", "zip_xcore"),
    }


def judge(v):
    """RULE.txt §3.3 規則 1〜4 を機械的に適用する。(行のリスト, 辞書) を返す。"""
    out = []
    T, P, R, K, S = v["T"], v["P"], v["R"], v["K"], v["S"]
    res = {}
    add_err = abs(P + R - T)
    additive = add_err <= ADDITIVITY_TOL * T
    out.append(f"規則 1 加算性: |P+R-T| = {us(add_err)} us（許容 {us(ADDITIVITY_TOL * T)} us）-> "
               + ("成立" if additive else "不成立（分解は参考扱い・判定は未確定）"))
    h4p = T - (K + S)
    out.append(f"規則 2 H4' = T-(K+S) = {us(h4p)} us（T の {100 * h4p / T:.1f}%）")
    h4a, h4b = P, R - (K + S)
    res.update(additive=additive, h4p=h4p, h4a=h4a, h4b=h4b)
    if not additive or h4p <= 0:
        verdict_a = verdict_b = "未確定"
    else:
        verdict_a = "支持" if h4a >= SUPPORT_SHARE * h4p else "未確定"
        verdict_b = "支持" if h4b >= SUPPORT_SHARE * h4p else "未確定"
    out.append(f"規則 3 H4a(prologue)=P = {us(h4a)} us（H4' の {100 * h4a / h4p:.1f}%）-> {verdict_a}"
               if h4p > 0 else "規則 3 H4' が非正のため判定不能 -> 未確定")
    if h4p > 0:
        out.append(f"        H4b(in-situ カーネル超過)=R-(K+S) = {us(h4b)} us（H4' の {100 * h4b / h4p:.1f}%）-> {verdict_b}")
    res.update(verdict_a=verdict_a, verdict_b=verdict_b)
    if verdict_b == "支持":
        x = R - S
        cands = {"cold(追い出し後)": v["Kc"], "xcore(他コア書き込み後)": K + (v["Zx"] - v["Zh"])}
        near = {k: c for k, c in cands.items() if x > 0 and abs(c - x) <= STATE_TOL * x}
        for k, c in cands.items():
            out.append(f"規則 4 候補 {k} = {us(c)} us（x=R-S={us(x)} us との差 {100 * (c - x) / x:+.1f}%）")
        if near:
            best = min(near, key=lambda k: abs(near[k] - x))
            out.append(f"        -> 整合: {best}")
            res["state"] = best
        else:
            out.append("        -> 未確定（±25% 以内の候補なし）")
            res["state"] = "未確定"
    else:
        out.append("規則 4 H4b が支持されないため状態の説明は付けない")
        res["state"] = "対象外"
    return out, res


def aggregate(multi, single=None, unpassed_multi=(), unpassed_single=()):
    lines = []
    for name, runs, un in (("multi", multi, unpassed_multi), ("single", single, unpassed_single)):
        if runs is None:
            continue
        lines.append(f"## 系列 {name}")
        if un:
            lines.append(f"【参考扱い】負荷ゲート未通過の run: {', '.join(un)}（RULE.txt §3.2。通常判定として扱わない）")
        lines.append("")
        lines.append("| arm | phase | median(us) | min-max(us) | q1/q3 median(us) | run1..5 median(us) |")
        lines.append("|---|---|---|---|---|---|")
        for arm, phases in EXPECTED.items():
            for ph in phases:
                m, lo, hi = med(runs, arm, ph)
                q1 = statistics.median([r[(arm, ph)]["q1_s"] for r in runs])
                q3 = statistics.median([r[(arm, ph)]["q3_s"] for r in runs])
                per = " / ".join(us(r[(arm, ph)]["median_s"]) for r in runs)
                lines.append(f"| {arm} | {ph} | {us(m)} | {us(lo)}-{us(hi)} | {us(q1)}/{us(q3)} | {per} |")
        lines.append("")
        lines.append("checksum: run 間・腕間一致 OK")
        # 補助情報（判定に使わない）: minflt と core_moves の run 合計
        aux = []
        for arm, phases in EXPECTED.items():
            for ph in phases:
                mf = [r[(arm, ph)].get("minflt_total") for r in runs]
                cm = [r[(arm, ph)].get("core_moves") for r in runs]
                if all(x is not None for x in mf):
                    aux.append(f"| {arm}/{ph} | {sum(mf)} | {sum(cm)} |")
        if aux:
            lines += ["", "補助（判定に使わない）: 計時区間中の minflt 合計と main スレッドのコア移動回数（5 run 合計）", "",
                      "| セル | minflt | core_moves |", "|---|---|---|"] + aux
        lines.append("")
        jl, res = judge(values(runs))
        lines += [f"### 判定（{name}）"] + jl + [""]
        if name == "multi":
            multi_res = res
    if single is not None:
        vm, vs = values(multi), values(single)
        a = vm["T"] - vm["K"]
        b = vs["T"] - vs["K"]
        lines.append("## 規則 5 書き手コアの判定")
        if unpassed_single or unpassed_multi:
            lines.append("multi／single のいずれかの系列に負荷ゲート未通過 run があるため 判定なし（RULE §2・§5）")
        else:
            ok = b <= WRITER_RATIO * a
            lines.append(f"(T_single-K_single) = {us(b)} us、(T_multi-K_multi) = {us(a)} us、比 {b / a:.2f}"
                         if a > 0 else f"(T_multi-K_multi) = {us(a)} us が非正のため判定なし")
            if a > 0:
                lines.append("-> " + ("マルチコア由来の状態を支持" if ok else "非支持"))
    return "\n".join(lines)


def main(argv):
    if len(argv) not in (2, 3):
        print(__doc__, file=sys.stderr)
        return 2
    try:
        multi, um = load_series(argv[1])
        single, us_ = load_series(argv[2]) if len(argv) == 3 else (None, ())
        print(aggregate(multi, single, um, us_))
    except AggError as e:
        print(f"集計エラー（fail-closed）: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
