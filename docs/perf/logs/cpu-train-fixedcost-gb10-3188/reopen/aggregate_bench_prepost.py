#!/usr/bin/env python3
"""イシュー #3188（再オープン分）: bench_prepost.sh の出力を集計する（標準ライブラリのみ・参考値、判定ゲートなし）。

使い方: aggregate_bench_prepost.py <out_dir>
        （<out_dir>/{pre,head}/round{1..5}/<cell>.jsonl と load_gate_status.txt を読む）
欠落セル・不正 JSON・round 数不一致・phase 集合の不一致は fail-closed（exit 1）。
checksum は pre／head の一致を報告する（不一致でも値はそのまま表に出す）。
"""
import json
import math
import os
import statistics
import sys

LABELS = ["pre", "head"]
ROUNDS = 5
CELLS = [
    "train_reuse_total", "train_reuse_phases",
    "infer_reuse_total", "infer_reuse_phases",
    "infer_fresh_total", "infer_fresh_phases",
]
# セルごとの期待 phase 集合（固定。入力同士の一致だけでは全ファイルから欠落しても通るため。fail-closed）
EXPECTED_PHASES = {
    "train_reuse_phases": ("tape_build", "leaf_register", "forward_resident", "loss_readout",
                           "backward", "device_update", "tape_drop", "step_total"),
    "infer_reuse_phases": ("predict_resident", "host_copy", "checksum", "iter_total"),
    "infer_fresh_phases": ("predict", "host_copy", "checksum", "iter_total"),
}


class AggError(Exception):
    pass


def load_cell(path, phases):
    rows = []
    try:
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    m = float(r["median_s"])
                    ck = float(r["checksum"])
                except (ValueError, KeyError, TypeError) as e:
                    raise AggError(f"{path}:{i}: 不正な行 ({e})")
                if not math.isfinite(m) or m <= 0.0:
                    raise AggError(f"{path}:{i}: median_s が有限の正値でない")
                rows.append((r.get("phase", "total"), m, ck))
    except OSError as e:
        raise AggError(f"{path}: 読み込み不可 ({e})")
    if not rows:
        raise AggError(f"{path}: 行なし")
    names = [n for n, _, _ in rows]
    if len(set(names)) != len(names):
        raise AggError(f"{path}: phase 重複")
    if not phases and names != ["total"]:
        raise AggError(f"{path}: total セルが 1 行でない ({names})")
    if phases and "total" in names:
        raise AggError(f"{path}: phases セルに phase 列がない")
    return {n: (m, ck) for n, m, ck in rows}


def load_all(d):
    data = {}
    for label in LABELS:
        for r in range(1, ROUNDS + 1):
            for c in CELLS:
                data[(label, r, c)] = load_cell(
                    os.path.join(d, label, f"round{r}", f"{c}.jsonl"), c.endswith("_phases"))
    for c in CELLS:  # phase 集合は全 round・全ラベルで一致
        sets = {tuple(data[(l, r, c)]) for l in LABELS for r in range(1, ROUNDS + 1)}
        if len(sets) != 1:
            raise AggError(f"{c}: phase 集合が round/ラベル間で不一致 {sorted(sets)}")
        exp = EXPECTED_PHASES.get(c)
        if exp is not None and set(next(iter(sets))) != set(exp):
            raise AggError(f"{c}: phase 集合が期待値と不一致 {sorted(next(iter(sets)))} != {sorted(exp)}")
    gate = {}
    path = os.path.join(d, "load_gate_status.txt")
    try:
        with open(path, encoding="utf-8") as f:
            for ln in f:
                p = ln.split()
                if not p:
                    continue
                if len(p) != 2 or not p[1].startswith("gate=") or p[1][5:] not in ("pass", "unpassed") or p[0] in gate:
                    raise AggError(f"{path}: 不正な行 {ln!r}")
                gate[p[0]] = p[1][5:]
    except OSError as e:
        raise AggError(f"{path}: 読み込み不可 ({e})")
    if set(gate) != {f"round{r}" for r in range(1, ROUNDS + 1)}:
        raise AggError(f"{path}: round1..round{ROUNDS} の全行が必要")
    return data, gate


def us(x):
    return f"{x * 1e6:.1f}"


def aggregate(data, gate):
    out = []
    un = sorted(k for k, v in gate.items() if v == "unpassed")
    if un:
        out.append(f"【参考扱い】負荷ゲート未通過の round: {', '.join(un)}")
        out.append("")
    out.append("round 順序: 奇数 round は pre→head、偶数 round は head→pre。値は各 round の median_s の 5 round 中央値。")
    out.append("")
    out.append("| セル | phase | pre median(us) | pre min-max | head median(us) | head min-max | head/pre | checksum pre/head |")
    out.append("|---|---|---|---|---|---|---|---|")
    totals = {}
    for c in CELLS:
        for ph in data[("pre", 1, c)]:
            cols = {}
            for l in LABELS:
                vals = [data[(l, r, c)][ph][0] for r in range(1, ROUNDS + 1)]
                cols[l] = (statistics.median(vals), min(vals), max(vals))
            cks = {l: {data[(l, r, c)][ph][1] for r in range(1, ROUNDS + 1)} for l in LABELS}
            if cks["pre"] == cks["head"] and len(cks["pre"]) == 1:
                ck = "一致"
            else:
                ck = f"不一致 pre={sorted(cks['pre'])} head={sorted(cks['head'])}"
            pm, hm = cols["pre"][0], cols["head"][0]
            out.append(f"| {c} | {ph} | {us(pm)} | {us(cols['pre'][1])}-{us(cols['pre'][2])} | {us(hm)} | "
                       f"{us(cols['head'][1])}-{us(cols['head'][2])} | {hm / pm:.3f} | {ck} |")
            if ph == "total":
                totals[(c, "pre")], totals[(c, "head")] = pm, hm
    out.append("")
    out.append("infer の reuse/fresh 比（total セル。1 未満なら reuse の方が速い）:")
    for l in LABELS:
        r, f = totals[("infer_reuse_total", l)], totals[("infer_fresh_total", l)]
        out.append(f"- {l}: reuse {us(r)} us / fresh {us(f)} us = {r / f:.3f}")
    return "\n".join(out)


def main(argv):
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    try:
        print(aggregate(*load_all(argv[1])))
    except AggError as e:
        print(f"集計エラー（fail-closed）: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
