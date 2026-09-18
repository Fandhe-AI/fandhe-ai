#!/usr/bin/env python3
"""イシュー #1980／#1981: `--phases` 5 run（1 run = 1 JSONL）の中央値フェーズ表を作る。

各 run の各 (device, task, mode, phase) について JSONL の `median_s`（トップレベルキー。RULE.txt の「stats.median_s」は同じ値を指す表記ゆれ）
（プロセス内 80 iters の中央値）を取り、5 run 間の中央値・最小・最大と、
合計フェーズ（train は `step_total`・infer は `iter_total`）に対する比を
出力する。残差トップ 3 は合計フェーズを除いた比の降順上位 3 件。
python3 標準ライブラリのみ。使い方: aggregate.py <dir> [--devices cpu,metal] [--machine "Apple M4 Max"] > aggregate.md
（`--devices` 既定は Mac 分の cpu,metal。GB10 分は `--devices cpu,cuda --machine "DGX Spark GB10"`。
既定引数での出力は #2021 収録の m4max/aggregate.md と byte 同一）
"""
import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

TOTAL = {"train_phases": "step_total", "infer_phases": "iter_total"}
# README・RULE.txt が契約する 8 セル（cpu／metal × train／infer × fresh／reuse）。
# orchestrate は個々の実行失敗を記録して続行するため、観測された行だけから
# セル集合を作ると全 run で欠落したセル（例: metal 全滅）を静かに除いた表が
# 出てしまう。期待集合を明示し、run ごとの欠落を fail-closed に検出する。
def expected_cells(devices):
    return sorted(
        (dev, task, mode)
        for dev in devices
        for task in ("train_phases", "infer_phases")
        for mode in ("fresh", "reuse")
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--devices", default="cpu,metal", help="期待するデバイス集合（カンマ区切り。既定 cpu,metal）")
    ap.add_argument("--machine", default="Apple M4 Max", help="見出しに書く実機名")
    args = ap.parse_args()
    devices = [x for x in args.devices.split(",") if x]
    if not devices:
        print("error: --devices が空", file=sys.stderr)
        return 1
    EXPECTED_CELLS = expected_cells(devices)
    d = Path(args.dir)
    files = [d / f"run{i}.jsonl" for i in range(1, 6)]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        print(f"error: 5 run 揃っていない（不足: {missing}）", file=sys.stderr)
        return 1
    vals = defaultdict(list)  # (device, task, mode) -> phase -> [median_s per run]
    order = {}
    phase_sets = {}  # run 名 -> {cell -> frozenset(phase)}。run 間のフェーズ集合一致を検査する
    for f in files:
        seen = set()
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("task") not in TOTAL:
                continue
            key = (r["device"], r["task"], r["mode"], r["phase"])
            if key in seen:
                print(f"error: {f.name} に重複行 {key}", file=sys.stderr)
                return 1
            seen.add(key)
            vals[key].append(r["median_s"])
            order[key] = r["phase_index"]
        per_cell = defaultdict(set)
        for k in seen:
            per_cell[k[:3]].add(k[3])
        missing_cells = [c for c in EXPECTED_CELLS if c not in per_cell]
        if missing_cells:
            print(f"error: {f.name} に欠落セル {missing_cells}（期待 {len(EXPECTED_CELLS)} セル）", file=sys.stderr)
            return 1
        extra_cells = sorted(c for c in per_cell if c not in EXPECTED_CELLS)
        if extra_cells:
            print(f"error: {f.name} に期待外のセル {extra_cells}", file=sys.stderr)
            return 1
        for c, ph in per_cell.items():
            if TOTAL[c[1]] not in ph:
                print(f"error: {f.name} の {c} に合計フェーズ {TOTAL[c[1]]} がない", file=sys.stderr)
                return 1
        phase_sets[f.name] = {c: frozenset(ph) for c, ph in per_cell.items()}
    ref_name = files[0].name
    for name, ps in phase_sets.items():
        for c in EXPECTED_CELLS:
            if ps[c] != phase_sets[ref_name][c]:
                diff = sorted(ps[c] ^ phase_sets[ref_name][c])
                print(f"error: {name} と {ref_name} で {c} のフェーズ集合が不一致 {diff}", file=sys.stderr)
                return 1
    cells = EXPECTED_CELLS
    print(f"# train／infer `--phases` 5 run 中央値（registry `fandhe-ai =0.9.0`・{args.machine}）\n")
    for cell in cells:
        dev, task, mode = cell
        phases = sorted((k for k in vals if k[:3] == cell), key=lambda k: order[k])
        bad = [k[3] for k in phases if len(vals[k]) != 5]
        if bad:
            print(f"error: {cell} の {bad} が 5 run 揃っていない", file=sys.stderr)
            return 1
        total_key = cell + (TOTAL[task],)
        total = statistics.median(vals[total_key])
        print(f"## {task} / {dev} / {mode}\n")
        print("| phase | 中央値 (µs) | min–max (µs) | 合計比 |")
        print("|---|---:|---|---:|")
        ranked = []
        for k in phases:
            med = statistics.median(vals[k])
            ratio = med / total
            print(f"| {k[3]} | {med*1e6:.1f} | {min(vals[k])*1e6:.1f}–{max(vals[k])*1e6:.1f} | {ratio*100:.1f}% |")
            if k[3] != TOTAL[task]:
                ranked.append((ratio, k[3], med))
        ranked.sort(reverse=True)
        top = "・".join(f"{name}（{ratio*100:.1f}%・{med*1e6:.1f} µs）" for ratio, name, med in ranked[:3])
        covered = sum(m for _, _, m in ranked)
        print(f"\n- トップ 3: {top}")
        print(f"- フェーズ和／合計: {covered/total*100:.1f}%（差分は各フェーズ中央値の非加法性を含み、固定費は未測定）\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
