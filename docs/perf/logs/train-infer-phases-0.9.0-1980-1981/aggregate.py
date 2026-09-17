#!/usr/bin/env python3
"""イシュー #1980／#1981: `--phases` 5 run（1 run = 1 JSONL）の中央値フェーズ表を作る。

各 run の各 (device, task, mode, phase) について JSONL の `median_s`（トップレベルキー。RULE.txt の「stats.median_s」は同じ値を指す表記ゆれ）
（プロセス内 80 iters の中央値）を取り、5 run 間の中央値・最小・最大と、
合計フェーズ（train は `step_total`・infer は `iter_total`）に対する比を
出力する。残差トップ 3 は合計フェーズを除いた比の降順上位 3 件。
python3 標準ライブラリのみ。使い方: aggregate.py <dir> > aggregate.md
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

TOTAL = {"train_phases": "step_total", "infer_phases": "iter_total"}


def main() -> int:
    d = Path(sys.argv[1])
    files = [d / f"run{i}.jsonl" for i in range(1, 6)]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        print(f"error: 5 run 揃っていない（不足: {missing}）", file=sys.stderr)
        return 1
    vals = defaultdict(list)  # (device, task, mode) -> phase -> [median_s per run]
    order = {}
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
    cells = sorted({k[:3] for k in vals})
    print("# train／infer `--phases` 5 run 中央値（registry `fandhe-ai =0.9.0`・Apple M4 Max）\n")
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
        print(f"- フェーズ和／合計: {covered/total*100:.1f}%（差分は計測区間外の固定費）\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
