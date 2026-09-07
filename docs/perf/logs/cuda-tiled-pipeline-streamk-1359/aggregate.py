#!/usr/bin/env python3
"""イシュー #1359 ゲート C・D 集計スクリプト（Python3 標準ライブラリのみ）。

`gemm_tiled_pipeline_persistent_bench --sizes 1024,2048,4096 --tile both
--blocks-per-sm auto --streamk on` の 5 回独立プロセス起動ログ
（gateC_streamk_auto_run{1..5}.log）を入力に取り、以下を計算する:

- ゲート C: 64x64 タイルの streamk_over_pipeline3（streamk_gpu_only_tflops /
  同一 run・同一 size の pipeline3_gpu_only_tflops）の 5 回中央値。
- ゲート D: 同一 run・同一 size の 64x64.streamk_gpu_only_tflops /
  128x64.pipeline3_gpu_only_tflops（結線候補である 128×64 pipeline 経路との
  比較。ゲート C の streamk_over_pipeline3 列はカーネル自身の 64x64 pipeline
  非 streamk 版との比であり分母が異なるため別途計算する）の 5 回中央値。

判定基準（docs/perf/cuda-gemm-tiled-pipeline-streamk.md §6.1。実測前に事前
宣言済みで本スクリプトでは変更しない）:
  - ゲート C: N=1024 で streamk_over_pipeline3 中央値 >= 1.05、
    N=2048 で >= 1.00（かつ 0.95 未満の形状なし）。
  - ゲート D: 64x64.streamk_gpu_only_tflops / 128x64.pipeline3_gpu_only_tflops
    の中央値が N=1024・N=2048 とも >= 1.00。
"""
import re
import statistics
import sys

LINE_RE = re.compile(
    r"size=(\d+) tile=(\S+) (\S+)"
)


def parse(path):
    """1 run のログから size×tile×metric -> 値 の辞書を作る。"""
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("size="):
                continue
            parts = line.split()
            size = int(parts[0].split("=")[1])
            tile = parts[1].split("=")[1]
            metrics = {}
            for p in parts[2:]:
                if "=" not in p:
                    break
                k, v = p.split("=", 1)
                if v == "n/a":
                    continue
                try:
                    metrics[k] = float(v)
                except ValueError:
                    pass
            # 同一 (size, tile) に対し pipeline3/persistent 行と streamk 行の
            # 2 行が出力されるため、既存エントリへマージする（上書きしない）。
            out.setdefault((size, tile), {}).update(metrics)
    return out


def main():
    if len(sys.argv) < 2:
        print("usage: aggregate.py <run1.log> [run2.log ...]", file=sys.stderr)
        sys.exit(1)

    runs = [parse(p) for p in sys.argv[1:]]
    n_runs = len(runs)
    sizes = [1024, 2048, 4096]

    # ゲート C: 64x64 の streamk_over_pipeline3（ログ自身の出力値をそのまま使う）
    gate_c = {}
    for size in sizes:
        vals = []
        for r in runs:
            m = r.get((size, "64x64"), {})
            if "streamk_over_pipeline3" in m:
                vals.append(m["streamk_over_pipeline3"])
        gate_c[size] = vals

    # ゲート D: 64x64.streamk_gpu_only_tflops / 128x64.pipeline3_gpu_only_tflops
    gate_d = {}
    for size in sizes:
        vals = []
        for r in runs:
            m64 = r.get((size, "64x64"), {})
            m128 = r.get((size, "128x64"), {})
            if "streamk_gpu_only_tflops" in m64 and "pipeline3_gpu_only_tflops" in m128:
                vals.append(m64["streamk_gpu_only_tflops"] / m128["pipeline3_gpu_only_tflops"])
        gate_d[size] = vals

    # 参考: 各 TFLOPS 列の中央値
    ref_cols = ["pipeline3_gpu_only_tflops", "persistent_gpu_only_tflops", "streamk_gpu_only_tflops"]
    ref = {}
    for size in sizes:
        for tile in ("64x64", "128x64"):
            for col in ref_cols:
                vals = [r.get((size, tile), {}).get(col) for r in runs]
                vals = [v for v in vals if v is not None]
                if vals:
                    ref[(size, tile, col)] = vals

    print(f"# n_runs={n_runs}")
    print()
    print("## ゲート C（64x64 streamk_over_pipeline3。5 回中央値）")
    print()
    print("| N | 値（5 run） | 中央値 | 判定基準 | 結果 |")
    print("|---|---|---|---|---|")
    thresholds_c = {1024: 1.05, 2048: 1.00, 4096: None}
    for size in sizes:
        vals = gate_c[size]
        if len(vals) != n_runs:
            print(f"| {size} | ERROR: {len(vals)}/{n_runs} samples | - | - | INCOMPLETE |")
            continue
        med = statistics.median(vals)
        th = thresholds_c[size]
        if th is None:
            verdict = "参考（判定に使わない）"
        else:
            verdict = "PASS" if med >= th else "FAIL"
        vals_str = ", ".join(f"{v:.4f}" for v in vals)
        th_str = f">= {th}" if th is not None else "参考"
        print(f"| {size} | {vals_str} | {med:.4f} | {th_str} | {verdict} |")

    print()
    print("## ゲート D（64x64.streamk_gpu_only_tflops / 128x64.pipeline3_gpu_only_tflops。5 回中央値）")
    print()
    print("| N | 値（5 run） | 中央値 | 判定基準 | 結果 |")
    print("|---|---|---|---|---|")
    thresholds_d = {1024: 1.00, 2048: 1.00, 4096: None}
    for size in sizes:
        vals = gate_d[size]
        if len(vals) != n_runs:
            print(f"| {size} | ERROR: {len(vals)}/{n_runs} samples | - | - | INCOMPLETE |")
            continue
        med = statistics.median(vals)
        th = thresholds_d[size]
        if th is None:
            verdict = "参考（判定に使わない）"
        else:
            verdict = "PASS" if med >= th else "FAIL"
        vals_str = ", ".join(f"{v:.4f}" for v in vals)
        th_str = f">= {th}" if th is not None else "参考"
        print(f"| {size} | {vals_str} | {med:.4f} | {th_str} | {verdict} |")

    print()
    print("## 参考: 各 TFLOPS 列の中央値")
    print()
    print("| N | tile | 列 | 中央値 |")
    print("|---|---|---|---|")
    for size in sizes:
        for tile in ("64x64", "128x64"):
            for col in ref_cols:
                key = (size, tile, col)
                if key in ref:
                    med = statistics.median(ref[key])
                    print(f"| {size} | {tile} | {col} | {med:.4f} |")


if __name__ == "__main__":
    main()
