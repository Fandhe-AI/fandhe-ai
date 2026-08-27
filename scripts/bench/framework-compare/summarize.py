#!/usr/bin/env python3
"""JSONL 計測結果 → Markdown 表の集計ツール。

使い方:
    python3 summarize.py [JSONL ...] [--out FILE]

- 入力: results/raw/ の JSONL（省略時は results/raw/*.jsonl を全件）。
  環境ごとのファイル（例: results.jsonl = Apple M4 Max / macOS、
  results-dgx.jsonl = DGX Spark）をそれぞれ独立のセクションとして表化する。
  表化するデバイス列（cpu / metal / cuda）は各ファイルに実在する行から導出する
  （macOS 前提の固定デバイス集合をハードコードしない）。
- 出力: 既定は標準出力。`--out FILE` を明示した場合のみファイルへ書き込む。
  コミット済みの results/summary.md（複数環境を統合した一次データ。環境情報・
  備考は人間が追記済み）を既定動作で上書きしない。
- 環境情報（チップ・OS 等）は入力 JSONL からは分からないため出力に含めない
  （リモート環境の JSONL をローカルのホスト情報でラベル付けしない）。
  環境の正は results/summary.md・results/versions.txt・run_all*.log。
"""

import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

FRAMEWORKS = ["fandhe-ai", "candle", "burn"]
DEVICE_ORDER = ["cpu", "metal", "cuda"]
DEVICE_LABEL = {"cpu": "CPU", "metal": "Metal", "cuda": "CUDA"}


def fmt_ms(s):
    if s >= 1.0:
        return f"{s:.3f} s"
    if s >= 1e-3:
        return f"{s * 1e3:.3f} ms"
    return f"{s * 1e6:.1f} µs"


def load_rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def get(rows, fw, task, device, size=None):
    for r in rows:
        if r["framework"] == fw and r["task"] == task and r["device"] == device:
            if size is None or r["size"] == size:
                return r
    return None


def devices_in(rows, task):
    present = {r["device"] for r in rows if r["task"] == task}
    return [d for d in DEVICE_ORDER if d in present]


def section(path, rows):
    lines = []
    rel = os.path.relpath(path, HERE)
    lines.append(f"## 集計対象: {rel}\n")

    versions = {r["framework"]: r["version"] for r in rows}
    lines.append("| フレームワーク | バージョン |")
    lines.append("| --- | --- |")
    for fw in FRAMEWORKS:
        lines.append(f"| {fw} | {versions.get(fw, '?')} |")
    lines.append("")

    lines.append("### (a) GEMM（C = A×B、f32、正方行列）\n")
    for device in devices_in(rows, "gemm"):
        sizes = sorted(
            {r["size"] for r in rows if r["task"] == "gemm" and r["device"] == device}
        )
        lines.append(f"#### {DEVICE_LABEL[device]}\n")
        lines.append("| N | フレームワーク | 中央値 | Q1 | Q3 | GFLOP/s |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for n in sizes:
            for fw in FRAMEWORKS:
                r = get(rows, fw, "gemm", device, n)
                if r:
                    lines.append(
                        f"| {n} | {fw} | {fmt_ms(r['median_s'])} | {fmt_ms(r['q1_s'])} | {fmt_ms(r['q3_s'])} | {r['gflops']:.1f} |"
                    )
                else:
                    lines.append(f"| {n} | {fw} | 計測不可 | - | - | - |")
        lines.append("")

    lines.append(
        "### (b) MLP 学習（784→256→10、ReLU、バッチ 64、MSE、SGD lr=0.01、1 ステップあたり時間）\n"
    )
    lines.append("| デバイス | フレームワーク | 中央値 | Q1 | Q3 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for device in devices_in(rows, "train"):
        for fw in FRAMEWORKS:
            r = get(rows, fw, "train", device)
            if r:
                lines.append(
                    f"| {device} | {fw} | {fmt_ms(r['median_s'])} | {fmt_ms(r['q1_s'])} | {fmt_ms(r['q3_s'])} |"
                )
            else:
                lines.append(f"| {device} | {fw} | 計測不可 | - | - |")
    lines.append("")

    lines.append("### (c) 推論スループット（同 MLP forward のみ、バッチ 64）\n")
    lines.append("| デバイス | フレームワーク | 中央値 | Q1 | Q3 | 推論/秒 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for device in devices_in(rows, "infer"):
        for fw in FRAMEWORKS:
            r = get(rows, fw, "infer", device)
            if r:
                lines.append(
                    f"| {device} | {fw} | {fmt_ms(r['median_s'])} | {fmt_ms(r['q1_s'])} | {fmt_ms(r['q3_s'])} | {r['throughput_per_s']:.0f} |"
                )
            else:
                lines.append(f"| {device} | {fw} | 計測不可 | - | - | - |")
    lines.append("")
    return lines


def main():
    parser = argparse.ArgumentParser(
        description="framework-compare の JSONL 計測結果を Markdown 表へ集計する"
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="入力 JSONL（省略時は results/raw/*.jsonl を全件）",
    )
    parser.add_argument(
        "--out",
        help="出力先ファイル（省略時は標準出力。コミット済み summary.md を既定で上書きしない）",
    )
    args = parser.parse_args()

    inputs = args.inputs or sorted(glob.glob(os.path.join(HERE, "results/raw/*.jsonl")))
    if not inputs:
        print("error: 入力 JSONL がありません（results/raw/*.jsonl）", file=sys.stderr)
        return 1

    lines = ["# ベンチマーク集計（summarize.py 生成）\n"]
    for path in inputs:
        rows = load_rows(path)
        if not rows:
            lines.append(f"## 集計対象: {os.path.relpath(path, HERE)}\n")
            lines.append("（有効な行なし）\n")
            continue
        lines.extend(section(path, rows))

    skip_logs = sorted(glob.glob(os.path.join(HERE, "results/raw/skipped*.log")))
    lines.append("## 実行時失敗（skipped*.log）\n")
    any_skip = False
    for sl in skip_logs:
        for line in open(sl):
            line = line.strip()
            if line:
                any_skip = True
                lines.append(f"- **{os.path.basename(sl)}**: {line}")
    if not any_skip:
        lines.append("- なし（skipped*.log は空または不在）")
    lines.append("")

    text = "\n".join(lines) + "\n"
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
