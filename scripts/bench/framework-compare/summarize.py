#!/usr/bin/env python3
"""Generate results/summary.md from results/raw/results.jsonl (+ skipped.log)."""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "results/raw/results.jsonl")
SKIP = os.path.join(HERE, "results/raw/skipped.log")
OUT = os.path.join(HERE, "results/summary.md")

rows = [json.loads(l) for l in open(RAW) if l.strip()]
skipped = [l.strip() for l in open(SKIP)] if os.path.exists(SKIP) else []

FRAMEWORKS = ["fandhe-ai", "candle", "burn"]
VERSIONS = {r["framework"]: r["version"] for r in rows}


def fmt_ms(s):
    if s >= 1.0:
        return f"{s:.3f} s"
    if s >= 1e-3:
        return f"{s * 1e3:.3f} ms"
    return f"{s * 1e6:.1f} µs"


def get(fw, task, device, size=None):
    for r in rows:
        if r["framework"] == fw and r["task"] == task and r["device"] == device:
            if size is None or r["size"] == size:
                return r
    return None


def sw(chip):
    return subprocess.run(chip, shell=True, capture_output=True, text=True).stdout.strip()


chip = sw("sysctl -n machdep.cpu.brand_string")
os_ver = sw("sw_vers -productVersion")
darwin = sw("uname -r")
cargo_v = sw("cargo --version")

lines = []
lines.append("# ベンチマーク結果サマリー（fandhe-ai-introduction）\n")
lines.append("## 環境\n")
lines.append(f"- チップ: {chip}")
lines.append(f"- OS: macOS {os_ver}（Darwin {darwin}）")
lines.append(f"- ツールチェーン: {cargo_v}（`--release` ビルド）")
lines.append("- 計測日: 2026-08-28")
lines.append("- 計測プロトコル: warmup 20 回 → 計測 20 回（学習は 100 ステップ中先頭 20 を warmup、残り 80 を計測）。中央値・Q1・Q3 を記録")
lines.append("- 同期: 計測区間終端で結果テンソルをホストへ実体化し全要素を読み出す（checksum として記録）")
lines.append("- 入力データ: xorshift64* の同一シード・同一生成式で全フレームワーク共通\n")

lines.append("## 採用バージョン\n")
lines.append("| フレームワーク | クレート | バージョン |")
lines.append("| --- | --- | --- |")
lines.append(f"| fandhe-ai | fandhe-ai (facade) | {VERSIONS.get('fandhe-ai', '?')} |")
lines.append(f"| candle | candle-core (metal feature) | {VERSIONS.get('candle', '?')} |")
lines.append(f"| Burn | burn (ndarray / wgpu backend) | {VERSIONS.get('burn', '?')} |\n")

lines.append("## (a) GEMM（C = A×B、f32、正方行列）\n")
for device, sizes in [("cpu", [256, 512, 1024, 2048]), ("metal", [256, 512, 1024, 2048, 4096])]:
    dev_label = "CPU" if device == "cpu" else "Metal"
    lines.append(f"### {dev_label}\n")
    lines.append("| N | フレームワーク | 中央値 | Q1 | Q3 | GFLOP/s |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for n in sizes:
        for fw in FRAMEWORKS:
            r = get(fw, "gemm", device, n)
            if r:
                lines.append(
                    f"| {n} | {fw} | {fmt_ms(r['median_s'])} | {fmt_ms(r['q1_s'])} | {fmt_ms(r['q3_s'])} | {r['gflops']:.1f} |"
                )
            else:
                lines.append(f"| {n} | {fw} | 計測不可 | - | - | - |")
    lines.append("")

lines.append("## (b) MLP 学習（784→256→10、ReLU、バッチ 64、MSE、SGD lr=0.01、1 ステップあたり時間）\n")
lines.append("| デバイス | フレームワーク | 中央値 | Q1 | Q3 |")
lines.append("| --- | --- | --- | --- | --- |")
for device in ["cpu", "metal"]:
    for fw in FRAMEWORKS:
        r = get(fw, "train", device)
        if r:
            lines.append(
                f"| {device} | {fw} | {fmt_ms(r['median_s'])} | {fmt_ms(r['q1_s'])} | {fmt_ms(r['q3_s'])} |"
            )
        else:
            lines.append(f"| {device} | {fw} | 計測不可 | - | - |")
lines.append("")

lines.append("## (c) 推論スループット（同 MLP forward のみ、バッチ 64）\n")
lines.append("| デバイス | フレームワーク | 中央値 | Q1 | Q3 | 推論/秒 |")
lines.append("| --- | --- | --- | --- | --- | --- |")
for device in ["cpu", "metal"]:
    for fw in FRAMEWORKS:
        r = get(fw, "infer", device)
        if r:
            lines.append(
                f"| {device} | {fw} | {fmt_ms(r['median_s'])} | {fmt_ms(r['q1_s'])} | {fmt_ms(r['q3_s'])} | {r['throughput_per_s']:.0f} |"
            )
        else:
            lines.append(f"| {device} | {fw} | 計測不可 | - | - | - |")
lines.append("")

lines.append("## 計測不可・未計測項目\n")
lines.append("- **CUDA（全フレームワーク）**: 計測不可。本環境（Apple M4 Max / macOS）に CUDA デバイスが存在しない")
lines.append("- **tch-rs（全タスク）**: 未計測。libtorch 依存のため（導入が制限時間内に完了しない見込みで省略）")
for s in skipped:
    lines.append(f"- **実行時失敗**: {s}")
if not skipped:
    lines.append("- 実行時に失敗した組み合わせ: なし（skipped.log は空）")
lines.append("")

lines.append("## 備考\n")
lines.append("- GEMM の入力は全フレームワークで同一（checksum が一致することを JSONL で確認できる）")
lines.append("- 学習・推論の重みは candle / Burn は共有 RNG で同一。fandhe-ai は `Sequential::add_linear` の内部初期化（シード指定）のため重みの値は異なるが、同一アーキテクチャ・同一入力・同一バッチであり実行時間の比較には影響しない")
lines.append("- fandhe-ai の学習ループは公開 API のみ（`compat::Sequential` + `tape.backward` + 手動 SGD）。パラメータ更新はホスト側で `param - lr * grad` を計算して `apply_parameters` で書き戻す実装であり、フレームワークにより更新方式が異なる（candle: `Var::set`、Burn: `from_inner + require_grad`）")

open(OUT, "w").write("\n".join(lines) + "\n")
print(f"wrote {OUT}")
