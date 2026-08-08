#!/usr/bin/env python3
"""TASK-8.3a（イシュー #155）: Transformer 複合ワークロード PyTorch 側ベースライン計測。

`crates/bench-harness/tests/transformer_workload.rs`（Rust 側・自作コア経路）と
同一形状・同一計測プロトコルで計測し、対 PyTorch 比の算出根拠を得る
（`docs/perf/transformer-workload-measurement.md` 参照）。

## 形状（PoC-8 定義。PoC-5 流用）

d_model=512, n_heads=8, d_ff=2048, batch=8, seq_len=128, num_layers=1,
activation=gelu, norm_first=False（post-norm）, layer_norm_eps=1e-5

## PoC-5 先例からの変更点

`docs/spec/03-poc/poc-5-performance/code/pytorch/transformer_block_bench_torch.py`
（読み取り参照のみ。`docs/spec/` は編集しない）を土台に、TASK-8.1 計測プロトコル
（warmup 20 回以上・計測 20 回以上・中央値／Q1・Q3 記録・決定的シード・
device 対応の完了待ち同期）へ合わせて新規作成した:

- warmup/reps を PoC-5 の 10/10 から 20/20 以上（既定 20/20）へ引き上げ
- `torch.manual_seed` による決定的シード固定を追加（PoC-5 は `torch.randn` に
  シード指定なし。bench-harness 側の `Xorshift64Star` 決定的シードとの整合方針に合わせる）
- `dropout=0.1` を `0.0` に変更（`nn.Module.eval()` で無効化はされるが、Rust 側
  forward 実装に dropout 相当の処理が存在しないため計測経路を完全に一致させる）
- device を `cpu`/`cuda`/`mps` から選択可能にし、device ごとの完了待ち同期
  （`torch.cuda.synchronize()`／`torch.mps.synchronize()`）を追加
- 出力を `docs/perf/cuda-tensor-core-measurement.md` 系の実測記録と揃うよう
  JSON（`bench_harness::BenchReport` と同形の median_secs/q1_secs/q3_secs）に変更

## 使用バージョン

PyTorch 2.13.0（`docs/spec/04-requirements.md` REQ-8 表と同一バージョンを明記する）。
本スクリプトはリポジトリの Cargo 依存には一切影響しない（計測専用の一時 Python
環境で実行する想定。`deps-policy.md` の対象外。venv・wheel はコミットしない）。

## セキュリティ（OWASP A03）

外部入力は `argparse` の限定的な引数（device・warmup・iters・出力パス）のみを
受け取り、シェル展開・`eval`・`os.system` を一切使わない。
"""

from __future__ import annotations

import argparse
import json
import sys
import time

MIN_ITERATIONS = 20  # TASK-8.1（`docs/spec/05-tasks.md`）の計測プロトコル下限と同一。

D_MODEL = 512
N_HEADS = 8
D_FF = 2048
BATCH = 8
SEQ_LEN = 128
SEED = 155_083  # Rust 側（bench-harness の SEED 定数）とイシュー番号を揃えた値。
# 決定的シードの値自体を一致させる意味はない（RNG アルゴリズムが異なるため入力の
# ビット列は一致しない）。「イシュー番号由来の固定シードを使う」運用のみを揃える。


def median_q1_q3(samples: list[float]) -> tuple[float, float, float]:
    """`bench_harness::stats::median_q1_q3`（`crates/bench-harness/src/stats.rs`）と
    同じ分位点定義。

    ソート後、`idx = round(p * (n - 1))`（`p = 0.5/0.25/0.75`）番目の要素をそのまま
    採用する（線形補間なし・区間の中央値の平均でもない）。PR #345 レビュー
    （Bugbot 指摘）で、以前の実装（下半分・上半分それぞれの中央値を平均する
    median-of-halves 方式）が `bench_harness::stats::median_q1_q3` の実際の定義と
    一致していないと判明したため修正した。`round` は Python 標準の
    round-half-to-even ではなく Rust `f64::round`（round-half-away-from-zero）と
    同じ丸めが必要である。`p * (n - 1)` がちょうど `.5` に当たる（タイになる）
    ケースは珍しくない: 例えば `p=0.5` かつ `n` が偶数のとき `n-1` は奇数となり
    `(n-1)/2` は必ず `.5` タイになる（本ベンチマークの `n=20` もこのケース。
    `idx=9.5` で `round-half-to-even` なら 10、`round-half-away-from-zero` でも
    10 のため今回のサンプルでは両者が偶然一致するが、`n=10` 等では 4 対 5 に
    分岐する）。境界ケースを取りこぼさないよう `math.floor(x + 0.5)` で明示的に
    round-half-away-from-zero を実装する。
    """
    import math

    s = sorted(samples)
    n = len(s)

    def pick(p: float) -> float:
        idx = int(math.floor(p * (n - 1) + 0.5))
        idx = min(idx, n - 1)
        return s[idx]

    return pick(0.5), pick(0.25), pick(0.75)


def resolve_device(device_arg: str):
    import torch

    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("device=cuda が指定されましたが CUDA が利用できません")
        return torch.device("cuda")
    if device_arg == "mps":
        if not torch.backends.mps.is_available():
            raise SystemExit("device=mps が指定されましたが MPS が利用できません")
        return torch.device("mps")
    raise SystemExit(f"未知の device: {device_arg}")


def sync_for(device_arg: str) -> None:
    """device 対応の「ホスト転送を伴わない完了待ち」（REQ-8 が定める同期方式統一）。"""
    import torch

    if device_arg == "cuda":
        torch.cuda.synchronize()
    elif device_arg == "mps":
        torch.mps.synchronize()
    # cpu は同期不要（`bench_harness::sync::CpuSync` と同じ no-op 契約）。


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transformer 複合ワークロード PyTorch ベースライン計測（TASK-8.3a）"
    )
    parser.add_argument(
        "--device", choices=["cpu", "cuda", "mps"], default="cpu", help="計測デバイス"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=MIN_ITERATIONS,
        help=f"warmup 回数（TASK-8.1 下限 {MIN_ITERATIONS} 未満は拒否）",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=MIN_ITERATIONS,
        help=f"計測回数（TASK-8.1 下限 {MIN_ITERATIONS} 未満は拒否）",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="JSON 出力先パス（未指定時は stdout のみ）"
    )
    args = parser.parse_args()

    # TASK-8.1 計測プロトコル下限（20/20）を回避する経路を設けない
    # （`bench_harness::protocol::MeasurementConfig::new` と同じ fail-closed 方針。
    # `.claude/rules/security.md` A08）。
    if args.warmup < MIN_ITERATIONS or args.iters < MIN_ITERATIONS:
        raise SystemExit(
            f"warmup・iters とも {MIN_ITERATIONS} 回以上が必須（TASK-8.1）。"
            f"指定値: warmup={args.warmup}, iters={args.iters}"
        )

    import torch
    import torch.nn as nn

    torch.manual_seed(SEED)

    device = resolve_device(args.device)

    layer = nn.TransformerEncoderLayer(
        d_model=D_MODEL,
        nhead=N_HEADS,
        dim_feedforward=D_FF,
        dropout=0.0,
        activation="gelu",
        layer_norm_eps=1e-5,
        batch_first=True,
        norm_first=False,
    )
    encoder = nn.TransformerEncoder(layer, num_layers=1).to(device)
    encoder.eval()

    input_tensor = torch.randn(BATCH, SEQ_LEN, D_MODEL, device=device)

    with torch.no_grad():
        for _ in range(args.warmup):
            out = encoder(input_tensor)
            sync_for(args.device)
            del out

        samples_secs: list[float] = []
        for _ in range(args.iters):
            start = time.perf_counter()
            out = encoder(input_tensor)
            sync_for(args.device)
            elapsed = time.perf_counter() - start
            samples_secs.append(elapsed)
            del out

    median, q1, q3 = median_q1_q3(samples_secs)

    report = {
        "schema_version": "1",
        "name": "transformer-block-forward-pytorch",
        "backend": args.device,
        "framework": "pytorch",
        "framework_version": torch.__version__,
        "warmup": args.warmup,
        "iters": args.iters,
        "median_secs": median,
        "q1_secs": q1,
        "q3_secs": q3,
        "samples_secs": samples_secs,
    }

    output_json = json.dumps(report)
    print(output_json)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_json)

    return 0


if __name__ == "__main__":
    sys.exit(main())
