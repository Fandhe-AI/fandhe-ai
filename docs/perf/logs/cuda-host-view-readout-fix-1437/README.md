# イシュー #1437 実測ログ

`docs/perf/cuda-host-view-readout-small-shape-regression.md` §13 以降の
根拠データ。

## 構成

- `env_info.txt`: 実機環境情報（内部ホスト名は含めない）
- `layer-b/`: Layer B 分離計測（`readout_regression_diag_tests_1436.rs`
  の単一腕プロセス分離実行。`LegacyToVec`／`BorrowedKeepAlive`／
  `PretouchedFreshDest` の 3 腕 × N=1024/2048/4096）の生ログ
- `results-dgx-gemm-gate-1437after-off.jsonl`: Layer A（framework-compare
  実践規模）`off@after`（結線後 HEAD・`host-view-readout` feature 無効。
  N=1024/2048/4096 reuse × 5 run）
- `results-dgx-gemm-gate-1437after-on.jsonl`: 同 `on@after`（結線後 HEAD・
  `host-view-readout` feature 有効。同一プロトコル）
- `results-dgx-gemm-gate-0.7.0-official-base.jsonl`: `off@base`（正式系列
  `fandhe-ai =0.7.0` registry 版。#1185 で 2026-09-06 に計測済みの既存
  ファイルをそのまま再利用。新規計測は行っていない——出力（bit 単位）
  は本イシューの変更の影響を受けないため〈`readback_with` の `Fresh`
  分岐自体は変更していない〉同一系列として扱ってよいが、これは出力の
  同一性であり性能の同一性ではない点に注意（`readback()` の既定
  `ReadbackDest` は `host-view-readout` feature の有効・無効に関わらず
  無条件に `PretouchedFresh` へ切り替わっているため〈`memory.rs:575`〉、
  `off@after` は `off@base`（旧 `Fresh` 既定）と異なる宛先確保方式を
  通り、Gate 2 上でわずかな超過が観測されている。詳細な再評価は
  `docs/perf/cuda-host-view-readout-small-shape-regression.md` §13.3
  を参照）
- `manifest-dgx-gemm-gate-1437after-{off,on}.json`: 上記 2 系列のビルド
  sha256・依存解決元記録（`run_gemm_gate.sh` が生成する manifest。§本文
  「バイナリ同一性検証」節参照）

## 再現手順

```bash
# off@after
cd scripts/bench/framework-compare
GEMM_GATE_PATCH_FACADE_PATH=<HEAD ツリーの crates/facade 絶対パス> \
  bash run_gemm_gate_cuda.sh 1437after-off

# on@after
GEMM_GATE_PATCH_FACADE_PATH=<HEAD ツリーの crates/facade 絶対パス> \
  GEMM_GATE_BENCH_FANDHE_FEATURES=host-view-readout \
  bash run_gemm_gate_cuda.sh 1437after-on

# Layer B（単一腕プロセス分離実行の例。N=1024 PretouchedFreshDest）
cargo test -p fandhe-ai-backend-cuda --release --lib \
  readout_regression_diag_n1024_pretouched_fresh_dest -- --ignored --nocapture --test-threads=1
```
