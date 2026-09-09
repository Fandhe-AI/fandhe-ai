#!/bin/bash
# Layer B（crates/backend-cpu 内側フェーズ分解診断テスト）5 run 実測スクリプト
# （イシュー #1481・Apple M4 Max。#1301 の run_layerB_m4max.sh と同型。
# 引数 $1 は対象ツリーの絶対パス（off 腕 = 実装 worktree・on 腕 = スクラッチ
# 複製ツリー）・$2 は arm ラベル（off|on）。ログは本ディレクトリ配下へ書く
# ため WORKTREE ではなく off 腕（実装 worktree）の本ディレクトリを LOG に使う）。
set -euo pipefail
WORKTREE="$1"
ARM="$2"
LOG_DIR="$(cd "$(dirname "$0")" && pwd)"
for run in 1 2 3 4 5; do
  echo "run=$run $(uptime)" >> "$LOG_DIR/uptime-m4max.log"
  (cd "$WORKTREE" && cargo test -p fandhe-ai-backend-cpu --release --lib -- --ignored \
    gemm_reuse_phase_diag_cpu --nocapture --test-threads=1) \
    > "$LOG_DIR/layerB-m4max-$ARM-run$run.log" 2>&1
done
echo DONE
