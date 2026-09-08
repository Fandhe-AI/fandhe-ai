#!/bin/bash
# Layer B（crates/backend-cpu 内側フェーズ分解診断テスト）5 run 実測スクリプト
# （イシュー #1301・Apple M4 Max。#1292 の run_layerB_m4max.sh と同型。
# 引数 $1 は arm ラベル（off|on）。
set -euo pipefail
WORKTREE="$1"
ARM="$2"
LOG="$WORKTREE/docs/perf/logs/cpu-matmul-fixed-cost-1301"
for run in 1 2 3 4 5; do
  echo "run=$run $(uptime)" >> "$LOG/uptime-m4max.log"
  (cd "$WORKTREE" && cargo test -p fandhe-ai-backend-cpu --release --lib -- --ignored \
    gemm_reuse_phase_diag_cpu --nocapture --test-threads=1) \
    > "$LOG/layerB-m4max-$ARM-run$run.log" 2>&1
done
echo DONE
