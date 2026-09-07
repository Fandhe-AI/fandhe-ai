#!/bin/bash
# Layer B（crates/backend-cpu 内側フェーズ分解診断テスト）5 run 実測スクリプト（イシュー #1292・M4 Max）。
set -euo pipefail
cd <worktree>
LOG=<worktree>/docs/perf/logs/cpu-gemm-reuse-phase-1292
for run in 1 2 3 4 5; do
  echo "run=$run $(uptime)" >> "$LOG/uptime-m4max.log"
  cargo test -p fandhe-ai-backend-cpu --release --lib -- --ignored \
    gemm_reuse_phase_diag_cpu --nocapture --test-threads=1 \
    > "$LOG/layerB-m4max-run$run.log" 2>&1
done
echo DONE
