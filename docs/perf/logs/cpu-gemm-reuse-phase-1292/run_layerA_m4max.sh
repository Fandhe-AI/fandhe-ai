#!/bin/bash
# Layer A（framework-compare `gemm --mode reuse --phases`）5 run × N=512/1024/2048 実測スクリプト（イシュー #1292・M4 Max）。
set -euo pipefail
cd <worktree>/scripts/bench/framework-compare
LOG=<worktree>/docs/perf/logs/cpu-gemm-reuse-phase-1292
OUT="$LOG/layerA-phases-m4max.jsonl"
rm -f "$OUT"
for run in 1 2 3 4 5; do
  for N in 512 1024 2048; do
    echo "run=$run N=$N $(uptime)" >> "$LOG/uptime-m4max.log"
    cargo run --release -p bench-fandhe -- --task gemm --device cpu --size "$N" --mode reuse --phases --out "$OUT"
  done
done
echo DONE
wc -l "$OUT"
