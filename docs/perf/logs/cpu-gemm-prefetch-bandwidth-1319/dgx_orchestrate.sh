#!/usr/bin/env bash
set -uo pipefail
source ~/.cargo/env 2>/dev/null
cd ~/work/ab-1319/rust-ai-library || exit 1
export CARGO_TARGET_DIR=$HOME/work/target-ab-1319
OUT=~/work/ab-1319/logs
mkdir -p "$OUT"
for i in 1 2 3 4 5; do
  cargo test -p fandhe-ai-backend-cpu --release --lib -- --ignored microkernel_residency_diag --nocapture > "$OUT/resid-dgx-run$i.txt" 2>&1
done
for i in 1 2 3 4 5; do
  cargo test -p fandhe-ai-backend-cpu --release --lib -- --ignored achievable_bandwidth_diag --nocapture > "$OUT/bw-dgx-run$i.txt" 2>&1
done
cargo test -p fandhe-ai-backend-cpu --release --lib -- --ignored gemm_blis_variant_ab_1024_2048 --nocapture > "$OUT/ab-1024-2048-dgx-run1.txt" 2>&1
echo DONE > "$OUT/DONE_MARKER"
