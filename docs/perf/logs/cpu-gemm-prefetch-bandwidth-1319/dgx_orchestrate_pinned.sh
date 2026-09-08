#!/usr/bin/env bash
set -uo pipefail
source ~/.cargo/env 2>/dev/null
cd ~/work/ab-1319/rust-ai-library || exit 1
export CARGO_TARGET_DIR=$HOME/work/target-ab-1319
OUT=~/work/ab-1319/logs
mkdir -p "$OUT"
BIN=$(find "$CARGO_TARGET_DIR/release/deps" -maxdepth 1 -type f -name 'fandhe_ai_backend_cpu-*' -perm -u+x ! -name '*.d' | head -1)
for i in 1 2 3 4 5; do
  taskset -c 5-9,15-19 "$BIN" --ignored microkernel_residency_diag --nocapture > "$OUT/resid-dgx-bigpin-run$i.txt" 2>&1
done
echo DONE > "$OUT/DONE_MARKER_PINNED"
