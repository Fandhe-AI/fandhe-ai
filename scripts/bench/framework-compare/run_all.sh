#!/bin/bash
# Run the full benchmark sweep. Results: results/raw/results.jsonl
# Failures are recorded in results/raw/skipped.log (never fabricated).
set -u
cd "$(dirname "$0")"

OUT=results/raw/results.jsonl
SKIP=results/raw/skipped.log
mkdir -p results/raw
: > "$OUT"
: > "$SKIP"

run() { # run <binary> <task> <device> <size>
  local bin=$1 task=$2 device=$3 size=$4
  echo "== $bin $task $device size=$size =="
  if ! "./target/release/$bin" --task "$task" --device "$device" --size "$size" --out "$OUT" 2>err.tmp; then
    echo "$bin task=$task device=$device size=$size : $(cat err.tmp)" >> "$SKIP"
    echo "  -> FAILED (recorded in $SKIP)"
  fi
  rm -f err.tmp
}

cargo build --release 2>&1 | tail -1

for bin in bench-fandhe bench-candle bench-burn; do
  # (a) GEMM
  for n in 256 512 1024 2048; do
    run "$bin" gemm cpu "$n"
  done
  for n in 256 512 1024 2048 4096; do
    run "$bin" gemm metal "$n"
  done
  # (b) MLP training, (c) inference
  for dev in cpu metal; do
    run "$bin" train "$dev" 64
    run "$bin" infer "$dev" 64
  done
done

echo "done. results in $OUT ; failures (if any) in $SKIP"
