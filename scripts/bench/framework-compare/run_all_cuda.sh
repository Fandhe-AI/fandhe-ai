#!/bin/bash
# CUDA-host sweep (e.g. DGX Spark): cuda + cpu for all three frameworks.
# bench-candle / bench-burn are built with --no-default-features --features cuda
# (their default `metal` feature is macOS-only). Failures are recorded in
# results/raw/skipped-cuda.log (never fabricated).
set -u
cd "$(dirname "$0")"

OUT=results/raw/results-cuda.jsonl
SKIP=results/raw/skipped-cuda.log
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

build() { # build <crate> [extra cargo args...]
  local crate=$1; shift
  echo "== build $crate $* =="
  if ! cargo build --release -p "$crate" "$@" 2>build-err.tmp; then
    tail -40 build-err.tmp
    echo "$crate BUILD FAILED: $(tail -3 build-err.tmp | tr '\n' ' ')" >> "$SKIP"
    echo "  -> BUILD FAILED (recorded in $SKIP)"
    rm -f build-err.tmp
    return 1
  fi
  rm -f build-err.tmp
}

BINS=()
build bench-fandhe && BINS+=(bench-fandhe)
build bench-candle --no-default-features --features cuda && BINS+=(bench-candle)
build bench-burn --no-default-features --features cuda && BINS+=(bench-burn)

for bin in "${BINS[@]}"; do
  # (a) GEMM
  for n in 256 512 1024 2048 4096; do
    run "$bin" gemm cuda "$n"
  done
  for n in 256 512 1024 2048; do
    run "$bin" gemm cpu "$n"
  done
  # (b) MLP training, (c) inference
  for dev in cuda cpu; do
    run "$bin" train "$dev" 64
    run "$bin" infer "$dev" 64
  done
done

echo "done. results in $OUT ; failures (if any) in $SKIP"
