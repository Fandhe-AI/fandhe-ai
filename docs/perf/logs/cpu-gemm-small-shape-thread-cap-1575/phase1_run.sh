#!/bin/bash
# イシュー #1575 Phase 1: 同一バイナリ（SMALL_SHAPE_CAP_ENABLED=true でビルド
# 済みの bench-fandhe）を RAYON_NUM_THREADS の有無で on/off 切替し、
# run 単位で interleave 計測する。
set -u
BIN=./target/release/bench-fandhe
OUTDIR="$1"
mkdir -p "$OUTDIR"

# 判定対象セル（train/infer cpu fresh/reuse）＋参考セル（gemm cpu 512/1024/2048 fresh/reuse）。
CELLS=(
  "train:fresh"
  "train:reuse"
  "infer:fresh"
  "infer:reuse"
  "gemm:fresh:512"
  "gemm:reuse:512"
  "gemm:fresh:1024"
  "gemm:reuse:1024"
  "gemm:fresh:2048"
  "gemm:reuse:2048"
)

run_one() {
  local task=$1 mode=$2 size=$3 arm=$4 run_idx=$5
  local out="$OUTDIR/${task}_${mode}_${size}_${arm}_run${run_idx}.jsonl"
  if [[ "$arm" == "after" ]]; then
    # after: RAYON_NUM_THREADS 未設定（cap 有効・eligible なら発火）
    if [[ "$task" == "gemm" ]]; then
      env -u RAYON_NUM_THREADS "$BIN" --task "$task" --device cpu --mode "$mode" --size "$size" --out "$out"
    else
      env -u RAYON_NUM_THREADS "$BIN" --task "$task" --device cpu --mode "$mode" --out "$out"
    fi
  else
    # before: RAYON_NUM_THREADS=16 を明示設定（cap 無効。env_override）
    if [[ "$task" == "gemm" ]]; then
      RAYON_NUM_THREADS=16 "$BIN" --task "$task" --device cpu --mode "$mode" --size "$size" --out "$out"
    else
      RAYON_NUM_THREADS=16 "$BIN" --task "$task" --device cpu --mode "$mode" --out "$out"
    fi
  fi
}

for cell in "${CELLS[@]}"; do
  IFS=':' read -r task mode size <<< "$cell"
  for run_idx in 1 2 3 4 5; do
    # run 単位で順序を反転する（奇数 run: after→before・偶数 run: before→after）。
    if (( run_idx % 2 == 1 )); then
      run_one "$task" "$mode" "${size:-}" after "$run_idx"
      run_one "$task" "$mode" "${size:-}" before "$run_idx"
    else
      run_one "$task" "$mode" "${size:-}" before "$run_idx"
      run_one "$task" "$mode" "${size:-}" after "$run_idx"
    fi
  done
  echo "done: $cell"
done
