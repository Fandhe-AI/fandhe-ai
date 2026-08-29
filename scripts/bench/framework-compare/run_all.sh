#!/bin/bash
# Run the full benchmark sweep. Results: results/raw/results.jsonl
# Failures are recorded in results/raw/skipped.log (never fabricated).
# pipefail: ビルド失敗を tail へのパイプで握り潰さない（下の build ガード参照）
set -uo pipefail
cd "$(dirname "$0")"

OUT=results/raw/results.jsonl
SKIP=results/raw/skipped.log
mkdir -p results/raw
: > "$OUT"
: > "$SKIP"

run() { # run <binary> <task> <device> <size> [mode]
  local bin=$1 task=$2 device=$3 size=$4 mode=${5:-fresh}
  echo "== $bin $task $device size=$size mode=$mode =="
  if ! "./target/release/$bin" --task "$task" --device "$device" --size "$size" --mode "$mode" --out "$OUT" 2>err.tmp; then
    echo "$bin task=$task device=$device size=$size mode=$mode : $(cat err.tmp)" >> "$SKIP"
    echo "  -> FAILED (recorded in $SKIP)"
  fi
  rm -f err.tmp
}

# ビルド失敗時はここで中断する（古い target/release バイナリを現行ツリーの結果として
# 計測・記録しないため。pipefail により tail 越しでも cargo の失敗が伝播する）
if ! cargo build --release 2>&1 | tail -20; then
  echo "BUILD FAILED: aborting sweep (stale binaries must not be measured)" >&2
  exit 1
fi

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

# (a') GEMM — デバイス/tape 再利用モード（イシュー #925。bench-fandhe の
# gemm タスクのみ対応。bench-candle / bench-burn は MEASURE_ERROR で
# fail-fast し skipped.log に記録される既存機構に乗る）
for n in 256 512 1024 2048 4096; do
  run bench-fandhe gemm metal "$n" reuse
done

# (b') MLP 学習 — デバイス常駐更新モード（イシュー #957/#958/#959。bench-fandhe の
# train タスクのみ対応。(a') と同じ理由で bench-candle / bench-burn はループに
# 含めない: reuse モードは必ず MEASURE_ERROR で fail-fast する仕様のため、対象外の
# 2 バイナリまで含めると既知の対象外失敗が skipped.log の実際の計測失敗と混在し
# 判別しづらくなる。codex-review 指摘 #944 discussion_r3877595038 と同じ理由）
for dev in cpu metal; do
  run bench-fandhe train "$dev" 64 reuse
done

echo "done. results in $OUT ; failures (if any) in $SKIP"
