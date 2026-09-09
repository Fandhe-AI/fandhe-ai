#!/bin/bash
# CUDA-host sweep (e.g. DGX Spark): cuda + cpu for all three frameworks.
# bench-candle / bench-burn are built with --no-default-features --features cuda
# (their default `metal` feature is macOS-only). Failures are recorded in
# results/raw/skipped-cuda.log (never fabricated).
set -u
cd "$(dirname "$0")"

# イシュー #1438 P0 是正（codex-review 指摘 PRRT_kwDOTuUCJc6gH59Q）に加え
# PR #1452 codex-review P1 是正（PRRT_kwDOTuUCJc6gIbMM）: `run_all.sh` と
# 同じ理由で `GEMM_GATE_PATCH_FACADE_PATH`（任意。`crates/facade` への
# path patch）を導入した（bench-candle／bench-burn は fandhe-ai に依存
# しないため patch 対象外）。承認ピン `fandhe-ai =0.8.0`（#1487）への
# 更新後も参考系列（HEAD ソース）計測用として維持し、未指定時は registry
# 解決（承認ピン）のまま通常ビルドする。

# A03 インジェクション対策（run_gemm_gate.sh と同一方針）。
CARGO_CONFIG_ARGS=()
if [[ -n "${GEMM_GATE_PATCH_FACADE_PATH:-}" ]]; then
  if [[ "$GEMM_GATE_PATCH_FACADE_PATH" == *'"'* || "$GEMM_GATE_PATCH_FACADE_PATH" == *'\'* ]]; then
    echo "ERROR: GEMM_GATE_PATCH_FACADE_PATH に '\"' または '\\' を含めることはできない" >&2
    exit 1
  fi
  CARGO_CONFIG_ARGS+=(--config "patch.crates-io.fandhe-ai.path=\"${GEMM_GATE_PATCH_FACADE_PATH}\"")
fi

OUT=results/raw/results-cuda.jsonl
SKIP=results/raw/skipped-cuda.log
mkdir -p results/raw
: > "$OUT"
: > "$SKIP"

# PR #1452 codex-review P2 指摘（PRRT_kwDOTuUCJc6gKes-）: 本スクリプトも
# `bench-fandhe` のみに `CARGO_CONFIG_ARGS`（`GEMM_GATE_PATCH_FACADE_PATH`
# 経由の invocation-only `--config patch.crates-io.fandhe-ai.path=...`）を
# 付けて `cargo build` するため、patch 解決過程で本 workspace の
# `Cargo.lock`（承認済みピン固定）が書き換わったまま残りうる。`run_all.sh`
# と同一設計（`bench_fandhe_lock_restore.sh`。#1487 で
# `bench_fandhe_pin_guard.sh` から分離。EXIT trap 経由で終了経路に依らず
# 必ず復元する。未指定時も無害）で解消する。
source ./bench_fandhe_lock_restore.sh
bench_fandhe_setup_lock_restore_trap

run() { # run <binary> <task> <device> <size> [mode] [extra_flag]
  local bin=$1 task=$2 device=$3 size=$4 mode=${5:-fresh} extra_flag=${6:-}
  echo "== $bin $task $device size=$size mode=$mode extra=${extra_flag:-none} =="
  if ! "./target/release/$bin" --task "$task" --device "$device" --size "$size" --mode "$mode" ${extra_flag:+"$extra_flag"} --out "$OUT" 2>err.tmp; then
    echo "$bin task=$task device=$device size=$size mode=$mode extra=${extra_flag:-none} : $(cat err.tmp)" >> "$SKIP"
    echo "  -> FAILED (recorded in $SKIP)"
  fi
  rm -f err.tmp
}

build() { # build <crate> [extra cargo args...]
  local crate=$1; shift
  local extra_args=("$@")
  # bench-fandhe のみ patch 対象（fandhe-ai に依存するのは bench-fandhe だけ。
  # bench-candle／bench-burn へ付けても cargo は無害な unused-patch 警告を
  # 出すのみだが、意図を明確にするため対象を絞る）。
  if [[ "$crate" == "bench-fandhe" ]]; then
    extra_args+=("${CARGO_CONFIG_ARGS[@]}")
  fi
  echo "== build $crate ${extra_args[*]} =="
  if ! cargo build --release -p "$crate" "${extra_args[@]}" 2>build-err.tmp; then
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

# (a') GEMM — デバイス/tape 再利用モード（イシュー #925。bench-fandhe の
# gemm タスクのみ対応。bench-candle / bench-burn は reuse モードを必ず
# MEASURE_ERROR で fail-fast する仕様のため、対象外の 2 バイナリまで
# ループに含めると計 10 件（5 サイズ×2 バイナリ）の既知の対象外失敗が
# skipped-cuda.log の「Failures」に混じり実際の計測失敗と判別しづらくなる
# （codex-review 指摘 #944 discussion_r3877595038）。BINS に bench-fandhe が
# 含まれる場合に限り、bench-fandhe のみを対象にこのループを実行する
if [[ " ${BINS[*]} " == *" bench-fandhe "* ]]; then
  for n in 256 512 1024 2048 4096; do
    run bench-fandhe gemm cuda "$n" reuse
  done
  # (b') MLP 学習 — デバイス常駐更新モード（イシュー #957/#958/#959）。上と同じ
  # ガード（BINS に bench-fandhe が含まれる場合のみ）・同じ理由（対象外の
  # bench-candle / bench-burn の既知失敗で skipped-cuda.log を汚さない）。
  for dev in cuda cpu; do
    run bench-fandhe train "$dev" 64 reuse
  done
  # (b'') MLP 学習 1 step のフェーズ分解（イシュー #1009）。上と同じガード・
  # 同じ理由（--phases は必ず MEASURE_ERROR で fail-fast する仕様のため
  # bench-candle / bench-burn は対象外）。
  for dev in cuda cpu; do
    run bench-fandhe train "$dev" 64 fresh --phases
    run bench-fandhe train "$dev" 64 reuse --phases
  done
  # (c') 推論 — デバイス常駐パラメータ更新モード（イシュー #1217）。上と
  # 同じガード・同じ理由。
  for dev in cuda cpu; do
    run bench-fandhe infer "$dev" 64 reuse
  done
  # (c'') 推論 1 反復のフェーズ分解（イシュー #1217）。上と同じガード・
  # 同じ理由。
  for dev in cuda cpu; do
    run bench-fandhe infer "$dev" 64 fresh --phases
    run bench-fandhe infer "$dev" 64 reuse --phases
  done
fi

echo "done. results in $OUT ; failures (if any) in $SKIP"
