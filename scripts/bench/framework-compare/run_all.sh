#!/bin/bash
# Run the full benchmark sweep. Results: results/raw/results.jsonl
# Failures are recorded in results/raw/skipped.log (never fabricated).
# pipefail: ビルド失敗を tail へのパイプで握り潰さない（下の build ガード参照）
set -uo pipefail
cd "$(dirname "$0")"

# イシュー #1438 P0 是正（codex-review 指摘 PRRT_kwDOTuUCJc6gH59Q）に加え
# PR #1452 codex-review P1 是正（PRRT_kwDOTuUCJc6gIbMM）で導入した
# `GEMM_GATE_PATCH_FACADE_PATH`（任意。`crates/facade` への path patch）は
# 承認ピン `fandhe-ai =0.8.0`（#1487）への更新後も参考系列（HEAD ソース）
# 計測用として維持する。指定時のみ `--config patch.crates-io.fandhe-ai.
# path=...` でビルドする（deps-policy.md 第 9 区分の承認済みピン固定は
# Cargo.lock へ永続化しない invocation 限定の `--config` のため壊さない）。
# 未指定時は registry 解決（承認ピン）のまま通常ビルドする。

# A03 インジェクション対策（run_gemm_gate.sh と同一方針）: TOML 文字列値へ
# 埋め込むため、二重引用符・バックスラッシュを含む値は不正な `--config` を
# 生成しうるので拒否する。
CARGO_CONFIG_ARGS=()
if [[ -n "${GEMM_GATE_PATCH_FACADE_PATH:-}" ]]; then
  if [[ "$GEMM_GATE_PATCH_FACADE_PATH" == *'"'* || "$GEMM_GATE_PATCH_FACADE_PATH" == *'\'* ]]; then
    echo "ERROR: GEMM_GATE_PATCH_FACADE_PATH に '\"' または '\\' を含めることはできない" >&2
    exit 1
  fi
  CARGO_CONFIG_ARGS+=(--config "patch.crates-io.fandhe-ai.path=\"${GEMM_GATE_PATCH_FACADE_PATH}\"")
fi

OUT=results/raw/results.jsonl
SKIP=results/raw/skipped.log
mkdir -p results/raw
: > "$OUT"
: > "$SKIP"

# PR #1452 codex-review P2 指摘（PRRT_kwDOTuUCJc6gKI3z）: `GEMM_GATE_
# PATCH_FACADE_PATH` 指定時、invocation-only のはずの patch 解決過程で本
# workspace の `Cargo.lock`（承認済みピン固定）が書き換わったまま残る。
# `bench_fandhe_lock_restore.sh` 共有のバックアップ・復元 EXIT trap
# （`run_ab_gemm_metal.sh` と同一設計。#1487 で `bench_fandhe_pin_guard.sh`
# から分離）で必ず元へ戻す（未指定時も無害）。
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

# ビルド失敗時はここで中断する（古い target/release バイナリを現行ツリーの結果として
# 計測・記録しないため。pipefail により tail 越しでも cargo の失敗が伝播する）
if ! cargo build --release "${CARGO_CONFIG_ARGS[@]}" 2>&1 | tail -20; then
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

# (b'') MLP 学習 1 step のフェーズ分解（イシュー #1009。bench-fandhe の
# train タスクのみ対応。(a')/(b') と同じ理由で bench-candle / bench-burn は
# ループに含めない: --phases は必ず MEASURE_ERROR で fail-fast する仕様の
# ため、対象外の 2 バイナリまで含めると既知の対象外失敗が skipped.log の
# 実際の計測失敗と混在し判別しづらくなる）
for dev in cpu metal; do
  run bench-fandhe train "$dev" 64 fresh --phases
  run bench-fandhe train "$dev" 64 reuse --phases
done

# (c') 推論 — デバイス常駐パラメータ更新モード（イシュー #1217。bench-fandhe の
# infer タスクのみ対応。(a')/(b') と同じ理由で bench-candle / bench-burn は
# ループに含めない）
for dev in cpu metal; do
  run bench-fandhe infer "$dev" 64 reuse
done

# (c'') 推論 1 反復のフェーズ分解（イシュー #1217。bench-fandhe の infer
# タスクのみ対応。(b'') と同じ理由で bench-candle / bench-burn はループに
# 含めない）
for dev in cpu metal; do
  run bench-fandhe infer "$dev" 64 fresh --phases
  run bench-fandhe infer "$dev" 64 reuse --phases
done

echo "done. results in $OUT ; failures (if any) in $SKIP"
