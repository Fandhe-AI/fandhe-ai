#!/bin/bash
# イシュー #1353: CUDA managed memory 配置（#1352。`--managed`）有無の
# GB10 実践規模 A/B 計測。
#
# `bench-fandhe` は既定ビルド（`managed-placement` feature 無効・
# crates.io 公開版 `fandhe-ai =0.7.0` ピン）では `--managed` を常に
# MEASURE_ERROR で拒否する（`set_cuda_managed_memory_enabled` API 自体が
# 0.7.0 ピンに未収録のため。`bench-fandhe/src/main.rs` dispatch 参照）。
# 本スクリプトは `managed-placement` feature を有効化し、かつ
# `AB_PATCH_FACADE_PATH`（未リリースの HEAD `crates/facade` への path
# patch。deps-policy.md 第 9 区分は registry 取得元のみを許容するため、
# この patch は本スクリプトの CLI 引数としてのみ与え、
# `scripts/bench/framework-compare/Cargo.toml`／`.cargo/config.toml` へは
# コミットしない）を要求する。
#
# 呼び出し例（GB10 実機。ユーザー承認・別セッション）:
#   AB_PATCH_FACADE_PATH=/home/<user>/work/rust-ai-library-run-1353/crates/facade \
#     bash run_ab_managed_cuda.sh head-abc1234
#
# 出力は run_ab_train_cuda.sh と同じ「失敗を捏造しない」方針
# （skipped ログへ記録。security.md A08）。
set -u
cd "$(dirname "$0")"

LABEL=${1:-}

# A03 インジェクション対策: ラベルはファイル名・パスに直接埋め込むため、
# 英数字・`._-` のみを許可する allowlist で検証する。
if [[ -z "$LABEL" || ! "$LABEL" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "usage: $0 <label>  (label must match [A-Za-z0-9._-]+, e.g. head-abc1234)" >&2
  echo "  env AB_PATCH_FACADE_PATH=<absolute path to crates/facade> is required" >&2
  exit 1
fi

# `AB_PATCH_FACADE_PATH` の検証（A03・A08）: 未設定・相対パス・
# Cargo.toml 不在・crate 名不一致のいずれかなら fail-closed で exit 1。
# 0.7.0 ピンには `set_cuda_managed_memory_enabled` API 自体が無いため、
# patch なしの計測は「--managed が常に MEASURE_ERROR」という無意味な結果
# にしかならない（bench-fandhe 側の契約テストで別途検証済み）。
if [[ -z "${AB_PATCH_FACADE_PATH:-}" ]]; then
  echo "error: AB_PATCH_FACADE_PATH is required (absolute path to an unreleased HEAD's crates/facade with set_cuda_managed_memory_enabled; issue #1353)" >&2
  exit 1
fi
if [[ "$AB_PATCH_FACADE_PATH" != /* ]]; then
  echo "error: AB_PATCH_FACADE_PATH must be an absolute path (got: $AB_PATCH_FACADE_PATH)" >&2
  exit 1
fi
if [[ ! -f "$AB_PATCH_FACADE_PATH/Cargo.toml" ]]; then
  echo "error: AB_PATCH_FACADE_PATH/Cargo.toml not found ($AB_PATCH_FACADE_PATH)" >&2
  exit 1
fi
if ! grep -qE '^\s*name\s*=\s*"fandhe-ai"\s*$' "$AB_PATCH_FACADE_PATH/Cargo.toml"; then
  echo "error: AB_PATCH_FACADE_PATH/Cargo.toml does not declare name = \"fandhe-ai\" ($AB_PATCH_FACADE_PATH)" >&2
  exit 1
fi
PATCH_CONFIG="patch.crates-io.fandhe-ai.path=\"${AB_PATCH_FACADE_PATH}\""

OUT="results/raw/results-dgx-managed-ab-${LABEL}.jsonl"
SKIP="results/raw/skipped-dgx-managed-ab-${LABEL}.log"
mkdir -p results/raw
: > "$OUT"
: > "$SKIP"

ANY_FAILED=0

# `Cargo.lock` を退避し、異常終了含め終了時に必ず復元する（trap）。
# `[patch]` は CLI 引数（`--config`）のみで与え、`Cargo.lock`／
# `.cargo/config.toml` は変更しない契約（deps-policy.md 第 9 区分）。
LOCK_BACKUP="$(mktemp)"
cp Cargo.lock "$LOCK_BACKUP"
restore_lock() {
  cp "$LOCK_BACKUP" Cargo.lock
  rm -f "$LOCK_BACKUP"
}
trap restore_lock EXIT

echo "== build bench-fandhe (managed-placement feature + path patch) =="
if ! cargo build --release -p bench-fandhe --features managed-placement --config "$PATCH_CONFIG" 2>build-err.tmp; then
  tail -40 build-err.tmp
  echo "bench-fandhe BUILD FAILED: $(tail -3 build-err.tmp | tr '\n' ' ')" >> "$SKIP"
  rm -f build-err.tmp
  echo "done. results in $OUT ; failures (if any) in $SKIP"
  exit 1
fi
rm -f build-err.tmp

# #1166 事故対応と同型のハードゲート: `cargo tree` で `fandhe-ai` が実際に
# path 解決されていることを確認する（registry 解決なら patch が効いて
# おらず、以降の計測が意味を持たないため fail-closed で止める）。
TREE_OUTPUT="$(cargo tree -p bench-fandhe --depth 1 --config "$PATCH_CONFIG" 2>&1)"
if ! echo "$TREE_OUTPUT" | grep -qE 'fandhe-ai v[0-9.]+ \(.*crates/facade\)'; then
  echo "error: fandhe-ai did not resolve to the path-patched crates/facade; cargo tree:" >&2
  echo "$TREE_OUTPUT" >&2
  exit 1
fi

BIN_SHA_BEFORE="$(shasum -a 256 target/release/bench-fandhe 2>/dev/null || sha256sum target/release/bench-fandhe)"
echo "bench-fandhe sha256: $BIN_SHA_BEFORE"

run() { # run <task> <device> <size> <mode> <managed_flag: 0|1> [extra_flag]
  local task=$1 device=$2 size=$3 mode=$4 managed_flag=$5 extra_flag=${6:-}
  local flags=(--task "$task" --device "$device" --size "$size" --mode "$mode" --out "$OUT")
  if [[ "$managed_flag" == "1" ]]; then
    flags+=(--managed)
  fi
  if [[ -n "$extra_flag" ]]; then
    flags+=("$extra_flag")
  fi
  echo "== bench-fandhe $task $device size=$size mode=$mode managed=$managed_flag extra=${extra_flag:-none} =="
  # ビルド直後に記録した sha256 と再照合する（計測中の意図しない再ビルド・
  # バイナリ差し替えを検出する。#1166 事故対応と同型）。
  local sha_now
  sha_now="$(shasum -a 256 target/release/bench-fandhe 2>/dev/null || sha256sum target/release/bench-fandhe)"
  if [[ "$sha_now" != "$BIN_SHA_BEFORE" ]]; then
    echo "error: bench-fandhe binary changed mid-measurement (sha256 mismatch)" >&2
    exit 1
  fi
  if ! ./target/release/bench-fandhe "${flags[@]}" 2>err.tmp; then
    echo "task=$task device=$device size=$size mode=$mode managed=$managed_flag extra=${extra_flag:-none} : $(cat err.tmp)" >> "$SKIP"
    echo "  -> FAILED (recorded in $SKIP)"
    ANY_FAILED=$((ANY_FAILED + 1))
  fi
  rm -f err.tmp
}

# (a) gemm cuda N in {1024,2048,4096} x mode in {fresh,reuse}: 各 run で
# off -> on を連続起動 x 5 run（同一プロセス起動間隔を最小化し熱・クロック
# 偏りを off/on 間で揃える）。
for size in 1024 2048 4096; do
  for mode in fresh reuse; do
    for i in 1 2 3 4 5; do
      run gemm cuda "$size" "$mode" 0
      run gemm cuda "$size" "$mode" 1
    done
  done
done

# (b) train cuda 64 x mode in {fresh,reuse}: 同様 x 5 run。
for mode in fresh reuse; do
  for i in 1 2 3 4 5; do
    run train cuda 64 "$mode" 0
    run train cuda 64 "$mode" 1
  done
done

# (c) infer cuda 64 reuse（副次）x 5 run。
for i in 1 2 3 4 5; do
  run infer cuda 64 reuse 0
  run infer cuda 64 reuse 1
done

# (d) 診断（単発・各 1 回）: train fresh/reuse --phases・gemm 4096 reuse
# --phases・infer reuse --phases を off/on 各 1 回。
run train cuda 64 fresh 0 --phases
run train cuda 64 fresh 1 --phases
run train cuda 64 reuse 0 --phases
run train cuda 64 reuse 1 --phases
run gemm cuda 4096 reuse 0 --phases
run gemm cuda 4096 reuse 1 --phases
run infer cuda 64 reuse 0 --phases
run infer cuda 64 reuse 1 --phases

echo "done. results in $OUT ; failures (if any) in $SKIP"

if [[ "$ANY_FAILED" -gt 0 ]]; then
  echo "FAILED: $ANY_FAILED run(s) failed; see $SKIP" >&2
  exit 1
fi
