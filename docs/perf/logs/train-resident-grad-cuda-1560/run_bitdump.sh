#!/usr/bin/env bash
# イシュー #1560: CUDA resident weight 勾配経路（#1559）の結線前後で
# `crates/facade/tests/cuda_graph_step_bit_identity.rs::eager_baseline`
# （opt-in OFF・`Tape::param_grads_to_host` の戻り値を含む step ごとの
# loss・dinput・grad・param・final.param のビット表現を印字する既存
# テスト。#1480 で追加）を実行し、`^(step\[|final\.param\[)` 行
# （`scripts/verify-cuda-graph-step-bit-identity.sh::extract_bits_lines`
# と同一正規表現）を抽出して before/after ツリー間で diff する。
#
# `eager_baseline` 自体は before（#1569 マージ直前の main）・after
# （本ブランチ）両ツリーに存在する（`a68c08b0` 以降。#1480）ため、新規
# テスト追加は不要——本スクリプトは「同一テストを 2 つの独立ツリーで
# 実行して出力を突き合わせる」オーケストレーションに徹する
# （`verify-cuda-graph-step-bit-identity.sh` が同一ツリー内の
# eager_baseline/graph_capture を比較するのとは axis が異なる）。
#
# 出力ファイル名に「verify」を含めない（#1480 env_info の隔離ガード事故
# 〈同名プレフィックスによる誤収集〉を再発させないため）。
#
# 使い方（GB10 実機）:
#   BEFORE_TREE=/home/<user>/work/rust-ai-library-run-1560-before \
#   AFTER_TREE=/home/<user>/work/rust-ai-library-run-1560-after \
#     ./run_bitdump.sh
set -u
SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${SELF_DIR}/bitdump"
mkdir -p "$OUT_DIR"

if [[ -z "${BEFORE_TREE:-}" || -z "${AFTER_TREE:-}" ]]; then
  echo "usage: BEFORE_TREE=<absolute path> AFTER_TREE=<absolute path> $0" >&2
  exit 1
fi
for v in BEFORE_TREE AFTER_TREE; do
  path="${!v}"
  if [[ "$path" != /* ]]; then
    echo "error: $v must be an absolute path (got: $path)" >&2
    exit 1
  fi
  if [[ ! -f "$path/Cargo.toml" ]]; then
    echo "error: $v/Cargo.toml not found ($path)" >&2
    exit 1
  fi
done

extract_bits_lines() {
  grep -E '^(step\[|final\.param\[)'
}

run_eager_baseline() { # run_eager_baseline <tree_path>
  local tree=$1
  (cd "$tree" && cargo test -p fandhe-ai --release --test cuda_graph_step_bit_identity \
    -- --ignored --nocapture --exact eager_baseline)
}

echo "[1/2] before ツリーで eager_baseline を実行する ($BEFORE_TREE)..." >&2
before_raw="$(run_eager_baseline "$BEFORE_TREE")" && before_rc=0 || before_rc=$?
printf '%s\n' "$before_raw" >"$OUT_DIR/before_raw.log"
if [[ "$before_rc" -ne 0 ]]; then
  echo "エラー: before ツリーの eager_baseline が失敗した（exit ${before_rc}）" >&2
  exit 1
fi
before_bits="$(printf '%s\n' "$before_raw" | extract_bits_lines)"
printf '%s\n' "$before_bits" >"$OUT_DIR/before_bits.txt"
if [[ -z "$before_bits" ]]; then
  echo "エラー: before の出力からビット行を抽出できなかった" >&2
  exit 1
fi

echo "[2/2] after ツリーで eager_baseline を実行する ($AFTER_TREE)..." >&2
after_raw="$(run_eager_baseline "$AFTER_TREE")" && after_rc=0 || after_rc=$?
printf '%s\n' "$after_raw" >"$OUT_DIR/after_raw.log"
if [[ "$after_rc" -ne 0 ]]; then
  echo "エラー: after ツリーの eager_baseline が失敗した（exit ${after_rc}）" >&2
  exit 1
fi
after_bits="$(printf '%s\n' "$after_raw" | extract_bits_lines)"
printf '%s\n' "$after_bits" >"$OUT_DIR/after_bits.txt"
if [[ -z "$after_bits" ]]; then
  echo "エラー: after の出力からビット行を抽出できなかった" >&2
  exit 1
fi

diff_output="$(diff "$OUT_DIR/before_bits.txt" "$OUT_DIR/after_bits.txt")" && diff_rc=0 || diff_rc=$?
printf '%s\n' "$diff_output" >"$OUT_DIR/bitdump_diff.txt"

before_lines="$(wc -l <"$OUT_DIR/before_bits.txt" | tr -d ' ')"
after_lines="$(wc -l <"$OUT_DIR/after_bits.txt" | tr -d ' ')"
echo "before_lines=$before_lines after_lines=$after_lines" >"$OUT_DIR/line_counts.txt"

if [[ "$diff_rc" -eq 0 ]]; then
  echo "OK: before/after は ${before_lines} 行すべて bit 同一（#1480 実績値 4782 行と突合すること）" >&2
  exit 0
elif [[ "$diff_rc" -eq 1 ]]; then
  echo "NG: before/after の出力が bit 同一でない（diff は $OUT_DIR/bitdump_diff.txt を参照）" >&2
  exit 1
else
  echo "エラー: diff の実行自体が失敗した（exit ${diff_rc}）。bit 同一性は判定不能" >&2
  exit "$diff_rc"
fi
