#!/usr/bin/env bash
# イシュー #1689: CUDA 推論 forward チェーン単一同期化（#1579／#1688）の
# before/after 2 ツリー間で `predict_resident_bit_dump_cuda`（bench 形状
# の `predict_resident` 出力を `out[<i>].bits=<hex>` 形式で 1 要素 1 行
# 印字する `#[ignore]` テスト。`crates/facade/tests/predict_device_chain_
# cuda_bit_identity.rs`）を実行し、`^out\[` 行を抽出して diff する（R2）。
#
# `predict_device_chain_cuda_bit_identity.rs` は本イシューで新設した
# ファイルのため before ツリー（`edb85c43`）には存在しない。本スクリプト
# は after ツリーの同ファイルを before ツリーの同パスへコピーしてから
# 実行する（公開 API のみに依存するテストのため、before ツリーの
# `crates/facade`／`bench-harness` 公開面で問題なくコンパイルできる想定。
# `docs/perf/logs/train-resident-grad-cuda-1560/run_bitdump.sh` は逆に
# 「両ツリーに既存」のテストを比較する方式だったため、本イシューでは
# コピー手順が追加で必要になる点が異なる）。
#
# 期待行数は `crates/facade/tests/predict_device_chain_cuda_bit_
# identity.rs::bench_shape_model_and_input` の形状（`BATCH=64`・
# `D_OUT=10`）から機械的に決まる 640 行（= 64 * 10）。
#
# 使い方（GB10 実機）:
#   BEFORE_TREE=/home/<user>/work/rust-ai-library-run-1689-before \
#   AFTER_TREE=/home/<user>/work/rust-ai-library-run-1689-after \
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

TEST_REL_PATH="crates/facade/tests/predict_device_chain_cuda_bit_identity.rs"
if [[ ! -f "$AFTER_TREE/$TEST_REL_PATH" ]]; then
  echo "error: after ツリーに $TEST_REL_PATH が見つからない（$AFTER_TREE）" >&2
  exit 1
fi
if [[ ! -f "$BEFORE_TREE/$TEST_REL_PATH" ]]; then
  echo "before ツリーに $TEST_REL_PATH が存在しないため after ツリーからコピーする（イシュー #1689 新設ファイル）" >&2
  mkdir -p "$(dirname "$BEFORE_TREE/$TEST_REL_PATH")"
  cp "$AFTER_TREE/$TEST_REL_PATH" "$BEFORE_TREE/$TEST_REL_PATH"
fi

extract_bits_lines() {
  grep -E '^out\['
}

run_bit_dump() { # run_bit_dump <tree_path>
  local tree=$1
  (cd "$tree" && cargo test -p fandhe-ai --release --test predict_device_chain_cuda_bit_identity \
    -- --ignored --nocapture --exact predict_resident_bit_dump_cuda)
}

echo "[1/2] before ツリーで predict_resident_bit_dump_cuda を実行する ($BEFORE_TREE)..." >&2
before_raw="$(run_bit_dump "$BEFORE_TREE")" && before_rc=0 || before_rc=$?
printf '%s\n' "$before_raw" >"$OUT_DIR/before_raw.log"
if [[ "$before_rc" -ne 0 ]]; then
  echo "エラー: before ツリーの predict_resident_bit_dump_cuda が失敗した（exit ${before_rc}）" >&2
  exit 1
fi
before_bits="$(printf '%s\n' "$before_raw" | extract_bits_lines)"
printf '%s\n' "$before_bits" >"$OUT_DIR/before_bits.txt"
if [[ -z "$before_bits" ]]; then
  echo "エラー: before の出力からビット行を抽出できなかった" >&2
  exit 1
fi

echo "[2/2] after ツリーで predict_resident_bit_dump_cuda を実行する ($AFTER_TREE)..." >&2
after_raw="$(run_bit_dump "$AFTER_TREE")" && after_rc=0 || after_rc=$?
printf '%s\n' "$after_raw" >"$OUT_DIR/after_raw.log"
if [[ "$after_rc" -ne 0 ]]; then
  echo "エラー: after ツリーの predict_resident_bit_dump_cuda が失敗した（exit ${after_rc}）" >&2
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

# 受け入れ条件は「640 行・差分 0」（`BATCH=64 * D_OUT=10`）。diff が 0
# （テキストとして一致）でも、before/after が同一箇所で行を欠落させた
# まま揃っていれば `diff` は差分なしと報告してしまう（#1560 `run_
# bitdump.sh` と同じ懸念・codex-review [P2] 指摘の踏襲）。成功と判定
# する前に両ファイルが期待行数 640 と一致することを検査し、一致しなけ
# れば「diff は 0 でも判定不能」として非 0 終了にする。
readonly EXPECTED_BITDUMP_LINES=640
if [[ "$before_lines" -ne "$EXPECTED_BITDUMP_LINES" || "$after_lines" -ne "$EXPECTED_BITDUMP_LINES" ]]; then
  echo "判定不能: before_lines=${before_lines} after_lines=${after_lines} が期待行数 ${EXPECTED_BITDUMP_LINES} と一致しない（diff_rc=${diff_rc}。$OUT_DIR/line_counts.txt を参照）" >&2
  exit 2
fi

if [[ "$diff_rc" -eq 0 ]]; then
  echo "OK: before/after は ${before_lines} 行すべて bit 同一" >&2
  exit 0
elif [[ "$diff_rc" -eq 1 ]]; then
  echo "NG: before/after の出力が bit 同一でない（diff は $OUT_DIR/bitdump_diff.txt を参照）" >&2
  exit 1
else
  echo "エラー: diff の実行自体が失敗した（exit ${diff_rc}）。bit 同一性は判定不能" >&2
  exit "$diff_rc"
fi
