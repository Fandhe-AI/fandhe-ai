#!/usr/bin/env bash
# イシュー #1771: Conv1d／Conv2d の Metal 実機 parity・実測ランブック
# （`run_ignored_tests_cuda.sh` の Metal 版。対象・判定規則は同一
# README「事前登録判定規則」節を参照）。
#
# 使い方（Apple Silicon 実機・リポジトリルートで実行）:
#   docs/perf/logs/conv-realdevice-1771/run_ignored_tests_metal.sh
#
# 出力: docs/perf/logs/conv-realdevice-1771/metal/ignored/*.log
set -u

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "ERROR: Metal 実機（Darwin arm64）でのみ実行可能（現在: $(uname -sm)）" >&2
  exit 1
fi

SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${SELF_DIR}/metal/ignored"
mkdir -p "$OUT_DIR"
REPO_ROOT="${REPO_ROOT:-$(cd "$SELF_DIR/../../../.." && pwd)}"
cd "$REPO_ROOT" || exit 1

ANY_FAILED=0

run_case() { # run_case <log_name> <cmd...>
  local log_name=$1
  shift
  echo "== $log_name ==" | tee "$OUT_DIR/${log_name}.log"
  if ! "$@" >>"$OUT_DIR/${log_name}.log" 2>&1; then
    echo "  -> FAILED (see $OUT_DIR/${log_name}.log)" | tee -a "$OUT_DIR/${log_name}.log"
    ANY_FAILED=$((ANY_FAILED + 1))
    return 1
  fi
  return 0
}

# 1) im2col／col2im（#1768。bit 完全一致契約）。
#    `--test-threads=1` は run-to-run 決定性検査（README「事前登録判定
#    規則」4)）のため全ケース共通で付与する（並列実行だと `bits=`／
#    `fold_bits=` 行の出力順が起動ごとに変わりうる。PR #1882 レビュー
#    指摘）。
run_case im2col_col2im_parity \
  cargo test -p fandhe-ai-backend-metal --release --test im2col_col2im_parity -- --ignored --nocapture --test-threads=1

# 2) conv2d forward／backward（#1768。`Var::conv2d` 直叩き。REQ-2 複合判定）。
#    `conv2d_backend_parity.rs` は `cuda_*`／`metal_*` の両テストを同一
#    バイナリに含む（`cuda_*` 側は `cfg(target_os = "macos")` で除外
#    されないため、フィルタなしで `--ignored` を実行すると Apple
#    Silicon 上でも CUDA 実機必須テストまで拾ってしまい ANY_FAILED が
#    水増しされる。PR #1882 レビュー指摘）。`metal_` 部分一致フィルタで
#    Metal 側のみへ限定する。
run_case conv2d_backend_parity \
  cargo test -p fandhe-ai --release --test conv2d_backend_parity -- --ignored --nocapture --test-threads=1 metal_

# 3) conv1d forward／backward・「特化」契約（#1769。`Var::conv1d` 直叩き）。
#    同上の理由で `metal_` フィルタを付ける。
run_case conv1d_backend_parity \
  cargo test -p fandhe-ai --release --test conv1d_backend_parity -- --ignored --nocapture --test-threads=1 metal_

# 4) nn 層（compat::Sequential）forward／backward／「特化」契約／学習
#    ループ record-only（イシュー #1771 新設）。同上の理由で `metal_`
#    フィルタを付ける。
run_case nn_conv_backend_parity \
  cargo test -p fandhe-ai --release --test nn_conv_backend_parity -- --ignored --nocapture --test-threads=1 metal_

echo "done. logs in $OUT_DIR"
if [[ "$ANY_FAILED" -gt 0 ]]; then
  echo "FAILED: $ANY_FAILED test group(s) failed; see $OUT_DIR" >&2
  exit 1
fi
