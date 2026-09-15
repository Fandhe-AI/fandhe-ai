#!/usr/bin/env bash
# イシュー #1771: Conv1d／Conv2d の CUDA 実機 parity・実測ランブック。
# 対象 `#[ignore]` テスト（#1766／#1767 の既存分 + #1771 新設の nn
# 層〈compat::Sequential〉backward／conv1d／「特化」契約／学習ループ
# record-only）を個別プロセスで実行しログを保存する
# （`infer-chain-single-sync-cuda-1689/run_ignored_tests.sh` と同型）。
#
# 使い方（GB10 実機・リポジトリルートで実行）:
#   docs/perf/logs/conv-realdevice-1771/run_ignored_tests_cuda.sh
#
# 出力: docs/perf/logs/conv-realdevice-1771/cuda/ignored/*.log
set -u
SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${SELF_DIR}/cuda/ignored"
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

# 1) im2col／col2im（#1766。bit 完全一致契約）。
run_case im2col_col2im_parity \
  cargo test -p fandhe-ai-backend-cuda --release --test im2col_col2im_parity -- --ignored --nocapture

# 2) conv2d forward／backward（#1766。`Var::conv2d` 直叩き。REQ-2 複合判定）。
run_case conv2d_backend_parity \
  cargo test -p fandhe-ai --release --test conv2d_backend_parity -- --ignored --nocapture

# 3) conv1d forward／backward・「特化」契約（#1767。`Var::conv1d` 直叩き）。
run_case conv1d_backend_parity \
  cargo test -p fandhe-ai --release --test conv1d_backend_parity -- --ignored --nocapture

# 4) nn 層（compat::Sequential）forward／backward／「特化」契約／学習
#    ループ record-only（イシュー #1771 新設。同一ファイル内の 6 件を
#    まとめて実行する）。
run_case nn_conv_backend_parity \
  cargo test -p fandhe-ai --release --test nn_conv_backend_parity -- --ignored --nocapture

echo "done. logs in $OUT_DIR"
if [[ "$ANY_FAILED" -gt 0 ]]; then
  echo "FAILED: $ANY_FAILED test group(s) failed; see $OUT_DIR" >&2
  exit 1
fi
