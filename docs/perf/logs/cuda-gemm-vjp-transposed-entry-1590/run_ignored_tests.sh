#!/usr/bin/env bash
# イシュー #1590: CUDA GEMM VJP NT／TN 転置入口（#1214）の非後退確認対象
# `#[ignore]` テスト群を GB10 実機で個別プロセス実行し、ログを保存する
# （R1。README「対象テスト」節）。
#
# `gemm_transposed_perf` は判定対象ではなく §3.2 の正式補助 A/B と兼用
# するため 5 回個別プロセス起動し、`aggregate_aux_ab.py` へ渡すログを
# 保存する。
#
# 使い方（GB10 実機・リポジトリルートで実行、または本スクリプトを直接
# 実行してもよい）:
#   docs/perf/logs/cuda-gemm-vjp-transposed-entry-1590/run_ignored_tests.sh
#
# 出力: docs/perf/logs/cuda-gemm-vjp-transposed-entry-1590/ignored/*.log
#       docs/perf/logs/cuda-gemm-vjp-transposed-entry-1590/aux/gemm_transposed_perf_run{1..5}.log
set -u
SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${SELF_DIR}/ignored"
AUX_DIR="${SELF_DIR}/aux"
mkdir -p "$OUT_DIR" "$AUX_DIR"
REPO_ROOT="${REPO_ROOT:-$(cd "$SELF_DIR/../../../.." && pwd)}"
cd "$REPO_ROOT" || exit 1

ANY_FAILED=0
AUX_LAUNCHES=${AUX_LAUNCHES:-5}

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

# gemm_transposed_parity（5 件。#1214 の bit 完全一致・REQ-2 複合判定）
run_case gemm_transposed_parity \
  cargo test -p fandhe-ai-backend-cuda --release --test gemm_transposed_parity -- \
  --ignored --nocapture --test-threads=1

# gemm_transposed_perf（2 件。§3.2 の正式補助 A/B 兼用のため 5 回
# 個別プロセス起動して `aux/` へ保存する）
for i in $(seq 1 "$AUX_LAUNCHES"); do
  echo "== gemm_transposed_perf run$i ==" | tee "$AUX_DIR/gemm_transposed_perf_run${i}.log"
  if ! cargo test -p fandhe-ai-backend-cuda --release --test gemm_transposed_perf -- \
      --ignored --nocapture --test-threads=1 >>"$AUX_DIR/gemm_transposed_perf_run${i}.log" 2>&1; then
    echo "  -> FAILED (see $AUX_DIR/gemm_transposed_perf_run${i}.log)" | tee -a "$AUX_DIR/gemm_transposed_perf_run${i}.log"
    ANY_FAILED=$((ANY_FAILED + 1))
  fi
done

# gemm_fp32_strict_into_parity（#1559 の NT/TN 経路が #1214 の入口の上に
# 成立していることの非後退確認。存在しない場合は skip 扱いにせず失敗と
# して記録する: テストファイル自体が消えているのは重大な回帰のため）
run_case gemm_fp32_strict_into_parity \
  cargo test -p fandhe-ai-backend-cuda --release --test gemm_fp32_strict_into_parity -- \
  --ignored --nocapture --test-threads=1

# transpose_parity（GPU 側 smem 転置カーネル自体の非後退確認）
run_case transpose_parity \
  cargo test -p fandhe-ai-backend-cuda --release --test transpose_parity -- \
  --ignored --nocapture --test-threads=1

# repack_count_tests（`--lib`。env-adaptive。GB10 実機では実経路で走り
# NT/TN ルーティング健全性の到達確認になる。`pub(crate)` カウンタへは
# 触れない）
run_case repack_count_tests \
  cargo test -p fandhe-ai-backend-cuda --release --lib repack_count_tests -- --nocapture

# device_param_store_backend_parity（CUDA 対象 2 件。既存回帰の非後退
# 確認。#1560／#1689 と同じ対象）
run_case device_param_store_backend_parity_cuda_100steps \
  cargo test -p fandhe-ai --release --test device_param_store_backend_parity -- \
  --ignored --nocapture --exact device_resident_matches_host_sgd_on_cuda_across_100_steps
run_case device_param_store_backend_parity_cuda_grad_readout \
  cargo test -p fandhe-ai --release --test device_param_store_backend_parity -- \
  --ignored --nocapture --exact grad_readout_contract_on_cuda

echo "done. logs in $OUT_DIR (ignored) / $AUX_DIR (aux 5-launch)"
if [[ "$ANY_FAILED" -gt 0 ]]; then
  echo "FAILED: $ANY_FAILED test group(s) failed; see $OUT_DIR / $AUX_DIR" >&2
  exit 1
fi
