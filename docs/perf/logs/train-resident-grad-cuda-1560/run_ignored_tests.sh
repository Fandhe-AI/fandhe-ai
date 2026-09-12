#!/usr/bin/env bash
# イシュー #1560: #1559（CUDA resident weight 勾配経路）の非後退確認
# 対象 `#[ignore]` テスト群を GB10 実機で個別プロセス実行し、ログを
# 保存する（R1）。`--exact` が必要なものは個別プロセス、
# `graph_capture_real_device*` は `--test-threads=1`（同一 GPU への
# 複数 CUDA コンテキスト初期化を避けるため。既存規約）。
#
# 使い方（GB10 実機・リポジトリルートで実行）:
#   docs/perf/logs/train-resident-grad-cuda-1560/run_ignored_tests.sh
#
# 出力: docs/perf/logs/train-resident-grad-cuda-1560/ignored/*.log
set -u
SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${SELF_DIR}/ignored"
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
  fi
}

# R1-1: gemm_fp32_strict_into_parity（#1559 新規・4 件）
run_case gemm_fp32_strict_into_parity \
  cargo test -p fandhe-ai-backend-cuda --release --test gemm_fp32_strict_into_parity -- --ignored --nocapture

# R1-2: gemm_transposed_parity（NT/TN 入口の非後退・5 件）
run_case gemm_transposed_parity \
  cargo test -p fandhe-ai-backend-cuda --release --test gemm_transposed_parity -- --ignored --nocapture

# R1-3: device_param_store_backend_parity（CUDA 対象 2 件。#1569 で
# resident_capable=true に変わった契約を含む）
run_case device_param_store_backend_parity_cuda_100steps \
  cargo test -p fandhe-ai --release --test device_param_store_backend_parity -- \
  --ignored --nocapture --exact device_resident_matches_host_sgd_on_cuda_across_100_steps
run_case device_param_store_backend_parity_cuda_grad_readout \
  cargo test -p fandhe-ai --release --test device_param_store_backend_parity -- \
  --ignored --nocapture --exact grad_readout_contract_on_cuda

# R1-4: cuda_graph_step_bit_identity（eager_baseline は run_bitdump.sh が
# before/after 比較として別途実行するため、ここでは after ツリー単体の
# 非後退確認として 3 テストを実行する）
run_case cuda_graph_step_bit_identity_eager_baseline \
  cargo test -p fandhe-ai --release --test cuda_graph_step_bit_identity -- \
  --ignored --nocapture --exact eager_baseline
run_case cuda_graph_step_bit_identity_graph_capture \
  env FANDHE_AI_CUDA_GRAPH_STEP=1 cargo test -p fandhe-ai --release --test cuda_graph_step_bit_identity -- \
  --ignored --nocapture --exact graph_capture
run_case cuda_graph_step_bit_identity_graph_capture_loop \
  cargo test -p fandhe-ai --release --test cuda_graph_step_bit_identity -- \
  --ignored --nocapture --exact graph_capture_completes_training_loop_without_error

# R1-5: graph_capture_real_device 系（backend-cuda。`graph_capture_
# real_device.rs`／`graph_capture_real_device_optin_off.rs` の 2 バイナリ。
# 同一 GPU への複数 CUDA コンテキスト初期化を避けるため
# `--test-threads=1` はテストランナー引数として `--` の後に置く）
run_case graph_capture_real_device \
  cargo test -p fandhe-ai-backend-cuda --release --test graph_capture_real_device -- --ignored --nocapture --test-threads=1
run_case graph_capture_real_device_optin_off \
  cargo test -p fandhe-ai-backend-cuda --release --test graph_capture_real_device_optin_off -- --ignored --nocapture --test-threads=1

# R1-6: memory.rs の upload_into 系ユニット（非 ignore・実機ライブラリテスト）
run_case backend_cuda_upload_into_unit \
  cargo test -p fandhe-ai-backend-cuda --release --lib -- upload_into

echo "done. logs in $OUT_DIR"
if [[ "$ANY_FAILED" -gt 0 ]]; then
  echo "FAILED: $ANY_FAILED test group(s) failed; see $OUT_DIR" >&2
  exit 1
fi
