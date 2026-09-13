#!/usr/bin/env bash
# イシュー #1689: CUDA 推論 forward チェーン単一同期化（#1579／#1688）の
# 非後退確認対象 `#[ignore]` テスト群を GB10 実機で個別プロセス実行し、
# ログを保存する。
#
# 実行順（事前登録規則。README「事前登録判定規則」節）:
#   R0（前提ゲート）: linear_forward_device_real_device の 4 件＋
#     record-only bench が green であること。R0 が失敗した場合は R1
#     以降を実行せず「R0 未達」として打ち切る（本スクリプトは非 0 で
#     終了する）
#   R1（chain bit 一致）: predict_device_chain_cuda_bit_identity の
#     `#[ignore]` 5 件
#   その他: 既存回帰（device_param_store_backend_parity の CUDA 対象
#     2 件）
#
# 使い方（GB10 実機・リポジトリルートで実行）:
#   docs/perf/logs/infer-chain-single-sync-cuda-1689/run_ignored_tests.sh
#
# 出力: docs/perf/logs/infer-chain-single-sync-cuda-1689/ignored/*.log
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
    return 1
  fi
  return 0
}

# R0（前提ゲート）: `linear_forward_device_real_device`（イシュー #1216。
# #1688 が新設する chain 経路の基礎カーネル）の `#[ignore]` 4 件。この
# 前提が崩れている場合、chain 経路自体の正しさを検証できないため R1
# 以降を実行せず打ち切る（README「事前登録判定規則」節）。
if ! run_case linear_forward_device_matches_cpu_reference_on_real_device \
    cargo test -p fandhe-ai-backend-cuda --release --test linear_forward_device_real_device -- \
    --ignored --nocapture --exact linear_forward_device_matches_cpu_reference_on_real_device
then
  echo "R0 未達: linear_forward_device_matches_cpu_reference_on_real_device が失敗したため R1 以降を打ち切る" >&2
  echo "done (R0 aborted). logs in $OUT_DIR"
  exit 1
fi
if ! run_case linear_forward_device_matches_gemm_resident_rhs_act_bit_exact_on_real_device \
    cargo test -p fandhe-ai-backend-cuda --release --test linear_forward_device_real_device -- \
    --ignored --nocapture --exact linear_forward_device_matches_gemm_resident_rhs_act_bit_exact_on_real_device
then
  echo "R0 未達: linear_forward_device_matches_gemm_resident_rhs_act_bit_exact_on_real_device が失敗したため R1 以降を打ち切る" >&2
  echo "done (R0 aborted). logs in $OUT_DIR"
  exit 1
fi
if ! run_case linear_forward_device_two_layer_chain_matches_cpu_reference_on_real_device \
    cargo test -p fandhe-ai-backend-cuda --release --test linear_forward_device_real_device -- \
    --ignored --nocapture --exact linear_forward_device_two_layer_chain_matches_cpu_reference_on_real_device
then
  echo "R0 未達: linear_forward_device_two_layer_chain_matches_cpu_reference_on_real_device が失敗したため R1 以降を打ち切る" >&2
  echo "done (R0 aborted). logs in $OUT_DIR"
  exit 1
fi
if ! run_case linear_forward_device_rejects_shape_mismatches_and_handles_empty_input_on_real_device \
    cargo test -p fandhe-ai-backend-cuda --release --test linear_forward_device_real_device -- \
    --ignored --nocapture --exact linear_forward_device_rejects_shape_mismatches_and_handles_empty_input_on_real_device
then
  echo "R0 未達: linear_forward_device_rejects_shape_mismatches_and_handles_empty_input_on_real_device が失敗したため R1 以降を打ち切る" >&2
  echo "done (R0 aborted). logs in $OUT_DIR"
  exit 1
fi
if ! run_case linear_forward_device_bench_cuda \
    cargo test -p fandhe-ai-backend-cuda --release --test linear_forward_device_real_device -- \
    --ignored --nocapture --exact linear_forward_device_bench_cuda
then
  echo "R0 未達: linear_forward_device_bench_cuda が失敗したため R1 以降を打ち切る" >&2
  echo "done (R0 aborted). logs in $OUT_DIR"
  exit 1
fi

# R1（chain bit 一致。イシュー #1689 新設）: `predict_device_chain_cuda_
# bit_identity` の `#[ignore]` 5 件（小形状 2 種＋bench 形状の chain/
# legacy bit 一致・CPU 参照 REQ-2 複合判定・bit ダンプ生成の動作確認）。
run_case predict_device_chain_matches_legacy_path_bit_exact_on_cuda_relu_fusion \
  cargo test -p fandhe-ai --release --test predict_device_chain_cuda_bit_identity -- \
  --ignored --nocapture --exact predict_device_chain_matches_legacy_path_bit_exact_on_cuda_relu_fusion
run_case predict_device_chain_matches_legacy_path_bit_exact_on_cuda_no_activation_fusion \
  cargo test -p fandhe-ai --release --test predict_device_chain_cuda_bit_identity -- \
  --ignored --nocapture --exact predict_device_chain_matches_legacy_path_bit_exact_on_cuda_no_activation_fusion
run_case predict_device_chain_matches_legacy_path_bit_exact_on_cuda_bench_shape \
  cargo test -p fandhe-ai --release --test predict_device_chain_cuda_bit_identity -- \
  --ignored --nocapture --exact predict_device_chain_matches_legacy_path_bit_exact_on_cuda_bench_shape
run_case predict_resident_matches_cpu_reference_on_cuda \
  cargo test -p fandhe-ai --release --test predict_device_chain_cuda_bit_identity -- \
  --ignored --nocapture --exact predict_resident_matches_cpu_reference_on_cuda
run_case predict_resident_bit_dump_cuda \
  cargo test -p fandhe-ai --release --test predict_device_chain_cuda_bit_identity -- \
  --ignored --nocapture --exact predict_resident_bit_dump_cuda

# 既存回帰（`device_param_store_backend_parity` の CUDA 対象 2 件。
# #1569 で resident_capable=true に変わった契約を含む。#1560
# `run_ignored_tests.sh` R1-3 と同じ非後退確認）。
run_case device_param_store_backend_parity_cuda_100steps \
  cargo test -p fandhe-ai --release --test device_param_store_backend_parity -- \
  --ignored --nocapture --exact device_resident_matches_host_sgd_on_cuda_across_100_steps
run_case device_param_store_backend_parity_cuda_grad_readout \
  cargo test -p fandhe-ai --release --test device_param_store_backend_parity -- \
  --ignored --nocapture --exact grad_readout_contract_on_cuda

echo "done. logs in $OUT_DIR"
if [[ "$ANY_FAILED" -gt 0 ]]; then
  echo "FAILED: $ANY_FAILED test group(s) failed; see $OUT_DIR" >&2
  exit 1
fi
