#!/usr/bin/env bash
# イシュー #1898: `grad_readout_contract_on_metal` の是正（AC 3。CUDA 側
# 非後退確認）用の GB10 実機再実測ランブック。
#
# CUDA 側の bias slot 契約は本イシューで変更していない
# （`StrictBiasExpectation::HostRouted`。weight slot は `Some`・bias slot
# は `None` のまま。#1559 と同じ挙動）。`grad_readout_contract_on_cuda`
# が引き続き pass することのみを確認する（non-regression）。
#
# 使い方（DGX Spark GB10 等・リポジトリルートで実行）:
#   docs/perf/logs/grad-readout-contract-1898/run_ignored_tests_cuda.sh
#
# 出力: docs/perf/logs/grad-readout-contract-1898/cuda/ignored/*.log
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

run_case device_param_store_backend_parity_cuda \
  cargo test -p fandhe-ai --release --test device_param_store_backend_parity -- --ignored --nocapture --test-threads=1 on_cuda

echo "==== summary ===="
if [[ $ANY_FAILED -eq 0 ]]; then
  echo "ALL PASS"
else
  echo "FAILED: ${ANY_FAILED} case(s). see ${OUT_DIR}/*.log"
fi
exit $ANY_FAILED
