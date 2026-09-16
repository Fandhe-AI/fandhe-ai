#!/usr/bin/env bash
# イシュー #1898: `grad_readout_contract_on_metal` の main 上 FAIL
# （回帰窓 e851e91a..565300e4）是正の Metal 実機再実測ランブック
# （`docs/perf/logs/conv-realdevice-1771/run_ignored_tests_metal.sh`・
# `docs/perf/logs/metal-reduce-sum-wiring-1896/run_ignored_tests_metal.sh`
# と同型構成）。
#
# 対象は `crates/facade/tests/device_param_store_backend_parity.rs`・
# `device_param_store_metal_mixed_shape_grad.rs`（facade クレート
# `fandhe-ai` 配下）。`make test-ignored-metal`（`-p fandhe-ai-backend-metal`
# 限定）の対象外のため専用スクリプトとする（`make test-ignored-metal-facade`
# と同一コマンド。単体でも呼べるよう独立させている）。
#
# 使い方（Apple Silicon 実機・リポジトリルートで実行）:
#   docs/perf/logs/grad-readout-contract-1898/run_ignored_tests_metal.sh
#
# 出力: docs/perf/logs/grad-readout-contract-1898/metal/ignored/*.log
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

# 1) device_param_store_backend_parity.rs の Metal 側テスト（is-a
#    `--test-threads=1` 直列。`grad_readout_contract_on_metal`〈本イシュー
#    の是正対象〉・`device_resident_matches_host_sgd_on_metal_across_
#    100_steps`〈非後退〉の 2 件）。
run_case device_param_store_backend_parity_metal \
  cargo test -p fandhe-ai --release --test device_param_store_backend_parity -- --ignored --nocapture --test-threads=1 on_metal

# 2) device_param_store_metal_mixed_shape_grad.rs（イシュー #1898 で
#    bias slot 期待を全 4 slot Some へ更新した回帰テスト）。
run_case device_param_store_metal_mixed_shape_grad \
  cargo test -p fandhe-ai --release --test device_param_store_metal_mixed_shape_grad -- --ignored --nocapture --test-threads=1

echo "==== summary ===="
if [[ $ANY_FAILED -eq 0 ]]; then
  echo "ALL PASS"
else
  echo "FAILED: ${ANY_FAILED} case(s). see ${OUT_DIR}/*.log"
fi
exit $ANY_FAILED
