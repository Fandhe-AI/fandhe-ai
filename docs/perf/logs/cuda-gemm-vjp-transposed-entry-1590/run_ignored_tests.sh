#!/usr/bin/env bash
# イシュー #1590: CUDA GEMM VJP NT／TN 転置入口（#1214）の非後退確認対象
# `#[ignore]` テスト群を GB10 実機で個別プロセス実行し、ログを保存する
# （R1。README「対象テスト」節）。
#
# `gemm_transposed_perf` は判定対象ではなく §3.2 の正式補助 A/B と兼用
# するため 5 回個別プロセス起動し、`aggregate_aux_ab.py` へ渡すログを
# 保存する。**この 5 起動は R1（本スクリプトの他のテスト群）とは別の
# ツリーで実行しなければならない**（PR #1812 Cursor Bugbot 指摘）:
# R1 の一部（`gemm_fp32_strict_into_parity` 等）は #1214 マージコミット
# （`ab0b77d0`）より後発の API に依存するため HEAD（本ブランチ・
# `REPO_ROOT`）でしか実行できない一方、§3.2 の正式補助 A/B は #1214
# マージコミット自身のツリー（`ab0b77d0`。README「比較対象 2 腕」の
# after 腕）で計測しなければ、post-#1214 の CUDA 変更が混入した数値を
# 正式値として記録してしまう。そのため両者を `AUX_TREE`（省略可。after
# ツリーの絶対パス）で明示的に分離する。
#
# 使い方（GB10 実機・リポジトリルートで実行、または本スクリプトを直接
# 実行してもよい）:
#   AUX_TREE=/absolute/path/to/after \
#     docs/perf/logs/cuda-gemm-vjp-transposed-entry-1590/run_ignored_tests.sh
#
#   `AUX_TREE` を省略すると R1（`ignored/`）のみを実行し、正式補助 A/B
#   （`aux/`）は fail-closed でスキップする（HEAD で代用して事実と異なる
#   系列を正式値として記録することを防ぐため。security.md A08）。
#
# 出力: docs/perf/logs/cuda-gemm-vjp-transposed-entry-1590/ignored/*.log
#       docs/perf/logs/cuda-gemm-vjp-transposed-entry-1590/aux/gemm_transposed_perf_run{1..5}.log
#       （`AUX_TREE` 指定時のみ生成）
set -u
SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${SELF_DIR}/ignored"
AUX_DIR="${SELF_DIR}/aux"
mkdir -p "$OUT_DIR" "$AUX_DIR"
REPO_ROOT="${REPO_ROOT:-$(cd "$SELF_DIR/../../../.." && pwd)}"
cd "$REPO_ROOT" || exit 1

ANY_FAILED=0
AUX_LAUNCHES=${AUX_LAUNCHES:-5}
AUX_TREE="${AUX_TREE:-}"
if [[ -n "$AUX_TREE" ]]; then
  if [[ "$AUX_TREE" != /* ]]; then
    echo "error: AUX_TREE must be an absolute path (got: $AUX_TREE)" >&2
    exit 1
  fi
  if [[ ! -f "$AUX_TREE/crates/backend-cuda/tests/gemm_transposed_perf.rs" ]]; then
    echo "error: AUX_TREE/crates/backend-cuda/tests/gemm_transposed_perf.rs not found ($AUX_TREE); this must be the #1214 merge-commit tree (after 腕)" >&2
    exit 1
  fi
fi

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
# 個別プロセス起動して `aux/` へ保存する）。R1 とは異なりこの 5 起動は
# **after ツリー（`AUX_TREE`。#1214 マージコミット `ab0b77d0` 自身）で
# 実行しなければならない**（本スクリプトの他の R1 ケースは HEAD 限定の
# ため、REPO_ROOT 自体を after へ切り替える方式は採らない。PR #1812
# Cursor Bugbot 指摘）。`AUX_TREE` 未指定時は HEAD の数値を正式値として
# 記録することを避けるため fail-closed でスキップする。
if [[ -z "$AUX_TREE" ]]; then
  echo "skip: gemm_transposed_perf の正式補助 A/B（aux/）は AUX_TREE 未指定のためスキップ（AUX_TREE=<after ツリー絶対パス> を指定して再実行すること）" | tee "$AUX_DIR/SKIPPED.txt"
else
  for i in $(seq 1 "$AUX_LAUNCHES"); do
    echo "== gemm_transposed_perf run$i == (tree=$AUX_TREE)" | tee "$AUX_DIR/gemm_transposed_perf_run${i}.log"
    if ! ( cd "$AUX_TREE" && cargo test -p fandhe-ai-backend-cuda --release --test gemm_transposed_perf -- \
        --ignored --nocapture --test-threads=1 ) >>"$AUX_DIR/gemm_transposed_perf_run${i}.log" 2>&1; then
      echo "  -> FAILED (see $AUX_DIR/gemm_transposed_perf_run${i}.log)" | tee -a "$AUX_DIR/gemm_transposed_perf_run${i}.log"
      ANY_FAILED=$((ANY_FAILED + 1))
    fi
  done
fi

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
