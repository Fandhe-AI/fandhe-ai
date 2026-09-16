#!/usr/bin/env bash
# イシュー #1896: `MetalBackendOps::sum` 結線・2026-09-16 実機実測
# （#1902。`docs/perf/logs/metal-realdevice-phase2-2026-09-16/README.md`
# §3.1）で判定不能だった 11 テストの再実測ランブック
# （`docs/perf/logs/conv-realdevice-1771/run_ignored_tests_metal.sh` と
# 同型構成）。
#
# 使い方（Apple Silicon 実機・リポジトリルートで実行）:
#   docs/perf/logs/metal-reduce-sum-wiring-1896/run_ignored_tests_metal.sh
#
# 出力: docs/perf/logs/metal-reduce-sum-wiring-1896/metal/ignored/*.log
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

# 0) 非 `#[ignore]` lib／統合テスト（macOS 通常 `cargo test`。pre-push
#    フックと同一）。書き換えた `max_remains_unsupported_*` 群・新規
#    `sum_rejects_out_of_range_dim_before_touching_device` 等が pass
#    することを確認する（R0 相当）。
run_case backend_metal_all_features \
  cargo test -p fandhe-ai-backend-metal --all-features

# 1) reduce_parity（#1895。起動 API 直叩き。bit 完全一致）。
run_case reduce_parity \
  cargo test -p fandhe-ai-backend-metal --release --test reduce_parity -- --ignored --nocapture --test-threads=1

# 2) backend_ops_real_device の新規 sum #[ignore] テスト（#1896。
#    `BackendOps` 経由・0 サイズ契約・非 contiguous・範囲外 dim・NaN・
#    決定性）。
run_case backend_ops_sum_bit_exact \
  cargo test -p fandhe-ai-backend-metal --release --test backend_ops_real_device -- --ignored --nocapture --test-threads=1 backend_ops_sum_matches_cpu_bit_exact

# 3) typed_ops_f16_parity／typed_ops_bf16_parity の新規 sum #[ignore]
#    テスト（#1896）。
run_case typed_ops_f16_sum \
  cargo test -p fandhe-ai-backend-metal --release --test typed_ops_f16_parity -- --ignored --nocapture --test-threads=1 sum_matches_f32_backend_ops_rounded_bit_exact
run_case typed_ops_bf16_sum \
  cargo test -p fandhe-ai-backend-metal --release --test typed_ops_bf16_parity -- --ignored --nocapture --test-threads=1 sum_matches_f32_backend_ops_rounded_bit_exact

# 4) §3.1 由来の 11 テスト（2026-09-16 実測〈#1902〉で `sum` 未実装に
#    より判定不能だったもの。`metal_` フィルタで CUDA 側テストを除外
#    する。`docs/perf/logs/conv-realdevice-1771/run_ignored_tests_metal.sh`
#    と同じ理由）。
run_case conv2d_backend_parity \
  cargo test -p fandhe-ai --release --test conv2d_backend_parity -- --ignored --nocapture --test-threads=1 metal_
run_case conv1d_backend_parity \
  cargo test -p fandhe-ai --release --test conv1d_backend_parity -- --ignored --nocapture --test-threads=1 metal_
run_case nn_conv_backend_parity \
  cargo test -p fandhe-ai --release --test nn_conv_backend_parity -- --ignored --nocapture --test-threads=1 metal_
run_case gather_scatter_parity \
  cargo test -p fandhe-ai-backend-metal --release --test gather_scatter_parity -- --ignored --nocapture --test-threads=1
run_case constant_pad_parity \
  cargo test -p fandhe-ai-backend-metal --release --test constant_pad_parity -- --ignored --nocapture --test-threads=1
run_case interpolate_parity \
  cargo test -p fandhe-ai-backend-metal --release --test interpolate_parity -- --ignored --nocapture --test-threads=1
run_case attention_backend_parity \
  cargo test -p fandhe-ai --release --test attention_backend_parity -- --ignored --nocapture --test-threads=1 metal_
run_case no_grad_detach_backend_parity \
  cargo test -p fandhe-ai --release --test no_grad_detach_backend_parity -- --ignored --nocapture --test-threads=1 metal_
run_case backward_accumulate_backend_parity \
  cargo test -p fandhe-ai --release --test backward_accumulate_backend_parity -- --ignored --nocapture --test-threads=1 metal_

# 5) facade sum／mean parity（本 PR 新設。イシュー #1896）。
run_case reduce_backend_parity_metal \
  cargo test -p fandhe-ai --release --test reduce_backend_parity -- --ignored --nocapture --test-threads=1 metal_

# 6) `make test-ignored-metal` 相当のフル実行（既知 FAIL（#1897／
#    #1898／#1899・`command_batching` 並列干渉）を除き全 pass すること
#    の非後退確認。1 バイナリの FAIL で以降未実行になるのを避けるため
#    `--no-fail-fast` を付ける）。
run_case full_ignored_metal \
  cargo test -p fandhe-ai-backend-metal --release --all-features --no-fail-fast -- --ignored --nocapture

{
  echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "sw_vers: $(sw_vers -productVersion 2>/dev/null || echo unknown)"
  echo "chip: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
  echo "rustc: $(rustc --version)"
  echo "uptime: $(uptime)"
  echo "git_head: $(git rev-parse --short HEAD)"
} >"$OUT_DIR/../env_info.txt"

echo "done. logs in $OUT_DIR"
if [[ "$ANY_FAILED" -gt 0 ]]; then
  echo "FAILED: $ANY_FAILED test group(s) failed; see $OUT_DIR" >&2
  exit 1
fi
