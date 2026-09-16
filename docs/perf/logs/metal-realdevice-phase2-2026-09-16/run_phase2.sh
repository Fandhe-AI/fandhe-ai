#!/usr/bin/env bash
# facade（metal_ フィルタ）と backend-metal（--all-features）の parity 群を順次実行しログ保存
set -u
export CARGO_TARGET_DIR=$HOME/work/target-fandhe-ai
LOG=docs/perf/logs/metal-realdevice-phase2-2026-09-16
mkdir -p "$LOG"
SUMMARY=$LOG/summary.tsv
: > "$SUMMARY"
run() { # run <log_name> <cmd...>
  local name=$1; shift
  echo "== $name: $*" | tee "$LOG/$name.log"
  "$@" >> "$LOG/$name.log" 2>&1
  local rc=$?
  local res; res=$(grep -E '^test result:' "$LOG/$name.log" | tail -1)
  printf '%s\t%s\t%s\n' "$name" "rc=$rc" "$res" | tee -a "$SUMMARY"
}
BM="cargo test -p fandhe-ai-backend-metal --release --all-features --test"
FA="cargo test -p fandhe-ai --release --test"
for t in sort_topk_parity scan_parity gather_scatter_parity cast_parity unique_parity constant_pad_parity interpolate_parity gemm_batched_parity scalar_op_parity typed_ops_f16_parity typed_ops_bf16_parity; do
  run "backend-metal_$t" $BM $t -- --ignored --nocapture
done
for t in sort_topk_backend_parity index_ops_backend_parity cast_backend_parity unique_backend_parity constant_pad_backend_parity interpolate_backend_parity batched_matmul_backend_parity scalar_ops_backend_parity scalar_unary_transcendental_backend_parity activation_gelu_softplus_backend_parity bce_backend_parity nll_kl_div_backend_parity huber_backend_parity dropout_backend_parity attention_backend_parity mha_backend_parity compat_sequential_layers_backend_parity no_grad_detach_backend_parity backward_accumulate_backend_parity device_transfer_backend_parity; do
  run "facade_$t" $FA $t -- --ignored --nocapture metal_
done
run "facade_device_enumeration" $FA device_enumeration -- --nocapture
echo "ALL DONE $(date -u +%FT%TZ)"
