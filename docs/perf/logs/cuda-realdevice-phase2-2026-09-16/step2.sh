#!/usr/bin/env bash
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH CARGO_TARGET_DIR=$HOME/work/target-fandhe-ai
cd ~/work/rust-ai-library-run || exit 1
until grep -q "^done\." ~/work/cuda-phase2/make-test-ignored-cuda.log; do sleep 20; done
O=$HOME/work/cuda-phase2; S=$O/summary.tsv; : > $S
run() { local name=$1; shift; echo "== $name: $*" | tee "$O/$name.log"; "$@" >> "$O/$name.log" 2>&1; local rc=$?; local res; res=$(grep -E "^test result:" "$O/$name.log" | tail -1); printf "%s\t%s\t%s\n" "$name" "rc=$rc" "$res" | tee -a $S; }
echo "### step2 start $(date -u +%FT%TZ)"; uptime; nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader
# 2a conv runbook (2 runs + determinism)
C=docs/perf/logs/conv-realdevice-1771
rm -rf "${C}/cuda/ignored"; bash $C/run_ignored_tests_cuda.sh; echo "conv run1 rc=$?"; rm -rf "${C}/cuda/ignored-run1"; mv $C/cuda/ignored $C/cuda/ignored-run1
bash $C/run_ignored_tests_cuda.sh; echo "conv run2 rc=$?"; rm -rf "${C}/cuda/ignored-run2"; mv $C/cuda/ignored $C/cuda/ignored-run2
bash $C/check_determinism.sh --expect-logs im2col_col2im_parity,conv2d_backend_parity,conv1d_backend_parity,nn_conv_backend_parity $C/cuda/ignored-run1 $C/cuda/ignored-run2 > $C/cuda/check_determinism.log 2>&1; echo "determinism rc=$?"; cat $C/cuda/check_determinism.log
# 2b batchnorm
run backend-cuda_batch_norm_parity cargo test -p fandhe-ai-backend-cuda --release --all-features --test batch_norm_parity -- --ignored --nocapture
run facade_batch_norm_backend_parity cargo test -p fandhe-ai --release --test batch_norm_backend_parity -- --ignored --nocapture cuda_
# 2c pooling
run backend-cuda_pooling_real_device cargo test -p fandhe-ai-backend-cuda --release --all-features --lib pooling::pooling_real_device_tests -- --ignored --nocapture
run facade_pooling_backend_parity cargo test -p fandhe-ai --release --test pooling_backend_parity -- --ignored --nocapture cuda_
# 2d backend-cuda
for t in sort_topk_parity scan_parity gather_scatter_parity cast_parity cast_ops_contract unique_parity constant_pad_parity interpolate_parity gemm_batched_parity typed_ops_f16_parity typed_ops_bf16_parity typed_ops_f64_contract rnn_cell_parity rmsnorm_parity rmsnorm_backward_parity scalar_op_parity mse_parity where_masked_fill_parity; do
  run backend-cuda_$t cargo test -p fandhe-ai-backend-cuda --release --all-features --test $t -- --ignored --nocapture
done
run backend-cuda_lib_gemm_batched cargo test -p fandhe-ai-backend-cuda --release --all-features --lib -- --ignored --nocapture gemm_batched
# 2e facade
for t in sort_topk_backend_parity index_ops_backend_parity cast_backend_parity unique_backend_parity constant_pad_backend_parity interpolate_backend_parity batched_matmul_backend_parity scalar_ops_backend_parity scalar_unary_transcendental_backend_parity activation_gelu_softplus_backend_parity bce_backend_parity nll_kl_div_backend_parity huber_backend_parity dropout_backend_parity attention_backend_parity mha_backend_parity compat_sequential_layers_backend_parity no_grad_detach_backend_parity backward_accumulate_backend_parity norm_backend_parity var_norm_backend_parity device_transfer_backend_parity scan_ops_backend_parity; do
  run facade_$t cargo test -p fandhe-ai --release --test $t -- --ignored --nocapture cuda_
done
run facade_device_enumeration cargo test -p fandhe-ai --release --test device_enumeration -- --nocapture
{ echo "hostname: masked"; echo "date: $(date -u +%FT%TZ)"; echo "rev: $(cat .rev-stamp)"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; nvcc --version | tail -2; rustc -V; cargo -V; uname -srm; uptime; nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader; } > $O/env_info.txt
echo "### step2 done $(date -u +%FT%TZ)"; echo "done2."
