#!/usr/bin/env bash
# イシュー #3190 再開分 GB10 CUDA セッション 3 / 段 1: B4 再計測 -> B5 前提ゲート (a) parity -> (b) ptxas。
# CARGO_TARGET_DIR は export しない (第 1 セッションの事故の再発防止)。ビルドと計測は直列。
set -u
export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
unset RAYON_NUM_THREADS
unset CARGO_TARGET_DIR
R=$HOME/work/r3190-reopen
OUT=$R/out
mkdir -p "$OUT/b5"
step() { echo "$(date -u +%FT%TZ) $*" >>"$OUT/steps-session3.log"; }

step "B4 (gb10-r2) orchestrate start"
(cd "$R/runner/docs/perf/logs/cuda-gemm-small-launch-cost-2109" && LOG_DIR=$OUT/b4 RUN_NSYS=1 ./orchestrate.sh >"$OUT/b4-orch.log" 2>&1)
step "B4 orchestrate rc=$?"

for arm in base head; do
  step "B5(a) parity $arm start"
  (cd "$R/b5-$arm" && cargo test -p fandhe-ai-backend-cuda --release \
    --test cpu_cuda_mma_parity --test gemm_mma --test parity_nonregression \
    --test specialized_mma_parity --test gemm_auto --test dispatch_boundary \
    -- --ignored --test-threads=1 >"$OUT/b5/parity-$arm.log" 2>&1)
  step "B5(a) parity $arm integration rc=$?"
  (cd "$R/b5-$arm" && cargo test -p fandhe-ai-backend-cuda --release --lib kernels_mma >>"$OUT/b5/parity-$arm.log" 2>&1)
  step "B5(a) parity $arm lib kernels_mma rc=$?"
done

for arm in base head; do
  step "B5(b) ptxas $arm start"
  mkdir -p "$OUT/ptx-$arm"
  (cd "$R/b5-$arm" && cargo run -p fandhe-ai-backend-cuda --release --example mma_ptx_dump \
    --features internal-diagnostics -- --out-dir "$OUT/ptx-$arm" >"$OUT/b5/ptxdump-$arm.log" 2>&1)
  step "B5(b) ptx dump $arm rc=$?"
  ptxas -arch=sm_121 -v "$OUT/ptx-$arm/mma_f16_base.ptx" -o /dev/null >"$OUT/b5/ptxas-$arm.log" 2>&1
  step "B5(b) ptxas $arm rc=$?"
done
step "STAGE1 DONE"
touch "$R/stage1.done"
