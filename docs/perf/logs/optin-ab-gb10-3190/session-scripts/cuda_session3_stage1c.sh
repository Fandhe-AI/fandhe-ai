#!/usr/bin/env bash
# イシュー #3190 再開分 GB10 CUDA セッション 3 / 段 1c: B5 前提ゲート (a) の再訂正版 (RULE-reopen-3.txt。--no-fail-fast)。
# 段 1b のログは parity-attempt2-*.log へ改名して残す (差し替えない)。
set -u
export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
unset RAYON_NUM_THREADS
unset CARGO_TARGET_DIR
R=$HOME/work/r3190-reopen
OUT=$R/out
step() { echo "$(date -u +%FT%TZ) $*" >>"$OUT/steps-session3.log"; }
for arm in base head; do
  mv "$OUT/b5/parity-$arm.log" "$OUT/b5/parity-attempt2-$arm.log"
done
for arm in base head; do
  step "B5(a)v3 parity $arm start"
  (cd "$R/b5-$arm" && cargo test -p fandhe-ai-backend-cuda --release --features internal-diagnostics --no-fail-fast \
    --test cpu_cuda_mma_parity --test gemm_mma --test parity_nonregression \
    --test specialized_mma_parity --test gemm_auto --test dispatch_boundary \
    -- --ignored --test-threads=1 >"$OUT/b5/parity-$arm.log" 2>&1)
  step "B5(a)v3 parity $arm integration rc=$?"
  (cd "$R/b5-$arm" && cargo test -p fandhe-ai-backend-cuda --release --lib kernels_mma >>"$OUT/b5/parity-$arm.log" 2>&1)
  step "B5(a)v3 parity $arm lib kernels_mma rc=$?"
done
step "STAGE1c DONE"
touch "$R/stage1c.done"
