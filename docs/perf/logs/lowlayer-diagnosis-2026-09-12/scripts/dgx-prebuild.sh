#!/usr/bin/env bash
# DGX: framework-compare（ピン 0.8.0）と backend-cuda 診断テストの事前ビルド（ベンチと分離）
set -u
export PATH="${HOME}/.cargo/bin:/usr/local/cuda/bin:${PATH}"
cd "${HOME}/work/rust-ai-library-run"
echo "start $(date -u +%FT%TZ)"
( cd scripts/bench/framework-compare && cargo build --release -p bench-fandhe 2>&1 | tail -2 && cargo build --release -p bench-candle --no-default-features --features cuda 2>&1 | tail -2 && cargo build --release -p bench-burn --no-default-features --features cuda 2>&1 | tail -2 )
echo "fc-build rc=$?"
cargo test -p fandhe-ai-backend-cuda --release --no-run --test async_ordering_real_device --test tma_probe_real_device --test gemm_transposed_parity --test gemm_transposed_perf 2>&1 | tail -3
echo "diag-build rc=$?"
echo "prebuild-done. $(date -u +%FT%TZ)"
