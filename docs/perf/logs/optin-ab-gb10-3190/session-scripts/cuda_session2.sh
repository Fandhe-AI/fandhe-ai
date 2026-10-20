#!/usr/bin/env bash
# イシュー #3190 GB10 CUDA セッション 2 (第 1 回は CARGO_TARGET_DIR の export がスクリプトの
# bench-fandhe 探索を壊して A/B が 0 件計測で失敗した。計測値は生成されていない。再実行は別 label)。
set -u
export PATH=$HOME/.cargo/bin:$PATH
unset RAYON_NUM_THREADS
unset CARGO_TARGET_DIR
R=$HOME/work/r3190
OUT=$R/cuda-out2
mkdir -p "$OUT"
BEFORE=$R/before
RUNNER=$R/runner
step() { echo "$(date -u +%FT%TZ) $*" >>"$OUT/steps.log"; }

# ---- B3 A/B (別 label)
step "B3 AB start"
(cd "$RUNNER/scripts/bench/framework-compare" && AB_PATCH_FACADE_PATH=$RUNNER/crates/facade bash run_ab_infer_graph_cuda.sh 2115-gb10-3190b >"$OUT/b3-ab.log" 2>&1)
step "B3 AB rc=$?"
# ---- B4 (別の空ディレクトリ)
step "B4 start"
mkdir -p "$OUT/b4"
(cd "$RUNNER/docs/perf/logs/cuda-gemm-small-launch-cost-2109" && LOG_DIR=$OUT/b4 RUN_NSYS=1 ./orchestrate.sh >"$OUT/b4-orch.log" 2>&1)
step "B4 orchestrate rc=$?"
# ---- B6
step "B6 bitmatch start"
(cd "$BEFORE" && CARGO_TARGET_DIR=$R/target-tests cargo test -p fandhe-ai-backend-cuda --release --all-features \
  --test readback_reuse_bit_match_2108 -- --ignored --nocapture --test-threads=1 >"$OUT/b6-bitmatch.log" 2>&1)
step "B6 bitmatch rc=$? passed=$(grep -c '\.\.\. ok' "$OUT/b6-bitmatch.log")"
(cd "$RUNNER/scripts/bench/framework-compare" && AB_READBACK_SIZES=n256 AB_PATCH_FACADE_PATH=$RUNNER/crates/facade bash run_ab_readback_reuse_cuda.sh head-d89a093-3190-n256 >"$OUT/b6-ab.log" 2>&1)
step "B6 AB rc=$?"
# ---- 診断（判定に使わない）: R1(b) を --test-threads=1 で再実行
step "DIAG R1b single-thread start"
(cd "$BEFORE" && FANDHE_AI_CUDA_GRAPH_INFER=1 CARGO_TARGET_DIR=$R/target-tests cargo test -p fandhe-ai --release \
  --test predict_device_chain_cuda_bit_identity -- --ignored --nocapture --test-threads=1 >"$OUT/diag-r1b-single-thread.log" 2>&1)
step "DIAG R1b single-thread rc=$?"
step "ALL DONE"
touch "$R/cuda2.done"
