#!/usr/bin/env bash
# イシュー #3190 GB10 CUDA セッション (B3 -> B4 -> B6)。ノード上で setsid nohup 起動する。
# 各ステップの rc を steps.log に残し、失敗しても次へ進む（判定は RULE のとおり機械的に当てはめる）。
set -u
export PATH=$HOME/.cargo/bin:$PATH
unset RAYON_NUM_THREADS
R=$HOME/work/r3190
OUT=$R/cuda-out
mkdir -p "$OUT"
export CARGO_TARGET_DIR=$R/target-tests
BEFORE=$R/before
RUNNER=$R/runner
step() { echo "$(date -u +%FT%TZ) $*" >>"$OUT/steps.log"; }

cd "$BEFORE" || exit 1
# ---- B3 R0
step "B3 R0 start"
cargo test -p fandhe-ai-backend-cuda --release --test linear_forward_device_real_device -- --ignored --nocapture >"$OUT/b3-r0-linear.log" 2>&1
step "B3 R0 linear rc=$?"
cargo test -p fandhe-ai-backend-cuda --release --test graph_capture_real_device -- --ignored --nocapture --test-threads=1 >"$OUT/b3-r0-graph.log" 2>&1
step "B3 R0 graph rc=$?"
# ---- B3 R1
FANDHE_AI_CUDA_GRAPH_INFER=1 cargo test -p fandhe-ai-backend-cuda --release \
  --test infer_graph_capture_real_device -- --ignored --nocapture --test-threads=1 >"$OUT/b3-r1-capture.log" 2>&1
step "B3 R1 capture rc=$? pass_lines=$(grep -c 'INFER_CAPTURE_R1: PASS' "$OUT/b3-r1-capture.log")"
FANDHE_AI_CUDA_GRAPH_INFER=1 cargo test -p fandhe-ai --release \
  --test predict_device_chain_cuda_bit_identity -- --ignored --nocapture >"$OUT/b3-r1-bitid.log" 2>&1
step "B3 R1 bitid rc=$?"
# ---- B3 R2
for b in 64 1024 4096; do
  ok=1
  for mode in off on; do
    if [ "$mode" = on ]; then export FANDHE_AI_CUDA_GRAPH_INFER=1; else unset FANDHE_AI_CUDA_GRAPH_INFER; fi
    if ! cargo test -p fandhe-ai --release --test predict_device_chain_cuda_graph_bit_identity \
        -- --ignored --exact --nocapture "predict_resident_bit_dump_cuda_graph_batch_$b" \
        >"$OUT/b3-r2-raw-$mode-$b.txt" 2>&1; then
      step "R2-FAIL batch=$b: test failed mode=$mode"; ok=0; break
    fi
    grep '^out\[' "$OUT/b3-r2-raw-$mode-$b.txt" >"$OUT/b3-r2-dump-$mode-$b.txt"
    if [ "$(wc -l <"$OUT/b3-r2-dump-$mode-$b.txt")" -ne $((b * 10)) ]; then
      step "R2-FAIL batch=$b: dump 行数不正 mode=$mode"; ok=0; break
    fi
  done
  if [ "$ok" = 1 ]; then
    if diff "$OUT/b3-r2-dump-off-$b.txt" "$OUT/b3-r2-dump-on-$b.txt" >/dev/null; then step "R2-PASS batch=$b"; else step "R2-FAIL batch=$b: diff"; fi
  fi
done
unset FANDHE_AI_CUDA_GRAPH_INFER
# ---- B3 A/B
step "B3 AB start"
(cd "$RUNNER/scripts/bench/framework-compare" && AB_PATCH_FACADE_PATH=$BEFORE/crates/facade bash run_ab_infer_graph_cuda.sh 2115-gb10-3190 >"$OUT/b3-ab.log" 2>&1)
step "B3 AB rc=$?"
# ---- B4
step "B4 start"
mkdir -p "$OUT/b4"
(cd "$RUNNER/docs/perf/logs/cuda-gemm-small-launch-cost-2109" && LOG_DIR=$OUT/b4 RUN_NSYS=1 ./orchestrate.sh >"$OUT/b4-orch.log" 2>&1)
step "B4 orchestrate rc=$?"
# aggregate.py は未マスクパスを fail-closed で拒否するため、マスク後にローカルで実行する
# ---- B6
step "B6 bitmatch start"
cargo test -p fandhe-ai-backend-cuda --release --test readback_reuse_bit_match_2108 -- --ignored --nocapture >"$OUT/b6-bitmatch.log" 2>&1
step "B6 bitmatch rc=$?"
(cd "$RUNNER/scripts/bench/framework-compare" && AB_READBACK_SIZES=n256 AB_PATCH_FACADE_PATH=$BEFORE/crates/facade bash run_ab_readback_reuse_cuda.sh head-d89a093-3190-n256 >"$OUT/b6-ab.log" 2>&1)
step "B6 AB rc=$?"
step "ALL DONE"
touch "$R/cuda.done"
