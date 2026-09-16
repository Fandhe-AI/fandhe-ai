#!/usr/bin/env bash
set -u; export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
R=$HOME/work/rust-ai-library-run-1585; LOG=$HOME/work/phase2-1585; mkdir -p "$LOG"; P="$LOG/progress.log"
step(){ echo "== $1 $(date -u +%FT%TZ) load=$(cut -d' ' -f1 /proc/loadavg) gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null)" | tee -a "$P"; }
run(){ local name=$1; shift; step "$name start"; if "$@" >"$LOG/$name.log" 2>&1; then echo "   $name rc=0" >>"$P"; else echo "   $name rc=$? (see $name.log)" >>"$P"; fi; step "$name end"; }
D=$R/docs/perf/logs/cuda-h2d-pinned-staging-1585
# Layer B（5 プロセス起動・record_only: uptime を前後で記録）
run layerB env CARGO_TARGET_DIR=$HOME/work/target-fandhe-ai bash "$D/run_layer_b.sh"
# Layer A（専有ゲート既定 ON。不成立なら 1 回だけ record_only）
FC=$R/scripts/bench/framework-compare
run layerA-gated env AB_PATCH_FACADE_PATH=$R/crates/facade bash -c "cd $FC && bash run_ab_pinned_h2d_cuda.sh pinned-h2d-1585"
if grep -q "undetermined\|gate not met\|GATE" "$LOG/layerA-gated.log" 2>/dev/null && ! ls $FC/compare-pinned-h2d-pinned-h2d-1585-gemm.md >/dev/null 2>&1; then
  echo "   layerA: gate not met -> single record_only rerun" >>"$P"
  run layerA-record_only env AB_LOAD_GATE_MODE=record_only AB_PATCH_FACADE_PATH=$R/crates/facade bash -c "cd $FC && bash run_ab_pinned_h2d_cuda.sh pinned-h2d-1585-ro"
fi
uptime >> "$LOG/uptime_end.txt"
echo "ALL DONE 1585 $(date -u +%FT%TZ)" | tee -a "$P"
