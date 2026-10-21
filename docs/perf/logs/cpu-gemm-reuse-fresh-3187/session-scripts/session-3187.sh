#!/usr/bin/env bash
# #3187 GB10 セッション: (1) 帰属 5 run (2) MINFLT 補助 1 run (3) 大コア affinity A/B（14 セル x 5 round）。
# ビルドを並走させないため直列に実行する。各ステップの rc を steps.log へ追記し、最後に終端マーカーを書く。
export PATH=$HOME/.cargo/bin:$PATH
unset RAYON_NUM_THREADS CARGO_TARGET_DIR AB_LOAD_GATE
R=$HOME/work/r3187
LOG=$R/steps.log
cd "$R/runner" || exit 1
echo "$(date -u +%FT%TZ) start" >>"$LOG"
bash docs/perf/logs/cpu-gemm-reuse-fresh-3187/orchestrate.sh gb10 >"$R/orch-gb10.log" 2>&1
echo "$(date -u +%FT%TZ) orchestrate-gb10 rc=$?" >>"$LOG"
GEMM_REUSE_MINFLT=1 bash docs/perf/logs/cpu-gemm-reuse-fresh-3187/orchestrate.sh gb10 \
  --out "$R/runner/docs/perf/logs/cpu-gemm-reuse-fresh-3187/gb10-minflt" >"$R/orch-minflt.log" 2>&1
echo "$(date -u +%FT%TZ) orchestrate-minflt rc=$?" >>"$LOG"
AB_BEFORE_FACADE_PATH=$R/before/crates/facade AB_AFTER_FACADE_PATH=$R/after/crates/facade \
  bash "$R/runner/scripts/bench/framework-compare/run_ab_gb10_affinity_cpu.sh" 2117-gb10-3187 >"$R/ab-2117.log" 2>&1
echo "$(date -u +%FT%TZ) ab-2117 rc=$?" >>"$LOG"
echo "done." >>"$LOG"
