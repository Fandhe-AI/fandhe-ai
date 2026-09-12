#!/usr/bin/env bash
# DGX Spark GB10 実測キャンペーン（2026-09-12）。事前ビルド済み前提・ビルドとベンチを重ねない。
# 1) framework-compare 全セル（ピン fandhe-ai =0.8.0）  2) 診断テスト  3) CPU train/infer の RAYON_NUM_THREADS スイープ
set -u
export PATH="${HOME}/.cargo/bin:/usr/local/cuda/bin:${PATH}"
ROOT="${HOME}/work/rust-ai-library-run"
FC="${ROOT}/scripts/bench/framework-compare"
LOGD="${HOME}/work/dgx-0.8.0-logs"; mkdir -p "${LOGD}"
cd "${FC}"
echo "start $(date -u +%FT%TZ)"; uptime; nvidia-smi --query-gpu=utilization.gpu,temperature.gpu --format=csv,noheader
# --- 1) 全セル
uptime > "${LOGD}/uptime_before_all.txt"
bash ./run_all_cuda.sh > "${LOGD}/run_all_cuda.log" 2>&1
cp results/raw/results-cuda.jsonl "${LOGD}/results-dgx-0.8.0.jsonl"; cp results/raw/skipped-cuda.log "${LOGD}/skipped-dgx-0.8.0.log"
uptime > "${LOGD}/uptime_after_all.txt"
echo "stage1-done $(date -u +%FT%TZ) rows=$(wc -l < results/raw/results-cuda.jsonl) skipped=$(wc -l < results/raw/skipped-cuda.log)"
# --- 2) 診断テスト（実機 #[ignore]）
cd "${ROOT}"
for t in async_ordering_real_device tma_probe_real_device gemm_transposed_parity gemm_transposed_perf; do
  echo "== ${t} =="
  cargo test -p fandhe-ai-backend-cuda --release --features internal-diagnostics --test "${t}" -- --ignored --nocapture --test-threads=1 > "${LOGD}/${t}.log" 2>&1
  echo "${t} rc=$? $(grep -E '^test result' "${LOGD}/${t}.log" | tail -1)"
done
echo "stage2-done $(date -u +%FT%TZ)"
# --- 3) RAYON_NUM_THREADS スイープ（cpu train/infer × fresh/reuse、各 3 プロセス起動）
cd "${FC}"
SW="${LOGD}/rayon-sweep.jsonl"; : > "${SW}"
for T in 4 8 10 20; do
  for run in 1 2 3; do
    for task in train infer; do
      for mode in fresh reuse; do
        RAYON_NUM_THREADS="${T}" ./target/release/bench-fandhe --task "${task}" --device cpu --size 64 --mode "${mode}" --out "${SW}" 2>>"${LOGD}/rayon-sweep.err" \
          && echo "T=${T} run=${run} ${task} ${mode} ok" || echo "T=${T} run=${run} ${task} ${mode} FAIL"
      done
    done
  done
done
echo "stage3-done $(date -u +%FT%TZ)"
uptime > "${LOGD}/uptime_after_sweep.txt"
echo "done."
