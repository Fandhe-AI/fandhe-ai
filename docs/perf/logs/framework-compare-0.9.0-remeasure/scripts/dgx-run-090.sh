#!/usr/bin/env bash
# DGX GB10 再計測（正式系列 registry ピン =0.9.0）: run_all_cuda.sh 全セル ＋ CPU gemm reuse/4096 ＋ Python 参照 FW
set -uo pipefail
export PATH="${HOME}/.cargo/bin:/usr/local/cuda/bin:${PATH}"
ROOT="${HOME}/work/rust-ai-library-run"; FC="${ROOT}/scripts/bench/framework-compare"
LOGD="${HOME}/work/dgx-090-logs"; mkdir -p "${LOGD}"
SHA="$(cat "${ROOT}/.head-sha")"
cd "${FC}"
echo "start $(date -u +%FT%TZ) sha=${SHA}"; uptime > "${LOGD}/uptime_before_all.txt"
bash ./run_all_cuda.sh > "${LOGD}/run_all_cuda.log" 2>&1; echo "run_all rc=$?"
cp results/raw/results-cuda.jsonl "${LOGD}/results-dgx-0.9.0.jsonl"; cp results/raw/skipped-cuda.log "${LOGD}/skipped-dgx-0.9.0.log"
grep -E 'fandhe-ai v' "${LOGD}/run_all_cuda.log" | head -3
uptime > "${LOGD}/uptime_after_all.txt"
echo "stage1-done $(date -u +%FT%TZ) rows=$(wc -l < results/raw/results-cuda.jsonl) skipped=$(wc -l < results/raw/skipped-cuda.log)"
EX="${LOGD}/results-dgx-0.9.0-extra.jsonl"; : > "${EX}"
for n in 256 512 1024 2048 4096; do ./target/release/bench-fandhe --task gemm --device cpu --size "${n}" --mode reuse --out "${EX}" 2>>"${LOGD}/extra.err" || echo "reuse ${n} FAIL"; done
./target/release/bench-fandhe --task gemm --device cpu --size 4096 --mode fresh --out "${EX}" 2>>"${LOGD}/extra.err" || echo "fresh 4096 FAIL"
for bin in bench-candle bench-burn; do ./target/release/${bin} --task gemm --device cpu --size 4096 --mode fresh --out "${EX}" 2>>"${LOGD}/extra.err" || echo "${bin} 4096 FAIL"; done
echo "stage2-done $(date -u +%FT%TZ) rows=$(wc -l < "${EX}")"
grep -E 'fandhe-ai v0\.9\.0$' "${LOGD}/run_all_cuda.log" | head -2 || echo "WARN: registry 0.9.0 line not found in log"
echo "done. $(date -u +%FT%TZ)"
