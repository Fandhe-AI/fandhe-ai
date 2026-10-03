#!/usr/bin/env bash
# DGX GB10 再計測（registry ピン =0.10.0）: run_all_cuda.sh 全セル ＋ CPU gemm reuse/4096 ＋ Python 参照 FW（同一セッション）
set -uo pipefail
export PATH="${HOME}/.cargo/bin:/usr/local/cuda/bin:${PATH}"
ROOT="${HOME}/work/rust-ai-library-run"; FC="${ROOT}/scripts/bench/framework-compare"
LOGD="${LOGD:-${HOME}/work/dgx-0100-logs}"; mkdir -p "${LOGD}"
SHA="$(cat "${ROOT}/.head-sha")"
cd "${FC}"
echo "start $(date -u +%FT%TZ) sha=${SHA}"; uptime > "${LOGD}/uptime_before_all.txt"
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader > "${LOGD}/gpu_util_before.txt"
bash ./run_all_cuda.sh > "${LOGD}/run_all_cuda.log" 2>&1; echo "run_all rc=$?"
cp results/raw/results-cuda.jsonl "${LOGD}/results-dgx-0.10.0.jsonl"; cp results/raw/skipped-cuda.log "${LOGD}/skipped-dgx-0.10.0.log"
uptime > "${LOGD}/uptime_after_all.txt"
echo "stage1-done $(date -u +%FT%TZ) rows=$(wc -l < results/raw/results-cuda.jsonl) skipped=$(wc -l < results/raw/skipped-cuda.log)"
EX="${LOGD}/results-dgx-0.10.0-extra.jsonl"; : > "${EX}"
for n in 256 512 1024 2048 4096; do ./target/release/bench-fandhe --task gemm --device cpu --size "${n}" --mode reuse --out "${EX}" 2>>"${LOGD}/extra.err" || echo "reuse ${n} FAIL"; done
./target/release/bench-fandhe --task gemm --device cpu --size 4096 --mode fresh --out "${EX}" 2>>"${LOGD}/extra.err" || echo "fresh 4096 FAIL"
for bin in bench-candle bench-burn; do ./target/release/${bin} --task gemm --device cpu --size 4096 --mode fresh --out "${EX}" 2>>"${LOGD}/extra.err" || echo "${bin} 4096 FAIL"; done
echo "stage2-done $(date -u +%FT%TZ) rows=$(wc -l < "${EX}")"
grep -E 'fandhe-ai v0\.10\.0$' "${LOGD}/run_all_cuda.log" | head -2 || echo "WARN: registry 0.10.0 line not found in log"
# stage3: Python 参照 FW（PyTorch CPU/CUDA・TensorFlow CPU・SciPy）
PY="${HOME}/work/.venv-bench/bin/python"; BP="${HOME}/work/bench_py.py"
PYOUT="${LOGD}/results-dgx-py-0.10.0.jsonl"; : > "${PYOUT}"
uptime > "${LOGD}/uptime_before_py.txt"
prun() { echo "== $* =="; "${PY}" "${BP}" "$@" --out "${PYOUT}" > /dev/null 2>>"${LOGD}/py.err" || echo "  -> FAILED $*"; }
for fw in pytorch tensorflow scipy; do
  for n in 256 512 1024 2048 4096; do prun --framework "${fw}" --task gemm --device cpu --size "${n}"; done
  prun --framework "${fw}" --task train --device cpu --size 64
  prun --framework "${fw}" --task infer --device cpu --size 64
done
for n in 256 512 1024 2048 4096; do prun --framework pytorch --task gemm --device cuda --size "${n}"; done
prun --framework pytorch --task train --device cuda --size 64
prun --framework pytorch --task infer --device cuda --size 64
uptime > "${LOGD}/uptime_after_py.txt"
echo "stage3-done $(date -u +%FT%TZ) pyrows=$(wc -l < "${PYOUT}")"
echo "done. $(date -u +%FT%TZ)"
