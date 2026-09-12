#!/usr/bin/env bash
# DGX: Python 参照フレームワーク（PyTorch CPU/CUDA・TensorFlow CPU・SciPy）を framework-compare と同一プロトコルで計測
set -u
cd "${HOME}/work"
PY="${HOME}/work/.venv-bench/bin/python"
LOGD="${HOME}/work/dgx-0.8.0-logs"; mkdir -p "${LOGD}"
OUT="${LOGD}/results-dgx-py-0.8.0.jsonl"; : > "${OUT}"
echo "py-start $(date -u +%FT%TZ)"
run() { echo "== $* =="; "${PY}" ./bench_py.py "$@" --out "${OUT}" 2>>"${LOGD}/py.err" || echo "  -> FAILED"; }
for fw in pytorch tensorflow scipy; do
  for n in 256 512 1024 2048 4096; do run --framework "${fw}" --task gemm --device cpu --size "${n}"; done
  run --framework "${fw}" --task train --device cpu --size 64
  run --framework "${fw}" --task infer --device cpu --size 64
done
for n in 256 512 1024 2048 4096; do run --framework pytorch --task gemm --device cuda --size "${n}"; done
run --framework pytorch --task train --device cuda --size 64
run --framework pytorch --task infer --device cuda --size 64
echo "py-done $(date -u +%FT%TZ) rows=$(wc -l < "${OUT}")"
