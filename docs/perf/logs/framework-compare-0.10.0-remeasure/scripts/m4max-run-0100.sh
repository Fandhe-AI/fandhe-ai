#!/usr/bin/env bash
# M4 Max 再計測（registry ピン =0.10.0）。0.9.0 の m4max-run-090.sh と同一セル ＋ Python 参照 FW（同一ラウンド内で計測）。
set -uo pipefail
ROOT="${ROOT:?repo root}"
D="${D:?remeasure dir}"
FC="${ROOT}/scripts/bench/framework-compare"
SHA="$(git -C "${ROOT}" rev-parse --short HEAD)"
LOGD="${1:?logdir}"; mkdir -p "${LOGD}"
OUT="${LOGD}/results-m4max-0.10.0.jsonl"; SKIP="${LOGD}/skipped-m4max-0.10.0.log"
PYOUT="${LOGD}/results-m4max-py-0.10.0.jsonl"
: > "${OUT}"; : > "${SKIP}"; : > "${PYOUT}"
cd "${FC}"
source ./bench_fandhe_lock_restore.sh
bench_fandhe_setup_lock_restore_trap
echo "start $(date -u +%FT%TZ) sha=${SHA}"; uptime | tee "${LOGD}/uptime_before.txt"
if [[ -z "${SKIP_BUILD:-}" ]]; then if ! cargo build --release 2>&1 | tail -5; then echo "BUILD FAILED"; exit 1; fi; fi
cargo tree -p bench-fandhe --depth 1 | grep -E 'fandhe-ai v' | tee "${LOGD}/tree.txt"
grep -q 'fandhe-ai v0.10.0$' "${LOGD}/tree.txt" || { echo "registry pin 0.10.0 not resolved"; exit 1; }
echo "build-done $(date -u +%FT%TZ)"; uptime | tee "${LOGD}/uptime_after_build.txt"
run() { local bin=$1 task=$2 dev=$3 size=$4 mode=${5:-fresh}
  echo "== ${bin} ${task} ${dev} ${size} ${mode} =="
  "./target/release/${bin}" --task "${task}" --device "${dev}" --size "${size}" --mode "${mode}" --out "${OUT}" 2>"${LOGD}/err.tmp" \
    || echo "${bin} ${task} ${dev} ${size} ${mode}: $(cat "${LOGD}/err.tmp")" >> "${SKIP}"; }
for bin in bench-fandhe bench-candle bench-burn; do
  for n in 256 512 1024 2048; do run "${bin}" gemm cpu "${n}"; done
  for n in 256 512 1024 2048 4096; do run "${bin}" gemm metal "${n}"; done
  for dev in cpu metal; do run "${bin}" train "${dev}" 64; run "${bin}" infer "${dev}" 64; done
done
for n in 256 512 1024 2048 4096; do run bench-fandhe gemm metal "${n}" reuse; done
for n in 256 512 1024 2048; do run bench-fandhe gemm cpu "${n}" reuse; done
for dev in cpu metal; do run bench-fandhe train "${dev}" 64 reuse; run bench-fandhe infer "${dev}" 64 reuse; done
echo "rust-done $(date -u +%FT%TZ)"
# Python 参照 FW（bench_py.py。PyTorch=venv-torch・SciPy=venv-torch・TensorFlow=venv-tf）
pyrun() { local py=$1; shift; echo "== py $* =="
  "${py}" "${D}/py/bench_py.py" "$@" --out "${PYOUT}" > /dev/null 2>"${LOGD}/err.tmp" \
    || echo "py $*: $(tail -3 "${LOGD}/err.tmp" | tr '\n' ' ')" >> "${SKIP}"; }
PT="${D}/../venv-torch/bin/python"; TF="${D}/../venv-tf/bin/python"
for pair in "pytorch:${PT}" "tensorflow:${TF}"; do fw="${pair%%:*}"; py="${pair#*:}"
  for n in 256 512 1024 2048; do pyrun "${py}" --framework "${fw}" --task gemm --device cpu --size "${n}"; done
  for n in 256 512 1024 2048 4096; do pyrun "${py}" --framework "${fw}" --task gemm --device metal --size "${n}"; done
  for dev in cpu metal; do pyrun "${py}" --framework "${fw}" --task train --device "${dev}" --size 64; pyrun "${py}" --framework "${fw}" --task infer --device "${dev}" --size 64; done
done
for n in 256 512 1024 2048; do pyrun "${PT}" --framework scipy --task gemm --device cpu --size "${n}"; done
pyrun "${PT}" --framework scipy --task train --device cpu --size 64; pyrun "${PT}" --framework scipy --task infer --device cpu --size 64
rm -f "${LOGD}/err.tmp"
uptime | tee "${LOGD}/uptime_after.txt"
echo "done. rows=$(wc -l < "${OUT}") pyrows=$(wc -l < "${PYOUT}") skipped=$(wc -l < "${SKIP}") $(date -u +%FT%TZ)"
