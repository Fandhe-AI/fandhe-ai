#!/usr/bin/env bash
# イシュー #1980／#1981（GB10 分）: registry `fandhe-ai =0.9.0` の
# `bench-fandhe --task train|infer --phases`（cpu／cuda × fresh／reuse）を
# DGX Spark GB10 で 5 プロセス系列（1 run = 1 JSONL）として取得する。
# `orchestrate_m4max.sh` の GB10 版。事前ビルド済みの
# `scripts/bench/framework-compare/target/release/bench-fandhe`（path patch
# なし・registry ピンのまま）を使い、本スクリプトは cargo を起動しない。
# 各 run 開始前に「load1 < 1.0 かつ GPU utilization 0%」を 30 秒間隔・最大
# 30 分待つ（gb10/RULE.txt。専有機のためゲートは Mac 分〈load1 < 8.0〉より
# 厳しい）。取得不能（/proc/loadavg・nvidia-smi が読めない）は gate=unavailable
# として記録し run は実行する（系列は参考扱い）。
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "${HERE}/../../../.." && pwd)
BIN="${ROOT}/scripts/bench/framework-compare/target/release/bench-fandhe"
OUT="${HERE}/gb10"
if [ ! -x "${BIN}" ]; then
  echo "bench-fandhe が未ビルド: cd scripts/bench/framework-compare && cargo build --release -p bench-fandhe" >&2
  exit 1
fi
mkdir -p "${OUT}"
for i in 1 2 3 4 5; do
  if [ -e "${OUT}/run${i}.jsonl" ]; then
    echo "run${i}.jsonl が既に存在する（差し替え禁止）" >&2
    exit 1
  fi
done
read_gate() {
  # 標準出力: "<load1> <gpu_util%>"。読めない値は NA
  l1=$(awk '{print $1}' /proc/loadavg 2>/dev/null)
  gu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  case "${l1}" in ''|*[!0-9.]*) l1="NA" ;; esac
  case "${gu}" in ''|*[!0-9]*) gu="NA" ;; esac
  echo "${l1} ${gu}"
}
for i in 1 2 3 4 5; do
  waited=0
  status="timeout"
  while [ "${waited}" -lt 1800 ]; do
    set -- $(read_gate); l1=$1; gu=$2
    if [ "${l1}" = "NA" ] || [ "${gu}" = "NA" ]; then status="unavailable"; break; fi
    ok=$(awk -v a="${l1}" -v g="${gu}" 'BEGIN{print (a<1.0 && g==0)?1:0}')
    if [ "${ok}" = "1" ]; then status="pass"; break; fi
    sleep 30
    waited=$((waited + 30))
  done
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
  echo "run${i} gate=${status} load1=${l1} gpu_util=${gu} compute_apps=${apps} waited_s=${waited} at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${OUT}/load_gate.log"
  for dev in cpu cuda; do
    for task in train infer; do
      for mode in fresh reuse; do
        "${BIN}" --task "${task}" --device "${dev}" --mode "${mode}" --phases \
          --out "${OUT}/run${i}.jsonl" 2>> "${OUT}/run${i}.err" \
          || echo "run${i} FAILED dev=${dev} task=${task} mode=${mode}" >> "${OUT}/load_gate.log"
      done
    done
  done
  set -- $(read_gate)
  echo "run${i} end_load1=$1 end_gpu_util=$2" >> "${OUT}/load_gate.log"
done
echo "series done" >> "${OUT}/load_gate.log"
