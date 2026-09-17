#!/bin/sh
# イシュー #1980／#1981（Mac 分）: registry `fandhe-ai =0.9.0` の
# `bench-fandhe --task train|infer --phases`（cpu／metal × fresh／reuse）を
# 5 プロセス系列（1 run = 1 JSONL）で取得する。事前ビルド済みの
# `scripts/bench/framework-compare/target/release/bench-fandhe`（path patch
# なし・registry ピンのまま）を使い、本スクリプトは cargo を起動しない。
# 各 run 開始前に load1 < 8.0 を 30 秒間隔・最大 30 分待つ（RULE.txt）。
# bash 3.2 安全のため変数は必ず ${VAR} で参照する。
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "${HERE}/../../../.." && pwd)
BIN="${ROOT}/scripts/bench/framework-compare/target/release/bench-fandhe"
OUT="${HERE}/m4max"
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
for i in 1 2 3 4 5; do
  waited=0
  status="timeout"
  while [ "${waited}" -lt 1800 ]; do
    l1=$(sysctl -n vm.loadavg | awk '{print $2}')
    ok=$(awk -v a="${l1}" 'BEGIN{print (a<8.0)?1:0}')
    if [ "${ok}" = "1" ]; then status="pass"; break; fi
    sleep 30
    waited=$((waited + 30))
  done
  echo "run${i} gate=${status} load1=${l1} waited_s=${waited} at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${OUT}/load_gate.log"
  for dev in cpu metal; do
    for task in train infer; do
      for mode in fresh reuse; do
        "${BIN}" --task "${task}" --device "${dev}" --mode "${mode}" --phases \
          --out "${OUT}/run${i}.jsonl" 2>> "${OUT}/run${i}.err" \
          || echo "run${i} FAILED dev=${dev} task=${task} mode=${mode}" >> "${OUT}/load_gate.log"
      done
    done
  done
  echo "run${i} end_load1=$(sysctl -n vm.loadavg | awk '{print $2}')" >> "${OUT}/load_gate.log"
done
echo "series done" >> "${OUT}/load_gate.log"
