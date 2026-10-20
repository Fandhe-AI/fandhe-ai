#!/usr/bin/env bash
# イシュー #3185: GB10 で PyTorch（cuBLAS）の純カーネル時間（FP32・TF32）と checksum を測る実行ループ。
# 規則の正は同ディレクトリの RULE.txt。#3179 の run-macmini.sh と同型（負荷ゲート・run ごとの起動順
# ローテーション・ログ非上書き・版固定）。GB10 版はゲートが load1 < 1.0 かつ GPU 利用率 0%（RULE 項 9）。
#
# 使い方（GB10 上。ビルド済みであること。cargo build は本スクリプトでは走らせない）:
#   VENV_PY=<.venv-bench/bin/python> BENCH_PY=<bench_py.py> DIAG=<opp_kernel_cuda_diag.py> \
#   REF_JSONL=<results-dgx-py-0.11.0-median5.jsonl> REPO_DIR=<転送ツリー> OUT=<出力ディレクトリ(未作成)> \
#   [NODE_BENCH_PY=<ノード上の既存 bench_py.py>] bash run-gb10.sh
set -euo pipefail
export LC_ALL=C

: "${VENV_PY:?}" "${BENCH_PY:?}" "${DIAG:?}" "${REF_JSONL:?}" "${REPO_DIR:?}" "${OUT:?}"
GATE_LOAD=1.0
if [ -e "${OUT}" ]; then
  echo "OUT が既にある（上書きしない）: ${OUT}" >&2
  exit 2
fi
mkdir -p "${OUT}"
GATE_LOG="${OUT}/gate.log"

# 版の固定（RULE 項 2）。torch が 2.14.0+cu130 でなければ停止。
"${VENV_PY}" -c 'import sys,torch,numpy; v=(sys.version.split()[0],torch.__version__,numpy.__version__); print(*v); sys.exit(0 if torch.__version__=="2.14.0+cu130" else 4)' > "${OUT}/versions.txt" \
  || { echo "venv の torch が RULE 項 2 の 2.14.0+cu130 と不一致" >&2; exit 4; }
{
  # 内部ホスト名は出さない（uname -n は使わない。RULE 項 1）。
  nvidia-smi --query-gpu=name,driver_version --format=csv
  nvcc --version || true
  uname -m
  "${VENV_PY}" -c 'import torch,numpy; print(torch.__config__.show()); print(torch.version.cuda, torch.backends.cudnn.version())'
  # 標準入力経由でハッシュのみ出す（sha256sum はパスを併記し、絶対ホームパスが env.txt に残るため。RULE 項 1）。
  echo "bench_py.py: $(sha256sum < "${BENCH_PY}" | cut -d' ' -f1)"
  if [ -n "${NODE_BENCH_PY:-}" ] && [ -f "${NODE_BENCH_PY}" ]; then echo "node bench_py.py: $(sha256sum < "${NODE_BENCH_PY}" | cut -d' ' -f1)"; fi
  cat "${REPO_DIR}/.rev-stamp"
} > "${OUT}/env.txt" 2>&1

wait_gate() { # $1=ラベル。load1 < GATE_LOAD かつ GPU 利用率 0% になるまで 30 秒間隔で最大 30 分。
  local i=0 load util
  while [ "${i}" -le 60 ]; do
    load="$(uptime | sed -E 's/.*load averages?: *([0-9.]+).*/\1/')"
    util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')"
    case "${load}" in ''|*[!0-9.]*) echo "GATE-INVALID $1 load=${load}" >> "${GATE_LOG}"; exit 5 ;; esac
    case "${util}" in ''|*[!0-9]*) echo "GATE-INVALID $1 util=${util}" >> "${GATE_LOG}"; exit 5 ;; esac
    if awk -v l="${load}" -v t="${GATE_LOAD}" 'BEGIN{exit !(l<t)}' && [ "${util}" = 0 ]; then
      echo "GATE-PASS $1 load1=${load} gpu_util=${util}" >> "${GATE_LOG}"
      return 0
    fi
    i=$((i + 1))
    sleep 30
  done
  echo "GATE-TIMEOUT $1" >> "${GATE_LOG}"
  exit 6
}

# 腕: "arm"。1 腕 = 1 プロセス。run ごとに起動順をローテーションする（RULE 項 10）。
ARMS="fp32 tf32 cpu"
N_ARMS=3
run=1
while [ "${run}" -le 5 ]; do
  mkdir -p "${OUT}/run${run}"
  rot=$(( (run - 1) % N_ARMS ))
  ordered=""
  for pass in 1 2; do
    j=0
    for a in ${ARMS}; do
      if { [ "${pass}" = 1 ] && [ "${j}" -ge "${rot}" ]; } || { [ "${pass}" = 2 ] && [ "${j}" -lt "${rot}" ]; }; then
        ordered="${ordered} ${a}"
      fi
      j=$((j + 1))
    done
  done
  for arm in ${ordered}; do
    wait_gate "run${run}/${arm}"
    uptime > "${OUT}/run${run}/uptime_before_${arm}.txt"
    rc=0
    "${VENV_PY}" "${DIAG}" measure --bench-py "${BENCH_PY}" --arm "${arm}" --run "${run}" \
      --out "${OUT}/run${run}/${arm}.jsonl" || rc=$?
    if [ "${rc}" != 0 ]; then
      exit "${rc}"
    fi
    uptime > "${OUT}/run${run}/uptime_after_${arm}.txt"
  done
  run=$((run + 1))
done

# CPU 側 BLAS 経路（RULE 項 13）。
wait_gate "cpu-blas-probe"
"${VENV_PY}" "${DIAG}" cpu-blas-probe --out-dir "${OUT}/cpu-blas"

# Python の計測を終えてから fandhe 側（ビルド済み。RULE 項 7・10）。test result: ok が無い run は parse が例外。
mkdir -p "${OUT}/fandhe" "${OUT}/fandhe_rows"
for r in 1 2 3 4 5; do
  wait_gate "fandhe/layerb/r${r}"
  (cd "${REPO_DIR}" && cargo test --release -p fandhe-ai-backend-cuda --lib gemm_reuse_phase_diag_tests::gemm_reuse_phase_diag_select \
    -- --ignored --exact --test-threads=1 --nocapture) > "${OUT}/fandhe/layerb-run${r}.log" 2>&1
  "${VENV_PY}" "${DIAG}" parse-fandhe --log "${OUT}/fandhe/layerb-run${r}.log" --kind layerb --run "${r}" \
    --out "${OUT}/fandhe_rows/layerb-run${r}.jsonl"
  wait_gate "fandhe/diag2109/r${r}"
  (cd "${REPO_DIR}" && cargo test --release -p fandhe-ai-backend-cuda --lib gemm_small_launch_cost_diag_tests::gemm_small_launch_cost_diag \
    -- --ignored --exact --test-threads=1 --nocapture) > "${OUT}/fandhe/diag2109-run${r}.log" 2>&1
  "${VENV_PY}" "${DIAG}" parse-fandhe --log "${OUT}/fandhe/diag2109-run${r}.log" --kind diag2109 --run "${r}" \
    --out "${OUT}/fandhe_rows/diag2109-run${r}.jsonl"
done

"${VENV_PY}" "${DIAG}" aggregate --runs-dir "${OUT}" --reference-py-jsonl "${REF_JSONL}" \
  --out-json "${OUT}/aggregate.json" --out-md "${OUT}/aggregate.md"
