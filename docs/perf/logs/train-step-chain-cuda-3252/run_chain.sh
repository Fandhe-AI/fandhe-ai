#!/usr/bin/env bash
# イシュー #3252: CUDA の学習チェーン（DeviceParamStore::train_step_device_chain）の GB10 実機テスト
# （RULE.txt の B1〜B3）を独立プロセスで 5 回ずつ実行するノード側スクリプト。
# 使い方: run_chain.sh <出力ディレクトリ> [run 開始番号 run 終了番号]。PATH に cargo と nvcc を通しておく。
# 各ログ末尾に rc=<n> を書く。foreign の cargo／rustc を run 前後で検出したら <name>.run<N>.invalid に理由を書く
# （RULE.txt: 同番号で取り直す）。構成は docs/perf/logs/train-step-chain-metal-3251/run_chain.sh と同方針。
set -euo pipefail

OUT="${1:?usage: run_chain.sh <outdir> [first last]}"
FIRST="${2:-1}"
LAST="${3:-5}"
mkdir -p "${OUT}"

F="compat::sequential::train_step_chain_tests_3251::cuda::"
SPECS="B1|fandhe-ai-backend-cuda|train_step_chain_diag_tests_3252::train_chain_window_has_no_sync_alloc_or_h2d_and_loss_read_is_one_sync
B2|fandhe-ai|${F}cuda_chain_matches_tape_path_bit_exact_two_layers
B3|fandhe-ai|${F}cuda_chain_matches_tape_path_bit_exact_three_layers"

foreign_builds() {
  ps -eo comm= | grep -E '^(cargo|rustc)$' || true
}

env_snapshot() {
  {
    echo "--- $1 $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    uname -srm
    rustc --version
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
    nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader || true
    foreign_builds
  } >> "${OUT}/env_info.txt" 2>&1
}

while IFS='|' read -r NAME CRATE FILTER; do
  [ -n "${NAME}" ] || continue
  N="${FIRST}"
  while [ "${N}" -le "${LAST}" ]; do
    LOG="${OUT}/${NAME}.run${N}.log"
    if [ -e "${LOG}" ] && [ ! -e "${OUT}/${NAME}.run${N}.invalid" ]; then
      echo "skip: ${LOG} は有効な既存 run のため保持する" >&2
      N=$((N + 1))
      continue
    fi
    if [ -e "${LOG}" ] || [ -e "${OUT}/${NAME}.run${N}.invalid" ]; then
      mkdir -p "${OUT}/attempts"
      K=1
      while [ -e "${OUT}/attempts/${NAME}.run${N}.attempt${K}.log" ] || [ -e "${OUT}/attempts/${NAME}.run${N}.attempt${K}.invalid" ]; do K=$((K + 1)); done
      [ -e "${LOG}" ] && mv "${LOG}" "${OUT}/attempts/${NAME}.run${N}.attempt${K}.log"
      [ -e "${OUT}/${NAME}.run${N}.invalid" ] && mv "${OUT}/${NAME}.run${N}.invalid" "${OUT}/attempts/${NAME}.run${N}.attempt${K}.invalid"
    fi
    env_snapshot "before ${NAME} run${N}"
    PRE="$(foreign_builds)"
    set +e
    cargo test -p "${CRATE}" --lib --release "${FILTER}" -- --ignored --exact --test-threads=1 --nocapture > "${LOG}" 2>&1 < /dev/null
    RC=$?
    set -e
    echo "rc=${RC}" >> "${LOG}"
    env_snapshot "after ${NAME} run${N}"
    POST="$(foreign_builds)"
    if [ -n "${PRE}${POST}" ]; then
      echo "foreign cargo/rustc detected: pre=[${PRE}] post=[${POST}]" > "${OUT}/${NAME}.run${N}.invalid"
    fi
    N=$((N + 1))
  done
done <<< "${SPECS}"
echo "done" > "${OUT}/DONE"
