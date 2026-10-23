#!/usr/bin/env bash
# イシュー #3251: 学習ステップのチェーン（DeviceParamStore::train_step_device_chain）の Mac 実機テスト
# （RULE.txt の A1〜A3）を独立プロセスで 5 回ずつ実行するノード側スクリプト。bash 3.2 互換（macOS 標準）。
# 使い方: run_chain.sh <出力ディレクトリ> [run 開始番号 run 終了番号]
# 各ログ末尾に rc=<n> を書く。foreign の cargo／rustc を run 前後で検出したら <name>.run<N>.invalid に理由を書く
# （RULE.txt: 同番号で取り直す）。構成は #3224 の run_parity.sh（metal）と同じ方針の簡略版。
set -euo pipefail

OUT="${1:?usage: run_chain.sh <outdir> [first last]}"
FIRST="${2:-1}"
LAST="${3:-5}"
mkdir -p "${OUT}"

P="compat::sequential::train_step_chain_tests_3251::metal::"
SPECS="A1|${P}metal_chain_matches_tape_path_bit_exact_two_layers
A2|${P}metal_chain_matches_tape_path_bit_exact_three_layers
A3|${P}metal_chain_step_issues_exactly_one_wait"

foreign_builds() {
  ps -axo comm= | grep -E '(^|/)(cargo|rustc)$' || true
}

env_snapshot() {
  {
    echo "--- $1 $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    uname -srm
    rustc --version
    foreign_builds
  } >> "${OUT}/env_info.txt" 2>&1
}

while IFS='|' read -r NAME FILTER; do
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
    cargo test -p fandhe-ai --lib --release "${FILTER}" -- --ignored --exact --test-threads=1 --nocapture > "${LOG}" 2>&1 < /dev/null
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
