#!/usr/bin/env bash
# イシュー #3256: 常駐チェーンの公開入口の実機 parity・同期回数テスト（RULE.txt の M1〜M6／C1〜C9）を
# 独立プロセスで 5 回ずつ実行するノード側スクリプト。
# 使い方: run_rows.sh <出力ディレクトリ> <metal|cuda> [run 開始番号 run 終了番号]。PATH に cargo（CUDA は nvcc も）を通す。
# 各ログ末尾に rc=<n> を書く。foreign の cargo／rustc を run 前後で検出したら <name>.run<N>.invalid に理由を書く
# （RULE.txt: 同番号で取り直す）。構成は docs/perf/logs/train-step-chain-cuda-3252/run_chain.sh と同方針。
# ターゲット列は `--lib` か `--test resident_train_step`。固定文字列の SPECS だけを展開し外部入力は使わない。
# 注意: `--lib` を一律に固定すると integration test が「0 tests run・rc=0」で素通りするため、
# 判定側（README の集計）で `running 1 test` と `1 passed` の有無を必ず確かめる（RULE.txt）。
# macOS の bash 3.2 でも動くよう、変数は ${VAR} の形で書く。
set -euo pipefail

OUT="${1:?usage: run_rows.sh <outdir> <metal|cuda> [first last]}"
MODE="${2:?usage: run_rows.sh <outdir> <metal|cuda> [first last]}"
FIRST="${3:-1}"
LAST="${4:-5}"
mkdir -p "${OUT}"

IT="--test resident_train_step"
F="compat::sequential::train_step_chain_tests_3251::"
case "${MODE}" in
  metal)
    SPECS="M1|fandhe-ai|${IT}|metal_public_api_chain_matches_tape_path_bit_exact_two_layers
M2|fandhe-ai|${IT}|metal_public_api_chain_matches_tape_path_bit_exact_three_layers
M3|fandhe-ai|${IT}|metal_public_api_unread_steps_then_final_params_match_tape_path
M4|fandhe-ai|${IT}|metal_upload_resident_with_cpu_tape_is_device_mismatch
M5|fandhe-ai|--lib|${F}metal::metal_chain_step_issues_exactly_one_wait
M6|fandhe-ai|--lib|${F}metal::metal_wait_device_param_store_issues_exactly_one_wait"
    ;;
  cuda)
    SPECS="C1|fandhe-ai|${IT}|cuda_public_api_chain_matches_tape_path_bit_exact_two_layers
C2|fandhe-ai|${IT}|cuda_public_api_chain_matches_tape_path_bit_exact_three_layers
C3|fandhe-ai|${IT}|cuda_public_api_unread_steps_then_final_params_match_tape_path
C4|fandhe-ai|${IT}|cuda_upload_resident_with_cpu_tape_is_device_mismatch
C5|fandhe-ai|${IT}|cuda_tf32_mode_chain_matches_tape_path_bit_exact
C6|fandhe-ai-backend-cuda|--lib|train_step_chain_diag_tests_3252::train_chain_window_has_no_sync_alloc_or_h2d_and_loss_read_is_one_sync
C7|fandhe-ai-backend-cuda|--lib|train_step_chain_diag_tests_3252::public_entry_call_sequence_window_has_no_sync_and_read_is_one_sync
C8|fandhe-ai-backend-cuda|--lib|train_step_chain_diag_tests_3252::tf32_mode_chain_outcome_is_recorded
C9|fandhe-ai|--lib|${F}cuda::cuda_wait_device_param_store_and_read_after_internal_tape_dropped"
    ;;
  *) echo "mode は metal か cuda" >&2; exit 2 ;;
esac

foreign_builds() {
  ps -eo comm= | grep -E '^(cargo|rustc)$' || true
}

env_snapshot() {
  {
    echo "--- $1 $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    uname -srm
    rustc --version
    if [ "${MODE}" = "cuda" ]; then
      nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
      nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader || true
    fi
    foreign_builds
  } >> "${OUT}/env_info.txt" 2>&1
}

while IFS='|' read -r NAME CRATE TARGET FILTER; do
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
    # shellcheck disable=SC2086  # TARGET は固定文字列（`--lib`／`--test <name>`）で意図した単語分割
    cargo test -p "${CRATE}" ${TARGET} --release "${FILTER}" -- --ignored --exact --test-threads=1 --nocapture > "${LOG}" 2>&1 < /dev/null
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
