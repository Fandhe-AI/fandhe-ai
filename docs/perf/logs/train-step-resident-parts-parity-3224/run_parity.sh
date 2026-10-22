#!/usr/bin/env bash
# イシュー #3224: 1 機体ぶんの実機 parity テスト（RULE.txt の M1〜M5／C1〜C5）を
# 独立プロセスで 5 回ずつ実行するノード側スクリプト。bash 3.2 互換（macOS 標準）。
# 使い方: run_parity.sh <metal|cuda> <出力ディレクトリ> [run 開始番号 run 終了番号]
# 各ログ末尾に rc=<n> を書く（summarize.py が厳密に検査する）。foreign の cargo／rustc を
# run 前後で検出したら <name>.run<N>.invalid に理由を書く（RULE.txt: 同番号で取り直す）。
set -euo pipefail

PLATFORM="${1:?usage: run_parity.sh <metal|cuda> <outdir> [first last]}"
OUT="${2:?outdir required}"
FIRST="${3:-1}"
LAST="${4:-5}"
mkdir -p "${OUT}"

# 1 行 = 名前|cargo のターゲット引数|--exact の有無（テスト名はここの固定値だけ。外部入力は使わない）。
case "${PLATFORM}" in
  metal)
    PKG="fandhe-ai-backend-metal"
    SPECS="relu_mask_device_parity|--test relu_mask_device_parity|
mse_device_parity|--test mse_device_parity|
gemm_device_input_parity|--test gemm_device_input_parity|
train_step_parts_no_wait_3224|--test train_step_parts_no_wait_3224|
gemm_device_tracked_token|--lib gemm_device_tracked_registers_token_and_does_not_wait|--exact"
    ;;
  cuda)
    PKG="fandhe-ai-backend-cuda"
    SPECS="relu_mask_device_parity|--test relu_mask_device_parity|
mse_device_parity|--test mse_device_parity|
gemm_device_input_parity|--test gemm_device_input_parity|
gemm_device_input_diag_tests_3223|--lib gemm_device_input_diag_tests_3223|
train_step_parts_sync_diag_tests_3224|--lib train_step_parts_sync_diag_tests_3224|"
    ;;
  *) echo "unknown platform: ${PLATFORM}" >&2; exit 2 ;;
esac

# cargo／rustc の稼働（テスト実行前後の汚染検知。自分のテスト実行中の cargo は含まない時点で呼ぶ）。
foreign_builds() {
  ps -axo comm= | grep -E '(^|/)(cargo|rustc)$' || true
}

env_snapshot() {
  {
    echo "--- $1 $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    uname -a
    rustc --version
    if [ "${PLATFORM}" = "cuda" ]; then
      nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv || true
      nvidia-smi --query-gpu=utilization.gpu --format=csv || true
    fi
    foreign_builds
  } >> "${OUT}/env_info.txt" 2>&1
}

printf '%s\n' "${SPECS}" | while IFS='|' read -r NAME ARGS EXACT; do
  [ -n "${NAME}" ] || continue
  N="${FIRST}"
  while [ "${N}" -le "${LAST}" ]; do
    LOG="${OUT}/${NAME}.run${N}.log"
    env_snapshot "before ${NAME} run${N}"
    PRE="$(foreign_builds)"
    set +e
    # ARGS・EXACT は上の固定値。語分割を意図して無引用。
    cargo test -p "${PKG}" --release ${ARGS} -- --ignored ${EXACT} --test-threads=1 --nocapture > "${LOG}" 2>&1 < /dev/null
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
done
echo "done" > "${OUT}/DONE"
