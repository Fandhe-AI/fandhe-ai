#!/usr/bin/env bash
# イシュー #3224: 1 機体ぶんの実機 parity テスト（RULE.txt の M1〜M5／C1〜C5）を
# 独立プロセスで 5 回ずつ実行するノード側スクリプト。bash 3.2 互換（macOS 標準）。
# 使い方: run_parity.sh <metal|cuda> <出力ディレクトリ> [run 開始番号 run 終了番号]
# 各ログ末尾に rc=<n> を書く（summarize.py が厳密に検査する）。foreign の cargo／rustc を
# run 前後で検出したら <name>.run<N>.invalid に理由を書く（RULE.txt: 同番号で取り直す）。
# CUDA は nvidia-smi の compute プロセスも起動時のベースラインとの差で検出する。起動時に許可外のプロセスがあれば中止し
# （許可は env PARITY_RESIDENT_APP_REGEX の常駐サービス名のみ）、run 前後の nvidia-smi 失敗は invalid にする。
# 同じ番号を取り直せるのは現行の .invalid がある run だけ。.invalid の無い既存ログ（有効な run）は保持してそのテスト・番号だけ skip し、残りの取り直しは続行する。取り直すとき、前の試行のログ・invalid 理由は <出力>/attempts/ へ退避して残す
# （全 invalid の件数と理由を保存する契約）。採用結果の有効性は現行の .invalid の有無だけで決まる。
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
gemm_device_tracked_token|--lib ops::tests::gemm_device_tracked_registers_token_and_does_not_wait|--exact"
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

# GB10 の compute プロセス一覧（pid,name のみ。used_memory は変動するため比較に使わない）。
# nvidia-smi の失敗は空一覧と区別するため終了コードをそのまま返す（失敗を握りつぶさない）。
compute_apps() {
  if [ "${PLATFORM}" = "cuda" ]; then
    local out
    out="$(nvidia-smi --query-compute-apps=pid,name --format=csv,noheader 2>/dev/null)" || return 1
    printf '%s\n' "${out}" | sort
  fi
}

# 起動時に存在してよい compute プロセスは、確認済みの常駐サービスだけ（env PARITY_RESIDENT_APP_REGEX。
# 拡張正規表現を「pid, name」行の name 部へ完全一致で当てる。既定は空 = 起動時に 1 件でもあれば中止）。
# RULE.txt「計測中は別の GPU 負荷を並走させない」を満たすため、先行する別ジョブの負荷は許容しない。
RESIDENT_RE="${PARITY_RESIDENT_APP_REGEX:-}"
BASE_APPS=""
if [ "${PLATFORM}" = "cuda" ]; then
  if ! BASE_APPS="$(compute_apps)"; then
    echo "nvidia-smi による起動時の compute プロセス取得に失敗したため中止する" >&2
    exit 3
  fi
  while IFS= read -r LINE; do
    [ -n "${LINE}" ] || continue
    APP_NAME="${LINE#*, }"
    if [ -z "${RESIDENT_RE}" ] || ! printf '%s\n' "${APP_NAME}" | grep -qxE "${RESIDENT_RE}"; then
      echo "起動時に許可外の compute プロセスが存在するため中止する: ${LINE}（許可は PARITY_RESIDENT_APP_REGEX で明示）" >&2
      exit 3
    fi
  done <<< "${BASE_APPS}"
fi

# ベースラインに無い compute プロセス（新規の GPU 負荷）を stdout へ返す。取得失敗は return 1。
new_compute_apps() {
  local cur
  cur="$(compute_apps)" || return 1
  printf '%s\n' "${cur}" | grep -vxF -f <(printf '%s\n' "${BASE_APPS}") || true
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

# here-string で回す（pipe のサブシェルにしない。サブシェルだと途中の失敗が親へ伝わらず DONE が書かれる）。
while IFS='|' read -r NAME ARGS EXACT; do
  [ -n "${NAME}" ] || continue
  N="${FIRST}"
  while [ "${N}" -le "${LAST}" ]; do
    LOG="${OUT}/${NAME}.run${N}.log"
    # 取り直し: 現行の .invalid がある run だけが対象。.invalid の無い既存ログ（有効な pass／fail）は
    # 上書き・再実行しない（RULE.txt「それ以外の理由での再実行・選別はしない」「1 run でも fail なら fail」）。
    if [ -e "${LOG}" ] && [ ! -e "${OUT}/${NAME}.run${N}.invalid" ]; then
      echo "skip: ${LOG} は有効な既存 run（.invalid なし）のため保持して再実行しない。取り直しは invalid の run のみ" >&2
      N=$((N + 1))
      continue
    fi
    # 前の試行のログと invalid 理由を attempts/ へ退避し、現行の .invalid を外す。
    if [ -e "${LOG}" ] || [ -e "${OUT}/${NAME}.run${N}.invalid" ]; then
      mkdir -p "${OUT}/attempts"
      K=1
      while [ -e "${OUT}/attempts/${NAME}.run${N}.attempt${K}.log" ] || [ -e "${OUT}/attempts/${NAME}.run${N}.attempt${K}.invalid" ]; do K=$((K + 1)); done
      [ -e "${LOG}" ] && mv "${LOG}" "${OUT}/attempts/${NAME}.run${N}.attempt${K}.log"
      [ -e "${OUT}/${NAME}.run${N}.invalid" ] && mv "${OUT}/${NAME}.run${N}.invalid" "${OUT}/attempts/${NAME}.run${N}.attempt${K}.invalid"
    fi
    env_snapshot "before ${NAME} run${N}"
    PRE="$(foreign_builds)"
    GPU_ERR=""
    PRE_GPU="$(new_compute_apps)" || { GPU_ERR="pre"; PRE_GPU=""; }
    set +e
    # ARGS・EXACT は上の固定値。語分割を意図して無引用。
    cargo test -p "${PKG}" --release ${ARGS} -- --ignored ${EXACT} --test-threads=1 --nocapture > "${LOG}" 2>&1 < /dev/null
    RC=$?
    set -e
    echo "rc=${RC}" >> "${LOG}"
    env_snapshot "after ${NAME} run${N}"
    POST="$(foreign_builds)"
    POST_GPU="$(new_compute_apps)" || { GPU_ERR="${GPU_ERR:+${GPU_ERR},}post"; POST_GPU=""; }
    REASON=""
    if [ -n "${PRE}${POST}" ]; then
      REASON="foreign cargo/rustc detected: pre=[${PRE}] post=[${POST}]"
    fi
    if [ -n "${PRE_GPU}${POST_GPU}" ]; then
      REASON="${REASON:+${REASON}; }new compute process detected: pre=[${PRE_GPU}] post=[${POST_GPU}]"
    fi
    if [ -n "${GPU_ERR}" ]; then
      REASON="${REASON:+${REASON}; }nvidia-smi failed (${GPU_ERR}): GPU 負荷を確認できていない"
    fi
    if [ -n "${REASON}" ]; then
      echo "${REASON}" > "${OUT}/${NAME}.run${N}.invalid"
    fi
    N=$((N + 1))
  done
done <<< "${SPECS}"
echo "done" > "${OUT}/DONE"
