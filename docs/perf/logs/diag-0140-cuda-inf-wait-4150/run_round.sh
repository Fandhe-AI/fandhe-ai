#!/usr/bin/env bash
# #4150 の 1 round（<cuda-node> = DGX GB10。計測キット。PR-1 時点では未実行）。
# 役割: RULE.txt 項 3 の手順を 1 round 分実行し、round{i}/{gate.txt, raw/chain.txt} を作る。
#   順序は precheck → ビルド → gate → example 実行（1 プロセス・P/E/I/F 腕）。ビルドはゲートの前に済ませ、計測中に並走させない。
#   ゲート通過後は cargo を呼ばず（fingerprint 検査等のホスト負荷を避ける）、ビルド済みバイナリを直接起動する（nsys_aux_4150.sh と同じ BIN）。
#   target dir は ROOT/target に固定する（利用者の CARGO_TARGET_DIR で別の場所に出て古いバイナリを起動するのを防ぐ）。
# 派生元: diag-0140-torch-gb10-mlp-4149/run_round.sh（GB10 のゲート・実機ロックの作法）。
# 環境変数: ROOT=計測ツリー（tree_sha の git archive 展開先。.rev-stamp に完全 SHA）
#           EXPECT_SHA=RULE.txt の tree_sha  EXPECT_EXAMPLE_SHA=RULE.txt の example_sha256
#           HW_LOCK_OWNER_FILE=実機ロックの owner ファイル（空でないこと）  HW_LOCK_HELD=1（実機ロック取得済みの宣言）
# 引数: $1=round 番号（1 始まり）$2=記録先 round ディレクトリ（絶対パス。round{i}）
# bash 3.2／5 互換。run の差し替え・上書きは禁止。外部入力をシェル文字列へ展開しない（変数は常に引用符つきで引数として渡す）。
set -uo pipefail
validate_abs() { # $1=変数名 $2=値。絶対パスのみ。引用符・バックスラッシュ・改行を拒否（A03）
  case "$2" in /*) ;; *) echo "$1 は絶対パスのみ: $2"; exit 2;; esac
  case "$2" in *\"*|*\'*|*\\*|*$'\n'*) echo "$1 に禁止文字を含む"; exit 2;; esac
}
ROOT="${ROOT:?計測ツリー}"; SHA="${EXPECT_SHA:?RULE の tree_sha}"; ESHA="${EXPECT_EXAMPLE_SHA:?RULE の example_sha256}"
LOCKF="${HW_LOCK_OWNER_FILE:?実機ロックの owner ファイル}"
RN="${1:?round 番号}"; RD="${2:?round ディレクトリ}"
validate_abs ROOT "${ROOT}"; validate_abs LOCKF "${LOCKF}"; validate_abs RD "${RD}"
case "${RN}" in ''|*[!0-9]*) echo "round 番号は数字のみ"; exit 2;; esac
case "${SHA}" in *[!0-9a-f]*|'') echo "EXPECT_SHA は hex のみ"; exit 2;; esac
case "${ESHA}" in *[!0-9a-f]*|'') echo "EXPECT_EXAMPLE_SHA は hex のみ"; exit 2;; esac
[ "${HW_LOCK_HELD:-}" = "1" ] || { echo "実機ロック未取得（HW_LOCK_HELD=1 が必要）"; exit 2; }
[ -z "${LD_PRELOAD:-}" ] || { echo "LD_PRELOAD が非空（計時腕では拒否）"; exit 2; }
EX="${ROOT}/crates/backend-cuda/examples/cuda_infer_chain_event_diag_4150.rs"
BIN="${ROOT}/target/release/examples/cuda_infer_chain_event_diag_4150"
[ ! -e "${RD}/raw" ] || { echo "${RD}/raw が既にある（上書き禁止）"; exit 2; }

# ---- precheck（fail-closed）
[ -s "${LOCKF}" ] || { echo "実機ロックの owner が空（ロックなしの値は採らない）"; exit 1; }
nvidia-smi --query-gpu=name --format=csv,noheader | grep -q 'GB10' || { echo "機体が GB10 ではない"; exit 1; }
[ "$(cat "${ROOT}/.rev-stamp")" = "${SHA}" ] || { echo ".rev-stamp が RULE と不一致"; exit 1; }
echo "${ESHA}  ${EX}" | sha256sum -c - > /dev/null || { echo "example の sha256 不一致"; exit 1; }
mkdir -p "${RD}/raw"
# ビルドはゲートの前（計測中のビルド並走を避ける）。
( cd "${ROOT}" && CARGO_TARGET_DIR="${ROOT}/target" cargo build --release --locked -p fandhe-ai-backend-cuda \
    --features internal-diagnostics --example cuda_infer_chain_event_diag_4150 ) 2> "${RD}/raw/build.err" \
  || { echo "build 失敗"; exit 1; }
[ -x "${BIN}" ] || { echo "ビルド済みバイナリがない: ${BIN}"; exit 1; }

# ---- ゲート: compute プロセス 0 件・GPU 使用率 0 が 3 サンプル連続・load1 < 1.0 を 30 秒間隔で最大 30 分（#4147・#4149 と同一）。
gate() {
  local i=0 apps util l zero=0
  while [ "${i}" -le 60 ]; do
    if ! apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>&1); then echo "INVALID nvidia-smi-apps-failed" > "${RD}/gate.txt"; return 2; fi
    if ! util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>&1); then echo "INVALID nvidia-smi-util-failed" > "${RD}/gate.txt"; return 2; fi
    l=$(awk '{print $1}' /proc/loadavg)
    if ! printf '%s' "${l}" | grep -Eq '^[0-9]+(\.[0-9]+)?$'; then echo "INVALID load1-not-numeric" > "${RD}/gate.txt"; return 2; fi
    if [ -z "${apps}" ] && [ "$(printf '%s' "${util}" | tr -d ' ')" = "0" ] && awk -v x="${l}" 'BEGIN{exit !(x < 1.0)}'; then
      zero=$((zero + 1))
    else
      zero=0
    fi
    if [ "${zero}" -ge 3 ]; then echo "PASS load1=${l} compute-apps=0 gpu-util=0x3 hw-lock=held $(date -u +%FT%TZ)" > "${RD}/gate.txt"; return 0; fi
    sleep 30; i=$((i + 1))
  done
  echo "TIMEOUT load1=${l} $(date -u +%FT%TZ)" > "${RD}/gate.txt"; return 1
}
gate || { echo "ゲート不通過: $(cat "${RD}/gate.txt")"; exit 1; }

# ゲート後はビルド済みバイナリを直接起動する（cargo run の fingerprint 処理をゲート通過後に走らせない）。
[ -x "${BIN}" ] || { echo "ビルド済みバイナリがない: ${BIN}"; exit 1; }
( cd "${ROOT}" && "${BIN}" --round "${RN}" ) > "${RD}/raw/chain.txt" 2> "${RD}/raw/chain.err" \
  || { echo "example 失敗"; exit 1; }
echo "round ${RN} done $(date -u +%FT%TZ)"
