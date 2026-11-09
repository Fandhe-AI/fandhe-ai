#!/usr/bin/env bash
# #4152 の 1 round（<metal-node> = Mac mini・Apple M4。計測キット。PR-1 時点では未実行）。
# 役割: RULE.txt 項 3 の手順を 1 round 分実行し、round{i}/{gate.txt, raw/{t1,native}.txt} を作る。
#   順序は gate → t1（判定系列）→ native（record_only）。奇数 round は n を昇順、偶数 round は降順で計る。
# 派生元: docs/perf/logs/diag-0140-torch-cpu-mlp-macmini-4148/run_round.sh と同じ不変条件
#   （機体 Apple M4・.rev-stamp 照合・絶対パス検証・hw-lock 宣言・既存 raw/ の上書き拒否）。
# 環境変数: ROOT=計測ツリー（tree_sha の git archive 展開先。.rev-stamp に完全 SHA。ビルド済みの
#             target/release/examples/cpu_gemm_nr_tail_4152 を持つ。ビルドは round の前に済ませる）
#           EXPECT_SHA=RULE.txt の tree_sha  HW_LOCK_HELD=1（hw-lock を取得済みであることの宣言）
# 引数: $1=round 番号（1 始まり）$2=記録先 round ディレクトリ（絶対パス。round{i}）
# bash 3.2 互換。計測中はビルド・集計・転送を並走させない。run の差し替え・上書きは禁止。
# 外部入力をシェル文字列へ展開しない（変数は常に引用符つきで引数として渡す）。
set -uo pipefail
validate_abs() { # $1=変数名 $2=値。絶対パスのみ。引用符・バックスラッシュ・改行を拒否（A03）
  case "$2" in /*) ;; *) echo "$1 は絶対パスのみ: $2"; exit 2;; esac
  case "$2" in *\"*|*\'*|*\\*|*$'\n'*) echo "$1 に禁止文字を含む"; exit 2;; esac
}
ROOT="${ROOT:?計測ツリー}"; SHA="${EXPECT_SHA:?RULE の tree_sha}"
RN="${1:?round 番号}"; RD="${2:?round ディレクトリ}"
validate_abs ROOT "${ROOT}"; validate_abs RD "${RD}"
case "${RN}" in ''|*[!0-9]*) echo "round 番号は数字のみ"; exit 2;; esac
case "${SHA}" in *[!0-9a-f]*) echo "EXPECT_SHA は小文字 hex のみ"; exit 2;; esac
[ "${HW_LOCK_HELD:-}" = "1" ] || { echo "hw-lock 未取得（HW_LOCK_HELD=1 が必要）"; exit 2; }
BIN="${ROOT}/target/release/examples/cpu_gemm_nr_tail_4152"
[ ! -e "${RD}/raw" ] || { echo "${RD}/raw が既にある（上書き禁止）"; exit 2; }
mkdir -p "${RD}/raw" || exit 2

# ---- precheck（fail-closed）
[ "$(sysctl -n machdep.cpu.brand_string)" = "Apple M4" ] || { echo "機体が Apple M4 ではない"; exit 1; }
[ "$(cat "${ROOT}/.rev-stamp")" = "${SHA}" ] || { echo ".rev-stamp が RULE と不一致"; exit 1; }
[ -x "${BIN}" ] || { echo "example が未ビルド: ${BIN}"; exit 1; }

# ---- ゲート: load1 < 3.0 を 30 秒間隔で最大 30 分（hw-lock は上で宣言確認）
gate() {
  local i=0 raw l
  while [ "${i}" -le 60 ]; do
    if ! raw=$(sysctl -n vm.loadavg); then echo "INVALID sysctl-failed" > "${RD}/gate.txt"; return 2; fi
    l=$(printf '%s' "${raw}" | tr -d '{}' | awk '{print $1}')
    if ! printf '%s' "${l}" | grep -Eq '^[0-9]+(\.[0-9]+)?$'; then echo "INVALID load1-not-numeric" > "${RD}/gate.txt"; return 2; fi
    if awk -v x="${l}" 'BEGIN{exit !(x < 3.0)}'; then echo "PASS load1=${l} hw-lock=held $(date -u +%FT%TZ)" > "${RD}/gate.txt"; return 0; fi
    sleep 30; i=$((i + 1))
  done
  echo "TIMEOUT load1=${l} $(date -u +%FT%TZ)" > "${RD}/gate.txt"; return 1
}
gate || { echo "ゲート不通過: $(cat "${RD}/gate.txt")"; exit 1; }

if [ $((RN % 2)) -eq 1 ]; then ORDER=asc; else ORDER=desc; fi
RAYON_NUM_THREADS=1 "${BIN}" "r${RN}" t1 "${ORDER}" > "${RD}/raw/t1.txt" 2> "${RD}/raw/t1.err" || { echo "t1 失敗"; exit 1; }
"${BIN}" "r${RN}" native "${ORDER}" > "${RD}/raw/native.txt" 2> "${RD}/raw/native.err" || { echo "native 失敗"; exit 1; }
echo "round ${RN} done ${ORDER} $(date -u +%FT%TZ)"
