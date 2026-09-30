#!/bin/bash
# イシュー #2118: SME_MIN_K 候補（64／128／256）の Apple M4 Max 再実測
# オーケストレーション（R4 格子 1 系列 + R1／R2 を候補ごと）。判定規則は
# 同ディレクトリの RULE.txt（事前登録）が正。#1978 の
# `cpu-gemm-sme-fmopa-1587/orchestrate_m4max.sh`（R4）と
# `scripts/bench/framework-compare/run_ab_sme_cpu.sh`（R1／R2）を、
# フォークせずラップして候補ごとの after ツリーを渡す（#2117 と同型の
# 「Linux で基盤まで作り実機セッションで実測」運用）。
#
# 使い方:
#   orchestrate_m4max.sh [--dry-run] r4            # R4 格子（候補共通・1 系列）
#   orchestrate_m4max.sh [--dry-run] r1 <64|128|256>
#   orchestrate_m4max.sh [--dry-run] all           # r4 → r1 64 → 128 → 256
# 収録先: <このディレクトリ>/m4max/（生ログは差し替え禁止。既存があれば停止）。
# 事前に事前登録コミット（lib_trees.sh の SME2118_REGISTERED_BASE）を git archive した一時ツリーで release ビルドするため、計測中に
# ビルドを並走させない。main の SME 定数は変更しない（本番切替は #2119）。
# bash 3.2（macOS 標準）でも動く書き方に限る。
# 失敗の伝播（RULE.txt §13）: pipefail でパイプ先頭（git archive 等）の失敗も検出し、収録・マスク・
# コピーの戻り値は全て確認して非ゼロで返す（set -e は使わず個別に `|| return 1`）。
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=./lib_trees.sh
. "${HERE}/lib_trees.sh"
REPO=$(git -C "${HERE}" rev-parse --show-toplevel) || exit 1
OUTD="${HERE}/m4max"

DRY=0
ARGS=""
for a in "$@"; do
  if [ "$a" = "--dry-run" ]; then DRY=1; else ARGS="${ARGS} ${a}"; fi
done
# shellcheck disable=SC2086
set -- ${ARGS}
SUB=${1:-}
KARG=${2:-}
case "${SUB}" in
  r4 | all) ;;
  r1) sme2118_validate_k "${KARG}" || exit 2 ;;
  *) echo "usage: $0 [--dry-run] r4 | r1 <64|128|256> | all" >&2; exit 2 ;;
esac

CMD="cargo test -p fandhe-ai-backend-cpu --release --lib -- --ignored sme_vs_neon_ab_r4_grid --nocapture --test-threads=1"

# 負荷ゲート: load1 < 8.0 を 30 秒間隔で最大 30 分。負荷取得不能は unavailable
# （正式系列から外し参考扱い。RULE.txt §11）。結果は status 変数へ。
wait_load_gate() {
  local logf=$1 tag=$2 waited=0 l1 ok
  status="timeout"
  while [ "${waited}" -lt 1800 ]; do
    l1=$(sysctl -n vm.loadavg 2>/dev/null | awk '{print $2}')
    case "${l1}" in
      '' | *[!0-9.]*) status="unavailable"; l1="NA"; break ;;
    esac
    ok=$(awk -v a="${l1}" 'BEGIN{print (a<8.0)?1:0}')
    if [ "${ok}" = "1" ]; then status="pass"; break; fi
    sleep 30
    waited=$((waited + 30))
  done
  echo "${tag} gate=${status} load1=${l1} waited_s=${waited} at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"${logf}"
}

do_r4() {
  if [ "${DRY}" = "1" ]; then
    echo "[dry-run] git archive ${SME2118_REGISTERED_BASE} -> <work>/before; (cd <work>/before && ${CMD}) x5 -> ${OUTD}/sme_r4_grid_run{1..5}.log (+ .raw.log)"
    return 0
  fi
  local i f work rc grc status
  mkdir -p "${OUTD}" || return 1
  for i in 1 2 3 4 5; do
    for f in "sme_r4_grid_run${i}.log" "sme_r4_grid_run${i}.raw.log"; do
      if [ -e "${OUTD}/${f}" ]; then echo "${f} が既に存在する（差し替え禁止）" >&2; return 1; fi
    done
  done
  sme2118_resolve_base "${REPO}" || return 1
  work=$(mktemp -d) || return 1
  mkdir -p "${work}/before" || return 1
  git -C "${REPO}" archive "${SME2118_REGISTERED_BASE}" | tar -x -C "${work}/before" || return 1
  {
    echo "head=${SME2118_REGISTERED_BASE}"
    echo "current_head=$(git -C "${REPO}" rev-parse HEAD)"
  } >"${OUTD}/r4_head.txt"
  (cd "${work}/before" && CARGO_TARGET_DIR="${work}/target" cargo test -p fandhe-ai-backend-cpu --release --lib --no-run) \
    || { echo "R4 ビルド失敗" >&2; return 1; }
  for i in 1 2 3 4 5; do
    wait_load_gate "${OUTD}/load_gate_r4.log" "run${i}"
    (cd "${work}/before" && CARGO_TARGET_DIR="${work}/target" ${CMD}) >"${work}/raw${i}.log" 2>&1
    rc=$?
    sme2118_mask "${work}" <"${work}/raw${i}.log" >"${OUTD}/sme_r4_grid_run${i}.raw.log" \
      || { echo "run${i}: raw ログのマスク収録に失敗" >&2; return 1; }
    grep -E "^(variant=|test |SME )" "${OUTD}/sme_r4_grid_run${i}.raw.log" >"${OUTD}/sme_r4_grid_run${i}.log"
    grc=$?
    echo "run${i} exit=${rc} grep_exit=${grc} end_load1=$(sysctl -n vm.loadavg 2>/dev/null | awk '{print $2}')" >>"${OUTD}/load_gate_r4.log"
    if [ "${rc}" -ne 0 ] || [ "${grc}" -ne 0 ]; then
      echo "run${i}: 計測プロセス失敗（exit=${rc} grep_exit=${grc}）。.raw.log を確認" >&2
      return 1
    fi
  done
  echo "series done" >>"${OUTD}/load_gate_r4.log"
  rm -rf "${work}"
}

do_r1() {
  local k=$1 label="2118-m4max-k$1" patch="${HERE}/on-arm-k$1.patch" work bench
  if [ "${DRY}" = "1" ]; then
    echo "[dry-run] K=${k}: git archive ${SME2118_REGISTERED_BASE} -> <work>/{before,after}; patch ${patch} を after のみへ適用し指紋差分 1 件・定数行を assert"
    echo "[dry-run]   AB_BEFORE_FACADE_PATH=<work>/before/crates/facade AB_AFTER_FACADE_PATH=<work>/after/crates/facade AB_DEVICE=cpu bash run_ab_sme_cpu.sh ${label}"
    echo "[dry-run]   -> mask して ${OUTD}/r1r2/k${k}/ へ収録・env_info.txt"
    return 0
  fi
  sme2118_validate_label "${label}" || return 1
  if [ -e "${OUTD}/r1r2/k${k}" ] && ls "${OUTD}/r1r2/k${k}"/* >/dev/null 2>&1; then
    echo "K=${k} の収録が既に存在する（差し替え禁止。再実行は別 LABEL で）" >&2
    return 1
  fi
  mkdir -p "${OUTD}/r1r2/k${k}" || return 1
  work=$(mktemp -d) || return 1
  sme2118_prepare_trees "${REPO}" "${work}" "${k}" "${patch}" "${OUTD}/r1r2/k${k}" || return 1
  bench="${work}/before/scripts/bench/framework-compare"
  (cd "${bench}" && AB_BEFORE_FACADE_PATH="${work}/before/crates/facade" \
    AB_AFTER_FACADE_PATH="${work}/after/crates/facade" AB_DEVICE=cpu \
    bash run_ab_sme_cpu.sh "${label}") >"${work}/run_ab.log" 2>&1
  local rc=$?
  local mask_rc=0 collect_rc=0
  sme2118_mask "${work}" <"${work}/run_ab.log" >"${OUTD}/r1r2/k${k}/run_ab.log" || mask_rc=1
  # 収録の失敗（必須成果物の欠損・コピー失敗）は collect_exit へ記録し、下で非ゼロ終了へ伝播する。
  # aggregate.py は collect_exit=0 の記録がない系列を判定不能にする（RULE.txt §13）。
  sme2118_collect_r1r2 "${bench}" "${label}" "${OUTD}/r1r2/k${k}" "${work}" || collect_rc=1
  [ "${mask_rc}" -eq 0 ] || collect_rc=1
  {
    echo "label=${label} K=${k} run_ab_exit=${rc} collect_exit=${collect_rc}"
    echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "uname=$(uname -srm)"
    echo "chip=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
    echo "rustc=$(rustc -V)"
    echo "cargo=$(cargo -V)"
    echo "hostname=masked"
  } >"${OUTD}/r1r2/k${k}/env_info.txt"
  rm -rf "${work}"
  if [ "${rc}" -ne 0 ]; then
    echo "run_ab_sme_cpu.sh が非ゼロ終了（exit=${rc}）。run_ab.log と skipped ログを確認" >&2
    return 1
  fi
  if [ "${collect_rc}" -ne 0 ]; then
    echo "成果物の収録に失敗（必須成果物の欠損またはコピー失敗）。標準エラーの一覧と ${OUTD}/r1r2/k${k}/ を確認" >&2
    return 1
  fi
}

case "${SUB}" in
  r4) do_r4 || exit 1 ;;
  r1) do_r1 "${KARG}" || exit 1 ;;
  all)
    do_r4 || exit 1
    for k in 64 128 256; do do_r1 "${k}" || exit 1; done
    ;;
esac
exit 0
