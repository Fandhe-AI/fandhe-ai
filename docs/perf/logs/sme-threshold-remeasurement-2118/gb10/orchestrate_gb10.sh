#!/bin/bash
# イシュー #2118: SME_MIN_K 候補ごとの GB10（DGX Spark・Grace CPU・SME 非対応）
# 非後退再確認オーケストレーション。判定規則は
# ../RULE.txt §8〜§9（事前登録。語彙は 1587/gb10/RULE-gb10.txt を継承）。
# #1978 残の `cpu-gemm-sme-fmopa-1587/gb10/orchestrate_gb10.sh` と同じ
# R0（sme_report プローブ）→ RT（after 腕の既存テスト）→ R1/R2
# （run_ab_sme_cpu.sh）の手順を、K・パッチ・LABEL を引数化し、`git archive HEAD`
# 由来の一時ツリー（before／after(K)）で行う。main の SME 定数は変更しない。
#
# 使い方: orchestrate_gb10.sh [--dry-run] <64|128|256>
#   順序は k64 → k128 → k256（RULE.txt §10）。差し替え禁止（既存収録があれば停止）。
# 収録先: <このディレクトリ>/k<K>/。マスク（RULE.txt §12）を経てから保存する。
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PARENT=$(cd "${HERE}/.." && pwd)
# shellcheck source=../lib_trees.sh
. "${PARENT}/lib_trees.sh"
REPO=$(git -C "${HERE}" rev-parse --show-toplevel) || exit 1
export PATH="${HOME}/.cargo/bin:${PATH}"

DRY=0
K=""
for a in "$@"; do
  if [ "$a" = "--dry-run" ]; then DRY=1; else K="$a"; fi
done
sme2118_validate_k "${K}" || { echo "usage: $0 [--dry-run] <64|128|256>" >&2; exit 2; }
LABEL="2118-gb10-k${K}"
sme2118_validate_label "${LABEL}" || exit 2
PATCH="${PARENT}/on-arm-k${K}.patch"
OUT="${HERE}/k${K}"
KNOWN_FAIL="gemm_blis::tests::sme_production_enabled_is_false_pending_measurement"

if [ "${DRY}" = "1" ]; then
  echo "[dry-run] K=${K} LABEL=${LABEL} patch=${PATCH} out=${OUT}"
  echo "[dry-run] 1. 外側専有ゲート（load1<1.0 かつ gpu_util==0・最大 20 回・30 秒間隔）を記録"
  echo "[dry-run] 2. git archive HEAD -> <work>/{before,after}・パッチ適用・指紋差分 1 件と定数行を assert"
  echo "[dry-run] 3. R0: 両腕で sme_report() が kernel_enabled: false（1587/gb10/sme-probe-* を再利用）"
  echo "[dry-run] 4. RT: after 腕で cargo test -p fandhe-ai-backend-cpu --release。FAIL は ${KNOWN_FAIL} の 1 件のみ許容"
  echo "[dry-run] 5. R1/R2: run_ab_sme_cpu.sh ${LABEL}（AB_DEVICE=cpu）"
  echo "[dry-run] 6. マスクして ${OUT}/ へ収録・env_info.txt"
  exit 0
fi

if [ -e "${OUT}" ] && ls "${OUT}"/* >/dev/null 2>&1; then
  echo "error: ${OUT} に既存収録がある（差し替え禁止。再実行は別 LABEL で）" >&2
  exit 1
fi
mkdir -p "${OUT}" || exit 1
WORK=$(mktemp -d) || exit 1
LOGD="${OUT}"

gate() { # 外側専有ゲート（記録のみ。不通過でも続行し参考扱い）
  local i l1 gu
  for i in $(seq 1 20); do
    l1=$(cut -d' ' -f1 /proc/loadavg)
    gu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    if awk -v l="${l1}" -v g="${gu:-x}" 'BEGIN{exit !(l<1.0 && g==0)}'; then
      echo "start attempt=${i} load1=${l1} gpu_util=${gu} pass" | tee -a "${LOGD}/load_gate_outer.log"
      return 0
    fi
    echo "start attempt=${i} load1=${l1} gpu_util=${gu:-NA} wait" >>"${LOGD}/load_gate_outer.log"
    sleep 30
  done
  echo "start attempt=20 load1=${l1} gpu_util=${gu:-NA} fail(reference)" | tee -a "${LOGD}/load_gate_outer.log"
  return 1
}

echo "start $(date -u +%FT%TZ) K=${K}"
uptime >"${LOGD}/uptime_before.txt"
sme2118_prepare_trees "${REPO}" "${WORK}" "${K}" "${PATCH}" "${LOGD}" || exit 1
gate || echo "外側専有ゲート不通過: この系列は参考扱い（RULE.txt §8）"

# R0: sme_report プローブ（両腕。プローブ定義は #1978 の 1587/gb10 を再利用）
: >"${LOGD}/sme_report.txt"
for t in before after; do
  P="${WORK}/probe-${t}"
  mkdir -p "${P}/src" || exit 1
  sed "s#TREE_PLACEHOLDER#${WORK}/${t}#" "${REPO}/docs/perf/logs/cpu-gemm-sme-fmopa-1587/gb10/sme-probe-Cargo.toml" >"${P}/Cargo.toml"
  cp "${REPO}/docs/perf/logs/cpu-gemm-sme-fmopa-1587/gb10/sme-probe-main.rs" "${P}/src/main.rs"
  (cd "${P}" && CARGO_TARGET_DIR="${WORK}/target-probe" cargo run --release -q) >"${WORK}/probe_${t}.log" 2>&1 \
    || { echo "probe ${t} 失敗 -> 中止"; tail -20 "${WORK}/probe_${t}.log"; exit 1; }
  sme2118_mask "${WORK}" <"${WORK}/probe_${t}.log" >"${LOGD}/sme_probe_${t}.log"
  echo "${t}: $(grep sme_report= "${LOGD}/sme_probe_${t}.log")" | tee -a "${LOGD}/sme_report.txt"
done
if ! grep -q 'kernel_enabled: false' "${LOGD}/sme_probe_after.log" \
  || ! grep -q 'kernel_enabled: false' "${LOGD}/sme_probe_before.log"; then
  echo "R0 不成立（kernel_enabled が false でない）-> 想定外・計測中止（記録のみ）"
  exit 1
fi

# RT: after 腕の既存テスト（非 #[ignore]）。既知 FAIL 1 件のみ許容（RULE.txt §9）
(cd "${WORK}/after" && CARGO_TARGET_DIR="${WORK}/target-after" cargo test -p fandhe-ai-backend-cpu --release) >"${WORK}/cargo_test_after.log" 2>&1
echo "RT rc=$?" >"${LOGD}/rt_result.txt"
grep -E '^test result|FAILED|panicked' "${WORK}/cargo_test_after.log" | sme2118_mask "${WORK}" >"${LOGD}/cargo_test_after.summary.log"
FAILS=$(grep -E '^test .* \.\.\. FAILED$' "${WORK}/cargo_test_after.log" | sed -e 's/^test //' -e 's/ \.\.\. FAILED$//' | sort -u)
if [ -z "${FAILS}" ] || [ "${FAILS}" = "${KNOWN_FAIL}" ]; then
  echo "rt_verdict=pass known_fail_only=$([ -n "${FAILS}" ] && echo yes || echo none)" >>"${LOGD}/rt_result.txt"
else
  echo "rt_verdict=regression-suspect（既知 FAIL 以外あり）" >>"${LOGD}/rt_result.txt"
  printf '%s\n' "${FAILS}" >>"${LOGD}/rt_result.txt"
fi
cat "${LOGD}/rt_result.txt"

# R1/R2: framework-compare A/B（内側 load ゲート 8.0 はスクリプト固定）
BENCH="${WORK}/before/scripts/bench/framework-compare"
(cd "${BENCH}" && AB_BEFORE_FACADE_PATH="${WORK}/before/crates/facade" \
  AB_AFTER_FACADE_PATH="${WORK}/after/crates/facade" AB_DEVICE=cpu \
  bash run_ab_sme_cpu.sh "${LABEL}") >"${WORK}/run_ab.log" 2>&1
RC=$?
echo "run_ab rc=${RC}" | tee -a "${LOGD}/rt_result.txt"
sme2118_mask "${WORK}" <"${WORK}/run_ab.log" >"${LOGD}/run_ab.log"
sme2118_collect_r1r2 "${BENCH}" "${LABEL}" "${LOGD}/r1r2" "${WORK}"
{
  echo "date_utc=$(date -u +%FT%TZ)"
  echo "label=${LABEL} K=${K}"
  echo "uname=$(uname -srm)"
  echo "rustc=$(rustc -V)"
  echo "cargo=$(cargo -V)"
  echo "cpu_model=$(lscpu | grep -i 'model name' | head -1 | sed 's/ \+/ /g')"
  echo "nproc=$(nproc)"
  echo "rayon_num_threads=${RAYON_NUM_THREADS:-unset}"
  echo "outer_gate=load1<1.0 && gpu_util==0 (load_gate_outer.log)"
  echo "hostname=masked"
} >"${LOGD}/env_info.txt"
uptime >"${LOGD}/uptime_after.txt"
rm -rf "${WORK}"
echo "done. $(date -u +%FT%TZ)"
[ "${RC}" -eq 0 ] || exit 1
