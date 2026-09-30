#!/bin/bash
# イシュー #2118: SME_MIN_K 候補ごとの GB10（DGX Spark・Grace CPU・SME 非対応）
# 非後退再確認オーケストレーション。判定規則は
# ../RULE.txt §8〜§9（事前登録。語彙は 1587/gb10/RULE-gb10.txt を継承）。
# #1978 残の `cpu-gemm-sme-fmopa-1587/gb10/orchestrate_gb10.sh` と同じ
# R0（sme_report プローブ）→ RT（after 腕の既存テスト）→ R1/R2
# （run_ab_sme_cpu.sh）の手順を、K・パッチ・LABEL を引数化し、事前登録コミット（SME2118_REGISTERED_BASE）の `git archive`
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
  echo "[dry-run] 2. git archive ${SME2118_REGISTERED_BASE} -> <work>/{before,after}・パッチ適用・指紋差分 1 件と定数行を assert"
  echo "[dry-run] 3. R0: 両腕で sme_report() が kernel_enabled: false（1587/gb10/sme-probe-* を再利用）"
  echo "[dry-run] 4. RT: after 腕で cargo test -p fandhe-ai-backend-cpu --release。FAIL は ${KNOWN_FAIL} の 1 件のみ許容（終了コード・結果行を検証し fail-closed）"
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
  sed "s#TREE_PLACEHOLDER#${WORK}/${t}#" "${REPO}/docs/perf/logs/cpu-gemm-sme-fmopa-1587/gb10/sme-probe-Cargo.toml" >"${P}/Cargo.toml" \
    || { echo "probe ${t} の Cargo.toml 生成に失敗 -> 中止"; exit 1; }
  cp "${REPO}/docs/perf/logs/cpu-gemm-sme-fmopa-1587/gb10/sme-probe-main.rs" "${P}/src/main.rs" \
    || { echo "probe ${t} の main.rs コピーに失敗 -> 中止"; exit 1; }
  (cd "${P}" && CARGO_TARGET_DIR="${WORK}/target-probe" cargo run --release -q) >"${WORK}/probe_${t}.log" 2>&1 \
    || { echo "probe ${t} 失敗 -> 中止"; tail -20 "${WORK}/probe_${t}.log"; exit 1; }
  sme2118_mask "${WORK}" <"${WORK}/probe_${t}.log" >"${LOGD}/sme_probe_${t}.log" \
    || { echo "probe ${t} ログのマスク収録に失敗 -> 中止"; exit 1; }
  echo "${t}: $(grep sme_report= "${LOGD}/sme_probe_${t}.log")" | tee -a "${LOGD}/sme_report.txt"
done
if ! grep -q 'kernel_enabled: false' "${LOGD}/sme_probe_after.log" \
  || ! grep -q 'kernel_enabled: false' "${LOGD}/sme_probe_before.log"; then
  echo "R0 不成立（kernel_enabled が false でない）-> 想定外・計測中止（記録のみ）"
  exit 1
fi

# RT: after 腕の既存テスト（非 #[ignore]）。既知 FAIL 1 件のみ許容（RULE.txt §9）
# --no-fail-fast: 既知 FAIL のあるテストバイナリで打ち切られず全バイナリの結果行を得る。
(cd "${WORK}/after" && CARGO_TARGET_DIR="${WORK}/target-after" cargo test -p fandhe-ai-backend-cpu --release --no-fail-fast) >"${WORK}/cargo_test_after.log" 2>&1
TEST_RC=$?
echo "RT rc=${TEST_RC}" >"${LOGD}/rt_result.txt"
# grep は一致 0 件で rc=1 になるため rc は見ず、収録（マスク）の成否だけを確認する。
grep -E '^test result|FAILED|panicked' "${WORK}/cargo_test_after.log" >"${WORK}/cargo_test_after.summary.raw"
sme2118_mask "${WORK}" <"${WORK}/cargo_test_after.summary.raw" >"${LOGD}/cargo_test_after.summary.log" \
  || { echo "RT サマリのマスク収録に失敗 -> 中止"; exit 1; }
FAILS=$(grep -E '^test .* \.\.\. FAILED$' "${WORK}/cargo_test_after.log" | sed -e 's/^test //' -e 's/ \.\.\. FAILED$//' | sort -u)
RESULT_LINES=$(grep -cE '^test result: ' "${WORK}/cargo_test_after.log")
# 異常終了行 = コンパイル失敗、または終了状態が 101 以外の `process didn't exit successfully`。
# cargo の通常のテスト失敗フッタ（error: test failed・exit status: 101）は異常に数えない（lib_trees.sh）。
ABNORMAL=$(sme2118_rt_abnormal_count "${WORK}/cargo_test_after.log")
# fail-closed（RULE.txt §8〜§9）: 終了コードと結果行の存在を検証する。FAIL 行の有無だけで pass にしない。
#  - rc=0: FAIL 行なし・結果行 1 件以上のときのみ pass
#  - rc!=0: rc=101（テスト失敗）かつ FAIL が既知 1 件のみ・結果行 1 件以上・異常終了行なしのときのみ pass（既知 FAIL 許容）
#  それ以外（起動・コンパイル失敗、途中終了、結果行なし、想定外 FAIL）は regression-suspect
RT_REASON=""
if [ "${RESULT_LINES}" -lt 1 ]; then
  RT_REASON="test result 行なし（起動失敗・途中終了の疑い）"
elif [ "${TEST_RC}" -eq 0 ]; then
  [ -z "${FAILS}" ] || RT_REASON="rc=0 だが FAIL 行あり"
elif [ "${TEST_RC}" -ne 101 ] || [ "${ABNORMAL}" -gt 0 ]; then
  RT_REASON="rc=${TEST_RC}・異常終了行=${ABNORMAL}（テスト失敗以外の終了）"
elif [ "${FAILS}" != "${KNOWN_FAIL}" ]; then
  RT_REASON="既知 FAIL 以外あり、または FAIL 名を抽出できない"
fi
if [ -z "${RT_REASON}" ]; then
  echo "rt_verdict=pass known_fail_only=$([ -n "${FAILS}" ] && echo yes || echo none)" >>"${LOGD}/rt_result.txt"
else
  echo "rt_verdict=regression-suspect（${RT_REASON}）" >>"${LOGD}/rt_result.txt"
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
MASK_RC=0
sme2118_mask "${WORK}" <"${WORK}/run_ab.log" >"${LOGD}/run_ab.log" || MASK_RC=1
# 収録の失敗（必須成果物の欠損・コピー失敗）は rt_result.txt の `collect rc=` へ記録し、末尾で非ゼロ終了へ
# 伝播する。aggregate.py は `collect rc=0` の記録がない系列を判定不能にする（RULE.txt §13）。
COLLECT_RC=0
sme2118_collect_r1r2 "${BENCH}" "${LABEL}" "${LOGD}/r1r2" "${WORK}" || COLLECT_RC=1
[ "${MASK_RC}" -eq 0 ] || COLLECT_RC=1
echo "collect rc=${COLLECT_RC}" | tee -a "${LOGD}/rt_result.txt"
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
[ "${RC}" -eq 0 ] || { echo "run_ab_sme_cpu.sh が非ゼロ終了（rc=${RC}）" >&2; exit 1; }
[ "${COLLECT_RC}" -eq 0 ] || { echo "成果物の収録に失敗（collect rc=${COLLECT_RC}）" >&2; exit 1; }
