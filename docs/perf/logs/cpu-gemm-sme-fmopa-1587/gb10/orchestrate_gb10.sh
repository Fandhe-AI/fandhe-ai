#!/usr/bin/env bash
# GB10 #1978 残: SME_PRODUCTION_ENABLED=true 腕の非 SME 環境フォールバック非後退確認。
# 判定規則は docs/perf/logs/cpu-gemm-sme-fmopa-1587/gb10/RULE-gb10.txt（事前登録）。
set -uo pipefail
export PATH="${HOME}/.cargo/bin:${PATH}"
W="${HOME}/work"
BEFORE="${W}/rust-ai-library-run"
AFTER="${W}/rust-ai-library-1978-after"
LOGD="${W}/gb10-1978"; mkdir -p "${LOGD}"
LABEL="1978-gb10"
SHA="$(cat "${BEFORE}/.rev-stamp")"
[[ -n "${SHA}" ]] || { echo "ERROR: .rev-stamp が空"; exit 1; }
echo "start $(date -u +%FT%TZ) sha=${SHA}"
uptime > "${LOGD}/uptime_before.txt"

gate() { # 外側専有ゲート: load1 < 1.0 かつ gpu_util 0%。最大 20 回・30 秒間隔。
  local tag=$1 i l1 gu
  for i in $(seq 1 20); do
    l1=$(cut -d' ' -f1 /proc/loadavg)
    gu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')
    if awk -v l="${l1}" -v g="${gu}" 'BEGIN{exit !(l<1.0 && g==0)}'; then
      echo "${tag} attempt=${i} load1=${l1} gpu_util=${gu} pass" | tee -a "${LOGD}/load_gate_outer.log"; return 0
    fi
    echo "${tag} attempt=${i} load1=${l1} gpu_util=${gu} wait" >> "${LOGD}/load_gate_outer.log"; sleep 30
  done
  echo "${tag} attempt=20 load1=${l1} gpu_util=${gu} fail" | tee -a "${LOGD}/load_gate_outer.log"; return 1
}

# after ツリー: before の複製 + on-arm.patch（SME_PRODUCTION_ENABLED のみ反転）
echo "-- prepare after tree ($(date -u +%FT%TZ))"
rsync -a --delete --exclude target --exclude 'target-*' --exclude .git "${BEFORE}/" "${AFTER}/" || { echo "rsync after rc=$?"; exit 1; }
PATCH="${BEFORE}/docs/perf/logs/cpu-gemm-sme-fmopa-1587/on-arm.patch"
sha256sum "${PATCH}" | sed 's#/home/[^/]*#<home>#g' > "${LOGD}/patch_sha256.txt"
( cd "${AFTER}" && patch -p1 --forward < "${PATCH}" ) > "${LOGD}/patch_apply.log" 2>&1 || { echo "patch rc=$? -> 中止"; cat "${LOGD}/patch_apply.log"; exit 1; }
echo "${SHA}+on-arm.patch" > "${AFTER}/.rev-stamp"
{
  echo "before: $(grep -n 'SME_PRODUCTION_ENABLED: bool' "${BEFORE}/crates/backend-cpu/src/gemm_blis/mod.rs")"
  echo "after:  $(grep -n 'SME_PRODUCTION_ENABLED: bool' "${AFTER}/crates/backend-cpu/src/gemm_blis/mod.rs")"
} > "${LOGD}/gate_constant.txt"; cat "${LOGD}/gate_constant.txt"

# ツリー指紋（Mac の git ls-files 一覧に対する sha256sum。ビルド前に採取）
for t in before after; do
  T="${BEFORE}"; [[ "${t}" == after ]] && T="${AFTER}"
  ( cd "${T}" && tr '\n' '\0' < "${W}/filelist-1978-gb10.txt" | xargs -0 sha256sum ) > "${LOGD}/fp-${t}.txt" 2>/dev/null
done
diff "${LOGD}/fp-before.txt" "${LOGD}/fp-after.txt" > "${LOGD}/fp-diff.txt"; echo "fp diff lines=$(wc -l < "${LOGD}/fp-diff.txt")"

gate "start"

# R0: sme_report プローブ（両腕）
echo "-- R0 sme_report ($(date -u +%FT%TZ))"
: > "${LOGD}/sme_report.txt"
for t in before after; do
  T="${BEFORE}"; [[ "${t}" == after ]] && T="${AFTER}"
  P="${W}/sme-probe-1978-${t}"; rm -rf "${P}"; cp -r "${W}/sme-probe-1978" "${P}"
  sed -i "s#TREE_PLACEHOLDER#${T}#" "${P}/Cargo.toml"
  ( cd "${P}" && cargo run --release -q ) > "${LOGD}/sme_probe_${t}.log" 2>&1 || { echo "probe ${t} rc=$? -> 中止"; tail -20 "${LOGD}/sme_probe_${t}.log"; exit 1; }
  echo "${t}: $(grep sme_report= "${LOGD}/sme_probe_${t}.log")" | tee -a "${LOGD}/sme_report.txt"
done
grep -m1 Features /proc/cpuinfo | grep -oE '\b(sme|smef32f32|sve2?)\b' | sort -u | tr '\n' ' ' | sed 's/^/cpuinfo_tokens: /;s/$/\n/' >> "${LOGD}/sme_report.txt"
grep -q 'kernel_enabled: false' "${LOGD}/sme_probe_after.log" || { echo "R0: kernel_enabled が false でない -> 想定外・中止"; exit 1; }

# RT: after 腕の既存テスト群（非 ignore）
echo "-- RT cargo test after ($(date -u +%FT%TZ))"
( cd "${AFTER}" && CARGO_TARGET_DIR="${W}/target-1978-after" cargo test -p fandhe-ai-backend-cpu --release ) > "${LOGD}/cargo_test_after.log" 2>&1; echo "RT rc=$?" | tee -a "${LOGD}/cargo_test_after.log"
grep -E '^test result|FAILED|panicked' "${LOGD}/cargo_test_after.log" | head -20

# R1/R2: framework-compare A/B（内側 load ゲート 8.0 はスクリプト固定）
echo "-- R1/R2 run_ab_sme_cpu.sh ($(date -u +%FT%TZ))"
cd "${BEFORE}/scripts/bench/framework-compare" || exit 1
AB_BEFORE_FACADE_PATH="${BEFORE}/crates/facade" AB_AFTER_FACADE_PATH="${AFTER}/crates/facade" AB_DEVICE=cpu \
  bash run_ab_sme_cpu.sh "${LABEL}" > "${LOGD}/run_ab.log" 2>&1; echo "run_ab rc=$?" | tee -a "${LOGD}/run_ab.log"
mkdir -p "${LOGD}/r1r2"
cp results/raw/*"${LABEL}"* "${LOGD}/r1r2/" 2>/dev/null
rm -f "${LOGD}"/r1r2/bench-fandhe-1978-*
cp compare-*-1978-cpu-"${LABEL}".md compare-*-1978-cpu-"${LABEL}".err "${LOGD}/r1r2/" 2>/dev/null
ls "${LOGD}/r1r2"
for t in gemm train infer; do echo "== ${t}"; grep -E '^\|' "compare-${t}-1978-cpu-${LABEL}.md" | head -40; done

{
  echo "date_utc=$(date -u +%FT%TZ)"
  echo "rev_stamp_before=${SHA}"
  echo "rev_stamp_after=$(cat "${AFTER}/.rev-stamp")"
  echo "uname=$(uname -srm)"
  echo "rustc=$(rustc -V)"; echo "cargo=$(cargo -V)"
  echo "cpu_model=$(/usr/bin/lscpu | grep -i 'model name' | head -1 | sed 's/ \+/ /g')"
  echo "nproc=$(nproc)"
  echo "rayon_num_threads=${RAYON_NUM_THREADS:-unset}"
  echo "tree_before=<home>/work/rust-ai-library-run"
  echo "tree_after=<home>/work/rust-ai-library-1978-after"
  echo "outer_gate=load1<1.0 && gpu_util==0 (load_gate_outer.log)"
} > "${LOGD}/env_info.txt"
uptime > "${LOGD}/uptime_after.txt"
echo "done. $(date -u +%FT%TZ)"
