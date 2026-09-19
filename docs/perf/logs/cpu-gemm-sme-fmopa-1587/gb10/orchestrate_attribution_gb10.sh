#!/usr/bin/env bash
# GB10 #2053（規則・パッチは #2052）: gemm cpu 1024/reuse の 5/5 一貫後退（#1978 残・§5.5）の帰属切り分け。
# 3 腕（before=main／after′=on-arm-prime.patch／after=on-arm.patch）・2 組（before vs after′・after′ vs after）の
# プロセス分離 A/B を run_ab_sme_cpu.sh（5 round・起動順反転・内側 load ゲート 8.0）で実行する。
# 判定規則は同ディレクトリの RULE-attribution.txt（事前登録）。既存の orchestrate_gb10.sh（2 腕）は変更しない。
set -uo pipefail
export PATH="${HOME}/.cargo/bin:${PATH}"
W="${HOME}/work"
BEFORE="${W}/rust-ai-library-run"
PRIME="${W}/rust-ai-library-2053-prime"
AFTER="${W}/rust-ai-library-2053-after"
LOGD="${W}/gb10-2053"; mkdir -p "${LOGD}"
GB10D="docs/perf/logs/cpu-gemm-sme-fmopa-1587/gb10"
SHA="$(cat "${BEFORE}/.rev-stamp")"
[[ -n "${SHA}" ]] || { echo "ERROR: .rev-stamp が空"; exit 1; }
[[ -f "${W}/filelist-2053-gb10.txt" ]] || { echo "ERROR: ${W}/filelist-2053-gb10.txt が無い"; exit 1; }
echo "start $(date -u +%FT%TZ) sha=${SHA}"
uptime > "${LOGD}/uptime_before.txt"

gate() { # 外側専有ゲート: load1 < 1.0 かつ gpu_util 0%。最大 20 回・30 秒間隔（不通過でも実行し「参考」表記）
  local tag=$1 i l1 gu
  for i in $(seq 1 20); do
    l1=$(cut -d' ' -f1 /proc/loadavg)
    gu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')
    if awk -v l="${l1}" -v g="${gu}" 'BEGIN{exit !(l<1.0 && g==0)}'; then
      echo "${tag} attempt=${i} load1=${l1} gpu_util=${gu} pass" | tee -a "${LOGD}/load_gate_outer.log"; return 0
    fi
    echo "${tag} attempt=${i} load1=${l1} gpu_util=${gu} wait" >> "${LOGD}/load_gate_outer.log"; sleep 30
  done
  echo "${tag} attempt=20 load1=${l1} gpu_util=${gu} fail (参考実行へ続行)" | tee -a "${LOGD}/load_gate_outer.log"; return 1
}

# 腕ツリー: before の複製 + パッチ（prime=on-arm-prime.patch／after=on-arm.patch）
: > "${LOGD}/patch_sha256.txt"
prepare_arm() { # $1=ツリー $2=パッチ相対パス $3=腕名
  local T=$1 P="${BEFORE}/${2}" name=$3
  echo "-- prepare ${name} tree ($(date -u +%FT%TZ))"
  rsync -a --delete --exclude target --exclude 'target-*' --exclude .git "${BEFORE}/" "${T}/" || { echo "rsync ${name} rc=$?"; exit 1; }
  sha256sum "${P}" | sed 's#/home/[^/]*#<home>#g' >> "${LOGD}/patch_sha256.txt"
  ( cd "${T}" && patch -p1 --forward < "${P}" ) > "${LOGD}/patch_apply_${name}.log" 2>&1 || { echo "patch ${name} rc=$? -> 中止"; cat "${LOGD}/patch_apply_${name}.log"; exit 1; }
  echo "${SHA}+$(basename "${P}")" > "${T}/.rev-stamp"
}
prepare_arm "${PRIME}" "${GB10D}/on-arm-prime.patch" prime
prepare_arm "${AFTER}" "docs/perf/logs/cpu-gemm-sme-fmopa-1587/on-arm.patch" after
{
  for t in before prime after; do
    T="${BEFORE}"; [[ "${t}" == prime ]] && T="${PRIME}"; [[ "${t}" == after ]] && T="${AFTER}"
    echo "${t}: $(grep -n 'SME_PRODUCTION_ENABLED: bool' "${T}/crates/backend-cpu/src/gemm_blis/mod.rs")"
    echo "${t}: $(grep -n 'let Some(kernel) = ' "${T}/crates/backend-cpu/src/gemm_blis/mod.rs" | head -1)"
  done
} > "${LOGD}/gate_constant.txt"; cat "${LOGD}/gate_constant.txt"

# ツリー指紋（3 腕。各パッチ腕と before の差分は mod.rs の 1 件のみ。不一致なら計測前に中止）
for t in before prime after; do
  T="${BEFORE}"; [[ "${t}" == prime ]] && T="${PRIME}"; [[ "${t}" == after ]] && T="${AFTER}"
  ( cd "${T}" && tr '\n' '\0' < "${W}/filelist-2053-gb10.txt" | xargs -0 sha256sum ) > "${LOGD}/fp-${t}.txt" 2>/dev/null
done
[[ $(wc -l < "${LOGD}/fp-before.txt") -gt 100 ]] || { echo "ERROR: 指紋の行数が少なすぎる"; exit 1; }
for t in prime after; do
  diff "${LOGD}/fp-before.txt" "${LOGD}/fp-${t}.txt" > "${LOGD}/fp-diff-${t}.txt"
  NCH=$(grep -cE "^[<>]" "${LOGD}/fp-diff-${t}.txt"); NMOD=$(grep -cE "^[<>].*crates/backend-cpu/src/gemm_blis/mod\.rs$" "${LOGD}/fp-diff-${t}.txt")
  echo "fp diff before..${t}: changed_lines=${NCH} mod_lines=${NMOD}"
  [[ "${NCH}" -eq 2 && "${NMOD}" -eq 2 ]] || { echo "ERROR: ${t} の指紋差分が mod.rs の 1 件でない -> 中止"; head "${LOGD}/fp-diff-${t}.txt"; exit 1; }
done

gate "start" || true

# R0: sme_report プローブ（3 腕）
echo "-- R0 sme_report ($(date -u +%FT%TZ))"
: > "${LOGD}/sme_report.txt"
for t in before prime after; do
  T="${BEFORE}"; [[ "${t}" == prime ]] && T="${PRIME}"; [[ "${t}" == after ]] && T="${AFTER}"
  P="${W}/sme-probe-2053-${t}"; rm -rf "${P}"; cp -r "${W}/sme-probe-1978" "${P}"
  sed -i "s#TREE_PLACEHOLDER#${T}#" "${P}/Cargo.toml"
  ( cd "${P}" && cargo run --release -q ) > "${LOGD}/sme_probe_${t}.log" 2>&1 || { echo "probe ${t} rc=$? -> 中止"; tail -20 "${LOGD}/sme_probe_${t}.log"; exit 1; }
  echo "${t}: $(grep sme_report= "${LOGD}/sme_probe_${t}.log")" | tee -a "${LOGD}/sme_report.txt"
  grep -q 'os_flag: false, svl_bytes: None, kernel_enabled: false' "${LOGD}/sme_probe_${t}.log" || { echo "R0 不成立（${t}）-> undetermined・中止"; exit 1; }
done

# 2 組 A/B（run_ab_sme_cpu.sh。LABEL は新規・既存 JSONL があれば fail-closed 停止）
run_pair() { # $1=LABEL $2=AB_BEFORE ツリー $3=AB_AFTER ツリー $4=組名
  local label=$1 bt=$2 at=$3 name=$4
  echo "-- A/B ${name} (${label}) $(date -u +%FT%TZ)"
  gate "${name}" || true
  ( cd "${BEFORE}/scripts/bench/framework-compare" && \
    AB_BEFORE_FACADE_PATH="${bt}/crates/facade" AB_AFTER_FACADE_PATH="${at}/crates/facade" AB_DEVICE=cpu \
    bash run_ab_sme_cpu.sh "${label}" ) > "${LOGD}/run_ab-${name}.log" 2>&1; echo "run_ab ${name} rc=$?" | tee -a "${LOGD}/run_ab-${name}.log"
  local FC="${BEFORE}/scripts/bench/framework-compare"
  mkdir -p "${LOGD}/r1r2-${name}"
  cp "${FC}"/results/raw/*"${label}"* "${LOGD}/r1r2-${name}/" 2>/dev/null
  rm -f "${LOGD}/r1r2-${name}"/bench-fandhe-*
  cp "${FC}"/compare-*-"${label}".md "${FC}"/compare-*-"${label}".err "${LOGD}/r1r2-${name}/" 2>/dev/null
  for t in gemm train infer; do echo "== ${name} ${t}"; grep -E '^\|' "${FC}/compare-${t}-1978-cpu-${label}.md" 2>/dev/null | head -40; done
}
run_pair "2053-p1" "${BEFORE}" "${PRIME}" p1
run_pair "2053-p2" "${PRIME}" "${AFTER}" p2

{
  echo "date_utc=$(date -u +%FT%TZ)"
  echo "rev_stamp_before=${SHA}"
  echo "rev_stamp_prime=$(cat "${PRIME}/.rev-stamp")"
  echo "rev_stamp_after=$(cat "${AFTER}/.rev-stamp")"
  echo "uname=$(uname -srm)"
  echo "rustc=$(rustc -V)"; echo "cargo=$(cargo -V)"
  echo "cpu_model=$(/usr/bin/lscpu | grep -i 'model name' | head -1 | sed 's/ \+/ /g')"
  echo "nproc=$(nproc)"
  echo "rayon_num_threads=${RAYON_NUM_THREADS:-unset}"
  echo "tree_before=<home>/work/rust-ai-library-run"
  echo "tree_prime=<home>/work/rust-ai-library-2053-prime"
  echo "tree_after=<home>/work/rust-ai-library-2053-after"
  echo "outer_gate=load1<1.0 && gpu_util==0 (load_gate_outer.log。記録のみ)"
} > "${LOGD}/env_info.txt"
uptime > "${LOGD}/uptime_after.txt"
echo "done. $(date -u +%FT%TZ)"
