#!/usr/bin/env bash
# GB10 #2050: SME_PRODUCTION_ENABLED=false（before=main）／true（after=on-arm.patch）両腕の
# 全出力 bit 同一（RB）・run-to-run 決定性（RR）を実測する。判定規則は同ディレクトリの
# RULE-bitdump.txt（事前登録）。#1978 残の orchestrate_gb10.sh と同型で、R1/R2 は再実行しない。
set -uo pipefail
export PATH="${HOME}/.cargo/bin:${PATH}"
W="${HOME}/work"
BEFORE="${W}/rust-ai-library-run"
AFTER="${W}/rust-ai-library-2050-after"
LOGD="${W}/gb10-2050"; mkdir -p "${LOGD}"
SHA="$(cat "${BEFORE}/.rev-stamp")"
[[ -n "${SHA}" ]] || { echo "ERROR: .rev-stamp が空"; exit 1; }
echo "start $(date -u +%FT%TZ) sha=${SHA}"
uptime > "${LOGD}/uptime_before.txt"

gate() { # 外側専有ゲート（記録のみ。RULE-bitdump.txt: 不通過でも実行し「参考」表記）
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

# after ツリー: before の複製 + on-arm.patch
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

# ツリー指紋（差分は mod.rs の 1 件が期待。filelist 不在・差分件数不一致は 30 分級ビルドの前に中止）
[[ -f "${W}/filelist-2050-gb10.txt" ]] || { echo "ERROR: ${W}/filelist-2050-gb10.txt が無い"; exit 1; }
for t in before after; do
  T="${BEFORE}"; [[ "${t}" == after ]] && T="${AFTER}"
  ( cd "${T}" && tr '\n' '\0' < "${W}/filelist-2050-gb10.txt" | xargs -0 sha256sum ) > "${LOGD}/fp-${t}.txt" 2>/dev/null
done
diff "${LOGD}/fp-before.txt" "${LOGD}/fp-after.txt" > "${LOGD}/fp-diff.txt"; echo "fp diff lines=$(wc -l < "${LOGD}/fp-diff.txt")"
NCH=$(grep -cE "^[<>]" "${LOGD}/fp-diff.txt"); NMOD=$(grep -cE "^[<>].*crates/backend-cpu/src/gemm_blis/mod\.rs$" "${LOGD}/fp-diff.txt")
[[ "${NCH}" -eq 2 && "${NMOD}" -eq 2 ]] || { echo "ERROR: 指紋差分が mod.rs の 1 件でない（changed_lines=${NCH} mod_lines=${NMOD}）-> 中止"; cat "${LOGD}/fp-diff.txt" | head; exit 1; }
[[ $(wc -l < "${LOGD}/fp-before.txt") -gt 100 ]] || { echo "ERROR: 指紋の行数が少なすぎる（filelist 空？）"; exit 1; }

gate "start" || true

# R0: sme_report プローブ（両腕。#1978 のプローブ crate を再利用）
echo "-- R0 sme_report ($(date -u +%FT%TZ))"
: > "${LOGD}/sme_report.txt"
for t in before after; do
  T="${BEFORE}"; [[ "${t}" == after ]] && T="${AFTER}"
  P="${W}/sme-probe-2050-${t}"; rm -rf "${P}"; cp -r "${W}/sme-probe-1978" "${P}"
  sed -i "s#TREE_PLACEHOLDER#${T}#" "${P}/Cargo.toml"
  ( cd "${P}" && cargo run --release -q ) > "${LOGD}/sme_probe_${t}.log" 2>&1 || { echo "probe ${t} rc=$? -> 中止"; tail -20 "${LOGD}/sme_probe_${t}.log"; exit 1; }
  echo "${t}: $(grep sme_report= "${LOGD}/sme_probe_${t}.log")" | tee -a "${LOGD}/sme_report.txt"
done
for t in before after; do
  grep -q 'os_flag: false, svl_bytes: None, kernel_enabled: false' "${LOGD}/sme_probe_${t}.log" || { echo "R0 不成立（${t}）-> undetermined・中止"; exit 1; }
done

# RB: run_bitdump.sh（before ツリー内のコピーを実行。出力は before ツリーの gb10/bitdump/）
echo "-- RB run_bitdump.sh ($(date -u +%FT%TZ))"
RB="${BEFORE}/docs/perf/logs/cpu-gemm-sme-fmopa-1587/gb10/run_bitdump.sh"
BEFORE_TREE="${BEFORE}" AFTER_TREE="${AFTER}" bash "${RB}" > "${LOGD}/run_bitdump.log" 2>&1; RB_RC=$?
echo "run_bitdump rc=${RB_RC}" | tee -a "${LOGD}/run_bitdump.log"
BD="${BEFORE}/docs/perf/logs/cpu-gemm-sme-fmopa-1587/gb10/bitdump"
cat "${BD}/line_counts.txt" "${BD}/summary.txt" 2>/dev/null

# RR: after 腕 2 回目（run-to-run 決定性。判定には用いない）
echo "-- RR after rerun ($(date -u +%FT%TZ))"
( cd "${AFTER}" && cargo test -p fandhe-ai --release --test cpu_sme_gate_bit_dump -- --ignored --nocapture --exact dump_cpu_sme_gate_bits ) > "${LOGD}/after_rerun_raw.log" 2>&1; echo "RR rc=$?"
grep -E '^out\[' "${LOGD}/after_rerun_raw.log" > "${LOGD}/after_rerun_bits.txt"
{
  echo "after_rerun_lines=$(wc -l < "${LOGD}/after_rerun_bits.txt" | tr -d ' ')"
  echo "after_rerun_sha256=$(sha256sum "${LOGD}/after_rerun_bits.txt" | awk '{print $1}')"
  echo "after_first_sha256=$(sha256sum "${BD}/after_bits.txt" | awk '{print $1}')"
} > "${LOGD}/rerun_after_sha256.txt"; cat "${LOGD}/rerun_after_sha256.txt"

# 回収用の小さい成果物（dump 本体は除外）
mkdir -p "${LOGD}/bitdump"
for f in line_counts.txt summary.txt bitdump_cmp.txt; do cp "${BD}/${f}" "${LOGD}/bitdump/" 2>/dev/null; done
head -50 "${BD}/bitdump_diff.txt" > "${LOGD}/bitdump/bitdump_diff.txt" 2>/dev/null
for t in before after; do grep -vE '^out\[' "${BD}/${t}_raw.log" > "${LOGD}/bitdump/${t}_raw.summary.log"; done
grep -vE '^out\[' "${LOGD}/after_rerun_raw.log" > "${LOGD}/bitdump/after_rerun_raw.summary.log"
rm -f "${LOGD}/after_rerun_raw.log" "${LOGD}/after_rerun_bits.txt"

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
  echo "tree_after=<home>/work/rust-ai-library-2050-after"
  echo "outer_gate=load1<1.0 && gpu_util==0 (load_gate_outer.log。記録のみ)"
} > "${LOGD}/env_info.txt"
uptime > "${LOGD}/uptime_after.txt"
echo "done. $(date -u +%FT%TZ) run_bitdump_rc=${RB_RC}"
