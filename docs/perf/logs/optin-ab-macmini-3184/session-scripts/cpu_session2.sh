#!/usr/bin/env bash
# イシュー #3184 CPU 系セッション 2（G1 再走 + G2 Stage 1）。nohup で切り離し起動する。
# 背景: 初回 G1 系列（label g1）は継承スクリプト内のビルド直後に round 負荷ゲートを読むため、
# 自身のビルド由来の load1（5.65）が補遺 RULE 3 の machine_gate_ok（全 round < 3.0）を満たさなかった。
# そこで「ウォーム走行（label *w。record_only 扱い・旧系列として残す）でビルドを済ませ、target を
# 別 label 名へ改名して no-op ビルドにし、負荷が落ち着いてから正式 label（*f）で走らせる」。
# 継承スクリプト・閾値は不変。ウォーム系列も正式系列も全て収録する（run の差し替えではなく別 label の再実行）。
set -u
export PATH=${HOME}/.cargo/bin:${PATH}
unset RAYON_NUM_THREADS CARGO_TARGET_DIR
. "$(dirname "$0")/lib.sh"
B=${R}/before; A=${R}/after
PATCHES=${B}/docs/perf/logs/optin-ab-macmini-3184
FC=${B}/scripts/bench/framework-compare
T=${1:-16384}

rename_targets() { # rename_targets <from-label> <to-label>
  for d in "${FC}"/target-ab-*"$1"*; do
    [ -d "${d}" ] || continue
    mv "${d}" "$(echo "${d}" | sed "s/$1/$2/")"
  done
}
collect_cpu() { # collect_cpu <label-fragment> <dest>
  mkdir -p "$2"
  mv "${FC}"/results/raw/*"$1"* "$2/" 2>/dev/null
  mv "${FC}"/compare-*"$1"* "$2/" 2>/dev/null
}

START=${2:-g1w}
git -C "${A}" checkout -- .
git -C "${A}" apply --check "${PATCHES}/on-host-arena.patch" >"${COL}/macmini/patch_apply-g1b.log" 2>&1 \
  && git -C "${A}" apply "${PATCHES}/on-host-arena.patch" >>"${COL}/macmini/patch_apply-g1b.log" 2>&1
step "G1b patch rc=$?"
if [ "${START}" = g1w ]; then
# ---- G1 ウォーム -> 正式
outer_gate g1w
( cd "${FC}" && AB_BEFORE_FACADE_PATH=${B}/crates/facade AB_AFTER_FACADE_PATH=${A}/crates/facade \
  bash run_ab_tape_arena_cpu.sh 3184-macmini-g1w >"${COL}/host-arena/stdout-g1w.log" 2>&1 )
step "G1w rc=$?"
collect_cpu 3184-macmini-g1w "${COL}/host-arena"
rename_targets 3184-macmini-g1w 3184-macmini-g1f
fi
sleep 120
outer_gate g1f
( cd "${FC}" && AB_BEFORE_FACADE_PATH=${B}/crates/facade AB_AFTER_FACADE_PATH=${A}/crates/facade \
  bash run_ab_tape_arena_cpu.sh 3184-macmini-g1f >"${COL}/host-arena/stdout-g1f.log" 2>&1 )
step "G1f rc=$?"
collect_cpu 3184-macmini-g1f "${COL}/host-arena"
rm -rf "${FC}"/target-ab-*
git -C "${A}" checkout -- .
step "status before=$(git -C "${B}" status --porcelain | wc -l) after=$(git -C "${A}" status --porcelain | wc -l)"

# ---- G2 Stage 1（T_macmini = ${T}）
sed "s/@T@/${T}/" "${PATCHES}/on-reduction.patch" >"${R}/on-reduction.filled.patch"
shasum -a 256 "${R}/on-reduction.filled.patch" >"${COL}/reduction/on-reduction-filled.sha256"
git -C "${A}" apply --check "${R}/on-reduction.filled.patch" >"${COL}/reduction/patch_apply.log" 2>&1 \
  && git -C "${A}" apply "${R}/on-reduction.filled.patch" >>"${COL}/reduction/patch_apply.log" 2>&1
step "G2s1 patch rc=$?"
mkdir -p "${COL}/reduction/stage1"
outer_gate g2w
( cd "${FC}" && AB_THRESHOLD_T=${T} AB_BEFORE_FACADE_PATH=${B}/crates/facade AB_AFTER_FACADE_PATH=${A}/crates/facade \
  bash run_ab_reduction_threshold_cpu.sh 3184-macmini-g2w >"${COL}/reduction/stage1/stdout-g2w.log" 2>&1 )
step "G2w rc=$?"
collect_cpu 3184-macmini-g2w "${COL}/reduction/stage1"
rename_targets 3184-macmini-g2w 3184-macmini-g2f
sleep 120
outer_gate g2f
( cd "${FC}" && AB_THRESHOLD_T=${T} AB_BEFORE_FACADE_PATH=${B}/crates/facade AB_AFTER_FACADE_PATH=${A}/crates/facade \
  bash run_ab_reduction_threshold_cpu.sh 3184-macmini-g2f >"${COL}/reduction/stage1/stdout-g2f.log" 2>&1 )
step "G2f rc=$?"
collect_cpu 3184-macmini-g2f "${COL}/reduction/stage1"
rm -rf "${FC}"/target-ab-*
git -C "${A}" checkout -- .
step "status before=$(git -C "${B}" status --porcelain | wc -l) after=$(git -C "${A}" status --porcelain | wc -l)"
step "CPU2 DONE"
echo done >"${R}/cpu2.done"
