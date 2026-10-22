#!/usr/bin/env bash
# イシュー #3184 再開分 CPU 系セッション（G1 -> G2 Stage 1）。nohup で切り離し起動する（RULE-reopen.txt 3・4・8）。
# 方針: 継承スクリプトの round 前に hook（AB_ROUND_COOLDOWN_MAX_LOAD1=3.0）が load1 < 3.0 を待つので、
#   ビルド負荷も round 1 前の待機で吸収される。初回のようなウォーム走行・target 改名は使わない。
#   各 label は 1 回だけ実行する（差し替え・追加禁止）。G2 Stage 0 は再計測しない（T_macmini = 16384 を継承）。
set -u
export PATH=${HOME}/.cargo/bin:${PATH}
unset RAYON_NUM_THREADS CARGO_TARGET_DIR
. "$(dirname "$0")/lib_r2.sh"
B=${R}/before; A=${R}/after
PATCHES=${B}/docs/perf/logs/optin-ab-macmini-3184
FC=${B}/scripts/bench/framework-compare
T=16384
collect_cpu() { # collect_cpu <label-fragment> <dest>
  mkdir -p "$2"
  mv "${FC}"/results/raw/*"$1"* "$2/" 2>/dev/null
  mv "${FC}"/compare-*"$1"* "$2/" 2>/dev/null
}

# ---- G1
git -C "${A}" checkout -- .
git -C "${A}" apply --check "${PATCHES}/on-host-arena.patch" >"${M}/patch_apply-g1.log" 2>&1 \
  && git -C "${A}" apply "${PATCHES}/on-host-arena.patch" >>"${M}/patch_apply-g1.log" 2>&1
step "G1 patch rc=$?"
outer_gate g1r2
step "G1 start"
mkdir -p "${COL}/host-arena-r2"
( cd "${FC}" && AB_BEFORE_FACADE_PATH=${B}/crates/facade AB_AFTER_FACADE_PATH=${A}/crates/facade \
  bash run_ab_tape_arena_cpu.sh 3184-macmini-r2-g1 >"${COL}/host-arena-r2/stdout-g1.log" 2>&1 )
step "G1 rc=$?"
collect_cpu 3184-macmini-r2-g1 "${COL}/host-arena-r2"
rm -rf "${FC}"/target-ab-*
git -C "${A}" checkout -- .
step "status before=$(git -C "${B}" status --porcelain | wc -l) after=$(git -C "${A}" status --porcelain | wc -l)"

# ---- G2 Stage 1（T_macmini = ${T}）
mkdir -p "${COL}/reduction/stage1-r2"
sed "s/@T@/${T}/" "${PATCHES}/on-reduction.patch" >"${R}/on-reduction.filled-r2.patch"
shasum -a 256 "${R}/on-reduction.filled-r2.patch" >"${COL}/reduction/stage1-r2/on-reduction-filled.sha256"
git -C "${A}" apply --check "${R}/on-reduction.filled-r2.patch" >"${COL}/reduction/stage1-r2/patch_apply.log" 2>&1 \
  && git -C "${A}" apply "${R}/on-reduction.filled-r2.patch" >>"${COL}/reduction/stage1-r2/patch_apply.log" 2>&1
step "G2s1 patch rc=$?"
outer_gate g2r2
step "G2s1 start"
( cd "${FC}" && AB_THRESHOLD_T=${T} AB_BEFORE_FACADE_PATH=${B}/crates/facade AB_AFTER_FACADE_PATH=${A}/crates/facade \
  bash run_ab_reduction_threshold_cpu.sh 3184-macmini-r2-g2 >"${COL}/reduction/stage1-r2/stdout-g2.log" 2>&1 )
step "G2s1 rc=$?"
collect_cpu 3184-macmini-r2-g2 "${COL}/reduction/stage1-r2"
rm -rf "${FC}"/target-ab-*
git -C "${A}" checkout -- .
step "status before=$(git -C "${B}" status --porcelain | wc -l) after=$(git -C "${A}" status --porcelain | wc -l)"
step "CPU-R2 DONE"
echo done >"${R}/cpu-r2.done"
