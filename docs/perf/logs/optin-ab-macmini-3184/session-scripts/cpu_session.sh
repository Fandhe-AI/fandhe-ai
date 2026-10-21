#!/usr/bin/env bash
# イシュー #3184 CPU 系セッション（G1 -> G2 Stage 0）。nohup で切り離し起動する。
# 配置: ~/work/optin-3184/session/（before/after clone の外。clone を汚さない）。
# 各ステップの rc を steps.log に残し、失敗しても次へ進む（判定は RULE のとおり機械的）。
set -u
export PATH=${HOME}/.cargo/bin:${PATH}
unset RAYON_NUM_THREADS CARGO_TARGET_DIR
. "$(dirname "$0")/lib.sh"
B=${R}/before; A=${R}/after
PATCHES=${B}/docs/perf/logs/optin-ab-macmini-3184
FC=${B}/scripts/bench/framework-compare

# ---- G1
outer_gate g1
step "G1 start"
git -C "${A}" apply --check "${PATCHES}/on-host-arena.patch" >"${COL}/macmini/patch_apply-g1.log" 2>&1 \
  && git -C "${A}" apply "${PATCHES}/on-host-arena.patch" >>"${COL}/macmini/patch_apply-g1.log" 2>&1
step "G1 patch rc=$?"
shasum -a 256 "${PATCHES}/on-host-arena.patch" >>"${COL}/macmini/patch_apply-g1.log"
mkdir -p "${COL}/host-arena"
( cd "${FC}" && AB_BEFORE_FACADE_PATH=${B}/crates/facade AB_AFTER_FACADE_PATH=${A}/crates/facade \
  bash run_ab_tape_arena_cpu.sh 3184-macmini-g1 >"${COL}/host-arena/stdout-g1.log" 2>&1 )
step "G1 AB rc=$?"
git -C "${A}" checkout -- .
mv "${FC}"/results/raw/*3184-macmini-g1* "${COL}/host-arena/" 2>/dev/null
rm -rf "${FC}"/target-ab-3184-* "${FC}"/target-ab-2104-3184-*
df -h "${HOME}" | tail -1 >>"${COL}/macmini/df-after-g1.txt"
step "status before=$(git -C "${B}" status --porcelain | wc -l) after=$(git -C "${A}" status --porcelain | wc -l)"

# ---- G2 Stage 0
outer_gate g2s0
step "G2 stage0 start"
( cd "${B}" && bash docs/perf/logs/elemental-reduction-ab-2102/run_reduction_sweep.sh macmini "${COL}/reduction/stage0" >"${COL}/macmini/stdout-g2s0.log" 2>&1 )
step "G2 stage0 rc=$?"
step "CPU1 DONE"
echo done >"${R}/cpu1.done"
