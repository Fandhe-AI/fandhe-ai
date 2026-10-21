#!/usr/bin/env bash
# イシュー #3184 再開分 GB10 セッション（G3 の GB10 対照セル: cpu・cuda × train {fresh, reuse}）。
# ノード上で setsid nohup 起動する（RULE-reopen.txt 5・8）。Linux では Metal コードがコンパイルされないため
# after 腕（TRAIN_FORWARD_ENCODE_ONLY_DEFAULT_ENABLED = true の patch）は実質 A/A の対照になる。
# 継承ゲート 8.0 のまま（Mac mini 用の 3.0 クールダウンは GB10 に適用しない。補遺 RULE 2 は Mac mini 固有条件）。
# CARGO_TARGET_DIR は export しない（clone 内の target を使う）。各ステップの rc を steps.log に残す。
set -u
export PATH=/usr/local/cuda/bin:${HOME}/.cargo/bin:${PATH}
unset RAYON_NUM_THREADS CARGO_TARGET_DIR AB_ROUND_COOLDOWN_MAX_LOAD1
R=${HOME}/work/r3184-gb10
B=${R}/before; A=${R}/after
OUT=${R}/out
mkdir -p "${OUT}"
FC=${B}/scripts/bench/framework-compare
PATCH=${B}/docs/perf/logs/optin-ab-macmini-3184/on-train-forward-encode.patch
step() { echo "$(date -u +%FT%TZ) $*" >>"${OUT}/steps.log"; }

git -C "${A}" checkout -- .
git -C "${A}" apply --check "${PATCH}" >"${OUT}/patch_apply-g3.log" 2>&1 && git -C "${A}" apply "${PATCH}" >>"${OUT}/patch_apply-g3.log" 2>&1
step "patch rc=$?"
shasum -a 256 "${PATCH}" >>"${OUT}/patch_apply-g3.log" 2>/dev/null || sha256sum "${PATCH}" >>"${OUT}/patch_apply-g3.log"
for dev in cpu cuda; do
  {
    echo "--- date: $(date -u +%FT%TZ)"; echo "--- uptime: $(uptime)"
    echo "--- procs (cargo|bench|rustc|nsys)"; ps -eo pid,etime,args | grep -E 'cargo|bench|rustc|nsys' | grep -v grep || true
  } >"${OUT}/preflight-procs-${dev}.txt" 2>&1
  step "G3 ${dev} start"
  ( cd "${FC}" && AB_DEVICE=${dev} AB_BEFORE_FACADE_PATH=${B}/crates/facade AB_AFTER_FACADE_PATH=${A}/crates/facade \
    bash run_ab_train_forward_encode_metal.sh "3184-gb10-g3-${dev}" >"${OUT}/stdout-g3-${dev}.log" 2>&1 )
  step "G3 ${dev} rc=$?"
done
mkdir -p "${OUT}/encode-only-gb10"
mv "${FC}"/results/raw/*3184-gb10-g3* "${OUT}/encode-only-gb10/" 2>/dev/null
mv "${FC}"/compare-*3184-gb10-g3* "${OUT}/encode-only-gb10/" 2>/dev/null
git -C "${A}" checkout -- .
step "status before=$(git -C "${B}" status --porcelain | wc -l) after=$(git -C "${A}" status --porcelain | wc -l)"
step "GB10 G3 DONE"
touch "${R}/gb10.done"
