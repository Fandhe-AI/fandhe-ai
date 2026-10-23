#!/usr/bin/env bash
# イシュー #3195: 3 者 bit 一致テスト（RULE.txt §6 FAIL 条件）→ Stage 0 を 5 独立プロセスで直列実行。
# CARGO_TARGET_DIR・RAYON_NUM_THREADS は unset。ビルドと計測は直列。
# 使い方: bash stage0.sh <作業ディレクトリ>
set -u
unset CARGO_TARGET_DIR RAYON_NUM_THREADS
export PATH=$HOME/.cargo/bin:$PATH
R=$1
OUT=$R/out
cd "$R/runner" || exit 1
step() { echo "$(date -u +%FT%TZ) $*" >>"$OUT/steps-stage0.log"; }
step "bit-exact start"
cargo test -p fandhe-ai-backend-cpu --release --lib mse::tests -- --nocapture >"$OUT/bit-exact.log" 2>&1
rc=$?
step "bit-exact rc=$rc"
echo "rc=$rc" >>"$OUT/bit-exact.log"
if [ "$rc" -ne 0 ]; then
  step "STOP (bit-exact failed -> FAIL)"
  touch "$R/stage0.done"
  exit 0
fi
for i in 1 2 3 4 5; do
  step "stage0 run$i start"
  cargo test -p fandhe-ai-backend-cpu --release --lib mse::tests::mse_backward_threshold_sweep -- --ignored --nocapture >"$OUT/stage0-run$i.log" 2>&1
  step "stage0 run$i rc=$?"
done
step "STAGE0 DONE"
touch "$R/stage0.done"
