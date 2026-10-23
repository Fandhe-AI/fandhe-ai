#!/usr/bin/env bash
# イシュー #3195: Stage 1（framework-compare A/B）。前提ゲート成立後のみ実行する。
# 使い方: bash stage1.sh <作業ディレクトリ> <macmini|gb10> <label>
# CARGO_TARGET_DIR は export しない。ハーネス成果物は runner の scripts/bench/framework-compare/ 配下に出る。
set -u
unset CARGO_TARGET_DIR RAYON_NUM_THREADS
export PATH=$HOME/.cargo/bin:$PATH
R=$1
KIND=$2
LABEL=$3
case "$KIND" in
  macmini) GATE=3.0 ;;
  gb10) GATE=1.0 ;;
  *) echo "bad kind" >&2; exit 1 ;;
esac
cd "$R/runner/scripts/bench/framework-compare" || exit 1
AB_BEFORE_FACADE_PATH=$R/before/crates/facade \
AB_AFTER_FACADE_PATH=$R/after/crates/facade \
AB_THRESHOLD_T=4096 AB_HOST_KIND=$KIND AB_ROUND_COOLDOWN_MAX_LOAD1=$GATE \
  bash run_ab_mse_backward_threshold_cpu.sh "$LABEL" >"$R/out/stage1-$LABEL.stdout.log" 2>&1
echo "rc=$?" >>"$R/out/stage1-$LABEL.stdout.log"
touch "$R/stage1-$LABEL.done"
