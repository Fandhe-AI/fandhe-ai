#!/bin/sh
# Layer B（crates/backend-cpu 内側フェーズ分解診断テスト）5 run 実測スクリプト
# （イシュー #1481・DGX Spark GB10。#1301 の run_layerB_dgx.sh と同型。
# 引数 $1 は arm ラベル（off|on）。ノード上の隔離ディレクトリ
# $HOME/work/fc-1481/{off,on} で実行する前提。GNU time が居れば minor page
# fault 数も付随記録する）。
set -eu
ARM="$1"
cd "$HOME/work/fc-1481/$ARM"
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
LOG="$HOME/work/fc-1481-logs"
mkdir -p "$LOG"
for run in 1 2 3 4 5; do
  echo "run=$run $(uptime)" >> "$LOG/uptime-dgx.log"
  if command -v /usr/bin/time >/dev/null 2>&1; then
    /usr/bin/time -v -o "$LOG/time-v-dgx-$ARM-run$run.log" cargo test -p fandhe-ai-backend-cpu --release --lib -- --ignored \
      gemm_reuse_phase_diag_cpu --nocapture --test-threads=1 \
      > "$LOG/layerB-dgx-$ARM-run$run.log" 2>&1
  else
    cargo test -p fandhe-ai-backend-cpu --release --lib -- --ignored \
      gemm_reuse_phase_diag_cpu --nocapture --test-threads=1 \
      > "$LOG/layerB-dgx-$ARM-run$run.log" 2>&1
  fi
done
echo DONE
