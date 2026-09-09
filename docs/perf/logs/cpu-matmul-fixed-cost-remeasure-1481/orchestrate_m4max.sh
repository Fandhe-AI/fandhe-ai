#!/bin/bash
# Apple M4 Max 側オーケストレーション（イシュー #1481・このセッションの
# ホスト自身でローカル実行）。
# (i) 専有ゲート（1 分 load average < 6.0 を 30 秒間隔で 2 回連続・
#     最大 30 試行。不成立なら GATE_NOT_PASSED.marker を書いて終了し、
#     ユーザー承認事項どおり verdict=undetermined を 1 回だけ記録する
#     判断は呼び出し側〈本エージェント〉に委ねる。待ち続けない）
# (ii) Layer A（off=実装 worktree → on=スクラッチ複製ツリーの順）
# (iii) Layer B（run_layerB_m4max.sh off → on）
# (iv) ALL_DONE.marker
set -uo pipefail
LOG_DIR="$(cd "$(dirname "$0")" && pwd)"
OFF_TREE="/Users/nancy/fandhe/library/rust-ai-library/.claude/worktrees/wf_dbb339eb-c60-13"
ON_TREE="/private/tmp/claude-501/-Users-nancy-fandhe-library-rust-ai-library/bac57b76-f1a4-4186-aea8-8f5e06b5dc10/scratchpad/fc-1481-on"

# --- (i) 専有ゲート ---
GATE_OK=0
CONSEC=0
i=1
while [ "$i" -le 30 ]; do
  LOAD1=$(uptime | sed -E 's/.*load averages?: ([0-9.]+).*/\1/')
  OK=$(awk -v l="$LOAD1" 'BEGIN{print (l < 6.0) ? 1 : 0}')
  echo "gate try=$i load1=$LOAD1 ok=$OK $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG_DIR/gate-m4max.log"
  if [ "$OK" = "1" ]; then
    CONSEC=$((CONSEC + 1))
  else
    CONSEC=0
  fi
  if [ "$CONSEC" -ge 2 ]; then
    GATE_OK=1
    break
  fi
  i=$((i + 1))
  sleep 30
done

if [ "$GATE_OK" != "1" ]; then
  echo "gate not passed after 30 tries" > "$LOG_DIR/GATE_NOT_PASSED.marker"
  exit 0
fi

# --- (ii) Layer A ---
cd "$OFF_TREE/scripts/bench/framework-compare"
SHA=$(cat "$OFF_TREE/.rev-stamp" 2>/dev/null || git -C "$OFF_TREE" rev-parse HEAD)

echo "layerA off start $(date -u +%Y-%m-%dT%H:%M:%SZ) $(uptime)" >> "$LOG_DIR/gate-m4max.log"
GEMM_GATE_CPU_NODE_TAG=m4max-cpu GEMM_GATE_PATCH_FACADE_PATH="$OFF_TREE/crates/facade" \
  bash run_gemm_gate_cpu.sh "head-${SHA}-1481-pzero-off" \
  > "$LOG_DIR/run_gemm_gate_cpu-m4max-off.log" 2>&1
echo "layerA off end $(date -u +%Y-%m-%dT%H:%M:%SZ) $(uptime)" >> "$LOG_DIR/gate-m4max.log"

echo "layerA on start $(date -u +%Y-%m-%dT%H:%M:%SZ) $(uptime)" >> "$LOG_DIR/gate-m4max.log"
GEMM_GATE_CPU_NODE_TAG=m4max-cpu GEMM_GATE_PATCH_FACADE_PATH="$ON_TREE/crates/facade" \
  bash run_gemm_gate_cpu.sh "head-${SHA}-1481-pzero-on" \
  > "$LOG_DIR/run_gemm_gate_cpu-m4max-on.log" 2>&1
echo "layerA on end $(date -u +%Y-%m-%dT%H:%M:%SZ) $(uptime)" >> "$LOG_DIR/gate-m4max.log"

# --- (iii) Layer B ---
bash "$LOG_DIR/run_layerB_m4max.sh" "$OFF_TREE" off > "$LOG_DIR/layerB-m4max-off-driver.log" 2>&1
bash "$LOG_DIR/run_layerB_m4max.sh" "$ON_TREE" on > "$LOG_DIR/layerB-m4max-on-driver.log" 2>&1

echo "all done $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$LOG_DIR/ALL_DONE.marker"
