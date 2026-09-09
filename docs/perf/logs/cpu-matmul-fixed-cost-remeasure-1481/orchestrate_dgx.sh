#!/bin/sh
# DGX Spark GB10 側オーケストレーション（イシュー #1481）。
# (i) 専有ゲート（1 分 load average < 6.0 を 30 秒間隔で 2 回連続・
#     最大 30 試行。不成立なら GATE_NOT_PASSED.marker を書いて終了）
# (ii) uptime 30 秒ポーラをバックグラウンド起動
# (iii) Layer A（off → on の順。off ツリーの scripts/bench/framework-compare
#      から実行し、on 腕は GEMM_GATE_PATCH_FACADE_PATH で on ツリーの
#      crates/facade を指す）
# (iv) Layer B（run_layerB_dgx.sh off → on）
# (v) ALL_DONE.marker
set -u
LOG="$HOME/work/fc-1481-logs"
mkdir -p "$LOG"
OFF="$HOME/work/fc-1481/off"
ON="$HOME/work/fc-1481/on"
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"

# --- (i) 専有ゲート ---
GATE_OK=0
CONSEC=0
i=1
while [ "$i" -le 30 ]; do
  LOAD1=$(uptime | sed -E 's/.*load average: ([0-9.]+),.*/\1/')
  OK=$(awk -v l="$LOAD1" 'BEGIN{print (l < 6.0) ? 1 : 0}')
  echo "gate try=$i load1=$LOAD1 ok=$OK $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG/gate-dgx.log"
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
  echo "gate not passed after 30 tries" > "$LOG/GATE_NOT_PASSED.marker"
  exit 0
fi

# --- (ii) uptime 30 秒ポーラ（バックグラウンド） ---
(
  while true; do
    echo "poll $(date -u +%Y-%m-%dT%H:%M:%SZ) $(uptime)" >> "$LOG/uptime-dgx.log"
    sleep 30
  done
) &
POLLER_PID=$!

# --- (iii) Layer A ---
cd "$OFF/scripts/bench/framework-compare"
SHA=$(cat "$OFF/.rev-stamp" 2>/dev/null || echo unknown)

echo "layerA off start $(date -u +%Y-%m-%dT%H:%M:%SZ) $(uptime)" >> "$LOG/gate-dgx.log"
GEMM_GATE_CPU_NODE_TAG=dgx-cpu GEMM_GATE_PATCH_FACADE_PATH="$OFF/crates/facade" \
  bash run_gemm_gate_cpu.sh "head-${SHA}-1481-pzero-off" \
  > "$LOG/run_gemm_gate_cpu-dgx-off.log" 2>&1
echo "layerA off end $(date -u +%Y-%m-%dT%H:%M:%SZ) $(uptime)" >> "$LOG/gate-dgx.log"

echo "layerA on start $(date -u +%Y-%m-%dT%H:%M:%SZ) $(uptime)" >> "$LOG/gate-dgx.log"
GEMM_GATE_CPU_NODE_TAG=dgx-cpu GEMM_GATE_PATCH_FACADE_PATH="$ON/crates/facade" \
  bash run_gemm_gate_cpu.sh "head-${SHA}-1481-pzero-on" \
  > "$LOG/run_gemm_gate_cpu-dgx-on.log" 2>&1
echo "layerA on end $(date -u +%Y-%m-%dT%H:%M:%SZ) $(uptime)" >> "$LOG/gate-dgx.log"

# --- (iv) Layer B ---
sh "$OFF/docs/perf/logs/cpu-matmul-fixed-cost-remeasure-1481/run_layerB_dgx.sh" off \
  > "$LOG/layerB-dgx-off-driver.log" 2>&1
sh "$OFF/docs/perf/logs/cpu-matmul-fixed-cost-remeasure-1481/run_layerB_dgx.sh" on \
  > "$LOG/layerB-dgx-on-driver.log" 2>&1

kill "$POLLER_PID" 2>/dev/null || true

echo "all done $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$LOG/ALL_DONE.marker"
