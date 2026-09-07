#!/usr/bin/env bash
# イシュー #1305: Apple M4 Max ローカル実行のスイープ一括実行オーケストレーションスクリプト。
set -uo pipefail

WORKDIR="/Users/nancy/fandhe/library/rust-ai-library/.claude/worktrees/wf_f889d48c-a2d-87"
BIN="$WORKDIR/scripts/bench/oss-gemm-compare/target/release/oss-gemm-compare"
LOGDIR="$WORKDIR/docs/perf/logs/cpu-gemm-rayon-sweep-1305/out-m4max"
SCRIPTDIR="$WORKDIR/docs/perf/logs/cpu-gemm-rayon-sweep-1305"
mkdir -p "$LOGDIR"

( while :; do date -u +"%Y-%m-%dT%H:%M:%SZ" >> "$LOGDIR/uptime-m4max.log"; uptime >> "$LOGDIR/uptime-m4max.log"; sleep 30; done ) &
POLLER_PID=$!

{
  echo "=== gate start $(date -u +%FT%TZ) ==="
  attempts=0
  consec=0
  while [ "$attempts" -lt 60 ]; do
    load1=$(uptime | sed -E 's/.*load averages?: ([0-9.]+).*/\1/')
    echo "attempt=$attempts load1=$load1"
    ok=$(awk -v l="$load1" 'BEGIN{print (l<4.0)?1:0}')
    if [ "$ok" = "1" ]; then
      consec=$((consec+1))
    else
      consec=0
    fi
    if [ "$consec" -ge 2 ]; then
      echo "gate PASSED at attempt=$attempts"
      break
    fi
    attempts=$((attempts+1))
    sleep 60
  done
  if [ "$consec" -lt 2 ]; then
    echo "gate NOT PASSED after max wait (共有負荷下として続行)"
  fi
  echo "who: $(who)"
  ps -Ao pcpu,pid,comm -r | head -15
  echo "=== gate end $(date -u +%FT%TZ) ==="
} > "$LOGDIR/gate-m4max.log" 2>&1

BIN="$BIN" SIZES=1024,2048,4096 THREADS="8 12 16 1 2 4 10 14" RUNS=5 \
  OUT="$LOGDIR/rayon_sweep_m4max.log" \
  bash "$SCRIPTDIR/run_sweep.sh"

BIN="$BIN" SIZES=1920 THREADS="8 12 16" RUNS=5 \
  OUT="$LOGDIR/control_1920_m4max.log" \
  bash "$SCRIPTDIR/run_sweep.sh"

kill "$POLLER_PID" 2>/dev/null || true
echo "done." > "$LOGDIR/ALL_DONE.marker"
echo "all sweeps complete $(date -u +%FT%TZ)" >> "$LOGDIR/uptime-m4max.log"
