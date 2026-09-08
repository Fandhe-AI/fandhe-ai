#!/usr/bin/env bash
# イシュー #1313: 2D 動的分配（TwoDDynamic）本番結線可否判断のための
# Apple M4 Max 専有ゲート付き再計測オーケストレーションスクリプト。
#
# #1312（docs/perf/cpu-gemm-2d-dynamic-partition-ab.md）が「M4 Max の
# 専有ゲートが通過できるタイミングで再実行」を推奨したことを受け、
# 同一の Tier 1 判定基準（1 分 load average <6.0 を 2 回連続・最大 30 分）
# で 1 回だけ試みる（本イシューの Phase 0）。#1312 と異なり、ゲート不通過
# 時は計測を続行せず GATE_NOT_PASSED.marker を書いて終了する（共有負荷下の
# 数値を Branch 判定の根拠として積み上げないため。計画 §Phase 0）。
#
# WORKDIR／CARGO_TARGET_DIR／LOGDIR／SCRIPTDIR は環境変数で上書き可能。
set -uo pipefail

WORKDIR="${WORKDIR:-/Users/nancy/fandhe/library/rust-ai-library/.claude/worktrees/wf_209997e6-39f-98}"
LOGDIR="${LOGDIR:-$WORKDIR/docs/perf/logs/cpu-gemm-2d-dynamic-wiring-1313}"
SCRIPTDIR="${SCRIPTDIR:-$LOGDIR}"
GATE_THRESHOLD="6.0"
MAX_ATTEMPTS=30

if [ ! -f "$SCRIPTDIR/run_ab.sh" ]; then
  echo "エラー: SCRIPTDIR='$SCRIPTDIR' に run_ab.sh が見つからない" >&2
  exit 1
fi

mkdir -p "$LOGDIR"
cd "$WORKDIR"

( while :; do date -u +"%Y-%m-%dT%H:%M:%SZ" >> "$LOGDIR/uptime-m4max.log"; uptime >> "$LOGDIR/uptime-m4max.log"; sleep 30; done ) &
POLLER_PID=$!

GATE_PASSED=0
{
  echo "=== gate start $(date -u +%FT%TZ) (threshold=$GATE_THRESHOLD, max_attempts=$MAX_ATTEMPTS) ==="
  attempts=0
  consec=0
  while [ "$attempts" -lt "$MAX_ATTEMPTS" ]; do
    load1=$(uptime | sed -E 's/.*load averages?: ([0-9.]+).*/\1/')
    echo "attempt=$attempts load1=$load1"
    ok=$(awk -v l="$load1" -v t="$GATE_THRESHOLD" 'BEGIN{print (l<t)?1:0}')
    if [ "$ok" = "1" ]; then
      consec=$((consec+1))
    else
      consec=0
    fi
    if [ "$consec" -ge 2 ]; then
      GATE_PASSED=1
      echo "gate PASSED at attempt=$attempts"
      break
    fi
    attempts=$((attempts+1))
    sleep 60
  done
  if [ "$GATE_PASSED" -ne 1 ]; then
    echo "gate NOT PASSED after max wait (threshold=$GATE_THRESHOLD)"
  fi
  echo "who: $(who)"
  ps -Ao pcpu,pid,comm -r | head -15
  echo "=== gate end $(date -u +%FT%TZ) ==="
} > "$LOGDIR/gate-m4max.log" 2>&1

kill "$POLLER_PID" 2>/dev/null || true

if [ "$GATE_PASSED" -ne 1 ]; then
  echo "gate not passed" > "$LOGDIR/GATE_NOT_PASSED.marker"
  echo "gate not passed $(date -u +%FT%TZ)" >> "$LOGDIR/uptime-m4max.log"
  exit 0
fi

# 前提 1: lib テスト全体が green であること
cargo test -p fandhe-ai-backend-cpu --lib --release > "$LOGDIR/unit-test-m4max.txt" 2>&1
UNIT_RC=$?
echo "unit test rc=$UNIT_RC" >> "$LOGDIR/unit-test-m4max.txt"
if [ "$UNIT_RC" -ne 0 ]; then
  echo "prereq unit test failed" > "$LOGDIR/PREREQ_FAILED.marker"
  exit 1
fi

# 前提 2: 大形状 bit 完全一致（#[ignore]）
cargo test -p fandhe-ai-backend-cpu --release --lib \
  -- --ignored gemm_blis_two_d_dynamic_matches_row_panel_bit_exact_large --nocapture \
  > "$LOGDIR/bit-exact-large-m4max.txt" 2>&1
BITEXACT_RC=$?
echo "bit-exact-large rc=$BITEXACT_RC" >> "$LOGDIR/bit-exact-large-m4max.txt"
if [ "$BITEXACT_RC" -ne 0 ]; then
  echo "prereq bit-exact-large failed" > "$LOGDIR/PREREQ_FAILED.marker"
  exit 1
fi

THREADS=default RUNS=5 MACHINE=m4max LOGDIR="$LOGDIR" \
  bash "$SCRIPTDIR/run_ab.sh"
AB_RC=$?

echo "done." > "$LOGDIR/ALL_DONE.marker"
echo "all done rc=$AB_RC $(date -u +%FT%TZ)" >> "$LOGDIR/uptime-m4max.log"
