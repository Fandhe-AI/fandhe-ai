#!/usr/bin/env bash
# イシュー #1253: 排他環境（load average < 2・他 GPU プロセスなし）で
# phase 1 のみモード（#1251）を 3 回実行するオーケストレーター。
#
# attempt 1（本ディレクトリの orchestrate.log 冒頭・wait_gate.log）は
# 最大 3 時間待っても load average が 2 未満へ収束せず TIMEOUT した。
# 本スクリプトは attempt 2 として同一ゲート閾値のまま再試行するが、
# 有限の待機上限（wait_gate.sh の MAX_WAIT_SECS）で区切る。ゲート通過時
# のみ 3 回の phase1-only 計測を実行し、各回の実行前後 uptime／プロセス
# 確認結果・phase1_round_stats を保存する。
set -uo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DERIVED_WORKDIR="$(cd "$SELF_DIR/../../../.." && pwd)"
WORKDIR="${WORKDIR:-$DERIVED_WORKDIR}"
LOGDIR="${LOGDIR:-$SELF_DIR}"
ATTEMPT="${ATTEMPT:-2}"
GATE_LOG="$LOGDIR/wait_gate_attempt${ATTEMPT}.log"

cd "$WORKDIR" || { echo "エラー: WORKDIR='$WORKDIR' への移動に失敗した" >&2; exit 1; }

echo "$(date +"%Y-%m-%dT%H:%M:%S%z") orchestrator attempt${ATTEMPT} start" >> "$LOGDIR/orchestrate.log"
echo "$(date +"%Y-%m-%dT%H:%M:%S%z") attempt ${ATTEMPT}: waiting for gate (see $(basename "$GATE_LOG"))" >> "$LOGDIR/orchestrate.log"

GATE_RESULT=$(LOGDIR="$LOGDIR" OUT="$GATE_LOG" bash "$LOGDIR/wait_gate.sh")

echo "$(date +"%Y-%m-%dT%H:%M:%S%z") attempt ${ATTEMPT}: gate result=${GATE_RESULT}" >> "$LOGDIR/orchestrate.log"

if [ "$GATE_RESULT" != "PASSED" ]; then
  echo "ORCHESTRATOR_RESULT=TIMEOUT valid_runs=0 attempt=${ATTEMPT}" >> "$LOGDIR/orchestrate.log"
  echo "TIMEOUT" > "$LOGDIR/DONE_TIMEOUT_ATTEMPT${ATTEMPT}"
  exit 0
fi

valid_runs=0
for n in 1 2 3; do
  {
    echo "=== run${n} pre-check $(date +"%Y-%m-%dT%H:%M:%S%z") ==="
    uptime
    echo "--- GPU/build 系プロセス（cargo/rustc/python3） ---"
    ps -Ao pid,pcpu,comm | grep -E '(cargo|rustc|python3)$' | grep -v grep || echo "(none)"
  } > "$LOGDIR/uptime_before_run${n}.txt"

  cargo run -p fandhe-ai-backend-metal --example gemm_transpose_route_ab_bench --release \
    -- --phase1-only > "$LOGDIR/phase1_run${n}.log" 2>&1
  RC=$?

  {
    echo "=== run${n} post-check $(date +"%Y-%m-%dT%H:%M:%S%z") rc=${RC} ==="
    uptime
    echo "--- GPU/build 系プロセス（cargo/rustc/python3） ---"
    ps -Ao pid,pcpu,comm | grep -E '(cargo|rustc|python3)$' | grep -v grep || echo "(none)"
  } > "$LOGDIR/uptime_after_run${n}.txt"

  if [ "$RC" -eq 0 ]; then
    valid_runs=$((valid_runs + 1))
  else
    echo "run${n} failed rc=${RC}" >> "$LOGDIR/orchestrate.log"
  fi
done

echo "$(date +"%Y-%m-%dT%H:%M:%S%z") attempt ${ATTEMPT}: done valid_runs=${valid_runs}/3" >> "$LOGDIR/orchestrate.log"
echo "ORCHESTRATOR_RESULT=DONE valid_runs=${valid_runs}" >> "$LOGDIR/orchestrate.log"
