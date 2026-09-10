#!/bin/sh
# Apple M4 Max（本セッションのホスト自身）側オーケストレーション（イシュー
# #1490）。正式系列 `fandhe-ai =0.8.0`（registry 解決）のみを計測する単一
# 系列版（Metal GEMM 計測経路の v0.8.0 ↔ origin/main 差分ゼロにつき参考系列は
# 計測しない。詳細は同ディレクトリ attribution.md）。
#
# `docs/perf/logs/cpu-gemm-candle-gate-0.8.0-1488/orchestrate_m4max.sh`（PR
# #1506 是正最終版）の専有ゲート・状態永続化ロジックをそのまま複製し、
# Metal 向けに以下だけ変更する:
#   - `GEMM_GATE_CPU_NODE_TAG` は不要（`run_gemm_gate_metal.sh` が
#     device=metal から NODE_TAG=m4max を無条件確定する。run_gemm_gate.sh
#     の NODE_TAG 決定ロジック参照）
#   - 起動コマンドを `bash run_gemm_gate_metal.sh "$LABEL"` に変更
#   - 既定ラベルを `0.8.0-1490` に変更
#   - 計測前後に `pmset -g therm` を記録する（#1309 の Metal 実機実測記録
#     方式を踏襲。サーマルスロットリング状態を事後確認できるようにする）
#
# 実行対象ツリーは呼び出し元の cwd（scripts/bench/framework-compare）と
# する。ログ出力先は環境変数 LOG で上書きできる。
set -u
LOG="${LOG:-$(pwd)/../../../docs/perf/logs/metal-gemm-candle-gate-0.8.0-1490}"
mkdir -p "$LOG" || { echo "orchestrate_m4max: cannot create LOG dir '$LOG'" >&2; exit 1; }
LOG=$(cd "$LOG" && pwd) || { echo "orchestrate_m4max: cannot resolve LOG dir" >&2; exit 1; }
if [ -z "$LOG" ] || [ "$LOG" = "/" ]; then
  echo "orchestrate_m4max: refusing to use LOG='$LOG'" >&2
  exit 1
fi

rm -f "$LOG/ALL_DONE_m4max.marker" "$LOG/MEASUREMENT_FAILED_m4max.marker" "$LOG/GATE_NOT_PASSED_m4max.marker"

# --- 専有ゲートの進行状態はセッション単位（session-start スタンプ）で永続化
#     し、プロセス再起動をまたいで「最大 10 試行・約 30 分経過時間上限」を
#     強制する（#1488/#1506 是正版と同一設計。真に独立した再計測をやり直す
#     場合は、このスタンプファイルごと新しい LOG ディレクトリで実行する）。
STAMP="$LOG/session-start-m4max.stamp"
STATE="$LOG/gate-state-m4max"
CAP_S=1800
MAX_ATTEMPTS=10
GATE_LOG="$LOG/gate-m4max.log"

refuse() {
  echo "$1" > "$LOG/GATE_NOT_PASSED_m4max.marker"
  echo "$2 $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$GATE_LOG"
  exit 0
}

is_uint() {
  case "$1" in
    ''|*[!0-9]*) return 1 ;;
    *) return 0 ;;
  esac
}

save_state() {
  _content="$1 $2 $3 $4"
  if ! printf '%s\n' "$_content" > "$STATE.tmp" || ! mv -f "$STATE.tmp" "$STATE"; then
    rm -f "$STATE.tmp"
    refuse "gate state save failed (fail-closed; no measurement)" \
      "gate state save failed: content='${_content}'"
  fi
  if [ "$(cat "$STATE" 2>/dev/null)" != "$_content" ]; then
    refuse "gate state verify-after-save failed (fail-closed; no measurement)" \
      "gate state verify-after-save failed: content='${_content}'"
  fi
}

NOW_EPOCH=$(date -u +%s)
if [ ! -f "$STAMP" ]; then
  save_state 0 60 $((NOW_EPOCH + 60)) 0
  echo "$NOW_EPOCH" > "$STAMP"
fi
SESSION_START=$(cat "$STAMP" 2>/dev/null)
if ! is_uint "$SESSION_START"; then
  refuse "session stamp unreadable or invalid (fail-closed; no measurement)" \
    "gate session stamp unreadable or invalid: '${SESSION_START}'"
fi
ELAPSED=$((NOW_EPOCH - SESSION_START))
if [ "$ELAPSED" -ge "$CAP_S" ]; then
  refuse "session elapsed=${ELAPSED}s >= cap=${CAP_S}s at invocation start; refusing further attempts (no silent restart-reset)" \
    "gate cap exceeded before any attempt: elapsed=${ELAPSED}s cap=${CAP_S}s"
fi

if [ ! -f "$STATE" ]; then
  refuse "session stamp exists but gate state file is missing; cannot restore session (fail-closed; no measurement)" \
    "gate state missing for existing session: elapsed=${ELAPSED}s"
fi
STATE_LINE=$(cat "$STATE" 2>/dev/null)
set -- $STATE_LINE
if [ "$#" -ne 4 ] || ! is_uint "$1" || ! is_uint "$2" || ! is_uint "$3" || ! is_uint "$4" \
  || [ "$2" -lt 1 ] || [ "$4" -gt 1 ]; then
  refuse "gate state file unreadable or invalid ('${STATE_LINE}'); cannot restore session (fail-closed; no measurement)" \
    "gate state invalid for existing session: '${STATE_LINE}' elapsed=${ELAPSED}s"
fi
ATTEMPT=$1
BACKOFF_S=$2
NEXT_EPOCH=$3
PASS_COUNT=$4
if [ "$ATTEMPT" -ge "$MAX_ATTEMPTS" ]; then
  refuse "session cumulative attempts=${ATTEMPT} >= max=${MAX_ATTEMPTS} at invocation start; refusing further attempts (no silent restart-reset)" \
    "gate attempt cap exceeded before any attempt: attempts=${ATTEMPT} max=${MAX_ATTEMPTS} elapsed=${ELAPSED}s"
fi
if [ "$ATTEMPT" -gt 0 ]; then
  if [ "$PASS_COUNT" -eq 1 ] && [ "$NOW_EPOCH" -gt "$NEXT_EPOCH" ]; then
    echo "gate restore: first-pass confirmation window (30s) already elapsed; dropping pass_count to 0 (fail-closed) elapsed=${ELAPSED}s $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$GATE_LOG"
    PASS_COUNT=0
    NEXT_EPOCH=$((NOW_EPOCH + BACKOFF_S))
    save_state "$ATTEMPT" "$BACKOFF_S" "$NEXT_EPOCH" "$PASS_COUNT"
  fi
  echo "gate restore: attempts=${ATTEMPT} backoff=${BACKOFF_S}s next_epoch=${NEXT_EPOCH} pass_count=${PASS_COUNT} elapsed=${ELAPSED}s $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$GATE_LOG"
fi

# --- 専有ゲート本体: 60 秒開始・不合格時のみ backoff_s を 1.5 倍・1 回目
#     合格直後の 2 回目確認だけは 30 秒固定・最大 10 試行。閾値は #1309 の
#     4.0 ではなく #1488（本ホストの CPU 計測。Metal より背景負荷に敏感な
#     経路への採用値）と同一の 6.0 を使う（計画 §4 規則 3）。
GATE_OK=0
while [ "$ATTEMPT" -lt "$MAX_ATTEMPTS" ]; do
  NOW_EPOCH=$(date -u +%s)
  ELAPSED=$((NOW_EPOCH - SESSION_START))
  REMAINING=$((CAP_S - ELAPSED))
  if [ "$REMAINING" -le 0 ]; then
    echo "gate cap reached mid-loop: elapsed=${ELAPSED}s cap=${CAP_S}s $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$GATE_LOG"
    break
  fi
  WAIT_S=$((NEXT_EPOCH - NOW_EPOCH))
  if [ "$WAIT_S" -lt 0 ]; then
    WAIT_S=0
  fi
  if [ "$WAIT_S" -gt "$REMAINING" ]; then
    echo "gate insufficient remaining time: next wait=${WAIT_S}s > remaining=${REMAINING}s (elapsed=${ELAPSED}s cap=${CAP_S}s); not shortening the interval $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$GATE_LOG"
    break
  fi
  ATTEMPT=$((ATTEMPT + 1))
  save_state "$ATTEMPT" "$BACKOFF_S" "$NEXT_EPOCH" "$PASS_COUNT"
  sleep "$WAIT_S"
  NOW_EPOCH=$(date -u +%s)
  ELAPSED=$((NOW_EPOCH - SESSION_START))
  if [ "$ELAPSED" -ge "$CAP_S" ]; then
    echo "gate cap reached after wait: attempt=$ATTEMPT elapsed=${ELAPSED}s cap=${CAP_S}s; sample not evaluated $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$GATE_LOG"
    break
  fi
  LOAD1=$(uptime | sed -E 's/.*load averages?: ([0-9.]+)[, ].*/\1/')
  OK=$(awk -v l="$LOAD1" 'BEGIN{print (l != "" && l == l+0 && l < 6.0) ? 1 : 0}')
  echo "gate attempt=$ATTEMPT (session cumulative) wait=${WAIT_S}s load1=$LOAD1 ok=$OK elapsed=${ELAPSED}s $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$GATE_LOG"
  if [ "$OK" = "1" ]; then
    PASS_COUNT=$((PASS_COUNT + 1))
  else
    PASS_COUNT=0
  fi
  if [ "$PASS_COUNT" -ge 2 ]; then
    GATE_OK=1
    break
  fi
  if [ "$PASS_COUNT" -eq 1 ]; then
    NEXT_EPOCH=$((NOW_EPOCH + 30))
  else
    BACKOFF_S=$(awk -v w="$BACKOFF_S" 'BEGIN{printf "%d", w*1.5}')
    NEXT_EPOCH=$((NOW_EPOCH + BACKOFF_S))
  fi
  save_state "$ATTEMPT" "$BACKOFF_S" "$NEXT_EPOCH" "$PASS_COUNT"
done

if [ "$GATE_OK" != "1" ]; then
  echo "gate not passed within elapsed-time cap (${CAP_S}s) / ${MAX_ATTEMPTS} attempts (session cumulative=${ATTEMPT})" > "$LOG/GATE_NOT_PASSED_m4max.marker"
  exit 0
fi

# --- 並走プロセス確認（バイナリ名の完全一致。#1309/#1488 と同じ理由で
#     `pgrep -x` を使う） ---
{
  echo "=== proc check $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  pgrep -x -l 'bench-fandhe' || true
  pgrep -x -l 'bench-candle' || true
} >> "$LOG/gate-m4max.log"
if pgrep -x 'bench-fandhe' > /dev/null 2>&1 || pgrep -x 'bench-candle' > /dev/null 2>&1; then
  echo "sibling bench process detected; treating gate as not passed" > "$LOG/GATE_NOT_PASSED_m4max.marker"
  exit 0
fi

# --- サーマル状態（計測前）・uptime 30 秒ポーラ（バックグラウンド） ---
pmset -g therm > "$LOG/pmset_therm_before.txt" 2>&1 || true
(
  while true; do
    echo "poll $(date -u +%Y-%m-%dT%H:%M:%SZ) $(uptime)" >> "$LOG/uptime-m4max.log"
    sleep 30
  done
) &
POLLER_PID=$!

fail_measurement() {
  kill "$POLLER_PID" 2>/dev/null || true
  pmset -g therm > "$LOG/pmset_therm_after.txt" 2>&1 || true
  echo "$1 failed $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$LOG/MEASUREMENT_FAILED_m4max.marker"
  exit 1
}

# --- 正式系列計測 ---
# LABEL は環境変数で上書き可能（既定 0.8.0-1490）。
LABEL="${GEMM_GATE_LABEL:-0.8.0-1490}"
echo "formal start $(date -u +%Y-%m-%dT%H:%M:%SZ) $(uptime)" >> "$LOG/gate-m4max.log"
if ! bash run_gemm_gate_metal.sh "$LABEL" \
  > "$LOG/run_gemm_gate_metal-m4max-${LABEL}.log" 2>&1; then
  fail_measurement "formal series"
fi
echo "formal end $(date -u +%Y-%m-%dT%H:%M:%SZ) $(uptime)" >> "$LOG/gate-m4max.log"

kill "$POLLER_PID" 2>/dev/null || true
pmset -g therm > "$LOG/pmset_therm_after.txt" 2>&1 || true

echo "all done $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$LOG/ALL_DONE_m4max.marker"
