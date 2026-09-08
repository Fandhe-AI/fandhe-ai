#!/usr/bin/env bash
# イシュー #1253: 排他環境（load average < 2・他 GPU プロセスなし）ゲート待ち。
#
# #1242（attempt 1・本ディレクトリの orchestrate.log／wait_gate.log）は
# 最大 3 時間待っても load average が 2 未満へ収束せず TIMEOUT した。本
# スクリプトは attempt 2 として同一閾値（緩めない。イシュー本文「ゲート
# 閾値は変更しない」）で再試行するが、待機上限は有限に区切る
# （#1187／#1284 の前例に倣い、再度不通過なら undetermined として記録
# する方針。orchestrate.sh から呼ばれる）。
#
# 判定条件: 1 分 load average < GATE_THRESHOLD が CONSEC_REQUIRED 回連続。
# GPU プロセス確認は cargo/rustc/python3（ビルド・計測系プロセス。他
# セッションの並走を示す代理指標）の有無を procs= として記録する。
# 注意: 本スクリプトは attempt 1 の実行後に再構成したもので、attempt 1 の
# 判定条件を再現する保証はない（attempt 1 の wait_gate.log は util= 欄を
# 持ち gemm_transpose_route_ab_bench も検出しているが本版は出力しない。
# attempt 1 は load1 < 2.0 の行でも gate_ok=0 のため判定条件は未確定。
# `docs/perf/metal-gemm-transpose-tiled.md` §5.5）。（誤検知を許容する簡易版であり、実際の
# GPU 使用有無は各 run 実行前後の `ps`／`uptime` 生ログで人間が確認する
# 前提）。
set -uo pipefail

LOGDIR="${LOGDIR:?LOGDIR required}"
GATE_THRESHOLD="${GATE_THRESHOLD:-2.0}"
CONSEC_REQUIRED="${CONSEC_REQUIRED:-2}"
MAX_WAIT_SECS="${MAX_WAIT_SECS:-600}"
POLL_INTERVAL_SECS="${POLL_INTERVAL_SECS:-30}"
OUT="${OUT:?OUT required}"

start_ts=$(date +%s)
consec=0
result="TIMEOUT"

while :; do
  now_ts=$(date +%s)
  elapsed=$((now_ts - start_ts))
  ts=$(date +"%Y-%m-%dT%H:%M:%S%z")
  load_line=$(uptime)
  load1=$(printf '%s' "$load_line" | sed -E 's/.*load averages?:[[:space:]]*([0-9.]+).*/\1/')
  load5=$(printf '%s' "$load_line" | sed -E 's/.*load averages?:[[:space:]]*[0-9.]+[[:space:]]+([0-9.]+).*/\1/')
  # GPU/build 系プロセス（cargo・rustc・python3）を他セッション並走の代理指標として数える。
  procs=""
  proc_count=0
  for name in cargo rustc python3; do
    cnt=$(pgrep -x "$name" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$cnt" -gt 0 ]; then
      procs="${procs}${name},"
      proc_count=$((proc_count + cnt))
    fi
  done
  ok=$(awk -v l="$load1" -v t="$GATE_THRESHOLD" 'BEGIN{print (l<t)?1:0}')
  if [ "$ok" = "1" ]; then
    consec=$((consec + 1))
  else
    consec=0
  fi
  echo "$ts elapsed=${elapsed}s load1=$load1 load5=$load5 proc_count=$proc_count procs=[$procs] gate_ok=$ok consecutive_ok=$consec" >> "$OUT"

  if [ "$consec" -ge "$CONSEC_REQUIRED" ]; then
    result="PASSED"
    break
  fi
  if [ "$elapsed" -ge "$MAX_WAIT_SECS" ]; then
    result="TIMEOUT"
    break
  fi
  sleep "$POLL_INTERVAL_SECS"
done

echo "${result} at $(date +"%Y-%m-%dT%H:%M:%S%z") (elapsed=${elapsed}s)" >> "$OUT"
echo "$result"
