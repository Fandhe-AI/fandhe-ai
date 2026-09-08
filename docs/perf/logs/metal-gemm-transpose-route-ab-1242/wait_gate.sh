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
# 判定条件: 1 分 load average < GATE_THRESHOLD **かつ**
# cargo/rustc/python3/gemm_transpose_route_ab_bench
# （ビルド・計測系プロセス、および対象バイナリの直接起動を含む。
# 他セッションの並走を示す代理指標）が 0 件、の両方が CONSEC_REQUIRED
# 回連続で成立すること。
#
# PR #1459 codex-review 指摘の是正（イシュー #1253）: 従来版は
# proc_count を計測・ログには記録するものの gate_ok（PASSED 判定）へは
# load1 のみを使っており、「他 GPU プロセスなし」という排他計測契約の
# 半分（プロセス条件）が判定へ反映されていなかった。本版は
# proc_count == 0 も gate_ok の必須条件へ加える（load1 のみでの緩和は
# 行わない）。
#
# PR #1459 codex-review 再指摘の是正（イシュー #1253）: 排他検査の対象が
# cargo/rustc/python3（`cargo run` 経由の起動を前提とした代理指標）のみ
# だったため、他セッションが `gemm_transpose_route_ab_bench` を `cargo
# run` を介さず直接（例: `target/release/examples/...` を直接実行）起動
# した場合に検出できなかった。本版はビルド済みバイナリ名
# `gemm_transpose_route_ab_bench` も監視対象へ加える
# （`crates/backend-metal/Cargo.toml` の example 名。同型箇所
# orchestrate.sh:75/97/125 も同時に是正する）。
#
# 注意: 本スクリプトは attempt 1 の実行後に再構成したもので、attempt 1 の
# 判定条件を再現する保証はない（attempt 1 の wait_gate.log は util= 欄を
# 持ち gemm_transpose_route_ab_bench も検出しているが本版は出力しない。
# attempt 1 は load1 < 2.0 の行でも gate_ok=0 のため判定条件は未確定。
# `docs/perf/metal-gemm-transpose-tiled.md` §5.6）。
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
  # GPU/build 系プロセス（cargo・rustc・python3、および対象バイナリ直接起動
  # gemm_transpose_route_ab_bench）を他セッション並走の代理指標として数える。
  procs=""
  proc_count=0
  for name in cargo rustc python3 gemm_transpose_route_ab_bench; do
    cnt=$(pgrep -x "$name" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$cnt" -gt 0 ]; then
      procs="${procs}${name},"
      proc_count=$((proc_count + cnt))
    fi
  done
  # load1 とプロセス不在の両方を満たしたときのみ gate_ok=1
  # （プロセス条件を落とすと排他計測契約の片側しか検査しなくなる）。
  ok=$(awk -v l="$load1" -v t="$GATE_THRESHOLD" -v p="$proc_count" 'BEGIN{print (l<t && p==0)?1:0}')
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
