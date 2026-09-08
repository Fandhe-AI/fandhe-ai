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
#
# PR #1459 codex-review／Cursor Bugbot 指摘の是正（イシュー #1253）:
# (1) `gemm_transpose_route_ab_bench` は `required-features =
#     ["internal-diagnostics"]`（`crates/backend-metal/Cargo.toml`）を
#     要求するため、`--features internal-diagnostics` なしの `cargo run`
#     は計測開始前に失敗し valid_runs=0 になりうる。本版は明示指定する。
# (2) 排他計測契約（実行直前・実行中の load average < 2・他 GPU プロセス
#     なし）は従来 run 実行前後の静的な uptime／ps 確認と終了コードのみで
#     判定しており、実行中に他セッションが割り込んだケースを検出できな
#     かった。本版は各 run のバックグラウンド実行中、`wait_gate.sh` と
#     同一閾値（`GATE_THRESHOLD`／`POLL_INTERVAL_SECS`）で load average・
#     他 GPU/build 系プロセス（cargo/rustc/python3。自 run が起動した
#     `cargo run` 自身のプロセスツリーは自プロセスとして除外する）を
#     ポーリング監視し、逸脱（breach）を検出した run は終了コードに関わ
#     らず valid_runs に含めない（`phase1_run${n}_monitor.log` に記録）。
#
# PR #1459 codex-review 再指摘の是正（イシュー #1253）:
# (3) 上記 (2) の監視対象が cargo/rustc/python3 のみだったため、他セッシ
#     ョンが `gemm_transpose_route_ab_bench` を `cargo run` を介さず直接
#     起動した場合に検出できなかった。本版はビルド済みバイナリ名
#     `gemm_transpose_route_ab_bench` も pre-check／監視／post-check の
#     全 3 箇所（`wait_gate.sh` と同型の是正）へ加える。
# (4) 監視中に BREACH を検出した run は valid_runs へは含めないが、
#     `phase1_run${n}.log`（生の計測出力）は削除せず残す設計のため、
#     `aggregate.py` 側が集計時に本 run の監視結果（`phase1_run${n}_
#     monitor.log` の BREACH 有無）を必ず参照し、排他条件不成立の計測を
#     有効な計測と同列に集計・表示しないようにする（`aggregate.py` 側の
#     是正）。
#
# PR #1459 codex-review 再々指摘の是正（イシュー #1253・P1）: 待機フェーズ
# （wait_gate.sh の GATE_THRESHOLD）と実行中監視フェーズ（本スクリプトの
# 従来 MONITOR_GATE_THRESHOLD）が別名の変数として重複定義されており、
# 例えば GATE_THRESHOLD=1.0 で再実行しても実行中監視は既定 2.0 のままに
# なり得た（AGENTS.md の閾値分散定義禁止に反する）。本版は
# `gate_common.sh`（wait_gate.sh と共有する単一定義）を source して
# GATE_THRESHOLD を得て、実行中監視ループでもそのまま同名で使う
# （MONITOR_GATE_THRESHOLD は廃止）。
set -uo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DERIVED_WORKDIR="$(cd "$SELF_DIR/../../../.." && pwd)"
WORKDIR="${WORKDIR:-$DERIVED_WORKDIR}"
LOGDIR="${LOGDIR:-$SELF_DIR}"
ATTEMPT="${ATTEMPT:-2}"
GATE_LOG="$LOGDIR/wait_gate_attempt${ATTEMPT}.log"

# shellcheck source=gate_common.sh
source "$SELF_DIR/gate_common.sh"
MONITOR_POLL_INTERVAL_SECS="${MONITOR_POLL_INTERVAL_SECS:-30}"

cd "$WORKDIR" || { echo "エラー: WORKDIR='$WORKDIR' への移動に失敗した" >&2; exit 1; }

echo "$(date +"%Y-%m-%dT%H:%M:%S%z") orchestrator attempt${ATTEMPT} start" >> "$LOGDIR/orchestrate.log"
echo "$(date +"%Y-%m-%dT%H:%M:%S%z") attempt ${ATTEMPT}: waiting for gate (see $(basename "$GATE_LOG"))" >> "$LOGDIR/orchestrate.log"

GATE_RESULT=$(LOGDIR="$LOGDIR" OUT="$GATE_LOG" GATE_THRESHOLD="$GATE_THRESHOLD" bash "$LOGDIR/wait_gate.sh")

echo "$(date +"%Y-%m-%dT%H:%M:%S%z") attempt ${ATTEMPT}: gate result=${GATE_RESULT}" >> "$LOGDIR/orchestrate.log"

if [ "$GATE_RESULT" != "PASSED" ]; then
  echo "ORCHESTRATOR_RESULT=TIMEOUT valid_runs=0 attempt=${ATTEMPT}" >> "$LOGDIR/orchestrate.log"
  echo "TIMEOUT" > "$LOGDIR/DONE_TIMEOUT_ATTEMPT${ATTEMPT}"
  exit 0
fi

# self_pid_tree: 与えた root PID とその子孫すべての PID を列挙する
# （macOS には pstree が無いため pgrep -P で再帰的に辿る自前実装）。
# run 実行中監視で「他 GPU/build 系プロセス」を判定する際、自 run が
# 起動した cargo/rustc 自身を誤検知しないよう除外するために使う。
self_pid_tree() {
  local root="$1"
  echo "$root"
  local child
  for child in $(pgrep -P "$root" 2>/dev/null); do
    self_pid_tree "$child"
  done
}

valid_runs=0
for n in 1 2 3; do
  {
    echo "=== run${n} pre-check $(date +"%Y-%m-%dT%H:%M:%S%z") ==="
    uptime
    echo "--- GPU/build 系プロセス（cargo/rustc/python3/gemm_transpose_route_ab_bench） ---"
    # Cursor Bugbot 指摘の是正（イシュー #1253・Medium）: comm 列は Darwin で
    # 15 文字に切り詰まるため gemm_transpose_route_ab_bench（29 文字）を
    # 末尾一致で検出できない。診断用ログのため comm ではなく完全なコマンド
    # ライン（command 列）を対象にする。
    ps -Ao pid,pcpu,command | grep -E '(cargo|rustc|python3|gemm_transpose_route_ab_bench)' | grep -v grep || echo "(none)"
  } > "$LOGDIR/uptime_before_run${n}.txt"

  MONITOR_LOG="$LOGDIR/phase1_run${n}_monitor.log"
  : > "$MONITOR_LOG"

  cargo run -p fandhe-ai-backend-metal --example gemm_transpose_route_ab_bench --release \
    --features internal-diagnostics \
    -- --phase1-only > "$LOGDIR/phase1_run${n}.log" 2>&1 &
  BENCH_PID=$!

  # run 実行中の排他計測契約（load average < 2・他 GPU プロセスなし）を
  # ポーリング監視する。bench 自身のプロセスツリーは self_pid_tree で
  # 除外したうえで cargo/rustc/python3/gemm_transpose_route_ab_bench の
  # 残存を「他プロセス」とみなす（gemm_transpose_route_ab_bench を含める
  # のは、他セッションが本バイナリを cargo run を介さず直接起動した場合
  # も検出するため。#1459 codex-review 再指摘）。
  breach=0
  while kill -0 "$BENCH_PID" 2>/dev/null; do
    self_pids=" $(self_pid_tree "$BENCH_PID" | tr '\n' ' ') "
    ts=$(date +"%Y-%m-%dT%H:%M:%S%z")
    load_line=$(uptime)
    load1=$(printf '%s' "$load_line" | sed -E 's/.*load averages?:[[:space:]]*([0-9.]+).*/\1/')
    other_procs=""
    other_count=0
    # Cursor Bugbot 指摘の是正（イシュー #1253・Medium）: `gemm_transpose_
    # route_ab_bench`（29 文字）は Darwin の comm が 15 文字までしか保持
    # しないため `pgrep -x`（comm 完全一致）では検出できない（wait_gate.sh
    # と同型の是正。同スクリプトのコメント参照）。当該バイナリのみ
    # `pgrep -f`（コマンドライン全体照合）で検出する。
    for name in cargo rustc python3; do
      for pid in $(pgrep -x "$name" 2>/dev/null); do
        case "$self_pids" in
          *" $pid "*) ;;
          *)
            other_procs="${other_procs}${name}:${pid},"
            other_count=$((other_count + 1))
            ;;
        esac
      done
    done
    for pid in $(pgrep -f '(^|/)gemm_transpose_route_ab_bench([[:space:]]|$)' 2>/dev/null); do
      case "$self_pids" in
        *" $pid "*) ;;
        *)
          other_procs="${other_procs}gemm_transpose_route_ab_bench:${pid},"
          other_count=$((other_count + 1))
          ;;
      esac
    done
    load_ok=$(awk -v l="$load1" -v t="$GATE_THRESHOLD" 'BEGIN{print (l<t)?1:0}')
    if [ "$load_ok" != "1" ] || [ "$other_count" -gt 0 ]; then
      breach=1
      echo "$ts load1=$load1 other_count=$other_count other_procs=[$other_procs] BREACH" >> "$MONITOR_LOG"
    else
      echo "$ts load1=$load1 other_count=$other_count other_procs=[$other_procs] ok" >> "$MONITOR_LOG"
    fi
    sleep "$MONITOR_POLL_INTERVAL_SECS"
  done

  wait "$BENCH_PID"
  RC=$?

  {
    echo "=== run${n} post-check $(date +"%Y-%m-%dT%H:%M:%S%z") rc=${RC} ==="
    uptime
    echo "--- GPU/build 系プロセス（cargo/rustc/python3/gemm_transpose_route_ab_bench） ---"
    # 上記 uptime_before_run と同型の是正（comm 切り詰め対策・command 列使用）。
    ps -Ao pid,pcpu,command | grep -E '(cargo|rustc|python3|gemm_transpose_route_ab_bench)' | grep -v grep || echo "(none)"
  } > "$LOGDIR/uptime_after_run${n}.txt"

  if [ "$RC" -eq 0 ] && [ "$breach" -eq 0 ]; then
    valid_runs=$((valid_runs + 1))
  else
    echo "run${n} failed rc=${RC} breach=${breach}" >> "$LOGDIR/orchestrate.log"
  fi
done

echo "$(date +"%Y-%m-%dT%H:%M:%S%z") attempt ${ATTEMPT}: done valid_runs=${valid_runs}/3" >> "$LOGDIR/orchestrate.log"
echo "ORCHESTRATOR_RESULT=DONE valid_runs=${valid_runs}" >> "$LOGDIR/orchestrate.log"
