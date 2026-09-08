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
#
# PR #1459 codex-review 四度目の指摘の是正（イシュー #1253・P2）:
# `wait_gate.sh` の起動元を `$LOGDIR/wait_gate.sh` としていたため、再利用時
# に `LOGDIR` を本ディレクトリ以外（既存記録と分けた出力用ディレクトリ等）
# へ変更すると、その出力先ディレクトリ配下に存在しない `wait_gate.sh` を
# 探して `bash` が「No such file or directory」で失敗する。`LOGDIR` は
# `wait_gate.sh`（`OUT`／`GATE_LOG` の書き出し先）に渡す出力先引数であり、
# スクリプト自体の配置元とは独立であるべきなので、起動元は常に
# `SELF_DIR/wait_gate.sh` に固定する。また従来は `bash ...` の終了コードを
# 見ずに標準出力（`GATE_RESULT`）のみで分岐していたため、上記のような
# 起動失敗（空の標準出力）も「ゲート待ちの TIMEOUT」と誤記録されていた。
# 本版は起動コマンドの終了コード（`GATE_RC`）を別途保持し、非 0（起動
# 失敗）と `TIMEOUT`（正常起動した上でのゲート不成立）を区別して記録する。
#
# PR #1459 codex-review 五度目の指摘の是正（イシュー #1253・P2）:
# 実行中監視ループは反復冒頭で自プロセスツリー（`self_pids`）をスナップ
# ショットし、その後 `pgrep` で列挙した PID を同スナップショットと照合して
# いた。スナップショット取得から `pgrep` 列挙までの間に自 `cargo` が
# `rustc` やベンチ本体を新たに起動すると、その自子孫 PID はスナップ
# ショットに無いため他セッションとして数えられ、1 回の誤検出で
# `breach=1` が固定されて排他条件を満たす計測が無効化されうる（TOCTOU）。
# 本版はスナップショット不一致の PID について `is_self_descendant`
# （`ps -o ppid=` で祖先を辿り `BENCH_PID` に到達するかを列挙時点で確認）
# を追加で適用し、自子孫と確定した PID を除外する。祖先を辿る途中で
# 消滅した PID（列挙と照合の間に終了した短命プロセス）は他セッションとも
# 自子孫とも確定できないため `vanished=[...]` として記録のみ行い
# `other_count` へは加算しない（load average 条件は別途判定されるため、
# 消滅済みプロセスを breach 根拠にはしない）。同型の照合箇所 2 箇所
# （cargo/rustc/python3 用と `gemm_transpose_route_ab_bench` 用）を共通
# 関数 `classify_pid` に集約し、判定規則の分散定義を避ける。
# 集計側 `aggregate.py` も同指摘で `LOGDIR`（第 1 引数または環境変数）を
# 参照するよう是正し、本スクリプトと入出力契約を揃えた
# （`LOGDIR=/path/to/out bash orchestrate.sh` の後に
# `LOGDIR=/path/to/out python3 aggregate.py`）。
#
# PR #1459 codex-review 六度目の指摘の是正（イシュー #1253・P2）:
# (5) `LOGDIR` が相対パスの場合、従来は `cd "$WORKDIR"` 後に解決されて
#     出力先が WORKDIR 基準になる一方、`aggregate.py` は呼び出し元の作業
#     ディレクトリ基準で解決するため、リポジトリ外から同じ相対 `LOGDIR`
#     を両者へ渡すと別ディレクトリを参照していた。本版は `cd` の前に
#     `LOGDIR` を呼び出し元基準の絶対パスへ正規化する（ディレクトリは
#     `mkdir -p` で作成）。
# (6) 同じ `LOGDIR` で成功した attempt の後に再試行してゲート待ちが
#     TIMEOUT すると、過去の `phase1_run*`／monitor／uptime ファイルが
#     残ったままになり、集計側が attempt を識別せずに読むと過去の成功
#     を今回の有効計測として表示していた。本版は run 単位の全記録を
#     `attempt${ATTEMPT}_` 接頭辞付きで保存し（`attempt${ATTEMPT}_
#     phase1_run${n}.log`／`_monitor.log`・`attempt${ATTEMPT}_uptime_
#     {before,after}_run${n}.txt`）、run ループ完了時に完了記録
#     `DONE_ATTEMPT${ATTEMPT}`（`DONE valid_runs=N`）を書く。同じ
#     `ATTEMPT` の記録（run 記録・DONE／TIMEOUT／GATE_LAUNCH_ERROR
#     マーカー）が既に存在する場合は上書きせず fail-closed に中止する
#     （`ATTEMPT` を変えるか、旧記録を明示的に退避してから再実行する）。
#     `aggregate.py` 側は同じ `ATTEMPT` 接頭辞の記録のみ読み、完了記録の
#     `valid_runs` と照合する。
set -uo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DERIVED_WORKDIR="$(cd "$SELF_DIR/../../../.." && pwd)"
WORKDIR="${WORKDIR:-$DERIVED_WORKDIR}"
LOGDIR="${LOGDIR:-$SELF_DIR}"
# LOGDIR は cd "$WORKDIR" より前に呼び出し元基準の絶対パスへ正規化する
# （相対指定時に aggregate.py〈呼び出し元 cwd 基準〉と別ディレクトリを
# 参照しないため。PR #1459 codex-review 六度目の指摘の是正）。
mkdir -p "$LOGDIR" || { echo "エラー: LOGDIR='$LOGDIR' を作成できない" >&2; exit 1; }
LOGDIR="$(cd "$LOGDIR" && pwd)" || { echo "エラー: LOGDIR の正規化に失敗した" >&2; exit 1; }
ATTEMPT="${ATTEMPT:-2}"
GATE_LOG="$LOGDIR/wait_gate_attempt${ATTEMPT}.log"
RUN_PREFIX="$LOGDIR/attempt${ATTEMPT}_"

# 同じ ATTEMPT の記録が既にある場合は上書きせず中止する（過去 attempt の
# 記録が今回の集計へ混入・消失するのを防ぐ fail-closed。PR #1459
# codex-review 六度目の指摘の是正）。
for existing in "$RUN_PREFIX"* "$LOGDIR/DONE_ATTEMPT${ATTEMPT}" \
  "$LOGDIR/DONE_TIMEOUT_ATTEMPT${ATTEMPT}" "$LOGDIR/DONE_GATE_LAUNCH_ERROR_ATTEMPT${ATTEMPT}"; do
  if [ -e "$existing" ]; then
    echo "エラー: attempt${ATTEMPT} の記録 '$existing' が既に存在する。ATTEMPT を変えるか旧記録を退避してから再実行すること" >&2
    exit 1
  fi
done

# shellcheck source=gate_common.sh
source "$SELF_DIR/gate_common.sh"
MONITOR_POLL_INTERVAL_SECS="${MONITOR_POLL_INTERVAL_SECS:-30}"

cd "$WORKDIR" || { echo "エラー: WORKDIR='$WORKDIR' への移動に失敗した" >&2; exit 1; }

echo "$(date +"%Y-%m-%dT%H:%M:%S%z") orchestrator attempt${ATTEMPT} start" >> "$LOGDIR/orchestrate.log"
echo "$(date +"%Y-%m-%dT%H:%M:%S%z") attempt ${ATTEMPT}: waiting for gate (see $(basename "$GATE_LOG"))" >> "$LOGDIR/orchestrate.log"

# wait_gate.sh は常に SELF_DIR（本スクリプトの配置元）から起動する。LOGDIR
# は出力先引数として渡すのみで、起動元の探索には使わない（LOGDIR を出力用
# ディレクトリへ変更した再利用時に「No such file or directory」で失敗する
# のを防ぐ。PR #1459 codex-review 四度目の指摘の是正）。
GATE_RESULT=$(LOGDIR="$LOGDIR" OUT="$GATE_LOG" GATE_THRESHOLD="$GATE_THRESHOLD" bash "$SELF_DIR/wait_gate.sh")
GATE_RC=$?

echo "$(date +"%Y-%m-%dT%H:%M:%S%z") attempt ${ATTEMPT}: gate result=${GATE_RESULT} rc=${GATE_RC}" >> "$LOGDIR/orchestrate.log"

# 起動コマンド自体の失敗（GATE_RC != 0。スクリプト不在・source 失敗等）は
# 正常に起動したうえでのゲート不成立（TIMEOUT）とは異なる事象であり、
# 混同すると「ゲート待ちを試みたが不成立だった」と誤解される。両者を
# 区別して記録する（PR #1459 codex-review 四度目の指摘の是正）。
if [ "$GATE_RC" -ne 0 ]; then
  echo "ORCHESTRATOR_RESULT=GATE_LAUNCH_ERROR valid_runs=0 attempt=${ATTEMPT} rc=${GATE_RC}" >> "$LOGDIR/orchestrate.log"
  echo "GATE_LAUNCH_ERROR rc=${GATE_RC}" > "$LOGDIR/DONE_GATE_LAUNCH_ERROR_ATTEMPT${ATTEMPT}"
  exit 1
fi

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

# is_self_descendant: 与えた PID の祖先を `ps -o ppid=` で辿り、root（自
# run の BENCH_PID）に到達すれば 0（自子孫）、PID 1／0 まで到達しても root
# に当たらなければ 1（他プロセス）、途中で PID が消滅して祖先を確定
# できなければ 2（判定不能・消滅）を返す。self_pid_tree のスナップ
# ショット取得後に自 cargo が起動した rustc／ベンチ本体を、列挙時点の
# 祖先関係で改めて自子孫と判定するために使う（PR #1459 codex-review
# 五度目の指摘の是正）。祖先の探索深さは異常な循環に備えて 64 で打ち切る。
is_self_descendant() {
  local pid="$1" root="$2" depth=0 ppid
  while [ "$pid" -gt 1 ] && [ "$depth" -lt 64 ]; do
    if [ "$pid" -eq "$root" ]; then
      return 0
    fi
    ppid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d '[:space:]')
    if [ -z "$ppid" ]; then
      return 2
    fi
    pid="$ppid"
    depth=$((depth + 1))
  done
  return 1
}

# classify_pid: 監視ループで列挙した 1 PID を「自子孫（除外）」「他プロ
# セス（other_procs へ加算）」「消滅（vanished へ記録のみ）」へ分類する。
# まず反復冒頭のスナップショット（self_pids）で高速に除外し、不一致の
# PID のみ is_self_descendant で祖先関係を確認する（スナップショット後に
# 起動した自子孫の誤検出を防ぐ）。結果は other_procs／other_count／
# vanished_procs（呼び出し側のシェル変数）へ反映する。
classify_pid() {
  local name="$1" pid="$2"
  case "$self_pids" in
    *" $pid "*) return 0 ;;
  esac
  is_self_descendant "$pid" "$BENCH_PID"
  case $? in
    0) ;;
    2) vanished_procs="${vanished_procs}${name}:${pid}," ;;
    *)
      other_procs="${other_procs}${name}:${pid},"
      other_count=$((other_count + 1))
      ;;
  esac
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
  } > "${RUN_PREFIX}uptime_before_run${n}.txt"

  MONITOR_LOG="${RUN_PREFIX}phase1_run${n}_monitor.log"
  : > "$MONITOR_LOG"

  cargo run -p fandhe-ai-backend-metal --example gemm_transpose_route_ab_bench --release \
    --features internal-diagnostics \
    -- --phase1-only > "${RUN_PREFIX}phase1_run${n}.log" 2>&1 &
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
    vanished_procs=""
    # Cursor Bugbot 指摘の是正（イシュー #1253・Medium）: `gemm_transpose_
    # route_ab_bench`（29 文字）は Darwin の comm が 15 文字までしか保持
    # しないため `pgrep -x`（comm 完全一致）では検出できない（wait_gate.sh
    # と同型の是正。同スクリプトのコメント参照）。当該バイナリのみ
    # `pgrep -f`（コマンドライン全体照合）で検出する。
    # 各 PID の自子孫／他プロセス／消滅の分類は classify_pid（スナップ
    # ショット照合 + 祖先関係の再確認）に集約する（PR #1459 codex-review
    # 五度目の指摘の是正）。
    for name in cargo rustc python3; do
      for pid in $(pgrep -x "$name" 2>/dev/null); do
        classify_pid "$name" "$pid"
      done
    done
    for pid in $(pgrep -f '(^|/)gemm_transpose_route_ab_bench([[:space:]]|$)' 2>/dev/null); do
      classify_pid gemm_transpose_route_ab_bench "$pid"
    done
    load_ok=$(awk -v l="$load1" -v t="$GATE_THRESHOLD" 'BEGIN{print (l<t)?1:0}')
    if [ "$load_ok" != "1" ] || [ "$other_count" -gt 0 ]; then
      breach=1
      echo "$ts load1=$load1 other_count=$other_count other_procs=[$other_procs] vanished=[$vanished_procs] BREACH" >> "$MONITOR_LOG"
    else
      echo "$ts load1=$load1 other_count=$other_count other_procs=[$other_procs] vanished=[$vanished_procs] ok" >> "$MONITOR_LOG"
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
  } > "${RUN_PREFIX}uptime_after_run${n}.txt"

  if [ "$RC" -eq 0 ] && [ "$breach" -eq 0 ]; then
    valid_runs=$((valid_runs + 1))
  else
    echo "run${n} failed rc=${RC} breach=${breach}" >> "$LOGDIR/orchestrate.log"
  fi
done

echo "$(date +"%Y-%m-%dT%H:%M:%S%z") attempt ${ATTEMPT}: done valid_runs=${valid_runs}/3" >> "$LOGDIR/orchestrate.log"
echo "ORCHESTRATOR_RESULT=DONE valid_runs=${valid_runs}" >> "$LOGDIR/orchestrate.log"
# 完了記録（aggregate.py が同じ ATTEMPT の集計結果と照合する）。
echo "DONE valid_runs=${valid_runs}" > "$LOGDIR/DONE_ATTEMPT${ATTEMPT}"
