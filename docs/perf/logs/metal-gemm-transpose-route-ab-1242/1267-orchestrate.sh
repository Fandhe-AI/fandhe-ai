#!/usr/bin/env bash
# イシュー #1267: 改善後プロトコル（#1264/#1265。`crates/bench-harness::
# env_guard`・`gemm_transpose_route_ab_bench` の実行前ガード CLI 引数
# `--max-load-avg`／`--gpu-watch`／`--guard-wait-secs`／`--guard-max-
# attempts`／`--env-info-out`／`--guard-only`）を用いて phase 1 → phase 2
# （30 セル A/B）を排他環境で完走させ、`verdict` を確定するための
# オーケストレーター。
#
# #1253／#1261 の教訓（`docs/perf/metal-gemm-transpose-tiled.md` §5.6・
# §5.7）を踏まえ、本スクリプトは 2 段の排他性チェックを持つ:
#   (1) 外側ゲート（本スクリプト。`gate_common.sh`／`wait_gate.sh` を再利用
#       ——同ファイルは編集しない。#1253 の記録用ファイルと役割が同じ
#       ため）: 本計測を「起動してよいか」の粗い事前判定。
#   (2) 内側ガード（`gemm_transpose_route_ab_bench` 本体。#1264/#1265 で
#       実装済みの `env_guard` API・バックオフ再試行）: phase 1 開始直前・
#       phase 2 開始直前の 2 回、`--max-load-avg`／`--gpu-watch` で再判定
#       し、`Fail` ならバックオフ再試行する。
# さらに (3) 本計測の実行中（GPU 計測が進行している間）は #1253 の
# `orchestrate.sh` と同型の監視ループで load average・GPU/build 系
# プロセスをポーリングし、逸脱（BREACH）を検出した run は無効
# （`valid=0`）として扱う（fail-closed。#1253 の設計をそのまま踏襲——
# 詳細な理由は `orchestrate.sh` 冒頭コメント参照。本スクリプトは対象
# バイナリ名を `gemm_transpose_route_ab_bench` に統一し、監視対象
# プロセス名も同じ watchlist を使う）。
#
# 事前宣言パラメータ（緩める方向の変更は不可。#1267 実装計画 §3
# Step 1）:
#   外側ゲート: GATE_THRESHOLD=2.0・CONSEC_REQUIRED=2・
#     POLL_INTERVAL_SECS=30・MAX_WAIT_SECS（既定 7200＝2 時間。環境変数で
#     上書き可——本計測実行時に実際に用いた値を env_info／docs へ明示する
#     契約は変更しない）
#   本計測（内側ガード）: --max-load-avg=2.0 --gpu-watch=python
#     --gpu-watch=torch --gpu-watch=mlx --gpu-watch=gemm_ --gpu-watch=bench
#     --guard-wait-secs=60 --guard-max-attempts=20
#     （待機列 60,90,135,202,300×16 ≈ 1.5h 上限。env_guard.rs のバック
#     オフ実装に従う）
#
# セキュリティ（OWASP A03）: 外部コマンドは固定引数のみで起動し、シェル
# 経由の任意実行はしない。ホスト名・ユーザー名の絶対パスはログへ書く前に
# 必ず SANITIZE_SED を通す（#1261 の方針を踏襲）。
#
# PR #1462 codex-review 是正: 監視ループはバイナリ起動直後（内側ガード
# のバックオフ待機区間を含む）から始まるため、待機中の負荷逸脱と
# phase 1／phase 2 実測区間中の逸脱を区別せずに breach を立てていた
# （詳細は `current_phase_state` 関数コメント参照）。実測区間の判定に
# は OUT_LOG（本計測プロセスの標準出力）中の `== env_guard(` ／
# `env_guard_result=` マーカーを用いる。
set -uo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGDIR="${LOGDIR:-$SELF_DIR}"
mkdir -p "$LOGDIR" || { echo "エラー: LOGDIR='$LOGDIR' を作成できない" >&2; exit 1; }
LOGDIR="$(cd "$LOGDIR" && pwd)" || { echo "エラー: LOGDIR の正規化に失敗した" >&2; exit 1; }
REPO_ROOT="$(cd "${SELF_DIR}/../../../.." && pwd)"
BINARY="${REPO_ROOT}/target/release/examples/gemm_transpose_route_ab_bench"

ATTEMPT="${ATTEMPT:-1}"
RUN_PREFIX="$LOGDIR/1267-attempt${ATTEMPT}-"

# 同じ ATTEMPT の記録が既にある場合は上書きせず中止する（#1253
# orchestrate.sh と同じ fail-closed 方針）。
for existing in "${RUN_PREFIX}"* \
  "$LOGDIR/1267-DONE_ATTEMPT${ATTEMPT}" \
  "$LOGDIR/1267-DONE_TIMEOUT_ATTEMPT${ATTEMPT}" \
  "$LOGDIR/1267-DONE_GATE_LAUNCH_ERROR_ATTEMPT${ATTEMPT}"; do
  if [ -e "$existing" ]; then
    echo "エラー: attempt${ATTEMPT} の記録 '$existing' が既に存在する。ATTEMPT を変えるか旧記録を退避してから再実行すること" >&2
    exit 1
  fi
done

if [ ! -x "${BINARY}" ]; then
    echo "ERROR: バイナリが見つからない（先に cargo build --release --features internal-diagnostics が必要）: ${BINARY}" >&2
    exit 1
fi

# shellcheck source=gate_common.sh
source "$SELF_DIR/gate_common.sh"
SANITIZE_SED='s#/Users/[^/[:space:]]+#/Users/<user>#g'

# ---------------------------------------------------------------------
# (1) 外側ゲート: wait_gate.sh を再利用する（#1253 の記録用ファイルは
# 編集しない。LOGDIR／OUT／GATE_THRESHOLD／CONSEC_REQUIRED／
# MAX_WAIT_SECS／POLL_INTERVAL_SECS を環境変数で渡す契約は同スクリプトの
# 既存仕様のまま）。
# ---------------------------------------------------------------------
CONSEC_REQUIRED="${CONSEC_REQUIRED:-2}"
POLL_INTERVAL_SECS="${POLL_INTERVAL_SECS:-30}"
MAX_WAIT_SECS="${MAX_WAIT_SECS:-7200}"
# attempt 接尾辞付きパス（`RUN_PREFIX`）にする。既存 `orchestrate.sh`
# （`GATE_LOG="$LOGDIR/wait_gate_attempt${ATTEMPT}.log"`）と同じ理由——
# 非接尾辞パスのままだと 2 回目以降の attempt 実行時に前の attempt の
# 外側ゲート記録が上書きで失われる（PR #1462 codex-review／Bugbot 指摘。
# #1267 実測自体はこのスクリプトを未経由のため実測データへの影響はない）。
GATE_LOG="${RUN_PREFIX}gate.log"
: > "$GATE_LOG"

gate_start_unix=$(date +%s)
echo "gate_start_unix=${gate_start_unix} attempt=${ATTEMPT} GATE_THRESHOLD=${GATE_THRESHOLD} CONSEC_REQUIRED=${CONSEC_REQUIRED} POLL_INTERVAL_SECS=${POLL_INTERVAL_SECS} MAX_WAIT_SECS=${MAX_WAIT_SECS}" >> "$GATE_LOG"

GATE_RESULT=$(LOGDIR="$LOGDIR" OUT="$GATE_LOG" GATE_THRESHOLD="$GATE_THRESHOLD" \
  CONSEC_REQUIRED="$CONSEC_REQUIRED" POLL_INTERVAL_SECS="$POLL_INTERVAL_SECS" \
  MAX_WAIT_SECS="$MAX_WAIT_SECS" bash "$SELF_DIR/wait_gate.sh")
GATE_RC=$?
gate_end_unix=$(date +%s)
gate_elapsed=$((gate_end_unix - gate_start_unix))
echo "gate_end_unix=${gate_end_unix} gate_elapsed_secs=${gate_elapsed} gate_result=${GATE_RESULT} gate_rc=${GATE_RC}" >> "$GATE_LOG"

if [ "$GATE_RC" -ne 0 ]; then
  echo "ORCHESTRATOR_RESULT=GATE_LAUNCH_ERROR valid=0 attempt=${ATTEMPT} rc=${GATE_RC}" >> "$LOGDIR/1267-orchestrate-stdout.log"
  echo "GATE_LAUNCH_ERROR rc=${GATE_RC} elapsed_secs=${gate_elapsed}" > "$LOGDIR/1267-DONE_GATE_LAUNCH_ERROR_ATTEMPT${ATTEMPT}"
  exit 1
fi

if [ "$GATE_RESULT" != "PASSED" ]; then
  echo "ORCHESTRATOR_RESULT=TIMEOUT valid=0 attempt=${ATTEMPT} elapsed_secs=${gate_elapsed}" >> "$LOGDIR/1267-orchestrate-stdout.log"
  echo "TIMEOUT elapsed_secs=${gate_elapsed}" > "$LOGDIR/1267-DONE_TIMEOUT_ATTEMPT${ATTEMPT}"
  echo "外側ゲート不通過（elapsed=${gate_elapsed}s）。計測せず終了する。" >&2
  exit 0
fi

# ---------------------------------------------------------------------
# 実行前後スナップショット
# ---------------------------------------------------------------------
snapshot() {
    local label="$1"
    {
        echo "=== ${label} unix=$(date +%s) ==="
        echo "--- uptime ---"
        uptime
        echo "--- pmset -g therm ---"
        pmset -g therm
        echo "--- GPU/build 系プロセス件数（cargo|rustc|python3|gemm_transpose_route_ab_bench） ---"
        ps -Ao pid,pcpu,command | grep -E '(cargo|rustc|python3|gemm_transpose_route_ab_bench)' | grep -v grep | sed -E "${SANITIZE_SED}" | wc -l | tr -d ' '
    }
}
snapshot "before attempt${ATTEMPT}" > "${RUN_PREFIX}uptime_before.txt"
# 単純な `uptime`／`pmset -g therm` 出力も attempt 接尾辞付きパスへ
# 書く（前段の GATE_LOG と同じ是正理由。非接尾辞パスのままだと 2 回目
# 以降の attempt 実行時に前の attempt の記録が上書きで失われる）。
uptime > "${RUN_PREFIX}uptime_before_plain.txt"
pmset -g therm > "${RUN_PREFIX}pmset_therm_before.txt" 2>&1

# ---------------------------------------------------------------------
# (2)+(3) 本計測: バイナリ内蔵ガード（--max-load-avg 等。#1264/#1265）を
# 有効化して起動し、実行中は BREACH 監視を並走させる（#1253 と同型）。
# ---------------------------------------------------------------------
self_pid_tree() {
  local root="$1"
  echo "$root"
  local child
  for child in $(pgrep -P "$root" 2>/dev/null); do
    self_pid_tree "$child"
  done
}

is_self_descendant() {
  local pid="$1" root="$2" depth=0 ppid ps_rc
  while [ "$pid" -gt 1 ] && [ "$depth" -lt 64 ]; do
    if [ "$pid" -eq "$root" ]; then
      return 0
    fi
    ppid=$(ps -o ppid= -p "$pid" 2>/dev/null)
    ps_rc=$?
    ppid=$(printf '%s' "$ppid" | tr -d '[:space:]')
    if [ "$ps_rc" -eq 1 ] && [ -z "$ppid" ]; then
      return 2
    fi
    if [ "$ps_rc" -ne 0 ] || [ -z "$ppid" ]; then
      return 3
    fi
    case "$ppid" in
      *[!0-9]*) return 3 ;;
    esac
    pid="$ppid"
    depth=$((depth + 1))
  done
  return 1
}

classify_pid() {
  local name="$1" pid="$2"
  case "$self_pids" in
    *" $pid "*) return 0 ;;
  esac
  is_self_descendant "$pid" "$BENCH_PID"
  case $? in
    0) ;;
    2) vanished_procs="${vanished_procs}${name}:${pid}," ;;
    3)
      enum_error=1
      other_procs="${other_procs}${name}:${pid}:PS_ERROR,"
      ;;
    *)
      other_procs="${other_procs}${name}:${pid},"
      other_count=$((other_count + 1))
      ;;
  esac
}

MONITOR_LOG="${RUN_PREFIX}monitor.log"
: > "$MONITOR_LOG"
MONITOR_POLL_INTERVAL_SECS="${MONITOR_POLL_INTERVAL_SECS:-30}"

OUT_LOG="${RUN_PREFIX}run.log"

# PR #1462 codex-review 指摘: 監視ループはバイナリ起動直後（内側ガード
# `env_guard` のバックオフ待機の最中）から始まり、待機中の負荷も
# 実測中の負荷と同じ扱いで breach=1 を不可逆に記録していた。外側
# ゲート通過後に一時的に負荷が上がり、内側ガードが待機してから
# phase 1／phase 2 の実測自体は排他的に成功した場合でも valid=0 に
# なってしまい、内側ガードの再試行設計が結果に反映されない。
#
# `${BINARY}` は `env_guard` のブロックを `== env_guard(<label>) ==`
# で開始し、判定確定時に `env_guard_result=<pass|fail...>` を書く
# （`env_guard.rs`。`crates/bench-harness` 側の契約。本スクリプトは
# 変更しない）。この 2 種のマーカーのうち OUT_LOG 中で最後に現れた
# 方を見て、現在が「ガード待機中（guard_wait）」か「実測中
# （measuring。直近の env_guard が通過し、次の env_guard ブロックが
# まだ始まっていない区間）」かを判定する。まだ 1 つも
# `env_guard_result=` が現れていなければ guard_wait とみなす
# （fail-closed。実測開始の証拠がない間は実測中とみなさない）。
current_phase_state() {
  local out_log="$1" last_start last_result
  last_start=$(grep -n '^== env_guard(' "$out_log" 2>/dev/null | tail -1 | cut -d: -f1)
  last_result=$(grep -n '^env_guard_result=' "$out_log" 2>/dev/null | tail -1 | cut -d: -f1)
  if [ -z "$last_result" ]; then
    echo "guard_wait"
    return
  fi
  if [ -n "$last_start" ] && [ "$last_start" -gt "$last_result" ]; then
    echo "guard_wait"
  else
    echo "measuring"
  fi
}
"${BINARY}" \
  --max-load-avg=2.0 \
  --gpu-watch=python --gpu-watch=torch --gpu-watch=mlx --gpu-watch=gemm_ --gpu-watch=bench \
  --guard-wait-secs=60 --guard-max-attempts=20 \
  --env-info-out="${RUN_PREFIX}env_info.txt" \
  > "$OUT_LOG" 2>&1 &
BENCH_PID=$!

breach=0
while kill -0 "$BENCH_PID" 2>/dev/null; do
  self_pids=" $(self_pid_tree "$BENCH_PID" | tr '\n' ' ') "
  ts=$(date +"%Y-%m-%dT%H:%M:%S%z")
  load_error=0
  if loads=$(read_load_or_fail); then
    load1=${loads%% *}
  else
    load_error=1
    load1="NA"
  fi
  other_procs=""
  other_count=0
  vanished_procs=""
  enum_error=0
  for name in cargo rustc python3; do
    if list=$(pgrep_or_fail -x "$name"); then
      for pid in $list; do
        classify_pid "$name" "$pid"
      done
    else
      enum_error=1
    fi
  done
  if list=$(pgrep_or_fail -f '(^|/)gemm_transpose_route_ab_bench([[:space:]]|$)'); then
    for pid in $list; do
      classify_pid gemm_transpose_route_ab_bench "$pid"
    done
  else
    enum_error=1
  fi
  load_ok=$(awk -v l="$load1" -v t="$GATE_THRESHOLD" -v le="$load_error" 'BEGIN{print (le==0 && l<t)?1:0}')
  # PR #1462 codex-review 指摘: `breach` は実測区間（phase_state=
  # measuring）で検出された逸脱のみで不可逆に立てる。ガード待機中
  # （guard_wait）の逸脱は監視ログには記録する（可視性のため）が
  # `valid` の判定へは影響させない——内側ガード自身がバックオフで
  # 再試行し、実測を開始できる状態まで待つ設計だから。
  phase_state=$(current_phase_state "$OUT_LOG")
  if [ "$enum_error" -ne 0 ] || [ "$load_error" -ne 0 ]; then
    [ "$phase_state" = "measuring" ] && breach=1
    echo "$ts load1=$load1 load_error=$load_error other_count=$other_count other_procs=[$other_procs] vanished=[$vanished_procs] enum_error=$enum_error phase=$phase_state UNDETERMINED" >> "$MONITOR_LOG"
  elif [ "$load_ok" != "1" ] || [ "$other_count" -gt 0 ]; then
    [ "$phase_state" = "measuring" ] && breach=1
    echo "$ts load1=$load1 load_error=0 other_count=$other_count other_procs=[$other_procs] vanished=[$vanished_procs] enum_error=0 phase=$phase_state BREACH" >> "$MONITOR_LOG"
  else
    echo "$ts load1=$load1 load_error=0 other_count=$other_count other_procs=[$other_procs] vanished=[$vanished_procs] enum_error=0 phase=$phase_state ok" >> "$MONITOR_LOG"
  fi
  sleep "$MONITOR_POLL_INTERVAL_SECS"
done

wait "$BENCH_PID"
RC=$?
sed -E -i '' "${SANITIZE_SED}" "$OUT_LOG" 2>/dev/null || true

snapshot "after attempt${ATTEMPT}" > "${RUN_PREFIX}uptime_after.txt"
uptime > "${RUN_PREFIX}uptime_after_plain.txt"
pmset -g therm > "${RUN_PREFIX}pmset_therm_after.txt" 2>&1

valid=0
if [ "$RC" -eq 0 ] && [ "$breach" -eq 0 ]; then
  valid=1
fi

echo "run_exit_code=${RC} breach=${breach} valid=${valid}" >> "$LOGDIR/1267-orchestrate-stdout.log"
echo "ORCHESTRATOR_RESULT=DONE valid=${valid} attempt=${ATTEMPT}" >> "$LOGDIR/1267-orchestrate-stdout.log"
echo "DONE valid=${valid} exit_code=${RC} breach=${breach}" > "$LOGDIR/1267-DONE_ATTEMPT${ATTEMPT}"
