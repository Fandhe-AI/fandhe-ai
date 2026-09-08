#!/usr/bin/env bash
# イシュー #1255: 「排他環境（load average < 2・他 GPU プロセスなし）」の
# 確保を待たず、逆に「負荷環境（load average の目安 3 以上）」で
# `--phase1-only` モード（#1251）を 3 回実行し、サイズ別 spread・単発
# スパイクの分布を記録するオーケストレーター。
#
# 位置づけ: 親 #1249 は排他環境（#1253）・負荷環境（本イシュー）の双方で
# 同一プロトコル（`--phase1-only`）を実行し spread 分布を比較する計画
# だったが、#1253（PR #1459）は排他ゲートに 2 回とも到達できず
# valid_runs=0 のまま終わった（`docs/perf/metal-gemm-transpose-tiled.md`
# §5.6）。本スクリプトは負荷環境側の実データを独立に取得する
# （既存の `orchestrate.sh`／`wait_gate.sh` は「load average < 2」への
# 収束を待つ排他ゲート専用のため、逆向きの条件には流用しない。#1261 と
# 同じ理由で `1255-` 接頭辞の独立スクリプトとする。計画 §3.1）。
#
# ゲート契約（事前宣言。計測後に変更しない。PR #1448 の教訓）:
# - 待機（有界・逆向きゲート）: 30 秒間隔で load average(1 分) を記録し、
#   load1 >= HIGH_LOAD_THRESHOLD(3.0) が 2 回連続で成立したら run ループへ
#   進む。MAX_WAIT_SECS(600) 到達時は成立していなくても run ループへ進む
#   （負荷は他セッション由来であり合成しない。合成負荷は計測対象の意味を
#   変えるため不採用。計画 §3.2）。
# - run 分類: 各 run の実行中 30 秒間隔サンプルの load1 の中央値で分類し
#   `1255-phase1_run${n}_monitor.log`／`1255-DONE` に記録する。
#     high: 中央値 >= 3.0（AC1 の 3 回にカウントする「負荷環境成立」）
#     mid:  2.0 <= 中央値 < 3.0（記録のみ）
#     low:  中央値 < 2.0（記録のみ。排他環境成立を意味しない——他プロセス
#           条件を検査していないため）
#     undetermined: `read_load_or_fail` 失敗が 1 回でもあれば判定不能
# - 追加実行の上限: high が 3 件未達なら MAX_RUNS(5) まで追加実行する。
#   5 run で 3 件に満たなくても、得られた件数のまま打ち切り「AC1 未達
#   （負荷環境 N/3）」として記録する（fail-closed。合成負荷での水増しは
#   しない）。
# - 実行フラグは `--phase1-only --env-info-out=<path>` のみ。
#   `--max-load-avg` は指定しない（gated モードは負荷下でバックオフ待機
#   → `verdict=undetermined` 終了となり目的と逆）。`--gpu-timestamps` も
#   指定しない（対照ワークロードが計装版に置換され #1187／#1253 との
#   比較可能性が崩れる）。`--min-warmup-secs` も未指定（既定 3 秒・
#   ROUNDS 10・COOLDOWN 8 秒のまま）。
set -uo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DERIVED_WORKDIR="$(cd "$SELF_DIR/../../../.." && pwd)"
WORKDIR="${WORKDIR:-$DERIVED_WORKDIR}"
# LOGDIR は cd "$WORKDIR" より前に呼び出し元基準の絶対パスへ正規化する
# （相対指定時に 1255-aggregate.py〈呼び出し元 cwd 基準〉と別ディレクトリ
# を参照しないため。#1253 orchestrate.sh 六度目の是正と同型）。
LOGDIR="${LOGDIR:-$SELF_DIR}"
mkdir -p "$LOGDIR" || { echo "エラー: LOGDIR='$LOGDIR' を作成できない" >&2; exit 1; }
LOGDIR="$(cd "$LOGDIR" && pwd)" || { echo "エラー: LOGDIR の正規化に失敗した" >&2; exit 1; }

# 同じ実行の記録が既に存在する場合は上書きせず中止する（fail-closed。
# #1253 orchestrate.sh の ATTEMPT 別排他と同型。本スクリプトは 1 回限り
# の実行を想定するため接頭辞は固定 `1255-` のみ）。
for existing in "$LOGDIR"/1255-phase1_run*.log "$LOGDIR/1255-DONE"; do
  if [ -e "$existing" ]; then
    echo "エラー: #1255 の記録 '$existing' が既に存在する。旧記録を退避してから再実行すること" >&2
    exit 1
  fi
done

# shellcheck source=gate_common.sh
source "$SELF_DIR/gate_common.sh"

HIGH_LOAD_THRESHOLD="${HIGH_LOAD_THRESHOLD:-3.0}"
MID_LOAD_THRESHOLD="${MID_LOAD_THRESHOLD:-2.0}"
POLL_INTERVAL_SECS="${POLL_INTERVAL_SECS:-30}"
MAX_WAIT_SECS="${MAX_WAIT_SECS:-600}"
MAX_RUNS="${MAX_RUNS:-5}"
TARGET_HIGH_RUNS="${TARGET_HIGH_RUNS:-3}"

WAIT_LOG="$LOGDIR/1255-wait_load.log"
: > "$WAIT_LOG"

cd "$WORKDIR" || { echo "エラー: WORKDIR='$WORKDIR' への移動に失敗した" >&2; exit 1; }

echo "$(date +"%Y-%m-%dT%H:%M:%S%z") orchestrator(#1255) start HIGH_LOAD_THRESHOLD=${HIGH_LOAD_THRESHOLD} MID_LOAD_THRESHOLD=${MID_LOAD_THRESHOLD} POLL_INTERVAL_SECS=${POLL_INTERVAL_SECS} MAX_WAIT_SECS=${MAX_WAIT_SECS} MAX_RUNS=${MAX_RUNS} TARGET_HIGH_RUNS=${TARGET_HIGH_RUNS}" >> "$LOGDIR/1255-orchestrate.log"

# ---------------------------------------------------------------------
# 待機フェーズ（逆向き・有界ゲート）: load1 >= HIGH_LOAD_THRESHOLD が
# 2 回連続で成立するまで、または MAX_WAIT_SECS に達するまで待つ。
# 負荷は他セッション由来のみを対象とし、合成負荷は生成しない
# （計画 §3.2「合成負荷は不採用」）。
# ---------------------------------------------------------------------
start_ts=$(date +%s)
consecutive_high=0
wait_result="TIMEOUT"
while true; do
  now_ts=$(date +%s)
  elapsed=$((now_ts - start_ts))
  if loads=$(read_load_or_fail); then
    load1=${loads%% *}
    load_error=0
  else
    load1="NA"
    load_error=1
  fi
  high_ok=0
  if [ "$load_error" -eq 0 ]; then
    high_ok=$(awk -v l="$load1" -v t="$HIGH_LOAD_THRESHOLD" 'BEGIN{print (l>=t)?1:0}')
  fi
  if [ "$high_ok" = "1" ]; then
    consecutive_high=$((consecutive_high + 1))
  else
    consecutive_high=0
  fi
  echo "$(date +"%Y-%m-%dT%H:%M:%S%z") elapsed=${elapsed}s load1=${load1} load_error=${load_error} high_ok=${high_ok} consecutive_high=${consecutive_high}" >> "$WAIT_LOG"
  if [ "$consecutive_high" -ge 2 ]; then
    wait_result="HIGH_CONFIRMED"
    break
  fi
  if [ "$elapsed" -ge "$MAX_WAIT_SECS" ]; then
    wait_result="TIMEOUT"
    break
  fi
  sleep "$POLL_INTERVAL_SECS"
done
echo "$(date +"%Y-%m-%dT%H:%M:%S%z") wait phase result=${wait_result} elapsed=$(( $(date +%s) - start_ts ))s" >> "$LOGDIR/1255-orchestrate.log"
# TIMEOUT でも run ループへは進む（MAX_WAIT_SECS 到達時は成立していなくても
# 開始する。計画 §3.2）。ここでは中止しない。

# median_of: 標準入力から 1 行 1 数値を読み中央値を出力する（bc 不使用・
# awk のみ。空入力の場合は空文字を出力する）。run 分類（§3.2）でのみ使う。
median_of() {
  awk '
    /^[0-9.]+$/ { vals[n++] = $0 }
    END {
      if (n == 0) { exit 0 }
      # 単純挿入ソート（サンプル数は 1 run あたり高々数十件のため十分）
      for (i = 0; i < n; i++) { sorted[i] = vals[i] }
      for (i = 1; i < n; i++) {
        key = sorted[i]; j = i - 1
        while (j >= 0 && sorted[j] > key) { sorted[j+1] = sorted[j]; j-- }
        sorted[j+1] = key
      }
      if (n % 2 == 1) { print sorted[int(n/2)] }
      else { print (sorted[n/2 - 1] + sorted[n/2]) / 2 }
    }
  '
}

# ---------------------------------------------------------------------
# run ループ: high 分類が TARGET_HIGH_RUNS(3) 件に達するまで、または
# MAX_RUNS(5) に達するまで phase1-only を実行する。
# ---------------------------------------------------------------------
high_runs=0
valid_runs=0
runs_executed=0
for n in $(seq 1 "$MAX_RUNS"); do
  if [ "$high_runs" -ge "$TARGET_HIGH_RUNS" ]; then
    break
  fi
  runs_executed=$n

  {
    echo "=== run${n} pre-check $(date +"%Y-%m-%dT%H:%M:%S%z") ==="
    uptime
  } > "$LOGDIR/1255-uptime_before_run${n}.txt"
  pmset -g therm > "$LOGDIR/1255-pmset_therm_before_run${n}.txt" 2>&1 || true

  MONITOR_LOG="$LOGDIR/1255-phase1_run${n}_monitor.log"
  : > "$MONITOR_LOG"

  cargo run -p fandhe-ai-backend-metal --example gemm_transpose_route_ab_bench --release \
    --features internal-diagnostics \
    -- --phase1-only "--env-info-out=$LOGDIR/1255-env_info_run${n}.txt" \
    > "$LOGDIR/1255-phase1_run${n}.log" 2>&1 &
  BENCH_PID=$!

  # 実行中の 30 秒間隔サンプリング（ゲートではなく分類用の記録のみ。
  # §3.2 の run 分類はこの monitor log の load1 中央値で行う）。
  while kill -0 "$BENCH_PID" 2>/dev/null; do
    ts=$(date +"%Y-%m-%dT%H:%M:%S%z")
    if loads=$(read_load_or_fail); then
      load1=${loads%% *}
      load_error=0
    else
      load1="NA"
      load_error=1
    fi
    echo "$ts load1=$load1 load_error=$load_error" >> "$MONITOR_LOG"
    sleep "$POLL_INTERVAL_SECS"
  done

  wait "$BENCH_PID"
  RC=$?

  {
    echo "=== run${n} post-check $(date +"%Y-%m-%dT%H:%M:%S%z") rc=${RC} ==="
    uptime
  } > "$LOGDIR/1255-uptime_after_run${n}.txt"
  pmset -g therm > "$LOGDIR/1255-pmset_therm_after_run${n}.txt" 2>&1 || true

  # run 有効条件（計画 §3.2）: 終了コード 0・`phase1_round_stats` 行が
  # 5 サイズ分揃う・`verdict=not_evaluated` 行あり・`mode=phase1_only` 行あり。
  round_stats_count=$(grep -c '^phase1_round_stats size=' "$LOGDIR/1255-phase1_run${n}.log" || true)
  has_verdict=$(grep -c '^verdict=not_evaluated' "$LOGDIR/1255-phase1_run${n}.log" || true)
  has_mode=$(grep -c '^mode=phase1_only' "$LOGDIR/1255-phase1_run${n}.log" || true)
  run_valid=0
  if [ "$RC" -eq 0 ] && [ "$round_stats_count" -eq 5 ] && [ "$has_verdict" -ge 1 ] && [ "$has_mode" -ge 1 ]; then
    run_valid=1
    valid_runs=$((valid_runs + 1))
  fi

  # run 分類（§3.2）: monitor log の load1 中央値。1 サンプルでも
  # load_error=1 があれば undetermined。
  load_error_count=$(grep -c 'load_error=1' "$MONITOR_LOG" || true)
  sample_count=$(grep -c '^' "$MONITOR_LOG" || true)
  if [ "$load_error_count" -gt 0 ] || [ "$sample_count" -eq 0 ]; then
    load_class="undetermined"
    median_load1="NA"
  else
    median_load1=$(grep -oE 'load1=[0-9.]+' "$MONITOR_LOG" | sed 's/load1=//' | median_of)
    if [ -z "$median_load1" ]; then
      load_class="undetermined"
      median_load1="NA"
    else
      load_class=$(awk -v m="$median_load1" -v hi="$HIGH_LOAD_THRESHOLD" -v mid="$MID_LOAD_THRESHOLD" \
        'BEGIN{ if (m>=hi) print "high"; else if (m>=mid) print "mid"; else print "low" }')
    fi
  fi

  if [ "$run_valid" -eq 1 ] && [ "$load_class" = "high" ]; then
    high_runs=$((high_runs + 1))
  fi

  echo "run${n} rc=${RC} run_valid=${run_valid} round_stats_count=${round_stats_count} load_class=${load_class} median_load1=${median_load1} sample_count=${sample_count} load_error_count=${load_error_count}" >> "$LOGDIR/1255-orchestrate.log"
  echo "RUN_CLASSIFICATION run=${n} run_valid=${run_valid} load_class=${load_class} median_load1=${median_load1}" >> "$MONITOR_LOG"
done

echo "$(date +"%Y-%m-%dT%H:%M:%S%z") orchestrator(#1255) done runs_executed=${runs_executed} valid_runs=${valid_runs} high_runs=${high_runs}/${TARGET_HIGH_RUNS}" >> "$LOGDIR/1255-orchestrate.log"
echo "DONE valid_runs=${valid_runs} high_runs=${high_runs} runs_executed=${runs_executed} target_high_runs=${TARGET_HIGH_RUNS} max_runs=${MAX_RUNS} wait_result=${wait_result}" > "$LOGDIR/1255-DONE"
