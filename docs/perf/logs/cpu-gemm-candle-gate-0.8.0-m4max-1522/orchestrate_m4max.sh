#!/bin/sh
# Apple M4 Max（本セッションのホスト自身）側オーケストレーション
# （イシュー #1522。派生元は docs/perf/logs/cpu-gemm-candle-gate-0.8.0-1488/
# orchestrate_m4max.sh〈PR #1506 是正版〉。差分は (a) LOG／LABEL 既定の
# 変更、(b) 専有ゲートの opt-out〈GEMM_GATE_LOAD_GATE_MODE=record_only〉の
# 追加、(c) pmset -g therm の before/after 記録の追加のみで、exclusive
# 既定時の専有ゲート本体ロジック〈session-start スタンプ・gate-state・
# 60 秒開始 1.5 倍バックオフ・30 秒固定確認・最大 10 試行・1800 秒上限〉は
# 1488 版と完全同一。1488 側のファイル自体は証跡のため編集しない）。
# 正式系列 `fandhe-ai =0.8.0`（registry 解決）のみを計測する単一系列版
# （詳細は orchestrate_dgx.sh 冒頭コメントと同じ。CPU 計測経路の
# v0.8.0 ↔ origin/main 差分ゼロにつき参考系列は計測しない）。
#
# ルート #1519 のユーザー指示（Metal・M4 Max 側は「現在の環境で測れる値で
# 大丈夫」＝専有ゲートを要件にしない。共有負荷下であることと計測中の
# load average 推移を記録し、判定規則は計測前に事前登録・計測後に緩和
# しない）を受け、兄弟イシュー #1520（`run_ab_readout_metal.sh` の
# `AB_LOAD_GATE_MODE=record_only`）と同型の opt-out 方式を本スクリプトへ
# 導入する。既定 `exclusive` は 1488 版の挙動を一切変えない
# （後方互換。将来また専有環境で再計測する場合はそのまま使える）。
#
# 実行対象ツリーは呼び出し元の cwd
# （scripts/bench/framework-compare）とする（DGX 版と異なり隔離転送は
# 行わず worktree 上でそのまま実行する）。ログ出力先は環境変数 LOG で
# 上書きできる。
set -u
LOG="${LOG:-$(pwd)/../../../docs/perf/logs/cpu-gemm-candle-gate-0.8.0-m4max-1522}"
# 新規の再計測ディレクトリ（§24.10 の設計どおり stamp ごと新しい LOG で実行する）
# を作成し、解決に失敗した場合は空の LOG で `/` 直下へ書き込まないよう即終了する
# （PR #1506 Cursor Bugbot 指摘 PRRT_kwDOTuUCJc6g7bsn 対応）。
mkdir -p "$LOG" || { echo "orchestrate_m4max: cannot create LOG dir '$LOG'" >&2; exit 1; }
LOG=$(cd "$LOG" && pwd) || { echo "orchestrate_m4max: cannot resolve LOG dir" >&2; exit 1; }
if [ -z "$LOG" ] || [ "$LOG" = "/" ]; then
  echo "orchestrate_m4max: refusing to use LOG='$LOG'" >&2
  exit 1
fi

rm -f "$LOG/ALL_DONE_m4max.marker" "$LOG/MEASUREMENT_FAILED_m4max.marker" "$LOG/GATE_NOT_PASSED_m4max.marker"

# --- GEMM_GATE_LOAD_GATE_MODE（イシュー #1522。#1520 と同型の opt-out
#     方式）: `exclusive`（既定。1488 版と完全同一の専有ゲート）または
#     `record_only`（専有ゲートを要件にせず load average を記録するのみ）
#     の 2 値 allowlist。それ以外は何も書き込まず fail-closed で exit 1
#     とする（判定規則を計測後に緩めない設計と対になる、無効値の暗黙
#     フォールバック禁止）。 ---
GEMM_GATE_LOAD_GATE_MODE="${GEMM_GATE_LOAD_GATE_MODE:-exclusive}"
case "$GEMM_GATE_LOAD_GATE_MODE" in
  exclusive|record_only) ;;
  *)
    echo "orchestrate_m4max: GEMM_GATE_LOAD_GATE_MODE must be 'exclusive' or 'record_only' (got: $GEMM_GATE_LOAD_GATE_MODE)" >&2
    exit 1
    ;;
esac


GATE_LOG="$LOG/gate-m4max.log"

if [ "$GEMM_GATE_LOAD_GATE_MODE" = "exclusive" ]; then
# --- 是正（PR #1506 codex-review 指摘対応。スレッド PRRT_kwDOTuUCJc6g6N9L）:
#     試行 1〜3（本ディレクトリの gate-m4max-attempt{1,2}.log・gate-m4max.log
#     が実際の記録）はいずれもこのスクリプト自体は「1 プロセス内で最大 10
#     試行・約 30 分上限」しか強制しておらず、プロセスを丸ごと再起動すれば
#     試行カウンタが 0 から再開できてしまい、事前宣言した「約 30 分上限で
#     不成立なら undetermined として打ち切る」終了条件をプロセス再起動が
#     実質的にバイパスできる欠陥があった。以下はこの欠陥を、プロセス再起動を
#     またいで通用する経過時間上限（`$LOG/session-start-m4max.stamp` に
#     最初の起動時刻を 1 度だけ記録し、以後の起動はそこからの経過時間で
#     判定する）として修正したもの。真に独立した再計測をやり直す場合は、
#     このスタンプファイルごと新しい LOG ディレクトリで実行すること（同一
#     ディレクトリでの暗黙リセットを禁止する設計）。 ---
# --- 追加是正（PR #1506 codex-review・Cursor Bugbot 指摘対応。スレッド
#     PRRT_kwDOTuUCJc6g6tbf〈累積試行数〉・PRRT_kwDOTuUCJc6g6tbc〈待機後の
#     期限再確認〉・PRRT_kwDOTuUCJc6g6vMQ〈確認間隔の短縮禁止〉・
#     PRRT_kwDOTuUCJc6g7Kcp〈状態の fail-closed 読み書き〉・
#     PRRT_kwDOTuUCJc6g7Kcq〈バックオフ状態の永続化〉）:
#     (a) 専有ゲートの進行状態（累積試行数・現在のバックオフ秒・次回サンプル
#         採取可能時刻・連続合格数）をスタンプと同じセッション単位で
#         `$LOG/gate-state-m4max` に永続化し、プロセス再起動をまたいで
#         「最大 10 試行」と「宣言どおりのバックオフ系列」を継続適用する
#         （経過時間上限だけを永続化した版では、再起動ごとに試行数が 0 へ戻り
#         同一セッションで 10 試行を超過できた〈m4max-redo-pr1506/ の記録は
#         3 + 9 = 累積 12 試行〉うえ、バックオフも 60 秒へ戻っていた）。
#     (b) 状態の読み書きは fail-closed とする: スタンプが存在するのに状態
#         ファイルが欠落・空・不正な場合は「状態を復元できない再起動」として
#         試行を一切行わず undetermined で終了する（0 へ戻さない）。保存は
#         一時ファイルへ書いて mv する原子的更新とし、書き戻しの読み取り検証に
#         失敗すれば計測せず終了する。保存は sleep の前に行う（待機中に kill
#         されても試行として数える）。
#     (c) 残り時間が次の待機時間（1 回目合格直後の 30 秒固定確認・不合格時の
#         バックオフ）より短い場合は待機時間を短縮せず、その時点で終了する。
#     (d) 待機後に時刻を再取得し、期限へ到達していればそのサンプルで合否を
#         判定せず終了する。gate ログの elapsed はサンプル採取時点の値。
#     (e) 再起動時、1 回目合格（連続合格数 1）の状態は次回サンプル時刻がまだ
#         到来していない場合に限り引き継ぐ（30 秒固定間隔をそのまま守れる）。
#         到来済みなら 30 秒間隔を守れないため合格数を 0 へ戻し（fail-closed）、
#         復元したバックオフ秒で待機する。 ---
STAMP="$LOG/session-start-m4max.stamp"
STATE="$LOG/gate-state-m4max"
CAP_S=1800
MAX_ATTEMPTS=10

refuse() {
  # 専有ゲートを不成立（undetermined）として終了する。$1 はマーカー本文・
  # $2 は gate ログ本文。
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

# 状態を原子的に保存し、書き戻しを読み取り検証する。失敗時は計測せず終了する。
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
  # 新規セッション: 初期状態（累積 0 試行・バックオフ 60 秒・次回サンプルは
  # 60 秒後・連続合格 0）を先に保存してからスタンプを置く。
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

# 状態の復元（fail-closed: 欠落・空・不正は 0 へ戻さず終了する）。
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
  # 再起動による復元。1 回目合格は次回サンプル時刻が未到来の場合のみ引き継ぐ。
  if [ "$PASS_COUNT" -eq 1 ] && [ "$NOW_EPOCH" -gt "$NEXT_EPOCH" ]; then
    echo "gate restore: first-pass confirmation window (30s) already elapsed; dropping pass_count to 0 (fail-closed) elapsed=${ELAPSED}s $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$GATE_LOG"
    PASS_COUNT=0
    NEXT_EPOCH=$((NOW_EPOCH + BACKOFF_S))
    save_state "$ATTEMPT" "$BACKOFF_S" "$NEXT_EPOCH" "$PASS_COUNT"
  fi
  echo "gate restore: attempts=${ATTEMPT} backoff=${BACKOFF_S}s next_epoch=${NEXT_EPOCH} pass_count=${PASS_COUNT} elapsed=${ELAPSED}s $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$GATE_LOG"
fi

# --- (i) 専有ゲート（計画 §3 規則 4・`docs/perf/logs/
#     metal-gemm-candle-gate-1309/wait_gate.sh` 方式と同一のバックオフ系列:
#     60 秒開始・不合格時のみ backoff_s を 1.5 倍・1 回目合格直後の 2 回目
#     確認だけは宣言どおり 30 秒固定・最大 10 試行。試行数・経過時間・
#     バックオフ・次回サンプル時刻はいずれも session-start スタンプと同一
#     セッションの永続化状態で管理し打ち切る（プロセス内の値ではない）。
#     待機時間は短縮しない（残り時間が足りなければ終了）。閾値は #1309 の
#     4.0 ではなく本イシューの計画どおり 6.0 を使う。 ---
GATE_OK=0
while [ "$ATTEMPT" -lt "$MAX_ATTEMPTS" ]; do
  NOW_EPOCH=$(date -u +%s)
  ELAPSED=$((NOW_EPOCH - SESSION_START))
  REMAINING=$((CAP_S - ELAPSED))
  if [ "$REMAINING" -le 0 ]; then
    echo "gate cap reached mid-loop: elapsed=${ELAPSED}s cap=${CAP_S}s $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$GATE_LOG"
    break
  fi
  # 次回サンプル採取可能時刻までの待機時間（再起動復元時は既に一部経過して
  # いることがある。経過分だけ短くなるのは間隔の短縮ではない）。
  WAIT_S=$((NEXT_EPOCH - NOW_EPOCH))
  if [ "$WAIT_S" -lt 0 ]; then
    WAIT_S=0
  fi
  if [ "$WAIT_S" -gt "$REMAINING" ]; then
    # 待機時間を短縮しない: 宣言した間隔（30 秒固定確認・バックオフ）を守れない
    # 場合は合格とせず終了する。
    echo "gate insufficient remaining time: next wait=${WAIT_S}s > remaining=${REMAINING}s (elapsed=${ELAPSED}s cap=${CAP_S}s); not shortening the interval $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$GATE_LOG"
    break
  fi
  ATTEMPT=$((ATTEMPT + 1))
  save_state "$ATTEMPT" "$BACKOFF_S" "$NEXT_EPOCH" "$PASS_COUNT"
  sleep "$WAIT_S"
  # 待機後に時刻を再取得し、期限へ到達していればこのサンプルで判定しない。
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
    # 1 回目合格: 宣言どおり 30 秒後に 2 回目を確認する（バックオフしない）。
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
else
  # --- record_only（イシュー #1522・ルート #1519）: 専有ゲートを待たず
  #     現在の load average を 1 行記録し、直ちに次の並走プロセス確認へ
  #     進む。session-start スタンプ・gate-state は作成しない
  #     （exclusive 経路の再開・打ち切りロジックと混同させないため）。
  #     判定入力にはしない（計画 §4 規則 4）。 ---
  GATE_OK=1
  RAW_UPTIME="$(uptime 2>/dev/null)" || RAW_UPTIME=""
  LOAD1=$(printf '%s\n' "$RAW_UPTIME" | sed -E 's/.*load averages?: ([0-9.]+)[, ].*/\1/')
  if awk -v l="$LOAD1" 'BEGIN{exit !(l != "" && l == l+0 && l >= 0)}' 2>/dev/null; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) mode=record_only load1=$LOAD1（専有ゲート要件なし。イシュー #1522・ルート #1519） $RAW_UPTIME" >> "$GATE_LOG"
  else
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) mode=record_only load1_invalid=${LOAD1:-<empty>}（専有ゲート要件なし。イシュー #1522・ルート #1519） $RAW_UPTIME" >> "$GATE_LOG"
  fi
fi

# --- (ii) 並走プロセス確認 ---
# バイナリ名の完全一致（pgrep -x）を使う（DGX 側 orchestrate_dgx.sh と同じ
# 理由。`-f` は他セッションの監視ループ自体のコマンドライン文字列に
# 部分文字列として誤反応しうることを DGX 側実行で確認済み）。
{
  echo "=== proc check $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  pgrep -x -l 'bench-fandhe' || true
  pgrep -x -l 'bench-candle' || true
} >> "$LOG/gate-m4max.log"
if pgrep -x 'bench-fandhe' > /dev/null 2>&1 || pgrep -x 'bench-candle' > /dev/null 2>&1; then
  echo "sibling bench process detected; treating gate as not passed" > "$LOG/GATE_NOT_PASSED_m4max.marker"
  exit 0
fi

# --- pmset -g therm（before）記録（イシュー #1522。#1475/#1490 と同じ
#     記録項目。失敗しても計測は止めない）。 ---
pmset -g therm > "$LOG/pmset_therm_before-m4max.txt" 2>&1 || true

# --- (iii) uptime 30 秒ポーラ（バックグラウンド） ---
(
  while true; do
    echo "poll $(date -u +%Y-%m-%dT%H:%M:%SZ) $(uptime)" >> "$LOG/uptime-m4max.log"
    sleep 30
  done
) &
POLLER_PID=$!

fail_measurement() {
  kill "$POLLER_PID" 2>/dev/null || true
  echo "$1 failed $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$LOG/MEASUREMENT_FAILED_m4max.marker"
  exit 1
}

# --- (iv) 正式系列計測 ---
# LABEL は環境変数で上書き可能にする（既定 0.8.0-1522。計画 §4 規則 3
# 〈ワンショット規則〉に従い、性能以外の失敗〈ビルド・manifest 不一致・
# ホスト照合失敗・並走 bench 検出〉での再試行は別ラベル
# `0.8.0-1522-r2` 等を明示的に渡し、失敗した試行の証跡を上書きしない）。
LABEL="${GEMM_GATE_LABEL:-0.8.0-1522}"
echo "formal start $(date -u +%Y-%m-%dT%H:%M:%SZ) $(uptime)" >> "$LOG/gate-m4max.log"
if ! GEMM_GATE_CPU_NODE_TAG=m4max-cpu \
  bash run_gemm_gate_cpu.sh "$LABEL" \
  > "$LOG/run_gemm_gate_cpu-m4max-${LABEL}.log" 2>&1; then
  fail_measurement "formal series"
fi
echo "formal end $(date -u +%Y-%m-%dT%H:%M:%SZ) $(uptime)" >> "$LOG/gate-m4max.log"

kill "$POLLER_PID" 2>/dev/null || true

# --- pmset -g therm（after）記録（イシュー #1522） ---
pmset -g therm > "$LOG/pmset_therm_after-m4max.txt" 2>&1 || true

echo "all done $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$LOG/ALL_DONE_m4max.marker"
