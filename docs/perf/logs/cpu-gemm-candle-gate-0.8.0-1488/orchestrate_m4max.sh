#!/bin/sh
# Apple M4 Max（本セッションのホスト自身）側オーケストレーション（イシュー
# #1488）。正式系列 `fandhe-ai =0.8.0`（registry 解決）のみを計測する単一
# 系列版（詳細は orchestrate_dgx.sh 冒頭コメントと同じ。CPU 計測経路の
# v0.8.0 ↔ origin/main 差分ゼロにつき参考系列は計測しない）。
#
# 実行対象ツリーは呼び出し元の cwd
# （scripts/bench/framework-compare）とする（DGX 版と異なり隔離転送は
# 行わず worktree 上でそのまま実行する）。ログ出力先は環境変数 LOG で
# 上書きできる。
set -u
LOG="${LOG:-$(pwd)/../../../docs/perf/logs/cpu-gemm-candle-gate-0.8.0-1488}"
LOG=$(cd "$LOG" && pwd)

rm -f "$LOG/ALL_DONE_m4max.marker" "$LOG/MEASUREMENT_FAILED_m4max.marker" "$LOG/GATE_NOT_PASSED_m4max.marker"

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
#     期限再確認〉・PRRT_kwDOTuUCJc6g6vMQ〈確認間隔の短縮禁止〉）:
#     (a) 試行数もスタンプと同じセッション単位で `$LOG/attempts-m4max.count`
#         に累積保存し、プロセス再起動をまたいで「最大 10 試行」を適用する
#         （経過時間上限だけを永続化した版では、再起動ごとに試行数が 0 へ
#         戻り同一セッションで 10 試行を超過できた。m4max-redo-pr1506/ の
#         記録は 3 + 9 = 累積 12 試行）。カウントは sleep の前に書き込む
#         （待機中に kill されても試行として数える fail-closed 方向）。
#     (b) 残り時間が次の待機時間（1 回目合格直後の 30 秒固定確認・不合格時の
#         バックオフ）より短い場合は待機時間を短縮せず、その時点で終了する。
#         待機時間を残り時間へクランプする実装では、連続 2 回確認の 30 秒
#         間隔が短縮されたり、期限到達後のサンプルで合格判定されうる。
#     (c) 待機後に時刻を再取得し、期限へ到達していればそのサンプルで合否を
#         判定せず終了する。gate ログの elapsed はサンプル採取時点の値。 ---
STAMP="$LOG/session-start-m4max.stamp"
COUNT_FILE="$LOG/attempts-m4max.count"
CAP_S=1800
MAX_ATTEMPTS=10
NOW_EPOCH=$(date -u +%s)
if [ ! -f "$STAMP" ]; then
  echo "$NOW_EPOCH" > "$STAMP"
fi
SESSION_START=$(cat "$STAMP")
ELAPSED=$((NOW_EPOCH - SESSION_START))
if [ "$ELAPSED" -ge "$CAP_S" ]; then
  echo "session elapsed=${ELAPSED}s >= cap=${CAP_S}s at invocation start; refusing further attempts (no silent restart-reset)" \
    > "$LOG/GATE_NOT_PASSED_m4max.marker"
  echo "gate cap exceeded before any attempt: elapsed=${ELAPSED}s cap=${CAP_S}s $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG/gate-m4max.log"
  exit 0
fi
ATTEMPT=0
if [ -f "$COUNT_FILE" ]; then
  ATTEMPT=$(cat "$COUNT_FILE")
  case "$ATTEMPT" in
    ''|*[!0-9]*) ATTEMPT=0 ;;
  esac
fi
if [ "$ATTEMPT" -ge "$MAX_ATTEMPTS" ]; then
  echo "session cumulative attempts=${ATTEMPT} >= max=${MAX_ATTEMPTS} at invocation start; refusing further attempts (no silent restart-reset)" \
    > "$LOG/GATE_NOT_PASSED_m4max.marker"
  echo "gate attempt cap exceeded before any attempt: attempts=${ATTEMPT} max=${MAX_ATTEMPTS} elapsed=${ELAPSED}s $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG/gate-m4max.log"
  exit 0
fi

# --- (i) 専有ゲート（計画 §3 規則 4・`docs/perf/logs/
#     metal-gemm-candle-gate-1309/wait_gate.sh` 方式と同一のバックオフ系列:
#     60 秒開始・不合格時のみ backoff_s を 1.5 倍・1 回目合格直後の 2 回目
#     確認だけは宣言どおり 30 秒固定・最大 10 試行。試行数・経過時間はいずれも
#     session-start スタンプと同一セッションの累積値で打ち切る（プロセス内の
#     値ではない）。待機時間は短縮しない（残り時間が足りなければ終了）。閾値は
#     #1309 の 4.0 ではなく本イシューの計画どおり 6.0 を使う。 ---
GATE_OK=0
PASS_COUNT=0
WAIT_S=60
BACKOFF_S=60
while [ "$ATTEMPT" -lt "$MAX_ATTEMPTS" ]; do
  NOW_EPOCH=$(date -u +%s)
  ELAPSED=$((NOW_EPOCH - SESSION_START))
  REMAINING=$((CAP_S - ELAPSED))
  if [ "$REMAINING" -le 0 ]; then
    echo "gate cap reached mid-loop: elapsed=${ELAPSED}s cap=${CAP_S}s $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG/gate-m4max.log"
    break
  fi
  if [ "$WAIT_S" -gt "$REMAINING" ]; then
    # 待機時間を短縮しない: 宣言した間隔（30 秒固定確認・バックオフ）を守れない
    # 場合は合格とせず終了する。
    echo "gate insufficient remaining time: next wait=${WAIT_S}s > remaining=${REMAINING}s (elapsed=${ELAPSED}s cap=${CAP_S}s); not shortening the interval $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG/gate-m4max.log"
    break
  fi
  ATTEMPT=$((ATTEMPT + 1))
  echo "$ATTEMPT" > "$COUNT_FILE"
  sleep "$WAIT_S"
  # 待機後に時刻を再取得し、期限へ到達していればこのサンプルで判定しない。
  NOW_EPOCH=$(date -u +%s)
  ELAPSED=$((NOW_EPOCH - SESSION_START))
  if [ "$ELAPSED" -ge "$CAP_S" ]; then
    echo "gate cap reached after wait: attempt=$ATTEMPT elapsed=${ELAPSED}s cap=${CAP_S}s; sample not evaluated $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG/gate-m4max.log"
    break
  fi
  LOAD1=$(uptime | sed -E 's/.*load averages?: ([0-9.]+)[, ].*/\1/')
  OK=$(awk -v l="$LOAD1" 'BEGIN{print (l != "" && l == l+0 && l < 6.0) ? 1 : 0}')
  echo "gate attempt=$ATTEMPT (session cumulative) wait=${WAIT_S}s load1=$LOAD1 ok=$OK elapsed=${ELAPSED}s $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG/gate-m4max.log"
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
    WAIT_S=30
  else
    BACKOFF_S=$(awk -v w="$BACKOFF_S" 'BEGIN{printf "%d", w*1.5}')
    WAIT_S=$BACKOFF_S
  fi
done

if [ "$GATE_OK" != "1" ]; then
  echo "gate not passed within elapsed-time cap (${CAP_S}s) / ${MAX_ATTEMPTS} attempts (session cumulative=${ATTEMPT})" > "$LOG/GATE_NOT_PASSED_m4max.marker"
  exit 0
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
# LABEL は環境変数で上書き可能にする（既定 0.8.0-1488。PR #1506 の独立
# 再計測は別ラベル `0.8.0-1488-redo1506` を渡し、元の試行 1〜3 が残した
# `results-m4max-cpu-gemm-gate-0.8.0-1488.jsonl` 等の証跡を上書きしない）。
LABEL="${GEMM_GATE_LABEL:-0.8.0-1488}"
echo "formal start $(date -u +%Y-%m-%dT%H:%M:%SZ) $(uptime)" >> "$LOG/gate-m4max.log"
if ! GEMM_GATE_CPU_NODE_TAG=m4max-cpu \
  bash run_gemm_gate_cpu.sh "$LABEL" \
  > "$LOG/run_gemm_gate_cpu-m4max-${LABEL}.log" 2>&1; then
  fail_measurement "formal series"
fi
echo "formal end $(date -u +%Y-%m-%dT%H:%M:%SZ) $(uptime)" >> "$LOG/gate-m4max.log"

kill "$POLLER_PID" 2>/dev/null || true

echo "all done $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$LOG/ALL_DONE_m4max.marker"
