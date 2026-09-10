#!/bin/sh
# DGX Spark GB10 側オーケストレーション（イシュー #1488）。
# 正式系列 `fandhe-ai =0.8.0`（registry 解決）のみを計測する単一系列版
# （#1321/#1481 の off/on・Layer A/B 構成とは異なり、v0.8.0 と origin/main
# の CPU 計測経路〈crates/backend-cpu, facade, autodiff, tensor-core〉に
# 差分が無いことを計画時に確認済みのため参考系列は計測しない。計画
# `docs/perf/logs/cpu-gemm-candle-gate-0.8.0-1488/diff_v0.8.0_*_cpu_path.txt`
# 参照）。
#
# (i) 専有ゲート（1 分 load average < 6.0 を 2 回連続。60 秒開始・不合格時のみ
#     1.5 倍バックオフ・1 回目合格直後の 2 回目確認は 30 秒固定・最大 10 試行・
#     約 30 分上限。試行数・バックオフ・期限はセッション状態として永続化し
#     プロセス再起動をまたいで適用する。不成立なら GATE_NOT_PASSED.marker を
#     書いて終了。計画 §3 規則 4・§24.1。PR #1506 で m4max 版と同一化）
# (ii) 並走プロセス確認（#1489/#1490 等の bench 系プロセス）
# (iii) uptime 30 秒ポーラをバックグラウンド起動
# (iv) 正式系列計測（GEMM_GATE_PATCH_FACADE_PATH 未指定＝registry 解決）
# (v) ALL_DONE.marker（計測が成功した場合のみ）
#
# 対象ツリー・ログ出力先は環境変数 TREE・LOG で上書きできる
# （既定はノード上の隔離ディレクトリ $HOME/work/fc-1488/tree・
# $HOME/work/fc-1488/logs。個人ホームパス・UUID を含まない）。
set -u
LOG="${LOG:-$HOME/work/fc-1488/logs}"
TREE="${TREE:-$HOME/work/fc-1488/tree}"
mkdir -p "$LOG"

rm -f "$LOG/ALL_DONE.marker" "$LOG/MEASUREMENT_FAILED.marker" "$LOG/GATE_NOT_PASSED.marker"
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"

# --- (i) 専有ゲート（PR #1506 codex-review 指摘 PRRT_kwDOTuUCJc6g7ZEx 対応:
#     本イシューの DGX 記録〈gate-dgx.log〉は固定 30 秒間隔の旧ループで取得し
#     try=1/2 の 2 回連続合格で通過したため記録済みの値は有効だが、再現用
#     スクリプトとしては §24.1 が両実機に宣言したバックオフ系列〈60 秒開始・
#     不合格時のみ 1.5 倍・1 回目合格直後の 2 回目確認は 30 秒固定・最大 10
#     試行・約 30 分上限〉と異なっていた。以下は orchestrate_m4max.sh と同一の
#     セッション状態永続化つきゲート〈stamp・gate-state の fail-closed 読み書き・
#     待機時間の非短縮・待機後の期限再確認〉へ揃えたもの。 ---
STAMP="$LOG/session-start-dgx.stamp"
STATE="$LOG/gate-state-dgx"
CAP_S=1800
MAX_ATTEMPTS=10
GATE_LOG="$LOG/gate-dgx.log"

refuse() {
  # 専有ゲートを不成立（undetermined）として終了する。$1 はマーカー本文・
  # $2 は gate ログ本文。
  echo "$1" > "$LOG/GATE_NOT_PASSED.marker"
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
  echo "gate not passed within elapsed-time cap (${CAP_S}s) / ${MAX_ATTEMPTS} attempts (session cumulative=${ATTEMPT})" > "$LOG/GATE_NOT_PASSED.marker"
  exit 0
fi

# --- (ii) 並走プロセス確認 ---
# バイナリ名の完全一致（pgrep -x）を使う。`-f`（コマンドライン全体の部分
# 一致）は、他セッションの `pgrep -f "bench-(fandhe|candle|burn)"` のような
# 監視ループ自体のコマンドライン文字列に「bench-fandhe」「bench-candle」が
# 部分文字列として含まれるだけで誤検出する（実機で確認済み: 13 日前からの
# 無関係な残留 `bash -c 'while pgrep ... "bench-(fandhe|candle|burn)" ...'`
# ループに `-f` が誤反応した）。`-x` は実行中バイナリの完全一致のみを見る
# ため、この種の誤検出を避けられる。
{
  echo "=== proc check $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  pgrep -x -l 'bench-fandhe' || true
  pgrep -x -l 'bench-candle' || true
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>&1 || echo "nvidia-smi unavailable"
} >> "$LOG/gate-dgx.log"
if pgrep -x 'bench-fandhe' > /dev/null 2>&1 || pgrep -x 'bench-candle' > /dev/null 2>&1; then
  echo "sibling bench process detected; treating gate as not passed" > "$LOG/GATE_NOT_PASSED.marker"
  exit 0
fi

# --- (iii) uptime 30 秒ポーラ（バックグラウンド） ---
(
  while true; do
    echo "poll $(date -u +%Y-%m-%dT%H:%M:%SZ) $(uptime)" >> "$LOG/uptime-dgx.log"
    sleep 30
  done
) &
POLLER_PID=$!

fail_measurement() {
  kill "$POLLER_PID" 2>/dev/null || true
  echo "$1 failed $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$LOG/MEASUREMENT_FAILED.marker"
  exit 1
}

# --- (iv) 正式系列計測 ---
cd "$TREE/scripts/bench/framework-compare" || fail_measurement "cd tree"

echo "formal start $(date -u +%Y-%m-%dT%H:%M:%SZ) $(uptime)" >> "$LOG/gate-dgx.log"
if ! GEMM_GATE_CPU_NODE_TAG=dgx-cpu \
  bash run_gemm_gate_cpu.sh "0.8.0-1488" \
  > "$LOG/run_gemm_gate_cpu-dgx-0.8.0-1488.log" 2>&1; then
  fail_measurement "formal series"
fi
echo "formal end $(date -u +%Y-%m-%dT%H:%M:%SZ) $(uptime)" >> "$LOG/gate-dgx.log"

kill "$POLLER_PID" 2>/dev/null || true

echo "all done $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$LOG/ALL_DONE.marker"
