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

# --- (i) 専有ゲート（是正版。計画 §3 規則 4・`docs/perf/logs/
#     metal-gemm-candle-gate-1309/wait_gate.sh` 方式と同一のバックオフ系列:
#     60 秒開始・不合格時のみ backoff_s を 1.5 倍・1 回目合格直後の 2 回目
#     確認だけは宣言どおり 30 秒固定・最大 10 試行（合計約 30 分上限）。
#     試行 1 回目（`gate-m4max-attempt1.log` に保存済み）は固定 30 秒間隔・
#     10 試行〈約 5 分〉のみで、計画が事前宣言したバックオフ系列と異なって
#     いたため、正しい系列で再試行する〉。閾値は #1309 の 4.0 ではなく本
#     イシューの計画どおり 6.0 を使う。 ---
GATE_OK=0
PASS_COUNT=0
ATTEMPT=0
WAIT_S=60
BACKOFF_S=60
while [ "$ATTEMPT" -lt 10 ]; do
  ATTEMPT=$((ATTEMPT + 1))
  sleep "$WAIT_S"
  LOAD1=$(uptime | sed -E 's/.*load averages?: ([0-9.]+)[, ].*/\1/')
  OK=$(awk -v l="$LOAD1" 'BEGIN{print (l != "" && l == l+0 && l < 6.0) ? 1 : 0}')
  echo "gate attempt=$ATTEMPT wait=${WAIT_S}s load1=$LOAD1 ok=$OK $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG/gate-m4max.log"
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
  echo "gate not passed after 10 attempts (backoff schedule)" > "$LOG/GATE_NOT_PASSED_m4max.marker"
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
echo "formal start $(date -u +%Y-%m-%dT%H:%M:%SZ) $(uptime)" >> "$LOG/gate-m4max.log"
if ! GEMM_GATE_CPU_NODE_TAG=m4max-cpu \
  bash run_gemm_gate_cpu.sh "0.8.0-1488" \
  > "$LOG/run_gemm_gate_cpu-m4max-0.8.0-1488.log" 2>&1; then
  fail_measurement "formal series"
fi
echo "formal end $(date -u +%Y-%m-%dT%H:%M:%SZ) $(uptime)" >> "$LOG/gate-m4max.log"

kill "$POLLER_PID" 2>/dev/null || true

echo "all done $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$LOG/ALL_DONE_m4max.marker"
