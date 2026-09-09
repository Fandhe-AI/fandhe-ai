#!/bin/sh
# DGX Spark GB10 側オーケストレーション（イシュー #1481）。
# (i) 専有ゲート（1 分 load average < 6.0 を 30 秒間隔で 2 回連続・
#     最大 30 試行。不成立なら GATE_NOT_PASSED.marker を書いて終了）
# (ii) uptime 30 秒ポーラをバックグラウンド起動
# (iii) Layer A（off → on の順。off ツリーの scripts/bench/framework-compare
#      から実行し、on 腕は GEMM_GATE_PATCH_FACADE_PATH で on ツリーの
#      crates/facade を指す）
# (iv) Layer B（run_layerB_dgx.sh off → on）
# (v) ALL_DONE.marker（Layer A/B の 4 呼び出しすべてが成功した場合のみ）
#
# PR #1501 codex-review P3 是正: 当初は `set -u`（errexit なし。`/bin/sh`
# 実行のため dash 等 POSIX sh には `pipefail` が無く `set -e` も
# バックグラウンドポーラの後始末と相性が悪いため、`errexit` には頼らず
# 各呼び出しの終了コードを明示的に確認する方式を採る）のまま Layer A/B
# の各呼び出しの終了コードを確認していなかったため、計測コマンドが
# 失敗しても後続ステップがそのまま実行され続け、最終的に
# `ALL_DONE.marker` が生成されて（終了コード 0 で）不完全な計測を
# 完了として扱ってしまう欠陥があった。各呼び出しを `if ! ...; then` で
# 明示的に確認し、失敗時はバックグラウンドポーラを停止したうえで
# `MEASUREMENT_FAILED.marker` を書いて `ALL_DONE.marker` を生成せずに
# 非 0 で終了するよう是正した。
set -u
LOG="$HOME/work/fc-1481-logs"
mkdir -p "$LOG"
OFF="$HOME/work/fc-1481/off"
ON="$HOME/work/fc-1481/on"
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"

# --- (i) 専有ゲート ---
GATE_OK=0
CONSEC=0
i=1
while [ "$i" -le 30 ]; do
  LOAD1=$(uptime | sed -E 's/.*load average: ([0-9.]+),.*/\1/')
  OK=$(awk -v l="$LOAD1" 'BEGIN{print (l < 6.0) ? 1 : 0}')
  echo "gate try=$i load1=$LOAD1 ok=$OK $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG/gate-dgx.log"
  if [ "$OK" = "1" ]; then
    CONSEC=$((CONSEC + 1))
  else
    CONSEC=0
  fi
  if [ "$CONSEC" -ge 2 ]; then
    GATE_OK=1
    break
  fi
  i=$((i + 1))
  sleep 30
done

if [ "$GATE_OK" != "1" ]; then
  echo "gate not passed after 30 tries" > "$LOG/GATE_NOT_PASSED.marker"
  exit 0
fi

# --- (ii) uptime 30 秒ポーラ（バックグラウンド） ---
(
  while true; do
    echo "poll $(date -u +%Y-%m-%dT%H:%M:%SZ) $(uptime)" >> "$LOG/uptime-dgx.log"
    sleep 30
  done
) &
POLLER_PID=$!

# 失敗確認ヘルパ: 呼び出しが非 0 終了した場合にバックグラウンドポーラを
# 停止したうえで MEASUREMENT_FAILED.marker を書いて非 0 で終了する
# （ALL_DONE.marker は生成しない。PR #1501 codex-review P3 是正の核心）。
fail_measurement() {
  kill "$POLLER_PID" 2>/dev/null || true
  echo "$1 failed $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$LOG/MEASUREMENT_FAILED.marker"
  exit 1
}

# --- (iii) Layer A ---
cd "$OFF/scripts/bench/framework-compare"
SHA=$(cat "$OFF/.rev-stamp" 2>/dev/null || echo unknown)

echo "layerA off start $(date -u +%Y-%m-%dT%H:%M:%SZ) $(uptime)" >> "$LOG/gate-dgx.log"
if ! GEMM_GATE_CPU_NODE_TAG=dgx-cpu GEMM_GATE_PATCH_FACADE_PATH="$OFF/crates/facade" \
  bash run_gemm_gate_cpu.sh "head-${SHA}-1481-pzero-off" \
  > "$LOG/run_gemm_gate_cpu-dgx-off.log" 2>&1; then
  fail_measurement "layerA off"
fi
echo "layerA off end $(date -u +%Y-%m-%dT%H:%M:%SZ) $(uptime)" >> "$LOG/gate-dgx.log"

echo "layerA on start $(date -u +%Y-%m-%dT%H:%M:%SZ) $(uptime)" >> "$LOG/gate-dgx.log"
if ! GEMM_GATE_CPU_NODE_TAG=dgx-cpu GEMM_GATE_PATCH_FACADE_PATH="$ON/crates/facade" \
  bash run_gemm_gate_cpu.sh "head-${SHA}-1481-pzero-on" \
  > "$LOG/run_gemm_gate_cpu-dgx-on.log" 2>&1; then
  fail_measurement "layerA on"
fi
echo "layerA on end $(date -u +%Y-%m-%dT%H:%M:%SZ) $(uptime)" >> "$LOG/gate-dgx.log"

# --- (iv) Layer B ---
if ! sh "$OFF/docs/perf/logs/cpu-matmul-fixed-cost-remeasure-1481/run_layerB_dgx.sh" off \
  > "$LOG/layerB-dgx-off-driver.log" 2>&1; then
  fail_measurement "layerB off"
fi
if ! sh "$OFF/docs/perf/logs/cpu-matmul-fixed-cost-remeasure-1481/run_layerB_dgx.sh" on \
  > "$LOG/layerB-dgx-on-driver.log" 2>&1; then
  fail_measurement "layerB on"
fi

kill "$POLLER_PID" 2>/dev/null || true

echo "all done $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$LOG/ALL_DONE.marker"
