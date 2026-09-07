#!/usr/bin/env bash
# イシュー #1305: DGX Spark GB10 上でのスイープ一括実行オーケストレーションスクリプト。
# ~/work/sweep-1305/ 上で setsid nohup により切り離して実行する。
#
# 実行時の実測記録（アーカイブ）であり、当時の実行環境の絶対パスを既定値として
# 残すが、再実行環境ではパスが異なりうるため WORKDIR／CARGO_TARGET_DIR／BIN／
# LOGDIR／SWEEP_SCRIPT_DIR を環境変数で上書き可能にし、実行前に BIN・
# SWEEP_SCRIPT_DIR の存在を確認する
# （base AGENTS.md「ハードコード回避」・codex-review 指摘 #1429 対応）。
set -uo pipefail

WORKDIR="${WORKDIR:-$HOME/work/sweep-1305}"
cd "$WORKDIR"
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$HOME/work/target-oss-1305}"
BIN="${BIN:-$CARGO_TARGET_DIR/release/oss-gemm-compare}"
LOGDIR="${LOGDIR:-$WORKDIR/out}"
SWEEP_SCRIPT_DIR="${SWEEP_SCRIPT_DIR:-$WORKDIR/sweep-scripts}"

if [ ! -x "$BIN" ]; then
  echo "エラー: BIN='$BIN' が存在しないか実行可能ではない。CARGO_TARGET_DIR または BIN を指定すること" >&2
  exit 1
fi
if [ ! -f "$SWEEP_SCRIPT_DIR/run_sweep.sh" ]; then
  echo "エラー: SWEEP_SCRIPT_DIR='$SWEEP_SCRIPT_DIR' に run_sweep.sh が見つからない。WORKDIR または SWEEP_SCRIPT_DIR を指定すること" >&2
  exit 1
fi

mkdir -p "$LOGDIR"

# 実行中 load average の推移を記録するバックグラウンドポーラー
( while :; do date -u +"%Y-%m-%dT%H:%M:%SZ" >> "$LOGDIR/uptime-dgx.log"; uptime >> "$LOGDIR/uptime-dgx.log"; sleep 30; done ) &
POLLER_PID=$!

# 専有状態ゲート: 1 分平均が 2.0 未満で 2 回連続を確認するまで最大 30 分待機
{
  echo "=== gate start $(date -u +%FT%TZ) ==="
  attempts=0
  consec=0
  while [ "$attempts" -lt 60 ]; do
    load1=$(uptime | sed -E 's/.*load average: ([0-9.]+).*/\1/')
    echo "attempt=$attempts load1=$load1"
    ok=$(awk -v l="$load1" 'BEGIN{print (l<2.0)?1:0}')
    if [ "$ok" = "1" ]; then
      consec=$((consec+1))
    else
      consec=0
    fi
    if [ "$consec" -ge 2 ]; then
      echo "gate PASSED at attempt=$attempts"
      break
    fi
    attempts=$((attempts+1))
    sleep 60
  done
  if [ "$consec" -lt 2 ]; then
    echo "gate NOT PASSED after max wait (共有負荷下として続行)"
  fi
  echo "who: $(who)"
  ps -eo pcpu,pid,comm --sort=-pcpu | head -15
  echo "=== gate end $(date -u +%FT%TZ) ==="
} > "$LOGDIR/gate-dgx.log" 2>&1

# 主スイープ: 8 -> T_big(10) -> 全コア(20) -> 1 -> 2 -> 4 -> 12 -> 16 の順
BIN="$BIN" SIZES=1024,2048,4096 THREADS="8 10 20 1 2 4 12 16" RUNS=5 \
  OUT="$LOGDIR/rayon_sweep_dgx.log" \
  bash "$SWEEP_SCRIPT_DIR/run_sweep.sh"

# 補助軸 A: RAYON_NUM_THREADS=10 固定で big pin / little pin / no pin の 3 条件
BIN="$BIN" SIZES=1024,2048,4096 THREADS="10" RUNS=5 TASKSET="5-9,15-19" \
  OUT="$LOGDIR/taskset_big_dgx.log" \
  bash "$SWEEP_SCRIPT_DIR/run_sweep.sh"

BIN="$BIN" SIZES=1024,2048,4096 THREADS="10" RUNS=5 TASKSET="0-4,10-14" \
  OUT="$LOGDIR/taskset_little_dgx.log" \
  bash "$SWEEP_SCRIPT_DIR/run_sweep.sh"

BIN="$BIN" SIZES=1024,2048,4096 THREADS="10" RUNS=5 \
  OUT="$LOGDIR/taskset_nopin_dgx.log" \
  bash "$SWEEP_SCRIPT_DIR/run_sweep.sh"

cat "$LOGDIR/taskset_big_dgx.log" "$LOGDIR/taskset_little_dgx.log" "$LOGDIR/taskset_nopin_dgx.log" > "$LOGDIR/taskset_pin_dgx.log"

# 補助軸 B: 制御形状 N=1920（8/10/12/16/20 いずれでも m/T が 8 の倍数に整列）
BIN="$BIN" SIZES=1920 THREADS="8 10 20" RUNS=5 \
  OUT="$LOGDIR/control_1920_dgx.log" \
  bash "$SWEEP_SCRIPT_DIR/run_sweep.sh"

kill "$POLLER_PID" 2>/dev/null || true
echo "done." > "$LOGDIR/ALL_DONE.marker"
echo "all sweeps complete $(date -u +%FT%TZ)" >> "$LOGDIR/uptime-dgx.log"
