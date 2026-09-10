#!/bin/sh
# DGX Spark GB10 側オーケストレーション（イシュー #1488）。
# 正式系列 `fandhe-ai =0.8.0`（registry 解決）のみを計測する単一系列版
# （#1321/#1481 の off/on・Layer A/B 構成とは異なり、v0.8.0 と origin/main
# の CPU 計測経路〈crates/backend-cpu, facade, autodiff, tensor-core〉に
# 差分が無いことを計画時に確認済みのため参考系列は計測しない。計画
# `docs/perf/logs/cpu-gemm-candle-gate-0.8.0-1488/diff_v0.8.0_*_cpu_path.txt`
# 参照）。
#
# (i) 専有ゲート（1 分 load average < 6.0 を 30 秒間隔で 2 回連続・
#     最大 10 試行。不成立なら GATE_NOT_PASSED.marker を書いて終了。
#     計画 §3 規則 4: 最大 10 試行・約 30 分上限）
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

# --- (i) 専有ゲート ---
GATE_OK=0
CONSEC=0
i=1
while [ "$i" -le 10 ]; do
  LOAD1=$(uptime | sed -E 's/.*load average: ([0-9.]+),.*/\1/')
  OK=$(awk -v l="$LOAD1" 'BEGIN{print (l != "" && l < 6.0) ? 1 : 0}')
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
  echo "gate not passed after 10 tries" > "$LOG/GATE_NOT_PASSED.marker"
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
