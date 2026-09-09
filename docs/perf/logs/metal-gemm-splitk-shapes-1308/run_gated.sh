#!/bin/sh
# イシュー #1308: run{1..5}.log を負荷ゲート付きで採取するオーケストレーション。
# 計画 §3.4: 1 分 load average <= 4.0 のときのみ起動。超過時は 60s から開始し
# 1.5 倍ずつ増加する間隔で最大 10 回再試行し、全試行を env_info.txt へ記録する。
set -eu

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
LOG_DIR="$REPO_ROOT/docs/perf/logs/metal-gemm-splitk-shapes-1308"
ENV_INFO="$LOG_DIR/env_info.txt"
RUN_IDX="$1"
MAX_LOAD=4.0
INTERVAL=60

cd "$REPO_ROOT"

attempt=0
while :; do
  attempt=$((attempt + 1))
  LOAD1=$(uptime | sed -E 's/.*load averages?: ([0-9.]+).*/\1/')
  TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  echo "run${RUN_IDX} attempt=${attempt} ts=${TS} load1=${LOAD1}" | tee -a "$ENV_INFO"
  # awk で浮動小数点比較（sh 標準では bc 非搭載環境があるため）
  OK=$(awk -v l="$LOAD1" -v m="$MAX_LOAD" 'BEGIN{print (l<=m)?1:0}')
  if [ "$OK" = "1" ]; then
    break
  fi
  if [ "$attempt" -ge 10 ]; then
    echo "run${RUN_IDX}: 10 回とも load average 超過。負荷下のまま実行する（§3.4 方針）" | tee -a "$ENV_INFO"
    break
  fi
  echo "run${RUN_IDX}: load average ${LOAD1} > ${MAX_LOAD}。${INTERVAL}s 待機して再試行" | tee -a "$ENV_INFO"
  sleep "$INTERVAL"
  INTERVAL=$(awk -v i="$INTERVAL" 'BEGIN{printf "%d", i*1.5}')
done

uptime | tee -a "$ENV_INFO"
cargo run -p fandhe-ai-backend-metal --example gemm_splitk_shapes_bench --release \
  > "$LOG_DIR/run${RUN_IDX}.log" 2>&1
echo "run${RUN_IDX} completed at $(date -u +"%Y-%m-%dT%H:%M:%SZ")" | tee -a "$ENV_INFO"
