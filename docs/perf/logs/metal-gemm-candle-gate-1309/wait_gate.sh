#!/bin/bash
# イシュー #1309: 参考系列開始前の負荷ゲート再判定。
# 事前宣言規則（§3-4）: 1 分 load average < 4.0 を 30 秒間隔で 2 回連続確認。
# 待機は 60 秒開始・1.5 倍ずつ増加・最大 10 回（合計約 30 分上限）。
LOGDIR="$(cd "$(dirname "$0")" && pwd)"
wait_s=60
pass_count=0
attempt=0
gate_result="unmet"
while [ $attempt -lt 10 ]; do
  attempt=$((attempt+1))
  sleep "$wait_s"
  load1=$(uptime | awk -F'load averages: ' '{print $2}' | awk '{print $1}')
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) attempt=$attempt wait=${wait_s}s load1=$load1" >> "$LOGDIR/gate-m4max.log"
  cmp=$(awk -v l="$load1" 'BEGIN{print (l<4.0)?"1":"0"}')
  if [ "$cmp" = "1" ]; then
    pass_count=$((pass_count+1))
  else
    pass_count=0
  fi
  if [ "$pass_count" -ge 2 ]; then
    gate_result="passed"
    break
  fi
  wait_s=$(awk -v w="$wait_s" 'BEGIN{printf "%d", w*1.5}')
done
echo "gate_result=$gate_result attempts=$attempt" >> "$LOGDIR/gate-m4max.log"
echo "$gate_result"
