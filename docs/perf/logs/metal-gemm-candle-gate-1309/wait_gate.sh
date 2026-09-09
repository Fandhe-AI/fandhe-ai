#!/bin/bash
# イシュー #1309: 参考系列開始前の負荷ゲート再判定。
# 事前宣言規則（§3-4）: 1 分 load average < 4.0 を 30 秒間隔で 2 回連続確認。
# 待機は 60 秒開始・1.5 倍ずつ増加・最大 10 回（合計約 30 分上限）。
#
# 是正（PR #1467 Cursor Bugbot 指摘）: バックオフ用の待機時間（60s, 90s, 135s, ...）
# を「合格後の次サンプルまでの間隔」としてそのまま使ってしまうと、宣言した
# 「30 秒間隔で連続 2 回」を満たさないまま合格判定してしまう（1 回合格した後
# 数分空けて 2 回目を取り、それを「連続」扱いしてしまう）。
# よって: 1 回目合格の直後は 30 秒固定で 2 回目を確認する（宣言どおりの間隔）。
# 不合格時のみバックオフ（60s 開始・1.5 倍）で待って再試行する。
set -u
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
  if [ "$pass_count" -eq 1 ]; then
    # 1 回目合格: 宣言どおり 30 秒後に 2 回目を確認する（バックオフしない）。
    wait_s=30
  else
    # 不合格（0 回目に戻った）: バックオフして再試行する。
    wait_s=$(awk -v w="$wait_s" 'BEGIN{printf "%d", w*1.5}')
  fi
done
echo "gate_result=$gate_result attempts=$attempt" >> "$LOGDIR/gate-m4max.log"
echo "$gate_result"
