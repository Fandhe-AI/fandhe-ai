#!/bin/bash
# イシュー #1309: 30 秒間隔で uptime を記録するバックグラウンドサンプラー。
# 計測中の負荷推移を証跡として残す（事前宣言した負荷ゲート判定の根拠）。
while true; do
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $(uptime)"
  sleep 30
done
