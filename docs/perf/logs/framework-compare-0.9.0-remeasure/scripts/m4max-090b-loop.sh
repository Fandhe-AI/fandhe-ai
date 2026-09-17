#!/usr/bin/env bash
# 系列 B: 事前宣言ゲート「run 開始前 load1 < 8.0（系列 A の最小 8.46 未満）」を各 run で最大 30 分待つ。
# 待機超過なら GATE-TIMEOUT を記録して打ち切り（系列 A を採用）。
D="<scratch>/remeasure"
B="$D/m4max-090b"; mkdir -p "$B"
gate() { local t=0; while :; do l=$(sysctl -n vm.loadavg | awk '{print $2}'); echo "gate $(date -u +%FT%TZ) load1=$l" >> "$B/gate.log"
  awk -v l="$l" 'BEGIN{exit !(l<8.0)}' && return 0; t=$((t+60)); [ $t -ge 1800 ] && return 1; sleep 60; done; }
for i in 1 2 3 4 5; do
  if ! gate; then echo "GATE-TIMEOUT before run$i"; break; fi
  mkdir -p "$B/run$i"; SKIP_BUILD=1 bash "$D/m4max-run-090.sh" "$B/run$i" > "$B/run$i/run.log" 2>&1
  echo "run$i: $(tail -1 "$B/run$i/run.log")"
done; echo ALL-DONE
