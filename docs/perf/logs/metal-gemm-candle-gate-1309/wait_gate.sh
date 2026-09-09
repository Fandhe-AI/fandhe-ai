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
#
# 是正（PR #1467 Cursor Bugbot 指摘・再修正）: 上記の「1 回目合格時は
# wait_s を 30 に上書きする」実装のまま、直後のサンプルが不合格になると
# 30*1.5=45s からバックオフを再開してしまい、宣言したバックオフ系列
# （60/90/135s, ...）から外れる（負荷が閾値付近で揺れると 10 回のリトライ
# 予算を早期に消費する）。よって、バックオフ用の待機時間 backoff_s と
# 「合格後の次サンプルまでの間隔」を別変数に分離した。1 回目合格直後の
# 次サンプルは常に 30 秒固定（wait_s）で取り、backoff_s 自体は合格・不合格
# に関わらず変更しない。不合格時のみ wait_s に backoff_s を採用し、
# backoff_s を 1.5 倍する。
#
# 是正（PR #1467 codex-review P2 指摘）: uptime コマンドが失敗するか
# 出力形式が想定と異なると load1 が空文字列になり、`awk -v l="$load1"
# 'BEGIN{print (l<4.0)?...}'` は空文字列を数値 0 として扱うため
# (0<4.0) が真になり、負荷を確認できないまま合格扱いしてしまっていた。
# uptime の終了コードと load1 が数値であることを検証し、失敗時はその
# サンプルを不合格として扱う（pass_count をリセットしバックオフする）。
set -u
LOGDIR="$(cd "$(dirname "$0")" && pwd)"
wait_s=60
backoff_s=60
pass_count=0
attempt=0
gate_result="unmet"
while [ $attempt -lt 10 ]; do
  attempt=$((attempt+1))
  sleep "$wait_s"
  uptime_out=$(uptime 2>&1)
  uptime_status=$?
  load1=""
  if [ "$uptime_status" -eq 0 ]; then
    load1=$(printf '%s\n' "$uptime_out" | awk -F'load averages: ' '{print $2}' | awk '{print $1}')
  fi
  is_numeric=$(awk -v l="$load1" 'BEGIN{if (l == "" ) {print "0"; exit} print (l == l+0) ? "1" : "0"}')
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) attempt=$attempt wait=${wait_s}s uptime_status=${uptime_status} load1=$load1" >> "$LOGDIR/gate-m4max.log"
  if [ "$uptime_status" -eq 0 ] && [ "$is_numeric" = "1" ]; then
    cmp=$(awk -v l="$load1" 'BEGIN{print (l<4.0)?"1":"0"}')
  else
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) attempt=$attempt: uptime 取得失敗または load1 が数値でないため不合格として扱う" >> "$LOGDIR/gate-m4max.log"
    cmp="0"
  fi
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
    # backoff_s 自体は変更しない（直後に不合格へ戻っても宣言系列を維持する）。
    wait_s=30
  else
    # 不合格（0 回目に戻った。1 回目合格直後の確認が失敗した場合を含む）:
    # backoff_s を先に 1.5 倍してから wait_s へ採用することで、確認用の
    # 30s による中断の有無に関わらず宣言系列（60/90/135s, ...）を維持する。
    backoff_s=$(awk -v w="$backoff_s" 'BEGIN{printf "%d", w*1.5}')
    wait_s=$backoff_s
  fi
done
echo "gate_result=$gate_result attempts=$attempt" >> "$LOGDIR/gate-m4max.log"
echo "$gate_result"
