#!/usr/bin/env bash
# DGX GB10 追加 4 ラウンド（run2〜run5）。run1 は dgx-run-0100.sh の初回実行（dgx-0100-logs 直下 → run1/ へ移設）
set -uo pipefail
B="${HOME}/work/dgx-0100-logs"
for i in 2 3 4 5; do
  # 失敗したラウンドで止める（計測後に追加。実行時の版は 8c152036。dgx-run-0100.sh 冒頭参照）
  if ! LOGD="${B}/run${i}" bash "${HOME}/work/dgx-run-0100.sh" > "${B}/run${i}.log" 2>&1; then
    echo "run${i}: FAILED $(tail -1 "${B}/run${i}.log")"; exit 1
  fi
  echo "run${i}: $(tail -1 "${B}/run${i}.log")"
done
echo "loop done. $(date -u +%FT%TZ)"
