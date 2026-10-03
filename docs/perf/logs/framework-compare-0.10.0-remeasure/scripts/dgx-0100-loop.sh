#!/usr/bin/env bash
# DGX GB10 追加 4 ラウンド（run2〜run5）。run1 は dgx-run-0100.sh の初回実行（dgx-0100-logs 直下 → run1/ へ移設）
set -uo pipefail
B="${HOME}/work/dgx-0100-logs"
for i in 2 3 4 5; do
  LOGD="${B}/run${i}" bash "${HOME}/work/dgx-run-0100.sh" > "${B}/run${i}.log" 2>&1
  echo "run${i}: $(tail -1 "${B}/run${i}.log")"
done
echo "loop done. $(date -u +%FT%TZ)"
