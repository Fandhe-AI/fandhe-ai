#!/usr/bin/env bash
# 系列 B: 事前宣言ゲート「run 開始前 load1 < 8.0」を各 run で最大 30 分待つ（0.9.0 系列 B と同一条件）。
# 待機超過なら GATE-TIMEOUT を記録して打ち切り（系列 A を採用）。
# 失敗検知（計測後に PR #2498 の codex P1 を受けて追加）: m4max-run-0110.sh が非零で終わったら
# 系列を打ち切って非零で終了し、最後に完走数が 5 であることを確認する。2026-10-03 の計測を実際に
# 回したのは追加前の版（コミット 8c152036 の本ファイル）で、10 run すべての完走は README の
# 「M4 スクリプトの失敗検知（事後確認）」で確認済み。
B="${D:?}/m4max-0110b"; mkdir -p "${B}"
gate() { local t=0; while :; do l=$(sysctl -n vm.loadavg | awk '{print $2}'); echo "gate $(date -u +%FT%TZ) load1=${l}" >> "${B}/gate.log"
  awk -v l="${l}" 'BEGIN{exit !(l<8.0)}' && return 0; t=$((t+60)); [ ${t} -ge 1800 ] && return 1; sleep 60; done; }
done_n=0
for i in 1 2 3 4 5; do
  if ! gate; then echo "GATE-TIMEOUT before run${i}"; echo "SERIES-ABORTED completed=${done_n}/5（系列 B は参考扱い）"; exit 1; fi
  mkdir -p "${B}/run${i}"; SKIP_BUILD=1 bash "${D}/m4max-run-0110.sh" "${B}/run${i}" > "${B}/run${i}/run.log" 2>&1; rc=$?
  if [ "${rc}" -ne 0 ]; then echo "run${i}: FAILED rc=${rc} $(tail -1 "${B}/run${i}/run.log")"; echo "SERIES-ABORTED completed=${done_n}/5（系列 B は参考扱い）"; exit 1; fi
  done_n=$((done_n+1))
  echo "run${i}: $(tail -1 "${B}/run${i}/run.log")"
done
[ "${done_n}" = 5 ] || { echo "SERIES-INCOMPLETE completed=${done_n}/5"; exit 1; }
echo "ALL-DONE completed=5/5"
