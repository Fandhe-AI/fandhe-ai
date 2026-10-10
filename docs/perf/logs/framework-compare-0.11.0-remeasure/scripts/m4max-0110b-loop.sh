#!/usr/bin/env bash
# 系列 B: 事前宣言ゲート「run 開始前 load1 < 8.0」を各 run で最大 30 分待つ（0.9.0 系列 B と同一条件）。
# 待機超過なら GATE-TIMEOUT を記録して打ち切り（系列 A を採用）。
# 失敗検知（計測後に PR #2498 の codex P1 を受けて追加）: m4max-run-0110.sh が非零で終わったら
# 系列を打ち切って非零で終了し、最後に完走数が 5 であることを確認する（0.10.0 の計測後に追加）。
# 0.11.0 の計測（2026-10-10・main 624d0ee4）は、この検知を含む版で回した。
# 0.10.0 計測時の経緯（コミット 8c152036 の版で回したこと・事後確認）は `docs/perf/logs/framework-compare-0.10.0-remeasure/README.md` を参照。
B="${D:?}/m4max-0110b"; mkdir -p "${B}"
# 負荷値の検証（fail-closed）: sysctl の失敗や出力形式の変化で load1 が数値として取れない場合は、
# ゲートを通過させず GATE-INVALID を記録して系列を打ち切る（awk は空文字列を 0 と比較するため）。
# この検証は 0.11.0 の計測後に PR #2963 の codex P2 を受けて追加した。同計測の gate.log の 5 判定は
# すべて数値（load1=4.55〜5.14）であり、検証の有無で判定は変わらない。
gate() { local t=0 raw l; while :; do
  if ! raw=$(sysctl -n vm.loadavg); then echo "gate $(date -u +%FT%TZ) GATE-INVALID sysctl-failed" >> "${B}/gate.log"; return 2; fi
  l=$(printf '%s\n' "${raw}" | awk '{print $2}')
  echo "gate $(date -u +%FT%TZ) load1=${l}" >> "${B}/gate.log"
  if ! printf '%s' "${l}" | grep -Eq '^[0-9]+(\.[0-9]+)?$'; then echo "gate $(date -u +%FT%TZ) GATE-INVALID load1-not-numeric" >> "${B}/gate.log"; return 2; fi
  awk -v l="${l}" 'BEGIN{exit !(l<8.0)}' && return 0; t=$((t+60)); [ ${t} -ge 1800 ] && return 1; sleep 60; done; }
done_n=0
for i in 1 2 3 4 5; do
  grc=0; gate || grc=$?
  if [ "${grc}" -eq 2 ]; then echo "GATE-INVALID before run${i}"; echo "SERIES-ABORTED completed=${done_n}/5（系列 B は参考扱い）"; exit 1; fi
  if [ "${grc}" -ne 0 ]; then echo "GATE-TIMEOUT before run${i}"; echo "SERIES-ABORTED completed=${done_n}/5（系列 B は参考扱い）"; exit 1; fi
  mkdir -p "${B}/run${i}"; SKIP_BUILD=1 bash "${D}/m4max-run-0110.sh" "${B}/run${i}" > "${B}/run${i}/run.log" 2>&1; rc=$?
  if [ "${rc}" -ne 0 ]; then echo "run${i}: FAILED rc=${rc} $(tail -1 "${B}/run${i}/run.log")"; echo "SERIES-ABORTED completed=${done_n}/5（系列 B は参考扱い）"; exit 1; fi
  done_n=$((done_n+1))
  echo "run${i}: $(tail -1 "${B}/run${i}/run.log")"
done
[ "${done_n}" = 5 ] || { echo "SERIES-INCOMPLETE completed=${done_n}/5"; exit 1; }
echo "ALL-DONE completed=5/5"
