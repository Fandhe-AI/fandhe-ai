#!/usr/bin/env bash
# 系列 A: ゲートなし 5 run（run1 のみビルド込み）。
# 失敗検知（計測後に PR #2498 の codex P1 を受けて追加）: m4max-run-0110.sh が非零で終わったら
# 系列を打ち切って非零で終了し、最後に完走数が 5 であることを確認する（0.10.0 の計測後に追加）。
# 0.11.0 の計測（2026-10-10・main 624d0ee4）は、この検知を含む版で回した。
# 0.10.0 計測時の経緯（コミット 8c152036 の版で回したこと・事後確認）は `docs/perf/logs/framework-compare-0.10.0-remeasure/README.md` を参照。
A="${D:?}/m4max-0110a"; mkdir -p "${A}"
for i in 1 2 3 4 5; do
  mkdir -p "${A}/run${i}"
  if [ "${i}" = 1 ]; then bash "${D}/m4max-run-0110.sh" "${A}/run${i}" > "${A}/run${i}/run.log" 2>&1; rc=$?
  else SKIP_BUILD=1 bash "${D}/m4max-run-0110.sh" "${A}/run${i}" > "${A}/run${i}/run.log" 2>&1; rc=$?; fi
  if [ "${rc}" -ne 0 ]; then echo "run${i}: FAILED rc=${rc} $(tail -1 "${A}/run${i}/run.log")"; echo "SERIES-ABORTED completed=$((i-1))/5"; exit 1; fi
  echo "run${i}: $(tail -1 "${A}/run${i}/run.log")"
done
n=$(ls -d "${A}"/run[1-5] 2>/dev/null | wc -l | tr -d ' ')
[ "${n}" = 5 ] || { echo "SERIES-INCOMPLETE completed=${n}/5"; exit 1; }
echo "ALL-DONE completed=5/5"
