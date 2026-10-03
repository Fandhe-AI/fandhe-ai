#!/usr/bin/env bash
# 系列 A: ゲートなし 5 run（run1 のみビルド込み）。
A="${D:?}/m4max-0100a"; mkdir -p "${A}"
for i in 1 2 3 4 5; do
  mkdir -p "${A}/run${i}"
  if [ "${i}" = 1 ]; then bash "${D}/m4max-run-0100.sh" "${A}/run${i}" > "${A}/run${i}/run.log" 2>&1
  else SKIP_BUILD=1 bash "${D}/m4max-run-0100.sh" "${A}/run${i}" > "${A}/run${i}/run.log" 2>&1; fi
  echo "run${i}: $(tail -1 "${A}/run${i}/run.log")"
done; echo ALL-DONE
