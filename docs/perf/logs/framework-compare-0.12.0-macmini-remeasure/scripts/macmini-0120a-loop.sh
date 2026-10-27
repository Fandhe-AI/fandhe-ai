#!/usr/bin/env bash
# 系列 A（対照・ゲートなし）5 run。run1 のみビルド込み（ビルドは系列開始前に 1 回。ログは build.log・判定に使わない）。
# 0.11.0 の macmini-0110a-loop.sh からの派生（イシュー #3273。草案: 実機未実行）。
# 失敗検知: macmini-run-0120.sh が非零で終わったら系列を打ち切って非零で終了し、最後に完走数 5 を確認する。
# 必要な環境変数（macmini-run-0120.sh へ引き継ぐ）: ROOT・OLD_ROOT・D・EXPECT_NEW_SHA・EXPECT_OLD_SHA。
A="${D:?}/macmini-0120a"; mkdir -p "${A}"
for i in 1 2 3 4 5; do
  mkdir -p "${A}/run${i}"
  if [ "${i}" = 1 ]; then bash "${D}/macmini-run-0120.sh" "${A}/run${i}" > "${A}/run${i}/run.log" 2>&1; rc=$?
  else SKIP_BUILD=1 bash "${D}/macmini-run-0120.sh" "${A}/run${i}" > "${A}/run${i}/run.log" 2>&1; rc=$?; fi
  if [ "${rc}" -ne 0 ]; then echo "run${i}: FAILED rc=${rc} $(tail -1 "${A}/run${i}/run.log")"; echo "SERIES-ABORTED completed=$((i-1))/5"; exit 1; fi
  echo "run${i}: $(tail -1 "${A}/run${i}/run.log")"
done
n=$(ls -d "${A}"/run[1-5] 2>/dev/null | wc -l | tr -d ' ')
[ "${n}" = 5 ] || { echo "SERIES-INCOMPLETE completed=${n}/5"; exit 1; }
echo "ALL-DONE completed=5/5"
