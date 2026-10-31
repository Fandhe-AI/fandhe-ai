#!/usr/bin/env bash
# 系列 B（唯一の系列・正式）: 事前宣言ゲート「run 開始前 load1 < 3.0」を各 run で最大 30 分待つ。計測キット（未実測）。
# 0.12.0 の macmini-0120b-loop.sh からの派生。0.13.0 は系列 A（対照・ゲートなし）を行わない。したがって
#   - ビルドは系列 A の run1 ではなく、本ループが系列開始前に 1 回だけ行う（macmini-run-0130.sh を BUILD_ONLY=1 で呼ぶ）。
#     ビルド直後の負荷はゲートが落ち着くまで待つので測定窓には入らない。
#   - 「ノイズ帯」（系列 A・B で区分が反転した行）は算出できない。RULE.txt とスコアボードに「対照系列なし・未算出」と明記する。
# 待機超過・数値でない値は GATE-TIMEOUT／GATE-INVALID で系列を打ち切る。5/5 完走したときだけ採用する（それ以外は打ち切りとして記録し、取り直しは RULE.v2.txt）。
# 必要な環境変数: ROOT・D・EXPECT_SHA。bash 3.2 互換（${VAR} 形式・setsid なし。バックグラウンド起動は nohup のサブシェルで）。
B="${D:?}/macmini-0130b"; mkdir -p "${B}"
# 負荷値の検証（fail-closed）: sysctl の失敗や出力形式の変化で load1 が数値として取れない場合は、
# ゲートを通過させず GATE-INVALID を記録して系列を打ち切る（awk は空文字列を 0 と比較するため）。
gate() { local t=0 raw l; while :; do
  if ! raw=$(sysctl -n vm.loadavg); then echo "gate $(date -u +%FT%TZ) GATE-INVALID sysctl-failed" >> "${B}/gate.log"; return 2; fi
  l=$(printf '%s\n' "${raw}" | awk '{print $2}')
  echo "gate $(date -u +%FT%TZ) load1=${l}" >> "${B}/gate.log"
  if ! printf '%s' "${l}" | grep -Eq '^[0-9]+(\.[0-9]+)?$'; then echo "gate $(date -u +%FT%TZ) GATE-INVALID load1-not-numeric" >> "${B}/gate.log"; return 2; fi
  awk -v l="${l}" 'BEGIN{exit !(l<3.0)}' && return 0; t=$((t+60)); [ ${t} -ge 1800 ] && return 1; sleep 60; done; }
# 事前ビルド（系列開始前に 1 回）。失敗したら系列を始めない
mkdir -p "${B}/build"
if ! ROOT="${ROOT:?}" D="${D}" EXPECT_SHA="${EXPECT_SHA:?}" BUILD_ONLY=1 bash "${D}/macmini-run-0130.sh" "${B}/build" > "${B}/build/run.log" 2>&1; then
  echo "BUILD/PRECHECK FAILED: $(tail -1 "${B}/build/run.log")"; exit 1
fi
FC="${ROOT}/scripts/bench/framework-compare"
for bin in bench-fandhe bench-candle bench-burn; do [ -x "${FC}/target/release/${bin}" ] || { echo "BUILD FAILED (${bin} missing)"; exit 1; }; done
done_n=0
for i in 1 2 3 4 5; do
  grc=0; gate || grc=$?
  if [ "${grc}" -eq 2 ]; then echo "GATE-INVALID before run${i}"; echo "SERIES-ABORTED completed=${done_n}/5"; exit 1; fi
  if [ "${grc}" -ne 0 ]; then echo "GATE-TIMEOUT before run${i}"; echo "SERIES-ABORTED completed=${done_n}/5"; exit 1; fi
  mkdir -p "${B}/run${i}"; SKIP_BUILD=1 bash "${D}/macmini-run-0130.sh" "${B}/run${i}" > "${B}/run${i}/run.log" 2>&1; rc=$?
  if [ "${rc}" -ne 0 ]; then echo "run${i}: FAILED rc=${rc} $(tail -1 "${B}/run${i}/run.log")"; echo "SERIES-ABORTED completed=${done_n}/5"; exit 1; fi
  done_n=$((done_n+1))
  echo "run${i}: $(tail -1 "${B}/run${i}/run.log")"
done
[ "${done_n}" = 5 ] || { echo "SERIES-INCOMPLETE completed=${done_n}/5"; exit 1; }
echo "ALL-DONE completed=5/5"
