#!/usr/bin/env bash
# 系列 B（正式）: 事前宣言ゲート「run 開始前 load1 < 3.0」を各 run で最大 30 分待つ。待機超過・数値でない値は GATE-TIMEOUT／GATE-INVALID で系列を打ち切る。
# 5/5 完走したときだけ B を正式とし、それ以外は A を正式・B を参考とする（RULE.txt）。0.11.0 の macmini-0110b-loop.sh からの派生（イシュー #3273。草案: 実機未実行）。
# B は SKIP_BUILD=1 で回すため、系列 A の run1（ビルド込み）を先に実行しておくこと。
# 必要な環境変数: ROOT・OLD_ROOT・D・EXPECT_NEW_SHA・EXPECT_OLD_SHA。
B="${D:?}/macmini-0120b"; mkdir -p "${B}"
# 負荷値の検証（fail-closed）: sysctl の失敗や出力形式の変化で load1 が数値として取れない場合は、
# ゲートを通過させず GATE-INVALID を記録して系列を打ち切る（awk は空文字列を 0 と比較するため）。
# この検証は 0.11.0 の計測後に追加されたもので、0.12.0 では最初から有効（弱めない）。
gate() { local t=0 raw l; while :; do
  if ! raw=$(sysctl -n vm.loadavg); then echo "gate $(date -u +%FT%TZ) GATE-INVALID sysctl-failed" >> "${B}/gate.log"; return 2; fi
  l=$(printf '%s\n' "${raw}" | awk '{print $2}')
  echo "gate $(date -u +%FT%TZ) load1=${l}" >> "${B}/gate.log"
  if ! printf '%s' "${l}" | grep -Eq '^[0-9]+(\.[0-9]+)?$'; then echo "gate $(date -u +%FT%TZ) GATE-INVALID load1-not-numeric" >> "${B}/gate.log"; return 2; fi
  awk -v l="${l}" 'BEGIN{exit !(l<3.0)}' && return 0; t=$((t+60)); [ ${t} -ge 1800 ] && return 1; sleep 60; done; }
# 事前検査: SKIP_BUILD=1 で回すため旧腕バイナリが無ければ開始前に fail する
[ -x "${OLD_ROOT:?}/../target-old/release/bench-fandhe" ] || { echo "旧腕バイナリなし。系列 A の run1（ビルド込み）を先に実行すること"; exit 1; }
done_n=0
for i in 1 2 3 4 5; do
  grc=0; gate || grc=$?
  if [ "${grc}" -eq 2 ]; then echo "GATE-INVALID before run${i}"; echo "SERIES-ABORTED completed=${done_n}/5（系列 B は参考扱い）"; exit 1; fi
  if [ "${grc}" -ne 0 ]; then echo "GATE-TIMEOUT before run${i}"; echo "SERIES-ABORTED completed=${done_n}/5（系列 B は参考扱い）"; exit 1; fi
  mkdir -p "${B}/run${i}"; SKIP_BUILD=1 bash "${D}/macmini-run-0120.sh" "${B}/run${i}" > "${B}/run${i}/run.log" 2>&1; rc=$?
  if [ "${rc}" -ne 0 ]; then echo "run${i}: FAILED rc=${rc} $(tail -1 "${B}/run${i}/run.log")"; echo "SERIES-ABORTED completed=${done_n}/5（系列 B は参考扱い）"; exit 1; fi
  done_n=$((done_n+1))
  echo "run${i}: $(tail -1 "${B}/run${i}/run.log")"
done
[ "${done_n}" = 5 ] || { echo "SERIES-INCOMPLETE completed=${done_n}/5"; exit 1; }
echo "ALL-DONE completed=5/5"
