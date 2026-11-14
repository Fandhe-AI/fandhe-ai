#!/usr/bin/env bash
# 系列 B（唯一の系列）: 各 run の前に「他ジョブ CLEAN かつ load1 < 3.0」を 30 秒間隔・最大 30 分待ち、各 run の後に他ジョブの有無を確かめる。
# 計測キット（道具のみ・RULE 未固定・未計測。#4177）。0.14.0 の macmini-0140b-loop.sh の写し。
# 0.14.0 版からの変更点:
#   (1) 他ジョブゲートを足した（所有者承認 2026-11-14。別プロジェクトが hw-lock を取らずに GPU を使うため。#4168 と同じ扱い）。
#       foreign_gate_wrapper.sh 経由で sha256 固定済みの foreign_job_gate.py・ab_foreign_gate.sh を呼ぶ（中身は変えない）:
#       - 各 run の前: CLEAN かつ load1 < 3.0 になるまで 30 秒間隔・最大 30 分待つ（0.14.0 の load1 の run 前待ちは、この待ちに含まれる。閾値は同じ 3.0）。
#         超えたら GATE-TIMEOUT、数値が取れない・ゲート自身の失敗は GATE-INVALID で系列を打ち切る。
#       - 各 run の後: FOREIGN（またはゲート自身の失敗）ならその run を無効にして系列を打ち切る（部分系列で続ける経路は作らない）。
#       gate.log の文法は #4168 と同じ（foreign_gate_wrapper.sh の冒頭）。ゲートの出力は foreign-gate/ に残る。
#   (2) 版の文字列（作業ディレクトリ名 macmini-0150b・呼び先 macmini-run-0150.sh）。呼び先は本スクリプトの隣（SD）から取る。
# 0.14.0 と同じ点: 系列 A（対照）は行わない（ノイズ帯は「対照系列なし・未算出」）。ビルドは系列開始前に 1 回だけ
#   （macmini-run-0150.sh を BUILD_ONLY=1 で呼ぶ）。5/5 完走したときだけ採用し、それ以外は打ち切りとして記録する（取り直しは RULE.v2.txt）。
# 必要な環境変数: ROOT・D・EXPECT_SHA（任意: PY_VENV_DIR。run へ引き継ぐ）。bash 3.2 互換（${VAR} 形式・setsid なし）。
SD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
B="${D:?}/macmini-0150b"; mkdir -p "${B}"
# shellcheck source=/dev/null
. "${SD}/foreign_gate_wrapper.sh"
GATE_LOAD1=3.0
# 他ジョブゲートの準備（sha256 検査・出力ディレクトリの新規作成）。失敗したら系列を始めない（ビルドより前）
fgw_init "${ROOT:?}" "${B}/foreign-gate" || { echo "FOREIGN-GATE PRECHECK FAILED"; exit 1; }
# 事前ビルド（系列開始前に 1 回）。失敗したら系列を始めない
mkdir -p "${B}/build"
if ! ROOT="${ROOT:?}" D="${D}" EXPECT_SHA="${EXPECT_SHA:?}" BUILD_ONLY=1 bash "${SD}/macmini-run-0150.sh" "${B}/build" > "${B}/build/run.log" 2>&1; then
  echo "BUILD/PRECHECK FAILED: $(tail -1 "${B}/build/run.log")"; exit 1
fi
FC="${ROOT}/scripts/bench/framework-compare"
for bin in bench-fandhe bench-candle bench-burn; do [ -x "${FC}/target/release/${bin}" ] || { echo "BUILD FAILED (${bin} missing)"; exit 1; }; done
done_n=0
for i in 1 2 3 4 5; do
  grc=0; fgw_before "${i}" "${GATE_LOAD1}" "${B}/gate.log" || grc=$?
  if [ "${grc}" -eq 2 ]; then echo "GATE-INVALID before run${i}"; echo "SERIES-ABORTED completed=${done_n}/5"; exit 1; fi
  if [ "${grc}" -ne 0 ]; then echo "GATE-TIMEOUT before run${i}"; echo "SERIES-ABORTED completed=${done_n}/5"; exit 1; fi
  mkdir -p "${B}/run${i}"; SKIP_BUILD=1 bash "${SD}/macmini-run-0150.sh" "${B}/run${i}" > "${B}/run${i}/run.log" 2>&1; rc=$?
  if [ "${rc}" -ne 0 ]; then echo "run${i}: FAILED rc=${rc} $(tail -1 "${B}/run${i}/run.log")"; echo "SERIES-ABORTED completed=${done_n}/5"; exit 1; fi
  arc=0; fgw_after "${i}" "${B}/gate.log" || arc=$?
  if [ "${arc}" -ne 0 ]; then
    echo "SERIES-ABORTED completed=${done_n}/5 (run${i} は無効: $(fg_after_reason))"; exit 1
  fi
  done_n=$((done_n+1))
  echo "run${i}: $(tail -1 "${B}/run${i}/run.log")"
done
[ "${done_n}" = 5 ] || { echo "SERIES-INCOMPLETE completed=${done_n}/5"; exit 1; }
echo "ALL-DONE completed=5/5"
