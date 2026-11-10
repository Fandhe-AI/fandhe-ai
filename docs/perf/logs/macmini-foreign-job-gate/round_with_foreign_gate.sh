#!/usr/bin/env bash
# Mac mini（<metal-node>）の 1 round を、他ジョブの前提ゲート（foreign_job_gate.py）で前後から挟んで実行するラッパー。
# 役割: 各 issue の run_round.sh（#4151・#4153・#4154 ほか。本体は変えない）を呼ぶ前と呼んだ後に、計測対象以外の重いジョブ
#   （caffeinate・%CPU 25 以上・重い名前・続いている GPU 使用）がいないことを 1 回ずつ確かめる。2026-11-10 に hw-lock を取らない
#   他ジョブが計測と重なった件（各 issue の所有者判断コメント「他ジョブと重なった系列は測り直す」）の再発防止で、
#   load1 のゲート・閾値・分類式・確度の規則はそのまま（ゲートを厳しくする方向だけの追加）。
# 呼び出し元: 計測ドライバ（hw-lock を取った側）が round ごとに
#   bash round_with_foreign_gate.sh <run_round.sh> <round 番号> <round ディレクトリ> <env ディレクトリ>
#   と呼ぶ。run_round.sh へ渡す環境変数（HW_LOCK_HELD=1・EXPECT_SHA 等）はそのまま引き継ぐ。
# 開始前（before）: 30 秒間隔で最大 60 回（30 分）確かめ、CLEAN になったら run_round.sh を始める。各回の結果を
#   <env>/foreign-gate/r<i>-before-a<k>.txt に残す。60 回とも FOREIGN なら run_round.sh を呼ばず、gate.txt を
#   `TIMEOUT foreign-job before <UTC>` にして exit 1（load1 ゲートの TIMEOUT と同じ扱い。差し替え・再実行はしない）。
# 終了後（after）: run_round.sh が返った直後に 1 回確かめ、<env>/foreign-gate/r<i>-after.txt に残す。FOREIGN なら
#   gate.txt の 1 語目が PASS の round を `INVALID foreign-after ...` に書き換え（元の行は "# was: " で残す。#4153 の
#   run_round.sh の EXIT trap と同じ形）。round 本体が 0 で終わり PASS だった round なら exit 4（その round だけ INVALID。
#   drive_series.sh は次の round へ進む）。round 本体が非 0・gate.txt が無い場合は INVALID にして exit 5（系列を止める）。
# ゲート自身の失敗（exit 2）は fail-closed: 開始前なら gate.txt を `INVALID foreign-gate-error before` にして exit 1、
#   終了後なら PASS の round を `INVALID foreign-gate-error after` にして exit 5（系列を止める）。
# 集計（aggregate_diag.py・aggregate_4151.py・aggregate_4154.py 等）は gate.txt の 1 語目 PASS／TIMEOUT／INVALID を
#   既存のとおり読み、PASS 以外を正式系列から外す。本ラッパーは新しい 1 語目を作らない。
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)" || exit 2
GATE="${HERE}/foreign_job_gate.py"
[ "$#" -eq 4 ] || { echo "使い方: round_with_foreign_gate.sh <run_round.sh> <round 番号> <round ディレクトリ> <env ディレクトリ>"; exit 2; }
RR="$1"; RN="$2"; RD="$3"; ENVD="$4"
for p in "${RR}" "${RD}" "${ENVD}"; do
  case "${p}" in /*) ;; *) echo "絶対パスのみ: ${p}"; exit 2;; esac
  case "${p}" in *\"*|*\'*|*\\*|*\$*|*\`*|*$'\n'*|*' '*) echo "禁止文字を含む: ${p}"; exit 2;; esac
done
case "${RN}" in ''|*[!0-9]*) echo "round 番号は数字のみ"; exit 2;; esac
[ -f "${RR}" ] || { echo "run_round.sh がない: ${RR}"; exit 2; }
[ -f "${GATE}" ] || { echo "foreign_job_gate.py がない"; exit 2; }
[ ! -e "${RD}/gate.txt" ] || { echo "${RD}/gate.txt が既にある（上書き禁止）"; exit 2; }
FG="${ENVD}/foreign-gate"
mkdir -p "${FG}" || exit 2
utc() { date -u +%FT%TZ; }

# ---- 開始前
k=1
while :; do
  python3 -I "${GATE}" --phase before --out "${FG}/r${RN}-before-a${k}.txt"
  g=$?
  if [ "${g}" -eq 0 ]; then
    echo "foreign-gate before CLEAN attempt=${k} $(utc)"
    break
  fi
  if [ "${g}" -ne 1 ]; then
    mkdir -p "${RD}" && echo "INVALID foreign-gate-error before rc=${g} $(utc)" > "${RD}/gate.txt"
    echo "foreign-gate before ERROR rc=${g}"; exit 1
  fi
  if [ "${k}" -ge 60 ]; then
    mkdir -p "${RD}" && echo "TIMEOUT foreign-job before $(utc)" > "${RD}/gate.txt"
    echo "foreign-gate before TIMEOUT attempts=${k}"; exit 1
  fi
  echo "foreign-gate before FOREIGN attempt=${k} $(utc)（30 秒後に再確認）"
  k=$((k + 1))
  sleep 30
done

# ---- round 本体（本体の終了コードはそのまま返す）
bash "${RR}" "${RN}" "${RD}"
rc=$?

# ---- 終了後
python3 -I "${GATE}" --phase after --out "${FG}/r${RN}-after.txt"
g=$?
if [ "${g}" -eq 0 ]; then
  echo "foreign-gate after CLEAN $(utc)"
  exit "${rc}"
fi
why="foreign-after"; [ "${g}" -ne 1 ] && why="foreign-gate-error after rc=${g}"
first=""; [ -f "${RD}/gate.txt" ] && first=$(awk 'NR==1{print $1}' "${RD}/gate.txt")
if [ ! -f "${RD}/gate.txt" ]; then
  mkdir -p "${RD}" && echo "INVALID ${why} $(utc)" > "${RD}/gate.txt"
elif [ "${first}" = "PASS" ]; then
  { echo "INVALID ${why} $(utc)"; sed 's/^/# was: /' "${RD}/gate.txt"; } > "${RD}/.gate.txt.tmp" \
    && mv "${RD}/.gate.txt.tmp" "${RD}/gate.txt"
fi
# 続けてよいのは「round 本体が 0 で終わり PASS だった round に、終了後ゲートが FOREIGN を返した」場合だけ（exit 4）。
# その round だけを INVALID にし、次の round は開始前ゲートで他ジョブが去るのを待つ（所有者判断「後で見つかったら、その round は INVALID」）。
# round 本体の失敗・gate.txt が無い（precheck で止まった）・ゲート自身の失敗は exit 5 で系列を止める（想定外として報告）。
if [ "${g}" -eq 1 ] && [ "${rc}" -eq 0 ] && [ "${first}" = "PASS" ]; then
  echo "foreign-gate after ${why}（round ${RN} は INVALID・次の round へ）"
  exit 4
fi
echo "foreign-gate after ${why}（round ${RN} は INVALID・round の終了コード ${rc}・系列を止める）"
exit 5
