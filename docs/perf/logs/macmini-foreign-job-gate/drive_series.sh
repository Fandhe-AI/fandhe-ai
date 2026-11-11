#!/usr/bin/env bash
# 系列ドライバ（<metal-node> 側で nohup の切り離し起動。他ジョブと重なった系列の測り直し〈remeasure-1〉で #4153・#4151・#4154 が共用）。
# hw-lock を取り、系列の始め方（load1 が抜けるのを待つ）を記録してから、各 round を他ジョブの前提ゲートのラッパー
# （同じディレクトリの round_with_foreign_gate.sh）で挟んで 1..N の順に回し、最後にロックを返す。
# 使い方: drive_series.sh <issue 番号> <系列ディレクトリ（絶対パス・未作成）> <rounds> <run_round.sh（絶対パス）> [VAR=VAL ...]
#   VAR=VAL は run_round.sh へ渡す環境変数（HW_LOCK_HELD=1・HW_LOCK_OWNER_FILE は自動）。
# 系列の始め方: ロック取得後、load1 を 30 秒ごとに読み、1.5 未満になったら始める（最大 40 回＝20 分。届かなければ記録して始める。
#   round ごとの load1 < 3.0 のゲートは run_round.sh 側で従来どおり）。待った秒数・各回の load1 を env/series-start.txt に残す。
# 進め方: ラッパーが 0 → 次の round へ。ラッパーが 1 かつ gate.txt の 1 語目が TIMEOUT（開始前の他ジョブ・load1、run_round.sh の load1）→
#   そのまま記録して次の round へ（TIMEOUT でもラッパーが 5 なら終了後ゲートの失敗・FOREIGN なので止める）。
#   ラッパーが 4 かつ gate.txt の 1 行目が「INVALID foreign-after」か「INVALID foreign-gap」（round 本体は正常終了・終了後に他ジョブ、
#   または開始前の確認と計測の開始の間に隙間）→ その round は INVALID のまま次の round へ。
#   それ以外の非 0（round 本体の失敗・ゲート自身の失敗など）→ 以降の round を回さず停止（想定外として報告）。差し替え・再実行はしない。
# 中断（SIGTERM・SIGINT・SIGHUP）: 実行中の round の子孫（ラッパー・run_round.sh・計測プロセス）へ SIGTERM を送り、全部が終わるのを
#   最大 DRIVE_KILL_WAIT_S 秒（既定 60）待つ。終わったら、その round の gate.txt を「INVALID interrupted-<信号> <UTC>」にして（PASS の元の行は
#   「# was: 」で残す）、ロックを外して exit 1。終わらない子孫が残ったら、ロックを外さずに（計測が生きている間は専有を保つ）
#   「LOCK-KEPT」を drive.status に書いて exit 6。ロックの後始末は人が子孫の終了を確かめてから行う。
# 経路の self-test は test_round_with_foreign_gate.sh（ロックは HOME 配下なので、テストは一時 HOME で動かす）。bash 3.2／5 互換。
set -u
[ "$#" -ge 4 ] || { echo "使い方: drive_series.sh <issue 番号> <系列ディレクトリ> <rounds> <run_round.sh> [VAR=VAL ...]"; exit 2; }
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)" || exit 2
ISSUE="$1"; SER="$2"; N="$3"; RR="$4"; shift 4
case "${ISSUE}" in ''|*[!0-9]*) echo "issue 番号は数字のみ"; exit 2;; esac
case "${N}" in ''|*[!0-9]*|0) echo "rounds は 1 以上の数字のみ"; exit 2;; esac
for p in "${SER}" "${RR}"; do
  case "${p}" in /*) ;; *) echo "絶対パスのみ: ${p}"; exit 2;; esac
done
[ -f "${RR}" ] || { echo "run_round.sh がない: ${RR}"; exit 2; }
for kv in "$@"; do  # env へ渡すのは VAR=VAL だけ（"=" の無い引数が env のコマンドとして実行されるのを防ぐ）
  case "${kv}" in [A-Z_]*=*) ;; *) echo "VAR=VAL の形でない引数: ${kv}"; exit 2;; esac
  case "${kv%%=*}" in *[!A-Z0-9_]*) echo "変数名は大文字英数字と _ のみ: ${kv%%=*}"; exit 2;; esac
done
LOCK="$HOME/work/.fandhe-hw-lock"
ST="${SER}/drive.status"
utc() { date -u +%FT%TZ; }
[ ! -e "${SER}" ] || { echo "${SER} が既にある（上書きしない）"; exit 2; }
if ! mkdir "${LOCK}" 2>/dev/null; then echo "LOCK-BUSY $(cat "${LOCK}/owner" 2>/dev/null)"; exit 3; fi
printf '#%s measure %s %s\n' "${ISSUE}" "$(basename "${SER}")" "$(utc)" > "${LOCK}/owner"
trap 'rm -f "${LOCK}/owner"; rmdir "${LOCK}"; echo "lock released $(utc)" >> "${ST}"' EXIT
cpid=""; i=0
descendants() {  # $1 の子孫の pid（$1 自身は含まない）を 1 行 1 個で出す
  ps -A -o pid=,ppid= | awk -v root="$1" '{ p[$1] = $2 } END {
    n = 1; q[1] = root; seen[root] = 1
    for (h = 1; h <= n; h++) for (c in p) if (p[c] == q[h] && !(c in seen)) { seen[c] = 1; q[++n] = c; print c } }'
}
on_signal() {  # $1=信号名
  trap '' TERM INT HUP
  echo "SIGNAL $1 $(utc) round ${i}" >> "${ST}"
  if [ -n "${cpid}" ]; then
    local w=0 alive all
    all="${cpid} $(descendants "${cpid}" | tr '\n' ' ')"
    while :; do
      # 新しく生まれた子孫も拾いながら SIGTERM を送り、全部が消えるのを待つ
      # shellcheck disable=SC2086  # pid の並びを語に分けて渡す
      all=$(for p in ${all}; do echo "${p}"; descendants "${p}"; done | sort -un | tr '\n' ' ')
      alive=""
      for p in ${all}; do kill -0 "${p}" 2>/dev/null && alive="${alive} ${p}"; done
      [ -z "${alive}" ] && break
      # shellcheck disable=SC2086  # pid の並びを語に分けて渡す
      kill -TERM ${alive} 2>/dev/null
      if [ "${w}" -ge "${DRIVE_KILL_WAIT_S:-60}" ]; then
        echo "LOCK-KEPT children-alive:${alive} $(utc)（ロックは外さない）" >> "${ST}"
        trap - EXIT; exit 6
      fi
      sleep 1; w=$((w + 1))
    done
    wait "${cpid}" 2>/dev/null
    local RD="${SER}/round${i}" first=""
    [ -f "${RD}/gate.txt" ] && first=$(awk 'NR==1{print $1}' "${RD}/gate.txt")
    if [ ! -f "${RD}/gate.txt" ]; then
      mkdir -p "${RD}" && echo "INVALID interrupted-$1 $(utc)" > "${RD}/gate.txt"
    elif [ "${first}" = "PASS" ]; then
      { echo "INVALID interrupted-$1 $(utc)"; sed 's/^/# was: /' "${RD}/gate.txt"; } > "${RD}/gate.txt.int" \
        && mv -f "${RD}/gate.txt.int" "${RD}/gate.txt" \
        || echo "RECORD-FAILED round ${i} gate.txt を INVALID にできない（PASS のまま。集計しない）" >> "${ST}"
    fi
  fi
  echo "STOP round ${i} signal=$1" >> "${ST}"
  exit 1
}
trap 'on_signal TERM' TERM
trap 'on_signal INT' INT
trap 'on_signal HUP' HUP
mkdir -p "${SER}/env" || exit 2
echo "start $(utc) owner=$(cat "${LOCK}/owner")" >> "${ST}"
SS="${SER}/env/series-start.txt"
t0=$(date +%s); echo "lock-acquired $(utc)" > "${SS}"
k=1; reached=no
while [ "${k}" -le 40 ]; do
  # 読み方は #4151・#4154 の run_round.sh の gate() と同じ（#4153 は awk '{print $2}' で同じ値）。取れない・数値でなければ届いたとみなさない（round ごとのラッパーが INVALID にして止める）
  l1=$(sysctl -n vm.loadavg 2>/dev/null | tr -d '{}' | awk '{print $1}')
  echo "sample ${k} $(utc) load1=${l1}" >> "${SS}"
  if printf '%s' "${l1}" | grep -Eq '^[0-9]+(\.[0-9]+)?$' && awk -v v="${l1}" 'BEGIN{exit !(v < 1.5)}'; then reached=yes; break; fi
  k=$((k + 1)); [ "${k}" -le 40 ] && sleep 30
done
echo "series-start $(utc) waited_s=$(( $(date +%s) - t0 )) load1=${l1} reached_below_1.5=${reached}" >> "${SS}"
i=1
while [ "${i}" -le "${N}" ]; do
  # 背景で起こして wait する（wait 中に届いた信号で on_signal が子孫を止められるように）
  env -u RAYON_NUM_THREADS -u CARGO_TARGET_DIR HW_LOCK_HELD=1 HW_LOCK_OWNER_FILE="${LOCK}/owner" "$@" \
    bash "${KIT}/round_with_foreign_gate.sh" "${RR}" "${i}" "${SER}/round${i}" "${SER}/env" > "${SER}/round${i}.log" 2>&1 < /dev/null &
  cpid=$!
  wait "${cpid}"
  rc=$?
  cpid=""
  l=$(head -n 1 "${SER}/round${i}/gate.txt" 2>/dev/null | head -c 200)
  g=$(printf '%s\n' "${l}" | awk '{print $1}')
  echo "round ${i} rc=${rc} gate=${g:-none} $(utc)" >> "${ST}"
  cont=no
  [ "${rc}" -eq 0 ] && cont=yes
  # TIMEOUT はラッパーが 1 で終わったときだけ続ける（終了後ゲートの失敗・FOREIGN は gate.txt が TIMEOUT のままでも exit 5 で止める）
  [ "${g}" = "TIMEOUT" ] && [ "${rc}" -eq 1 ] && cont=yes
  case "${l}" in "INVALID foreign-after "*|"INVALID foreign-gap "*) [ "${rc}" -eq 4 ] && cont=yes ;; esac
  if [ "${cont}" != "yes" ]; then echo "STOP round ${i}" >> "${ST}"; exit 1; fi
  i=$((i + 1))
done
echo "all done $(utc)" >> "${ST}"
