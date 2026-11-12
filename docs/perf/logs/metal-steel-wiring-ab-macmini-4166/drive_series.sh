#!/bin/bash
# #4166 の系列 1 本（RULE.txt 項 4〜6）。所有者の「準備ができた」の連絡を受けてから、計測セッションが nohup で起動する。
# 順に確かめ、1 つでも成り立たなければ計測プロセスを 1 本も起動せずに止める（その試みは系列に数えない）:
#   1. 他ジョブの前提ゲート G1〜G4（展開ツリーの foreign_job_gate.py --phase before が CLEAN〈終了コード 0〉）と、
#      basename がちょうど `Safari` のプロセスが無いこと（#4160 の RULE.v2.txt 項 4 (b) 1。補助プロセスは条件にしない）
#   2. mediaanalysisd: 2 分間隔で 3 回、%CPU が 5 未満（basename の完全一致。複数なら最大値。無ければ通過）
#   3. hw-lock を取り、load1 を 30 秒ごとに読む（1.5 未満で進む。最大 40 回で、届かなければそのまま進んで記録する）
#   4. 1. と同じ確認を、起動の直前（hw-lock の下）にもう一度
#   5. ツリーの照合（runner が書く results/ と target-ab-4160-* を除いた指紋と、配置時の manifest）
#   6. 腕の checkout と道具（RULE.txt 項 4 (b) 6）: after が clean（未追跡を含め git status が空）、before の差分が tile.rs の
#      1 行だけ（未追跡なし）、両者の HEAD が同じ、cargo・rustc・python3・git が PATH で見つかる。runner も同じ検査をするが、
#      ここで落とせば系列のラベルを使わずに済む（厳しくする方向だけの追加）。
# 通れば、セル A/B を 1 系列（5 round）回し、全ゲート通過かどうかをゲートの行と終了コードだけで決める（値は見ない）。
# 使い方: drive_series.sh <W> <k> <試みの番号 n> <run|dry>
#   dry は 1〜5 だけを通す（hw-lock は取って外す。計測は起動しない。出力は W/precheck-dry-<n>.txt）。
# 出力（run）: 始めなかった試み → W/prestart-<k>-<n>.txt。起動した系列 → W/formal-<k>/（series-start.txt・ab.out）と
#   W/formal-<k>.status、W/drive.log に 1 行の判定。runner の記録は W/tree/scripts/bench/framework-compare/results/raw/。
set -u
W=$1; K=$2; N=$3; MODE=$4
case "${K}" in 1|2|3) ;; *) echo "k は 1〜3" >&2; exit 2;; esac
case "${N}" in ''|*[!0-9]*) echo "n は数字" >&2; exit 2;; esac
FG="${W}/tree/docs/perf/logs/macmini-foreign-job-gate/foreign_job_gate.py"
FC="${W}/tree/scripts/bench/framework-compare"
LOG="${W}/drive.log"
EXP_AB=$(cat "${W}/fp-deploy-ab.txt")
SER="formal-${K}"; LABEL="formal-${K}"
utc() { date -u +%FT%TZ; }
case "${MODE}" in run) A="${W}/attempt-${K}-${N}.txt";; dry) A="${W}/precheck-dry-${N}.txt";; *) echo "mode?" >&2; exit 2;; esac
[ -e "${A}" ] && { echo "exists: ${A}" >&2; exit 2; }
[ "${MODE}" = run ] && [ -e "${W}/${SER}" ] && { echo "exists: ${W}/${SER}" >&2; exit 2; }
# 始めなかった試みの記録（prestart）も使用済みの試み番号として拒否する（RULE.txt 項 6。記録を上書きしない）。
[ "${MODE}" = run ] && [ -e "${W}/prestart-${K}-${N}.txt" ] && { echo "exists: ${W}/prestart-${K}-${N}.txt" >&2; exit 2; }
: > "${A}"
say() { echo "$*" >> "${A}"; }
LOCKED=0; LOCK="${HOME}/work/.fandhe-hw-lock"
release() { if [ "${LOCKED}" = 1 ]; then rm -f "${LOCK}/owner"; rmdir "${LOCK}"; LOCKED=0; say "lock released $(utc)"; fi; }
notstart() {
  say "result=not-started reason=$1 $(utc)"; release
  if [ "${MODE}" = run ]; then mv "${A}" "${W}/prestart-${K}-${N}.txt"; echo "${SER} attempt ${N} NOT-STARTED reason=$1 $(utc)" >> "${LOG}"; fi
  exit 5
}
proc_check() { # proc_check <回目>
  local g="${W}/.pc-gate-${K}-${N}-$1.txt" grc app
  rm -f "${g}"
  python3 -I "${FG}" --phase before --out "${g}" > /dev/null 2>&1; grc=$?
  app=$(ps -A -o comm= | awk -F/ '$NF=="Safari"' | wc -l | tr -d ' ')
  say "proc-check$1 $(utc) safari_app=${app} gate_rc=${grc}"
  say "  safari-named (pcpu basename):"
  ps -A -o pcpu=,comm= | awk '{p=$1; $1=""; sub(/^ +/,""); n=split($0,a,"/"); if (a[n] ~ /Safari/) print "    " p " " a[n]}' | LC_ALL=C sort -k2 >> "${A}"
  say "  gate output:"; [ -f "${g}" ] && sed 's/^/    /' "${g}" >> "${A}"; rm -f "${g}"
  [ "${app}" = 0 ] && [ "${grc}" = 0 ]
}
mediaanalysisd_pcpu() { # 最大 %CPU（basename の完全一致）。無ければ absent、ps 失敗は ps-failed、非数値は not-numeric
  local out
  out=$(ps -A -o pcpu=,comm=) || { echo ps-failed; return; }
  [ -n "${out}" ] || { echo ps-empty; return; }
  printf '%s\n' "${out}" | awk '{p=$1; $1=""; sub(/^ +/,""); n=split($0,a,"/"); if (a[n]=="mediaanalysisd") { if (p !~ /^[0-9]+(\.[0-9]+)?$/) bad=1; else if (!f || p+0>m) m=p+0; f=1 } }
    END { if (bad) print "not-numeric"; else if (!f) print "absent"; else print m }'
}
say "${SER} attempt=${N} mode=${MODE} begin $(utc)"
say "env macos=$(sw_vers -productVersion) hw_model=$(sysctl -n hw.model) rustc=$(rustc -V 2>&1)"
# 1. 他ジョブの前提ゲート G1〜G4 と Safari（1 回目）
proc_check 1 || notstart proc-check1
# 2. mediaanalysisd
ok=1
for i in 1 2 3; do
  v=$(mediaanalysisd_pcpu)
  say "mediaanalysisd check${i} $(utc) pcpu=${v}"
  case "${v}" in
    absent) ;;
    ps-failed|ps-empty|not-numeric) ok=0 ;;
    *) awk -v x="${v}" 'BEGIN{exit !(x < 5)}' || ok=0 ;;
  esac
  [ "${i}" -lt 3 ] && sleep 120
done
n=$(log show --last 10m --style compact --predicate 'process == "spotlightknowledged" AND eventMessage CONTAINS "Submitting text processing"' 2>/dev/null | awk 'NR>1' | wc -l | tr -d ' ')
say "spotlightknowledged_submitting_last10m=${n} idle_ok=${ok}"
[ "${ok}" = 1 ] || notstart mediaanalysisd
# 3. hw-lock と load1
if ! mkdir "${LOCK}" 2>/dev/null; then say "LOCK-BUSY owner=$(cat "${LOCK}/owner" 2>/dev/null)"; notstart lock-busy; fi
LOCKED=1; trap release EXIT
printf '#4166 measure %s %s\n' "${SER}-${MODE}" "$(utc)" > "${LOCK}/owner"
say "lock-acquired $(utc) owner=$(cat "${LOCK}/owner")"
t0=$(date +%s); c=0
while :; do
  c=$((c + 1)); l=$(sysctl -n vm.loadavg | awk '{print $2}'); say "load1 sample ${c} $(utc) load1=${l}"
  if awk -v a="${l}" 'BEGIN{exit !(a<1.5)}'; then say "load1-wait done waited_s=$(( $(date +%s) - t0 )) load1=${l} reached_below_1.5=yes"; break; fi
  if [ "${c}" -ge 40 ]; then say "load1-wait done waited_s=$(( $(date +%s) - t0 )) load1=${l} reached_below_1.5=no"; break; fi
  sleep 30
done
# 4. 1. と同じ確認（2 回目。起動の直前・hw-lock の下）
proc_check 2 || notstart proc-check2
# 5. ツリーの照合。合わなければ __pycache__ の一覧を残す（消すのは所有者の了承の後。RULE.txt 項 5 (a)）。
fpab=$(bash "${W}/rule/fp.sh" "${W}/tree" ab)
( cd "${W}/tree" && shasum -a 256 -c --quiet "${W}/manifest-deploy.txt" ) > "${W}/.mf-${K}-${N}.txt" 2>&1; mrc=$?
say "tree-check $(utc) fingerprint_ab_excluded: ${fpab} expected: ${EXP_AB} manifest_check_rc=${mrc}"
rm -f "${W}/.mf-${K}-${N}.txt"
if [ "${fpab}" != "${EXP_AB}" ] || [ "${mrc}" != 0 ]; then
  say "  __pycache__ in tree:"; ( cd "${W}/tree" && find . -path '*/__pycache__/*' -type f | LC_ALL=C sort | sed 's/^/    /' ) >> "${A}"
  notstart tree-mismatch
fi
# 6. 腕の checkout と道具
TILE=crates/backend-metal/src/tile.rs
ast=$(git -C "${W}/after" status --porcelain 2>&1); ahead=$(git -C "${W}/after" rev-parse HEAD 2>&1)
bst=$(git -C "${W}/before" status --porcelain 2>&1); bhead=$(git -C "${W}/before" rev-parse HEAD 2>&1)
bnum=$(git -C "${W}/before" diff --numstat HEAD 2>&1 | tr '\t' ' ')
tools=ok; for t in cargo rustc python3 git; do command -v "${t}" > /dev/null 2>&1 || tools="missing-${t}"; done
say "checkout-check $(utc) after_status_lines=$(printf '%s' "${ast}" | grep -c .) before_status=[$(printf '%s' "${bst}" | tr '\n' ';')] before_numstat=[${bnum}] head_equal=$([ "${ahead}" = "${bhead}" ] && echo yes || echo no) tools=${tools}"
if [ -n "${ast}" ] || [ "${bst}" != " M ${TILE}" ] || [ "${bnum}" != "1 1 ${TILE}" ] || [ "${ahead}" != "${bhead}" ]; then notstart checkout-mismatch; fi
[ "${tools}" = ok ] || notstart "${tools}"
if [ "${MODE}" = dry ]; then say "result=dry-ok (no measurement started) $(utc)"; exit 0; fi
# 起動
mkdir "${W}/${SER}" || notstart outdir-exists
say "series-start $(utc)"
mv "${A}" "${W}/${SER}/series-start.txt"; A="${W}/${SER}/series-start.txt"
ST="${W}/${SER}.status"
echo "${SER} begin $(utc)" >> "${LOG}"
echo "start $(utc) owner=$(cat "${LOCK}/owner")" >> "${ST}"
# PYTHONDONTWRITEBYTECODE=1 は RULE.txt 項 5 (a) の指示どおり渡す。ただし runner と比較スクリプトは `python3 -I` で
# 起動するため、この環境変数は効かない（-I は -E を含む。9eba21749 で確かめた）。
( cd "${FC}" && env -u RAYON_NUM_THREADS -u CARGO_TARGET_DIR PYTHONDONTWRITEBYTECODE=1 AB_DIRECTION=post-wiring \
    FOREIGN_GATE="${FG}" AB_BEFORE_FACADE_PATH="${W}/before/crates/facade" AB_AFTER_FACADE_PATH="${W}/after/crates/facade" \
    bash run_ab_steel_large_n_gemm_metal.sh "${LABEL}" ) > "${W}/${SER}/ab.out" 2>&1
echo "ab rc=$? $(utc)" >> "${ST}"
echo "all done $(utc)" >> "${ST}"
# 全ゲート通過の判定（RULE.txt 項 6。ゲートの行と終了コードだけ。値は見ない）
AG="${FC}/results/raw/load-gate-4160-metal-${LABEL}.log"
SK="${FC}/results/raw/skipped-4160-metal-${LABEL}.log"
abc=$(grep -c '^ab rc=0 ' "${ST}")
an=$(grep -c 'gate=' "${AG}" 2>/dev/null); ap=$(grep -cE '(^|[[:space:]])gate=pass([[:space:]]|$)' "${AG}" 2>/dev/null)
sk=missing; [ -f "${SK}" ] && sk=$(wc -c < "${SK}" | tr -d ' ')
v=notpass
if [ "${abc}" = 1 ] && [ "${an:-0}" -gt 0 ] && [ "${an}" = "${ap}" ] && [ "${sk}" = 0 ]; then v=pass; fi
echo "${SER} verdict=${v} ab_rc0=${abc} ab_gate=${ap}/${an} skipped_bytes=${sk} $(utc)" >> "${LOG}"
