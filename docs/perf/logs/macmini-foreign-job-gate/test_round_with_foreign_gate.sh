#!/usr/bin/env bash
# round_with_foreign_gate.sh と drive_series.sh の経路の self-test（実機・ロック不要。macOS／Linux の bash で動く）。
# 役割: 一時ディレクトリへラッパーを複製し、foreign_job_gate.py をスタブ（終了コードを順に返す）に、sleep を即時終了の
#   スタブに差し替えて、開始前 CLEAN・開始前 TIMEOUT・開始前のゲート失敗・終了後 FOREIGN・終了後のゲート失敗・
#   round 本体の失敗・開始前の load1 の待ちと TIMEOUT・開始前の確認と計測の開始の隙間（foreign-gap）の各経路で、
#   gate.txt の 1 語目とラッパーの終了コードを確かめる（sysctl もスタブにして load1 を順に返す）。drive_series.sh は一時 HOME（ロックの場所）と
#   sysctl のスタブ（load1 を順に返す）で、全 round 完走・終了後 FOREIGN は INVALID のまま次の round へ・ゲート失敗と round 失敗で停止・
#   開始前 TIMEOUT は次の round へ・ロック使用中・
#   引数の拒否・load1 が 1.5 未満に届かない場合の記録を確かめる。判定基準そのものの self-test は
#   `python3 -I foreign_job_gate.py --self-test`。
# 使い方: bash test_round_with_foreign_gate.sh   （成功で "self-test ok"・exit 0）
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)" || exit 2
T="$(mktemp -d)" || exit 2
trap 'rm -rf "${T}"' EXIT
mkdir -p "${T}/kit" "${T}/bin"
cp "${HERE}/round_with_foreign_gate.sh" "${HERE}/drive_series.sh" "${T}/kit/"
# スタブ: 呼ばれるたびに ${T}/codes の先頭行の終了コードを返し、--out に 1 行書く
cat > "${T}/kit/foreign_job_gate.py" <<'EOF'
import sys, os
codes = os.environ["STUB_CODES"]
with open(codes) as f:
    lines = f.read().split()
rc = int(lines[0]) if lines else 0
with open(codes, "w") as f:
    f.write("\n".join(lines[1:]))
out = sys.argv[sys.argv.index("--out") + 1]
with open(out, "x") as f:
    f.write(("CLEAN" if rc == 0 else "FOREIGN") + " stub\n")
sys.exit(rc)
EOF
printf '#!/bin/sh\nexit 0\n' > "${T}/bin/sleep"; chmod +x "${T}/bin/sleep"
# sysctl のスタブ: ${LOADS} の先頭行を load1 として返す（最後の 1 行は繰り返す）
cat > "${T}/bin/sysctl" <<'STUB'
#!/bin/sh
v=$(head -n 1 "${LOADS}")
if [ "$(wc -l < "${LOADS}")" -gt 1 ]; then tail -n +2 "${LOADS}" > "${LOADS}.n" && mv "${LOADS}.n" "${LOADS}"; fi
echo "{ ${v} ${v} ${v} }"
STUB
chmod +x "${T}/bin/sysctl"
# 偽の run_round.sh: gate.txt に PASS を書き、FAKE_RC で終わる（FAKE_NOGATE=1 なら gate.txt を書かずに終わる＝precheck で止まった形）
cat > "${T}/run_round.sh" <<'EOF'
[ "${FAKE_NOGATE:-0}" = "1" ] && exit "${FAKE_RC:-1}"
ts=$(python3 -I -c 'import sys, time; print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + int(sys.argv[1]))))' "${FAKE_PASS_SHIFT:-0}")
mkdir -p "$2" && echo "PASS load1=1.00 hw-lock=held ${FAKE_PASS_TS:-${ts}}" > "$2/gate.txt"
exit "${FAKE_RC:-0}"
EOF
fail=0
case_() {  # case_ <名前> <スタブの終了コード列> <FAKE_RC> <期待 exit> <期待 gate 1 語目> <期待 "# was:" 行数>
  local name="$1" codes="$2" frc="$3" wexit="$4" wfirst="$5" wwas="$6" rd="${T}/s-$1/round1" env="${T}/s-$1/env"
  printf '%s' "${codes}" | tr -s ' ' '\n' > "${T}/codes-${name}"
  printf '%s\n' "${CASE_LOADS:-1.00}" | tr -s ' ' '\n' > "${T}/cloads-${name}"
  PATH="${T}/bin:${PATH}" STUB_CODES="${T}/codes-${name}" FAKE_RC="${frc}" FAKE_NOGATE="${FAKE_NOGATE:-0}" LOADS="${T}/cloads-${name}" \
    bash "${T}/kit/round_with_foreign_gate.sh" "${T}/run_round.sh" 1 "${rd}" "${env}" > "${T}/log-${name}" 2>&1
  local got=$? first was
  first=$(awk 'NR==1{print $1}' "${rd}/gate.txt" 2>/dev/null)
  was=$(grep -c '^# was: ' "${rd}/gate.txt" 2>/dev/null)
  if [ "${got}" != "${wexit}" ] || [ "${first:-none}" != "${wfirst}" ] || [ "${was:-0}" != "${wwas}" ]; then
    echo "NG ${name}: exit=${got}（期待 ${wexit}） gate=${first:-none}（期待 ${wfirst}） was=${was:-0}（期待 ${wwas}）"
    sed 's/^/  | /' "${T}/log-${name}"; fail=1
  fi
}
case_ clean        "0 0"   0 0 PASS    0
case_ wait-then-ok "1 1 0 0" 0 0 PASS  0
case_ round-fails  "0 0"   1 1 PASS    0
case_ after-foreign "0 1"  0 4 INVALID 1
case_ after-error  "0 2"   0 5 INVALID 1
case_ fail-and-foreign "0 1" 1 5 INVALID 1
FAKE_NOGATE=1 case_ nogate-foreign "0 1" 1 5 INVALID 0
FAKE_NOGATE=1 case_ nogate-clean "0 0" 1 1 none 0
CASE_LOADS="3.50 1.00" case_ load-wait "0 0 0" 0 0 PASS 0
CASE_LOADS="x 1.00" case_ load-nonnum "0 0 0" 0 0 PASS 0
CASE_LOADS="3.50" case_ load-timeout "$(printf '0 %.0s' $(seq 1 60))" 0 1 TIMEOUT 0
FAKE_PASS_SHIFT=120 case_ gap-late "0 0" 0 4 INVALID 1
FAKE_PASS_SHIFT=-120 case_ gap-early "0 0" 0 4 INVALID 1
FAKE_PASS_SHIFT=30 case_ gap-within "0 0" 0 0 PASS 0
FAKE_PASS_TS=garbage case_ gap-unreadable "0 0" 0 4 INVALID 1
gl() { head -n 1 "${T}/s-$1/round1/gate.txt" | awk '{print $1, $2}'; }
[ "$(gl load-timeout)" = "TIMEOUT load1" ] || { echo "NG load-timeout: TIMEOUT の理由が load1 でない"; fail=1; }
[ "$(gl gap-late)" = "INVALID foreign-gap" ] || { echo "NG gap-late: foreign-gap でない"; fail=1; }
[ "$(find "${T}/s-load-wait/env/foreign-gate" -name 'r1-before-a*' | wc -l | tr -d ' ')" = "2" ] || { echo "NG load-wait: 待った回の記録が 2 件でない"; fail=1; }
grep -q '^wrapper_load1 3.50$' "${T}/s-load-wait/env/foreign-gate/r1-before-a1.txt" || { echo "NG load-wait: load1 を記録していない"; fail=1; }
case_ before-error "2"     0 1 INVALID 0
case_ before-timeout "$(printf '1 %.0s' $(seq 1 60))" 0 1 TIMEOUT 0
[ "$(gl before-timeout)" = "TIMEOUT foreign-job" ] || { echo "NG before-timeout: TIMEOUT の理由が foreign-job でない"; fail=1; }
# 開始前の待ちは各回のファイルを残す（wait-then-ok は a1〜a3）
[ "$(find "${T}/s-wait-then-ok/env/foreign-gate" -name 'r1-before-a*' | wc -l | tr -d ' ')" = "3" ] || { echo "NG before の記録が 3 件でない"; fail=1; }
[ -f "${T}/s-wait-then-ok/env/foreign-gate/r1-after.txt" ] || { echo "NG after の記録がない"; fail=1; }
# 開始前で止まった round は run_round.sh を呼ばない（raw 等を作らない）・after を記録しない
[ ! -e "${T}/s-before-timeout/env/foreign-gate/r1-after.txt" ] || { echo "NG TIMEOUT なのに after を記録した"; fail=1; }
[ "$(find "${T}/s-before-timeout/env/foreign-gate" -name 'r1-before-a*' | wc -l | tr -d ' ')" = "60" ] || { echo "NG TIMEOUT までの記録が 60 件でない"; fail=1; }
# 既に gate.txt がある round ディレクトリは拒否する
mkdir -p "${T}/s-dup/round1" && echo "PASS x" > "${T}/s-dup/round1/gate.txt"
printf '0\n0\n' > "${T}/codes-dup"
STUB_CODES="${T}/codes-dup" bash "${T}/kit/round_with_foreign_gate.sh" "${T}/run_round.sh" 1 "${T}/s-dup/round1" "${T}/s-dup/env" > /dev/null 2>&1
[ $? -eq 2 ] || { echo "NG 既存 gate.txt を拒否しない"; fail=1; }
# 相対パスは拒否する
STUB_CODES="${T}/codes-dup" bash "${T}/kit/round_with_foreign_gate.sh" run_round.sh 1 "${T}/x/round1" "${T}/x/env" > /dev/null 2>&1
[ $? -eq 2 ] || { echo "NG 相対パスを拒否しない"; fail=1; }
# ---- drive_series.sh
# 偽の run_round.sh（ドライバ用）: 受け取った環境変数を残し、gate.txt に FAKE_GATE（既定 PASS の行）を書いて FAKE_RC で終わる
cat > "${T}/run_round_d.sh" <<'STUB'
mkdir -p "$2" && echo "${FAKE_GATE:-PASS load1=1.00 hw-lock=held $(date -u +%FT%TZ)}" > "$2/gate.txt"
echo "${HW_LOCK_HELD:-} ${PASSVAR:-} $([ -s "${HW_LOCK_OWNER_FILE:-/nonexistent}" ] && echo owner-ok)" > "$2/env.seen"
exit "${FAKE_RC:-0}"
STUB
dcase() {  # dcase <名前> <ゲートの終了コード列> <load1 列> <rounds> <期待 exit> [追加引数...]
  local name="$1" codes="$2" loads="$3" n="$4" wexit="$5"; shift 5
  local h="${T}/h-${name}"; mkdir -p "${h}/work"
  printf '%s' "${codes}" | tr -s ' ' '\n' > "${T}/dcodes-${name}"
  printf '%s\n' "${loads}" | tr -s ' ' '\n' > "${T}/loads-${name}"
  HOME="${h}" PATH="${T}/bin:${PATH}" STUB_CODES="${T}/dcodes-${name}" LOADS="${T}/loads-${name}" \
    bash "${T}/kit/drive_series.sh" 4153 "${h}/ser" "${n}" "${T}/run_round_d.sh" "$@" > "${T}/dlog-${name}" 2>&1
  local got=$?
  [ "${got}" = "${wexit}" ] || { echo "NG drive ${name}: exit=${got}（期待 ${wexit}）"; sed 's/^/  | /' "${T}/dlog-${name}"; fail=1; }
  if [ "${name}" != "busy" ] && [ -e "${h}/work/.fandhe-hw-lock" ]; then echo "NG drive ${name}: ロックが残った"; fail=1; fi
}
dfirst() { awk 'NR==1{print $1}' "${T}/h-$1/ser/round$2/gate.txt" 2>/dev/null; }
dcase ok "0 0 0 0" "2.00 1.20" 2 0 PASSVAR=x
grep -q '^all done' "${T}/h-ok/ser/drive.status" || { echo "NG drive ok: all done がない"; fail=1; }
grep -q 'waited_s=.* load1=1.20 reached_below_1.5=yes' "${T}/h-ok/ser/env/series-start.txt" || { echo "NG drive ok: series-start"; fail=1; }
[ "$(cat "${T}/h-ok/ser/round2/env.seen" 2>/dev/null)" = "1 x owner-ok" ] || { echo "NG drive ok: 環境変数が渡らない"; fail=1; }
[ "$(dfirst ok 2)" = "PASS" ] || { echo "NG drive ok: round2"; fail=1; }
dcase afterforeign "0 1 0 0" "1.00" 2 0
if [ "$(dfirst afterforeign 1)" != "INVALID" ] || [ "$(dfirst afterforeign 2)" != "PASS" ] || ! grep -q '^all done' "${T}/h-afterforeign/ser/drive.status"; then
  echo "NG drive afterforeign: 終了後 FOREIGN の round を INVALID にして次へ進まない"; fail=1
fi
dcase aftererror "0 2 0 0" "1.00" 2 1
if [ "$(dfirst aftererror 1)" != "INVALID" ] || [ -e "${T}/h-aftererror/ser/round2" ] || ! grep -q '^STOP round 1' "${T}/h-aftererror/ser/drive.status"; then
  echo "NG drive aftererror: 終了後ゲートの失敗で止まらない"; fail=1
fi
FAKE_GATE="PASS load1=1.00 hw-lock=held 2000-01-01T00:00:00Z" dcase gap "0 0 0 0" "1.00" 2 0
if [ "$(dfirst gap 1)" != "INVALID" ] || ! grep -q '^all done' "${T}/h-gap/ser/drive.status"; then
  echo "NG drive gap: foreign-gap の round を INVALID にして次へ進まない"; fail=1
fi
FAKE_RC=1 dcase roundfail "0 1 0 0" "1.00" 2 1
[ ! -e "${T}/h-roundfail/ser/round2" ] || { echo "NG drive roundfail: round 本体の失敗＋終了後 FOREIGN で止まらない"; fail=1; }
FAKE_RC=4 FAKE_GATE="INVALID run_round-exit=4 last-step=x" dcase rr4 "0 0 0 0" "1.00" 2 1
[ ! -e "${T}/h-rr4/ser/round2" ] || { echo "NG drive rr4: round 本体の終了コード 4 を他ジョブと取り違えた"; fail=1; }
dcase timeout "$(printf '1 %.0s' $(seq 1 60))0 0" "1.00" 2 0
if [ "$(dfirst timeout 1)" != "TIMEOUT" ] || [ "$(dfirst timeout 2)" != "PASS" ]; then echo "NG drive timeout: TIMEOUT の次へ進まない"; fail=1; fi
dcase noreach "0 0" "2.00" 1 0
if [ "$(grep -c '^sample ' "${T}/h-noreach/ser/env/series-start.txt")" != "40" ] \
  || ! grep -q 'reached_below_1.5=no' "${T}/h-noreach/ser/env/series-start.txt"; then
  echo "NG drive noreach: 40 回で始めない"; fail=1
fi
dcase nonnum "0 0" "x 1.00" 1 0
grep -q 'load1=1.00 reached_below_1.5=yes' "${T}/h-nonnum/ser/env/series-start.txt" || { echo "NG drive nonnum: 数値でない load1 を届いたとみなした"; fail=1; }
mkdir -p "${T}/h-busy/work/.fandhe-hw-lock"
dcase busy "0 0" "1.00" 1 3
[ ! -e "${T}/h-busy/ser" ] || { echo "NG drive busy: ロック使用中に系列ディレクトリを作った"; fail=1; }
dcase badarg "0 0" "1.00" 1 2 notakv
dcase badname "0 0" "1.00" 1 2 lower=1
mkdir -p "${T}/h-exists/ser"
dcase exists "0 0" "1.00" 1 2
[ "${fail}" -eq 0 ] && echo "self-test ok"
exit "${fail}"
