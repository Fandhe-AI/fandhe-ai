#!/usr/bin/env bash
# round_with_foreign_gate.sh の経路の self-test（実機・ロック不要。macOS／Linux の bash で動く）。
# 役割: 一時ディレクトリへラッパーを複製し、foreign_job_gate.py をスタブ（終了コードを順に返す）に、sleep を即時終了の
#   スタブに差し替えて、開始前 CLEAN・開始前 TIMEOUT・開始前のゲート失敗・終了後 FOREIGN・終了後のゲート失敗・
#   round 本体の失敗の各経路で、gate.txt の 1 語目とラッパーの終了コードを確かめる。判定基準そのものの self-test は
#   `python3 -I foreign_job_gate.py --self-test`。
# 使い方: bash test_round_with_foreign_gate.sh   （成功で "self-test ok"・exit 0）
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)" || exit 2
T="$(mktemp -d)" || exit 2
trap 'rm -rf "${T}"' EXIT
mkdir -p "${T}/kit" "${T}/bin"
cp "${HERE}/round_with_foreign_gate.sh" "${T}/kit/"
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
# 偽の run_round.sh: gate.txt に PASS を書き、FAKE_RC で終わる
cat > "${T}/run_round.sh" <<'EOF'
mkdir -p "$2" && echo "PASS load1=1.00 hw-lock=held 2026-01-01T00:00:00Z" > "$2/gate.txt"
exit "${FAKE_RC:-0}"
EOF
fail=0
case_() {  # case_ <名前> <スタブの終了コード列> <FAKE_RC> <期待 exit> <期待 gate 1 語目> <期待 "# was:" 行数>
  local name="$1" codes="$2" frc="$3" wexit="$4" wfirst="$5" wwas="$6" rd="${T}/s-$1/round1" env="${T}/s-$1/env"
  printf '%s' "${codes}" | tr -s ' ' '\n' > "${T}/codes-${name}"
  PATH="${T}/bin:${PATH}" STUB_CODES="${T}/codes-${name}" FAKE_RC="${frc}" \
    bash "${T}/kit/round_with_foreign_gate.sh" "${T}/run_round.sh" 1 "${rd}" "${env}" > "${T}/log-${name}" 2>&1
  local got=$? first was
  first=$(awk 'NR==1{print $1}' "${rd}/gate.txt" 2>/dev/null)
  was=$(grep -c '^# was: ' "${rd}/gate.txt" 2>/dev/null)
  if [ "${got}" != "${wexit}" ] || [ "${first}" != "${wfirst}" ] || [ "${was:-0}" != "${wwas}" ]; then
    echo "NG ${name}: exit=${got}（期待 ${wexit}） gate=${first:-none}（期待 ${wfirst}） was=${was:-0}（期待 ${wwas}）"
    sed 's/^/  | /' "${T}/log-${name}"; fail=1
  fi
}
case_ clean        "0 0"   0 0 PASS    0
case_ wait-then-ok "1 1 0 0" 0 0 PASS  0
case_ round-fails  "0 0"   1 1 PASS    0
case_ after-foreign "0 1"  0 4 INVALID 1
case_ after-error  "0 2"   0 4 INVALID 1
case_ before-error "2"     0 1 INVALID 0
case_ before-timeout "$(printf '1 %.0s' $(seq 1 60))" 0 1 TIMEOUT 0
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
[ "${fail}" -eq 0 ] && echo "self-test ok"
exit "${fail}"
