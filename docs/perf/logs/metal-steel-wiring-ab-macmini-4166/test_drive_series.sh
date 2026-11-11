#!/bin/bash
# drive_series.sh・prereq_tests.sh・fp.sh の self-test（実機・GPU・hw-lock の実体を使わない）。
# ps・sysctl・sleep・log・sw_vers・rustc・cargo を PATH の先頭の偽物に差し替え、HOME を一時ディレクトリにする。
# 他ジョブのゲートと runner は一時の展開ツリーに置いた偽物。確かめること:
#   - 確認の各段（ゲート FOREIGN／失敗・Safari・mediaanalysisd の境界と非数値・hw-lock 使用中・ツリー不一致）で
#     計測を起動せず、試みを prestart-<k>-<n>.txt に残し、formal-<k>/ を作らず、hw-lock を残さないこと
#   - ツリー不一致のとき __pycache__ の一覧を残すこと
#   - 全ゲート通過の判定（rc=0・全行 gate=pass・skipped が空）と、それぞれが崩れたときの notpass
#   - runner へ AB_DIRECTION=post-wiring・PYTHONDONTWRITEBYTECODE=1・FOREIGN_GATE・両 facade のパスを渡すこと
#   - prereq_tests.sh: 件数の一致・0 件の単位・--exact の名前違い（0 件）・失敗を見分けること
# 使い方: bash test_drive_series.sh
# check の第 2 引数は eval する文字列で、その中で使う変数（r・FCR・base・rc・PRC・PW）を shellcheck は見られない。
# shellcheck disable=SC2034
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
T=$(mktemp -d "${TMPDIR:-/tmp}/t4166.XXXXXX")
trap 'rm -rf "${T}"' EXIT
FAILS=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; FAILS=$((FAILS + 1)); fi; }

BIN="${T}/bin"; mkdir -p "${BIN}" "${T}/home/work"
cat > "${BIN}/ps" <<'EOF'
#!/bin/bash
[ -n "${PS_FAIL:-}" ] && exit 1
printf '%b' "${PS_OUT:-  0.0 /usr/sbin/cfprefsd\n}"
EOF
cat > "${BIN}/sysctl" <<'EOF'
#!/bin/bash
case "$*" in *vm.loadavg*) echo "{ ${LOAD1:-1.00} 1.00 1.00 }";; *hw.model*) echo Mac16,10;; *) exit 1;; esac
EOF
printf '#!/bin/bash\nexit 0\n' > "${BIN}/sleep"
printf '#!/bin/bash\necho header\n' > "${BIN}/log"
printf '#!/bin/bash\necho 26.1\n' > "${BIN}/sw_vers"
printf '#!/bin/bash\necho rustc 1.95.0\n' > "${BIN}/rustc"
chmod +x "${BIN}"/*
export PATH="${BIN}:${PATH}" HOME="${T}/home"
LOCK="${HOME}/work/.fandhe-hw-lock"

# 一時の展開ツリー W（deploy.sh が作る形のうち drive_series.sh が読むものだけ）
mkw() { # mkw <名前> -> W
  local w="${T}/$1.noindex"
  mkdir -p "${w}/tree/docs/perf/logs/macmini-foreign-job-gate" "${w}/tree/scripts/bench/framework-compare" "${w}/rule" \
    "${w}/before/crates/facade" "${w}/after/crates/facade"
  cat > "${w}/tree/docs/perf/logs/macmini-foreign-job-gate/foreign_job_gate.py" <<'EOF'
import os, sys
out = sys.argv[sys.argv.index("--out") + 1]
rc = int(os.environ.get("FAKE_FG_RC", "0"))
if rc != 2:
    open(out, "w").write(("FOREIGN" if rc else "CLEAN") + " before x\n")
sys.exit(rc)
EOF
  cat > "${w}/tree/scripts/bench/framework-compare/run_ab_steel_large_n_gemm_metal.sh" <<'EOF'
#!/bin/bash
mkdir -p results/raw
L=$1
env | LC_ALL=C sort > "results/raw/env-${L}.txt"
g="results/raw/load-gate-4160-metal-${L}.log"
echo "threshold=3.0 rule_threshold=3.0 series=official" > "${g}"
for r in 1 2 3 4 5; do echo "round${r} gate=pass threshold=3.0 load1=1.0 waited_s=0 at=x" >> "${g}"; done
[ -n "${FAKE_GATE_EXTRA:-}" ] && echo "${FAKE_GATE_EXTRA}" >> "${g}"
printf '%s' "${FAKE_SKIP:-}" > "results/raw/skipped-4160-metal-${L}.log"
exit "${FAKE_AB_RC:-0}"
EOF
  echo 'x' > "${w}/tree/scripts/bench/framework-compare/compare_gemm_ab.py"
  cp "${HERE}/fp.sh" "${HERE}/drive_series.sh" "${w}/rule/"
  bash "${w}/rule/fp.sh" "${w}/tree" ab > "${w}/fp-deploy-ab.txt"
  ( cd "${w}/tree" && find . -type f | LC_ALL=C sort | tr '\n' '\0' | xargs -0 shasum -a 256 ) > "${w}/manifest-deploy.txt"
  echo "${w}"
}
drive() { bash "$1/rule/drive_series.sh" "$@" > "${T}/last.out" 2>&1; }
last_line() { tail -1 "$1/drive.log" 2>/dev/null; }

# --- dry: 全部通る
W=$(mkw dry)
drive "${W}" 1 1 dry; rc=$?
check "dry: rc 0" '[ "${rc}" = 0 ]'
check "dry: dry-ok" 'grep -q "^result=dry-ok" "${W}/precheck-dry-1.txt"'
check "dry: 計測を起動しない" '[ ! -d "${W}/formal-1" ] && [ ! -d "${W}/tree/scripts/bench/framework-compare/results" ]'
check "dry: hw-lock を残さない" '[ ! -e "${LOCK}" ]'
check "dry: 確認の順序" '[ "$(grep -oE "^(proc-check1|mediaanalysisd check1|lock-acquired|proc-check2|tree-check)" "${W}/precheck-dry-1.txt" | tr "\n" ,)" = "proc-check1,mediaanalysisd check1,lock-acquired,proc-check2,tree-check," ]'

# --- 始めない分岐（run モード）
notstart_case() { # notstart_case <名前> <期待する reason>  （環境変数は呼び出し側で設定）
  local w r=$2; w=$(mkw "$1")
  drive "${w}" 1 1 run; local rc=$?
  check "$1: rc 5" '[ "${rc}" = 5 ]'
  check "$1: reason=$2" 'grep -q "^result=not-started reason=${r} " "${w}/prestart-1-1.txt"'
  check "$1: formal-1 を作らない" '[ ! -e "${w}/formal-1" ] && [ ! -d "${w}/tree/scripts/bench/framework-compare/results" ]'
  check "$1: drive.log に NOT-STARTED" 'grep -q "^formal-1 attempt 1 NOT-STARTED reason=${r} " "${w}/drive.log"'
  check "$1: hw-lock を残さない" '[ ! -e "${LOCK}" ]'
}
FAKE_FG_RC=1 notstart_case foreign proc-check1
FAKE_FG_RC=2 notstart_case gateerr proc-check1
PS_OUT='  0.0 /Applications/Safari.app/Contents/MacOS/Safari\n' notstart_case safari proc-check1
PS_OUT='  5.0 /System/Library/PrivateFrameworks/MediaAnalysis.framework/mediaanalysisd\n' notstart_case ma5 mediaanalysisd
PS_OUT='  1.0 /x/mediaanalysisd\n  7.5 /y/mediaanalysisd\n' notstart_case mamax mediaanalysisd
PS_OUT='  abc /x/mediaanalysisd\n' notstart_case manan mediaanalysisd
W=$(mkw masub); PS_OUT='  50.0 /x/mediaanalysisd-helper\n  50.0 /x/com.apple.mediaanalysisd.x\n' drive "${W}" 1 1 dry; rc=$?
check "masub: 名前の部分一致は mediaanalysisd と見なさない" '[ "${rc}" = 0 ] && grep -q "^mediaanalysisd check1 .* pcpu=absent$" "${W}/precheck-dry-1.txt"'
mkdir "${LOCK}"; echo other > "${LOCK}/owner"
W=$(mkw busy); drive "${W}" 1 1 run; rc=$?
check "busy: rc 5・lock-busy" '[ "${rc}" = 5 ] && grep -q "reason=lock-busy" "${W}/prestart-1-1.txt"'
check "busy: 他人の hw-lock を消さない" '[ "$(cat "${LOCK}/owner")" = other ]'
rm -rf "${LOCK}"
W=$(mkw pyc); mkdir "${W}/tree/scripts/bench/framework-compare/__pycache__"
echo z > "${W}/tree/scripts/bench/framework-compare/__pycache__/checksum_contract.cpython-312.pyc"
drive "${W}" 2 1 run; rc=$?
check "pyc: tree-mismatch" '[ "${rc}" = 5 ] && grep -q "reason=tree-mismatch" "${W}/prestart-2-1.txt"'
check "pyc: __pycache__ の一覧を残す" 'grep -q "__pycache__/checksum_contract.cpython-312.pyc" "${W}/prestart-2-1.txt"'
check "pyc: __pycache__ を消さない" '[ -f "${W}/tree/scripts/bench/framework-compare/__pycache__/checksum_contract.cpython-312.pyc" ]'
# 2 回目の確認だけが落ちる（1 回目の後に Safari が起動した）は、ps の偽物を回数で切り替えて確かめる
cat > "${BIN}/ps" <<'EOF'
#!/bin/bash
c=$(cat "${HOME}/ps.count" 2>/dev/null || echo 0); c=$((c + 1)); echo "${c}" > "${HOME}/ps.count"
if [ "${c}" -gt 5 ]; then printf '  0.0 /Applications/Safari.app/Contents/MacOS/Safari\n'; else printf '  0.0 /usr/sbin/cfprefsd\n'; fi
EOF
chmod +x "${BIN}/ps"; rm -f "${HOME}/ps.count"
W=$(mkw second); drive "${W}" 1 1 run; rc=$?
check "second: proc-check2 で始めない" '[ "${rc}" = 5 ] && grep -q "reason=proc-check2" "${W}/prestart-1-1.txt"'
check "second: hw-lock を外した記録" 'grep -q "^lock released" "${W}/prestart-1-1.txt" && [ ! -e "${LOCK}" ]'
cat > "${BIN}/ps" <<'EOF'
#!/bin/bash
[ -n "${PS_FAIL:-}" ] && exit 1
printf '%b' "${PS_OUT:-  0.0 /usr/sbin/cfprefsd\n}"
EOF
chmod +x "${BIN}/ps"

# --- 起動する系列と全ゲート通過の判定
W=$(mkw run1); drive "${W}" 1 1 run; rc=$?
FCR="${W}/tree/scripts/bench/framework-compare/results/raw"
check "run1: verdict=pass" 'last_line "${W}" | grep -q "^formal-1 verdict=pass ab_rc0=1 ab_gate=5/5 skipped_bytes=0 "'
check "run1: series-start.txt に移す" '[ -f "${W}/formal-1/series-start.txt" ] && [ ! -e "${W}/attempt-1-1.txt" ]'
check "run1: ab rc=0 を状態に残す" 'grep -q "^ab rc=0 " "${W}/formal-1.status"'
check "run1: ラベルは formal-1" '[ -f "${FCR}/load-gate-4160-metal-formal-1.log" ]'
check "run1: AB_DIRECTION=post-wiring" 'grep -qx "AB_DIRECTION=post-wiring" "${FCR}/env-formal-1.txt"'
check "run1: PYTHONDONTWRITEBYTECODE=1" 'grep -qx "PYTHONDONTWRITEBYTECODE=1" "${FCR}/env-formal-1.txt"'
check "run1: FOREIGN_GATE は展開ツリーのもの" 'grep -qx "FOREIGN_GATE=${W}/tree/docs/perf/logs/macmini-foreign-job-gate/foreign_job_gate.py" "${FCR}/env-formal-1.txt"'
check "run1: before／after の facade" 'grep -qx "AB_BEFORE_FACADE_PATH=${W}/before/crates/facade" "${FCR}/env-formal-1.txt" && grep -qx "AB_AFTER_FACADE_PATH=${W}/after/crates/facade" "${FCR}/env-formal-1.txt"'
check "run1: CARGO_TARGET_DIR を渡さない" '! grep -q "^CARGO_TARGET_DIR=" "${FCR}/env-formal-1.txt"'
check "run1: hw-lock を外す" '[ ! -e "${LOCK}" ]'
# 2 系列目: 1 系列目の results/ があってもツリー照合は合う（除外の対象）
drive "${W}" 2 1 run; rc=$?
check "run1→2: 2 系列目も起動する" 'last_line "${W}" | grep -q "^formal-2 verdict=pass "'
drive "${W}" 2 2 run; rc=$?
check "同じ系列は 2 度起動しない" '[ "${rc}" = 2 ]'
W=$(mkw abrc); FAKE_AB_RC=1 drive "${W}" 1 1 run
check "ab rc=1 は notpass" 'last_line "${W}" | grep -q "^formal-1 verdict=notpass ab_rc0=0 "'
W=$(mkw cellgate); FAKE_GATE_EXTRA="round3 gate=invalid reason=foreign-after phase=after cell=candle-N4096-fresh at=x" drive "${W}" 1 1 run
check "セルの foreign-after は notpass" 'last_line "${W}" | grep -q "verdict=notpass ab_rc0=1 ab_gate=5/6 "'
W=$(mkw skip); FAKE_SKIP="compare: 比較不能" drive "${W}" 1 1 run
check "skipped が空でなければ notpass" 'last_line "${W}" | grep -q "verdict=notpass .*skipped_bytes=[1-9]"'
W=$(mkw passlike); FAKE_GATE_EXTRA="round3 gate=pass-reference threshold=x" drive "${W}" 1 1 run
check "gate=pass-reference は pass と数えない" 'last_line "${W}" | grep -q "verdict=notpass ab_rc0=1 ab_gate=5/6 "'

# --- 引数
W=$(mkw args)
bash "${W}/rule/drive_series.sh" "${W}" 4 1 run > /dev/null 2>&1; rc=$?
check "k=4 を拒否" '[ "${rc}" = 2 ]'
bash "${W}/rule/drive_series.sh" "${W}" 1 x run > /dev/null 2>&1; rc=$?
check "n が数字でなければ拒否" '[ "${rc}" = 2 ]'

# --- fp.sh
F="${T}/fptree"; mkdir -p "${F}/scripts/bench/framework-compare/results" "${F}/.git"
echo a > "${F}/a"; echo r > "${F}/.rev-stamp"; echo g > "${F}/.git/x"
base=$(bash "${HERE}/fp.sh" "${F}")
echo b > "${F}/scripts/bench/framework-compare/results/r"; mkdir -p "${F}/scripts/bench/framework-compare/target-ab-4160-formal-1-before"
echo c > "${F}/scripts/bench/framework-compare/target-ab-4160-formal-1-before/c"
check "fp: .git・.rev-stamp を除く" '[ "$(echo "${base}" | sed "s/.*files=//")" = 1 ]'
check "fp ab: results と target-ab-4160-* を除く" '[ "$(bash "${HERE}/fp.sh" "${F}" ab)" = "${base}" ]'
check "fp: ab なしでは除かない" '[ "$(bash "${HERE}/fp.sh" "${F}")" != "${base}" ]'
mkdir -p "${F}/x/__pycache__"; echo p > "${F}/x/__pycache__/m.pyc"
check "fp ab: __pycache__ は除かない" '[ "$(bash "${HERE}/fp.sh" "${F}" ab)" != "${base}" ]'
bash "${HERE}/fp.sh" "${F}" bogus > /dev/null 2>&1; rc=$?
check "fp: 不明な引数を拒否" '[ "${rc}" = 2 ]'

# --- prereq_tests.sh（cargo の偽物。単位ごとの list の件数と本実行の結果を FAKE_* で決める）
cat > "${BIN}/cargo" <<'EOF'
#!/bin/bash
args=" $* "
unit=lib
case "${args}" in *" --test "*) unit=$(echo "${args}" | sed 's/.* --test \([^ ]*\) .*/\1/');; esac
case "${args}" in *steel_large_n_default_is_on*) unit=steel;; *gemm_tile_macmini_table_diag_tests::*) unit=table;; esac
listed=3; [ "${unit}" = linalg_parity_inputs_guard ] && listed=0
[ "${unit}" = steel ] && listed=${FAKE_STEEL_LISTED:-2}
case "${args}" in
  *" --list "*) i=0; while [ "${i}" -lt "${listed}" ]; do echo "t${i}: test"; i=$((i + 1)); done; exit 0;;
esac
passed=${listed}; failed=0
if [ "${unit}" = "${FAKE_FAIL_UNIT:-none}" ]; then passed=$((listed - 1)); failed=1; fi
if [ "${unit}" = "${FAKE_SHORT_UNIT:-none}" ]; then passed=$((listed - 1)); fi
echo "test result: $([ "${failed}" = 0 ] && echo ok || echo FAILED). ${passed} passed; ${failed} failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s"
[ "${failed}" = 0 ]
EOF
chmod +x "${BIN}/cargo"
pq() { local w="${T}/pq-$1.noindex"; mkdir -p "${w}/after"; shift; env "$@" bash "${HERE}/prereq_tests.sh" "${w}" > /dev/null 2>&1; PRC=$?; PW=${w}; }
pq ok FAKE_NONE=1
check "prereq: all-pass" '[ "${PRC}" = 0 ] && grep -q "^all-pass " "${PW}/prereq/result.txt"'
check "prereq: 61 単位" '[ "$(($(wc -l < "${PW}/prereq/summary.tsv") - 1))" = 61 ]'
check "prereq: 0 件の単位は pass-no-ignored" 'grep -q "^linalg_parity_inputs_guard	0	0	0	0	pass-no-ignored$" "${PW}/prereq/summary.tsv"'
check "prereq: hw-lock を外す" '[ ! -e "${LOCK}" ]'
pq fail FAKE_FAIL_UNIT=gemm_splitk_parity
check "prereq: 失敗を拾う" '[ "${PRC}" = 1 ] && grep -q "^fail " "${PW}/prereq/result.txt" && grep -q "^gemm_splitk_parity	3	2	1	1	fail$" "${PW}/prereq/summary.tsv"'
pq short FAKE_SHORT_UNIT=softmax_parity
check "prereq: passed が件数に届かなければ fail" '[ "${PRC}" = 1 ] && grep -q "^softmax_parity	3	2	0	0	fail$" "${PW}/prereq/summary.tsv"'
pq steel0 FAKE_STEEL_LISTED=0
check "prereq: --exact で 0 件なら fail" '[ "${PRC}" = 1 ]'
mkdir "${LOCK}"; pq busy FAKE_NONE=1
check "prereq: hw-lock 使用中なら流さない" '[ "${PRC}" = 5 ] && grep -q "^lock-busy" "${PW}/prereq/result.txt" && [ ! -e "${PW}/prereq/summary.tsv" ]'
rm -rf "${LOCK}"

if [ "${FAILS}" -ne 0 ]; then echo "self-test FAILED: ${FAILS}"; exit 1; fi
echo "self-test ok"
