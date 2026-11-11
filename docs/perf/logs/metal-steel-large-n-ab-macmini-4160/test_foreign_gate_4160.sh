#!/usr/bin/env bash
# #4160 の起動スクリプト 2 本に組み込んだ他ジョブの前提ゲート（scripts/bench/framework-compare/ab_foreign_gate.sh）の self-test。
# 役割: foreign_job_gate.py を偽物（呼び出し番号ごとに CLEAN／FOREIGN／失敗／出力不一致を返す）に、sleep・sysctl・cargo・
#   rustc・sw_vers をスタブに差し替え、実機・ロック・ビルドなしで次を確かめる（実機には接続しない）:
#   - orchestrate_macmini.sh を一時ツリーで通しで実行し、load_gate.log が aggregate.py の parse_gate（変更なし）で
#     全 CLEAN なら正式系列、終了後 FOREIGN（foreign-after）・開始前 60 回 FOREIGN（timeout）・load1 超過（timeout）・
#     ゲート失敗・出力不一致（fail-closed）なら正式系列にならないこと（正のケースと負のケースの両方）
#   - run_ab_steel_large_n_gemm_metal.sh の wait_load_gate／cell_gate_before／cell_gate_after を本体から切り出して実行し、
#     通過だけなら `round<N> gate=pass` の 5 行のまま（aggregate.py の ab 判定が正式）、異常なら正式でなくなること
#   - FOREIGN_GATE 未指定・相対パスで起動しないこと
# 使い方: bash test_foreign_gate_4160.sh   （成功で "self-test ok"・exit 0。macOS／Linux の bash で動く）
# shellcheck disable=SC2034,SC2086,SC2012,SC1091  # 切り出した関数が読む変数・意図した単語分割・ls の件数
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)" || exit 2
ROOT="$(cd "${HERE}/../../../.." && pwd -P)" || exit 2
FC="${ROOT}/scripts/bench/framework-compare"
T="$(mktemp -d)" || exit 2
trap 'rm -rf "${T}"' EXIT
fail=0
ng() { echo "NG: $*"; fail=1; }

mkdir -p "${T}/bin"
# 偽ゲート: ${STUB_RULES} の行 "N rc" または "N-M rc"（rc: 0 CLEAN・1 FOREIGN・2 失敗・m 終了コード 0 だが 1 行目が FOREIGN）で
# N 回目の呼び出しの結果を決める（既定 CLEAN）。呼び出し回数は ${STUB_COUNT} に残す。
cat > "${T}/fake_gate.py" <<'EOF'
import os, sys
cf, rf = os.environ["STUB_COUNT"], os.environ.get("STUB_RULES", "")
n = (int(open(cf).read()) if os.path.exists(cf) else 0) + 1
open(cf, "w").write(str(n))
rc = "0"
if rf and os.path.exists(rf):
    for ln in open(rf):
        p = ln.split()
        if len(p) != 2:
            continue
        lo, _, hi = p[0].partition("-")
        if int(lo) <= n <= int(hi or lo):
            rc = p[1]
phase = sys.argv[sys.argv.index("--phase") + 1]
out = sys.argv[sys.argv.index("--out") + 1]
if rc == "2":
    sys.exit(2)
with open(out, "x") as f:
    f.write(("FOREIGN" if rc in ("1", "m") else "CLEAN") + " " + phase + " stub\n")
sys.exit(1 if rc == "1" else 0)
EOF
# sleep は回数だけ数えて即時終了。sysctl は ${LOADS} の先頭行を load1 として返す（最後の 1 行は繰り返す）
cat > "${T}/bin/sleep" <<'EOF'
#!/bin/sh
echo x >> "${SLEEP_COUNT}"
exit 0
EOF
cat > "${T}/bin/sysctl" <<'EOF'
#!/bin/sh
if [ "$1" = "-n" ] && [ "$2" = "vm.loadavg" ]; then
  v=$(head -n 1 "${LOADS}")
  if [ "$(wc -l < "${LOADS}")" -gt 1 ]; then tail -n +2 "${LOADS}" > "${LOADS}.n" && mv "${LOADS}.n" "${LOADS}"; fi
  echo "{ ${v} ${v} ${v} }"
fi
exit 0
EOF
cat > "${T}/bin/cargo" <<'EOF'
#!/bin/sh
case "$1" in
  build) mkdir -p "${CARGO_TARGET_DIR}/release" && printf '#!/bin/sh\nexit 0\n' > "${CARGO_TARGET_DIR}/release/bench-candle" && chmod +x "${CARGO_TARGET_DIR}/release/bench-candle" ;;
  tree) echo "+-- candle-core v0.11.0" ;;
  *) echo "test result: ok." ;;
esac
exit 0
EOF
for c in rustc sw_vers; do printf '#!/bin/sh\necho stub\n' > "${T}/bin/${c}"; done
chmod +x "${T}"/bin/*

# 集計器（変更なし）で load_gate.log が正式系列かを判定する（parse_gate を直接呼ぶ）
official() { # official <load_gate.log> -> true|false
  python3 -I - "${HERE}/aggregate.py" "$1" <<'EOF'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("aggregate_4160", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
ok, _ = m.parse_gate(sys.argv[2])
print("true" if ok else "false")
EOF
}

# ---- orchestrate_macmini.sh の通し実行（一時ツリー）
run_orch() { # run_orch <名前> <rules> <loads> -> ${T}/<名前>/out に出力。終了コードを返す
  local n="$1" tree="${T}/$1/tree"
  mkdir -p "${tree}/docs/perf/logs/x" "${tree}/scripts/bench/framework-compare"
  cp "${HERE}/orchestrate_macmini.sh" "${tree}/docs/perf/logs/x/"
  cp "${FC}/ab_foreign_gate.sh" "${tree}/scripts/bench/framework-compare/"
  echo "fixed-at-utc: 2026-01-01T00:00:00Z" > "${tree}/docs/perf/logs/x/RULE.txt"
  printf 'abc\nparent=def\n' > "${tree}/.rev-stamp"
  printf '%s' "$2" > "${T}/$n/rules"
  printf '%s\n' $3 > "${T}/$n/loads"
  rm -f "${T}/$n/count" "${T}/$n/sleeps"
  (
    cd "${tree}/docs/perf/logs/x" || exit 99
    PATH="${T}/bin:${PATH}" STUB_COUNT="${T}/$n/count" STUB_RULES="${T}/$n/rules" SLEEP_COUNT="${T}/$n/sleeps" \
      LOADS="${T}/$n/loads" CARGO_TARGET_DIR="${T}/$n/target" OUT="${T}/$n/out" FOREIGN_GATE="${4-${T}/fake_gate.py}" \
      sh orchestrate_macmini.sh > "${T}/$n/log" 2>&1
  )
}
calls() { cat "${T}/$1/count" 2> /dev/null || echo 0; }
sleeps() { wc -l < "${T}/$1/sleeps" 2> /dev/null | tr -d ' ' || echo 0; }

# 正: すべて CLEAN → 正式系列。before 39（前提 4 + 計測 35）+ after 35 = 74 回、foreign-gate/ に 74 ファイル
mkdir -p "${T}/clean"
run_orch clean "" "0.5" || { ng "clean: orchestrate が失敗"; sed 's/^/  | /' "${T}/clean/log"; }
[ "$(official "${T}/clean/out/load_gate.log")" = "true" ] || ng "clean: 正式系列にならない"
[ "$(calls clean)" = "74" ] || ng "clean: ゲート呼び出し $(calls clean) 回（期待 74）"
[ "$(ls "${T}/clean/out/foreign-gate" | wc -l | tr -d ' ')" = "74" ] || ng "clean: foreign-gate/ のファイル数"
grep -q '^foreign_job_gate_sha256=[0-9a-f]\{64\}$' "${T}/clean/out/env_info.txt" || ng "clean: env_info に sha256 が無い"
grep -rq "${T}" "${T}/clean/out/load_gate.log" "${T}/clean/out/env_info.txt" && ng "clean: ログにゲートの絶対パスが出た"
grep -q 'reason=' "${T}/clean/out/load_gate.log" && ng "clean: 通過の行に reason が付いた"

# 負: 終了後 FOREIGN（3 回目＝prereq 4 の次の run1-trace の after は 6 回目）
mkdir -p "${T}/after"
run_orch after "6 1" "0.5"
grep -q '^run1-trace gate=invalid reason=foreign-after phase=after ' "${T}/after/out/load_gate.log" || ng "after: foreign-after の行が無い"
grep -q '^series done' "${T}/after/out/load_gate.log" || ng "after: 系列が最後まで進んでいない（既存の GATE-INVALID と同じく続行）"
[ "$(official "${T}/after/out/load_gate.log")" = "false" ] || ng "after: 正式系列になってしまう"

# 負: 開始前に 60 回 FOREIGN → timeout（呼び出し 5〜64 回目）。ちょうど 60 回で打ち切る
mkdir -p "${T}/tmo"
run_orch tmo "5-64 1" "0.5"
grep -q '^run1-trace gate=timeout .*waited_s=1800 reason=foreign-job ' "${T}/tmo/out/load_gate.log" || ng "tmo: timeout foreign-job の行が無い"
[ "$(calls tmo)" = "133" ] || ng "tmo: ゲート呼び出し $(calls tmo) 回（期待 133＝60 回で打ち切り）"
[ "$(official "${T}/tmo/out/load_gate.log")" = "false" ] || ng "tmo: 正式系列になってしまう"

# 負: ゲートは CLEAN でも load1 が 3.0 以上のまま 60 回 → timeout reason=load1（既存の load1 ゲートの timeout と同じ扱い）
mkdir -p "${T}/load"
# 前提 4 回の読み取り 4 行（0.5）→ run1-trace の 60 回は 5.0 → 以降は 0.5
run_orch load "" "0.5 0.5 0.5 0.5 $(yes 5.0 | head -60 | tr '\n' ' ')0.5"
grep -q '^run1-trace gate=timeout load1=5.0 waited_s=1800 reason=load1 ' "${T}/load/out/load_gate.log" || ng "load: timeout load1 の行が無い"
[ "$(official "${T}/load/out/load_gate.log")" = "false" ] || ng "load: 正式系列になってしまう"

# 負: 開始前のゲート失敗（待たずに fail-closed）と終了後のゲート失敗・出力不一致
mkdir -p "${T}/err"
run_orch err "5 2" "0.5"
grep -q '^run1-trace gate=invalid .*reason=foreign-gate-error phase=before ' "${T}/err/out/load_gate.log" || ng "err: before の foreign-gate-error が無い"
[ "$(official "${T}/err/out/load_gate.log")" = "false" ] || ng "err: 正式系列になってしまう"
mkdir -p "${T}/err2"
run_orch err2 "6 2" "0.5"
grep -q '^run1-trace gate=invalid reason=foreign-gate-error phase=after ' "${T}/err2/out/load_gate.log" || ng "err2: after の foreign-gate-error が無い"
[ "$(official "${T}/err2/out/load_gate.log")" = "false" ] || ng "err2: 正式系列になってしまう"
mkdir -p "${T}/mism"
run_orch mism "6 m" "0.5"
grep -q '^run1-trace gate=invalid reason=foreign-gate-error phase=after ' "${T}/mism/out/load_gate.log" || ng "mism: 出力不一致が error 扱いにならない"

# 起動しない: FOREIGN_GATE 未指定・相対パス・存在しないパス
mkdir -p "${T}/nogate"
run_orch nogate "" "0.5" "" && ng "nogate: FOREIGN_GATE 未指定で起動した"
[ ! -e "${T}/nogate/out/load_gate.log" ] || ng "nogate: 計測の記録ができている"
mkdir -p "${T}/rel"
run_orch rel "" "0.5" "fake_gate.py" && ng "rel: 相対パスで起動した"
mkdir -p "${T}/missing"
run_orch missing "" "0.5" "${T}/none.py" && ng "missing: 存在しないパスで起動した"

# dry-run は FOREIGN_GATE なしで通り、計画にゲートが出る
sh "${HERE}/orchestrate_macmini.sh" --dry-run 2>&1 | grep -q '他ジョブの前提ゲート' || ng "dry-run: ゲートの表示が無い"

# ---- run_ab の関数（本体から切り出して実行する）
extract() { # extract <関数名> -> 本体の "name() {" から次の "^}" まで
  awk -v n="$1" '$0 ~ "^"n"\\(\\) \\{" {p=1} p {print} p && /^\}/ {exit}' "${FC}/run_ab_steel_large_n_gemm_metal.sh"
}
ab_run() { # ab_run <名前> <rules> <loads> <series> <シナリオ関数>: 5 round 分の GATE_LOG を作る
  local n="$1"
  mkdir -p "${T}/$n"
  printf '%s' "$2" > "${T}/$n/rules"
  printf '%s\n' $3 > "${T}/$n/loads"
  (
    PATH="${T}/bin:${PATH}" STUB_COUNT="${T}/$n/count" STUB_RULES="${T}/$n/rules" SLEEP_COUNT="${T}/$n/sleeps" LOADS="${T}/$n/loads"
    export PATH STUB_COUNT STUB_RULES SLEEP_COUNT LOADS
    FOREIGN_GATE="${T}/fake_gate.py"
    # shellcheck source=/dev/null
    source "${FC}/ab_foreign_gate.sh"
    fg_require "${FOREIGN_GATE}" "${T}/$n/fg" || exit 99
    LOAD_GATE=3.0; GATE_SERIES="${4:-official}"; DEVICE=metal; GATE_LOG="${T}/$n/gate.log"; CUR_ROUND=0
    : > "${GATE_LOG}"
    for f in wait_load_gate cell_gate_before cell_gate_after; do
      eval "$(extract "${f}")"
    done
    for r in 1 2 3 4 5; do
      CUR_ROUND=${r}
      wait_load_gate "${r}"
      for cell in before-N256-fresh after-N256-fresh candle-N256-fresh; do
        cell_gate_before "${cell}"
        cell_gate_after "${cell}"
      done
    done
  )
}
ab_official() { # ab_official <gate.log> -> aggregate.py の load_ab と同じ判定（"series=official" 相当を除く round 行の 5 行・全 pass）
  python3 -I - "$1" <<'EOF'
import re, sys
t = open(sys.argv[1]).read()
g = re.findall(r"^round(\d+) gate=(\S+)", t, re.MULTILINE)
print("true" if sorted(int(x[0]) for x in g) == [1, 2, 3, 4, 5] and all(x[1] == "pass" for x in g) else "false")
EOF
}
[ -n "$(extract wait_load_gate)" ] && [ -n "$(extract cell_gate_before)" ] && [ -n "$(extract cell_gate_after)" ] || ng "ab: 関数を切り出せない"
ab_run abclean "" "0.5" || ng "ab clean: 実行失敗"
[ "$(ab_official "${T}/abclean/gate.log")" = "true" ] || { ng "ab clean: 正式にならない"; cat "${T}/abclean/gate.log"; }
[ "$(wc -l < "${T}/abclean/gate.log" | tr -d ' ')" = "5" ] || ng "ab clean: gate.log が 5 行でない"
# round 2 の最初の計測プロセス（呼び出し: round1 = 1 + 6、round2 の先頭 = 8、プロセス前 = 9、後 = 10）の後で FOREIGN
ab_run abafter "10 1" "0.5"
grep -q '^round2 gate=invalid reason=foreign-after phase=after cell=before-N256-fresh ' "${T}/abafter/gate.log" || ng "ab after: foreign-after の行が無い"
[ "$(ab_official "${T}/abafter/gate.log")" = "false" ] || ng "ab after: 正式になってしまう"
ab_run abtmo "9-68 1" "0.5"
grep -q '^round2 gate=timeout reason=foreign-job cell=before-N256-fresh load1=0.5 waited_s=1800 ' "${T}/abtmo/gate.log" || ng "ab tmo: timeout の行が無い"
[ "$(ab_official "${T}/abtmo/gate.log")" = "false" ] || ng "ab tmo: 正式になってしまう"
ab_run aberr "8 2" "0.5"
grep -q '^round2 gate=invalid threshold=3.0 load1=NA waited_s=0 reason=foreign-gate-error phase=before ' "${T}/aberr/gate.log" || ng "ab err: round 先頭の失敗の行が無い"
[ "$(ab_official "${T}/aberr/gate.log")" = "false" ] || ng "ab err: 正式になってしまう"

# aggregate.py 自体の self-test（変更なしで通ること）
python3 -I "${HERE}/aggregate.py" --self-test > /dev/null 2>&1 || ng "aggregate.py --self-test が通らない"

# FOREIGN_GATE 未指定の run_ab は計測前に止まる（引数検査より後・facade 検査より前）
FOREIGN_GATE="" bash "${FC}/run_ab_steel_large_n_gemm_metal.sh" 4160 > "${T}/ab_nogate.log" 2>&1 && ng "run_ab: FOREIGN_GATE 未指定で起動した"
grep -q 'FOREIGN_GATE' "${T}/ab_nogate.log" || ng "run_ab: 未指定の理由が出ない"

if [ "${fail}" -ne 0 ]; then exit 1; fi
echo "self-test ok"
