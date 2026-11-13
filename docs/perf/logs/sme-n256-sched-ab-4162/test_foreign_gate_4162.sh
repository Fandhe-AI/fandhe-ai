#!/usr/bin/env bash
# #4162 の Mac mini キットに足した他ジョブの前提ゲートと系列の前の確認（Safari・mediaanalysisd）の self-test（実機には接続しない）。
# 役割: foreign_job_gate.py を偽物（呼び出し番号ごとに CLEAN／FOREIGN／失敗／出力不一致を返す）に、sleep・sysctl・ps をスタブに
#   差し替え、ロック・ビルドなしで次を確かめる（形は #4156 の test_foreign_gate_4156.sh〈PR #4258〉と同じ）:
#   - orchestrate_macmini.sh の wait_load_gate／round_gate_after を本体から切り出して 5 round 回し、全 CLEAN なら従来と同じ
#     `round<N> gate=pass gate_ok=1 ...` の 5 行だけ（GATE_NOT_OK=0）、終了後 FOREIGN・開始前 60 回 FOREIGN・load1 超過・
#     ゲート失敗・出力不一致・load1 が読めないなら GATE_NOT_OK が 1 以上になること
#   - その load_gate.log を変更なしの aggregate.py（judge）へ渡し、全 CLEAN は判定まで進み（ADOPT）、開始前の不通過と
#     終了後の FOREIGN はどちらも undetermined になること（#4162 には run_ab が無いので、終了後は同じ round の重複行と gate_ok=0 で外れる）
#   - check_safari（basename がちょうど Safari・補助プロセスは条件にしない・ps の失敗と空は不成立）と check_mediaanalysisd
#     （basename の完全一致・5 未満・不在は通過・ps の失敗と非数値は不成立）、series_start_check（不成立なら prestart を残して止まる）
#   - FOREIGN_GATE の未指定・相対パス・存在しないパス・禁止文字でロックを取る前に止まること、呼び出し順
#     （ロック → P-TREE → ゲートの準備 → ビルド → 系列の前の確認 → round〈前のゲート → 腕 → PyTorch → 後のゲート〉→ 収録）と dry-run
# 使い方: bash test_foreign_gate_4162.sh   （成功で "self-test ok"・exit 0。macOS／Linux の bash で動く）
# shellcheck disable=SC2034,SC2086,SC1091  # 切り出した関数が読む変数・意図した単語分割
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)" || exit 2
ROOT="$(cd "${HERE}/../../../.." && pwd -P)" || exit 2
FC="${ROOT}/scripts/bench/framework-compare"
ORCH="${HERE}/orchestrate_macmini.sh"
T="$(mktemp -d)" || exit 2
trap 'rm -rf "${T}"' EXIT
fail=0
ng() { echo "NG: $*"; fail=1; }

mkdir -p "${T}/bin"
# 偽ゲート: ${STUB_RULES} の行 "N rc" または "N-M rc"（rc: 0 CLEAN・1 FOREIGN・2 失敗・m 終了コード 0 だが 1 行目が FOREIGN）で
# N 回目の呼び出しの結果を決める（既定 CLEAN）。呼び出し回数は ${STUB_COUNT} に残す（#4156・#4160 の self-test と同じ偽物）。
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
# sleep は回数だけ数えて即時終了。sysctl は ${LOADS} の先頭行を load1 として返す（最後の 1 行は繰り返す）。
# ps は ${PS_OUT} の中身を返す（${PS_FAIL} が 1 なら失敗）
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
cat > "${T}/bin/ps" <<'EOF'
#!/bin/sh
[ "${PS_FAIL:-0}" = "1" ] && exit 1
cat "${PS_OUT}"
EOF
chmod +x "${T}"/bin/*

extract() { # extract <ファイル> <関数名> -> 本体の "name() {" から次の "^}" まで
  awk -v n="$2" '$0 ~ "^"n"\\(\\) \\{" {p=1} p {print} p && /^\}/ {exit}' "$1"
}
for f in wait_load_gate round_gate_after check_safari check_mediaanalysisd series_start_check; do
  [ -n "$(extract "${ORCH}" "${f}")" ] || ng "orchestrate: 関数 ${f} を切り出せない"
done

# ---- round のゲート（5 round を回して load_gate.log と GATE_NOT_OK を作る）
ab_run() { # ab_run <名前> <rules> <loads>
  local n="$1"
  mkdir -p "${T}/$n"
  printf '%s' "$2" > "${T}/$n/rules"
  printf '%s\n' $3 > "${T}/$n/loads"
  (
    PATH="${T}/bin:${PATH}" STUB_COUNT="${T}/$n/count" STUB_RULES="${T}/$n/rules" SLEEP_COUNT="${T}/$n/sleeps" LOADS="${T}/$n/loads"
    export PATH STUB_COUNT STUB_RULES SLEEP_COUNT LOADS
    FOREIGN_GATE="${T}/fake_gate.py"; LOAD_GATE=3.0; OUT="${T}/$n/out"; GATE_NOT_OK=0
    mkdir -p "${OUT}"
    # shellcheck source=/dev/null
    . "${FC}/ab_foreign_gate.sh"
    fg_require "${FOREIGN_GATE}" "${OUT}/foreign-gate" || exit 99
    : > "${OUT}/load_gate.log"
    for f in wait_load_gate round_gate_after; do
      eval "$(extract "${ORCH}" "${f}")"
    done
    for r in 1 2 3 4 5; do
      wait_load_gate "${r}"
      round_gate_after "${r}"
    done
    echo "${GATE_NOT_OK}" > "${T}/$n/not_ok"
  )
}
glog() { echo "${T}/$1/out/load_gate.log"; }
calls() { cat "${T}/$1/count" 2> /dev/null || echo 0; }
notok() { cat "${T}/$1/not_ok" 2> /dev/null || echo NA; }
nlines() { wc -l < "$(glog "$1")" | tr -d ' '; }

# 変更なしの aggregate.py の judge で判定を出す。fixture（_make）は t4 が判定 2 セルで速い ADOPT(t4) の系列で、
# load_gate.log だけを <ログ> に差し替える
verdict() { # verdict <load_gate.log>
  python3 -I - "${HERE}/aggregate.py" "$1" <<'EOF'
import importlib.util, os, shutil, sys, tempfile
spec = importlib.util.spec_from_file_location("aggregate_4162", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with tempfile.TemporaryDirectory() as td:
    m._make(td, {"t4": {("gemm", 256, "fresh"): 0.9, ("gemm", 256, "reuse"): 0.92}})
    shutil.copyfile(sys.argv[2], os.path.join(td, "load_gate.log"))
    r = m.judge(td)
    print(r["verdict"], " | ".join(r["problems"]))
EOF
}

PASS_RE='^round[1-5] gate=pass gate_ok=1 threshold=3\.0 load1=0\.5 waited_s=0 at=[0-9T:Z-]*$'

# 正: すべて CLEAN → pass 5 行だけ・GATE_NOT_OK=0・呼び出し 10 回（各 round 前 1 + 後 1）・aggregate は判定まで進む（ADOPT）
ab_run clean "" "0.5"
[ "$(nlines clean)" = "5" ] || ng "clean: load_gate.log が 5 行でない"
[ "$(grep -c "${PASS_RE}" "$(glog clean)")" = "5" ] || { ng "clean: pass の行が従来の書式でない"; cat "$(glog clean)"; }
[ "$(notok clean)" = "0" ] || ng "clean: GATE_NOT_OK=$(notok clean)"
[ "$(calls clean)" = "10" ] || ng "clean: ゲート呼び出し $(calls clean) 回（期待 10）"
nfiles() { local n=0 f; for f in "$@"; do [ -f "${f}" ] && n=$((n + 1)); done; echo "${n}"; }
[ "$(nfiles "${T}"/clean/out/foreign-gate/round[1-5]-start-before-a1.txt)" = "5" ] || ng "clean: 開始前の出力が 5 件でない"
[ "$(nfiles "${T}"/clean/out/foreign-gate/round[1-5]-after.txt)" = "5" ] || ng "clean: 終了後の出力が 5 件でない"
case "$(verdict "$(glog clean)")" in "ADOPT(t4) "*) ;; *) ng "clean: aggregate.py が判定まで進まない: $(verdict "$(glog clean)")" ;; esac

# 負: round 2 の終了後に FOREIGN（呼び出し 4 回目）→ invalid 行・GATE_NOT_OK=1・系列は最後まで進む・aggregate は undetermined
ab_run after "4 1" "0.5"
grep -q '^round2 gate=invalid gate_ok=0 threshold=3\.0 load1=NA waited_s=0 at=[0-9T:Z-]* reason=foreign-after phase=after$' "$(glog after)" || ng "after: foreign-after の行が無い"
[ "$(notok after)" = "1" ] || ng "after: GATE_NOT_OK=$(notok after)（期待 1）"
grep -q '^round5 gate=pass gate_ok=1 ' "$(glog after)" || ng "after: 系列が最後まで進んでいない"
v_after="$(verdict "$(glog after)")"
case "${v_after}" in undetermined*) ;; *) ng "after: 終了後の FOREIGN で正式から外れない: ${v_after}" ;; esac
case "${v_after}" in *"round2 が重複"*"P-GATE 不成立: round2"*) ;; *) ng "after: 外れる理由が重複行と P-GATE でない: ${v_after}" ;; esac

# 負: round 2 の開始前に 60 回 FOREIGN（呼び出し 3〜62 回目）→ timeout reason=foreign-job。ちょうど 60 回で打ち切る
ab_run tmo "3-62 1" "0.5"
grep -q '^round2 gate=timeout gate_ok=0 threshold=3\.0 load1=0\.5 waited_s=1800 at=[0-9T:Z-]* reason=foreign-job$' "$(glog tmo)" || ng "tmo: timeout foreign-job の行が無い"
[ "$(calls tmo)" = "69" ] || ng "tmo: ゲート呼び出し $(calls tmo) 回（期待 69＝60 回で打ち切り）"
[ "$(notok tmo)" = "1" ] || ng "tmo: GATE_NOT_OK=$(notok tmo)"
case "$(verdict "$(glog tmo)")" in undetermined*"P-GATE 不成立: round2"*) ;; *) ng "tmo: P-GATE で正式から外れない" ;; esac

# 負: ゲートは CLEAN でも load1 が 3.0 以上のまま 60 回 → timeout reason=load1（従来の load1 の timeout と同じ扱い）
ab_run load "" "0.5 $(yes 5.0 | head -60 | tr '\n' ' ')0.5"
grep -q '^round2 gate=timeout gate_ok=0 threshold=3\.0 load1=5\.0 waited_s=1800 at=[0-9T:Z-]* reason=load1$' "$(glog load)" || ng "load: timeout load1 の行が無い"
case "$(verdict "$(glog load)")" in undetermined*) ;; *) ng "load: P-GATE で正式から外れない" ;; esac

# 負: 開始前のゲート失敗（待たずに invalid）・終了後のゲート失敗・出力不一致（fail-closed）
ab_run err "3 2" "0.5"
grep -q '^round2 gate=invalid gate_ok=0 threshold=3\.0 load1=NA waited_s=0 at=[0-9T:Z-]* reason=foreign-gate-error phase=before$' "$(glog err)" || ng "err: before の foreign-gate-error が無い"
[ "$(notok err)" = "1" ] || ng "err: GATE_NOT_OK=$(notok err)"
case "$(verdict "$(glog err)")" in undetermined*) ;; *) ng "err: P-GATE で正式から外れない" ;; esac
ab_run err2 "4 2" "0.5"
grep -q '^round2 gate=invalid gate_ok=0 .* reason=foreign-gate-error phase=after$' "$(glog err2)" || ng "err2: after の foreign-gate-error が無い"
[ "$(notok err2)" = "1" ] || ng "err2: GATE_NOT_OK=$(notok err2)"
case "$(verdict "$(glog err2)")" in undetermined*) ;; *) ng "err2: 正式から外れない" ;; esac
ab_run mism "4 m" "0.5"
grep -q '^round2 gate=invalid gate_ok=0 .* reason=foreign-gate-error phase=after$' "$(glog mism)" || ng "mism: 出力不一致が error 扱いにならない"

# load1 が読めない → unavailable（従来と同じ語）・gate_ok=0
ab_run unav "" "x"
grep -q '^round1 gate=unavailable gate_ok=0 threshold=3\.0 load1=NA waited_s=0 at=[0-9T:Z-]* reason=load1-unavailable$' "$(glog unav)" || ng "unav: unavailable の行が無い"
case "$(verdict "$(glog unav)")" in undetermined*) ;; *) ng "unav: P-GATE で正式から外れない" ;; esac

# ---- 系列の前の確認（関数を切り出してスタブの ps で回す）
pscase() { # pscase <名前> <関数> <ps の出力（\n 区切り）> [ps を失敗させる] -> 終了コードを ${T}/<名前>/rc に
  local n="$1"
  mkdir -p "${T}/$n"
  printf "$3" > "${T}/$n/ps"
  (
    PATH="${T}/bin:${PATH}" PS_OUT="${T}/$n/ps" PS_FAIL="${4:-0}" SLEEP_COUNT="${T}/$n/sleeps"
    export PATH PS_OUT PS_FAIL SLEEP_COUNT
    eval "$(extract "${ORCH}" "$2")"
    "$2" "${T}/$n/out"
    echo $? > "${T}/$n/rc"
  )
}
mrc() { cat "${T}/$1/rc"; }
msl() { wc -l < "${T}/$1/sleeps" 2> /dev/null | tr -d ' ' || echo 0; }
MDA_PATH="/System/Library/PrivateFrameworks/MediaAnalysis.framework/Versions/A/mediaanalysisd"
pscase m_low check_mediaanalysisd " 0.0 /usr/sbin/cfprefsd\n 1.2 ${MDA_PATH}\n"
[ "$(mrc m_low)" = "0" ] || ng "mda low: 5 未満で不成立"
[ "$(grep -c 'pcpu=1\.2$' "${T}/m_low/out")" = "3" ] || ng "mda low: 3 回の記録が無い"
[ "$(msl m_low)" = "2" ] || ng "mda low: 間隔の待ちが 2 回でない（$(msl m_low)）"
pscase m_absent check_mediaanalysisd ' 0.0 /usr/sbin/cfprefsd\n 3.0 /usr/libexec/mediaanalysisd-helper\n 9.0 /usr/libexec/xmediaanalysisd\n'
[ "$(mrc m_absent)" = "0" ] || ng "mda absent: 不在（部分一致だけ）で不成立"
grep -q 'pcpu=absent$' "${T}/m_absent/out" || ng "mda absent: absent と記録しない（部分一致を拾った？）"
pscase m_high check_mediaanalysisd " 12.5 ${MDA_PATH}\n"
[ "$(mrc m_high)" = "1" ] || ng "mda high: 5 以上で成立した"
pscase m_edge check_mediaanalysisd ' 5.0 mediaanalysisd\n'
[ "$(mrc m_edge)" = "1" ] || ng "mda edge: ちょうど 5.0 で成立した（5 未満を要求）"
pscase m_max check_mediaanalysisd ' 0.1 mediaanalysisd\n 7.0 mediaanalysisd\n'
[ "$(mrc m_max)" = "1" ] || ng "mda max: 複数のうち最大値で判定しない"
pscase m_nan check_mediaanalysisd ' abc mediaanalysisd\n'
[ "$(mrc m_nan)" = "1" ] || ng "mda nan: 非数値で成立した"
pscase m_psfail check_mediaanalysisd '' 1
[ "$(mrc m_psfail)" = "1" ] || ng "mda psfail: ps の失敗で成立した"
pscase m_empty check_mediaanalysisd ''
[ "$(mrc m_empty)" = "1" ] || ng "mda empty: ps の出力が空で成立した"

SAFARI_HELPERS=' 0.0 /System/Library/Frameworks/SafariServices.framework/com.apple.Safari.SafeBrowsing.Service\n 0.0 /System/Library/PrivateFrameworks/SafariShared.framework/SafariBookmarksSyncAgent\n 0.0 /usr/libexec/SafariLaunchAgent\n'
pscase s_helpers check_safari " 0.0 /usr/sbin/cfprefsd\n${SAFARI_HELPERS}"
[ "$(mrc s_helpers)" = "0" ] || ng "safari helpers: 補助プロセスだけで不成立"
grep -q 'safari_app=0$' "${T}/s_helpers/out" || ng "safari helpers: safari_app=0 を記録しない"
[ "$(grep -c '^  safari-named pcpu=' "${T}/s_helpers/out")" = "3" ] || ng "safari helpers: 名前に Safari を含むプロセスの一覧が 3 件でない"
grep -q '/' "${T}/s_helpers/out" && ng "safari helpers: 記録に絶対パスが残る（basename だけを書く）"
pscase s_app check_safari " 0.0 /usr/sbin/cfprefsd\n 3.4 /Applications/Safari.app/Contents/MacOS/Safari\n${SAFARI_HELPERS}"
[ "$(mrc s_app)" = "1" ] || ng "safari app: Safari アプリ本体があるのに成立した"
grep -q 'safari_app=1$' "${T}/s_app/out" || ng "safari app: safari_app=1 を記録しない"
pscase s_psfail check_safari '' 1
[ "$(mrc s_psfail)" = "1" ] || ng "safari psfail: ps の失敗で成立した"
pscase s_empty check_safari ''
[ "$(mrc s_empty)" = "1" ] || ng "safari empty: ps の出力が空で成立した"

# series_start_check: 成立なら out に result=started・prestart なし。不成立なら out を prestart へ移して exit 1（後続は動かない）
sscase() { # sscase <名前> <ps の出力>
  local n="$1"
  mkdir -p "${T}/$n"
  printf "$2" > "${T}/$n/ps"
  (
    PATH="${T}/bin:${PATH}" PS_OUT="${T}/$n/ps" PS_FAIL=0 SLEEP_COUNT="${T}/$n/sleeps"
    export PATH PS_OUT PS_FAIL SLEEP_COUNT
    for f in check_safari check_mediaanalysisd series_start_check; do eval "$(extract "${ORCH}" "${f}")"; done
    series_start_check "${T}/$n/series-start.txt" "${T}/$n/prestart.txt"
    echo reached > "${T}/$n/reached"
  ) > "${T}/$n/log" 2>&1
  echo $? > "${T}/$n/rc"
}
sscase ss_ok " 0.0 /usr/sbin/cfprefsd\n 0.0 ${MDA_PATH}\n${SAFARI_HELPERS}"
[ "$(mrc ss_ok)" = "0" ] && [ -f "${T}/ss_ok/reached" ] || ng "ss ok: 成立なのに先へ進まない"
grep -q '^result=started ' "${T}/ss_ok/series-start.txt" || ng "ss ok: result=started が無い"
[ ! -e "${T}/ss_ok/prestart.txt" ] || ng "ss ok: 成立なのに prestart がある"
sscase ss_safari " 3.4 /Applications/Safari.app/Contents/MacOS/Safari\n 0.0 ${MDA_PATH}\n"
[ "$(mrc ss_safari)" = "1" ] || ng "ss safari: 非 0 で止まらない"
[ ! -e "${T}/ss_safari/reached" ] || ng "ss safari: 止まらずに先へ進んだ"
grep -q '^result=not-started reason=safari ' "${T}/ss_safari/prestart.txt" 2> /dev/null || ng "ss safari: prestart に reason=safari が無い"
[ ! -e "${T}/ss_safari/series-start.txt" ] || ng "ss safari: series-start.txt が残る（prestart へ移していない）"
grep -q 'mediaanalysisd check' "${T}/ss_safari/prestart.txt" 2> /dev/null && ng "ss safari: Safari で不成立なのに mediaanalysisd を見た"
sscase ss_mda " 0.0 /usr/sbin/cfprefsd\n 40.0 ${MDA_PATH}\n"
[ "$(mrc ss_mda)" = "1" ] || ng "ss mda: 非 0 で止まらない"
[ ! -e "${T}/ss_mda/reached" ] || ng "ss mda: 止まらずに先へ進んだ"
grep -q '^result=not-started reason=mediaanalysisd ' "${T}/ss_mda/prestart.txt" 2> /dev/null || ng "ss mda: prestart に reason=mediaanalysisd が無い"

# ---- orchestrate_macmini.sh の起動時の検査（HOME を一時ディレクトリに差し替え、実物の hw-lock に触れない）
mkdir -p "${T}/home/work" "${T}/base"
orch() { # orch <名前> <FOREIGN_GATE> -> 終了コード
  env -u RAYON_NUM_THREADS HOME="${T}/home" BASE="${T}/base" WORK="${T}/work-$1" TORCH_PY="/usr/bin/true" FOREIGN_GATE="$2" \
    bash "${ORCH}" > "${T}/orch-$1.log" 2>&1
}
cp "${T}/fake_gate.py" "${T}/a b.py"
for c in "nogate|" "rel|fake_gate.py" "missing|${T}/none.py" "space|${T}/a b.py"; do
  name="${c%%|*}"; g="${c#*|}"
  orch "${name}" "${g}" && ng "orch ${name}: 起動した"
  grep -q 'FOREIGN_GATE' "${T}/orch-${name}.log" || ng "orch ${name}: FOREIGN_GATE の理由が出ない"
  [ ! -e "${T}/home/work/.fandhe-hw-lock" ] || ng "orch ${name}: ロックを取った"
done
# 正しいパスなら FOREIGN_GATE の検査を通り、次の RULE.txt の検査（写しの BASE に RULE.txt は置かない）で止まる。ロックはまだ取らない
orch okgate "${T}/fake_gate.py" && ng "orch okgate: RULE.txt なしで起動した"
grep -q 'RULE.txt が無い' "${T}/orch-okgate.log" || ng "orch okgate: RULE.txt の検査まで進まない"
[ ! -e "${T}/home/work/.fandhe-hw-lock" ] || ng "orch okgate: ロックを取った"

# 呼び出し順: ゲートのパス検査 → ロック → P-TREE → ゲートの準備 → ビルド → 系列の前の確認 → round のループ → 収録
ln_of() { grep -nF -- "$1" "${ORCH}" | head -n 1 | cut -d: -f1; }
l_chk=$(ln_of '[ -f "${FOREIGN_GATE}" ] ||')
l_lock=$(ln_of 'until mkdir "${LOCK_DIR}"')
l_tree=$(ln_of 'gate P-TREE all ok')
l_req=$(ln_of 'fg_require "${FOREIGN_GATE}" "${OUT}/foreign-gate"')
l_build=$(ln_of 'for t in ${TREES}; do bench_build')
l_ss=$(ln_of 'series_start_check "${OUT}/series-start.txt" "${WORK}/prestart.txt"')
l_res=$(ln_of 'for a in ${ARMS}; do : > ')
l_loop=$(ln_of 'while [ "${r}" -le "${ROUNDS}" ]; do')
l_col=$(ln_of 'for f in "${OUT}/foreign-gate"/*; do')
for x in "${l_chk}" "${l_lock}" "${l_tree}" "${l_req}" "${l_build}" "${l_ss}" "${l_res}" "${l_loop}" "${l_col}"; do
  [ -n "${x}" ] || ng "orch: 呼び出し箇所を見つけられない"
done
prev=0
for x in "${l_chk}" "${l_lock}" "${l_tree}" "${l_req}" "${l_build}" "${l_ss}" "${l_res}" "${l_loop}" "${l_col}"; do
  [ -n "${x}" ] && [ "${prev}" -lt "${x}" ] || { ng "orch: 呼び出し順が パス検査 → ロック → P-TREE → ゲート準備 → ビルド → 系列の前の確認 → 結果ファイル → round → 収録 でない"; break; }
  prev="${x}"
done
# round のループの中: 前のゲート → 腕 → PyTorch → 後のゲート
awk '/^while \[ "\$\{r\}" -le "\$\{ROUNDS\}" \]; do$/ {p=1} p {print} p && /^done$/ {exit}' "${ORCH}" > "${T}/loop.txt"
l_w=$(grep -nxF '    wait_load_gate "${r}"' "${T}/loop.txt" | cut -d: -f1)
l_arm=$(grep -nxF '        run_arm "${arm}"' "${T}/loop.txt" | cut -d: -f1)
l_t=$(grep -nxF '    torch_cell gemm 256; torch_cell gemm 512; torch_cell train 64' "${T}/loop.txt" | cut -d: -f1)
l_a=$(grep -nxF '    round_gate_after "${r}"' "${T}/loop.txt" | cut -d: -f1)
if [ -z "${l_w}" ] || [ -z "${l_arm}" ] || [ -z "${l_t}" ] || [ -z "${l_a}" ] \
  || ! [ "${l_w}" -lt "${l_arm}" ] || ! [ "${l_arm}" -lt "${l_t}" ] || ! [ "${l_t}" -lt "${l_a}" ]; then
  ng "orch: round ループの呼び出し順が wait_load_gate → 腕 → PyTorch → round_gate_after でない"
fi
# 既存の出力があれば止まる（上書き禁止）の対象にゲートの出力・系列の前の記録・prestart が入っている
for p in '"${OUT}/foreign-gate"' '"${OUT}/series-start.txt"' '"${WORK}/prestart.txt"'; do
  grep -qF -- "[ -e ${p} ]" "${ORCH}" || ng "orch: 上書き禁止の検査に ${p} が無い"
done
grep -qF 'echo "foreign_job_gate_sha256=' "${ORCH}" || ng "orch: ゲートの sha256 を残していない"

# dry-run は FOREIGN_GATE なしで通り、計画に追加分が出る
env -u FOREIGN_GATE bash "${ORCH}" --dry-run > "${T}/dry.log" 2>&1 || ng "dry-run: 失敗"
for w in '他ジョブの前提ゲート' 'FOREIGN_GATE' 'Safari' 'mediaanalysisd' 'prestart'; do
  grep -q "${w}" "${T}/dry.log" || ng "dry-run: ${w} の表示が無い"
done

# 既存の self-test（変更なしで通ること）
python3 -I "${HERE}/aggregate.py" --self-test > /dev/null 2>&1 || ng "aggregate.py --self-test が通らない"
python3 -I "${HERE}/aggregate.py" --check-template > /dev/null 2>&1 || ng "aggregate.py --check-template が通らない"

if [ "${fail}" -ne 0 ]; then exit 1; fi
echo "self-test ok"
