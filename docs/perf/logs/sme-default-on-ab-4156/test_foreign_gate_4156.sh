#!/usr/bin/env bash
# #4156 の Mac mini 腕に足した他ジョブの前提ゲートと mediaanalysisd の確認の self-test（実機には接続しない）。
# 役割: foreign_job_gate.py を偽物（呼び出し番号ごとに CLEAN／FOREIGN／失敗／出力不一致を返す）に、sleep・sysctl・ps をスタブに
#   差し替え、ロック・ビルドなしで次を確かめる（形は #4160 の test_foreign_gate_4160.sh と同じ）:
#   - run_ab_sme_default_on_4156_cpu.sh の wait_load_gate／wait_load_gate_fg／round_gate_after を本体から切り出して 5 round 回し、
#     AB_HOST=macmini で全 CLEAN なら従来と同じ `round<N> gate=pass gate_ok=1 ...` の 5 行だけ（GATE_NOT_OK=0）、
#     終了後 FOREIGN・開始前 60 回 FOREIGN・load1 超過・ゲート失敗・出力不一致なら GATE_NOT_OK が 1 以上（A/B 本体は非 0 終了）になること
#   - その行を変更なしの aggregate.py（evaluate）へ渡し、全 CLEAN は正式、開始前の不通過は P-GATE で、終了後の FOREIGN は
#     `run_ab exit=1`（GATE_NOT_OK による非 0 終了）の P-RUNAB で正式でなくなること
#   - AB_HOST=gb10 ではゲートを呼ばず、ログの行が従来の書式のままであること
#   - orchestrate_macmini.sh の check_mediaanalysisd（basename の完全一致・5 未満・不在は通過・ps の失敗と非数値は不成立）と、
#     FOREIGN_GATE の未指定・相対パス・存在しないパスでロックを取る前に止まること、呼び出し順（ロック → P-RT → mediaanalysisd → A/B）
# 使い方: bash test_foreign_gate_4156.sh   （成功で "self-test ok"・exit 0。macOS／Linux の bash で動く）
# shellcheck disable=SC2034,SC2086,SC1091  # 切り出した関数が読む変数・意図した単語分割
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)" || exit 2
ROOT="$(cd "${HERE}/../../../.." && pwd -P)" || exit 2
FC="${ROOT}/scripts/bench/framework-compare"
RUN_AB="${FC}/run_ab_sme_default_on_4156_cpu.sh"
ORCH="${HERE}/orchestrate_macmini.sh"
T="$(mktemp -d)" || exit 2
trap 'rm -rf "${T}"' EXIT
fail=0
ng() { echo "NG: $*"; fail=1; }

mkdir -p "${T}/bin"
# 偽ゲート: ${STUB_RULES} の行 "N rc" または "N-M rc"（rc: 0 CLEAN・1 FOREIGN・2 失敗・m 終了コード 0 だが 1 行目が FOREIGN）で
# N 回目の呼び出しの結果を決める（既定 CLEAN）。呼び出し回数は ${STUB_COUNT} に残す（#4160 の self-test と同じ偽物）。
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
for f in wait_load_gate wait_load_gate_fg round_gate_after; do
  [ -n "$(extract "${RUN_AB}" "${f}")" ] || ng "run_ab: 関数 ${f} を切り出せない"
done
[ -n "$(extract "${ORCH}" check_mediaanalysisd)" ] || ng "orchestrate: check_mediaanalysisd を切り出せない"

# ---- run_ab の round ゲート（5 round を回して GATE_LOG と GATE_NOT_OK を作る）
ab_run() { # ab_run <名前> <host> <rules> <loads> [series]
  local n="$1"
  mkdir -p "${T}/$n"
  printf '%s' "$3" > "${T}/$n/rules"
  printf '%s\n' $4 > "${T}/$n/loads"
  (
    PATH="${T}/bin:${PATH}" STUB_COUNT="${T}/$n/count" STUB_RULES="${T}/$n/rules" SLEEP_COUNT="${T}/$n/sleeps" LOADS="${T}/$n/loads"
    export PATH STUB_COUNT STUB_RULES SLEEP_COUNT LOADS
    FOREIGN_GATE="${T}/fake_gate.py"
    if [ "$2" = macmini ]; then
      FG_ENABLED=1; LOAD_GATE=3.0
      # shellcheck source=/dev/null
      source "${FC}/ab_foreign_gate.sh"
      fg_require "${FOREIGN_GATE}" "${T}/$n/fg" || exit 99
    else
      FG_ENABLED=0; LOAD_GATE=8.0
    fi
    GATE_SERIES="${5:-official}"; GATE_LOG="${T}/$n/gate.log"; GATE_NOT_OK=0
    : > "${GATE_LOG}"
    for f in wait_load_gate wait_load_gate_fg round_gate_after; do
      eval "$(extract "${RUN_AB}" "${f}")"
    done
    for r in 1 2 3 4 5; do
      wait_load_gate "${r}"
      round_gate_after "${r}"
    done
    echo "${GATE_NOT_OK}" > "${T}/$n/not_ok"
  )
}
calls() { cat "${T}/$1/count" 2> /dev/null || echo 0; }
notok() { cat "${T}/$1/not_ok" 2> /dev/null || echo NA; }
nlines() { wc -l < "${T}/$1/gate.log" | tr -d ' '; }

# 変更なしの aggregate.py の evaluate で Mac mini の判定を出す。<内側ログ> を A/B の load_gate-*.log に、
# <run_ab の終了コード> を load_gate.log の `run_ab exit=` に入れる（他の記録は aggregate.py の fixture のまま）
verdict() { # verdict <内側ログ> <run_ab の終了コード>
  python3 -I - "${HERE}/aggregate.py" "$1" "$2" <<'EOF'
import importlib.util, os, sys, tempfile
spec = importlib.util.spec_from_file_location("aggregate_4156", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
inner, rc = open(sys.argv[2]).read(), sys.argv[3]
with tempfile.TemporaryDirectory() as hd, tempfile.TemporaryDirectory() as td:
    m.HERE = hd
    sha = m._mk_rule(hd, mode="GB10-GATE-ONLY-STRUCTURAL", nb="adopted", ok=True)
    m._fixture(td, "macmini", sha)
    with open(os.path.join(td, "ab", "load_gate-4156-sme-cpu-4156-r1.log"), "w") as f:
        f.write("threshold=3.0 rule_threshold=3.0 series=official\n" + inner)
    p = os.path.join(td, "load_gate.log")
    lg = open(p).read().replace("run_ab exit=0\n", "run_ab exit=%s\n" % rc)
    open(p, "w").write(lg)
    try:
        print(m.evaluate(td, "macmini")["verdict"])
    except m.AggError as e:
        print("undetermined(AggError: %s)" % e)
EOF
}

PASS_RE='^round[1-5] gate=pass gate_ok=1 threshold=3\.0 load1=0\.5 waited_s=0 at=[0-9T:Z-]*$'

# 正: すべて CLEAN → pass 5 行だけ・GATE_NOT_OK=0・呼び出し 10 回（各 round 前 1 + 後 1）・aggregate は正式（ADOPT）
ab_run clean macmini "" "0.5"
[ "$(nlines clean)" = "5" ] || ng "clean: gate.log が 5 行でない"
[ "$(grep -c "${PASS_RE}" "${T}/clean/gate.log")" = "5" ] || { ng "clean: pass の行が従来の書式でない"; cat "${T}/clean/gate.log"; }
[ "$(notok clean)" = "0" ] || ng "clean: GATE_NOT_OK=$(notok clean)"
[ "$(calls clean)" = "10" ] || ng "clean: ゲート呼び出し $(calls clean) 回（期待 10）"
nfiles() { local n=0 f; for f in "$@"; do [ -f "${f}" ] && n=$((n + 1)); done; echo "${n}"; }
[ "$(nfiles "${T}"/clean/fg/round[1-5]-start-before-a1.txt)" = "5" ] || ng "clean: 開始前の出力が 5 件でない"
[ "$(nfiles "${T}"/clean/fg/round[1-5]-after.txt)" = "5" ] || ng "clean: 終了後の出力が 5 件でない"
case "$(verdict "${T}/clean/gate.log" 0)" in ADOPT*) ;; *) ng "clean: aggregate.py が正式（ADOPT）にしない: $(verdict "${T}/clean/gate.log" 0)" ;; esac

# 負: round 2 の終了後に FOREIGN（呼び出し 4 回目）→ invalid 行・GATE_NOT_OK=1・系列は最後まで進む
ab_run after macmini "4 1" "0.5"
grep -q '^round2 gate=invalid gate_ok=0 threshold=3\.0 load1=NA waited_s=0 at=[0-9T:Z-]* reason=foreign-after phase=after$' "${T}/after/gate.log" || ng "after: foreign-after の行が無い"
[ "$(notok after)" = "1" ] || ng "after: GATE_NOT_OK=$(notok after)（期待 1）"
grep -q '^round5 gate=pass gate_ok=1 ' "${T}/after/gate.log" || ng "after: 系列が最後まで進んでいない"
# aggregate.py の P-GATE は pass 行だけを数えるので、この系列を正式から外すのは run_ab の非 0 終了（P-RUNAB）。両方を確かめる
case "$(verdict "${T}/after/gate.log" 1)" in undetermined*) ;; *) ng "after: run_ab exit=1 でも正式になる" ;; esac
case "$(verdict "${T}/after/gate.log" 0)" in ADOPT*) ;; *) ng "after: P-RUNAB だけで外れる前提が崩れた（aggregate.py が変わった？）" ;; esac

# 負: round 2 の開始前に 60 回 FOREIGN（呼び出し 3〜62 回目）→ timeout reason=foreign-job。ちょうど 60 回で打ち切る
ab_run tmo macmini "3-62 1" "0.5"
grep -q '^round2 gate=timeout gate_ok=0 threshold=3\.0 load1=0\.5 waited_s=1800 at=[0-9T:Z-]* reason=foreign-job$' "${T}/tmo/gate.log" || ng "tmo: timeout foreign-job の行が無い"
[ "$(calls tmo)" = "69" ] || ng "tmo: ゲート呼び出し $(calls tmo) 回（期待 69＝60 回で打ち切り）"
[ "$(notok tmo)" = "1" ] || ng "tmo: GATE_NOT_OK=$(notok tmo)"
case "$(verdict "${T}/tmo/gate.log" 0)" in undetermined*) ;; *) ng "tmo: P-GATE で正式から外れない" ;; esac

# 負: ゲートは CLEAN でも load1 が 3.0 以上のまま 60 回 → timeout reason=load1（従来の load1 の timeout と同じ扱い）
ab_run load macmini "" "0.5 $(yes 5.0 | head -60 | tr '\n' ' ')0.5"
grep -q '^round2 gate=timeout gate_ok=0 threshold=3\.0 load1=5\.0 waited_s=1800 at=[0-9T:Z-]* reason=load1$' "${T}/load/gate.log" || ng "load: timeout load1 の行が無い"
case "$(verdict "${T}/load/gate.log" 0)" in undetermined*) ;; *) ng "load: P-GATE で正式から外れない" ;; esac

# 負: 開始前のゲート失敗（待たずに invalid）・終了後のゲート失敗・出力不一致（fail-closed）
ab_run err macmini "3 2" "0.5"
grep -q '^round2 gate=invalid gate_ok=0 threshold=3\.0 load1=NA waited_s=0 at=[0-9T:Z-]* reason=foreign-gate-error phase=before$' "${T}/err/gate.log" || ng "err: before の foreign-gate-error が無い"
[ "$(notok err)" = "1" ] || ng "err: GATE_NOT_OK=$(notok err)"
ab_run err2 macmini "4 2" "0.5"
grep -q '^round2 gate=invalid gate_ok=0 .* reason=foreign-gate-error phase=after$' "${T}/err2/gate.log" || ng "err2: after の foreign-gate-error が無い"
[ "$(notok err2)" = "1" ] || ng "err2: GATE_NOT_OK=$(notok err2)"
ab_run mism macmini "4 m" "0.5"
grep -q '^round2 gate=invalid gate_ok=0 .* reason=foreign-gate-error phase=after$' "${T}/mism/gate.log" || ng "mism: 出力不一致が error 扱いにならない"

# load1 が読めない → unavailable（従来と同じ語）・gate_ok=0
ab_run unav macmini "" "x"
grep -q '^round1 gate=unavailable gate_ok=0 threshold=3\.0 load1=NA waited_s=0 at=[0-9T:Z-]* reason=load1-unavailable$' "${T}/unav/gate.log" || ng "unav: unavailable の行が無い"

# 緩和閾値の系列（reference）の通過は従来どおり pass-reference・gate_ok=0
ab_run ref macmini "" "0.5" reference
[ "$(grep -c '^round[1-5] gate=pass-reference gate_ok=0 ' "${T}/ref/gate.log")" = "5" ] || ng "ref: pass-reference にならない"

# GB10: ゲートを呼ばない・行は従来の書式（2 回目の round だけ 1 回待つ）
ab_run gb10 gb10 "" "0.5 9.0 0.5"
[ "$(calls gb10)" = "0" ] || ng "gb10: ゲートを呼んだ（$(calls gb10) 回）"
sed 's/ at=[0-9T:Z-]*$//' "${T}/gb10/gate.log" > "${T}/gb10/norm"
# 9.0 は round2 の 1 回目で読まれ（8.0 以上で 30 秒待つ）、2 回目の 0.5 で通る
printf '%s\n' "round1 gate=pass gate_ok=1 threshold=8.0 load1=0.5 waited_s=0" \
  "round2 gate=pass gate_ok=1 threshold=8.0 load1=0.5 waited_s=30" \
  "round3 gate=pass gate_ok=1 threshold=8.0 load1=0.5 waited_s=0" \
  "round4 gate=pass gate_ok=1 threshold=8.0 load1=0.5 waited_s=0" \
  "round5 gate=pass gate_ok=1 threshold=8.0 load1=0.5 waited_s=0" > "${T}/gb10/want"
cmp -s "${T}/gb10/norm" "${T}/gb10/want" || { ng "gb10: 行が従来の書式と違う"; diff "${T}/gb10/want" "${T}/gb10/norm"; }
[ "$(notok gb10)" = "0" ] || ng "gb10: GATE_NOT_OK=$(notok gb10)"

# run_ab の round ループで、開始前は wait_load_gate、終了後は PyTorch の同席計測の後に round_gate_after を呼ぶ
awk '/^for run_i in \$\(seq 1 "\$ROUNDS"\); do$/ {p=1} p {print} p && /^done$/ {exit}' "${RUN_AB}" > "${T}/loop.txt"
l_w=$(grep -n '^  wait_load_gate "\$run_i"$' "${T}/loop.txt" | cut -d: -f1)
l_t=$(grep -n 'run_torch_round "\$run_i"$' "${T}/loop.txt" | cut -d: -f1)
l_a=$(grep -n '^  round_gate_after "\$run_i"$' "${T}/loop.txt" | cut -d: -f1)
if [ -z "${l_w}" ] || [ -z "${l_t}" ] || [ -z "${l_a}" ] || ! [ "${l_w}" -lt "${l_t}" ] || ! [ "${l_t}" -lt "${l_a}" ]; then
  ng "run_ab: round ループの呼び出し順が wait_load_gate → PyTorch → round_gate_after でない"
fi

# run_ab の起動時の検査: macmini で FOREIGN_GATE 未指定なら止まる・gb10 では FOREIGN_GATE を要求しない
FOREIGN_GATE="" AB_HOST=macmini bash "${RUN_AB}" 4156-t > "${T}/ab_nogate.log" 2>&1 && ng "run_ab: macmini で FOREIGN_GATE 未指定のまま起動した"
grep -q 'FOREIGN_GATE' "${T}/ab_nogate.log" || ng "run_ab: 未指定の理由が出ない"
FOREIGN_GATE="" AB_HOST=gb10 bash "${RUN_AB}" > "${T}/ab_gb10.log" 2>&1
grep -q 'FOREIGN_GATE' "${T}/ab_gb10.log" && ng "run_ab: gb10 で FOREIGN_GATE を要求した"
grep -q '^usage:' "${T}/ab_gb10.log" || ng "run_ab: gb10 の従来の引数検査に届かない"

# ---- orchestrate_macmini.sh の mediaanalysisd の確認（関数を切り出してスタブの ps で回す）
mda() { # mda <名前> <ps の出力（\n 区切り）> [ps を失敗させる] -> 終了コードを ${T}/<名前>/rc に
  local n="$1"
  mkdir -p "${T}/$n"
  printf "$2" > "${T}/$n/ps"
  (
    PATH="${T}/bin:${PATH}" PS_OUT="${T}/$n/ps" PS_FAIL="${3:-0}" SLEEP_COUNT="${T}/$n/sleeps"
    export PATH PS_OUT PS_FAIL SLEEP_COUNT
    eval "$(extract "${ORCH}" check_mediaanalysisd)"
    check_mediaanalysisd "${T}/$n/out"
    echo $? > "${T}/$n/rc"
  )
}
mrc() { cat "${T}/$1/rc"; }
msl() { wc -l < "${T}/$1/sleeps" 2> /dev/null | tr -d ' ' || echo 0; }
mda m_low ' 0.0 /usr/sbin/cfprefsd\n 1.2 /System/Library/PrivateFrameworks/MediaAnalysis.framework/Versions/A/mediaanalysisd\n'
[ "$(mrc m_low)" = "0" ] || ng "mda low: 5 未満で不成立"
[ "$(grep -c 'pcpu=1\.2$' "${T}/m_low/out")" = "3" ] || ng "mda low: 3 回の記録が無い"
[ "$(msl m_low)" = "2" ] || ng "mda low: 間隔の待ちが 2 回でない（$(msl m_low)）"
mda m_absent ' 0.0 /usr/sbin/cfprefsd\n 3.0 /usr/libexec/mediaanalysisd-helper\n 9.0 /usr/libexec/xmediaanalysisd\n'
[ "$(mrc m_absent)" = "0" ] || ng "mda absent: 不在（部分一致だけ）で不成立"
grep -q 'pcpu=absent$' "${T}/m_absent/out" || ng "mda absent: absent と記録しない（部分一致を拾った？）"
mda m_high ' 12.5 /System/Library/PrivateFrameworks/MediaAnalysis.framework/Versions/A/mediaanalysisd\n'
[ "$(mrc m_high)" = "1" ] || ng "mda high: 5 以上で成立した"
mda m_edge ' 5.0 mediaanalysisd\n'
[ "$(mrc m_edge)" = "1" ] || ng "mda edge: ちょうど 5.0 で成立した（5 未満を要求）"
mda m_max ' 0.1 mediaanalysisd\n 7.0 mediaanalysisd\n'
[ "$(mrc m_max)" = "1" ] || ng "mda max: 複数のうち最大値で判定しない"
mda m_nan ' abc mediaanalysisd\n'
[ "$(mrc m_nan)" = "1" ] || ng "mda nan: 非数値で成立した"
mda m_psfail '' 1
[ "$(mrc m_psfail)" = "1" ] || ng "mda psfail: ps の失敗で成立した"
mda m_empty ''
[ "$(mrc m_empty)" = "1" ] || ng "mda empty: ps の出力が空で成立した"

# ---- orchestrate_macmini.sh の起動時の検査（HOME を一時ディレクトリに差し替え、実物の hw-lock に触れない）
mkdir -p "${T}/home/work" "${T}/base/docs/perf/logs/sme-default-on-ab-4156"
# 実物の RULE.v2.txt の §9 は記入済みのため、GB10-MODE だけを未決に戻した写しを置く（P-OWNER で止まることを確かめる）
sed 's/^GB10-MODE: .*/GB10-MODE: <未決>/' "${HERE}/RULE.v2.txt" > "${T}/base/docs/perf/logs/sme-default-on-ab-4156/RULE.v2.txt"
grep -q '^GB10-MODE: <未決>$' "${T}/base/docs/perf/logs/sme-default-on-ab-4156/RULE.v2.txt" || ng "orch: 未決の RULE.v2.txt の写しを作れない"
orch() { # orch <名前> <FOREIGN_GATE> -> 終了コード
  HOME="${T}/home" BASE="${T}/base" WORK="${T}/work-$1" TORCH_PY="/usr/bin/true" FOREIGN_GATE="$2" \
    bash "${ORCH}" > "${T}/orch-$1.log" 2>&1
}
for c in "nogate|" "rel|fake_gate.py" "missing|${T}/none.py"; do
  name="${c%%|*}"; g="${c#*|}"
  orch "${name}" "${g}" && ng "orch ${name}: 起動した"
  grep -q 'FOREIGN_GATE' "${T}/orch-${name}.log" || ng "orch ${name}: FOREIGN_GATE の理由が出ない"
  [ ! -e "${T}/home/work/.fandhe-hw-lock" ] || ng "orch ${name}: ロックを取った"
done
# 正しいパスなら FOREIGN_GATE の検査を通り、次の P-OWNER（写しの RULE.v2.txt §9 が未決）で止まる。ロックはまだ取らない
orch okgate "${T}/fake_gate.py" && ng "orch okgate: P-OWNER 未決のまま起動した"
grep -q 'P-OWNER 不成立' "${T}/orch-okgate.log" || ng "orch okgate: P-OWNER まで進まない"
[ ! -e "${T}/home/work/.fandhe-hw-lock" ] || ng "orch okgate: ロックを取った"

# 呼び出し順: ロック → P-RT（rt_off）→ mediaanalysisd → A/B。A/B に FOREIGN_GATE を渡す・ゲートの出力を収録する
ln_of() { grep -n -- "$1" "${ORCH}" | head -n 1 | cut -d: -f1; }
l_lock=$(ln_of 'until mkdir "${LOCK_DIR}"')
l_rt=$(ln_of 'echo "rt_off exit=')
l_mda=$(ln_of 'if check_mediaanalysisd "${OUT}/mediaanalysisd.txt"')
l_ab=$(ln_of 'bash run_ab_sme_default_on_4156_cpu.sh "${LABEL}"')
l_fg=$(ln_of 'FG_SRC=')
if [ -z "${l_lock}" ] || [ -z "${l_rt}" ] || [ -z "${l_mda}" ] || [ -z "${l_ab}" ] || [ -z "${l_fg}" ]; then
  ng "orch: 呼び出し箇所を見つけられない"
elif ! [ "${l_lock}" -lt "${l_rt}" ] || ! [ "${l_rt}" -lt "${l_mda}" ] || ! [ "${l_mda}" -lt "${l_ab}" ] || ! [ "${l_ab}" -lt "${l_fg}" ]; then
  ng "orch: 呼び出し順が ロック → P-RT → mediaanalysisd → A/B → 収録 でない"
fi
grep -q 'FOREIGN_GATE="${FOREIGN_GATE}" AB_HOST=macmini' "${ORCH}" || ng "orch: A/B に FOREIGN_GATE を渡していない"

# dry-run は FOREIGN_GATE なしで通り、計画に両方が出る
bash "${ORCH}" --dry-run > "${T}/dry.log" 2>&1 || ng "dry-run: 失敗"
grep -q '他ジョブの前提ゲート' "${T}/dry.log" || ng "dry-run: ゲートの表示が無い"
grep -q 'mediaanalysisd' "${T}/dry.log" || ng "dry-run: mediaanalysisd の表示が無い"

# 既存の self-test（変更なしで通ること）
bash "${ORCH}" --self-test-leak > /dev/null 2>&1 || ng "orchestrate --self-test-leak が通らない"
python3 -I "${HERE}/aggregate.py" --self-test > /dev/null 2>&1 || ng "aggregate.py --self-test が通らない"

if [ "${fail}" -ne 0 ]; then exit 1; fi
echo "self-test ok"
