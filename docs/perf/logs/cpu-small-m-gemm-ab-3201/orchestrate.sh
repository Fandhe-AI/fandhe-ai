#!/bin/bash
# イシュー #3201: m が小さい GEMM の分割方式（G1 job 統合＋A pack 共有／G2 常駐 W の事前 pack／
# GB10 affinity）を、実機（Mac mini〈macmini〉／GB10〈gb10〉）で N 腕 A/B するオーケストレーション。
# 判定規則は同ディレクトリ RULE.txt・RULE-addendum.txt（計測前に固定。本スクリプトは規則を実装するだけで変えない）。
# 前例: ../cpu-jobgrid-3177/orchestrate.sh（負荷ゲート・.rev-stamp・P-TREE・上書き禁止・失敗即停止）と
# scripts/bench/framework-compare/run_ab_thread_policy_macmini_cpu.sh（bench-fandhe の path patch 計測・
# PyTorch 同席計測）。既存 runner は before/after の 2 腕型で N 腕ローテーションに使えないため本スクリプトを置く。
# macOS の bash 3.2 で動くよう変数は ${VAR}・mapfile／連想配列／jq は使わない。
#
# 使い方（BASE は rsync した main ツリー〈.rev-stamp 付き〉、WORK は同期ツリーの外、いずれも絶対パス）:
#   BASE=<絶対> WORK=<絶対> AB_TORCH_PYTHON=<絶対> bash orchestrate.sh <macmini|gb10> --dry-run
#   BASE=<絶対> WORK=<絶対> AB_TORCH_PYTHON=<絶対> bash orchestrate.sh <macmini|gb10>
# ログは ${WORK}/<machine> へ出す。ホスト名・ユーザー名・ホーム絶対パスは収録物へ書かない（mask）。
# 失敗したら即停止し FAILED を残す（DONE は全工程成功時だけ）。run の差し替え・再実行はしない。
set -u
export LC_ALL=C

MACHINE="${1:-}"
DRY=0
[ "${2:-}" = "--dry-run" ] && DRY=1
case "${MACHINE}" in
    macmini)
        GATE_MAX_LOAD1="3.0"
        ARMS="off merge mergeprepack"
        ;;
    gb10)
        GATE_MAX_LOAD1="10.0"
        ARMS="off merge mergeprepack off-aff merge-aff mergeprepack-aff"
        ;;
    *) echo "usage: orchestrate.sh <macmini|gb10> [--dry-run]" >&2; exit 2 ;;
esac
GATE_WAIT_S=30
GATE_TRIES=60
RUNS="1 2 3 4 5"
PKG="fandhe-ai-backend-cpu"
EX="cpu_jobgrid_bench"
REPO_LOG="docs/perf/logs/cpu-small-m-gemm-ab-3201"
BENCH_PY_REL="docs/perf/logs/lowlayer-diagnosis-2026-09-12/bench_py.py"
F_G1="crates/backend-cpu/src/gemm_blis/small_m.rs"
F_G2="crates/backend-cpu/src/gemm_blis/prepack.rs"
F_AFF="crates/backend-cpu/src/gb10_affinity.rs"

# 腕 -> 当てる patch 種別（merge／prepack／aff）。off は複製しない（BASE をそのまま使う）。
patches_of() {
    case "$1" in
        off) echo "" ;;
        merge) echo "merge" ;;
        mergeprepack) echo "merge prepack" ;;
        off-aff) echo "aff" ;;
        merge-aff) echo "merge aff" ;;
        mergeprepack-aff) echo "merge prepack aff" ;;
        *) echo "?" ;;
    esac
}
# 腕 -> ゲート期待値（true|false）
want_g1() { case "$1" in merge*|mergeprepack*) echo true ;; *) echo false ;; esac; }
want_g2() { case "$1" in mergeprepack*) echo true ;; *) echo false ;; esac; }
want_aff() { case "$1" in *-aff) echo true ;; *) echo false ;; esac; }
tree_of() { if [ "$1" = "off" ]; then echo "${BASE}"; else echo "${WORK}/arm-$1"; fi; }

if [ "${DRY}" = "1" ]; then
    echo "# dry-run: machine=${MACHINE} 腕=[${ARMS}] 負荷ゲート load1 < ${GATE_MAX_LOAD1}（${GATE_WAIT_S}s 間隔・最大 ${GATE_TRIES} 回）"
    echo "1. env_info・uname・.rev-stamp の収録"
    echo "2. P-TREE: BASE から target・.git を除いて腕ごとに複製し patch 適用（off は BASE）"
    for a in ${ARMS}; do echo "   arm=${a} patches=[$(patches_of "${a}")] G1=$(want_g1 "${a}") G2=$(want_g2 "${a}") aff=$(want_aff "${a}")"; done
    echo "   tree_verify（ファイル一覧一致・変更ファイル集合が patch 対象と完全一致）・patch_sha256・ゲート定数の読み取り照合"
    echo "3. ビルド（逐次・計測前に全部）: 腕ごとに ${EX}、bench-fandhe（共有 target・path patch・exe を腕別にコピー）"
    echo "4. 機構発火の事前確認（fail-closed）: small_m_shared_a_report・resident_w_prepack_fire_report"
    [ "${MACHINE}" = "gb10" ] && echo "   gb10_affinity_report_dump（EXPECT_GB10_AFFINITY_POOL_ACTIVE=1|0）"
    echo "5. bit 一致ログ（off のみ）: small_m::tests・prepack::tests・ops::resident_w_prepack_tests"
    set -- ${ARMS}
    n=$#
    for r in ${RUNS}; do
        set -- ${ARMS}
        shift_n=$(( (r - 1) % n ))
        order="$*"
        i=0
        while [ "${i}" -lt "${shift_n}" ]; do first="${order%% *}"; rest="${order#* }"; order="${rest} ${first}"; i=$((i + 1)); done
        echo "6. round ${r}: 腕順=[${order}]（腕ごと: ゲート→${EX} threads→bench-fandhe train/infer×fresh/reuse）→ PyTorch train/infer"
    done
    echo "7. DONE センチネル（全工程成功時のみ）"
    exit 0
fi

for v in BASE WORK AB_TORCH_PYTHON; do
    eval "val=\${${v}:-}"
    case "${val}" in
        /*) ;;
        *) echo "error: ${v} は絶対パスで指定すること" >&2; exit 1 ;;
    esac
done
[ -x "${AB_TORCH_PYTHON}" ] || { echo "error: AB_TORCH_PYTHON が実行可能でない" >&2; exit 1; }
if [ -n "${RAYON_NUM_THREADS:-}" ]; then
    echo "error: RAYON_NUM_THREADS が設定されている。unset して再実行すること" >&2
    exit 1
fi
[ -f "${BASE}/${BENCH_PY_REL}" ] || { echo "error: bench_py.py が無い" >&2; exit 1; }

OUT="${WORK}/${MACHINE}"
mkdir -p "${WORK}" || exit 1
if [ -e "${OUT}" ]; then
    echo "既存の出力先があるため停止（上書き禁止）: ${OUT}" >&2
    exit 1
fi
mkdir -p "${OUT}/ab" "${OUT}/fc" "${OUT}/prereq" "${WORK}/bin" || exit 1

HOSTN="$(hostname 2>/dev/null || echo __nohost__)"
HOSTS="$(hostname -s 2>/dev/null || echo __nohost__)"
mask() { sed -e "s#${HOME}#<home>#g" -e "s#${WORK}#<work>#g" -e "s#${BASE}#<base>#g" -e "s#$(whoami)#<user>#g" -e "s#${HOSTN}#masked#g" -e "s#${HOSTS}#masked#g"; }
die() {
    echo "FAILED: $*" >&2
    echo "$* at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | mask > "${OUT}/FAILED"
    exit 1
}
load1_now() {
    if [ -r /proc/loadavg ]; then
        raw="$(awk '{print $1}' /proc/loadavg)"
    else
        raw="$(sysctl -n vm.loadavg 2>/dev/null | awk '{print $2}')"
    fi
    case "${raw}" in
        ''|*[!0-9.]*|.*|*.*.*) echo NA; return 1 ;;
    esac
    echo "${raw}"
}
gate() { # gate <series> <run> <arm>
    waited=0; tries=0
    while :; do
        if l1="$(load1_now)"; then
            ok="$(awk -v l="${l1}" -v m="${GATE_MAX_LOAD1}" 'BEGIN{print (l < m) ? 1 : 0}')"
        else
            ok=0
        fi
        if [ "${ok}" = "1" ]; then
            echo "START series=$1 run=$2 arm=$3 gate=pass load1=${l1} waited_s=${waited} at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${OUT}/load_gate.log"
            return 0
        fi
        tries=$((tries + 1))
        if [ "${tries}" -ge "${GATE_TRIES}" ]; then
            echo "START series=$1 run=$2 arm=$3 gate=unavailable load1=${l1} waited_s=${waited} at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${OUT}/load_gate.log"
            return 1
        fi
        sleep "${GATE_WAIT_S}"
        waited=$((waited + GATE_WAIT_S))
    done
}
sha_of() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'; else shasum -a 256 "$1" | awk '{print $1}'; fi; }
read_gate() { # read_gate <file> <CONST> -> true|false|(空)。読み取りのみでソースは実行しない
    sed -nE "s/^[[:space:]]*pub(\([a-z]+\))?[[:space:]]+const[[:space:]]+$2[[:space:]]*:[[:space:]]*bool[[:space:]]*=[[:space:]]*(true|false)[[:space:]]*;.*/\2/p" "$1" | head -n1
}

# 1. 環境記録（uname は -n を使わない）
{
    echo "machine=${MACHINE}"
    uname -srvm
    if [ "$(uname -s)" = "Darwin" ]; then
        echo "chip=$(sysctl -n machdep.cpu.brand_string)"
        echo "ncpu=$(sysctl -n hw.ncpu)"
        sw_vers
        for p in 0 1; do
            sysctl -n hw.perflevel${p}.name hw.perflevel${p}.logicalcpu 2>/dev/null | tr '\n' ' ' | sed "s/^/perflevel${p}: /"
            echo
        done
    else
        lscpu | grep -E '^(Model name|CPU\(s\)|Thread|Core|Socket|Architecture)'
        grep PRETTY_NAME /etc/os-release
    fi
    rustc -Vv
    cargo -V
    echo "gate_max_load1=${GATE_MAX_LOAD1}"
    echo "rayon_num_threads=unset"
    echo "bench_py_sha256=$(sha_of "${BASE}/${BENCH_PY_REL}")"
    echo "torch=$("${AB_TORCH_PYTHON}" -c 'import torch; print(torch.__version__, torch.get_num_threads())' 2>&1 | tail -n1)"
} 2>&1 | mask > "${OUT}/env_info.txt"
uname -srvm > "${OUT}/uname.txt"
cp "${BASE}/.rev-stamp" "${OUT}/.rev-stamp" || die ".rev-stamp が無い"

# 2. P-TREE（腕ツリーの生成・指紋検査・ゲート定数の照合）
: > "${OUT}/tree_verify.txt"
: > "${OUT}/patch_sha256.txt"
: > "${OUT}/patch_apply.log"
for k in merge prepack aff; do
    echo "${k} $(sha_of "${BASE}/${REPO_LOG}/ab_patch_${k}.diff")" >> "${OUT}/patch_sha256.txt"
done
(cd "${BASE}" && find . -type f -not -path './target*' -not -path './.git/*' | sort) > "${WORK}/files_base.txt"
for arm in ${ARMS}; do
    pk="$(patches_of "${arm}")"
    [ "${pk}" = "?" ] && die "未知の腕: ${arm}"
    TREE="$(tree_of "${arm}")"
    want_files=""
    if [ "${arm}" != "off" ]; then
        [ -e "${TREE}" ] && die "既存の腕ツリーがあるため停止: ${TREE}"
        mkdir -p "${TREE}" || die "mkdir 失敗"
        (cd "${BASE}" && tar cf - --exclude=./target --exclude=./.git .) | (cd "${TREE}" && tar xf -) || die "複製に失敗(${arm})"
        for k in ${pk}; do
            (cd "${TREE}" && patch -p1 --forward < "${BASE}/${REPO_LOG}/ab_patch_${k}.diff") >> "${OUT}/patch_apply.log" 2>&1 \
                || die "patch 適用に失敗（${arm}/${k}）-> P-TREE 不成立"
            case "${k}" in
                merge) want_files="${want_files}./${F_G1}
" ;;
                prepack) want_files="${want_files}./${F_G2}
" ;;
                aff) want_files="${want_files}./${F_AFF}
" ;;
            esac
        done
        (cd "${TREE}" && find . -type f -not -path './target*' -not -path './.git/*' -not -name '*.orig' | sort) > "${WORK}/files_${arm}.txt"
        cmp -s "${WORK}/files_base.txt" "${WORK}/files_${arm}.txt" || die "ファイル一覧が不一致（${arm}）-> P-TREE 不成立"
        : > "${WORK}/changed_${arm}.txt"
        while read -r f; do
            cmp -s "${BASE}/${f}" "${TREE}/${f}" || echo "${f}" >> "${WORK}/changed_${arm}.txt"
        done < "${WORK}/files_base.txt"
        want_sorted="$(printf '%s' "${want_files}" | sort)"
        got_sorted="$(sort "${WORK}/changed_${arm}.txt")"
        [ "${want_sorted}" = "${got_sorted}" ] || die "変更ファイル集合が patch 対象と不一致（${arm}）-> P-TREE 不成立"
        echo "${arm} changed=$(echo "${got_sorted}" | tr '\n' ',')" >> "${OUT}/tree_verify.txt"
    else
        echo "off changed=(BASE そのまま)" >> "${OUT}/tree_verify.txt"
    fi
    g1="$(read_gate "${TREE}/${F_G1}" SMALL_M_JOB_MERGE_ENABLED)"
    g2="$(read_gate "${TREE}/${F_G2}" RESIDENT_W_PREPACK_ENABLED)"
    ga="$(read_gate "${TREE}/${F_AFF}" GB10_AFFINITY_ENABLED)"
    [ "${g1}" = "$(want_g1 "${arm}")" ] || die "G1 定数が期待と不一致（${arm}: ${g1:-unreadable}）"
    [ "${g2}" = "$(want_g2 "${arm}")" ] || die "G2 定数が期待と不一致（${arm}: ${g2:-unreadable}）"
    [ "${ga}" = "$(want_aff "${arm}")" ] || die "affinity 定数が期待と不一致（${arm}: ${ga:-unreadable}）"
    echo "${arm} G1=${g1} G2=${g2} aff=${ga}" >> "${OUT}/tree_verify.txt"
done
echo "P-TREE ok"

# 3. ビルド（全部を逐次に済ませてから計測。ビルドと計測は並走させない）
for arm in ${ARMS}; do
    TREE="$(tree_of "${arm}")"
    (cd "${TREE}" && CARGO_TARGET_DIR="${WORK}/target-${arm}" cargo build --release -p "${PKG}" --example "${EX}") \
        > "${OUT}/build-${arm}-example.log" 2>&1 || { tail -20 "${OUT}/build-${arm}-example.log" >&2; die "example ビルド失敗(${arm})"; }
    FC="${TREE}/scripts/bench/framework-compare"
    cp "${FC}/Cargo.lock" "${WORK}/Cargo.lock.${arm}.bak" || die "Cargo.lock 退避失敗"
    PCFG="patch.crates-io.fandhe-ai.path=\"${TREE}/crates/facade\""
    (cd "${FC}" && cargo build --release -p bench-fandhe --target-dir "${WORK}/target-fc" --config "${PCFG}") \
        > "${OUT}/build-${arm}-bench-fandhe.log" 2>&1 || { tail -20 "${OUT}/build-${arm}-bench-fandhe.log" >&2; cp "${WORK}/Cargo.lock.${arm}.bak" "${FC}/Cargo.lock"; die "bench-fandhe ビルド失敗(${arm})"; }
    cp "${WORK}/target-fc/release/bench-fandhe" "${WORK}/bin/bench-fandhe-${arm}" || die "exe コピー失敗(${arm})"
    (cd "${FC}" && cargo tree -p bench-fandhe --depth 1 --config "${PCFG}") > "${OUT}/tree-${arm}.txt" 2>&1
    cp "${WORK}/Cargo.lock.${arm}.bak" "${FC}/Cargo.lock"
    grep -qE 'fandhe-ai v[0-9.]+ \(.*crates/facade\)' "${OUT}/tree-${arm}.txt" || die "fandhe-ai が腕の crates/facade に解決されていない(${arm})"
    echo "${arm} $(sha_of "${WORK}/bin/bench-fandhe-${arm}")" >> "${OUT}/bench_sha256.txt"
done
for arm in ${ARMS}; do mask < "${OUT}/tree-${arm}.txt" > "${OUT}/tree-${arm}.txt.m" && mv "${OUT}/tree-${arm}.txt.m" "${OUT}/tree-${arm}.txt"; done
echo "build ok"

# 4. 機構発火の事前確認（fail-closed。出力は prereq/<arm>.log へ。RULE-addendum 1・2）
# 全件実行（--lib のフィルタなし）はしない。ゲート定数 1 件の report テストだけを指定する。
run_report() { # run_report <arm> <label> <cargo test args...> ; 出力を prereq/<arm>.log へ追記
    arm="$1"; label="$2"; shift 2
    TREE="$(tree_of "${arm}")"
    echo "### ${label}" >> "${OUT}/prereq/${arm}.log"
    (cd "${TREE}" && CARGO_TARGET_DIR="${WORK}/target-${arm}" "$@") > "${WORK}/prereq.tmp" 2>&1
    rc=$?
    mask < "${WORK}/prereq.tmp" | sed "s/^test [^ ]* \.\.\. //" >> "${OUT}/prereq/${arm}.log"
    echo "exit=${rc}" >> "${OUT}/prereq/${arm}.log"
    return ${rc}
}
for arm in ${ARMS}; do
    : > "${OUT}/prereq/${arm}.log"
    w1="$(want_g1 "${arm}")"; w2="$(want_g2 "${arm}")"
    # G1: small_m_shared_a_report
    run_report "${arm}" "G1 small_m_shared_a_report" cargo test -p "${PKG}" --release --lib -- --ignored small_m_shared_a_report --nocapture --test-threads=1 \
        || die "G1 report 失敗（${arm}）。前提不成立（prereq/${arm}.log）"
    lines="$(grep -c '^probe=small_m_shared_a ' "${OUT}/prereq/${arm}.log")"
    [ "${lines}" -gt 0 ] || die "G1 report に probe 行が無い（${arm}）"
    grep -q 'running 1 test' "${OUT}/prereq/${arm}.log" || die "G1 report が 1 件指定になっていない（${arm}）"
    bad="$(grep '^probe=small_m_shared_a ' "${OUT}/prereq/${arm}.log" | grep -vc "gate=${w1} ")"
    [ "${bad}" -eq 0 ] || die "G1 の gate 値が期待 ${w1} と食い違う行がある（${arm}）"
    if [ "${w1}" = "true" ]; then
        # 適格形状（eligible=true）が l1_fwd・l2_fwd・l1_dx・l2_dx のいずれかに最低 1 行あり、
        # 適格行は a_shared_bands == row_bands であること（テスト内 assert は適格時のみ働くため行を照合する）
        elig="$(grep '^probe=small_m_shared_a ' "${OUT}/prereq/${arm}.log" | grep -c ' eligible=true ')"
        [ "${elig}" -gt 0 ] || die "G1 ON なのに適格行が無い（${arm}）"
        badshare="$(grep '^probe=small_m_shared_a ' "${OUT}/prereq/${arm}.log" | grep ' eligible=true ' | awk '{rb="";sb="";for(i=1;i<=NF;i++){if($i ~ /^row_bands=/)rb=substr($i,11);if($i ~ /^a_shared_bands=/)sb=substr($i,16)} if(rb==""||sb==""||rb!=sb)c++} END{print c+0}')"
        [ "${badshare}" -eq 0 ] || die "G1 適格行で a_shared_bands != row_bands（${arm}）"
    else
        elig="$(grep '^probe=small_m_shared_a ' "${OUT}/prereq/${arm}.log" | grep -c ' eligible=true ')"
        [ "${elig}" -eq 0 ] || die "G1 OFF なのに適格行がある（${arm}）"
    fi
    # G2: resident_w_prepack_fire_report（ON なら builds==1・hits>=1 はテスト内 assert。OFF は cache_len==0）
    run_report "${arm}" "G2 resident_w_prepack_fire_report" cargo test -p "${PKG}" --release --lib -- --ignored resident_w_prepack_fire_report --nocapture --test-threads=1 \
        || die "G2 report 失敗（${arm}）。前提不成立（prereq/${arm}.log）"
    grep -q 'running 1 test' "${OUT}/prereq/${arm}.log" || die "G2 report が 1 件指定になっていない（${arm}）"
    n2="$(grep -c '^probe=resident_w_prepack ' "${OUT}/prereq/${arm}.log")"
    [ "${n2}" -eq 4 ] || die "G2 probe 行が 4 でない（${arm}: ${n2}）"
    bad="$(grep '^probe=resident_w_prepack ' "${OUT}/prereq/${arm}.log" | grep -vc "gate=${w2} ")"
    [ "${bad}" -eq 0 ] || die "G2 の gate 値が期待 ${w2} と食い違う（${arm}）"
    if [ "${w2}" = "true" ]; then
        bad="$(grep '^probe=resident_w_prepack ' "${OUT}/prereq/${arm}.log" | grep -vc ' builds=1 hits=[1-9]')"
        [ "${bad}" -eq 0 ] || die "G2 ON なのに builds=1・hits>=1 でない行がある（${arm}）"
    else
        bad="$(grep '^probe=resident_w_prepack ' "${OUT}/prereq/${arm}.log" | grep -vc ' cache_len=0 ')"
        [ "${bad}" -eq 0 ] || die "G2 OFF なのに cache_len != 0（${arm}）"
    fi
    # affinity（GB10 のみ）
    if [ "${MACHINE}" = "gb10" ]; then
        wa="$(want_aff "${arm}")"; ev=0; [ "${wa}" = "true" ] && ev=1
        TREE="$(tree_of "${arm}")"
        echo "### gb10_affinity_report_dump EXPECT=${ev}" >> "${OUT}/prereq/${arm}.log"
        (cd "${TREE}" && CARGO_TARGET_DIR="${WORK}/target-${arm}" EXPECT_GB10_AFFINITY_POOL_ACTIVE="${ev}" \
            cargo test -p "${PKG}" --release --test gb10_affinity_report -- --ignored --nocapture) > "${WORK}/prereq.tmp" 2>&1
        rc=$?
        mask < "${WORK}/prereq.tmp" | sed "s/^test [^ ]* \.\.\. //" >> "${OUT}/prereq/${arm}.log"
        echo "exit=${rc}" >> "${OUT}/prereq/${arm}.log"
        [ "${rc}" -eq 0 ] || die "affinity report 失敗（${arm}）。前提不成立"
        grep -q "^gb10_affinity_report enabled=${wa} " "${OUT}/prereq/${arm}.log" || die "affinity report の enabled が期待 ${wa} と不一致（${arm}）"
    fi
    echo "prereq ok: ${arm}"
done
rm -f "${WORK}/prereq.tmp"

# 5. bit 一致ログ（off＝BASE のみ。ゲート OFF のまま _with(true) 経路を強制するテスト群）
(cd "${BASE}" && CARGO_TARGET_DIR="${WORK}/target-off" cargo test -p "${PKG}" --release --lib -- \
    small_m::tests prepack::tests ops::resident_w_prepack_tests) > "${WORK}/bitmatch.tmp" 2>&1
rc=$?
mask < "${WORK}/bitmatch.tmp" > "${OUT}/bitmatch.log"
echo "exit=${rc}" >> "${OUT}/bitmatch.log"
rm -f "${WORK}/bitmatch.tmp"
[ "${rc}" -eq 0 ] || die "bit 一致テストが失敗（FAIL。bitmatch.log）"
grep -qE 'test result: ok\. [1-9][0-9]* passed; 0 failed' "${OUT}/bitmatch.log" || die "bit 一致テストの pass 件数を確認できない"
echo "bitmatch ok"

# 実行ヘルパー
header() { # header <file> <series> <run> <arm>
    echo "# series=$2 run=$3 arm=$4 rev=$(head -n1 "${OUT}/.rev-stamp") start=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$1"
}
guard_new() { [ -e "$1" ] && die "既存ログがあるため停止（上書き禁止）: $1"; return 0; }
count_rows() { # count_rows <jsonl> <mode> -> 行数
    python3 -I -c '
import json, sys
n = 0
for line in open(sys.argv[1]):
    line = line.strip()
    if line and json.loads(line).get("mode", "fresh") == sys.argv[2]:
        n += 1
print(n)' "$1" "$2"
}
run_fc_cell() { # run_fc_cell <arm> <task> <mode>
    arm="$1"; task="$2"; mode="$3"
    f="${OUT}/fc/results-${arm}-${task}.jsonl"
    [ -f "${f}" ] || : > "${f}"
    before="$(count_rows "${f}" "${mode}")"
    "${WORK}/bin/bench-fandhe-${arm}" --task "${task}" --size 64 --device cpu --mode "${mode}" --out "${f}" 2>> "${OUT}/fc/err-${arm}.log" >/dev/null \
        || die "bench-fandhe 失敗（${arm} ${task} ${mode}）"
    after="$(count_rows "${f}" "${mode}")"
    [ "${after}" -eq $((before + 1)) ] || die "記録行数が +1 でない（${arm} ${task} ${mode}: ${before}->${after}）"
}

# 6. 計測（round 1〜5。腕の起動順は round ごとにローテーション。腕ごとに独立プロセス）
set -- ${ARMS}
NARMS=$#
for r in ${RUNS}; do
    set -- ${ARMS}
    shift_n=$(( (r - 1) % NARMS ))
    order="$*"
    i=0
    while [ "${i}" -lt "${shift_n}" ]; do first="${order%% *}"; rest="${order#* }"; order="${rest} ${first}"; i=$((i + 1)); done
    echo "ORDER run=${r} order=${order}" >> "${OUT}/load_gate.log"
    for arm in ${order}; do
        log="${OUT}/ab/run${r}-${arm}.log"
        guard_new "${log}"
        if ! gate ab "${r}" "${arm}"; then
            echo "unavailable: ab run${r} ${arm}（負荷ゲート不成立）" >&2
            continue
        fi
        header "${log}" ab "${r}" "${arm}"
        "${WORK}/target-${arm}/release/examples/${EX}" "${arm}" threads >> "${log}" 2> "${log}.err"
        rc=$?
        [ "${rc}" -eq 0 ] || die "${EX} 失敗（run${r} ${arm} exit=${rc}）"
        for task in train infer; do
            for mode in fresh reuse; do run_fc_cell "${arm}" "${task}" "${mode}"; done
        done
        echo "END series=ab run=${r} arm=${arm} exit=0 end_load1=$(load1_now || true)" >> "${OUT}/load_gate.log"
    done
    # 同じ round の PyTorch（fresh のみ。判定には使わない）
    if gate torch "${r}" torch; then
        for task in train infer; do
            "${AB_TORCH_PYTHON}" "${BASE}/${BENCH_PY_REL}" --framework pytorch --task "${task}" --device cpu --size 64 \
                --out "${OUT}/fc/results-torch-${task}.jsonl" >/dev/null 2>> "${OUT}/fc/err-torch.log" \
                || die "PyTorch 計測失敗（run${r} ${task}）"
        done
        echo "END series=torch run=${r} arm=torch exit=0 end_load1=$(load1_now || true)" >> "${OUT}/load_gate.log"
    else
        echo "unavailable: torch run${r}" >&2
    fi
    echo "round ${r}: load1=$(load1_now || true)" >> "${OUT}/uptime.log"
done

for f in "${OUT}"/*/*.err "${OUT}"/fc/err-*.log; do
    [ -e "${f}" ] && [ ! -s "${f}" ] && rm -f "${f}"
done
echo "finished at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${OUT}/DONE"
echo "完了"
