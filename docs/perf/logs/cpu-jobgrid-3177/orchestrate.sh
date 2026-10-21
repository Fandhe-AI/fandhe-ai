#!/bin/bash
# イシュー #3177: m=64 級 GEMM の job 分割・再 pack・確保の計数と単スレッド GFLOPS・A/B を、
# 実機（Mac mini〈macmini〉／GB10〈gb10〉）で一気通貫に回すオーケストレーション。
# 判定規則は同ディレクトリ RULE.txt（計測前に固定済み。本スクリプトは規則を実装するだけで変えない）。
# 前例: docs/perf/logs/sme-macmini-3176/orchestrate_macmini.sh（負荷ゲート・.raw.log/.log 分離・
# 既存ログ上書き禁止・失敗即停止・P-TREE）。macOS の bash 3.2 で動くよう変数は ${VAR} と書く。
#
# 使い方（BASE は rsync で同期した base ツリー〈main そのまま。.rev-stamp 付き〉、WORK は同期ツリーの外）:
#   BASE=<絶対パス> WORK=<絶対パス> bash orchestrate.sh <macmini|gb10> --dry-run   # 計画のみ表示
#   BASE=<絶対パス> WORK=<絶対パス> bash orchestrate.sh <macmini|gb10>             # 実行
# 腕ツリー（jpw1・cap4・cap8）は BASE から target を除いて複製し ab_patch_*.diff を当てる。腕ツリーでは
# `cargo test` を実行しない（cap 腕の下限テストは構造上落ちるため。example のビルドだけを行う）。
# ログは ${WORK}/<machine> へ出す。ホスト名・ユーザー名は収録物へ書かない（マスク規則）。
set -u
export LC_ALL=C

MACHINE="${1:-}"
DRY=0
[ "${2:-}" = "--dry-run" ] && DRY=1
case "${MACHINE}" in
    macmini) GATE_MAX_LOAD1="5.0" ;;
    gb10) GATE_MAX_LOAD1="10.0" ;;
    *) echo "usage: orchestrate.sh <macmini|gb10> [--dry-run]" >&2; exit 2 ;;
esac
GATE_WAIT_S=30
GATE_TRIES=60
RUNS="1 2 3 4 5"
ARMS="base jpw1 cap4 cap8"
PKG="fandhe-ai-backend-cpu"
EX="cpu_jobgrid_bench"
COUNT_TEST="gemm_blis::jobgrid_probe::tests::jobgrid_probe_mlp_shapes"
REPO_LOG="docs/perf/logs/cpu-jobgrid-3177"

if [ "${DRY}" = "1" ]; then
    echo "# dry-run: machine=${MACHINE} 負荷ゲート load1 < ${GATE_MAX_LOAD1}（${GATE_WAIT_S}s 間隔・最大 ${GATE_TRIES} 回）"
    echo "1. env_info・uname・.rev-stamp の収録"
    echo "2. P-TREE: BASE から target を除いて arm-{jpw1,cap4,cap8} へ複製し patch 適用。差分 1 件（jpw1=mod.rs／cap=partition.rs）と patch sha256 を検査"
    echo "3. ビルド（逐次・計測前に全部）: base の example と lib テスト --no-run、各腕の example のみ"
    for r in ${RUNS}; do echo "4. 計数 run${r}: ${COUNT_TEST}（RAYON_NUM_THREADS unset・base）"; done
    for r in ${RUNS}; do echo "5. 単スレッド run${r}: RAYON_NUM_THREADS=1 ${EX} base threads"; done
    for r in ${RUNS}; do echo "6. serial_ref run${r}: ${EX} base serial_ref"; done
    for r in ${RUNS}; do echo "7. A/B run${r}: 腕 ${ARMS} を run ごとにローテーション（RAYON_NUM_THREADS unset）"; done
    echo "8. DONE センチネル"
    exit 0
fi

for v in BASE WORK; do
    eval "val=\${${v}:-}"
    case "${val}" in
        /*) ;;
        *) echo "error: ${v} は絶対パスで指定すること" >&2; exit 1 ;;
    esac
done
if [ -n "${RAYON_NUM_THREADS:-}" ]; then
    echo "error: RAYON_NUM_THREADS が設定されている。unset して再実行すること" >&2
    exit 1
fi

OUT="${WORK}/${MACHINE}"
mkdir -p "${WORK}" || exit 1
if [ -e "${OUT}" ]; then
    echo "既存の出力先があるため停止（上書き禁止）: ${OUT}" >&2
    exit 1
fi
mkdir -p "${OUT}/count" "${OUT}/st" "${OUT}/serial_ref" "${OUT}/ab" || exit 1

mask() { sed -e "s#${HOME}#<home>#g" -e "s#${WORK}#<work>#g" -e "s#${BASE}#<base>#g" -e "s#$(whoami)#<user>#g"; }
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

# 1. 環境記録（内部ホスト名・ユーザー名は書かない。uname は -n を使わない）
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
} 2>&1 | mask > "${OUT}/env_info.txt"
uname -srvm > "${OUT}/uname.txt"
cp "${BASE}/.rev-stamp" "${OUT}/.rev-stamp" || { echo ".rev-stamp が無い" >&2; exit 1; }

# 2. P-TREE（arm ツリーの生成と指紋検査）
: > "${OUT}/tree_verify.txt"
: > "${OUT}/patch_sha256.txt"
: > "${OUT}/patch_apply.log"
for arm in jpw1 cap4 cap8; do
    PATCH="${BASE}/${REPO_LOG}/ab_patch_${arm}.diff"
    TREE="${WORK}/arm-${arm}"
    [ -e "${TREE}" ] && { echo "既存の腕ツリーがあるため停止: ${TREE}" >&2; exit 1; }
    mkdir -p "${TREE}" || exit 1
    (cd "${BASE}" && tar cf - --exclude=./target --exclude=./.git .) | (cd "${TREE}" && tar xf -) || { echo "複製に失敗" >&2; exit 1; }
    if command -v sha256sum >/dev/null 2>&1; then
        sha="$(sha256sum "${PATCH}" | awk '{print $1}')"
    else
        sha="$(shasum -a 256 "${PATCH}" | awk '{print $1}')"
    fi
    echo "${arm} ${sha}" >> "${OUT}/patch_sha256.txt"
    (cd "${TREE}" && patch -p1 --forward < "${PATCH}") >> "${OUT}/patch_apply.log" 2>&1 \
        || { echo "patch 適用に失敗（${arm}）-> P-TREE 不成立" >&2; exit 1; }
    case "${arm}" in
        jpw1) want="crates/backend-cpu/src/gemm_blis/mod.rs" ;;
        *) want="crates/backend-cpu/src/gemm_blis/partition.rs" ;;
    esac
    (cd "${BASE}" && find . -type f -not -path './target*' -not -path './.git/*' | sort) > "${WORK}/files_base.txt"
    (cd "${TREE}" && find . -type f -not -path './target*' -not -path './.git/*' -not -name '*.orig' | sort) > "${WORK}/files_${arm}.txt"
    cmp -s "${WORK}/files_base.txt" "${WORK}/files_${arm}.txt" || { echo "ファイル一覧が不一致（${arm}）-> P-TREE 不成立" >&2; exit 1; }
    : > "${WORK}/changed_${arm}.txt"
    while read -r f; do
        cmp -s "${BASE}/${f}" "${TREE}/${f}" || echo "${f}" >> "${WORK}/changed_${arm}.txt"
    done < "${WORK}/files_base.txt"
    if [ "$(cat "${WORK}/changed_${arm}.txt")" != "./${want}" ]; then
        echo "差分ファイルが ${want} の 1 件だけではない（${arm}）-> P-TREE 不成立" >&2
        cat "${WORK}/changed_${arm}.txt" >&2
        exit 1
    fi
    echo "${arm} ./${want}" >> "${OUT}/tree_verify.txt"
done
echo "P-TREE ok"

# 3. ビルド（全部を逐次に済ませてから計測。ビルドと計測は並走させない）
build() { # build <label> <tree> <args...>
    label="$1"; tree="$2"; shift 2
    (cd "${tree}" && CARGO_TARGET_DIR="${WORK}/target-${label}" cargo "$@") > "${OUT}/build-${label}-$1.log" 2>&1 \
        || { echo "ビルド失敗: ${label} $*" >&2; tail -20 "${OUT}/build-${label}-$1.log" >&2; exit 1; }
}
build base "${BASE}" build --release -p "${PKG}" --example "${EX}"
build base "${BASE}" test -p "${PKG}" --release --lib --no-run
for arm in jpw1 cap4 cap8; do
    build "${arm}" "${WORK}/arm-${arm}" build --release -p "${PKG}" --example "${EX}"
done
echo "build ok"

# 実行ヘルパー。ログ先頭に腕・run・rev・開始時刻を書く（# 始まりの行は集計が読み飛ばす）
header() { # header <file> <series> <run> <arm>
    echo "# series=$2 run=$3 arm=$4 rev=$(head -n1 "${OUT}/.rev-stamp") start=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$1"
}
guard_new() { [ -e "$1" ] && { echo "既存ログがあるため停止（上書き禁止）: $1" >&2; exit 1; }; return 0; }
bin_of() { # bin_of <arm>
    if [ "$1" = "base" ]; then echo "${WORK}/target-base/release/examples/${EX}"; else echo "${WORK}/target-$1/release/examples/${EX}"; fi
}
run_bench() { # run_bench <series> <run> <arm> <logfile> <threads|serial_ref> [env RAYON_NUM_THREADS=1]
    series="$1"; run="$2"; arm="$3"; log="$4"; mode="$5"; st="${6:-}"
    guard_new "${log}"
    if ! gate "${series}" "${run}" "${arm}"; then
        echo "unavailable: ${series} run${run} ${arm}（負荷ゲート不成立）" >&2
        return 0
    fi
    header "${log}" "${series}" "${run}" "${arm}"
    if [ -n "${st}" ]; then
        RAYON_NUM_THREADS=1 "$(bin_of "${arm}")" "${arm}" "${mode}" >> "${log}" 2> "${log}.err"
    else
        "$(bin_of "${arm}")" "${arm}" "${mode}" >> "${log}" 2> "${log}.err"
    fi
    rc=$?
    echo "END series=${series} run=${run} arm=${arm} exit=${rc} end_load1=$(load1_now || true)" >> "${OUT}/load_gate.log"
    if [ "${rc}" -ne 0 ]; then echo "FAILED: ${log} (exit=${rc})" >&2; exit 1; fi
}

# 4. 計数（cfg(test) あり。計数専用。native 腕は機体の既定プール）
for r in ${RUNS}; do
    log="${OUT}/count/run${r}.log"; raw="${OUT}/count/run${r}.raw.log"
    guard_new "${log}"
    if ! gate count "${r}" base; then echo "unavailable: count run${r}" >&2; continue; fi
    (cd "${BASE}" && CARGO_TARGET_DIR="${WORK}/target-base" cargo test -p "${PKG}" --release --lib -- --ignored \
        "${COUNT_TEST}" --exact --nocapture --test-threads=1) 2>&1 | mask > "${raw}"
    rc=${PIPESTATUS[0]}
    header "${log}" count "${r}" base
    sed 's/^test [^ ]* \.\.\. //' "${raw}" | grep -E '^probe=jobgrid ' >> "${log}"
    grc=$?
    echo "END series=count run=${r} arm=base exit=${rc} grep_exit=${grc} end_load1=$(load1_now || true)" >> "${OUT}/load_gate.log"
    if [ "${rc}" -ne 0 ] || [ "${grc}" -ne 0 ]; then echo "FAILED: count run${r} (exit=${rc} grep=${grc})" >&2; exit 1; fi
done

# 5. 単スレッド GFLOPS（RAYON_NUM_THREADS=1）と 6. 直列参照
for r in ${RUNS}; do run_bench st "${r}" base "${OUT}/st/run${r}.log" threads st1; done
for r in ${RUNS}; do run_bench serial_ref "${r}" base "${OUT}/serial_ref/run${r}.log" serial_ref; done

# 7. A/B（既定プール。腕の起動順は run ごとにローテーション）
for r in ${RUNS}; do
    set -- ${ARMS}
    shift_n=$(( (r - 1) % 4 ))
    order="$*"
    i=0
    while [ "${i}" -lt "${shift_n}" ]; do
        first="${order%% *}"; rest="${order#* }"
        order="${rest} ${first}"
        i=$((i + 1))
    done
    echo "ab run=${r} order=${order}" >> "${OUT}/load_gate.log"
    for arm in ${order}; do
        run_bench ab "${r}" "${arm}" "${OUT}/ab/run${r}-${arm}.log" threads
    done
done

for f in "${OUT}"/*/*.err; do
    [ -s "${f}" ] || rm -f "${f}"
done
echo "finished at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${OUT}/DONE"
echo "完了"
