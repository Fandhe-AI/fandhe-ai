#!/bin/bash
# イシュー #3614: 6 形状の A pack・B pack・kernel の時間内訳を実機（Mac mini〈macmini〉／
# GB10〈gb10〉）で測るランナー。判定規則は同ディレクトリ RULE.txt（計測前に固定済み。本スクリプトは
# 規則を実装するだけで変えない）。手順は #3177 の orchestrate.sh（負荷ゲート・マスク・上書き禁止・
# 腕ツリー検査）と同型。macOS の bash 3.2 で動くよう変数は ${VAR} と書く。
#
# 実機ロック（~/work/.fandhe-hw-lock）の取得と解放は呼び出し側（README の手順）が行い、保持している
# 間に本スクリプトを走らせる。ホスト名・ユーザー名は収録物へ書かない（マスク規則）。
#
# 使い方（BASE は rsync で同期した main ツリー〈.rev-stamp 付き。診断パッチ未適用〉、WORK は同期ツリーの外）:
#   BASE=<絶対パス> WORK=<絶対パス> bash run_breakdown.sh <macmini|gb10> --dry-run
#   BASE=<絶対パス> WORK=<絶対パス> bash run_breakdown.sh <macmini|gb10>
# timed 腕は BASE から target を除いて複製し diag_timing.patch を当てる。timed 腕では cargo test を実行しない。
set -u
export LC_ALL=C

MACHINE="${1:-}"
DRY=0
[ "${2:-}" = "--dry-run" ] && DRY=1
case "${MACHINE}" in
    macmini) GATE_MAX_LOAD1="5.0" ;;
    gb10) GATE_MAX_LOAD1="10.0" ;;
    *) echo "usage: run_breakdown.sh <macmini|gb10> [--dry-run]" >&2; exit 2 ;;
esac
GATE_WAIT_S=30
GATE_TRIES=60
ROUNDS="1 2 3 4 5"
PKG="fandhe-ai-backend-cpu"
EX="cpu_pack_kernel_breakdown"
LOGDIR="docs/perf/logs/cpu-pack-kernel-breakdown-3614"
# 4 プロセス: <arm>:<mode>。round ごとに先頭をずらす（ローテーション）。
PROCS="base:native base:t1 timed:native timed:t1"

if [ "${DRY}" = "1" ]; then
    echo "# dry-run: machine=${MACHINE} 負荷ゲート load1 < ${GATE_MAX_LOAD1}（${GATE_WAIT_S}s 間隔・最大 ${GATE_TRIES} 回）"
    echo "1. env_info・.rev-stamp の収録"
    echo "2. P-TREE: BASE から target を除いて timed ツリーへ複製し diag_timing.patch を適用。sha256 を patch_sha256.txt と突合"
    echo "3. ビルド（逐次・計測前に全部）: base と timed の example（release）"
    for r in ${ROUNDS}; do echo "4. round${r}: ${PROCS} を $((r - 1)) 個ぶん回転した順で実行（t1 は RAYON_NUM_THREADS=1）"; done
    echo "5. DONE センチネル"
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
mkdir -p "${OUT}" || exit 1

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
gate() { # gate <round> <arm> <mode>
    waited=0; tries=0
    while :; do
        if l1="$(load1_now)"; then
            ok="$(awk -v l="${l1}" -v m="${GATE_MAX_LOAD1}" 'BEGIN{print (l < m) ? 1 : 0}')"
        else
            ok=0
        fi
        if [ "${ok}" = "1" ]; then
            echo "START round=$1 arm=$2 mode=$3 gate=pass load1=${l1} waited_s=${waited} at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${OUT}/load_gate.log"
            return 0
        fi
        tries=$((tries + 1))
        if [ "${tries}" -ge "${GATE_TRIES}" ]; then
            echo "START round=$1 arm=$2 mode=$3 gate=unavailable load1=${l1} waited_s=${waited} at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${OUT}/load_gate.log"
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
cp "${BASE}/.rev-stamp" "${OUT}/.rev-stamp" || { echo "error: ${BASE}/.rev-stamp がない" >&2; exit 1; }

# 2. P-TREE: timed ツリーを BASE から複製してパッチを当てる。
TIMED="${WORK}/timed-tree-${MACHINE}"
# 前回実行の残骸が混ざると診断パッチ以外の差を含む腕で計測しうるため、既存なら停止し毎回空の複製先を使う。
if [ -e "${TIMED}" ]; then echo "error: ${TIMED} が既に存在する（前回の残骸を混ぜない。削除してから再実行すること）" >&2; exit 1; fi
mkdir -p "${TIMED}" || exit 1
( cd "${BASE}" && tar --exclude=./target --exclude=./.git -cf - . ) | ( cd "${TIMED}" && tar -xf - ) || exit 1
PATCH="${BASE}/${LOGDIR}/diag_timing.patch"
if command -v sha256sum >/dev/null 2>&1; then
    got="$(sha256sum "${PATCH}" | awk '{print $1}')"
else
    got="$(shasum -a 256 "${PATCH}" | awk '{print $1}')"
fi
want="$(awk '{print $1}' "${BASE}/${LOGDIR}/patch_sha256.txt")"
if [ "${got}" != "${want}" ]; then
    echo "error: diag_timing.patch の sha256 が patch_sha256.txt と食い違う（got=${got} want=${want}）" >&2
    exit 1
fi
cp "${BASE}/${LOGDIR}/patch_sha256.txt" "${OUT}/patch_sha256.txt"
( cd "${TIMED}" && git apply --whitespace=nowarn "${PATCH}" ) || { echo "error: パッチ適用に失敗" >&2; exit 1; }
{
    echo "patch_sha256=${got}"
    echo "base_tree_files=$( cd "${BASE}" && find crates -name '*.rs' -not -path '*/target/*' | wc -l | tr -d ' ')"
    echo "changed_files_vs_base:"
    ( cd "${BASE}" && find crates -type f -not -path '*/target/*' | sort ) | while read -r f; do
        if [ ! -e "${TIMED}/${f}" ] || ! cmp -s "${BASE}/${f}" "${TIMED}/${f}"; then echo "  ${f}"; fi
    done
    ( cd "${TIMED}" && find crates -type f -not -path '*/target/*' | sort ) | while read -r f; do
        [ -e "${BASE}/${f}" ] || echo "  ${f} (timed のみ)"
    done
} 2>&1 | mask > "${OUT}/tree_verify.txt"

# 3. ビルド（逐次。計測と並走させない）
# cargo の Compiling 行はソースの絶対パスを含むため mask を通す。pipefail で cargo の終了コードを保つ。
( set -o pipefail; cd "${BASE}" && cargo build --release -p "${PKG}" --example "${EX}" 2>&1 | mask > "${OUT}/build-base.log" ) || { echo "error: base のビルド失敗" >&2; exit 1; }
( set -o pipefail; cd "${TIMED}" && cargo build --release -p "${PKG}" --example "${EX}" 2>&1 | mask > "${OUT}/build-timed.log" ) || { echo "error: timed のビルド失敗" >&2; exit 1; }
BIN_BASE="${BASE}/target/release/examples/${EX}"
BIN_TIMED="${TIMED}/target/release/examples/${EX}"

# 4. round 1..5。1 round の 4 プロセスは起動順を round ごとに回転する。
set -- ${PROCS}
for r in ${ROUNDS}; do
    shift_n=$(( (r - 1) % 4 ))
    order=""
    i=0
    for p in "$@"; do
        idx=$(( (i + 4 - shift_n) % 4 ))
        order="${order}${idx}:${p} "
        i=$((i + 1))
    done
    # idx 昇順に並べ替える（ローテーション後の起動順）
    ordered="$(echo "${order}" | tr ' ' '\n' | grep -v '^$' | sort -t: -k1,1n | cut -d: -f2-)"
    for p in ${ordered}; do
        arm="${p%%:*}"
        mode="${p##*:}"
        log="${OUT}/round${r}-${arm}-${mode}.log"
        if [ -e "${log}" ]; then echo "既存ログがあるため停止: ${log}" >&2; exit 1; fi
        if ! gate "${r}" "${arm}" "${mode}"; then
            echo "UNAVAILABLE (load gate)" > "${log}.unavailable"
            continue
        fi
        if [ "${arm}" = "base" ]; then bin="${BIN_BASE}"; else bin="${BIN_TIMED}"; fi
        # pipefail でバイナリの異常終了を検出する。失敗時は .tmp を破棄しログを確定せず、DONE も作らず停止する。
        if [ "${mode}" = "t1" ]; then
            ( set -o pipefail; RAYON_NUM_THREADS=1 "${bin}" "${arm}" 2>&1 | mask > "${log}.tmp" )
        else
            ( set -o pipefail; "${bin}" "${arm}" 2>&1 | mask > "${log}.tmp" )
        fi
        rc=$?
        if [ "${rc}" -ne 0 ]; then
            rm -f "${log}.tmp"
            echo "error: 計測失敗 round=${r} arm=${arm} mode=${mode} rc=${rc}（ログ未確定・DONE なし）" >&2
            exit 1
        fi
        mv "${log}.tmp" "${log}"
    done
done

date -u +%Y-%m-%dT%H:%M:%SZ > "${OUT}/DONE"
echo "完了: ${OUT}"
