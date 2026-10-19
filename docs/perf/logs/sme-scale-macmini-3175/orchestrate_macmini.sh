#!/bin/sh
# Mac mini の SME 多スレッド頭打ち・micro-call 固定費の計測オーケストレーション（イシュー #3175）。
# 判定規則は同ディレクトリ RULE.txt（計測前に単独コミット済み）。前例は
# docs/perf/logs/cpu-gemm-sme-fmopa-1587/orchestrate_m4max.sh（負荷ゲート・.raw.log/.log 分離・
# 既存ログ上書き禁止・失敗即停止）と同一方針。macOS の bash 3.2/sh で動くよう変数は ${VAR} と書く。
#
# 使い方（リポジトリルートで）:
#   sh docs/perf/logs/sme-scale-macmini-3175/orchestrate_macmini.sh --dry-run   # 計画のみ表示
#   sh docs/perf/logs/sme-scale-macmini-3175/orchestrate_macmini.sh             # 実行（Mac mini 上）
# 事前に `cargo test -p fandhe-ai-backend-cpu --release --lib --no-run` を済ませる
# （計測中にビルドを並走させない）。CARGO_TARGET_DIR は呼び出し側の環境変数に従う。
#
# 系列: A=スレッド掃引（T=1,2,4,10 × 5 run）/ B=micro-call 固定費（5 run）/
#       C=E コア側（taskpolicy -b・T=1 × 5 run。taskpolicy が無ければ省略して記録）。
set -u

OUT="docs/perf/logs/sme-scale-macmini-3175"
RUNS="1 2 3 4 5"
THREADS="1 2 4 10"
GATE_MAX_LOAD1="5.0"
GATE_WAIT_S=30
GATE_TRIES=60
TEST_PREFIX="gemm_blis::tests"
DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

# run 番号に応じてスレッド数の並びを回転させる（順序偏りの均し）
rotate_threads() {
    r="$1"
    set -- ${THREADS}
    n=$#
    shift_by=$(( (r - 1) % n ))
    j=0
    head_part=""
    tail_part=""
    for t in "$@"; do
        if [ "${j}" -lt "${shift_by}" ]; then
            tail_part="${tail_part} ${t}"
        else
            head_part="${head_part} ${t}"
        fi
        j=$((j + 1))
    done
    echo "${head_part}${tail_part}"
}

load1_now() {
    sysctl -n vm.loadavg | awk '{print $2}'
}

# gate <series> <run> <T> : 通過なら 0、取れなければ 1 を返し、START 行を load_gate.log へ書く
gate() {
    series="$1"; run="$2"; t="$3"
    waited=0
    tries=0
    while :; do
        l1="$(load1_now)"
        ok="$(awk -v l="${l1}" -v m="${GATE_MAX_LOAD1}" 'BEGIN{print (l < m) ? 1 : 0}')"
        if [ "${ok}" = "1" ]; then
            echo "START series=${series} run=${run} T=${t} gate=pass load1=${l1} waited_s=${waited} at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${OUT}/load_gate.log"
            return 0
        fi
        tries=$((tries + 1))
        if [ "${tries}" -ge "${GATE_TRIES}" ]; then
            echo "START series=${series} run=${run} T=${t} gate=unavailable load1=${l1} waited_s=${waited} at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${OUT}/load_gate.log"
            return 1
        fi
        sleep "${GATE_WAIT_S}"
        waited=$((waited + GATE_WAIT_S))
    done
}

# run_one <series> <run> <T> <logbase> <testname> <grep-pattern> [wrapper...]
run_one() {
    series="$1"; run="$2"; t="$3"; base="$4"; test="$5"; pat="$6"; shift 6
    gate "${series}" "${run}" "${t}"
    RAYON_NUM_THREADS="${t}" "$@" cargo test -p fandhe-ai-backend-cpu --release --lib -- --ignored \
        "${TEST_PREFIX}::${test}" --exact --nocapture --test-threads=1 > "${OUT}/${base}.raw.log" 2>&1
    rc=$?
    grep -E "${pat}" "${OUT}/${base}.raw.log" > "${OUT}/${base}.log"
    grc=$?
    echo "END series=${series} run=${run} T=${t} exit=${rc} grep_exit=${grc} end_load1=$(load1_now)" >> "${OUT}/load_gate.log"
    if [ "${rc}" -ne 0 ] || [ "${grc}" -ne 0 ]; then
        echo "FAILED: ${base} (exit=${rc} grep_exit=${grc})" >&2
        exit 1
    fi
}

SWEEP_PAT='^(num_threads=|rayon_current_num_threads=|variant=)'
PROBE_PAT='^probe=microcall '

plan() {
    for r in ${RUNS}; do
        for t in $(rotate_threads "${r}"); do echo "A thread_t${t}_run${r} T=${t}"; done
        echo "B microcall_run${r} T=1"
    done
    if command -v taskpolicy >/dev/null 2>&1; then
        for r in ${RUNS}; do echo "C ecore_t1_run${r} T=1 (taskpolicy -b)"; done
    else
        echo "C 省略（taskpolicy 無し）"
    fi
}

if [ "${DRY}" = "1" ]; then
    echo "# dry-run: 実行計画（負荷ゲート load1 < ${GATE_MAX_LOAD1}・${GATE_WAIT_S}s 間隔・最大 ${GATE_TRIES} 回）"
    plan
    exit 0
fi

# 既存ログがあれば停止（差し替え・上書き禁止）
if ls "${OUT}"/thread_t*_run*.log "${OUT}"/microcall_run*.log "${OUT}"/ecore_t1_run*.log "${OUT}"/load_gate.log >/dev/null 2>&1; then
    echo "既存ログがあるため停止（上書き禁止）" >&2
    exit 1
fi

for r in ${RUNS}; do
    for t in $(rotate_threads "${r}"); do
        run_one thread "${r}" "${t}" "thread_t${t}_run${r}" sme_vs_neon_ab_thread_sweep "${SWEEP_PAT}" env
    done
    run_one microcall "${r}" 1 "microcall_run${r}" sme_microcall_fixed_cost_probe "${PROBE_PAT}" env
done

if command -v taskpolicy >/dev/null 2>&1; then
    for r in ${RUNS}; do
        run_one ecore "${r}" 1 "ecore_t1_run${r}" sme_vs_neon_ab_thread_sweep "${SWEEP_PAT}" taskpolicy -b env
    done
else
    echo "ecore_unavailable (taskpolicy 無し)" >> "${OUT}/load_gate.log"
fi
echo "完了"
