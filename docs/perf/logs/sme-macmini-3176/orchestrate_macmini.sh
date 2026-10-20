#!/bin/bash
# Mac mini（Apple M4）で SME カーネルの ON／OFF を測る一連の計測オーケストレーション（イシュー #3176）。
# 判定規則は同ディレクトリ RULE.txt（計測前に単独コミット済み）。前例は
# docs/perf/logs/sme-scale-macmini-3175/orchestrate_macmini.sh（負荷ゲート・.raw.log/.log 分離・
# 既存ログ上書き禁止・失敗即停止）と docs/perf/logs/sme-threshold-remeasurement-2118/orchestrate_m4max.sh
# （after ツリー生成・P-TREE・P-R0）。macOS の bash 3.2 で動くよう変数は ${VAR} と書く。
#
# 起動順: P-TREE（after 生成と指紋）→ P-R0（両腕の sme_report）→ R4 格子 5 run → 長方形 5 run → A/B。
# after では `--lib` を実行しない（ゲートが true の単体テスト〈ドリフトガード〉が構造上落ちるため）。
# R4・長方形は before ツリーで走らせる（診断経路は本番定数を参照しないので before で足りる）。
#
# 使い方（before は bundle から clone したツリー〈差分検証に git が要る。README 参照〉のルートで）:
#   BEFORE=<before ツリー絶対パス> TORCH_PY=<venv-torch/bin/python 絶対パス> WORK=<作業ディレクトリ絶対パス> \
#     bash docs/perf/logs/sme-macmini-3176/orchestrate_macmini.sh --dry-run   # 計画のみ表示
#     bash docs/perf/logs/sme-macmini-3176/orchestrate_macmini.sh             # 実行（Mac mini 上）
# WORK は同期ツリーの外に置く（target もそこへ置く）。ホスト名・ユーザー名は収録物へ書かない（マスク規則）。
set -u
export LC_ALL=C

REPO_LOG="docs/perf/logs/sme-macmini-3176"
PATCH_SHA256_EXPECTED="${PATCH_SHA256_EXPECTED:-}"
RUNS="1 2 3 4 5"
GATE_MAX_LOAD1="3.0"
GATE_WAIT_S=30
GATE_TRIES=60
TEST_PREFIX="gemm_blis::tests"
DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

if [ "${DRY}" = "1" ]; then
    echo "# dry-run: 実行計画（負荷ゲート load1 < ${GATE_MAX_LOAD1}・${GATE_WAIT_S}s 間隔・最大 ${GATE_TRIES} 回）"
    echo "1. P-TREE: before から target を除いて after へ複製し on-arm.patch を適用。差分ファイルが mod.rs の 1 件だけで、"
    echo "   after の定数行が true、patch の sha256 が RULE 記載と一致することを検査（tree_verify.txt・patch_sha256.txt・gate_constant.txt）"
    echo "2. P-R0: 両腕で sme_report() が kernel_enabled: true であること（1587/gb10/sme-probe-* を再利用。sme_report.txt）"
    for r in ${RUNS}; do echo "3. R4 格子 run${r}: sme_vs_neon_ab_r4_grid（before・32 点）"; done
    for r in ${RUNS}; do echo "4. 長方形 run${r}: sme_vs_neon_ab_mlp_rect（before・8 点）"; done
    echo "5. A/B: scripts/bench/framework-compare/run_ab_sme_macmini_cpu.sh 3176-r1（5 round・PyTorch 同席）→ macmini/r1r2/ へマスク収録"
    exit 0
fi

for v in BEFORE TORCH_PY WORK; do
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

OUT="${REPO_LOG}/macmini"
AFTER="${WORK}/after"
mkdir -p "${OUT}" "${WORK}" || exit 1
for f in "${OUT}"/*.log "${OUT}"/load_gate.log; do
    if [ -e "${f}" ]; then
        echo "既存ログがあるため停止（上書き禁止）: ${f}" >&2
        exit 1
    fi
done
if [ -e "${AFTER}" ]; then
    echo "既存の after ツリーがあるため停止: ${AFTER}" >&2
    exit 1
fi

mask() { sed -e "s#${HOME}#<home>#g" -e "s#${WORK}#<work>#g" -e "s#${BEFORE}#<before>#g"; }
load1_now() { sysctl -n vm.loadavg | awk '{print $2}'; }

gate() { # gate <series> <run>
    waited=0; tries=0
    while :; do
        l1="$(load1_now)"
        ok="$(awk -v l="${l1}" -v m="${GATE_MAX_LOAD1}" 'BEGIN{print (l < m) ? 1 : 0}')"
        if [ "${ok}" = "1" ]; then
            echo "START series=$1 run=$2 gate=pass load1=${l1} waited_s=${waited} at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${OUT}/load_gate.log"
            return 0
        fi
        tries=$((tries + 1))
        if [ "${tries}" -ge "${GATE_TRIES}" ]; then
            echo "START series=$1 run=$2 gate=unavailable load1=${l1} waited_s=${waited} at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${OUT}/load_gate.log"
            return 1
        fi
        sleep "${GATE_WAIT_S}"
        waited=$((waited + GATE_WAIT_S))
    done
}

# P-TREE: after ツリーの生成と指紋検査（失敗は undetermined として記録し停止）
# run_ab_sme_macmini_cpu.sh が「同一コミット・定数 1 行のみ差分」を検査するため、after は before の clone にする
git clone -q "${BEFORE}" "${AFTER}" || { echo "after 複製に失敗" >&2; exit 1; }
(cd "${AFTER}" && patch -p1 --forward < "${BEFORE}/${REPO_LOG}/on-arm.patch") > "${OUT}/patch_apply.log" 2>&1 \
    || { echo "patch 適用に失敗 -> P-TREE 不成立" >&2; exit 1; }
shasum -a 256 "${BEFORE}/${REPO_LOG}/on-arm.patch" | awk '{print $1}' > "${OUT}/patch_sha256.txt"
if [ -n "${PATCH_SHA256_EXPECTED}" ] && [ "$(cat "${OUT}/patch_sha256.txt")" != "${PATCH_SHA256_EXPECTED}" ]; then
    echo "patch の sha256 が RULE 記載と不一致 -> P-TREE 不成立" >&2
    exit 1
fi
(cd "${BEFORE}" && find . -type f -not -path './.git/*' -not -path './target*' -not -name .rev-stamp | sort) > "${WORK}/files_before.txt"
(cd "${AFTER}" && find . -type f -not -path './.git/*' -not -path './target*' -not -name .rev-stamp | sort) > "${WORK}/files_after.txt"
cmp -s "${WORK}/files_before.txt" "${WORK}/files_after.txt" || { echo "ファイル一覧が不一致 -> P-TREE 不成立" >&2; exit 1; }
: > "${OUT}/tree_verify.txt"
while read -r f; do
    if ! cmp -s "${BEFORE}/${f}" "${AFTER}/${f}"; then echo "${f}" >> "${OUT}/tree_verify.txt"; fi
done < "${WORK}/files_before.txt"
if [ "$(cat "${OUT}/tree_verify.txt")" != "./crates/backend-cpu/src/gemm_blis/mod.rs" ]; then
    echo "差分ファイルが mod.rs の 1 件だけではない -> P-TREE 不成立" >&2
    cat "${OUT}/tree_verify.txt" >&2
    exit 1
fi
grep -n '^const SME_PRODUCTION_ENABLED: bool' "${AFTER}/crates/backend-cpu/src/gemm_blis/mod.rs" > "${OUT}/gate_constant.txt"
grep -q 'SME_PRODUCTION_ENABLED: bool = true;' "${OUT}/gate_constant.txt" || { echo "after の定数が true ではない" >&2; exit 1; }
echo "P-TREE ok"

# P-R0: 両腕で SME が有効（kernel_enabled: true）であること
for t in before after; do
    P="${WORK}/probe-${t}"
    mkdir -p "${P}/src"
    if [ "${t}" = "before" ]; then TREE="${BEFORE}"; else TREE="${AFTER}"; fi
    sed "s#TREE_PLACEHOLDER#${TREE}#" "${BEFORE}/docs/perf/logs/cpu-gemm-sme-fmopa-1587/gb10/sme-probe-Cargo.toml" > "${P}/Cargo.toml" || exit 1
    cp "${BEFORE}/docs/perf/logs/cpu-gemm-sme-fmopa-1587/gb10/sme-probe-main.rs" "${P}/src/main.rs" || exit 1
    (cd "${P}" && CARGO_TARGET_DIR="${WORK}/target-probe" cargo run --release -q) > "${WORK}/probe_${t}.log" 2>&1 \
        || { echo "probe ${t} 失敗" >&2; tail -20 "${WORK}/probe_${t}.log" >&2; exit 1; }
    mask < "${WORK}/probe_${t}.log" > "${OUT}/sme_probe_${t}.log"
    echo "${t}: $(grep sme_report= "${OUT}/sme_probe_${t}.log")" >> "${OUT}/sme_report.txt"
done
if ! grep -q 'kernel_enabled: true' "${OUT}/sme_probe_before.log" || ! grep -q 'kernel_enabled: true' "${OUT}/sme_probe_after.log"; then
    echo "P-R0 不成立（kernel_enabled が true でない）-> 計測中止（undetermined）" >&2
    exit 1
fi
echo "P-R0 ok"

# run_one <series> <run> <logbase> <testname> <grep-pattern>
run_one() {
    series="$1"; run="$2"; base="$3"; test="$4"; pat="$5"
    gate "${series}" "${run}"
    (cd "${BEFORE}" && CARGO_TARGET_DIR="${WORK}/target-before" cargo test -p fandhe-ai-backend-cpu --release --lib -- --ignored \
        "${TEST_PREFIX}::${test}" --exact --nocapture --test-threads=1) > "${OUT}/${base}.raw.log" 2>&1
    rc=$?
    # libtest は先頭の出力を "test <name> ... " と同じ行に出すため接頭辞を除いてから行頭アンカーで抽出する（#3175 の教訓）
    sed 's/^test [^ ]* \.\.\. //' "${OUT}/${base}.raw.log" | grep -E "${pat}" > "${OUT}/${base}.log"
    grc=$?
    echo "END series=${series} run=${run} exit=${rc} grep_exit=${grc} end_load1=$(load1_now)" >> "${OUT}/load_gate.log"
    if [ "${rc}" -ne 0 ] || [ "${grc}" -ne 0 ]; then
        echo "FAILED: ${base} (exit=${rc} grep_exit=${grc})" >&2
        exit 1
    fi
}
PAT='^(num_threads=|variant=)'
for r in ${RUNS}; do run_one r4 "${r}" "r4_run${r}" sme_vs_neon_ab_r4_grid "${PAT}"; done
for r in ${RUNS}; do run_one rect "${r}" "rect_run${r}" sme_vs_neon_ab_mlp_rect "${PAT}"; done

# A/B（5 round・PyTorch 同席）。results/raw の 3176 系ファイルを macmini/r1r2/ へマスク収録する
(cd "${BEFORE}/scripts/bench/framework-compare" && \
    AB_BEFORE_FACADE_PATH="${BEFORE}/crates/facade" AB_AFTER_FACADE_PATH="${AFTER}/crates/facade" \
    AB_TORCH_PYTHON="${TORCH_PY}" bash run_ab_sme_macmini_cpu.sh 3176-r1) > "${OUT}/run_ab.log" 2>&1
echo "run_ab exit=$?" >> "${OUT}/load_gate.log"
mkdir -p "${OUT}/r1r2"
for f in "${BEFORE}"/scripts/bench/framework-compare/results/raw/*3176*; do
    case "${f}" in
        */bench-fandhe-*) ;;
        *) mask < "${f}" > "${OUT}/r1r2/$(basename "${f}")" ;;
    esac
done
for f in "${BEFORE}"/scripts/bench/framework-compare/compare-*3176*; do
    [ -e "${f}" ] && mask < "${f}" > "${OUT}/r1r2/$(basename "${f}")"
done
echo "完了"
