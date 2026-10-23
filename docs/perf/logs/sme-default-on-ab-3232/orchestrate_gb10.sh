#!/bin/bash
# GB10（DGX Spark・SME 非搭載）で SME 既定 ON（main）と OFF の非後退を確かめるオーケストレーション（イシュー #3232。再開条件 2）。
# 判定規則は同ディレクトリ RULE.txt。前例は docs/perf/logs/cpu-gemm-sme-fmopa-1587/gb10/orchestrate_gb10.sh（外側ゲート・R0・RT）と
# 同 run_bitdump.sh（bit 同一確認。出力先が自ディレクトリ固定のため本スクリプトに同じ手順を組み込む）。
# GB10 は kernel_enabled が false（P-R0）のため、全セルが非到達のガードになる。ON/OFF が同じ NEON 経路を通ることの確認が目的。
#
# 腕: ON = 本スクリプトを置いたツリー（bundle から clone した main）、OFF = ON を clone して off-arm.patch を当てたツリー。
# 起動順: ロック → 外側ゲート → P-TREE → P-R0 → RT（ON のみ。OFF では --lib を実行しない）→ A/B（AB_HOST=gb10）→ bitdump → 収録。
#
# 使い方（ON ツリーのルートで。PATH は ${HOME}/.cargo/bin を先頭へ明示する）:
#   ON=<ON ツリー絶対パス> WORK=<作業ディレクトリ絶対パス> [TORCH_PY=<python 絶対パス>] \
#     bash docs/perf/logs/sme-default-on-ab-3232/orchestrate_gb10.sh --dry-run   # 計画のみ表示（副作用なし）
#     setsid nohup bash docs/perf/logs/sme-default-on-ab-3232/orchestrate_gb10.sh > <ログ> 2>&1 &   # 実行（GB10 上）
set -u
export LC_ALL=C
export PATH="${HOME}/.cargo/bin:${PATH}"

REPO_LOG="docs/perf/logs/sme-default-on-ab-3232"
LABEL="3232-r1"
GATE_FILE="crates/backend-cpu/src/gemm_blis/mod.rs"
LOCK_DIR="${HOME}/work/.fandhe-hw-lock"
EXPECTED_BITDUMP_LINES=6726847
DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

if [ "${DRY}" = "1" ]; then
    echo "# dry-run: 実行計画（副作用なし）"
    echo "0. ロック: mkdir ${LOCK_DIR}（60 秒おきに再試行）→ 外側ゲート load1 < 1.0 かつ GPU 0%（最大 20 回・30 秒間隔。load_gate_outer.log）"
    echo "1. P-TREE: ON を clone して off-arm.patch を適用（sha256 照合・git apply --check が先）。差分が ${GATE_FILE} の 1 件だけ"
    echo "2. P-R0: 両腕の sme_report() が kernel_enabled: false であること（外れたら中止）"
    echo "3. RT: ON ツリーで cargo test -p fandhe-ai-backend-cpu --release（OFF では --lib を実行しない）"
    echo "4. A/B: run_ab_sme_default_on_cpu.sh ${LABEL}（AB_HOST=gb10・内側ゲート 8.0・5 round。TORCH_PY があれば PyTorch 同席）"
    echo "5. bitdump: cpu_sme_gate_bit_dump の ^out[ 行を ON/OFF で比較（期待 ${EXPECTED_BITDUMP_LINES} 行。summary と cmp 結果のみ収録・dump 本体は収録しない）"
    echo "6. 収録: マスクして ${REPO_LOG}/gb10 へ複製。target-ab-* を削除し、ロックを解放"
    exit 0
fi

for v in ON WORK; do
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
TORCH_PY="${TORCH_PY:-}"

OUT="${WORK}/gb10"
FINAL="${ON}/${REPO_LOG}/gb10"
OFF="${WORK}/off"
mkdir -p "${WORK}" "${OUT}/bitdump" || exit 1
if [ -e "${FINAL}" ] || [ -e "${OFF}" ] || [ -e "${OUT}/load_gate.log" ]; then
    echo "既存の収録先・OFF ツリー・ログがあるため停止（上書き禁止）" >&2
    exit 1
fi

mask() { sed -e "s#${HOME}#<home>#g" -e "s#${WORK}#<work>#g" -e "s#${ON}#<on>#g"; }

until mkdir "${LOCK_DIR}" 2>/dev/null; do sleep 60; done
echo "#3232 $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${LOCK_DIR}/owner"
echo "acquired at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${OUT}/hwlock.log"
release_lock() {
    echo "released at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${OUT}/hwlock.log"
    rm -rf "${LOCK_DIR}"
}
trap release_lock EXIT

# 外側専有ゲート（記録）: load1 < 1.0 かつ GPU 使用率 0%
outer_gate() {
    i=1
    while [ "${i}" -le 20 ]; do
        l1="$(cut -d' ' -f1 /proc/loadavg)"
        gu="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
        if awk -v l="${l1}" -v g="${gu:-x}" 'BEGIN{exit !(l < 1.0 && g == 0)}'; then
            echo "start attempt=${i} load1=${l1} gpu_util=${gu} pass" >> "${OUT}/load_gate_outer.log"
            return 0
        fi
        echo "start attempt=${i} load1=${l1} gpu_util=${gu:-NA} wait" >> "${OUT}/load_gate_outer.log"
        sleep 30
        i=$((i + 1))
    done
    echo "start attempt=20 fail" >> "${OUT}/load_gate_outer.log"
    return 1
}
outer_gate || { echo "外側ゲート不成立 -> 計測中止" >&2; exit 1; }

# P-TREE
git clone -q "${ON}" "${OFF}" || { echo "OFF 複製に失敗" >&2; exit 1; }
PATCH="${ON}/docs/perf/logs/sme-default-on-3231/off-arm.patch"
EXPECTED="$(sed -n 's/^off-arm\.patch sha256: \([0-9a-f]\{64\}\)$/\1/p' "${ON}/${REPO_LOG}/RULE.txt" | head -n1)"
case "${EXPECTED}" in
    ????????????????????????????????????????????????????????????????) ;;
    *) echo "RULE.txt から patch の sha256 を取得できない -> P-TREE 不成立" >&2; exit 1 ;;
esac
sha256sum "${PATCH}" | awk '{print $1}' > "${OUT}/patch_sha256.txt" || exit 1
[ "$(cat "${OUT}/patch_sha256.txt")" = "${EXPECTED}" ] || { echo "patch の sha256 が RULE 記載と不一致 -> P-TREE 不成立" >&2; exit 1; }
(cd "${OFF}" && git apply --check "${PATCH}" && git apply "${PATCH}") > "${OUT}/patch_apply.log" 2>&1 \
    || { echo "patch 適用に失敗 -> P-TREE 不成立" >&2; exit 1; }
git -C "${OFF}" diff --name-only HEAD > "${OUT}/tree_verify.txt"
[ "$(cat "${OUT}/tree_verify.txt")" = "${GATE_FILE}" ] || { echo "差分ファイルが ${GATE_FILE} の 1 件だけではない -> P-TREE 不成立" >&2; exit 1; }
grep -n '^const SME_PRODUCTION_ENABLED: bool' "${OFF}/${GATE_FILE}" > "${OUT}/gate_constant.txt"
grep -n '^const SME_PRODUCTION_ENABLED: bool' "${ON}/${GATE_FILE}" >> "${OUT}/gate_constant.txt"
[ "$(sed -n 1p "${OUT}/gate_constant.txt" | sed 's/^[0-9]*://')" = "const SME_PRODUCTION_ENABLED: bool = false;" ] || { echo "OFF の定数が false ではない" >&2; exit 1; }
[ "$(sed -n 2p "${OUT}/gate_constant.txt" | sed 's/^[0-9]*://')" = "const SME_PRODUCTION_ENABLED: bool = true;" ] || { echo "ON の定数が true ではない" >&2; exit 1; }
git -C "${ON}" rev-parse HEAD > "${OUT}/rev-stamp.txt"
echo "$(git -C "${ON}" rev-parse HEAD)+off-arm.patch" >> "${OUT}/rev-stamp.txt"
git -C "${ON}" ls-files | sort | sha256sum | awk '{print $1}' > "${OUT}/tree_fingerprint.txt"
echo "P-TREE ok"

# P-R0（GB10 は SME 非搭載のため両腕 kernel_enabled: false が期待値）
: > "${OUT}/sme_report.txt"
for t in off on; do
    P="${WORK}/probe-${t}"
    mkdir -p "${P}/src"
    if [ "${t}" = "off" ]; then TREE="${OFF}"; else TREE="${ON}"; fi
    sed "s#TREE_PLACEHOLDER#${TREE}#" "${ON}/docs/perf/logs/cpu-gemm-sme-fmopa-1587/gb10/sme-probe-Cargo.toml" > "${P}/Cargo.toml" || exit 1
    cp "${ON}/docs/perf/logs/cpu-gemm-sme-fmopa-1587/gb10/sme-probe-main.rs" "${P}/src/main.rs" || exit 1
    (cd "${P}" && CARGO_TARGET_DIR="${WORK}/target-probe" cargo run --release -q) > "${WORK}/probe_${t}.log" 2>&1 \
        || { echo "probe ${t} 失敗" >&2; tail -20 "${WORK}/probe_${t}.log" >&2; exit 1; }
    mask < "${WORK}/probe_${t}.log" > "${OUT}/sme_probe_${t}.log"
    echo "${t}: $(grep sme_report= "${OUT}/sme_probe_${t}.log")" >> "${OUT}/sme_report.txt"
done
if ! grep -q 'kernel_enabled: false' "${OUT}/sme_probe_off.log" || ! grep -q 'kernel_enabled: false' "${OUT}/sme_probe_on.log"; then
    echo "P-R0 不成立（kernel_enabled が false でない）-> 計測中止（undetermined）" >&2
    exit 1
fi
echo "P-R0 ok"

# RT: ON ツリーのみ（OFF では --lib を実行しない）。全文は収録せず要約だけ残す
(cd "${ON}" && CARGO_TARGET_DIR="${WORK}/target-rt-on" cargo test -p fandhe-ai-backend-cpu --release) > "${WORK}/rt_on.raw.log" 2>&1
rt_rc=$?
echo "rt_on exit=${rt_rc}" >> "${OUT}/load_gate.log"
grep -E '^test result|FAILED|panicked' "${WORK}/rt_on.raw.log" | mask > "${OUT}/rt_on.log"
rm -rf "${WORK}/target-rt-on"

# A/B
(cd "${ON}/scripts/bench/framework-compare" && \
    AB_HOST=gb10 AB_BEFORE_FACADE_PATH="${OFF}/crates/facade" AB_AFTER_FACADE_PATH="${ON}/crates/facade" \
    AB_TORCH_PYTHON="${TORCH_PY}" bash run_ab_sme_default_on_cpu.sh "${LABEL}") > "${OUT}/run_ab.log" 2>&1
run_ab_rc=$?
echo "run_ab exit=${run_ab_rc}" >> "${OUT}/load_gate.log"

# bitdump（ON/OFF の ^out[ 行の行数・sha256・cmp。dump 本体は収録しない。#3231 の M4 Max 値は記録のみで判定しない）
bd_rc=0
for t in off on; do
    if [ "${t}" = "off" ]; then TREE="${OFF}"; else TREE="${ON}"; fi
    (cd "${TREE}" && CARGO_TARGET_DIR="${WORK}/target-bd" cargo test -p fandhe-ai --release --test cpu_sme_gate_bit_dump \
        -- --ignored --nocapture --exact dump_cpu_sme_gate_bits) 2>&1 | grep -E '^out\[' > "${WORK}/bits_${t}.txt"
    [ "${PIPESTATUS[0]}" = "0" ] || bd_rc=1
done
n_off="$(wc -l < "${WORK}/bits_off.txt" | tr -d ' ')"
n_on="$(wc -l < "${WORK}/bits_on.txt" | tr -d ' ')"
{
    echo "off_lines=${n_off} on_lines=${n_on} expected=${EXPECTED_BITDUMP_LINES}"
    echo "off_sha256=$(sha256sum "${WORK}/bits_off.txt" | awk '{print $1}')"
    echo "on_sha256=$(sha256sum "${WORK}/bits_on.txt" | awk '{print $1}')"
} > "${OUT}/bitdump/summary.txt"
cmp "${WORK}/bits_off.txt" "${WORK}/bits_on.txt" > "${OUT}/bitdump/cmp.txt" 2>&1
cmp_rc=$?
if [ "${n_off}" != "${EXPECTED_BITDUMP_LINES}" ] || [ "${n_on}" != "${EXPECTED_BITDUMP_LINES}" ]; then bd_rc=1; fi
echo "bitdump exit=${bd_rc} cmp_exit=${cmp_rc}" >> "${OUT}/load_gate.log"
rm -f "${WORK}/bits_off.txt" "${WORK}/bits_on.txt"
rm -rf "${WORK}/target-bd" "${WORK}/target-probe"

# 収録（マスク）
collect_rc=0
for f in "${OUT}/run_ab.log" "${OUT}/patch_apply.log"; do
    [ -e "${f}" ] || continue
    if mask < "${f}" > "${f}.masked"; then mv "${f}.masked" "${f}"; else rm -f "${f}.masked"; collect_rc=1; fi
done
mkdir -p "${OUT}/ab" || collect_rc=1
for f in "${ON}"/scripts/bench/framework-compare/results/raw/*3232*; do
    [ -e "${f}" ] || { collect_rc=1; continue; }
    case "${f}" in
        */bench-fandhe-*) ;;
        *) mask < "${f}" > "${OUT}/ab/$(basename "${f}")" || collect_rc=1 ;;
    esac
done
for f in "${ON}"/scripts/bench/framework-compare/compare-*3232*; do
    [ -e "${f}" ] && { mask < "${f}" > "${OUT}/ab/$(basename "${f}")" || collect_rc=1; }
done
echo "collect exit=${collect_rc}" >> "${OUT}/load_gate.log"
echo "released-pending at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${OUT}/hwlock.log"
cp -R "${OUT}" "${FINAL}" || { echo "収録先への複製に失敗: ${FINAL}" >&2; exit 1; }
rm -rf "${ON}"/scripts/bench/framework-compare/target-ab-3232-*
[ "${collect_rc}" = "0" ] || { echo "収録に失敗（collect exit=${collect_rc}）" >&2; exit 1; }
[ "${run_ab_rc}" = "0" ] || { echo "A/B 実行が失敗（run_ab exit=${run_ab_rc}）。ログは収録済み" >&2; exit "${run_ab_rc}"; }
echo "完了"
