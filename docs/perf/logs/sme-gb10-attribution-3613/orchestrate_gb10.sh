#!/bin/bash
# GB10（DGX Spark・SME 非搭載）で SME 既定 ON の 2 セル後退を 3 腕 A/B で切り分けるオーケストレーション（イシュー #3613）。
# 判定規則は同ディレクトリ RULE.txt。下敷きは ../sme-default-on-ab-3232/orchestrate_gb10.sh（ロック・外側ゲート・P-R0・bitdump の手順）。
#
# 腕: before = BASE（本スクリプトを置いたクリーンなツリー = bundle から clone した登録コミット）、
#     prime = BASE の clone + on-arm-prime.patch、after = BASE の clone + on-arm.patch。
# 起動順: ロック → 外側ゲート → P-TREE（sha 照合・差分検証）→ P-R0 → P-RT（before のみ）→ A/B → bitdump（3 腕）→ 収録。
# P-RT を before だけで行う理由: prime・after では定数 true のためドリフト検出テスト
# sme_production_disabled_after_gb10_reject が作りの上で必ず落ちる（RULE.txt §3）。
#
# 使い方（BASE ツリーのルートで。PATH は ${HOME}/.cargo/bin を先頭へ明示する）:
#   BASE=<BASE ツリー絶対パス> WORK=<作業ディレクトリ絶対パス> bash docs/perf/logs/sme-gb10-attribution-3613/orchestrate_gb10.sh --dry-run
#   BASE=... WORK=... setsid nohup bash docs/perf/logs/sme-gb10-attribution-3613/orchestrate_gb10.sh > <ログ> 2>&1 &
set -u
export LC_ALL=C
export PATH="${HOME}/.cargo/bin:${PATH}"

REPO_LOG="docs/perf/logs/sme-gb10-attribution-3613"
LABEL="3613-r1"
GATE_FILE="crates/backend-cpu/src/gemm_blis/mod.rs"
LOCK_DIR="${HOME}/work/.fandhe-hw-lock"
EXPECTED_BITDUMP_LINES=6726847
DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

if [ "${DRY}" = "1" ]; then
    echo "# dry-run: 実行計画（副作用なし）"
    echo "0. ロック: mkdir ${LOCK_DIR}（60 秒おきに再試行）→ 外側ゲート load1 < 1.0 かつ GPU 0%（最大 20 回・30 秒間隔）"
    echo "1. P-TREE: BASE を 2 回 clone し on-arm-prime.patch / on-arm.patch を適用（RULE.txt の sha256 照合・git apply --check が先）。差分は ${GATE_FILE} の 1 件だけ"
    echo "2. P-R0: 3 腕の sme_report() が kernel_enabled: false であること（外れたら中止）"
    echo "3. P-RT: before のみ cargo test -p fandhe-ai-backend-cpu --release"
    echo "4. A/B: run_ab_sme_attribution_gb10_cpu.sh ${LABEL}（3 腕・巡回順固定・内側ゲート 8.0・5 round）"
    echo "5. bitdump: cpu_sme_gate_bit_dump の ^out[ 行を 3 腕で比較（期待 ${EXPECTED_BITDUMP_LINES} 行・sha256 一致・cmp 差分 0）"
    echo "6. 収録: マスクして ${REPO_LOG}/gb10 へ複製。target-ab-* を削除し、ロックを解放"
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

OUT="${WORK}/gb10"
FINAL="${BASE}/${REPO_LOG}/gb10"
PRIME="${WORK}/prime"
AFTER="${WORK}/after"
mkdir -p "${WORK}" "${OUT}/bitdump" || exit 1
if [ -e "${FINAL}" ] || [ -e "${PRIME}" ] || [ -e "${AFTER}" ] || [ -e "${OUT}/load_gate.log" ]; then
    echo "既存の収録先・腕ツリー・ログがあるため停止（上書き禁止）" >&2
    exit 1
fi
if ls "${BASE}"/scripts/bench/framework-compare/results/raw/*3613* >/dev/null 2>&1; then
    echo "既存の 3613 計測記録があるため停止" >&2
    exit 1
fi

mask() { sed -e "s#${HOME}#<home>#g" -e "s#${WORK}#<work>#g" -e "s#${BASE}#<base>#g"; }

until mkdir "${LOCK_DIR}" 2>/dev/null; do sleep 60; done
echo "#3613 $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${LOCK_DIR}/owner"
echo "acquired at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${OUT}/hwlock.log"
release_lock() {
    echo "released at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${OUT}/hwlock.log"
    cp "${OUT}/hwlock.log" "${FINAL}/hwlock.log" 2>/dev/null
    rm -rf "${LOCK_DIR}"
}
trap release_lock EXIT

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

# P-TREE: before = BASE（クリーン）。prime／after は BASE の clone へパッチ適用（sha256 を RULE.txt と照合してから）
expected_sha() { sed -n "s/^$1 sha256: \([0-9a-f]\{64\}\)\$/\1/p" "${BASE}/${REPO_LOG}/RULE.txt" | head -n1; }
: > "${OUT}/patch_sha256.txt"
: > "${OUT}/patch_apply.log"
: > "${OUT}/tree_verify.txt"
for arm in prime after; do
    if [ "${arm}" = "prime" ]; then
        P="${BASE}/${REPO_LOG}/on-arm-prime.patch"; T="${PRIME}"; K="on-arm-prime.patch"; WANT=4
    else
        P="${BASE}/${REPO_LOG}/on-arm.patch"; T="${AFTER}"; K="on-arm.patch"; WANT=2
    fi
    EXPECTED="$(expected_sha "${K}")"
    case "${EXPECTED}" in
        ????????????????????????????????????????????????????????????????) ;;
        *) echo "RULE.txt から ${K} の sha256 を取得できない -> P-TREE 不成立" >&2; exit 1 ;;
    esac
    GOT="$(sha256sum "${P}" | awk '{print $1}')"
    echo "${K} ${GOT}" >> "${OUT}/patch_sha256.txt"
    [ "${GOT}" = "${EXPECTED}" ] || { echo "${K} の sha256 が RULE 記載と不一致 -> P-TREE 不成立" >&2; exit 1; }
    git clone -q "${BASE}" "${T}" || { echo "${arm} 複製に失敗" >&2; exit 1; }
    (cd "${T}" && git apply --check "${P}" && git apply "${P}") >> "${OUT}/patch_apply.log" 2>&1 \
        || { echo "${K} の適用に失敗 -> P-TREE 不成立" >&2; exit 1; }
    [ "$(git -C "${T}" diff --name-only HEAD)" = "${GATE_FILE}" ] || { echo "${arm} の差分ファイルが ${GATE_FILE} の 1 件だけではない" >&2; exit 1; }
    N="$(git -C "${T}" diff -U0 HEAD -- "${GATE_FILE}" | grep -cE '^[-+][^-+]')"
    [ "${N}" = "${WANT}" ] || { echo "${arm} の変更行が ${WANT} 本ではない（${N}）" >&2; exit 1; }
    echo "${arm} changed_lines=${N}" >> "${OUT}/tree_verify.txt"
done
[ -z "$(git -C "${BASE}" status --porcelain --untracked-files=no)" ] || { echo "before(BASE) がクリーンではない" >&2; exit 1; }
{
    for t in before prime after; do
        case "${t}" in before) TR="${BASE}" ;; prime) TR="${PRIME}" ;; *) TR="${AFTER}" ;; esac
        echo "${t} $(grep -n '^const SME_PRODUCTION_ENABLED: bool' "${TR}/${GATE_FILE}")"
    done
} > "${OUT}/gate_constant.txt"
if ! grep -q 'before .*= false;' "${OUT}/gate_constant.txt" || ! grep -q 'prime .*= true;' "${OUT}/gate_constant.txt" \
    || ! grep -q 'after .*= true;' "${OUT}/gate_constant.txt"; then
    echo "定数の値が想定（before=false・prime=true・after=true）ではない" >&2
    exit 1
fi
{
    echo "before $(git -C "${BASE}" rev-parse HEAD)"
    echo "prime $(git -C "${PRIME}" rev-parse HEAD)+on-arm-prime.patch"
    echo "after $(git -C "${AFTER}" rev-parse HEAD)+on-arm.patch"
} > "${OUT}/rev-stamp.txt"
git -C "${BASE}" ls-files | sort | sha256sum | awk '{print $1}' > "${OUT}/tree_fingerprint.txt"
echo "rc=0" >> "${OUT}/tree_verify.txt"
echo "P-TREE ok"

# P-R0（SME 非搭載のため 3 腕とも kernel_enabled: false が期待値）
: > "${OUT}/sme_report.txt"
for t in before prime after; do
    case "${t}" in before) TREE="${BASE}" ;; prime) TREE="${PRIME}" ;; *) TREE="${AFTER}" ;; esac
    P="${WORK}/probe-${t}"
    mkdir -p "${P}/src"
    sed "s#TREE_PLACEHOLDER#${TREE}#" "${BASE}/docs/perf/logs/cpu-gemm-sme-fmopa-1587/gb10/sme-probe-Cargo.toml" > "${P}/Cargo.toml" || exit 1
    cp "${BASE}/docs/perf/logs/cpu-gemm-sme-fmopa-1587/gb10/sme-probe-main.rs" "${P}/src/main.rs" || exit 1
    (cd "${P}" && CARGO_TARGET_DIR="${WORK}/target-probe" cargo run --release -q) > "${WORK}/probe_${t}.log" 2>&1 \
        || { echo "probe ${t} 失敗" >&2; tail -20 "${WORK}/probe_${t}.log" >&2; exit 1; }
    mask < "${WORK}/probe_${t}.log" > "${OUT}/sme_probe_${t}.log"
    echo "${t}: $(grep sme_report= "${OUT}/sme_probe_${t}.log")" >> "${OUT}/sme_report.txt"
    if ! grep -q 'kernel_enabled: false' "${OUT}/sme_probe_${t}.log"; then
        echo "P-R0 不成立（${t}）-> 計測中止（記録のみ残す）" >&2
        exit 1
    fi
done
echo "P-R0 ok"

# P-RT: before のみ
(cd "${BASE}" && CARGO_TARGET_DIR="${WORK}/target-rt-before" cargo test -p fandhe-ai-backend-cpu --release) > "${WORK}/rt_before.raw.log" 2>&1
rt_rc=$?
echo "rt_before exit=${rt_rc}" >> "${OUT}/load_gate.log"
grep -E '^test result|FAILED|panicked' "${WORK}/rt_before.raw.log" | mask > "${OUT}/rt_before.log"
rm -rf "${WORK}/target-rt-before"

# A/B
(cd "${BASE}/scripts/bench/framework-compare" && \
    AB_HOST=gb10 AB_BEFORE_FACADE_PATH="${BASE}/crates/facade" AB_PRIME_FACADE_PATH="${PRIME}/crates/facade" \
    AB_AFTER_FACADE_PATH="${AFTER}/crates/facade" bash run_ab_sme_attribution_gb10_cpu.sh "${LABEL}") > "${OUT}/run_ab.log" 2>&1
run_ab_rc=$?
echo "run_ab exit=${run_ab_rc}" >> "${OUT}/load_gate.log"

# bitdump（3 腕。dump 本体は収録しない）
bd_rc=0
for t in before prime after; do
    case "${t}" in before) TREE="${BASE}" ;; prime) TREE="${PRIME}" ;; *) TREE="${AFTER}" ;; esac
    (cd "${TREE}" && CARGO_TARGET_DIR="${WORK}/target-bd" cargo test -p fandhe-ai --release --test cpu_sme_gate_bit_dump \
        -- --ignored --nocapture --exact dump_cpu_sme_gate_bits) 2>&1 | grep -E '^out\[' > "${WORK}/bits_${t}.txt"
    [ "${PIPESTATUS[0]}" = "0" ] || bd_rc=1
done
{
    echo "expected=${EXPECTED_BITDUMP_LINES}"
    for t in before prime after; do
        echo "${t}_lines=$(wc -l < "${WORK}/bits_${t}.txt" | tr -d ' ')"
        echo "${t}_sha256=$(sha256sum "${WORK}/bits_${t}.txt" | awk '{print $1}')"
    done
} > "${OUT}/bitdump/summary.txt"
: > "${OUT}/bitdump/cmp.txt"
cmp_rc=0
for t in prime after; do
    cmp "${WORK}/bits_before.txt" "${WORK}/bits_${t}.txt" >> "${OUT}/bitdump/cmp.txt" 2>&1 || cmp_rc=1
done
for t in before prime after; do
    [ "$(wc -l < "${WORK}/bits_${t}.txt" | tr -d ' ')" = "${EXPECTED_BITDUMP_LINES}" ] || bd_rc=1
done
echo "bitdump exit=${bd_rc} cmp_exit=${cmp_rc}" >> "${OUT}/load_gate.log"
rm -f "${WORK}"/bits_*.txt
rm -rf "${WORK}/target-bd" "${WORK}/target-probe"

# 収録（マスク）
collect_rc=0
for f in "${OUT}/run_ab.log" "${OUT}/patch_apply.log" "${OUT}/bitdump/cmp.txt"; do
    [ -e "${f}" ] || continue
    if mask < "${f}" > "${f}.masked"; then mv "${f}.masked" "${f}"; else rm -f "${f}.masked"; collect_rc=1; fi
done
mkdir -p "${OUT}/ab" || collect_rc=1
for f in "${BASE}"/scripts/bench/framework-compare/results/raw/*3613*; do
    [ -e "${f}" ] || { collect_rc=1; continue; }
    case "${f}" in
        */bench-fandhe-*) ;;
        *) mask < "${f}" > "${OUT}/ab/$(basename "${f}")" || collect_rc=1 ;;
    esac
done
# 腕別バイナリの sha256（参考記録。帰属の根拠には使わない。RULE.txt §2）
cat "${OUT}"/ab/sha-3613-* > "${OUT}/bench_sha256.txt" 2>/dev/null || collect_rc=1
for f in "${BASE}"/scripts/bench/framework-compare/compare-*3613*; do
    if [ -e "${f}" ]; then mask < "${f}" > "${OUT}/ab/$(basename "${f}")" || collect_rc=1; fi
done
echo "collect exit=${collect_rc}" >> "${OUT}/load_gate.log"
echo "released-pending at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${OUT}/hwlock.log"
cp -R "${OUT}" "${FINAL}" || { echo "収録先への複製に失敗: ${FINAL}" >&2; exit 1; }
rm -rf "${BASE}"/scripts/bench/framework-compare/target-ab-3613-*
[ "${collect_rc}" = "0" ] || { echo "収録に失敗（collect exit=${collect_rc}）" >&2; exit 1; }
[ "${run_ab_rc}" = "0" ] || { echo "A/B 実行が失敗（run_ab exit=${run_ab_rc}）。ログは収録済み" >&2; exit "${run_ab_rc}"; }
echo "完了"
