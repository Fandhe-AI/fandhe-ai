#!/bin/bash
# Mac mini（Apple M4）で #3236 の l2_dx を反復数・round 数を増やして測り直すオーケストレーション（イシュー #3615・RULE-reopen-1）。
# 判定規則は同ディレクトリ RULE.txt 末尾の RULE-reopen-1（計測前にコミット済み）。
# orchestrate_macmini.sh（#3236。変更しない）の派生: 負荷ゲート・P-TREE・P-R0・P-BIT・失敗即停止・マスク収録は同一作法。
# 変更点: LABEL=3236-reopen-r1・ROUNDS=11・計測は cpu_sme_thin_m64_reopen（WARMUP 200／ITERS 2000）だけ・
#         framework-compare と記録 run（RAYON_NUM_THREADS=1）は行わない・収録先は macmini-reopen。
# macOS の bash 3.2 で動くよう変数は ${VAR} と書く（mapfile・連想配列・jq は使わない）。
#
# 使い方（before ツリーのルートで）:
#   ON_MAIN=<main ツリー絶対パス> WORK=<作業ディレクトリ絶対パス> \
#     bash docs/perf/logs/sme-thin-m64-ab-3236/orchestrate_macmini_reopen.sh --dry-run   # 計画のみ表示（副作用なし）
#     bash docs/perf/logs/sme-thin-m64-ab-3236/orchestrate_macmini_reopen.sh             # 実行（Mac mini 上）
# 完了マーカー: 最終行 `done.`、各段の `p_bit exit=` `gemm exit=` `collect exit=` は load_gate.log に残る。
set -u
export LC_ALL=C

REPO_LOG="docs/perf/logs/sme-thin-m64-ab-3236"
LABEL="3236-reopen-r1"
ROUNDS=11
GATE_FILE="crates/backend-cpu/src/gemm_blis/mod.rs"
LOCK_DIR="${HOME}/work/.fandhe-hw-lock"
DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

case "${LABEL}" in
    *[!A-Za-z0-9_-]*) echo "error: LABEL が allowlist 外" >&2; exit 1 ;;
esac

if [ "${DRY}" = "1" ]; then
    echo "# dry-run: 実行計画（副作用なし）label=${LABEL}"
    echo "0. ロック: mkdir ${LOCK_DIR}（取れなければ 60 秒おきに再試行）。owner に '#3615 <UTC>' を書き hwlock.log へ取得・解放時刻を残す"
    echo "1. P-TREE: main を clone して on-arm.patch を適用（sha256 は RULE.txt の値と照合・git apply --check が先）。差分ファイルが ${GATE_FILE} の 1 件だけで before が false・after が true。SME_PRODUCTION_ENABLED は両腕とも false"
    echo "2. P-R0: 両腕で sme_report() が kernel_enabled: true（外れたら中止）"
    echo "3. P-BIT: main ツリーのみで cargo test -p fandhe-ai-backend-cpu --release --lib sme_thin_m64 と -- --ignored sme_thin_m64 --test-threads=1"
    echo "4. ビルド: 両ツリーで cpu_sme_thin_m64_reopen を release ビルド（計測前に完了。ビルドと計測を並べない）"
    echo "5. 計測: gemm（l2_dx・l1_fwd・l1_dx）と kgrid を ${ROUNDS} round（round ごとに load1 < 3.0 のゲート・起動順は交互に反転）"
    echo "6. 収録: マスク（ホスト名・\$HOME・WORK・ツリーパス）して ${REPO_LOG}/macmini-reopen へ複製。ロックを解放"
    exit 0
fi

for v in ON_MAIN WORK; do
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

MAIN="${ON_MAIN}"
OUT="${WORK}/macmini-reopen"
FINAL="${MAIN}/${REPO_LOG}/macmini-reopen"
AFTER="${WORK}/on-arm"
GD="${OUT}/gemm"
mkdir -p "${WORK}" "${OUT}" "${GD}" || exit 1
if [ -e "${FINAL}" ] || [ -e "${AFTER}" ] || [ -e "${OUT}/load_gate.log" ]; then
    echo "既存の収録先・after ツリー・ログがあるため停止（上書き禁止）" >&2
    exit 1
fi

mask() { sed -e "s#${HOME}#<home>#g" -e "s#${WORK}#<work>#g" -e "s#${MAIN}#<main>#g"; }

# 実機ロック（P-LOCK）。解放は trap で必ず行う
until mkdir "${LOCK_DIR}" 2>/dev/null; do sleep 60; done
echo "#3615 $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${LOCK_DIR}/owner"
echo "acquired at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${OUT}/hwlock.log"
release_lock() {
    echo "released at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${OUT}/hwlock.log"
    rm -rf "${LOCK_DIR}"
}
trap release_lock EXIT

# P-TREE
git clone -q "${MAIN}" "${AFTER}" || { echo "after 複製に失敗" >&2; exit 1; }
PATCH="${MAIN}/${REPO_LOG}/on-arm.patch"
EXPECTED="$(sed -n 's/^on-arm\.patch sha256: \([0-9a-f]\{64\}\)$/\1/p' "${MAIN}/${REPO_LOG}/RULE.txt" | head -n1)"
case "${EXPECTED}" in
    ????????????????????????????????????????????????????????????????) ;;
    *) echo "RULE.txt から patch の sha256 を取得できない -> P-TREE 不成立" >&2; exit 1 ;;
esac
shasum -a 256 "${PATCH}" | awk '{print $1}' > "${OUT}/patch_sha256.txt" || exit 1
[ "$(cat "${OUT}/patch_sha256.txt")" = "${EXPECTED}" ] || { echo "patch の sha256 が RULE 記載と不一致 -> P-TREE 不成立" >&2; exit 1; }
(cd "${AFTER}" && git apply --check "${PATCH}" && git apply "${PATCH}") > "${OUT}/patch_apply.log" 2>&1 \
    || { echo "patch 適用に失敗 -> P-TREE 不成立" >&2; exit 1; }
git -C "${AFTER}" diff --name-only HEAD > "${OUT}/tree_verify.txt"
[ "$(cat "${OUT}/tree_verify.txt")" = "${GATE_FILE}" ] || { echo "差分ファイルが ${GATE_FILE} の 1 件だけではない -> P-TREE 不成立" >&2; exit 1; }
git -C "${MAIN}" status --porcelain --untracked-files=no > "${OUT}/main_status.txt"
[ ! -s "${OUT}/main_status.txt" ] || { echo "main ツリーに未コミット変更がある -> P-TREE 不成立" >&2; exit 1; }
grep -n '^const SME_THIN_M64_ENABLED: bool' "${MAIN}/${GATE_FILE}" > "${OUT}/gate_constant.txt"
grep -n '^const SME_THIN_M64_ENABLED: bool' "${AFTER}/${GATE_FILE}" >> "${OUT}/gate_constant.txt"
[ "$(sed -n 1p "${OUT}/gate_constant.txt" | sed 's/^[0-9]*://')" = "const SME_THIN_M64_ENABLED: bool = false;" ] || { echo "before の定数が false ではない" >&2; exit 1; }
[ "$(sed -n 2p "${OUT}/gate_constant.txt" | sed 's/^[0-9]*://')" = "const SME_THIN_M64_ENABLED: bool = true;" ] || { echo "after の定数が true ではない" >&2; exit 1; }
for t in "${MAIN}" "${AFTER}"; do
    grep -q '^const SME_PRODUCTION_ENABLED: bool = false;$' "${t}/${GATE_FILE}" || { echo "SME_PRODUCTION_ENABLED が両腕とも false ではない" >&2; exit 1; }
done
git -C "${MAIN}" rev-parse HEAD > "${OUT}/rev-stamp.txt"
echo "$(git -C "${MAIN}" rev-parse HEAD)+on-arm.patch" >> "${OUT}/rev-stamp.txt"
git -C "${MAIN}" ls-files | sort | shasum -a 256 | awk '{print $1}' > "${OUT}/tree_fingerprint.txt"
echo "P-TREE ok"

# P-R0
: > "${OUT}/sme_report.txt"
for t in before after; do
    P="${WORK}/probe-${t}"
    mkdir -p "${P}/src"
    if [ "${t}" = "before" ]; then TREE="${MAIN}"; else TREE="${AFTER}"; fi
    sed "s#TREE_PLACEHOLDER#${TREE}#" "${MAIN}/docs/perf/logs/cpu-gemm-sme-fmopa-1587/gb10/sme-probe-Cargo.toml" > "${P}/Cargo.toml" || exit 1
    cp "${MAIN}/docs/perf/logs/cpu-gemm-sme-fmopa-1587/gb10/sme-probe-main.rs" "${P}/src/main.rs" || exit 1
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

# P-BIT（main ツリーのみ。after では --lib を実行しない）。計測より前に完了させる
p_bit_rc=0
(cd "${MAIN}" && CARGO_TARGET_DIR="${WORK}/target-main" cargo test -p fandhe-ai-backend-cpu --release --lib sme_thin_m64 \
    && CARGO_TARGET_DIR="${WORK}/target-main" cargo test -p fandhe-ai-backend-cpu --release --lib -- --ignored sme_thin_m64 --test-threads=1) > "${WORK}/p_bit.log" 2>&1 || p_bit_rc=$?
mask < "${WORK}/p_bit.log" > "${OUT}/p_bit.log"
echo "p_bit exit=${p_bit_rc}" >> "${OUT}/load_gate.log"
[ "${p_bit_rc}" = "0" ] || { echo "P-BIT 不成立 -> 計測中止（undetermined）" >&2; exit 1; }
echo "P-BIT ok"

# 全 binary を先にビルド（計測中はビルドを並走させない）
BIN_M="${WORK}/target-main/release/examples"
BIN_A="${WORK}/target-after/release/examples"
(cd "${MAIN}" && CARGO_TARGET_DIR="${WORK}/target-main" cargo build --release -p fandhe-ai-backend-cpu --example cpu_sme_thin_m64_reopen) > "${WORK}/build_before.log" 2>&1 || { echo "before ビルド失敗" >&2; exit 1; }
(cd "${AFTER}" && CARGO_TARGET_DIR="${WORK}/target-after" cargo build --release -p fandhe-ai-backend-cpu --example cpu_sme_thin_m64_reopen) > "${WORK}/build_after.log" 2>&1 || { echo "after ビルド失敗" >&2; exit 1; }
for d in "${BIN_M}" "${BIN_A}"; do
    [ -x "${d}/cpu_sme_thin_m64_reopen" ] || { echo "binary がない: ${d}" >&2; exit 1; }
done
shasum -a 256 "${BIN_M}/cpu_sme_thin_m64_reopen" "${BIN_A}/cpu_sme_thin_m64_reopen" | awk '{print $1}' | sed 's#$# cpu_sme_thin_m64_reopen#' >> "${OUT}/binary_sha256.txt"
echo "build ok"

# 負荷ゲート（load1 < 3.0・30 秒間隔・最大 30 分）
GATE_LOG="${GD}/load_gate.log"
: > "${GATE_LOG}"
wait_gate() { # wait_gate <round>
    local r=$1 waited=0 load1
    while :; do
        load1="$(uptime | sed 's/.*load averages*: *//' | awk '{gsub(",","",$1); print $1}')"
        # load1 が空・非数値だと awk の l+0 が 0 になりゲートを素通りするため、数値形式を先に検証する（RULE.txt P-GATE）
        if ! printf '%s' "${load1}" | grep -Eq '^[0-9]+([.][0-9]+)?$'; then
            echo "round${r} gate=load1-unavailable gate_ok=0 threshold=3.0 load1=${load1:-NA} waited_s=${waited} at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${GATE_LOG}"
            return 1
        fi
        if awk -v l="${load1}" 'BEGIN{exit !(l+0 < 3.0)}'; then
            echo "round${r} gate=pass gate_ok=1 threshold=3.0 load1=${load1} waited_s=${waited} at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${GATE_LOG}"
            return 0
        fi
        if [ "${waited}" -ge 1800 ]; then
            echo "round${r} gate=timeout gate_ok=0 threshold=3.0 load1=${load1} waited_s=${waited} at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${GATE_LOG}"
            return 1
        fi
        sleep 30
        waited=$((waited + 30))
    done
}

# gemm・kgrid の ROUNDS round（起動順は交互に反転）
gemm_rc=0
r=1
while [ "${r}" -le "${ROUNDS}" ]; do
    wait_gate "${r}" || gemm_rc=1
    if [ $((r % 2)) = 1 ]; then ORDER="before after"; else ORDER="after before"; fi
    for arm in ${ORDER}; do
        if [ "${arm}" = "before" ]; then D="${BIN_M}"; else D="${BIN_A}"; fi
        for mode in gemm kgrid; do
            "${D}/cpu_sme_thin_m64_reopen" "${arm}" "${mode}" > "${GD}/${mode}-${arm}-r${r}.txt" 2> "${GD}/${mode}-${arm}-r${r}.err" || gemm_rc=1
        done
    done
    r=$((r + 1))
done
echo "gemm exit=${gemm_rc}" >> "${OUT}/load_gate.log"

# 収録（マスク）
mask_rc=0
for f in "${OUT}/patch_apply.log" "${GD}"/*.err; do
    [ -e "${f}" ] || continue
    if mask < "${f}" > "${f}.masked"; then mv "${f}.masked" "${f}"; else rm -f "${f}.masked"; mask_rc=1; fi
done
collect_rc=${mask_rc}
echo "collect exit=${collect_rc}" >> "${OUT}/load_gate.log"
echo "released-pending at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${OUT}/hwlock.log"
cp -R "${OUT}" "${FINAL}" || { echo "収録先への複製に失敗: ${FINAL}" >&2; exit 1; }
rm -rf "${WORK}/target-probe"
[ "${collect_rc}" = "0" ] || { echo "収録に失敗（collect exit=${collect_rc}）" >&2; exit 1; }
[ "${gemm_rc}" = "0" ] || { echo "計測系列が失敗（gemm exit=${gemm_rc}）。ログは収録済み" >&2; exit 1; }
echo "done."
