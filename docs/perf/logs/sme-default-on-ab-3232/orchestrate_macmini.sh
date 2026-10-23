#!/bin/bash
# Mac mini（Apple M4）で SME 既定 ON（main）と OFF（定数 1 行を戻したもの）を測るオーケストレーション（イシュー #3232）。
# 判定規則は同ディレクトリ RULE.txt（計測前に単独コミット済み）。前例は
# docs/perf/logs/sme-macmini-3176/orchestrate_macmini.sh（負荷ゲート・P-TREE・P-R0・失敗即停止・マスク収録）。
# macOS の bash 3.2 で動くよう変数は ${VAR} と書く（mapfile・連想配列・jq は使わない）。
#
# 腕: ON = 本スクリプトを置いたツリー（bundle から clone した main。クリーン）、
#     OFF = ON を clone して off-arm.patch を当てたツリー（P-TREE が 1 ファイル 1 行を検査）。
# 起動順: ロック取得 → P-TREE → P-R0 → A/B（run_ab_sme_default_on_cpu.sh AB_HOST=macmini）→ 収録 → ロック解放。
# OFF ツリーでは `--lib` を実行しない（ゲートが true のドリフト検出テストが構造上落ちるため。RULE §3 P-RT）。
#
# 使い方（ON ツリーのルートで）:
#   ON=<ON ツリー絶対パス> TORCH_PY=<venv-torch/bin/python 絶対パス> WORK=<作業ディレクトリ絶対パス> \
#     bash docs/perf/logs/sme-default-on-ab-3232/orchestrate_macmini.sh --dry-run   # 計画のみ表示（副作用なし）
#     bash docs/perf/logs/sme-default-on-ab-3232/orchestrate_macmini.sh             # 実行（Mac mini 上）
# WORK は同期ツリーの外に置く（target もそこへ置く）。ログは両 checkout の外（${WORK}/macmini）へ出し、
# すべて終わってから ${ON}/${REPO_LOG}/macmini へ収録する（計測中に ON ツリーを汚さない。
# P-TREE のファイル一覧比較と A/B の未追跡ファイル検査が通るため）。ホスト名・ユーザー名は収録物へ書かない。
set -u
export LC_ALL=C

REPO_LOG="docs/perf/logs/sme-default-on-ab-3232"
LABEL="3232-r1"
GATE_FILE="crates/backend-cpu/src/gemm_blis/mod.rs"
LOCK_DIR="${HOME}/work/.fandhe-hw-lock"
DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

if [ "${DRY}" = "1" ]; then
    echo "# dry-run: 実行計画（副作用なし）"
    echo "0. ロック: mkdir ${LOCK_DIR}（取れなければ 60 秒おきに再試行）。owner に '#3232 <UTC>' を書き hwlock.log へ取得・解放時刻を残す"
    echo "1. P-TREE: ON を clone して off-arm.patch を適用（sha256 は RULE.txt の値と照合・git apply --check が先）。差分ファイルが ${GATE_FILE} の 1 件だけで OFF が false・ON が true であること"
    echo "2. P-R0: 両腕で sme_report() が kernel_enabled: true であること（cpu-gemm-sme-fmopa-1587/gb10/sme-probe-* を流用。外れたら中止）"
    echo "3. A/B: scripts/bench/framework-compare/run_ab_sme_default_on_cpu.sh ${LABEL}（AB_HOST=macmini・5 round・PyTorch 同席・負荷ゲート load1 < 3.0）"
    echo "4. 収録: マスク（ホスト名・\$HOME・WORK・ツリーパス）して ${REPO_LOG}/macmini へ複製。target-ab-* と probe の target を削除し、ロックを解放"
    exit 0
fi

for v in ON TORCH_PY WORK; do
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

OUT="${WORK}/macmini"
FINAL="${ON}/${REPO_LOG}/macmini"
OFF="${WORK}/off"
mkdir -p "${WORK}" "${OUT}" || exit 1
if [ -e "${FINAL}" ] || [ -e "${OFF}" ] || [ -e "${OUT}/load_gate.log" ]; then
    echo "既存の収録先・OFF ツリー・ログがあるため停止（上書き禁止）" >&2
    exit 1
fi

mask() { sed -e "s#${HOME}#<home>#g" -e "s#${WORK}#<work>#g" -e "s#${ON}#<on>#g"; }

# 実機ロック（P-LOCK）。解放は trap で必ず行う
until mkdir "${LOCK_DIR}" 2>/dev/null; do sleep 60; done
echo "#3232 $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${LOCK_DIR}/owner"
echo "acquired at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${OUT}/hwlock.log"
release_lock() {
    echo "released at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${OUT}/hwlock.log"
    rm -rf "${LOCK_DIR}"
}
trap release_lock EXIT

# P-TREE
git clone -q "${ON}" "${OFF}" || { echo "OFF 複製に失敗" >&2; exit 1; }
PATCH="${ON}/${REPO_LOG}/../sme-default-on-3231/off-arm.patch"
EXPECTED="$(sed -n 's/^off-arm\.patch sha256: \([0-9a-f]\{64\}\)$/\1/p' "${ON}/${REPO_LOG}/RULE.txt" | head -n1)"
case "${EXPECTED}" in
    ????????????????????????????????????????????????????????????????) ;;
    *) echo "RULE.txt から patch の sha256 を取得できない -> P-TREE 不成立" >&2; exit 1 ;;
esac
shasum -a 256 "${PATCH}" | awk '{print $1}' > "${OUT}/patch_sha256.txt" || exit 1
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
git -C "${ON}" ls-files | sort | shasum -a 256 | awk '{print $1}' > "${OUT}/tree_fingerprint.txt"
echo "P-TREE ok"

# P-R0
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
if ! grep -q 'kernel_enabled: true' "${OUT}/sme_probe_off.log" || ! grep -q 'kernel_enabled: true' "${OUT}/sme_probe_on.log"; then
    echo "P-R0 不成立（kernel_enabled が true でない）-> 計測中止（undetermined）" >&2
    exit 1
fi
echo "P-R0 ok"

# A/B（OFF = before・ON = after）。Cargo.lock は A/B スクリプトが trap で元へ戻す
(cd "${ON}/scripts/bench/framework-compare" && \
    AB_HOST=macmini AB_BEFORE_FACADE_PATH="${OFF}/crates/facade" AB_AFTER_FACADE_PATH="${ON}/crates/facade" \
    AB_TORCH_PYTHON="${TORCH_PY}" bash run_ab_sme_default_on_cpu.sh "${LABEL}") > "${OUT}/run_ab.log" 2>&1
run_ab_rc=$?
echo "run_ab exit=${run_ab_rc}" >> "${OUT}/load_gate.log"
mask_rc=0
for f in "${OUT}/run_ab.log" "${OUT}/patch_apply.log"; do
    [ -e "${f}" ] || continue
    if mask < "${f}" > "${f}.masked"; then mv "${f}.masked" "${f}"; else rm -f "${f}.masked"; mask_rc=1; fi
done
collect_rc=${mask_rc}
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
rm -rf "${ON}"/scripts/bench/framework-compare/target-ab-3232-* "${WORK}/target-probe"
[ "${collect_rc}" = "0" ] || { echo "収録に失敗（collect exit=${collect_rc}）" >&2; exit 1; }
[ "${run_ab_rc}" = "0" ] || { echo "A/B 実行が失敗（run_ab exit=${run_ab_rc}）。ログは収録済み" >&2; exit "${run_ab_rc}"; }
echo "完了"
