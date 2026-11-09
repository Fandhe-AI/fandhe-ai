#!/bin/bash
# GB10（DGX Spark・SME 非搭載）での SME 既定 ON の A/B（構造ガード）（イシュー #4156）。
# 役割: SME `fmopa` カーネル既定 ON の「新しい事前登録 A/B」を GB10で測るオーケストレーション。
# 判定規則は同ディレクトリ RULE.txt（計測前に単独コミット済み）。#3232 の orchestrate_gb10.sh を派生元とし、
# 腕の組み立てを逆にした: OFF = 本スクリプトを置いたツリー（BASE。main の clone・クリーン・定数 false）、
# ON = BASE を clone して on-arm.patch（定数 true の 1 行）を当てた計測専用ツリー。main へコミットしない。
# 追加ゲート: P-OWNER（所有者決定が RULE §9 に転記済みか。未決なら計測を始めない）・
# P-RT（両腕。ON は sme_production_disabled_after_gb10_reject を除外）・P-BIT（両腕 bitdump の一致）。
# 起動順: P-OWNER → ロック → 外側ゲート → P-TREE → P-R0 → P-RT → A/B（AB_HOST=gb10）→ P-BIT → マスク収録。
# 既定の切り替えは行わない（定数は計測専用ツリーでのみ true）。
# macOS の bash 3.2 でも動くよう変数は ${VAR} と書く（mapfile・連想配列・jq は使わない）。
#
# 使い方（BASE ツリーのルートで）:
#   BASE=<クリーンな clone の絶対パス> WORK=<作業ディレクトリ絶対パス> [TORCH_PY=<python 絶対パス>] \
#     bash docs/perf/logs/sme-default-on-ab-4156/orchestrate_gb10.sh --dry-run   # 計画のみ表示（副作用なし）
#     bash docs/perf/logs/sme-default-on-ab-4156/orchestrate_gb10.sh             # 実行（GB10上）
# WORK は同期ツリーの外に置く（target もそこへ置く）。ログは BASE の外（${WORK}/gb10）へ出し、
# 終わってから ${BASE}/${REPO_LOG}/gb10 へ収録する。ホスト名・ユーザー名は収録物へ書かない。
set -u
export LC_ALL=C
export PATH="${HOME}/.cargo/bin:${PATH}"

REPO_LOG="docs/perf/logs/sme-default-on-ab-4156"
HOST="gb10"
LABEL="4156-r1"
GATE_FILE="crates/backend-cpu/src/gemm_blis/mod.rs"
LOCK_DIR="${HOME}/work/.fandhe-hw-lock"
EXPECTED_BITDUMP_LINES=6726847
RT_SKIP_TEST="sme_production_disabled_after_gb10_reject"
EXPECT_KERNEL="false"
DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

if command -v sha256sum >/dev/null 2>&1; then
    sha() { sha256sum "$@"; }
else
    sha() { shasum -a 256 "$@"; }
fi

# P-OWNER: RULE.txt §9 に所有者決定が転記されているかを読むだけで検査する（ソースは実行しない）。
# 値はアンカー付き正規表現で取り出し、列挙値と URL の形式に完全一致しなければ不成立（fail-closed）。
OWNER_URL_RE='^https://github\.com/Fandhe-AI/fandhe-ai/(issues|pull)/[0-9]+#issuecomment-[0-9]+$'
# 各キーはちょうど 1 行だけを要求する（重複・欠落は空を返して不成立にする。aggregate.py の read_owner と同じ契約）。
owner_field() {
    local n
    n="$(grep -c "^$1: *" "$2")"
    if [ "${n}" != "1" ]; then echo "P-OWNER 不成立: $1 が ${n} 件（1 件ちょうどを要求）" >&2; return 0; fi
    sed -n "s/^$1: *\(.*[^ ]\) *\$/\1/p" "$2"
}
check_owner() { # check_owner <RULE.txt> -> 0=成立 1=不成立（理由は stderr）
    local rule=$1 mode nb und murl nurl uurl u rc=0
    mode="$(owner_field GB10-MODE "${rule}")"
    nb="$(owner_field NOISE-BAND "${rule}")"
    und="$(owner_field UNDETERMINED-HANDLING "${rule}")"
    murl="$(owner_field GB10-MODE-URL "${rule}")"
    nurl="$(owner_field NOISE-BAND-URL "${rule}")"
    uurl="$(owner_field UNDETERMINED-HANDLING-URL "${rule}")"
    case "${mode}" in GB10-RECORD | GB10-NBSEP | GB10-GATE-ONLY-STRUCTURAL) ;; *) echo "P-OWNER 不成立: GB10-MODE が未決または列挙外 (${mode:-空})" >&2; rc=1 ;; esac
    case "${nb}" in adopted | not-adopted) ;; *) echo "P-OWNER 不成立: NOISE-BAND が未決または列挙外 (${nb:-空})" >&2; rc=1 ;; esac
    case "${und}" in auto-reremeasure | return-to-owner) ;; *) echo "P-OWNER 不成立: UNDETERMINED-HANDLING が未決または列挙外 (${und:-空})" >&2; rc=1 ;; esac
    for u in "${murl}" "${nurl}" "${uurl}"; do
        printf '%s\n' "${u}" | grep -Eq "${OWNER_URL_RE}" || { echo "P-OWNER 不成立: 所有者回答コメントの URL が欠落または形式不正 (${u:-空})" >&2; rc=1; }
    done
    return ${rc}
}

if [ "${DRY}" = "1" ]; then
    echo "# dry-run: 実行計画（副作用なし）"
    echo "0. P-OWNER: ${REPO_LOG}/RULE.txt §9 の GB10-MODE・NOISE-BAND・UNDETERMINED-HANDLING と各回答コメント URL が記入済みであること（未決ならロック取得前に停止）。prereg_append.txt の rule_sha256 が RULE.txt と一致すること"
    if [ -n "${BASE:-}" ] && [ -f "${BASE}/${REPO_LOG}/RULE.txt" ]; then
        if check_owner "${BASE}/${REPO_LOG}/RULE.txt"; then echo "   (現在の RULE.txt: P-OWNER 成立)"; else echo "   (現在の RULE.txt: P-OWNER 不成立。実行モードは計測を始めない)"; fi
    fi
    echo "1. ロック: mkdir ${LOCK_DIR}（60 秒おきに再試行）。owner に '#4156 <UTC>' を書き hwlock.log へ取得・解放時刻を残す"
    echo "1b. 外側ゲート: load1 < 1.0・GPU 使用率 0%・nvidia-smi --query-compute-apps が 0 行（最大 20 回・30 秒間隔。内側ゲートは 8.0）"
    echo "2. P-TREE: BASE を clone して on-arm.patch を適用（sha256 は RULE.txt の値と照合・git apply --check が先）。差分が ${GATE_FILE} の 1 件だけで OFF=false・ON=true"
    echo "3. P-R0: 両腕の sme_report() が kernel_enabled: ${EXPECT_KERNEL} であること（外れたら中止）"
    echo "4. P-RT: ON は cargo test -p fandhe-ai-backend-cpu --release -- --skip ${RT_SKIP_TEST}、OFF はスキップなしの同コマンド"
    echo "5. A/B: scripts/bench/framework-compare/run_ab_sme_default_on_4156_cpu.sh ${LABEL}（AB_HOST=${HOST}・5 round）"
    echo "6. P-BIT: cpu_sme_gate_bit_dump の ^out[ 行を ON/OFF で比較（期待 ${EXPECTED_BITDUMP_LINES} 行・sha256 一致・cmp 差分 0）"
    echo "7. 収録: マスクして ${REPO_LOG}/${HOST} へ複製。実ホスト名等の grep が 0 件でなければ停止。target-ab-* を削除しロックを解放"
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
TORCH_PY="${TORCH_PY:-}"
RULE="${BASE}/${REPO_LOG}/RULE.txt"
# P-OWNER（ロック取得より前。未決なら何も始めない）
check_owner "${RULE}" || { echo "P-OWNER 不成立 -> 計測を始めない（undetermined）。RULE.txt §9 へ所有者決定を転記してから再実行すること" >&2; exit 1; }
PREREG="${BASE}/${REPO_LOG}/prereg_append.txt"
[ -f "${PREREG}" ] || { echo "prereg_append.txt が無い -> P-OWNER 不成立" >&2; exit 1; }
[ "$(sed -n 's/^rule_sha256=\([0-9a-f]\{64\}\)$/\1/p' "${PREREG}" | head -n1)" = "$(sha "${RULE}" | awk '{print $1}')" ] \
    || { echo "prereg_append.txt の rule_sha256 が現在の RULE.txt と不一致 -> P-OWNER 不成立" >&2; exit 1; }
# prepared_at の形式検証（ロック取得より前。欠落・形式不正・未来時刻は計測を始めない。RULE §4）
PREPARED_AT="$(sed -n 's/^prepared_at=\([0-9]\{4\}-[0-9]\{2\}-[0-9]\{2\}T[0-9]\{2\}:[0-9]\{2\}:[0-9]\{2\}Z\)$/\1/p' "${PREREG}" | head -n1)"
[ -n "${PREPARED_AT}" ] || { echo "prereg_append.txt の prepared_at が欠落または形式不正 (YYYY-MM-DDTHH:MM:SSZ) -> P-OWNER 不成立" >&2; exit 1; }
NOW_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
[ "${PREPARED_AT}" \< "${NOW_UTC}" ] || { echo "prereg_append.txt の prepared_at (${PREPARED_AT}) が現在時刻 (${NOW_UTC}) 以降 -> P-OWNER 不成立" >&2; exit 1; }

OUT="${WORK}/${HOST}"
FINAL="${BASE}/${REPO_LOG}/${HOST}"
ON="${WORK}/on"
mkdir -p "${WORK}" "${OUT}/bitdump" || exit 1
if [ -e "${FINAL}" ] || [ -e "${ON}" ] || [ -e "${OUT}/load_gate.log" ]; then
    echo "既存の収録先・ON ツリー・ログがあるため停止（上書き禁止）" >&2
    exit 1
fi

mask() { sed -e "s#${HOME}#<home>#g" -e "s#${WORK}#<work>#g" -e "s#${BASE}#<base>#g"; }

# 実機ロック（P-LOCK）。解放は trap で必ず行う
until mkdir "${LOCK_DIR}" 2>/dev/null; do sleep 60; done
echo "#4156 $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${LOCK_DIR}/owner"
ACQUIRED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "acquired at=${ACQUIRED_AT}" >> "${OUT}/hwlock.log"
release_lock() {
    echo "released at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${OUT}/hwlock.log"
    rm -rf "${LOCK_DIR}"
}
trap release_lock EXIT
# 取得直後の順序検証（prepared_at < acquired でなければ後続へ進まない。RULE §4）
[ "${PREPARED_AT}" \< "${ACQUIRED_AT}" ] || { echo "prepared_at (${PREPARED_AT}) が hwlock 取得時刻 (${ACQUIRED_AT}) より前でない -> P-OWNER 不成立" >&2; exit 1; }
cp "${PREREG}" "${OUT}/prereg_append.txt"

# 外側専有ゲート（記録）: load1 < 1.0・GPU 使用率 0%・compute プロセス 0 件（最大 20 回・30 秒間隔）
outer_gate() {
    i=1
    while [ "${i}" -le 20 ]; do
        l1="$(cut -d' ' -f1 /proc/loadavg)"
        gu="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
        # クエリ失敗（非ゼロ終了）を 0 行と取り違えないよう終了コードを確認する。失敗時は np を空にして awk 側で不成立にする
        if np_raw="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)"; then
            np="$(printf '%s' "${np_raw}" | grep -c . || true)"
        else
            np=""
        fi
        if awk -v l="${l1}" -v g="${gu:-x}" -v n="${np:-x}" 'BEGIN{exit !(l < 1.0 && g == 0 && n == 0)}'; then
            echo "start attempt=${i} load1=${l1} gpu_util=${gu} compute_apps=${np} pass" >> "${OUT}/load_gate_outer.log"
            return 0
        fi
        echo "start attempt=${i} load1=${l1} gpu_util=${gu:-NA} compute_apps=${np:-NA(query-failed)} wait" >> "${OUT}/load_gate_outer.log"
        sleep 30
        i=$((i + 1))
    done
    echo "start attempt=20 fail" >> "${OUT}/load_gate_outer.log"
    return 1
}
outer_gate || { echo "外側ゲート不成立 -> 計測中止" >&2; exit 1; }

# P-TREE
git clone -q "${BASE}" "${ON}" || { echo "ON 複製に失敗" >&2; exit 1; }
PATCH="${BASE}/${REPO_LOG}/on-arm.patch"
EXPECTED="$(sed -n 's/^on-arm\.patch sha256: \([0-9a-f]\{64\}\)$/\1/p' "${RULE}" | head -n1)"
case "${EXPECTED}" in
    ????????????????????????????????????????????????????????????????) ;;
    *) echo "RULE.txt から patch の sha256 を取得できない -> P-TREE 不成立" >&2; exit 1 ;;
esac
sha "${PATCH}" | awk '{print $1}' > "${OUT}/patch_sha256.txt" || exit 1
[ "$(cat "${OUT}/patch_sha256.txt")" = "${EXPECTED}" ] || { echo "patch の sha256 が RULE 記載と不一致 -> P-TREE 不成立" >&2; exit 1; }
(cd "${ON}" && git apply --check "${PATCH}" && git apply "${PATCH}") > "${OUT}/patch_apply.log" 2>&1 \
    || { echo "patch 適用に失敗 -> P-TREE 不成立" >&2; exit 1; }
git -C "${ON}" diff --name-only HEAD > "${OUT}/tree_verify.txt"
[ "$(cat "${OUT}/tree_verify.txt")" = "${GATE_FILE}" ] || { echo "差分ファイルが ${GATE_FILE} の 1 件だけではない -> P-TREE 不成立" >&2; exit 1; }
[ -z "$(git -C "${BASE}" status --porcelain --untracked-files=no)" ] || { echo "BASE（OFF）に未コミット変更がある -> P-TREE 不成立" >&2; exit 1; }
grep -n '^const SME_PRODUCTION_ENABLED: bool' "${BASE}/${GATE_FILE}" > "${OUT}/gate_constant.txt"
grep -n '^const SME_PRODUCTION_ENABLED: bool' "${ON}/${GATE_FILE}" >> "${OUT}/gate_constant.txt"
[ "$(sed -n 1p "${OUT}/gate_constant.txt" | sed 's/^[0-9]*://')" = "const SME_PRODUCTION_ENABLED: bool = false;" ] || { echo "OFF の定数が false ではない" >&2; exit 1; }
[ "$(sed -n 2p "${OUT}/gate_constant.txt" | sed 's/^[0-9]*://')" = "const SME_PRODUCTION_ENABLED: bool = true;" ] || { echo "ON の定数が true ではない" >&2; exit 1; }
git -C "${BASE}" rev-parse HEAD > "${OUT}/rev-stamp.txt"
echo "$(git -C "${BASE}" rev-parse HEAD)+on-arm.patch" >> "${OUT}/rev-stamp.txt"
git -C "${BASE}" ls-files | sort | sha | awk '{print $1}' > "${OUT}/tree_fingerprint.txt"
# 登録コミットと計測コミットの間で計測に効くパスが動いていないこと（#3613 §8 の教訓）は、実機セッションが
# RULE の登録基準 SHA と rev-stamp の差分（crates/・scripts/bench/・Cargo.*・rust-toolchain.toml）で確認し README に記録する
echo "P-TREE ok"

# P-R0
: > "${OUT}/sme_report.txt"
for t in off on; do
    P="${WORK}/probe-${t}"
    mkdir -p "${P}/src"
    if [ "${t}" = "off" ]; then TREE="${BASE}"; else TREE="${ON}"; fi
    sed "s#TREE_PLACEHOLDER#${TREE}#" "${BASE}/docs/perf/logs/cpu-gemm-sme-fmopa-1587/gb10/sme-probe-Cargo.toml" > "${P}/Cargo.toml" || exit 1
    cp "${BASE}/docs/perf/logs/cpu-gemm-sme-fmopa-1587/gb10/sme-probe-main.rs" "${P}/src/main.rs" || exit 1
    (cd "${P}" && CARGO_TARGET_DIR="${WORK}/target-probe" cargo run --release -q) > "${WORK}/probe_${t}.log" 2>&1 \
        || { echo "probe ${t} 失敗" >&2; tail -20 "${WORK}/probe_${t}.log" >&2; exit 1; }
    mask < "${WORK}/probe_${t}.log" > "${OUT}/sme_probe_${t}.log"
    echo "${t}: $(grep sme_report= "${OUT}/sme_probe_${t}.log")" >> "${OUT}/sme_report.txt"
done
if ! grep -q "kernel_enabled: ${EXPECT_KERNEL}" "${OUT}/sme_probe_off.log" || ! grep -q "kernel_enabled: ${EXPECT_KERNEL}" "${OUT}/sme_probe_on.log"; then
    echo "P-R0 不成立（kernel_enabled が ${EXPECT_KERNEL} でない）-> 計測中止（undetermined）" >&2
    exit 1
fi
echo "P-R0 ok"

# P-RT（両腕）。ON は定数 true では作りの上で必ず落ちるドリフト検出 1 件だけを計測前に固定した名前で除外する。全文は収録せず要約だけ残す
(cd "${ON}" && CARGO_TARGET_DIR="${WORK}/target-rt-on" cargo test -p fandhe-ai-backend-cpu --release -- --skip "${RT_SKIP_TEST}") > "${WORK}/rt_on.raw.log" 2>&1
echo "rt_on exit=$? skip=${RT_SKIP_TEST}" >> "${OUT}/load_gate.log"
grep -E '^test result|FAILED|panicked' "${WORK}/rt_on.raw.log" | mask > "${OUT}/rt_on.log"
rm -rf "${WORK}/target-rt-on"
(cd "${BASE}" && CARGO_TARGET_DIR="${WORK}/target-rt-off" cargo test -p fandhe-ai-backend-cpu --release) > "${WORK}/rt_off.raw.log" 2>&1
echo "rt_off exit=$?" >> "${OUT}/load_gate.log"
grep -E '^test result|FAILED|panicked' "${WORK}/rt_off.raw.log" | mask > "${OUT}/rt_off.log"
rm -rf "${WORK}/target-rt-off"

# A/B（before = OFF = BASE・after = ON）
(cd "${BASE}/scripts/bench/framework-compare" && \
    AB_HOST=gb10 AB_BEFORE_FACADE_PATH="${BASE}/crates/facade" AB_AFTER_FACADE_PATH="${ON}/crates/facade" \
    AB_TORCH_PYTHON="${TORCH_PY}" bash run_ab_sme_default_on_4156_cpu.sh "${LABEL}") > "${OUT}/run_ab.log" 2>&1
run_ab_rc=$?
echo "run_ab exit=${run_ab_rc}" >> "${OUT}/load_gate.log"

# P-BIT（ON/OFF の ^out[ 行の行数・sha256・cmp。dump 本体は収録しない）
bd_rc=0
for t in off on; do
    if [ "${t}" = "off" ]; then TREE="${BASE}"; else TREE="${ON}"; fi
    (cd "${TREE}" && CARGO_TARGET_DIR="${WORK}/target-bd" cargo test -p fandhe-ai --release --test cpu_sme_gate_bit_dump \
        -- --ignored --nocapture --exact dump_cpu_sme_gate_bits) 2>&1 | grep -E '^out\[' > "${WORK}/bits_${t}.txt"
    [ "${PIPESTATUS[0]}" = "0" ] || bd_rc=1
done
n_off="$(wc -l < "${WORK}/bits_off.txt" | tr -d ' ')"
n_on="$(wc -l < "${WORK}/bits_on.txt" | tr -d ' ')"
{
    echo "off_lines=${n_off} on_lines=${n_on} expected=${EXPECTED_BITDUMP_LINES}"
    echo "off_sha256=$(sha "${WORK}/bits_off.txt" | awk '{print $1}')"
    echo "on_sha256=$(sha "${WORK}/bits_on.txt" | awk '{print $1}')"
} > "${OUT}/bitdump/summary.txt"
cmp "${WORK}/bits_off.txt" "${WORK}/bits_on.txt" > "${OUT}/bitdump/cmp.txt" 2>&1
cmp_rc=$?
if [ "${n_off}" != "${EXPECTED_BITDUMP_LINES}" ] || [ "${n_on}" != "${EXPECTED_BITDUMP_LINES}" ]; then bd_rc=1; fi
echo "bitdump exit=${bd_rc} cmp_exit=${cmp_rc}" >> "${OUT}/load_gate.log"
rm -f "${WORK}/bits_off.txt" "${WORK}/bits_on.txt"
rm -rf "${WORK}/target-bd" "${WORK}/target-probe"

# 収録（マスク）
collect_rc=0
for f in "${OUT}/run_ab.log" "${OUT}/patch_apply.log" "${OUT}/bitdump/cmp.txt"; do
    [ -e "${f}" ] || continue
    if mask < "${f}" > "${f}.masked"; then mv "${f}.masked" "${f}"; else rm -f "${f}.masked"; collect_rc=1; fi
done
mkdir -p "${OUT}/ab" || collect_rc=1
for f in "${BASE}"/scripts/bench/framework-compare/results/raw/*4156*; do
    [ -e "${f}" ] || { collect_rc=1; continue; }
    case "${f}" in
        */bench-fandhe-*) ;;
        *) mask < "${f}" > "${OUT}/ab/$(basename "${f}")" || collect_rc=1 ;;
    esac
done
for f in "${BASE}"/scripts/bench/framework-compare/compare-*4156*; do
    [ -e "${f}" ] && { mask < "${f}" > "${OUT}/ab/$(basename "${f}")" || collect_rc=1; }
done
# 実ホスト名・ユーザー名・IP・絶対ホームの残りを収録前に検査する（0 件でなければ fail-closed）
LEAK_RE="/Users/|/home/|$(hostname 2>/dev/null || echo __nohost__)|$(id -un 2>/dev/null || echo __nouser__)|[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}"
if grep -rEq "${LEAK_RE}" "${OUT}"; then
    echo "マスク漏れの疑い（ホスト名・ユーザー名・絶対パス・IP）-> 収録を中止" >&2
    grep -rEl "${LEAK_RE}" "${OUT}" | mask >&2
    collect_rc=1
fi
echo "collect exit=${collect_rc}" >> "${OUT}/load_gate.log"
echo "released-pending at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${OUT}/hwlock.log"
if [ "${collect_rc}" = "0" ]; then
    cp -R "${OUT}" "${FINAL}" || { echo "収録先への複製に失敗: ${FINAL}" >&2; exit 1; }
fi
rm -rf "${BASE}"/scripts/bench/framework-compare/target-ab-4156-*
[ "${collect_rc}" = "0" ] || { echo "収録に失敗（collect exit=${collect_rc}）。生ログは ${OUT} に残る（公開しないこと）" >&2; exit 1; }
[ "${run_ab_rc}" = "0" ] || { echo "A/B 実行が失敗（run_ab exit=${run_ab_rc}）。ログは収録済み" >&2; exit "${run_ab_rc}"; }
echo "完了"
