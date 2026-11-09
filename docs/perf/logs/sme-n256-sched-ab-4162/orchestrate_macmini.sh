#!/bin/bash
# Mac mini（Apple M4）での N=256 SME スレッド数・分配の 7 腕 A/B（イシュー #4162）。
# 役割: 事前登録規則 RULE.txt（計測前に単独コミット済み）に従い、前提ゲート → 7 腕のビルド → 5 round 計測 →
# マスク収録までを 1 本で行う実行キット。派生元: docs/perf/logs/sme-default-on-ab-4156/orchestrate_macmini.sh
# （Mac mini のゲート・ロック・マスク）と docs/perf/logs/cpu-infer-jpw1-gb10-ab-4161/orchestrate.sh（腕ツリー・path patch）。
# 腕: base（SME ON）・aa（base と同一バイナリの A/A）・t4・t6・jpw1・jpw4・t4-jpw1。判定は aggregate.py。
# 既定の切り替えは行わない（定数は計測専用ツリーでだけ書き換える。main へコミットしない）。
# macOS の bash 3.2 でも動くよう mapfile・連想配列・jq は使わない。
#
# 使い方（BASE ツリーのルートで）:
#   BASE=<クリーンな clone の絶対パス> WORK=<作業ディレクトリ絶対パス> TORCH_PY=<venv-torch/bin/python 絶対パス> \
#     bash docs/perf/logs/sme-n256-sched-ab-4162/orchestrate_macmini.sh --dry-run   # 計画のみ（副作用なし）
#     bash docs/perf/logs/sme-n256-sched-ab-4162/orchestrate_macmini.sh             # 実行（Mac mini 上）
# WORK は同期ツリーの外に置く（target もそこ）。ログは BASE の外（${WORK}/out）へ出し、終了後に
# ${BASE}/${REPO_LOG}/macmini へ収録する。ホスト名・ユーザー名は収録物へ書かない。
set -u
export LC_ALL=C

REPO_LOG="docs/perf/logs/sme-n256-sched-ab-4162"
LOCK_DIR="${HOME}/work/.fandhe-hw-lock"
GATE_FILE="crates/backend-cpu/src/gemm_blis/mod.rs"
SCHED_FILE="crates/backend-cpu/src/gemm_blis/sme_small_sched.rs"
ROUNDS=5
# 腕（固定配列。外部入力からパスやコマンドを組み立てない）。aa は base のバイナリを使う。
ARMS="base aa t4 t6 jpw1 jpw4 t4-jpw1"
TREES="base t4 t6 jpw1 jpw4 t4-jpw1"
# P-RT で落ちるのが作りの上で当然のドリフト検出テスト（計測前に名前を固定）
RT_SKIP_ON="sme_production_disabled_after_gb10_reject"
RT_SKIP_CAND="sme_small_sched_disabled_by_default"
FIRE_TEST="gemm_blis::sme_small_sched::tests::report_sme_small_sched_probe"
BIT_TEST="gemm_blis::sme_small_sched::tests::sme_small_sched_bit_exact_vs_default_when_sme_available"
DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

if command -v sha256sum >/dev/null 2>&1; then
    sha() { sha256sum "$@"; }
else
    sha() { shasum -a 256 "$@"; }
fi

# validate_abs: 絶対パスで、引用符・バックスラッシュ・改行・空白を含まないこと（A03）
validate_abs() { # validate_abs <名前> <値>
    case "$2" in
        /*) ;;
        *) echo "error: $1 は絶対パスで指定すること" >&2; return 1 ;;
    esac
    case "$2" in
        *[\"\'\\\`\$\ ]* | *"
"*) echo "error: $1 に使えない文字（引用符・バックスラッシュ・空白・改行・\$・\`）がある" >&2; return 1 ;;
    esac
    return 0
}

if [ "${DRY}" = "1" ]; then
    echo "# dry-run: 実行計画（副作用なし）"
    echo "0. RULE.txt: プレースホルダ（<未記入>）が残っていないこと、fixed-at-utc が現在より前、各 sha256（patch 6 本・aggregate.py）が実ファイルと一致"
    echo "1. P-LOCK: mkdir ${LOCK_DIR}（60 秒おきに再試行）。取得・解放時刻を hwlock.log へ"
    echo "2. P-TREE: BASE を ${TREES} の 6 ツリーへ clone。on-arm.patch（全ツリー）と arm-<腕>.patch（候補のみ）を git apply --check のうえ適用。差分ファイルが ${GATE_FILE} と ${SCHED_FILE} の期待どおりか。BASE と tree_sha の間で crates/・scripts/bench/・Cargo.*・rust-toolchain.toml に差分がないか。cargo tree で fandhe-ai が path patch の facade に解決するか"
    echo "3. P-R0: 全ツリーで sme_report().kernel_enabled == true（定数 SME_PRODUCTION_ENABLED == true は P-TREE が確認）"
    echo "4. P-FIRE: 各ツリーで ${FIRE_TEST} を --ignored --exact 1 件。256^3 の発火数は候補 >=1・base/aa =0、512^3 と (784,256,64) は全腕 0"
    echo "5. P-RT: base/aa は cargo test -p fandhe-ai-backend-cpu --release -- --skip ${RT_SKIP_ON}。候補は --skip ${RT_SKIP_CAND} も付ける"
    echo "6. P-BIT: 各ツリーで ${BIT_TEST} を実行（SKIP 出力なら undetermined）"
    echo "7. ビルド（逐次・計測前）: 各ツリーの bench-fandhe を path patch の facade で release ビルド"
    echo "8. A/B: ${ROUNDS} round。round ごとに ${ARMS} の起動順をローテーション。load1 < 3.0（30 秒間隔・最大 30 分）。セルは gemm 256/512 x fresh/reuse と train 64 x fresh/reuse。各 round 後に PyTorch（記録のみ）"
    echo "9. 収録: gates.txt を作り、マスクして ${REPO_LOG}/macmini へ複製。ホスト名等の grep が 0 件でなければ停止。ロックを解放"
    exit 0
fi

for v in BASE WORK TORCH_PY; do
    eval "val=\${${v}:-}"
    # shellcheck disable=SC2154
    validate_abs "${v}" "${val}" || exit 1
done
[ -x "${TORCH_PY}" ] || { echo "error: TORCH_PY が実行可能でない" >&2; exit 1; }
if [ -n "${RAYON_NUM_THREADS:-}" ]; then
    echo "error: RAYON_NUM_THREADS が設定されている（専用プールが env_override で無効化される）。unset して再実行" >&2
    exit 1
fi

RULE="${BASE}/${REPO_LOG}/RULE.txt"
[ -f "${RULE}" ] || { echo "error: RULE.txt が無い（RULE.template.txt を埋めて単独コミットしてから実行）" >&2; exit 1; }
if grep -q '<未記入>' "${RULE}"; then echo "error: RULE.txt にプレースホルダ <未記入> が残っている" >&2; exit 1; fi
rule_val() { # rule_val <キー>: "キー: 値" の行をちょうど 1 件だけ許す
    local n
    n="$(grep -c "^$1: " "${RULE}")"
    [ "${n}" = "1" ] || { echo ""; return 0; }
    sed -n "s/^$1: \(.*[^ ]\) *\$/\1/p" "${RULE}"
}
FIXED_AT="$(rule_val fixed-at-utc | sed -n 's/^\([0-9]\{4\}-[0-9]\{2\}-[0-9]\{2\}T[0-9]\{2\}:[0-9]\{2\}:[0-9]\{2\}Z\).*/\1/p')"
[ -n "${FIXED_AT}" ] || { echo "error: fixed-at-utc が欠落または形式不正" >&2; exit 1; }
NOW_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
[ "${FIXED_AT}" \< "${NOW_UTC}" ] || { echo "error: fixed-at-utc が現在時刻以降" >&2; exit 1; }
TREE_SHA="$(rule_val tree_sha)"
case "${TREE_SHA}" in
    ????????????????????????????????????????) ;;
    *) echo "error: RULE の tree_sha が 40 桁でない" >&2; exit 1 ;;
esac
# patch・aggregate.py の sha256 を RULE と照合する（RULE 側は "<ファイル名> sha256: <64 桁>"）
D="${BASE}/${REPO_LOG}"
check_sha() { # check_sha <ファイル名>
    local want got
    want="$(sed -n "s/^$1 sha256: \([0-9a-f]\{64\}\)\$/\1/p" "${RULE}" | head -n1)"
    got="$(sha "${D}/$1" | awk '{print $1}')"
    [ -n "${want}" ] && [ "${want}" = "${got}" ] || { echo "error: $1 の sha256 が RULE 記載と不一致 -> P-TREE 不成立" >&2; return 1; }
}
for f in on-arm.patch aggregate.py arm-t4.patch arm-t6.patch arm-jpw1.patch arm-jpw4.patch arm-t4-jpw1.patch; do
    check_sha "${f}" || exit 1
done

OUT="${WORK}/out"
FINAL="${D}/macmini"
AB="${OUT}/ab"
mkdir -p "${WORK}" "${AB}" || exit 1
if [ -e "${FINAL}" ] || [ -e "${OUT}/gates.txt" ] || [ -e "${WORK}/trees" ]; then
    echo "既存の収録先・ログ・ツリーがあるため停止（上書き禁止。別の WORK で再実行）" >&2
    exit 1
fi
mkdir -p "${WORK}/trees" || exit 1
mask() { sed -e "s#${HOME}#<home>#g" -e "s#${WORK}#<work>#g" -e "s#${BASE}#<base>#g" -e "s#$(hostname 2>/dev/null || echo __nohost__)#<metal-node>#g"; }

# P-LOCK: 解放は trap で必ず行う
until mkdir "${LOCK_DIR}" 2>/dev/null; do sleep 60; done
echo "#4162 $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${LOCK_DIR}/owner"
ACQUIRED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "acquired at=${ACQUIRED_AT}" >> "${OUT}/hwlock.log"
release_lock() {
    echo "released at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${OUT}/hwlock.log"
    rm -rf "${LOCK_DIR}"
}
trap release_lock EXIT
[ "${FIXED_AT}" \< "${ACQUIRED_AT}" ] || { echo "fixed-at-utc がロック取得時刻より前でない" >&2; exit 1; }

GATES="${OUT}/gates.txt"
: > "${GATES}"
gate() { echo "$1 $2 $3" >> "${GATES}"; } # gate <ゲート> <腕|all> <ok|fail|undetermined>
die() { echo "error: $*" >&2; exit 1; }

# P-TREE
[ -z "$(git -C "${BASE}" status --porcelain --untracked-files=no)" ] || die "BASE に未コミット変更がある -> P-TREE 不成立"
[ -z "$(git -C "${BASE}" ls-files --others --exclude-standard)" ] || die "BASE に未追跡ファイルがある -> P-TREE 不成立"
git -C "${BASE}" cat-file -e "${TREE_SHA}^{commit}" 2>/dev/null || die "tree_sha ${TREE_SHA} が BASE に存在しない -> P-TREE 不成立"
if ! git -C "${BASE}" diff --quiet "${TREE_SHA}" HEAD -- crates scripts/bench Cargo.toml Cargo.lock rust-toolchain.toml; then
    die "tree_sha と BASE の間で計測に効くパスに差分がある -> P-TREE 不成立（undetermined）"
fi
git -C "${BASE}" rev-parse HEAD > "${OUT}/rev-stamp.txt"
git -C "${BASE}" ls-files | sort | sha | awk '{print $1}' > "${OUT}/tree_fingerprint.txt"
for t in ${TREES}; do
    T="${WORK}/trees/${t}"
    git clone -q "${BASE}" "${T}" || die "clone 失敗: ${t}"
    (cd "${T}" && git apply --check "${D}/on-arm.patch" && git apply "${D}/on-arm.patch") || die "on-arm.patch 適用失敗: ${t}"
    if [ "${t}" != "base" ]; then
        (cd "${T}" && git apply --check "${D}/arm-${t}.patch" && git apply "${D}/arm-${t}.patch") || die "arm-${t}.patch 適用失敗"
    fi
    names="$(git -C "${T}" diff --name-only HEAD | sort | tr '\n' ' ')"
    if [ "${t}" = "base" ]; then want="${GATE_FILE} "; else want="${GATE_FILE} ${SCHED_FILE} "; fi
    [ "${names}" = "${want}" ] || die "ツリー ${t} の差分ファイルが期待と違う: ${names}"
    [ -z "$(git -C "${T}" ls-files --others --exclude-standard)" ] || die "ツリー ${t} に未追跡ファイルがある"
    grep -q '^const SME_PRODUCTION_ENABLED: bool = true;' "${T}/${GATE_FILE}" || die "ツリー ${t} の SME 定数が true でない"
done
echo "P-TREE ok"
gate P-TREE all ok

# P-R0 / P-FIRE / P-RT / P-BIT（ツリーごと。target は WORK に置く）
: > "${OUT}/gates_detail.log"
for t in ${TREES}; do
    T="${WORK}/trees/${t}"
    P="${WORK}/probe-${t}"
    mkdir -p "${P}/src"
    sed "s#TREE_PLACEHOLDER#${T}#" "${BASE}/docs/perf/logs/cpu-gemm-sme-fmopa-1587/gb10/sme-probe-Cargo.toml" > "${P}/Cargo.toml" || die "probe 準備失敗"
    cp "${BASE}/docs/perf/logs/cpu-gemm-sme-fmopa-1587/gb10/sme-probe-main.rs" "${P}/src/main.rs" || die "probe 準備失敗"
    (cd "${P}" && CARGO_TARGET_DIR="${WORK}/target-probe" cargo run --release -q) > "${WORK}/probe_${t}.log" 2>&1 || { tail -20 "${WORK}/probe_${t}.log" >&2; die "probe ${t} 失敗"; }
    grep -q 'kernel_enabled: true' "${WORK}/probe_${t}.log" || die "P-R0 不成立（${t}: kernel_enabled が true でない）"
    mask < "${WORK}/probe_${t}.log" > "${OUT}/sme_probe_${t}.log"

    LIBT="${WORK}/target-lib-${t}"
    # P-FIRE
    (cd "${T}" && CARGO_TARGET_DIR="${LIBT}" cargo test -p fandhe-ai-backend-cpu --release --lib -- --ignored --exact --nocapture "${FIRE_TEST}") > "${WORK}/fire_${t}.log" 2>&1
    f256="$(sed -n 's/^FIRE256=\([0-9]*\)$/\1/p' "${WORK}/fire_${t}.log" | head -n1)"
    f512="$(sed -n 's/^FIRE512=\([0-9]*\)$/\1/p' "${WORK}/fire_${t}.log" | head -n1)"
    f784="$(sed -n 's/^FIRE784x256x64=\([0-9]*\)$/\1/p' "${WORK}/fire_${t}.log" | head -n1)"
    echo "${t}: FIRE256=${f256:-NA} FIRE512=${f512:-NA} FIRE784x256x64=${f784:-NA}" >> "${OUT}/gates_detail.log"
    fire=fail
    if [ -z "${f256}" ] || [ -z "${f512}" ] || [ -z "${f784}" ]; then
        fire=undetermined
    elif [ "${f512}" = "0" ] && [ "${f784}" = "0" ]; then
        if [ "${t}" = "base" ]; then [ "${f256}" = "0" ] && fire=ok; else [ "${f256}" -ge 1 ] && fire=ok; fi
    fi
    # P-RT
    if [ "${t}" = "base" ]; then
        (cd "${T}" && CARGO_TARGET_DIR="${LIBT}" cargo test -p fandhe-ai-backend-cpu --release -- --skip "${RT_SKIP_ON}") > "${WORK}/rt_${t}.raw.log" 2>&1
    else
        (cd "${T}" && CARGO_TARGET_DIR="${LIBT}" cargo test -p fandhe-ai-backend-cpu --release -- --skip "${RT_SKIP_ON}" --skip "${RT_SKIP_CAND}") > "${WORK}/rt_${t}.raw.log" 2>&1
    fi
    rt_rc=$?
    grep -E '^test result|FAILED|panicked' "${WORK}/rt_${t}.raw.log" | mask > "${OUT}/rt_${t}.log"
    rt=ok; [ "${rt_rc}" = "0" ] || rt=fail
    # P-BIT
    (cd "${T}" && CARGO_TARGET_DIR="${LIBT}" cargo test -p fandhe-ai-backend-cpu --release --lib -- --ignored --exact --nocapture "${BIT_TEST}") > "${WORK}/bit_${t}.log" 2>&1
    bit_rc=$?
    bit=ok
    if grep -q '^SKIP' "${WORK}/bit_${t}.log"; then bit=undetermined; elif [ "${bit_rc}" != "0" ]; then bit=fail; fi
    echo "${t}: fire=${fire} rt=${rt} bit=${bit}" >> "${OUT}/gates_detail.log"
    gate P-FIRE "${t}" "${fire}"; gate P-RT "${t}" "${rt}"; gate P-BIT "${t}" "${bit}"
    if [ "${t}" = "base" ]; then gate P-FIRE aa "${fire}"; gate P-RT aa "${rt}"; gate P-BIT aa "${bit}"; fi
    rm -rf "${LIBT}"
done
rm -rf "${WORK}/target-probe"
gate P-R0 all ok
echo "P-R0 / P-FIRE / P-RT / P-BIT 記録済み"

# ビルド（逐次・計測前）
bench_build() { # bench_build <ツリー名>
    local t=$1 T tdir
    T="${WORK}/trees/$1"
    tdir="${WORK}/target-ab-${t}"
    (cd "${T}/scripts/bench/framework-compare" && cargo build --release -p bench-fandhe --target-dir "${tdir}" \
        --config "patch.crates-io.fandhe-ai.path=\"${T}/crates/facade\"") > "${WORK}/build_${t}.log" 2>&1 \
        || { tail -30 "${WORK}/build_${t}.log" >&2; die "bench-fandhe のビルド失敗: ${t}"; }
    (cd "${T}/scripts/bench/framework-compare" && cargo tree -p bench-fandhe --depth 1 \
        --config "patch.crates-io.fandhe-ai.path=\"${T}/crates/facade\"") 2>&1 | grep -qE 'fandhe-ai v[0-9.]+ \(.*crates/facade\)' \
        || die "fandhe-ai が path patch の facade に解決しない: ${t}"
    [ -x "${tdir}/release/bench-fandhe" ] || die "bench-fandhe の実行ファイルが無い: ${t}"
}
for t in ${TREES}; do bench_build "${t}"; done
echo "ビルド完了"

BENCH_PY="${BASE}/docs/perf/logs/lowlayer-diagnosis-2026-09-12/bench_py.py"
[ -f "${BENCH_PY}" ] || die "bench_py.py が無い"
bin_of() { # bin_of <腕>
    if [ "$1" = "aa" ]; then echo "${WORK}/target-ab-base/release/bench-fandhe"; else echo "${WORK}/target-ab-$1/release/bench-fandhe"; fi
}
for a in ${ARMS}; do : > "${AB}/results-${a}-gemm.jsonl"; : > "${AB}/results-${a}-train.jsonl"; done
: > "${AB}/results-torch-gemm.jsonl"; : > "${AB}/results-torch-train.jsonl"
RUNAB_FAIL=0
run_cell() { # run_cell <腕> <task> <size> <mode>
    local arm=$1 task=$2 size=$3 mode=$4 f n0 n1
    f="${AB}/results-${arm}-${task}.jsonl"
    n0="$(wc -l < "${f}" | tr -d ' ')"
    if ! "$(bin_of "${arm}")" --task "${task}" --size "${size}" --device cpu --mode "${mode}" --out "${f}" 2>>"${OUT}/cell_err.log"; then
        echo "FAILED arm=${arm} ${task} ${size} ${mode}" >> "${OUT}/cell_err.log"; RUNAB_FAIL=$((RUNAB_FAIL + 1)); return 0
    fi
    n1="$(wc -l < "${f}" | tr -d ' ')"
    [ "${n1}" -eq $((n0 + 1)) ] || { echo "ROW MISMATCH arm=${arm} ${task} ${size} ${mode}" >> "${OUT}/cell_err.log"; RUNAB_FAIL=$((RUNAB_FAIL + 1)); }
}
run_arm() { # run_arm <腕>
    local m
    for m in fresh reuse; do run_cell "$1" gemm 256 "${m}"; run_cell "$1" gemm 512 "${m}"; done
    for m in fresh reuse; do run_cell "$1" train 64 "${m}"; done
}
LOAD_GATE=3.0
: > "${OUT}/load_gate.log"
wait_load_gate() { # wait_load_gate <round>
    local waited=0 status=timeout l1 ok gate_ok=0
    while [ "${waited}" -lt 1800 ]; do
        l1="$(sysctl -n vm.loadavg 2>/dev/null | awk '{print $2}')"
        if [ -z "${l1}" ] || echo "${l1}" | grep -q '[^0-9.]'; then status=unavailable; l1=NA; break; fi
        ok="$(awk -v a="${l1}" -v t="${LOAD_GATE}" 'BEGIN{print (a<t)?1:0}')"
        if [ "${ok}" = "1" ]; then status=pass; break; fi
        sleep 30; waited=$((waited + 30))
    done
    [ "${status}" = "pass" ] && gate_ok=1
    echo "round${1} gate=${status} gate_ok=${gate_ok} threshold=${LOAD_GATE} load1=${l1:-NA} waited_s=${waited} at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${OUT}/load_gate.log"
}
TORCH_FAILED=0
torch_cell() { # torch_cell <task> <size>
    "${TORCH_PY}" "${BENCH_PY}" --framework pytorch --task "$1" --device cpu --size "$2" --out "${AB}/results-torch-$1.jsonl" >/dev/null 2>>"${OUT}/torch_err.log" \
        || TORCH_FAILED=$((TORCH_FAILED + 1))
}
set -- ${ARMS}
NARMS=$#
r=1
while [ "${r}" -le "${ROUNDS}" ]; do
    wait_load_gate "${r}"
    # 起動順を round ごとにローテーションする（順序効果の相殺）
    i=0
    while [ "${i}" -lt "${NARMS}" ]; do
        idx=$(( (i + r - 1) % NARMS + 1 ))
        eval "arm=\${${idx}}"
        run_arm "${arm}"
        i=$((i + 1))
    done
    torch_cell gemm 256; torch_cell gemm 512; torch_cell train 64
    r=$((r + 1))
done
if [ "${RUNAB_FAIL}" = "0" ]; then gate P-CELLS-RUNAB all ok; else gate P-CELLS-RUNAB all fail; fi

# 収録（マスク）
collect_rc=0
mkdir -p "${OUT}/masked" || collect_rc=1
for f in "${AB}"/*.jsonl; do
    mask < "${f}" > "${OUT}/masked/$(basename "${f}")" || collect_rc=1
done
for f in "${OUT}/cell_err.log" "${OUT}/torch_err.log"; do
    [ -e "${f}" ] && { mask < "${f}" > "${f}.m" && mv "${f}.m" "${f}"; }
done
rm -rf "${AB}"; mv "${OUT}/masked" "${AB}" || collect_rc=1
LEAK_RE="/Users/|/home/|$(hostname 2>/dev/null || echo __nohost__)|$(id -un 2>/dev/null || echo __nouser__)|[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}"
if grep -rEq "${LEAK_RE}" "${OUT}"; then
    echo "マスク漏れの疑い（ホスト名・ユーザー名・絶対パス・IP）-> 収録を中止" >&2
    grep -rEl "${LEAK_RE}" "${OUT}" | mask >&2
    collect_rc=1
fi
if [ "${collect_rc}" = "0" ]; then gate P-COLLECT all ok; else gate P-COLLECT all fail; fi
echo "released-pending at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${OUT}/hwlock.log"
gate P-LOCK all ok
if [ "${collect_rc}" = "0" ]; then
    cp -R "${OUT}" "${FINAL}" || die "収録先への複製に失敗: ${FINAL}"
fi
[ "${collect_rc}" = "0" ] || die "収録に失敗（生ログは ${OUT} に残る。公開しないこと）"
echo "torch_failed=${TORCH_FAILED}（記録専用）"
echo "完了。python3 -I ${REPO_LOG}/aggregate.py ${REPO_LOG}/macmini > ${REPO_LOG}/aggregate-macmini.md で集計する"
