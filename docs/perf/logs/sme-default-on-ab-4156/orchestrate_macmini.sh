#!/bin/bash
# Mac mini（Apple M4）での SME 既定 ON の A/B（イシュー #4156）。
# 役割: SME `fmopa` カーネル既定 ON の「新しい事前登録 A/B」を Mac mini（Apple M4）で測るオーケストレーション。
# 判定規則は同ディレクトリ RULE.txt（計測前に単独コミット済み）。#3232 の orchestrate_macmini.sh を派生元とし、
# 腕の組み立てを逆にした: OFF = 本スクリプトを置いたツリー（BASE。main の clone・クリーン・定数 false）、
# ON = BASE を clone して on-arm.patch（定数 true の 1 行）を当てた計測専用ツリー。main へコミットしない。
# 追加ゲート: P-OWNER（所有者決定が RULE §9 に転記済みか。未決なら計測を始めない）・
# P-RT（両腕。ON は sme_production_disabled_after_gb10_reject を除外）・P-BIT（両腕 bitdump の一致）。
# 起動順: P-OWNER → ロック → P-TREE → P-R0 → P-RT → mediaanalysisd の確認 → A/B（AB_HOST=macmini）→ P-BIT → マスク収録。
# 他ジョブの前提ゲート（2026-11-11 追加。#4160 の orchestrate_macmini.sh と同じ部品）: FOREIGN_GATE（foreign_job_gate.py の絶対パス。必須）を
#   A/B 本体へ渡し、A/B 本体が各 round の前（他ジョブ CLEAN かつ load1 < 3.0 を同じループで待つ）と後（1 回）に挟む。
#   ゲートの出力は A/B の後に ${OUT}/foreign-gate へ収録し、ゲートの sha256 を ${OUT}/foreign_gate.txt に残す。
# mediaanalysisd の確認（2026-11-11 追加。#4146 の RULE.v2.txt 項 8 (c) と同じ）: A/B の直前に 2 分間隔で 3 回、
#   mediaanalysisd の %CPU が 5 未満であること（プロセスがなければ通過）。不成立なら A/B を始めずに止める。
# この 2 つはゲートを厳しくする方向だけの追加で、判定式・閾値・round 数・ほかの段の手順は変えない。RULE.txt の固定後の
#   追加のため、この版で測る系列は #4156 の RULE.v2 で登録する。
# 既定の切り替えは行わない（定数は計測専用ツリーでのみ true）。
# macOS の bash 3.2 でも動くよう変数は ${VAR} と書く（mapfile・連想配列・jq は使わない）。
#
# 使い方（BASE ツリーのルートで）:
#   BASE=<クリーンな clone の絶対パス> WORK=<作業ディレクトリ絶対パス> TORCH_PY=<venv-torch/bin/python 絶対パス> \
#   FOREIGN_GATE=<foreign_job_gate.py の絶対パス（docs/perf/logs/macmini-foreign-job-gate/。tree の外の kit に置く）> \
#     bash docs/perf/logs/sme-default-on-ab-4156/orchestrate_macmini.sh --dry-run   # 計画のみ表示（副作用なし）
#     bash docs/perf/logs/sme-default-on-ab-4156/orchestrate_macmini.sh             # 実行（Mac mini（Apple M4）上）
# WORK は同期ツリーの外に置く（target もそこへ置く）。ログは BASE の外（${WORK}/macmini）へ出し、
# 終わってから ${BASE}/${REPO_LOG}/macmini へ収録する。ホスト名・ユーザー名は収録物へ書かない。
set -u
export LC_ALL=C

REPO_LOG="docs/perf/logs/sme-default-on-ab-4156"
HOST="macmini"
LABEL="4156-r1"
GATE_FILE="crates/backend-cpu/src/gemm_blis/mod.rs"
LOCK_DIR="${HOME}/work/.fandhe-hw-lock"
EXPECTED_BITDUMP_LINES=6726847
RT_SKIP_TEST="sme_production_disabled_after_gb10_reject"
EXPECT_KERNEL="true"
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

# 収録前のマスク漏れ検査の正規表現（#4156 所有者判断 2026-11-11）。
# ユーザー名の項だけを語境界つきにする。語の文字は英数字・下線・ハイフンとし、前後がそのどれでもないときだけ一致する
# （ハイフンでつながった識別子の一部は別の語として扱う）。ほかの項（/Users/・/home/・
# ホスト名・IPv4）は従来どおり。境界は GNU grep・macOS の grep の両方で動く ERE の文字クラスで書く（\b は使わない）。
# --self-test-leak で、植えた値（ユーザー名・ホーム・ホスト名・IP・絶対パス）が変更後も検出されることを確かめる。
leak_re() { # leak_re <hostname> <username>
    printf '%s' "/Users/|/home/|$1|(^|[^A-Za-z0-9_-])$2([^A-Za-z0-9_-]|\$)|[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}"
}
self_test_leak() { # 値は表示しない（件数と PASS/FAIL だけ）
    local d re n=0 bad=0 host user f
    d="$(mktemp -d "${TMPDIR:-/tmp}/leak-selftest.XXXXXX")" || return 1
    # 組 1 は合成値、組 2 は実行環境の実値（ファイルは一時ディレクトリにだけ置き、終わったら消す）
    for pair in "thost-01.example tuser9" "$(hostname 2>/dev/null || echo __nohost__) $(id -un 2>/dev/null || echo __nouser__)"; do
        host="${pair% *}"; user="${pair#* }"
        re="$(leak_re "${host}" "${user}")"
        # 検出されるべき行（1 ファイル 1 行）
        i=0
        for line in "run by ${user} at 00:00" "${user}" "${user}: done" "owner=${user}" "path ${user}/x" \
            "home /home/someone/work" "home /Users/someone/work" "on ${host} now" "addr 10.1.2.3" "file /Users/x/y.log"; do
            i=$((i + 1)); f="${d}/pos-${i}"; printf '%s\n' "${line}" > "${f}"
            n=$((n + 1)); grep -Eq "${re}" "${f}" || { bad=$((bad + 1)); echo "self-test-leak FAIL: 検出されるべき行 ${i}（通し ${n} 件目）" >&2; }
        done
        # ディレクトリ再帰（本番と同じ grep -rEq）でも検出されること
        n=$((n + 1)); grep -rEq "${re}" "${d}" || { bad=$((bad + 1)); echo "self-test-leak FAIL: grep -r で検出されない" >&2; }
        rm -f "${d}"/pos-*
        # 検出されないべき行: ユーザー名が英数字・下線・ハイフンに隣接し、別の語の一部になっている
        i=0
        for line in "x${user}y" "${user}_z" "z_${user}" "${user}9q" "${user}-z" "z-${user}"; do
            i=$((i + 1)); f="${d}/neg-${i}"; printf '%s\n' "${line}" > "${f}"
            n=$((n + 1)); if grep -Eq "${re}" "${f}"; then
                # 実値の組では、ホスト名の項など別の項が当たることがある。合成値の組だけを判定に使う
                if [ "${host}" = "thost-01.example" ]; then bad=$((bad + 1)); echo "self-test-leak FAIL: 検出されないべき行 ${i}" >&2; fi
            fi
        done
        rm -f "${d}"/neg-*
    done
    rm -rf "${d}"
    if [ "${bad}" = "0" ]; then echo "self-test-leak ok (${n} 件)"; return 0; fi
    echo "self-test-leak NG (${bad}/${n} 件)" >&2; return 1
}
if [ "${1:-}" = "--self-test-leak" ]; then
    self_test_leak; exit $?
fi

# mediaanalysisd の確認（#4146 の RULE.v2.txt 項 8 (c) と同じ基準）。2 分間隔で 3 回、%CPU が 5 未満であること。
# 名前は ps の comm の basename の完全一致で見る（部分一致にしない）。複数あれば最大値。プロセスがなければ absent で通過。
# ps の失敗・出力が空・%CPU が数値でないときは不成立（fail-closed）。各回の結果を <出力ファイル> に 1 行ずつ足す。
check_mediaanalysisd() { # check_mediaanalysisd <出力ファイル> -> 0=成立 1=不成立
    local out=$1 i v ps_out
    for i in 1 2 3; do
        ps_out="$(ps -A -o pcpu=,comm= 2>/dev/null)" || { echo "mediaanalysisd check${i} at=$(date -u +%Y-%m-%dT%H:%M:%SZ) ps-failed" >> "${out}"; return 1; }
        [ -n "${ps_out}" ] || { echo "mediaanalysisd check${i} at=$(date -u +%Y-%m-%dT%H:%M:%SZ) ps-empty" >> "${out}"; return 1; }
        v="$(printf '%s\n' "${ps_out}" | awk '{
            c = $0; sub(/^[ \t]*[^ \t]+[ \t]+/, "", c); n = split(c, a, "/")
            if (a[n] == "mediaanalysisd") {
                if ($1 !~ /^[0-9]+(\.[0-9]+)?$/) bad = 1
                else if (!f || $1 + 0 > m + 0) m = $1
                f = 1
            }
        } END { if (bad) print "not-numeric"; else if (f) print m; else print "absent" }')"
        echo "mediaanalysisd check${i} at=$(date -u +%Y-%m-%dT%H:%M:%SZ) pcpu=${v}" >> "${out}"
        case "${v}" in
            absent) ;;
            '' | not-numeric) return 1 ;;
            *) awk -v x="${v}" 'BEGIN { exit !(x < 5) }' || return 1 ;;
        esac
        [ "${i}" -lt 3 ] && sleep 120
    done
    return 0
}

if [ "${DRY}" = "1" ]; then
    echo "# dry-run: 実行計画（副作用なし）"
    echo "0. P-OWNER: ${REPO_LOG}/RULE.txt §9 の GB10-MODE・NOISE-BAND・UNDETERMINED-HANDLING と各回答コメント URL が記入済みであること（未決ならロック取得前に停止）。prereg_append.txt の rule_sha256 が RULE.txt と一致すること"
    if [ -n "${BASE:-}" ] && [ -f "${BASE}/${REPO_LOG}/RULE.txt" ]; then
        if check_owner "${BASE}/${REPO_LOG}/RULE.txt"; then echo "   (現在の RULE.txt: P-OWNER 成立)"; else echo "   (現在の RULE.txt: P-OWNER 不成立。実行モードは計測を始めない)"; fi
    fi
    echo "1. ロック: mkdir ${LOCK_DIR}（60 秒おきに再試行）。owner に '#4156 <UTC>' を書き hwlock.log へ取得・解放時刻を残す"

    echo "2. P-TREE: BASE を clone して on-arm.patch を適用（sha256 は RULE.txt の値と照合・git apply --check が先）。差分が ${GATE_FILE} の 1 件だけで OFF=false・ON=true"
    echo "3. P-R0: 両腕の sme_report() が kernel_enabled: ${EXPECT_KERNEL} であること（外れたら中止）"
    echo "4. P-RT: ON は cargo test -p fandhe-ai-backend-cpu --release -- --skip ${RT_SKIP_TEST}、OFF はスキップなしの同コマンド"
    echo "4b. mediaanalysisd の確認: 2 分間隔で 3 回 %CPU 5 未満（basename の完全一致・不在は通過・ps の失敗は不成立）。不成立なら A/B を始めずに止める"
    echo "5. A/B: scripts/bench/framework-compare/run_ab_sme_default_on_4156_cpu.sh ${LABEL}（AB_HOST=${HOST}・5 round）"
    echo "   他ジョブの前提ゲート: FOREIGN_GATE 必須。各 round の前は他ジョブ CLEAN かつ load1<3.0 を同じループで待ち（30 秒間隔・最大 60 回）、round の後に 1 回確認（FOREIGN は gate=invalid reason=foreign-after → A/B は非 0 終了）"
    echo "6. P-BIT: cpu_sme_gate_bit_dump の ^out[ 行を ON/OFF で比較（期待 ${EXPECTED_BITDUMP_LINES} 行・sha256 一致・cmp 差分 0）"
    echo "7. 収録: マスクして ${REPO_LOG}/${HOST} へ複製。実ホスト名等の grep が 0 件でなければ停止。target-ab-* を削除しロックを解放"
    exit 0
fi

for v in BASE WORK TORCH_PY FOREIGN_GATE; do
    eval "val=\${${v}:-}"
    case "${val}" in
        /*) ;;
        *) echo "error: ${v} は絶対パスで指定すること" >&2; exit 1 ;;
    esac
done
case "${FOREIGN_GATE}" in
    *\"* | *\'* | *\\* | *\$* | *\`* | *' '*) echo "error: FOREIGN_GATE に禁止文字を含む" >&2; exit 1 ;;
esac
[ -f "${FOREIGN_GATE}" ] || { echo "error: FOREIGN_GATE のファイルが無い" >&2; exit 1; }
if [ -n "${RAYON_NUM_THREADS:-}" ]; then
    echo "error: RAYON_NUM_THREADS が設定されている。unset して再実行すること" >&2
    exit 1
fi

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
if [ -e "${FINAL}" ] || [ -e "${ON}" ] || [ -e "${OUT}/load_gate.log" ] || [ -e "${OUT}/mediaanalysisd.txt" ] || [ -e "${OUT}/foreign-gate" ]; then
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
echo "foreign_job_gate_sha256=$(sha "${FOREIGN_GATE}" | awk '{print $1}')" > "${OUT}/foreign_gate.txt"

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

# mediaanalysisd の確認（A/B の直前。不成立なら A/B を始めない。系列は始まっていない）
if check_mediaanalysisd "${OUT}/mediaanalysisd.txt"; then
    echo "mediaanalysisd exit=0" >> "${OUT}/load_gate.log"
else
    echo "mediaanalysisd exit=1" >> "${OUT}/load_gate.log"
    echo "mediaanalysisd が止まっていない（または確認できない）-> A/B を始めない。記録は ${OUT}/mediaanalysisd.txt（公開しないこと）" >&2
    exit 1
fi

# A/B（before = OFF = BASE・after = ON）。各 round を他ジョブの前提ゲートで前後から挟むのは A/B 本体（FOREIGN_GATE を渡す）
(cd "${BASE}/scripts/bench/framework-compare" && \
    FOREIGN_GATE="${FOREIGN_GATE}" AB_HOST=macmini AB_BEFORE_FACADE_PATH="${BASE}/crates/facade" AB_AFTER_FACADE_PATH="${ON}/crates/facade" \
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
# 他ジョブの前提ゲートの出力（A/B 本体が results/raw/foreign-gate/4156-sme-cpu-<LABEL>/ に残す。ゲート自身がマスク済みだが念のため mask を通す）
FG_SRC="${BASE}/scripts/bench/framework-compare/results/raw/foreign-gate/4156-sme-cpu-${LABEL}"
if [ -d "${FG_SRC}" ] && mkdir "${OUT}/foreign-gate"; then
    for f in "${FG_SRC}"/*; do
        [ -f "${f}" ] || { collect_rc=1; continue; }
        mask < "${f}" > "${OUT}/foreign-gate/$(basename "${f}")" || collect_rc=1
    done
else
    collect_rc=1
fi
# 実ホスト名・ユーザー名・IP・絶対ホームの残りを収録前に検査する（0 件でなければ fail-closed）
LEAK_RE="$(leak_re "$(hostname 2>/dev/null || echo __nohost__)" "$(id -un 2>/dev/null || echo __nouser__)")"
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
