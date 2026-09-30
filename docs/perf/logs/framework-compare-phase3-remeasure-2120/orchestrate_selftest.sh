#!/usr/bin/env bash
# orchestrate_selftest.sh — 計測ツリーの出典照合（tree_provenance.sh・orchestrate.sh の計測前停止）の self-test（イシュー #2120・PR #2466 レビュー）。
#
# 役割: 一時 git リポジトリで「.rev-stamp と git rev-parse HEAD の不一致」「作業ツリーの変更（dirty）」等を作り、
#   verify_tree_provenance が停止（非 0）すること・計測手順が意図的に加える変更（ツリールートの未追跡 .rev-stamp・
#   docs/ 配下の LOGD の未追跡ファイル・submodule）だけを許すことを確かめる。さらに orchestrate.sh 本体を cargo のスタブ付きで
#   起動し、照合に失敗したとき cargo（ビルド）・switches.sh を呼ぶ前に非 0 で停止することを確かめる（E2E）。
#   腕 B の規定 commit（650359799aa5…）は一時リポジトリでは作れないため、完全一致の照合は単体ケース（期待 commit を
#   一時リポジトリの HEAD にして一致・不一致の両方）で確かめ、本番の定数を上書きする抜け道は設けない。
# 使い方: bash orchestrate_selftest.sh   （一時ファイルは ${TMPDIR:-/tmp} 配下に作り、終了時に消す。成功で 0）
# 依存: bash（3.2 可）・git・coreutils 相当（mktemp・grep）。ネットワーク・cargo は使わない。
set -uo pipefail

SELF_DIR="$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd -P)" || exit 1
# shellcheck source=tree_provenance.sh
source "${SELF_DIR}/tree_provenance.sh"

T="$(mktemp -d "${TMPDIR:-/tmp}/prov-selftest.XXXXXX")" || exit 1
T="$(CDPATH='' cd -- "${T}" && pwd -P)" || exit 1
trap 'rm -rf "${T}"' EXIT
export GIT_CONFIG_NOSYSTEM=1 HOME="${T}/home"
mkdir -p "${HOME}"
gitc() { git -c user.name=selftest -c user.email=selftest@example.invalid -c core.hooksPath=/dev/null \
  -c commit.gpgsign=false -c init.defaultBranch=main -c protocol.file.allow=always "$@"; }

N=0; FAILS=0
pass() { N=$((N + 1)); echo "ok  $1"; }
fail() { N=$((N + 1)); FAILS=$((FAILS + 1)); echo "NG  $1: $2" >&2; }

# mkrepo <dir>: crates/・docs/ を持つ 1 コミットのリポジトリを作り、ルートに .rev-stamp（短縮 sha）を書く
mkrepo() {
  local d=$1
  mkdir -p "${d}/crates" "${d}/docs/perf/logs"
  gitc init -q "${d}" || return 1
  echo 'fn main() {}' > "${d}/crates/x.rs"
  echo 'doc' > "${d}/docs/a.md"
  echo 'kept' > "${d}/docs/perf/logs/tracked.txt"
  gitc -C "${d}" add -A && gitc -C "${d}" commit -q -m init || return 1
  git -C "${d}" rev-parse --short HEAD > "${d}/.rev-stamp"
}
head_of() { git -C "$1" rev-parse HEAD; }
stamp_of() { cat "$1/.rev-stamp"; }

# expect_ok <名前> <tree> <stamp> <expect|-> <logd>
expect_ok() {
  local name=$1 err
  if err="$( (verify_tree_provenance X "$2" "$3" "$4" "$5" && [[ "${PROV_HEAD}" == "$(head_of "$2")" ]]) 2>&1 >/dev/null)"; then
    pass "${name}"
  else
    fail "${name}" "照合が通らない: ${err}"
  fi
}
# expect_ng <名前> <期待する理由の断片> <tree> <stamp> <expect|-> <logd>
expect_ng() {
  local name=$1 hint=$2 err
  if err="$( (verify_tree_provenance X "$3" "$4" "$5" "$6") 2>&1 >/dev/null)"; then
    fail "${name}" "停止しない"
  elif [[ "${err}" != *"${hint}"* ]]; then
    fail "${name}" "想定した理由で停止していない: ${err}"
  else
    pass "${name}"
  fi
}

OUTSIDE="${T}/logd-outside"; mkdir -p "${OUTSIDE}"

# --- 単体: 正常系（計測手順が意図的に加える変更だけがある状態） ---
R="${T}/ok"; mkrepo "${R}" || exit 1
expect_ok 'U-正常（短縮 stamp・未追跡 .rev-stamp のみ）' "${R}" "$(stamp_of "${R}")" - "${OUTSIDE}"
expect_ok 'U-正常（40 桁 stamp）' "${R}" "$(head_of "${R}")" - "${OUTSIDE}"
expect_ok 'U-正常（期待 commit と完全一致）' "${R}" "$(stamp_of "${R}")" "$(head_of "${R}")" "${OUTSIDE}"
mkdir -p "${R}/docs/perf/logs/gb10/run1"; echo x > "${R}/docs/perf/logs/gb10/run1/A.jsonl"; echo x > "${R}/docs/perf/logs/gb10/gate.log"
expect_ok 'U-正常（LOGD がツリーの docs/ 配下・その未追跡ファイル）' "${R}" "$(stamp_of "${R}")" - "${R}/docs/perf/logs/gb10"

# --- 単体: stamp と HEAD の照合 ---
R="${T}/stamp"; mkrepo "${R}" || exit 1
expect_ng 'U-stamp と HEAD が一致しない' '一致しない' "${R}" 'deadbeef' - "${OUTSIDE}"
expect_ng 'U-stamp が 16 進でない' '16 進' "${R}" 'main' - "${OUTSIDE}"
expect_ng 'U-stamp が 6 桁（短すぎる）' '16 進' "${R}" "$(stamp_of "${R}" | cut -c1-6)" - "${OUTSIDE}"
expect_ng 'U-期待 commit と不一致（腕 B の完全一致照合）' '規定の commit' "${R}" "$(stamp_of "${R}")" 650359799aa5e8e6d493bf3a38d6e59732062863 "${OUTSIDE}"
OLD_STAMP="$(stamp_of "${R}")"; OLD_HEAD="$(head_of "${R}")"
echo 'fn main() { }' > "${R}/crates/x.rs"; gitc -C "${R}" commit -q -am move
expect_ng 'U-古い stamp を残したまま HEAD が進んだ' '一致しない' "${R}" "${OLD_STAMP}" - "${OUTSIDE}"
expect_ng 'U-計測中に HEAD が進んだ（全 run 後の再照合）' '一致しない' "${R}" "${OLD_STAMP}" "${OLD_HEAD}" "${OUTSIDE}"

# --- 単体: 作業ツリーの変更（dirty） ---
R="${T}/dirty-mod"; mkrepo "${R}" || exit 1; echo 'changed' >> "${R}/crates/x.rs"
expect_ng 'U-追跡ファイルの変更' '作業ツリーに変更がある' "${R}" "$(stamp_of "${R}")" - "${OUTSIDE}"
R="${T}/dirty-staged"; mkrepo "${R}" || exit 1; echo 'n' > "${R}/crates/new.rs"; gitc -C "${R}" add crates/new.rs
expect_ng 'U-ステージ済みの追加' '作業ツリーに変更がある' "${R}" "$(stamp_of "${R}")" - "${OUTSIDE}"
R="${T}/dirty-untracked"; mkrepo "${R}" || exit 1; echo 'n' > "${R}/crates/extra.rs"
expect_ng 'U-未追跡ファイル（.rev-stamp 以外）' '作業ツリーに変更がある' "${R}" "$(stamp_of "${R}")" - "${OUTSIDE}"
R="${T}/dirty-substamp"; mkrepo "${R}" || exit 1; echo 'x' > "${R}/docs/.rev-stamp"
expect_ng 'U-ルート以外の .rev-stamp は例外にしない' '作業ツリーに変更がある' "${R}" "$(stamp_of "${R}")" - "${OUTSIDE}"
R="${T}/dirty-deleted"; mkrepo "${R}" || exit 1; rm "${R}/docs/a.md"
expect_ng 'U-追跡ファイルの削除' '作業ツリーに変更がある' "${R}" "$(stamp_of "${R}")" - "${OUTSIDE}"
R="${T}/dirty-assume"; mkrepo "${R}" || exit 1; git -C "${R}" update-index --assume-unchanged crates/x.rs; echo 'hidden' >> "${R}/crates/x.rs"
expect_ng 'U-assume-unchanged で隠した変更' 'assume-unchanged' "${R}" "$(stamp_of "${R}")" - "${OUTSIDE}"
R="${T}/dirty-skip"; mkrepo "${R}" || exit 1; git -C "${R}" update-index --skip-worktree crates/x.rs; echo 'hidden' >> "${R}/crates/x.rs"
expect_ng 'U-skip-worktree で隠した変更' 'skip-worktree' "${R}" "$(stamp_of "${R}")" - "${OUTSIDE}"

# --- 単体: LOGD をツリー内に置く場合の範囲 ---
R="${T}/logd"; mkrepo "${R}" || exit 1
mkdir -p "${R}/crates/out"; echo x > "${R}/crates/out/A.jsonl"
expect_ng 'U-LOGD がツリー内の docs/ 外' 'docs/ 配下ではない' "${R}" "$(stamp_of "${R}")" - "${R}/crates/out"
rm -rf "${R}/crates/out"
expect_ng 'U-LOGD がツリーのルート' 'ルートそのもの' "${R}" "$(stamp_of "${R}")" - "${R}"
echo 'edited' >> "${R}/docs/perf/logs/tracked.txt"
expect_ng 'U-LOGD 配下の追跡ファイルの変更は例外にしない' '作業ツリーに変更がある' "${R}" "$(stamp_of "${R}")" - "${R}/docs/perf/logs"

# --- 単体: git 作業ツリーでない・ルートでない ---
R="${T}/plain"; mkdir -p "${R}/crates"; echo 'deadbeef' > "${R}/.rev-stamp"
expect_ng 'U-git 作業ツリーでない（.git なしで転送）' 'git 作業ツリーではない' "${R}" 'deadbeef' - "${OUTSIDE}"
R="${T}/outer"; mkrepo "${R}" || exit 1; mkdir -p "${R}/docs/inner"
expect_ng 'U-別リポジトリのサブディレクトリ（HEAD の借用）' 'ルートではない' "${R}/docs/inner" "$(stamp_of "${R}")" - "${OUTSIDE}"

# --- 単体: submodule（docs/spec 相当）はビルドに使わないため照合から除外する ---
S="${T}/sub-src"; mkrepo "${S}" || exit 1
R="${T}/withsub"; mkrepo "${R}" || exit 1
if gitc -C "${R}" submodule add -q "${S}" docs/spec >/dev/null 2>&1 && gitc -C "${R}" commit -q -m sub; then
  git -C "${R}" rev-parse --short HEAD > "${R}/.rev-stamp"
  echo 'moved' >> "${R}/docs/spec/docs/a.md"; gitc -C "${R}/docs/spec" commit -q -am moved
  expect_ok 'U-submodule の差分は照合から除外' "${R}" "$(stamp_of "${R}")" - "${OUTSIDE}"
else
  fail 'U-submodule の差分は照合から除外' 'submodule を作れない'
fi

# --- E2E: orchestrate.sh 本体が照合失敗でビルド・スイッチ抽出の前に停止する ---
STUB="${T}/stub"; mkdir -p "${STUB}"
MARK="${T}/cargo-called"
printf '#!/usr/bin/env bash\necho called >> "%s"\nexit 1\n' "${MARK}" > "${STUB}/cargo"; chmod +x "${STUB}/cargo"
# e2e <名前> <期待する理由の断片> <HEAD_TREE> <PRE_TREE>
e2e() {
  local name=$1 hint=$2 logd rc err
  logd="${T}/e2e-logd-${N}"
  rm -f "${MARK}"
  err="$(env -i PATH="${STUB}:${PATH}" HOME="${HOME}" GIT_CONFIG_NOSYSTEM=1 TMPDIR="${T}" \
    HEAD_TREE="$3" PRE_TREE="$4" LOGD="${logd}" bash "${SELF_DIR}/orchestrate.sh" gb10 2>&1 >/dev/null)"; rc=$?
  if [[ "${rc}" -eq 0 ]]; then fail "${name}" "orchestrate.sh が停止しない"
  elif [[ "${err}" != *"${hint}"* ]]; then fail "${name}" "想定した理由で停止していない: ${err}"
  elif [[ -e "${MARK}" ]]; then fail "${name}" "照合失敗後に cargo（ビルド）が呼ばれた"
  elif [[ -e "${logd}/switches-B.txt" || -e "${logd}/switches-C.txt" || -e "${logd}/build.log" || -e "${logd}/env_info.txt" ]]; then
    fail "${name}" "照合失敗後に計測の出力が作られた: $(ls "${logd}")"
  else pass "${name}"; fi
}
PRE="${T}/e2e-pre"; mkrepo "${PRE}" || exit 1; echo 65035979 > "${PRE}/.rev-stamp"
HT="${T}/e2e-head-ok"; mkrepo "${HT}" || exit 1
e2e 'E2E-腕 B の stamp（65035979）とツリーの HEAD が一致しない → 停止' '腕 B: .rev-stamp（65035979）と' "${HT}" "${PRE}"
HT="${T}/e2e-head-stale"; mkrepo "${HT}" || exit 1; echo 'deadbeef' > "${HT}/.rev-stamp"
e2e 'E2E-腕 C の stamp と HEAD が一致しない → 停止' '腕 C: .rev-stamp（deadbeef）と' "${HT}" "${PRE}"
HT="${T}/e2e-head-dirty"; mkrepo "${HT}" || exit 1; echo 'changed' >> "${HT}/crates/x.rs"
e2e 'E2E-腕 C の作業ツリーが dirty → 停止' '腕 C: 作業ツリーに変更がある' "${HT}" "${PRE}"
HT="${T}/e2e-head-plain"; mkdir -p "${HT}"; echo 'deadbeef' > "${HT}/.rev-stamp"
e2e 'E2E-腕 C が git 作業ツリーでない → 停止' '腕 C: ' "${HT}" "${PRE}"

echo "orchestrate self-test: ${N} ケース中 NG ${FAILS}"
[[ "${FAILS}" -eq 0 ]]
