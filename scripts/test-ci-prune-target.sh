#!/usr/bin/env bash
# scripts/ci-prune-target.sh の回帰テスト（旧構造・新構造・想定外構造の fixture）。
# usage: scripts/test-ci-prune-target.sh（引数なし。cargo と jq が必要。ネットワーク不要）
#
# 一時ディレクトリに 2 メンバー（memb-a・memb-b）の最小 workspace を作り、本物の
# ci-prune-target.sh と lib.sh を複製して実行する（lib.sh が自身の親をルートへ cd するため、
# リポジトリの target には触れない）。fixture の target は次の構造を模す:
#   old: Cargo 1.99 以前（.fingerprint/・deps/・build/<package>-<hex>）
#   new: Cargo 1.100 以降（build/<package>/<hex>/{fingerprint,out}）
#   unknown: どちらでもない（警告が出て rc 0 であること）
# 検証: メンバーの成果物だけが消え、crates.io 依存（dep-x）の成果物が残ること。
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./lib.sh
source "${DIR}/lib.sh"
reject_args "$(basename "$0")" "$@"

fail=0
check() { # check <説明> <条件コマンド...>
  local desc="$1"
  shift
  if "$@"; then echo "OK: ${desc}"; else
    echo "NG: ${desc}" >&2
    fail=1
  fi
}
# 複数パスをまとめて判定する（`check ... && check ...` のように連結すると失敗を取りこぼす）。
exist_all() {
  local x
  for x in "$@"; do [ -e "${x}" ] || return 1; done
}
absent_all() {
  local x
  for x in "$@"; do [ ! -e "${x}" ] || return 1; done
}
contains() { grep -qF -- "$2" "$1"; }
no_warning() { ! grep -qF -- "::warning::" "$1"; }
rc_is() { [ "${rc}" -eq "$1" ]; }

H1=0123456789abcdef
H2=fedcba9876543210
H3=aaaaaaaaaaaaaaaa

# make_ws <dir>: 2 メンバーの workspace（依存なし）と prune スクリプトの複製を作る。
make_ws() {
  local w="$1" m
  mkdir -p "${w}/scripts"
  cp "${DIR}/ci-prune-target.sh" "${DIR}/lib.sh" "${w}/scripts/"
  printf '[workspace]\nresolver = "2"\nmembers = ["memb-a", "memb-b"]\n' >"${w}/Cargo.toml"
  for m in memb-a memb-b; do
    mkdir -p "${w}/${m}/src"
    printf '[package]\nname = "%s"\nversion = "0.1.0"\nedition = "2021"\n' "${m}" >"${w}/${m}/Cargo.toml"
    : >"${w}/${m}/src/lib.rs"
  done
  (cd "${w}" && cargo generate-lockfile --offline >/dev/null 2>&1)
}

# run_prune <ws> <log>: 実行して終了コードを rc に入れる。
run_prune() {
  rc=0
  (cd "$1" && bash scripts/ci-prune-target.sh) >"$2" 2>&1 || rc=$?
}

tmp="$(mktemp -d)"
trap 'rm -rf -- "${tmp}"' EXIT

# --- 新構造（Cargo 1.100） ---
w="${tmp}/new"
make_ws "${w}"
t="${w}/target/debug"
mkdir -p "${t}/build/memb-a/${H1}/out" "${t}/build/memb-a/${H2}/fingerprint" \
  "${t}/build/memb-b/${H1}/out" "${t}/build/dep-x/${H1}/out" "${t}/build/dep-x/${H2}/fingerprint" \
  "${t}/examples" "${t}/incremental/x" "${w}/target/tmp" "${w}/target/doc"
: >"${t}/build/dep-x/${H1}/out/libdep_x-${H1}.rlib"
: >"${t}/build/memb-a/${H1}/out/memb_a-${H1}"
: >"${t}/memb-a"
: >"${t}/.cargo-lock"
run_prune "${w}" "${tmp}/new.log"
check "新構造: rc 0" rc_is 0
check "新構造: メンバー memb-a が消える" absent_all "${t}/build/memb-a"
check "新構造: メンバー memb-b が消える" absent_all "${t}/build/memb-b"
check "新構造: 依存 dep-x の成果物が残る" exist_all "${t}/build/dep-x/${H1}/out/libdep_x-${H1}.rlib"
check "新構造: 依存 dep-x の fingerprint が残る" exist_all "${t}/build/dep-x/${H2}/fingerprint"
check "新構造: uplift 成果物が消える" absent_all "${t}/memb-a"
check "新構造: ドットファイルが残る" exist_all "${t}/.cargo-lock"
check "新構造: examples・incremental・tmp・doc が消える" absent_all "${t}/examples" "${t}/incremental" "${w}/target/tmp" "${w}/target/doc"
check "新構造: 警告が出ない" no_warning "${tmp}/new.log"

# --- 旧構造（Cargo 1.99 以前） ---
w="${tmp}/old"
make_ws "${w}"
t="${w}/target/debug"
mkdir -p "${t}/.fingerprint/memb-a-${H1}" "${t}/.fingerprint/dep-x-${H2}" "${t}/.fingerprint/cast-${H3}" \
  "${t}/build/memb-a-${H1}" "${t}/build/dep-x-${H2}" "${t}/deps" "${t}/examples" "${t}/incremental"
: >"${t}/deps/libmemb_a-${H1}.rlib"
: >"${t}/deps/libmemb_a-${H1}.rmeta"
: >"${t}/deps/libdep_x-${H2}.rlib"
# 統合テスト名 cast が依存 crate cast と同名: hex で判定するので依存側（H3）は残る。
: >"${t}/deps/cast-${H1}"
: >"${t}/deps/libcast-${H3}.rlib"
: >"${t}/memb-a"
run_prune "${w}" "${tmp}/old.log"
check "旧構造: rc 0" rc_is 0
check "旧構造: メンバーの fingerprint・build が消える" absent_all "${t}/.fingerprint/memb-a-${H1}" "${t}/build/memb-a-${H1}"
check "旧構造: メンバーの deps が消える" absent_all "${t}/deps/libmemb_a-${H1}.rlib" "${t}/deps/libmemb_a-${H1}.rmeta"
check "旧構造: 同名テスト cast（メンバーの hex）が消える" absent_all "${t}/deps/cast-${H1}"
check "旧構造: 依存の fingerprint・deps が残る" exist_all "${t}/.fingerprint/dep-x-${H2}" "${t}/deps/libdep_x-${H2}.rlib" "${t}/build/dep-x-${H2}"
check "旧構造: 同名の依存 cast（別 hex）が残る" exist_all "${t}/deps/libcast-${H3}.rlib" "${t}/.fingerprint/cast-${H3}"
check "旧構造: uplift 成果物・examples・incremental が消える" absent_all "${t}/memb-a" "${t}/examples" "${t}/incremental"
check "旧構造: 警告が出ない" no_warning "${tmp}/old.log"

# --- 想定外構造: 警告を出し、落ちず、何も消さない ---
w="${tmp}/unknown"
make_ws "${w}"
t="${w}/target/debug"
mkdir -p "${t}/somewhere/else"
: >"${t}/somewhere/else/artifact"
run_prune "${w}" "${tmp}/unknown.log"
check "想定外構造: rc 0" rc_is 0
check "想定外構造: ::warning:: を出す" contains "${tmp}/unknown.log" "::warning::debug:"
check "想定外構造: 未知のディレクトリは消さない" exist_all "${t}/somewhere/else/artifact"

# --- メンバーのユニットが無い新構造: 警告 ---
w="${tmp}/nomember"
make_ws "${w}"
t="${w}/target/debug"
mkdir -p "${t}/build/dep-x/${H1}/out"
run_prune "${w}" "${tmp}/nomember.log"
check "メンバー 0 件: rc 0 と警告" contains "${tmp}/nomember.log" "::warning::debug:"
check "メンバー 0 件: 依存が残る" exist_all "${t}/build/dep-x/${H1}/out"

# --- メンバーと同名の依存が lock にある: fail-closed ---
w="${tmp}/dup"
make_ws "${w}"
printf '\n[[package]]\nname = "memb-a"\nversion = "9.9.9"\nsource = "registry+https://github.com/rust-lang/crates.io-index"\n' >>"${w}/Cargo.lock"
mkdir -p "${w}/target/debug/build/memb-a/${H1}/out"
run_prune "${w}" "${tmp}/dup.log"
check "同名依存: 中止（rc 1）" rc_is 1
check "同名依存: 何も消さない" exist_all "${w}/target/debug/build/memb-a/${H1}/out"

# --- build/<package> が symlink: 辿らず消さない ---
w="${tmp}/symlink"
make_ws "${w}"
t="${w}/target/debug"
mkdir -p "${t}/build/dep-x/${H1}/out" "${tmp}/outside/${H1}"
: >"${tmp}/outside/keep"
ln -s "${tmp}/outside" "${t}/build/memb-a"
run_prune "${w}" "${tmp}/symlink.log"
check "symlink: 外部の実体が無傷" exist_all "${tmp}/outside/keep"

# --- 新構造で build/<package> が symlink: fail-closed（rc 非 0・何も消さない） ---
# 走査用の部分シェル（`$(...) || true`）の中で die しても主シェルが続行しないこと。
w="${tmp}/symlink-new"
make_ws "${w}"
t="${w}/target/debug"
mkdir -p "${t}/build/memb-b/${H1}/out" "${t}/build/dep-x/${H1}/out" "${tmp}/outside2/${H1}"
: >"${tmp}/outside2/keep"
: >"${t}/memb-b"
ln -s "${tmp}/outside2" "${t}/build/memb-a"
run_prune "${w}" "${tmp}/symlink-new.log"
check "新構造 symlink: 中止（rc 非 0）" [ "${rc}" -ne 0 ]
check "新構造 symlink: 中止理由を出す" contains "${tmp}/symlink-new.log" "symlink のため prune を中止します"
check "新構造 symlink: 外部の実体が無傷" exist_all "${tmp}/outside2/keep"
check "新構造 symlink: 何も消さない" exist_all "${t}/build/memb-b/${H1}/out" "${t}/build/dep-x/${H1}/out" "${t}/memb-b" "${t}/build/memb-a"

[ "${fail}" -eq 0 ] || die "ci-prune-target.sh の回帰テストに失敗しました"
echo "OK: ci-prune-target.sh の回帰テストすべて成功"
