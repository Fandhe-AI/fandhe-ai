#!/usr/bin/env bash
# ci.yml の target キャッシュ保存前に、workspace メンバーの成果物を target/ から削除する
# （CI 専用。`ci-detect.sh` と同じく Makefile からは呼ばない）。
#
# 目的: target キャッシュのキーはソースを含まず（toolchain 版数 + Cargo.lock）、保存は
# キー未保存時の 1 回だけのため、メンバー crate の成果物（統合テスト・example バイナリ、
# メンバーの rlib/rmeta、incremental、rustdoc 出力）は次の run では必ず古く再ビルドされ
# キャッシュ価値が無い。統合テストが約 760 ファイルあり、`cargo test --workspace` が
# テストバイナリごとに依存グラフ全体を静的リンクするため、build-no-cuda-toolkit の target
# キャッシュは約 9.8GB（圧縮後）に達し、リポジトリのキャッシュ上限を単独で使い切って他の
# キャッシュを毎回追い出していた。依存 crate（crates.io 側）の成果物だけを残す。
#
# 判定方法（名前の一致ではなくハッシュで判定する）:
#   1. `cargo metadata --no-deps` で workspace メンバーの package 名を得る
#   2. `.fingerprint/<package>-<16 桁 hex>` のうち package 名がメンバーのものを削除し、
#      その hex（ユニットのメタデータハッシュ）を集める。`build/<package>-<hex>` も同様
#   3. `deps/` は `<target 名>-<hex>[.拡張子…]` のうち hex が 2. の集合に含まれるものだけを
#      削除する。target 名で判定すると、統合テスト名と crates.io 依存の crate 名が同じ
#      場合（例: テスト `cast` と依存 `cast`）に依存側の成果物まで消してしまうため
#   4. プロファイル直下の通常ファイル（ルートユニット＝メンバーの uplift 成果物と .d）と
#      `examples/`・`incremental/` を削除する。依存 crate の成果物は uplift されない
#   5. `doc/`（`cargo doc --no-deps` の出力はメンバーのみ）と `tmp/`（統合テストの
#      CARGO_TARGET_TMPDIR）を削除する
# メンバーと同名の crates.io 依存が Cargo.lock にある場合は 2. が依存側を巻き込むため、
# fail-closed で中止する。削除前後のサイズとプロファイル別内訳を出力する（キャッシュ
# サイズの計測用。`.claude/rules/github-actions.md`「キャッシュ方針」節）。
# Fandhe-AI/actions の rust-base-ci（clippy / test ジョブの保存前 prune）と目的は同じ。
#
# 呼び出し規約: cargo を実行する最後のステップの後・`actions/cache/save` の前に置く。
# 失敗時（前段ステップの失敗）は保存しないため `if: always()` は付けない。
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./lib.sh
source "${DIR}/lib.sh"

reject_args "$(basename "$0")" "$@"

command -v jq >/dev/null 2>&1 || die "jq が見つかりません（target の prune に必要です）"

target_dir="${ROOT_DIR}/target"
if [ ! -d "${target_dir}" ]; then
  log "target が存在しないため prune をスキップします"
  exit 0
fi
[ -L "${target_dir}" ] && die "target が symlink のため prune を中止します"
real_target="$(cd "${target_dir}" && pwd -P)"

# assert_real_dir <dir>: 列挙・削除の起点にするディレクトリ自身が symlink でなく、実体が
# target 配下にあることを検証する（親ディレクトリの symlink 経由で target 外を列挙・
# 削除しないため。find は既定で symlink を辿らないが、cd した先は辿るため明示検査する）。
assert_real_dir() {
  local real
  [ -L "$1" ] && die "symlink のため prune を中止します: $1"
  real="$(cd "$1" && pwd -P)" || die "ディレクトリへ移動できません: $1"
  case "${real}" in
    "${real_target}" | "${real_target}"/*) ;;
    *) die "実体が target 配下にないため prune を中止します: $1 -> ${real}" ;;
  esac
}

# size_of <path>: du -s の KiB 値（存在しなければ 0）。
size_of() {
  if [ -e "$1" ]; then du -sk "$1" | awk '{print $1}'; else echo 0; fi
}
# fmt_mib <KiB>
fmt_mib() {
  awk -v k="$1" 'BEGIN { printf "%.0f MiB", k / 1024 }'
}

before_kib="$(size_of "${target_dir}")"

work="$(mktemp -d)"
trap 'rm -rf -- "${work}"' EXIT

# メンバー package 名（1 行 1 件）。--no-deps は依存解決・ネットワーク不要。
cargo metadata --no-deps --format-version 1 --locked | jq -r '.packages[].name' | sort -u >"${work}/members"
[ -s "${work}/members" ] || die "cargo metadata から workspace メンバーを取得できませんでした"
if grep -vqE '^[A-Za-z0-9_-]+$' "${work}/members"; then
  die "想定外の package 名を検出したため prune を中止します: $(grep -vE '^[A-Za-z0-9_-]+$' "${work}/members" | head -n 1)"
fi

# メンバーと同名の crates.io 依存（Cargo.lock で source を持つ package）があれば中止する。
awk '/^\[\[package\]\]/ { name = "" } /^name = / { name = $3 } /^source = / { gsub(/"/, "", name); print name }' Cargo.lock |
  sort -u >"${work}/lock-deps"
dup="$(comm -12 "${work}/members" "${work}/lock-deps" | head -n 1)"
[ -n "${dup}" ] && die "workspace メンバーと同名の依存 crate があるため prune を中止します: ${dup}"

removed=0
queue=()
# queue_rm <path> <base>: 境界・symlink を検査して削除キューへ積む（削除は flush_rm で一括）。
queue_rm() {
  case "$1" in
    "$2"/*) ;;
    *) die "削除対象が想定ディレクトリ配下ではありません: $1" ;;
  esac
  [ -L "$1" ] && die "symlink は削除対象にしません: $1"
  queue+=("$1")
}
flush_rm() {
  [ "${#queue[@]}" -eq 0 ] && return 0
  printf '%s\0' "${queue[@]}" | xargs -0 rm -rf --
  removed=$((removed + ${#queue[@]}))
  queue=()
}

# list_names <dir>: 直下のエントリ名を 1 行 1 件で出す（後段の sed が厳密な正規表現で
# 絞り込むため、改行を含む名前は一致せず対象外になる）。
list_names() {
  assert_real_dir "$1"
  (cd "$1" && find . -mindepth 1 -maxdepth 1) | sed 's|^\./||'
}

prune_profile() { # $1=プロファイルディレクトリ（target[/<triple>]/{debug,release}）
  local profile="$1" d n n_units=0 n_deps=0
  assert_real_dir "${profile}"
  for d in .fingerprint build deps; do
    if [ -e "${profile}/${d}" ] || [ -L "${profile}/${d}" ]; then assert_real_dir "${profile}/${d}"; fi
  done
  : >"${work}/hashes"
  # 2. .fingerprint/ と build/ のメンバーユニット（`<package>-<hex>` の package 名で判定し、
  #    hex を集める）。名前は空白・改行を含まない形（sed の正規表現で検証）だけを扱う。
  for d in .fingerprint build; do
    [ -d "${profile}/${d}" ] || continue
    while read -r n h; do
      queue_rm "${profile}/${d}/${n}" "${profile}"
      printf '%s\n' "${h}" >>"${work}/hashes"
      n_units=$((n_units + 1))
    done < <(list_names "${profile}/${d}" |
      sed -nE 's/^([A-Za-z0-9_-]+)-([0-9a-f]{16})$/\1 \2 &/p' |
      awk 'NR == FNR { m[$1]; next } ($1 in m) { print $3, $2 }' "${work}/members" -)
  done
  # 3. deps/ はメンバーユニットの hex に一致するものだけ。
  if [ -d "${profile}/deps" ]; then
    while IFS= read -r n; do
      queue_rm "${profile}/deps/${n}" "${profile}"
      n_deps=$((n_deps + 1))
    done < <(list_names "${profile}/deps" |
      sed -nE 's/^(lib)?[A-Za-z0-9_]+-([0-9a-f]{16})(\.[^ ]*)?$/\2 &/p' |
      awk 'NR == FNR { h[$1]; next } ($1 in h) { print $2 }' "${work}/hashes" -)
    if [ "${n_units}" -gt 0 ] && [ "${n_deps}" -eq 0 ]; then
      # ハッシュ対応（.fingerprint と deps のファイル名）が cargo の変更で崩れた場合の検知。
      # 削除漏れはキャッシュが大きくなるだけで正しさには影響しないため警告に留める。
      echo "::warning::${profile#"${target_dir}"/}: メンバーの fingerprint ${n_units} 件に対応する deps/ の成果物が見つかりませんでした（cargo のファイル名規則の変更を確認してください）"
    fi
  fi
  # 4. uplift 成果物（プロファイル直下の通常ファイル。.cargo-lock 等のドットファイルは残す）と
  #    examples/・incremental/。
  while IFS= read -r -d '' n; do
    queue_rm "${n}" "${profile}"
  done < <(find "${profile}" -mindepth 1 -maxdepth 1 -type f ! -name '.*' -print0)
  for d in examples incremental; do
    if [ -e "${profile}/${d}" ]; then queue_rm "${profile}/${d}" "${profile}"; fi
  done
  flush_rm
}

# target/{debug,release} と、--target 指定時の target/<triple>/{debug,release}。
profiles=()
while IFS= read -r -d '' d; do
  # target/debug/release のようなプロファイル配下の同名ディレクトリは対象外にする。
  case "${d%/*}" in
    */debug | */release) continue ;;
  esac
  profiles+=("${d}")
done < <(find "${target_dir}" -mindepth 1 -maxdepth 2 -type d \( -name debug -o -name release \) -print0)

# 計測（削除前）: プロファイルごとのサイズと deps/ のファイル数。
summary=()
for p in "${profiles[@]}"; do
  rel="${p#"${target_dir}"/}"
  summary+=("${rel}: $(fmt_mib "$(size_of "${p}")")（deps $(find "${p}/deps" -mindepth 1 -maxdepth 1 2>/dev/null | wc -l | tr -d ' ') 件）")
done

for p in "${profiles[@]}"; do
  prune_profile "${p}"
done
# doc/（`cargo doc --no-deps` の出力はメンバーのみ）と tmp/（CARGO_TARGET_TMPDIR）。
while IFS= read -r -d '' d; do
  queue_rm "${d}" "${target_dir}"
done < <(find "${target_dir}" -mindepth 1 -maxdepth 2 -type d \( -name doc -o -name tmp \) -print0)
flush_rm

after_kib="$(size_of "${target_dir}")"

{
  echo "prune 完了: ${removed} エントリを削除"
  echo "target: $(fmt_mib "${before_kib}") → $(fmt_mib "${after_kib}")（ディスク上。キャッシュの Cache Size は圧縮後）"
  echo "削除前のプロファイル別内訳:"
  printf '  %s\n' "${summary[@]}"
} >&2

if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
  {
    echo "### target prune（キャッシュ保存前）"
    echo ""
    echo "- target: $(fmt_mib "${before_kib}") → $(fmt_mib "${after_kib}")（ディスク上）"
    printf -- '- 削除前 %s\n' "${summary[@]}"
  } >>"${GITHUB_STEP_SUMMARY}"
fi
