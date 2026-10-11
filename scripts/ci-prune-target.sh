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

# size_of <path>: du -s の KiB 値（存在しなければ 0）。
size_of() {
  if [ -e "$1" ]; then du -sk "$1" | awk '{print $1}'; else echo 0; fi
}
# fmt_mib <KiB>
fmt_mib() {
  awk -v k="$1" 'BEGIN { printf "%.0f MiB", k / 1024 }'
}

before_kib="$(size_of "${target_dir}")"

# メンバー package 名。--no-deps は依存解決・ネットワーク不要（workspace 内のみを見る）。
members="|"
while IFS= read -r n; do
  # 区切り文字・glob 文字を含まないことの保証（case パターンへ埋め込むため）。
  [[ "${n}" =~ ^[A-Za-z0-9_-]+$ ]] || die "想定外の package 名を検出したため prune を中止します: ${n}"
  members+="${n}|"
done < <(cargo metadata --no-deps --format-version 1 --locked | jq -r '.packages[].name' | sort -u)
[ "${members}" = "|" ] && die "cargo metadata から workspace メンバーを取得できませんでした"

# メンバーと同名の crates.io 依存（Cargo.lock で source を持つ package）があれば中止する。
while IFS= read -r n; do
  case "${members}" in
    *"|${n}|"*) die "workspace メンバーと同名の依存 crate があるため prune を中止します: ${n}" ;;
  esac
done < <(awk '/^\[\[package\]\]/ { name = "" } /^name = / { name = $3 } /^source = / { gsub(/"/, "", name); print name }' Cargo.lock)

removed=0
safe_rm() { # $1=削除対象 $2=許可される親ディレクトリ（境界チェック）
  local p="$1" base="$2"
  case "${p}" in
    "${base}"/*) ;;
    *) die "削除対象が想定ディレクトリ配下ではありません: ${p}" ;;
  esac
  [ -L "${p}" ] && die "symlink は削除対象にしません: ${p}"
  rm -rf -- "${p}"
  removed=$((removed + 1))
}
prune_profile() { # $1=プロファイルディレクトリ（target[/<triple>]/{debug,release}）
  local profile="$1" d e b pkg h hashes="|" n_units=0 n_deps=0
  [ -L "${profile}" ] && die "${profile} が symlink のため prune を中止します"
  # 2. .fingerprint/ と build/ のメンバーユニット（package 名で判定し、hex を集める）。
  for d in .fingerprint build; do
    [ -d "${profile}/${d}" ] || continue
    while IFS= read -r -d '' e; do
      b="${e##*/}"
      pkg="${b%-*}"
      h="${b##*-}"
      [[ "${h}" =~ ^[0-9a-f]{16}$ ]] || continue
      case "${members}" in
        *"|${pkg}|"*)
          hashes+="${h}|"
          safe_rm "${e}" "${profile}"
          n_units=$((n_units + 1))
          ;;
      esac
    done < <(find "${profile}/${d}" -mindepth 1 -maxdepth 1 -print0)
  done
  # 3. deps/ はメンバーユニットの hex に一致するものだけ。
  if [ -d "${profile}/deps" ]; then
    while IFS= read -r -d '' e; do
      b="${e##*/}"
      [[ "${b}" =~ ^[A-Za-z0-9_]+-([0-9a-f]{16})(\..*)?$ ]] || continue
      case "${hashes}" in
        *"|${BASH_REMATCH[1]}|"*)
          safe_rm "${e}" "${profile}"
          n_deps=$((n_deps + 1))
          ;;
      esac
    done < <(find "${profile}/deps" -mindepth 1 -maxdepth 1 -print0)
  fi
  if [ "${n_units}" -gt 0 ] && [ "${n_deps}" -eq 0 ] && [ -d "${profile}/deps" ]; then
    # ハッシュ対応（.fingerprint と deps のファイル名）が cargo の変更で崩れた場合の検知。
    # 削除漏れはキャッシュが大きくなるだけで正しさには影響しないため警告に留める。
    echo "::warning::${profile#"${target_dir}"/}: メンバーの fingerprint ${n_units} 件に対応する deps/ の成果物が見つかりませんでした（cargo のファイル名規則の変更を確認してください）"
  fi
  # 4. uplift 成果物（プロファイル直下の通常ファイル。.cargo-lock 等のドットファイルは残す）と
  #    examples/・incremental/。
  while IFS= read -r -d '' e; do
    safe_rm "${e}" "${profile}"
  done < <(find "${profile}" -mindepth 1 -maxdepth 1 -type f ! -name '.*' -print0)
  for d in examples incremental; do
    if [ -e "${profile}/${d}" ]; then safe_rm "${profile}/${d}" "${profile}"; fi
  done
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
  safe_rm "${d}" "${target_dir}"
done < <(find "${target_dir}" -mindepth 1 -maxdepth 2 -type d \( -name doc -o -name tmp \) -print0)

after_kib="$(size_of "${target_dir}")"

{
  echo "prune 完了: ${removed} エントリを削除"
  echo "target: $(fmt_mib "${before_kib}") → $(fmt_mib "${after_kib}")（未圧縮。キャッシュ保存サイズは zstd 圧縮後）"
  echo "削除前のプロファイル別内訳:"
  printf '  %s\n' "${summary[@]}"
} >&2

if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
  {
    echo "### target prune（キャッシュ保存前）"
    echo ""
    echo "- target: $(fmt_mib "${before_kib}") → $(fmt_mib "${after_kib}")（未圧縮）"
    printf -- '- 削除前 %s\n' "${summary[@]}"
  } >>"${GITHUB_STEP_SUMMARY}"
fi
