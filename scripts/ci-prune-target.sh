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
# 削除ロジックは Fandhe-AI/actions の rust-base-ci（clippy / test ジョブの保存前 prune。
# 同リポジトリ #131〜#134）と同じ判定（cargo metadata --no-deps のメンバー package 名・
# target 名に一致する .fingerprint/ build/ deps/ のエントリと uplift 成果物）を移植し、
# 本リポジトリ向けに次を加えた:
#   - release プロファイル（verification-gates の `cargo test --release`）も対象にする
#   - `doc/`（`cargo doc --no-deps` の出力はメンバーのみ）と `tmp/`（統合テストの
#     CARGO_TARGET_TMPDIR）を削除する
#   - 削除前後のサイズ・deps/ のファイル数を出力する（キャッシュサイズの計測用。
#     `.claude/rules/github-actions.md`「キャッシュ方針」節）
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

# メンバー package 名 + 全 target 名（lib/bin/test/bench/example）。--no-deps は依存解決・
# ネットワーク不要（workspace 内のみを見る）。
raw=()
while IFS= read -r n; do
  [ -n "${n}" ] && raw+=("${n}")
done < <(cargo metadata --no-deps --format-version 1 --locked \
  | jq -r '.packages[] | .name, (.targets[] | select(.kind | index("custom-build") | not) | .name)' \
  | sort -u)
[ "${#raw[@]}" -eq 0 ] && die "cargo metadata から workspace メンバーを取得できませんでした"

# 正規表現へ無エスケープで埋め込むため、名前は Unicode 英数字・`_`・`-` のみ許可する
# （正規表現メタ文字を含まないことの保証。rust-base-ci と同じ検査）。
pkgs=()
names=()
for n in "${raw[@]}"; do
  if ! jq -n --arg n "${n}" '$n | test("^[\\p{L}\\p{N}_-]+$")' | grep -q '^true$'; then
    die "想定外の crate / target 名を検出したため prune を中止します: ${n}"
  fi
  pkgs+=("${n}")
  names+=("${n}" "${n//-/_}")
done
alt_pkgs="$(printf '%s\n' "${pkgs[@]}" | sort -u | paste -sd '|' -)"
alt_names="$(printf '%s\n' "${names[@]}" | sort -u | paste -sd '|' -)"
re_unit="^(${alt_pkgs})-[0-9a-f]{16}$"                           # .fingerprint/ build/（package 名）
re_dep="^(lib)?(${alt_names})-[0-9a-f]{16}(\..+)?$"              # deps/（target 名・`_` 正規化）
re_top="^(lib)?(${alt_names})(\.(d|rlib|rmeta|so|a|dylib|exe|pdb))?$" # <profile> 直下の uplift 成果物

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
prune_entries() { # $1=dir $2=regex $3=base
  local dir="$1" re="$2" base="$3" e b
  [ -d "${dir}" ] || return 0
  while IFS= read -r -d '' e; do
    b="${e##*/}"
    if [[ "${b}" =~ ${re} ]]; then safe_rm "${e}" "${base}"; fi
  done < <(find "${dir}" -mindepth 1 -maxdepth 1 -print0)
}
prune_profile() { # $1=プロファイルディレクトリ（target[/<triple>]/{debug,release}）
  local profile="$1" d e b
  [ -L "${profile}" ] && die "${profile} が symlink のため prune を中止します"
  for d in examples incremental; do
    if [ -e "${profile}/${d}" ]; then safe_rm "${profile}/${d}" "${profile}"; fi
  done
  prune_entries "${profile}/.fingerprint" "${re_unit}" "${profile}"
  prune_entries "${profile}/build" "${re_unit}" "${profile}"
  prune_entries "${profile}/deps" "${re_dep}" "${profile}"
  while IFS= read -r -d '' e; do
    b="${e##*/}"
    if [[ "${b}" =~ ${re_top} ]] || [[ "${b}" == *.d ]]; then safe_rm "${e}" "${profile}"; fi
  done < <(find "${profile}" -mindepth 1 -maxdepth 1 -type f -print0)
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
