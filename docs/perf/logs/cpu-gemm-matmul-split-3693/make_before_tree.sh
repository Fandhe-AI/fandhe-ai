#!/usr/bin/env bash
# イシュー #3693: after の展開木（git archive）から before の木を作る。
# before = after + before-arm.patch（resident_input.rs の呼び出し 1 行を download_owned から download に戻す）。
# 判定規則は同ディレクトリの RULE.txt が正。before は main へ入れない計測専用の木で、push しない。
# 手順: <after_tree> を <before_tree>（新規）へコピー → パッチ適用。再現性は patch の sha256 と
#       build_split_arms.sh の diff -r 検査（差が resident_input.rs 1 ファイルだけ）で担保する。.rev-stamp の 1 行目は
#       「<after の sha>+before-arm.patch(<patch の sha256 先頭 12 桁>)」、2 行目は dirty=0。
# 使い方: make_before_tree.sh <after_tree> <before_tree>
# bash 3.2 互換。
set -euo pipefail

[ $# -eq 2 ] || { echo "usage: $0 <after_tree> <before_tree>" >&2; exit 2; }
after_tree="$1"
before_tree="$2"
for p in "${after_tree}" "${before_tree}"; do
  case "${p}" in /*) ;; *) echo "絶対パスのみ許可: ${p}" >&2; exit 2 ;; esac
  case "${p}" in *[!A-Za-z0-9._/-]*) echo "許可外の文字を含むパス: ${p}" >&2; exit 2 ;; esac
done
[ -d "${after_tree}" ] || { echo "after の木が無い" >&2; exit 2; }
[ -f "${after_tree}/.rev-stamp" ] || { echo ".rev-stamp が無い: ${after_tree}" >&2; exit 1; }
[ "$(sed -n 2p "${after_tree}/.rev-stamp")" = "dirty=0" ] || { echo "after の dirty が 0 でない" >&2; exit 1; }
if [ -e "${before_tree}" ]; then echo "before の木が既に存在する: ${before_tree}" >&2; exit 5; fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
patch_file="${script_dir}/before-arm.patch"
[ -f "${patch_file}" ] || { echo "before-arm.patch が無い" >&2; exit 1; }

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'; else shasum -a 256 "$1" | awk '{print $1}'; fi
}

mkdir -p "${before_tree}"
(cd "${after_tree}" && tar -cf - --exclude='target*' .) | (cd "${before_tree}" && tar -xf -)
(cd "${before_tree}" && patch -p1 --no-backup-if-mismatch <"${patch_file}")

psha="$(sha256_of "${patch_file}")"
printf '%s+before-arm.patch(%s)\ndirty=0\n' "$(sed -n 1p "${after_tree}/.rev-stamp")" "$(printf '%s' "${psha}" | cut -c1-12)" >"${before_tree}/.rev-stamp"
echo "before-arm.patch sha256: ${psha}"
echo "完了: ${before_tree}"
