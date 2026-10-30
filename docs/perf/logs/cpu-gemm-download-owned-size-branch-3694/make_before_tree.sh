#!/usr/bin/env bash
# イシュー #3694: A/B の before の木を作る（RULE.txt「腕」が正）。
# before = a7aac2f0227add243845418e0897ac0675c5b622 のアーカイブ展開木（base_tree）に、計測ハーネスだけを
# after の木から「置換」した木。置換対象は scripts/bench/framework-compare/（target* を除く全体）と
# crates/facade/tests/cpu_gemm_resident_reuse_diag.rs。ライブラリコードは a7aac2f0 のまま触れない。
# 置換は rm -rf してから cp する（上書きの混ぜ合わせで after 側に無いファイルが残るのを防ぐ）。
# 再現性: 置換した全ファイルについて「相対パスを LC_ALL=C sort し、`<sha256>  <path>` の一覧全体を sha256 した値」の
# 先頭 12 桁を .rev-stamp 1 行目に入れる（`a7aac2f0…+harness-overlay(<12 桁>)`）。2 行目は dirty=0。
# 呼び出し元: build_ab_arms.sh の前段（手動）。before は main へ入れない計測専用の木で、push しない。
# 使い方: make_before_tree.sh <base_tree> <after_tree> <before_tree>
#   base_tree: a7aac2f0 のアーカイブ展開木（.rev-stamp の 1 行目が a7aac2f0 の完全 sha・2 行目 dirty=0 を事前に置く）
#   after_tree: after の木（.rev-stamp の 2 行目 dirty=0）  before_tree: 新規（既存なら中止）
# bash 3.2 互換。
set -euo pipefail

BASE_SHA="a7aac2f0227add243845418e0897ac0675c5b622"

[ $# -eq 3 ] || { echo "usage: $0 <base_tree> <after_tree> <before_tree>" >&2; exit 2; }
base_tree="$1"
after_tree="$2"
before_tree="$3"
for p in "${base_tree}" "${after_tree}" "${before_tree}"; do
  case "${p}" in /*) ;; *) echo "絶対パスのみ許可: ${p}" >&2; exit 2 ;; esac
  case "${p}" in *[!A-Za-z0-9._/-]*) echo "許可外の文字を含むパス: ${p}" >&2; exit 2 ;; esac
done
[ -d "${base_tree}" ] && [ -d "${after_tree}" ] || { echo "入力の木が無い" >&2; exit 2; }
[ -f "${base_tree}/.rev-stamp" ] && [ -f "${after_tree}/.rev-stamp" ] || { echo ".rev-stamp が無い" >&2; exit 1; }
[ "$(sed -n 1p "${base_tree}/.rev-stamp")" = "${BASE_SHA}" ] || { echo "base の .rev-stamp 1 行目が ${BASE_SHA} でない" >&2; exit 1; }
[ "$(sed -n 2p "${base_tree}/.rev-stamp")" = "dirty=0" ] || { echo "base の dirty が 0 でない" >&2; exit 1; }
[ "$(sed -n 2p "${after_tree}/.rev-stamp")" = "dirty=0" ] || { echo "after の dirty が 0 でない" >&2; exit 1; }
[ -d "${after_tree}/scripts/bench/framework-compare" ] && [ -f "${after_tree}/crates/facade/tests/cpu_gemm_resident_reuse_diag.rs" ] || { echo "after に置換元が無い" >&2; exit 1; }
if [ -e "${before_tree}" ]; then echo "before の木が既に存在する: ${before_tree}" >&2; exit 5; fi

export LC_ALL=C
sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'; else shasum -a 256 "$1" | awk '{print $1}'; fi
}

# 置換対象ファイルの「sha256  相対パス」一覧（相対パス順）。$1=木のルート
overlay_list() {
  local root="$1" f
  (cd "${root}" && { find scripts/bench/framework-compare -type f -not -path '*/target*'; echo crates/facade/tests/cpu_gemm_resident_reuse_diag.rs; } | sort) | while IFS= read -r f; do
    printf '%s  %s\n' "$(sha256_of "${root}/${f}")" "${f}"
  done
}

mkdir -p "${before_tree}"
(cd "${base_tree}" && tar -cf - --exclude='target*' .) | (cd "${before_tree}" && tar -xf -)

# framework-compare は丸ごと置換（after 側に無いファイルを残さない）。
rm -rf "${before_tree}/scripts/bench/framework-compare"
mkdir -p "${before_tree}/scripts/bench"
(cd "${after_tree}/scripts/bench" && tar -cf - --exclude='target*' framework-compare) | (cd "${before_tree}/scripts/bench" && tar -xf -)
cp "${after_tree}/crates/facade/tests/cpu_gemm_resident_reuse_diag.rs" "${before_tree}/crates/facade/tests/cpu_gemm_resident_reuse_diag.rs"

# 置換後の内容が after と一致することを確かめ、その一覧の sha256 を指紋にする。
list="$(overlay_list "${before_tree}")"
after_list="$(overlay_list "${after_tree}")"
[ "${list}" = "${after_list}" ] || { echo "置換後の内容が after と一致しない" >&2; exit 1; }
tmp="$(mktemp)"
printf '%s\n' "${list}" >"${tmp}"
osha="$(sha256_of "${tmp}")"
rm -f "${tmp}"
printf '%s+harness-overlay(%s)\ndirty=0\n' "${BASE_SHA}" "$(printf '%s' "${osha}" | cut -c1-12)" >"${before_tree}/.rev-stamp"
echo "overlay sha256: ${osha}"
echo "完了: ${before_tree}"
