#!/usr/bin/env bash
# イシュー #3701: runner の木（部品を入れたコミットの git archive 展開木）から after の木を作る。
# after = runner + probe-3701.patch（計測専用パッチ。backend-cpu に S1・S2 の内側の計時点を足す。main には入れない）。
# 判定規則は同ディレクトリの RULE.txt が正。入力の木に split_probe.rs が無いことを先に検査する
# （main の本番コードに計時点が無いことの機械検査）。
# .rev-stamp の 1 行目は「<runner の sha>+probe-3701.patch(<patch の sha256 先頭 12 桁>)」、2 行目は dirty=0。
# runner の木のルートには .rev-stamp（1 行目 sha・2 行目 dirty=0）が必須。
# 使い方: make_after_tree.sh <runner_tree> <after_tree>   （after_tree は新規。既存なら中止）
# bash 3.2 互換。
set -euo pipefail
export LC_ALL=C

[ $# -eq 2 ] || { echo "usage: $0 <runner_tree> <after_tree>" >&2; exit 2; }
runner_tree="$1"
after_tree="$2"
for p in "${runner_tree}" "${after_tree}"; do
  case "${p}" in /*) ;; *) echo "絶対パスのみ許可: ${p}" >&2; exit 2 ;; esac
  case "${p}" in *[!A-Za-z0-9._/-]*) echo "許可外の文字を含むパス: ${p}" >&2; exit 2 ;; esac
done
[ -d "${runner_tree}" ] || { echo "runner の木が無い" >&2; exit 2; }
[ -f "${runner_tree}/.rev-stamp" ] || { echo ".rev-stamp が無い: ${runner_tree}" >&2; exit 1; }
[ "$(sed -n 2p "${runner_tree}/.rev-stamp")" = "dirty=0" ] || { echo "runner の dirty が 0 でない" >&2; exit 1; }
if [ -e "${runner_tree}/crates/backend-cpu/src/split_probe.rs" ]; then
  echo "runner の木に split_probe.rs が既にある（main に計時点が入っている）" >&2
  exit 1
fi
if [ -e "${after_tree}" ]; then echo "after の木が既に存在する: ${after_tree}" >&2; exit 5; fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
patch_file="${script_dir}/probe-3701.patch"
[ -f "${patch_file}" ] || { echo "probe-3701.patch が無い" >&2; exit 1; }

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'; else shasum -a 256 "$1" | awk '{print $1}'; fi
}

mkdir -p "${after_tree}"
(cd "${runner_tree}" && tar -cf - --exclude='target*' .) | (cd "${after_tree}" && tar -xf -)
(cd "${after_tree}" && patch -p1 --no-backup-if-mismatch <"${patch_file}")

psha="$(sha256_of "${patch_file}")"
printf '%s+probe-3701.patch(%s)\ndirty=0\n' "$(sed -n 1p "${runner_tree}/.rev-stamp")" "$(printf '%s' "${psha}" | cut -c1-12)" >"${after_tree}/.rev-stamp"
echo "probe-3701.patch sha256: ${psha}"
echo "完了: ${after_tree}"
