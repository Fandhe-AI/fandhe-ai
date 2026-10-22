#!/usr/bin/env bash
# イシュー #3183: Mac mini へ転送する実行ツリーを `git archive` から作る（ステージング。転送は rsync）。
# `git archive` の展開物には .git（worktree 由来の gitfile を含む）も追跡外のファイル（.env*・*.local.md 等）も入らない。
# 第 1 ラウンドは worktree の .git ファイルが rsync の除外 `.git/`（末尾スラッシュ付きはディレクトリのみ）をすり抜けて
# 転送され、ノードの env_info の git_head が空になった。その再発防止として、ツリー直下に
#   .rev-stamp    : 1 行目 = ref のコミット SHA（overlay 指定時は 2 行目 overlay=<path>@<HEAD の SHA>）
#   .tree-manifest: `sha256  path` の一覧（.rev-stamp と .tree-manifest を除く全ファイル。LC_ALL=C ソート）
# を書く。orchestrate.sh がノード側で .rev-stamp を読み .tree-manifest を照合する（fail-closed）。
#
# 使い方: stage-tree.sh <ref> <dest-dir> [--overlay <repo 内のパス>]
#   <dest-dir> は存在する空ディレクトリ（呼び出し側が mktemp -d で作り、同じ呼び出しの中で後始末する）。
#   --overlay: HEAD のファイル 1 つを展開物へ重ねる（v0.11.0 ツリーへ HEAD の診断テストを重ねる用途）。
# rsync の例: rsync -a --delete --exclude '.git' --exclude '.env*' --exclude 'real-hardware-verification-env.local.md' \
#   "<dest-dir>/" "$METAL_NODE":~/work/rust-ai-library-run/
#   （`--exclude '.git'` は末尾スラッシュなし。gitfile も除く。実値の METAL_NODE はコミットしない）
set -euo pipefail

usage() { echo "usage: $0 <ref> <dest-dir> [--overlay <path>]" >&2; exit 2; }
[ $# -ge 2 ] || usage
ref="$1"
dest="$2"
shift 2
overlay=""
while [ $# -gt 0 ]; do
  case "$1" in
    --overlay)
      [ $# -ge 2 ] || usage
      overlay="$2"
      shift 2
      ;;
    *) usage ;;
  esac
done
[ -d "${dest}" ] || { echo "dest が無い: ${dest}" >&2; exit 2; }
[ -z "$(ls -A "${dest}")" ] || { echo "dest が空でない（上書き禁止）: ${dest}" >&2; exit 2; }

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${repo_root}"
ref_sha="$(git rev-parse --verify "${ref}^{commit}")"
git archive "${ref_sha}" | tar -x -C "${dest}"
{
  echo "${ref_sha}"
  if [ -n "${overlay}" ]; then
    head_sha="$(git rev-parse --verify HEAD)"
    mkdir -p "${dest}/$(dirname "${overlay}")"
    git show "HEAD:${overlay}" >"${dest}/${overlay}"
    echo "overlay=${overlay}@${head_sha}"
  fi
} >"${dest}/.rev-stamp"
(cd "${dest}" && find . -type f ! -name .rev-stamp ! -name .tree-manifest -print0 | LC_ALL=C sort -z \
  | xargs -0 shasum -a 256 | sed -e 's|  \./|  |' >.tree-manifest)
echo "staged ${ref_sha} -> ${dest} ($(wc -l <"${dest}/.tree-manifest" | tr -d ' ') files)"
