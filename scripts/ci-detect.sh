#!/usr/bin/env bash
# ci.yml の各固有ジョブ共通の `detect` ステップ本体。GITHUB_OUTPUT へ `has_cargo=true|false` を
# 出力する。後続の cargo 系ステップは `if: steps.detect.outputs.has_cargo == 'true'` で
# ガードされているため、false を出すとジョブは checkout と軽量な self-test のみで success になる。
#
# false を出す条件:
#   1. Cargo.toml が無い（M0 以前の冪等セルフヒール方針のフェイルセーフ。.claude/rules/ci.md）
#   2. pull_request で、変更が docs 系ファイルのみ（下の is_docs_only_path）。
#      必須チェック（ruleset の required status checks）はジョブ名で参照されるため、
#      workflow 単位の `paths` フィルタではなく「ジョブは起動して成功する」形で重い処理だけを
#      省く。差分は checkout の fetch-depth: 2 で取得したマージコミットの第 1 親との比較で
#      求める。差分を取得できない・空の場合は fail-closed で true（通常どおり実行）にする。
#      push は ci.yml の `paths-ignore` で workflow ごと省くため本スクリプトでは判定しない。
#
# 環境変数: EVENT_NAME（github.event_name。未設定なら docs 判定をしない）
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./lib.sh
source "${DIR}/lib.sh"

# is_docs_only_path <path>: ビルド・テスト・検査の結果に影響しない docs 系パスか。
# site/ は docs-site クレートのテストが実読込するため、crates/ 配下の md は
# include_str!/doc 属性で取り込まれうるため、いずれも対象外（= コード扱い）。
is_docs_only_path() {
  case "$1" in
    site/* | crates/*) return 1 ;;
    docs/* | .claude/* | .agents/* | skills-lock.json | LICENSE-APACHE | LICENSE-MIT) return 0 ;;
    *.md) return 0 ;;
    *) return 1 ;;
  esac
}

# docs_only: 変更ファイルが 1 件以上あり、全て docs 系なら 0。
docs_only() {
  local files f
  files="$(git diff --name-only HEAD^1 HEAD 2>/dev/null)" || return 1
  [ -n "${files}" ] || return 1
  while IFS= read -r f; do
    is_docs_only_path "${f}" || return 1
  done <<<"${files}"
  return 0
}

emit() {
  echo "has_cargo=$1" >>"${GITHUB_OUTPUT:-/dev/stdout}"
}

if ! has_cargo; then
  emit false
  echo "::notice::Cargo.toml が未追加のため cargo 系ステップをスキップしました（workspace 作成後に有効化されます）"
elif [ "${EVENT_NAME:-}" = "pull_request" ] && docs_only; then
  emit false
  echo "::notice::docs 系ファイルのみの変更のため cargo 系ステップをスキップしました（scripts/ci-detect.sh）"
else
  emit true
fi
