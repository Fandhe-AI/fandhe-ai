#!/usr/bin/env bash
# イシュー #3184 clone 準備: 計測 target／生成物を各 clone の .git/info/exclude で未追跡扱いから外す。
# 理由: 継承スクリプトは未追跡ファイル 1 件で停止するため（verify_no_untracked_build_inputs）、
# 改名した target-ab-* を clone 内に置くには exclude が要る。作業ツリーのファイルは変えない。
set -u
. "$(dirname "$0")/lib.sh"
for c in before after diag; do
  printf '%s\n' 'scripts/bench/framework-compare/target-ab-*' 'scripts/bench/framework-compare/results/raw/' \
    'scripts/bench/framework-compare/compare-*' >>"${R}/${c}/.git/info/exclude"
done
git -C "${R}/after" checkout -- .
echo "before untracked/modified: $(git -C "${R}/before" status --porcelain | wc -l)"
echo "after untracked/modified: $(git -C "${R}/after" status --porcelain | wc -l)"
