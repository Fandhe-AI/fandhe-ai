#!/bin/bash
# #4166 のツリー指紋（RULE.txt 項 1 (c)。#4160 の RULE.txt 項 14 (a) と同じ求め方）。
# 展開ツリーで .git 以下と .rev-stamp を除く全ファイルの相対パスを LC_ALL=C で整列し、各ファイルの
# `shasum -a 256` の出力を順に連結した全体の sha256。対象ツリー（9eba21749 の git archive）には本 issue の
# RULE.txt が入らないので、#4160 のような自己参照の除外はない。
# 第 2 引数に ab を渡すと、A/B の runner が書く scripts/bench/framework-compare/results/ と target-ab-4160-* を除く
# （runner は名前に 4160 を固定で使う）。`__pycache__` は除かない（RULE.txt 項 5 (a)）。
# 使い方: bash fp.sh <展開ツリー> [ab]  → `fingerprint=<sha256> files=<件数>`
set -euo pipefail
cd "$1"
FC=./scripts/bench/framework-compare
if [ "${2:-}" = ab ]; then
  list=$(find . -type f ! -path './.git/*' ! -name '.rev-stamp' ! -path "${FC}/results/*" ! -path "${FC}/target-ab-4160-*" | LC_ALL=C sort)
elif [ -z "${2:-}" ]; then
  list=$(find . -type f ! -path './.git/*' ! -name '.rev-stamp' | LC_ALL=C sort)
else
  echo "usage: fp.sh <tree> [ab]" >&2
  exit 2
fi
n=$(printf '%s\n' "${list}" | wc -l | tr -d ' ')
h=$(printf '%s\n' "${list}" | tr '\n' '\0' | xargs -0 shasum -a 256 | shasum -a 256 | cut -d' ' -f1)
echo "fingerprint=${h} files=${n}"
