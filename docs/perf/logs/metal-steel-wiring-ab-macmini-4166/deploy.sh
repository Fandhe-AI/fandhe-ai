#!/bin/bash
# #4166 の展開（RULE.txt 項 1。Mac mini 上で 1 回だけ実行する。計測はしない・hw-lock は取らない）。
# 入力: 9eba21749 の git archive（送信側で `git archive --format=tar 9eba21749` したもの）と、RULE.txt を固定した
# コミットの本ディレクトリの git archive（rule.tar）。
# 作るもの（展開先 W は名前が .noindex で終わる新しいディレクトリ。既にあれば止まる）:
#   W/tree   : archive + .rev-stamp（1 行: 9eba21749 の完全な SHA）。runner・ゲート・比較スクリプトはここのものを使う。
#   W/after  : 同じ archive を git init で 1 コミットにした checkout（clean。定数は true）。
#   W/before : W/after の clone に、tile.rs の STEEL_LARGE_N_ENABLED を true→false にする未コミットの 1 行だけを当てたもの。
#   W/rule   : 本ディレクトリ（drive_series.sh・prereq_tests.sh・fp.sh・judge_ab.py）。sha256 は RULE.txt 項 1 (d) で照合する。
#   W/fp-deploy.txt・W/fp-deploy-ab.txt・W/manifest-deploy.txt: 配置時の指紋と manifest（#4160 と同じ形）。
# 使い方: bash deploy.sh <W> <tree.tar> <rule.tar> <期待する指紋>
set -euo pipefail
W=$1; TAR=$2; RULETAR=$3; EXPECT=$4
SHA=9eba21749767147d100399962e3d41e54c8365df
TILE=crates/backend-metal/src/tile.rs
LINE_ON='pub(crate) const STEEL_LARGE_N_ENABLED: bool = true;'
LINE_OFF='pub(crate) const STEEL_LARGE_N_ENABLED: bool = false;'
case "${W}" in /*.noindex) ;; *) echo "error: W は絶対パスで .noindex で終わること: ${W}" >&2; exit 2;; esac
[ -e "${W}" ] && { echo "error: 既にある: ${W}" >&2; exit 2; }
mkdir "${W}" "${W}/tree" "${W}/after" "${W}/rule"
tar -x -f "${TAR}" -C "${W}/tree"
tar -x -f "${RULETAR}" -C "${W}/rule" --strip-components=4
printf '%s\n' "${SHA}" > "${W}/tree/.rev-stamp"
got=$(bash "${W}/rule/fp.sh" "${W}/tree")
echo "${got}" > "${W}/fp-deploy.txt"
bash "${W}/rule/fp.sh" "${W}/tree" ab > "${W}/fp-deploy-ab.txt"
( cd "${W}/tree" && find . -type f ! -name '.rev-stamp' | LC_ALL=C sort | tr '\n' '\0' | xargs -0 shasum -a 256 ) > "${W}/manifest-deploy.txt"
fp=$(echo "${got}" | sed 's/^fingerprint=\([0-9a-f]*\) .*/\1/')
if [ "${fp}" != "${EXPECT}" ]; then
  echo "error: 指紋が RULE.txt と違う: ${got}" >&2
  exit 3
fi
# after: 同じ archive を 1 コミットに。before: その clone + 1 行。
tar -x -f "${TAR}" -C "${W}/after"
(
  cd "${W}/after"
  git init -q
  git add -A
  git -c user.name=ab-4166 -c user.email=ab-4166@invalid -c commit.gpgsign=false commit -q -m "archive of ${SHA}"
)
git clone -q "${W}/after" "${W}/before"
[ "$(grep -cxF "${LINE_ON}" "${W}/before/${TILE}")" = 1 ] || { echo "error: tile.rs に定数の行がちょうど 1 つ無い" >&2; exit 3; }
sed -i '' "s/^${LINE_ON}\$/${LINE_OFF}/" "${W}/before/${TILE}"
[ "$(git -C "${W}/before" diff --numstat | tr '\t' ' ')" = "1 1 ${TILE}" ] || { echo "error: before の差分が 1 行ではない" >&2; exit 3; }
[ -z "$(git -C "${W}/after" status --porcelain)" ] || { echo "error: after が clean ではない" >&2; exit 3; }
[ "$(git -C "${W}/after" rev-parse HEAD)" = "$(git -C "${W}/before" rev-parse HEAD)" ] || { echo "error: HEAD が違う" >&2; exit 3; }
{
  echo "deployed $(date -u +%FT%TZ)"
  echo "target_sha=${SHA}"
  cat "${W}/fp-deploy.txt"
  echo "ab_excluded $(cat "${W}/fp-deploy-ab.txt")"
  echo "checkout_head=$(git -C "${W}/after" rev-parse HEAD)"
  echo "before_diff:"; git -C "${W}/before" diff | sed 's/^/  /'
  echo "rule_files:"; ( cd "${W}/rule" && shasum -a 256 drive_series.sh prereq_tests.sh fp.sh judge_ab.py ) | sed 's/^/  /'
} > "${W}/deploy.txt"
echo "deploy ok"
