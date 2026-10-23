#!/usr/bin/env bash
# イシュー #3195: 実機上の clone 作成（runner / before / after）と preflight 記録。
# CARGO_TARGET_DIR は export しない（#3190 の事故の再発防止）。
# 使い方: bash setup.sh <作業ディレクトリ> （事前に <作業ディレクトリ>/r3195.bundle を置く）
set -eu
unset CARGO_TARGET_DIR RAYON_NUM_THREADS
export PATH=$HOME/.cargo/bin:$PATH
R=$1
OUT=$R/out
mkdir -p "$OUT"
RUNNER_SHA=c8bb8448d
BASE_SHA=fd1ca01a9
for d in runner before after; do
  rm -rf "$R/$d"
  git init -q "$R/$d"
  git -C "$R/$d" fetch -q "$R/r3195.bundle" 'refs/*:refs/bundle/*'
done
git -C "$R/runner" checkout -q --detach "$RUNNER_SHA"
git -C "$R/before" checkout -q --detach "$BASE_SHA"
git -C "$R/after" checkout -q --detach "$BASE_SHA"
# after: 定数宣言 1 行だけを = 4096; にした未コミット変更（RULE.txt §4）
sed -i.bak -E 's/^(pub\(crate\) const MSE_BACKWARD_PARALLEL_MIN_ELEMS: usize = )0;/\14096;/' "$R/after/crates/backend-cpu/src/mse.rs"
rm -f "$R/after/crates/backend-cpu/src/mse.rs.bak"
{
  echo "runner : $(git -C "$R/runner" rev-parse HEAD) status=[$(git -C "$R/runner" status --porcelain | wc -l | tr -d ' ')]"
  echo "before : $(git -C "$R/before" rev-parse HEAD) status=[$(git -C "$R/before" status --porcelain | wc -l | tr -d ' ')]"
  echo "after  : $(git -C "$R/after" rev-parse HEAD) status=[$(git -C "$R/after" status --porcelain | tr '\n' ' ')] (before と同一 sha + mse.rs 定数 1 行。未コミット)"
  echo "after-diff:"; git -C "$R/after" diff --stat; git -C "$R/after" diff -U0 | grep '^[+-][^+-]'
} | tee "$OUT/rev-stamp.txt"
{
  echo "--- date: $(date -u +%FT%TZ)"
  echo "--- uptime: $(uptime)"
  echo "--- procs (cargo|rustc|bench)"
  ps -axo pid,etime,command | grep -E 'cargo|rustc|bench' | grep -v grep || true
} >"$OUT/preflight-procs.txt" 2>&1
cat "$OUT/preflight-procs.txt"
{
  echo "--- uname: $(uname -srm)"
  echo "--- rustc: $(rustc -V)  cargo: $(cargo -V)"
  echo "--- RAYON_NUM_THREADS=[${RAYON_NUM_THREADS:-unset}] CARGO_TARGET_DIR=[${CARGO_TARGET_DIR:-unset}]"
  echo "--- tools: jq=$(command -v jq || echo none) python3=$(command -v python3 || echo none) sha=$(command -v sha256sum || command -v shasum || echo none)"
} >"$OUT/env_info.txt" 2>&1
cat "$OUT/env_info.txt"
