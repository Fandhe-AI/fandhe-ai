#!/usr/bin/env bash
# イシュー #3190 再開分: GB10 上の clone 作成 (runner / b5-base / b5-head)。CARGO_TARGET_DIR は export しない。
set -eu
unset CARGO_TARGET_DIR RAYON_NUM_THREADS
R=$HOME/work/r3190-reopen
cd "$R"
OUT=$R/out
mkdir -p "$OUT"
RUNNER_SHA=fbf6ecbbe
BASE_SHA=230937f5f
for d in runner b5-base b5-head; do
  rm -rf "$R/$d"
  git init -q "$R/$d"
  git -C "$R/$d" fetch -q "$R/r3190.bundle" 'refs/*:refs/bundle/*'
done
git -C "$R/runner" checkout -q --detach "$RUNNER_SHA"
git -C "$R/b5-base" checkout -q --detach "$BASE_SHA"
git -C "$R/b5-head" checkout -q --detach "$BASE_SHA"
git -C "$R/b5-head" apply "$R/runner/docs/perf/logs/optin-ab-gb10-3190/on-boundary-predicate.patch"
{
  echo "runner  : $(git -C "$R/runner" rev-parse HEAD) status=[$(git -C "$R/runner" status --porcelain | wc -l)]"
  echo "b5-base : $(git -C "$R/b5-base" rev-parse HEAD) status=[$(git -C "$R/b5-base" status --porcelain | wc -l)]"
  echo "b5-head : $(git -C "$R/b5-head" rev-parse HEAD) status=[$(git -C "$R/b5-head" status --porcelain | tr '\n' ' ')] (base + on-boundary-predicate.patch。未コミット)"
  echo "b5-base-vs-origin-main backend-cuda src/examples diff (tests 以外) lines: $(git -C "$R/runner" diff 230937f5f refs/bundle/remotes/origin/main -- crates/backend-cuda/src crates/backend-cuda/examples | wc -l)"
  echo "origin/main = $(git -C "$R/runner" rev-parse refs/bundle/remotes/origin/main)"
} | tee "$OUT/rev-stamp.txt"
{
  echo "--- date: $(date -u +%FT%TZ)"
  echo "--- uptime: $(uptime)"
  echo "--- nvidia-smi"; nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv
  echo "--- procs (cargo|bench|rustc|nsys|nvidia-smi)"; ps -eo pid,etime,args | grep -E 'cargo|bench|rustc|nsys|nvidia-smi' | grep -v grep || true
} >"$OUT/preflight-procs.txt" 2>&1
cat "$OUT/preflight-procs.txt"
