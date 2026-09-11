#!/usr/bin/env bash
# #1555 Metal resident weight-grad staging の framework-compare train A/B
# before = origin/main の crates/facade（path patch）、after = branch worktree の crates/facade
# 5 round・run 単位で before/after を順序反転・record_only（専有ゲートなし・uptime 記録のみ）
set -euo pipefail
FC=<masked>
BEFORE_FACADE=<masked>
AFTER_FACADE=<masked>
OUT=${OUT:?}
LABEL=${LABEL:-1555}
ROUNDS=${ROUNDS:-5}
mkdir -p "$OUT"
cd "$FC"
LOCK_BACKUP="$(mktemp)"; cp Cargo.lock "$LOCK_BACKUP"
trap 'cp "$LOCK_BACKUP" "$FC/Cargo.lock"; rm -f "$LOCK_BACKUP"' EXIT

build_arm() { # build_arm <arm> <facade_path> <target_dir>
  local arm=$1 facade=$2 tdir=$3 msg exe
  msg="$(mktemp)"
  cargo build --release -p bench-fandhe --target-dir "$tdir" --message-format=json \
    --config "patch.crates-io.fandhe-ai.path=\"${facade}\"" >"$msg" 2>"$OUT/build-$arm.err"
  exe="$(jq -rs '[.[] | select(.reason == "compiler-artifact" and .target.name == "bench-fandhe" and (.target.kind[]? == "bin") and .executable != null)] | last | .executable // empty' "$msg")"
  rm -f "$msg"
  [[ -n "$exe" && -f "$exe" ]] || { echo "build $arm: exe not found" >&2; exit 1; }
  cp "$exe" "$OUT/bench-fandhe-$arm"
  cargo tree -p bench-fandhe --depth 1 --config "patch.crates-io.fandhe-ai.path=\"${facade}\"" | grep -E '^[│├└─ ]*fandhe-ai ' >"$OUT/tree-$arm.txt"
  shasum -a 256 "$OUT/bench-fandhe-$arm" >"$OUT/sha-$arm.txt"
}

if [[ "${PHASE:-all}" == "build" || "${PHASE:-all}" == "all" ]]; then
  echo "== build before =="; build_arm before "$BEFORE_FACADE" "$FC/target-ab-1555-before"
  echo "== build after  =="; build_arm after  "$AFTER_FACADE"  "$FC/target-ab-1555-after"
  cat "$OUT"/tree-*.txt
fi
[[ "${PHASE:-all}" == "build" ]] && exit 0

run_train() { # run_train <arm> <mode> [--phases]
  local arm=$1 mode=$2 extra=${3:-} suffix=train
  [[ -n "$extra" ]] && suffix=phases
  echo "== train metal mode=$mode arm=$arm $extra =="
  "$OUT/bench-fandhe-$arm" --task train --device metal --mode "$mode" $extra --out "$OUT/results-$arm-$LABEL-$suffix.jsonl"
}
uptime >>"$OUT/uptime.log"; pmset -g therm >"$OUT/pmset_therm_before.txt" 2>&1 || true
for run_i in $(seq 1 "$ROUNDS"); do
  for mode in fresh reuse; do
    if (( run_i % 2 == 1 )); then
      run_train before "$mode"; run_train after "$mode"
    else
      run_train after "$mode"; run_train before "$mode"
    fi
  done
  echo "run $run_i: $(uptime)" | tee -a "$OUT/uptime.log"
done
# --phases は診断用（compare_gemm_ab.py の診断表は phase ごとに 1 行を採るため各腕 1 回。#1548 と同型）
for mode in fresh reuse; do
  run_train before "$mode" --phases; run_train after "$mode" --phases
done
echo "phases: $(uptime)" | tee -a "$OUT/uptime.log"
pmset -g therm >"$OUT/pmset_therm_after.txt" 2>&1 || true
python3 compare_gemm_ab.py --task train --threshold 1.00 --per-run --phases \
  "$OUT/results-before-$LABEL-phases.jsonl" "$OUT/results-after-$LABEL-phases.jsonl" \
  "$OUT/results-before-$LABEL-train.jsonl" "$OUT/results-after-$LABEL-train.jsonl" \
  >"$OUT/compare-train-$LABEL.md" 2>"$OUT/compare-train-$LABEL.err" || echo "compare exit=$?" | tee -a "$OUT/compare-train-$LABEL.err"
echo DONE_AB
