#!/usr/bin/env bash
# イシュー #1577 framework-compare train A/B（reuse backward の ReLU
# マスク stride 対応）。before = origin/main の crates/facade（git
# archive で取り出した非 git ツリー・path patch）、after = 本ブランチ
# worktree の crates/facade。5 round・run 単位で before/after を順序
# 反転・record_only（専有ゲートなし・uptime 記録のみ）。
set -euo pipefail
FC=${FC:?}
BEFORE_FACADE=${BEFORE_FACADE:?}
AFTER_FACADE=${AFTER_FACADE:?}
OUT=${OUT:?}
LABEL=${LABEL:-1577}
ROUNDS=${ROUNDS:-5}
DEVICE=${DEVICE:?}
case "$DEVICE" in
  cpu|metal|cuda) ;;
  *) echo "invalid DEVICE: $DEVICE" >&2; exit 1 ;;
esac
mkdir -p "$OUT"
cd "$FC"
LOCK_BACKUP="$(mktemp)"; cp Cargo.lock "$LOCK_BACKUP"
trap 'cp "$LOCK_BACKUP" "$FC/Cargo.lock"; rm -f "$LOCK_BACKUP"' EXIT

build_arm() {
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
  echo "== build before =="; build_arm before "$BEFORE_FACADE" "$FC/target-ab-1577-before"
  echo "== build after  =="; build_arm after  "$AFTER_FACADE"  "$FC/target-ab-1577-after"
  cat "$OUT"/tree-*.txt
fi
[[ "${PHASE:-all}" == "build" ]] && exit 0

run_train() {
  local arm=$1 mode=$2 extra=${3:-} suffix=train
  [[ -n "$extra" ]] && suffix=phases
  echo "== train $DEVICE mode=$mode arm=$arm $extra =="
  "$OUT/bench-fandhe-$arm" --task train --device "$DEVICE" --mode "$mode" $extra --out "$OUT/results-$arm-$LABEL-$suffix.jsonl"
}
uptime >>"$OUT/uptime.log"
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
for mode in fresh reuse; do
  run_train before "$mode" --phases; run_train after "$mode" --phases
done
echo "phases: $(uptime)" | tee -a "$OUT/uptime.log"
python3 compare_gemm_ab.py --task train --device "$DEVICE" --threshold 1.00 --per-run --phases \
  "$OUT/results-before-$LABEL-phases.jsonl" "$OUT/results-after-$LABEL-phases.jsonl" \
  "$OUT/results-before-$LABEL-train.jsonl" "$OUT/results-after-$LABEL-train.jsonl" \
  >"$OUT/compare-train-$LABEL-$DEVICE.md" 2>"$OUT/compare-train-$LABEL-$DEVICE.err" || echo "compare exit=$?" | tee -a "$OUT/compare-train-$LABEL-$DEVICE.err"
echo DONE_AB
