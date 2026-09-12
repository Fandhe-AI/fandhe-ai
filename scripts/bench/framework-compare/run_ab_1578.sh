#!/usr/bin/env bash
# イシュー #1578: mse_loss_backward の要素数しきい値による逐次
# フォールバック（`MSE_BACKWARD_PARALLEL_MIN_ELEMS`）の framework-compare
# train A/B。before = origin/main（マージ base）の `crates/facade`、
# after = 本ブランチの `crates/facade` を、それぞれ
# `[patch.crates-io.fandhe-ai]` path patch した 2 本の `bench-fandhe`
# バイナリとして CPU device 限定でビルドし、5 round・run 単位で起動順を
# 反転しながら交互実行する（`run_ab_resident_grad_cuda.sh` の CPU 移植・
# 簡略版。GPU 依存〈nvidia-smi／pmset〉は持たない）。
#
# `[patch]` は本スクリプトの CLI 引数（環境変数）としてのみ与え、
# `scripts/bench/framework-compare/Cargo.toml`／`Cargo.lock`へは
# コミットしない（deps-policy.md 第 9 区分は registry 取得元のみを許容）。
# `Cargo.lock` は `bench_fandhe_lock_restore.sh` の共有ヘルパーで
# 退避・EXIT trap 復元する。
#
# 呼び出し例:
#   AB_BEFORE_FACADE_PATH=/path/to/before/crates/facade \
#   AB_AFTER_FACADE_PATH=/path/to/after/crates/facade \
#     bash run_ab_1578.sh 1578
set -u
cd "$(dirname "$0")"
# shellcheck source=./bench_fandhe_lock_restore.sh
source ./bench_fandhe_lock_restore.sh

LABEL=${1:-}
ROUNDS=${AB_ROUNDS:-5}

# A03 インジェクション対策: ラベルはファイル名・パスに直接埋め込むため、
# 英数字・`._-` のみを許可する allowlist で検証する。
if [[ -z "$LABEL" || ! "$LABEL" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "usage: $0 <label>  (label must match [A-Za-z0-9._-]+, e.g. 1578)" >&2
  echo "  env AB_BEFORE_FACADE_PATH / AB_AFTER_FACADE_PATH (absolute paths) are required" >&2
  exit 1
fi

validate_facade_path() {
  local var_name=$1 path=$2
  if [[ -z "$path" ]]; then
    echo "error: $var_name is required (absolute path to a crates/facade checkout; issue #1578)" >&2
    exit 1
  fi
  if [[ "$path" != /* ]]; then
    echo "error: $var_name must be an absolute path (got: $path)" >&2
    exit 1
  fi
  if [[ ! -f "$path/Cargo.toml" ]]; then
    echo "error: $var_name/Cargo.toml not found ($path)" >&2
    exit 1
  fi
  if ! grep -qE '^\s*name\s*=\s*"fandhe-ai"\s*$' "$path/Cargo.toml"; then
    echo "error: $var_name/Cargo.toml does not declare name = \"fandhe-ai\" ($path)" >&2
    exit 1
  fi
}
validate_facade_path AB_BEFORE_FACADE_PATH "${AB_BEFORE_FACADE_PATH:-}"
validate_facade_path AB_AFTER_FACADE_PATH "${AB_AFTER_FACADE_PATH:-}"
BEFORE_FACADE="$AB_BEFORE_FACADE_PATH"
AFTER_FACADE="$AB_AFTER_FACADE_PATH"

# 判定対象デバイス（既定 cpu）。ガードセル計測（metal／cuda）にも
# 同一スクリプトを流用できるよう環境変数で切り替える。
DEVICE=${AB_DEVICE:-cpu}
case "$DEVICE" in
  cpu | metal | cuda) ;;
  *)
    echo "error: AB_DEVICE must be one of cpu|metal|cuda (got: $DEVICE)" >&2
    exit 1
    ;;
esac

OUT="results/raw"
mkdir -p "$OUT"
SKIP="$OUT/skipped-1578-${DEVICE}-${LABEL}.log"
: >"$SKIP"
ANY_FAILED=0

for _reset_arm in before after; do
  for _reset_suffix in train phases; do
    : >"$OUT/results-${_reset_arm}-${LABEL}-${DEVICE}-${_reset_suffix}.jsonl"
  done
done

bench_fandhe_setup_lock_restore_trap

build_arm() { # build_arm <arm> <facade_path> <target_dir>
  local arm=$1 facade=$2 tdir=$3 patch_config exe msg
  patch_config="patch.crates-io.fandhe-ai.path=\"${facade}\""
  msg="$(mktemp)"
  if ! cargo build --release -p bench-fandhe --target-dir "$tdir" --message-format=json \
    --config "$patch_config" >"$msg" 2>"$OUT/build-${arm}-${LABEL}-${DEVICE}.err"; then
    tail -40 "$OUT/build-${arm}-${LABEL}-${DEVICE}.err"
    echo "bench-fandhe BUILD FAILED ($arm): $(tail -3 "$OUT/build-${arm}-${LABEL}-${DEVICE}.err" | tr '\n' ' ')" >>"$SKIP"
    rm -f "$msg"
    exit 1
  fi
  exe="$(jq -rs '[.[] | select(.reason == "compiler-artifact" and .target.name == "bench-fandhe" and (.target.kind[]? == "bin") and .executable != null)] | last | .executable // empty' "$msg")"
  rm -f "$msg"
  if [[ -z "$exe" || ! -f "$exe" ]]; then
    echo "error: build $arm: exe not found" >&2
    exit 1
  fi
  cp "$exe" "$OUT/bench-fandhe-1578-${arm}-${LABEL}"
  local tree_output
  tree_output="$(cargo tree -p bench-fandhe --depth 1 --config "$patch_config" 2>&1)"
  if ! echo "$tree_output" | grep -qE 'fandhe-ai v[0-9.]+ \(.*crates/facade\)'; then
    echo "error: fandhe-ai did not resolve to the path-patched crates/facade ($arm); cargo tree:" >&2
    echo "$tree_output" >&2
    exit 1
  fi
  echo "$tree_output" >"$OUT/tree-1578-${arm}-${LABEL}.txt"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$OUT/bench-fandhe-1578-${arm}-${LABEL}" >"$OUT/sha-1578-${arm}-${LABEL}.txt"
  else
    shasum -a 256 "$OUT/bench-fandhe-1578-${arm}-${LABEL}" >"$OUT/sha-1578-${arm}-${LABEL}.txt"
  fi
}

echo "== build before ==(facade=$BEFORE_FACADE)"
build_arm before "$BEFORE_FACADE" "target-ab-1578-${LABEL}-before"
echo "== build after  ==(facade=$AFTER_FACADE)"
build_arm after "$AFTER_FACADE" "target-ab-1578-${LABEL}-after"
cat "$OUT"/tree-1578-*-"${LABEL}".txt

BIN_BEFORE="$OUT/bench-fandhe-1578-before-${LABEL}"
BIN_AFTER="$OUT/bench-fandhe-1578-after-${LABEL}"

run_train() { # run_train <arm> <mode> [--phases]
  local arm=$1 mode=$2 extra=${3:-} suffix=train bin
  [[ -n "$extra" ]] && suffix=phases
  if [[ "$arm" == "before" ]]; then bin="$BIN_BEFORE"; else bin="$BIN_AFTER"; fi
  echo "== train $DEVICE mode=$mode arm=$arm ${extra:-} =="
  if ! "$bin" --task train --size 64 --device "$DEVICE" --mode "$mode" ${extra:+$extra} \
    --out "$OUT/results-${arm}-${LABEL}-${DEVICE}-${suffix}.jsonl" 2>"$OUT/err-${arm}-${LABEL}-${DEVICE}.tmp"; then
    echo "arm=$arm mode=$mode extra=${extra:-none} : $(cat "$OUT/err-${arm}-${LABEL}-${DEVICE}.tmp")" >>"$SKIP"
    echo "  -> FAILED (recorded in $SKIP)"
    ANY_FAILED=$((ANY_FAILED + 1))
  fi
  rm -f "$OUT/err-${arm}-${LABEL}-${DEVICE}.tmp"
}

: >"$OUT/uptime-1578-${DEVICE}-${LABEL}.log"
uptime >>"$OUT/uptime-1578-${DEVICE}-${LABEL}.log"

for run_i in $(seq 1 "$ROUNDS"); do
  for mode in fresh reuse; do
    if ((run_i % 2 == 1)); then
      run_train before "$mode"
      run_train after "$mode"
    else
      run_train after "$mode"
      run_train before "$mode"
    fi
  done
  echo "run $run_i: $(uptime)" | tee -a "$OUT/uptime-1578-${DEVICE}-${LABEL}.log"
done

for mode in fresh reuse; do
  run_train before "$mode" "--phases"
  run_train after "$mode" "--phases"
done
echo "phases: $(uptime)" | tee -a "$OUT/uptime-1578-${DEVICE}-${LABEL}.log"

# 事前登録した受け入れ契約は reuse のみを必須判定とする（fresh は対照／参考）。
python3 compare_gemm_ab.py --device "$DEVICE" --task train --threshold 1.00 --per-run --modes reuse \
  --require-checksum-exact --phases \
  "$OUT/results-before-${LABEL}-${DEVICE}-phases.jsonl" "$OUT/results-after-${LABEL}-${DEVICE}-phases.jsonl" \
  "$OUT/results-before-${LABEL}-${DEVICE}-train.jsonl" "$OUT/results-after-${LABEL}-${DEVICE}-train.jsonl" \
  >"compare-train-1578-${DEVICE}.md" 2>"compare-train-1578-${DEVICE}.err"
COMPARE_EXIT=$?
if [[ "$COMPARE_EXIT" -ne 0 ]]; then
  echo "compare exit=$COMPARE_EXIT" | tee -a "compare-train-1578-${DEVICE}.err"
  ANY_FAILED=$((ANY_FAILED + 1))
fi

python3 compare_gemm_ab.py --device "$DEVICE" --task train --threshold 1.00 --per-run --modes fresh --phases \
  "$OUT/results-before-${LABEL}-${DEVICE}-phases.jsonl" "$OUT/results-after-${LABEL}-${DEVICE}-phases.jsonl" \
  "$OUT/results-before-${LABEL}-${DEVICE}-train.jsonl" "$OUT/results-after-${LABEL}-${DEVICE}-train.jsonl" \
  >"compare-train-1578-${DEVICE}-fresh-reference.md" 2>"compare-train-1578-${DEVICE}-fresh-reference.err" ||
  echo "fresh reference compare exit=$? (informational only; not part of the judged A/B)" \
    | tee -a "compare-train-1578-${DEVICE}-fresh-reference.err"

echo "done. results in $OUT ; failures (if any) in $SKIP"
if [[ "$ANY_FAILED" -gt 0 ]]; then
  echo "FAILED: $ANY_FAILED run(s) failed; see $SKIP" >&2
  exit 1
fi
