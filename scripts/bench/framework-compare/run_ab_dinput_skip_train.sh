#!/usr/bin/env bash
# イシュー #3247: #3246（d_input GEMM の省略）の前後を train セルで A/B する両機体共用ランナー。
# 規則の正本: docs/perf/logs/dinput-skip-ab-3247/RULE.txt（計測前に固定。本スクリプトはその
# sha256 を env_info へ記録し、judge_dinput_skip_ab.py が現 RULE と突合する）。
#
# before／after を別ツリーの `crates/facade` への `[patch.crates-io.fandhe-ai]` path patch
# （`--config` 引数のみ。Cargo.toml／Cargo.lock へはコミットしない）で別々にビルドする。両腕は
# 同じ runner ツリーの bench ソース（`bench-var-no-grad.patch` 適用済み）を使うため、違いは
# facade だけになる。5 round・round ごとに起動順を反転。判定は judge 側で行う。
#
# 呼び出し例:
#   AB_PROFILE=macmini AB_BEFORE_FACADE_PATH=<abs>/before/crates/facade \
#   AB_AFTER_FACADE_PATH=<abs>/after/crates/facade \
#   AB_ROUND_COOLDOWN_MAX_LOAD1=3.0 bash run_ab_dinput_skip_train.sh 3247-macmini
# bash 3.2 互換（mapfile・連想配列・eval 不使用）。jq 不使用。失敗は捏造せず skipped へ記録し非 0 終了。
set -u
cd "$(dirname "$0")" || exit 1
# shellcheck source=./bench_fandhe_lock_restore.sh
source ./bench_fandhe_lock_restore.sh
# shellcheck source=./ab_round_cooldown.sh
source ./ab_round_cooldown.sh

LABEL=${1:-}
ROUNDS=${AB_ROUNDS:-5}
PROFILE=${AB_PROFILE:-}

if [[ -z "$LABEL" || ! "$LABEL" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "usage: AB_PROFILE=macmini|gb10 $0 <label>  (label: [A-Za-z0-9._-]+)" >&2
  exit 1
fi
if [[ "$ROUNDS" != "5" ]]; then
  echo "error: AB_ROUNDS は 5 固定（RULE.txt。got: ${ROUNDS}）" >&2
  exit 1
fi
case "$PROFILE" in
  macmini) DEVICES="cpu metal"; RULE_LOAD_GATE=3.0 ;;
  gb10) DEVICES="cpu cuda"; RULE_LOAD_GATE=1.0 ;;
  *) echo "error: AB_PROFILE は macmini か gb10（got: ${PROFILE:-unset}）" >&2; exit 1 ;;
esac

validate_facade_path() { # <var_name> <path>
  local var_name=$1 path=$2
  if [[ -z "$path" || "$path" != /* ]]; then
    echo "error: ${var_name} は絶対パス必須（got: ${path:-unset}）" >&2; exit 1
  fi
  # --config 文字列へ埋め込むため引用符・バックスラッシュ・改行を拒否する（A03）。
  if [[ "$path" == *\"* || "$path" == *\\* || "$path" == *$'\n'* ]]; then
    echo "error: ${var_name} に使えない文字を含む" >&2; exit 1
  fi
  if [[ ! -f "$path/Cargo.toml" ]] || ! grep -qE '^[[:space:]]*name[[:space:]]*=[[:space:]]*"fandhe-ai"[[:space:]]*$' "$path/Cargo.toml"; then
    echo "error: ${var_name}/Cargo.toml が name = \"fandhe-ai\" を宣言していない" >&2; exit 1
  fi
}
validate_facade_path AB_BEFORE_FACADE_PATH "${AB_BEFORE_FACADE_PATH:-}"
validate_facade_path AB_AFTER_FACADE_PATH "${AB_AFTER_FACADE_PATH:-}"
BEFORE_FACADE=$AB_BEFORE_FACADE_PATH
AFTER_FACADE=$AB_AFTER_FACADE_PATH
BEFORE_ROOT="$(cd "$BEFORE_FACADE/../.." && pwd)"
AFTER_ROOT="$(cd "$AFTER_FACADE/../.." && pwd)"
RUNNER_ROOT="$(cd ../../.. && pwd)"
RULE_FILE="$RUNNER_ROOT/docs/perf/logs/dinput-skip-ab-3247/RULE.txt"
PATCH_FILE="$RUNNER_ROOT/docs/perf/logs/dinput-skip-ab-3247/bench-var-no-grad.patch"
MAIN_RS="bench-fandhe/src/main.rs"

# 機体同定（fail-closed）。Mac mini は M4 完全一致のみ（M4 Max 開発機の値を混ぜない）。
if [[ "$PROFILE" == "macmini" ]]; then
  brand="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || true)"
  if [[ "$brand" != "Apple M4" ]]; then
    echo "error: macmini プロファイルは 'Apple M4' のみ（got: ${brand:-unknown}）" >&2; exit 1
  fi
else
  if [[ "$(uname -m)" != "aarch64" ]]; then echo "error: gb10 は aarch64 のみ" >&2; exit 1; fi
fi

sha256_of() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'; else shasum -a 256 "$1" | awk '{print $1}'; fi; }
[[ -f "$RULE_FILE" && -f "$PATCH_FILE" ]] || { echo "error: RULE.txt / patch が無い" >&2; exit 1; }
RULE_SHA="$(sha256_of "$RULE_FILE")"
PATCH_SHA="$(sha256_of "$PATCH_FILE")"

OUT="results/raw"
mkdir -p "$OUT"
SKIP="$OUT/skipped-dinput-skip-${LABEL}.log"
ENV_INFO="$OUT/env_info-${LABEL}.txt"
# 同じラベルの既存出力があれば停止（run の差し替え・追記禁止。A08）。
if [[ -e "$SKIP" || -e "$ENV_INFO" ]]; then
  echo "error: ラベル ${LABEL} の出力が既に存在する。新しいラベルを使うこと" >&2; exit 1
fi
: >"$SKIP"
ANY_FAILED=0
unset RAYON_NUM_THREADS

# ドリフトガード: 各ツリー直下の `.rev-stamp` 1 行目が実 HEAD と一致すること。
read_rev_stamp() { if [[ -f "$1/.rev-stamp" ]]; then head -n1 "$1/.rev-stamp"; fi; }
STAMP_BEFORE="$(read_rev_stamp "$BEFORE_ROOT")"
STAMP_AFTER="$(read_rev_stamp "$AFTER_ROOT")"
STAMP_RUNNER="$(read_rev_stamp "$RUNNER_ROOT")"
check_stamp() { # <root> <stamp>
  local real
  real="$(git -C "$1" rev-parse HEAD 2>/dev/null || true)"
  if [[ -z "$2" || "$2" != "$real" ]]; then
    echo "error: $1 の .rev-stamp が無い、または HEAD と不一致（stamp=${2:-none} head=${real:-none}）" >&2; exit 1
  fi
}
# RULE.txt が登録した before／after の SHA（`- before = <40hex>` 行）を正として両腕 HEAD を照合し、
# 追跡ファイルの未コミット変更（腕が登録外コミット・改変になる事故）も拒否する。
rule_sha_of() { grep -E "^- $1[[:space:]]*=[[:space:]]*[0-9a-f]{40}" "$RULE_FILE" | head -n1 | grep -oE '[0-9a-f]{40}' | head -n1; }
EXPECT_BEFORE="$(rule_sha_of before)"
EXPECT_AFTER="$(rule_sha_of after)"
if [[ -z "$EXPECT_BEFORE" || -z "$EXPECT_AFTER" ]]; then
  echo "error: RULE.txt から before/after の SHA を読めない" >&2; exit 1
fi
check_arm() { # <root> <expected_sha>
  local real dirty
  real="$(git -C "$1" rev-parse HEAD 2>/dev/null || true)"
  if [[ "$real" != "$2" ]]; then
    echo "error: $1 の HEAD が RULE.txt の登録 SHA と不一致（head=${real:-none} expected=$2）" >&2; exit 1
  fi
  dirty="$(git -C "$1" status --porcelain --untracked-files=no 2>/dev/null || echo unknown)"
  if [[ -n "$dirty" ]]; then
    echo "error: $1 の追跡ファイルに未コミット変更がある" >&2; exit 1
  fi
}
check_arm "$BEFORE_ROOT" "$EXPECT_BEFORE"
check_arm "$AFTER_ROOT" "$EXPECT_AFTER"
check_stamp "$BEFORE_ROOT" "$STAMP_BEFORE"
check_stamp "$AFTER_ROOT" "$STAMP_AFTER"
check_stamp "$RUNNER_ROOT" "$STAMP_RUNNER"

# precheck（fail-closed。RULE.txt [precheck]）。
precheck_fail() { echo "precheck FAILED: $1" | tee -a "$SKIP" >&2; exit 1; }
grep -qE 'input_requires_grad' "$AFTER_FACADE/../autodiff/src/grad.rs" || precheck_fail "after に input_requires_grad の分岐が無い"
if grep -qE 'input_requires_grad' "$BEFORE_FACADE/../autodiff/src/grad.rs"; then precheck_fail "before に input_requires_grad の分岐がある"; fi
n_nograd="$(grep -cE '^[[:space:]]*let x = tape\.var_no_grad\(&x_data\);' "$MAIN_RS")"
n_var="$(grep -cE '^[[:space:]]*let x = tape\.var\(&x_data\);' "$MAIN_RS")"
[[ "$n_nograd" == "2" ]] || precheck_fail "var_no_grad 行が 2 件でない（${n_nograd}）。パッチ未適用の可能性"
# fresh 2 件（run_train／measure_train_phases）のほか infer 系が別にあるため下限のみ検査する。
[[ "$n_var" -ge 2 ]] || precheck_fail "fresh の var 行が不足（${n_var}）"
status_lines="$(git -C "$RUNNER_ROOT" status --porcelain | grep -v '^??' || true)"
[[ "$status_lines" == " M scripts/bench/framework-compare/bench-fandhe/src/main.rs" ]] \
  || precheck_fail "runner ツリーの変更が main.rs 1 件だけでない（${status_lines}）"
git -C "$RUNNER_ROOT" apply --reverse --check "$PATCH_FILE" || precheck_fail "patch が runner ツリーと一致しない"
echo "precheck=assert ok"

bench_fandhe_setup_lock_restore_trap

build_arm() { # <arm> <facade>
  local arm=$1 facade=$2 tdir="target-ab-dinput-skip-${LABEL}-$1" cfg tree
  cfg="patch.crates-io.fandhe-ai.path=\"${facade}\""
  if ! cargo build --release -p bench-fandhe --target-dir "$tdir" --config "$cfg" 2>"$OUT/build-${arm}-${LABEL}.err"; then
    tail -40 "$OUT/build-${arm}-${LABEL}.err"
    echo "bench-fandhe BUILD FAILED (${arm})" >>"$SKIP"; exit 1
  fi
  tree="$(cargo tree -p bench-fandhe --depth 1 --config "$cfg" 2>&1)"
  if ! echo "$tree" | grep -qE 'fandhe-ai v[0-9.]+ \(.*crates/facade\)'; then
    echo "error: fandhe-ai が path patch に解決されていない (${arm})" >&2; echo "$tree" >&2; exit 1
  fi
  echo "$tree" >"$OUT/tree-${arm}-${LABEL}.txt"
  cp "$tdir/release/bench-fandhe" "$OUT/bench-fandhe-${arm}-${LABEL}"
  echo "$(sha256_of "$OUT/bench-fandhe-${arm}-${LABEL}")  bench-fandhe-${arm}-${LABEL}" >"$OUT/sha-${arm}-${LABEL}.txt"
}
# 全ビルドを計測の前に済ませる（計測中のビルド並走禁止）。
build_arm before "$BEFORE_FACADE"
build_arm after "$AFTER_FACADE"
BIN_BEFORE="$OUT/bench-fandhe-before-${LABEL}"
BIN_AFTER="$OUT/bench-fandhe-after-${LABEL}"

{
  echo "profile=${PROFILE}"
  echo "host=masked"
  echo "uname=$(uname -srm)"
  if [[ "$PROFILE" == "macmini" ]]; then echo "cpu_brand=$(sysctl -n machdep.cpu.brand_string)"; else echo "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)"; fi
  echo "rule_sha256=${RULE_SHA}"
  echo "patch_sha256=${PATCH_SHA}"
  echo "rev_stamp_head_before=${STAMP_BEFORE}"
  echo "rev_stamp_head_after=${STAMP_AFTER}"
  echo "rev_stamp_head_runner=${STAMP_RUNNER}"
  echo "bin_sha256_before=$(awk '{print $1}' "$OUT/sha-before-${LABEL}.txt")"
  echo "bin_sha256_after=$(awk '{print $1}' "$OUT/sha-after-${LABEL}.txt")"
  echo "precheck=assert"
  echo "rounds=${ROUNDS}"
} >"$ENV_INFO"

# 専有ゲート。GB10 は load1<1.0（CUDA はさらに GPU 使用率 0% を 3 サンプル連続）、Mac mini は load1<3.0。
gate_ok_once() { # <dev> -> 0/1
  local l1 util n
  l1="$(ab_cooldown_read_load1)"
  [[ "$l1" =~ ^[0-9]+(\.[0-9]+)?$ ]] || return 1
  [[ "$(awk -v a="$l1" -v t="$RULE_LOAD_GATE" 'BEGIN{print (a<t)?1:0}')" == "1" ]] || return 1
  if [[ "$PROFILE" == "gb10" && "$1" == "cuda" ]]; then
    n=0
    while [[ "$n" -lt 3 ]]; do
      util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ')"
      [[ "$util" == "0" ]] || return 1
      n=$((n + 1))
      if [[ "$n" -lt 3 ]]; then sleep 2; fi
    done
  fi
  return 0
}
wait_gate() { # <dev> <round> <logfile>
  local waited=0 ok=0
  while :; do
    if gate_ok_once "$1"; then ok=1; break; fi
    if [[ "$waited" -ge 1800 ]]; then break; fi
    sleep 30; waited=$((waited + 30))
  done
  if [[ "$ok" == "1" ]]; then
    echo "round$2 dev=$1 gate=pass gate_ok=1 threshold=${RULE_LOAD_GATE} waited_s=${waited} load1=$(ab_cooldown_read_load1)" >>"$3"
  else
    echo "round$2 dev=$1 gate=fail gate_ok=0 threshold=${RULE_LOAD_GATE} waited_s=${waited} load1=$(ab_cooldown_read_load1)" >>"$3"
  fi
}

run_one() { # <dev> <arm> <mode> <round> [--phases]
  local dev=$1 arm=$2 mode=$3 r=$4 extra=${5:-} bin suffix=train errf
  if [[ -n "$extra" ]]; then suffix=phases; fi
  if [[ "$arm" == "before" ]]; then bin=$BIN_BEFORE; else bin=$BIN_AFTER; fi
  errf="$OUT/digest-${arm}-${LABEL}-${dev}-${mode}-r${r}.log"
  if [[ -n "$extra" ]]; then errf="$OUT/stderr-phases-${arm}-${LABEL}-${dev}-${mode}.log"; fi
  echo "== train ${dev} mode=${mode} arm=${arm} round=${r} ${extra} =="
  if ! BENCH_PARAM_DIGEST=1 "$bin" --task train --device "$dev" --mode "$mode" ${extra:+$extra} \
      --out "$OUT/results-${arm}-${LABEL}-${dev}-${suffix}.jsonl" 2>"$errf"; then
    echo "dev=${dev} arm=${arm} mode=${mode} round=${r} extra=${extra:-none}: $(tr '\n' ' ' <"$errf")" >>"$SKIP"
    ANY_FAILED=$((ANY_FAILED + 1))
  fi
}

pm_log() { if [[ "$PROFILE" == "macmini" ]]; then pmset -g therm >"$OUT/pmset_therm_$1-${LABEL}.txt" 2>&1 || true; fi; }
pm_log before
for dev in $DEVICES; do
  for arm in before after; do
    : >"$OUT/results-${arm}-${LABEL}-${dev}-train.jsonl"
    : >"$OUT/results-${arm}-${LABEL}-${dev}-phases.jsonl"
  done
  GATE_LOG="$OUT/load_gate-${LABEL}-${dev}.log"
  echo "series=official threshold=${RULE_LOAD_GATE} rule_threshold=${RULE_LOAD_GATE} precheck=assert" >"$GATE_LOG"
  ab_round_cooldown_init "$OUT/cooldown-${LABEL}-${dev}.log"
  for r in 1 2 3 4 5; do
    ab_round_cooldown "$r"
    wait_gate "$dev" "$r" "$GATE_LOG"
    for mode in fresh reuse; do
      if (( r % 2 == 1 )); then
        run_one "$dev" before "$mode" "$r"; run_one "$dev" after "$mode" "$r"
      else
        run_one "$dev" after "$mode" "$r"; run_one "$dev" before "$mode" "$r"
      fi
    done
    echo "dev=${dev} run ${r}: $(uptime)" | tee -a "$OUT/uptime-${LABEL}.log"
  done
  for mode in fresh reuse; do
    run_one "$dev" before "$mode" 0 --phases
    run_one "$dev" after "$mode" 0 --phases
  done
  for mode in reuse fresh; do
    python3 compare_gemm_ab.py --device "$dev" --task train --threshold 1.00 --per-run --modes "$mode" \
      --require-checksum-exact --phases \
      "$OUT/results-before-${LABEL}-${dev}-phases.jsonl" "$OUT/results-after-${LABEL}-${dev}-phases.jsonl" \
      "$OUT/results-before-${LABEL}-${dev}-train.jsonl" "$OUT/results-after-${LABEL}-${dev}-train.jsonl" \
      >"compare-train-${LABEL}-${dev}-${mode}.md" 2>"compare-train-${LABEL}-${dev}-${mode}.err"
    rc=$?
    # 終了コードは参考。最終判定は judge_dinput_skip_ab.py（RULE 準拠）が行う。
    echo "compare dev=${dev} mode=${mode} exit=${rc}" | tee -a "compare-train-${LABEL}-${dev}-${mode}.err"
  done
done
pm_log after
echo "done. results in ${OUT}; skipped log: ${SKIP}"
if [[ "$ANY_FAILED" -gt 0 ]]; then echo "FAILED: ${ANY_FAILED} run(s)" >&2; exit 1; fi
