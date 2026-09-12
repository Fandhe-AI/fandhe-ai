#!/usr/bin/env bash
# イシュー #1563: Metal `Op::LinearResident` VJP の d_weight（encode-
# only）を d_input（`gemm_resident_lhs`。同期点を持つ）より前へ移し、
# 同じ層の d_weight コマンドを d_input の同期点へ合流させた変更
# （層内合流。`docs/backend-metal-command-batching-design.md` §7.4）の
# framework-compare train A/B。
#
# before = 本ブランチのベース（origin/main。#1563 適用前）、after = 本
# ブランチ（`crates/*/src` は評価対象コミットと同一）を、それぞれ別
# ツリーへ展開した `crates/facade` への `[patch.crates-io.fandhe-ai]`
# path patch で 2 本の `bench-fandhe` バイナリとしてビルドし、5 round・
# run 単位で起動順を反転しながら交互実行する（CUDA 版 #1560
# `run_ab_resident_grad_cuda.sh` の Metal 移植。nvidia-smi → pmset -g
# therm、`--device cuda` → `--device metal` に置換）。
#
# `[patch]` は本スクリプトの CLI 引数（`--config`）としてのみ与え、
# `scripts/bench/framework-compare/Cargo.toml`／`Cargo.lock`／
# `.cargo/config.toml` へはコミットしない（deps-policy.md 第 9 区分は
# registry 取得元のみを許容するため）。`Cargo.lock` は
# `bench_fandhe_lock_restore.sh` の共有ヘルパーで退避・EXIT trap 復元する。
#
# 呼び出し例（M4 Max 実機。ユーザー承認・別セッション）:
#   AB_BEFORE_FACADE_PATH=/home/<user>/work/rust-ai-library-run-1563-before/crates/facade \
#   AB_AFTER_FACADE_PATH=/home/<user>/work/rust-ai-library-run-1563-after/crates/facade \
#     bash run_ab_dinput_sync_metal.sh 1563
#
# 出力は他の run_ab_*.sh と同じ「失敗を捏造しない」方針（skipped ログへ
# 記録・非 0 終了。security.md A08）。専有ゲートは設けず record_only
# 運用（イシュー #1519 系のユーザー指示に従う）。
set -u
cd "$(dirname "$0")"
# shellcheck source=./bench_fandhe_lock_restore.sh
source ./bench_fandhe_lock_restore.sh

LABEL=${1:-}
ROUNDS=${AB_ROUNDS:-5}

# A03 インジェクション対策: ラベルはファイル名・パスに直接埋め込むため、
# 英数字・`._-` のみを許可する allowlist で検証する。
if [[ -z "$LABEL" || ! "$LABEL" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "usage: $0 <label>  (label must match [A-Za-z0-9._-]+, e.g. 1563)" >&2
  echo "  env AB_BEFORE_FACADE_PATH / AB_AFTER_FACADE_PATH (absolute paths) are required" >&2
  exit 1
fi

# `AB_{BEFORE,AFTER}_FACADE_PATH` の検証（A03・A08）: 未設定・相対パス・
# Cargo.toml 不在・crate 名不一致のいずれかなら fail-closed で exit 1。
validate_facade_path() {
  local var_name=$1 path=$2
  if [[ -z "$path" ]]; then
    echo "error: $var_name is required (absolute path to a crates/facade checkout; issue #1563)" >&2
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

OUT="results/raw"
mkdir -p "$OUT"
SKIP="$OUT/skipped-metal-dinput-sync-ab-${LABEL}.log"
: > "$SKIP"
ANY_FAILED=0

# `run_train` の出力先（bench-common が `OpenOptions::append(true)` で
# 書き込む）を実行開始時に必ず空へ初期化する。append 方式のため、
# 中断後の再実行や同一ラベルでの再実行があると過去の計測行が残存し、
# `compare_gemm_ab.py` の「各セルちょうど 5 件」検証が判定不能になる
# （`run_ab_resident_grad_cuda.sh` と同型の codex-review 指摘対応）。
for _reset_arm in before after; do
  for _reset_suffix in train phases; do
    : > "$OUT/results-${_reset_arm}-${LABEL}-${_reset_suffix}.jsonl"
  done
done

# `Cargo.lock` を退避し、異常終了含め終了時に必ず復元する。
bench_fandhe_setup_lock_restore_trap

build_arm() { # build_arm <arm> <facade_path> <target_dir>
  local arm=$1 facade=$2 tdir=$3 patch_config exe msg
  patch_config="patch.crates-io.fandhe-ai.path=\"${facade}\""
  msg="$(mktemp)"
  if ! cargo build --release -p bench-fandhe --target-dir "$tdir" --message-format=json \
      --config "$patch_config" >"$msg" 2>"$OUT/build-${arm}-${LABEL}.err"; then
    tail -40 "$OUT/build-${arm}-${LABEL}.err"
    echo "bench-fandhe BUILD FAILED ($arm): $(tail -3 "$OUT/build-${arm}-${LABEL}.err" | tr '\n' ' ')" >>"$SKIP"
    rm -f "$msg"
    exit 1
  fi
  exe="$(jq -rs '[.[] | select(.reason == "compiler-artifact" and .target.name == "bench-fandhe" and (.target.kind[]? == "bin") and .executable != null)] | last | .executable // empty' "$msg")"
  rm -f "$msg"
  if [[ -z "$exe" || ! -f "$exe" ]]; then
    echo "error: build $arm: exe not found" >&2
    exit 1
  fi
  cp "$exe" "$OUT/bench-fandhe-${arm}-${LABEL}"
  # #1166 事故対応と同型のハードゲート: `cargo tree` で `fandhe-ai` が
  # 実際に path 解決されていることを確認する。
  local tree_output
  tree_output="$(cargo tree -p bench-fandhe --depth 1 --config "$patch_config" 2>&1)"
  if ! echo "$tree_output" | grep -qE 'fandhe-ai v[0-9.]+ \(.*crates/facade\)'; then
    echo "error: fandhe-ai did not resolve to the path-patched crates/facade ($arm); cargo tree:" >&2
    echo "$tree_output" >&2
    exit 1
  fi
  echo "$tree_output" >"$OUT/tree-${arm}-${LABEL}.txt"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$OUT/bench-fandhe-${arm}-${LABEL}" >"$OUT/sha-${arm}-${LABEL}.txt"
  else
    shasum -a 256 "$OUT/bench-fandhe-${arm}-${LABEL}" >"$OUT/sha-${arm}-${LABEL}.txt"
  fi
}

echo "== build before ==(facade=$BEFORE_FACADE)"
build_arm before "$BEFORE_FACADE" "target-ab-dinput-sync-${LABEL}-before"
echo "== build after  ==(facade=$AFTER_FACADE)"
build_arm after "$AFTER_FACADE" "target-ab-dinput-sync-${LABEL}-after"
cat "$OUT"/tree-*-"${LABEL}".txt

BIN_BEFORE="$OUT/bench-fandhe-before-${LABEL}"
BIN_AFTER="$OUT/bench-fandhe-after-${LABEL}"

run_train() { # run_train <arm> <mode> [--phases]
  local arm=$1 mode=$2 extra=${3:-} suffix=train bin
  [[ -n "$extra" ]] && suffix=phases
  if [[ "$arm" == "before" ]]; then bin="$BIN_BEFORE"; else bin="$BIN_AFTER"; fi
  echo "== train metal mode=$mode arm=$arm ${extra:-} =="
  if ! "$bin" --task train --device metal --mode "$mode" ${extra:+$extra} \
      --out "$OUT/results-${arm}-${LABEL}-${suffix}.jsonl" 2>"$OUT/err-${arm}-${LABEL}.tmp"; then
    echo "arm=$arm mode=$mode extra=${extra:-none} : $(cat "$OUT/err-${arm}-${LABEL}.tmp")" >>"$SKIP"
    echo "  -> FAILED (recorded in $SKIP)"
    ANY_FAILED=$((ANY_FAILED + 1))
  fi
  rm -f "$OUT/err-${arm}-${LABEL}.tmp"
}

: >"$OUT/uptime-${LABEL}.log"
uptime >>"$OUT/uptime-${LABEL}.log"
pmset -g therm >"$OUT/pmset_therm_before-${LABEL}.txt" 2>&1 || true

for run_i in $(seq 1 "$ROUNDS"); do
  for mode in fresh reuse; do
    if (( run_i % 2 == 1 )); then
      run_train before "$mode"; run_train after "$mode"
    else
      run_train after "$mode"; run_train before "$mode"
    fi
  done
  echo "run $run_i: $(uptime)" | tee -a "$OUT/uptime-${LABEL}.log"
done

# `--phases` は診断用（compare_gemm_ab.py の診断表は phase ごとに 1 行を
# 採るため各腕 1 回。#1548／#1555／#1560 と同型）。
for mode in fresh reuse; do
  run_train before "$mode" "--phases"; run_train after "$mode" "--phases"
done
echo "phases: $(uptime)" | tee -a "$OUT/uptime-${LABEL}.log"
pmset -g therm >"$OUT/pmset_therm_after-${LABEL}.txt" 2>&1 || true

# 事前登録した受け入れ契約は reuse のみを必須判定とする（train reuse の
# `step_total` が主ゲート。fresh は対照／参考。`--modes reuse` を明示し、
# fresh 参考行が必須判定へ混入するのを防ぐ。#1560 と同型）。この呼び出し
# の終了コード（0=非後退・2=判定不能・3=後退）を保存し、非 0 終了を
# `ANY_FAILED` へ反映してスクリプト全体の終了コードへ伝播させる。
# `--require-checksum-exact` を明示指定する: 事前登録規則は reuse セルの
# checksum 完全一致（`checksum_exact_match`）を必須としているが、
# `compare_gemm_ab.py` は複合誤差判定（`checksum_composite_match`）のみを
# 終了コードへ反映する既定契約のため、これを付けない限り checksum 不一致
# でも性能比が threshold 内なら成功扱いになってしまう（#1560 と同型の
# codex-review [P1] 指摘対応）。`--phases`（nargs=2）の直後に
# `--require-checksum-exact` を置くと argparse が誤消費するため、フラグ系
# を先に並べ `--phases` を最後（ファイル引数の直前）に置く。
python3 compare_gemm_ab.py --device metal --task train --threshold 1.00 --per-run --modes reuse \
  --require-checksum-exact --phases \
  "$OUT/results-before-${LABEL}-phases.jsonl" "$OUT/results-after-${LABEL}-phases.jsonl" \
  "$OUT/results-before-${LABEL}-train.jsonl" "$OUT/results-after-${LABEL}-train.jsonl" \
  >"compare-train-${LABEL}.md" 2>"compare-train-${LABEL}.err"
COMPARE_EXIT=$?
if [[ "$COMPARE_EXIT" -ne 0 ]]; then
  echo "compare exit=$COMPARE_EXIT" | tee -a "compare-train-${LABEL}.err"
  ANY_FAILED=$((ANY_FAILED + 1))
fi

# fresh は対照（事前登録どおり非判定）の参考表として別ファイルへ出力する。
# この呼び出しの終了コードは判定に用いないため `ANY_FAILED` へは反映しない。
python3 compare_gemm_ab.py --device metal --task train --threshold 1.00 --per-run --modes fresh --phases \
  "$OUT/results-before-${LABEL}-phases.jsonl" "$OUT/results-after-${LABEL}-phases.jsonl" \
  "$OUT/results-before-${LABEL}-train.jsonl" "$OUT/results-after-${LABEL}-train.jsonl" \
  >"compare-train-${LABEL}-fresh-reference.md" 2>"compare-train-${LABEL}-fresh-reference.err" ||
  echo "fresh reference compare exit=$? (informational only; not part of the judged A/B)" \
    | tee -a "compare-train-${LABEL}-fresh-reference.err"

echo "done. results in $OUT ; failures (if any) in $SKIP"
if [[ "$ANY_FAILED" -gt 0 ]]; then
  echo "FAILED: $ANY_FAILED run(s) failed; see $SKIP" >&2
  exit 1
fi
