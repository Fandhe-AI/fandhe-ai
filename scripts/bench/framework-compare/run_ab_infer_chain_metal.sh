#!/usr/bin/env bash
# イシュー #1580: Metal 推論 forward チェーン（`Sequential::
# predict_resident`）を「層境界ごとに D2H→H2D」から「入力を 1 回だけ
# upload・各層は encode-only・チェーン末尾で 1 回だけ download」へ
# 置き換えた変更の framework-compare 実践規模 A/B。
#
# before = 本ブランチのベース（origin/main。#1580 適用前）、after = 本
# ブランチ（`crates/*/src` は評価対象コミットと同一）を、それぞれ別
# ツリーへ展開した `crates/facade` への `[patch.crates-io.fandhe-ai]`
# path patch で 2 本の `bench-fandhe` バイナリとしてビルドし、5 round・
# run 単位で起動順を反転しながら交互実行する（`run_ab_dinput_sync_
# metal.sh` の `--task infer` 版）。
#
# `--task infer` は `compare_gemm_ab.py` が `--task train` 限定でしか
# `--phases` を受け付けないため（同スクリプトの既存ガード。他イシューと
# 共有するツールへの変更は本イシューのスコープ外とし衝突を避ける）、
# 判定は本スクリプト内の `judge_infer_ab.py`（JSONL から `task:"infer"`
# 行を直接読み `median_s`／`checksum` を突き合わせる自己完結ロジック）で
# 行う。
#
# `[patch]` は本スクリプトの CLI 引数（`--config`）としてのみ与え、
# `scripts/bench/framework-compare/Cargo.toml`／`Cargo.lock`／
# `.cargo/config.toml` へはコミットしない（deps-policy.md 第 9 区分は
# registry 取得元のみを許容するため）。`Cargo.lock` は
# `bench_fandhe_lock_restore.sh` の共有ヘルパーで退避・EXIT trap 復元する。
#
# 呼び出し例（M4 Max 実機。ユーザー承認・別セッション）:
#   AB_BEFORE_FACADE_PATH=/home/<user>/work/rust-ai-library-run-1580-before/crates/facade \
#   AB_AFTER_FACADE_PATH=/home/<user>/work/rust-ai-library-run-1580-after/crates/facade \
#     bash run_ab_infer_chain_metal.sh 1580
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
  echo "usage: $0 <label>  (label must match [A-Za-z0-9._-]+, e.g. 1580)" >&2
  echo "  env AB_BEFORE_FACADE_PATH / AB_AFTER_FACADE_PATH (absolute paths) are required" >&2
  exit 1
fi

# `AB_{BEFORE,AFTER}_FACADE_PATH` の検証（A03・A08）: 未設定・相対パス・
# Cargo.toml 不在・crate 名不一致のいずれかなら fail-closed で exit 1。
validate_facade_path() {
  local var_name=$1 path=$2
  if [[ -z "$path" ]]; then
    echo "error: $var_name is required (absolute path to a crates/facade checkout; issue #1580)" >&2
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
SKIP="$OUT/skipped-metal-infer-chain-ab-${LABEL}.log"
: > "$SKIP"
ANY_FAILED=0

# `run_infer` の出力先（bench-common が `OpenOptions::append(true)` で
# 書き込む）を実行開始時に必ず空へ初期化する。append 方式のため、
# 中断後の再実行や同一ラベルでの再実行があると過去の計測行が残存し、
# 判定が壊れる（`run_ab_dinput_sync_metal.sh` と同型の対処）。
for _reset_arm in before after; do
  : > "$OUT/results-${_reset_arm}-${LABEL}-infer.jsonl"
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
  # codex-review 指摘: `set -u` のみでは `cp` の終了コードを検査しない
  # ため、同一ラベルで再実行して既存バイナリへの上書きコピーに失敗
  # した場合でも処理が継続し、後続の cargo tree・実行対象が古いバイナリ
  # のまま計測が続行しうる。計測対象の同一性を保証するため終了コードを
  # 明示検査して非 0 で中止する。
  if ! cp "$exe" "$OUT/bench-fandhe-${arm}-${LABEL}"; then
    echo "error: cp failed ($arm): $exe -> $OUT/bench-fandhe-${arm}-${LABEL}" >&2
    exit 1
  fi
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
build_arm before "$BEFORE_FACADE" "target-ab-infer-chain-${LABEL}-before"
echo "== build after  ==(facade=$AFTER_FACADE)"
build_arm after "$AFTER_FACADE" "target-ab-infer-chain-${LABEL}-after"
cat "$OUT"/tree-*-"${LABEL}".txt

BIN_BEFORE="$OUT/bench-fandhe-before-${LABEL}"
BIN_AFTER="$OUT/bench-fandhe-after-${LABEL}"

run_infer() { # run_infer <arm> <mode>
  local arm=$1 mode=$2 bin
  if [[ "$arm" == "before" ]]; then bin="$BIN_BEFORE"; else bin="$BIN_AFTER"; fi
  echo "== infer metal mode=$mode arm=$arm =="
  if ! "$bin" --task infer --device metal --mode "$mode" \
      --out "$OUT/results-${arm}-${LABEL}-infer.jsonl" 2>"$OUT/err-${arm}-${LABEL}.tmp"; then
    echo "arm=$arm mode=$mode : $(cat "$OUT/err-${arm}-${LABEL}.tmp")" >>"$SKIP"
    echo "  -> FAILED (recorded in $SKIP)"
    ANY_FAILED=$((ANY_FAILED + 1))
  fi
  rm -f "$OUT/err-${arm}-${LABEL}.tmp"
}

: >"$OUT/uptime-${LABEL}.log"
uptime >>"$OUT/uptime-${LABEL}.log"
pmset -g therm >"$OUT/pmset_therm_before-${LABEL}.txt" 2>&1 || true

# 判定セル: `--mode reuse`。対照: `--mode fresh`（事前登録どおり非判定・
# 参考記録のみ）。
for run_i in $(seq 1 "$ROUNDS"); do
  for mode in fresh reuse; do
    if (( run_i % 2 == 1 )); then
      run_infer before "$mode"; run_infer after "$mode"
    else
      run_infer after "$mode"; run_infer before "$mode"
    fi
  done
  echo "run $run_i: $(uptime)" | tee -a "$OUT/uptime-${LABEL}.log"
done
pmset -g therm >"$OUT/pmset_therm_after-${LABEL}.txt" 2>&1 || true

# `compare_gemm_ab.py` は `--task train` 限定の `--phases` ガードを持ち、
# 他イシューと共有するため本イシューでは変更しない（計画・§本スクリプト
# 冒頭コメント参照）。代わりに JSONL を直接読む自己完結の判定スクリプト
# `judge_infer_ab.py` で判定する: `--mode reuse` の全 round 中央値比
# （after/before）が `<= 1.00`・checksum が全 round で完全一致すること
# を事前登録した非後退規則とする（fresh は対照・非判定）。
python3 judge_infer_ab.py --before "$OUT/results-before-${LABEL}-infer.jsonl" \
  --after "$OUT/results-after-${LABEL}-infer.jsonl" \
  --rounds "$ROUNDS" \
  >"compare-infer-${LABEL}.md" 2>"compare-infer-${LABEL}.err"
JUDGE_EXIT=$?
cat "compare-infer-${LABEL}.md"
if [[ "$JUDGE_EXIT" -ne 0 ]]; then
  echo "judge exit=$JUDGE_EXIT" | tee -a "compare-infer-${LABEL}.err"
  ANY_FAILED=$((ANY_FAILED + 1))
fi

echo "done. results in $OUT ; failures (if any) in $SKIP"
if [[ "$ANY_FAILED" -gt 0 ]]; then
  echo "FAILED: $ANY_FAILED run(s) failed; see $SKIP" >&2
  exit 1
fi
