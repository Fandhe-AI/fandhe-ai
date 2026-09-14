#!/usr/bin/env bash
# イシュー #1590: CUDA GEMM VJP 専用 NT／TN 転置入口（#1214。マージ
# コミット `ab0b77d0`）の train fresh／reuse A/B。
#
# 既存の `run_ab_resident_grad_cuda.sh`／`run_ab_infer_chain_cuda.sh` は
# 「単一チェックアウト（本ブランチ HEAD）の bench-fandhe ハーネスへ
# `crates/facade` だけを before／after 2 本 path patch する」方式だが、
# 本イシューは HEAD のハーネスが #1214 マージ直前の facade（旧 API 集合）
# ではビルド不能（後発 API 呼び出しを含むため）なので採用しない。
# 代わりに **before／after それぞれ独立に丸ごと展開したツリー**
# （`git archive 82058501` 相当／`git archive ab0b77d0` 相当。#1214
# マージ直前の main／マージコミット自身）を受け取り、各ツリー**自身の**
# `scripts/bench/framework-compare/` 配下で `cargo build` を実行する
# （ハーネスと facade が常に同一ツリー由来＝バージョン不整合が起きない）。
# その上で `crates/facade` への `[patch.crates-io.fandhe-ai]` path patch
# （同一ツリー内の facade を指す。CLI `--config` のみ・非コミット）を使い、
# ツリー内の `bench-fandhe`（registry 版 `fandhe-ai =0.6.0` 依存）を
# ツリー内 facade ソースへ差し替えてビルドする（既存 run_ab_*.sh と同じ
# 目的の patch だが、対象が「本リポジトリ」ではなく「各アームのツリー」
# である点が異なる）。
#
# 本スクリプト自身（このファイル・`compare_gemm_ab.py` 等）は **この
# リポジトリ（呼び出し元ツリー）の framework-compare を一切 cargo で
# ビルド・実行しない**（`cd "$(dirname "$0")"` はしない。build_arm は
# 各アームのツリーへ `cd` してから cargo を実行し、その場でも同様に
# `[patch]` は CLI 引数のみで manifest／lock へは書かない）。これにより
# 本リポジトリの `scripts/bench/framework-compare/Cargo.lock`（承認済み
# ピン。deps-policy.md 第 9 区分）は本スクリプトの実行では一切書き換わら
# ず、`check_framework_compare` の fail-closed 契約検査を迂回しない。
# `compare_gemm_ab.py` による判定のみ、このリポジトリ（呼び出し元）の
# コピーを使う（JSONL の読み取りのみで cargo を要さないため安全）。
#
# 呼び出し例（GB10 実機。ユーザー承認・別セッション）:
#   AB_BEFORE_TREE=/home/<user>/work/rust-ai-library-1590-before \
#   AB_AFTER_TREE=/home/<user>/work/rust-ai-library-1590-after \
#     bash scripts/bench/framework-compare/run_ab_vjp_transposed_cuda.sh 1590
#
# 出力は他の run_ab_*.sh と同じ「失敗を捏造しない」方針（skipped ログへ
# 記録・非 0 終了。security.md A08）。
set -u
SELF_DIR="$(cd "$(dirname "$0")" && pwd)"

LABEL=${1:-}
ROUNDS=${AB_ROUNDS:-5}

# A03 インジェクション対策: ラベルはファイル名・パスに直接埋め込むため、
# 英数字・`._-` のみを許可する allowlist で検証する（既存 run_ab_*.sh と
# 同一方針）。
if [[ -z "$LABEL" || ! "$LABEL" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "usage: $0 <label>  (label must match [A-Za-z0-9._-]+, e.g. 1590)" >&2
  echo "  env AB_BEFORE_TREE / AB_AFTER_TREE (absolute paths to whole-repo checkouts) are required" >&2
  exit 1
fi

# `AB_{BEFORE,AFTER}_TREE` の検証（A03・A08）: 未設定・相対パス・
# `"` を含むパス（`--config` の TOML 文字列へ埋め込むため）・
# `crates/facade/Cargo.toml` 不在／crate 名不一致・
# `scripts/bench/framework-compare/Cargo.toml` 不在のいずれかなら
# fail-closed で exit 1。
validate_tree() {
  local var_name=$1 tree=$2
  if [[ -z "$tree" ]]; then
    echo "error: $var_name is required (absolute path to a whole-repo checkout; issue #1590)" >&2
    exit 1
  fi
  if [[ "$tree" != /* ]]; then
    echo "error: $var_name must be an absolute path (got: $tree)" >&2
    exit 1
  fi
  if [[ "$tree" == *'"'* || "$tree" == *$'\n'* ]]; then
    echo "error: $var_name must not contain a double-quote or newline (got: $tree)" >&2
    exit 1
  fi
  if [[ ! -f "$tree/crates/facade/Cargo.toml" ]]; then
    echo "error: $var_name/crates/facade/Cargo.toml not found ($tree)" >&2
    exit 1
  fi
  if ! grep -qE '^\s*name\s*=\s*"fandhe-ai"\s*$' "$tree/crates/facade/Cargo.toml"; then
    echo "error: $var_name/crates/facade/Cargo.toml does not declare name = \"fandhe-ai\" ($tree)" >&2
    exit 1
  fi
  if [[ ! -f "$tree/scripts/bench/framework-compare/Cargo.toml" ]]; then
    echo "error: $var_name/scripts/bench/framework-compare/Cargo.toml not found ($tree)" >&2
    exit 1
  fi
}
validate_tree AB_BEFORE_TREE "${AB_BEFORE_TREE:-}"
validate_tree AB_AFTER_TREE "${AB_AFTER_TREE:-}"
# Cursor Bugbot 指摘（イシュー #1590 PR #1812）: 末尾スラッシュを含む絶対パスを
# そのまま使うと、後段の `cargo tree` ハードゲート（`${tree}/crates/facade`
# パターン）が二重スラッシュを要求してしまい、`fandhe-ai` が実際には正しく
# 解決できているのに不一致で誤って中断しうる。ここで正規化しておく。
BEFORE_TREE="${AB_BEFORE_TREE%/}"
AFTER_TREE="${AB_AFTER_TREE%/}"

OUT="$SELF_DIR/results/raw"
mkdir -p "$OUT"
SKIP="$OUT/skipped-dgx-vjp-transposed-ab-${LABEL}.log"
: > "$SKIP"
ANY_FAILED=0

# `run_train` の出力先（bench-common が `OpenOptions::append(true)` で
# 書き込む）を実行開始時に必ず空へ初期化する（既存 run_ab_*.sh と同じ
# 理由。中断後の再実行での過去計測行の残存を防ぐ）。
for _reset_arm in before after; do
  for _reset_suffix in train phases; do
    : > "$OUT/results-${_reset_arm}-${LABEL}-${_reset_suffix}.jsonl"
  done
done

build_arm() { # build_arm <arm> <tree>
  local arm=$1 tree=$2 patch_config exe msg tdir
  tdir="${tree}/scripts/bench/framework-compare/target-ab-vjp-transposed-${LABEL}-${arm}"
  patch_config="patch.crates-io.fandhe-ai.path=\"${tree}/crates/facade\""
  msg="$(mktemp)"
  # `cd` はサブシェル限定（`(...)`）にし、本関数の呼び出し元（本スクリプト
  # 自体のカレントディレクトリ）へ影響を残さない。cargo はこのツリー
  # **自身の** `Cargo.toml`／`Cargo.lock` を読み書きし、本リポジトリの
  # framework-compare には一切触れない。
  if ! ( cd "${tree}/scripts/bench/framework-compare" && \
      cargo build --release -p bench-fandhe --target-dir "$tdir" --message-format=json \
      --config "$patch_config" ) >"$msg" 2>"$OUT/build-${arm}-${LABEL}.err"; then
    tail -40 "$OUT/build-${arm}-${LABEL}.err"
    echo "bench-fandhe BUILD FAILED ($arm, tree=$tree): $(tail -3 "$OUT/build-${arm}-${LABEL}.err" | tr '\n' ' ')" >>"$SKIP"
    rm -f "$msg"
    exit 1
  fi
  exe="$(jq -rs '[.[] | select(.reason == "compiler-artifact" and .target.name == "bench-fandhe" and (.target.kind[]? == "bin") and .executable != null)] | last | .executable // empty' "$msg")"
  rm -f "$msg"
  if [[ -z "$exe" || ! -f "$exe" ]]; then
    echo "error: build $arm: exe not found (tree=$tree)" >&2
    exit 1
  fi
  if ! cp "$exe" "$OUT/bench-fandhe-${arm}-${LABEL}"; then
    echo "error: build $arm: failed to copy built binary from $exe to $OUT/bench-fandhe-${arm}-${LABEL} (tree=$tree)" >&2
    exit 1
  fi
  # #1166 事故対応と同型のハードゲート: `cargo tree` で `fandhe-ai` が
  # このツリー自身の path-patched facade へ実際に解決されていることを
  # 確認する（別ツリーの facade を誤って参照していないことの検証）。
  # codex-review／Bugbot 指摘（イシュー #1590 PR #1812）: `$tree` は
  # 未信頼な絶対パス（正規表現メタ文字・末尾スラッシュを含みうる）の
  # ため、そのまま `grep -E` パターンへ埋め込まず ERE メタ文字を
  # エスケープしてリテラル一致させる（誤って不一致になり、正しく解決
  # できているのに計測前に fail-closed で中断してしまうのを防ぐ）。
  local tree_escaped tree_output
  tree_escaped="$(printf '%s' "$tree" | sed -e 's/[.[\^$(){}+*?|]/\\&/g')"
  tree_output="$( cd "${tree}/scripts/bench/framework-compare" && \
    cargo tree -p bench-fandhe --depth 1 --config "$patch_config" 2>&1 )"
  if ! echo "$tree_output" | grep -qE "fandhe-ai v[0-9.]+ \(${tree_escaped}/crates/facade\)"; then
    echo "error: fandhe-ai did not resolve to the path-patched crates/facade within its own tree ($arm, tree=$tree); cargo tree:" >&2
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

echo "== build before ==(tree=$BEFORE_TREE)"
build_arm before "$BEFORE_TREE"
echo "== build after  ==(tree=$AFTER_TREE)"
build_arm after "$AFTER_TREE"
cat "$OUT"/tree-*-"${LABEL}".txt

BIN_BEFORE="$OUT/bench-fandhe-before-${LABEL}"
BIN_AFTER="$OUT/bench-fandhe-after-${LABEL}"

# codex-review 指摘（イシュー #1590 PR #1812）: 両腕のリリースビルド
# （上記 `build_arm` 2 回）は数分規模の CPU 負荷を伴い、`orchestrate.sh`
# が計測開始「前」に確認した専有ゲート（1 分 load average < 1.0 かつ
# `utilization.gpu == 0 %`）の前提を、計測ループ開始時点では保証しない
# （ビルド自体の CPU 負荷・ビルド所要時間中の他プロセス起動の余地）。
# そのためビルド完了後・計測ループ（下記 `for run_i in ...`）直前に、
# `orchestrate.sh` と同一のゲート条件を再実行する（`AB_LOAD_GATE_MODE`
# はオーケストレータ経由で環境変数として引き継がれるため同一の
# opt-out 判断を尊重する）。
POST_BUILD_GATE_LOG="$OUT/gate-postbuild-${LABEL}.log"
POST_BUILD_GATE_MODE="${AB_LOAD_GATE_MODE:-gated}"
if [[ "$POST_BUILD_GATE_MODE" == "record_only" ]]; then
  {
    echo "post-build gate protocol: record_only（専有ゲート opt-out。ユーザー明示指示）"
    echo "start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    uptime
  } >"$POST_BUILD_GATE_LOG"
else
  PB_MAX_SAMPLES=20
  PB_NEED_CONSEC=3
  PB_INTERVAL=30
  PB_LOAD_MAX="1.0"
  {
    echo "post-build gate protocol: load1 < ${PB_LOAD_MAX} && util.gpu == 0% for ${PB_NEED_CONSEC} consecutive samples (interval ${PB_INTERVAL}s, max ${PB_MAX_SAMPLES})"
    echo "start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } >"$POST_BUILD_GATE_LOG"
  pb_consec=0
  pb_passed=0
  for pb_i in $(seq 1 "$PB_MAX_SAMPLES"); do
    pb_load1=$(cut -d' ' -f1 /proc/loadavg)
    pb_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    pb_apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
    pb_ok=0
    if awk -v l="$pb_load1" -v m="$PB_LOAD_MAX" 'BEGIN{exit !(l<m)}' && [[ "$pb_util" == "0" ]]; then pb_ok=1; fi
    echo "sample=$pb_i ts=$(date -u +%H:%M:%SZ) load1=$pb_load1 util_gpu=${pb_util}% compute_apps=$pb_apps gate_ok=$pb_ok" >>"$POST_BUILD_GATE_LOG"
    if [[ "$pb_ok" == "1" ]]; then pb_consec=$((pb_consec+1)); else pb_consec=0; fi
    if [[ "$pb_consec" -ge "$PB_NEED_CONSEC" ]]; then pb_passed=1; break; fi
    sleep "$PB_INTERVAL"
  done
  if [[ "$pb_passed" != "1" ]]; then
    echo "verdict=undetermined (post-build gate not satisfied within ${PB_MAX_SAMPLES} samples)" >>"$POST_BUILD_GATE_LOG"
    echo "done. verdict=undetermined" >>"$POST_BUILD_GATE_LOG"
    echo "post-build gate not satisfied; see $POST_BUILD_GATE_LOG" >>"$SKIP"
    echo "post-build gate not satisfied (verdict=undetermined); see $POST_BUILD_GATE_LOG" >&2
    exit 0
  fi
  echo "post-build gate passed at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$POST_BUILD_GATE_LOG"
  uptime >>"$POST_BUILD_GATE_LOG"
fi

run_train() { # run_train <arm> <mode> [--phases]
  local arm=$1 mode=$2 extra=${3:-} suffix=train bin
  [[ -n "$extra" ]] && suffix=phases
  if [[ "$arm" == "before" ]]; then bin="$BIN_BEFORE"; else bin="$BIN_AFTER"; fi
  echo "== train cuda mode=$mode arm=$arm ${extra:-} =="
  if ! "$bin" --task train --device cuda --size 64 --mode "$mode" ${extra:+$extra} \
      --out "$OUT/results-${arm}-${LABEL}-${suffix}.jsonl" 2>"$OUT/err-${arm}-${LABEL}.tmp"; then
    echo "arm=$arm mode=$mode extra=${extra:-none} : $(cat "$OUT/err-${arm}-${LABEL}.tmp")" >>"$SKIP"
    echo "  -> FAILED (recorded in $SKIP)"
    ANY_FAILED=$((ANY_FAILED + 1))
  fi
  rm -f "$OUT/err-${arm}-${LABEL}.tmp"
}

: >"$OUT/uptime-${LABEL}.log"
uptime >>"$OUT/uptime-${LABEL}.log"
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits >"$OUT/nvidia-smi-before-${LABEL}.txt" 2>&1 || true
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader >>"$OUT/nvidia-smi-before-${LABEL}.txt" 2>&1 || true

for run_i in $(seq 1 "$ROUNDS"); do
  for mode in fresh reuse; do
    if (( run_i % 2 == 1 )); then
      run_train before "$mode"; run_train after "$mode"
    else
      run_train after "$mode"; run_train before "$mode"
    fi
  done
  echo "run $run_i: $(uptime)" | tee -a "$OUT/uptime-${LABEL}.log"
  nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits >>"$OUT/nvidia-smi-during-${LABEL}.txt" 2>&1 || true
done

# `--phases` は診断用（`backward`／`step_total` 等。compare_gemm_ab.py の
# 診断表は phase ごとに 1 行を採るため各腕・各 mode 1 回。既存 run_ab_*.sh
# と同型）。fresh／reuse 双方を対象とする（計画 §3.3「両 mode とも判定
# 対象」）。
for mode in fresh reuse; do
  run_train before "$mode" "--phases"; run_train after "$mode" "--phases"
done
echo "phases: $(uptime)" | tee -a "$OUT/uptime-${LABEL}.log"
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits >"$OUT/nvidia-smi-after-${LABEL}.txt" 2>&1 || true

# 事前登録判定規則（イシュー #1590・`docs/perf/
# cuda-gemm-vjp-transposed-entry.md` §3.3）: fresh／reuse の**両方**を
# Tier 1 必須判定とする（#1560／#1689 が reuse のみを判定対象にしたのと
# 異なり、fresh も `matmul_vjp` 経由で NT／TN 入口へ到達するため）。
# `compare_gemm_ab.py --phases` の診断表は単一 mode 前提のため、mode
# ごとに 2 回呼ぶ（既存 run_ab_*.sh と同じ制約）。`--require-checksum-
# exact` は #2 数値一致契約（bit 完全一致設計）に基づき両 mode とも
# 必須指定する。フラグ系を `--phases`（nargs=2）より前に置く（argparse
# の値消費事故を避けるため。既存 run_ab_*.sh と同じ理由）。
COMPARE_PY="$SELF_DIR/compare_gemm_ab.py"
for mode in fresh reuse; do
  python3 "$COMPARE_PY" --device cuda --task train --threshold 1.00 --per-run --modes "$mode" \
    --require-checksum-exact --phases \
    "$OUT/results-before-${LABEL}-phases.jsonl" "$OUT/results-after-${LABEL}-phases.jsonl" \
    "$OUT/results-before-${LABEL}-train.jsonl" "$OUT/results-after-${LABEL}-train.jsonl" \
    >"$SELF_DIR/compare-train-${LABEL}-${mode}.md" 2>"$SELF_DIR/compare-train-${LABEL}-${mode}.err"
  COMPARE_EXIT=$?
  if [[ "$COMPARE_EXIT" -ne 0 ]]; then
    echo "compare ($mode) exit=$COMPARE_EXIT" | tee -a "$SELF_DIR/compare-train-${LABEL}-${mode}.err"
    ANY_FAILED=$((ANY_FAILED + 1))
  fi
done

echo "done. results in $OUT ; failures (if any) in $SKIP"
if [[ "$ANY_FAILED" -gt 0 ]]; then
  echo "FAILED: $ANY_FAILED run(s) failed; see $SKIP" >&2
  exit 1
fi
