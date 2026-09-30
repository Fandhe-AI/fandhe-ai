#!/usr/bin/env bash
# イシュー #2115: CUDA 推論 forward チェーンの CUDA Graph capture opt-in
# （`FANDHE_AI_CUDA_GRAPH_INFER`・既定 OFF）の framework-compare infer A/B。
# 判定規則は実測前に固定した `docs/perf/logs/cuda-infer-chain-graphcapture-2115/
# RULE.txt`（事後に緩めない）、記録は `docs/perf/infer-chain-graphcapture-cuda-ab.md`。
#
# 同一バイナリ・環境変数トグル方式（`run_ab_graph_cuda.sh` と同型）: before 腕は
# 環境変数を**未設定**、after 腕は `FANDHE_AI_CUDA_GRAPH_INFER=1` で起動する
# （opt-in は最初の CUDA デバイス初期化前に決まるため、腕ごとに別プロセス）。
# セルは infer × {fresh, reuse} × batch {64, 1024, 4096}
# （`bench-fandhe --infer-batch`）。5 round・プロセス独立起動・round ごとに
# 起動順を反転する。fresh セルは推論チェーンへ到達しない（`model.forward`
# 経路）ため、fresh は「created stream 化のみ」の効果を見る対照で、RULE.txt
# に事前帰属を記載済み。
#
# `AB_PATCH_FACADE_PATH`（本機構を含む HEAD の `crates/facade` への絶対パス）は
# 必須。`[patch]` は `--config` の CLI 引数としてのみ与え、`Cargo.toml`／
# `Cargo.lock`／`.cargo/config.toml` へはコミットしない（deps-policy.md 第 9 区分。
# `Cargo.lock` は `bench_fandhe_lock_restore.sh` で退避・EXIT trap 復元）。
#
# 専有ゲート（RULE.txt）: load1 < 1.0 かつ `nvidia-smi` の utilization.gpu が
# 0% の状態が 30 秒間隔の 3 サンプル連続で成立すること。最大 20 サンプルで
# 不成立なら undetermined を 1 回記録して終了する（再試行ループなし）。
# `AB_LOAD_GATE_MODE=record_only` はユーザーの明示指示があるときだけ使う
# （ゲート不成立でも計測は進めるが、判定は undetermined 扱いになる）。
#
# 呼び出し例（GB10 実機。ユーザー承認・別セッション。<user> は各自の値）:
#   AB_PATCH_FACADE_PATH=/home/<user>/work/rust-ai-library-run-2115/crates/facade \
#     bash run_ab_infer_graph_cuda.sh 2115-gb10-run1
#
# 失敗を捏造しない（skipped ログへ記録・非 0 終了。security.md A08）。
set -u
cd "$(dirname "$0")"
# shellcheck source=./bench_fandhe_lock_restore.sh
source ./bench_fandhe_lock_restore.sh

LABEL=${1:-}
ROUNDS=${AB_ROUNDS:-5}

# A03 インジェクション対策: ラベルはファイル名・パスへ直接埋め込むため
# 英数字・`._-` のみの allowlist で検証する。
if [[ -z "$LABEL" || ! "$LABEL" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "usage: $0 <label>  (label must match [A-Za-z0-9._-]+, e.g. 2115-gb10-run1)" >&2
  echo "  env AB_PATCH_FACADE_PATH=<absolute path to crates/facade> is required" >&2
  exit 1
fi
# 判定は 5 run 中央値が前提（RULE.txt）。5 以外は拒否する。
if [[ "$ROUNDS" != "5" ]]; then
  echo "error: AB_ROUNDS must be 5 (RULE.txt fixes the run count; got: $ROUNDS)" >&2
  exit 1
fi

# `AB_PATCH_FACADE_PATH` の検証（A03・A08）。
if [[ -z "${AB_PATCH_FACADE_PATH:-}" ]]; then
  echo "error: AB_PATCH_FACADE_PATH is required (absolute path to a crates/facade checkout containing issue #2115)" >&2
  exit 1
fi
if [[ "$AB_PATCH_FACADE_PATH" != /* ]]; then
  echo "error: AB_PATCH_FACADE_PATH must be an absolute path (got: $AB_PATCH_FACADE_PATH)" >&2
  exit 1
fi
if [[ ! -f "$AB_PATCH_FACADE_PATH/Cargo.toml" ]]; then
  echo "error: AB_PATCH_FACADE_PATH/Cargo.toml not found ($AB_PATCH_FACADE_PATH)" >&2
  exit 1
fi
if ! grep -qE '^\s*name\s*=\s*"fandhe-ai"\s*$' "$AB_PATCH_FACADE_PATH/Cargo.toml"; then
  echo "error: AB_PATCH_FACADE_PATH/Cargo.toml does not declare name = \"fandhe-ai\" ($AB_PATCH_FACADE_PATH)" >&2
  exit 1
fi
PATCH_CONFIG="patch.crates-io.fandhe-ai.path=\"${AB_PATCH_FACADE_PATH}\""

# 学習側 opt-in（`FANDHE_AI_CUDA_GRAPH_STEP`）が環境に残っていると、
# 両腕の created stream 化が同時に起きて before 腕が汚染されるため拒否する。
# 推論側の環境変数も、腕ごとに本スクリプトが明示的に制御する。
if [[ -n "${FANDHE_AI_CUDA_GRAPH_STEP:-}" ]]; then
  echo "error: FANDHE_AI_CUDA_GRAPH_STEP is set in the environment (would confound the before arm); unset it" >&2
  exit 1
fi
if [[ -n "${FANDHE_AI_CUDA_GRAPH_INFER:-}" ]]; then
  echo "error: FANDHE_AI_CUDA_GRAPH_INFER is set in the environment; this script controls it per arm (unset it)" >&2
  exit 1
fi

OUT="results/raw"
mkdir -p "$OUT"
SKIP="$OUT/skipped-dgx-infer-graph-ab-${LABEL}.log"
BEFORE_JSONL="$OUT/results-before-${LABEL}-infer.jsonl"
AFTER_JSONL="$OUT/results-after-${LABEL}-infer.jsonl"

# 同一ラベルの既存結果があれば停止する（append 方式のため、混入すると
# 「各セルちょうど 5 件」検証が崩れる。上書きで隠さない）。
if [[ -s "$BEFORE_JSONL" || -s "$AFTER_JSONL" || -s "$SKIP" ]]; then
  echo "error: results for label '$LABEL' already exist under $OUT; choose a new label" >&2
  exit 1
fi
: > "$SKIP"
: > "$BEFORE_JSONL"
: > "$AFTER_JSONL"
ANY_FAILED=0

# `Cargo.lock` を退避し、異常終了含め終了時に必ず復元する。
bench_fandhe_setup_lock_restore_trap

# --- 専有ゲート ---------------------------------------------------------
GATE_MODE=${AB_LOAD_GATE_MODE:-enforce}
gate_sample_ok() {
  local load1 util
  load1="$(awk '{print $1}' /proc/loadavg 2>/dev/null || echo 999)"
  util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
  [[ -n "$util" ]] || return 1
  awk -v l="$load1" 'BEGIN { exit !(l < 1.0) }' && [[ "$util" == "0" ]]
}
GATE_LOG="$OUT/gate-${LABEL}.log"
: > "$GATE_LOG"
GATE_PASSED=0
consecutive=0
for sample in $(seq 1 20); do
  if gate_sample_ok; then
    consecutive=$((consecutive + 1))
    echo "gate sample $sample: ok ($consecutive/3)" >> "$GATE_LOG"
  else
    consecutive=0
    echo "gate sample $sample: busy" >> "$GATE_LOG"
  fi
  if [[ "$consecutive" -ge 3 ]]; then GATE_PASSED=1; break; fi
  sleep 30
done
if [[ "$GATE_PASSED" -ne 1 ]]; then
  echo "undetermined: exclusive-use gate not satisfied (see $GATE_LOG)" | tee -a "$SKIP" >&2
  if [[ "$GATE_MODE" != "record_only" ]]; then
    exit 1
  fi
  echo "AB_LOAD_GATE_MODE=record_only: continuing; verdict MUST be recorded as undetermined" | tee -a "$SKIP" >&2
fi

# --- ビルド -------------------------------------------------------------
echo "== build bench-fandhe (path patch) =="
if ! cargo build --release -p bench-fandhe --config "$PATCH_CONFIG" 2>"$OUT/build-${LABEL}.err"; then
  tail -40 "$OUT/build-${LABEL}.err"
  echo "bench-fandhe BUILD FAILED: $(tail -3 "$OUT/build-${LABEL}.err" | tr '\n' ' ')" >> "$SKIP"
  exit 1
fi
# #1166 事故対応と同型のハードゲート: path 解決の確認。
TREE_OUTPUT="$(cargo tree -p bench-fandhe --depth 1 --config "$PATCH_CONFIG" 2>&1)"
if ! echo "$TREE_OUTPUT" | grep -qE 'fandhe-ai v[0-9.]+ \(.*crates/facade\)'; then
  echo "error: fandhe-ai did not resolve to the path-patched crates/facade; cargo tree:" >&2
  echo "$TREE_OUTPUT" >&2
  exit 1
fi
echo "$TREE_OUTPUT" > "$OUT/tree-${LABEL}.txt"
BIN="$OUT/bench-fandhe-${LABEL}"
cp target/release/bench-fandhe "$BIN"
if command -v sha256sum >/dev/null 2>&1; then
  BIN_SHA="$(sha256sum "$BIN")"
else
  BIN_SHA="$(shasum -a 256 "$BIN")"
fi
echo "$BIN_SHA" > "$OUT/sha-${LABEL}.txt"

nvidia-smi --query-gpu=name,driver_version --format=csv,noheader > "$OUT/env-gpu-${LABEL}.txt" 2>&1 || true
uptime > "$OUT/uptime-${LABEL}.log"

# --- 計測 ---------------------------------------------------------------
run_cell() { # run_cell <arm: before|after> <mode> <batch>
  local arm=$1 mode=$2 batch=$3 jsonl
  if [[ "$arm" == "before" ]]; then jsonl="$BEFORE_JSONL"; else jsonl="$AFTER_JSONL"; fi
  echo "== infer cuda mode=$mode batch=$batch arm=$arm =="
  local rc
  if [[ "$arm" == "after" ]]; then
    env FANDHE_AI_CUDA_GRAPH_INFER=1 "$BIN" --task infer --device cuda --mode "$mode" \
      --infer-batch "$batch" --out "$jsonl" 2>"$OUT/err-${LABEL}.tmp"
    rc=$?
  else
    "$BIN" --task infer --device cuda --mode "$mode" \
      --infer-batch "$batch" --out "$jsonl" 2>"$OUT/err-${LABEL}.tmp"
    rc=$?
  fi
  if [[ "$rc" -ne 0 ]]; then
    echo "arm=$arm mode=$mode batch=$batch : $(cat "$OUT/err-${LABEL}.tmp")" >> "$SKIP"
    echo "  -> FAILED (recorded in $SKIP)"
    ANY_FAILED=$((ANY_FAILED + 1))
  fi
  rm -f "$OUT/err-${LABEL}.tmp"
}

for run_i in $(seq 1 "$ROUNDS"); do
  for batch in 64 1024 4096; do
    for mode in fresh reuse; do
      if (( run_i % 2 == 1 )); then
        run_cell before "$mode" "$batch"; run_cell after "$mode" "$batch"
      else
        run_cell after "$mode" "$batch"; run_cell before "$mode" "$batch"
      fi
    done
  done
  echo "run $run_i: $(uptime)" | tee -a "$OUT/uptime-${LABEL}.log"
  nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits >> "$OUT/nvidia-smi-during-${LABEL}.txt" 2>&1 || true
done

# --- 判定（RULE.txt の ADOPT 条件のうち機械判定できる部分）---------------
# 全 6 セルの 5 run 中央値 ratio<=1.00・checksum 完全一致・run 単位比の
# 併記。R0〜R2 と「reuse の少なくとも 1 セルで 5/5 run 一貫 ratio<1.00」の
# 確認は `compare-infer-graph-<label>.md` の per-run 列と README 手順で行う。
python3 compare_gemm_ab.py --task infer --device cuda --sizes infer-batches \
  --threshold 1.00 --per-run --require-checksum-exact \
  "$BEFORE_JSONL" "$AFTER_JSONL" \
  >"compare-infer-graph-${LABEL}.md" 2>"compare-infer-graph-${LABEL}.err"
COMPARE_EXIT=$?
if [[ "$COMPARE_EXIT" -ne 0 ]]; then
  echo "compare exit=$COMPARE_EXIT" | tee -a "compare-infer-graph-${LABEL}.err"
  ANY_FAILED=$((ANY_FAILED + 1))
fi

echo "done. results in $OUT ; failures (if any) in $SKIP"
if [[ "$GATE_PASSED" -ne 1 ]]; then
  echo "NOTE: gate not satisfied (record_only) -> verdict is undetermined regardless of compare exit" >&2
  exit 1
fi
if [[ "$ANY_FAILED" -gt 0 ]]; then
  echo "FAILED: $ANY_FAILED run(s) failed; see $SKIP" >&2
  exit 1
fi
