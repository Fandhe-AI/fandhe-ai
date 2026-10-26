#!/usr/bin/env bash
# イシュー #3261: 学習ステップ全体の CUDA Graph capture（#3260。opt-in・既定 OFF）を GB10 の train CUDA で A/B する。
# GB10（aarch64・CUDA）専用ランナー。規則の正本: docs/perf/logs/train-step-graph-ab-3261/RULE.txt
# （計測前に固定。本スクリプトは sha256 を env_info へ記録し、judge_train_graph_ab.py が現 RULE と突合する）。
#
# 1 ツリー・1 ビルド・1 バイナリ: bench-fandhe は runner ツリーの `crates/facade` への
# `[patch.crates-io.fandhe-ai]` path patch（`--config` 引数のみ。コミットしない）で 1 回だけビルドし、
# off／on の差は環境変数 FANDHE_AI_CUDA_GRAPH_TRAIN だけにする（on = `=1`、off = `=0` を明示。#3594 の既定 ON 化後も off 腕が ON にならないようにする）。
# セルは {chain, reuse, fresh} × {off, on}。判定セルは chain（capture が届く唯一の経路。計測専用パッチ
# docs/perf/logs/resident-chain-ab-3257/bench-resident-chain.patch が `--mode chain` を足す）。
# 計測の前に前提ゲート R0〜R2（実機テスト）を同じ HEAD・同じ機体で実行する（RULE.txt [前提ゲート]）。
#
# 呼び出し例:
#   AB_FACADE_PATH=<abs>/crates/facade AB_TORCH_PY=<abs>/venv-torch/bin/python AB_BENCH_PY=<abs>/bench_py.py \
#   bash run_ab_train_graph_cuda.sh 3261-gb10
# bash 3.2 互換（mapfile・連想配列・eval 不使用）。jq 不使用。失敗は捏造せず skipped へ記録し非 0 終了。
set -u
cd "$(dirname "$0")" || exit 1
# shellcheck source=./bench_fandhe_lock_restore.sh
source ./bench_fandhe_lock_restore.sh

LABEL=${1:-}
ROUNDS=${AB_ROUNDS:-5}
LOAD_GATE=1.0

if [[ -z "$LABEL" || ! "$LABEL" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "usage: $0 <label>  (label: [A-Za-z0-9._-]+)" >&2
  exit 1
fi
if [[ "$ROUNDS" != "5" ]]; then
  echo "error: AB_ROUNDS は 5 固定（RULE.txt。got: ${ROUNDS}）" >&2
  exit 1
fi

# 環境変数の漏れを拒否（off 腕の汚染防止。A08）。腕ごとに本スクリプトが明示して起動する。
for v in FANDHE_AI_CUDA_GRAPH_TRAIN FANDHE_AI_CUDA_GRAPH_STEP FANDHE_AI_CUDA_GRAPH_INFER; do
  if [[ -n "${!v+x}" ]]; then
    echo "error: 継承環境に ${v} が設定されている。unset してから実行すること" >&2; exit 1
  fi
done

# 絶対パス・引用符／バックスラッシュ／改行の拒否（--config 文字列へ埋め込むため。A03）。
validate_abs_path() { # <var_name> <path>
  local var_name=$1 path=$2
  if [[ -z "$path" || "$path" != /* ]]; then
    echo "error: ${var_name} は絶対パス必須（got: ${path:-unset}）" >&2; exit 1
  fi
  if [[ "$path" == *\"* || "$path" == *\\* || "$path" == *$'\n'* ]]; then
    echo "error: ${var_name} に使えない文字を含む" >&2; exit 1
  fi
}
validate_abs_path AB_FACADE_PATH "${AB_FACADE_PATH:-}"
validate_abs_path AB_TORCH_PY "${AB_TORCH_PY:-}"
validate_abs_path AB_BENCH_PY "${AB_BENCH_PY:-}"
FACADE=$AB_FACADE_PATH
if [[ ! -f "$FACADE/Cargo.toml" ]] || ! grep -qE '^[[:space:]]*name[[:space:]]*=[[:space:]]*"fandhe-ai"[[:space:]]*$' "$FACADE/Cargo.toml"; then
  echo "error: AB_FACADE_PATH/Cargo.toml が name = \"fandhe-ai\" を宣言していない" >&2; exit 1
fi
[[ -x "$AB_TORCH_PY" && -f "$AB_BENCH_PY" ]] || { echo "error: AB_TORCH_PY／AB_BENCH_PY が実在しない" >&2; exit 1; }
RUNNER_ROOT="$(cd ../../.. && pwd)"
if [[ "$(cd "$FACADE/../.." && pwd)" != "$RUNNER_ROOT" ]]; then
  echo "error: AB_FACADE_PATH が runner ツリーの crates/facade でない（1 ツリー方式）" >&2; exit 1
fi
RULE_FILE="$RUNNER_ROOT/docs/perf/logs/train-step-graph-ab-3261/RULE.txt"
PATCH_FILE="$RUNNER_ROOT/docs/perf/logs/resident-chain-ab-3257/bench-resident-chain.patch"

# 機体同定（fail-closed）。GB10 の aarch64 のみ。
if [[ "$(uname -m)" != "aarch64" ]]; then echo "error: aarch64（GB10）のみ" >&2; exit 1; fi
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)"
if [[ "$GPU_NAME" != *GB10* ]]; then
  echo "error: nvidia-smi の GPU 名に GB10 を含まない（got: ${GPU_NAME:-unknown}）" >&2; exit 1
fi

sha256_of() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'; else shasum -a 256 "$1" | awk '{print $1}'; fi; }
[[ -f "$RULE_FILE" && -f "$PATCH_FILE" ]] || { echo "error: RULE.txt / patch が無い" >&2; exit 1; }
RULE_SHA="$(sha256_of "$RULE_FILE")"
PATCH_SHA="$(sha256_of "$PATCH_FILE")"
BENCHPY_SHA="$(sha256_of "$AB_BENCH_PY")"
# RULE.txt に値で固定したパッチ sha との一致（行頭アンカー。コメント行では素通りさせない）。
RULE_PATCH_SHA="$(sed -n 's/^  patch_sha256: \([0-9a-f]\{64\}\)$/\1/p' "$RULE_FILE" | head -n1)"
if [[ -z "$RULE_PATCH_SHA" || "$RULE_PATCH_SHA" != "$PATCH_SHA" ]]; then
  echo "error: RULE.txt の patch_sha256 と実パッチの sha256 が一致しない" >&2; exit 1
fi

OUT="results/raw"
mkdir -p "$OUT"
SKIP="$OUT/skipped-train-graph-${LABEL}.log"
ENV_INFO="$OUT/env_info-${LABEL}.txt"
# 同じラベルの既存出力があれば停止（run の差し替え・追記禁止。A08）。
if [[ -e "$SKIP" || -e "$ENV_INFO" ]]; then
  echo "error: ラベル ${LABEL} の出力が既に存在する。新しいラベルを使うこと" >&2; exit 1
fi
: >"$SKIP"
ANY_FAILED=0
unset RAYON_NUM_THREADS

# ドリフトガード: .rev-stamp 1 行目が実 HEAD と一致、2 行目が dirty=0。
STAMP_RUNNER="$(head -n1 "$RUNNER_ROOT/.rev-stamp" 2>/dev/null || true)"
STAMP_DIRTY="$(sed -n 2p "$RUNNER_ROOT/.rev-stamp" 2>/dev/null || true)"
REAL_HEAD="$(git -C "$RUNNER_ROOT" rev-parse HEAD 2>/dev/null || true)"
if [[ -z "$STAMP_RUNNER" || "$STAMP_RUNNER" != "$REAL_HEAD" || "$STAMP_DIRTY" != "dirty=0" ]]; then
  echo "error: .rev-stamp が無い・HEAD と不一致・dirty=0 でない（stamp=${STAMP_RUNNER:-none} head=${REAL_HEAD:-none}）" >&2; exit 1
fi

# precheck（fail-closed。grep は行頭アンカー付き。コメント行で素通りさせない）。
precheck_fail() { echo "precheck FAILED: $1" | tee -a "$SKIP" >&2; exit 1; }
MAIN_RS="bench-fandhe/src/main.rs"
grep -qE '^[[:space:]]*\("train", "chain", false\) =>' "$MAIN_RS" || precheck_fail "main.rs に (train, chain) の分岐が無い。パッチ未適用の可能性"
grep -qE '^[[:space:]]*[^/[:space:]].*FANDHE_AI_CUDA_GRAPH_TRAIN' "$RUNNER_ROOT/crates/backend-cuda/src/graph.rs" \
  || precheck_fail "backend-cuda の graph.rs に FANDHE_AI_CUDA_GRAPH_TRAIN の読み出しが無い"
status_lines="$(git -C "$RUNNER_ROOT" status --porcelain | grep -v '^??' | sort || true)"
want_status=" M scripts/bench/framework-compare/bench-common/src/lib.rs
 M scripts/bench/framework-compare/bench-fandhe/src/main.rs"
[[ "$status_lines" == "$want_status" ]] \
  || precheck_fail "runner ツリーの変更が bench-common/lib.rs・bench-fandhe/main.rs の 2 件だけでない（${status_lines}）"
git -C "$RUNNER_ROOT" apply --reverse --check "$PATCH_FILE" || precheck_fail "patch が runner ツリーと一致しない"
# 完全一致検証: 一時 index に HEAD + 所定パッチだけを適用し、対象 2 ファイルの全内容が作業ツリーと一致すること。
TMP_IDX="$(mktemp "${TMPDIR:-/tmp}/ab-idx.XXXXXX")" || precheck_fail "一時 index を作れない"
rm -f "$TMP_IDX"
if ! GIT_INDEX_FILE="$TMP_IDX" git -C "$RUNNER_ROOT" read-tree HEAD \
  || ! GIT_INDEX_FILE="$TMP_IDX" git -C "$RUNNER_ROOT" apply --cached "$PATCH_FILE"; then
  rm -f "$TMP_IDX"; precheck_fail "HEAD へ所定パッチを適用できない"
fi
if ! GIT_INDEX_FILE="$TMP_IDX" git -C "$RUNNER_ROOT" diff --quiet -- \
  scripts/bench/framework-compare/bench-common/src/lib.rs \
  scripts/bench/framework-compare/bench-fandhe/src/main.rs; then
  rm -f "$TMP_IDX"; precheck_fail "対象 2 ファイルの内容が HEAD + 所定パッチと完全一致しない"
fi
rm -f "$TMP_IDX"
echo "precheck=assert ok"

bench_fandhe_setup_lock_restore_trap

PATCH_CFG="patch.crates-io.fandhe-ai.path=\"${FACADE}\""
TDIR_F="target-ab-train-graph-${LABEL}-fandhe"
TDIR_C="target-ab-train-graph-${LABEL}-candle"
TDIR_G="${RUNNER_ROOT}/target-ab-train-graph-${LABEL}-gates"
ROOT_MANIFEST="$RUNNER_ROOT/Cargo.toml"
# 全ビルドを計測の前に済ませる（計測中のビルド並走禁止）。
if ! cargo build --release -p bench-fandhe --target-dir "$TDIR_F" --config "$PATCH_CFG" 2>"$OUT/build-fandhe-${LABEL}.err"; then
  tail -40 "$OUT/build-fandhe-${LABEL}.err"; echo "bench-fandhe BUILD FAILED" >>"$SKIP"; exit 1
fi
tree="$(cargo tree -p bench-fandhe --depth 1 --config "$PATCH_CFG" 2>&1)"
if ! echo "$tree" | grep -qE 'fandhe-ai v[0-9.]+ \(.*crates/facade\)'; then
  echo "error: fandhe-ai が path patch に解決されていない" >&2; echo "$tree" >&2; exit 1
fi
echo "$tree" >"$OUT/tree-${LABEL}.txt"
if ! cargo build --release -p bench-candle --no-default-features --features cuda --target-dir "$TDIR_C" 2>"$OUT/build-candle-${LABEL}.err"; then
  tail -40 "$OUT/build-candle-${LABEL}.err"; echo "bench-candle BUILD FAILED" >>"$SKIP"; exit 1
fi
# 前提ゲート用テストバイナリも計測の前にビルドする（既定 feature・--release）。
build_gate_tests() {
  cargo test --manifest-path "$ROOT_MANIFEST" --locked --release --target-dir "$TDIR_G" \
    -p fandhe-ai-backend-cuda --lib --no-run \
  && cargo test --manifest-path "$ROOT_MANIFEST" --locked --release --target-dir "$TDIR_G" \
    -p fandhe-ai-backend-cuda --test graph_capture_real_device --test graph_capture_real_device_optin_off --no-run \
  && cargo test --manifest-path "$ROOT_MANIFEST" --locked --release --target-dir "$TDIR_G" \
    -p fandhe-ai --test cuda_graph_train_step_bit_identity --no-run
}
if ! build_gate_tests 2>"$OUT/build-gates-${LABEL}.err"; then
  tail -40 "$OUT/build-gates-${LABEL}.err"; echo "gate test BUILD FAILED" >>"$SKIP"; exit 1
fi
BIN_F="$OUT/bench-fandhe-${LABEL}"; BIN_C="$OUT/bench-candle-${LABEL}"
cp "$TDIR_F/release/bench-fandhe" "$BIN_F"
cp "$TDIR_C/release/bench-candle" "$BIN_C"
echo "$(sha256_of "$BIN_F")  bench-fandhe-${LABEL}" >"$OUT/sha-fandhe-${LABEL}.txt"
echo "$(sha256_of "$BIN_C")  bench-candle-${LABEL}" >"$OUT/sha-candle-${LABEL}.txt"

{
  echo "profile=gb10"
  echo "host=masked"
  echo "uname=$(uname -srm)"
  echo "gpu=${GPU_NAME}"
  echo "nvidia_driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1)"
  echo "rustc=$(rustc --version 2>/dev/null)"
  echo "rule_sha256=${RULE_SHA}"
  echo "patch_sha256=${PATCH_SHA}"
  echo "bench_py_sha256=${BENCHPY_SHA}"
  echo "torch_python=masked"
  echo "rev_stamp_head_runner=${STAMP_RUNNER}"
  echo "bin_sha256=$(awk '{print $1}' "$OUT/sha-fandhe-${LABEL}.txt")"
  echo "bin_candle_sha256=$(awk '{print $1}' "$OUT/sha-candle-${LABEL}.txt")"
  echo "arms=same_binary_env_toggle"
  echo "precheck=assert"
  echo "rounds=${ROUNDS}"
  echo "tf32=off"
} >"$ENV_INFO"

# ---- 前提ゲート R0〜R2（RULE.txt [前提ゲート]。各ログに rc= 行を残す。不合格なら計測しない） ----
# gate_run <log> <env_mode: off|on> <cmd...>: 環境変数は off で TRAIN=0（明示）・STEP/INFER 未設定、on で TRAIN=1 のみ。
gate_run() {
  local log=$1 mode=$2 rc
  shift 2
  if [[ "$mode" == "on" ]]; then
    env -u FANDHE_AI_CUDA_GRAPH_STEP -u FANDHE_AI_CUDA_GRAPH_INFER FANDHE_AI_CUDA_GRAPH_TRAIN=1 "$@" >"$log" 2>&1
  else
    env -u FANDHE_AI_CUDA_GRAPH_STEP -u FANDHE_AI_CUDA_GRAPH_INFER FANDHE_AI_CUDA_GRAPH_TRAIN=0 "$@" >"$log" 2>&1
  fi
  rc=$?
  echo "rc=${rc}" >>"$log"
  return "$rc"
}
CT=(cargo test --manifest-path "$ROOT_MANIFEST" --locked --release --target-dir "$TDIR_G")
PREGATE_OK=1
gate_check() { # <name> <rc>
  if [[ "$2" -ne 0 ]]; then
    echo "pregate ${1} FAILED rc=${2}" >>"$SKIP"; PREGATE_OK=0
  fi
}
gate_run "$OUT/r0-3252-${LABEL}.log" off "${CT[@]}" -p fandhe-ai-backend-cuda --lib train_step_chain_diag_tests_3252 -- --ignored --test-threads=1 --nocapture
gate_check r0-3252 $?
gate_run "$OUT/r0-real-device-${LABEL}.log" off "${CT[@]}" -p fandhe-ai-backend-cuda --test graph_capture_real_device -- --ignored --nocapture --test-threads=1
gate_check r0-real-device $?
gate_run "$OUT/r0-optin-off-${LABEL}.log" off "${CT[@]}" -p fandhe-ai-backend-cuda --test graph_capture_real_device_optin_off -- --ignored --nocapture
gate_check r0-optin-off $?
gate_run "$OUT/r1-facade-${LABEL}.log" on "${CT[@]}" -p fandhe-ai --test cuda_graph_train_step_bit_identity -- --ignored --nocapture --exact in_process_capture_matches_plain_chain_bit_exact
gate_check r1-facade $?
gate_run "$OUT/r1-3260-${LABEL}.log" on "${CT[@]}" -p fandhe-ai-backend-cuda --lib train_step_graph_diag_tests_3260 -- --ignored --test-threads=1 --nocapture
gate_check r1-3260 $?
gate_run "$OUT/r2-off-${LABEL}.log" off "${CT[@]}" -p fandhe-ai --test cuda_graph_train_step_bit_identity -- --ignored --nocapture --exact dump_bits_for_two_process_diff
gate_check r2-off $?
gate_run "$OUT/r2-on-${LABEL}.log" on "${CT[@]}" -p fandhe-ai --test cuda_graph_train_step_bit_identity -- --ignored --nocapture --exact dump_bits_for_two_process_diff
gate_check r2-on $?
grep '^BITS' "$OUT/r2-off-${LABEL}.log" >"$OUT/bits-off-${LABEL}.txt" || true
grep '^BITS' "$OUT/r2-on-${LABEL}.log" >"$OUT/bits-on-${LABEL}.txt" || true
if [[ ! -s "$OUT/bits-off-${LABEL}.txt" ]] || ! diff "$OUT/bits-off-${LABEL}.txt" "$OUT/bits-on-${LABEL}.txt" >"$OUT/r2-diff-${LABEL}.txt"; then
  echo "pregate r2-diff FAILED（BITS が空、または off／on で差がある）" >>"$SKIP"; PREGATE_OK=0
fi
if [[ "$PREGATE_OK" != "1" ]]; then
  # 判定ツールへ「計測欠測ではなく前提ゲート不合格で停止した」ことを渡す印（R1・R2 の失敗は REJECT。judge が生ログで再確認する）
  cp "$SKIP" "$OUT/pregate-stop-${LABEL}.txt"
  echo "undetermined: pregate（前提ゲート不合格のため計測しない。詳細は ${SKIP}）" | tee -a "$SKIP" >&2
  exit 1
fi

# ---- 計測 ----
# 専有ゲート。load1<1.0 かつ GPU 使用率 0% を 30 秒間隔で 3 サンプル連続（RULE.txt [専有ゲート]）。
GATE_START=$SECONDS
gate_ok_once() {
  local l1 util n
  n=0
  while [[ "$n" -lt 3 ]]; do
    # 実時間の期限（RULE.txt 最大 30 分）を検査より前に確認し、期限後の通過を受理しない
    [[ $((SECONDS - GATE_START)) -le 1800 ]] || return 1
    # 各サンプルで load1 と GPU 使用率の両方を確認する（RULE.txt: 3 サンプル連続）
    l1="$(uptime | sed -n 's/.*load average[s]*: *\([0-9.]*\).*/\1/p')"
    [[ "$l1" =~ ^[0-9]+(\.[0-9]+)?$ ]] || return 1
    [[ "$(awk -v a="$l1" -v t="$LOAD_GATE" 'BEGIN{print (a<t)?1:0}')" == "1" ]] || return 1
    util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ')"
    [[ "$util" == "0" ]] || return 1
    n=$((n + 1))
    if [[ "$n" -lt 3 ]]; then sleep 30; fi
  done
  return 0
}
wait_gate() { # <round> <logfile>
  local waited=0 ok=0 l1
  GATE_START=$SECONDS
  while :; do
    if gate_ok_once; then ok=1; break; fi
    # サンプル間の待機を含む実経過時間で期限を判定する
    if [[ $((SECONDS - GATE_START)) -ge 1800 ]]; then break; fi
    sleep 30
  done
  waited=$((SECONDS - GATE_START))
  l1="$(uptime | sed -n 's/.*load average[s]*: *\([0-9.]*\).*/\1/p')"
  if [[ "$ok" == "1" ]]; then
    echo "round$1 dev=cuda gate=pass gate_ok=1 threshold=${LOAD_GATE} waited_s=${waited} load1=${l1}" >>"$2"
  else
    echo "round$1 dev=cuda gate=fail gate_ok=0 threshold=${LOAD_GATE} waited_s=${waited} load1=${l1}" >>"$2"
  fi
}

run_arm() { # <arm> <round>
  local arm=$1 r=$2 errf res mode digest
  res="$OUT/results-${arm}-${LABEL}-cuda-train.jsonl"
  errf="$OUT/digest-${arm}-${LABEL}-cuda-train-r${r}.log"
  echo "== train cuda arm=${arm} round=${r} =="
  case "$arm" in
    candle) "$BIN_C" --task train --device cuda --size 64 --out "$res" 2>"$errf" ;;
    torch) "$AB_TORCH_PY" "$AB_BENCH_PY" --framework pytorch --task train --device cuda --size 64 --out "$res" >/dev/null 2>"$errf" ;;
    *)
      mode="${arm%-*}"
      digest=0
      if [[ "$mode" == "chain" || "$mode" == "reuse" ]]; then digest=1; fi
      case "$arm" in
        *-on) env -u FANDHE_AI_CUDA_GRAPH_STEP -u FANDHE_AI_CUDA_GRAPH_INFER FANDHE_AI_CUDA_GRAPH_TRAIN=1 \
                BENCH_PARAM_DIGEST="$digest" "$BIN_F" --task train --device cuda --mode "$mode" --out "$res" 2>"$errf" ;;
        *-off) env -u FANDHE_AI_CUDA_GRAPH_STEP -u FANDHE_AI_CUDA_GRAPH_INFER FANDHE_AI_CUDA_GRAPH_TRAIN=0 \
                BENCH_PARAM_DIGEST="$digest" "$BIN_F" --task train --device cuda --mode "$mode" --out "$res" 2>"$errf" ;;
      esac
      ;;
  esac
  # shellcheck disable=SC2181
  if [[ $? -ne 0 ]]; then
    echo "task=train dev=cuda arm=${arm} round=${r}: $(tail -3 "$errf" | tr '\n' ' ')" >>"$SKIP"
    ANY_FAILED=$((ANY_FAILED + 1))
  fi
}

# 起動順の回転（RULE.txt 固定。L の位置 3*(r-1) mod 8 から巡回）。
ARMS=(chain-off chain-on reuse-off reuse-on fresh-off fresh-on candle torch)
order_for_round() {
  local r=$1 start i out=""
  start=$(( (3 * (r - 1)) % 8 ))
  i=0
  while [[ "$i" -lt 8 ]]; do
    out="${out}${ARMS[$(( (start + i) % 8 ))]} "
    i=$((i + 1))
  done
  echo "$out"
}

for arm in "${ARMS[@]}"; do : >"$OUT/results-${arm}-${LABEL}-cuda-train.jsonl"; done
GATE_LOG="$OUT/load_gate-${LABEL}-cuda.log"
echo "series=official threshold=${LOAD_GATE} rule_threshold=${LOAD_GATE} precheck=assert" >"$GATE_LOG"
{ uptime; nvidia-smi; } >"$OUT/nvidia-smi-before-${LABEL}.txt" 2>&1 || true
for r in 1 2 3 4 5; do
  wait_gate "$r" "$GATE_LOG"
  for arm in $(order_for_round "$r"); do run_arm "$arm" "$r"; done
  echo "run ${r}: $(uptime)" | tee -a "$OUT/uptime-${LABEL}.log"
done
{ uptime; nvidia-smi; } >"$OUT/nvidia-smi-after-${LABEL}.txt" 2>&1 || true
echo "done. results in ${OUT}; skipped log: ${SKIP}"
if [[ "$ANY_FAILED" -gt 0 ]]; then echo "FAILED: ${ANY_FAILED} run(s)" >&2; exit 1; fi
