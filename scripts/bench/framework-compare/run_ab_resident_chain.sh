#!/usr/bin/env bash
# イシュー #3257: 常駐チェーン（train_step_resident／predict_resident_input）を現行経路と並べて計測する
# 両機体共用ランナー（参考系列。正式な 6 フレームワーク同席計測は #3273）。
# 規則の正本: docs/perf/logs/resident-chain-ab-3257/RULE.txt（計測前に固定。本スクリプトは sha256 を
# env_info へ記録し、judge_resident_chain_ab.py が現 RULE と突合する）。
#
# 1 ツリー・1 ビルド・2 モード: bench-fandhe は runner ツリーの `crates/facade` への
# `[patch.crates-io.fandhe-ai]` path patch（`--config` 引数のみ。Cargo.toml／Cargo.lock へは
# コミットしない）で 1 回だけビルドし、before = `--mode reuse`、after = `--mode chain`
# （bench-resident-chain.patch が追加。runner ツリーにだけ適用）で同じバイナリから呼ぶ。
# 相手（candle・PyTorch）も同じ round の中で測る。5 round・round ごとに 4 腕の起動順を回転。
#
# 呼び出し例:
#   AB_PROFILE=macmini AB_FACADE_PATH=<abs>/crates/facade AB_TORCH_PY=<abs>/venv-torch/bin/python \
#   AB_BENCH_PY=<abs>/bench_py.py AB_ROUND_COOLDOWN_MAX_LOAD1=3.0 bash run_ab_resident_chain.sh 3257-macmini
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
  macmini) DEVICES="cpu metal"; RULE_LOAD_GATE=3.0; CANDLE_FEATURE_ARGS=() ;; # 既定 feature（metal）
  gb10) DEVICES="cpu cuda"; RULE_LOAD_GATE=1.0; CANDLE_FEATURE_ARGS=(--no-default-features --features cuda) ;; # Linux では metal 依存を除き CUDA でビルド（run_all_cuda.sh と同型）
  *) echo "error: AB_PROFILE は macmini か gb10（got: ${PROFILE:-unset}）" >&2; exit 1 ;;
esac

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
RULE_FILE="$RUNNER_ROOT/docs/perf/logs/resident-chain-ab-3257/RULE.txt"
PATCH_FILE="$RUNNER_ROOT/docs/perf/logs/resident-chain-ab-3257/bench-resident-chain.patch"

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
BENCHPY_SHA="$(sha256_of "$AB_BENCH_PY")"

OUT="results/raw"
mkdir -p "$OUT"
SKIP="$OUT/skipped-resident-chain-${LABEL}.log"
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

# precheck（fail-closed。RULE.txt [precheck]）。
precheck_fail() { echo "precheck FAILED: $1" | tee -a "$SKIP" >&2; exit 1; }
SRC="$FACADE/src"
grep -rqE '^[[:space:]]*pub fn train_step_resident\(' "$SRC" || precheck_fail "facade に train_step_resident が無い"
grep -rqE '^[[:space:]]*pub fn predict_resident_input\(' "$SRC" || precheck_fail "facade に predict_resident_input が無い"
grep -rqE '^[[:space:]]*pub fn upload_resident\(' "$SRC" || precheck_fail "facade に upload_resident が無い"
grep -rqE '^impl ResidentLoss \{' "$SRC" || precheck_fail "facade に impl ResidentLoss が無い"
grep -rqE '^[[:space:]]*pub fn read\(&self, store: &mut DeviceParamStore\)' "$SRC" || precheck_fail "ResidentLoss::read が無い"
MAIN_RS="bench-fandhe/src/main.rs"
grep -qE '^[[:space:]]*\("train", "chain", false\) =>' "$MAIN_RS" || precheck_fail "main.rs に (train, chain) の分岐が無い。パッチ未適用の可能性"
grep -qE '^[[:space:]]*\("infer", "chain", false\) =>' "$MAIN_RS" || precheck_fail "main.rs に (infer, chain) の分岐が無い"
status_lines="$(git -C "$RUNNER_ROOT" status --porcelain | grep -v '^??' | sort || true)"
want_status=" M scripts/bench/framework-compare/bench-common/src/lib.rs
 M scripts/bench/framework-compare/bench-fandhe/src/main.rs"
[[ "$status_lines" == "$want_status" ]] \
  || precheck_fail "runner ツリーの変更が bench-common/lib.rs・bench-fandhe/main.rs の 2 件だけでない（${status_lines}）"
git -C "$RUNNER_ROOT" apply --reverse --check "$PATCH_FILE" || precheck_fail "patch が runner ツリーと一致しない"
# 完全一致検証: 一時 index に HEAD + 所定パッチだけを適用し、対象 2 ファイルの全内容が作業ツリーと
# 一致することを確認する（逆適用チェックだけでは同ファイルの別位置の変更を見逃す）。
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
TDIR_F="target-ab-chain-${LABEL}-fandhe"
TDIR_C="target-ab-chain-${LABEL}-candle"
# 全ビルドを計測の前に済ませる（計測中のビルド並走禁止）。
if ! cargo build --release -p bench-fandhe --target-dir "$TDIR_F" --config "$PATCH_CFG" 2>"$OUT/build-fandhe-${LABEL}.err"; then
  tail -40 "$OUT/build-fandhe-${LABEL}.err"; echo "bench-fandhe BUILD FAILED" >>"$SKIP"; exit 1
fi
tree="$(cargo tree -p bench-fandhe --depth 1 --config "$PATCH_CFG" 2>&1)"
if ! echo "$tree" | grep -qE 'fandhe-ai v[0-9.]+ \(.*crates/facade\)'; then
  echo "error: fandhe-ai が path patch に解決されていない" >&2; echo "$tree" >&2; exit 1
fi
echo "$tree" >"$OUT/tree-${LABEL}.txt"
if ! cargo build --release -p bench-candle ${CANDLE_FEATURE_ARGS[@]+"${CANDLE_FEATURE_ARGS[@]}"} --target-dir "$TDIR_C" 2>"$OUT/build-candle-${LABEL}.err"; then
  tail -40 "$OUT/build-candle-${LABEL}.err"; echo "bench-candle BUILD FAILED" >>"$SKIP"; exit 1
fi
BIN_F="$OUT/bench-fandhe-${LABEL}"; BIN_C="$OUT/bench-candle-${LABEL}"
cp "$TDIR_F/release/bench-fandhe" "$BIN_F"
cp "$TDIR_C/release/bench-candle" "$BIN_C"
echo "$(sha256_of "$BIN_F")  bench-fandhe-${LABEL}" >"$OUT/sha-fandhe-${LABEL}.txt"
echo "$(sha256_of "$BIN_C")  bench-candle-${LABEL}" >"$OUT/sha-candle-${LABEL}.txt"

{
  echo "profile=${PROFILE}"
  echo "host=masked"
  echo "uname=$(uname -srm)"
  if [[ "$PROFILE" == "macmini" ]]; then echo "cpu_brand=$(sysctl -n machdep.cpu.brand_string)"; else echo "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)"; fi
  echo "rule_sha256=${RULE_SHA}"
  echo "patch_sha256=${PATCH_SHA}"
  echo "bench_py_sha256=${BENCHPY_SHA}"
  echo "torch_python=masked"
  echo "rev_stamp_head_runner=${STAMP_RUNNER}"
  echo "bin_sha256=$(awk '{print $1}' "$OUT/sha-fandhe-${LABEL}.txt")"
  echo "bin_candle_sha256=$(awk '{print $1}' "$OUT/sha-candle-${LABEL}.txt")"
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

run_arm() { # <dev> <task> <arm> <round>
  local dev=$1 task=$2 arm=$3 r=$4 errf res
  res="$OUT/results-${arm}-${LABEL}-${dev}-${task}.jsonl"
  errf="$OUT/digest-${arm}-${LABEL}-${dev}-${task}-r${r}.log"
  echo "== ${task} ${dev} arm=${arm} round=${r} =="
  case "$arm" in
    before) BENCH_PARAM_DIGEST=1 "$BIN_F" --task "$task" --device "$dev" --mode reuse --out "$res" 2>"$errf" ;;
    after) BENCH_PARAM_DIGEST=1 "$BIN_F" --task "$task" --device "$dev" --mode chain --out "$res" 2>"$errf" ;;
    candle) "$BIN_C" --task "$task" --device "$dev" --size 64 --out "$res" 2>"$errf" ;;
    torch) "$AB_TORCH_PY" "$AB_BENCH_PY" --framework pytorch --task "$task" --device "$dev" --size 64 --out "$res" >/dev/null 2>"$errf" ;;
  esac
  # shellcheck disable=SC2181
  if [[ $? -ne 0 ]]; then
    echo "dev=${dev} task=${task} arm=${arm} round=${r}: $(tail -3 "$errf" | tr '\n' ' ')" >>"$SKIP"
    ANY_FAILED=$((ANY_FAILED + 1))
  fi
}

# 起動順の回転（RULE.txt 固定。B=before A=after C=candle T=torch）。r1: B A C T / r2: A C T B / r3: C T B A / r4: T B A C / r5: r1 と同じ。
order_for_round() {
  case $(( ($1 - 1) % 4 )) in
    0) echo "before after candle torch" ;;
    1) echo "after candle torch before" ;;
    2) echo "candle torch before after" ;;
    3) echo "torch before after candle" ;;
  esac
}

pm_log() { if [[ "$PROFILE" == "macmini" ]]; then pmset -g therm >"$OUT/pmset_therm_$1-${LABEL}.txt" 2>&1 || true; fi; }
pm_log before
for dev in $DEVICES; do
  for arm in before after candle torch; do
    for task in train infer; do : >"$OUT/results-${arm}-${LABEL}-${dev}-${task}.jsonl"; done
  done
  GATE_LOG="$OUT/load_gate-${LABEL}-${dev}.log"
  echo "series=official threshold=${RULE_LOAD_GATE} rule_threshold=${RULE_LOAD_GATE} precheck=assert" >"$GATE_LOG"
  ab_round_cooldown_init "$OUT/cooldown-${LABEL}-${dev}.log"
  for r in 1 2 3 4 5; do
    ab_round_cooldown "$r"
    wait_gate "$dev" "$r" "$GATE_LOG"
    for task in train infer; do
      for arm in $(order_for_round "$r"); do run_arm "$dev" "$task" "$arm" "$r"; done
    done
    echo "dev=${dev} run ${r}: $(uptime)" | tee -a "$OUT/uptime-${LABEL}.log"
  done
done
pm_log after
echo "done. results in ${OUT}; skipped log: ${SKIP}"
if [[ "$ANY_FAILED" -gt 0 ]]; then echo "FAILED: ${ANY_FAILED} run(s)" >&2; exit 1; fi
