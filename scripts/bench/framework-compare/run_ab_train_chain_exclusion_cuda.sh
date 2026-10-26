#!/usr/bin/env bash
# イシュー #3596: 学習チェーンと Graph capture の ordinal 単位排他（#3597）の非後退を GB10 の train CUDA chain セルで確かめる。
# GB10（aarch64・CUDA）専用ランナー。規則の正本: docs/perf/logs/train-chain-exclusion-3596/RULE.txt
# （計測前に固定。本スクリプトは sha256 を env_info へ記録し、judge_train_chain_exclusion_ab.py が現 RULE と突合する）。
#
# 排他に当たる実行時トグルは無いため 2 ツリー方式: before（excl0 = #3597 の直前）と after（excl1 = 本ランナーのツリー）を
# それぞれ `crates/facade` への `[patch.crates-io.fandhe-ai]` path patch（`--config` 引数のみ。コミットしない）で別ビルドする。
# 両ツリーには計測専用パッチ bench-resident-chain.patch（`--mode chain`）を適用済みであること。
# 腕は {excl0, excl1} × {graph off, on}。on = FANDHE_AI_CUDA_GRAPH_TRAIN=1 のみ、off = 3 変数とも未設定。
# 前提ゲート g1〜g4（after ツリーの実機テスト）を計測の前に実行し、不合格なら計測しない。
#
# 呼び出し例:
#   AB_BEFORE_FACADE_PATH=<abs>/before/crates/facade AB_AFTER_FACADE_PATH=<abs>/after/crates/facade \
#   bash run_ab_train_chain_exclusion_cuda.sh 3596-gb10
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
if [[ "$AFTER_ROOT" != "$RUNNER_ROOT" ]]; then
  echo "error: AB_AFTER_FACADE_PATH が runner ツリーの crates/facade でない（after = runner）" >&2; exit 1
fi
if [[ "$BEFORE_ROOT" == "$AFTER_ROOT" ]]; then
  echo "error: before と after が同一ツリー" >&2; exit 1
fi
RULE_FILE="$RUNNER_ROOT/docs/perf/logs/train-chain-exclusion-3596/RULE.txt"
PATCH_FILE="$RUNNER_ROOT/docs/perf/logs/resident-chain-ab-3257/bench-resident-chain.patch"
ROOT_MANIFEST="$RUNNER_ROOT/Cargo.toml"

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
# RULE.txt に値で固定した値の読み出しは行頭アンカー（コメント行・本文中の言及では素通りさせない）。
RULE_PATCH_SHA="$(sed -n 's/^  patch_sha256: \([0-9a-f]\{64\}\)$/\1/p' "$RULE_FILE" | head -n1)"
EXPECT_BEFORE="$(sed -n 's/^- before = \([0-9a-f]\{40\}\)$/\1/p' "$RULE_FILE" | head -n1)"
CRATES_REF="$(sed -n 's/^- after_crates_identical_to = \([0-9a-f]\{40\}\)$/\1/p' "$RULE_FILE" | head -n1)"
if [[ -z "$RULE_PATCH_SHA" || "$RULE_PATCH_SHA" != "$PATCH_SHA" ]]; then
  echo "error: RULE.txt の patch_sha256 と実パッチの sha256 が一致しない" >&2; exit 1
fi
if [[ -z "$EXPECT_BEFORE" || -z "$CRATES_REF" ]]; then
  echo "error: RULE.txt から before SHA／after_crates_identical_to を読めない" >&2; exit 1
fi

OUT="results/raw"
mkdir -p "$OUT"
SKIP="$OUT/skipped-train-chain-exclusion-${LABEL}.log"
ENV_INFO="$OUT/env_info-${LABEL}.txt"
# 同じラベルの既存出力があれば停止（run の差し替え・追記禁止。A08）。
if [[ -e "$SKIP" || -e "$ENV_INFO" ]]; then
  echo "error: ラベル ${LABEL} の出力が既に存在する。新しいラベルを使うこと" >&2; exit 1
fi
: >"$SKIP"
ANY_FAILED=0
unset RAYON_NUM_THREADS

precheck_fail() { echo "precheck FAILED: $1" | tee -a "$SKIP" >&2; exit 1; }

# ドリフトガード: 各ツリーの .rev-stamp 1 行目が実 HEAD と一致、2 行目が dirty=0。
stamp_head() { # <root>
  local stamp real dirty
  stamp="$(head -n1 "$1/.rev-stamp" 2>/dev/null || true)"
  dirty="$(sed -n 2p "$1/.rev-stamp" 2>/dev/null || true)"
  real="$(git -C "$1" rev-parse HEAD 2>/dev/null || true)"
  if [[ -z "$stamp" || "$stamp" != "$real" || "$dirty" != "dirty=0" ]]; then
    echo "error: $1 の .rev-stamp が無い・HEAD と不一致・dirty=0 でない（stamp=${stamp:-none} head=${real:-none}）" >&2; exit 1
  fi
  echo "$stamp"
}
STAMP_BEFORE="$(stamp_head "$BEFORE_ROOT")" || exit 1
STAMP_AFTER="$(stamp_head "$AFTER_ROOT")" || exit 1
if [[ "$STAMP_BEFORE" != "$EXPECT_BEFORE" ]]; then
  precheck_fail "before の HEAD が RULE.txt の登録 SHA と不一致（head=${STAMP_BEFORE} expected=${EXPECT_BEFORE}）"
fi
# after は crates/ が RULE の基準コミットと同一であること（RULE 自身のコミット SHA は固定できないため）。
if ! git -C "$AFTER_ROOT" cat-file -e "${CRATES_REF}^{commit}" 2>/dev/null; then
  precheck_fail "after ツリーに基準コミット ${CRATES_REF} が無い（git bundle に履歴を含めること）"
fi
git -C "$AFTER_ROOT" diff --quiet "$CRATES_REF" -- crates || precheck_fail "after の crates/ が基準コミットと同一でない"
if git -C "$BEFORE_ROOT" diff --quiet "$CRATES_REF" -- crates 2>/dev/null; then
  precheck_fail "before の crates/ が after と同一（腕が区別できていない）"
fi
STAMP_RUNNER="$STAMP_AFTER"

# 両ツリーが HEAD + 所定パッチだけの状態であること（一時 index による完全一致検証）。
check_patched_tree() { # <root>
  local root=$1 status_lines want_status tmp_idx
  git -C "$root" apply --reverse --check "$PATCH_FILE" || precheck_fail "patch が ${root} と一致しない"
  status_lines="$(git -C "$root" status --porcelain | grep -v '^??' | sort || true)"
  want_status=" M scripts/bench/framework-compare/bench-common/src/lib.rs
 M scripts/bench/framework-compare/bench-fandhe/src/main.rs"
  [[ "$status_lines" == "$want_status" ]] || precheck_fail "${root} の変更が bench-common/lib.rs・bench-fandhe/main.rs の 2 件だけでない（${status_lines}）"
  tmp_idx="$(mktemp "${TMPDIR:-/tmp}/ab-idx.XXXXXX")" || precheck_fail "一時 index を作れない"
  rm -f "$tmp_idx"
  if ! GIT_INDEX_FILE="$tmp_idx" git -C "$root" read-tree HEAD \
    || ! GIT_INDEX_FILE="$tmp_idx" git -C "$root" apply --cached "$PATCH_FILE"; then
    rm -f "$tmp_idx"; precheck_fail "${root} の HEAD へ所定パッチを適用できない"
  fi
  if ! GIT_INDEX_FILE="$tmp_idx" git -C "$root" diff --quiet -- \
    scripts/bench/framework-compare/bench-common/src/lib.rs \
    scripts/bench/framework-compare/bench-fandhe/src/main.rs; then
    rm -f "$tmp_idx"; precheck_fail "${root} の対象 2 ファイルが HEAD + 所定パッチと完全一致しない"
  fi
  rm -f "$tmp_idx"
  grep -qE '^[[:space:]]*\("train", "chain", false\) =>' "$root/scripts/bench/framework-compare/bench-fandhe/src/main.rs" \
    || precheck_fail "${root} の main.rs に (train, chain) の分岐が無い"
}
check_patched_tree "$BEFORE_ROOT"
check_patched_tree "$AFTER_ROOT"
# 排他の有無（after にだけ begin_device_chain_scope が存在する）を行頭アンカーで確認する。
grep -rqE '^[[:space:]]*(pub )?fn begin_device_chain_scope' "$AFTER_ROOT/crates" \
  || precheck_fail "after に begin_device_chain_scope が無い"
if grep -rqE '^[[:space:]]*(pub )?fn begin_device_chain_scope' "$BEFORE_ROOT/crates"; then
  precheck_fail "before に begin_device_chain_scope がある（排他導入前でない）"
fi
echo "precheck=assert ok"

bench_fandhe_setup_lock_restore_trap

build_arm() { # <arm: excl1（runner ツリー）> <facade>
  local arm=$1 facade=$2 tdir="target-ab-chain-excl-${LABEL}-$1" cfg tree
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
# 全ビルドを計測の前に済ませる（計測中のビルド並走禁止）。bench は各ツリーの bench ソースを使うため、ビルドは各ツリーの作業ディレクトリで行う。
build_arm excl1 "$AFTER_FACADE"
# before は bench ソース（パッチ適用済み）が別ツリーにあるため、そのツリーの framework-compare でビルドして成果物を回収する。
build_before() {
  local facade=$BEFORE_FACADE tdir="target-ab-chain-excl-${LABEL}-excl0" cfg tree
  local fc="$BEFORE_ROOT/scripts/bench/framework-compare"
  local out_abs
  out_abs="$(pwd)/$OUT"
  cfg="patch.crates-io.fandhe-ai.path=\"${facade}\""
  if ! ( cd "$fc" && cargo build --release -p bench-fandhe --target-dir "$tdir" --config "$cfg" ) 2>"$OUT/build-excl0-${LABEL}.err"; then
    tail -40 "$OUT/build-excl0-${LABEL}.err"
    echo "bench-fandhe BUILD FAILED (excl0)" >>"$SKIP"; exit 1
  fi
  tree="$(cd "$fc" && cargo tree -p bench-fandhe --depth 1 --config "$cfg" 2>&1)"
  if ! echo "$tree" | grep -qE 'fandhe-ai v[0-9.]+ \(.*crates/facade\)'; then
    echo "error: fandhe-ai が path patch に解決されていない (excl0)" >&2; echo "$tree" >&2; exit 1
  fi
  echo "$tree" >"$OUT/tree-excl0-${LABEL}.txt"
  cp "$fc/$tdir/release/bench-fandhe" "$out_abs/bench-fandhe-excl0-${LABEL}"
  echo "$(sha256_of "$out_abs/bench-fandhe-excl0-${LABEL}")  bench-fandhe-excl0-${LABEL}" >"$OUT/sha-excl0-${LABEL}.txt"
}
build_before

# 前提ゲート用テストバイナリも計測の前にビルドする（after ツリー・既定 feature・--release）。
TDIR_G="${RUNNER_ROOT}/target-ab-chain-excl-${LABEL}-gates"
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
BIN_0="$OUT/bench-fandhe-excl0-${LABEL}"
BIN_1="$OUT/bench-fandhe-excl1-${LABEL}"

{
  echo "profile=gb10"
  echo "host=masked"
  echo "uname=$(uname -srm)"
  echo "gpu=${GPU_NAME}"
  echo "nvidia_driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1)"
  echo "rustc=$(rustc --version 2>/dev/null)"
  echo "rule_sha256=${RULE_SHA}"
  echo "patch_sha256=${PATCH_SHA}"
  echo "rev_stamp_head_before=${STAMP_BEFORE}"
  echo "rev_stamp_head_after=${STAMP_AFTER}"
  echo "rev_stamp_head_runner=${STAMP_RUNNER}"
  echo "crates_identical_to=${CRATES_REF}"
  echo "bin_sha256_excl0=$(awk '{print $1}' "$OUT/sha-excl0-${LABEL}.txt")"
  echo "bin_sha256_excl1=$(awk '{print $1}' "$OUT/sha-excl1-${LABEL}.txt")"
  echo "arms=two_trees_env_toggle"
  echo "precheck=assert"
  echo "rounds=${ROUNDS}"
  echo "tf32=off"
} >"$ENV_INFO"

# ---- 前提ゲート g1〜g4（RULE.txt [前提ゲート]。各ログに rc= 行を残す。不合格なら計測しない） ----
gate_run() { # <log> <env_mode: off|on> <cmd...>
  local log=$1 mode=$2 rc
  shift 2
  if [[ "$mode" == "on" ]]; then
    env -u FANDHE_AI_CUDA_GRAPH_STEP -u FANDHE_AI_CUDA_GRAPH_INFER FANDHE_AI_CUDA_GRAPH_TRAIN=1 "$@" >"$log" 2>&1
  else
    env -u FANDHE_AI_CUDA_GRAPH_TRAIN -u FANDHE_AI_CUDA_GRAPH_STEP -u FANDHE_AI_CUDA_GRAPH_INFER "$@" >"$log" 2>&1
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
gate_run "$OUT/g1-unfenced-${LABEL}.log" on "${CT[@]}" -p fandhe-ai-backend-cuda --lib \
  train_step_graph_diag_tests_3260::unfenced_chain_and_capture_threads_do_not_poison_and_stay_bit_exact -- --ignored --exact --test-threads=1 --nocapture
gate_check g1-unfenced $?
gate_run "$OUT/g2-3260-${LABEL}.log" on "${CT[@]}" -p fandhe-ai-backend-cuda --lib train_step_graph_diag_tests_3260 -- --ignored --test-threads=1 --nocapture
gate_check g2-3260 $?
gate_run "$OUT/g3-3252-${LABEL}.log" off "${CT[@]}" -p fandhe-ai-backend-cuda --lib train_step_chain_diag_tests_3252 -- --ignored --test-threads=1 --nocapture
gate_check g3-3252 $?
gate_run "$OUT/g3-real-device-${LABEL}.log" off "${CT[@]}" -p fandhe-ai-backend-cuda --test graph_capture_real_device -- --ignored --nocapture --test-threads=1
gate_check g3-real-device $?
gate_run "$OUT/g3-optin-off-${LABEL}.log" off "${CT[@]}" -p fandhe-ai-backend-cuda --test graph_capture_real_device_optin_off -- --ignored --nocapture
gate_check g3-optin-off $?
gate_run "$OUT/g4-facade-${LABEL}.log" on "${CT[@]}" -p fandhe-ai --test cuda_graph_train_step_bit_identity -- --ignored --nocapture --exact in_process_capture_matches_plain_chain_bit_exact
gate_check g4-facade $?
gate_run "$OUT/g4-off-${LABEL}.log" off "${CT[@]}" -p fandhe-ai --test cuda_graph_train_step_bit_identity -- --ignored --nocapture --exact dump_bits_for_two_process_diff
gate_check g4-off $?
gate_run "$OUT/g4-on-${LABEL}.log" on "${CT[@]}" -p fandhe-ai --test cuda_graph_train_step_bit_identity -- --ignored --nocapture --exact dump_bits_for_two_process_diff
gate_check g4-on $?
grep '^BITS' "$OUT/g4-off-${LABEL}.log" >"$OUT/bits-off-${LABEL}.txt" || true
grep '^BITS' "$OUT/g4-on-${LABEL}.log" >"$OUT/bits-on-${LABEL}.txt" || true
if [[ ! -s "$OUT/bits-off-${LABEL}.txt" ]] || ! diff "$OUT/bits-off-${LABEL}.txt" "$OUT/bits-on-${LABEL}.txt" >"$OUT/g4-diff-${LABEL}.txt"; then
  echo "pregate g4-diff FAILED（BITS が空、または off／on で差がある）" >>"$SKIP"; PREGATE_OK=0
fi
if [[ "$PREGATE_OK" != "1" ]]; then
  # 判定ツールへ「計測欠測ではなく前提ゲート不合格で停止した」ことを渡す印（judge が生ログで再確認する）
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
    [[ $((SECONDS - GATE_START)) -le 1800 ]] || return 1
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

run_arm() { # <arm: excl{0,1}-{off,on}> <round>
  local arm=$1 r=$2 errf res bin
  res="$OUT/results-${arm}-${LABEL}-cuda-train.jsonl"
  errf="$OUT/digest-${arm}-${LABEL}-cuda-train-r${r}.log"
  case "$arm" in
    excl0-*) bin=$BIN_0 ;;
    excl1-*) bin=$BIN_1 ;;
    *) echo "error: 未知の腕 ${arm}" >&2; exit 1 ;;
  esac
  echo "== train cuda chain arm=${arm} round=${r} =="
  case "$arm" in
    *-on) env -u FANDHE_AI_CUDA_GRAPH_STEP -u FANDHE_AI_CUDA_GRAPH_INFER FANDHE_AI_CUDA_GRAPH_TRAIN=1 \
            BENCH_PARAM_DIGEST=1 "$bin" --task train --device cuda --mode chain --out "$res" 2>"$errf" ;;
    *-off) env -u FANDHE_AI_CUDA_GRAPH_TRAIN -u FANDHE_AI_CUDA_GRAPH_STEP -u FANDHE_AI_CUDA_GRAPH_INFER \
            BENCH_PARAM_DIGEST=1 "$bin" --task train --device cuda --mode chain --out "$res" 2>"$errf" ;;
  esac
  # shellcheck disable=SC2181
  if [[ $? -ne 0 ]]; then
    echo "task=train dev=cuda arm=${arm} round=${r}: $(tail -3 "$errf" | tr '\n' ' ')" >>"$SKIP"
    ANY_FAILED=$((ANY_FAILED + 1))
  fi
}

# 起動順の回転（RULE.txt 固定。L の位置 (r-1) mod 4 から巡回）。
ARMS=(excl0-off excl1-off excl0-on excl1-on)
order_for_round() {
  local r=$1 start i out=""
  start=$(( (r - 1) % 4 ))
  i=0
  while [[ "$i" -lt 4 ]]; do
    out="${out}${ARMS[$(( (start + i) % 4 ))]} "
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
