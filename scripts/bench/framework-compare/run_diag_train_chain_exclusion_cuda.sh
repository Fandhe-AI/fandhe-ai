#!/usr/bin/env bash
# イシュー #3596 r4: 既定 OFF の chain セルに残った約 0.9 µs の差（RULE-r3 UNDETERMINED）の出どころを GB10 で切り分ける診断ランナー。
# 手順の正本: docs/perf/logs/train-chain-exclusion-3596/DIAG-r4.txt（計測前に固定。本スクリプトは sha256 を env_info へ記録し、
# summarize_diag_train_chain_exclusion.py が DIAG と突合する）。判定ではなく分類を出す診断であり、judge ツールの入力ではない。
# run_ab_train_chain_exclusion_cuda.sh は 2 腕固定・排他導入前後の precheck を持ち A/A・revert 腕を拒否するため流用せず、
# 同じ検証（env 漏れ拒否・facade パス検証・.rev-stamp・crates 同一性・一時 index による完全一致）を 7 腕向けに持つ。既存ランナーは変更しない。
#
# 腕（すべて chain セル・train CUDA・size=64・graph 環境変数 3 つ未設定）:
#   aa0 / aa1 = before の同一バイナリ、excl0-rebuild = before の再ビルド、excl1 = after、
#   rv-scopecall / rv-kindlookup / rv-graphlock = after に計測専用 revert パッチ 1 本（diag-r4/）を当てたもの。
# 呼び出し例:
#   DIAG_NAME=DIAG-r4.txt AB_BEFORE_FACADE_PATH=<abs>/before/crates/facade AB_AFTER_FACADE_PATH=<abs>/after/crates/facade \
#   bash run_diag_train_chain_exclusion_cuda.sh 3596-diag-r4-gb10
# bash 3.2 互換（mapfile・連想配列・eval 不使用）。jq 不使用。失敗は捏造せず skipped へ記録し非 0 終了。
set -u
cd "$(dirname "$0")" || exit 1
# shellcheck source=./bench_fandhe_lock_restore.sh
source ./bench_fandhe_lock_restore.sh

LABEL=${1:-}
ROUNDS=${AB_ROUNDS:-5}
LOAD_GATE=1.0
RV_ARMS=(rv-scopecall rv-kindlookup rv-graphlock)
ARMS=(aa0 aa1 excl0-rebuild excl1 rv-scopecall rv-kindlookup rv-graphlock)
NARMS=7

if [[ -z "$LABEL" || ! "$LABEL" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "usage: $0 <label>  (label: [A-Za-z0-9._-]+)" >&2
  exit 1
fi
if [[ "$ROUNDS" != "5" ]]; then
  echo "error: AB_ROUNDS は 5 固定（DIAG-r4.txt。got: ${ROUNDS}）" >&2
  exit 1
fi
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
# DIAG ファイル名は固定の正規表現のみ受け、固定ディレクトリ配下でのみ解決する（パス走査防止。A03）。
DIAG_NAME="${DIAG_NAME-DIAG-r4.txt}"
if [[ ! "$DIAG_NAME" =~ ^DIAG-[a-z0-9]+\.txt$ ]]; then
  echo "error: DIAG_NAME は DIAG-<英小文字数字>.txt のファイル名のみ（got: ${DIAG_NAME}）" >&2; exit 1
fi
LOGDIR="$RUNNER_ROOT/docs/perf/logs/train-chain-exclusion-3596"
DIAG_FILE="$LOGDIR/${DIAG_NAME}"
PATCH_FILE="$RUNNER_ROOT/docs/perf/logs/resident-chain-ab-3257/bench-resident-chain.patch"

if [[ "$(uname -m)" != "aarch64" ]]; then echo "error: aarch64（GB10）のみ" >&2; exit 1; fi
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)"
if [[ "$GPU_NAME" != *GB10* ]]; then
  echo "error: nvidia-smi の GPU 名に GB10 を含まない（got: ${GPU_NAME:-unknown}）" >&2; exit 1
fi

sha256_of() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'; else shasum -a 256 "$1" | awk '{print $1}'; fi; }
[[ -f "$DIAG_FILE" && -f "$PATCH_FILE" ]] || { echo "error: DIAG / bench patch が無い" >&2; exit 1; }
DIAG_SHA="$(sha256_of "$DIAG_FILE")"
PATCH_SHA="$(sha256_of "$PATCH_FILE")"
# DIAG に値で固定した値の読み出しは行頭アンカー（コメント行・本文中の言及では素通りさせない）。
DIAG_PATCH_SHA="$(sed -n 's/^  patch_sha256: \([0-9a-f]\{64\}\)$/\1/p' "$DIAG_FILE" | head -n1)"
EXPECT_BEFORE="$(sed -n 's/^- before = \([0-9a-f]\{40\}\)$/\1/p' "$DIAG_FILE" | head -n1)"
CRATES_REF="$(sed -n 's/^- after_crates_identical_to = \([0-9a-f]\{40\}\)$/\1/p' "$DIAG_FILE" | head -n1)"
if [[ -z "$DIAG_PATCH_SHA" || "$DIAG_PATCH_SHA" != "$PATCH_SHA" ]]; then
  echo "error: DIAG の patch_sha256 と実パッチの sha256 が一致しない" >&2; exit 1
fi
if [[ -z "$EXPECT_BEFORE" || -z "$CRATES_REF" ]]; then
  echo "error: DIAG から before SHA／after_crates_identical_to を読めない" >&2; exit 1
fi
# revert パッチ 3 本の sha256 を DIAG の登録値と照合する。
RV_SHA_LINES=""
for rv in "${RV_ARMS[@]}"; do
  f="$LOGDIR/diag-r4/${rv}.patch"
  [[ -f "$f" ]] || { echo "error: ${f} が無い" >&2; exit 1; }
  want="$(sed -n "s/^  ${rv}_sha256: \([0-9a-f]\{64\}\)\$/\1/p" "$DIAG_FILE" | head -n1)"
  got="$(sha256_of "$f")"
  if [[ -z "$want" || "$want" != "$got" ]]; then
    echo "error: ${rv}.patch の sha256 が DIAG の登録値と一致しない" >&2; exit 1
  fi
  RV_SHA_LINES="${RV_SHA_LINES}${rv}_sha256=${got}
"
done

OUT="results/raw"
mkdir -p "$OUT"
SKIP="$OUT/skipped-diag-train-chain-exclusion-${LABEL}.log"
ENV_INFO="$OUT/env_info-diag-${LABEL}.txt"
if [[ -e "$SKIP" || -e "$ENV_INFO" ]]; then
  echo "error: ラベル ${LABEL} の出力が既に存在する。新しいラベルを使うこと" >&2; exit 1
fi
: >"$SKIP"
ANY_FAILED=0
unset RAYON_NUM_THREADS

precheck_fail() { echo "precheck FAILED: $1" | tee -a "$SKIP" >&2; exit 1; }

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
  precheck_fail "before の HEAD が DIAG の登録 SHA と不一致（head=${STAMP_BEFORE} expected=${EXPECT_BEFORE}）"
fi
if ! git -C "$AFTER_ROOT" cat-file -e "${CRATES_REF}^{commit}" 2>/dev/null; then
  precheck_fail "after ツリーに基準コミット ${CRATES_REF} が無い（git bundle に履歴を含めること）"
fi
git -C "$AFTER_ROOT" diff --quiet "$CRATES_REF" -- crates || precheck_fail "after の crates/ が基準コミットと同一でない"
if git -C "$BEFORE_ROOT" diff --quiet "$CRATES_REF" -- crates 2>/dev/null; then
  precheck_fail "before の crates/ が after と同一（腕が区別できていない）"
fi

# ツリーが「HEAD + bench パッチ（+ 当該 revert パッチ）」だけであることを、一時 index による完全一致で検証する。
check_patched_tree() { # <root> [<extra_patch>]
  local root=$1 extra=${2:-} tmp_idx
  git -C "$root" apply --reverse --check "$PATCH_FILE" 2>/dev/null || precheck_fail "bench patch が ${root} と一致しない"
  tmp_idx="$(mktemp "${TMPDIR:-/tmp}/diag-idx.XXXXXX")" || precheck_fail "一時 index を作れない"
  rm -f "$tmp_idx"
  if ! GIT_INDEX_FILE="$tmp_idx" git -C "$root" read-tree HEAD \
    || ! GIT_INDEX_FILE="$tmp_idx" git -C "$root" apply --cached "$PATCH_FILE"; then
    rm -f "$tmp_idx"; precheck_fail "${root} の HEAD へ bench パッチを適用できない"
  fi
  if [[ -n "$extra" ]] && ! GIT_INDEX_FILE="$tmp_idx" git -C "$root" apply --cached "$extra"; then
    rm -f "$tmp_idx"; precheck_fail "${root} の HEAD へ ${extra} を適用できない"
  fi
  # 追跡ファイル全体が一時 index と完全一致すること。
  if ! GIT_INDEX_FILE="$tmp_idx" git -C "$root" diff --quiet; then
    rm -f "$tmp_idx"; precheck_fail "${root} の作業ツリーが HEAD + 所定パッチと完全一致しない"
  fi
  rm -f "$tmp_idx"
  grep -qE '^[[:space:]]*\("train", "chain", false\) =>' "$root/scripts/bench/framework-compare/bench-fandhe/src/main.rs" \
    || precheck_fail "${root} の main.rs に (train, chain) の分岐が無い"
}
check_patched_tree "$BEFORE_ROOT"
check_patched_tree "$AFTER_ROOT"
grep -rqE '^[[:space:]]*(pub )?fn begin_device_chain_scope' "$AFTER_ROOT/crates" \
  || precheck_fail "after に begin_device_chain_scope が無い"
if grep -rqE '^[[:space:]]*(pub )?fn begin_device_chain_scope' "$BEFORE_ROOT/crates"; then
  precheck_fail "before に begin_device_chain_scope がある（排他導入前でない）"
fi
echo "precheck=assert ok"

bench_fandhe_setup_lock_restore_trap

# before ツリーの Cargo.lock も path patch 付き cargo が書き換えうる。独立に退避・復元し、共有ヘルパーの trap（after 側）へ連鎖させる。
# revert 腕の worktree は終了時に必ず片付ける。
BEFORE_LOCK="$BEFORE_ROOT/scripts/bench/framework-compare/Cargo.lock"
BEFORE_LOCK_BACKUP="$(mktemp)" || { echo "error: 退避用の一時ファイルを作れない" >&2; exit 1; }
if ! cp "$BEFORE_LOCK" "$BEFORE_LOCK_BACKUP" || ! cmp -s "$BEFORE_LOCK" "$BEFORE_LOCK_BACKUP"; then
  rm -f "$BEFORE_LOCK_BACKUP"
  echo "error: before の Cargo.lock の退避に失敗した" >&2; exit 1
fi
WT_LIST=() # 空白を含むパスでも分割されないよう配列で保持する
cleanup_then_after_trap() {
  local code=$? wt
  if cp "$BEFORE_LOCK_BACKUP" "$BEFORE_LOCK"; then
    rm -f "$BEFORE_LOCK_BACKUP"
  else
    echo "error: before の Cargo.lock の復元に失敗した。バックアップを保持する: $BEFORE_LOCK_BACKUP" >&2
    [[ "$code" -eq 0 ]] && code=1
  fi
  # bash 3.2 + set -u で空配列展開が unbound になるため ${arr[@]+...} 形式で展開する
  for wt in ${WT_LIST[@]+"${WT_LIST[@]}"}; do git -C "$AFTER_ROOT" worktree remove --force "$wt" >/dev/null 2>&1 || true; done
  (exit "$code")
  bench_fandhe_restore_lock_trap
}
trap cleanup_then_after_trap EXIT

# <arm> <root>: root の framework-compare で path patch 付きビルドし、成果物を $OUT へ回収する。
build_in_tree() {
  local arm=$1 root=$2 tdir="target-diag-${LABEL}-$1" cfg tree fc out_abs
  fc="$root/scripts/bench/framework-compare"
  out_abs="$(pwd)/$OUT"
  cfg="patch.crates-io.fandhe-ai.path=\"${root}/crates/facade\""
  if ! ( cd "$fc" && cargo build --release -p bench-fandhe --target-dir "$tdir" --config "$cfg" ) 2>"$OUT/build-diag-${arm}-${LABEL}.err"; then
    tail -40 "$OUT/build-diag-${arm}-${LABEL}.err"
    echo "bench-fandhe BUILD FAILED (${arm})" >>"$SKIP"; exit 1
  fi
  tree="$(cd "$fc" && cargo tree -p bench-fandhe --depth 1 --config "$cfg" 2>&1)"
  if ! echo "$tree" | grep -qE 'fandhe-ai v[0-9.]+ \(.*crates/facade\)'; then
    echo "error: fandhe-ai が path patch に解決されていない (${arm})" >&2; echo "$tree" >&2; exit 1
  fi
  echo "$tree" >"$OUT/tree-diag-${arm}-${LABEL}.txt"
  cp "$fc/$tdir/release/bench-fandhe" "$out_abs/bench-fandhe-diag-${arm}-${LABEL}"
  echo "$(sha256_of "$out_abs/bench-fandhe-diag-${arm}-${LABEL}")  bench-fandhe-diag-${arm}-${LABEL}" >"$OUT/sha-diag-${arm}-${LABEL}.txt"
}

rv_tree_path() { echo "$(dirname "$AFTER_ROOT")/diag-wt-${LABEL}-$1"; }
# revert 腕のツリー: after の HEAD から detached worktree を作り、bench パッチ + 当該 revert パッチを当てて完全一致を検証する。
make_rv_tree() { # <arm>
  local arm=$1 wt
  wt="$(rv_tree_path "$arm")"
  [[ ! -e "$wt" ]] || precheck_fail "worktree ${wt} が既に存在する"
  git -C "$AFTER_ROOT" worktree add --detach "$wt" HEAD >/dev/null 2>&1 || precheck_fail "worktree ${wt} を作れない"
  WT_LIST+=("$wt")
  git -C "$wt" apply "$PATCH_FILE" || precheck_fail "${arm}: bench patch を適用できない"
  git -C "$wt" apply "$LOGDIR/diag-r4/${arm}.patch" || precheck_fail "${arm}: revert patch を適用できない"
  check_patched_tree "$wt" "$LOGDIR/diag-r4/${arm}.patch"
}

# 全ビルドを計測の前に済ませる（計測中のビルド並走禁止）。
build_in_tree excl0 "$BEFORE_ROOT"
build_in_tree excl0-rebuild "$BEFORE_ROOT"
build_in_tree excl1 "$AFTER_ROOT"
for rv in "${RV_ARMS[@]}"; do
  make_rv_tree "$rv"
  build_in_tree "$rv" "$(rv_tree_path "$rv")"
done

# aa0 / aa1 は excl0 の同一ファイルを参照する（sha256 一致を env_info に残す）。
bin_for() { # <arm>
  case "$1" in
    aa0|aa1) echo "$OUT/bench-fandhe-diag-excl0-${LABEL}" ;;
    excl0-rebuild|excl1|rv-scopecall|rv-kindlookup|rv-graphlock) echo "$OUT/bench-fandhe-diag-$1-${LABEL}" ;;
    *) echo "error: 未知の腕 $1" >&2; exit 1 ;;
  esac
}

{
  echo "profile=gb10"
  echo "host=masked"
  echo "uname=$(uname -srm)"
  echo "gpu=${GPU_NAME}"
  echo "nvidia_driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1)"
  echo "rustc=$(rustc --version 2>/dev/null)"
  echo "diag_sha256=${DIAG_SHA}"
  echo "patch_sha256=${PATCH_SHA}"
  printf '%s' "$RV_SHA_LINES"
  echo "rev_stamp_head_before=${STAMP_BEFORE}"
  echo "rev_stamp_head_after=${STAMP_AFTER}"
  echo "crates_identical_to=${CRATES_REF}"
  for a in "${ARMS[@]}"; do echo "bin_sha256_${a}=$(sha256_of "$(bin_for "$a")")"; done
  echo "precheck=assert"
  echo "rounds=${ROUNDS}"
  echo "tf32=off"
} >"$ENV_INFO"

# ---- 計測 ----
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
    return 1
  fi
}

run_arm() { # <arm> <round>
  local arm=$1 r=$2 errf res bin
  res="$OUT/results-diag-${arm}-${LABEL}-cuda-train.jsonl"
  errf="$OUT/digest-diag-${arm}-${LABEL}-cuda-train-r${r}.log"
  bin="$(bin_for "$arm")"
  echo "== diag train cuda chain arm=${arm} round=${r} =="
  env -u FANDHE_AI_CUDA_GRAPH_TRAIN -u FANDHE_AI_CUDA_GRAPH_STEP -u FANDHE_AI_CUDA_GRAPH_INFER \
    BENCH_PARAM_DIGEST=1 "$bin" --task train --device cuda --mode chain --out "$res" 2>"$errf"
  # shellcheck disable=SC2181
  if [[ $? -ne 0 ]]; then
    echo "task=train dev=cuda arm=${arm} round=${r}: $(tail -3 "$errf" | tr '\n' ' ')" >>"$SKIP"
    ANY_FAILED=$((ANY_FAILED + 1))
  fi
}

# 起動順の回転（DIAG-r4.txt 固定。L の位置 (r-1) mod 7 から巡回）。
order_for_round() {
  local r=$1 start i out=""
  start=$(( (r - 1) % NARMS ))
  i=0
  while [[ "$i" -lt "$NARMS" ]]; do
    out="${out}${ARMS[$(( (start + i) % NARMS ))]} "
    i=$((i + 1))
  done
  echo "$out"
}

for arm in "${ARMS[@]}"; do : >"$OUT/results-diag-${arm}-${LABEL}-cuda-train.jsonl"; done
GATE_LOG="$OUT/load_gate-diag-${LABEL}-cuda.log"
echo "series=official threshold=${LOAD_GATE} rule_threshold=${LOAD_GATE} precheck=assert" >"$GATE_LOG"
{ uptime; nvidia-smi; } >"$OUT/nvidia-smi-before-diag-${LABEL}.txt" 2>&1 || true
for r in 1 2 3 4 5; do
  # 専有ゲート不通過は診断不能（DIAG-r4.txt の専有ゲート契約）。負荷下で計測を続けず、
  # 記録して計測を停止し、ANY_FAILED へ加算して非 0 終了させる。
  if ! wait_gate "$r" "$GATE_LOG"; then
    echo "round${r}: exclusivity gate failed; diagnosis impossible, stopping measurement" >>"$SKIP"
    echo "round${r}: exclusivity gate failed; stopping" >&2
    ANY_FAILED=$((ANY_FAILED + 1))
    break
  fi
  for arm in $(order_for_round "$r"); do run_arm "$arm" "$r"; done
  echo "run ${r}: $(uptime)" | tee -a "$OUT/uptime-diag-${LABEL}.log"
done
{ uptime; nvidia-smi; } >"$OUT/nvidia-smi-after-diag-${LABEL}.txt" 2>&1 || true
echo "done. results in ${OUT}; skipped log: ${SKIP}"
if [[ "$ANY_FAILED" -gt 0 ]]; then echo "FAILED: ${ANY_FAILED} run(s)" >&2; exit 1; fi
