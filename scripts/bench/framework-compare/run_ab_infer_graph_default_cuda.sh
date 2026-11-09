#!/usr/bin/env bash
# イシュー #4163: CUDA Graph（推論チェーン capture。#2115）を既定にした場合の影響を GB10 で A/B する。
# GB10（aarch64・CUDA）専用ランナー。規則の正本: docs/perf/logs/cuda-infer-graph-default-ab-4163/RULE.txt
# （計測前に単独コミットで固定。本スクリプトは sha256 を env_info へ記録し、aggregate_4163.py が現 RULE と突合する）。
#
# 1 ツリー・1 ビルド・1 バイナリ: bench-fandhe は runner ツリーの `crates/facade` への
# `[patch.crates-io.fandhe-ai]` path patch（`--config` 引数のみ。コミットしない）で 1 回だけビルドし、
# off／on の差は環境変数 FANDHE_AI_CUDA_GRAPH_INFER だけにする（on = `=1`、off = 未設定）。
# 既定 ON の const 反転ビルドは正しさの R3 にだけ使う（ビルドごとの配置差を性能 A/B に混ぜない）。
# セル（11）: infer {fresh, reuse（常駐入力。本番・スコアボード経路）, reuse-host（--diag-host-input-reuse。
# capture が届く診断腕）} × batch {64, 1024, 4096} と train {fresh, reuse} × size 64。
# 計測の前に前提ゲート R0〜R3（run_gates_4163.sh）を同じ HEAD・同じ機体で実行する。
#
# 呼び出し例（GB10 実機。実機ロック保持下。HW_LOCK_HELD=1 と owner ファイルが要る）:
#   HW_LOCK_HELD=1 HW_LOCK_OWNER_FILE=<abs> AB_FACADE_PATH=<abs>/crates/facade \
#     bash run_ab_infer_graph_default_cuda.sh 4163-gb10
# bash 3.2 互換（mapfile・連想配列・eval 不使用）。jq 不使用。失敗は捏造せず skipped へ記録し非 0 終了。
set -u
# cd 後は $0 が相対パスのままだと自身を指さなくなるため、cd 前に自身の絶対パスを固定する（runner_sha256 の照合・記録で共用）。
SELF_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
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
for v in FANDHE_AI_CUDA_GRAPH_INFER FANDHE_AI_CUDA_GRAPH_STEP FANDHE_AI_CUDA_GRAPH_TRAIN; do
  if [[ -n "${!v+x}" ]]; then
    echo "error: 継承環境に ${v} が設定されている。unset してから実行すること" >&2; exit 1
  fi
done

# 実機ロック（#4150 の run_round.sh と同じ方式）。
if [[ "${HW_LOCK_HELD:-}" != "1" || -z "${HW_LOCK_OWNER_FILE:-}" || ! -e "${HW_LOCK_OWNER_FILE:-/nonexistent}" ]]; then
  echo "error: 実機ロック未保持（HW_LOCK_HELD=1 と実在する HW_LOCK_OWNER_FILE が要る）" >&2; exit 1
fi

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
FACADE=$AB_FACADE_PATH
if [[ ! -f "$FACADE/Cargo.toml" ]] || ! grep -qE '^[[:space:]]*name[[:space:]]*=[[:space:]]*"fandhe-ai"[[:space:]]*$' "$FACADE/Cargo.toml"; then
  echo "error: AB_FACADE_PATH/Cargo.toml が name = \"fandhe-ai\" を宣言していない" >&2; exit 1
fi
RUNNER_ROOT="$(cd ../../.. && pwd)"
if [[ "$(cd "$FACADE/../.." && pwd)" != "$RUNNER_ROOT" ]]; then
  echo "error: AB_FACADE_PATH が runner ツリーの crates/facade でない（1 ツリー方式）" >&2; exit 1
fi
LOGDIR="$RUNNER_ROOT/docs/perf/logs/cuda-infer-graph-default-ab-4163"
RULE_FILE="$LOGDIR/RULE.txt"
GATES_SH="$LOGDIR/run_gates_4163.sh"

# 機体同定（fail-closed）。GB10 の aarch64 のみ。
if [[ "$(uname -m)" != "aarch64" ]]; then echo "error: aarch64（GB10）のみ" >&2; exit 1; fi
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)"
if [[ "$GPU_NAME" != *GB10* ]]; then
  echo "error: nvidia-smi の GPU 名に GB10 を含まない（got: ${GPU_NAME:-unknown}）" >&2; exit 1
fi

sha256_of() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'; else shasum -a 256 "$1" | awk '{print $1}'; fi; }
[[ -f "$RULE_FILE" && -f "$GATES_SH" ]] || { echo "error: RULE.txt / run_gates_4163.sh が無い" >&2; exit 1; }
RULE_SHA="$(sha256_of "$RULE_FILE")"
# RULE.txt に固定した sha256 と実ファイルの一致（行頭アンカー。コメント行では素通りさせない）。
rule_pinned() { # <key> <file>
  local want
  want="$(sed -n "s/^  $1: \\([0-9a-f]\\{64\\}\\)\$/\\1/p" "$RULE_FILE" | head -n1)"
  [[ -n "$want" && "$want" == "$(sha256_of "$2")" ]]
}
for pair in "runner_sha256:$SELF_PATH" "gates_sha256:$GATES_SH" "patch_sha256:$LOGDIR/default-on.patch" \
            "assumption_list_sha256:$LOGDIR/default-off-assumption-tests.txt" \
            "aggregate_sha256:$LOGDIR/aggregate_4163.py" "bench_main_sha256:bench-fandhe/src/main.rs"; do
  rule_pinned "${pair%%:*}" "${pair#*:}" || { echo "error: RULE.txt の ${pair%%:*} と実ファイルの sha256 が一致しない" >&2; exit 1; }
done

OUT="results/raw"
mkdir -p "$OUT"
SKIP="$OUT/skipped-infer-graph-default-${LABEL}.log"
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
# 計測対象ソースの同一性。RULE.txt は計測前に単独コミットされ HEAD が source_commit から進むため、HEAD 完全一致は要求しない。
# 代わりに「source_commit（フル SHA）が HEAD の祖先で、RULE.txt 以外の差分が 0」を検証する
# （ソース・ランナー・ゲート・パッチ・集計器のいずれが変わっても停止する）。
RULE_SRC="$(sed -n 's/^  source_commit: \([0-9a-f]\{40\}\)$/\1/p' "$RULE_FILE" | head -n1)"
if [[ -z "$RULE_SRC" ]] || ! git -C "$RUNNER_ROOT" cat-file -e "${RULE_SRC}^{commit}" 2>/dev/null; then
  echo "error: RULE.txt の source_commit が無い・リポジトリに存在しない" >&2; exit 1
fi
if ! git -C "$RUNNER_ROOT" merge-base --is-ancestor "$RULE_SRC" "$REAL_HEAD"; then
  echo "error: source_commit が runner の HEAD の祖先でない" >&2; exit 1
fi
if ! git -C "$RUNNER_ROOT" diff --quiet "$RULE_SRC" "$REAL_HEAD" -- . ':(exclude)docs/perf/logs/cuda-infer-graph-default-ab-4163/RULE.txt'; then
  echo "error: source_commit から HEAD までに RULE.txt 以外の差分がある（計測対象ソースが変わっている）" >&2; exit 1
fi
# 作業ツリーは無変更（パッチは当てない。診断腕はコミット済みの main.rs に入っている）。
if [[ -n "$(git -C "$RUNNER_ROOT" status --porcelain | grep -v '^??' || true)" ]]; then
  echo "error: runner ツリーに未コミットの変更がある" >&2; exit 1
fi

bench_fandhe_setup_lock_restore_trap

PATCH_CFG="patch.crates-io.fandhe-ai.path=\"${FACADE}\""
TDIR_F="target-ab-infer-graph-default-${LABEL}-fandhe"
TDIR_G="${RUNNER_ROOT}/target-ab-infer-graph-default-${LABEL}-gates"
# 全ビルドを計測の前に済ませる（計測中のビルド並走禁止）。既定 feature・--release（internal-diagnostics は使わない）。
if ! cargo build --release -p bench-fandhe --target-dir "$TDIR_F" --config "$PATCH_CFG" 2>"$OUT/build-fandhe-${LABEL}.err"; then
  tail -40 "$OUT/build-fandhe-${LABEL}.err"; echo "bench-fandhe BUILD FAILED" >>"$SKIP"; exit 1
fi
tree="$(cargo tree -p bench-fandhe --depth 1 --config "$PATCH_CFG" 2>&1)"
if ! echo "$tree" | grep -qE 'fandhe-ai v[0-9.]+ \(.*crates/facade\)'; then
  echo "error: fandhe-ai が path patch に解決されていない" >&2; echo "$tree" >&2; exit 1
fi
echo "$tree" >"$OUT/tree-${LABEL}.txt"
BIN_F="$OUT/bench-fandhe-${LABEL}"
cp "$TDIR_F/release/bench-fandhe" "$BIN_F"
BIN_SHA="$(sha256_of "$BIN_F")"
echo "${BIN_SHA}  bench-fandhe-${LABEL}" >"$OUT/sha-fandhe-${LABEL}.txt"

{
  echo "profile=gb10"
  echo "host=masked"
  echo "uname=$(uname -srm)"
  echo "gpu=${GPU_NAME}"
  echo "nvidia_driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1)"
  echo "rustc=$(rustc --version 2>/dev/null)"
  echo "rule_sha256=${RULE_SHA}"
  echo "runner_sha256=$(sha256_of "$SELF_PATH")"
  echo "gates_sha256=$(sha256_of "$GATES_SH")"
  echo "patch_sha256=$(sha256_of "$LOGDIR/default-on.patch")"
  echo "assumption_list_sha256=$(sha256_of "$LOGDIR/default-off-assumption-tests.txt")"
  echo "aggregate_sha256=$(sha256_of "$LOGDIR/aggregate_4163.py")"
  echo "bench_main_sha256=$(sha256_of bench-fandhe/src/main.rs)"
  echo "rev_stamp_head_runner=${STAMP_RUNNER}"
  echo "source_commit=${RULE_SRC}"
  echo "source_diff_clean=1"
  echo "bin_sha256=${BIN_SHA}"
  echo "arms=same_binary_env_toggle"
  echo "rounds=${ROUNDS}"
  echo "tf32=off"
} >"$ENV_INFO"

# ---- 前提ゲート R0〜R3（ログに rc= を残す。判定は aggregate_4163.py。計測の前に全部済ませる） ----
if ! bash "$GATES_SH" "$LABEL" "$OUT" "$RUNNER_ROOT" "$TDIR_G"; then
  cp "$OUT/skipped-gates-${LABEL}.log" "$OUT/pregate-stop-${LABEL}.txt" 2>/dev/null || true
  echo "undetermined: pregate（前提ゲート不合格のため計測しない。詳細は ${OUT}/skipped-gates-${LABEL}.log）" | tee -a "$SKIP" >&2
  exit 1
fi

# ---- 計測 ----
# 専有ゲート（round ごと）: compute プロセス 0 かつ GPU 使用率 0% かつ load1 < 1.0 が 30 秒間隔で 3 サンプル連続。最大 30 分。
GATE_START=$SECONDS
gate_ok_once() {
  local l1 util apps n
  n=0
  while [[ "$n" -lt 3 ]]; do
    [[ $((SECONDS - GATE_START)) -le 1800 ]] || return 1
    l1="$(uptime | sed -n 's/.*load average[s]*: *\([0-9.]*\).*/\1/p')"
    [[ "$l1" =~ ^[0-9]+(\.[0-9]+)?$ ]] || return 1
    [[ "$(awk -v a="$l1" -v t="$LOAD_GATE" 'BEGIN{print (a<t)?1:0}')" == "1" ]] || return 1
    util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ')"
    [[ "$util" == "0" ]] || return 1
    apps="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)"
    [[ "$apps" == "0" ]] || return 1
    n=$((n + 1))
    if [[ "$n" -lt 3 ]]; then sleep 30; fi
  done
  return 0
}
wait_gate() { # <round> <logfile>。不通過なら undetermined を記録して非 0 終了（record_only の逃げ道なし）
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
    echo "undetermined: 専有ゲート不通過（round $1）" | tee -a "$SKIP" >&2
    exit 1
  fi
}

# run_cell <cell> <arm> <round>。cell 名は固定の allowlist（infer-{fresh,reuse,host}-{64,1024,4096}・train-{fresh,reuse}）。
run_cell() {
  local cell=$1 arm=$2 r=$3 res errf rc
  res="$OUT/results-${arm}-${LABEL}-${cell}.jsonl"
  errf="$OUT/stderr-${arm}-${LABEL}-${cell}-r${r}.log"
  local cmd=("$BIN_F" --device cuda --out "$res")
  case "$cell" in
    infer-fresh-*) cmd+=(--task infer --mode fresh --infer-batch "${cell##*-}") ;;
    infer-reuse-*) cmd+=(--task infer --mode reuse --infer-batch "${cell##*-}") ;;
    infer-host-*) cmd+=(--task infer --mode reuse --infer-batch "${cell##*-}" --diag-host-input-reuse) ;;
    train-fresh) cmd+=(--task train --mode fresh --size 64) ;;
    train-reuse) cmd+=(--task train --mode reuse --size 64) ;;
    *) echo "internal error: unknown cell" >&2; exit 1 ;;
  esac
  echo "round=${r} cell=${cell} arm=${arm}" >>"$OUT/order-${LABEL}.txt"
  echo "== ${cell} arm=${arm} round=${r} =="
  if [[ "$arm" == "on" ]]; then
    env -u FANDHE_AI_CUDA_GRAPH_STEP -u FANDHE_AI_CUDA_GRAPH_TRAIN FANDHE_AI_CUDA_GRAPH_INFER=1 "${cmd[@]}" 2>"$errf"
  else
    env -u FANDHE_AI_CUDA_GRAPH_STEP -u FANDHE_AI_CUDA_GRAPH_TRAIN -u FANDHE_AI_CUDA_GRAPH_INFER "${cmd[@]}" 2>"$errf"
  fi
  rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "cell=${cell} arm=${arm} round=${r} rc=${rc}: $(tail -3 "$errf" | tr '\n' ' ')" >>"$SKIP"
    ANY_FAILED=$((ANY_FAILED + 1))
  fi
}

# 起動順（RULE.txt 固定）: 基準のセル列を round ごとに 3*(r-1) mod 11 だけ巡回し、
# 各セルは off／on を連続起動する。先後は奇数 round が off→on、偶数 round が on→off。
CELLS=(infer-fresh-64 infer-fresh-1024 infer-fresh-4096 infer-reuse-64 infer-reuse-1024 infer-reuse-4096
       infer-host-64 infer-host-1024 infer-host-4096 train-fresh train-reuse)
NCELLS=${#CELLS[@]}
for cell in "${CELLS[@]}"; do for arm in off on; do : >"$OUT/results-${arm}-${LABEL}-${cell}.jsonl"; done; done
: >"$OUT/order-${LABEL}.txt"
GATE_LOG="$OUT/load_gate-${LABEL}-cuda.log"
echo "series=official threshold=${LOAD_GATE} rule_threshold=${LOAD_GATE}" >"$GATE_LOG"
{ uptime; nvidia-smi; } >"$OUT/nvidia-smi-before-${LABEL}.txt" 2>&1 || true
for r in 1 2 3 4 5; do
  wait_gate "$r" "$GATE_LOG"
  start=$(( (3 * (r - 1)) % NCELLS ))
  if (( r % 2 == 1 )); then first=off; second=on; else first=on; second=off; fi
  i=0
  while [[ "$i" -lt "$NCELLS" ]]; do
    cell="${CELLS[$(( (start + i) % NCELLS ))]}"
    run_cell "$cell" "$first" "$r"
    run_cell "$cell" "$second" "$r"
    i=$((i + 1))
  done
  echo "run ${r}: $(uptime)" | tee -a "$OUT/uptime-${LABEL}.log"
done
{ uptime; nvidia-smi; } >"$OUT/nvidia-smi-after-${LABEL}.txt" 2>&1 || true
echo "done. results in ${OUT}; skipped log: ${SKIP}"
if [[ "$ANY_FAILED" -gt 0 ]]; then echo "FAILED: ${ANY_FAILED} run(s)" >&2; exit 1; fi
