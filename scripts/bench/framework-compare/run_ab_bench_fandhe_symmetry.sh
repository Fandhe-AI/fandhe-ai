#!/usr/bin/env bash
# イシュー #3266: bench-fandhe 是正（#3265）の前後を、相手フレームワーク同席で併記計測する両機体共用ランナー。
# 規則の正本: docs/perf/logs/bench-fandhe-symmetry-ab-3266/RULE.txt（計測前に固定。sha256 を env_info へ記録し
# judge_bench_fandhe_symmetry_ab.py が現 RULE と突合する）。判定は同 judge が行い、本スクリプトは計測と記録だけを担う。
# 派生元: docs/perf/logs/framework-compare-0.11.0-macmini-remeasure/scripts/macmini-run-0110.sh（セル構成・期待行数・系列 B ゲート）、
#   同 framework-compare-0.11.0-remeasure/scripts/dgx-run-0110.sh（GB10 のゲート・Python 参照）、run_ab_dinput_skip_train.sh（腕・precheck の作法）。
# 腕: old=#3265 直前（RULE の old SHA）／new=#3265（同 new SHA）の bench-fandhe。どちらも registry の fandhe-ai =0.11.0
#   （path patch なし・--locked）。相手（candle・burn・Python）は new ツリーのバイナリを共用する。
# 呼び出し例:
#   AB_PROFILE=macmini AB_GATE=1 AB_OLD_ROOT=<abs>/old AB_NEW_ROOT=<abs>/new AB_RUNNER_ROOT=<abs>/runner \
#   AB_PY_TORCH=<abs>/venv-torch/bin/python AB_PY_TF=<abs>/venv-tf/bin/python bash run_ab_bench_fandhe_symmetry.sh series-b
#   AB_PROFILE=gb10 AB_GATE=1 ... AB_PY=<abs>/python bash run_ab_bench_fandhe_symmetry.sh gb10
# bash 3.2 互換（mapfile・連想配列・eval・jq 不使用）。失敗は skipped ログへ記録し非 0 終了。
set -uo pipefail

LABEL=${1:-}
PROFILE=${AB_PROFILE:-}
ROUNDS=${AB_ROUNDS:-5}
GATE=${AB_GATE:-1}

if [[ -z "$LABEL" || ! "$LABEL" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "usage: AB_PROFILE=macmini|gb10 AB_GATE=0|1 $0 <label>  (label: [A-Za-z0-9._-]+)" >&2; exit 1
fi
if [[ "$ROUNDS" != "5" ]]; then echo "error: AB_ROUNDS は 5 固定（RULE.txt。got: ${ROUNDS}）" >&2; exit 1; fi
if [[ "$GATE" != "0" && "$GATE" != "1" ]]; then echo "error: AB_GATE は 0 か 1" >&2; exit 1; fi
case "$PROFILE" in
  macmini) RULE_LOAD_GATE=3.0 ;;
  gb10) RULE_LOAD_GATE=1.0
    if [[ "$GATE" != "1" ]]; then echo "error: gb10 は AB_GATE=1 必須" >&2; exit 1; fi ;;
  *) echo "error: AB_PROFILE は macmini か gb10（got: ${PROFILE:-unset}）" >&2; exit 1 ;;
esac

# 外部入力のパスは絶対パスに限り、引用符・バックスラッシュ・改行を拒否する（A03）。
validate_abs() { # <var_name> <path>
  if [[ -z "$2" || "$2" != /* || "$2" == *\"* || "$2" == *\\* || "$2" == *$'\n'* ]]; then
    echo "error: $1 は引用符・バックスラッシュ・改行を含まない絶対パス必須" >&2; exit 1
  fi
}
validate_abs AB_OLD_ROOT "${AB_OLD_ROOT:-}"
validate_abs AB_NEW_ROOT "${AB_NEW_ROOT:-}"
validate_abs AB_RUNNER_ROOT "${AB_RUNNER_ROOT:-}"
OLD_ROOT=$AB_OLD_ROOT; NEW_ROOT=$AB_NEW_ROOT; RUNNER_ROOT=$AB_RUNNER_ROOT
if [[ "$PROFILE" == "macmini" ]]; then
  validate_abs AB_PY_TORCH "${AB_PY_TORCH:-}"; validate_abs AB_PY_TF "${AB_PY_TF:-}"
  PY_PT=$AB_PY_TORCH; PY_TF=$AB_PY_TF
  brand="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || true)"
  if [[ "$brand" != "Apple M4" ]]; then echo "error: macmini プロファイルは 'Apple M4' のみ（got: ${brand:-unknown}）" >&2; exit 1; fi
else
  validate_abs AB_PY "${AB_PY:-}"; PY_PT=$AB_PY; PY_TF=$AB_PY
  if [[ "$(uname -m)" != "aarch64" ]]; then echo "error: gb10 は aarch64 のみ" >&2; exit 1; fi
fi

sha256_of() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'; else shasum -a 256 "$1" | awk '{print $1}'; fi; }
FC_REL=scripts/bench/framework-compare
RULE_FILE="$RUNNER_ROOT/docs/perf/logs/bench-fandhe-symmetry-ab-3266/RULE.txt"
BENCH_PY_REL=docs/perf/logs/lowlayer-diagnosis-2026-09-12/bench_py.py
BENCH_PY_SHA=9fd2c08f37d66b5bb41ececab8a5b3daaa36b55609770502ed3b35ae2ce2409b
[[ -f "$RULE_FILE" ]] || { echo "error: RULE.txt が無い" >&2; exit 1; }
RULE_SHA="$(sha256_of "$RULE_FILE")"

OUT="$RUNNER_ROOT/$FC_REL/results/raw/ab3266-${LABEL}"
if [[ -e "$OUT" ]]; then echo "error: ラベル ${LABEL} の出力が既に存在する（差し替え禁止）: $OUT" >&2; exit 1; fi
mkdir -p "$OUT"
SKIP="$OUT/skipped.log"; : >"$SKIP"
unset RAYON_NUM_THREADS
precheck_fail() { echo "precheck FAILED: $1" | tee -a "$SKIP" >&2; exit 1; }

# RULE.txt の `- old = <40hex>`／`- new = <40hex>` と `.rev-stamp` を照合（腕の取り違え防止）。
rule_sha_of() { grep -E "^- $1[[:space:]]*=[[:space:]]*[0-9a-f]{40}" "$RULE_FILE" | head -n1 | grep -oE '[0-9a-f]{40}' | head -n1; }
EXP_OLD="$(rule_sha_of old)"; EXP_NEW="$(rule_sha_of new)"
[[ -n "$EXP_OLD" && -n "$EXP_NEW" ]] || precheck_fail "RULE.txt から old/new SHA を読めない"
stamp_of() { if [[ -f "$1/.rev-stamp" ]]; then head -n1 "$1/.rev-stamp"; fi; }
[[ "$(stamp_of "$OLD_ROOT")" == "$EXP_OLD" ]] || precheck_fail "old の .rev-stamp が RULE の old SHA と不一致"
[[ "$(stamp_of "$NEW_ROOT")" == "$EXP_NEW" ]] || precheck_fail "new の .rev-stamp が RULE の new SHA と不一致"
STAMP_RUNNER="$(stamp_of "$RUNNER_ROOT")"
[[ -n "$STAMP_RUNNER" ]] || precheck_fail "runner の .rev-stamp が無い"

# old/new で bench-fandhe 本体以外が完全一致すること（相手バイナリを共用できる前提）。
for rel in Cargo.lock Cargo.toml bench-fandhe/Cargo.toml bench-candle bench-burn bench-common; do
  diff -r "$OLD_ROOT/$FC_REL/$rel" "$NEW_ROOT/$FC_REL/$rel" >/dev/null 2>&1 || precheck_fail "old と new で $rel が一致しない"
done
# old は是正前の関数を持ち new は持たない（行頭アンカー付き grep）。
OLD_MAIN="$OLD_ROOT/$FC_REL/bench-fandhe/src/main.rs"; NEW_MAIN="$NEW_ROOT/$FC_REL/bench-fandhe/src/main.rs"
grep -qE '^fn readout_uses_borrowed_view\(device: &str' "$OLD_MAIN" || precheck_fail "old に readout_uses_borrowed_view が無い（腕の取り違え）"
if grep -qE '^fn readout_uses_borrowed_view\(device: &str' "$NEW_MAIN"; then precheck_fail "new に readout_uses_borrowed_view がある（腕の取り違え）"; fi
[[ "$(sha256_of "$RUNNER_ROOT/$BENCH_PY_REL")" == "$BENCH_PY_SHA" ]] || precheck_fail "bench_py.py の sha256 が登録値と不一致"
echo "precheck=ok"

# ビルド（計測の前にすべて済ませる。計測中のビルド並走禁止）。
build() { # <root> <target-dir-name> <crate> [cargo args...]
  local root=$1 tdir="$OUT/target-$2" crate=$3; shift 3
  ( cd "$root/$FC_REL" && cargo build --release --locked -p "$crate" --target-dir "$tdir" "$@" ) 2>"$OUT/build-$2-$crate.err" \
    || { tail -40 "$OUT/build-$2-$crate.err"; echo "BUILD FAILED $2 $crate" >>"$SKIP"; exit 1; }
}
if [[ "$PROFILE" == "gb10" ]]; then export PATH="${HOME}/.cargo/bin:/usr/local/cuda/bin:${PATH}"; fi
build "$OLD_ROOT" old bench-fandhe
build "$NEW_ROOT" new bench-fandhe
if [[ "$PROFILE" == "gb10" ]]; then
  build "$NEW_ROOT" new-cuda bench-candle --no-default-features --features cuda
  build "$NEW_ROOT" new-cuda bench-burn --no-default-features --features cuda
  OPP_DIR="$OUT/target-new-cuda/release"
else
  build "$NEW_ROOT" new bench-candle
  build "$NEW_ROOT" new bench-burn
  OPP_DIR="$OUT/target-new/release"
fi
BIN_OLD="$OUT/target-old/release/bench-fandhe"; BIN_NEW="$OUT/target-new/release/bench-fandhe"
for arm in old new; do
  if [[ "$arm" == "old" ]]; then root=$OLD_ROOT; else root=$NEW_ROOT; fi
  tree="$(cd "$root/$FC_REL" && cargo tree --locked -p bench-fandhe --depth 1 2>&1)"
  echo "$tree" | grep -qE 'fandhe-ai v0\.11\.0$' || precheck_fail "$arm の fandhe-ai が 0.11.0 に解決されない"
  echo "$tree" >"$OUT/tree-$arm.txt"
done

{
  echo "profile=${PROFILE}"; echo "label=${LABEL}"; echo "gate=${GATE}"; echo "host=masked"
  echo "uname=$(uname -srm)"
  if [[ "$PROFILE" == "macmini" ]]; then echo "cpu_brand=$(sysctl -n machdep.cpu.brand_string)"; else echo "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)"; fi
  echo "rule_sha256=${RULE_SHA}"; echo "bench_py_sha256=$(sha256_of "$RUNNER_ROOT/$BENCH_PY_REL")"
  echo "rev_stamp_old=$(stamp_of "$OLD_ROOT")"; echo "rev_stamp_new=$(stamp_of "$NEW_ROOT")"; echo "rev_stamp_runner=${STAMP_RUNNER}"
  echo "cargo_lock_sha256=$(sha256_of "$NEW_ROOT/$FC_REL/Cargo.lock")"
  echo "bin_sha256_old=$(sha256_of "$BIN_OLD")"; echo "bin_sha256_new=$(sha256_of "$BIN_NEW")"
  echo "bin_sha256_candle=$(sha256_of "$OPP_DIR/bench-candle")"; echo "bin_sha256_burn=$(sha256_of "$OPP_DIR/bench-burn")"
  echo "python_pt=$("$PY_PT" -c 'import sys,torch;print(sys.version.split()[0],"torch",torch.__version__)' 2>&1 | tail -1)"
  echo "python_tf=$("$PY_TF" -c 'import tensorflow as t;print("tf",t.__version__)' 2>&1 | tail -1)"
  echo "python_scipy=$("$PY_PT" -c 'import scipy;print("scipy",scipy.__version__)' 2>&1 | tail -1)"
  echo "rounds=${ROUNDS}"; echo "utc=$(date -u +%FT%TZ)"
} >"$OUT/env_info.txt"

load1() {
  local l
  l=$(sysctl -n vm.loadavg 2>/dev/null | awk '{print $2}')
  if [[ -z "$l" && -r /proc/loadavg ]]; then l=$(awk '{print $1}' /proc/loadavg); fi
  printf '%s' "$l"
}
gpu_idle3() {
  local n=0 u
  while [[ "$n" -lt 3 ]]; do
    u="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ')"
    [[ "$u" == "0" ]] || return 1
    n=$((n + 1)); if [[ "$n" -lt 3 ]]; then sleep 2; fi
  done
  return 0
}
# ゲート: load1 < 閾値を 30 秒間隔・最大 30 分待つ（GB10 は GPU 使用率 0% 3 サンプルも要求）。数値でなければ GATE-INVALID。
gate_wait() { # <round>
  local waited=0 l ok
  while :; do
    l="$(load1)"
    if [[ ! "$l" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then echo "round$1 GATE-INVALID load1-not-numeric" >>"$OUT/gate.log"; return 2; fi
    ok="$(awk -v a="$l" -v t="$RULE_LOAD_GATE" 'BEGIN{print (a<t)?1:0}')"
    if [[ "$ok" == "1" && "$PROFILE" == "gb10" ]] && ! gpu_idle3; then ok=0; fi
    if [[ "$ok" == "1" ]]; then
      echo "round$1 gate=pass threshold=${RULE_LOAD_GATE} waited_s=${waited} load1=${l}" >>"$OUT/gate.log"; return 0
    fi
    if [[ "$waited" -ge 1800 ]]; then echo "round$1 GATE-TIMEOUT load1=${l}" >>"$OUT/gate.log"; return 1; fi
    sleep 30; waited=$((waited + 30))
  done
}
: >"$OUT/gate.log"

if [[ "$PROFILE" == "macmini" ]]; then
  GEMM_DEV="cpu:256,512,1024,2048 metal:256,512,1024,2048,4096"; MDEVS="cpu metal"
  EXP_ARM=26; EXP_OPP=22; EXP_PY=32
else
  GEMM_DEV="cuda:256,512,1024,2048,4096 cpu:256,512,1024,2048,4096"; MDEVS="cuda cpu"
  EXP_ARM=28; EXP_OPP=28; EXP_PY=28
fi
ANY_FAILED=0
PYF=""
# 1 セルを実行。失敗は skipped へ記録（捏造しない）。
cell() { # <bin> <out> <task> <dev> <size> <mode>
  local errf="$OUT/err.tmp"
  if ! "$1" --task "$3" --device "$4" --size "$5" --mode "$6" --out "$2" 2>"$errf"; then
    echo "$(basename "$1") $3 $4 $5 $6: $(tr '\n' ' ' <"$errf")" >>"$SKIP"; ANY_FAILED=$((ANY_FAILED + 1))
  fi
}
each_cell() { # <fn>: 判定セルを (task dev size) で列挙して fn を呼ぶ
  local spec dev sizes n
  for spec in $GEMM_DEV; do
    dev=${spec%%:*}; sizes=${spec#*:}
    for n in ${sizes//,/ }; do "$1" gemm "$dev" "$n"; done
  done
  for dev in $MDEVS; do "$1" train "$dev" 64; "$1" infer "$dev" 64; done
}
pyrun() { # <python> args...
  local py=$1; shift
  if ! "$py" "$RUNNER_ROOT/$BENCH_PY_REL" "$@" --out "$PYF" >/dev/null 2>"$OUT/err.tmp"; then
    echo "py $*: $(tail -3 "$OUT/err.tmp" | tr '\n' ' ')" >>"$SKIP"; ANY_FAILED=$((ANY_FAILED + 1))
  fi
}
RD=""
opp_cell() { local b; for b in bench-candle bench-burn; do cell "$OPP_DIR/$b" "$RD/results-opp.jsonl" "$1" "$2" "$3" fresh; done; }
fandhe_cell() {
  local mode
  for mode in fresh reuse; do
    if (( R % 2 == 1 )); then
      cell "$BIN_OLD" "$RD/results-old.jsonl" "$1" "$2" "$3" "$mode"; cell "$BIN_NEW" "$RD/results-new.jsonl" "$1" "$2" "$3" "$mode"
    else
      cell "$BIN_NEW" "$RD/results-new.jsonl" "$1" "$2" "$3" "$mode"; cell "$BIN_OLD" "$RD/results-old.jsonl" "$1" "$2" "$3" "$mode"
    fi
  done
}
py_cell() { # task dev size
  local fw
  if [[ "$PROFILE" == "macmini" ]]; then
    pyrun "$PY_PT" --framework pytorch --task "$1" --device "$2" --size "$3"
    pyrun "$PY_TF" --framework tensorflow --task "$1" --device "$2" --size "$3"
    if [[ "$2" == "cpu" ]]; then pyrun "$PY_PT" --framework scipy --task "$1" --device cpu --size "$3"; fi
  elif [[ "$2" == "cpu" ]]; then
    for fw in pytorch tensorflow scipy; do pyrun "$PY_PT" --framework "$fw" --task "$1" --device cpu --size "$3"; done
  else
    pyrun "$PY_PT" --framework pytorch --task "$1" --device cuda --size "$3"
  fi
}

R=0
for R in 1 2 3 4 5; do
  RD="$OUT/run$R"; mkdir -p "$RD"
  : >"$RD/results-old.jsonl"; : >"$RD/results-new.jsonl"; : >"$RD/results-opp.jsonl"; : >"$RD/results-py.jsonl"
  PYF="$RD/results-py.jsonl"
  if [[ "$GATE" == "1" ]]; then
    grc=0; gate_wait "$R" || grc=$?
    if [[ "$grc" -ne 0 ]]; then echo "SERIES-ABORTED completed=$((R - 1))/5 (gate rc=${grc})"; exit 1; fi
  fi
  uptime >"$RD/uptime_before.txt"
  if [[ "$PROFILE" == "macmini" ]]; then pmset -g therm >"$RD/pmset_before.txt" 2>&1 || true
  else nvidia-smi --query-compute-apps=pid,name --format=csv >"$RD/gpu_apps_before.txt" 2>&1 || true; fi
  # 順序: Rust の相手 → fandhe 両腕（セルごとに old/new を続け、先行腕はラウンドの奇偶で反転）→ Python
  each_cell opp_cell
  each_cell fandhe_cell
  each_cell py_cell
  uptime >"$RD/uptime_after.txt"
  if [[ "$PROFILE" == "macmini" ]]; then pmset -g therm >"$RD/pmset_after.txt" 2>&1 || true
  else nvidia-smi --query-compute-apps=pid,name --format=csv >"$RD/gpu_apps_after.txt" 2>&1 || true; fi
  rows_old=$(wc -l <"$RD/results-old.jsonl" | tr -d ' '); rows_new=$(wc -l <"$RD/results-new.jsonl" | tr -d ' ')
  rows_opp=$(wc -l <"$RD/results-opp.jsonl" | tr -d ' '); rows_py=$(wc -l <"$RD/results-py.jsonl" | tr -d ' ')
  echo "run${R}: old=${rows_old} new=${rows_new} opp=${rows_opp} py=${rows_py}"
  # burn Metal gemm N=512〜4096 の既知の記録拒否 4 件（0.11.0 と同じ）以外の skipped は想定外として打ち切る。
  unexpected=$(grep -vcE '^bench-burn gemm metal (512|1024|2048|4096) fresh: MEASURE_ERROR: gemm checksum is degenerate' "$SKIP")
  if [[ "$rows_old" != "$EXP_ARM" || "$rows_new" != "$EXP_ARM" || "$rows_opp" != "$EXP_OPP" || "$rows_py" != "$EXP_PY" ]]; then
    echo "SERIES-ABORTED completed=$((R - 1))/5 (期待行数 ${EXP_ARM}/${EXP_ARM}/${EXP_OPP}/${EXP_PY} と不一致)"; exit 1
  fi
  if [[ "$unexpected" != "0" ]]; then echo "SERIES-ABORTED completed=$((R - 1))/5 (想定外の失敗)"; exit 1; fi
done
rm -f "$OUT/err.tmp"
echo "ALL-DONE completed=5/5"
