#!/usr/bin/env bash
# イシュー #4168: 結線後（Phase 4 後）の main HEAD と v0.14.0 の前後比較を、相手フレームワーク同席で併記計測する両機体共用ランナー。
# 役割: 計測（ビルド・ゲート・5 ラウンド・記録）だけを担う。判定は judge_prepost_0150_4168.py が行う。
# 規則の正本: docs/perf/logs/prepost-0150-4168/RULE.txt（RULE.template.txt を実体化して計測前に単独コミットする。
#   sha256 を env_info へ記録し、判定器が現 RULE と突合する。`<要記入` が残っていれば本スクリプトは拒否する）。
# 派生元: run_ab_bench_fandhe_symmetry.sh（#3266。凍結。セル構成・期待行数・ゲート・ラウンド手順）、
#   run_ab_sme_default_on_4156_cpu.sh（path patch 腕のビルド作法）、bench_fandhe_lock_restore.sh（Cargo.lock の退避・復元）、
#   docs/perf/logs/framework-compare-0.14.0-remeasure/scripts/dgx-run-0140.sh（GB10 の compute プロセス 0 の専有確認）。
# 腕（両腕とも同じ BASE ツリーの同じ bench-fandhe ソース。差はライブラリだけ）:
#   before = registry の fandhe-ai =0.14.0（path patch なし・--locked）
#   after  = `--config patch.crates-io.fandhe-ai.path="<BASE>/crates/facade"`（BASE の main HEAD ソース）
#   相手（candle・burn・Python）は BASE からビルドして両腕で共用する。
# 呼び出し例:
#   AB_PROFILE=macmini AB_BASE_ROOT=<abs>/base AB_RUNNER_ROOT=<abs>/base AB_HWLOCK_OWNER=<owner> \
#     AB_PY_TORCH=<abs>/venv-torch/bin/python AB_PY_TF=<abs>/venv-tf/bin/python bash run_ab_prepost_0150_4168.sh series-b
#   AB_PROFILE=gb10 ... AB_PY=<abs>/python bash run_ab_prepost_0150_4168.sh gb10
#   Linux の事前確認（ビルド・計測なし）: AB_PROFILE=macmini AB_BASE_ROOT=<repo> AB_RUNNER_ROOT=<repo> bash run_ab_prepost_0150_4168.sh --dry-run <label>
# bash 3.2 互換（mapfile・連想配列・eval・jq 不使用）。失敗は skipped ログへ記録し非 0 終了。
set -uo pipefail

DRY=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY=1; shift; fi
LABEL=${1:-}
PROFILE=${AB_PROFILE:-}
ROUNDS=${AB_ROUNDS:-5}

if [[ -z "$LABEL" || ! "$LABEL" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "usage: AB_PROFILE=macmini|gb10 $0 [--dry-run] <label>  (label: [A-Za-z0-9._-]+)" >&2; exit 1
fi
if [[ "$ROUNDS" != "5" ]]; then echo "error: AB_ROUNDS は 5 固定（RULE.txt。got: ${ROUNDS}）" >&2; exit 1; fi
case "$PROFILE" in
  macmini) RULE_LOAD_GATE=3.0 ;;
  gb10) RULE_LOAD_GATE=1.0 ;;
  *) echo "error: AB_PROFILE は macmini か gb10（got: ${PROFILE:-unset}）" >&2; exit 1 ;;
esac

# 外部入力のパスは絶対パスに限り、引用符・バックスラッシュ・改行を拒否する（A03。cargo の --config へ埋め込むため厳格に）。
validate_abs() { # <var_name> <path>
  if [[ -z "$2" || "$2" != /* || "$2" == *\"* || "$2" == *\\* || "$2" == *$'\n'* || "$2" == *\'* ]]; then
    echo "error: $1 は引用符・バックスラッシュ・改行を含まない絶対パス必須" >&2; exit 1
  fi
}
validate_abs AB_BASE_ROOT "${AB_BASE_ROOT:-}"
validate_abs AB_RUNNER_ROOT "${AB_RUNNER_ROOT:-}"
BASE_ROOT=$AB_BASE_ROOT; RUNNER_ROOT=$AB_RUNNER_ROOT
if [[ "$DRY" == "0" ]]; then
  if [[ "$PROFILE" == "macmini" ]]; then
    validate_abs AB_PY_TORCH "${AB_PY_TORCH:-}"; validate_abs AB_PY_TF "${AB_PY_TF:-}"
    PY_PT=$AB_PY_TORCH; PY_TF=$AB_PY_TF
    brand="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || true)"
    if [[ "$brand" != "Apple M4" ]]; then echo "error: macmini プロファイルは 'Apple M4' のみ（got: ${brand:-unknown}）" >&2; exit 1; fi
  else
    validate_abs AB_PY "${AB_PY:-}"; PY_PT=$AB_PY; PY_TF=$AB_PY
    if [[ "$(uname -m)" != "aarch64" ]]; then echo "error: gb10 は aarch64 のみ" >&2; exit 1; fi
  fi
  # 実機ロック（owner 文字列が一致すること。取得・解放は orchestrate.sh）。ロックなしの値は採らない。
  OWNER="${AB_HWLOCK_OWNER:-}"
  [[ -n "$OWNER" && "$OWNER" != *$'\n'* ]] || { echo "error: AB_HWLOCK_OWNER 必須" >&2; exit 1; }
  [[ "$(cat "${HOME}/work/.fandhe-hw-lock/owner" 2>/dev/null)" == "$OWNER" ]] || { echo "error: 実機ロックの owner が一致しない（LOCK-NOT-MINE）" >&2; exit 1; }
fi

sha256_of() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'; else shasum -a 256 "$1" | awk '{print $1}'; fi; }
FC_REL=scripts/bench/framework-compare
LOGS_REL=docs/perf/logs/prepost-0150-4168
RULE_FILE="$RUNNER_ROOT/$LOGS_REL/RULE.txt"
BENCH_PY_REL=docs/perf/logs/lowlayer-diagnosis-2026-09-12/bench_py.py
BENCH_PY_SHA=9fd2c08f37d66b5bb41ececab8a5b3daaa36b55609770502ed3b35ae2ce2409b
BASELINE_COMMIT=18a7f4b87fe61029a179cb1922cc82928440f2c9   # 0.14.0 スコアボードの計測腕（#4063 マージ）
FACADE_PATH="$BASE_ROOT/crates/facade"
validate_abs FACADE_PATH "$FACADE_PATH"

if [[ "$DRY" == "1" ]]; then
  OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
else
  OUT="$RUNNER_ROOT/$FC_REL/results/raw/ab4168-${LABEL}"
  if [[ -e "$OUT" ]]; then echo "error: ラベル ${LABEL} の出力が既に存在する（差し替え禁止）: $OUT" >&2; exit 1; fi
  mkdir -p "$OUT"
fi
SKIP="$OUT/skipped.log"; : >"$SKIP"
unset RAYON_NUM_THREADS
precheck_fail() { echo "precheck FAILED: $1" | tee -a "$SKIP" >&2; exit 1; }

# ---- 事前確認 1: RULE.txt（未記入の拒否・after SHA と .rev-stamp の照合）
[[ -f "$RULE_FILE" ]] || precheck_fail "RULE.txt が無い（RULE.template.txt の実体化と単独コミットが先）"
if grep -q '<要記入' "$RULE_FILE"; then precheck_fail "RULE.txt に <要記入 が残っている"; fi
RULE_SHA="$(sha256_of "$RULE_FILE")"
rule_sha_of() { grep -E "^- $1[[:space:]]*=[[:space:]]*[0-9a-f]{40}([[:space:]]|\$)" "$RULE_FILE" | head -n1 | grep -oE '[0-9a-f]{40}' | head -n1; }
EXP_AFTER="$(rule_sha_of after)"
[[ -n "$EXP_AFTER" ]] || precheck_fail "RULE.txt から after SHA（'- after = <40hex>'）を読めない"
stamp_of() { if [[ -f "$1/.rev-stamp" ]]; then head -n1 "$1/.rev-stamp"; fi; }
if [[ "$DRY" == "0" ]]; then
  [[ "$(stamp_of "$BASE_ROOT")" == "$EXP_AFTER" ]] || precheck_fail "BASE の .rev-stamp が RULE の after SHA と不一致"
  STAMP_RUNNER="$(stamp_of "$RUNNER_ROOT")"
  [[ -n "$STAMP_RUNNER" ]] || precheck_fail "runner の .rev-stamp が無い"
  [[ "$(sha256_of "$RUNNER_ROOT/$BENCH_PY_REL")" == "$BENCH_PY_SHA" ]] || precheck_fail "bench_py.py の sha256 が登録値と不一致"
else
  echo "dry-run: .rev-stamp・bench_py.py の照合は実機でのみ行う（未検証）"
fi

# ---- 事前確認 2: ゲート定数（v0.14.0 計測腕の既存定数は同値・以後に増えた定数は false）
consts_of() { # <git-rev|WORKTREE>  -> 'path:NAME=value' を整列して出力
  if [[ "$1" == "WORKTREE" ]]; then
    ( cd "$BASE_ROOT" && grep -rnE 'const [A-Z_0-9]*ENABLED[A-Z_0-9]*: *bool *=' crates --include='*.rs' )
  else
    git -C "$BASE_ROOT" grep -nE 'const [A-Z_0-9]*ENABLED[A-Z_0-9]*: *bool *=' "$1" -- crates | sed -e "s/^$1://"
  fi | sed -nE 's/^([^:]+):[0-9]+:.*const ([A-Z_0-9]*ENABLED[A-Z_0-9]*): *bool *= *(true|false);.*/\1:\2=\3/p' | LC_ALL=C sort
}
if git -C "$BASE_ROOT" cat-file -e "$BASELINE_COMMIT^{commit}" 2>/dev/null; then
  consts_of "$BASELINE_COMMIT" >"$OUT/consts-baseline.txt"
  consts_of WORKTREE >"$OUT/consts.txt"
  [[ -s "$OUT/consts.txt" && -s "$OUT/consts-baseline.txt" ]] || precheck_fail "ゲート定数を抽出できない"
  while IFS= read -r line; do
    name=${line%%=*}; val=${line#*=}
    base_val="$(grep -F "$name=" "$OUT/consts-baseline.txt" | head -n1 | sed 's/.*=//')"
    if [[ -n "$base_val" ]]; then
      [[ "$base_val" == "$val" ]] || precheck_fail "ゲート定数が v0.14.0 計測腕から変わっている（結線あり）: $name $base_val -> $val"
    else
      [[ "$val" == "false" ]] || precheck_fail "v0.14.0 以降に増えた定数が false でない: $name=$val"
    fi
  done <"$OUT/consts.txt"
  echo "gate-consts=ok ($(wc -l <"$OUT/consts.txt" | tr -d ' ') 件)"
  # bench-fandhe の差分は記録する（機械的に断定はしない。README「事前確認」で人が読む）。両腕は同じソースを使う。
  git -C "$BASE_ROOT" diff "$BASELINE_COMMIT" HEAD --stat -- "$FC_REL/bench-fandhe" >"$OUT/bench-fandhe-diff-stat.txt" 2>&1 || true
else
  [[ "$DRY" == "1" ]] || precheck_fail "BASE に baseline コミット ${BASELINE_COMMIT} が無い（git archive ではなく clone を使う）"
  echo "dry-run: baseline コミットが無いためゲート定数の照合を省略（未検証）"
fi

# ---- ビルド・腕の解決確認（Cargo.lock は path patch で書き換わるため退避し、終了時に復元して差分 0 を確かめる）
PATCH_CONFIG="patch.crates-io.fandhe-ai.path=\"${FACADE_PATH}\""
LOCK_CLEAN_REL="$FC_REL/Cargo.lock"
LOCK_COPY="$(mktemp)"
cp "$BASE_ROOT/$LOCK_CLEAN_REL" "$LOCK_COPY" || { echo "error: Cargo.lock の退避に失敗" >&2; exit 1; }
restore_lock() { cp "$LOCK_COPY" "$BASE_ROOT/$LOCK_CLEAN_REL" 2>/dev/null; rm -f "$LOCK_COPY"; }
trap 'restore_lock; if [[ "$DRY" == "1" ]]; then rm -rf "$OUT"; fi' EXIT
FC_DIR="$BASE_ROOT/$FC_REL"

tree_checks() { # before/after の fandhe-ai 解決を検査
  local t
  t="$(cd "$FC_DIR" && cargo tree --locked -p bench-fandhe --depth 2 --target all 2>&1)" || { echo "$t" | tail -5; return 2; }
  echo "$t" >"$OUT/tree-before.txt"
  echo "$t" | grep -qE 'fandhe-ai v0\.14\.0$' || precheck_fail "before の fandhe-ai が registry 0.14.0 に解決されない"
  t="$(cd "$FC_DIR" && cargo tree -p bench-fandhe --depth 2 --target all --config "$PATCH_CONFIG" 2>&1)" || { echo "$t" | tail -5; return 2; }
  echo "$t" >"$OUT/tree-after.txt"
  # after: fandhe-ai と fandhe-ai-* のすべてが path に解決される（registry 形 'fandhe-ai... vX.Y.Z' で行が終わる行が 0 件）
  if echo "$t" | grep -E 'fandhe-ai[a-z-]* v[0-9.]+$' >/dev/null; then precheck_fail "after に registry へ解決された fandhe-ai 系クレートが残る（比較が無意味になる）"; fi
  local need
  for need in fandhe-ai fandhe-ai-tensor-core fandhe-ai-autodiff fandhe-ai-backend-cpu; do
    echo "$t" | grep -qE "${need} v[0-9.]+ \(${BASE_ROOT}/crates/" || precheck_fail "after で ${need} が BASE の crates/ へ解決されない"
  done
  return 0
}
tc=0; tree_checks || tc=$?
if [[ "$tc" -ne 0 ]]; then
  if [[ "$DRY" == "1" ]]; then echo "dry-run: cargo tree を実行できない環境（rc=$tc）。腕の解決確認は未検証"; else precheck_fail "cargo tree の実行に失敗"; fi
else
  echo "tree-checks=ok"
fi

if [[ "$DRY" == "1" ]]; then
  echo "dry-run: 以降（ビルド・ゲート・計測）は実機でのみ実行する。計画:"
  echo "  build before: cargo build --release --locked -p bench-fandhe"
  echo "  build opp   : bench-candle, bench-burn（gb10 は --no-default-features --features cuda）"
  echo "  build after : cargo build --release -p bench-fandhe --config '${PATCH_CONFIG}'（Cargo.lock は trap で復元）"
  echo "  rounds=5 ゲート(${PROFILE}) load1<${RULE_LOAD_GATE}"
  echo "DRY-RUN-OK"
  exit 0
fi

build() { # <target-dir-name> <crate> [cargo args...]（BASE の framework-compare で --locked ビルド）
  local arm=$1 crate=$2 tdir="$OUT/target-$1"; shift 2
  ( cd "$FC_DIR" && cargo build --release --locked -p "$crate" --target-dir "$tdir" "$@" ) 2>"$OUT/build-${arm}-${crate}.err" \
    || { tail -40 "$OUT/build-${arm}-${crate}.err"; echo "BUILD FAILED ${arm} ${crate}" >>"$SKIP"; exit 1; }
}
if [[ "$PROFILE" == "gb10" ]]; then export PATH="${HOME}/.cargo/bin:/usr/local/cuda/bin:${PATH}"; fi
build before bench-fandhe
if [[ "$PROFILE" == "gb10" ]]; then
  build opp-cuda bench-candle --no-default-features --features cuda
  build opp-cuda bench-burn --no-default-features --features cuda
  OPP_DIR="$OUT/target-opp-cuda/release"
else
  build opp bench-candle
  build opp bench-burn
  OPP_DIR="$OUT/target-opp/release"
fi
# after は --locked を付けない（path patch の解決で Cargo.lock が更新されるため。trap が終了時に復元する）
( cd "$FC_DIR" && cargo build --release -p bench-fandhe --target-dir "$OUT/target-after" --config "$PATCH_CONFIG" ) 2>"$OUT/build-after-bench-fandhe.err" \
  || { tail -40 "$OUT/build-after-bench-fandhe.err"; echo "BUILD FAILED after bench-fandhe" >>"$SKIP"; exit 1; }
BIN_BEFORE="$OUT/target-before/release/bench-fandhe"; BIN_AFTER="$OUT/target-after/release/bench-fandhe"
[[ -x "$BIN_BEFORE" && -x "$BIN_AFTER" ]] || precheck_fail "bench-fandhe のバイナリが無い"
[[ "$(sha256_of "$BIN_BEFORE")" != "$(sha256_of "$BIN_AFTER")" ]] || precheck_fail "before と after のバイナリが同一（腕が分かれていない）"
echo "precheck=ok"

{
  echo "profile=${PROFILE}"; echo "label=${LABEL}"; echo "host=masked"
  echo "uname=$(uname -srm)"
  if [[ "$PROFILE" == "macmini" ]]; then echo "cpu_brand=$(sysctl -n machdep.cpu.brand_string)"; else echo "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)"; fi
  echo "rule_sha256=${RULE_SHA}"; echo "bench_py_sha256=$(sha256_of "$RUNNER_ROOT/$BENCH_PY_REL")"
  echo "rev_stamp_after=$(stamp_of "$BASE_ROOT")"; echo "rev_stamp_runner=${STAMP_RUNNER}"; echo "baseline_commit=${BASELINE_COMMIT}"
  echo "cargo_lock_sha256=$(sha256_of "$LOCK_COPY")"
  echo "bin_sha256_before=$(sha256_of "$BIN_BEFORE")"; echo "bin_sha256_after=$(sha256_of "$BIN_AFTER")"
  echo "bin_sha256_candle=$(sha256_of "$OPP_DIR/bench-candle")"; echo "bin_sha256_burn=$(sha256_of "$OPP_DIR/bench-burn")"
  echo "python_pt=$("$PY_PT" -c 'import sys,torch;print(sys.version.split()[0],"torch",torch.__version__)' 2>&1 | tail -1)"
  echo "python_tf=$("$PY_TF" -c 'import tensorflow as t;print("tf",t.__version__)' 2>&1 | tail -1)"
  echo "python_scipy=$("$PY_PT" -c 'import scipy;print("scipy",scipy.__version__)' 2>&1 | tail -1)"
  echo "gate=1"; echo "rounds=${ROUNDS}"; echo "utc=$(date -u +%FT%TZ)"
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
# GB10 の専有確認（0.14.0 RULE と同じ fail-closed）: 取得失敗は EXCLUSIVE-INVALID、1 件でもあれば専有未達。
compute_apps_zero() { # <記録先>
  local apps rc
  apps="$(nvidia-smi --query-compute-apps=pid,name --format=csv,noheader 2>&1)"; rc=$?
  printf '%s rc=%s\n%s\n' "$(date -u +%FT%TZ)" "$rc" "$apps" >>"$1"
  [[ "$rc" -eq 0 ]] || return 2
  [[ -z "$apps" ]] || return 1
  return 0
}
# ゲート: load1 < 閾値を 30 秒間隔・最大 30 分待つ（GB10 は GPU 使用率 0% 3 サンプルと compute プロセス 0 も要求）。
gate_wait() { # <round>
  local waited=0 l ok rc
  if [[ "$PROFILE" == "gb10" ]]; then
    rc=0; compute_apps_zero "$OUT/compute_apps.log" || rc=$?
    if [[ "$rc" -eq 2 ]]; then echo "round$1 EXCLUSIVE-INVALID" >>"$OUT/gate.log"; return 2; fi
    if [[ "$rc" -eq 1 ]]; then echo "round$1 EXCLUSIVE-UNMET compute-process-present" >>"$OUT/gate.log"; return 3; fi
  fi
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
cell() { # <bin> <out> <task> <dev> <size> <mode>   失敗は skipped へ記録（捏造しない）。--diag-* は渡さない。
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
      cell "$BIN_BEFORE" "$RD/results-before.jsonl" "$1" "$2" "$3" "$mode"; cell "$BIN_AFTER" "$RD/results-after.jsonl" "$1" "$2" "$3" "$mode"
    else
      cell "$BIN_AFTER" "$RD/results-after.jsonl" "$1" "$2" "$3" "$mode"; cell "$BIN_BEFORE" "$RD/results-before.jsonl" "$1" "$2" "$3" "$mode"
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
  : >"$RD/results-before.jsonl"; : >"$RD/results-after.jsonl"; : >"$RD/results-opp.jsonl"; : >"$RD/results-py.jsonl"
  PYF="$RD/results-py.jsonl"
  grc=0; gate_wait "$R" || grc=$?
  if [[ "$grc" -ne 0 ]]; then echo "SERIES-ABORTED completed=$((R - 1))/5 (gate rc=${grc})"; exit 1; fi
  uptime >"$RD/uptime_before.txt"
  if [[ "$PROFILE" == "macmini" ]]; then pmset -g therm >"$RD/pmset_before.txt" 2>&1 || true
  else nvidia-smi --query-compute-apps=pid,name --format=csv >"$RD/gpu_apps_before.txt" 2>&1 || true; fi
  # 順序: Rust の相手 → fandhe 両腕（セルごとに before/after を続け、先行腕はラウンドの奇偶で反転）→ Python
  each_cell opp_cell
  each_cell fandhe_cell
  each_cell py_cell
  uptime >"$RD/uptime_after.txt"
  if [[ "$PROFILE" == "macmini" ]]; then pmset -g therm >"$RD/pmset_after.txt" 2>&1 || true
  else nvidia-smi --query-compute-apps=pid,name --format=csv >"$RD/gpu_apps_after.txt" 2>&1 || true; fi
  rows_b=$(wc -l <"$RD/results-before.jsonl" | tr -d ' '); rows_a=$(wc -l <"$RD/results-after.jsonl" | tr -d ' ')
  rows_opp=$(wc -l <"$RD/results-opp.jsonl" | tr -d ' '); rows_py=$(wc -l <"$RD/results-py.jsonl" | tr -d ' ')
  echo "run${R}: before=${rows_b} after=${rows_a} opp=${rows_opp} py=${rows_py}"
  unexpected=$(grep -vcE '^bench-burn gemm metal (512|1024|2048|4096) fresh: MEASURE_ERROR: gemm checksum is degenerate' "$SKIP")
  if [[ "$rows_b" != "$EXP_ARM" || "$rows_a" != "$EXP_ARM" || "$rows_opp" != "$EXP_OPP" || "$rows_py" != "$EXP_PY" ]]; then
    echo "SERIES-ABORTED completed=$((R - 1))/5 (期待行数 ${EXP_ARM}/${EXP_ARM}/${EXP_OPP}/${EXP_PY} と不一致)"; exit 1
  fi
  if [[ "$unexpected" != "0" ]]; then echo "SERIES-ABORTED completed=$((R - 1))/5 (想定外の失敗)"; exit 1; fi
done
rm -f "$OUT/err.tmp"
restore_lock
if ! git -C "$BASE_ROOT" diff --quiet -- "$LOCK_CLEAN_REL"; then echo "error: Cargo.lock に差分が残った" >&2; exit 1; fi
echo "ALL-DONE completed=5/5"
