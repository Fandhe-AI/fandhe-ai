#!/usr/bin/env bash
# イシュー #2100: CPU backward 非 GEMM 内訳の診断（train --phases の各 step ごとの
# DIAG_BACKWARD 行）を、1 台の機体で 5 run（独立プロセス）分収集する
# オーケストレーター。M4 Max（m4max）・DGX Spark GB10（gb10）・非公式スモーク
# 用の x86 Linux（smoke-x86）で共用する（機体差は負荷ゲートと env_info だけ）。
#
# 役割と契約:
#   - `prepare_tree.sh` が作った 2 本のツリー（計装あり／計装なし。同一 ref）の
#     `crates/facade` を `[patch.crates-io.fandhe-ai]` の path patch として
#     CLI 引数だけで与え、`scripts/bench/framework-compare` の `bench-fandhe`
#     を 2 本ビルドする（Cargo.toml／Cargo.lock はコミットしない。
#     deps-policy.md 第 9 区分は registry 取得元のみ許容。Cargo.lock は
#     `bench_fandhe_lock_restore.sh` で退避し EXIT trap で復元する。
#     `run_ab_1578.sh` と同方式）。
#   - 実測開始前に RULE.txt（判定規則。事前登録）を出力先へ書く。判定種別は
#     record_only。唯一の fail-closed 検査は計装あり／なしの JSONL checksum の
#     完全一致（`aggregate_backward_diag.py` が検査する）。
#   - 5 run はそれぞれ独立プロセス。run ごとに計装あり／なしの起動順を反転する。
#   - 各 run の前に負荷ゲート（m4max: load1 < 8.0／gb10: load1 < 1.0 かつ GPU
#     util 0%）を 30 秒間隔・最大 30 分待つ。不通過でも run は実施し、
#     gate.tsv に pass=0 を記録する（その系列は参考扱い）。smoke-x86 は
#     ゲートなし（記録のみ）。
#
# 使い方:
#   DIAG_INSTR_FACADE_PATH=<instr ツリー>/crates/facade \
#   DIAG_PLAIN_FACADE_PATH=<plain ツリー>/crates/facade \
#   DIAG_OUT_DIR=<出力先の絶対パス> \
#   [DIAG_DEVICES="cpu"]   # 参考セル: m4max は "cpu metal"、gb10 は "cpu cuda"
#   [DIAG_RUNS=5]         # m4max／gb10 は 5 固定。smoke-x86 のみ 1〜99 可
#     bash orchestrate_backward_diag.sh <m4max|gb10|smoke-x86>
#
# セキュリティ: 引数・環境変数は allowlist／絶対パス検査で検証し、eval は使わ
# ない。収録ログ・env_info は `$HOME`・ホスト名を <home>／<host> へマスクする。
set -u

MACHINE=${1:-}
case "$MACHINE" in
  m4max | gb10 | smoke-x86) ;;
  *)
    echo "usage: $0 <m4max|gb10|smoke-x86>" >&2
    exit 1
    ;;
esac

HERE="$(cd "$(dirname "$0")" && pwd)"
FC_DIR="$(cd "$HERE/../../../../scripts/bench/framework-compare" && pwd)"
AGG="$HERE/aggregate_backward_diag.py"

validate_abs_dir() { # validate_abs_dir <var_name> <path> <must_exist:0|1>
  local name=$1 path=$2 must=$3
  if [[ -z "$path" || "$path" != /* ]]; then
    echo "error: $name must be an absolute path (got: '$path')" >&2
    exit 1
  fi
  if [[ ! "$path" =~ ^[A-Za-z0-9._/@+-]+$ ]]; then
    echo "error: $name contains characters outside [A-Za-z0-9._/@+-] (got: '$path')" >&2
    exit 1
  fi
  if [[ "$must" == "1" && ! -d "$path" ]]; then
    echo "error: $name is not a directory ($path)" >&2
    exit 1
  fi
}
validate_facade_path() {
  local name=$1 path=$2
  validate_abs_dir "$name" "$path" 1
  if [[ ! -f "$path/Cargo.toml" ]] || ! grep -qE '^\s*name\s*=\s*"fandhe-ai"\s*$' "$path/Cargo.toml"; then
    echo "error: $name/Cargo.toml does not declare name = \"fandhe-ai\" ($path)" >&2
    exit 1
  fi
}
validate_facade_path DIAG_INSTR_FACADE_PATH "${DIAG_INSTR_FACADE_PATH:-}"
validate_facade_path DIAG_PLAIN_FACADE_PATH "${DIAG_PLAIN_FACADE_PATH:-}"
validate_abs_dir DIAG_OUT_DIR "${DIAG_OUT_DIR:-}" 0
INSTR_FACADE="$DIAG_INSTR_FACADE_PATH"
PLAIN_FACADE="$DIAG_PLAIN_FACADE_PATH"
OUT="$DIAG_OUT_DIR"
if [[ "$INSTR_FACADE" == "$PLAIN_FACADE" ]]; then
  echo "error: instr / plain facade paths must differ" >&2
  exit 1
fi
# 同一 ref 検査: 計装あり／なしのツリーが同一コミットから作られていなければ、
# checksum 一致・オーバーヘッド比の前提（計装差分のみの A/B）が崩れるため測定前に失敗させる。
INSTR_REV="$(git -C "$INSTR_FACADE" rev-parse HEAD 2>/dev/null || true)"
PLAIN_REV="$(git -C "$PLAIN_FACADE" rev-parse HEAD 2>/dev/null || true)"
if [[ -z "$INSTR_REV" || -z "$PLAIN_REV" || "$INSTR_REV" != "$PLAIN_REV" ]]; then
  echo "error: instr / plain trees must be at the same git HEAD (instr=${INSTR_REV:-unknown} plain=${PLAIN_REV:-unknown})" >&2
  exit 1
fi
if ! grep -q "mod diag;" "$INSTR_FACADE/../autodiff/src/lib.rs" 2>/dev/null; then
  echo "error: instr tree does not contain the diag patch (crates/autodiff/src/lib.rs)" >&2
  exit 1
fi
if grep -q "mod diag;" "$PLAIN_FACADE/../autodiff/src/lib.rs" 2>/dev/null; then
  echo "error: plain tree unexpectedly contains the diag patch" >&2
  exit 1
fi

RUNS=${DIAG_RUNS:-5}
if [[ ! "$RUNS" =~ ^[1-9][0-9]?$ ]]; then
  echo "error: DIAG_RUNS must be a small positive integer (got: '$RUNS')" >&2
  exit 1
fi
# 実機（m4max／gb10）は RULE.txt の 5 run 契約（5 回計測中央値）に従い 5 固定。
# 少ない run は smoke-x86（記録のみ）に限る。
if [[ "$MACHINE" != "smoke-x86" && "$RUNS" != "5" ]]; then
  echo "error: DIAG_RUNS must be 5 for $MACHINE (got: '$RUNS'; other values are smoke-x86 only)" >&2
  exit 1
fi
DEVICES=${DIAG_DEVICES:-cpu}
# cpu は必須セル。空・cpu 欠落は測定前に失敗させる（参考セルだけの収録を防ぐ）。
HAS_CPU=0
for d in $DEVICES; do
  [[ "$d" == "cpu" ]] && HAS_CPU=1
done
if [[ "$HAS_CPU" -ne 1 ]]; then
  echo "error: DIAG_DEVICES must be non-empty and include cpu (got: '$DEVICES')" >&2
  exit 1
fi
for d in $DEVICES; do
  case "$d" in
    cpu | metal | cuda) ;;
    *)
      echo "error: DIAG_DEVICES entries must be cpu|metal|cuda (got: '$d')" >&2
      exit 1
      ;;
  esac
done
GATE_INTERVAL_S=${DIAG_GATE_INTERVAL_S:-30}
GATE_MAX_S=${DIAG_GATE_MAX_S:-1800}
if [[ ! "$GATE_INTERVAL_S" =~ ^[1-9][0-9]*$ || ! "$GATE_MAX_S" =~ ^[0-9]+$ ]]; then
  # 待機間隔 0 は waited が増えず不通過時に無期限待機になるため、正の整数に限る
  echo "error: DIAG_GATE_INTERVAL_S は正の整数、DIAG_GATE_MAX_S は 0 以上の整数で指定する" >&2
  exit 1
fi

mkdir -p "$OUT"

# ---- マスク（内部ホスト名・ホームディレクトリを収録ログへ残さない） ----
HOST_NAME="$(hostname 2>/dev/null || echo unknown)"
# sed 正規表現の特殊文字（. * [ ] ^ $ \ / と区切りの #）をエスケープする
sed_escape() { printf '%s' "$1" | sed -e 's/[][\\.*^$/#]/\\&/g'; }
mask() { # stdin -> stdout
  # worktree・出力先・作業ディレクトリは $HOME 外（/tmp 等）にも置かれうるため、
  # より長い（具体的な）パスから先に置換し、最後に $HOME を置換する。
  # WORK は mktemp 後にのみ設定される（未設定・空はスキップ）。
  local args=()
  local pair path label
  for pair in \
    "${WORK:-}|<work>" \
    "${OUT:-}|<out>" \
    "${INSTR_FACADE%/crates/facade}|<instr-tree>" \
    "${PLAIN_FACADE%/crates/facade}|<plain-tree>"; do
    path="${pair%%|*}"
    label="${pair##*|}"
    # 空または "/" 単体は全置換になるためスキップ
    if [[ -n "$path" && "$path" != "/" ]]; then
      args+=(-e "s#$(sed_escape "$path")#$label#g")
    fi
  done
  args+=(-e "s#$(sed_escape "$HOME")#<home>#g")
  # 3 文字以下のホスト名は一般語の過剰置換を招くためマスクしない
  if [[ "${#HOST_NAME}" -ge 4 ]]; then
    args+=(-e "s#$(sed_escape "$HOST_NAME")#<host>#g")
  fi
  sed "${args[@]}"
}

# ---- 1. RULE.txt（実測開始前に固定。事後に緩めない） ----
RULE="$OUT/RULE.txt"
if [[ -e "$RULE" ]]; then
  echo "error: $RULE already exists (実測前に固定した規則は上書きしない。別ディレクトリを使う)" >&2
  exit 1
fi
{
  echo "# RULE.txt — イシュー #2100 backward 非 GEMM 内訳の診断（事前登録。実測前に固定）"
  echo "fixed_at_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "machine: $MACHINE"
  echo "devices: $DEVICES  (cpu = 必須セル。metal／cuda = 参考セル。いずれも record_only)"
  echo "modes: fresh reuse"
  echo "task: train --size 64 --phases（1 プロセス 100 step。うち先頭 20 step は warmup）"
  echo "runs: $RUNS（それぞれ独立プロセス。差し替え・追加起動はしない）"
  echo "decision: record_only（診断のみ。ADOPT／REJECT は判定しない）"
  echo "fail_closed_check: 計装あり／計装なしの JSONL checksum が全セル・全 run で完全一致"
  echo "  （不一致なら系列を無効とする。計装が数値を変えないことの証明）"
  echo "recorded_not_judged: 計装オーバーヘッド比（計装あり step_total ÷ 計装なし step_total）"
  echo "aggregation: 各 run で step 20..99 の中央値 → その $RUNS run の中央値"
  echo "load_gate: m4max = load1 < 8.0／gb10 = load1 < 1.0 かつ GPU util 0%／smoke-x86 = なし"
  echo "  30 秒間隔・最大 30 分待つ。1 run でも不通過ならその系列は参考扱い（gate.tsv 参照）"
  echo "not_reexecuted: 既存 REJECT と重複する実験（#1578 MSE 逐次しきい値の A/B 等）は再実行しない"
  echo "masking: ホスト名・絶対パスは <host>／<home>／<out>／<work>／<instr-tree>／<plain-tree> へマスクして収録する"
} >"$RULE"
echo "wrote $RULE"

# ---- 2. env_info.txt ----
{
  echo "machine: $MACHINE"
  echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "uname: $(uname -srm)"
  echo "rustc: $(rustc --version 2>&1)"
  echo "cargo: $(cargo --version 2>&1)"
  echo "instr_tree_rev: $(git -C "$INSTR_FACADE" rev-parse HEAD 2>&1)"
  echo "plain_tree_rev: $(git -C "$PLAIN_FACADE" rev-parse HEAD 2>&1)"
  case "$(uname -s)" in
    Darwin)
      echo "cpu: $(sysctl -n machdep.cpu.brand_string 2>&1)"
      echo "ncpu: $(sysctl -n hw.ncpu 2>&1)"
      echo "mem_bytes: $(sysctl -n hw.memsize 2>&1)"
      ;;
    *)
      echo "cpu: $(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | sed 's/.*: //' || true)"
      echo "ncpu: $(nproc 2>&1)"
      echo "mem_kb: $(awk '/MemTotal/ {print $2}' /proc/meminfo 2>/dev/null)"
      if command -v nvidia-smi >/dev/null 2>&1; then
        echo "gpu: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>&1 | head -1)"
      fi
      ;;
  esac
  echo "uptime_at_start: $(uptime)"
} 2>&1 | mask >"$OUT/env_info.txt"

# ---- 3. Cargo.lock の退避・復元（path patch による書き換え対策） ----
cd "$FC_DIR" || exit 1
# shellcheck source=../../../../scripts/bench/framework-compare/bench_fandhe_lock_restore.sh
source ./bench_fandhe_lock_restore.sh
bench_fandhe_setup_lock_restore_trap

WORK="$(mktemp -d)"
# EXIT trap: 作業ディレクトリを削除してから、Cargo.lock 復元ハンドラへ元の
# 終了コードを引き継ぐ（`bench_fandhe_restore_lock_trap` は `$?` を読む）。
diag_exit_trap() {
  local c=$?
  rm -rf "$WORK"
  (exit "$c")
  bench_fandhe_restore_lock_trap
}
trap diag_exit_trap EXIT
BIN_DIR="$WORK/bin"
mkdir -p "$BIN_DIR"

build_arm() { # build_arm <arm> <facade_path>
  local arm=$1 facade=$2 patch_config msg exe tree_output
  patch_config="patch.crates-io.fandhe-ai.path=\"${facade}\""
  msg="$WORK/msg-$arm.json"
  if ! cargo build --release -p bench-fandhe --target-dir "$WORK/target-$arm" \
    --message-format=json --config "$patch_config" >"$msg" 2>"$WORK/build-$arm.err"; then
    tail -40 "$WORK/build-$arm.err" | mask >&2
    echo "error: bench-fandhe build failed ($arm)" >&2
    exit 1
  fi
  exe="$(jq -rs '[.[] | select(.reason == "compiler-artifact" and .target.name == "bench-fandhe" and (.target.kind[]? == "bin") and .executable != null)] | last | .executable // empty' "$msg")"
  if [[ -z "$exe" || ! -f "$exe" ]]; then
    echo "error: build $arm: executable not found" >&2
    exit 1
  fi
  cp "$exe" "$BIN_DIR/bench-fandhe-$arm"
  tree_output="$(cargo tree -p bench-fandhe --depth 1 --config "$patch_config" 2>&1)"
  if ! echo "$tree_output" | grep -qE 'fandhe-ai v[0-9.]+ \(.*crates/facade\)'; then
    echo "error: fandhe-ai did not resolve to the path-patched crates/facade ($arm)" >&2
    echo "$tree_output" | mask >&2
    exit 1
  fi
  {
    echo "$tree_output"
    echo "sha256: $(bench_fandhe_sha256_of "$BIN_DIR/bench-fandhe-$arm")"
  } | sed -e "s#${facade%/crates/facade}#<${arm}-tree>#g" | mask >"$OUT/build-$arm.txt"
}
echo "== build instr =="
build_arm instr "$INSTR_FACADE"
echo "== build plain =="
build_arm plain "$PLAIN_FACADE"

# ---- 4. 負荷ゲート ----
load1() {
  LC_ALL=C uptime | sed -E 's/.*load averages?: *//' | awk -F'[ ,]+' '{print $1}'
}
gpu_util() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' %'
  else
    echo "na"
  fi
}
# load_below <load1> <閾値>: load1 が数値でなければ（空・解析失敗）不通過（fail-closed）。
# 空文字列は awk の比較で閾値未満と誤判定されるため、先に数値検証する
load_below() {
  awk -v l="$1" -v t="$2" 'BEGIN{ if (l !~ /^[0-9]+([.][0-9]+)?$/) exit 1; exit !((l + 0) < (t + 0)) }'
}
GATE_TSV="$OUT/gate.tsv"
printf 'run\tdevice\tmode\tload1\tgpu_util\tpass\n' >"$GATE_TSV"

wait_gate() { # wait_gate <run> <device> <mode>; 結果を gate.tsv へ記録
  local run=$1 dev=$2 mode=$3 waited=0 l g pass=0
  while :; do
    l="$(load1)"
    g="$(gpu_util)"
    case "$MACHINE" in
      m4max) load_below "$l" 8.0 && pass=1 ;;
      gb10) load_below "$l" 1.0 && [[ "$g" == "0" ]] && pass=1 ;;
      smoke-x86) pass=1 ;;
    esac
    if [[ "$pass" == "1" || "$waited" -ge "$GATE_MAX_S" ]]; then
      break
    fi
    sleep "$GATE_INTERVAL_S"
    waited=$((waited + GATE_INTERVAL_S))
  done
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$run" "$dev" "$mode" "$l" "$g" "$pass" >>"$GATE_TSV"
}

# ---- 5. 計測 ----
ANY_FAILED=0
run_one() { # run_one <arm> <run> <device> <mode>
  local arm=$1 run=$2 dev=$3 mode=$4 base tmp_err rc
  base="$OUT/run${run}-${dev}-${mode}-${arm}"
  tmp_err="$WORK/err-${arm}.tmp"
  : >"$base.jsonl"
  echo "== run $run device=$dev mode=$mode arm=$arm =="
  if [[ "$arm" == "instr" ]]; then
    FANDHE_DIAG_BACKWARD=1 "$BIN_DIR/bench-fandhe-instr" --task train --size 64 \
      --device "$dev" --mode "$mode" --phases --out "$base.jsonl" >/dev/null 2>"$tmp_err"
    rc=$?
    mask <"$tmp_err" >"$base.err"
  else
    "$BIN_DIR/bench-fandhe-plain" --task train --size 64 \
      --device "$dev" --mode "$mode" --phases --out "$base.jsonl" >/dev/null 2>"$tmp_err"
    rc=$?
    # 計装なしの stderr は診断対象外（失敗時のみ内容を残す）。
    if [[ "$rc" -ne 0 ]]; then mask <"$tmp_err" >"$base.err"; fi
  fi
  rm -f "$tmp_err"
  if [[ "$rc" -ne 0 ]]; then
    echo "  -> FAILED rc=$rc (arm=$arm run=$run device=$dev mode=$mode)" >&2
    ANY_FAILED=$((ANY_FAILED + 1))
  fi
}

for dev in $DEVICES; do
  for mode in fresh reuse; do
    for run in $(seq 1 "$RUNS"); do
      wait_gate "$run" "$dev" "$mode"
      if ((run % 2 == 1)); then
        run_one plain "$run" "$dev" "$mode"
        run_one instr "$run" "$dev" "$mode"
      else
        run_one instr "$run" "$dev" "$mode"
        run_one plain "$run" "$dev" "$mode"
      fi
    done
  done
done
echo "uptime_at_end: $(uptime)" | mask >>"$OUT/env_info.txt"

# ---- 6. 集計（checksum 同一性検査を含む fail-closed） ----
python3 "$AGG" --machine "$MACHINE" --in-dir "$OUT" --runs "$RUNS" --devices "$DEVICES" \
  --out-csv "$OUT/iterations.csv" --out-md "$OUT/aggregate.md"
AGG_RC=$?
if [[ "$AGG_RC" -ne 0 ]]; then
  echo "error: aggregate failed rc=$AGG_RC（checksum 不一致・行数不一致等。系列は無効）" >&2
  exit 1
fi
if [[ "$ANY_FAILED" -gt 0 ]]; then
  echo "error: $ANY_FAILED run(s) failed" >&2
  exit 1
fi
echo "done. results in $OUT"
