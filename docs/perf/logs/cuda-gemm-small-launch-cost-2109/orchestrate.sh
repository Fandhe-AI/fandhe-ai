#!/usr/bin/env bash
# イシュー #2109: CUDA gemm N=256 起動固定費の診断を GB10 実機で一括実行する
# オーケストレーション（親 #2099。事前登録規則は同ディレクトリの RULE.txt）。
# 構成は #1973（`../cuda-gemm-reuse-phase-1973/orchestrate.sh`）と同型:
#   - Layer A: `bench-fandhe --task gemm --device cuda --mode reuse --phases
#     --size 256`（registry ピン `fandhe-ai =0.9.0`）を 5 run
#   - HEAD path-patch Layer A（必須。RULE.txt 4）: 同一 bench を
#     `--config 'patch.crates-io.fandhe-ai.path=<HEAD crates/facade>'` でビルドし
#     phases／非 phases を 5 run ずつ（H5 の正式な突合相手・ratio の分子）
#   - AC-2: `--phases` なしの reuse N=256 を 5 run（checksum 突合）
#   - candle 参照（診断用。判定に使わない）: `bench-candle --mode fresh` 5 run
#   - Layer B: `gemm_small_launch_cost_diag`（crates/backend-cuda 非公開 API・
#     HEAD ツリー）を 5 プロセス
#   - 任意: `RUN_NSYS=1` で nsys の CUDA API 集計（cudarc 内部の event・
#     async alloc の件数を得る唯一の手段）。未実行なら「欠測」と記録する
# 専有ゲート: 各 run 前に load1 < 1.0 かつ GPU 利用率 0% を確認する
# （30 秒間隔・最大 30 分。結果は load_gate.log）。
# 収録時にホスト名を masked、$HOME を <home> へ置換する（RULE.txt 10）。
#
# 上書き禁止（RULE.txt 1: run1〜run5 の差し替え・追加起動・上書きをしない）:
#   出力先 LOG_DIR（既定は本ディレクトリ）に生成物ログが 1 つでもあれば、計測開始前に
#   exit 1 で停止する。再計測は空の別ディレクトリを LOG_DIR に指定する（既存ログは
#   触らない）。加えて `set -o noclobber` で `>` による既存ファイルの切り詰めも
#   シェルが拒否する（事前検査をすり抜けた場合の二重防止）。
#
# 使い方（GB10 実機。別セッション）:
#   ./orchestrate.sh
#   LOG_DIR=/path/to/new-empty-dir ./orchestrate.sh   # 別ディレクトリへ保存
#   ./orchestrate.sh --dry-run   # 経路解決と上書き検査のみ（実機不要）
set -euo pipefail
set -o noclobber

DRY_RUN=0
case "${1:-}" in
  "") ;;
  --dry-run) DRY_RUN=1 ;;
  *) echo "ERROR: 未知の引数: ${1}（許容: --dry-run）" >&2; exit 2 ;;
esac

SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SELF_DIR/../../../.." && pwd)"
WORK_DIR="${FRAMEWORK_COMPARE_DIR:-$REPO_ROOT/scripts/bench/framework-compare}"
LOG_DIR="${LOG_DIR:-$SELF_DIR}"
# パスは最初の `cd` より前に絶対パスへ正規化する（後段で `cd "$WORK_DIR"`／`cd "$REPO_ROOT"` するため、
# 相対のままだと既存ログ検査・load_gate.log・Layer A・Layer B が別ディレクトリへ分散する）。
# ディレクトリ未作成でも解決できるよう先頭が `/` でなければ起動時の cwd を前置する。
case "$LOG_DIR" in /*) ;; *) LOG_DIR="$PWD/$LOG_DIR" ;; esac
case "$WORK_DIR" in /*) ;; *) WORK_DIR="$PWD/$WORK_DIR" ;; esac
N=256
RUNS=5
RUN_NSYS="${RUN_NSYS:-0}"
LAYER_B_FILTER="gemm_small_launch_cost_diag_tests::gemm_small_launch_cost_diag"

if [[ ! -f "$WORK_DIR/Cargo.toml" ]]; then
  echo "ERROR: framework-compare が見つからない: $WORK_DIR" >&2
  exit 1
fi

# 事前登録済みの生成物ログ（aggregate.py の入力＋補助ログ）が 1 つでもあれば開始前に停止する。
# 計測終了後に env_info へ追記される「計測終了時刻」行も計測済みの印として扱う。
existing=()
for f in load_gate.log layerA-phases-N${N}.log layerA-ac2-N${N}.log head-phases-N${N}.log \
         head-ac2-N${N}.log candle-fresh-N${N}.log counts-exact.log nsys-cuda-api.log \
         $(seq -f 'layerB-run%g.log' 1 "$RUNS"); do
  [[ -e "$LOG_DIR/$f" ]] && existing+=("$f")
done
if [[ -f "$LOG_DIR/env_info.txt" ]] && grep -q '^計測終了時刻' "$LOG_DIR/env_info.txt"; then
  existing+=("env_info.txt（計測終了時刻あり）")
fi
if (( ${#existing[@]} > 0 )); then
  echo "ERROR: 出力先 $LOG_DIR に既存の計測ログがある。上書きは禁止（RULE.txt 1）: ${existing[*]}" >&2
  echo "ERROR: 再計測は空の別ディレクトリを LOG_DIR に指定すること" >&2
  exit 1
fi
mkdir -p "$LOG_DIR"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "dry-run: LOG_DIR=$LOG_DIR"
  echo "dry-run: WORK_DIR=$WORK_DIR"
  echo "dry-run: REPO_ROOT=$REPO_ROOT"
  echo "dry-run: N=$N runs=$RUNS RUN_NSYS=$RUN_NSYS"
  ls -la "$REPO_ROOT/crates/backend-cuda/src/gemm_small_launch_cost_diag_tests.rs"
  grep -n '^fandhe-ai' "$WORK_DIR/bench-fandhe/Cargo.toml" || true
  echo "dry-run: cargo build --release -p bench-fandhe"
  echo "dry-run: cargo build --release -p bench-fandhe --config 'patch.crates-io.fandhe-ai.path=\"$REPO_ROOT/crates/facade\"'"
  echo "dry-run: cargo build --release -p bench-candle --no-default-features --features cuda"
  echo "dry-run: cargo test --release -p fandhe-ai-backend-cuda --lib $LAYER_B_FILTER -- --ignored --exact --test-threads=1 --nocapture"
  echo "dry-run: OK"
  exit 0
fi

# 収録時マスク（ホスト名 → masked、$HOME → <home>）。
HOST_NAME="$(hostname)"
mask() {
  sed -e "s|${HOME}|<home>|g" -e "s|${HOST_NAME}|masked|g"
}

# 専有ゲート。通らなければ結果を load_gate.log に残し、その run を参考扱いにする。
gate() {
  local label="$1" waited=0 load util
  while :; do
    load="$(cut -d' ' -f1 /proc/loadavg)"
    util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
    if awk -v l="$load" 'BEGIN{exit !(l < 1.0)}' && [[ "$util" == "0" ]]; then
      echo "$label PASS load1=$load gpu_util=$util" >>"$LOG_DIR/load_gate.log"
      return 0
    fi
    if (( waited >= 1800 )); then
      echo "$label FAIL(参考扱い) load1=$load gpu_util=$util" >>"$LOG_DIR/load_gate.log"
      return 0
    fi
    sleep 30
    waited=$((waited + 30))
  done
}

: >"$LOG_DIR/load_gate.log"

cd "$WORK_DIR"
# path patch のビルドは framework-compare/Cargo.lock を書き換えるため必ず復元する
# （承認ピン固定。deps-policy.md 第 9 区分）。
source ./bench_fandhe_lock_restore.sh
bench_fandhe_setup_lock_restore_trap
BIN_DIR="$(mktemp -d)"
echo "== ビルド（release） =="
cargo build --release -p bench-fandhe
cp ./target/release/bench-fandhe "$BIN_DIR/bench-fandhe-registry"
echo "== ビルド（release・HEAD path-patch） =="
cargo build --release -p bench-fandhe \
  --config "patch.crates-io.fandhe-ai.path=\"$REPO_ROOT/crates/facade\""
cp ./target/release/bench-fandhe "$BIN_DIR/bench-fandhe-head"
# Cargo.lock はビルド直後に復元し trap を解除する（後段で cd するため EXIT trap の相対パスが外れる）。
bench_fandhe_restore_lock
trap - EXIT
cargo build --release -p bench-candle --no-default-features --features cuda

echo "== Layer A: reuse --phases N=$N =="
OUT="$LOG_DIR/layerA-phases-N${N}.log"
: >"$OUT"
for i in $(seq 1 "$RUNS"); do
  gate "layerA-phases run$i"
  echo "-- run $i/N=$N --" >>"$OUT"
  "$BIN_DIR/bench-fandhe-registry" --task gemm --device cuda --size "$N" \
    --mode reuse --phases 2>&1 | mask >>"$OUT"
done

echo "== AC-2: reuse（非 phases） N=$N =="
OUT="$LOG_DIR/layerA-ac2-N${N}.log"
: >"$OUT"
for i in $(seq 1 "$RUNS"); do
  gate "ac2 run$i"
  echo "-- run $i/N=$N --" >>"$OUT"
  "$BIN_DIR/bench-fandhe-registry" --task gemm --device cuda --size "$N" \
    --mode reuse 2>&1 | mask >>"$OUT"
done

echo "== HEAD path-patch Layer A: reuse --phases N=$N =="
OUT="$LOG_DIR/head-phases-N${N}.log"
: >"$OUT"
for i in $(seq 1 "$RUNS"); do
  gate "head-phases run$i"
  echo "-- run $i/N=$N --" >>"$OUT"
  "$BIN_DIR/bench-fandhe-head" --task gemm --device cuda --size "$N" \
    --mode reuse --phases 2>&1 | mask >>"$OUT"
done

echo "== HEAD path-patch AC-2: reuse（非 phases） N=$N =="
OUT="$LOG_DIR/head-ac2-N${N}.log"
: >"$OUT"
for i in $(seq 1 "$RUNS"); do
  gate "head-ac2 run$i"
  echo "-- run $i/N=$N --" >>"$OUT"
  "$BIN_DIR/bench-fandhe-head" --task gemm --device cuda --size "$N" \
    --mode reuse 2>&1 | mask >>"$OUT"
done

echo "== candle 参照（診断用） fresh N=$N =="
OUT="$LOG_DIR/candle-fresh-N${N}.log"
: >"$OUT"
for i in $(seq 1 "$RUNS"); do
  gate "candle run$i"
  echo "-- run $i/N=$N --" >>"$OUT"
  ./target/release/bench-candle --task gemm --device cuda --size "$N" \
    --mode fresh 2>&1 | mask >>"$OUT"
done

rm -rf "$BIN_DIR"

echo "== Layer B: 診断テスト（${RUNS} プロセス） =="
cd "$REPO_ROOT"
for i in $(seq 1 "$RUNS"); do
  gate "layerB run$i"
  cargo test --release -p fandhe-ai-backend-cuda --lib "$LAYER_B_FILTER" \
    -- --ignored --exact --test-threads=1 --nocapture 2>&1 | mask >"$LOG_DIR/layerB-run${i}.log"
done
echo "== 件数の厳密断言 =="
cargo test --release -p fandhe-ai-backend-cuda --lib gemm_small_launch_cost_diag_tests::gemm_small_launch_counts_exact \
  -- --ignored --exact --test-threads=1 2>&1 | mask >"$LOG_DIR/counts-exact.log"

if [[ "$RUN_NSYS" == "1" ]]; then
  echo "== 任意: nsys CUDA API 集計（Layer B 1 プロセス） =="
  if command -v nsys >/dev/null 2>&1; then
    nsys profile --trace=cuda --stats=true -o "$(mktemp -d)/gemm2109" \
      cargo test --release -p fandhe-ai-backend-cuda --lib "$LAYER_B_FILTER" \
      -- --ignored --exact --test-threads=1 --nocapture 2>&1 | mask >"$LOG_DIR/nsys-cuda-api.log"
  else
    echo "nsys 欠測（未導入）" >"$LOG_DIR/nsys-cuda-api.log"
  fi
else
  echo "nsys 欠測（RUN_NSYS=1 未指定）" >"$LOG_DIR/nsys-cuda-api.log"
fi

{
  echo "計測終了時刻（UTC）: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  uptime
  nvidia-smi --query-gpu=name,driver_version,utilization.gpu --format=csv || true
} 2>&1 | mask >>"$LOG_DIR/env_info.txt"

echo "done. 次: python3 $SELF_DIR/aggregate.py --log-dir $LOG_DIR"
