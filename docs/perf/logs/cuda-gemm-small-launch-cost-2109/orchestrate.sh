#!/usr/bin/env bash
# イシュー #2109: CUDA gemm N=256 起動固定費の診断を GB10 実機で一括実行する
# オーケストレーション（親 #2099。事前登録規則は同ディレクトリの RULE.txt）。
# 構成は #1973（`../cuda-gemm-reuse-phase-1973/orchestrate.sh`）と同型:
#   - Layer A: `bench-fandhe --task gemm --device cuda --mode reuse --phases
#     --size 256`（registry ピン `fandhe-ai =0.9.0`）を 5 run
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
# 使い方（GB10 実機。別セッション）:
#   ./orchestrate.sh
#   ./orchestrate.sh --dry-run   # 経路解決のみ（実機不要）
set -euo pipefail

DRY_RUN=0
case "${1:-}" in
  "") ;;
  --dry-run) DRY_RUN=1 ;;
  *) echo "ERROR: 未知の引数: ${1}（許容: --dry-run）" >&2; exit 2 ;;
esac

SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SELF_DIR/../../../.." && pwd)"
WORK_DIR="${FRAMEWORK_COMPARE_DIR:-$REPO_ROOT/scripts/bench/framework-compare}"
N=256
RUNS=5
RUN_NSYS="${RUN_NSYS:-0}"
LAYER_B_FILTER="gemm_small_launch_cost_diag_tests::gemm_small_launch_cost_diag"

if [[ ! -f "$WORK_DIR/Cargo.toml" ]]; then
  echo "ERROR: framework-compare が見つからない: $WORK_DIR" >&2
  exit 1
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "dry-run: WORK_DIR=$WORK_DIR"
  echo "dry-run: REPO_ROOT=$REPO_ROOT"
  echo "dry-run: N=$N runs=$RUNS RUN_NSYS=$RUN_NSYS"
  ls -la "$REPO_ROOT/crates/backend-cuda/src/gemm_small_launch_cost_diag_tests.rs"
  grep -n '^fandhe-ai' "$WORK_DIR/bench-fandhe/Cargo.toml" || true
  echo "dry-run: cargo build --release -p bench-fandhe"
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
      echo "$label PASS load1=$load gpu_util=$util" >>"$SELF_DIR/load_gate.log"
      return 0
    fi
    if (( waited >= 1800 )); then
      echo "$label FAIL(参考扱い) load1=$load gpu_util=$util" >>"$SELF_DIR/load_gate.log"
      return 0
    fi
    sleep 30
    waited=$((waited + 30))
  done
}

: >"$SELF_DIR/load_gate.log"

cd "$WORK_DIR"
echo "== ビルド（release） =="
cargo build --release -p bench-fandhe
cargo build --release -p bench-candle --no-default-features --features cuda

echo "== Layer A: reuse --phases N=$N =="
OUT="$SELF_DIR/layerA-phases-N${N}.log"
: >"$OUT"
for i in $(seq 1 "$RUNS"); do
  gate "layerA-phases run$i"
  echo "-- run $i/N=$N --" >>"$OUT"
  ./target/release/bench-fandhe --task gemm --device cuda --size "$N" \
    --mode reuse --phases 2>&1 | mask >>"$OUT"
done

echo "== AC-2: reuse（非 phases） N=$N =="
OUT="$SELF_DIR/layerA-ac2-N${N}.log"
: >"$OUT"
for i in $(seq 1 "$RUNS"); do
  gate "ac2 run$i"
  echo "-- run $i/N=$N --" >>"$OUT"
  ./target/release/bench-fandhe --task gemm --device cuda --size "$N" \
    --mode reuse 2>&1 | mask >>"$OUT"
done

echo "== candle 参照（診断用） fresh N=$N =="
OUT="$SELF_DIR/candle-fresh-N${N}.log"
: >"$OUT"
for i in $(seq 1 "$RUNS"); do
  gate "candle run$i"
  echo "-- run $i/N=$N --" >>"$OUT"
  ./target/release/bench-candle --task gemm --device cuda --size "$N" \
    --mode fresh 2>&1 | mask >>"$OUT"
done

echo "== Layer B: 診断テスト（${RUNS} プロセス） =="
cd "$REPO_ROOT"
for i in $(seq 1 "$RUNS"); do
  gate "layerB run$i"
  cargo test --release -p fandhe-ai-backend-cuda --lib "$LAYER_B_FILTER" \
    -- --ignored --exact --test-threads=1 --nocapture 2>&1 | mask >"$SELF_DIR/layerB-run${i}.log"
done
echo "== 件数の厳密断言 =="
cargo test --release -p fandhe-ai-backend-cuda --lib gemm_small_launch_cost_diag_tests::gemm_small_launch_counts_exact \
  -- --ignored --exact --test-threads=1 2>&1 | mask >"$SELF_DIR/counts-exact.log"

if [[ "$RUN_NSYS" == "1" ]]; then
  echo "== 任意: nsys CUDA API 集計（Layer B 1 プロセス） =="
  if command -v nsys >/dev/null 2>&1; then
    nsys profile --trace=cuda --stats=true -o "$(mktemp -d)/gemm2109" \
      cargo test --release -p fandhe-ai-backend-cuda --lib "$LAYER_B_FILTER" \
      -- --ignored --exact --test-threads=1 --nocapture 2>&1 | mask >"$SELF_DIR/nsys-cuda-api.log"
  else
    echo "nsys 欠測（未導入）" >"$SELF_DIR/nsys-cuda-api.log"
  fi
else
  echo "nsys 欠測（RUN_NSYS=1 未指定）" >"$SELF_DIR/nsys-cuda-api.log"
fi

{
  echo "計測終了時刻（UTC）: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  uptime
  nvidia-smi --query-gpu=name,driver_version,utilization.gpu --format=csv || true
} 2>&1 | mask >>"$SELF_DIR/env_info.txt"

echo "done. 次: python3 $SELF_DIR/aggregate.py"
