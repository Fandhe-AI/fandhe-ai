#!/usr/bin/env bash
# イシュー #1312: DGX Spark GB10 上での TwoDDynamic vs RowPanel 両実機 A/B
# 一括実行オーケストレーションスクリプト。~/work/ab-1312/ 上で setsid nohup
# により切り離して実行する想定（#1305/#1367 の dgx_orchestrate.sh と同型）。
#
# 前提条件を満たさない場合は計測を中止する（前提不成立を REJECT 判定より
# 先に検出する。計画 §5 Step 3-3）:
#   - `cargo test -p fandhe-ai-backend-cpu --lib gemm_blis --release` green
#   - `gemm_blis_two_d_dynamic_matches_row_panel_bit_exact_large`（--ignored）pass
set -uo pipefail

WORKDIR="${WORKDIR:-$HOME/work/ab-1312}"
cd "$WORKDIR"
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$HOME/work/target-ab-1312}"
LOGDIR="${LOGDIR:-$WORKDIR/out}"
AB_SCRIPT_DIR="${AB_SCRIPT_DIR:-$WORKDIR/ab-scripts}"
MACHINE="dgx"

if [ ! -f "$AB_SCRIPT_DIR/run_ab.sh" ]; then
  echo "エラー: AB_SCRIPT_DIR='$AB_SCRIPT_DIR' に run_ab.sh が見つからない" >&2
  exit 1
fi

mkdir -p "$LOGDIR"

# 実行中 load average の推移を記録するバックグラウンドポーラー
(
  while :; do
    date -u +"%Y-%m-%dT%H:%M:%SZ" >>"$LOGDIR/uptime-dgx.log"
    uptime >>"$LOGDIR/uptime-dgx.log"
    sleep 30
  done
) &
POLLER_PID=$!

# 専有状態ゲート: 1 分平均が 6.0 未満で 2 回連続を確認するまで最大 30 分待機
# （計画 §4.1 条件 7 は M4 Max 側の閾値〈6.0〉をそのまま DGX にも適用したもの。
#   #1305 の DGX 側スイープ〈dgx_orchestrate.sh〉は 2.0 を使っていたが、本スクリプトは
#   両実機で同一閾値〈6.0〉を用いる計画 §4.1 の記述に従う。DGX は実測時 load average が
#   概ね 0.00〜0.01 だったためどちらの閾値でも成立に変わりはない）
{
  echo "=== gate start $(date -u +%FT%TZ) ==="
  attempts=0
  consec=0
  while [ "$attempts" -lt 30 ]; do
    load1=$(uptime | sed -E 's/.*load average: ([0-9.]+).*/\1/')
    echo "attempt=$attempts load1=$load1"
    ok=$(awk -v l="$load1" 'BEGIN{print (l<6.0)?1:0}')
    if [ "$ok" = "1" ]; then
      consec=$((consec + 1))
    else
      consec=0
    fi
    if [ "$consec" -ge 2 ]; then
      echo "gate PASSED at attempt=$attempts"
      break
    fi
    attempts=$((attempts + 1))
    sleep 60
  done
  if [ "$consec" -lt 2 ]; then
    echo "gate NOT PASSED after max wait (共有負荷下として続行)"
  fi
  echo "who: $(who)"
  ps -eo pcpu,pid,comm --sort=-pcpu | head -15
  echo "=== gate end $(date -u +%FT%TZ) ==="
} >"$LOGDIR/gate-dgx.log" 2>&1

# 前提 1: 通常テスト green
cargo test -p fandhe-ai-backend-cpu --lib gemm_blis --release >"$LOGDIR/unit-test-dgx.txt" 2>&1
if [ $? -ne 0 ]; then
  echo "前提不成立: 通常テストが失敗（unit-test-dgx.txt 参照）" >"$LOGDIR/PRECONDITION_FAILED.marker"
  kill "$POLLER_PID" 2>/dev/null || true
  exit 1
fi

# 前提 2: bit 完全一致（大形状）
cargo test -p fandhe-ai-backend-cpu --release --lib \
  -- --ignored gemm_blis_two_d_dynamic_matches_row_panel_bit_exact_large --nocapture \
  >"$LOGDIR/bit-exact-large-dgx.txt" 2>&1
if [ $? -ne 0 ]; then
  echo "前提不成立: bit 完全一致テストが失敗（bit-exact-large-dgx.txt 参照）" >"$LOGDIR/PRECONDITION_FAILED.marker"
  kill "$POLLER_PID" 2>/dev/null || true
  exit 1
fi

# 主計測マトリクス: T=8, T=10, 既定（未設定=20）
# run_ab.sh の終了コードは呼び出し元へ伝播させる（計測プロセスの非 0 終了を
# 検出したら ALL_DONE.marker を作らず失敗のまま停止する。codex-review 指摘
# 対応: #1312 PR #1444 レビュースレッド）
LOGDIR="$LOGDIR" MACHINE="$MACHINE" THREADS="8 10 default" RUNS=5 \
  bash "$AB_SCRIPT_DIR/run_ab.sh"
if [ $? -ne 0 ]; then
  echo "計測失敗: run_ab.sh が非 0 終了（$LOGDIR/FAILURES.log 参照）" >"$LOGDIR/RUN_AB_FAILED.marker"
  kill "$POLLER_PID" 2>/dev/null || true
  exit 1
fi

# oss-gemm-compare（Tier 2。gemm crate 比較用）5 回独立実行
OSS_BIN="$CARGO_TARGET_DIR/oss-release/oss-gemm-compare"
if [ ! -x "$OSS_BIN" ]; then
  (
    cd "$WORKDIR/scripts/bench/oss-gemm-compare" &&
      cargo build --release --target-dir "$CARGO_TARGET_DIR/oss-release"
  ) >"$LOGDIR/oss-build-dgx.log" 2>&1
  OSS_BIN="$CARGO_TARGET_DIR/oss-release/release/oss-gemm-compare"
fi
for run in 1 2 3 4 5; do
  "$OSS_BIN" --sizes 1024,2048,4096 >"$LOGDIR/oss-dgx-run${run}.jsonl" 2>"$LOGDIR/oss-dgx-run${run}.log"
done

# 環境情報（内部ホスト名を含めない）
{
  rustc -V
  cat /etc/os-release 2>/dev/null | head -5
} >"$LOGDIR/rustc-version-dgx.txt" 2>&1
lscpu >"$LOGDIR/env_info_raw-dgx.txt" 2>&1 || true
# ホスト名行を除去（多層防御。lscpu 自体はホスト名を出力しないが、他行に
# 万一混入した場合に備える）
sed -i '/Hostname/Id' "$LOGDIR/env_info_raw-dgx.txt" 2>/dev/null || true

kill "$POLLER_PID" 2>/dev/null || true
echo "done." >"$LOGDIR/ALL_DONE.marker"
echo "all done $(date -u +%FT%TZ)" >>"$LOGDIR/uptime-dgx.log"
