#!/bin/bash
# Apple M4 Max 側オーケストレーション（イシュー #1481・このセッションの
# ホスト自身でローカル実行）。
# (i) 専有ゲート（1 分 load average < 6.0 を 30 秒間隔で 2 回連続・
#     最大 30 試行。不成立なら GATE_NOT_PASSED.marker を書いて終了し、
#     ユーザー承認事項どおり verdict=undetermined を 1 回だけ記録する
#     判断は呼び出し側〈本エージェント〉に委ねる。待ち続けない）
# (ii) Layer A（off=実装 worktree → on=スクラッチ複製ツリーの順）
#
# 対象ツリーは環境変数 OFF_TREE（off 腕＝実装 worktree の絶対パス）・
# ON_TREE（on 腕＝`on-arm.patch` 適用済みスクラッチ複製ツリーの絶対パス）
# で実行時に指定する（既定値なし。未指定なら usage を出して exit 1。
# PR #1501 codex-review P1 是正: 当初は個人 worktree とセッション UUID を
# 含む絶対パスをハードコードしており別 checkout で再計測できなかった。
# 実測時に使った具体値は env_info.txt へ記録する）。
#   使用例: OFF_TREE=/path/to/off ON_TREE=/path/to/on bash orchestrate_m4max.sh
# (iii) Layer B（run_layerB_m4max.sh off → on）
# (iv) ALL_DONE.marker（Layer A/B の 4 呼び出しすべてが成功した場合のみ）
#
# PR #1501 codex-review P3 是正: 当初は `set -uo pipefail`（errexit なし）
# のまま Layer A/B の各呼び出しの終了コードを確認していなかったため、
# 計測コマンドが失敗しても後続ステップがそのまま実行され続け、最終的に
# `ALL_DONE.marker` が生成されて（終了コード 0 で）不完全な計測を
# 完了として扱ってしまう欠陥があった。各呼び出しを `if ! ...; then`
# で明示的に確認し、失敗時は `MEASUREMENT_FAILED.marker` を書いて
# `ALL_DONE.marker` を生成せずに非 0 で終了するよう是正した。
set -uo pipefail
LOG_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -z "${OFF_TREE:-}" ] || [ -z "${ON_TREE:-}" ]; then
  echo "usage: OFF_TREE=<off 腕ツリーの絶対パス> ON_TREE=<on 腕ツリーの絶対パス> bash $0" >&2
  exit 1
fi
if [ ! -d "$OFF_TREE/scripts/bench/framework-compare" ] || [ ! -d "$ON_TREE/crates/facade" ]; then
  echo "error: OFF_TREE/ON_TREE がリポジトリツリーを指していない: OFF_TREE=$OFF_TREE ON_TREE=$ON_TREE" >&2
  exit 1
fi

# 状態マーカーの初期化（PR #1501 codex-review P2 是正: ログディレクトリを
# 再利用するため、前回実行の完了印・失敗印・ゲート不成立印が残ったままだと
# 今回の不完全な計測を完了と取り違える。実行ごとに開始時に消す）。
rm -f "$LOG_DIR/ALL_DONE.marker" "$LOG_DIR/MEASUREMENT_FAILED.marker" "$LOG_DIR/GATE_NOT_PASSED.marker"

# --- (i) 専有ゲート ---
GATE_OK=0
CONSEC=0
i=1
while [ "$i" -le 30 ]; do
  LOAD1=$(uptime | sed -E 's/.*load averages?: ([0-9.]+).*/\1/')
  OK=$(awk -v l="$LOAD1" 'BEGIN{print (l < 6.0) ? 1 : 0}')
  echo "gate try=$i load1=$LOAD1 ok=$OK $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG_DIR/gate-m4max.log"
  if [ "$OK" = "1" ]; then
    CONSEC=$((CONSEC + 1))
  else
    CONSEC=0
  fi
  if [ "$CONSEC" -ge 2 ]; then
    GATE_OK=1
    break
  fi
  i=$((i + 1))
  sleep 30
done

if [ "$GATE_OK" != "1" ]; then
  echo "gate not passed after 30 tries" > "$LOG_DIR/GATE_NOT_PASSED.marker"
  exit 0
fi

# 失敗確認ヘルパ: 呼び出しが非 0 終了した場合に MEASUREMENT_FAILED.marker
# を書いて非 0 で終了する（ALL_DONE.marker は生成しない。PR #1501
# codex-review P3 是正の核心）。
fail_measurement() {
  echo "$1 failed $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$LOG_DIR/MEASUREMENT_FAILED.marker"
  exit 1
}

# --- (ii) Layer A ---
cd "$OFF_TREE/scripts/bench/framework-compare"
SHA=$(cat "$OFF_TREE/.rev-stamp" 2>/dev/null || git -C "$OFF_TREE" rev-parse HEAD)

echo "layerA off start $(date -u +%Y-%m-%dT%H:%M:%SZ) $(uptime)" >> "$LOG_DIR/gate-m4max.log"
if ! GEMM_GATE_CPU_NODE_TAG=m4max-cpu GEMM_GATE_PATCH_FACADE_PATH="$OFF_TREE/crates/facade" \
  bash run_gemm_gate_cpu.sh "head-${SHA}-1481-pzero-off" \
  > "$LOG_DIR/run_gemm_gate_cpu-m4max-off.log" 2>&1; then
  fail_measurement "layerA off"
fi
echo "layerA off end $(date -u +%Y-%m-%dT%H:%M:%SZ) $(uptime)" >> "$LOG_DIR/gate-m4max.log"

echo "layerA on start $(date -u +%Y-%m-%dT%H:%M:%SZ) $(uptime)" >> "$LOG_DIR/gate-m4max.log"
if ! GEMM_GATE_CPU_NODE_TAG=m4max-cpu GEMM_GATE_PATCH_FACADE_PATH="$ON_TREE/crates/facade" \
  bash run_gemm_gate_cpu.sh "head-${SHA}-1481-pzero-on" \
  > "$LOG_DIR/run_gemm_gate_cpu-m4max-on.log" 2>&1; then
  fail_measurement "layerA on"
fi
echo "layerA on end $(date -u +%Y-%m-%dT%H:%M:%SZ) $(uptime)" >> "$LOG_DIR/gate-m4max.log"

# --- (iii) Layer B ---
if ! bash "$LOG_DIR/run_layerB_m4max.sh" "$OFF_TREE" off > "$LOG_DIR/layerB-m4max-off-driver.log" 2>&1; then
  fail_measurement "layerB off"
fi
if ! bash "$LOG_DIR/run_layerB_m4max.sh" "$ON_TREE" on > "$LOG_DIR/layerB-m4max-on-driver.log" 2>&1; then
  fail_measurement "layerB on"
fi

echo "all done $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$LOG_DIR/ALL_DONE.marker"
