#!/bin/bash
# Apple M4 Max 側オーケストレーション（イシュー #1481・このセッションの
# ホスト自身でローカル実行）。
# (i) 専有ゲート（1 分 load average < 6.0 を 30 秒間隔で 2 回連続・
#     最大 30 試行。不成立なら GATE_NOT_PASSED.marker を書いて終了し、
#     ユーザー承認事項どおり verdict=undetermined を 1 回だけ記録する
#     判断は呼び出し側〈本エージェント〉に委ねる。待ち続けない）
# (ii) Layer A（off=実装 worktree → on=スクラッチ複製ツリーの順）
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
OFF_TREE="/Users/nancy/fandhe/library/rust-ai-library/.claude/worktrees/wf_dbb339eb-c60-13"
ON_TREE="/private/tmp/claude-501/-Users-nancy-fandhe-library-rust-ai-library/bac57b76-f1a4-4186-aea8-8f5e06b5dc10/scratchpad/fc-1481-on"

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
