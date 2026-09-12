#!/bin/sh
# `d_input` 経路の同期境界・回収余地の定量化（イシュー #1562）の M4 Max
# 実機実測オーケストレーション。方針 A（`mnist_scale_train_reuse_metal_
# backward_dinput_phase`）・方針 B（`resident_lhs_dinput_phase_bench`。
# 通常版・`internal-diagnostics` 版の 2 回）の 3 テストを順に実行する。
# 各 `cargo test` 呼び出しには `--test-threads=1` を付ける（イシュー #1562
# codex-review 是正）: `resident_lhs_dinput_phase_bench.rs` は L1・L2 用に
# 独立の `MetalContext::new()` を持つ 2 テストを含み、既定のマルチ
# スレッド並列実行では同一物理 GPU を同時に奪い合い、互いの
# `synchronize()` 計測へ資源競合が混入しうる。3 コマンドとも直列化する
# ことで統一する。
# 引数は `--dry-run` のみを受け付ける（それ以外の外部入力を展開しない。
# `.claude/rules/security.md` A03）。
#
# 使い方（このディレクトリで実行する想定。worktree のパスは実行環境に
# 合わせる）:
#   sh orchestrate.sh            # 実行する
#   sh orchestrate.sh --dry-run  # 実行するコマンド列を表示するのみ
#
# 生成物: backward_phase.log・resident_lhs_phase.log・
# resident_lhs_phase_gpu_timestamps.log・env_info.txt・uptime_before.txt・
# uptime_after.txt（内部ホスト名は含めない。README.md の事前登録判定規則に
# 従い docs/backend-metal-command-batching-design.md §7.3.4 へ転記する）。

set -eu

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=1
fi

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../../../.." && pwd)

CMD_BACKWARD_PHASE="cargo test -p fandhe-ai --release --test mnist_scale_train_reuse_bench -- --ignored --nocapture --test-threads=1 mnist_scale_train_reuse_metal_backward_dinput_phase"
CMD_RESIDENT_LHS_PHASE="cargo test -p fandhe-ai-backend-metal --release --test resident_lhs_dinput_phase_bench -- --ignored --nocapture --test-threads=1"
CMD_RESIDENT_LHS_PHASE_GPU_TS="cargo test -p fandhe-ai-backend-metal --release --features internal-diagnostics --test resident_lhs_dinput_phase_bench -- --ignored --nocapture --test-threads=1"

if [ "$DRY_RUN" -eq 1 ]; then
    echo "=== dry-run: 実行するコマンド列（repo root: $REPO_ROOT） ==="
    echo "$CMD_BACKWARD_PHASE"
    echo "$CMD_RESIDENT_LHS_PHASE"
    echo "$CMD_RESIDENT_LHS_PHASE_GPU_TS"
    exit 0
fi

echo "=== env_info ==="
{
    uname -srm
    sw_vers
    rustc -V
    sysctl machdep.cpu.brand_string
    git -C "$REPO_ROOT" rev-parse HEAD
} >"$SCRIPT_DIR/env_info.txt" 2>&1
cat "$SCRIPT_DIR/env_info.txt"

uptime >"$SCRIPT_DIR/uptime_before.txt"

cd "$REPO_ROOT"

echo "=== 方針 A: backward 限定カウンタ・壁時間 ==="
# shellcheck disable=SC2086
eval "$CMD_BACKWARD_PHASE" >"$SCRIPT_DIR/backward_phase.log" 2>&1
tail -20 "$SCRIPT_DIR/backward_phase.log"

echo "=== 方針 B: d_input GEMM 単体隔離フェーズ分解 ==="
# shellcheck disable=SC2086
eval "$CMD_RESIDENT_LHS_PHASE" >"$SCRIPT_DIR/resident_lhs_phase.log" 2>&1
tail -20 "$SCRIPT_DIR/resident_lhs_phase.log"

echo "=== 方針 B（GPU タイムスタンプ内訳）==="
# shellcheck disable=SC2086
eval "$CMD_RESIDENT_LHS_PHASE_GPU_TS" >"$SCRIPT_DIR/resident_lhs_phase_gpu_timestamps.log" 2>&1
tail -20 "$SCRIPT_DIR/resident_lhs_phase_gpu_timestamps.log"

uptime >"$SCRIPT_DIR/uptime_after.txt"

echo DONE
