#!/bin/sh
# `d_input` 経路の同期境界回収の実装（イシュー #1563）を検証する M4 Max
# 実機実測オーケストレーション。before 腕（origin/main。#1563 適用前）・
# after 腕（本ブランチ）をそれぞれ別 worktree に展開した前提で、以下を
# 順に実行する:
#
#   1. bit 同一確認（両腕で `metal_reuse_step_grad_bit_dump` を実行し、
#      `^(step\[|final\.param\[)` 行を diff。差分 0 を期待）
#   2. `#[ignore]` 群の非後退確認（after 腕で実行）
#   3. カウンタ実測（after 腕で `mnist_scale_train_reuse_metal_batch_
#      counters`〈hard assert 11/8/8〉・`mnist_scale_train_reuse_metal_
#      backward_dinput_phase`〈record-only 仮説 5/3/3〉）
#   4. before 腕で #1562 の `docs/perf/logs/metal-dinput-sync-1562/
#      orchestrate.sh` を実行し同イシューの記入欄を埋める（追加コスト
#      なし。本オーケストレーションはこのステップを呼び出すのみで
#      1562 側のファイルは変更しない）
#   5. A/B（`scripts/bench/framework-compare/run_ab_dinput_sync_metal.sh`。
#      5 round・record_only）
#
# 引数は `--dry-run` のみを受け付ける（それ以外の外部入力を展開しない。
# `.claude/rules/security.md` A03）。before/after 腕のパスは環境変数
# `DINPUT_SYNC_BEFORE_REPO`／`DINPUT_SYNC_AFTER_REPO`（絶対パス。それぞれ
# の worktree のリポジトリルート）で指定する。
#
# 使い方（このディレクトリで実行する想定）:
#   DINPUT_SYNC_BEFORE_REPO=/abs/path/to/before-worktree \
#   DINPUT_SYNC_AFTER_REPO=/abs/path/to/after-worktree \
#     sh orchestrate.sh            # 実行する
#   sh orchestrate.sh --dry-run    # 実行するコマンド列を表示するのみ
#     （--dry-run はリポジトリパス環境変数なしでも動作する）
#
# 生成物: bitdump_before.log・bitdump_after.log・bitdump_diff.txt・
# ignored_after.log・batch_counters_after.log・backward_phase_after.log・
# ab/（`run_ab_dinput_sync_metal.sh` の出力一式。同スクリプトが
# `scripts/bench/framework-compare/results/raw/` 配下へ書くため、実行後に
# 手動でこのディレクトリの `ab/` へコピーする運用は README を参照）・
# env_info_before.txt・env_info_after.txt・uptime_before.txt・
# uptime_after.txt（内部ホスト名は含めない）。

set -eu

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=1
fi

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

BEFORE_REPO="${DINPUT_SYNC_BEFORE_REPO:-}"
AFTER_REPO="${DINPUT_SYNC_AFTER_REPO:-}"

BIT_DUMP_CMD="cargo test -p fandhe-ai --release --test metal_reuse_step_grad_bit_dump -- --ignored --nocapture"
IGNORED_CMD_MNIST="cargo test -p fandhe-ai --release --test mnist_scale_train_reuse_bench -- --ignored --nocapture --test-threads=1"
IGNORED_CMD_GEMM_PARITY="cargo test -p fandhe-ai-backend-metal --release --test gemm_resident_parity -- --ignored --nocapture"
IGNORED_CMD_STORE_PARITY="cargo test -p fandhe-ai --release --test device_param_store_backend_parity -- --ignored --nocapture"
IGNORED_CMD_COMMAND_BATCHING="cargo test -p fandhe-ai-backend-metal --release --test command_batching_bench -- --ignored --nocapture"
BATCH_COUNTERS_CMD="cargo test -p fandhe-ai --release --test mnist_scale_train_reuse_bench -- --ignored --nocapture --test-threads=1 mnist_scale_train_reuse_metal_batch_counters"
BACKWARD_PHASE_CMD="cargo test -p fandhe-ai --release --test mnist_scale_train_reuse_bench -- --ignored --nocapture --test-threads=1 mnist_scale_train_reuse_metal_backward_dinput_phase"

if [ "$DRY_RUN" -eq 1 ]; then
    echo "=== dry-run: 実行するコマンド列 ==="
    echo "[before repo] $BIT_DUMP_CMD"
    echo "[after repo]  $BIT_DUMP_CMD"
    echo "diff bitdump_before.log(filtered) bitdump_after.log(filtered)"
    echo "[after repo]  $IGNORED_CMD_MNIST"
    echo "[after repo]  $IGNORED_CMD_GEMM_PARITY"
    echo "[after repo]  $IGNORED_CMD_STORE_PARITY"
    echo "[after repo]  $IGNORED_CMD_COMMAND_BATCHING"
    echo "[after repo]  $BATCH_COUNTERS_CMD"
    echo "[after repo]  $BACKWARD_PHASE_CMD"
    echo "[before repo] sh docs/perf/logs/metal-dinput-sync-1562/orchestrate.sh"
    echo "[after repo]  AB_BEFORE_FACADE_PATH=\$BEFORE_REPO/crates/facade AB_AFTER_FACADE_PATH=\$AFTER_REPO/crates/facade bash scripts/bench/framework-compare/run_ab_dinput_sync_metal.sh 1563"
    exit 0
fi

if [ -z "$BEFORE_REPO" ] || [ -z "$AFTER_REPO" ]; then
    echo "error: DINPUT_SYNC_BEFORE_REPO / DINPUT_SYNC_AFTER_REPO must be set (absolute paths)" >&2
    exit 1
fi
case "$BEFORE_REPO" in
    /*) ;;
    *) echo "error: DINPUT_SYNC_BEFORE_REPO must be an absolute path" >&2; exit 1 ;;
esac
case "$AFTER_REPO" in
    /*) ;;
    *) echo "error: DINPUT_SYNC_AFTER_REPO must be an absolute path" >&2; exit 1 ;;
esac

echo "=== env_info (before) ==="
{
    uname -srm
    sw_vers
    rustc -V
    sysctl machdep.cpu.brand_string
    git -C "$BEFORE_REPO" rev-parse HEAD
} >"$SCRIPT_DIR/env_info_before.txt" 2>&1
cat "$SCRIPT_DIR/env_info_before.txt"

echo "=== env_info (after) ==="
{
    uname -srm
    sw_vers
    rustc -V
    sysctl machdep.cpu.brand_string
    git -C "$AFTER_REPO" rev-parse HEAD
} >"$SCRIPT_DIR/env_info_after.txt" 2>&1
cat "$SCRIPT_DIR/env_info_after.txt"

uptime >"$SCRIPT_DIR/uptime_before.txt"

echo "=== 1. bit 同一確認（before 腕） ==="
(cd "$BEFORE_REPO" && eval "$BIT_DUMP_CMD") >"$SCRIPT_DIR/bitdump_before.log" 2>&1
tail -5 "$SCRIPT_DIR/bitdump_before.log"

echo "=== 1. bit 同一確認（after 腕） ==="
(cd "$AFTER_REPO" && eval "$BIT_DUMP_CMD") >"$SCRIPT_DIR/bitdump_after.log" 2>&1
tail -5 "$SCRIPT_DIR/bitdump_after.log"

grep -E '^(step\[|final\.param\[)' "$SCRIPT_DIR/bitdump_before.log" >"$SCRIPT_DIR/bitdump_before_filtered.txt" || true
grep -E '^(step\[|final\.param\[)' "$SCRIPT_DIR/bitdump_after.log" >"$SCRIPT_DIR/bitdump_after_filtered.txt" || true
diff -u "$SCRIPT_DIR/bitdump_before_filtered.txt" "$SCRIPT_DIR/bitdump_after_filtered.txt" \
    >"$SCRIPT_DIR/bitdump_diff.txt" 2>&1 || true
if [ -s "$SCRIPT_DIR/bitdump_diff.txt" ]; then
    echo "WARNING: bitdump_diff.txt is non-empty (bit mismatch detected). See $SCRIPT_DIR/bitdump_diff.txt" >&2
else
    echo "bit dump: 0 diff lines (bit-identical)"
fi

echo "=== 2. #[ignore] 群非後退確認（after 腕） ==="
(cd "$AFTER_REPO" && eval "$IGNORED_CMD_MNIST") >"$SCRIPT_DIR/ignored_after_mnist.log" 2>&1
tail -20 "$SCRIPT_DIR/ignored_after_mnist.log"
(cd "$AFTER_REPO" && eval "$IGNORED_CMD_GEMM_PARITY") >"$SCRIPT_DIR/ignored_after_gemm_parity.log" 2>&1
tail -20 "$SCRIPT_DIR/ignored_after_gemm_parity.log"
(cd "$AFTER_REPO" && eval "$IGNORED_CMD_STORE_PARITY") >"$SCRIPT_DIR/ignored_after_store_parity.log" 2>&1
tail -20 "$SCRIPT_DIR/ignored_after_store_parity.log"
(cd "$AFTER_REPO" && eval "$IGNORED_CMD_COMMAND_BATCHING") >"$SCRIPT_DIR/ignored_after_command_batching.log" 2>&1
tail -20 "$SCRIPT_DIR/ignored_after_command_batching.log"

echo "=== 3. カウンタ実測（after 腕） ==="
(cd "$AFTER_REPO" && eval "$BATCH_COUNTERS_CMD") >"$SCRIPT_DIR/batch_counters_after.log" 2>&1
tail -20 "$SCRIPT_DIR/batch_counters_after.log"
(cd "$AFTER_REPO" && eval "$BACKWARD_PHASE_CMD") >"$SCRIPT_DIR/backward_phase_after.log" 2>&1
tail -20 "$SCRIPT_DIR/backward_phase_after.log"

echo "=== 3b. カウンタ実測（before 腕。再現確認用） ==="
(cd "$BEFORE_REPO" && eval "$BATCH_COUNTERS_CMD") >"$SCRIPT_DIR/batch_counters_before.log" 2>&1
tail -20 "$SCRIPT_DIR/batch_counters_before.log"
(cd "$BEFORE_REPO" && eval "$BACKWARD_PHASE_CMD") >"$SCRIPT_DIR/backward_phase_before.log" 2>&1
tail -20 "$SCRIPT_DIR/backward_phase_before.log"

echo "=== 4. #1562 記入欄を before 腕で埋める ==="
(cd "$BEFORE_REPO/docs/perf/logs/metal-dinput-sync-1562" && sh orchestrate.sh) \
    >"$SCRIPT_DIR/issue_1562_backfill.log" 2>&1 || \
    echo "WARNING: #1562 orchestrate.sh failed or partially failed; see $SCRIPT_DIR/issue_1562_backfill.log" >&2
tail -20 "$SCRIPT_DIR/issue_1562_backfill.log"

echo "=== 5. A/B（framework-compare train。5 round・record_only） ==="
mkdir -p "$SCRIPT_DIR/ab"
(
    cd "$AFTER_REPO/scripts/bench/framework-compare" && \
    AB_BEFORE_FACADE_PATH="$BEFORE_REPO/crates/facade" \
    AB_AFTER_FACADE_PATH="$AFTER_REPO/crates/facade" \
    bash run_ab_dinput_sync_metal.sh 1563
) >"$SCRIPT_DIR/ab/run_ab_1563.log" 2>&1 || \
    echo "WARNING: A/B script exited non-zero; see $SCRIPT_DIR/ab/run_ab_1563.log" >&2
tail -40 "$SCRIPT_DIR/ab/run_ab_1563.log"
# A/B 生成物（results/raw 配下・compare-train-1563*.md 等）はコピーせず
# `$AFTER_REPO/scripts/bench/framework-compare/` 配下に残す（README
# 「生成物の所在」参照）。

uptime >"$SCRIPT_DIR/uptime_after.txt"

echo DONE
