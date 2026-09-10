#!/bin/sh
# 公開入口の数値契約ゲート解除（イシュー #1513）の M4 Max 実機確認オーケストレーション。
# `SPLIT_K_NUMERIC_CONTRACT_APPROVED` を `true` へ切替後、`crates/backend-metal/
# tests/gemm_splitk_auto_entry_parity.rs` の `#[ignore]` テスト 2 件（自動判定
# 入口 `dispatch_split_k_strided_prepared` 自体の到達確認・非対象形状の
# フォールバック確認）と、既存 `gemm_splitk_bit_match.rs`・`gemm_splitk_parity.rs`
# の非後退を確認する。引数は取らない（外部入力を展開しない。`.claude/rules/
# security.md` A03）。
#
# 使い方（このディレクトリで実行する想定。worktree のパスは実行環境に合わせる）:
#   sh run_auto_entry.sh
#
# 生成物: auto_entry.log・bit_match.log・parity.log・env_info.txt・
# uptime_before.txt・uptime_after.txt・uptime_during.log（内部ホスト名は
# 含めない。`docs/perf/metal-gemm-splitk-two-pass.md` §5.9 の事前登録判定
# 規則に従い判定する）。

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

echo "=== env_info ==="
{
    uname -srm
    sw_vers
    rustc -V
    sysctl machdep.cpu.brand_string
    git -C "$SCRIPT_DIR/../../../.." rev-parse HEAD
} >"$SCRIPT_DIR/env_info.txt" 2>&1
cat "$SCRIPT_DIR/env_info.txt"

uptime >"$SCRIPT_DIR/uptime_before.txt"

# バックグラウンドで load average 推移を 10 秒間隔で記録する（実行中の負荷が
# 判定に影響していないかを事後確認するため。専有ゲートは不要 — parity・
# 到達確認テストは負荷非依存の事前登録判定規則。#1512 の `run_parity.sh`
# と同一方針）。
: >"$SCRIPT_DIR/uptime_during.log"
(
    while true; do
        uptime >>"$SCRIPT_DIR/uptime_during.log"
        sleep 10
    done
) &
SAMPLER_PID=$!
trap 'kill "$SAMPLER_PID" 2>/dev/null || true' EXIT

cd "$SCRIPT_DIR/../../../.."

# (1) 自動判定入口自体の受け入れテスト（本イシューの主対象。`required-features`
# なし・既定ビルドで到達可能）。
cargo test -p fandhe-ai-backend-metal --release \
    --test gemm_splitk_auto_entry_parity -- --ignored --nocapture \
    >"$SCRIPT_DIR/auto_entry.log" 2>&1

# (2) 既存 AC-1（run-to-run bit 同一）非後退確認。
cargo test -p fandhe-ai-backend-metal --release --features internal-diagnostics \
    --test gemm_splitk_bit_match -- --ignored --nocapture \
    >"$SCRIPT_DIR/bit_match.log" 2>&1

# (3) 既存 AC-2（split-K 経路自体の正しさ。`_with_plan` 版）非後退確認。
cargo test -p fandhe-ai-backend-metal --release --features internal-diagnostics \
    --test gemm_splitk_parity -- --ignored --nocapture \
    >"$SCRIPT_DIR/parity.log" 2>&1

uptime >"$SCRIPT_DIR/uptime_after.txt"

echo "=== auto_entry.log tail ==="
tail -20 "$SCRIPT_DIR/auto_entry.log"
echo "=== bit_match.log tail ==="
tail -10 "$SCRIPT_DIR/bit_match.log"
echo "=== parity.log tail ==="
tail -10 "$SCRIPT_DIR/parity.log"
echo DONE
