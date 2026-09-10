#!/bin/sh
# Metal f32 split-K parity 実測ベースライン非後退方式への切替（イシュー #1512）の
# M4 Max 実機確認オーケストレーション。`crates/backend-metal/tests/
# gemm_splitk_parity.rs` の `#[ignore]` テストを実行し、承認済みベースライン
# （`docs/backend-metal-splitk-parity-judgment-decision.md` §7）を上回らない
# ことを確認する。引数は取らない（外部入力を展開しない。`.claude/rules/
# security.md` A03）。
#
# 使い方（このディレクトリで実行する想定。worktree のパスは実行環境に合わせる）:
#   sh run_parity.sh
#
# 生成物: parity.log・env_info.txt・uptime_before.txt・uptime_after.txt・
# uptime_during.log（内部ホスト名は含めない。`docs/perf/metal-gemm-splitk-
# two-pass.md` §5.8 の事前登録判定規則に従い判定する）。

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
# 判定に影響していないかを事後確認するため。専有ゲートは不要 — parity は
# 負荷非依存の事前登録判定規則。`docs/perf/metal-gemm-splitk-two-pass.md`
# §5.8「共有負荷下で可（専有ゲート不要）」）。
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
cargo test -p fandhe-ai-backend-metal --release --features internal-diagnostics \
    --test gemm_splitk_parity -- --ignored --nocapture \
    >"$SCRIPT_DIR/parity.log" 2>&1

uptime >"$SCRIPT_DIR/uptime_after.txt"

echo "=== parity.log tail ==="
tail -20 "$SCRIPT_DIR/parity.log"
echo DONE
