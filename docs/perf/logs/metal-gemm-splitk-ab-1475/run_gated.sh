#!/bin/sh
# split-K vs classic 経路の M4 Max 実機 A/B（イシュー #1475）1 プロセス起動分の
# 実行ラッパー。負荷ゲート（`--max-load-avg`）自体は example 内蔵の
# `bench_harness::env_guard`（バックオフ再試行込み）に委ね、本スクリプトは
# 実行前後の `uptime`／`pmset -g therm` 記録と `runN.log` へのリダイレクトのみを
# 担当する（ゲートの二重実装を避ける。`docs/perf/logs/
# metal-gemm-splitk-shapes-1308/run_gated.sh` と同型の役割分担）。
#
# 使い方: ./run_gated.sh <run番号（1以上の整数）>
#
# セキュリティ（`.claude/rules/security.md` A03）: 引数は run 番号のみを
# 受け取り、数値以外（シェルメタ文字を含む値）は起動前に拒否する。
set -eu

RUN_NO="${1:-}"
case "$RUN_NO" in
    ''|*[!0-9]*)
        echo "usage: $0 <run番号（正の整数）>" >&2
        exit 1
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

uptime > "$SCRIPT_DIR/uptime_before_run${RUN_NO}.txt"
pmset -g therm > "$SCRIPT_DIR/pmset_therm_before_run${RUN_NO}.txt" 2>&1 || true

cd "$REPO_ROOT"
cargo run -p fandhe-ai-backend-metal --release --features internal-diagnostics \
    --example gemm_splitk_ab_bench -- --max-load-avg=4.0 \
    > "$SCRIPT_DIR/run${RUN_NO}.log" 2>&1

pmset -g therm > "$SCRIPT_DIR/pmset_therm_after_run${RUN_NO}.txt" 2>&1 || true
echo "run${RUN_NO} 完了: $SCRIPT_DIR/run${RUN_NO}.log"
