#!/bin/sh
# split-K vs classic 経路の A/B 5 run 正式確定（イシュー #1515・
# ルート #1509 のユーザー指示: 専有ゲートは受け入れ条件にしない）1 プロセス
# 起動分の実行ラッパー。`docs/perf/logs/metal-gemm-splitk-ab-1475/
# run_gated.sh`（負荷ゲート版）と異なり、本スクリプトは専有ゲートを実施
# しない（`gemm_splitk_ab_bench` 自身も `--max-load-avg` を渡さないため
# record_only 運用で動作する）。代わりに計測前後の `uptime`／
# `pmset -g therm` に加え、計測中の負荷推移をバックグラウンド `uptime`
# サンプラーで記録し、共有負荷下であったことを事後に確認できるようにする
# （受け入れ条件「計測中の load average 推移・並走プロセスの有無を
# env_info に記録する」の機械化）。
#
# 使い方: ./orchestrate.sh <run番号（1〜5 の整数）> [--dry-run]
#         DRY_RUN=1 ./orchestrate.sh <run番号>
#
# --dry-run（または環境変数 DRY_RUN=1）指定時は cargo を起動せず、実行
# コマンド列と出力先パスのみを表示する（Linux 上で流れを検証するため。
# Apple Silicon 実機なしでもオーケストレーション自体の構文・分岐を確認
# できる）。
#
# セキュリティ（`.claude/rules/security.md` A03）: 引数は run 番号
# （正の整数）のみを受け取り、それ以外（シェルメタ文字を含む値）は
# 起動前に拒否する。並走プロセスの記録はプロセス名の一致件数のみとし、
# コマンドライン全文・絶対パスは記録しない（内部情報の非混入）。
set -eu

RUN_NO="${1:-}"
case "$RUN_NO" in
    ''|*[!0-9]*)
        echo "usage: $0 <run番号（1〜5 の正の整数）> [--dry-run]" >&2
        exit 1
        ;;
esac
case "$RUN_NO" in
    0)
        echo "run番号は 1 以上を指定する（0 は不可）" >&2
        exit 1
        ;;
esac

DRY_RUN="${DRY_RUN:-0}"
shift || true
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        *)
            echo "未知の引数: '$arg'（既知の引数: --dry-run）" >&2
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

UPTIME_BEFORE="$SCRIPT_DIR/uptime_before_run${RUN_NO}.txt"
PMSET_BEFORE="$SCRIPT_DIR/pmset_therm_before_run${RUN_NO}.txt"
PMSET_AFTER="$SCRIPT_DIR/pmset_therm_after_run${RUN_NO}.txt"
RUN_LOG="$SCRIPT_DIR/run${RUN_NO}.log"
MONITOR_LOG="$SCRIPT_DIR/run${RUN_NO}_monitor.log"
PROCS_LOG="$SCRIPT_DIR/run${RUN_NO}_procs.txt"
ENV_INFO="$SCRIPT_DIR/env_info.txt"

# 並走プロセスの検出対象（固定 watchlist。コマンドライン全文・パスは
# 記録せず、プロセス名の一致件数のみを記録する。`.claude/rules/
# security.md` A01 対応）。
WATCHLIST="python torch mlx cargo gemm_ bench"

record_procs() {
    out="$1"
    {
        echo "# 固定 watchlist（$WATCHLIST）に一致するプロセス名の件数のみを記録する"
        echo "# （コマンドライン全文・絶対パスは含めない）。"
        for name in $WATCHLIST; do
            count=$(ps -axo comm= 2>/dev/null | grep -ic -- "$name" || true)
            echo "watchlist_proc name=${name} count=${count}"
        done
    } > "$out"
}

CMD="cargo run -p fandhe-ai-backend-metal --release --features internal-diagnostics --example gemm_splitk_ab_bench"

if [ "$DRY_RUN" = "1" ]; then
    echo "[dry-run] run${RUN_NO}"
    echo "[dry-run]   uptime_before -> $UPTIME_BEFORE"
    echo "[dry-run]   pmset_before  -> $PMSET_BEFORE"
    echo "[dry-run]   monitor_log   -> $MONITOR_LOG（バックグラウンド uptime サンプラー）"
    echo "[dry-run]   procs_log     -> $PROCS_LOG（watchlist: $WATCHLIST）"
    echo "[dry-run]   command       -> $CMD"
    echo "[dry-run]   run_log       -> $RUN_LOG"
    echo "[dry-run]   pmset_after   -> $PMSET_AFTER"
    exit 0
fi

uptime > "$UPTIME_BEFORE"
pmset -g therm > "$PMSET_BEFORE" 2>&1 || true
record_procs "$PROCS_LOG"

# バックグラウンド `uptime` サンプラー（10 秒間隔。
# `scripts/bench/framework-compare/run_ab_readout_metal.sh` の
# `UPTIME_SAMPLER_LOG` と同型の役割だが、`aggregate.py::parse_monitor_log`
# が解析できる `<UTC時刻> load1=<x> load5=<y> load15=<z>` 形式へ整形する）。
# macOS の `uptime` 出力（`load averages: X Y Z`）から抽出する（sed の
# 正規表現。GNU/BSD いずれの `sed -E` でも動作する範囲の表現に限定）。
: > "$MONITOR_LOG"
(
    while true; do
        TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
        LOADS=$(uptime | sed -E 's/.*load averages?: *([0-9.]+)[, ]+([0-9.]+)[, ]+([0-9.]+).*/\1 \2 \3/')
        # shellcheck disable=SC2086 # LOADS は sed 抽出後の数値 3 個
        # （またはマッチ失敗時の uptime 生出力）のみを含み、意図的な
        # 単語分割で 3 つの位置パラメータへ分解する。
        set -- $LOADS
        L1="${1:-NA}"
        L5="${2:-NA}"
        L15="${3:-NA}"
        echo "${TS} load1=${L1} load5=${L5} load15=${L15}" >> "$MONITOR_LOG"
        sleep 10
    done
) &
SAMPLER_PID=$!
trap 'kill "$SAMPLER_PID" 2>/dev/null || true' EXIT

cd "$REPO_ROOT"
# shellcheck disable=SC2086 # CMD は固定リテラル（外部入力を含まない）で
# あり、意図的な単語分割によってサブコマンドへ複数引数を渡す
# （`.claude/rules/security.md` A03: 展開対象はスクリプト内で構築した
# 固定文字列のみで、利用者入力や環境変数由来の値を含まない）。
$CMD > "$RUN_LOG" 2>&1 || {
    STATUS=$?
    echo "run${RUN_NO}: cargo run が非ゼロ終了（status=${STATUS}）。$RUN_LOG を確認する" >&2
    kill "$SAMPLER_PID" 2>/dev/null || true
    trap - EXIT
    exit "$STATUS"
}

kill "$SAMPLER_PID" 2>/dev/null || true
wait "$SAMPLER_PID" 2>/dev/null || true
trap - EXIT

pmset -g therm > "$PMSET_AFTER" 2>&1 || true

{
    echo "run${RUN_NO} completed at $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
} >> "$ENV_INFO"

echo "run${RUN_NO} 完了: $RUN_LOG（監視ログ: $MONITOR_LOG）"
