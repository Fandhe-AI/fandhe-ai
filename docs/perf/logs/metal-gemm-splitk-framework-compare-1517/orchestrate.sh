#!/bin/sh
# split-K 結線前後 framework-compare A/B（イシュー #1517・ルート #1509
# のユーザー指示: 専有ゲートは受け入れ条件にしない）の実行ラッパー。
# `docs/perf/logs/metal-gemm-splitk-ab-5run-1515/orchestrate.sh` と同型の
# 役割（バックグラウンド `uptime` サンプラー・watchlist プロセス件数記録・
# 二重起動防止ロック・`--dry-run`）だが、本イシューの実体は
# `run_ab_splitk_metal.sh <label>` を 1 回呼ぶだけ（5 round は同スクリプト
# が内部でループする）ため、run 番号引数は取らない（label のみ）。
#
# 使い方: ./orchestrate.sh <label> [--dry-run]
#         DRY_RUN=1 ./orchestrate.sh <label>
#   環境変数 AB_BEFORE_FACADE_PATH・AB_AFTER_FACADE_PATH は
#   `run_ab_splitk_metal.sh` と同じ契約で必須（呼び出し元が設定する）。
#
# セキュリティ（`.claude/rules/security.md` A03）: 引数は label
# （`[A-Za-z0-9._-]+`）のみを受け取り、それ以外は起動前に拒否する
# （`run_ab_splitk_metal.sh` 自身の allowlist 検証と二重になるが、本
# ラッパーが構築するログファイル名にも label を埋め込むため個別に検証
# する）。並走プロセスの記録はプロセス名の一致件数のみとし、コマンド
# ライン全文・絶対パスは記録しない（内部情報の非混入）。
set -eu

LABEL="${1:-}"
case "$LABEL" in
    '')
        echo "usage: $0 <label> [--dry-run]" >&2
        exit 1
        ;;
    *[!A-Za-z0-9._-]*)
        echo "label は英数字・'._-' のみ許可する（got: $LABEL）" >&2
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
FC_DIR="$(cd "$SCRIPT_DIR/../../../../scripts/bench/framework-compare" && pwd)"

UPTIME_BEFORE="$SCRIPT_DIR/uptime_before_${LABEL}.txt"
PMSET_BEFORE="$SCRIPT_DIR/pmset_therm_before_${LABEL}.txt"
PMSET_AFTER="$SCRIPT_DIR/pmset_therm_after_${LABEL}.txt"
RUN_LOG="$SCRIPT_DIR/run_${LABEL}.log"
MONITOR_LOG="$SCRIPT_DIR/monitor_${LABEL}.log"
PROCS_LOG="$SCRIPT_DIR/procs_${LABEL}.txt"
ENV_INFO="$SCRIPT_DIR/env_info.txt"

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

if [ "$DRY_RUN" = "1" ]; then
    echo "[dry-run] label=$LABEL"
    echo "[dry-run]   uptime_before -> $UPTIME_BEFORE"
    echo "[dry-run]   pmset_before  -> $PMSET_BEFORE"
    echo "[dry-run]   monitor_log   -> $MONITOR_LOG（バックグラウンド uptime サンプラー）"
    echo "[dry-run]   procs_log     -> $PROCS_LOG（watchlist: $WATCHLIST）"
    echo "[dry-run]   command       -> bash \"$FC_DIR/run_ab_splitk_metal.sh\" \"$LABEL\""
    echo "[dry-run]   run_log       -> $RUN_LOG"
    echo "[dry-run]   pmset_after   -> $PMSET_AFTER"
    exit 0
fi

# 既存成果物の確認（run の差し替え禁止。`docs/perf/metal-gemm-splitk-ab.md`
# §10.2 と同方針）。
for artifact in "$UPTIME_BEFORE" "$PMSET_BEFORE" "$PMSET_AFTER" "$RUN_LOG" "$MONITOR_LOG" "$PROCS_LOG"; do
    if [ -e "$artifact" ]; then
        echo "label=${LABEL}: 既存の成果物が見つかった（$artifact）。" \
             "同 label の再実行は成果物を手動で別名へ退避してから行う" \
             "（run の差し替え禁止）" >&2
        exit 1
    fi
done

# ロックは label 単位ではなく framework-compare ディレクトリ（$FC_DIR）
# 単位で取る。`run_ab_splitk_metal.sh` は Cargo.lock・path patch・
# before/after バイナリ等 $FC_DIR 配下の共有ファイルを書き換えるため、
# label が異なっていても同時実行すると互いの成果物を破壊しうる
# （codex-review 指摘。イシュー #1517 PR #1531）。ロックファイルは
# $FC_DIR 側に置き、label をまたいで単一のロックを共有する。
LOCK_DIR="$FC_DIR/.splitk-framework-compare-1517.lock"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "label=${LABEL}: framework-compare ディレクトリの実行が既に" \
         "進行中（ロック $LOCK_DIR が存在する。他 label を含め同時実行" \
         "しない）" >&2
    exit 1
fi

SAMPLER_PID=""
cleanup() {
    if [ -n "$SAMPLER_PID" ]; then
        kill "$SAMPLER_PID" 2>/dev/null || true
        wait "$SAMPLER_PID" 2>/dev/null || true
    fi
    rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap cleanup EXIT

uptime > "$UPTIME_BEFORE"
pmset -g therm > "$PMSET_BEFORE" 2>&1 || true
record_procs "$PROCS_LOG"

: > "$MONITOR_LOG"
(
    while true; do
        TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
        LOADS=$(uptime | sed -E 's/.*load averages?: *([0-9.]+)[, ]+([0-9.]+)[, ]+([0-9.]+).*/\1 \2 \3/')
        # shellcheck disable=SC2086
        set -- $LOADS
        L1="${1:-NA}"
        L5="${2:-NA}"
        L15="${3:-NA}"
        echo "${TS} load1=${L1} load5=${L5} load15=${L15}" >> "$MONITOR_LOG"
        sleep 10
    done
) &
SAMPLER_PID=$!

cd "$FC_DIR"
bash "$FC_DIR/run_ab_splitk_metal.sh" "$LABEL" > "$RUN_LOG" 2>&1 || {
    STATUS=$?
    echo "label=${LABEL}: run_ab_splitk_metal.sh が非ゼロ終了（status=${STATUS}）。$RUN_LOG を確認する" >&2
    exit "$STATUS"
}

kill "$SAMPLER_PID" 2>/dev/null || true
wait "$SAMPLER_PID" 2>/dev/null || true
SAMPLER_PID=""

pmset -g therm > "$PMSET_AFTER" 2>&1 || true

{
    echo "label=${LABEL} completed at $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
} >> "$ENV_INFO"

echo "label=${LABEL} 完了: $RUN_LOG（監視ログ: $MONITOR_LOG）"
