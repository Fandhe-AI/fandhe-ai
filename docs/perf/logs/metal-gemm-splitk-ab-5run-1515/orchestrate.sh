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
        echo "# 固定 watchlist（${WATCHLIST}）に一致するプロセス名の件数のみを記録する"
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
    echo "[dry-run]   monitor_log   -> ${MONITOR_LOG}（バックグラウンド uptime サンプラー）"
    echo "[dry-run]   procs_log     -> ${PROCS_LOG}（watchlist: ${WATCHLIST}）"
    echo "[dry-run]   command       -> $CMD"
    echo "[dry-run]   run_log       -> $RUN_LOG"
    echo "[dry-run]   pmset_after   -> $PMSET_AFTER"
    exit 0
fi

# 既存成果物の確認（イシュー #1529 codex-review P2 指摘: 同じ run 番号の
# 再実行によるログ上書き防止）。README の 5 回ループを途中中断後に先頭
# から再実行すると、確認なしに `>` で切り詰めて書き始めてしまうと完了済
# み run の測定値・中断の証跡が失われ、`docs/perf/metal-gemm-splitk-ab.md`
# §10.2 の「run の差し替え禁止」と整合しない。計測開始前（何も書き込む前）
# に当該 run 番号の成果物の有無を確認し、1 つでも既存なら何も書かず・
# 削除せず非ゼロ終了する（fail-closed）。同番号の再実行は成果物を手動で
# 別名へ退避してから行う。
for artifact in "$UPTIME_BEFORE" "$PMSET_BEFORE" "$PMSET_AFTER" "$RUN_LOG" "$MONITOR_LOG" "$PROCS_LOG"; do
    if [ -e "$artifact" ]; then
        echo "run${RUN_NO}: 既存の成果物が見つかった（${artifact}）。" \
             "同番号の再実行は成果物を手動で別名へ退避してから行う" \
             "（run の差し替え禁止。docs/perf/metal-gemm-splitk-ab.md §10.2）" >&2
        exit 1
    fi
done

# 同番号の同時実行を排他的に拒否する。`mkdir` は POSIX で単一のアトミック
# 排他作成操作であり、既にロックディレクトリが存在すれば非ゼロ終了する
# 性質を利用する（`set -C` の `>` 作成は run 途中の `>>` 追記との併用が
# ある本スクリプトでは扱いにくいため、専用ロックディレクトリ方式を採る）。
# ロック自体は正常・異常いずれの終了でも trap で解放するが、計測成果物
# （上記の既存確認対象）は解放対象に含めない（残す）。
LOCK_DIR="$SCRIPT_DIR/run${RUN_NO}.lock"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "run${RUN_NO}: 同番号の実行が既に進行中（ロック $LOCK_DIR が存在する）" >&2
    exit 1
fi

# バックグラウンド `uptime` サンプラーの PID とロックの解放をまとめて
# 扱う cleanup（EXIT trap）。`kill`／`rmdir` はいずれも `|| true` で
# 冪等化してあるため、成功・失敗いずれの終了経路でも安全に複数回呼び
# 出せる（以降のコード側で個別に `trap - EXIT` を呼んで無効化する必要
# がなくなり、ロック解放漏れを防ぐ）。
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
# 個別の `trap 'kill ...' EXIT` は張らない（既に `cleanup`〈ロック解放込み〉
# を trap 済みであり、シェルの trap はスロット 1 つのため再設定すると
# ロック解放が上書きされて消えてしまう。`cleanup` は `SAMPLER_PID` を
# 都度参照するため、ここでの代入のみで両方が正しく解放される）。

cd "$REPO_ROOT"
# shellcheck disable=SC2086 # CMD は固定リテラル（外部入力を含まない）で
# あり、意図的な単語分割によってサブコマンドへ複数引数を渡す
# （`.claude/rules/security.md` A03: 展開対象はスクリプト内で構築した
# 固定文字列のみで、利用者入力や環境変数由来の値を含まない）。
$CMD > "$RUN_LOG" 2>&1 || {
    STATUS=$?
    echo "run${RUN_NO}: cargo run が非ゼロ終了（status=${STATUS}）。$RUN_LOG を確認する" >&2
    exit "$STATUS"
}

kill "$SAMPLER_PID" 2>/dev/null || true
wait "$SAMPLER_PID" 2>/dev/null || true
SAMPLER_PID=""

pmset -g therm > "$PMSET_AFTER" 2>&1 || true

{
    echo "run${RUN_NO} completed at $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
} >> "$ENV_INFO"

echo "run${RUN_NO} 完了: ${RUN_LOG}（監視ログ: ${MONITOR_LOG}）"
