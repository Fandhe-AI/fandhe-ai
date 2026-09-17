#!/bin/sh
# 協調ロードの threadgroup メモリ格納位置 XOR swizzle 軸（イシュー #1970。
# 機構実装は `crates/backend-metal/src/{tile.rs, gemm.rs, pipeline.rs,
# spec_source.rs, shaders/gemm.metal}`）の実機（Apple Silicon）実行ラッパー。
# `docs/perf/logs/metal-gemm-thread-elements-ab-1694/orchestrate.sh` を雛形に
# する。
#
#   ./orchestrate.sh gate [--dry-run]
#       AC-1（bit 一致）の前提ゲート: `crates/backend-metal/src/gemm.rs`
#       の `smem_swizzle_bit_match_*` 6 本 + 既存 `coop_load_bit_match_*`
#       6 本（非後退確認）を `--exact` で列挙して実行する（診断テスト
#       `xor_swizzle_kernel_gpu_ab_production_sizes` を巻き込まないよう
#       フルパス名を明示指定する）。1 件でも FAIL なら打ち切る（事前登録
#       判定規則 1）。ログは本ディレクトリ直下の `gate_run.log` へ保存する。
#
#   ./orchestrate.sh <run番号（1〜5 の整数）> [--dry-run]
#       性能 A/B（`gemm_smem_swizzle_diag_tests::
#       xor_swizzle_kernel_gpu_ab_production_sizes`）を 1 プロセス起動
#       する。record_only 運用（専有ゲートなし。ルート #1509 と同じ運用）。
#       計測前後の uptime／pmset・計測中の負荷推移サンプラー・並走
#       プロセス watchlist 件数を記録する。
#
# --dry-run（または環境変数 DRY_RUN=1）指定時は cargo を起動せず、実行
# コマンド列と出力先パスのみを表示する（Apple Silicon 実機なしでも
# オーケストレーション自体の構文・分岐を検証できる。本 PR は Linux
# 実装環境のため実測は未実施 — README 参照）。
#
# セキュリティ（`.claude/rules/security.md` A03）: 引数は「gate」または
# 「run 番号（正の整数）」のみを受け取り、それ以外（シェルメタ文字を
# 含む値）は起動前に拒否する。並走プロセスの記録はプロセス名の一致件数
# のみとし、コマンドライン全文・絶対パスは記録しない（内部情報の非混入）。
set -eu

MODE="${1:-}"
if [ -z "$MODE" ]; then
    echo "usage: $0 <gate|run番号（1〜5 の正の整数）> [--dry-run]" >&2
    exit 1
fi

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

# gate 対象テストのフルパス名（`--exact` で列挙。診断テスト
# `xor_swizzle_kernel_gpu_ab_production_sizes` を巻き込まない）。
GATE_TESTS="
gemm::tests::smem_swizzle_bit_match_all_candidates
gemm::tests::smem_swizzle_bit_match_dispatch_auto
gemm::tests::smem_swizzle_transposed_bit_match
gemm::tests::smem_swizzle_bit_match_boundary_shape
gemm::tests::smem_swizzle_f16_path_is_noop
gemm::tests::smem_swizzle_default_matches_production_constants
gemm::tests::coop_load_bit_match_all_candidates
gemm::tests::coop_load_bit_match_dispatch_auto
gemm::tests::coop_load_transposed_bit_match
gemm::tests::coop_load_bit_match_boundary_shape
gemm::tests::coop_load_f16_path_is_noop
gemm::tests::coop_load_default_matches_production_constants
"

run_gate() {
    GATE_LOG="$SCRIPT_DIR/gate_run.log"
    if [ "$DRY_RUN" = "1" ]; then
        echo "[dry-run] gate: 12 テストを --exact で列挙して実行する（--test-threads=1）"
        for t in $GATE_TESTS; do
            echo "[dry-run]   $t"
        done
        echo "[dry-run]   log -> $GATE_LOG"
        return 0
    fi

    if [ -e "$GATE_LOG" ]; then
        echo "gate: 既存の成果物が見つかった（${GATE_LOG}）。手動で別名へ退避してから再実行する" >&2
        exit 1
    fi

    cd "$REPO_ROOT"
    # 複数フィルタの同時指定は Rust 標準テストハーネス（libtest）が
    # Rust 1.59 以降サポートする機能（`cargo test -- name1 name2 ...`
    # は各フィルタの和集合にマッチする。`--exact` で完全一致に限定）。
    # shellcheck disable=SC2086 # GATE_TESTS は固定リテラル集合（外部入力を含まない）
    cargo test -p fandhe-ai-backend-metal --release --lib \
        -- --ignored --exact --test-threads=1 --nocapture \
        $GATE_TESTS \
        > "$GATE_LOG" 2>&1 || {
        STATUS=$?
        echo "gate: FAIL（status=${STATUS}）。${GATE_LOG} を確認する。gate FAIL のため性能 A/B は実施しない（機構契約の不成立として記録するのみ）" >&2
        exit "$STATUS"
    }
    echo "gate: 全 12 テスト成功した。性能 A/B（run1〜run5）へ進める"
}

run_ab() {
    RUN_NO="$1"
    UPTIME_BEFORE="$SCRIPT_DIR/uptime_before_run${RUN_NO}.txt"
    PMSET_BEFORE="$SCRIPT_DIR/pmset_therm_before_run${RUN_NO}.txt"
    PMSET_AFTER="$SCRIPT_DIR/pmset_therm_after_run${RUN_NO}.txt"
    RUN_LOG="$SCRIPT_DIR/kernel_gpu_run${RUN_NO}.log"
    MONITOR_LOG="$SCRIPT_DIR/run${RUN_NO}_monitor.log"
    PROCS_LOG="$SCRIPT_DIR/run${RUN_NO}_procs.txt"
    ENV_INFO="$SCRIPT_DIR/env_info.txt"

    CMD="cargo test -p fandhe-ai-backend-metal --release --lib gemm_smem_swizzle_diag_tests::xor_swizzle_kernel_gpu_ab_production_sizes -- --ignored --nocapture --test-threads=1"

    if [ "$DRY_RUN" = "1" ]; then
        echo "[dry-run] run${RUN_NO}"
        echo "[dry-run]   uptime_before -> $UPTIME_BEFORE"
        echo "[dry-run]   pmset_before  -> $PMSET_BEFORE"
        echo "[dry-run]   monitor_log   -> ${MONITOR_LOG}（バックグラウンド uptime サンプラー）"
        echo "[dry-run]   procs_log     -> ${PROCS_LOG}（watchlist: ${WATCHLIST}）"
        echo "[dry-run]   command       -> $CMD"
        echo "[dry-run]   run_log       -> $RUN_LOG"
        echo "[dry-run]   pmset_after   -> $PMSET_AFTER"
        return 0
    fi

    for artifact in "$UPTIME_BEFORE" "$PMSET_BEFORE" "$PMSET_AFTER" "$RUN_LOG" "$MONITOR_LOG" "$PROCS_LOG"; do
        if [ -e "$artifact" ]; then
            echo "run${RUN_NO}: 既存の成果物が見つかった（${artifact}）。" \
                 "同番号の再実行は成果物を手動で別名へ退避してから行う" >&2
            exit 1
        fi
    done

    LOCK_DIR="$SCRIPT_DIR/run${RUN_NO}.lock"
    if ! mkdir "$LOCK_DIR" 2>/dev/null; then
        echo "run${RUN_NO}: 同番号の実行が既に進行中（ロック $LOCK_DIR が存在する）" >&2
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

    cd "$REPO_ROOT"
    # shellcheck disable=SC2086
    $CMD > "$RUN_LOG" 2>&1 || {
        STATUS=$?
        echo "run${RUN_NO}: cargo test が非ゼロ終了（status=${STATUS}）。$RUN_LOG を確認する" >&2
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
}

case "$MODE" in
    gate)
        run_gate
        ;;
    ''|*[!0-9]*)
        echo "usage: $0 <gate|run番号（1〜5 の正の整数）> [--dry-run]" >&2
        exit 1
        ;;
    0)
        echo "run番号は 1 以上を指定する（0 は不可）" >&2
        exit 1
        ;;
    *)
        run_ab "$MODE"
        ;;
esac
