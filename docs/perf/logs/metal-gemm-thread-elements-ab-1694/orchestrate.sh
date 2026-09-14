#!/bin/sh
# thread_elements() 方式 BlockMMA 候補（イシュー #1693。本 A/B はイシュー
# #1694）の実機（Apple Silicon）実行ラッパー。`docs/perf/logs/
# metal-gemm-splitk-ab-5run-1515/orchestrate.sh` を雛形に、本イシュー向け
# に 2 サブモードへ整理する:
#
#   ./orchestrate.sh gate [--dry-run]
#       R0（probe）→ R1（parity・正しさ）→ R2（非 staged 拒否）→ R3
#       （本番との bit 一致）を順に実行し、最初に失敗した時点で打ち切る
#       （`docs/perf/metal-gemm-thread-elements-candidate.md` §4／イシュー
#       #1694 issue コメントの事前登録判定規則 1）。ログは
#       `../metal-gemm-thread-elements-1693/`（#1693 が予約したファイル名
#       `probe_run.log`／`parity_run.log`／`all_staged_candidates_run.log`／
#       `bit_match_run.log`）へ保存する。
#
#   ./orchestrate.sh <run番号（1〜5 の整数）> [--dry-run]
#       性能 A/B（`te_kernel_gpu_ab_vs_production_select`）を 1 プロセス
#       起動する。record_only 運用（専有ゲートなし。ルート #1509 の
#       ユーザー指示・#1515／#1538 と同じ運用）。計測前後の uptime／
#       pmset・計測中の負荷推移サンプラー・並走プロセス watchlist 件数を
#       記録する。
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
GATE_LOG_DIR="$(cd "$SCRIPT_DIR/../metal-gemm-thread-elements-1693" 2>/dev/null && pwd || echo "$SCRIPT_DIR/../metal-gemm-thread-elements-1693")"

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

run_gate() {
    if [ "$DRY_RUN" = "1" ]; then
        echo "[dry-run] gate: R0 -> R1 -> R2 -> R3（失敗したら打ち切り）"
        echo "[dry-run]   R0 (probe)                  -> $GATE_LOG_DIR/probe_run.log"
        echo "[dry-run]     cargo test -p fandhe-ai-backend-metal --release --lib gemm::tests::te_layout_probe_matches_model -- --ignored --nocapture --test-threads=1"
        echo "[dry-run]   R1 (parity, 外部テスト)      -> $GATE_LOG_DIR/parity_run.log"
        echo "[dry-run]     cargo test -p fandhe-ai-backend-metal --release --test gemm_te_parity -- --ignored --nocapture --test-threads=1"
        echo "[dry-run]   R1 (all_staged_candidates)   -> $GATE_LOG_DIR/all_staged_candidates_run.log"
        echo "[dry-run]     cargo test -p fandhe-ai-backend-metal --release --lib gemm::tests::all_staged_candidates_match_te_cpu_reference_512_nn -- --ignored --nocapture --test-threads=1"
        echo "[dry-run]   R2 (非 staged 拒否・R3 に含めて外部テスト側で実行済み)"
        echo "[dry-run]   R3 (bit 一致)                -> $GATE_LOG_DIR/bit_match_run.log"
        echo "[dry-run]     cargo test -p fandhe-ai-backend-metal --release --test gemm_te_parity te_bit_match_with_production_dispatch_auto -- --ignored --nocapture --test-threads=1"
        return 0
    fi

    mkdir -p "$GATE_LOG_DIR"
    cd "$REPO_ROOT"

    echo "gate: R0 (probe) を実行する"
    cargo test -p fandhe-ai-backend-metal --release --lib \
        gemm::tests::te_layout_probe_matches_model \
        -- --ignored --nocapture --test-threads=1 \
        > "$GATE_LOG_DIR/probe_run.log" 2>&1 || {
        STATUS=$?
        echo "gate: R0 (probe) が失敗した（status=${STATUS}）。${GATE_LOG_DIR}/probe_run.log を確認する。R0 FAIL のため後続（R1〜R3）は実行しない" >&2
        exit "$STATUS"
    }

    echo "gate: R1 (parity, 外部テスト) を実行する"
    cargo test -p fandhe-ai-backend-metal --release --test gemm_te_parity \
        -- --ignored --nocapture --test-threads=1 \
        > "$GATE_LOG_DIR/parity_run.log" 2>&1 || {
        STATUS=$?
        echo "gate: R1 (parity) が失敗した（status=${STATUS}）。${GATE_LOG_DIR}/parity_run.log を確認する。R1 FAIL のため後続（R2〜R3）は実行しない" >&2
        exit "$STATUS"
    }

    echo "gate: R1 (all_staged_candidates_match_te_cpu_reference_512_nn) を実行する"
    cargo test -p fandhe-ai-backend-metal --release --lib \
        gemm::tests::all_staged_candidates_match_te_cpu_reference_512_nn \
        -- --ignored --nocapture --test-threads=1 \
        > "$GATE_LOG_DIR/all_staged_candidates_run.log" 2>&1 || {
        STATUS=$?
        echo "gate: R1 (all_staged_candidates) が失敗した（status=${STATUS}）。${GATE_LOG_DIR}/all_staged_candidates_run.log を確認する。R1 FAIL のため後続（R2〜R3）は実行しない" >&2
        exit "$STATUS"
    }

    echo "gate: R2 (非 staged 拒否) を実行する"
    cargo test -p fandhe-ai-backend-metal --release --test gemm_te_parity \
        te_rejects_non_staged_candidate \
        -- --ignored --nocapture --test-threads=1 \
        >> "$GATE_LOG_DIR/parity_run.log" 2>&1 || {
        STATUS=$?
        echo "gate: R2 (非 staged 拒否) が失敗した（status=${STATUS}）。${GATE_LOG_DIR}/parity_run.log を確認する（機構契約の不成立として記録し、性能 A/B は参考値扱いとする）" >&2
        exit "$STATUS"
    }

    echo "gate: R3 (本番との bit 一致) を実行する"
    cargo test -p fandhe-ai-backend-metal --release --test gemm_te_parity \
        te_bit_match_with_production_dispatch_auto \
        -- --ignored --nocapture --test-threads=1 \
        > "$GATE_LOG_DIR/bit_match_run.log" 2>&1 || {
        STATUS=$?
        echo "gate: R3 (bit 一致) が失敗した（status=${STATUS}）。${GATE_LOG_DIR}/bit_match_run.log を確認する（機構契約の不成立として記録し、性能 A/B は参考値扱いとする）" >&2
        exit "$STATUS"
    }

    echo "gate: R0〜R3 すべて成功した。性能 A/B（run1〜run5）へ進める"
}

run_ab() {
    RUN_NO="$1"
    UPTIME_BEFORE="$SCRIPT_DIR/uptime_before_run${RUN_NO}.txt"
    PMSET_BEFORE="$SCRIPT_DIR/pmset_therm_before_run${RUN_NO}.txt"
    PMSET_AFTER="$SCRIPT_DIR/pmset_therm_after_run${RUN_NO}.txt"
    RUN_LOG="$SCRIPT_DIR/kernel_gpu_te_ab_run${RUN_NO}.log"
    MONITOR_LOG="$SCRIPT_DIR/run${RUN_NO}_monitor.log"
    PROCS_LOG="$SCRIPT_DIR/run${RUN_NO}_procs.txt"
    ENV_INFO="$SCRIPT_DIR/env_info.txt"

    CMD="cargo test -p fandhe-ai-backend-metal --release --lib gemm_te_diag_tests::te_kernel_gpu_ab_vs_production_select -- --ignored --nocapture --test-threads=1"

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

    # 既存成果物の確認（イシュー #1515 と同じ判断: 同じ run 番号の再実行
    # によるログ上書き防止。計測開始前に既存があれば何も書かず非ゼロ
    # 終了する）。
    for artifact in "$UPTIME_BEFORE" "$PMSET_BEFORE" "$PMSET_AFTER" "$RUN_LOG" "$MONITOR_LOG" "$PROCS_LOG"; do
        if [ -e "$artifact" ]; then
            echo "run${RUN_NO}: 既存の成果物が見つかった（${artifact}）。" \
                 "同番号の再実行は成果物を手動で別名へ退避してから行う" >&2
            exit 1
        fi
    done

    # 同番号の同時実行を排他的に拒否する（`mkdir` の単一アトミック排他
    # 作成性を利用。イシュー #1515 と同じ設計）。
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

    # バックグラウンド `uptime` サンプラー（10 秒間隔。イシュー #1515 と
    # 同じ整形。macOS の `uptime` 出力〈load averages: X Y Z〉から抽出）。
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

    cd "$REPO_ROOT"
    # shellcheck disable=SC2086 # CMD は固定リテラル（外部入力を含まない）
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
