#!/bin/sh
# candle／MLX steel 差分由来の GEMM 候補（イシュー #2110。機構は
# `crates/backend-metal/src/{tile.rs, gemm.rs, pipeline.rs, spec_source.rs,
# shaders/gemm.metal}`、ハーネスは `gemm_steel_candidate_diag_tests.rs`）の
# 実機（Apple Silicon）実行ラッパー。実測は #2111 が行う。
# `docs/perf/logs/metal-gemm-smem-swizzle-ab-1970/orchestrate.sh` を雛形とし、
# 負荷運用のみ「専有ゲート」（RULE.txt 7.）へ変更している。
#
#   ./orchestrate.sh gate [--dry-run]
#       RULE.txt 1.（前提ゲート）: bit 一致 3 本 + parity 1 本を `--exact` で
#       実行する。1 件でも FAIL なら打ち切る（A/B は実施しない）。
#       ログは本ディレクトリ直下の `gate_run.log`。run 番号の実行は起動前にこのログを検証し、
#       ゲート未成立なら拒否する（verify_gate_log）。
#
#   ./orchestrate.sh <run番号（1〜5 の整数）> [--dry-run]
#       kernel_gpu A/B（`steel_candidate_kernel_gpu_ab_production_sizes`）を
#       1 プロセス起動する。起動前に load1 < 8.0 を 30 秒間隔・最大 30 分待つ
#       （`load_gate.log` に記録。timeout でも起動し、系列を参考扱いにする）。
#
# --dry-run（または DRY_RUN=1）は cargo を起動せず、実行コマンドと出力先のみ
# 表示する（Linux でも分岐・構文を検証できる）。
#
# セキュリティ（`.claude/rules/security.md` A03/A01）: 引数は「gate」または
# 「1〜5 の整数」と `--dry-run` のみ。それ以外は起動前に拒否する。並走プロセスは
# 固定 watchlist の件数のみ記録し、コマンドライン全文・絶対パスは記録しない。
# 保存ログ中のホスト名・ユーザー名・絶対パスは `<home>` 等へマスクしてから
# コミットすること（README 参照）。
set -eu

USAGE="usage: $0 <gate|run番号（1〜5 の整数）> [--dry-run]"
MODE="${1:-}"
if [ -z "$MODE" ]; then
    echo "$USAGE" >&2
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

# MODE を起動前に検証する（gate または 1〜5）。
case "$MODE" in
    gate | 1 | 2 | 3 | 4 | 5) ;;
    *)
        echo "$USAGE" >&2
        exit 1
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

LOAD_THRESHOLD=8
LOAD_POLL_SECS=30
LOAD_MAX_WAIT_SECS=1800

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

# load1 を整数部で返す（取得失敗時は 999）。
current_load1() {
    uptime | sed -E 's/.*load averages?: *([0-9]+)[.,0-9]*.*/\1/' | grep -E '^[0-9]+$' || echo 999
}

# 専有ゲート: load1 < LOAD_THRESHOLD を待つ。timeout でも 0 を返し、
# load_gate.log に TIMEOUT を記録する（aggregate.py が参考扱いにする）。
wait_load_gate() {
    run_no="$1"
    log="$SCRIPT_DIR/load_gate.log"
    waited=0
    while :; do
        l1=$(current_load1)
        if [ "$l1" -lt "$LOAD_THRESHOLD" ]; then
            echo "run${run_no} OK load1=${l1} waited=${waited}s" >> "$log"
            return 0
        fi
        if [ "$waited" -ge "$LOAD_MAX_WAIT_SECS" ]; then
            echo "run${run_no} TIMEOUT load1=${l1} waited=${waited}s" >> "$log"
            return 0
        fi
        sleep "$LOAD_POLL_SECS"
        waited=$((waited + LOAD_POLL_SECS))
    done
}

GATE_TESTS="
gemm_steel_candidate_diag_tests::unroll_load_on_off_bit_match_all_candidates
gemm_steel_candidate_diag_tests::unroll_load_on_off_bit_match_dispatch_auto
gemm_steel_candidate_diag_tests::unroll_load_on_off_bit_match_transposed
gemm_steel_candidate_diag_tests::steel_candidate_arms_match_cpu_reference
"

# 前提ゲート成立の機械検証（RULE.txt 1.。fail-closed）。gate_run.log が存在し、
# GATE_TESTS の全件が `... ok` で `0 failed`・FAILED/panicked なしであることを要求する。
# 不成立なら 0 以外を返す。A/B（run_ab）の起動前に呼ぶ。aggregate.py の check_gate_log と同条件。
verify_gate_log() {
    GATE_LOG="$SCRIPT_DIR/gate_run.log"
    if [ ! -f "$GATE_LOG" ]; then
        echo "前提ゲート未実施: ${GATE_LOG} が無い。先に ./orchestrate.sh gate を実行する（RULE.txt 1.）" >&2
        return 1
    fi
    if grep -Eq 'FAILED|panicked' "$GATE_LOG"; then
        echo "前提ゲート不成立: ${GATE_LOG} に FAILED/panicked がある。A/B は実施しない（RULE.txt 1.）" >&2
        return 1
    fi
    for t in $GATE_TESTS; do
        if ! grep -Fxq "test ${t} ... ok" "$GATE_LOG"; then
            echo "前提ゲート不成立: ${t} の成功行が無い。A/B は実施しない（RULE.txt 1.）" >&2
            return 1
        fi
    done
    if ! grep -Eq '^test result: ok\. 4 passed; 0 failed' "$GATE_LOG"; then
        echo "前提ゲート不成立: 'test result: ok. 4 passed; 0 failed' が無い（RULE.txt 1.）" >&2
        return 1
    fi
    return 0
}

run_gate() {
    GATE_LOG="$SCRIPT_DIR/gate_run.log"
    if [ "$DRY_RUN" = "1" ]; then
        echo "[dry-run] gate: 4 テストを --exact で実行する（--test-threads=1）"
        for t in $GATE_TESTS; do
            echo "[dry-run]   $t"
        done
        echo "[dry-run]   log -> $GATE_LOG"
        return 0
    fi
    if [ -e "$GATE_LOG" ]; then
        echo "gate: 既存の成果物がある（${GATE_LOG}）。手動で別名へ退避してから再実行する" >&2
        exit 1
    fi
    cd "$REPO_ROOT"
    # shellcheck disable=SC2086 # GATE_TESTS は固定リテラル集合（外部入力なし）
    cargo test -p fandhe-ai-backend-metal --release --lib \
        -- --ignored --exact --test-threads=1 --nocapture \
        $GATE_TESTS \
        > "$GATE_LOG" 2>&1 || {
        STATUS=$?
        echo "gate: FAIL（status=${STATUS}）。${GATE_LOG} を確認する。A/B は実施しない（RULE.txt 1.）" >&2
        exit "$STATUS"
    }
    echo "gate: 全 4 テスト成功した。A/B（run1〜run5）へ進める"
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

    CMD="cargo test -p fandhe-ai-backend-metal --release --lib gemm_steel_candidate_diag_tests::steel_candidate_kernel_gpu_ab_production_sizes -- --ignored --nocapture --test-threads=1"

    if [ "$DRY_RUN" = "1" ]; then
        echo "[dry-run] run${RUN_NO}"
        echo "[dry-run]   load_gate     -> load1<${LOAD_THRESHOLD} を ${LOAD_POLL_SECS}s 間隔・最大 ${LOAD_MAX_WAIT_SECS}s 待つ（load_gate.log）"
        echo "[dry-run]   uptime_before -> $UPTIME_BEFORE"
        echo "[dry-run]   pmset_before  -> $PMSET_BEFORE"
        echo "[dry-run]   monitor_log   -> ${MONITOR_LOG}"
        echo "[dry-run]   procs_log     -> ${PROCS_LOG}（watchlist: ${WATCHLIST}）"
        echo "[dry-run]   command       -> $CMD"
        echo "[dry-run]   run_log       -> $RUN_LOG"
        echo "[dry-run]   pmset_after   -> $PMSET_AFTER"
        return 0
    fi

    verify_gate_log || exit 1

    for artifact in "$UPTIME_BEFORE" "$PMSET_BEFORE" "$PMSET_AFTER" "$RUN_LOG" "$MONITOR_LOG" "$PROCS_LOG"; do
        if [ -e "$artifact" ]; then
            echo "run${RUN_NO}: 既存の成果物がある（${artifact}）。差し替えは RULE.txt 2. で禁止" >&2
            exit 1
        fi
    done

    LOCK_DIR="$SCRIPT_DIR/run${RUN_NO}.lock"
    if ! mkdir "$LOCK_DIR" 2>/dev/null; then
        echo "run${RUN_NO}: 同番号の実行が進行中（ロック $LOCK_DIR が存在する）" >&2
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

    wait_load_gate "$RUN_NO"
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
            echo "${TS} load1=${1:-NA} load5=${2:-NA} load15=${3:-NA}" >> "$MONITOR_LOG"
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
    echo "run${RUN_NO} completed at $(date -u +"%Y-%m-%dT%H:%M:%SZ")" >> "$ENV_INFO"
    echo "run${RUN_NO} 完了: ${RUN_LOG}（監視ログ: ${MONITOR_LOG}）"
}

case "$MODE" in
    gate) run_gate ;;
    *) run_ab "$MODE" ;;
esac
