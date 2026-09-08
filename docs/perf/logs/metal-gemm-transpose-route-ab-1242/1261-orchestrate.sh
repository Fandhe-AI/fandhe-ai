#!/bin/bash
# イシュー #1261: 分離計測（--gpu-timestamps）を排他環境で実行し、
# 原因候補 (a) コマンドバッファのスケジューリング・(b) ウォームアップ
# 不足・(c) pmset 電源状態 を切り分けるための一括オーケストレーション。
#
# 位置づけ: `crates/bench-harness::env_guard`（イシュー #1264/#1265。
# `AbConfig` の判定ロジック自体は変更しない）は example への結線・
# バックオフ再試行がまだ実装されていない（#1265 のスコープ）ため、本
# スクリプトは一回限りの計測実行用としてここ（docs/perf/logs 配下）に
# 置く。恒久機構化は #1265 が別途行う。
#
# フロー: 排他環境ゲート（§3.1）→ E1〜E3（--phase1-only
# --gpu-timestamps）→ W1〜W2（同 + --min-warmup-secs=9）。各 run 前後に
# pmset/uptime/GPU プロセスを記録し、run 中も一定間隔でサンプリングする。
#
# セキュリティ（OWASP A03）: 外部コマンドは固定引数のみで起動し、シェル
# 経由の任意実行はしない。ホスト名・ユーザー名・ホームディレクトリの
# 絶対パスはログへ書く前に必ず SANITIZE_SED を通す
# （`docs/real-hardware-verification-env.md` 方針。イシュー #1261 計画
# §4 ステップ 10）。
set -uo pipefail

# ---------------------------------------------------------------------
# パス解決（#1313 の m4max_orchestrate.sh と同型: 自身の配置から導出）
# ---------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGDIR="${SCRIPT_DIR}"
# リポジトリルートはこのログディレクトリの 4 階層上
# （docs/perf/logs/metal-gemm-transpose-route-ab-1242/ から遡る）。
REPO_ROOT="$(cd "${LOGDIR}/../../../.." && pwd)"
BINARY="${REPO_ROOT}/target/release/examples/gemm_transpose_route_ab_bench"

# ホームディレクトリ絶対パス・ユーザー名を伏字化する sed フィルタ
# （計画 §4 ステップ 10。cargo stderr・`ps`/`pgrep` の comm フル実行パス
# 混入対策。ログへ書き出す全経路でこれを通す）。
SANITIZE_SED='s#/Users/[^/[:space:]]+#/Users/<user>#g'

if [ ! -x "${BINARY}" ]; then
    echo "ERROR: バイナリが見つからない（先に cargo build --release が必要）: ${BINARY}" >&2
    exit 1
fi

# ---------------------------------------------------------------------
# 排他環境ゲート（計画 §3.1。ここでは env_guard は使わず独立実装——
# #1265 のスコープと衝突しないための一回限りの計測実行用スクリプト）
# ---------------------------------------------------------------------
GATE_LOAD_THRESHOLD="2.0"
GATE_INTERVAL_SECS=60
GATE_MAX_ATTEMPTS=30
GATE_LOG="${LOGDIR}/1261-gate.log"
: > "${GATE_LOG}"

# GPU プロセス watchlist（部分一致）。本スクリプト自身・本 example 以外の
# ビルド/ベンチ系プロセスを検出する（計画 §3.1「GPU プロセス確認」）。
GPU_WATCH_PATTERN='cargo|rustc|bench|gemm_|python|torch|mlx'

# load average の 1 分値を取り出す（macOS `uptime` の
# "load averages: 1.23 4.56 7.89" 形式）。
current_load1() {
    uptime | sed -E 's/.*load averages?: *([0-9.]+).*/\1/'
}

# 自プロセス（オーケストレータ）・本 example 以外の該当プロセスを
# 検出し、伏字化して1行ずつ出す（0 件なら何も出さない）。
#
# 注意（codex-review 指摘・PR #1457）: macOS（BSD）の `pgrep` に `-E`
# オプションは存在しない（`pgrep` は既定で拡張正規表現を解釈するため
# `-E` は不要かつ無効な引数でありコマンド自体が失敗する）。以前は
# `-E` を渡していたため `pgrep` が毎回失敗し、その失敗が
# `| grep -v ...` のパイプラインに吸収されて `procs=""`・
# `proc_count=0`（該当プロセスなしの意味に誤読される値）になっていた
# ——低負荷時であっても実際には他 GPU プロセスの有無を一切検査できて
# いなかった。`-E` を除去し、`pgrep` 自体の終了コードも呼び出し元
# （ゲートループ）で確認できるよう関数の戻り値として伝播する。
gpu_watch_processes() {
    local pgrep_out
    pgrep_out="$(pgrep -fl "${GPU_WATCH_PATTERN}" 2>/dev/null)"
    local pgrep_status=$?
    # pgrep の終了コード: 0=一致あり、1=一致なし、2 以上=起動失敗
    # （不正オプション等）。1（一致なし）は正常系として扱うが、
    # 2 以上は「検査できていない」ことを示すため呼び出し元へ伝える。
    if [ "${pgrep_status}" -ge 2 ]; then
        echo "PGREP_ERROR status=${pgrep_status}" >&2
        return 2
    fi
    printf '%s\n' "${pgrep_out}" \
        | grep -v "1261-orchestrate.sh" \
        | grep -v "gemm_transpose_route_ab_bench" \
        | grep -v '^$' \
        | sed -E "${SANITIZE_SED}"
    return 0
}

echo "gate_start_unix=$(date +%s)" >> "${GATE_LOG}"
gate_passed=0
prev_ok=0
for attempt in $(seq 1 "${GATE_MAX_ATTEMPTS}"); do
    load1="$(current_load1)"
    procs="$(gpu_watch_processes)"
    pgrep_check_status=$?
    proc_count=0
    if [ -n "${procs}" ]; then
        proc_count=$(printf '%s\n' "${procs}" | grep -c .)
    fi
    ok=0
    # pgrep_check_status != 0（起動失敗。「一致なし」を意味する 1 は
    # gpu_watch_processes 内部で正常系として吸収済みのためここでは
    # 現れない）の場合は「検査できていない」として fail-closed に
    # ok=0 とする（誤って proc_count=0 のままゲート通過させない）。
    if [ "${pgrep_check_status}" -eq 0 ] \
        && awk -v l="${load1}" -v t="${GATE_LOAD_THRESHOLD}" 'BEGIN{exit !(l<t)}'; then
        if [ "${proc_count}" -eq 0 ]; then
            ok=1
        fi
    fi
    {
        echo "attempt=${attempt} load1=${load1} proc_count=${proc_count} pgrep_check_status=${pgrep_check_status} ok=${ok}"
        if [ -n "${procs}" ]; then
            printf '%s\n' "${procs}" | sed 's/^/  matched: /'
        fi
    } >> "${GATE_LOG}"

    if [ "${ok}" -eq 1 ] && [ "${prev_ok}" -eq 1 ]; then
        gate_passed=1
        echo "gate_passed_at_attempt=${attempt}" >> "${GATE_LOG}"
        break
    fi
    prev_ok="${ok}"
    if [ "${attempt}" -lt "${GATE_MAX_ATTEMPTS}" ]; then
        sleep "${GATE_INTERVAL_SECS}"
    fi
done

if [ "${gate_passed}" -ne 1 ]; then
    echo "gate_not_passed" >> "${GATE_LOG}"
    : > "${LOGDIR}/1261-GATE_NOT_PASSED.marker"
    echo "排他環境ゲート不通過（${GATE_MAX_ATTEMPTS} 回試行）。計測せず終了する。" >&2
    exit 0
fi

# ---------------------------------------------------------------------
# 実行前後スナップショット（各 run 共通ヘルパ）
# ---------------------------------------------------------------------
snapshot() {
    local label="$1"
    {
        echo "=== ${label} unix=$(date +%s) ==="
        echo "--- uptime ---"
        uptime
        echo "--- pmset -g ---"
        pmset -g
        echo "--- pmset -g therm ---"
        pmset -g therm
        echo "--- pmset -g batt ---"
        pmset -g batt
        echo "--- gpu_watch_processes ---"
        gpu_watch_processes
    } | sed -E "${SANITIZE_SED}"
}

# 実行中サンプリングをバックグラウンドで開始し、PID を返す。
#
# 注意（codex-review 指摘・PR #1457）: 以下 2 つの無限ループはループ本体
# の出力を明示的にファイルへリダイレクトしているが、`( ... ) &` の
# サブシェル自体の stdout（fd 1）はこの関数の呼び出し元
# （`sampling_pids="$(start_during_sampling ...)"` というコマンド置換）
# のパイプ書き込み端を継承したままになる。ループが無限に回り続ける限り
# この fd 1 は閉じられないため、コマンド置換は PID を echo した後も
# パイプの読み取り側で EOF を待ち続け、`sampling_pids` への代入が
# 停止していた（排他ゲート通過後もベンチ本体へ到達できない）。
# `>/dev/null 2>&1` でサブシェル自体の fd 1/2 を明示的に切り離し
# （ループ内部のファイルへのリダイレクトは個々のコマンドに対して
# 別途行われているため、この変更はログ出力先には影響しない）、
# 継承されたパイプの書き込み端を確実に閉じる。
start_during_sampling() {
    local run_label="$1"
    local uptime_log="${LOGDIR}/1261-uptime-during-${run_label}.log"
    local pmset_log="${LOGDIR}/1261-pmset-during-${run_label}.log"
    : > "${uptime_log}"
    : > "${pmset_log}"
    (
        while true; do
            { echo "t=$(date +%s)"; uptime; } | sed -E "${SANITIZE_SED}" >> "${uptime_log}"
            sleep 30
        done
    ) >/dev/null 2>&1 &
    local uptime_pid=$!
    (
        while true; do
            {
                echo "t=$(date +%s)"
                pmset -g
                pmset -g therm
                pmset -g batt
            } | sed -E "${SANITIZE_SED}" >> "${pmset_log}"
            sleep 60
        done
    ) >/dev/null 2>&1 &
    local pmset_pid=$!
    echo "${uptime_pid} ${pmset_pid}"
}

stop_during_sampling() {
    local pids="$1"
    for pid in ${pids}; do
        kill "${pid}" 2>/dev/null
    done
    wait 2>/dev/null
}

# 1 run を実行する。$1=run_label（例: exclusive-run1） $2...=example への引数
run_one() {
    local run_label="$1"
    shift
    local out_log="${LOGDIR}/1261-${run_label}.log"

    snapshot "before ${run_label}" > "${LOGDIR}/1261-${run_label}-before.txt"
    local sampling_pids
    sampling_pids="$(start_during_sampling "${run_label}")"

    # `cargo run` はビルド出力（絶対パス）を含みうるため、事前ビルド済み
    # バイナリを直接起動する（計画 §4 ステップ 10 の理由）。
    "${BINARY}" "$@" > "${out_log}" 2>&1
    local exit_code=$?
    sed -E -i '' "${SANITIZE_SED}" "${out_log}"

    stop_during_sampling "${sampling_pids}"
    snapshot "after ${run_label}" > "${LOGDIR}/1261-${run_label}-after.txt"

    echo "run_label=${run_label} exit_code=${exit_code}" >> "${GATE_LOG}"
}

# ---------------------------------------------------------------------
# 計測本体: E1〜E3（排他環境・既定 MIN_WARMUP）→ W1〜W2
# （--min-warmup-secs=9。原因候補 (b) の切り分け）
# ---------------------------------------------------------------------
run_one "exclusive-run1" --phase1-only --gpu-timestamps
run_one "exclusive-run2" --phase1-only --gpu-timestamps
run_one "exclusive-run3" --phase1-only --gpu-timestamps
run_one "warmup9s-run1" --phase1-only --gpu-timestamps --min-warmup-secs=9
run_one "warmup9s-run2" --phase1-only --gpu-timestamps --min-warmup-secs=9

: > "${LOGDIR}/1261-ALL_DONE.marker"
echo "全 run 完了" >&2
