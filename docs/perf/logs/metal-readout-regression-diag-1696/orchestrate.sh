#!/bin/sh
# Metal readout legacy 後退の 4 腕診断（イシュー #1696）M4 Max 実機実測
# オーケストレーション。#1695 が実装した診断ハーネス
# （`crates/backend-metal/src/readout_regression_diag_tests_1695.rs`）の
# 単一腕・単一サイズ 12 テストを各 5 プロセス起動（主系列）・
# `*_n{1024,2048,4096}`（in-process 4 腕。副系列）を各 1 起動する。
#
# モジュール名付き完全修飾名 + `--exact` を必ず使う（README.md「単一
# テスト名フィルタの注意」参照。モジュール名を省いた部分一致では単腕
# 関数と一括腕関数の両方に一致し、同一プロセス内でまとめて実行されて
# しまいアロケータ状態の交絡が生じる。#1436 と同じ理由）。
#
# 引数は `--dry-run` のみを受け付ける（それ以外の外部入力を展開しない。
# `.claude/rules/security.md` A03）。
#
# 使い方（このディレクトリで実行する想定。worktree のパスは実行環境に
# 合わせる）:
#   sh orchestrate.sh            # 実行する
#   sh orchestrate.sh --dry-run  # 実行するコマンド列を表示するのみ
#
# 生成物: env_info.txt・uptime_before.txt／uptime_after.txt・
# uptime_sampler.log・pmset_therm_{before,after}.txt・
# layerB-n{N}-{arm}-run{1..5}.log（60 ファイル）・
# inprocess-n{N}-run1.log（3 ファイル）。集計は aggregate.py が行う
# （内部ホスト名は含めない。README.md の事前登録判定規則に従い
# docs/perf/metal-readout-legacy-regression-four-arm-diag.md §6〜§8・§10
# へ転記する）。

set -eu

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=1
fi

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../../../.." && pwd)

MODULE="readout_regression_diag_tests_1695"

# 主系列: 単一腕・単一サイズ（プロセス分離。各 5 run）
SIZES="1024 2048 4096"
ARMS="legacy_to_vec borrowed_keep_alive borrowed_with_dummy_alloc_free pretouched_reused_dest"

layer_b_cmd() {
    n="$1"
    arm="$2"
    echo "cargo test -p fandhe-ai-backend-metal --release --lib ${MODULE}::readout_regression_diag_n${n}_${arm} -- --ignored --nocapture --test-threads=1 --exact"
}

# 副系列: in-process 4 腕（各 1 run）
inprocess_cmd() {
    n="$1"
    echo "cargo test -p fandhe-ai-backend-metal --release --lib ${MODULE}::readout_regression_diag_n${n} -- --ignored --nocapture --test-threads=1 --exact"
}

if [ "$DRY_RUN" -eq 1 ]; then
    echo "=== dry-run: 実行するコマンド列（repo root: $REPO_ROOT） ==="
    for n in $SIZES; do
        for arm in $ARMS; do
            i=1
            while [ "$i" -le 5 ]; do
                echo "[layerB n=$n arm=$arm run=$i] $(layer_b_cmd "$n" "$arm")"
                i=$((i + 1))
            done
        done
    done
    for n in $SIZES; do
        echo "[inprocess n=$n run=1] $(inprocess_cmd "$n")"
    done
    exit 0
fi

echo "=== env_info ==="
{
    uname -srm
    sw_vers
    rustc -V
    sysctl machdep.cpu.brand_string
    sysctl hw.pagesize
    git -C "$REPO_ROOT" rev-parse HEAD
} >"$SCRIPT_DIR/env_info.txt" 2>&1
cat "$SCRIPT_DIR/env_info.txt"

uptime >"$SCRIPT_DIR/uptime_before.txt"
pmset -g therm >"$SCRIPT_DIR/pmset_therm_before.txt" 2>&1 || true

# 実行中の load average 推移を 30 秒間隔で記録する（record_only 運用。
# 専有ゲートには使わない参考記録。バックグラウンドで起動し末尾で止める）。
(
    while true; do
        date
        uptime
        sleep 30
    done
) >"$SCRIPT_DIR/uptime_sampler.log" 2>&1 &
SAMPLER_PID=$!
trap 'kill "$SAMPLER_PID" >/dev/null 2>&1 || true' EXIT

cd "$REPO_ROOT"

echo "=== 主系列: 単一腕・単一サイズ（各 5 run） ==="
for n in $SIZES; do
    for arm in $ARMS; do
        i=1
        while [ "$i" -le 5 ]; do
            log="$SCRIPT_DIR/layerB-n${n}-${arm}-run${i}.log"
            echo "--- n=$n arm=$arm run=$i ---"
            cmd=$(layer_b_cmd "$n" "$arm")
            # shellcheck disable=SC2086
            eval "$cmd" >"$log" 2>&1 || true
            tail -20 "$log"
            i=$((i + 1))
        done
    done
done

echo "=== 副系列: in-process 4 腕（各 1 run） ==="
for n in $SIZES; do
    log="$SCRIPT_DIR/inprocess-n${n}-run1.log"
    echo "--- n=$n (inprocess) ---"
    cmd=$(inprocess_cmd "$n")
    # shellcheck disable=SC2086
    eval "$cmd" >"$log" 2>&1 || true
    tail -20 "$log"
done

kill "$SAMPLER_PID" >/dev/null 2>&1 || true
trap - EXIT

uptime >"$SCRIPT_DIR/uptime_after.txt"
pmset -g therm >"$SCRIPT_DIR/pmset_therm_after.txt" 2>&1 || true

echo DONE
