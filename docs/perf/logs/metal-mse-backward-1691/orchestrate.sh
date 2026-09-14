#!/bin/sh
# Metal `mse_loss_backward` encode-only 化（イシュー #1690・実装は
# PR #1785）を検証する M4 Max 実機実測オーケストレーション（イシュー
# #1691）。before 腕（`84490ad1`。#1690 適用前）・after 腕（本ブランチ）
# をそれぞれ別 worktree に展開した前提で、以下を順に実行する:
#
#   (a) REQ-2 正しさ確認（after 腕で `mse_parity.rs::
#       mse_matches_cpu_across_shapes` を実行）
#   (b) bit 同一確認（両腕で `metal_reuse_step_grad_bit_dump` を実行し、
#       `^(step\[|final\.param\[)` 行を diff。抽出件数が独立の期待値
#       `EXPECTED_BITDUMP_LINES` と一致し項目ラベル集合が両腕で一致する
#       ことを先に検証したうえで、差分 0 行を期待。検証に失敗した場合は
#       「一致」ではなく「判定不能」として記録する〈#1563 方式の踏襲〉）
#   (c) カウンタ実測（after 腕で `mnist_scale_train_reuse_metal_batch_
#       counters`〈hard assert 11/7/7〉・before 腕で同テスト〈期待
#       11/8/8 再現〉・両腕で `mnist_scale_train_reuse_metal_backward_
#       dinput_phase`〈record-only・両腕とも仮説 5/3/3〉）
#   (d) backward マイクロベンチ 5 round・起動順反転
#       （`mse_backward_bench.rs::mse_backward_cases`。
#       `FANDHE_BENCH_DEVICE=metal`）
#   (f) 既存 `#[ignore]` 群非後退確認（after 腕）
#   (e) A/B（`scripts/bench/framework-compare/run_ab_mse_encode_metal.sh`。
#       5 round・record_only）
#
# 引数は `--dry-run` のみを受け付ける（それ以外の外部入力を展開しない。
# `.claude/rules/security.md` A03）。before/after 腕のパスは環境変数
# `MSE_AB_BEFORE_REPO`／`MSE_AB_AFTER_REPO`（絶対パス。それぞれの
# worktree のリポジトリルート）で指定する。
#
# 使い方（このディレクトリで実行する想定）:
#   MSE_AB_BEFORE_REPO=/abs/path/to/before-worktree \
#   MSE_AB_AFTER_REPO=/abs/path/to/after-worktree \
#     sh orchestrate.sh            # 実行する
#   sh orchestrate.sh --dry-run    # 実行するコマンド列を表示するのみ
#     （--dry-run はリポジトリパス環境変数なしでも動作する）
#
# 生成物: mse_parity_after.log・bitdump_before.log・bitdump_after.log・
# bitdump_before_filtered.txt・bitdump_after_filtered.txt・
# bitdump_before_labels.txt・bitdump_after_labels.txt・
# bitdump_label_diff.txt・bitdump_diff.txt・
# batch_counters_after.log・batch_counters_before.log・
# backward_phase_after.log・backward_phase_before.log・
# before_round{1..5}.log・after_round{1..5}.log・rounds.log・
# aggregate.md（`aggregate.py` の出力）・
# ignored_after_mse_parity.log・ignored_after_mnist.log・
# ignored_after_command_batching.log・ignored_after_command_batching_bench.log・
# ignored_after_gemm_resident_parity.log・
# ignored_after_device_param_store_backend_parity.log・
# ab/（`run_ab_mse_encode_metal.sh` の実行ログ。生成物本体
# 〈compare-train-1691.md・results/raw/* 等〉は
# `$AFTER_REPO/scripts/bench/framework-compare/` 配下に残るため、実測後
# 手動でこのディレクトリの `ab/` へコピーする運用）・
# env_info_before.txt・env_info_after.txt・uptime_before.txt・
# uptime_after.txt（内部ホスト名は含めない）。

set -eu

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=1
fi

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

BEFORE_REPO="${MSE_AB_BEFORE_REPO:-}"
AFTER_REPO="${MSE_AB_AFTER_REPO:-}"

MSE_PARITY_CMD="cargo test -p fandhe-ai-backend-metal --release --test mse_parity -- --ignored --nocapture"
BIT_DUMP_CMD="cargo test -p fandhe-ai --release --test metal_reuse_step_grad_bit_dump -- --ignored --nocapture"

# 独立の期待行数（`docs/perf/logs/metal-dinput-sync-1563/orchestrate.sh`
# と同一の実測値・同一テスト）。腕間比較とは独立にこの期待値との突合を
# 行う（両腕で同じ抽出漏れが起きた場合の見かけ上の一致を防ぐため。
# PR #1665 codex-review 指摘と同型の対策）。
EXPECTED_BITDUMP_LINES=4462

IGNORED_CMD_MSE_PARITY="$MSE_PARITY_CMD"
IGNORED_CMD_MNIST="cargo test -p fandhe-ai --release --test mnist_scale_train_reuse_bench -- --ignored --nocapture --test-threads=1"
IGNORED_CMD_COMMAND_BATCHING="cargo test -p fandhe-ai-backend-metal --release --test command_batching -- --ignored --nocapture"
IGNORED_CMD_COMMAND_BATCHING_BENCH="cargo test -p fandhe-ai-backend-metal --release --test command_batching_bench -- --ignored --nocapture"
IGNORED_CMD_GEMM_RESIDENT_PARITY="cargo test -p fandhe-ai-backend-metal --release --test gemm_resident_parity -- --ignored --nocapture"
# `device_param_store_backend_parity` は Metal・CUDA 両方の `#[ignore]`
# テスト（`device_resident_matches_host_sgd_on_{metal,cuda}_across_100_steps`・
# `grad_readout_contract_on_{metal,cuda}`）を同一バイナリに含む
# （`crates/facade/tests/device_param_store_backend_parity.rs`）。フィルタ
# なしで `--ignored` を実行すると M4 Max 実機上で CUDA 必須テストまで
# 選択されて失敗し、`set -eu` によりオーケストレーション本体（(e) の
# 主判定 A/B）へ到達できなくなる（codex-review [P1] 指摘対応。
# `mnist_scale_train_reuse_bench` は `#![cfg(target_os = "macos")]` で
# ファイル全体が macOS 限定のため CUDA テストを含まず対象外）。
# `on_metal` 部分一致フィルタで Metal 用 2 ケースへ限定する。
IGNORED_CMD_STORE_PARITY="cargo test -p fandhe-ai --release --test device_param_store_backend_parity -- --ignored --nocapture on_metal"

BATCH_COUNTERS_CMD="cargo test -p fandhe-ai --release --test mnist_scale_train_reuse_bench -- --ignored --nocapture --test-threads=1 mnist_scale_train_reuse_metal_batch_counters"
BACKWARD_PHASE_CMD="cargo test -p fandhe-ai --release --test mnist_scale_train_reuse_bench -- --ignored --nocapture --test-threads=1 mnist_scale_train_reuse_metal_backward_dinput_phase"

MSE_BENCH_CMD="FANDHE_BENCH_DEVICE=metal cargo test -p fandhe-ai --release --test mse_backward_bench -- --ignored --nocapture --exact mse_backward_cases"
ROUNDS=5

if [ "$DRY_RUN" -eq 1 ]; then
    echo "=== dry-run: 実行するコマンド列 ==="
    echo "[after repo]  $MSE_PARITY_CMD"
    echo "[before repo] $BIT_DUMP_CMD"
    echo "[after repo]  $BIT_DUMP_CMD"
    echo "diff bitdump_before_filtered.txt bitdump_after_filtered.txt"
    echo "[after repo]  $BATCH_COUNTERS_CMD"
    echo "[before repo] $BATCH_COUNTERS_CMD"
    echo "[after repo]  $BACKWARD_PHASE_CMD"
    echo "[before repo] $BACKWARD_PHASE_CMD"
    echo "[before/after repo x5 round, interleaved order] $MSE_BENCH_CMD"
    echo "python3 aggregate.py . $ROUNDS"
    echo "[after repo]  $IGNORED_CMD_MSE_PARITY"
    echo "[after repo]  $IGNORED_CMD_MNIST"
    echo "[after repo]  $IGNORED_CMD_COMMAND_BATCHING"
    echo "[after repo]  $IGNORED_CMD_COMMAND_BATCHING_BENCH"
    echo "[after repo]  $IGNORED_CMD_GEMM_RESIDENT_PARITY"
    echo "[after repo]  $IGNORED_CMD_STORE_PARITY"
    echo "[after repo]  AB_BEFORE_FACADE_PATH=\$BEFORE_REPO/crates/facade AB_AFTER_FACADE_PATH=\$AFTER_REPO/crates/facade bash scripts/bench/framework-compare/run_ab_mse_encode_metal.sh 1691"
    exit 0
fi

if [ -z "$BEFORE_REPO" ] || [ -z "$AFTER_REPO" ]; then
    echo "error: MSE_AB_BEFORE_REPO / MSE_AB_AFTER_REPO must be set (absolute paths)" >&2
    exit 1
fi
case "$BEFORE_REPO" in
    /*) ;;
    *) echo "error: MSE_AB_BEFORE_REPO must be an absolute path" >&2; exit 1 ;;
esac
case "$AFTER_REPO" in
    /*) ;;
    *) echo "error: MSE_AB_AFTER_REPO must be an absolute path" >&2; exit 1 ;;
esac
if [ ! -f "$BEFORE_REPO/Cargo.toml" ]; then
    echo "error: $BEFORE_REPO/Cargo.toml not found" >&2
    exit 1
fi
if [ ! -f "$AFTER_REPO/Cargo.toml" ]; then
    echo "error: $AFTER_REPO/Cargo.toml not found" >&2
    exit 1
fi

# BEFORE_REPO 配置手順（`docs/perf/logs/cuda-mse-backward-1692/orchestrate.sh`
# と同型の防御的コピー）: (d) で使う `mse_backward_bench.rs` は
# `crates/facade/tests/mse_backward_bench.rs`。84490ad1 時点で既に存在する
# ことを確認済み（`git diff 84490ad1 HEAD -- crates/facade/tests/
# mse_backward_bench.rs` は無差分）だが、将来 BEFORE_REPO がこのファイル
# 導入前のコミットを指すケースに備え、未配置の場合のみ AFTER_REPO 側の
# 内容をコピーして補う（既に配置済みなら上書きしない）。
BENCH_REL_PATH="crates/facade/tests/mse_backward_bench.rs"
if [ ! -f "$AFTER_REPO/$BENCH_REL_PATH" ]; then
    echo "error: $AFTER_REPO/$BENCH_REL_PATH not found" >&2
    exit 1
fi
if [ ! -f "$BEFORE_REPO/$BENCH_REL_PATH" ]; then
    echo "info: $BEFORE_REPO/$BENCH_REL_PATH が未配置のためコピーします" >&2
    mkdir -p "$(dirname "$BEFORE_REPO/$BENCH_REL_PATH")"
    cp "$AFTER_REPO/$BENCH_REL_PATH" "$BEFORE_REPO/$BENCH_REL_PATH"
fi

echo "=== env_info (before) ==="
{
    uname -srm
    sw_vers
    rustc -V
    sysctl machdep.cpu.brand_string
    git -C "$BEFORE_REPO" rev-parse HEAD
} >"$SCRIPT_DIR/env_info_before.txt" 2>&1
cat "$SCRIPT_DIR/env_info_before.txt"

echo "=== env_info (after) ==="
{
    uname -srm
    sw_vers
    rustc -V
    sysctl machdep.cpu.brand_string
    git -C "$AFTER_REPO" rev-parse HEAD
} >"$SCRIPT_DIR/env_info_after.txt" 2>&1
cat "$SCRIPT_DIR/env_info_after.txt"

uptime >"$SCRIPT_DIR/uptime_before.txt"

echo "=== (a) REQ-2 正しさ確認（after 腕・mse_parity） ==="
MSE_PARITY_OK=1
if ! (cd "$AFTER_REPO" && eval "$MSE_PARITY_CMD") >"$SCRIPT_DIR/mse_parity_after.log" 2>&1; then
    MSE_PARITY_OK=0
    echo "WARNING: mse_parity (after) failed; see $SCRIPT_DIR/mse_parity_after.log" >&2
fi
tail -20 "$SCRIPT_DIR/mse_parity_after.log"

echo "=== (b) bit 同一確認（before 腕） ==="
(cd "$BEFORE_REPO" && eval "$BIT_DUMP_CMD") >"$SCRIPT_DIR/bitdump_before.log" 2>&1
tail -5 "$SCRIPT_DIR/bitdump_before.log"

echo "=== (b) bit 同一確認（after 腕） ==="
(cd "$AFTER_REPO" && eval "$BIT_DUMP_CMD") >"$SCRIPT_DIR/bitdump_after.log" 2>&1
tail -5 "$SCRIPT_DIR/bitdump_after.log"

grep -E '^(step\[|final\.param\[)' "$SCRIPT_DIR/bitdump_before.log" >"$SCRIPT_DIR/bitdump_before_filtered.txt" || true
grep -E '^(step\[|final\.param\[)' "$SCRIPT_DIR/bitdump_after.log" >"$SCRIPT_DIR/bitdump_after_filtered.txt" || true

# 件数・項目集合の独立検証（`docs/perf/logs/metal-dinput-sync-1563/
# orchestrate.sh` と同型。fail-closed。誤って bit-identical と報告しない）。
BITDUMP_COUNT_BEFORE=$(wc -l <"$SCRIPT_DIR/bitdump_before_filtered.txt" | tr -d ' ')
BITDUMP_COUNT_AFTER=$(wc -l <"$SCRIPT_DIR/bitdump_after_filtered.txt" | tr -d ' ')
sed -E 's/ = .*$//' "$SCRIPT_DIR/bitdump_before_filtered.txt" | sort >"$SCRIPT_DIR/bitdump_before_labels.txt"
sed -E 's/ = .*$//' "$SCRIPT_DIR/bitdump_after_filtered.txt" | sort >"$SCRIPT_DIR/bitdump_after_labels.txt"
diff -u "$SCRIPT_DIR/bitdump_before_labels.txt" "$SCRIPT_DIR/bitdump_after_labels.txt" \
    >"$SCRIPT_DIR/bitdump_label_diff.txt" 2>&1 || true

BITDUMP_COUNT_OK=1
if [ "$BITDUMP_COUNT_BEFORE" -eq 0 ] || [ "$BITDUMP_COUNT_AFTER" -eq 0 ]; then
    echo "WARNING: bitdump filtered output is empty (before=$BITDUMP_COUNT_BEFORE after=$BITDUMP_COUNT_AFTER); grep may have failed to extract any line" >&2
    BITDUMP_COUNT_OK=0
fi
if [ "$BITDUMP_COUNT_BEFORE" -ne "$EXPECTED_BITDUMP_LINES" ] || [ "$BITDUMP_COUNT_AFTER" -ne "$EXPECTED_BITDUMP_LINES" ]; then
    echo "WARNING: bitdump filtered line count differs from expected ($EXPECTED_BITDUMP_LINES): before=$BITDUMP_COUNT_BEFORE after=$BITDUMP_COUNT_AFTER" >&2
    BITDUMP_COUNT_OK=0
fi
if [ -s "$SCRIPT_DIR/bitdump_label_diff.txt" ]; then
    echo "WARNING: bitdump_label_diff.txt is non-empty (item set differs between arms). See $SCRIPT_DIR/bitdump_label_diff.txt" >&2
    BITDUMP_COUNT_OK=0
fi

diff -u "$SCRIPT_DIR/bitdump_before_filtered.txt" "$SCRIPT_DIR/bitdump_after_filtered.txt" \
    >"$SCRIPT_DIR/bitdump_diff.txt" 2>&1 || true
BITDUMP_OK=1
if [ "$BITDUMP_COUNT_OK" -eq 0 ]; then
    BITDUMP_OK=0
    echo "bit dump: UNDETERMINED（件数・項目集合の検証に失敗したため bit 同一の判定を確定できない。$SCRIPT_DIR/bitdump_diff.txt・bitdump_label_diff.txt を参照）" >&2
elif [ -s "$SCRIPT_DIR/bitdump_diff.txt" ]; then
    BITDUMP_OK=0
    echo "WARNING: bitdump_diff.txt is non-empty (bit mismatch detected). See $SCRIPT_DIR/bitdump_diff.txt" >&2
else
    echo "bit dump: $BITDUMP_COUNT_BEFORE/$BITDUMP_COUNT_AFTER 行・0 diff lines (bit-identical)"
fi

echo "=== (c) カウンタ実測（after 腕。hard assert 11/7/7 期待） ==="
(cd "$AFTER_REPO" && eval "$BATCH_COUNTERS_CMD") >"$SCRIPT_DIR/batch_counters_after.log" 2>&1
tail -20 "$SCRIPT_DIR/batch_counters_after.log"
(cd "$AFTER_REPO" && eval "$BACKWARD_PHASE_CMD") >"$SCRIPT_DIR/backward_phase_after.log" 2>&1
tail -20 "$SCRIPT_DIR/backward_phase_after.log"

echo "=== (c) カウンタ実測（before 腕。11/8/8 再現確認） ==="
(cd "$BEFORE_REPO" && eval "$BATCH_COUNTERS_CMD") >"$SCRIPT_DIR/batch_counters_before.log" 2>&1
tail -20 "$SCRIPT_DIR/batch_counters_before.log"
(cd "$BEFORE_REPO" && eval "$BACKWARD_PHASE_CMD") >"$SCRIPT_DIR/backward_phase_before.log" 2>&1
tail -20 "$SCRIPT_DIR/backward_phase_before.log"

echo "=== (d) backward マイクロベンチ 5 round・起動順反転 ==="
# `set -eu` 下では「サブシェル呼び出し + リダイレクト」だけの単純コマンドが
# 非 0 終了すると、続く `xxx_rc=$?` 行の実行前に `set -e` がスクリプトを
# 即座に終了させてしまい、対をなす腕の実行や rc 判定・rounds.log 出力が
# スキップされる（Bugbot 指摘・#1691 レビュー是正）。各コマンドの周囲だけ
# `set +e`/`set -e` で挟み、rc を確実に捕捉してから通常の fail-closed 判定
# （下の if 節）へ渡す。
for round in $(seq 1 "$ROUNDS"); do
    echo "-- round ${round}/${ROUNDS} --"
    set +e
    if [ $((round % 2)) -eq 1 ]; then
        order="before_first"
        (cd "$BEFORE_REPO" && eval "$MSE_BENCH_CMD") >"$SCRIPT_DIR/before_round${round}.log" 2>&1
        before_rc=$?
        (cd "$AFTER_REPO" && eval "$MSE_BENCH_CMD") >"$SCRIPT_DIR/after_round${round}.log" 2>&1
        after_rc=$?
    else
        order="after_first"
        (cd "$AFTER_REPO" && eval "$MSE_BENCH_CMD") >"$SCRIPT_DIR/after_round${round}.log" 2>&1
        after_rc=$?
        (cd "$BEFORE_REPO" && eval "$MSE_BENCH_CMD") >"$SCRIPT_DIR/before_round${round}.log" 2>&1
        before_rc=$?
    fi
    set -e
    echo "round=${round} order=${order} before_rc=${before_rc} after_rc=${after_rc}" >>"$SCRIPT_DIR/rounds.log"
    if [ "$before_rc" -ne 0 ] || [ "$after_rc" -ne 0 ]; then
        echo "エラー: round ${round} で非 0 終了（before_rc=${before_rc} after_rc=${after_rc}）" >&2
        exit 1
    fi
done

AGGREGATE_OK=1
if ! python3 "$SCRIPT_DIR/aggregate.py" "$SCRIPT_DIR" "$ROUNDS" >"$SCRIPT_DIR/aggregate.md" 2>&1; then
    AGGREGATE_OK=0
    echo "WARNING: aggregate.py exited non-zero (regression ratio>1.00 or grad mismatch); see $SCRIPT_DIR/aggregate.md" >&2
fi
cat "$SCRIPT_DIR/aggregate.md"

echo "=== (f) #[ignore] 群非後退確認（after 腕） ==="
(cd "$AFTER_REPO" && eval "$IGNORED_CMD_MSE_PARITY") >"$SCRIPT_DIR/ignored_after_mse_parity.log" 2>&1
tail -20 "$SCRIPT_DIR/ignored_after_mse_parity.log"
(cd "$AFTER_REPO" && eval "$IGNORED_CMD_MNIST") >"$SCRIPT_DIR/ignored_after_mnist.log" 2>&1
tail -20 "$SCRIPT_DIR/ignored_after_mnist.log"
(cd "$AFTER_REPO" && eval "$IGNORED_CMD_COMMAND_BATCHING") >"$SCRIPT_DIR/ignored_after_command_batching.log" 2>&1
tail -20 "$SCRIPT_DIR/ignored_after_command_batching.log"
(cd "$AFTER_REPO" && eval "$IGNORED_CMD_COMMAND_BATCHING_BENCH") >"$SCRIPT_DIR/ignored_after_command_batching_bench.log" 2>&1
tail -20 "$SCRIPT_DIR/ignored_after_command_batching_bench.log"
(cd "$AFTER_REPO" && eval "$IGNORED_CMD_GEMM_RESIDENT_PARITY") >"$SCRIPT_DIR/ignored_after_gemm_resident_parity.log" 2>&1
tail -20 "$SCRIPT_DIR/ignored_after_gemm_resident_parity.log"
(cd "$AFTER_REPO" && eval "$IGNORED_CMD_STORE_PARITY") >"$SCRIPT_DIR/ignored_after_device_param_store_backend_parity.log" 2>&1
tail -20 "$SCRIPT_DIR/ignored_after_device_param_store_backend_parity.log"

echo "=== (e) A/B（framework-compare train。5 round・record_only） ==="
mkdir -p "$SCRIPT_DIR/ab"
AB_OK=1
if ! (
    cd "$AFTER_REPO/scripts/bench/framework-compare" && \
    AB_BEFORE_FACADE_PATH="$BEFORE_REPO/crates/facade" \
    AB_AFTER_FACADE_PATH="$AFTER_REPO/crates/facade" \
    bash run_ab_mse_encode_metal.sh 1691
) >"$SCRIPT_DIR/ab/run_ab_1691.log" 2>&1; then
    AB_OK=0
    echo "WARNING: A/B script exited non-zero (regression, checksum mismatch, or undetermined); see $SCRIPT_DIR/ab/run_ab_1691.log" >&2
fi
tail -40 "$SCRIPT_DIR/ab/run_ab_1691.log"
# A/B 生成物（results/raw 配下・compare-train-1691*.md 等）はコピーせず
# `$AFTER_REPO/scripts/bench/framework-compare/` 配下に残す（README
# 「生成物の所在」参照）。

uptime >"$SCRIPT_DIR/uptime_after.txt"

# 総合判定（README「事前登録判定規則」の総合判定節）: (a)(b)(f) pass かつ
# (c) 一致かつ (d)(e) すべて ratio<=1.00 → ADOPT（終了コード 0）。
# (e) または (d) に ratio>1.00、あるいは実機到達不能・件数検証失敗による
# UNDETERMINED → REJECT/UNDETERMINED のいずれでも非 0 終了とする
# （codex-review [P1] 指摘対応: 判定失敗・判定不能を WARNING 表示のみで
# 終了コード 0 のまま握り潰さない）。(c)(f) は非 0 終了時に `set -eu` が
# 即座にスクリプトを中断させるため、ここに到達した時点で既に pass 確定。
# (d) の round 単位失敗も同様に既に exit 1 済み。
OVERALL_OK=1
[ "$MSE_PARITY_OK" -eq 1 ] || OVERALL_OK=0
[ "$BITDUMP_OK" -eq 1 ] || OVERALL_OK=0
[ "$AGGREGATE_OK" -eq 1 ] || OVERALL_OK=0
[ "$AB_OK" -eq 1 ] || OVERALL_OK=0

if [ "$OVERALL_OK" -eq 1 ]; then
    echo "DONE（総合判定: ADOPT 相当。(a)(b)(c)(d)(e)(f) すべて pass・非後退）"
else
    echo "DONE（総合判定: NOT ADOPT。mse_parity_ok=$MSE_PARITY_OK bitdump_ok=$BITDUMP_OK aggregate_ok=$AGGREGATE_OK ab_ok=$AB_OK のいずれかが 0。詳細は各ログ・aggregate.md・ab/run_ab_1691.log を参照）" >&2
    exit 1
fi
