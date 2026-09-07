#!/usr/bin/env bash
#
# 単一 GPU 環境（DGX Spark GB10 等）向けの CUDA Graph step capture
# bit 同一性・自動比較ハーネス（codex-review P2 指摘対応・PR #1390。
# イシュー #1349）。
#
# 背景: crates/facade/tests/cuda_graph_step_bit_identity.rs の
# eager_baseline／graph_capture は、それぞれ「非 capture 経路（opt-in
# OFF）」「capture 経路（opt-in ON）」の loss・各 step 完了直後の
# パラメータ・最終パラメータをビット表現で標準出力へ印字するのみで、
# 自動比較は行わない（opt-in はプロセス内最初の CUDA デバイス初期化
# より前に固定する必要があるため、同一プロセス内で両経路を切り替える
# ことができず、2 プロセス構成にせざるを得ない。同ファイル冒頭コメント
# 参照）。2 GPU 機械比較（cuda_graph_step_two_gpu_bit_identity.rs）は
# 単一 GPU 環境では device_count() < 2 のため早期 return し成立しない。
#
# 本スクリプトは、その「2 プロセスの標準出力を目視・スクリプトで比較」
# という手作業部分を自動化する: 同一 GPU 上で eager_baseline
# （opt-in OFF）・graph_capture（opt-in ON）を順に別プロセスとして実行し、
# 両者が出力する `step[...].loss.bits` / `step[...].param[...][...].bits`
# / `final.param[...][...].bits` 行列（ラベル行 `=== ... ===` を除く）を
# 完全一致するか diff で検証する。不一致があれば非ゼロ終了・diff を
# 表示する（fail-closed。CI では実行しない — 実機〈CUDA〉必須のため
# 通常 CI ジョブの対象外。.claude/rules/ci.md「実機依存」節）。
#
# 実行方法（DGX Spark GB10 等 CUDA 実機。docs/real-hardware-
# verification-env.md の手順に従う。事前に cargo test --release
# --no-run でビルド済みにしておくとテスト自体の初回コンパイル待ちを
# 避けられるが必須ではない）:
#
#   scripts/verify-cuda-graph-step-bit-identity.sh
#
# 終了コード: 0 = bit 同一（一致）。1 = 不一致または実行時エラー。
set -euo pipefail

PACKAGE="fandhe-ai"
TEST_BIN="cuda_graph_step_bit_identity"

# 印字される行のうち機械比較対象のみを抽出する（cargo test 自体が出す
# "running 1 test" / "test <name> ... ok" / "test result: ..." 等の
# ノイズ行、および `=== ... ===` ラベル行（opt-in の有無で文言が異なり
# 意図的に不一致になるため比較対象から除く）を除外する。
extract_bits_lines() {
    grep -E '^(step\[|final\.param\[)'
}

run_variant() {
    local exact_fn="$1"
    local optin="$2"

    if [[ -n "$optin" ]]; then
        FANDHE_AI_CUDA_GRAPH_STEP="$optin" \
            cargo test -p "$PACKAGE" --release --test "$TEST_BIN" \
            -- --ignored --nocapture --exact "$exact_fn"
    else
        cargo test -p "$PACKAGE" --release --test "$TEST_BIN" \
            -- --ignored --nocapture --exact "$exact_fn"
    fi
}

# Cursor Bugbot 指摘対応（PR #1390）: `set -euo pipefail` 下で
# `var="$(run_variant ...)"`（`local` を伴わない top-level 代入）は、
# `run_variant`（＝ `cargo test`）が非ゼロ終了すると代入コマンド自体の
# 終了ステータスがそのまま非ゼロになり、`set -e` により後続の診断出力
# （捕捉した raw 出力の印字）へ到達する前にスクリプトが即終了して
# しまう（失敗原因が分からなくなる欠陥）。`A && B || C` の 1 個の複合
# コマンドとして終了ステータスを明示的に変数へ退避することで、
# `set -e` を発火させずに失敗を検出し、raw 出力を必ず印字してから
# `exit 1` する。
echo "[1/2] eager_baseline（opt-in OFF）を実行する..." >&2
eager_raw="$(run_variant eager_baseline "")" && eager_rc=0 || eager_rc=$?
if [[ "$eager_rc" -ne 0 ]]; then
    echo "エラー: eager_baseline（opt-in OFF）の cargo test が失敗した（exit ${eager_rc}）" >&2
    printf '%s\n' "$eager_raw" >&2
    exit 1
fi
eager_bits="$(printf '%s\n' "$eager_raw" | extract_bits_lines)"

if [[ -z "$eager_bits" ]]; then
    echo "エラー: eager_baseline の出力からビット行を抽出できなかった（テスト自体が失敗した可能性）" >&2
    printf '%s\n' "$eager_raw" >&2
    exit 1
fi

echo "[2/2] graph_capture（opt-in ON）を実行する..." >&2
graph_raw="$(run_variant graph_capture "1")" && graph_rc=0 || graph_rc=$?
if [[ "$graph_rc" -ne 0 ]]; then
    echo "エラー: graph_capture（opt-in ON）の cargo test が失敗した（exit ${graph_rc}）" >&2
    printf '%s\n' "$graph_raw" >&2
    exit 1
fi
graph_bits="$(printf '%s\n' "$graph_raw" | extract_bits_lines)"

if [[ -z "$graph_bits" ]]; then
    echo "エラー: graph_capture の出力からビット行を抽出できなかった（テスト自体が失敗した可能性）" >&2
    printf '%s\n' "$graph_raw" >&2
    exit 1
fi

# codex-review P2 指摘対応（PR #1390）: 従来は `diff ... || true` で
# diff の終了コードを常に握りつぶしており、diff 自体の実行エラー
# （diff 未インストール等・終了コード 2 以上）も「bit 同一」（exit 0）
# として扱ってしまっていた。diff の終了コードは仕様上 0（一致）／
# 1（不一致）／2 以上（実行エラー）の 3 値を持つため、`eager_rc` と
# 同じ「複合コマンドで終了ステータスを明示退避」する方式で 3 値を
# 区別する。
diff_output="$(diff <(printf '%s\n' "$eager_bits") <(printf '%s\n' "$graph_bits"))" && diff_rc=0 || diff_rc=$?

if [[ "$diff_rc" -eq 0 ]]; then
    line_count="$(printf '%s\n' "$eager_bits" | wc -l | tr -d ' ')"
    echo "OK: eager_baseline（opt-in OFF）と graph_capture（opt-in ON）は ${line_count} 行すべて bit 同一" >&2
    exit 0
elif [[ "$diff_rc" -eq 1 ]]; then
    echo "NG: eager_baseline（opt-in OFF）と graph_capture（opt-in ON）の出力が bit 同一でない" >&2
    printf '%s\n' "$diff_output" >&2
    exit 1
else
    echo "エラー: diff の実行自体が失敗した（exit ${diff_rc}）。bit 同一性は判定不能" >&2
    printf '%s\n' "$diff_output" >&2
    exit "$diff_rc"
fi
