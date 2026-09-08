#!/usr/bin/env bash
# イシュー #1312: TwoDDynamic vs RowPanel 両実機 A/B 計測の共通ドライバ。
# DGX Spark GB10・Apple M4 Max の両方から同一スクリプトを呼ぶ（実機固有の
# 差異は呼び出し側が渡す環境変数のみで吸収し、スクリプト自体は機種非依存）。
#
# 環境変数（すべて呼び出し側指定必須／明示デフォルトあり）:
#   WORKDIR          cargo プロジェクトルート（既定: カレントディレクトリ）
#   CARGO_TARGET_DIR ビルドキャッシュ先（既定: cargo 既定の ./target）
#   LOGDIR           出力ログ先（必須）
#   MACHINE           ログファイル名に埋め込む機種ラベル（dgx / m4max。必須）
#   THREADS          スペース区切りのスレッド数リスト。"default" は
#                    RAYON_NUM_THREADS 未設定（機種既定値）を意味する（必須）
#   RUNS             1 スレッド設定あたりの独立プロセス実行回数（既定 5）
set -uo pipefail

WORKDIR="${WORKDIR:-$(pwd)}"
cd "$WORKDIR"
LOGDIR="${LOGDIR:?LOGDIR required}"
MACHINE="${MACHINE:?MACHINE required}"
THREADS="${THREADS:?THREADS required}"
RUNS="${RUNS:-5}"
mkdir -p "$LOGDIR"

run_test() {
  local test_name="$1" out_prefix="$2" thread_label="$3" thread_arg="$4"
  local run out
  for run in $(seq 1 "$RUNS"); do
    out="$LOGDIR/${out_prefix}-${MACHINE}-T${thread_label}-run${run}.txt"
    if [ -n "$thread_arg" ]; then
      RAYON_NUM_THREADS="$thread_arg" cargo test -p fandhe-ai-backend-cpu --release --lib \
        -- --ignored "$test_name" --nocapture >"$out" 2>&1
    else
      cargo test -p fandhe-ai-backend-cpu --release --lib \
        -- --ignored "$test_name" --nocapture >"$out" 2>&1
    fi
    echo "  wrote $out (rc=$?)"
  done
}

for T in $THREADS; do
  if [ "$T" = "default" ]; then
    thread_arg=""
  else
    thread_arg="$T"
  fi
  echo "=== THREADS=$T ($(date -u +%FT%TZ)) ==="
  run_test gemm_blis_two_d_dynamic_ab_1024_2048 ab-1024-2048 "$T" "$thread_arg"
  run_test gemm_blis_two_d_dynamic_ab_4096 ab-4096 "$T" "$thread_arg"
done

echo "run_ab.sh done $(date -u +%FT%TZ)"
