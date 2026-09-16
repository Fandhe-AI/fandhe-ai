#!/usr/bin/env bash
# イシュー #1585 Layer B（H2D 単体マイクロ A/B）の実行スクリプト。
#
# `crates/backend-cuda/tests/pinned_h2d_upload_ab_1585.rs`
# （`pinned_h2d_upload_ab`。#[ignore] 実機テスト）を 5 プロセス起動
# （5 回独立プロセス）で実行し、`layer_b_run{1..5}.log` へ出力する。
# `aggregate.py` がこのログ群を集計して `aggregate.md` を書く
# （`docs/perf/cuda-h2d-pinned-staging.md` §3 の事前登録規則）。
#
# `CARGO_TARGET_DIR` は呼び出し元の環境変数をそのまま使う（本スクリプト
# では固定パスを書かない。未設定ならビルド標準の `target/` を使う）。
# 内部ホスト名・絶対パスはログへ書かない方針（`docs/real-hardware-
# verification-env.md`）に従い、本スクリプト自体もホスト名へ言及しない。
#
# 実行例（DGX Spark GB10 実機上のリポジトリルートから）:
#   ./docs/perf/logs/cuda-h2d-pinned-staging-1585/run_layer_b.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../../../.." && pwd)"

cd "${REPO_ROOT}"

TEST_CMD=(
  cargo test -p fandhe-ai-backend-cuda --release --features internal-diagnostics
  --test pinned_h2d_upload_ab_1585 -- --ignored --nocapture --test-threads=1
)

echo "=== uptime (before) ===" | tee "${HERE}/uptime_before.txt"
uptime | tee -a "${HERE}/uptime_before.txt"

for i in 1 2 3 4 5; do
  echo "=== run ${i}/5 ===" >&2
  "${TEST_CMD[@]}" 2>&1 | tee "${HERE}/layer_b_run${i}.log"
done

echo "=== uptime (after) ===" | tee "${HERE}/uptime_after.txt"
uptime | tee -a "${HERE}/uptime_after.txt"

python3 "${HERE}/aggregate.py" "${HERE}" 5
