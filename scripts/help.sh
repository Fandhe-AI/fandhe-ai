#!/usr/bin/env bash
# Make を介さない操作一覧。`make help` は Makefile の `## ` コメントから機械的に生成される
# のに対し、本ファイルは手動で保守する一覧であり、両者を自動で同期する仕組みは無い。
# Makefile のターゲットを追加・改名・削除したら同じ変更で本ファイルも更新すること。
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./lib.sh
source "${DIR}/lib.sh"

reject_args "$(basename "$0")" "$@"

cat <<'EOT'
このリポジトリの操作と、Make を使わずに直接実行する方法:

環境構築
  setup                       scripts/setup.sh                          サブモジュール → rustup → lefthook の一括構築
  rustup                      scripts/rustup.sh                         rustup を未導入の場合のみ導入
  submodule                   git submodule update --init               docs/spec サブモジュールの初期化・更新
  hooks                       scripts/hooks.sh                          lefthook の git hooks を導入

品質チェック
  fmt / fmt-check             cargo fmt --all [--check]                 整形 / 整形差分の検出
  lint                        scripts/lint.sh                           cargo clippy -D warnings
  test                        cargo test --workspace --all-features     実機依存の #[ignore] テストは除く
  test-ignored                cargo test --workspace --all-features -- --ignored --nocapture
  test-ignored-cuda           scripts/test-ignored.sh cuda              CUDA 実機専用（release）
  test-ignored-metal          cargo test -p fandhe-ai-backend-metal --release --all-features -- --ignored --nocapture
  test-ignored-metal-facade   scripts/test-ignored.sh metal-facade      Metal 実機専用: facade 契約テスト（release）
  build-cross                 scripts/build-cross.sh                    Linux / aarch64-apple-darwin の cargo build
  build-no-cuda               scripts/build-no-cuda.sh                  CUDA toolkit／ROCm 非搭載でのビルド・実行成立検証
  check-cross-metal-tests     scripts/check-cross.sh metal-tests        backend-metal の #[ignore] テスト型検査
  check-cross-cpu-tests       scripts/check-cross.sh cpu-tests          backend-cpu の cfg(test) 限定コード型検査
  check-cross-rocm-tests      scripts/check-cross.sh rocm-tests         backend-rocm の cfg 分岐を macOS ターゲットで型検査
  check-cross-windows-interop scripts/check-cross.sh windows-interop    Windows ターゲット clippy
  doc-warnings                scripts/doc-warnings.sh                   rustdoc 警告ゲート
  deny                        scripts/deny.sh                           cargo-deny 依存監査
  deps-forbidden              scripts/ci-checks.sh deps-forbidden       依存禁止リストの混入検査
  runner-policy               scripts/ci-checks.sh runner-policy        self-hosted 逆戻り防止検査
  guardrail-regression        scripts/ci-checks.sh guardrail-regression guardrail 2 層検証
  verification-gates          scripts/ci-checks.sh verification-gates   検証ゲート（build/test/clippy）
  verification-gate-bench     scripts/ci-checks.sh verification-gate-bench  bench ゲート単体
  startup-bench               scripts/bench-harness.sh startup          起動コスト計測（BACKEND・TRIALS）
  peak-memory-bench           scripts/bench-harness.sh peak-memory      GEMM ピークメモリ計測（BACKEND・TRIALS・SIZE）
  ci                          上記チェックの一括実行（Makefile の ci 依存順）

Docker
  docker-build                docker compose build
  docker-shell                docker compose run --rm dev
  docker-ci                   docker compose run --rm dev make ci

help                          scripts/help.sh                           この一覧を表示する

`make <target>` は上記と同じコマンドを呼ぶだけの薄い入口であり、
処理の定義は scripts/ 側にのみ存在する。
EOT
