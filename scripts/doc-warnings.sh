#!/usr/bin/env bash
# rustdoc 警告ゲート（make doc-warnings。イシュー #883）。ci.yml の build ジョブの
# rustdoc 2 step とローカルで同一コマンドを共用する。cfg（aarch64 NEON vs x86 AVX・
# macOS 限定モジュール）により警告集合がターゲットごとに異なるため Linux ホスト分と
# aarch64-apple-darwin クロス分の両方をゲートする。3 行目（イシュー #2289）は
# backend-cuda の internal-diagnostics feature 有効時の公開面を別途検査する
# （docs/crates-io-publishing-order.md §8.5）。
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./lib.sh
source "${DIR}/lib.sh"

reject_args "$(basename "$0")" "$@"

ensure_rust_target aarch64-apple-darwin
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked
RUSTDOCFLAGS="-D warnings" cargo doc -p fandhe-ai-backend-metal -p fandhe-ai-backend-cpu --no-deps --locked --target aarch64-apple-darwin
RUSTDOCFLAGS="-D warnings" cargo doc -p fandhe-ai-backend-cuda --all-features --no-deps --locked
