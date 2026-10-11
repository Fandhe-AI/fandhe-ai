#!/usr/bin/env bash
# macOS／Linux 両ターゲットの cargo build 検証（make build-cross。TASK-2.1b・イシュー #50）。
# macOS runner 未登録の代替として aarch64-apple-darwin へのクロスターゲットビルドで
# Metal 有効経路（cfg(target_os = "macos")）のコンパイルを検証する。クロス側は --lib のみ
# （bin のリンクには macOS のリンカ・SDK が要り Linux/コンテナには無い。ci.yml の build ジョブと同じ）。
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./lib.sh
source "${DIR}/lib.sh"

reject_args "$(basename "$0")" "$@"

ensure_rust_target aarch64-apple-darwin
cargo build --workspace --locked
cargo build --workspace --locked --target aarch64-apple-darwin --lib
