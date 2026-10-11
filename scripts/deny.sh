#!/usr/bin/env bash
# cargo deny check advisories bans licenses sources（make deny。依存監査。#353）。
# cargo-deny 未導入なら自動導入する。Cargo.toml または deny.toml が無ければスキップする。
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./lib.sh
source "${DIR}/lib.sh"

reject_args "$(basename "$0")" "$@"

if ! has_cargo || [ ! -f deny.toml ]; then
  echo "skip: Cargo.toml または deny.toml 未追加のため deny をスキップ"
  exit 0
fi
if ! command -v cargo-deny >/dev/null 2>&1; then
  echo "cargo-deny を導入します"
  cargo install cargo-deny --locked
fi
cargo deny --locked check advisories bans licenses sources
