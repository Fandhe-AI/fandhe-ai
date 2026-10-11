#!/usr/bin/env bash
# lefthook の git hooks を導入する（make hooks）。未導入なら lefthook 本体も導入する。
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./lib.sh
source "${DIR}/lib.sh"

reject_args "$(basename "$0")" "$@"

if ! command -v lefthook >/dev/null 2>&1; then
  echo "lefthook を導入します"
  if command -v brew >/dev/null 2>&1; then
    brew install lefthook
  elif command -v cargo >/dev/null 2>&1; then
    cargo install lefthook --locked
  else
    echo "brew / cargo が見つかりません。https://lefthook.dev/installation/ を参照してください" >&2
    exit 1
  fi
fi
lefthook install
