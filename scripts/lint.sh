#!/usr/bin/env bash
# cargo clippy -D warnings（make lint）。
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./lib.sh
source "${DIR}/lib.sh"

reject_args "$(basename "$0")" "$@"

cargo clippy --workspace --all-targets --all-features -- -D warnings
# イシュー #2299: `--workspace --all-features` だけでは backend-cuda の
# `internal-diagnostics` feature が常に有効化されるため、feature 無効（既定ビルド）時のみ
# 表面化する dead code をローカルで再現できない（ci.yml `build-no-cuda-toolkit` ジョブと
# 同じ理由）。CI と同じ条件をローカルでも再現するため 1 行追加する。
cargo clippy -p fandhe-ai-backend-cuda --all-targets -- -D warnings
