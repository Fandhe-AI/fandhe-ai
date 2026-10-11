#!/usr/bin/env bash
# 開発環境の一括構築（make setup）。サブモジュール → rustup → lefthook の順を保証する。
# 依存ターゲット並記だと make -j 実行時に順序が保証されず、cargo フォールバックを持つ hooks が
# rustup より先に走りうるため、順序をこのスクリプトで直列に固定する。
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./lib.sh
source "${DIR}/lib.sh"

reject_args "$(basename "$0")" "$@"

git submodule update --init
"${DIR}/rustup.sh"
"${DIR}/hooks.sh"
echo "setup 完了"
