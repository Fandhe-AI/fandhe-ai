#!/usr/bin/env bash
# scripts/*.sh が共有するヘルパー。直接実行せず source して使う:
#   DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#   source "${DIR}/lib.sh"
#
# source した時点でリポジトリルートへ cd する（`make <target>` でも `scripts/<name>.sh` の
# 直接実行でも同じ cwd で動かすため）。呼び出し元は `set -euo pipefail` を引き継ぎ、
# 失敗した終了コードはそのまま呼び出し元（人・make・CI・lefthook）へ伝播する。
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC2034 # source した側のスクリプトで参照する
readonly ROOT_DIR
cd "${ROOT_DIR}"

# make rustup 直後（同一シェルの PATH 未反映）でも cargo を拾えるよう ~/.cargo/bin を足す。
export PATH="${HOME}/.cargo/bin:${PATH}"

# log <message...>: 呼び出し元スクリプト名付きで stderr へ出す。
log() {
  echo "[$(basename "$0")] $*" >&2
}

# die <message...>: メッセージを出して終了コード 1 で止める。
die() {
  log "$*"
  exit 1
}

# reject_args <script-name> "$@": 引数を取らないスクリプトで未知の入力を黙って無視せず、
# usage を出して終了コード 2（誤用）で止める。
reject_args() {
  local script_name="$1"
  shift
  if [ "$#" -ne 0 ]; then
    echo "usage: ${script_name} (no arguments)" >&2
    exit 2
  fi
}

# has_cargo: Cargo.toml があるか（CI の detect ステップと同一の判定）。
has_cargo() {
  [ -f Cargo.toml ]
}

# ensure_rust_target <triple>: rustup target を未導入の場合のみ追加する（冪等）。
ensure_rust_target() {
  rustup target list --installed | grep -qx "$1" || rustup target add "$1"
}
