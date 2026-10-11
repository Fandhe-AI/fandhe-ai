#!/usr/bin/env bash
# 計測ハーネスの実行面（make startup-bench / peak-memory-bench）。CI には組み込まない。
# usage: bench-harness.sh <startup|peak-memory>
#
# 入力は環境変数 BACKEND（cpu|cuda|metal。既定 cpu）・TRIALS（数値。既定 5）・
# SIZE（peak-memory のみ数値。既定 4096）。Makefile 側は値を `export` して環境変数経由で
# 渡す: make の $(VAR) 展開はレシピ本文へのテキスト置換であり、値に `"` や `;` 等の
# シェル構文を含めるとコマンド注入が成立する（PR #360 codex-review P1）。本スクリプトは
# 環境変数をシェル変数として読み、許可リスト／数値のみを検証してから cargo へ二重引用符で
# 渡す。実機限定の cuda／metal は実機でのみ使う。
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./lib.sh
source "${DIR}/lib.sh"

BACKEND="${BACKEND:-cpu}"
TRIALS="${TRIALS:-5}"
SIZE="${SIZE:-4096}"

require_numeric() {
  case "$2" in
    ''|*[!0-9]*) echo "$1 は数値を指定する（指定値: ${2}）" >&2; exit 1 ;;
    *) ;;
  esac
}

case "$BACKEND" in
  cpu|cuda|metal) ;;
  *) echo "BACKEND は cpu|cuda|metal のいずれかを指定する（指定値: ${BACKEND}）" >&2; exit 1 ;;
esac
require_numeric TRIALS "$TRIALS"

case "${1:-}" in
  startup)
    # 起動コスト計測（TASK-13.1a・イシュー #170）。実測は兄弟イシュー #171 のスコープ。
    cargo build -p bench-harness --release --bins
    cargo run -p bench-harness --release --bin startup_bench -- --backend "$BACKEND" --trials "$TRIALS"
    ;;
  peak-memory)
    # GEMM ピークメモリ計測（TASK-14.2a・イシュー #178）。REQ-14 代表ワークロード
    # （M=N=K=4096）は数百 MB〜数 GB を確保するため CI に組み込まない
    # （CI のスモークは crates/bench-harness/tests/peak_memory_smoke.rs）。
    require_numeric SIZE "$SIZE"
    cargo build -p bench-harness --release --bins
    cargo run -p bench-harness --release --bin peak_memory_bench -- --backend "$BACKEND" --size "$SIZE" --trials "$TRIALS"
    ;;
  *)
    echo "usage: $(basename "$0") <startup|peak-memory>" >&2
    exit 2
    ;;
esac
