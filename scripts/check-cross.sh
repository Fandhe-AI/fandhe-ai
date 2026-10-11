#!/usr/bin/env bash
# クロスターゲットの型検査・lint（make check-cross-*）。cargo check / clippy はリンクを
# 行わないため、対象 OS の SDK が無い Linux 上でも成立する。
# usage: check-cross.sh <metal-tests|cpu-tests|windows-interop>
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./lib.sh
source "${DIR}/lib.sh"

case "${1:-}" in
  metal-tests)
    # TASK-1.8e（#42）: backend-metal の #[ignore] 実機テストを Linux CI でも型検査する。
    # --workspace --all-targets は bench-harness の dev-dependencies 経由で criterion →
    # alloca を引き込み、macOS クロスコンパイラ非搭載の Linux では
    # `cc: error: unrecognized command-line option '-arch'` で失敗する（実測済み）ため、
    # -p fandhe-ai-backend-metal --tests に限定する。
    ensure_rust_target aarch64-apple-darwin
    cargo check -p fandhe-ai-backend-metal --tests --target aarch64-apple-darwin
    # イシュー #3003: macOS・internal-diagnostics 無効の lib／lib test で出る
    # dead_code 警告の再発防止（Linux の --all-features clippy では見えない構成）。
    # --all-targets は criterion → alloca のクロス C ビルドで失敗するため --lib --tests。
    cargo clippy -p fandhe-ai-backend-metal --lib --tests --locked --target aarch64-apple-darwin -- -D warnings
    ;;
  cpu-tests)
    # イシュー #753（codex-review P2）: backend-cpu::gemm_blis::cache_params は
    # #[cfg(test)] 限定で、内部の sysctl_ffi（cfg(target_os = "macos")）は Linux の通常
    # cargo test では到達しない。--tests で cfg(test) を有効にしたまま
    # aarch64-apple-darwin へクロス型検査する（criterion → alloca 回避は metal-tests と同じ）。
    ensure_rust_target aarch64-apple-darwin
    cargo check -p fandhe-ai-backend-cpu --tests --target aarch64-apple-darwin
    ;;
  windows-interop)
    # onnx-interop の external data Windows 対応（#2349）と facade の Windows
    # ビルド可能性（#2390・#2391）を Linux 上で継続的に型検査・lint する。
    ensure_rust_target x86_64-pc-windows-msvc
    cargo clippy -p fandhe-ai-onnx-interop --lib --tests --locked --target x86_64-pc-windows-msvc -- -D warnings
    cargo clippy -p fandhe-ai --lib --tests --locked --target x86_64-pc-windows-msvc -- -D warnings
    ;;
  *)
    echo "usage: $(basename "$0") <metal-tests|cpu-tests|windows-interop>" >&2
    exit 2
    ;;
esac
