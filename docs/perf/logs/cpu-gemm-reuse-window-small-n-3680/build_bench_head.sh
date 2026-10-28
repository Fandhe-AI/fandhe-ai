#!/usr/bin/env bash
# イシュー #3680: 計測木の HEAD（現 main。#3672 の download_owned を含む）を path patch した bench-fandhe を
# 1 本だけ hw-lock の前に作る（計測中のビルド並走は禁止）。判定規則は同ディレクトリの RULE.txt が正。
# registry の fandhe-ai =0.12.0 には download_owned が無いため、#3671 の build_bench_arms.sh の
# bench-fandhe-base（registry 版）は流用しない。
# path patch は framework-compare/Cargo.lock を書き換えるため bench_fandhe_lock_restore.sh の退避・復元 trap を使い
# 終了時に元へ戻す（deps-policy.md 第 9 区分の承認ピンを汚さない）。
# 使い方: build_bench_head.sh <tree> <bindir>   （bindir は新規ディレクトリ。既存なら中止）
# 出力: <bindir>/{bench-fandhe-head,arms.txt}（arms.txt に sha256・rustc・cargo・rev-stamp）
# bash 3.2（macOS 標準）で動くよう ${VAR} 形式・mapfile／連想配列なしで書く。
set -euo pipefail

[ $# -eq 2 ] || { echo "usage: $0 <tree> <bindir>" >&2; exit 2; }
tree="$1"
bindir="$2"
for p in "${tree}" "${bindir}"; do
  case "${p}" in /*) ;; *) echo "絶対パスのみ許可: ${p}" >&2; exit 2 ;; esac
  case "${p}" in *[!A-Za-z0-9._/-]*) echo "許可外の文字を含むパス: ${p}" >&2; exit 2 ;; esac
done
[ -d "${tree}" ] || { echo "計測木が無い: ${tree}" >&2; exit 2; }
if [ -e "${bindir}" ]; then echo "bindir が既に存在する: ${bindir}" >&2; exit 5; fi

# 環境変数の混入を断つ（Mac の login 環境は CARGO_TARGET_DIR を設定している）。
unset CARGO_TARGET_DIR RAYON_NUM_THREADS GEMM_REUSE_MINFLT GLIBC_TUNABLES

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'; else shasum -a 256 "$1" | awk '{print $1}'; fi
}

[ -f "${tree}/.rev-stamp" ] || { echo ".rev-stamp が無い: ${tree}" >&2; exit 1; }
[ "$(sed -n 2p "${tree}/.rev-stamp")" = "dirty=0" ] || { echo ".rev-stamp の dirty が 0 でない: ${tree}" >&2; exit 1; }
facade="${tree}/crates/facade"
[ -d "${facade}" ] || { echo "crates/facade が無い: ${tree}" >&2; exit 1; }

mkdir -p "${bindir}"
bindir="$(cd "${bindir}" && pwd)"
patch_config="patch.crates-io.fandhe-ai.path=\"${facade}\""

(
  cd "${tree}/scripts/bench/framework-compare"
  # shellcheck source=/dev/null
  source ./bench_fandhe_lock_restore.sh
  bench_fandhe_setup_lock_restore_trap
  tree_out="$(cargo tree -p bench-fandhe --depth 1 --config "${patch_config}" 2>&1)"
  case "${tree_out}" in
    *"fandhe-ai v"*"${facade}"*) ;;
    *) echo "fandhe-ai が path patch の ${facade} に解決されていない" >&2; exit 1 ;;
  esac
  cargo build --release -p bench-fandhe --target-dir "${bindir}/target-bench-head" \
    --config "${patch_config}" 2>"${bindir}/build-bench-head.err"
  cp "${bindir}/target-bench-head/release/bench-fandhe" "${bindir}/bench-fandhe-head"
)

[ -x "${bindir}/bench-fandhe-head" ] || { echo "バイナリが無い: bench-fandhe-head" >&2; exit 1; }
{
  echo "rev_stamp: $(sed -n 1p "${tree}/.rev-stamp") $(sed -n 2p "${tree}/.rev-stamp")"
  echo "path_patch: ${facade}"
  echo "bench-fandhe-head_sha256: $(sha256_of "${bindir}/bench-fandhe-head")"
  echo "rustc -vV: $(rustc -vV | tr '\n' ';')"
  echo "cargo: $(cargo --version)"
} >"${bindir}/arms.txt"
echo "完了: ${bindir}（framework-compare/Cargo.lock は終了時に復元済み）"
