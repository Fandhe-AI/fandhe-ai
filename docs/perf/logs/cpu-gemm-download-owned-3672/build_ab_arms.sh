#!/usr/bin/env bash
# イシュー #3672: download_owned の A/B 用に、before／after 両腕のバイナリ 4 本を hw-lock の前に作る
# （計測中のビルド並走は禁止）。判定規則は同ディレクトリの RULE.txt ＋ RULE.v2.txt が正。
#   bench-fandhe（framework-compare。fandhe-ai は両腕とも各木の crates/facade への path patch）
#   cpu_gemm_resident_reuse_diag のテストバイナリ（各木で cargo test --no-run）
# RULE.txt: registry の fandhe-ai =0.12.0 は before に使わない。腕は git archive の展開木で、
# 各木のルートに .rev-stamp（1 行目 sha・2 行目 dirty=0）が必須。
# path patch は framework-compare/Cargo.lock を書き換えるため bench_fandhe_lock_restore.sh の退避・復元 trap を使い
# 終了時に元へ戻す（deps-policy.md 第 9 区分の承認ピンを汚さない）。
# 使い方: build_ab_arms.sh <before_tree> <after_tree> <bindir>   （bindir は新規ディレクトリ。既存なら中止）
# 出力: <bindir>/{bench-before,bench-after,diag-before,diag-after,arms.txt}（arms.txt に sha256・rustc・両木の rev-stamp）
# bash 3.2（macOS 標準）で動くよう ${VAR} 形式・mapfile／連想配列なしで書く。
set -euo pipefail

[ $# -eq 3 ] || { echo "usage: $0 <before_tree> <after_tree> <bindir>" >&2; exit 2; }
before_tree="$1"
after_tree="$2"
bindir="$3"
for p in "${before_tree}" "${after_tree}" "${bindir}"; do
  case "${p}" in /*) ;; *) echo "絶対パスのみ許可: ${p}" >&2; exit 2 ;; esac
  case "${p}" in *[!A-Za-z0-9._/-]*) echo "許可外の文字を含むパス: ${p}" >&2; exit 2 ;; esac
done
[ -d "${before_tree}" ] && [ -d "${after_tree}" ] || { echo "腕の木が無い" >&2; exit 2; }
if [ -e "${bindir}" ]; then echo "bindir が既に存在する: ${bindir}" >&2; exit 5; fi

# 環境変数の混入を断つ（Mac の login 環境は CARGO_TARGET_DIR を設定している）。
unset CARGO_TARGET_DIR RAYON_NUM_THREADS GEMM_REUSE_MINFLT GLIBC_TUNABLES

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'; else shasum -a 256 "$1" | awk '{print $1}'; fi
}

# 木の同定: .rev-stamp の 2 行目が dirty=0 であること。
for t in "${before_tree}" "${after_tree}"; do
  [ -f "${t}/.rev-stamp" ] || { echo ".rev-stamp が無い: ${t}" >&2; exit 1; }
  [ "$(sed -n 2p "${t}/.rev-stamp")" = "dirty=0" ] || { echo ".rev-stamp の dirty が 0 でない: ${t}" >&2; exit 1; }
done
if [ "$(sed -n 1p "${before_tree}/.rev-stamp")" = "$(sed -n 1p "${after_tree}/.rev-stamp")" ]; then
  echo "before と after の sha が同じ" >&2
  exit 1
fi

# RULE.txt のハード条件: 診断テストと framework-compare が両腕で同一（違えば中止）。
if ! diff -q "${before_tree}/crates/facade/tests/cpu_gemm_resident_reuse_diag.rs" \
  "${after_tree}/crates/facade/tests/cpu_gemm_resident_reuse_diag.rs" >/dev/null; then
  echo "ハード条件違反: cpu_gemm_resident_reuse_diag.rs が腕の間で異なる" >&2
  exit 1
fi
if ! diff -r "${before_tree}/scripts/bench/framework-compare" "${after_tree}/scripts/bench/framework-compare" >/dev/null; then
  echo "ハード条件違反: scripts/bench/framework-compare が腕の間で異なる" >&2
  exit 1
fi

mkdir -p "${bindir}"
bindir="$(cd "${bindir}" && pwd)"

find_exe() { # $1=target 名。stdin: cargo --message-format=json
  python3 -c '
import json, sys
want = sys.argv[1]
exe = None
for line in sys.stdin:
    try:
        m = json.loads(line)
    except ValueError:
        continue
    if m.get("reason") == "compiler-artifact" and m.get("executable") and m.get("target", {}).get("name") == want:
        exe = m["executable"]
if exe is None:
    sys.exit(1)
print(exe)
' "$1"
}

build_arm() { # $1=before|after $2=木
  local arm="$1" tree="$2" facade exe patch_config tree_out
  facade="${tree}/crates/facade"
  [ -d "${facade}" ] || { echo "crates/facade が無い: ${tree}" >&2; exit 1; }
  patch_config="patch.crates-io.fandhe-ai.path=\"${facade}\""
  # bench-fandhe（サブシェルで Cargo.lock の退避・復元 trap を持たせる）
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
    exe="$(cargo build --release -p bench-fandhe --target-dir "${bindir}/target-bench-${arm}" \
      --message-format=json --config "${patch_config}" 2>"${bindir}/build-bench-${arm}.err" | find_exe bench-fandhe)"
    cp "${exe}" "${bindir}/bench-${arm}"
  )
  # 診断テストバイナリ（ルート workspace。--locked）
  (
    cd "${tree}"
    exe="$(cargo test --release --locked -p fandhe-ai --test cpu_gemm_resident_reuse_diag --no-run \
      --target-dir "${bindir}/target-diag-${arm}" --message-format=json \
      2>"${bindir}/build-diag-${arm}.err" | find_exe cpu_gemm_resident_reuse_diag)"
    cp "${exe}" "${bindir}/diag-${arm}"
  )
}

build_arm before "${before_tree}"
build_arm after "${after_tree}"

for b in bench-before bench-after diag-before diag-after; do
  [ -x "${bindir}/${b}" ] || { echo "バイナリが無い: ${b}" >&2; exit 1; }
done
{
  echo "before_rev_stamp: $(sed -n 1p "${before_tree}/.rev-stamp") $(sed -n 2p "${before_tree}/.rev-stamp")"
  echo "after_rev_stamp: $(sed -n 1p "${after_tree}/.rev-stamp") $(sed -n 2p "${after_tree}/.rev-stamp")"
  for b in bench-before bench-after diag-before diag-after; do
    echo "${b}_sha256: $(sha256_of "${bindir}/${b}")"
  done
  echo "rustc -vV: $(rustc -vV | tr '\n' ';')"
  echo "cargo: $(cargo --version)"
} >"${bindir}/arms.txt"
echo "完了: ${bindir}（各木の framework-compare/Cargo.lock は終了時に復元済み）"
