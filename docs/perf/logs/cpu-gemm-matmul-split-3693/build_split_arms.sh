#!/usr/bin/env bash
# イシュー #3693: before／after 両木の診断テストバイナリ 2 本を hw-lock の前に作る
# （計測中のビルド並走は禁止）。判定規則は同ディレクトリの RULE.txt が正。
# 構成は docs/perf/logs/cpu-gemm-download-owned-3672/build_ab_arms.sh の診断部分と同型（bench は作らない）。
# 両木のハード条件（RULE.txt）: (1) diff -r（.rev-stamp・target* を除く）の差が
#   crates/autodiff/src/optim/resident_input.rs の 1 ファイルだけ  (2) 診断テストファイルが同一。違えば中止する。
# 各木のルートに .rev-stamp（1 行目 sha・2 行目 dirty=0）が必須。registry の fandhe-ai =0.12.0 は使わない。
# 使い方: build_split_arms.sh <before_tree> <after_tree> <bindir>   （bindir は新規ディレクトリ。既存なら中止）
# 出力: <bindir>/{diag-before,diag-after,arms.txt}（arms.txt に sha256・rustc・両木の rev-stamp）
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

DIAG=crates/facade/tests/cpu_gemm_resident_matmul_split_diag.rs
ONLY=crates/autodiff/src/optim/resident_input.rs
if ! diff -q "${before_tree}/${DIAG}" "${after_tree}/${DIAG}" >/dev/null; then
  echo "ハード条件違反: 診断テストが腕の間で異なる" >&2
  exit 1
fi
# 差の集合が resident_input.rs 1 ファイルだけであること（.rev-stamp・target* は除く）。
diff_list="$(diff -rq -x .rev-stamp -x 'target*' "${before_tree}" "${after_tree}" || true)"
want="Files ${before_tree}/${ONLY} and ${after_tree}/${ONLY} differ"
if [ "${diff_list}" != "${want}" ]; then
  echo "ハード条件違反: 木の差が ${ONLY} の 1 ファイルだけでない" >&2
  printf '%s\n' "${diff_list}" >&2
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
  local arm="$1" tree="$2" exe
  # 診断テストバイナリ（ルート workspace。--locked）
  (
    cd "${tree}"
    exe="$(cargo test --release --locked -p fandhe-ai --test cpu_gemm_resident_matmul_split_diag --no-run \
      --target-dir "${bindir}/target-diag-${arm}" --message-format=json \
      2>"${bindir}/build-diag-${arm}.err" | find_exe cpu_gemm_resident_matmul_split_diag)"
    cp "${exe}" "${bindir}/diag-${arm}"
  )
}

build_arm before "${before_tree}"
build_arm after "${after_tree}"

for b in diag-before diag-after; do
  [ -x "${bindir}/${b}" ] || { echo "バイナリが無い: ${b}" >&2; exit 1; }
done
{
  echo "before_rev_stamp: $(sed -n 1p "${before_tree}/.rev-stamp") $(sed -n 2p "${before_tree}/.rev-stamp")"
  echo "after_rev_stamp: $(sed -n 1p "${after_tree}/.rev-stamp") $(sed -n 2p "${after_tree}/.rev-stamp")"
  for b in diag-before diag-after; do
    echo "${b}_sha256: $(sha256_of "${bindir}/${b}")"
  done
  echo "rustc -vV: $(rustc -vV | tr '\n' ';')"
  echo "cargo: $(cargo --version)"
} >"${bindir}/arms.txt"
echo "完了: ${bindir}"
