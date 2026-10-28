#!/usr/bin/env bash
# イシュー #3671: bench 側の腕 bench_base／bench_headpatch のバイナリを hw-lock の前に作る（計測中のビルド並走は禁止）。
# 判定規則は RULE.txt ＋ RULE.v2.txt が正。bench-fandhe のソース・窓・emit は変えない。
#   base:      registry fandhe-ai =0.12.0（framework-compare の承認ピンそのまま）→ <bindir>/bench-fandhe-base
#   headpatch: --config 'patch.crates-io.fandhe-ai.path=<repo>/crates/facade' で HEAD の path ビルド → <bindir>/bench-fandhe-head
# path patch は framework-compare/Cargo.lock を書き換えるため bench_fandhe_lock_restore.sh の退避・復元 trap を使い、
# 終了時に元へ戻す（deps-policy.md 第 9 区分・check_framework_compare の承認ピンを汚さない）。
# 使い方: build_bench_arms.sh <bindir>   （bindir は新規ディレクトリ。既存ならば中止）
set -euo pipefail

[ $# -eq 1 ] || { echo "usage: $0 <bindir>" >&2; exit 2; }
bindir="$1"
if [ -e "$bindir" ]; then echo "bindir が既に存在する: $bindir" >&2; exit 5; fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../../../.." && pwd)"
fc="$repo_root/scripts/bench/framework-compare"
facade="$repo_root/crates/facade"
[ -d "$facade" ] || { echo "crates/facade が無い" >&2; exit 1; }

mkdir -p "$bindir"
bindir="$(cd "$bindir" && pwd)"
cd "$fc"
# shellcheck source=/dev/null
source ./bench_fandhe_lock_restore.sh
bench_fandhe_setup_lock_restore_trap

find_exe() { # stdin: cargo --message-format=json。bench-fandhe の実行ファイル 1 本を出す
  python3 -c '
import json, sys
exe = None
for line in sys.stdin:
    try:
        m = json.loads(line)
    except ValueError:
        continue
    if m.get("reason") == "compiler-artifact" and m.get("executable") and m.get("target", {}).get("name") == "bench-fandhe":
        exe = m["executable"]
if exe is None:
    sys.exit(1)
print(exe)
'
}

base_exe="$(cargo build --release --locked -p bench-fandhe --target-dir "$bindir/target-base" --message-format=json 2>"$bindir/build-base.err" | find_exe)"
cp "$base_exe" "$bindir/bench-fandhe-base"

patch_config="patch.crates-io.fandhe-ai.path=\"${facade}\""
head_exe="$(cargo build --release -p bench-fandhe --target-dir "$bindir/target-head" --message-format=json \
  --config "$patch_config" 2>"$bindir/build-head.err" | find_exe)"
cp "$head_exe" "$bindir/bench-fandhe-head"
# path patch が解決されたことを確認する（registry のままなら計測の意味が変わるため中止）。
tree_out="$(cargo tree -p bench-fandhe --depth 1 --config "$patch_config" 2>&1)"
case "$tree_out" in
  *"fandhe-ai v"*"$facade"*) ;;
  *) echo "fandhe-ai が path patch の crates/facade に解決されていない" >&2; exit 1 ;;
esac
{
  echo "base_sha256: $(bench_fandhe_sha256_of "$bindir/bench-fandhe-base")"
  echo "head_sha256: $(bench_fandhe_sha256_of "$bindir/bench-fandhe-head")"
  echo "rustc -vV: $(rustc -vV | tr '\n' ';')"
  echo "cargo: $(cargo --version)"
} >"$bindir/bench_arms.txt"
echo "完了: $bindir（Cargo.lock は終了時に復元される）"
