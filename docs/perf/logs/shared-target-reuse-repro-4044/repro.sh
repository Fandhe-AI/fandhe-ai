#!/usr/bin/env bash
# イシュー #4044: 実機ノードの共有 CARGO_TARGET_DIR で、workspace メンバーの成果物が前の回から再利用される条件を開発機（Linux）で再現する。
# 手動実行専用（CI からは呼ばない）。実機は使わない。決定記録 docs/real-hardware-shared-target-reuse-decision.md §2 の根拠ログを作る。
# 使い方: bash repro.sh <repo-abs-path> <rev-X> <rev-Y> <out-abs-path>
#   X は古い側、Y は新しい側のコミット（crates/*/src に差分があること）。out は存在しないパス。
# 作業ディレクトリ（target・展開先）は $HOME/tmp-build-4044.XXXXXX に mktemp -d で作り、終了時に削除する。
# 各シナリオは cargo build -v -p fandhe-ai-tensor-core -p fandhe-ai-backend-cpu（dev プロファイル）の Fresh／Compiling を記録する。
set -euo pipefail
export LC_ALL=C
[ "$#" -eq 4 ] || { echo "usage: $0 <repo> <rev-X> <rev-Y> <out>" >&2; exit 2; }
REPO=$1 X=$2 Y=$3 OUT=$4
for p in "$REPO" "$OUT"; do [[ $p =~ ^/[A-Za-z0-9._/-]+$ ]] || { echo "絶対パス・許可文字のみ" >&2; exit 2; }; done
[[ $X =~ ^[0-9a-f]{7,40}$ && $Y =~ ^[0-9a-f]{7,40}$ ]] || { echo "rev は 16 進数" >&2; exit 2; }
[ ! -e "$OUT" ] || { echo "出力先が存在する" >&2; exit 2; }
PKGS="-p fandhe-ai-tensor-core -p fandhe-ai-backend-cpu"
W=$(mktemp -d "$HOME/tmp-build-4044.XXXXXX")
trap 'rm -rf "$W"' EXIT
mkdir -p "$OUT"
export CARGO_TARGET_DIR="$W/target" CARGO_INCREMENTAL=0

# unpack <rev> <dir> <touch-time>: git archive で展開し、全ファイルの mtime を固定する。
unpack() {
  mkdir -p "$2"
  git -C "$REPO" archive "$1" crates Cargo.toml Cargo.lock rust-toolchain.toml | tar -x -C "$2"
  find "$2" -exec touch -h -d "$3" {} +
}
# build <label> <dir>: Fresh／Compiling の行を記録する。fingerprint ログ（CARGO_LOG）も同じファイルに入る。
build() {
  local label=$1 dir=$2
  ( cd "$dir" && CARGO_LOG=cargo::core::compiler::fingerprint=info cargo build --offline -v $PKGS ) >"$OUT/build_$label.raw" 2>&1 \
    || { echo "build 失敗: $label" >&2; tail -20 "$OUT/build_$label.raw" >&2; exit 1; }
}
# rlib_report <label>: tensor-core の dev rlib の mtime と SHA-256（先頭 16 桁）。
rlib_report() {
  local d="$CARGO_TARGET_DIR/debug/deps" f
  for f in "$d"/libfandhe_ai_tensor_core-*.rlib; do
    [ -e "$f" ] || continue
    printf '%s\t%s\t%s\t%s\n' "$1" "$(stat -c %Y "$f")" "$(sha256sum "$f" | cut -c1-16)" "$(basename "$f")"
  done >>"$OUT/rlibs.tsv"
}
# summ <label>: パッケージ別に Fresh／Compiling を数える。
summ() {
  local l=$1 p
  for p in fandhe-ai-tensor-core fandhe-ai-backend-cpu; do
    printf '%s\t%s\t%s\t%s\n' "$l" "$p" \
      "$(grep -c "^ *Fresh $p " "$OUT/build_$l.raw" || true)" "$(grep -c "^ *Compiling $p " "$OUT/build_$l.raw" || true)"
  done >>"$OUT/summary.tsv"
}
printf 'label\tmtime\tsha256_16\tfile\n' >"$OUT/rlibs.tsv"
printf 'label\tpackage\tfresh\tcompiling\n' >"$OUT/summary.tsv"
T0="2026-01-01 00:00:00 +0000"
T1="2026-01-02 00:00:00 +0000"

# S0: 基準。X・Y を別ディレクトリへ展開（Y は古い時刻、X はそれより新しい時刻）。
unpack "$X" "$W/srcX" "$T1"
unpack "$Y" "$W/srcY" "$T0"
# S1: 同じ展開先 D へ X、続けて Y を rsync -a（mtime を保つ）で送ってビルドする。
mkdir -p "$W/D"
rsync -a --delete "$W/srcX/" "$W/D/"
build s1_x "$W/D"; summ s1_x; rlib_report s1_x
rsync -a --delete "$W/srcY/" "$W/D/"
build s1_y "$W/D"; summ s1_y; rlib_report s1_y
# S2: 別の展開先 D2 へ Y を送ってビルドする（展開先を変えても Fresh になるか）。
mkdir -p "$W/D2"
rsync -a --delete "$W/srcY/" "$W/D2/"
build s2_y_other_dir "$W/D2"; summ s2_y_other_dir; rlib_report s2_y_other_dir
# S3: 対照。mtime を保たずに X、続けて Y を送る。
rsync -a --delete --no-times "$W/srcX/" "$W/D/"
build s3_x_notimes "$W/D"; summ s3_x_notimes
rsync -a --delete --no-times "$W/srcY/" "$W/D/"
build s3_y_notimes "$W/D"; summ s3_y_notimes; rlib_report s3_y_notimes
# S4: 対策。X を mtime つきで送って再現状態を作り、Y を mtime つきで送ったうえで全メンバーを cargo clean -p してビルドする。
rm -rf "$W/target"
rsync -a --delete "$W/srcX/" "$W/D/"
build s4_prep_x "$W/D"
rsync -a --delete "$W/srcY/" "$W/D/"
( cd "$W/D" && cargo metadata --offline --no-deps --format-version 1 | python3 -c '
import json, re, sys
n = [p["name"] for p in json.load(sys.stdin)["packages"]]
assert n and all(re.fullmatch(r"[A-Za-z0-9_-]+", x) for x in n)
print("\n".join(n))' ) >"$OUT/members.txt"
: >"$OUT/clean_s4.raw"
( cd "$W/D"; while IFS= read -r m; do echo "== cargo clean -p $m" >>"$OUT/clean_s4.raw"; cargo clean --offline -p "$m" >>"$OUT/clean_s4.raw" 2>&1; done <"$OUT/members.txt" )
{ find "$CARGO_TARGET_DIR/debug" \( -name '*fandhe_ai_tensor_core*' -o -name '*fandhe-ai-tensor-core*' \) || true; } | sed "s|$W|<work>|" | sort >"$OUT/after_clean_leftovers.raw"
build s4_y_after_clean "$W/D"; summ s4_y_after_clean; rlib_report s4_y_after_clean
# S5: 新しい mtime の対照。Y のツリーを now に触ってから送る。
rm -rf "$W/target"
rsync -a --delete "$W/srcX/" "$W/D/"
build s5_prep_x "$W/D"
sleep 1
find "$W/srcY" -exec touch -h {} +
rsync -a --delete "$W/srcY/" "$W/D/"
build s5_y_fresh_mtime "$W/D"; summ s5_y_fresh_mtime; rlib_report s5_y_fresh_mtime

# マスク: 一時ディレクトリ・$HOME・ホスト名・ユーザー名。
UN=${USER:-$(id -un)}
HN=$(hostname)
esc() { printf '%s' "$1" | sed -e 's/[][\\.^$*+?(){}|/]/\\&/g'; }
for f in "$OUT"/*.raw; do
  b=$(basename "$f" .raw)
  sed -E -e "s/$(esc "$W")/<work>/g" -e "s/$(esc "$HOME")/<home>/g" -e "s/$(esc "$HN")/<dev-host>/g" \
    -e "s/(^|[^A-Za-z0-9_.-])$(esc "$UN")([^A-Za-z0-9_.-]|\$)/\1<user>\2/g" "$f" >"$OUT/$b.log"
  rm -f "$f"
done
if grep -lE "/home/|$(esc "$HN")" "$OUT"/*; then echo "マスク漏れ" >&2; exit 1; fi
( cd "$OUT" && find . -type f ! -name SHA256SUMS | sed 's|^\./||' | sort | xargs sha256sum >SHA256SUMS )
echo "done"
