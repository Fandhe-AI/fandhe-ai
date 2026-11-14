#!/usr/bin/env bash
# イシュー #4168: 収録だけのやり直し（計測はしない。実機ロックも取らない）。
# 役割: orchestrate.sh が「マスク検査に失敗。収録しない」で止まったとき（収録先に SHA256SUMS が無い）に限り、
#   無傷の生データ（results/raw/ab4168-<label>/）から、固定済みの判定器を再実行して収録をやり直す。
# 背景（2026-11-14 所有者判断「収録だけ別スクリプトで」）: orchestrate.sh のユーザー名の置換と漏れ検査が部分一致で、
#   記録に要る語の一部まで書き換えて止まった。orchestrate.sh・ランナー・判定器は RULE.txt の [道具の sha256] で固定済みなので
#   変えない。本スクリプトはその一覧の外にあり、計測・ゲート・判定式には関わらない。
# マスクは orchestrate.sh と同じ（ホーム・ホスト名・IPv4 の扱いは変えない）で、ユーザー名の項だけを語境界つきにする
#   （#4156 所有者判断 2026-11-11 と同じ定義: 語の文字は英数字・下線・ハイフン。前後がそのどれでもないときだけ一致する）。
# 確認（fail-closed）:
#   1. 固定済みの判定器（AB_RUNNER_ROOT 側。RULE の sha256 照合は判定器自身が行う）を生データに再実行し、
#      orchestrate.sh と同じ手順・同じ旧マスクで組み直したものが、止まった収録先と（hwlock.log を除き）バイト一致すること。
#      生データと判定器の出力が止まったときから変わっていないことの確認になる。値は表示しない。
#   2. 新しいマスクのあとの漏れ検査（ホーム・ホスト名は部分一致、ユーザー名は語境界つき、IPv4）が 0 件であること。
# 使い方:
#   AB_PROFILE=macmini|gb10 AB_RUNNER_ROOT=<abs> AB_LOCK_ACQUIRED=<UTC> AB_LOCK_RELEASED=<UTC> bash recollect.sh
#     AB_LOCK_* は orchestrate.sh のログ（lock released 行）と止まった収録先の hwlock.log（acquired=）の時刻。
#   bash recollect.sh --self-test   # マスクと漏れ検査の正規表現の自己検査（合成値と実行環境の値。値は表示しない）
# bash 3.2 互換。
set -uo pipefail

die() { echo "error: $*" >&2; exit 1; }

# ユーザー名の語境界つきの置換・検査（ERE。\b は使わない。GNU と BSD の sed／grep の両方で動く文字クラス）
re_escape() { printf '%s' "$1" | sed -e 's#[][\\.^$*+?(){}|/#]#\\&#g'; }
user_ere() { printf '(^|[^A-Za-z0-9_-])%s([^A-Za-z0-9_-]|$)' "$(re_escape "$1")"; }
# 新しいマスク（ユーザー名だけ語境界つき。隣り合う出現は 1 回の置換で取りこぼすため 2 回かける。残りは漏れ検査が拾う）
mask_new() { # mask_new <file> <home> <host> <host-short> <user>
  local f=$1 h=$2 hn=$3 hns=$4 un=$5 t ue rc
  local args=()
  # 式は配列で渡す（未クォートの展開に [ ] が入るとパス名展開を受けるため）
  if [[ -n "$h" ]]; then args+=(-e "s#$(re_escape "$h")#<home>#g"); fi
  if [[ -n "$hn" ]]; then args+=(-e "s#$(re_escape "$hn")#<host>#g"); fi
  if [[ -n "$hns" ]]; then args+=(-e "s#$(re_escape "$hns")#<host>#g"); fi
  if [[ -n "$un" ]]; then ue="$(user_ere "$un")"; args+=(-e "s#${ue}#\\1<user>\\2#g" -e "s#${ue}#\\1<user>\\2#g"); fi
  [[ "${#args[@]}" -gt 0 ]] || return 0
  t="$(mktemp)" || return 1
  LC_ALL=C sed -E "${args[@]}" "$f" >"$t" && cat "$t" >"$f"
  rc=$?; rm -f "$t"; return "$rc"
}
# 漏れ検査（ディレクトリ再帰。件数だけ返す。値は表示しない）
leak_count() { # leak_count <dir> <home> <host> <host-short> <user>
  local d=$1 n=0 needle
  for needle in "$2" "$3" "$4"; do
    [[ -n "$needle" ]] || continue
    if grep -rFq -- "$needle" "$d"; then echo "LEAK: '${needle:0:2}…' が収録物に残る" >&2; n=$((n + 1)); fi
  done
  if [[ -n "$5" ]] && grep -rEq -- "$(user_ere "$5")" "$d"; then echo "LEAK: ユーザー名（語境界つき）が収録物に残る" >&2; n=$((n + 1)); fi
  if grep -rEq '([0-9]{1,3}\.){3}[0-9]{1,3}' "$d" --include='*.txt' --include='*.log' --include='*.md'; then echo "LEAK: IP アドレスらしき文字列が残る" >&2; n=$((n + 1)); fi
  echo "$n"
}

self_test() {
  local d bad=0 n=0 pair host user line i f
  d="$(mktemp -d "${TMPDIR:-/tmp}/recollect-selftest.XXXXXX")" || return 1
  for pair in "thost-01.example tuser9" "$(hostname 2>/dev/null || echo __nohost__) $(id -un 2>/dev/null || echo __nouser__)"; do
    host="${pair% *}"; user="${pair#* }"
    # 検出・置換されるべき行（マスク後は漏れ 0 件になること）
    i=0
    for line in "run by ${user} at 00:00" "${user}" "${user}: done" "owner=${user}" "path ${user}/x" "${user} ${user} ${user}" "on ${host} now" "home /h0me/${user}/w"; do
      i=$((i + 1)); mkdir -p "$d/pos$i"; f="$d/pos$i/a.txt"; printf '%s\n' "$line" >"$f"
      n=$((n + 1)); [[ "$(leak_count "$d/pos$i" "/nonexistent-home" "$host" "" "$user" 2>/dev/null)" != "0" ]] || { bad=$((bad + 1)); echo "self-test FAIL: 検出されるべき行 $i" >&2; }
      mask_new "$f" "/nonexistent-home" "$host" "" "$user" || { bad=$((bad + 1)); echo "self-test FAIL: マスク失敗 $i" >&2; }
      n=$((n + 1)); [[ "$(leak_count "$d/pos$i" "/nonexistent-home" "$host" "" "$user" 2>/dev/null)" == "0" ]] || { bad=$((bad + 1)); echo "self-test FAIL: マスク後に残る行 $i" >&2; }
    done
    # 書き換えも検出もされないべき行（ユーザー名が別の語の一部）
    i=0
    for line in "x${user}y" "${user}_z" "z_${user}" "${user}9q" "${user}-z" "z-${user}" "z-${user}-w"; do
      i=$((i + 1)); mkdir -p "$d/neg$i"; f="$d/neg$i/a.txt"; printf '%s\n' "$line" >"$f"
      n=$((n + 1)); [[ "$(leak_count "$d/neg$i" "/nonexistent-home" "__nohost__" "" "$user" 2>/dev/null)" == "0" ]] || { bad=$((bad + 1)); echo "self-test FAIL: 検出されないべき行 $i" >&2; }
      mask_new "$f" "/nonexistent-home" "__nohost__" "" "$user"
      n=$((n + 1)); [[ "$(cat "$f")" == "$line" ]] || { bad=$((bad + 1)); echo "self-test FAIL: 書き換えられないべき行 $i" >&2; }
    done
    rm -rf "$d"/pos* "$d"/neg*
  done
  # ホームの置換（正規表現の特殊文字を含むパスでも文字どおりに置換される）
  mkdir -p "$d/home"; printf 'p /tmp/a.b+c/x\n' >"$d/home/a.log"
  mask_new "$d/home/a.log" "/tmp/a.b+c" "" "" ""; n=$((n + 1)); [[ "$(cat "$d/home/a.log")" == "p <home>/x" ]] || { bad=$((bad + 1)); echo "self-test FAIL: ホームの置換" >&2; }
  rm -rf "$d"
  echo "self-test: ${n} 件中 FAIL ${bad} 件"
  [[ "$bad" -eq 0 ]]
}

if [[ "${1:-}" == "--self-test" ]]; then self_test; exit $?; fi
[[ $# -eq 0 ]] || die "引数は --self-test だけ"

PROFILE=${AB_PROFILE:-}
case "$PROFILE" in
  macmini) LABEL=series-b ;;
  gb10) LABEL=gb10 ;;
  *) die "AB_PROFILE は macmini か gb10" ;;
esac
R=${AB_RUNNER_ROOT:-}
if [[ -z "$R" || "$R" != /* || "$R" == *\"* || "$R" == *\\* || "$R" == *\'* || "$R" == *$'\n'* || "$R" == *#* ]]; then
  die "AB_RUNNER_ROOT は引用符・バックスラッシュ・#・改行を含まない絶対パス必須"
fi
UTC_RE='^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$'
[[ "${AB_LOCK_ACQUIRED:-}" =~ $UTC_RE ]] || die "AB_LOCK_ACQUIRED は UTC（例 2026-11-14T03:11:44Z）"
[[ "${AB_LOCK_RELEASED:-}" =~ $UTC_RE ]] || die "AB_LOCK_RELEASED は UTC（例 2026-11-14T03:53:04Z）"

JUDGE="$R/scripts/bench/framework-compare/judge_prepost_0150_4168.py"
LOGS="$R/docs/perf/logs/prepost-0150-4168"
RULE="$LOGS/RULE.txt"
RAW="$R/scripts/bench/framework-compare/results/raw/ab4168-${LABEL}"
DEST="$LOGS/$PROFILE"
[[ -f "$JUDGE" && -f "$RULE" ]] || die "判定器か RULE.txt が無い（AB_RUNNER_ROOT は RULE を含むコミットの clone）"
[[ -d "$RAW" ]] || die "生データが無い: $RAW"
[[ -d "$DEST" ]] || die "止まった収録先が無い（本スクリプトは orchestrate.sh が収録の段で止まったときだけ使う）: $DEST"
[[ ! -e "$DEST/SHA256SUMS" ]] || die "収録先に SHA256SUMS がある（収録は完了している。やり直さない）"
[[ -d "$HOME/work/.fandhe-hw-lock" ]] && echo "note: 実機ロックが存在する（本スクリプトは計測しないので取らない・触らない）"

H="${HOME}"; HN="$(hostname 2>/dev/null || true)"; HNS="$(hostname -s 2>/dev/null || true)"; UN="$(id -un)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/recollect-4168.XXXXXX")" || die "一時ディレクトリを作れない"
trap 'rm -rf "$WORK"' EXIT

# orchestrate.sh と同じ組み立て（判定器の再実行 → ログと JSONL の複写）。<out> へ置く。
assemble() { # assemble <out>
  local out=$1 f i jrc
  mkdir -p "$out" || return 1
  local JARGS=(--profile "$PROFILE" --label "$LABEL" --series-dir "$RAW" --env-info "$RAW/env_info.txt" --rule "$RULE" --gate-log "$RAW/gate.log")
  if [[ "$PROFILE" == "gb10" ]]; then JARGS+=(--compute-log "$RAW/compute_apps.log"); fi
  python3 -I "$JUDGE" "${JARGS[@]}" --ratios-out "$out/ratios.tsv" >"$out/judge-4168-${PROFILE}.md"; jrc=$?
  echo "$jrc" >"$out/judge-4168-${PROFILE}.exit"
  for f in env_info.txt gate.log skipped.log tree-before.txt tree-after.txt consts.txt consts-baseline.txt consts-diff.txt bench-fandhe-diff-stat.txt compute_apps.log; do
    if [[ -f "$RAW/$f" ]]; then cp "$RAW/$f" "$out/$f" || return 1; fi
  done
  if [[ -d "$RAW/foreign-gate" ]]; then mkdir -p "$out/foreign-gate"; cp "$RAW/foreign-gate/"* "$out/foreign-gate/" || return 1; fi
  i=1
  while [[ "$i" -le 5 ]]; do
    if [[ -d "$RAW/run$i" ]]; then mkdir -p "$out/run$i"; cp "$RAW/run$i/"* "$out/run$i/" || return 1; fi
    i=$((i + 1))
  done
  echo "$jrc"
}

# 確認 1: 旧マスク（orchestrate.sh の mask_file と同じ式）で組み直したものが、止まった収録先とバイト一致する
OLD="$WORK/old"
JRC_OLD="$(assemble "$OLD")" || die "組み直しに失敗"
mask_old() {
  local f=$1 t
  t="$(mktemp)"
  LC_ALL=C sed -e "s#${H}#<home>#g" ${HN:+-e "s#${HN}#<host>#g"} ${HNS:+-e "s#${HNS}#<host>#g"} ${UN:+-e "s#${UN}#<user>#g"} "$f" >"$t" && cat "$t" >"$f"
  rm -f "$t"
}
find "$OLD" -type f \( -name '*.txt' -o -name '*.log' -o -name '*.md' -o -name '*.tsv' \) | while IFS= read -r f; do mask_old "$f"; done
if ! diff -rq -x hwlock.log "$OLD" "$DEST" >/dev/null 2>&1; then
  echo "error: 止まった収録先と、生データから組み直したものが一致しない（差分のあるファイル名だけ示す。中身は表示しない）" >&2
  diff -rq -x hwlock.log "$OLD" "$DEST" 2>&1 | sed -e "s#${WORK}#<work>#g" -e "s#${R}#<R>#g" >&2
  exit 1
fi
echo "check-old-collection=ok（止まった収録先と、生データ＋固定済み判定器＋旧マスクの組み直しがバイト一致。hwlock.log を除く）"

# 確認 2: 新しいマスクで収録し直し、漏れ 0 件を確かめる
NEW="$WORK/new"
JRC_NEW="$(assemble "$NEW")" || die "組み直しに失敗"
[[ "$JRC_NEW" == "$JRC_OLD" ]] || die "判定器の終了コードが再実行で変わった（${JRC_OLD} → ${JRC_NEW}）"
find "$NEW" -type f \( -name '*.txt' -o -name '*.log' -o -name '*.md' -o -name '*.tsv' \) | while IFS= read -r f; do mask_new "$f" "$H" "$HN" "$HNS" "$UN"; done
leaks="$(leak_count "$NEW" "$H" "$HN" "$HNS" "$UN")"
[[ "$leaks" == "0" ]] || die "マスク検査に失敗（${leaks} 項目）。収録しない（止まった収録先はそのまま残す）"

printf 'acquired=%s\nreleased=%s\n' "$AB_LOCK_ACQUIRED" "$AB_LOCK_RELEASED" >"$NEW/hwlock.log"
RC_SHA="$(if command -v sha256sum >/dev/null 2>&1; then sha256sum "${BASH_SOURCE[0]}"; else shasum -a 256 "${BASH_SOURCE[0]}"; fi | cut -d' ' -f1)"
printf 'recollected=%s\nrecollect_sh_sha256=%s\nreason=orchestrate.sh の収録の段がマスク検査で停止（計測と判定は完了済み）。2026-11-14 所有者判断で収録だけをやり直した\ncheck_old_collection=ok\njudge_exit=%s\nleaks=0\n' \
  "$(date -u +%FT%TZ)" "$RC_SHA" "$JRC_NEW" >"$NEW/recollect.log"
SUMS_TMP="$(mktemp)"
( cd "$NEW" && find . -type f | LC_ALL=C sort | while IFS= read -r f; do
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "$f"; else shasum -a 256 "$f"; fi
  done ) >"$SUMS_TMP" && cat "$SUMS_TMP" >"$NEW/SHA256SUMS"
rm -f "$SUMS_TMP"
# 止まった収録先を置き換える（確認 1 で生データから再現できることを確かめた後なので、失われるものは無い）
rm -rf "$DEST" && cp -R "$NEW" "$DEST" || die "収録先の置き換えに失敗"
echo "収録完了（recollect）: $DEST（judge=${JRC_NEW}）"
[[ "$JRC_NEW" == "0" ]]
