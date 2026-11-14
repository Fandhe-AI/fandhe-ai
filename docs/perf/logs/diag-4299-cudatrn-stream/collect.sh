#!/usr/bin/env bash
# イシュー #4299: G-CUDA-TRN の stream 種別 4 腕 A/B の起動役と収録（GB10〈<cuda-node>〉専用）。
# 役割: 実機ロックの取得・解放（時刻を hwlock.log へ記録）→ ランナー（scripts/bench/framework-compare/run_diag_4299_cudatrn_stream.sh）
#   → 判定器（judge_diag_4299.py）→ 生データの収録（マスク・漏れ検査・SHA256SUMS）を 1 回で行う。計測中はビルドも別負荷も並走させない。
# 前提: RULE.txt は計測より前に固定してコミット済み。AB_BASE_ROOT は RULE の BASE SHA の clone（.rev-stamp つき）、
#   AB_RUNNER_ROOT は RULE.txt を含むコミットの clone（.rev-stamp つき）。本スクリプト・ランナー・判定器は RULE の [道具の sha256] で固定する。
# マスク（#4168 の recollect.sh と同じ規則。#4156 所有者判断 2026-11-11 の語境界の定義）:
#   ホームの絶対パス → <home>、ホスト名（FQDN と短縮名）→ <cuda-node>（どちらも文字どおりの部分一致）、
#   ユーザー名 → <user>（語境界つき。語の文字は英数字・下線・ハイフン。前後がそのどれでもないときだけ一致する。
#   部分一致だと crate 名などの記録に要る語の一部まで書き換えるため。#4168 の既知の不具合）。
#   マスクの後に漏れ検査（ホーム・ホスト名は部分一致、ユーザー名は語境界つき、IPv4 らしき文字列）を行い、1 件でもあれば収録しない。
#   その後で hwlock.log を確定し SHA256SUMS を書く。実名は docs/real-hardware-verification-env.local.md を見るだけで転記しない。
# 使い方:
#   AB_BASE_ROOT=<abs> AB_RUNNER_ROOT=<abs> bash collect.sh             # 本計測（16 ラウンド）と収録
#   AB_BASE_ROOT=<abs> AB_RUNNER_ROOT=<abs> bash collect.sh --dry-run   # ロックを取り、ランナーの --dry-run（ビルドと事前確認だけ）
#   bash collect.sh --self-test   # マスク・漏れ検査・複写・SHA256SUMS の自己検査（合成値と実行環境の値。値は表示しない）
# bash 3.2 互換（mapfile・連想配列・eval 不使用）。変数の直後に全角文字を置かない（${VAR} と書く。#4168 は `$DEST（` で set -u が落ちた）。
set -uo pipefail

NROUNDS=16
LABEL=gb10
die() { echo "error: $*" >&2; exit 1; }

# ---- マスクと漏れ検査（recollect.sh〈#4168〉と同じ式。ホストの置換先だけ <cuda-node>）
# ユーザー名の語境界つきの置換・検査（ERE。\b は使わない。GNU と BSD の sed／grep の両方で動く文字クラス）
re_escape() { printf '%s' "$1" | sed -e 's#[][\\.^$*+?(){}|/#]#\\&#g'; }
user_ere() { printf '(^|[^A-Za-z0-9_-])%s([^A-Za-z0-9_-]|$)' "$(re_escape "$1")"; }
# 隣り合う出現は 1 回の置換で取りこぼすため、ユーザー名の式は 2 回かける（残りは漏れ検査が拾う）。
mask_new() { # mask_new <file> <home> <host> <host-short> <user>
  local f=$1 h=$2 hn=$3 hns=$4 un=$5 t ue rc
  local args=()
  # 式は配列で渡す（未クォートの展開に [ ] が入るとパス名展開を受けるため）
  if [[ -n "$h" ]]; then args+=(-e "s#$(re_escape "$h")#<home>#g"); fi
  if [[ -n "$hn" ]]; then args+=(-e "s#$(re_escape "$hn")#<cuda-node>#g"); fi
  if [[ -n "$hns" ]]; then args+=(-e "s#$(re_escape "$hns")#<cuda-node>#g"); fi
  if [[ -n "$un" ]]; then ue="$(user_ere "$un")"; args+=(-e "s#${ue}#\\1<user>\\2#g" -e "s#${ue}#\\1<user>\\2#g"); fi
  [[ "${#args[@]}" -gt 0 ]] || return 0
  t="$(mktemp)" || return 1
  LC_ALL=C sed -E "${args[@]}" "$f" >"$t" && cat "$t" >"$f"
  rc=$?; rm -f "$t"; return "$rc"
}
# 漏れ検査（ディレクトリ再帰。件数だけ標準出力へ返す。値は表示しない）。
# grep の終了コードは 0=検出・1=不一致・それ以外=検査失敗。検査失敗は「漏れなし」と扱わず rc 2 を返す（fail-closed）。
leak_count() { # leak_count <dir> <home> <host> <host-short> <user>
  local d=$1 n=0 needle g
  [[ -d "$d" ]] || { echo "error: 検査対象が無い" >&2; return 2; }
  for needle in "$2" "$3" "$4"; do
    [[ -n "$needle" ]] || continue
    grep -rFq -- "$needle" "$d"; g=$?
    case "$g" in 0) echo "LEAK: '${needle:0:2}…' が収録物に残る" >&2; n=$((n + 1)) ;; 1) ;; *) echo "error: 漏れ検査の grep が失敗（rc=${g}）" >&2; return 2 ;; esac
  done
  if [[ -n "$5" ]]; then
    grep -rEq -- "$(user_ere "$5")" "$d"; g=$?
    case "$g" in 0) echo "LEAK: ユーザー名（語境界つき）が収録物に残る" >&2; n=$((n + 1)) ;; 1) ;; *) echo "error: 漏れ検査の grep が失敗（rc=${g}）" >&2; return 2 ;; esac
  fi
  grep -rEq '([0-9]{1,3}\.){3}[0-9]{1,3}' "$d" --include='*.txt' --include='*.log' --include='*.md' --include='*.tsv'; g=$?
  case "$g" in 0) echo "LEAK: IP アドレスらしき文字列が残る" >&2; n=$((n + 1)) ;; 1) ;; *) echo "error: 漏れ検査の grep が失敗（rc=${g}）" >&2; return 2 ;; esac
  echo "$n"
}
# マスクの一括適用（txt/log/md/tsv。1 件でも失敗すれば非 0。パイプの while は失敗を握りつぶすのでプロセス置換で回す）
mask_tree() { # mask_tree <dir> <mask 関数名> [関数への追加引数...]
  local d=$1 fn=$2 f bad=0
  shift 2
  while IFS= read -r f; do
    if [[ "$f" == "__FIND_FAILED__" ]]; then bad=$((bad + 1)); continue; fi
    "$fn" "$f" "$@" || bad=$((bad + 1))
  done < <(find "$d" -type f \( -name '*.txt' -o -name '*.log' -o -name '*.md' -o -name '*.tsv' \) || echo "__FIND_FAILED__")
  [[ "$bad" -eq 0 ]] || { echo "error: マスクの適用に ${bad} 件失敗" >&2; return 1; }
}
# SHA256SUMS の作成（各ハッシュと書き込みの成否を確かめる）
sha_of() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1"; else shasum -a 256 "$1"; fi; }
write_sums() { # write_sums <dir>
  local d=$1 f tmp bad=0
  tmp="$(mktemp)" || return 1
  while IFS= read -r f; do
    [[ "$f" != "__FIND_FAILED__" ]] || { bad=$((bad + 1)); continue; }
    (cd "$d" && sha_of "$f") >>"$tmp" || bad=$((bad + 1))
  done < <(cd "$d" && find . -type f | LC_ALL=C sort || echo "__FIND_FAILED__")
  if [[ "$bad" -ne 0 || ! -s "$tmp" ]]; then rm -f "$tmp"; echo "error: SHA256SUMS の作成に失敗" >&2; return 1; fi
  cat "$tmp" >"$d/SHA256SUMS" || { rm -f "$tmp"; return 1; }
  rm -f "$tmp"
}

# ---- 生データの複写（ログと JSONL・起動記録だけ。target-*・バイナリ・build-*.err・各起動の *.err は収録しない）
copy_raw() { # copy_raw <raw> <dest>
  local raw=$1 dest=$2 f i rd
  for f in env_info.txt gate.log skipped.log tree-A.txt tree-B.txt compute_apps.log; do
    if [[ -f "$raw/$f" ]]; then cp "$raw/$f" "$dest/$f" || return 1; fi
  done
  i=1
  while [[ "$i" -le "$NROUNDS" ]]; do
    rd="$raw/round$i"
    if [[ -d "$rd" ]]; then
      mkdir -p "$dest/round$i" || return 1
      for f in "$rd"/*.jsonl "$rd"/launch.tsv "$rd"/uptime_before.txt "$rd"/uptime_after.txt "$rd"/gpu_apps_before.txt "$rd"/gpu_apps_after.txt; do
        if [[ -f "$f" ]]; then cp "$f" "$dest/round$i/" || return 1; fi
      done
    fi
    i=$((i + 1))
  done
}
# マスク → 漏れ検査（0 件でなければ非 0）
mask_and_check() { # mask_and_check <dir> <home> <host> <host-short> <user>
  local d=$1 leaks
  mask_tree "$d" mask_new "$2" "$3" "$4" "$5" || return 1
  leaks="$(leak_count "$d" "$2" "$3" "$4" "$5")" || return 1
  [[ "$leaks" == "0" ]] || { echo "error: マスク検査に失敗（${leaks} 項目）" >&2; return 1; }
}

self_test() {
  local d bad=0 n=0 pair host user line i f raw dest r arm mode
  d="$(mktemp -d "${TMPDIR:-/tmp}/collect4299-selftest.XXXXXX")" || return 1
  for pair in "thost-01.example tuser9" "$(hostname 2>/dev/null || echo __nohost__) $(id -un 2>/dev/null || echo __nouser__)"; do
    host="${pair% *}"; user="${pair#* }"
    # 検出・置換されるべき行（マスク後は漏れ 0 件になること）
    i=0
    for line in "run by ${user} at 00:00" "${user}" "${user}: done" "owner=${user}" "path ${user}/x" "${user} ${user} ${user}" "on ${host} now" "home /h0me/${user}/w"; do
      i=$((i + 1)); mkdir -p "$d/pos$i"; f="$d/pos$i/a.txt"; printf '%s\n' "$line" >"$f"
      n=$((n + 1)); [[ "$(leak_count "$d/pos$i" "/nonexistent-home" "$host" "" "$user" 2>/dev/null)" != "0" ]] || { bad=$((bad + 1)); echo "self-test FAIL: 検出されるべき行 ${i}" >&2; }
      mask_new "$f" "/nonexistent-home" "$host" "" "$user" || { bad=$((bad + 1)); echo "self-test FAIL: マスク失敗 ${i}" >&2; }
      n=$((n + 1)); [[ "$(leak_count "$d/pos$i" "/nonexistent-home" "$host" "" "$user" 2>/dev/null)" == "0" ]] || { bad=$((bad + 1)); echo "self-test FAIL: マスク後に残る行 ${i}" >&2; }
    done
    # 書き換えも検出もされないべき行（ユーザー名が別の語の一部。crate 名の一部など）
    i=0
    for line in "x${user}y" "${user}_z" "z_${user}" "${user}9q" "${user}-z" "z-${user}" "z-${user}-w"; do
      i=$((i + 1)); mkdir -p "$d/neg$i"; f="$d/neg$i/a.txt"; printf '%s\n' "$line" >"$f"
      n=$((n + 1)); [[ "$(leak_count "$d/neg$i" "/nonexistent-home" "__nohost__" "" "$user" 2>/dev/null)" == "0" ]] || { bad=$((bad + 1)); echo "self-test FAIL: 検出されないべき行 ${i}" >&2; }
      mask_new "$f" "/nonexistent-home" "__nohost__" "" "$user"
      n=$((n + 1)); [[ "$(cat "$f")" == "$line" ]] || { bad=$((bad + 1)); echo "self-test FAIL: 書き換えられないべき行 ${i}" >&2; }
    done
    rm -rf "$d"/pos* "$d"/neg*
  done
  # ホームの置換（正規表現の特殊文字を含むパスでも文字どおりに置換される）とホストの置換先
  mkdir -p "$d/home"; printf 'p /tmp/a.b+c/x on thost-01.example\n' >"$d/home/a.log"
  mask_new "$d/home/a.log" "/tmp/a.b+c" "thost-01.example" "thost-01" ""
  n=$((n + 1)); [[ "$(cat "$d/home/a.log")" == "p <home>/x on <cuda-node>" ]] || { bad=$((bad + 1)); echo "self-test FAIL: ホーム・ホストの置換" >&2; }
  # 検査失敗の扱い（root では読み書きの拒否が起きないので飛ばす）
  if [[ "$(id -u)" != "0" ]]; then
    mkdir -p "$d/unread"; printf 'x\n' >"$d/unread/a.txt"; chmod 000 "$d/unread/a.txt"
    n=$((n + 1)); leak_count "$d/unread" "/nonexistent-home" "__nohost__" "" "tuser9" >/dev/null 2>&1; [[ $? -eq 2 ]] || { bad=$((bad + 1)); echo "self-test FAIL: 読めないファイルで漏れ検査が失敗を返さない" >&2; }
    chmod 600 "$d/unread/a.txt"
    mkdir -p "$d/ro"; printf 'tuser9\n' >"$d/ro/a.txt"; chmod 444 "$d/ro/a.txt"
    n=$((n + 1)); if mask_tree "$d/ro" mask_new "/nonexistent-home" "" "" "tuser9" 2>/dev/null; then bad=$((bad + 1)); echo "self-test FAIL: 書けないファイルでマスクの一括適用が失敗を返さない" >&2; fi
    chmod 644 "$d/ro/a.txt"
  fi
  n=$((n + 1)); leak_count "$d/__absent__" "" "" "" "" >/dev/null 2>&1; [[ $? -eq 2 ]] || { bad=$((bad + 1)); echo "self-test FAIL: 検査対象が無いときに失敗を返さない" >&2; }
  # 収録の段の通し（合成の 16 ラウンド。複写 → マスク → 漏れ検査 → SHA256SUMS → sha256 の照合）
  raw="$d/raw"; dest="$d/dest"; mkdir -p "$raw" "$dest"
  printf 'uname=Linux x aarch64\nowner=tuser9\nroot=/tmp/fakehome-4299/work/base\n' >"$raw/env_info.txt"
  printf 'round1 gate=pass threshold=1.0 waited_s=0 load1=0.5\n' >"$raw/gate.log"
  : >"$raw/skipped.log"
  printf 'fandhe-ai v0.14.0 (/tmp/fakehome-4299/work/base/crates/facade)\nfandhe-ai-backend-cuda v0.14.0\n' >"$raw/tree-B.txt"
  printf '2026-01-01T00:00:00Z rc=0\n\n' >"$raw/compute_apps.log"
  r=1
  while [[ "$r" -le "$NROUNDS" ]]; do
    mkdir -p "$raw/round$r"
    for arm in A B C D; do
      for mode in reuse fresh; do
        printf '{"framework":"fandhe-ai","task":"train","median_s":0.000135,"checksum":0.080541,"mode":"%s"}\n' "$mode" >"$raw/round$r/${arm}-${mode}.jsonl"
        printf 'err on thost-01.example by tuser9\n' >"$raw/round$r/${arm}-${mode}.err"
      done
    done
    printf '%s\t1\tA\treuse\tnone\tbefore\t0\tt0\tt1\n' "$r" >"$raw/round$r/launch.tsv"
    printf ' 00:00 up 1 day, load average: 0.50, 0.40, 0.30\n' >"$raw/round$r/uptime_before.txt"
    r=$((r + 1))
  done
  mkdir -p "$raw/target-A/release"; printf 'bin\n' >"$raw/target-A/release/bench-fandhe"
  n=$((n + 1)); copy_raw "$raw" "$dest" || { bad=$((bad + 1)); echo "self-test FAIL: 複写" >&2; }
  n=$((n + 1)); [[ "$(find "$dest" -name '*.jsonl' | wc -l | tr -d ' ')" == "$((NROUNDS * 8))" ]] || { bad=$((bad + 1)); echo "self-test FAIL: JSONL の件数" >&2; }
  n=$((n + 1)); [[ -z "$(find "$dest" -name '*.err' -o -name 'target-*' -o -name 'bench-fandhe')" ]] || { bad=$((bad + 1)); echo "self-test FAIL: 収録しないものが複写された" >&2; }
  n=$((n + 1)); [[ "$(leak_count "$dest" "/tmp/fakehome-4299" "thost-01.example" "thost-01" "tuser9" 2>/dev/null)" != "0" ]] || { bad=$((bad + 1)); echo "self-test FAIL: マスク前に漏れを検出しない" >&2; }
  n=$((n + 1)); mask_and_check "$dest" "/tmp/fakehome-4299" "thost-01.example" "thost-01" "tuser9" 2>/dev/null || { bad=$((bad + 1)); echo "self-test FAIL: マスク後の漏れ検査" >&2; }
  n=$((n + 1))
  if ! { grep -q 'fandhe-ai-backend-cuda v0.14.0' "$dest/tree-B.txt" && grep -q '^owner=<user>$' "$dest/env_info.txt" \
    && grep -q '^root=<home>/work/base$' "$dest/env_info.txt"; }; then bad=$((bad + 1)); echo "self-test FAIL: マスクの結果" >&2; fi
  printf 'line 10.0.0.1\n' >"$dest/ip.md"
  n=$((n + 1)); [[ "$(leak_count "$dest" "" "" "" "" 2>/dev/null)" == "1" ]] || { bad=$((bad + 1)); echo "self-test FAIL: IPv4 らしき文字列を検出しない" >&2; }
  rm -f "$dest/ip.md"
  n=$((n + 1)); write_sums "$dest" || { bad=$((bad + 1)); echo "self-test FAIL: SHA256SUMS の作成" >&2; }
  n=$((n + 1)); [[ "$(wc -l <"$dest/SHA256SUMS" | tr -d ' ')" == "$(cd "$dest" && find . -type f ! -name SHA256SUMS | wc -l | tr -d ' ')" ]] || { bad=$((bad + 1)); echo "self-test FAIL: SHA256SUMS の行数" >&2; }
  if command -v sha256sum >/dev/null 2>&1; then
    n=$((n + 1)); (cd "$dest" && grep -v ' \./SHA256SUMS$' SHA256SUMS | sha256sum -c --quiet -) >/dev/null 2>&1 || { bad=$((bad + 1)); echo "self-test FAIL: SHA256SUMS の照合" >&2; }
  fi
  rm -rf "$d"
  echo "self-test: ${n} 件中 FAIL ${bad} 件"
  [[ "$bad" -eq 0 ]]
}

DRY=0
case "${1:-}" in
  --self-test) [[ $# -eq 1 ]] || die "--self-test は単独で使う"; self_test; exit $? ;;
  --dry-run) DRY=1; shift ;;
esac
[[ $# -eq 0 ]] || die "引数は --dry-run か --self-test だけ"

for v in AB_BASE_ROOT AB_RUNNER_ROOT; do
  p=${!v:-}
  if [[ -z "$p" || "$p" != /* || "$p" == *\"* || "$p" == *\\* || "$p" == *\'* || "$p" == *$'\n'* || "$p" == *#* ]]; then
    die "${v} は引用符・バックスラッシュ・#・改行を含まない絶対パス必須"
  fi
done
RUNNER_ROOT=$AB_RUNNER_ROOT
RUNNER="$RUNNER_ROOT/scripts/bench/framework-compare/run_diag_4299_cudatrn_stream.sh"
JUDGE="$RUNNER_ROOT/scripts/bench/framework-compare/judge_diag_4299.py"
LOGS="$RUNNER_ROOT/docs/perf/logs/diag-4299-cudatrn-stream"
RULE="$LOGS/RULE.txt"
RAW="$RUNNER_ROOT/scripts/bench/framework-compare/results/raw/diag4299-${LABEL}"
DEST="$LOGS/$LABEL"
[[ -f "$RUNNER" && -f "$JUDGE" && -f "$RULE" ]] || die "ランナー・判定器・RULE.txt のいずれかが無い"
if [[ "$DRY" == "0" ]]; then
  # 計測の前に確かめる（測ってから収録先が埋まっていると気付いても取り返せない）
  [[ ! -e "$DEST" ]] || die "収録先が既に存在する（差し替え禁止）: ${DEST}"
  [[ ! -e "$RAW" ]] || die "生データの出力先が既に存在する（差し替え禁止）: ${RAW}"
fi

LOCK_DIR="${HOME}/work/.fandhe-hw-lock"
OWNER="#4299 diag-cudatrn-stream $(date -u +%FT%TZ)"
mkdir -p "${HOME}/work"
mkdir "$LOCK_DIR" 2>/dev/null || die "実機ロックを取得できない（他の計測が保持中）"
printf '%s\n' "$OWNER" >"$LOCK_DIR/owner"
LOCK_T0="$(date -u +%FT%TZ)"
STAGE=""
release() {
  if [[ -n "$STAGE" ]]; then rm -rf "$STAGE"; fi
  if [[ "$(cat "$LOCK_DIR/owner" 2>/dev/null)" == "$OWNER" ]]; then rm -rf "$LOCK_DIR"; echo "lock released $(date -u +%FT%TZ)"; else echo "LOCK-LOST: 解放しない"; fi
}
trap release EXIT
trap 'exit 129' HUP; trap 'exit 130' INT; trap 'exit 143' TERM
export AB_HWLOCK_OWNER="$OWNER"

if [[ "$DRY" == "1" ]]; then
  bash "$RUNNER" --dry-run "$LABEL"; rc=$?
  echo "dry-run rc=${rc}"
  exit "$rc"
fi

bash "$RUNNER" "$LABEL"; rc=$?
echo "runner rc=${rc}"
[[ -d "$RAW" ]] || die "出力が無い: ${RAW}"

# 収録物は一時ディレクトリで組み立て、マスクと漏れ検査を通してから収録先へ写す（未マスクのものをリポジトリ内に置かない）。
STAGE="$(mktemp -d "${TMPDIR:-/tmp}/collect4299.XXXXXX")" || die "一時ディレクトリを作れない"
# 判定器（生成物は手で書き換えない）。RULE・道具の sha256・ゲートの整合も判定器が検査する。
python3 -I "$JUDGE" --series-dir "$RAW" --env-info "$RAW/env_info.txt" --rule "$RULE" --gate-log "$RAW/gate.log" \
  --compute-log "$RAW/compute_apps.log" --ratios-out "$STAGE/ratios.tsv" >"$STAGE/judge-4299-${LABEL}.md"; jrc=$?
echo "$jrc" >"$STAGE/judge-4299-${LABEL}.exit"
echo "judge rc=${jrc}"
copy_raw "$RAW" "$STAGE" || die "生データの複写に失敗（生データは ${RAW} に残る）"
CS_SHA="$(sha_of "${BASH_SOURCE[0]}" | cut -d' ' -f1)"
printf 'runner_rc=%s\njudge_rc=%s\ncollect_sh_sha256=%s\ncollected=%s\n' "$rc" "$jrc" "$CS_SHA" "$(date -u +%FT%TZ)" >"$STAGE/collect.log" \
  || die "collect.log を書けない"

H="${HOME}"; HN="$(hostname 2>/dev/null || true)"; HNS="$(hostname -s 2>/dev/null || true)"; UN="$(id -un)"
mask_and_check "$STAGE" "$H" "$HN" "$HNS" "$UN" || die "マスク検査に失敗。収録しない（生データは ${RAW} に残る）"
printf 'acquired=%s\nreleased=%s\n' "$LOCK_T0" "$(date -u +%FT%TZ)" >"$STAGE/hwlock.log" || die "hwlock.log を書けない"
write_sums "$STAGE" || die "SHA256SUMS を作れない。収録しない"
mkdir -p "$DEST" || die "収録先を作れない"
cp -R "$STAGE/." "$DEST/" || die "収録物の複写に失敗（${DEST} を確かめて削除する）"
diff -rq "$STAGE" "$DEST" >/dev/null 2>&1 || die "収録先が組み立てたものと一致しない（${DEST} を確かめて削除する）"
echo "収録完了: ${DEST}（runner rc=${rc} judge=${jrc}）"
[[ "$rc" -eq 0 && "$jrc" -eq 0 ]]
