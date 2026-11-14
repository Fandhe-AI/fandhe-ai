#!/usr/bin/env bash
# 0.15.0 同席計測（#4177 Mac mini・#4178 GB10）: 収録時のマスク・漏れ検査・SHA256SUMS の関数群（source 専用。両機体で同一内容）。
# 役割: collect.sh（収録）と self-test.sh（単体テスト）から source される。0.14.0 では手作業だったマスクの機械化。
#   docs/perf/logs/prepost-0150-4168/recollect.sh の語境界つきマスクを写した（ユーザー名の定義は #4156 所有者判断 2026-11-11 と同じ:
#   語の文字は英数字・下線・ハイフンで、前後がそのどれでもないときだけ一致する。ユーザー名が別の語の一部なら置換も検出もしない）。
# 置換先: ホーム → <home>、ホスト名（完全名・短縮名）→ 呼び出し元が渡す <metal-node>／<cuda-node>、ユーザー名 → <user>。
#   ホームとホスト名は部分一致（recollect.sh と同じ）。実名はこのファイルに書かず、呼び出し元が実行時に hostname・$HOME・id -un から取る。
# 漏れ検査は fail-closed: 1 件でも残れば collect.sh が非 0 で止まり、検査自体の失敗（読めない・grep 異常）も「漏れなし」と扱わない。
# bash 3.2 互換（連想配列・mapfile・${v,,} なし。`${VAR}` 形式。空配列を set -u で展開しない）。

# 正規表現の特殊文字を sed／ERE 用にエスケープ
re_escape() { printf '%s' "$1" | sed -e 's#[][\\.^$*+?(){}|/#]#\\&#g'; }
# ユーザー名の語境界つき ERE（\b は GNU と BSD で挙動が違うため使わない）
user_ere() { printf '(^|[^A-Za-z0-9_-])%s([^A-Za-z0-9_-]|$)' "$(re_escape "$1")"; }

# mask_file <file> <home> <host> <host-short> <user> <ノード置換先>: 1 ファイルをその場で置換。失敗は非 0。
# ユーザー名は隣り合う出現（`u u`）が 1 回の置換で取りこぼされるため 2 回かける。
mask_file() {
  local f=$1 h=$2 hn=$3 hns=$4 un=$5 np=$6 t ue rc n=0
  local a1="" a2="" a3="" a4="" a5=""
  [ -n "${h}" ] && { a1="s#$(re_escape "${h}")#<home>#g"; n=$((n + 1)); }
  [ -n "${hn}" ] && { a2="s#$(re_escape "${hn}")#${np}#g"; n=$((n + 1)); }
  [ -n "${hns}" ] && { a3="s#$(re_escape "${hns}")#${np}#g"; n=$((n + 1)); }
  if [ -n "${un}" ]; then
    ue="$(user_ere "${un}")"
    a4="s#${ue}#\\1<user>\\2#g"; a5="${a4}"; n=$((n + 1))
  fi
  [ "${n}" -gt 0 ] || return 0
  t="$(mktemp)" || return 1
  # 空の式は sed に渡さない（-e '' は無害だが、意図を明確にするため固定の 5 本のうち空はコメント式にする）
  LC_ALL=C sed -E -e "${a1:-#}" -e "${a2:-#}" -e "${a3:-#}" -e "${a4:-#}" -e "${a5:-#}" "${f}" > "${t}" && cat "${t}" > "${f}"
  rc=$?; rm -f "${t}"; return "${rc}"
}

# grep_files_ere <ERE> <ファイル一覧を 1 行 1 件で標準入力>: 0 = どれかが一致／1 = 不一致／2 = 検査失敗
grep_files_ere() {
  local f g any=1
  while IFS= read -r f; do
    [ -n "${f}" ] || continue
    grep -Eq -- "$1" "${f}"; g=$?
    case "${g}" in 0) any=0 ;; 1) ;; *) return 2 ;; esac
  done
  return "${any}"
}

# leak_count <dir> <home> <host> <host-short> <user>: 漏れの項目数を標準出力へ（値は出さない）。検査失敗は rc 2。
# ホーム・ホスト名は部分一致、ユーザー名は語境界つき、IPv4 らしき文字列は *.txt・*.log・*.md から探す。
# IPv4 検査は pip freeze の収録（pip-freeze-*）を除く（`nvidia-cudnn-cu13==9.20.0.48` のような 4 区切りの版番号に当たるため）。
leak_count() {
  local d=$1 n=0 needle g files
  [ -d "${d}" ] || { echo "error: 検査対象が無い" >&2; return 2; }
  for needle in "$2" "$3" "$4"; do
    [ -n "${needle}" ] || continue
    grep -rFq -- "${needle}" "${d}"; g=$?
    case "${g}" in 0) echo "LEAK: '$(printf '%s' "${needle}" | cut -c1-2)…' が収録物に残る" >&2; n=$((n + 1)) ;; 1) ;; *) echo "error: 漏れ検査の grep が失敗（rc=${g}）" >&2; return 2 ;; esac
  done
  if [ -n "$5" ]; then
    grep -rEq -- "$(user_ere "$5")" "${d}"; g=$?
    case "${g}" in 0) echo "LEAK: ユーザー名（語境界つき）が収録物に残る" >&2; n=$((n + 1)) ;; 1) ;; *) echo "error: 漏れ検査の grep が失敗（rc=${g}）" >&2; return 2 ;; esac
  fi
  files="$(find "${d}" -type f \( -name '*.txt' -o -name '*.log' -o -name '*.md' \) ! -name 'pip-freeze-*')" || { echo "error: find が失敗" >&2; return 2; }
  printf '%s\n' "${files}" | grep_files_ere '([0-9]{1,3}\.){3}[0-9]{1,3}'; g=$?
  case "${g}" in 0) echo "LEAK: IP アドレスらしき文字列が残る" >&2; n=$((n + 1)) ;; 1) ;; *) echo "error: IP 検査が失敗（rc=${g}）" >&2; return 2 ;; esac
  echo "${n}"
}

# mask_tree <dir> <home> <host> <host-short> <user> <ノード置換先>: txt/log/md/tsv/out/err を全件マスク。1 件でも失敗すれば非 0。
# jsonl は書き換えない（実測値そのもの。名前が入っていれば leak_count が止める）。
mask_tree() {
  local d=$1 f bad=0 list
  shift
  list="$(find "${d}" -type f \( -name '*.txt' -o -name '*.log' -o -name '*.md' -o -name '*.tsv' -o -name '*.out' -o -name '*.err' \))" || { echo "error: find が失敗" >&2; return 1; }
  while IFS= read -r f; do
    [ -n "${f}" ] || continue
    mask_file "${f}" "$@" || bad=$((bad + 1))
  done <<EOF
${list}
EOF
  [ "${bad}" -eq 0 ] || { echo "error: マスクの適用に ${bad} 件失敗" >&2; return 1; }
}

sha_of() { if command -v sha256sum > /dev/null 2>&1; then sha256sum "$1"; else shasum -a 256 "$1"; fi; }

# write_sums <dir>: dir 直下の SHA256SUMS を作る（README.md と SHA256SUMS 自身を除く全ファイル。`<sha>  ./相対パス`。0.14.0 と同じ範囲）。
write_sums() {
  local d=$1 f tmp bad=0 list
  tmp="$(mktemp)" || return 1
  list="$(cd "${d}" && find . -type f ! -path ./SHA256SUMS ! -path ./README.md | LC_ALL=C sort)" || { rm -f "${tmp}"; return 1; }
  while IFS= read -r f; do
    [ -n "${f}" ] || continue
    (cd "${d}" && sha_of "${f}") >> "${tmp}" || bad=$((bad + 1))
  done <<EOF
${list}
EOF
  if [ "${bad}" -ne 0 ] || [ ! -s "${tmp}" ]; then rm -f "${tmp}"; echo "error: SHA256SUMS の作成に失敗" >&2; return 1; fi
  cat "${tmp}" > "${d}/SHA256SUMS" || { rm -f "${tmp}"; return 1; }
  rm -f "${tmp}"
}
