#!/usr/bin/env bash
# イシュー #3173: GB10／Mac mini での HTTPS 取得の実機確認の実行スクリプト。手動実行専用（CI からは呼ばない）。
# 規則は同ディレクトリの RULE.txt（事前登録）が正。判定は各 cargo test の終了コード（exit_codes.txt）のみ。
# 宛先（URL・pin・不正 TLS・ブラックホール）は転送ツリー内の RULE.txt から厳密な正規表現で検証して読む
# （環境変数ファイルを source しない＝A03。実行した宛先と事前登録した宛先が一致することをスクリプトで保証する）。
#
# 終了コード: 0=全実行 exit 0／1=どれかが非 0 またはマスク漏れ・残留（記録は残る）／2=引数・前提不正（テスト未実行）。
# 使い方:
#   bash run_https_download.sh <tree-abs-path> <out-abs-path>   # ノード上で実行（CARGO_TARGET_DIR は呼び出し側で与える）
#   bash run_https_download.sh --fingerprint <dir>             # ツリー指紋（.rev-stamp 除く全ファイルの sha256 の sha256）を出す
#   bash run_https_download.sh --self-test                     # マスクの自己検査（ネットワーク・cargo 不使用）
# set -x は使わない（URL を端末に出さない）。
set -euo pipefail
export LC_ALL=C

sha_cmd() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$@"; else shasum -a 256 "$@"; fi
}

# .rev-stamp 以外の全ファイルの sha256 を相対パス順にまとめ、さらに sha256 を取る。
fingerprint() { # fingerprint <dir>
  (
    cd "$1"
    find . -type f ! -name .rev-stamp | LC_ALL=C sort | while IFS= read -r f; do sha_cmd "$f"; done | sha_cmd | cut -d' ' -f1
  )
}

# 識別子を単語境界つき（直前直後が [A-Za-z0-9_-] でない）で置換する。USER=fandhe でも fandhe-ai／fandhe_ai は化けない。
# 引数: <入力> <出力> と環境変数 M_HOME M_FQDN M_SHORT M_USER。
mask_file() {
  M_IN=$1 M_OUT=$2 perl -e '
    my ($i,$o)=($ENV{M_IN},$ENV{M_OUT});
    open(my $fh,"<",$i) or die; local $/; my $t=<$fh>; close $fh;
    my $b="(?<![A-Za-z0-9_-])"; my $a="(?![A-Za-z0-9_-])";
    for my $p (["M_HOME","<home>"],["M_FQDN","<node>"],["M_SHORT","<node>"],["M_USER","<user>"]) {
      my $v=$ENV{$p->[0]}; next if !defined($v) || $v eq "";
      $t =~ s/$b\Q$v\E$a/$p->[1]/g;
    }
    open(my $oh,">",$o) or die; print $oh $t; close $oh;'
}
# 残留検査。1 件でもあれば非 0。
leak_check() { # leak_check <file>
  M_FILE=$1 perl -e '
    open(my $fh,"<",$ENV{M_FILE}) or die; local $/; my $t=<$fh>; close $fh;
    my $b="(?<![A-Za-z0-9_-])"; my $a="(?![A-Za-z0-9_-])";
    for my $k (qw(M_HOME M_FQDN M_SHORT M_USER)) {
      my $v=$ENV{$k}; next if !defined($v) || $v eq "";
      exit 1 if $t =~ /$b\Q$v\E$a/;
    }
    exit 1 if $t =~ m{/home/|/Users/|X-Amz-|Signature=|\?[A-Za-z_]+=};
    exit 0;'
}

if [ "${1:-}" = "--fingerprint" ]; then
  { [ "$#" -eq 2 ] && [ -d "$2" ]; } || { echo "usage: $0 --fingerprint <dir>" >&2; exit 2; }
  fingerprint "$2"
  exit 0
fi

if [ "${1:-}" = "--self-test" ]; then
  T=$(mktemp -d)
  trap 'rm -rf "$T"' EXIT
  export M_HOME=/home/fandhe M_FQDN=node-x.example M_SHORT=node-x M_USER=fandhe
  cat >"$T/in" <<'IN'
   Compiling fandhe-ai v0.11.0 (/home/fandhe/work/rust-ai-library-run-3173/crates/facade)
   Compiling fandhe_ai_core v0.1.0
 target dir /home/fandhe/work/target-fandhe-ai-3173
 host node-x.example short node-x user fandhe done
IN
  mask_file "$T/in" "$T/out"
  grep -q 'Compiling fandhe-ai v0.11.0 (<home>/work/rust-ai-library-run-3173' "$T/out" || { echo "FAIL(a): クレート名" >&2; exit 1; }
  { grep -q 'fandhe_ai_core' "$T/out" && grep -q 'target-fandhe-ai-3173' "$T/out"; } || { echo "FAIL(a2): 識別子" >&2; exit 1; }
  grep -q 'host <node> short <node> user <user> done' "$T/out" || { echo "FAIL(b): 置換" >&2; exit 1; }
  leak_check "$T/out" || { echo "FAIL(b2): 正しいマスク結果が残留扱い" >&2; exit 1; }
  printf 'x fandhe y\n' >"$T/leak1"
  if leak_check "$T/leak1"; then echo "FAIL(c1): 単独ユーザー名を検出できない" >&2; exit 1; fi
  printf 'x /home/other/y\n' >"$T/leak2"
  if leak_check "$T/leak2"; then echo "FAIL(c2): /home/ を検出できない" >&2; exit 1; fi
  printf 'x node-x.example y\n' >"$T/leak3"
  if leak_check "$T/leak3"; then echo "FAIL(c3): ホスト名を検出できない" >&2; exit 1; fi
  printf 'u?token=abc\n' >"$T/leak4"
  if leak_check "$T/leak4"; then echo "FAIL(c4): クエリを検出できない" >&2; exit 1; fi
  echo "self-test ok"
  exit 0
fi

if [ "$#" -ne 2 ]; then
  echo "usage: $0 <tree-abs-path> <out-abs-path> | --fingerprint <dir> | --self-test" >&2
  exit 2
fi
TREE=$1
OUT=$2
for p in "$TREE" "$OUT"; do
  # 改行を含む値が通らないよう bash の全体一致で検査する（A03）。
  if ! [[ $p =~ ^/[A-Za-z0-9._/-]+$ ]]; then
    echo "絶対パスかつ許可文字のみが必要" >&2
    exit 2
  fi
done
[ -d "$TREE" ] || { echo "tree が存在しない" >&2; exit 2; }
[ ! -e "$OUT" ] || { echo "出力先が既に存在する（上書きしない）" >&2; exit 2; }
[ -f "$TREE/.rev-stamp" ] || { echo ".rev-stamp が無い" >&2; exit 2; }
STAMP_SIZE=$(wc -c <"$TREE/.rev-stamp" | tr -d ' ')
STAMP_BODY=$(cat "$TREE/.rev-stamp")
if [ "$STAMP_SIZE" -ne 41 ] || ! [[ $STAMP_BODY =~ ^[0-9a-f]{40}$ ]]; then
  echo ".rev-stamp が 40 桁の小文字 16 進数＋改行の 1 行だけでない" >&2
  exit 2
fi

# 宛先は RULE.txt の機械可読行（行頭 2 スペース＋キー）から読み、厳密に検証する。
RULE=$TREE/docs/perf/logs/https-download-3173/RULE.txt
[ -f "$RULE" ] || { echo "RULE.txt が無い" >&2; exit 2; }
rule_val() { sed -n "s/^  $1: \([^ ]*\).*/\1/p" "$RULE"; }
URL_RE='^https://[A-Za-z0-9._/-]+$'
ENV_URL=$(rule_val URL)
ENV_SHA=$(rule_val SHA256)
BLACKHOLE_URL=$(rule_val BLACKHOLE)
BAD_TLS_URLS=$(sed -n 's/^  BAD_TLS: //p' "$RULE")
[[ $ENV_URL =~ $URL_RE ]] || { echo "RULE の URL が不正" >&2; exit 2; }
[[ $ENV_SHA =~ ^[0-9a-f]{64}$ ]] || { echo "RULE の SHA256 が不正" >&2; exit 2; }
[[ $BLACKHOLE_URL =~ $URL_RE ]] || { echo "RULE の BLACKHOLE が不正" >&2; exit 2; }
[ -n "$BAD_TLS_URLS" ] || { echo "RULE の BAD_TLS が空" >&2; exit 2; }
for u in $BAD_TLS_URLS; do
  [[ $u =~ $URL_RE ]] || { echo "RULE の BAD_TLS が不正" >&2; exit 2; }
done

# マスク対象（ユーザー名・ホスト名のみ。ホームのパスもユーザー名を含むため含める）。
M_HOME=$HOME
M_FQDN=$(hostname)
M_SHORT=$(hostname -s 2>/dev/null || echo "$M_FQDN")
M_USER=${USER:-$(id -un)}
for v in "$M_FQDN" "$M_SHORT" "$M_USER"; do
  [[ $v =~ ^[A-Za-z0-9._-]+$ ]] || { echo "ホスト名・ユーザー名が許可文字外" >&2; exit 2; }
done
[[ $M_HOME =~ ^/[A-Za-z0-9._/-]+$ ]] || { echo "HOME が許可文字外" >&2; exit 2; }
export M_HOME M_FQDN M_SHORT M_USER

# 前提検査（失敗は exit 2＝テスト未実行。実行に数えない）。
if pgrep -x cargo >/dev/null 2>&1 || pgrep -x rustc >/dev/null 2>&1; then
  echo "他の cargo／rustc が実行中。待ってから再実行する" >&2
  exit 2
fi
OSNAME=$(uname -s)
case $OSNAME in
  Linux)
    CPU=$(lscpu 2>/dev/null | grep 'Model name' || true)
    [ -n "$CPU" ] || CPU=$(tr -d '\0' </proc/device-tree/model 2>/dev/null || true)
    [ -n "$CPU" ] || CPU=$(grep 'model name' /proc/cpuinfo 2>/dev/null | sort -u || true)
    OSDESC=$(. /etc/os-release 2>/dev/null && echo "${PRETTY_NAME:-}") || OSDESC=
    ;;
  Darwin)
    CPU=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || true)
    OSDESC="$(sw_vers -productName 2>/dev/null) $(sw_vers -productVersion 2>/dev/null)"
    ;;
  *) echo "未対応の OS" >&2; exit 2 ;;
esac
[ -n "$CPU" ] || { echo "cpu を取得できない（テスト未実行で中止）" >&2; exit 2; }
command -v cargo >/dev/null 2>&1 || { echo "cargo が無い" >&2; exit 2; }
command -v rustc >/dev/null 2>&1 || { echo "rustc が無い" >&2; exit 2; }
command -v perl >/dev/null 2>&1 || { echo "perl が無い" >&2; exit 2; }

mkdir -p "$OUT"
cp "$TREE/.rev-stamp" "$OUT/.rev-stamp"
FP=$(fingerprint "$TREE")

# 環境情報は allowlist で抽出する（env・set の全文は貼らない）。
{
  echo "uname: $(uname -sm)"
  echo "os: $OSDESC"
  echo "cpu:"
  echo "$CPU" | sed 's/^/  /'
  echo "rustc: $(rustc -V 2>/dev/null || echo 取得不能)"
  echo "cargo: $(cargo -V 2>/dev/null || echo 取得不能)"
  echo "perl: $(perl -v 2>/dev/null | sed -n 2p || true)"
  echo "make: $(make --version 2>/dev/null | head -1 || echo なし)"
  echo "system_openssl: $(openssl version 2>/dev/null || echo なし)"
  if command -v dpkg-query >/dev/null 2>&1; then
    echo "libssl-dev: $(dpkg-query -W -f='${Status}' libssl-dev 2>/dev/null || echo 未導入)"
  fi
  echo "tree_fingerprint_sha256: $FP"
} >"$OUT/env_info.raw"

cd "$TREE"
# ビルドを先に終え、計測中にコンパイルを並走させない（RULE.txt build_then_run）。
B0=$(date +%s)
set +e
cargo test -p fandhe-ai --release --locked --test model_download_real_network --no-run \
  --message-format=json >"$OUT/build_json.raw" 2>"$OUT/build.raw"
BRC=$?
set -e
echo "build rc=$BRC seconds=$(($(date +%s) - B0))" >"$OUT/exit_codes.raw"

# TLS 経路と vendored OpenSSL の証跡（RULE.txt tls_path。判定には使わない）。
HOST_TRIPLE=$(rustc -vV | sed -n 's/^host: //p')
{
  echo "host_triple: $HOST_TRIPLE"
  for c in openssl-src openssl-sys security-framework; do
    echo "== cargo tree -p fandhe-ai --locked -e normal,build --target $HOST_TRIPLE -i $c"
    cargo tree -p fandhe-ai --locked -e normal,build --target "$HOST_TRIPLE" -i "$c" 2>&1 | head -8 || true
  done
  BIN=$(grep '"name":"model_download_real_network"' "$OUT/build_json.raw" | sed -n 's/.*"executable":"\([^"]*\)".*/\1/p' | tail -1)
  if [ -n "$BIN" ] && [ -x "$BIN" ]; then
    if [ "$OSNAME" = Linux ]; then
      echo "binary_links_libssl_or_libcrypto: $(ldd "$BIN" 2>&1 | grep -c -E 'libssl|libcrypto' || true)（0 なら動的リンクなし）"
    else
      echo "binary_links_Security.framework: $(otool -L "$BIN" 2>&1 | grep -c 'Security.framework' || true)"
      echo "binary_links_libssl_or_libcrypto: $(otool -L "$BIN" 2>&1 | grep -c -E 'libssl|libcrypto' || true)"
    fi
  else
    echo "binary: 特定不能"
  fi
  if [ -n "${CARGO_TARGET_DIR:-}" ] && ls -d "$CARGO_TARGET_DIR"/release/build/openssl-sys-*/output >/dev/null 2>&1; then
    echo "openssl-sys_build_output_with_vendored=1: $(grep -l 'vendored=1' "$CARGO_TARGET_DIR"/release/build/openssl-sys-*/output 2>/dev/null | wc -l | tr -d ' ') ファイル"
  else
    echo "openssl-sys_build_output: なし（この構成では openssl-sys をビルドしていない、または CARGO_TARGET_DIR 未指定）"
  fi
} >"$OUT/tls_path.raw" 2>&1
rm -f "$OUT/build_json.raw"

run() { # run <label> <test-name> ; 環境変数は呼び出し側が与える
  set +e
  cargo test -p fandhe-ai --release --locked --test model_download_real_network \
    -- --ignored --nocapture --exact "$2" >"$OUT/$1.raw" 2>&1
  echo "$1 rc=$?" >>"$OUT/exit_codes.raw"
  set -e
}
if [ "$BRC" -eq 0 ]; then
  FANDHE_AI_DOWNLOAD_TEST_URL=$ENV_URL FANDHE_AI_DOWNLOAD_TEST_SHA256=$ENV_SHA \
    run r1 real_network_download_verifies_pin_and_reuses_cache
  FANDHE_AI_DOWNLOAD_TEST_URL=$ENV_URL FANDHE_AI_DOWNLOAD_TEST_SHA256=$ENV_SHA \
    run r2 real_network_wrong_pin_is_rejected
  i=0
  for u in $BAD_TLS_URLS; do
    i=$((i + 1))
    FANDHE_AI_DOWNLOAD_TEST_BAD_TLS_URL=$u run "r3_$i" real_network_bad_tls_certificate_is_rejected
  done
  FANDHE_AI_DOWNLOAD_TEST_BLACKHOLE_URL=$BLACKHOLE_URL run r4 real_network_blackhole_connect_times_out
fi

# リダイレクトの独立観測（-L なし）。Location の値は書き出さず、スキーム・ホスト変化の有無だけ残す。
{
  HDR=$(curl -sS -m 30 -o /dev/null -w '%{http_code} %{redirect_url}' -I "$ENV_URL" 2>&1 || true)
  CODE=${HDR%% *}
  LOC=${HDR#* }
  ORIG_HOST=${ENV_URL#https://}
  ORIG_HOST=${ORIG_HOST%%/*}
  if [ -n "$LOC" ] && [ "$LOC" != "$HDR" ]; then
    LOC_SCHEME=${LOC%%://*}
    LOC_HOST=${LOC#*://}
    LOC_HOST=${LOC_HOST%%/*}
    if [ "$LOC_HOST" = "$ORIG_HOST" ]; then SAME=same_host; else SAME=different_host; fi
    echo "redirect_probe: status=$CODE location_scheme=$LOC_SCHEME host=$SAME"
  else
    echo "redirect_probe: status=$CODE location=none"
  fi
} >"$OUT/redirect_probe.raw"

# マスクしてから収録し、残っていれば fail-closed で止める（raw は残さず削除する）。
FAIL=0
for f in "$OUT"/*.raw; do
  fin="${f%.raw}.log"
  mask_file "$f" "$fin"
  rm -f "$f"
  if ! leak_check "$fin" || grep -Fq -e "$ENV_URL" -e "$ENV_SHA" "$fin"; then
    echo "マスク漏れ・残留・署名付きクエリの疑い: $(basename "$fin")" >&2
    rm -f "$fin"
    FAIL=1
  fi
done
[ "$FAIL" -eq 0 ] || exit 1
mv "$OUT/exit_codes.log" "$OUT/exit_codes.txt"
mv "$OUT/env_info.log" "$OUT/env_info.txt"
mv "$OUT/redirect_probe.log" "$OUT/redirect_probe.txt"
mv "$OUT/tls_path.log" "$OUT/tls_path.txt"

cat "$OUT/exit_codes.txt"
# 記録とマスクを終えた後で、どれかが非 0 なら非 0 で終了する（握りつぶさない）。
if grep -Eq 'rc=[1-9]' "$OUT/exit_codes.txt"; then
  exit 1
fi
