#!/usr/bin/env bash
# イシュー #3173: GB10／Mac mini での HTTPS 取得の実機確認の実行スクリプト。手動実行専用（CI からは呼ばない）。
# 規則は同ディレクトリの RULE.txt（事前登録）が正。判定は各 cargo test の終了コード（exit_codes.txt）のみ。
#
# 終了コード: 0=全実行 exit 0／1=どれかが非 0 またはマスク漏れ・残留（記録は残る）／2=引数・前提不正。
# 使い方（ノード上）:
#   bash run_https_download.sh <tree-abs-path> <out-abs-path> <env-file-abs-path>
#   env-file: 同期ツリーの外の一時ファイル（mode 600）。ENV_URL／ENV_SHA／BAD_TLS_URLS／BLACKHOLE_URL を
#   export 形式で持つ（コマンドラインに URL を載せない。set -x は使わない）
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "usage: $0 <tree-abs-path> <out-abs-path> <env-file-abs-path>" >&2
  exit 2
fi
TREE=$1
OUT=$2
ENVF=$3
for p in "$TREE" "$OUT" "$ENVF"; do
  # 改行を含む値が grep で通らないよう bash の全体一致で検査する（A03）。
  if ! [[ $p =~ ^/[A-Za-z0-9._/-]+$ ]]; then
    echo "絶対パスかつ許可文字のみが必要" >&2
    exit 2
  fi
done
[ -d "$TREE" ] || { echo "tree が存在しない" >&2; exit 2; }
[ ! -e "$OUT" ] || { echo "出力先が既に存在する（上書きしない）" >&2; exit 2; }
[ -f "$ENVF" ] || { echo "env-file が無い" >&2; exit 2; }
[ -f "$TREE/.rev-stamp" ] || { echo ".rev-stamp が無い" >&2; exit 2; }
export LC_ALL=C
STAMP_SIZE=$(wc -c <"$TREE/.rev-stamp")
STAMP=$(
  cat "$TREE/.rev-stamp"
  printf x
)
STAMP=${STAMP%x}
STAMP_BODY=${STAMP%$'\n'}
if { [ "$STAMP_SIZE" -ne 40 ] && [ "$STAMP_SIZE" -ne 41 ]; } || ! [[ $STAMP_BODY =~ ^[0-9a-f]{40}$ ]]; then
  echo ".rev-stamp が 40 桁の小文字 16 進数 1 行だけでない" >&2
  exit 2
fi

# shellcheck disable=SC1090
. "$ENVF"
: "${ENV_URL:?}" "${ENV_SHA:?}" "${BAD_TLS_URLS:?}" "${BLACKHOLE_URL:?}"

# 兄弟イシューの計測を妨げない（RULE.txt concurrency）。
if pgrep -x cargo >/dev/null 2>&1 || pgrep -x rustc >/dev/null 2>&1; then
  echo "他の cargo／rustc が実行中。待ってから再実行する" >&2
  exit 2
fi

mkdir -p "$OUT"
cp "$TREE/.rev-stamp" "$OUT/.rev-stamp"

# 環境情報は allowlist で抽出する（env・set の全文は貼らない）。
{
  echo "uname: $(uname -sm)"
  if [ -r /etc/os-release ]; then
    echo "os: $(. /etc/os-release && echo "$PRETTY_NAME")"
  else
    echo "os: $( (sw_vers -productName; sw_vers -productVersion) 2>/dev/null | tr '\n' ' ')"
  fi
  echo "cpu: $( (grep -m1 'model name' /proc/cpuinfo || sysctl -n machdep.cpu.brand_string) 2>/dev/null || echo 取得不能)"
  echo "rustc: $( (rustc -V) 2>/dev/null || echo 取得不能)"
  echo "cargo: $( (cargo -V) 2>/dev/null || echo 取得不能)"
  echo "perl: $( (perl -v | sed -n 2p) 2>/dev/null || echo なし)"
  echo "make: $( (make --version | head -1) 2>/dev/null || echo なし)"
  echo "system_openssl: $( (openssl version) 2>/dev/null || echo なし)"
  if command -v dpkg-query >/dev/null 2>&1; then
    echo "libssl-dev: $( (dpkg-query -W -f='${Status}' libssl-dev) 2>/dev/null || echo 未導入)"
  fi
} >"$OUT/env_info.raw"

cd "$TREE"
{
  echo "== cargo tree -p fandhe-ai -e normal -i openssl-src（先頭 5 行）"
  cargo tree -p fandhe-ai -e normal -i openssl-src 2>&1 | head -5 || true
} >>"$OUT/env_info.raw"

# ビルドを先に終え、計測中にコンパイルを並走させない（RULE.txt build_then_run）。
B0=$(date +%s)
set +e
cargo test -p fandhe-ai --release --locked --test model_download_real_network --no-run >"$OUT/build.raw" 2>&1
BRC=$?
set -e
echo "build rc=$BRC seconds=$(($(date +%s) - B0))" >"$OUT/exit_codes.raw"

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

# マスクしてから収録し、残っていれば fail-closed で止める（残った場合は raw を残さず削除する）。
HN=$(hostname)
HS=$(hostname -s 2>/dev/null || echo "$HN")
mask() { # mask <raw> <final>
  sed -e "s|$HN|<node>|g" -e "s|$HS|<node>|g" -e "s|$HOME|<home>|g" -e "s|${USER:-__nouser__}|<user>|g" "$1" >"$2"
  rm -f "$1"
  for s in "$HN" "$HS" "$HOME" "${USER:-}" "$ENV_URL" "$ENV_SHA"; do
    if [ -n "$s" ] && grep -Fq -- "$s" "$2"; then
      echo "マスク漏れ・残留: $2" >&2
      rm -f "$2"
      exit 1
    fi
  done
  if grep -Eq 'X-Amz-|Signature=|\?[A-Za-z_]+=' "$2"; then
    echo "署名付きクエリの疑い: $2" >&2
    rm -f "$2"
    exit 1
  fi
}
for f in "$OUT"/*.raw; do
  mask "$f" "${f%.raw}.log"
done
mv "$OUT/exit_codes.log" "$OUT/exit_codes.txt"
mv "$OUT/env_info.log" "$OUT/env_info.txt"
mv "$OUT/redirect_probe.log" "$OUT/redirect_probe.txt"

cat "$OUT/exit_codes.txt"
# 記録とマスクを終えた後で、どれかが非 0 なら非 0 で終了する（握りつぶさない）。
if grep -Eq 'rc=[1-9]' "$OUT/exit_codes.txt"; then
  exit 1
fi
