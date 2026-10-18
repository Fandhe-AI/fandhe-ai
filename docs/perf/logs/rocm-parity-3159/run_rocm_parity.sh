#!/usr/bin/env bash
# イシュー #3159: ROCm 実機（MI300X・gfx942）parity の実行スクリプト。手動実行専用（CI からは呼ばない）。
# 規則は同ディレクトリの RULE.txt（事前登録）が正。判定は cargo test の終了コード（run*/exit_code）のみ。
#
# 使い方（ROCm ノード上）:
#   bash run_rocm_parity.sh <tree-abs-path> <out-abs-path>
#   tree: 展開済みリポジトリ（直下に .rev-stamp が必要）／out: 未存在の出力先
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 <tree-abs-path> <out-abs-path>" >&2
  exit 2
fi
TREE=$1
OUT=$2
# 引数は絶対パス + 文字種 allowlist で検査する（A03）。以降の展開は常にクォートする。
for p in "$TREE" "$OUT"; do
  if ! printf '%s' "$p" | grep -Eq '^/[A-Za-z0-9._/-]+$'; then
    echo "絶対パスかつ許可文字のみが必要" >&2
    exit 2
  fi
done
[ -d "$TREE" ] || { echo "tree が存在しない" >&2; exit 2; }
[ ! -e "$OUT" ] || { echo "出力先が既に存在する（上書きしない）" >&2; exit 2; }
[ -f "$TREE/.rev-stamp" ] || { echo ".rev-stamp が無い" >&2; exit 2; }
if ! grep -Eq '^[0-9a-f]{40}$' "$TREE/.rev-stamp"; then
  echo ".rev-stamp が 40 桁の 16 進数でない" >&2
  exit 2
fi
mkdir -p "$OUT/run1" "$OUT/run2"
cp "$TREE/.rev-stamp" "$OUT/.rev-stamp"

# 環境情報は allowlist で抽出する（env・set・rocminfo の全文は貼らない。Uuid・BDFID は取らない）。
{
  echo "rocm_version: $(cat /opt/rocm/.info/version 2>/dev/null || echo 取得不能)"
  echo "hipconfig: $( (hipconfig --version) 2>/dev/null || echo 取得不能)"
  echo "rustc: $( (rustc -V) 2>/dev/null || echo 取得不能)"
  echo "cargo: $( (cargo -V) 2>/dev/null || echo 取得不能)"
  echo "uname_m: $(uname -m)"
  echo "uname_r: $(uname -r)"
  if command -v rocminfo >/dev/null 2>&1; then
    rocminfo 2>/dev/null | grep -E '^[[:space:]]*(Name:[[:space:]]+gfx|Marketing Name:|Wavefront Size:|Compute Unit:|Workgroup Max Size:|Grid Max Size)' || true
  else
    echo "rocminfo: 取得不能"
  fi
} >"$OUT/env_info.raw"

cd "$TREE"
# ビルドを先に終え、計測中にコンパイルを並走させない（RULE.txt build_then_run）。
cargo test -p fandhe-ai-backend-rocm --release --no-run

for r in run1 run2; do
  set +e
  cargo test -p fandhe-ai-backend-rocm --release -- --ignored --nocapture --test-threads=1 \
    >"$OUT/$r/ignored.raw" 2>&1
  echo $? >"$OUT/$r/exit_code"
  set -e
done

# マスクしてから収録し、残っていれば fail-closed で止める（残った場合は raw を残さず削除する）。
HN=$(hostname)
mask() { # mask <raw> <final>
  sed -e "s|$HN|<rocm-node>|g" -e "s|$HOME|<home>|g" -e "s|${USER:-__nouser__}|<user>|g" "$1" >"$2"
  rm -f "$1"
  for s in "$HN" "$HOME"; do
    if [ -n "$s" ] && grep -Fq -- "$s" "$2"; then
      echo "マスク漏れ: $2" >&2
      rm -f "$2"
      exit 1
    fi
  done
}
mask "$OUT/env_info.raw" "$OUT/env_info.txt"
mask "$OUT/run1/ignored.raw" "$OUT/run1/ignored.log"
mask "$OUT/run2/ignored.raw" "$OUT/run2/ignored.log"
echo "done: run1=$(cat "$OUT/run1/exit_code") run2=$(cat "$OUT/run2/exit_code")"
