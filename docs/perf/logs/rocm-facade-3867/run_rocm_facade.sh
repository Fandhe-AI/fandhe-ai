#!/usr/bin/env bash
# イシュー #3867: ROCm を facade 経由（Device::Rocm）で MI300X で測る実行スクリプト。手動実行専用（CI からは呼ばない）。
# 規則は同ディレクトリの RULE.txt（事前登録）が正。#3159 の run_rocm_parity.sh の引数検査・.rev-stamp 検査・マスクを写し、
# 対象を facade の 4 本（1 本ずつ別プロセス・--list の 1 件照合・timeout 1800）に変えた。
#
# 終了コード: 0=4 本とも exit 0／1=どれかが非 0・--list 不一致・マスク漏れ（記録は残る）／2=引数・前提不正。
# 使い方（ROCm ノード上・#3159 の run1・run2 の後）:
#   bash run_rocm_facade.sh <tree-abs-path> <out-abs-path>
#   tree: 展開済みリポジトリ（直下に .rev-stamp が必要）／out: 未存在の出力先
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 <tree-abs-path> <out-abs-path>" >&2
  exit 2
fi
TREE=$1
OUT=$2
# 引数は絶対パス + 文字種 allowlist で検査する（A03）。改行を含む値が通らないよう bash の全体一致で検査する。
for p in "$TREE" "$OUT"; do
  if ! [[ $p =~ ^/[A-Za-z0-9._/-]+$ ]]; then
    echo "絶対パスかつ許可文字のみが必要" >&2
    exit 2
  fi
done
[ -d "$TREE" ] || { echo "tree が存在しない" >&2; exit 2; }
[ ! -e "$OUT" ] || { echo "出力先が既に存在する（上書きしない）" >&2; exit 2; }
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

TESTS=(
  rocm_available_devices_lists_rocm0_and_tape_for_ok
  rocm_facade_matmul_parity_vs_cpu
  rocm_facade_mlp_train_step_parity_vs_cpu
  rocm_tape_for_out_of_range_ordinal_is_err
)
mkdir -p "$OUT"
cp "$TREE/.rev-stamp" "$OUT/.rev-stamp"
{
  echo "rocm_version: $(cat /opt/rocm/.info/version 2>/dev/null || echo 取得不能)"
  echo "rustc: $( (rustc -V) 2>/dev/null || echo 取得不能)"
  echo "cargo: $( (cargo -V) 2>/dev/null || echo 取得不能)"
  echo "uname_m: $(uname -m)"
  echo "script_sha256: $(sha256sum "$0" | cut -d' ' -f1)"
} >"$OUT/env_info.raw"

cd "$TREE"
# ビルドを先に終え、計測中にコンパイルを並走させない（RULE.txt build_then_run 1）。
cargo test -p fandhe-ai --test rocm_facade_real_device --no-run

for t in "${TESTS[@]}"; do
  mkdir -p "$OUT/$t"
  # --list で対象がちょうど 1 件であることを確かめる（RULE.txt build_then_run 2）。不一致は undetermined。
  N=$(cargo test -p fandhe-ai --test rocm_facade_real_device -- --ignored --exact --list "$t" 2>/dev/null | grep -c ': test$' || true)
  echo "$N" >"$OUT/$t/list_count"
  if [ "$N" != 1 ]; then
    echo "list_mismatch" >"$OUT/$t/exit_code"
    continue
  fi
  set +e
  timeout 1800 cargo test -p fandhe-ai --test rocm_facade_real_device -- --ignored --exact --nocapture --test-threads=1 "$t" \
    >"$OUT/$t/test.raw" 2>&1
  echo $? >"$OUT/$t/exit_code"
  set -e
done

# マスクしてから収録し、残っていれば fail-closed で止める（残った場合は該当ファイルを削除する）。
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
RC_ALL=0
for t in "${TESTS[@]}"; do
  [ -f "$OUT/$t/test.raw" ] && mask "$OUT/$t/test.raw" "$OUT/$t/test.log"
  RC=$(cat "$OUT/$t/exit_code")
  echo "done: $t=$RC"
  [ "$RC" = 0 ] || RC_ALL=1
done
exit "$RC_ALL"
