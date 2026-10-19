#!/usr/bin/env bash
# イシュー #3164（2 回目）: Mac mini（Apple M4）での compat::Sequential 層 9 群（Metal 10 テスト）の実機 parity 再実測スクリプト。
# 手動実行専用（CI からは呼ばない）。規則は同ディレクトリの RULE.txt（事前登録）が正。
# ../metal-max-reduce-3475/run_metal_parity.sh の骨格（マスク・EXIT trap・pgrep 検査・1 行 .rev-stamp）を流用し、
# 対象テストは fandhe-ai の compat_sequential_*_backend_parity 9 ターゲットの 10 本に変えている（全件 1 回・判定テスト）。
# macOS 標準 bash 3.2 で動くよう、連想配列・mapfile・${var,,}・sed -i は使わない。
#
# 終了コード: 0=全 pass／1=fail またはマスク漏れ（記録は残る。マスク漏れのファイルは削除）／
#             2=引数・前提不正、または undetermined（--list の選択件数が 1 でない・テスト名残存検査の失敗を含む）。
# 使い方（Mac mini 上。zsh -lc で CARGO_TARGET_DIR=$HOME/work/target-fandhe-ai を指定して呼ぶ）:
#   bash run_metal_parity.sh <tree-abs-path> <out-abs-path>
#   tree: 展開済みリポジトリ（直下に 1 行形式の .rev-stamp が必要）／out: 未存在の出力先
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 <tree-abs-path> <out-abs-path>" >&2
  exit 2
fi
TREE=$1
OUT=$2
# 引数は絶対パス + 文字種 allowlist で検査する（A03）。改行を含む値は =~ の全体一致で拒否する。
for p in "$TREE" "$OUT"; do
  if ! [[ $p =~ ^/[A-Za-z0-9._/-]+$ ]]; then
    echo "絶対パスかつ許可文字のみが必要" >&2
    exit 2
  fi
done
if [ "$(uname -s)" != "Darwin" ]; then
  echo "Darwin（Mac mini）以外では実行しない" >&2
  exit 2
fi
[ -d "$TREE" ] || { echo "tree が存在しない" >&2; exit 2; }
[ ! -e "$OUT" ] || { echo "出力先が既に存在する（上書きしない）" >&2; exit 2; }
[ -f "$TREE/.rev-stamp" ] || { echo ".rev-stamp が無い" >&2; exit 2; }

# .rev-stamp は厳密に 1 行: 40 桁小文字 16 進数 + 改行 1 個のみ。CR・空白・余計な行は拒否する。
# dirty 検査は送信前にローカルで行い env_info.txt の説明に記録する（スタンプには含めない）。
export LC_ALL=C
STAMP=$(
  cat "$TREE/.rev-stamp"
  printf x
)
STAMP=${STAMP%x}
STAMP_SHA=${STAMP%$'\n'}
if ! [[ $STAMP_SHA =~ ^[0-9a-f]{40}$ ]] || [ "$STAMP" != "$STAMP_SHA"$'\n' ]; then
  echo ".rev-stamp が 1 行形式（40 桁小文字 16 進数のみ）でない" >&2
  exit 2
fi

# マスクしてから収録し、残っていれば fail-closed で止める（残ったファイルは raw ごと削除する）。
# 異常終了（set -e・exit 2 を含む）でも未処理の .raw が出力先に残らないよう、EXIT trap でも mask_all を呼ぶ。
HN=$(hostname)
UN=${USER:-__nouser__}
# ユーザー名を正規表現（sed -E／grep -E 共通）で安全に使うため、メタ文字をエスケープしておく。
UN_RE=$(printf '%s' "$UN" | sed -e 's#[][\\.^$*+?(){}|/]#\\&#g')
LEAK=0
mask() { # mask <raw> <final>
  # ユーザー名の直後は「区切り文字 または 行末」を境界とする（行末で終わる owner=<user> も置換する）。
  sed -E -e "s|$HN|<metal-node>|g" -e "s|$HOME|~|g" -e "s#${UN_RE}([^-A-Za-z0-9_]|\$)#<user>\1#g" "$1" >"$2"
  rm -f "$1"
  for s in "$HN" "$HOME" "/Users/"; do
    if [ -n "$s" ] && grep -Fq -- "$s" "$2"; then
      echo "マスク漏れ: $2" >&2
      rm -f "$2"
      LEAK=1
      break
    fi
  done
  # ユーザー名も同じ境界（行末含む）で残存検査する。部分文字列の誤検知は避ける。
  if [ -e "$2" ] && grep -Eq -- "(^|[^-A-Za-z0-9_])${UN_RE}([^-A-Za-z0-9_]|\$)" "$2"; then
    echo "マスク漏れ(user): $2" >&2
    rm -f "$2"
    LEAK=1
  fi
}
mask_all() {
  for raw in "$OUT"/*.raw "$OUT"/logs/*/*.raw; do
    [ -e "$raw" ] || continue
    # env_info.raw のみ収集手順・コミット済み成果物に合わせて .txt で出力する。
    case "$raw" in
      "$OUT/env_info.raw") mask "$raw" "$OUT/env_info.txt" ;;
      *) mask "$raw" "${raw%.raw}.log" ;;
    esac
  done
}
trap mask_all EXIT

mkdir -p "$OUT/logs"
cp "$TREE/.rev-stamp" "$OUT/rev-stamp.txt"

# 環境情報は allowlist で取得する。hostname は取らず masked と固定で書く。
env_info() { # env_info <label>
  echo "== $1 =="
  echo "date_utc: $(date -u +%Y-%m-%dT%H:%MZ)"
  echo "hostname: masked"
  echo "sw_vers: $( (sw_vers | tr '\n' ' ') 2>/dev/null || echo 取得不能)"
  echo "cpu: $( (sysctl -n machdep.cpu.brand_string) 2>/dev/null || echo 取得不能)"
  echo "rustc: $( (rustc -V) 2>/dev/null || echo 取得不能)"
  echo "cargo: $( (cargo -V) 2>/dev/null || echo 取得不能)"
  echo "arch: $(uname -m)"
  echo "load: $( (uptime | sed -e 's/.*load averages*: *//') 2>/dev/null || echo 取得不能)"
  echo "rev_stamp: $STAMP_SHA"
}
env_info before >"$OUT/env_info.raw"

# 他の cargo／rustc が走っていれば並走させない（RULE.txt）。30 秒間隔で最大 60 分待ち、消えなければ undetermined。
WAITED=0
while pgrep -x cargo >/dev/null 2>&1 || pgrep -x rustc >/dev/null 2>&1; do
  if [ "$WAITED" -ge 120 ]; then
    echo "他の cargo／rustc プロセスが 60 分待っても消えないため停止（undetermined）" >&2
    echo "concurrent_cargo_rustc: still-running-after-60min" >>"$OUT/env_info.raw"
    exit 2
  fi
  sleep 30
  WAITED=$((WAITED + 1))
done
echo "concurrent_cargo_rustc: none (waited_x30s=$WAITED)" >>"$OUT/env_info.raw"

cd "$TREE"
# 各コマンドは alarm 1800 秒で打ち切る（macOS に timeout はない。時間超過は rc 142）。
TMO() { perl -e 'alarm shift; exec @ARGV' 1800 "$@"; }
# ビルドを先に終え、計測中にコンパイルを並走させない（判定外）。失敗したら set -e で止まる。
{
  TMO cargo test -p fandhe-ai --test compat_sequential_activation_layers_backend_parity --no-run
  TMO cargo test -p fandhe-ai --test compat_sequential_activation_scalar_layers_backend_parity --no-run
  TMO cargo test -p fandhe-ai --test compat_sequential_adaptive_max_global_pool_backend_parity --no-run
  TMO cargo test -p fandhe-ai --test compat_sequential_conv_transpose2d_backend_parity --no-run
  TMO cargo test -p fandhe-ai --test compat_sequential_conv3d_backend_parity --no-run
  TMO cargo test -p fandhe-ai --test compat_sequential_dropout_embedding_bag_backend_parity --no-run
  TMO cargo test -p fandhe-ai --test compat_sequential_group_instance_norm_backend_parity --no-run
  TMO cargo test -p fandhe-ai --test compat_sequential_pixel_shuffle_backend_parity --no-run
  TMO cargo test -p fandhe-ai --test compat_sequential_spatial_layers_backend_parity --no-run
} >"$OUT/build.raw" 2>&1

# 行: 群|パッケージ|ターゲット引数|テスト関数名（関数名がログのファイル名になる）。
CASES='activation-layers-2529|fandhe-ai|--test compat_sequential_activation_layers_backend_parity|metal_activation_layers_parity
activation-scalar-layers-2679|fandhe-ai|--test compat_sequential_activation_scalar_layers_backend_parity|metal_activation_scalar_layers_parity
adaptive-max-global-pool-2527|fandhe-ai|--test compat_sequential_adaptive_max_global_pool_backend_parity|metal_adaptive_max_global_pool_bit_exact
conv-transpose2d-2523|fandhe-ai|--test compat_sequential_conv_transpose2d_backend_parity|metal_conv_transpose2d_forward_backward_matches_cpu
conv3d-2524|fandhe-ai|--test compat_sequential_conv3d_backend_parity|metal_conv3d_forward_backward_matches_cpu
dropout-embedding-bag-2528|fandhe-ai|--test compat_sequential_dropout_embedding_bag_backend_parity|metal_dropout_embedding_bag_bit_exact
group-instance-norm-2525|fandhe-ai|--test compat_sequential_group_instance_norm_backend_parity|metal_group_instance_norm_matches_cpu
pixel-shuffle-2526|fandhe-ai|--test compat_sequential_pixel_shuffle_backend_parity|metal_pixel_shuffle_layers_bit_exact
spatial-2522|fandhe-ai|--test compat_sequential_spatial_layers_backend_parity|metal_spatial_layers_nearest_bit_exact
spatial-2522|fandhe-ai|--test compat_sequential_spatial_layers_backend_parity|metal_spatial_layers_bilinear_matches_cpu'

printf 'group\ttarget\ttest\texit_code\trunning\tresult\tverdict\n' >"$OUT/summary.tsv"
ANY_FAIL=0
ANY_UNDET=0
while IFS='|' read -r GROUP PKG TARGET TEST; do
  FN=${TEST##*::}
  mkdir -p "$OUT/logs/$GROUP"
  # 選択の検証: --list で ": test" 終わりの行がちょうど 1 行であること（0 または 2 以上は実行しない）。
  # TARGET は上の固定文字列のみ（ユーザー入力ではない）なので単語分割で展開する。
  # shellcheck disable=SC2086
  N=$(TMO cargo test -p "$PKG" $TARGET -- --ignored --exact --list "$TEST" 2>/dev/null | grep -c ': test$' || true)
  if [ "$N" -ne 1 ]; then
    echo "--list の選択件数が 1 でない: $TEST ($N)" >&2
    printf '%s\t%s\t%s\t-\t-\t-\tundetermined\n' "$GROUP" "$PKG $TARGET" "$TEST" >>"$OUT/summary.tsv"
    ANY_UNDET=1
    continue
  fi
  set +e
  # shellcheck disable=SC2086
  TMO cargo test -p "$PKG" $TARGET -- --ignored --exact --nocapture --test-threads=1 "$TEST" \
    >"$OUT/logs/$GROUP/$FN.raw" 2>&1
  RC=$?
  set -e
  echo "rc=$RC" >>"$OUT/logs/$GROUP/$FN.raw"
  echo "$RC" >"$OUT/logs/$GROUP/$FN.exit_code"
  RUNNING=$(grep -c '^running 1 test$' "$OUT/logs/$GROUP/$FN.raw" || true)
  RESULT=$(grep -E '^test result: ' "$OUT/logs/$GROUP/$FN.raw" | head -n 1 || true)
  if [ "$RC" -eq 0 ] && [ "$RUNNING" -eq 1 ] && [[ $RESULT == "test result: ok. 1 passed; 0 failed"* ]]; then
    V=pass
  elif [ "$RC" -eq 142 ]; then
    V=undetermined
    ANY_UNDET=1
  else
    V=fail
    ANY_FAIL=1
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$GROUP" "$PKG $TARGET" "$TEST" "$RC" "$RUNNING" "${RESULT:-none}" "$V" >>"$OUT/summary.tsv"
done <<EOF
$CASES
EOF

env_info after >>"$OUT/env_info.raw"

mask_all

# summary.tsv はヘッダ + 10 行が揃っていること（欠損は例外扱い）。
ROWS=$(grep -c . "$OUT/summary.tsv" || true)
if [ "$ROWS" -ne 11 ]; then
  echo "summary.tsv が 11 行（ヘッダ + 10）でない: $ROWS" >&2
  exit 2
fi

# マスク後のテスト名残存検査: 実行した各ログに "test <フルパス> ... " がちょうど 1 行残ること。
# マスクでテスト名・crate 名が壊れていれば（またはログがマスク漏れで削除されていれば）記録を無効にする。
NAME_BAD=0
while IFS='|' read -r GROUP PKG TARGET TEST; do
  FN=${TEST##*::}
  if [ -e "$OUT/logs/$GROUP/$FN.exit_code" ]; then
    C=0
    if [ -e "$OUT/logs/$GROUP/$FN.log" ]; then
      C=$(grep -cF -- "test $TEST ... " "$OUT/logs/$GROUP/$FN.log" || true)
    fi
    if [ "$C" -ne 1 ]; then
      echo "テスト名残存検査に失敗: $GROUP/$FN ($C)" >&2
      NAME_BAD=1
    fi
  fi
done <<EOF
$CASES
EOF

echo "done: any_fail=$ANY_FAIL any_undetermined=$ANY_UNDET leak=$LEAK name_bad=$NAME_BAD"
if [ "$ANY_UNDET" -ne 0 ] || [ "$NAME_BAD" -ne 0 ]; then
  exit 2
fi
if [ "$LEAK" -ne 0 ] || [ "$ANY_FAIL" -ne 0 ]; then
  exit 1
fi
