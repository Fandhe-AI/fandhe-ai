#!/usr/bin/env bash
# イシュー #3165: DGX Spark GB10 での Transformer・Functional・可変長系列 7 群（8 テスト）の
# CUDA 実機 parity 実行スクリプト。手動実行専用（CI からは呼ばない）。
# 規則は同ディレクトリの RULE.txt（事前登録）が正。判定は各テストの cargo test 終了コードと出力件数。
# ../metal-facade-parity-3166/run_metal_parity.sh と同型で、Linux／CUDA 向けに直したもの。
#
# マスク規則（PR #3469 の反省で是正）: ホスト名・$HOME・ユーザー名だけを置換する。ユーザー名は前後とも
# 境界（英数字・_・.・- 以外）のときだけ置換し、crate 名・パッケージ名・ディレクトリ名
# （fandhe-ai・fandhe_ai*・target-fandhe-ai・Fandhe-AI 等）は変えない。回収ログは手で編集しない。
#
# 終了コード: 0=全 pass／1=fail またはマスク漏れ（記録は残る。マスク漏れのファイルは削除）／
#             2=引数・前提不正、または undetermined（--list の選択件数が 1 でない・timeout を含む）。
# 使い方（GB10 ノード上。PATH に cargo と nvcc、CARGO_TARGET_DIR=$HOME/work/target-fandhe-ai を指定して呼ぶ）:
#   bash run_cuda_parity.sh <tree-abs-path> <out-abs-path>
#   bash run_cuda_parity.sh --self-test        # マスクの不変条件だけを検査する（どの OS でも可。bash 3.2 対応）
# bash 3.2 で動くよう、連想配列・mapfile・${var,,}・sed -i は使わない。
set -euo pipefail

B='[^A-Za-z0-9_.-]' # ユーザー名の境界に使わない文字集合の否定形（- も . も _ も英数字も境界にしない）

# 正規表現（sed -E／grep -E 共通）のメタ文字と区切り文字 / をエスケープする。
esc_re() { printf '%s' "$1" | sed -e 's/[][\\.^$*+?(){}|/]/\\&/g'; }

# mask_stream <user> <host> <host-short> <home>: 標準入力をマスクして標準出力へ。
# 順序: ホスト名（長・短）→ $HOME → ユーザー名（前後境界付き・隣接取り逃しに備え 2 回）。
mask_stream() {
  local un=$1 hn=$2 hns=$3 home=$4 script=""
  if [ -n "$hn" ]; then script="${script}s/$(esc_re "$hn")/<cuda-node>/g"$'\n'; fi
  if [ -n "$hns" ]; then script="${script}s/$(esc_re "$hns")/<cuda-node>/g"$'\n'; fi
  if [ -n "$home" ] && [ "$home" != "/" ]; then script="${script}s/$(esc_re "$home")/~/g"$'\n'; fi
  local pat="(^|$B)$(esc_re "$un")($B|\$)"
  script="${script}s/${pat}/\\1<user>\\2/g"$'\n'"s/${pat}/\\1<user>\\2/g"
  sed -E -e "$script"
}

# leak_check <file> <user> <host> <host-short> <home>: 残存があれば 1、無ければ 0。
leak_check() {
  local f=$1 un=$2 hn=$3 hns=$4 home=$5 s
  for s in "$hn" "$hns" "$home" "/home/"; do
    if [ -n "$s" ] && [ "$s" != "/" ] && grep -Fq -- "$s" "$f"; then return 1; fi
  done
  if grep -Eq -- "(^|$B)$(esc_re "$un")($B|\$)" "$f"; then return 1; fi
  return 0
}

# 不変条件の検査。入力に与えたユーザー名・ホスト名・HOME で、変えてはいけない行がバイト単位で不変、
# 置換すべき行が期待どおりになり、置換後に残存が無いことを確かめる。失敗なら 1。
# 本実行の冒頭では実際の値、--self-test では架空の値（fandhe／node-x）で呼ぶ。
mask_selfcheck() { # mask_selfcheck <user> <host> <host-short> <home>
  local un=$1 hn=$2 hns=$3 home=$4 d rc=0
  d=$(mktemp -d)
  cat >"$d/keep.in" <<'KEEP'
   Compiling fandhe-ai v0.11.0 (~/work/rust-ai-library-run-3165/crates/facade)
   Compiling fandhe-ai-backend-cuda v0.11.0
Running unittests src/lib.rs (target/debug/deps/fandhe_ai-0123456789abcdef)
fandhe_ai_backend_cuda::device
~/work/target-fandhe-ai/debug/deps/fandhe_ai_backend_cuda-abc
repository: Fandhe-AI/fandhe-ai
KEEP
  : >"$d/chg.in"
  : >"$d/chg.want"
  # 各行: 入力 -> 期待。実際の値で組み立てる。
  printf '%s/x\n' "$home" >>"$d/chg.in"
  printf '~/x\n' >>"$d/chg.want"
  printf 'owner=%s\n' "$un" >>"$d/chg.in"
  printf 'owner=<user>\n' >>"$d/chg.want"
  printf '%s@%s\n' "$un" "$hn" >>"$d/chg.in"
  printf '<user>@<cuda-node>\n' >>"$d/chg.want"
  printf '%s\n' "$hn" >>"$d/chg.in"
  printf '<cuda-node>\n' >>"$d/chg.want"
  printf '%s %s\n' "$un" "$un" >>"$d/chg.in"
  printf '<user> <user>\n' >>"$d/chg.want"
  mask_stream "$un" "$hn" "$hns" "$home" <"$d/keep.in" >"$d/keep.out"
  mask_stream "$un" "$hn" "$hns" "$home" <"$d/chg.in" >"$d/chg.out"
  if ! cmp -s "$d/keep.in" "$d/keep.out"; then
    echo "selfcheck: 変えてはいけない行が変化した" >&2
    diff "$d/keep.in" "$d/keep.out" >&2 || true
    rc=1
  fi
  if ! cmp -s "$d/chg.want" "$d/chg.out"; then
    echo "selfcheck: 置換結果が期待と異なる" >&2
    diff "$d/chg.want" "$d/chg.out" >&2 || true
    rc=1
  fi
  if ! leak_check "$d/chg.out" "$un" "$hn" "$hns" "$home"; then
    echo "selfcheck: 置換後に残存がある" >&2
    rc=1
  fi
  rm -rf "$d"
  return "$rc"
}

if [ "${1:-}" = "--self-test" ]; then
  if [ "$#" -ne 1 ]; then
    echo "usage: $0 --self-test" >&2
    exit 2
  fi
  if mask_selfcheck fandhe node-x node-x /home/fandhe; then
    echo "self-test: pass"
    exit 0
  fi
  echo "self-test: FAIL" >&2
  exit 2
fi

if [ "$#" -ne 2 ]; then
  echo "usage: $0 <tree-abs-path> <out-abs-path> | --self-test" >&2
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
if [ "$(uname -s)" != "Linux" ]; then
  echo "Linux（GB10 ノード）以外では実行しない" >&2
  exit 2
fi
[ -d "$TREE" ] || { echo "tree が存在しない" >&2; exit 2; }
[ ! -e "$OUT" ] || { echo "出力先が既に存在する（上書きしない）" >&2; exit 2; }
[ -f "$TREE/.rev-stamp" ] || { echo ".rev-stamp が無い" >&2; exit 2; }

# .rev-stamp は「40 桁の小文字 16 進数 + 任意の末尾改行 1 つ」だけであること。
# 末尾に番兵 x を付けて $(...) の改行除去を避け、CR・空白・余計な行を全て拒否する。
export LC_ALL=C
STAMP_SIZE=$(wc -c <"$TREE/.rev-stamp" | tr -d ' ')
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

HN=$(hostname)
HNS=$(hostname -s 2>/dev/null || true)
UN=${USER:-$(id -un)}
# 実際の値でもマスクの不変条件を確かめる（ユーザー名やホスト名が crate 名と衝突する場合もここで止まる）。
if ! mask_selfcheck "$UN" "$HN" "$HNS" "$HOME"; then
  echo "実値でのマスク自己検査に失敗" >&2
  exit 2
fi

sha256() { # sha256 <file> -> 64 桁
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi
}
SELF=${BASH_SOURCE[0]}
SCRIPT_SHA=$(sha256 "$SELF")

# マスクしてから収録し、残っていれば fail-closed で止める（残ったファイルは raw ごと削除する）。
# 異常終了（set -e・exit 2 を含む）でも未処理の .raw が出力先に残らないよう、EXIT trap でも mask_all を呼ぶ。
LEAK=0
mask() { # mask <raw> <final>
  mask_stream "$UN" "$HN" "$HNS" "$HOME" <"$1" >"$2"
  rm -f "$1"
  if ! leak_check "$2" "$UN" "$HN" "$HNS" "$HOME"; then
    echo "マスク漏れ: $2" >&2
    rm -f "$2"
    LEAK=1
  fi
}
mask_all() {
  for raw in "$OUT"/*.raw "$OUT"/logs/*/*.raw; do
    [ -e "$raw" ] || continue
    # env_info.raw は .txt、それ以外は .log として出力する。
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
# compute-apps はプロセス名を書き出さず、件数と固定名 <resident-process> だけにする。
env_info() { # env_info <label>
  echo "== $1 =="
  echo "date_utc: $(date -u +%Y-%m-%dT%H:%MZ)"
  echo "hostname: masked"
  echo "gpu: $( (nvidia-smi --query-gpu=name,driver_version,utilization.gpu --format=csv,noheader) 2>/dev/null || echo 取得不能)"
  echo "compute_apps: $( (nvidia-smi --query-compute-apps=pid --format=csv,noheader | { grep -c . || true; }) 2>/dev/null) x <resident-process>"
  echo "nvcc: $( (nvcc --version | tail -n 1) 2>/dev/null || echo 取得不能)"
  echo "rustc: $( (rustc -V) 2>/dev/null || echo 取得不能)"
  echo "cargo: $( (cargo -V) 2>/dev/null || echo 取得不能)"
  echo "arch: $(uname -m)"
  echo "load: $( (uptime | sed -e 's/.*load averages*: *//') 2>/dev/null || echo 取得不能)"
  echo "rev_stamp: $STAMP_BODY"
  echo "script_sha256: $SCRIPT_SHA"
}
env_info before >"$OUT/env_info.raw"

# 他の cargo／rustc が走っていれば並走させない（RULE.txt）。
if pgrep -x cargo >/dev/null 2>&1 || pgrep -x rustc >/dev/null 2>&1; then
  echo "他の cargo／rustc プロセスが動作中のため停止" >&2
  exit 2
fi
echo "concurrent_cargo_rustc: none" >>"$OUT/env_info.raw"

cd "$TREE"
# ビルドを先に終え、計測中にコンパイルを並走させない（判定外）。失敗したら set -e で止まる。
{
  cargo test -p fandhe-ai --lib --no-run
  cargo test -p fandhe-ai --test compat_sequential_layers_backend_parity --no-run
  cargo test -p fandhe-ai --test merge_ops_backend_parity --no-run
  cargo test -p fandhe-ai --test packed_sequence_backend_parity --no-run
} >"$OUT/build.raw" 2>&1

# 行: 群|ターゲット引数|テストのフルパス（関数名がログのファイル名になる）。
CASES='mha-config-sequential-2530|--test compat_sequential_layers_backend_parity|cuda_multihead_attention_config_matches_cpu
transformer-decoder-sequential-2532|--test compat_sequential_layers_backend_parity|cuda_transformer_decoder_layer_matches_cpu
transformer-sequential-2533|--test compat_sequential_layers_backend_parity|cuda_transformer_matches_cpu
functional-graph-2665|--lib|compat::functional::tests::cuda_functional_graph_matches_cpu_reference
functional-fit-2667|--lib|compat::functional::fit_parity_tests::cuda_parameter_gradients_match_cpu_reference
merge-ops-2666|--test merge_ops_backend_parity|cuda_merge_ops_match_cpu_reference
merge-ops-2666|--lib|compat::functional::merge_tests::cuda_merge_graph_matches_cpu_reference
packed-sequence-2647|--test packed_sequence_backend_parity|cuda_packed_sequence_matches_cpu_reference'

printf 'group\ttarget\ttest\texit_code\trunning\tresult\tverdict\n' >"$OUT/summary.tsv"
ANY_FAIL=0
ANY_UNDET=0
while IFS='|' read -r GROUP TARGET TEST; do
  FN=${TEST##*::}
  mkdir -p "$OUT/logs/$GROUP"
  # 選択の検証: --list で ": test" 終わりの行がちょうど 1 行であること（0 または 2 以上は実行しない）。
  # TARGET は上の固定文字列のみ（ユーザー入力ではない）なので単語分割で展開する。
  # shellcheck disable=SC2086
  N=$(cargo test -p fandhe-ai $TARGET -- --ignored --exact --list "$TEST" 2>/dev/null | grep -c ': test$' || true)
  if [ "$N" -ne 1 ]; then
    echo "--list の選択件数が 1 でない: $TEST ($N)" >&2
    printf '%s\t%s\t%s\t-\t-\t-\tundetermined\n' "$GROUP" "$TARGET" "$TEST" >>"$OUT/summary.tsv"
    ANY_UNDET=1
    continue
  fi
  set +e
  # shellcheck disable=SC2086
  timeout 1800 cargo test -p fandhe-ai $TARGET -- --ignored --exact --nocapture --test-threads=1 "$TEST" \
    >"$OUT/logs/$GROUP/$FN.raw" 2>&1
  RC=$?
  set -e
  echo "$RC" >"$OUT/logs/$GROUP/$FN.exit_code"
  RUNNING=$(grep -c '^running 1 test$' "$OUT/logs/$GROUP/$FN.raw" || true)
  RESULT=$(grep -E '^test result: ' "$OUT/logs/$GROUP/$FN.raw" | head -n 1 || true)
  if [ "$RC" -eq 124 ]; then
    V=undetermined
    ANY_UNDET=1
  elif [ "$RC" -eq 0 ] && [ "$RUNNING" -eq 1 ] && [[ $RESULT == "test result: ok. 1 passed; 0 failed"* ]]; then
    V=pass
  else
    V=fail
    ANY_FAIL=1
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$GROUP" "$TARGET" "$TEST" "$RC" "$RUNNING" "${RESULT:-none}" "$V" >>"$OUT/summary.tsv"
done <<EOF
$CASES
EOF

env_info after >>"$OUT/env_info.raw"

mask_all

# summary.tsv はヘッダ + 8 行が揃っていること（欠損は例外扱い）。
ROWS=$(grep -c . "$OUT/summary.tsv" || true)
if [ "$ROWS" -ne 9 ]; then
  echo "summary.tsv が 9 行（ヘッダ + 8）でない: $ROWS" >&2
  exit 2
fi

# 提出物の完全性: 全成果物の SHA-256（回収後に sha256sum -c で無改変を確かめる）。
(
  cd "$OUT"
  find . -type f ! -name SHA256SUMS | sed -e 's|^\./||' | sort | while IFS= read -r f; do
    printf '%s  %s\n' "$(sha256 "$f")" "$f"
  done >SHA256SUMS
)

echo "done: any_fail=$ANY_FAIL any_undetermined=$ANY_UNDET leak=$LEAK"
if [ "$ANY_UNDET" -ne 0 ]; then
  exit 2
fi
if [ "$LEAK" -ne 0 ] || [ "$ANY_FAIL" -ne 0 ]; then
  exit 1
fi
