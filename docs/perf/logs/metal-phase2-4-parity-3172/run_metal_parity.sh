#!/usr/bin/env bash
# イシュー #3172: Mac mini（Apple M4）での Phase 2〜4 追加 8 機能（12 テスト）の Metal 実機 parity 実行スクリプト。
# 手動実行専用（CI からは呼ばない）。規則は同ディレクトリの RULE.txt（事前登録）が正。
# 判定は各テストの cargo test 終了コードと出力件数。paged attention（9〜12）は
# ../paged-attention-parity-3113/RULE.txt に従い独立プロセスで 5 回実行し PAGED_PARITY 行の中央値を集計する。
# 前例 ../metal-facade-parity-3166/run_metal_parity.sh（Darwin 限定・2 行形式 .rev-stamp・pgrep による並走検出）を踏襲する。
#
# 前回測定（PR #3474。所有者判断で不採用）の是正点: ホスト名の置換が境界なしの部分文字列一致で、
# テスト名の一部（local_response_norm の local）を壊していた。本スクリプトはホスト名にもユーザー名と同じ
# 前後境界を課し、(1) 本実行冒頭の実値での自己検査 (2) 事後の識別子保存検査（期待するテスト名が
# 逐語で残っていること）を必須にする。回収ログは手で編集しない。漏れ・化けはスクリプトを直し、
# 新コミットで再登録して測り直す。
#
# マスク規則: ホスト名（長・短）・$HOME・ユーザー名だけを置換する。ホスト名とユーザー名は前後とも
# 境界（英数字・_・.・- 以外）のときだけ置換し、crate 名・テスト名・ディレクトリ名
# （fandhe-ai・fandhe_ai*・target-fandhe-ai・Fandhe-AI・metal_*_matches_cpu 等）は変えない。
#
# 終了コード: 0=全 pass／1=fail またはマスク漏れ（記録は残る。マスク漏れのファイルは削除）／
#             2=引数・前提不正、undetermined（--list の選択が期待と異なる）、自己検査失敗、識別子保存検査失敗。
# 使い方（Mac mini 上。zsh -lc で CARGO_TARGET_DIR=$HOME/work/target-fandhe-ai を指定して呼ぶ）:
#   bash run_metal_parity.sh <tree-abs-path> <out-abs-path>
#   bash run_metal_parity.sh --self-test        # マスクの不変条件だけを検査する（どの OS でも可）
# bash 3.2 で動くよう、連想配列・mapfile・${var,,}・sed -i は使わない。
set -euo pipefail

B='[^A-Za-z0-9_.-]' # 識別子の一部と見なす文字（英数字・_・.・-）の否定形。これを境界とする。

# 正規表現（sed -E／grep -E 共通）のメタ文字と区切り文字 / をエスケープする。
esc_re() { printf '%s' "$1" | sed -e 's/[][\\.^$*+?(){}|/]/\\&/g'; }

# mask_stream <user> <host> <host-short> <home>: 標準入力をマスクして標準出力へ。
# 順序: ホスト名（長・短。境界付き）→ $HOME → ユーザー名（境界付き）。隣接取り逃しに備え各 2 回。
mask_stream() {
  local un=$1 hn=$2 hns=$3 home=$4 script="" pat
  if [ -n "$hn" ]; then
    pat="(^|$B)$(esc_re "$hn")($B|\$)"
    script="${script}s/${pat}/\\1<metal-node>\\2/g"$'\n'"s/${pat}/\\1<metal-node>\\2/g"$'\n'
  fi
  if [ -n "$hns" ] && [ "$hns" != "$hn" ]; then
    pat="(^|$B)$(esc_re "$hns")($B|\$)"
    script="${script}s/${pat}/\\1<metal-node>\\2/g"$'\n'"s/${pat}/\\1<metal-node>\\2/g"$'\n'
  fi
  if [ -n "$home" ] && [ "$home" != "/" ]; then script="${script}s/$(esc_re "$home")/~/g"$'\n'; fi
  pat="(^|$B)$(esc_re "$un")($B|\$)"
  script="${script}s/${pat}/\\1<user>\\2/g"$'\n'"s/${pat}/\\1<user>\\2/g"
  sed -E -e "$script"
}

# leak_check <file> <user> <host> <host-short> <home>: 残存があれば 1、無ければ 0。
# $HOME・/Users/ は境界なしの固定文字列で、ホスト名・ユーザー名はマスクと同じ境界付きで検査する
# （境界なしだと host=local が metal_local_response_norm に誤一致するため。PR #3482 指摘）。
leak_check() {
  local f=$1 un=$2 hn=$3 hns=$4 home=$5 s
  for s in "$home" "/Users/"; do
    if [ -n "$s" ] && [ "$s" != "/" ] && grep -Fq -- "$s" "$f"; then return 1; fi
  done
  for s in "$hn" "$hns" "$un"; do
    if [ -n "$s" ] && grep -Eq -- "(^|$B)$(esc_re "$s")($B|\$)" "$f"; then return 1; fi
  done
  return 0
}

# 12 テストの識別子（自己検査で共用）。
TEST_NAMES='metal_pool3d_forward_backward_matches_cpu
metal_conv_transpose3d_forward_backward_matches_cpu
metal_functional_max_unpool_2d_forward_backward_matches_cpu
metal_functional_max_unpool_3d_forward_backward_matches_cpu
metal_fold_unfold_forward_backward_matches_cpu
metal_local_response_norm_forward_backward_matches_cpu
metal_jvp_multi_matches_cpu_reference
metal_vmap_grad_matches_cpu_reference
metal_paged_attention_matrix_matches_cpu_reference
metal_paged_attention_tracked_matrix_matches_cpu_reference
metal_paged_attention_tolerance_constants_pinned
metal_paged_attention_forward_matches_cpu_tape'

# 不変条件の検査。与えたユーザー名・ホスト名・HOME で、変えてはいけない行がバイト単位で不変、
# 置換すべき行が期待どおりになり、置換後に残存が無いことを確かめる。失敗なら 1。
# 本実行の冒頭では実際の値、--self-test では架空の値で呼ぶ。
mask_selfcheck() { # mask_selfcheck <user> <host> <host-short> <home>
  local un=$1 hn=$2 hns=$3 home=$4 d rc=0 t
  d=$(mktemp -d)
  {
    cat <<'KEEP'
   Compiling fandhe-ai v0.11.0 (~/work/rust-ai-library-run-3172/crates/facade)
   Compiling fandhe-ai-backend-metal v0.11.0
     Running tests/compat_sequential_lrn_backend_parity.rs (~/work/target-fandhe-ai/debug/deps/compat_sequential_lrn_backend_parity-0123456789abcdef)
Running unittests src/lib.rs (target/debug/deps/fandhe_ai-0123456789abcdef)
fandhe_ai_backend_metal::device
~/work/target-fandhe-ai/release/deps/fandhe_ai_backend_metal-abc
repository: Fandhe-AI/fandhe-ai
PAGED_PARITY backend=metal-tracked case=legacy_bs1_b1 step=0 total=24 fail=0 max_abs=0e0 max_rel=0e0 mean_abs=0e0 judgement=strict
KEEP
    printf '%s\n' "$TEST_NAMES" | while IFS= read -r t; do
      printf 'test %s ... ok\n' "$t"
      printf 'test %s ... PAGED_PARITY backend=metal case=x step=0 total=1 fail=0 max_abs=0e0 max_rel=0e0 mean_abs=0e0 judgement=strict\n' "$t"
    done
  } >"$d/keep.in"
  : >"$d/chg.in"
  : >"$d/chg.want"
  # 各行: 入力 -> 期待。実際の値で組み立てる。
  printf '%s/x\n' "$home" >>"$d/chg.in"
  printf '~/x\n' >>"$d/chg.want"
  printf 'owner=%s\n' "$un" >>"$d/chg.in"
  printf 'owner=<user>\n' >>"$d/chg.want"
  printf '%s@%s\n' "$un" "$hn" >>"$d/chg.in"
  printf '<user>@<metal-node>\n' >>"$d/chg.want"
  printf '%s\n' "$hn" >>"$d/chg.in"
  printf '<metal-node>\n' >>"$d/chg.want"
  printf 'host=%s (short)\n' "$hns" >>"$d/chg.in"
  printf 'host=<metal-node> (short)\n' >>"$d/chg.want"
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
  # 識別子行（metal_local_response_norm 等）を leak_check が誤検出しない回帰ケース。
  if ! leak_check "$d/keep.out" "$un" "$hn" "$hns" "$home"; then
    echo "selfcheck: 識別子行を残存と誤検出した" >&2
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
  # 短縮ホスト名がテスト識別子の部分文字列になりうる一般語（local・response・norm）のケースを含める
  # （前回 PR #3474 の破損の再現防止）。
  ST=0
  mask_selfcheck fandhe node-x node-x /Users/fandhe || ST=1
  mask_selfcheck alice local local.lan /Users/alice || ST=1
  mask_selfcheck bob response response /Users/bob || ST=1
  mask_selfcheck carol norm norm.example /Users/carol || ST=1
  if [ "$ST" -eq 0 ]; then
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
if [ "$(uname -s)" != "Darwin" ]; then
  echo "Darwin（Mac mini）以外では実行しない" >&2
  exit 2
fi
[ -d "$TREE" ] || { echo "tree が存在しない" >&2; exit 2; }
[ ! -e "$OUT" ] || { echo "出力先が既に存在する（上書きしない）" >&2; exit 2; }
[ -f "$TREE/.rev-stamp" ] || { echo ".rev-stamp が無い" >&2; exit 2; }

# .rev-stamp は厳密に 2 行: 1 行目が 40 桁小文字 16 進数、2 行目が dirty=<数字>。CR・空白・余計な行は拒否する。
export LC_ALL=C
STAMP=$(
  cat "$TREE/.rev-stamp"
  printf x
)
STAMP=${STAMP%x}
STAMP_SHA=${STAMP%%$'\n'*}
STAMP_REST=${STAMP#*$'\n'}
STAMP_DIRTY=${STAMP_REST%$'\n'}
if ! [[ $STAMP_SHA =~ ^[0-9a-f]{40}$ ]] || [ "$STAMP_REST" = "$STAMP" ] \
  || ! [[ $STAMP_DIRTY =~ ^dirty=[0-9]+$ ]] || [ "$STAMP_REST" != "$STAMP_DIRTY"$'\n' ]; then
  echo ".rev-stamp が 2 行形式（40 桁 SHA + dirty=N）でない" >&2
  exit 2
fi
if [ "$STAMP_DIRTY" != "dirty=0" ]; then
  echo ".rev-stamp が dirty=0 でない（実測は無効）" >&2
  exit 2
fi

HN=$(hostname)
HNS=$(hostname -s 2>/dev/null || true)
UN=${USER:-$(id -un)}
# 実際の値でもマスクの不変条件を確かめる（識別子と衝突する値はここで止まり、cargo は一切起動しない）。
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
  local raw
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
  echo "rev_stamp: $STAMP_SHA $STAMP_DIRTY"
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
  cargo test -p fandhe-ai --test compat_sequential_pool3d_backend_parity --no-run
  cargo test -p fandhe-ai --test compat_sequential_conv_transpose3d_backend_parity --no-run
  cargo test -p fandhe-ai --test compat_functional_max_unpool_backend_parity --no-run
  cargo test -p fandhe-ai --test compat_sequential_fold_unfold_backend_parity --no-run
  cargo test -p fandhe-ai --test compat_sequential_lrn_backend_parity --no-run
  cargo test -p fandhe-ai --test functional_transforms_jvp_multi_backend_parity --no-run
  cargo test -p fandhe-ai --test functional_transforms_facade --no-run
  cargo test --release -p fandhe-ai-backend-metal --test paged_kv_real_device --no-run
  cargo test --release -p fandhe-ai --test paged_attention_facade --no-run
} >"$OUT/build.raw" 2>&1

printf 'group\tkind\trun\ttarget\ttest\texit_code\trunning\tresult\tverdict\n' >"$OUT/summary.tsv"
ANY_FAIL=0
ANY_UNDET=0
ROWS_EXPECTED=19 # ヘッダ + 単発 8 + paged 10 run

# record <group> <kind> <run> <target> <test> <rc> <running> <result> <verdict>
record() {
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$@" >>"$OUT/summary.tsv"
}

# run_single <group> <target-args> <test>: 1〜8（debug・1 テスト 1 プロセス・1 回）。
# target-args は本スクリプト内の固定文字列のみ（ユーザー入力ではない）なので単語分割で展開する。
run_single() {
  local group=$1 targs=$2 test=$3 n n1 rc running result v
  mkdir -p "$OUT/logs/$group"
  # shellcheck disable=SC2086
  n=$(cargo test -p fandhe-ai $targs -- --ignored --exact --list "$test" 2>/dev/null | grep -c ': test$' || true)
  # shellcheck disable=SC2086
  n1=$(cargo test -p fandhe-ai $targs -- --ignored --exact --list "$test" 2>/dev/null | grep -Fxc "$test: test" || true)
  if [ "$n" -ne 1 ] || [ "$n1" -ne 1 ]; then
    echo "--list の選択が 1 件でない: $test (all=$n exact=$n1)" >&2
    record "$group" single 1 "$targs" "$test" - - - undetermined
    ANY_UNDET=1
    return 0
  fi
  set +e
  # shellcheck disable=SC2086
  cargo test -p fandhe-ai $targs -- --ignored --exact --nocapture --test-threads=1 "$test" \
    >"$OUT/logs/$group/$test.raw" 2>&1
  rc=$?
  set -e
  echo "rc=$rc" >>"$OUT/logs/$group/$test.raw"
  running=$(grep -c '^running 1 test$' "$OUT/logs/$group/$test.raw" || true)
  result=$(grep -E '^test result: ' "$OUT/logs/$group/$test.raw" | head -n 1 || true)
  if [ "$rc" -eq 0 ] && [ "$running" -eq 1 ] && [[ $result == "test result: ok. 1 passed; 0 failed"* ]]; then
    v=pass
  else
    v=fail
    ANY_FAIL=1
  fi
  record "$group" single 1 "$targs" "$test" "$rc" "$running" "${result:-none}" "$v"
}

# run_paged <group> <pkg> <target> <expected-count> <i>: paged attention の 1 run（--release・独立プロセス）。
# 期待件数は backend 3・facade 1。判定は 5 回すべて pass。
run_paged() {
  local group=$1 pkg=$2 target=$3 count=$4 i=$5 base rc running result v
  base="$OUT/logs/$group/$target.run$i"
  set +e
  if [ "$pkg" = "fandhe-ai-backend-metal" ]; then
    cargo test --release -p "$pkg" --test "$target" -- --ignored --nocapture --test-threads=1 metal_paged_attention >"$base.raw" 2>&1
  else
    cargo test --release -p "$pkg" --test "$target" -- --ignored --exact --nocapture --test-threads=1 metal_paged_attention_forward_matches_cpu_tape >"$base.raw" 2>&1
  fi
  rc=$?
  set -e
  echo "rc=$rc" >>"$base.raw"
  running=$(grep -c "^running $count tests\?\$" "$base.raw" || true)
  result=$(grep -E '^test result: ' "$base.raw" | head -n 1 || true)
  if [ "$rc" -eq 0 ] && [ "$running" -eq 1 ] && [[ $result == "test result: ok. $count passed; 0 failed"* ]]; then
    v=pass
  else
    v=fail
    ANY_FAIL=1
  fi
  record "$group" paged-run "$i" "--release -p $pkg --test $target" "(${count} tests)" "$rc" "$running" "${result:-none}" "$v"
}

# paged の --list 検査。backend はフィルタ metal_paged_attention の選択が期待の 3 件と完全一致、
# facade は exact で 1 件であること。
PAGED_BACKEND_WANT='metal_paged_attention_matrix_matches_cpu_reference: test
metal_paged_attention_tolerance_constants_pinned: test
metal_paged_attention_tracked_matrix_matches_cpu_reference: test'
paged_list_ok() {
  local got
  got=$(cargo test --release -p fandhe-ai-backend-metal --test paged_kv_real_device -- --ignored --list metal_paged_attention 2>/dev/null | grep ': test$' | sort || true)
  if [ "$got" != "$PAGED_BACKEND_WANT" ]; then
    echo "backend の --list 選択が期待の 3 件と異なる" >&2
    return 1
  fi
  got=$(cargo test --release -p fandhe-ai --test paged_attention_facade -- --ignored --exact --list metal_paged_attention_forward_matches_cpu_tape 2>/dev/null | grep ': test$' || true)
  if [ "$got" != "metal_paged_attention_forward_matches_cpu_tape: test" ]; then
    echo "facade の --list 選択が期待の 1 件と異なる" >&2
    return 1
  fi
  return 0
}

# 1〜8（上から順に 1 本ずつ）。--release は付けない（各 README のコマンド形を正とする）。
run_single compat-sequential-pool3d-3029 "--test compat_sequential_pool3d_backend_parity" metal_pool3d_forward_backward_matches_cpu
run_single compat-sequential-conv-transpose3d-3033 "--test compat_sequential_conv_transpose3d_backend_parity" metal_conv_transpose3d_forward_backward_matches_cpu
run_single compat-sequential-max-unpool-3037 "--test compat_functional_max_unpool_backend_parity" metal_functional_max_unpool_2d_forward_backward_matches_cpu
run_single compat-sequential-max-unpool-3037 "--test compat_functional_max_unpool_backend_parity" metal_functional_max_unpool_3d_forward_backward_matches_cpu
run_single compat-sequential-fold-unfold-3041 "--test compat_sequential_fold_unfold_backend_parity" metal_fold_unfold_forward_backward_matches_cpu
run_single compat-sequential-local-response-norm-3045 "--test compat_sequential_lrn_backend_parity" metal_local_response_norm_forward_backward_matches_cpu
run_single functional-transforms-jvp-multi-3064 "--test functional_transforms_jvp_multi_backend_parity" metal_jvp_multi_matches_cpu_reference
run_single functional-transforms-vmap-grad-3073 "--test functional_transforms_facade" metal_vmap_grad_matches_cpu_reference

# 9〜12: paged attention。backend を 5 回 → facade を 5 回（RULE.txt で順序固定）。
PG=paged-attention-parity-3113
mkdir -p "$OUT/logs/$PG"
if paged_list_ok; then
  for i in 1 2 3 4 5; do run_paged "$PG" fandhe-ai-backend-metal paged_kv_real_device 3 "$i"; done
  for i in 1 2 3 4 5; do run_paged "$PG" fandhe-ai paged_attention_facade 1 "$i"; done
else
  ANY_UNDET=1
  for i in 1 2 3 4 5; do record "$PG" paged-run "$i" "backend" "(list mismatch)" - - - undetermined; done
  for i in 1 2 3 4 5; do record "$PG" paged-run "$i" "facade" "(list mismatch)" - - - undetermined; done
fi

env_info after >>"$OUT/env_info.raw"

mask_all

# PAGED_PARITY 行の 5 回中央値。run 欠損・行集合の不一致・metal／metal-tracked の片方欠落・数値パース不能は
# すべてエラー（exit 2）。時間計測ではないので run ごとの生値はログ（run 別）にそのまま残る。
paged_median() { # paged_median <dir>
  local i f
  for i in 1 2 3 4 5; do
    f="$1/paged_kv_real_device.run$i.log"
    [ -f "$f" ] || { echo "run ログ欠損: $f" >&2; return 2; }
    grep 'PAGED_PARITY ' "$f" | sed -E 's/^.*PAGED_PARITY /PAGED_PARITY /' | awk -v r="$i" '{print r "\t" $0}'
  done | awk -F'\t' '
    function bad(m) { print "paged_median: " m > "/dev/stderr"; err = 1; exit 2 }
    function isint(s) { return s ~ /^[0-9]+$/ }
    function isnum(s) { return s ~ /^[0-9]+(\.[0-9]+)?(e-?[0-9]+)?$/ }
    function med(arr, key,   j, k2, x, v) {
      for (j = 1; j <= 5; j++) v[j] = arr[key, j] + 0
      for (j = 2; j <= 5; j++) { x = v[j]; k2 = j - 1; while (k2 >= 1 && v[k2] > x) { v[k2 + 1] = v[k2]; k2-- } v[k2 + 1] = x }
      if (v[1] != v[5]) ident = 0
      return v[3]
    }
    {
      run = $1; n = split($2, t, " ")
      if (t[1] != "PAGED_PARITY") bad("行頭が不正")
      b = ""; c = ""; s = ""; tot = ""; fl = ""; ma = ""; mr = ""; me = ""; jd = ""
      for (i = 2; i <= n; i++) {
        p = index(t[i], "="); if (p == 0) bad("トークン不正: " t[i])
        k = substr(t[i], 1, p - 1); v = substr(t[i], p + 1)
        if (k == "backend") b = v; else if (k == "case") c = v; else if (k == "step") s = v
        else if (k == "total") tot = v; else if (k == "fail") fl = v; else if (k == "max_abs") ma = v
        else if (k == "max_rel") mr = v; else if (k == "mean_abs") me = v; else if (k == "judgement") jd = v
        else bad("未知のキー: " k)
      }
      if (b == "" || c == "" || jd == "") bad("必須キー欠損")
      if (!isint(s) || !isint(tot) || !isint(fl) || !isnum(ma) || !isnum(mr) || !isnum(me)) bad("数値パース不能: " $2)
      key = b SUBSEP c SUBSEP s
      if ((key, run) in seen) bad("同一 run に重複行: " b " " c " " s)
      seen[key, run] = 1
      if (!(key in cnt)) { order[++nk] = key; cnt[key] = 0; KB[key] = b; KC[key] = c; KS[key] = s }
      cnt[key]++
      TOT[key, run] = tot; FL[key, run] = fl; MA[key, run] = ma; MR[key, run] = mr; ME[key, run] = me; JD[key, run] = jd
      backends[b] = 1
    }
    END {
      if (err) exit 2
      if (nk == 0) bad("PAGED_PARITY 行が 0 件")
      if (!("metal" in backends) || !("metal-tracked" in backends)) bad("metal／metal-tracked の片方が欠落")
      print "backend\tcase\tstep\ttotal\tfail\tmax_abs_median\tmax_rel_median\tmean_abs_median\tjudgement\truns_identical"
      for (q = 1; q <= nk; q++) {
        key = order[q]
        if (cnt[key] != 5) bad("run 数が 5 でない: " KB[key] " " KC[key] " " KS[key] " (" cnt[key] ")")
        ident = 1
        t1 = med(TOT, key); f1 = med(FL, key); a1 = med(MA, key); r1 = med(MR, key); m1 = med(ME, key)
        for (j = 2; j <= 5; j++) if (JD[key, j] != JD[key, 1]) ident = 0
        printf "%s\t%s\t%s\t%d\t%d\t%.17g\t%.17g\t%.17g\t%s\t%s\n", KB[key], KC[key], KS[key], t1, f1, a1, r1, m1, JD[key, 1], (ident ? "yes" : "no")
      }
    }'
}

PMED_RC=0
if [ "$ANY_UNDET" -eq 0 ]; then
  paged_median "$OUT/logs/$PG" >"$OUT/logs/$PG/paged_parity_median.tsv" || PMED_RC=$?
fi

# 識別子保存検査: マスク後のログに期待するテスト名の "test <名前> ... " 行が逐語で残っていること。
# 無ければ fail-closed（マスクがテスト名を壊した、またはログ欠損）。ログは手で直さない。
IDENT_FAIL=0
ident_check() { # ident_check <file> <test-name>...
  local f=$1 t
  shift
  if [ ! -f "$f" ]; then
    echo "識別子保存検査: ログ欠損 $f" >&2
    IDENT_FAIL=1
    return 0
  fi
  for t in "$@"; do
    if ! grep -Fq -- "test $t ... " "$f"; then
      echo "識別子保存検査: $f に 'test $t ... ' が無い" >&2
      IDENT_FAIL=1
    fi
  done
}
# --list 不一致で実行していない場合は検査しない（undetermined として別に扱う）。
if [ "$ANY_UNDET" -eq 0 ]; then
  ident_check "$OUT/logs/compat-sequential-pool3d-3029/metal_pool3d_forward_backward_matches_cpu.log" metal_pool3d_forward_backward_matches_cpu
  ident_check "$OUT/logs/compat-sequential-conv-transpose3d-3033/metal_conv_transpose3d_forward_backward_matches_cpu.log" metal_conv_transpose3d_forward_backward_matches_cpu
  ident_check "$OUT/logs/compat-sequential-max-unpool-3037/metal_functional_max_unpool_2d_forward_backward_matches_cpu.log" metal_functional_max_unpool_2d_forward_backward_matches_cpu
  ident_check "$OUT/logs/compat-sequential-max-unpool-3037/metal_functional_max_unpool_3d_forward_backward_matches_cpu.log" metal_functional_max_unpool_3d_forward_backward_matches_cpu
  ident_check "$OUT/logs/compat-sequential-fold-unfold-3041/metal_fold_unfold_forward_backward_matches_cpu.log" metal_fold_unfold_forward_backward_matches_cpu
  ident_check "$OUT/logs/compat-sequential-local-response-norm-3045/metal_local_response_norm_forward_backward_matches_cpu.log" metal_local_response_norm_forward_backward_matches_cpu
  ident_check "$OUT/logs/functional-transforms-jvp-multi-3064/metal_jvp_multi_matches_cpu_reference.log" metal_jvp_multi_matches_cpu_reference
  ident_check "$OUT/logs/functional-transforms-vmap-grad-3073/metal_vmap_grad_matches_cpu_reference.log" metal_vmap_grad_matches_cpu_reference
  for i in 1 2 3 4 5; do
    ident_check "$OUT/logs/$PG/paged_kv_real_device.run$i.log" metal_paged_attention_matrix_matches_cpu_reference metal_paged_attention_tracked_matrix_matches_cpu_reference metal_paged_attention_tolerance_constants_pinned
    ident_check "$OUT/logs/$PG/paged_attention_facade.run$i.log" metal_paged_attention_forward_matches_cpu_tape
  done
fi

# summary.tsv はヘッダ + 18 行が揃っていること（欠損は例外扱い）。
ROWS=$(grep -c . "$OUT/summary.tsv" || true)
SUMMARY_BAD=0
if [ "$ROWS" -ne "$ROWS_EXPECTED" ]; then
  echo "summary.tsv が $ROWS_EXPECTED 行（ヘッダ + 18）でない: $ROWS" >&2
  SUMMARY_BAD=1
fi

# 提出物の完全性: 全成果物の SHA-256（回収後に shasum -a 256 -c で無改変を確かめる）。
(
  cd "$OUT"
  find . -type f ! -name SHA256SUMS | sed -e 's|^\./||' | sort | while IFS= read -r f; do
    printf '%s  %s\n' "$(sha256 "$f")" "$f"
  done >SHA256SUMS
)

echo "done: any_fail=$ANY_FAIL any_undetermined=$ANY_UNDET leak=$LEAK ident_fail=$IDENT_FAIL summary_bad=$SUMMARY_BAD median_rc=$PMED_RC"
if [ "$ANY_UNDET" -ne 0 ] || [ "$IDENT_FAIL" -ne 0 ] || [ "$SUMMARY_BAD" -ne 0 ] || [ "$PMED_RC" -ne 0 ]; then
  exit 2
fi
if [ "$LEAK" -ne 0 ] || [ "$ANY_FAIL" -ne 0 ]; then
  exit 1
fi
