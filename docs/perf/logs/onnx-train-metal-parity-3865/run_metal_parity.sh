#!/usr/bin/env bash
# イシュー #3865: ONNX モデルの学習（#3842 の公開経路 OnnxModel::bind_autograd）を Mac mini（Apple M4）の Metal の GPU tape で
# CPU tape と突き合わせる実機 parity 実行スクリプト。手動実行専用（CI からは呼ばない）。
# ../param-groups-resident-metal-parity-3863/run_metal_parity.sh（#3863）の複製で、ISSUE・BUILD_TARGETS・CASES・N_CASES と
# 自己検査の固定行のイシュー番号だけを変えた（GB10 版 ../onnx-train-cuda-parity-3864/run_cuda_parity.sh〈#3864〉と同じ変え方）。
# #3863 版は #3861 版に、package をケースごとに書く変更（GB10 版 #3862 と同じ）を足したもの。
# #3861 版は #3858 版に train_gpu_parity の TRAIN_PARITY 判定分岐を足したもの。以下は #3858 版の説明。
# 規則は同ディレクトリの RULE.txt（事前登録）が正。判定は各テストの cargo test 終了コードと出力件数。
# GB10 版 ../create-graph-c-cuda-parity-3856/run_cuda_parity.sh（#3856。PR #4020）の複製で、
# 実機ロック・2 行形式 .rev-stamp・<home> マスク・マスク／leak 検査の骨格は変えず、次の 5 点だけを変えた。
#   1. 対象テスト名を create_graph_<族>_metal_matches_cpu にする（CASES）
#   2. Darwin（Mac mini）限定にし、ホスト名のマスク先を <metal-node>、leak 検査の個人パスを /Users/ にする
#   3. env_info を Mac の allowlist（sw_vers・CPU 名・メモリ圧迫率）にする（../metal-autodiff-parity-3169 と同じ）
#   4. 外部 timeout を付けない（macOS に timeout(1) が無い。Metal 前例 ../metal-facade-parity-3166・
#      ../metal-autodiff-parity-3169 と同じ）。ハングしたら外から止め、その回と以降を undetermined にする（RULE.txt）
#   5. 他の cargo／rustc の検査を各テストの前にも行う（16GB 共有機で計測中のビルド並走を避ける）
#
# マスク規則: ホスト名・$HOME・ユーザー名だけを置換する。ユーザー名は前後とも境界（英数字・_・.・- 以外）の
# ときだけ置換し、crate 名・パッケージ名・ディレクトリ名（fandhe-ai・fandhe_ai*・target-fandhe-ai・Fandhe-AI 等）は
# 変えない。回収ログは手で編集しない。
#
# 終了コード: 0=全 pass／1=fail またはマスク漏れ（記録は残る。マスク漏れのファイルは削除）／
#             2=引数・前提不正、ロック取得不能、または undetermined（--list の選択件数が 1 でない・ロック喪失・他の cargo／rustc の並走・上記の undetermined）。
# 使い方（Mac mini 上。zsh -lc で PATH を通し、CARGO_TARGET_DIR=$HOME/work/target-fandhe-ai を指定して呼ぶ）:
#   bash run_metal_parity.sh <tree-abs-path> <out-abs-path>
#   bash run_metal_parity.sh --self-test        # マスクと .rev-stamp 検査の不変条件だけを検査する（どの OS でも可。bash 3.2 対応）
# bash 3.2 で動くよう、連想配列・mapfile・${var,,}・sed -i は使わない。
set -euo pipefail

ISSUE=3865
# 事前ビルドするテストターゲット（1 行 1 ターゲット。「-p <package> --test <target>」。固定文字列のみ）。
BUILD_TARGETS='-p fandhe-ai --test train_gpu_parity'
# 行: 群|ターゲット引数|テストのフルパス（関数名がログのファイル名になる）。固定文字列のみ（外部入力を展開しない）。
# 1 行目は雛形の scaffold 族（経路の自己検査。測定対象外）。2 行目が測定対象の onnx 族（#3842。2 ケース）。
CASES='scaffold-3778|-p fandhe-ai --test train_gpu_parity|train_parity_scaffold_metal_matches_cpu
onnx-train-3842|-p fandhe-ai --test train_gpu_parity|train_parity_onnx_metal_matches_cpu'
N_CASES=2

B='[^A-Za-z0-9_.-]' # ユーザー名の境界に使わない文字集合の否定形（- も . も _ も英数字も境界にしない）

# 正規表現（sed -E／grep -E 共通）のメタ文字と区切り文字 / をエスケープする。
esc_re() { printf '%s' "$1" | sed -e 's/[][\\.^$*+?(){}|/]/\\&/g'; }

# mask_stream <user> <host> <host-short> <home>: 標準入力をマスクして標準出力へ。
# 順序: ホスト名（長・短）→ $HOME → ユーザー名（前後境界付き・隣接取り逃しに備え 2 回）。
mask_stream() {
  local un=$1 hn=$2 hns=$3 home=$4 script=""
  if [ -n "$hn" ]; then script="${script}s/$(esc_re "$hn")/<metal-node>/g"$'\n'; fi
  if [ -n "$hns" ]; then script="${script}s/$(esc_re "$hns")/<metal-node>/g"$'\n'; fi
  if [ -n "$home" ] && [ "$home" != "/" ]; then script="${script}s/$(esc_re "$home")/<home>/g"$'\n'; fi
  local pat
  pat="(^|$B)$(esc_re "$un")($B|\$)"
  script="${script}s/${pat}/\\1<user>\\2/g"$'\n'"s/${pat}/\\1<user>\\2/g"
  sed -E -e "$script"
}

# leak_check <file> <user> <host> <host-short> <home>: 残存があれば 1、無ければ 0。
leak_check() {
  local f=$1 un=$2 hn=$3 hns=$4 home=$5 s
  for s in "$hn" "$hns" "$home" "/Users/"; do
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
   Compiling fandhe-ai v0.13.0 (<home>/work/rust-ai-library-run-3865/crates/facade)
   Compiling fandhe-ai-backend-metal v0.13.0
Running unittests src/lib.rs (target/debug/deps/fandhe_ai-0123456789abcdef)
fandhe_ai_backend_metal::device
~/work/target-fandhe-ai/debug/deps/fandhe_ai_backend_metal-abc
repository: Fandhe-AI/fandhe-ai
KEEP
  : >"$d/chg.in"
  : >"$d/chg.want"
  # 各行: 入力 -> 期待。実際の値で組み立てる。
  printf '%s/x\n' "$home" >>"$d/chg.in"
  printf '<home>/x\n' >>"$d/chg.want"
  printf 'owner=%s\n' "$un" >>"$d/chg.in"
  printf 'owner=<user>\n' >>"$d/chg.want"
  printf '%s@%s\n' "$un" "$hn" >>"$d/chg.in"
  printf '<user>@<metal-node>\n' >>"$d/chg.want"
  printf '%s\n' "$hn" >>"$d/chg.in"
  printf '<metal-node>\n' >>"$d/chg.want"
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

# check_rev_stamp <file>: 「40 桁小文字 16 進数\ndirty=0\n」（末尾改行は任意）だけなら 0 を返し、
# 1 行目を REV_HEAD に入れる。それ以外（dirty≠0・CR・空白・余計な行）は 1。
check_rev_stamp() {
  local f=$1 body line1 line2 rest
  body=$(
    cat "$f"
    printf x
  )
  body=${body%x}
  body=${body%$'\n'}
  line1=${body%%$'\n'*}
  rest=${body#*$'\n'}
  [ "$rest" != "$body" ] || return 1
  line2=$rest
  case "$line2" in *$'\n'*) return 1 ;; esac
  [[ $line1 =~ ^[0-9a-f]{40}$ ]] || return 1
  [ "$line2" = "dirty=0" ] || return 1
  REV_HEAD=$line1
  return 0
}

rev_stamp_selfcheck() {
  local d rc=0 h=0123456789abcdef0123456789abcdef01234567
  d=$(mktemp -d)
  printf '%s\ndirty=0\n' "$h" >"$d/ok1"
  printf '%s\ndirty=0' "$h" >"$d/ok2"
  printf '%s\ndirty=1\n' "$h" >"$d/ng1"
  printf '%s\n' "$h" >"$d/ng2"
  printf '%s\ndirty=0\nx\n' "$h" >"$d/ng3"
  printf '%s\r\ndirty=0\n' "$h" >"$d/ng4"
  printf '%s\ndirty=0\n' "0123456789ABCDEF0123456789abcdef01234567" >"$d/ng5"
  for f in ok1 ok2; do check_rev_stamp "$d/$f" || { echo "selfcheck: $f を拒否した" >&2; rc=1; }; done
  for f in ng1 ng2 ng3 ng4 ng5; do
    if check_rev_stamp "$d/$f"; then echo "selfcheck: $f を受理した" >&2; rc=1; fi
  done
  rm -rf "$d"
  return "$rc"
}

if [ "${1:-}" = "--self-test" ]; then
  if [ "$#" -ne 1 ]; then
    echo "usage: $0 --self-test" >&2
    exit 2
  fi
  if mask_selfcheck fandhe node-x node-x /Users/fandhe && rev_stamp_selfcheck; then
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

export LC_ALL=C
REV_HEAD=
if ! check_rev_stamp "$TREE/.rev-stamp"; then
  echo ".rev-stamp が 2 行形式（40 桁の小文字 16 進数／dirty=0）でない" >&2
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

# 実機ロック。取れなければ待たず・奪わず停止する（他セッションの計測を壊さない）。
LOCK_DIR="$HOME/work/.fandhe-hw-lock"
LOCK_HELD=0
LOCK_OWNER="#$ISSUE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "実機ロックを取得できない（他セッションが保持中）。待たずに停止" >&2
  exit 2
fi
LOCK_HELD=1
printf '%s\n' "$LOCK_OWNER" >"$LOCK_DIR/owner"
lock_is_mine() { [ -f "$LOCK_DIR/owner" ] && [ "$(cat "$LOCK_DIR/owner")" = "$LOCK_OWNER" ]; }

# マスクしてから収録し、残っていれば fail-closed で止める（残ったファイルは raw ごと削除する）。
# 異常終了（set -e・exit 2 を含む）でも未処理の .raw が出力先に残らず、ロックも必ず外れるよう EXIT trap で後始末する。
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
cleanup() {
  set +e
  [ -d "$OUT" ] && mask_all
  # 自分が取ったロックだけを外す（owner が自分のままのとき。失われていれば触らない）。
  if [ "$LOCK_HELD" -eq 1 ] && lock_is_mine; then rm -rf "$LOCK_DIR"; fi
}
trap cleanup EXIT

mkdir -p "$OUT/logs"
cp "$TREE/.rev-stamp" "$OUT/rev-stamp.txt"

# 環境情報は allowlist で取得する。hostname は取らず masked と固定で書く。プロセス名は書き出さない。
env_info() { # env_info <label>
  echo "== $1 =="
  echo "date_utc: $(date -u +%Y-%m-%dT%H:%MZ)"
  echo "hostname: masked"
  echo "sw_vers: $( (sw_vers | tr '\n' ' ') 2>/dev/null || echo 取得不能)"
  echo "cpu: $( (sysctl -n machdep.cpu.brand_string) 2>/dev/null || echo 取得不能)"
  echo "memsize_bytes: $( (sysctl -n hw.memsize) 2>/dev/null || echo 取得不能)"
  echo "memory_free_percent: $( (memory_pressure | sed -n -e 's/^System-wide memory free percentage: *//p') 2>/dev/null || echo 取得不能)"
  echo "rustc: $( (rustc -V) 2>/dev/null || echo 取得不能)"
  echo "cargo: $( (cargo -V) 2>/dev/null || echo 取得不能)"
  echo "arch: $(uname -m)"
  echo "load: $( (uptime | sed -e 's/.*load averages*: *//') 2>/dev/null || echo 取得不能)"
  echo "rev_stamp: $REV_HEAD"
  echo "script_sha256: $SCRIPT_SHA"
  if lock_is_mine; then echo "hw_lock: held owner=$LOCK_OWNER"; else echo "hw_lock: LOST"; fi
}
env_info before >"$OUT/env_info.raw"

# 他の cargo／rustc が走っていれば並走させない（RULE.txt）。事前ビルド前は停止、各テスト前は undetermined。
other_build_running() { pgrep -x cargo >/dev/null 2>&1 || pgrep -x rustc >/dev/null 2>&1; }
if other_build_running; then
  echo "他の cargo／rustc プロセスが動作中のため停止" >&2
  exit 2
fi
echo "concurrent_cargo_rustc: none" >>"$OUT/env_info.raw"

cd "$TREE"
# ビルドを先に終え、計測中にコンパイルを並走させない（判定外）。失敗したら set -e で止まる。
{
  while IFS= read -r t; do
    [ -n "$t" ] || continue
    # t は上の固定文字列のみ（ユーザー入力ではない）なので単語分割で展開する。
    # shellcheck disable=SC2086
    cargo test $t --no-run </dev/null
  done <<BT
$BUILD_TARGETS
BT
} >"$OUT/build.raw" 2>&1

printf 'group\ttarget\ttest\texit_code\trunning\tresult\tverdict\n' >"$OUT/summary.tsv"
ANY_FAIL=0
ANY_UNDET=0
LOCK_LOST=0
while IFS='|' read -r GROUP TARGET TEST; do
  FN=${TEST##*::}
  mkdir -p "$OUT/logs/$GROUP"
  # ロックを失ったら以降は測らない（ロックを持たずに測った値は採らない）。
  if [ "$LOCK_LOST" -eq 1 ] || ! lock_is_mine; then
    LOCK_LOST=1
    echo "実機ロックを失った: $TEST を実行しない" >&2
    printf '%s\t%s\t%s\t-\t-\t-\tundetermined\n' "$GROUP" "$TARGET" "$TEST" >>"$OUT/summary.tsv"
    ANY_UNDET=1
    continue
  fi
  # 他のビルドが並走していれば測らない（自分の cargo はこの時点で終了している）。
  if other_build_running; then
    echo "他の cargo／rustc が動作中: $TEST を実行しない" >&2
    printf '%s\t%s\t%s\t-\t-\t-\tundetermined\n' "$GROUP" "$TARGET" "$TEST" >>"$OUT/summary.tsv"
    ANY_UNDET=1
    continue
  fi
  # 選択の検証: --list で ": test" 終わりの行がちょうど 1 行であること（0 または 2 以上は実行しない）。
  # TARGET は上の固定文字列のみ（ユーザー入力ではない）なので単語分割で展開する。
  # shellcheck disable=SC2086
  N=$(cargo test $TARGET -- --ignored --exact --list "$TEST" 2>/dev/null | grep -c ': test$' || true)
  if [ "$N" -ne 1 ]; then
    echo "--list の選択件数が 1 でない: $TEST ($N)" >&2
    printf '%s\t%s\t%s\t-\t-\t-\tundetermined\n' "$GROUP" "$TARGET" "$TEST" >>"$OUT/summary.tsv"
    ANY_UNDET=1
    continue
  fi
  set +e
  # shellcheck disable=SC2086
  cargo test $TARGET -- --ignored --exact --nocapture --test-threads=1 "$TEST" \
    >"$OUT/logs/$GROUP/$FN.raw" 2>&1
  RC=$?
  set -e
  echo "rc=$RC" >>"$OUT/logs/$GROUP/$FN.raw"
  echo "$RC" >"$OUT/logs/$GROUP/$FN.exit_code"
  RUNNING=$(grep -c '^running 1 test$' "$OUT/logs/$GROUP/$FN.raw" || true)
  RESULT=$(grep -E '^test result: ' "$OUT/logs/$GROUP/$FN.raw" | head -n 1 || true)
  if ! lock_is_mine; then
    # 実行中にロックが外れた（他者に奪われた）回は採らない。
    LOCK_LOST=1
    V=undetermined
    ANY_UNDET=1
  elif [ "$RC" -eq 0 ] && [ "$RUNNING" -eq 1 ] && [[ $RESULT == "test result: ok. 1 passed; 0 failed"* ]]; then
    V=pass
  elif grep -Eq '^TRAIN_PARITY .* verdict=fail( |$)' "$OUT/logs/$GROUP/$FN.raw"; then
    V=fail
    ANY_FAIL=1
  elif grep -Eq '^TRAIN_PARITY .* verdict=undetermined( |$)|undetermined: 族 .* の登録が 0 件' "$OUT/logs/$GROUP/$FN.raw"; then
    # train_gpu_parity の undetermined（run 関数の Err・全ゼロの量・登録 0 件の族）。pass として扱わない。
    V=undetermined
    ANY_UNDET=1
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

# summary.tsv はヘッダ + N_CASES 行が揃っていること（欠損は例外扱い）。
ROWS=$(grep -c . "$OUT/summary.tsv" || true)
if [ "$ROWS" -ne $((N_CASES + 1)) ]; then
  echo "summary.tsv が $((N_CASES + 1)) 行（ヘッダ + $N_CASES）でない: $ROWS" >&2
  exit 2
fi

# 提出物の完全性: 全成果物の SHA-256（回収後に sha256sum -c で無改変を確かめる）。
(
  cd "$OUT"
  find . -type f ! -name SHA256SUMS | sed -e 's|^\./||' | sort | while IFS= read -r f; do
    printf '%s  %s\n' "$(sha256 "$f")" "$f"
  done >SHA256SUMS
)

echo "done: any_fail=$ANY_FAIL any_undetermined=$ANY_UNDET leak=$LEAK lock_lost=$LOCK_LOST"
if [ "$ANY_UNDET" -ne 0 ]; then
  exit 2
fi
if [ "$LEAK" -ne 0 ] || [ "$ANY_FAIL" -ne 0 ]; then
  exit 1
fi
