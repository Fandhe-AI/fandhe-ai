#!/usr/bin/env bash
# イシュー #4031 の測り直し（rerun）。前回 ../metal-batch-norm-2e20-4031/ は実行中に展開ツリーが別セッションの rsync で上書きされ、無効になった。
# このスクリプトは前回の ../metal-batch-norm-2e20-4031/run_metal_parity.sh の複製で、対象・判定（pass／fail）は変えず、次の点だけを足した:
#   R1. 実機ロックは送信側が転送の前に取り、owner を第 3 引数で受け取る。自分では取らず、owner が一致するときだけ引き継ぐ。
#       終了時は失敗でも trap で外す（検証より先に trap を置く）。出力先は自分が作った場合だけマスク処理する
#   R2. ソースの指紋（crates/・Cargo.toml・Cargo.lock・rust-toolchain.toml の各ファイルの SHA-256 を名前順に並べた一覧の SHA-256）を
#       EXPECTED_SOURCE_FP と照合する。ビルド前に不一致なら停止、テストの前後で不一致なら undetermined。git archive から同じ値を再現できる
#   R3. テストの前後で、ノードの .rev-stamp が開始時の内容と同じか、スクリプトの SHA-256 が開始時と同じかを確かめ、env_info.txt にも書く。
#       前回はこれらを開始時に 1 回しか読まず、上書きを検出できなかった
# 以下は前回版の説明。
# イシュー #4031: Metal の BatchNorm（train 経路）で 2e20 級の有限入力が途中で overflow しないことを、
# crates/backend-metal/tests/batch_norm_parity.rs の #[ignore] テスト 1 本で Mac mini（Apple M4）の Metal と CPU 参照を突き合わせて確かめる
# 実機 parity 実行スクリプト。手動実行専用（CI からは呼ばない）。規則は同ディレクトリの RULE.txt（事前登録）が正。
# ../optim-train-template-metal-parity-3863/run_metal_parity.sh（#3863 の 2 回目）の複製で、実機ロック・2 行形式 .rev-stamp・
# マスク／leak 検査・他の cargo／rustc の検査の骨格は変えず、次の点だけを変えた:
#   1. 対象を PACKAGE・TEST_TARGET・TEST の 1 本にし、--release を付ける（docs/perf/logs/metal-batch-norm-1736/README.md の #4031 節）
#   2. train_gpu_parity の TRAIN_PARITY 記録の集計（extract_records・tabulate_records・records_selfcheck・EXPECTED_RECORDS・
#      test_result.tsv）を外す。このテストは記録行を出さない。判定は exit code・running 件数・test result 行だけで決める
#   3. 出力名を batch_norm_parity_2e20_4031.log・exit_code・summary.tsv（1 行）にする（--list の出力は list.log）
#   4. 判定: exit 0・running 1 test・1 passed; 0 failed なら pass。exit 非 0 は fail。exit 0 でもそれ以外（0 本実行など）は undetermined
# 以下は #3858 版の説明。
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
#   bash run_metal_parity.sh <tree-abs-path> <out-abs-path> "<lock-owner>"   # lock-owner は「#4031 rerun <UTC>」。送信側が取ったロックの owner
#   bash run_metal_parity.sh --self-test        # マスクと .rev-stamp 検査の不変条件だけを検査する（どの OS でも可。bash 3.2 対応）
# bash 3.2 で動くよう、連想配列・mapfile・${var,,}・sed -i は使わない。
set -euo pipefail

# 対象（固定文字列のみ。外部入力を展開しない）。
PACKAGE=fandhe-ai-backend-metal
TEST_TARGET=batch_norm_parity
TEST=batch_norm_train_2e20_scale_has_no_intermediate_overflow_and_matches_cpu_reference
LOG_BASE=batch_norm_parity_2e20_4031
# ソースの指紋の期待値（RULE.txt の source_fingerprint。86797e1b7 の git archive から計算した）。
EXPECTED_SOURCE_FP=faf953ec518fde73cec4d3354e82941223306ffd2d2d16a419a36ff24e3f84c8

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
   Compiling fandhe-ai-backend-metal v0.13.0 (<home>/work/rust-ai-library-run-4031-rerun/crates/backend-metal)
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

# source_fingerprint <tree>: crates/・Cargo.toml・Cargo.lock・rust-toolchain.toml の全ファイルの SHA-256 を、名前順の一覧にしてさらに SHA-256 を取る。
source_fingerprint() {
  (cd "$1" && find crates Cargo.toml Cargo.lock rust-toolchain.toml -type f | LC_ALL=C sort | tr '\n' '\0' \
    | xargs -0 shasum -a 256 | shasum -a 256 | cut -d' ' -f1)
}

# 指紋の不変条件: 同じ内容なら同じ値、内容の変更・ファイルの追加で値が変わる。
fingerprint_selfcheck() {
  local d rc=0 a b c
  d=$(mktemp -d)
  mkdir -p "$d/crates/x"
  printf 'a\n' >"$d/crates/x/a.rs"
  printf 'c\n' >"$d/Cargo.toml"
  printf 'l\n' >"$d/Cargo.lock"
  printf 't\n' >"$d/rust-toolchain.toml"
  a=$(source_fingerprint "$d")
  [ "$a" = "$(source_fingerprint "$d")" ] || { echo "selfcheck: 指紋が再現しない" >&2; rc=1; }
  [[ $a =~ ^[0-9a-f]{64}$ ]] || { echo "selfcheck: 指紋の形式が不正" >&2; rc=1; }
  printf 'b\n' >"$d/crates/x/a.rs"
  b=$(source_fingerprint "$d")
  [ "$a" != "$b" ] || { echo "selfcheck: 内容の変更で指紋が変わらない" >&2; rc=1; }
  printf 'n\n' >"$d/crates/x/n.rs"
  c=$(source_fingerprint "$d")
  [ "$b" != "$c" ] || { echo "selfcheck: ファイルの追加で指紋が変わらない" >&2; rc=1; }
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
  if mask_selfcheck fandhe node-x node-x /Users/fandhe && rev_stamp_selfcheck && fingerprint_selfcheck; then
    echo "self-test: pass"
    exit 0
  fi
  echo "self-test: FAIL" >&2
  exit 2
fi

if [ "$#" -ne 3 ]; then
  echo "usage: $0 <tree-abs-path> <out-abs-path> <lock-owner> | --self-test" >&2
  exit 2
fi
TREE=$1
OUT=$2
LOCK_OWNER=$3

# 実機ロックの引き継ぎ（R1）。送信側が転送の前に取ったロックの owner を確かめる。以降の検証で止まっても trap で外す。
LOCK_DIR="$HOME/work/.fandhe-hw-lock"
LOCK_HELD=0
OUT_MINE=0
UN=
HN=
HNS=
LEAK=0
lock_is_mine() { [ -f "$LOCK_DIR/owner" ] && [ "$(cat "$LOCK_DIR/owner")" = "$LOCK_OWNER" ]; }
cleanup() {
  set +e
  [ "$OUT_MINE" -eq 1 ] && mask_all
  # 自分が引き継いだロックだけを外す（owner が自分のままのとき。失われていれば触らない）。
  if [ "$LOCK_HELD" -eq 1 ] && lock_is_mine; then rm -rf "$LOCK_DIR"; fi
}
trap cleanup EXIT
if ! [[ $LOCK_OWNER =~ ^#4031\ rerun\ [0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$ ]]; then
  echo "lock-owner の形式が不正" >&2
  exit 2
fi
if ! lock_is_mine; then
  echo "実機ロックの owner が一致しない。奪わず停止" >&2
  exit 2
fi
LOCK_HELD=1

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

# マスクしてから収録し、残っていれば fail-closed で止める（残ったファイルは raw ごと削除する）。
# 異常終了（set -e・exit 2 を含む）でも未処理の .raw が出力先に残らず、ロックも必ず外れるよう EXIT trap で後始末する。
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
  for raw in "$OUT"/*.raw; do
    [ -e "$raw" ] || continue
    # env_info.raw は .txt、それ以外は .log として出力する。
    case "$raw" in
      "$OUT/env_info.raw") mask "$raw" "$OUT/env_info.txt" ;;
      *) mask "$raw" "${raw%.raw}.log" ;;
    esac
  done
}
mkdir -p "$OUT"
OUT_MINE=1
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
  # その時点のツリーから読み直した値（R2・R3）。開始時の値と違えば上書きの証拠になる。
  echo "rev_stamp_now: $(sed -n 1p "$TREE/.rev-stamp" 2>/dev/null || echo 取得不能)"
  echo "script_sha256_now: $(sha256 "$SELF" 2>/dev/null || echo 取得不能)"
  echo "source_fingerprint_expected: $EXPECTED_SOURCE_FP"
  echo "source_fingerprint_now: $(source_fingerprint "$TREE" 2>/dev/null || echo 取得不能)"
}

# ツリーが開始時のままか（R2・R3）。.rev-stamp の内容・スクリプトの SHA-256・ソースの指紋がすべて同じなら 0。
tree_intact() {
  cmp -s "$TREE/.rev-stamp" "$OUT/rev-stamp.txt" \
    && [ "$(sha256 "$SELF")" = "$SCRIPT_SHA" ] \
    && [ "$(source_fingerprint "$TREE")" = "$EXPECTED_SOURCE_FP" ]
}
env_info before >"$OUT/env_info.raw"

# 他の cargo／rustc が走っていれば並走させない（RULE.txt）。事前ビルド前は停止、各テスト前は undetermined。
other_build_running() { pgrep -x cargo >/dev/null 2>&1 || pgrep -x rustc >/dev/null 2>&1; }
if other_build_running; then
  echo "他の cargo／rustc プロセスが動作中のため停止" >&2
  exit 2
fi
echo "concurrent_cargo_rustc: none" >>"$OUT/env_info.raw"
if ! tree_intact; then
  echo "ビルド前のツリーが期待と一致しない（指紋・.rev-stamp・スクリプト）。ビルドせず停止" >&2
  exit 2
fi

cd "$TREE"
# ビルドを先に終え、計測中にコンパイルを並走させない（判定外）。失敗したら set -e で止まる。
cargo test -p "$PACKAGE" --release --test "$TEST_TARGET" --no-run >"$OUT/build.raw" 2>&1

printf 'package\ttarget\ttest\texit_code\trunning\tresult\tverdict\n' >"$OUT/summary.tsv"
ANY_FAIL=0
ANY_UNDET=0
LOCK_LOST=0
RC=-
RUNNING=-
RESULT=-
V=undetermined
# 選択の検証結果（--release・--exact 付き。": test" 終わりの行がちょうど 1 行であること）。
cargo test -p "$PACKAGE" --release --test "$TEST_TARGET" -- --ignored --exact --list "$TEST" >"$OUT/list.raw" 2>&1 || true
N=$(grep -c ': test$' "$OUT/list.raw" || true)
if ! lock_is_mine; then
  # ロックを失ったら測らない（ロックを持たずに測った値は採らない）。
  LOCK_LOST=1
  echo "実機ロックを失った: $TEST を実行しない" >&2
elif other_build_running; then
  # 他のビルドが並走していれば測らない（自分の cargo はこの時点で終了している）。
  echo "他の cargo／rustc が動作中: $TEST を実行しない" >&2
elif ! tree_intact; then
  echo "テスト前のツリーが開始時と一致しない: $TEST を実行しない" >&2
elif [ "$N" -ne 1 ]; then
  echo "--list の選択件数が 1 でない: $TEST ($N)" >&2
else
  set +e
  cargo test -p "$PACKAGE" --release --test "$TEST_TARGET" -- --ignored --exact --nocapture --test-threads=1 "$TEST" \
    >"$OUT/$LOG_BASE.raw" 2>&1
  RC=$?
  set -e
  echo "rc=$RC" >>"$OUT/$LOG_BASE.raw"
  echo "$RC" >"$OUT/exit_code"
  RUNNING=$(grep -c '^running 1 test$' "$OUT/$LOG_BASE.raw" || true)
  RESULT=$(grep -E '^test result: ' "$OUT/$LOG_BASE.raw" | head -n 1 || true)
  if ! lock_is_mine; then
    # 実行中にロックが外れた（他者に奪われた）回は採らない。
    LOCK_LOST=1
    V=undetermined
  elif ! tree_intact; then
    # 実行中にツリーが変わった回は、テストの結果によらず採らない（R2・R3）。
    echo "テスト後のツリーが開始時と一致しない" >&2
    V=undetermined
  elif [ "$RC" -eq 0 ] && [ "$RUNNING" -eq 1 ] && [[ $RESULT == "test result: ok. 1 passed; 0 failed"* ]]; then
    V=pass
  elif [ "$RC" -eq 0 ]; then
    # exit 0 でも 1 本実行されて 1 passed でない回（0 本実行など）は pass にしない。
    V=undetermined
  else
    V=fail
  fi
fi
case "$V" in
  fail) ANY_FAIL=1 ;;
  undetermined) ANY_UNDET=1 ;;
esac
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$PACKAGE" "$TEST_TARGET" "$TEST" "$RC" "$RUNNING" "${RESULT:-none}" "$V" >>"$OUT/summary.tsv"

env_info after >>"$OUT/env_info.raw"

mask_all

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
