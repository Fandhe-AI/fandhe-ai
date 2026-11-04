#!/usr/bin/env bash
# イシュー #3860（再計測）: #4026 が train_gpu_parity.rs に登録した nn 族の学習ステップ（18 ケース）を DGX Spark GB10 の
# GPU tape で CPU tape と突き合わせる実機 parity 実行スクリプト。手動実行専用（CI からは呼ばない）。
# 規則は同ディレクトリの RULE.txt（事前登録）が正。判定は各テストの cargo test 終了コードと出力件数。
# ../onnx-train-cuda-parity-3864/run_cuda_parity.sh（#3864）の複製で、ISSUE・CASES・N_CASES と自己検査の固定行の
# 展開先名に加え、次の 4 点だけを変えた（マスク・leak 検査・.rev-stamp 検査・実機ロックのロジックは不変）。
#   - fail／undetermined の印の grep を行頭固定にしない（--nocapture では最初の TRAIN_PARITY 行が「test <名前> ... 」と
#     同じ行に出るため。#3864 README の注記）
#   - exit 0 でも、TRAIN_PARITY 行に verdict=pass 以外が 1 行でもあれば fail、EXPECTED_CASES の各ケースが
#     family=$FAMILY の記録行として 1 行以上出ていなければ undetermined にする（ケースの取りこぼしを pass にしない）
#   - テスト単位の判定表を tests.tsv（#3864 までの summary.tsv と同じ列）へ書き、summary.tsv には TRAIN_PARITY 行を
#     1 行 1 量で集計する（マスク後のログから作る）
#   - 対象はテスト 1 本だけ（scaffold の自己検査は走らせない）
# #3864 版は #3862 版の複製で、#3862 版は #3860 版（#3856 版に train_gpu_parity の undetermined 分岐を足したもの）に
# package をターゲット引数へ含める変更を入れたもの。
#
# マスク規則: ホスト名・$HOME・ユーザー名だけを置換する。ユーザー名は前後とも境界（英数字・_・.・- 以外）の
# ときだけ置換し、crate 名・パッケージ名・ディレクトリ名（fandhe-ai・fandhe_ai*・target-fandhe-ai・Fandhe-AI 等）は
# 変えない。回収ログは手で編集しない。
#
# 終了コード: 0=全 pass／1=fail またはマスク漏れ（記録は残る。マスク漏れのファイルは削除）／
#             2=引数・前提不正、ロック取得不能、または undetermined（--list の選択件数が 1 でない・timeout・ロック喪失・train_gpu_parity の undetermined）。
# 使い方（GB10 ノード上。PATH に cargo と nvcc、CARGO_TARGET_DIR=$HOME/work/target-fandhe-ai を指定して呼ぶ）:
#   bash run_cuda_parity.sh <tree-abs-path> <out-abs-path>
#   bash run_cuda_parity.sh --self-test        # マスクと .rev-stamp 検査の不変条件だけを検査する（どの OS でも可。bash 3.2 対応）
# bash 3.2 で動くよう、連想配列・mapfile・${var,,}・sed -i は使わない。
set -euo pipefail

ISSUE=3860
TIMEOUT_SEC=1800
# 事前ビルドするテストターゲット（1 行 1 ターゲット。「-p <package> --test <target>」。固定文字列のみ）。
BUILD_TARGETS='-p fandhe-ai --test train_gpu_parity'
# 行: 群|ターゲット引数|テストのフルパス（関数名がログのファイル名になる）。固定文字列のみ（外部入力を展開しない）。
# 測定対象は雛形の nn 族 1 本（#4026 が登録した 18 ケース）。
CASES='nn-train-4026|-p fandhe-ai --test train_gpu_parity|train_parity_nn_cuda_matches_cpu'
N_CASES=1
# 記録行の族名と、記録行に必ず現れるべきケース名（対象コミット 2f59b9638 の CASES の family "nn" 18 行。固定文字列のみ）。
FAMILY=nn
EXPECTED_CASES='nn_upsample_scale_factor_train_step_s1 nn_upsample_scale_factor_train_step_s2
nn_shrink_train_step_s1 nn_shrink_train_step_s2
nn_batch_norm3d_train_step_s1 nn_batch_norm3d_train_step_s2
nn_channel_shuffle_train_step_s1 nn_channel_shuffle_train_step_s2
nn_adaptive_pool3d_train_step_s1 nn_adaptive_pool3d_train_step_s2
nn_lp_pool_train_step_s1 nn_lp_pool_train_step_s2
nn_fractional_max_pool_train_step_s1 nn_fractional_max_pool_train_step_s2
nn_bilinear_train_step_s1 nn_bilinear_train_step_s2
nn_similarity_distance_train_step_s1 nn_similarity_distance_train_step_s2'
N_EXPECTED=18

# records <log>: TRAIN_PARITY 記録を 1 行 1 記録で標準出力へ（行頭に限らず拾う）。
records() { grep -o 'TRAIN_PARITY .*' "$1" || true; }

B='[^A-Za-z0-9_.-]' # ユーザー名の境界に使わない文字集合の否定形（- も . も _ も英数字も境界にしない）

# 正規表現（sed -E／grep -E 共通）のメタ文字と区切り文字 / をエスケープする。
esc_re() { printf '%s' "$1" | sed -e 's/[][\\.^$*+?(){}|/]/\\&/g'; }

# mask_stream <user> <host> <host-short> <home>: 標準入力をマスクして標準出力へ。
# 順序: ホスト名（長・短）→ $HOME → ユーザー名（前後境界付き・隣接取り逃しに備え 2 回）。
mask_stream() {
  local un=$1 hn=$2 hns=$3 home=$4 script=""
  if [ -n "$hn" ]; then script="${script}s/$(esc_re "$hn")/<cuda-node>/g"$'\n'; fi
  if [ -n "$hns" ]; then script="${script}s/$(esc_re "$hns")/<cuda-node>/g"$'\n'; fi
  if [ -n "$home" ] && [ "$home" != "/" ]; then script="${script}s/$(esc_re "$home")/<home>/g"$'\n'; fi
  local pat
  pat="(^|$B)$(esc_re "$un")($B|\$)"
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
   Compiling fandhe-ai v0.13.0 (<home>/work/rust-ai-library-run-3860-train/crates/facade)
   Compiling fandhe-ai-backend-cuda v0.13.0
Running unittests src/lib.rs (target/debug/deps/fandhe_ai-0123456789abcdef)
fandhe_ai_backend_cuda::device
~/work/target-fandhe-ai/debug/deps/fandhe_ai_backend_cuda-abc
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
  if mask_selfcheck fandhe node-x node-x /home/fandhe && rev_stamp_selfcheck; then
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
  echo "rev_stamp: $REV_HEAD"
  echo "script_sha256: $SCRIPT_SHA"
  if lock_is_mine; then echo "hw_lock: held owner=$LOCK_OWNER"; else echo "hw_lock: LOST"; fi
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
  while IFS= read -r t; do
    [ -n "$t" ] || continue
    # t は上の固定文字列のみ（ユーザー入力ではない）なので単語分割で展開する。
    # shellcheck disable=SC2086
    cargo test $t --no-run </dev/null
  done <<BT
$BUILD_TARGETS
BT
} >"$OUT/build.raw" 2>&1

printf 'group\ttarget\ttest\texit_code\trunning\tresult\tverdict\n' >"$OUT/tests.tsv"
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
    printf '%s\t%s\t%s\t-\t-\t-\tundetermined\n' "$GROUP" "$TARGET" "$TEST" >>"$OUT/tests.tsv"
    ANY_UNDET=1
    continue
  fi
  # 選択の検証: --list で ": test" 終わりの行がちょうど 1 行であること（0 または 2 以上は実行しない）。
  # TARGET は上の固定文字列のみ（ユーザー入力ではない）なので単語分割で展開する。
  # shellcheck disable=SC2086
  N=$(cargo test $TARGET -- --ignored --exact --list "$TEST" 2>/dev/null | grep -c ': test$' || true)
  if [ "$N" -ne 1 ]; then
    echo "--list の選択件数が 1 でない: $TEST ($N)" >&2
    printf '%s\t%s\t%s\t-\t-\t-\tundetermined\n' "$GROUP" "$TARGET" "$TEST" >>"$OUT/tests.tsv"
    ANY_UNDET=1
    continue
  fi
  set +e
  # shellcheck disable=SC2086
  timeout "$TIMEOUT_SEC" cargo test $TARGET -- --ignored --exact --nocapture --test-threads=1 "$TEST" \
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
  elif [ "$RC" -eq 124 ]; then
    V=undetermined
    ANY_UNDET=1
  elif [ "$RC" -eq 0 ] && [ "$RUNNING" -eq 1 ] && [[ $RESULT == "test result: ok. 1 passed; 0 failed"* ]]; then
    # exit 0 でも記録行を検査する: pass 以外の verdict が 1 行でもあれば fail、族の違う記録やケースの欠けは undetermined。
    NONPASS=$(records "$OUT/logs/$GROUP/$FN.raw" | grep -Evc ' verdict=pass$' || true)
    OTHERFAM=$(records "$OUT/logs/$GROUP/$FN.raw" | grep -Evc "^TRAIN_PARITY case=[A-Za-z0-9_]+ family=$FAMILY " || true)
    MISSING=0
    for c in $EXPECTED_CASES; do
      if ! records "$OUT/logs/$GROUP/$FN.raw" | grep -Eq "^TRAIN_PARITY case=$c family=$FAMILY "; then
        echo "記録行に現れないケース: $c" >&2
        MISSING=$((MISSING + 1))
      fi
    done
    if [ "$NONPASS" -ne 0 ]; then
      echo "exit 0 だが pass 以外の記録行が $NONPASS 行ある" >&2
      V=fail
      ANY_FAIL=1
    elif [ "$OTHERFAM" -ne 0 ] || [ "$MISSING" -ne 0 ]; then
      echo "記録行の族またはケースが期待と合わない（他族 $OTHERFAM 行・欠け $MISSING/$N_EXPECTED ケース）" >&2
      V=undetermined
      ANY_UNDET=1
    else
      V=pass
    fi
  elif grep -Eq 'TRAIN_PARITY .* verdict=fail( |$)' "$OUT/logs/$GROUP/$FN.raw"; then
    V=fail
    ANY_FAIL=1
  elif grep -Eq 'TRAIN_PARITY .* verdict=undetermined( |$)|undetermined: 族 .* の登録が 0 件' "$OUT/logs/$GROUP/$FN.raw"; then
    # train_gpu_parity の undetermined（run 関数の Err・全ゼロの量・登録 0 件の族）。pass として扱わない。
    V=undetermined
    ANY_UNDET=1
  else
    V=fail
    ANY_FAIL=1
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$GROUP" "$TARGET" "$TEST" "$RC" "$RUNNING" "${RESULT:-none}" "$V" >>"$OUT/tests.tsv"
done <<EOF
$CASES
EOF

env_info after >>"$OUT/env_info.raw"

mask_all

# tests.tsv はヘッダ + N_CASES 行が揃っていること（欠損は例外扱い）。
ROWS=$(grep -c . "$OUT/tests.tsv" || true)
if [ "$ROWS" -ne $((N_CASES + 1)) ]; then
  echo "tests.tsv が $((N_CASES + 1)) 行（ヘッダ + $N_CASES）でない: $ROWS" >&2
  exit 2
fi

# summary.tsv: マスク後の各ログの TRAIN_PARITY 記録を 1 行 1 量で集計する（key=value を列へ展開。無い鍵は -）。
# reason（undetermined 行）は値に空白を含みうるので行末までを 1 列にする。
printf 'test\tcase\tfamily\tquantity\tarm\troute\tjudge\ttotal\tfail_count\tmax_abs_diff\tmean_abs_diff\tmax_rel_err\tverdict\treason\n' \
  >"$OUT/summary.tsv"
while IFS='|' read -r GROUP TARGET TEST; do
  FN=${TEST##*::}
  [ -f "$OUT/logs/$GROUP/$FN.log" ] || continue
  records "$OUT/logs/$GROUP/$FN.log" | awk -v t="$FN" '
    BEGIN { OFS = "\t"; n = split("case family quantity arm route judge total fail_count max_abs_diff mean_abs_diff max_rel_err verdict reason", K, " ") }
    {
      for (k in v) delete v[k]
      for (i = 2; i <= NF; i++) {
        p = index($i, "="); if (p == 0) continue
        key = substr($i, 1, p - 1); val = substr($i, p + 1)
        if (key == "reason") { for (j = i + 1; j <= NF; j++) val = val " " $j; v[key] = val; break }
        v[key] = val
      }
      line = t
      for (j = 1; j <= n; j++) line = line OFS ((K[j] in v) ? v[K[j]] : "-")
      print line
    }' >>"$OUT/summary.tsv"
done <<EOF
$CASES
EOF

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
