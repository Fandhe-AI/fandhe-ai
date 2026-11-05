#!/usr/bin/env bash
# 実機 parity 実行スクリプトの雛形（イシュー #4044）。手動実行専用（CI からは呼ばない）。
# 役割: 共有 CARGO_TARGET_DIR（$HOME/work/target-fandhe-ai）で、検証対象の workspace メンバー成果物が前の回のものの再利用にならないことを
#   確かめたうえで、実機テストを 1 本ずつ別プロセスで実行し、exit code と出力件数だけで pass／fail／undetermined を決める。
# 背景: #4031 の測り直し（../metal-batch-norm-2e20-4031-rerun/）は、リンクされた rlib が前の回の再利用だったため無効になった。
#   原因と再現は docs/real-hardware-shared-target-reuse-decision.md §2（根拠ログ ../shared-target-reuse-repro-4044/）。
# 呼び出し文脈: 各回が docs/perf/logs/<名前>-<issue>/run_<cuda|metal>_parity.sh として複製し、下の「固定値」を埋める（README.md 参照）。
#   R3 はスクリプト 1 本の SHA-256 で実行内容を事前登録（RULE.txt）と対応づけるため、source する補助ファイルには分けない。
# 骨格は ../metal-batch-norm-2e20-4031-rerun2/run_metal_parity.sh（回収物。編集しない）の R1〜R5 で、次の 2 点を直して一般化した:
#   - R5 の対象 rlib を手書き glob（libfandhe_ai_*）でなく cargo metadata の各メンバーの lib ターゲット名から作る。
#     glob は facade 自身の lib（libfandhe_ai-<hash>.rlib）に一致しなかった。Phase 10 の facade テストはこれをリンクしている
#   - プロファイル（dev／release）と機体（cuda／metal）を固定値で切り替える。rerun2 は release 専用だった
# 規則:
#   R1 実機ロックは送信側が転送の前に取り、owner を第 3 引数で受け取る。自分では取らず、owner が一致するときだけ引き継ぐ。終了時は trap で外す
#   R2 ソースの指紋（crates/・Cargo.toml・Cargo.lock・rust-toolchain.toml の各ファイルの SHA-256 を名前順に並べた一覧の SHA-256）を EXPECTED_SOURCE_FP と照合する
#   R3 テストの前後で .rev-stamp・スクリプトの SHA-256・指紋が開始時と同じか確かめる
#   R4 --no-run の前に、全 workspace メンバーへ cargo clean [--release] -p <name> を実行する（crates.io の依存は消さない）。CARGO_TARGET_DIR は想定値以外なら停止する
#   R5 ビルド後、テストバイナリと全メンバーの lib 成果物（.rlib・.rmeta）の更新時刻がロック取得より秒単位で厳密に後であること、
#      REQUIRED_LIBS が揃うこと、実行したバイナリが確かめたバイナリと同じことを確かめる。1 つでも外れれば全テストを実行せず undetermined
#      ロック取得の時刻は、ノード上の owner ファイルの mtime（送信側の時計とのずれを受けない）
# マスク規則: ホスト名・$HOME・ユーザー名だけを置換する。回収ログは手で編集しない。
# 終了コード: 0=全 pass／1=fail またはマスク漏れ／2=引数・前提不正、ロック取得不能、固定値が未記入、または undetermined。
# 使い方（ノード上。CARGO_TARGET_DIR=$HOME/work/target-fandhe-ai を指定して呼ぶ）:
#   bash run_parity.sh <tree-abs-path> <out-abs-path> "<lock-owner>"
#   bash run_parity.sh --self-test   # 不変条件だけを検査する（どの OS でも可。bash 3.2 対応）
# bash 3.2 で動くよう、連想配列・mapfile・${var,,}・sed -i は使わない。macOS bash 3.2 での --self-test は Linux からは未検証（README.md）。
set -euo pipefail

# ---- 固定値（複製したら埋める。固定文字列のみ。外部入力を展開しない。__FILL__ や空のままだと --self-test 以外では開始前に exit 2） ----
NODE_KIND=cuda        # cuda | metal（マスク置換先・OS 検査・env_info・外部 timeout の有無が決まる）
PROFILE=dev          # dev | release（dev は cargo の test プロファイルで target/debug/deps。release は --release で target/release/deps）
PACKAGE=fandhe-ai          # 例: fandhe-ai
TEST_TARGET=train_gpu_parity      # 例: train_gpu_parity
TESTS=(train_parity_onnx_cuda_matches_cpu)          # 実行するテスト名（--exact）。1 本ずつ別プロセスで実行する
LOG_BASE=train_parity_onnx_cuda         # 回収ログ名の接頭辞（<LOG_BASE>_<番号>.log）
EXPECTED_SOURCE_FP=faf953ec518fde73cec4d3354e82941223306ffd2d2d16a419a36ff24e3f84c8   # RULE.txt の source_fingerprint
REQUIRED_LIBS=(fandhe_ai fandhe_ai_tensor_core fandhe_ai_autodiff fandhe_ai_backend_cpu fandhe_ai_backend_cuda fandhe_ai_backend_rocm fandhe_ai_onnx_interop bench_harness)  # 必ず存在すべき lib 名（ハイフンを _ にしたもの。例: fandhe_ai fandhe_ai_tensor_core）。facade を測るなら fandhe_ai を入れる
EXTERNAL_TIMEOUT=1800 # cuda のみ使う timeout(1) の値（例: 600）。metal は使わない（macOS に timeout(1) が無い）
# ロックの owner の形式（変更不要）。例: 「#4044 run 2026-11-05T01:02:03Z」
LOCK_OWNER_RE='^#[0-9]+\ [A-Za-z0-9._-]+\ [0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$'
# ---- 固定値ここまで ----

B='[^A-Za-z0-9_.-]' # ユーザー名の境界に使わない文字集合の否定形

# fixed_values_ok <kind> <profile> <package> <target> <fp> <ntests> <nlibs> <timeout>: 固定値が埋まっていれば 0（未記入・不正は 1）。
fixed_values_ok() {
  local kind=$1 profile=$2 package=$3 target=$4 fp=$5 nt=$6 nl=$7 to=$8
  case "$kind" in cuda | metal) ;; *) return 1 ;; esac
  case "$profile" in dev | release) ;; *) return 1 ;; esac
  [[ $package =~ ^[A-Za-z0-9_-]+$ && $package != __FILL__ ]] || return 1
  [[ $target =~ ^[A-Za-z0-9_]+$ && $target != __FILL__ ]] || return 1
  [[ $fp =~ ^[0-9a-f]{64}$ ]] || return 1
  [ "$nt" -ge 1 ] && [ "$nl" -ge 1 ] || return 1
  if [ "$kind" = cuda ]; then [[ $to =~ ^[0-9]+$ ]] || return 1; fi
  return 0
}
# all_filled: TESTS・REQUIRED_LIBS の各要素と LOG_BASE も埋まっているか。
all_filled() {
  local t l
  fixed_values_ok "$NODE_KIND" "$PROFILE" "$PACKAGE" "$TEST_TARGET" "$EXPECTED_SOURCE_FP" "${#TESTS[@]}" "${#REQUIRED_LIBS[@]}" "$EXTERNAL_TIMEOUT" || return 1
  [[ $LOG_BASE =~ ^[A-Za-z0-9_]+$ && $LOG_BASE != __FILL__ ]] || return 1
  for t in "${TESTS[@]}"; do [[ $t =~ ^[A-Za-z0-9_:@-]+$ && $t != __FILL__ ]] || return 1; done
  for l in "${REQUIRED_LIBS[@]}"; do [[ $l =~ ^[A-Za-z0-9_]+$ && $l != __FILL__ ]] || return 1; done
  return 0
}

# プロファイルから deps ディレクトリ名・clean／build の引数を決める。
deps_subdir() { case "$1" in release) echo release/deps ;; *) echo debug/deps ;; esac; }
profile_flag() { case "$1" in release) echo --release ;; *) echo "" ;; esac; }

# 正規表現（sed -E／grep -E 共通）のメタ文字と区切り文字 / をエスケープする。
esc_re() { printf '%s' "$1" | sed -e 's/[][\\.^$*+?(){}|/]/\\&/g'; }

# mask_stream <user> <host> <host-short> <home> <placeholder>: 標準入力をマスクして標準出力へ。
mask_stream() {
  local un=$1 hn=$2 hns=$3 home=$4 ph=$5 script=""
  if [ -n "$hn" ]; then script="${script}s/$(esc_re "$hn")/$ph/g"$'\n'; fi
  if [ -n "$hns" ]; then script="${script}s/$(esc_re "$hns")/$ph/g"$'\n'; fi
  if [ -n "$home" ] && [ "$home" != "/" ]; then script="${script}s/$(esc_re "$home")/<home>/g"$'\n'; fi
  local pat
  pat="(^|$B)$(esc_re "$un")($B|\$)"
  script="${script}s/${pat}/\\1<user>\\2/g"$'\n'"s/${pat}/\\1<user>\\2/g"
  sed -E -e "$script"
}

# leak_check <file> <user> <host> <host-short> <home> <personal-prefix>: 残存があれば 1、無ければ 0。
leak_check() {
  local f=$1 un=$2 hn=$3 hns=$4 home=$5 pre=$6 s
  for s in "$hn" "$hns" "$home" "$pre"; do
    if [ -n "$s" ] && [ "$s" != "/" ] && grep -Fq -- "$s" "$f"; then return 1; fi
  done
  if grep -Eq -- "(^|$B)$(esc_re "$un")($B|\$)" "$f"; then return 1; fi
  return 0
}

# マスクの不変条件。変えてはいけない行（crate 名・パッケージ名）がバイト単位で不変、置換すべき行が期待どおり、置換後に残存が無いこと。
mask_selfcheck() { # mask_selfcheck <user> <host> <host-short> <home> <placeholder> <personal-prefix>
  local un=$1 hn=$2 hns=$3 home=$4 ph=$5 pre=$6 d rc=0
  d=$(mktemp -d)
  cat >"$d/keep.in" <<'KEEP'
   Compiling fandhe-ai-tensor-core v0.13.0
Running unittests src/lib.rs (target/debug/deps/fandhe_ai-0123456789abcdef)
fandhe_ai_backend_metal::device
~/work/target-fandhe-ai/debug/deps/fandhe_ai_backend_metal-abc
repository: Fandhe-AI/fandhe-ai
KEEP
  : >"$d/chg.in"
  : >"$d/chg.want"
  printf '%s/x\n' "$home" >>"$d/chg.in"
  printf '<home>/x\n' >>"$d/chg.want"
  printf 'owner=%s\n' "$un" >>"$d/chg.in"
  printf 'owner=<user>\n' >>"$d/chg.want"
  printf '%s@%s\n' "$un" "$hn" >>"$d/chg.in"
  printf '<user>@%s\n' "$ph" >>"$d/chg.want"
  printf '%s %s\n' "$un" "$un" >>"$d/chg.in"
  printf '<user> <user>\n' >>"$d/chg.want"
  mask_stream "$un" "$hn" "$hns" "$home" "$ph" <"$d/keep.in" >"$d/keep.out"
  mask_stream "$un" "$hn" "$hns" "$home" "$ph" <"$d/chg.in" >"$d/chg.out"
  cmp -s "$d/keep.in" "$d/keep.out" || { echo "selfcheck: 変えてはいけない行が変化した" >&2; rc=1; }
  cmp -s "$d/chg.want" "$d/chg.out" || { echo "selfcheck: 置換結果が期待と異なる" >&2; rc=1; }
  leak_check "$d/chg.out" "$un" "$hn" "$hns" "$home" "$pre" || { echo "selfcheck: 置換後に残存がある" >&2; rc=1; }
  rm -rf "$d"
  return "$rc"
}

# source_fingerprint <tree>: crates/・Cargo.toml・Cargo.lock・rust-toolchain.toml の全ファイルの SHA-256 を名前順の一覧にし、さらに SHA-256 を取る。
sha256() { # sha256 <file> -> 64 桁
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi
}
source_fingerprint() {
  (cd "$1" && find crates Cargo.toml Cargo.lock rust-toolchain.toml -type f | LC_ALL=C sort | tr '\n' '\0' \
    | xargs -0 shasum -a 256 | shasum -a 256 | cut -d' ' -f1)
}
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

# check_rev_stamp <file>: 「40 桁小文字 16 進数\ndirty=0\n」（末尾改行は任意）だけなら 0 を返し、1 行目を REV_HEAD に入れる。
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
  local d rc=0 h=0123456789abcdef0123456789abcdef01234567 f
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

# mtime_epoch <file>: 更新時刻のエポック秒。mtime_utc <epoch>: 同じ時刻の UTC 表記。
mtime_epoch() {
  case "$(uname -s)" in
    Darwin) stat -f %m "$1" ;;
    *) stat -c %Y "$1" ;;
  esac
}
mtime_utc() {
  case "$(uname -s)" in
    Darwin) date -u -r "$1" +%Y-%m-%dT%H:%M:%SZ ;;
    *) date -u -d "@$1" +%Y-%m-%dT%H:%M:%SZ ;;
  esac
}

# members_and_libs: 標準入力の cargo metadata の JSON から、1 行ずつ「M <パッケージ名>」（全メンバー）と「L <lib 名>」（lib ターゲットを持つメンバーの lib 名。
# ハイフンは _）を出す。空、または名前の文字種が想定外なら失敗する（R4・R5。名前は cargo の引数やファイル名 glob に使うので allowlist で確かめる）。
members_and_libs() {
  python3 -c '
import json, re, sys
pk = json.load(sys.stdin)["packages"]
names = [p["name"] for p in pk]
libs = sorted({t["name"].replace("-", "_") for p in pk for t in p["targets"] if any(k in ("lib", "rlib") for k in t["kind"])})
if not names or not all(re.fullmatch(r"[A-Za-z0-9_-]+", n) for n in names + libs):
    sys.exit(1)
print("\n".join(["M " + n for n in names] + ["L " + l for l in libs]))
'
}

# test_bin_from_log <build-or-run-log> <Executable|Running> <test-target>: 該当行からバイナリのパスを出す。ちょうど 1 行でなければ何も出さない。
test_bin_from_log() {
  local lines
  lines=$(sed -n -E 's/^ *'"$2"' tests\/'"$3"'\.rs \((.*)\)$/\1/p' "$1")
  if [ -n "$lines" ] && [ "$(printf '%s\n' "$lines" | wc -l | tr -d ' ')" = "1" ]; then printf '%s\n' "$lines"; fi
}

# artifacts_report <deps-dir> <test-bin> <lock-epoch> <lib-names-file> <required-libs-file>: 成果物の一覧（TSV）を標準出力へ書く（R5）。
# テストバイナリと、メンバーの lib 名（<lib-names-file> の各行）ごとの lib<name>-<hash>.rlib・.rmeta の更新時刻がすべて lock-epoch より後（秒単位で厳密に後）なら 0。
# テストバイナリが無い、または <required-libs-file> の lib（.rlib）が 1 つも無ければ 1。
artifacts_report() {
  local deps=$1 bin=$2 lock=$3 libs=$4 req=$5 rc=0 list f m st l
  printf 'lock_mtime_epoch\t%s\n' "$lock"
  printf 'lock_mtime_utc\t%s\n' "$(mtime_utc "$lock")"
  printf 'mtime_epoch\tmtime_utc\tstatus\tpath\n'
  list=$(mktemp)
  if [ -z "$bin" ] || [ ! -f "$bin" ]; then
    printf -- '-\t-\tMISSING\ttest_binary:%s\n' "$bin"
    rc=1
  else
    printf '%s\n' "$bin" >>"$list"
  fi
  while IFS= read -r l; do
    [ -n "$l" ] || continue
    find "$deps" -maxdepth 1 -type f \( -name "lib$l-*.rlib" -o -name "lib$l-*.rmeta" \) >>"$list"
  done <"$libs"
  LC_ALL=C sort -u "$list" -o "$list"
  while IFS= read -r f; do
    m=$(mtime_epoch "$f")
    if [ "$m" -gt "$lock" ]; then
      st=after_lock
    else
      st=NOT_AFTER_LOCK
      rc=1
    fi
    printf '%s\t%s\t%s\t%s\n' "$m" "$(mtime_utc "$m")" "$st" "$f"
  done <"$list"
  while IFS= read -r l; do
    [ -n "$l" ] || continue
    if ! grep -Eq "/lib$l-[0-9a-f]+\.rlib\$" "$list"; then
      printf -- '-\t-\tMISSING\tlib%s\n' "$l"
      rc=1
    fi
  done <"$req"
  rm -f "$list"
  return "$rc"
}

# R4・R5 の不変条件。facade 自身の lib（libfandhe_ai-<hash>）が古い場合を含め、古い成果物・ロックと同じ秒・必須 lib の欠落・テストバイナリの欠落を拒否する。
build_check_selfcheck() {
  local d rc=0 lock n meta
  d=$(mktemp -d)
  mkdir -p "$d/deps"
  printf 'x\n' >"$d/lockfile"
  touch -t 202601010000.00 "$d/lockfile"
  lock=$(mtime_epoch "$d/lockfile")
  printf 'fandhe_ai\nfandhe_ai_tensor_core\nbench_harness\n' >"$d/libs"
  printf 'fandhe_ai\nfandhe_ai_tensor_core\n' >"$d/req"
  for n in libfandhe_ai libfandhe_ai_tensor_core libbench_harness; do
    touch -t 202601010001.00 "$d/deps/$n-0123456789abcdef.rlib" "$d/deps/$n-0123456789abcdef.rmeta"
  done
  touch -t 202601010001.00 "$d/deps/testbin"
  artifacts_report "$d/deps" "$d/deps/testbin" "$lock" "$d/libs" "$d/req" >/dev/null || { echo "selfcheck: 全部新しいのに拒否した" >&2; rc=1; }
  touch -t 202512312359.00 "$d/deps/libfandhe_ai-0123456789abcdef.rlib"
  if artifacts_report "$d/deps" "$d/deps/testbin" "$lock" "$d/libs" "$d/req" >/dev/null; then echo "selfcheck: 古い facade の rlib を受理した" >&2; rc=1; fi
  touch -t 202601010001.00 "$d/deps/libfandhe_ai-0123456789abcdef.rlib"
  touch -t 202512312359.00 "$d/deps/libfandhe_ai_tensor_core-0123456789abcdef.rmeta"
  if artifacts_report "$d/deps" "$d/deps/testbin" "$lock" "$d/libs" "$d/req" >/dev/null; then echo "selfcheck: 古い rmeta を受理した" >&2; rc=1; fi
  touch -t 202601010001.00 "$d/deps/libfandhe_ai_tensor_core-0123456789abcdef.rmeta"
  touch -t 202601010000.00 "$d/deps/libbench_harness-0123456789abcdef.rlib"
  if artifacts_report "$d/deps" "$d/deps/testbin" "$lock" "$d/libs" "$d/req" >/dev/null; then echo "selfcheck: ロックと同じ秒の rlib を受理した" >&2; rc=1; fi
  touch -t 202601010001.00 "$d/deps/libbench_harness-0123456789abcdef.rlib"
  touch -t 202512312359.00 "$d/deps/testbin"
  if artifacts_report "$d/deps" "$d/deps/testbin" "$lock" "$d/libs" "$d/req" >/dev/null; then echo "selfcheck: 古いテストバイナリを受理した" >&2; rc=1; fi
  touch -t 202601010001.00 "$d/deps/testbin"
  if artifacts_report "$d/deps" "" "$lock" "$d/libs" "$d/req" >/dev/null; then echo "selfcheck: テストバイナリなしを受理した" >&2; rc=1; fi
  rm -f "$d/deps/libfandhe_ai-0123456789abcdef.rlib" "$d/deps/libfandhe_ai-0123456789abcdef.rmeta"
  if artifacts_report "$d/deps" "$d/deps/testbin" "$lock" "$d/libs" "$d/req" >/dev/null; then echo "selfcheck: 必須 lib（facade）の欠落を受理した" >&2; rc=1; fi

  meta='{"packages":[{"name":"fandhe-ai","targets":[{"name":"fandhe_ai","kind":["lib"]},{"name":"t","kind":["test"]}]},{"name":"a-b","targets":[{"name":"a-b","kind":["bin"]}]},{"name":"bench-harness","targets":[{"name":"bench_harness","kind":["rlib"]}]}]}'
  [ "$(printf '%s' "$meta" | members_and_libs | tr '\n' ',')" = "M fandhe-ai,M a-b,M bench-harness,L bench_harness,L fandhe_ai," ] \
    || { echo "selfcheck: メンバーと lib 名を取り出せない" >&2; rc=1; }
  if printf '{"packages":[]}' | members_and_libs >/dev/null 2>&1; then echo "selfcheck: 空のメンバーを受理した" >&2; rc=1; fi
  if printf '{"packages":[{"name":"a b","targets":[]}]}' | members_and_libs >/dev/null 2>&1; then echo "selfcheck: 不正な名前を受理した" >&2; rc=1; fi

  printf '   Compiling x\n  Executable tests/tt.rs (/t/debug/deps/tt-ab)\n' >"$d/one"
  [ "$(test_bin_from_log "$d/one" Executable tt)" = "/t/debug/deps/tt-ab" ] || { echo "selfcheck: Executable 行を読めない" >&2; rc=1; }
  printf '  Executable tests/tt.rs (/a)\n  Executable tests/tt.rs (/b)\n' >"$d/two"
  [ -z "$(test_bin_from_log "$d/two" Executable tt)" ] || { echo "selfcheck: Executable が 2 行でも値を返した" >&2; rc=1; }
  printf '     Running tests/tt.rs (/t/x)\n' >"$d/run"
  [ "$(test_bin_from_log "$d/run" Running tt)" = "/t/x" ] || { echo "selfcheck: Running 行を読めない" >&2; rc=1; }
  [ "$(deps_subdir dev)" = "debug/deps" ] && [ "$(deps_subdir release)" = "release/deps" ] || { echo "selfcheck: deps ディレクトリの切り替えが誤り" >&2; rc=1; }
  [ -z "$(profile_flag dev)" ] && [ "$(profile_flag release)" = "--release" ] || { echo "selfcheck: プロファイル引数が誤り" >&2; rc=1; }
  rm -rf "$d"
  return "$rc"
}

# 固定値の検査自体の不変条件。未記入の雛形は拒否し、埋まった例は受理する。
fixed_selfcheck() {
  local rc=0 fp
  fp=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
  fixed_values_ok cuda dev fandhe-ai train_gpu_parity "$fp" 1 2 600 || { echo "selfcheck: 埋まった固定値を拒否した" >&2; rc=1; }
  fixed_values_ok metal release fandhe-ai x "$fp" 1 1 __FILL__ || { echo "selfcheck: metal は timeout 不要" >&2; rc=1; }
  fixed_values_ok __FILL__ dev fandhe-ai x "$fp" 1 1 600 && { echo "selfcheck: 未記入の NODE_KIND を受理した" >&2; rc=1; }
  fixed_values_ok cuda dev __FILL__ x "$fp" 1 1 600 && { echo "selfcheck: 未記入の PACKAGE を受理した" >&2; rc=1; }
  fixed_values_ok cuda dev fandhe-ai x __FILL__ 1 1 600 && { echo "selfcheck: 未記入の指紋を受理した" >&2; rc=1; }
  fixed_values_ok cuda dev fandhe-ai x "$fp" 1 0 600 && { echo "selfcheck: 空の REQUIRED_LIBS を受理した" >&2; rc=1; }
  fixed_values_ok cuda dev fandhe-ai x "$fp" 1 1 __FILL__ && { echo "selfcheck: cuda で timeout 未記入を受理した" >&2; rc=1; }
  return "$rc"
}

if [ "${1:-}" = "--self-test" ]; then
  if [ "$#" -ne 1 ]; then
    echo "usage: $0 --self-test" >&2
    exit 2
  fi
  if mask_selfcheck fandhe node-x node-x /home/fandhe '<cuda-node>' /home/ && mask_selfcheck fandhe node-x node-x /Users/fandhe '<metal-node>' /Users/ \
    && rev_stamp_selfcheck && fingerprint_selfcheck && build_check_selfcheck && fixed_selfcheck; then
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
if ! all_filled; then
  echo "固定値が未記入または不正（__FILL__ のまま・空・形式不正）。複製元の README.md の手順で埋める" >&2
  exit 2
fi
TREE=$1
OUT=$2
LOCK_OWNER=$3
case "$NODE_KIND" in
  cuda) PLACEHOLDER='<cuda-node>' PERSONAL_PREFIX=/home/ NEED_OS=Linux ;;
  *) PLACEHOLDER='<metal-node>' PERSONAL_PREFIX=/Users/ NEED_OS=Darwin ;;
esac
PROFILE_FLAG=$(profile_flag "$PROFILE")

# 実機ロックの引き継ぎ（R1）。以降の検証で止まっても trap で外す。
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
  if [ "$LOCK_HELD" -eq 1 ] && lock_is_mine; then rm -rf "$LOCK_DIR"; fi
}
trap cleanup EXIT
if ! [[ $LOCK_OWNER =~ $LOCK_OWNER_RE ]]; then
  echo "lock-owner の形式が不正" >&2
  exit 2
fi
if ! lock_is_mine; then
  echo "実機ロックの owner が一致しない。奪わず停止" >&2
  exit 2
fi
LOCK_HELD=1
# ロック取得の時刻（R5）。ノードの時計の値を使い、送信側の時計とのずれを受けない。
LOCK_EPOCH=$(mtime_epoch "$LOCK_DIR/owner")
[[ $LOCK_EPOCH =~ ^[0-9]+$ ]] || { echo "ロックの時刻を読めない" >&2; exit 2; }
# target は共有の 1 つだけ。これ以外では消さない・測らない（R4。誤った target の削除を防ぐ）。
if [ "${CARGO_TARGET_DIR:-}" != "$HOME/work/target-fandhe-ai" ]; then
  echo "CARGO_TARGET_DIR が想定と違う" >&2
  exit 2
fi
DEPS_DIR="$CARGO_TARGET_DIR/$(deps_subdir "$PROFILE")"

# 引数は絶対パス + 文字種 allowlist で検査する（A03）。改行を含む値は =~ の全体一致で拒否する。
for p in "$TREE" "$OUT"; do
  if ! [[ $p =~ ^/[A-Za-z0-9._/-]+$ ]]; then
    echo "絶対パスかつ許可文字のみが必要" >&2
    exit 2
  fi
done
if [ "$(uname -s)" != "$NEED_OS" ]; then
  echo "$NEED_OS 以外では実行しない（NODE_KIND=$NODE_KIND）" >&2
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
if ! mask_selfcheck "$UN" "$HN" "$HNS" "$HOME" "$PLACEHOLDER" "$PERSONAL_PREFIX"; then
  echo "実値でのマスク自己検査に失敗" >&2
  exit 2
fi

SELF=${BASH_SOURCE[0]}
SCRIPT_SHA=$(sha256 "$SELF")

# マスクしてから収録し、残っていれば fail-closed で止める。異常終了でも未処理の .raw が残らないよう EXIT trap で後始末する。
mask() { # mask <raw> <final>
  mask_stream "$UN" "$HN" "$HNS" "$HOME" "$PLACEHOLDER" <"$1" >"$2"
  rm -f "$1"
  if ! leak_check "$2" "$UN" "$HN" "$HNS" "$HOME" "$PERSONAL_PREFIX"; then
    echo "マスク漏れ: $2" >&2
    rm -f "$2"
    LEAK=1
  fi
}
mask_all() {
  for raw in "$OUT"/*.raw; do
    [ -e "$raw" ] || continue
    case "$raw" in
      "$OUT/env_info.raw") mask "$raw" "$OUT/env_info.txt" ;;
      "$OUT/build_artifacts.raw") mask "$raw" "$OUT/build_artifacts.txt" ;;
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
  if [ "$NODE_KIND" = metal ]; then
    echo "sw_vers: $( (sw_vers | tr '\n' ' ') 2>/dev/null || echo 取得不能)"
    echo "cpu: $( (sysctl -n machdep.cpu.brand_string) 2>/dev/null || echo 取得不能)"
    echo "memsize_bytes: $( (sysctl -n hw.memsize) 2>/dev/null || echo 取得不能)"
  else
    echo "gpu: $( (nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader) 2>/dev/null || echo 取得不能)"
    echo "kernel: $(uname -sr)"
  fi
  echo "rustc: $( (rustc -V) 2>/dev/null || echo 取得不能)"
  echo "cargo: $( (cargo -V) 2>/dev/null || echo 取得不能)"
  echo "arch: $(uname -m)"
  echo "profile: $PROFILE"
  echo "rev_stamp: $REV_HEAD"
  echo "script_sha256: $SCRIPT_SHA"
  if lock_is_mine; then echo "hw_lock: held owner=$LOCK_OWNER"; else echo "hw_lock: LOST"; fi
  echo "rev_stamp_now: $(sed -n 1p "$TREE/.rev-stamp" 2>/dev/null || echo 取得不能)"
  echo "script_sha256_now: $(sha256 "$SELF" 2>/dev/null || echo 取得不能)"
  echo "source_fingerprint_expected: $EXPECTED_SOURCE_FP"
  echo "source_fingerprint_now: $(source_fingerprint "$TREE" 2>/dev/null || echo 取得不能)"
}

# ツリーが開始時のままか（R2・R3）。
tree_intact() {
  cmp -s "$TREE/.rev-stamp" "$OUT/rev-stamp.txt" \
    && [ "$(sha256 "$SELF")" = "$SCRIPT_SHA" ] \
    && [ "$(source_fingerprint "$TREE")" = "$EXPECTED_SOURCE_FP" ]
}
env_info before >"$OUT/env_info.raw"

# 他の cargo／rustc が走っていれば並走させない。事前ビルド前は停止、各テスト前は undetermined。
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
# 全メンバーの成果物を消す（R4）。crates.io の依存は残す。失敗したら set -e で止まる。
META_OUT=$(cargo metadata --no-deps --format-version 1 | members_and_libs)
MEMBERS=$(printf '%s\n' "$META_OUT" | sed -n 's/^M //p')
LIBS_FILE=$(mktemp)
REQ_FILE=$(mktemp)
trap 'rm -f "$LIBS_FILE" "$REQ_FILE"; cleanup' EXIT
printf '%s\n' "$META_OUT" | sed -n 's/^L //p' >"$LIBS_FILE"
printf '%s\n' "${REQUIRED_LIBS[@]}" >"$REQ_FILE"
# REQUIRED_LIBS の綴り違いで検査が空振りしないよう、すべてメンバーの lib 名であることを確かめる。
while IFS= read -r l; do
  grep -Fxq -- "$l" "$LIBS_FILE" || { echo "REQUIRED_LIBS の $l が workspace メンバーの lib 名にない" >&2; exit 2; }
done <"$REQ_FILE"
{
  echo "profile: $PROFILE"
  echo "workspace_members:"
  printf '%s\n' "$MEMBERS"
} >"$OUT/clean.raw"
for name in $MEMBERS; do
  echo "== cargo clean $PROFILE_FLAG -p $name" >>"$OUT/clean.raw"
  # shellcheck disable=SC2086
  cargo clean $PROFILE_FLAG -p "$name" >>"$OUT/clean.raw" 2>&1
done

# ビルドを先に終え、計測中にコンパイルを並走させない（判定外）。失敗したら set -e で止まる。
# shellcheck disable=SC2086
cargo test -p "$PACKAGE" $PROFILE_FLAG --test "$TEST_TARGET" --no-run >"$OUT/build.raw" 2>&1

printf 'package\ttarget\ttest\texit_code\trunning\tresult\tverdict\n' >"$OUT/summary.tsv"
ANY_FAIL=0
ANY_UNDET=0
LOCK_LOST=0

# 選択の検証（--exact 付き。": test" 終わりの行がちょうど 1 行であること）を先に全テスト分行う。
i=0
for T in "${TESTS[@]}"; do
  i=$((i + 1))
  # shellcheck disable=SC2086
  cargo test -p "$PACKAGE" $PROFILE_FLAG --test "$TEST_TARGET" -- --ignored --exact --list "$T" >"$OUT/list_$i.raw" 2>&1 || true
done
# 成果物の確認（R5）。--list の再コンパイルも含めるため、ここで 1 回読む。
TEST_BIN=$(test_bin_from_log "$OUT/build.raw" Executable "$TEST_TARGET")
ART_OK=0
if artifacts_report "$DEPS_DIR" "$TEST_BIN" "$LOCK_EPOCH" "$LIBS_FILE" "$REQ_FILE" >"$OUT/build_artifacts.raw"; then
  ART_OK=1
fi
echo "build_artifacts_ok: $ART_OK" >>"$OUT/env_info.raw"

i=0
for T in "${TESTS[@]}"; do
  i=$((i + 1))
  RC=-
  RUNNING=-
  RESULT=-
  V=undetermined
  N=$(grep -c ': test$' "$OUT/list_$i.raw" || true)
  RAWLOG="$OUT/${LOG_BASE}_$i.raw"
  if ! lock_is_mine; then
    LOCK_LOST=1
    echo "実機ロックを失った: $T を実行しない" >&2
  elif other_build_running; then
    echo "他の cargo／rustc が動作中: $T を実行しない" >&2
  elif ! tree_intact; then
    echo "テスト前のツリーが開始時と一致しない: $T を実行しない" >&2
  elif [ "$N" -ne 1 ]; then
    echo "--list の選択件数が 1 でない: $T ($N)" >&2
  elif [ "$ART_OK" -ne 1 ]; then
    echo "成果物の確認に失敗（build_artifacts.txt）: $T を実行しない" >&2
  else
    set +e
    if [ "$NODE_KIND" = cuda ]; then
      # shellcheck disable=SC2086
      timeout "$EXTERNAL_TIMEOUT" cargo test -p "$PACKAGE" $PROFILE_FLAG --test "$TEST_TARGET" -- --ignored --exact --nocapture --test-threads=1 "$T" >"$RAWLOG" 2>&1
    else
      # shellcheck disable=SC2086
      cargo test -p "$PACKAGE" $PROFILE_FLAG --test "$TEST_TARGET" -- --ignored --exact --nocapture --test-threads=1 "$T" >"$RAWLOG" 2>&1
    fi
    RC=$?
    set -e
    echo "rc=$RC" >>"$RAWLOG"
    echo "$RC" >"$OUT/exit_code_$i"
    RUNNING=$(grep -c '^running 1 test$' "$RAWLOG" || true)
    RESULT=$(grep -E '^test result: ' "$RAWLOG" | head -n 1 || true)
    if ! lock_is_mine; then
      LOCK_LOST=1
    elif ! tree_intact; then
      echo "テスト後のツリーが開始時と一致しない" >&2
    elif [ "$(test_bin_from_log "$RAWLOG" Running "$TEST_TARGET")" != "$TEST_BIN" ]; then
      echo "実行したテストバイナリが確かめたものと違う" >&2
    elif [ "$RC" -eq 124 ]; then
      echo "外部 timeout: $T" >&2
    elif [ "$RC" -eq 0 ] && [ "$RUNNING" -eq 1 ] && [[ $RESULT == "test result: ok. 1 passed; 0 failed"* ]]; then
      V=pass
    elif [ "$RC" -eq 0 ]; then
      V=undetermined
    else
      V=fail
    fi
  fi
  case "$V" in
    fail) ANY_FAIL=1 ;;
    undetermined) ANY_UNDET=1 ;;
  esac
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$PACKAGE" "$TEST_TARGET" "$T" "$RC" "$RUNNING" "${RESULT:-none}" "$V" >>"$OUT/summary.tsv"
done

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
