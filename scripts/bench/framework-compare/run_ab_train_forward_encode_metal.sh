#!/usr/bin/env bash
# イシュー #2113（親 #1980 §17.4 施策 2）: Metal train forward の encode-only 合流
# （`crates/backend-metal/src/train_forward_encode_runtime.rs` の
# `TRAIN_FORWARD_ENCODE_ONLY_DEFAULT_ENABLED`）の framework-compare train／infer A/B。
# before = main の `crates/facade`、after = 同一コミットで上記定数だけを `true` に
# した計測専用 worktree の `crates/facade` を、それぞれ `[patch.crates-io.fandhe-ai]`
# path patch した 2 本の `bench-fandhe` としてビルドし、5 round・round 単位で
# 起動順を反転しながら交互実行する（`run_ab_tape_arena_cpu.sh`〈#2104〉の派生。ビルド・
# lock 復元・path patch 解決検証・load ゲートは同一）。判定対象は train {fresh, reuse}
# （主対象は metal × train reuse）。infer は `Sequential::predict_resident` が
# `predict_device_chain` を優先し本オーバーライドへ到達しないため record_only。
# AB_DEVICE: macOS の既定は metal（M4 Max）。GB10 は cpu／cuda を A/A 相当の対照として
# 実行する（Metal コードは Linux でコンパイルされない。RULE.txt 参照）。
# 定数を反転した worktree は計測専用であり main へはコミットしない（既定切替は
# 両機体 ADOPT 後の別 PR。`docs/perf/logs/metal-train-forward-encodeonly-2113/RULE.txt`）。
#
# 呼び出し例:
#   AB_BEFORE_FACADE_PATH=/path/to/before/crates/facade \
#   AB_AFTER_FACADE_PATH=/path/to/after/crates/facade \
#     bash run_ab_train_forward_encode_metal.sh 2113
set -u
cd "$(dirname "$0")" || exit 1
# shellcheck source=./bench_fandhe_lock_restore.sh
source ./bench_fandhe_lock_restore.sh

LABEL=${1:-}
# RULE.txt は「5 round 中央値・5/5 round 一貫性」を事前登録しており、
# compare_gemm_ab.py も各セルちょうど 5 件を要求する。round 数は 5 固定とし、
# AB_ROUNDS で 5 以外が指定された場合は fail-closed で拒否する（PR #2448 codex P1 指摘）。
ROUNDS=5
if [[ -n "${AB_ROUNDS:-}" && "${AB_ROUNDS}" != "5" ]]; then
  echo "error: AB_ROUNDS must be 5 (RULE.txt の事前登録は 5 round 固定。got: ${AB_ROUNDS})" >&2
  exit 1
fi

# A03 インジェクション対策: ラベルはファイル名・パスに直接埋め込むため、
# 英数字・`._-` のみを許可する allowlist で検証する。
if [[ -z "$LABEL" || ! "$LABEL" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "usage: $0 <label>  (label must match [A-Za-z0-9._-]+, e.g. 2113)" >&2
  echo "  env AB_BEFORE_FACADE_PATH / AB_AFTER_FACADE_PATH (absolute paths) are required" >&2
  exit 1
fi

validate_facade_path() {
  local var_name=$1 path=$2
  if [[ -z "$path" ]]; then
    echo "error: $var_name is required (absolute path to a crates/facade checkout; issue #2113)" >&2
    exit 1
  fi
  if [[ "$path" != /* ]]; then
    echo "error: $var_name must be an absolute path (got: $path)" >&2
    exit 1
  fi
  if [[ ! -f "$path/Cargo.toml" ]]; then
    echo "error: $var_name/Cargo.toml not found ($path)" >&2
    exit 1
  fi
  if ! grep -qE '^\s*name\s*=\s*"fandhe-ai"\s*$' "$path/Cargo.toml"; then
    echo "error: $var_name/Cargo.toml does not declare name = \"fandhe-ai\" ($path)" >&2
    exit 1
  fi
}
validate_facade_path AB_BEFORE_FACADE_PATH "${AB_BEFORE_FACADE_PATH:-}"
validate_facade_path AB_AFTER_FACADE_PATH "${AB_AFTER_FACADE_PATH:-}"
BEFORE_FACADE="$AB_BEFORE_FACADE_PATH"
AFTER_FACADE="$AB_AFTER_FACADE_PATH"

# A/B 両 checkout の `TRAIN_FORWARD_ENCODE_ONLY_DEFAULT_ENABLED` を計測前に検証する（fail-closed）。before=false・after=true 以外（同一 checkout・反転漏れ・
# 定数の取得不能）は比較として無効なため、ビルド前に停止する。
read_flag_default() { # read_flag_default <facade_path> -> true|false|(空)
  local alloc="$1/../backend-metal/src/train_forward_encode_runtime.rs"
  [[ -f "$alloc" ]] || return 0
  sed -nE 's/^[[:space:]]*pub(\(crate\))?[[:space:]]+const[[:space:]]+TRAIN_FORWARD_ENCODE_ONLY_DEFAULT_ENABLED[[:space:]]*:[[:space:]]*bool[[:space:]]*=[[:space:]]*(true|false)[[:space:]]*;.*/\2/p' "$alloc" | head -n1
}
BEFORE_FLAG_DEFAULT="$(read_flag_default "$BEFORE_FACADE")"
AFTER_FLAG_DEFAULT="$(read_flag_default "$AFTER_FACADE")"
if [[ "$BEFORE_FLAG_DEFAULT" != "false" || "$AFTER_FLAG_DEFAULT" != "true" ]]; then
  echo "error: TRAIN_FORWARD_ENCODE_ONLY_DEFAULT_ENABLED mismatch (before=${BEFORE_FLAG_DEFAULT:-unreadable}, after=${AFTER_FLAG_DEFAULT:-unreadable}); expected before=false / after=true" >&2
  exit 1
fi

# A/B 両 checkout が「同一コミット・after は定数 1 行のみ差分」であることをビルド前に
# 検証する（fail-closed。PR #2448 codex 指摘）。別コミット・他変更を含む checkout での
# 計測を防ぐ。
verify_no_untracked_build_inputs() { # verify_no_untracked_build_inputs <arm> <root>
  local arm=$1 root=$2 untracked ignored_hits
  # 未追跡（gitignore 対象外）はビルド影響の有無を問わず 1 件でもあれば停止する。
  # `ls-files --others` は個別ファイル単位で列挙する。
  untracked="$(git -C "$root" ls-files --others --exclude-standard)" || {
    echo "error: $arm checkout の未追跡ファイル一覧を取得できません" >&2
    exit 1
  }
  if [[ -n "$untracked" ]]; then
    echo "error: $arm checkout に未追跡ファイルがあります（ビルド影響の恐れ）:" >&2
    printf '%s\n' "$untracked" | head -20 >&2
    exit 1
  fi
  # gitignore 済みでもビルド結果を変えうるファイル（cargo 設定・toolchain 指定・
  # ソース・マニフェスト）は停止対象。`target*` 等の成果物ディレクトリは対象外。
  ignored_hits="$(git -C "$root" ls-files --others --ignored --exclude-standard)" || {
    echo "error: $arm checkout の無視済みファイル一覧を取得できません" >&2
    exit 1
  }
  ignored_hits="$(printf '%s\n' "$ignored_hits" | grep -vE '^(target[^/]*|scripts/bench/[^ ]*/target[^/]*)/' |
    grep -E '(^|/)\.cargo/|(^|/)rust-toolchain(\.toml)?$|(^|/)Cargo\.(toml|lock)$|(^|/)build\.rs$|\.rs$' || true)"
  if [[ -n "$ignored_hits" ]]; then
    echo "error: $arm checkout に gitignore 済みのビルド影響ファイルがあります:" >&2
    printf '%s\n' "$ignored_hits" | head -20 >&2
    exit 1
  fi
}

verify_same_commit_and_diff() {
  local before_root after_root bh ah names changed
  before_root="$(git -C "$BEFORE_FACADE" rev-parse --show-toplevel 2>/dev/null)" || {
    echo "error: before checkout is not a git worktree ($BEFORE_FACADE)" >&2
    exit 1
  }
  after_root="$(git -C "$AFTER_FACADE" rev-parse --show-toplevel 2>/dev/null)" || {
    echo "error: after checkout is not a git worktree ($AFTER_FACADE)" >&2
    exit 1
  }
  bh="$(git -C "$before_root" rev-parse HEAD)" || exit 1
  ah="$(git -C "$after_root" rev-parse HEAD)" || exit 1
  if [[ -z "$bh" || "$bh" != "$ah" ]]; then
    echo "error: before/after checkout の HEAD コミットが一致しません (before=$bh after=$ah)" >&2
    exit 1
  fi
  if [[ -n "$(git -C "$before_root" status --porcelain --untracked-files=no)" ]]; then
    echo "error: before checkout に未コミット変更があります" >&2
    exit 1
  fi
  # 未追跡・gitignore 済みのビルド影響ファイル（`.cargo/config.toml`・
  # `rust-toolchain*`・追加 `.rs`／`Cargo.*`／`build.rs` 等）は tracked diff で検出できず、
  # 「同一コミットで定数 1 行のみ差」の事前宣言を破るため、両 checkout で検出して停止する
  # （fail-closed。PR #2448 codex P1 指摘）。
  verify_no_untracked_build_inputs "before" "$before_root"
  verify_no_untracked_build_inputs "after" "$after_root"
  names="$(git -C "$after_root" diff --name-only HEAD)"
  if [[ "$names" != "crates/backend-metal/src/train_forward_encode_runtime.rs" ]]; then
    echo "error: after checkout の差分が train_forward_encode_runtime.rs のみではありません: ${names:-(空)}" >&2
    exit 1
  fi
  changed="$(git -C "$after_root" diff -U0 HEAD -- crates/backend-metal/src/train_forward_encode_runtime.rs | grep -E '^[-+][^-+]' || true)"
  if [[ "$(printf '%s\n' "$changed" | wc -l | tr -d ' ')" != "2" ]] ||
    ! printf '%s\n' "$changed" | grep -qE '^-.*TRAIN_FORWARD_ENCODE_ONLY_DEFAULT_ENABLED.*= false;' ||
    ! printf '%s\n' "$changed" | grep -qE '^\+.*TRAIN_FORWARD_ENCODE_ONLY_DEFAULT_ENABLED.*= true;'; then
    echo "error: after checkout の差分が TRAIN_FORWARD_ENCODE_ONLY_DEFAULT_ENABLED の false→true 1 行のみではありません" >&2
    exit 1
  fi
}
verify_same_commit_and_diff

# 判定対象デバイスは allowlist（metal／cpu／cuda）。それ以外は fail-closed で拒否する。
# metal は macOS 限定（bench-fandhe が MEASURE_ERROR で拒否する）。
if [[ "$(uname -s)" == "Darwin" ]]; then DEFAULT_DEVICE=metal; else DEFAULT_DEVICE=cpu; fi
DEVICE=${AB_DEVICE:-$DEFAULT_DEVICE}
case "$DEVICE" in
  metal | cpu | cuda) ;;
  *)
    echo "error: AB_DEVICE must be one of metal|cpu|cuda (got: $DEVICE)" >&2
    exit 1
    ;;
esac

OUT="results/raw"
mkdir -p "$OUT"
SKIP="$OUT/skipped-2113-${DEVICE}-${LABEL}.log"
ANY_FAILED=0
# infer は RULE.txt で record_only（判定に使わない）。計測不備は記録するが train 判定の
# 成否（ANY_FAILED）とは分離し、INFER_ISSUES へ計上する（PR #2457 codex P2 指摘）。
INFER_ISSUES=0
record_failure() { # record_failure <task>
  if [[ "$1" == "infer" ]]; then
    INFER_ISSUES=$((INFER_ISSUES + 1))
  else
    ANY_FAILED=$((ANY_FAILED + 1))
  fi
}

# 同じ LABEL の既存系列は消去しない（RULE.txt: run の差し替え・追加起動はしない）。
# 既存 JSONL を検出したら書き込み前に終了し、再実行は別ラベルで行わせる
# （R4 の orchestrate_m4max.sh と同じ fail-closed）。
for _reset_arm in before after; do
  for _reset_suffix in train infer; do
    _existing="$OUT/results-${_reset_arm}-${LABEL}-${DEVICE}-${_reset_suffix}.jsonl"
    if [[ -e "$_existing" ]]; then
      echo "error: 既存の計測記録 $_existing があります。差し替え禁止のため別の LABEL で実行してください" >&2
      exit 1
    fi
  done
done
if [[ -e "$SKIP" ]]; then
  echo "error: 既存の失敗記録 $SKIP があります。差し替え禁止のため別の LABEL で実行してください" >&2
  exit 1
fi
: >"$SKIP"
for _reset_arm in before after; do
  for _reset_suffix in train infer; do
    : >"$OUT/results-${_reset_arm}-${LABEL}-${DEVICE}-${_reset_suffix}.jsonl"
  done
done

bench_fandhe_setup_lock_restore_trap

build_arm() { # build_arm <arm> <facade_path> <target_dir>
  local arm=$1 facade=$2 tdir=$3 patch_config exe msg
  patch_config="patch.crates-io.fandhe-ai.path=\"${facade}\""
  msg="$(mktemp)"
  if ! cargo build --release -p bench-fandhe --target-dir "$tdir" --message-format=json \
    --config "$patch_config" >"$msg" 2>"$OUT/build-${arm}-${LABEL}-${DEVICE}.err"; then
    tail -40 "$OUT/build-${arm}-${LABEL}-${DEVICE}.err"
    echo "bench-fandhe BUILD FAILED ($arm): $(tail -3 "$OUT/build-${arm}-${LABEL}-${DEVICE}.err" | tr '\n' ' ')" >>"$SKIP"
    rm -f "$msg"
    exit 1
  fi
  exe="$(jq -rs '[.[] | select(.reason == "compiler-artifact" and .target.name == "bench-fandhe" and (.target.kind[]? == "bin") and .executable != null)] | last | .executable // empty' "$msg")"
  rm -f "$msg"
  if [[ -z "$exe" || ! -f "$exe" ]]; then
    echo "error: build $arm: exe not found" >&2
    exit 1
  fi
  cp "$exe" "$OUT/bench-fandhe-2113-${arm}-${LABEL}"
  local tree_output
  tree_output="$(cargo tree -p bench-fandhe --depth 1 --config "$patch_config" 2>&1)"
  if ! echo "$tree_output" | grep -qE 'fandhe-ai v[0-9.]+ \(.*crates/facade\)'; then
    echo "error: fandhe-ai did not resolve to the path-patched crates/facade ($arm); cargo tree:" >&2
    echo "$tree_output" >&2
    exit 1
  fi
  echo "$tree_output" >"$OUT/tree-2113-${arm}-${LABEL}.txt"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$OUT/bench-fandhe-2113-${arm}-${LABEL}" >"$OUT/sha-2113-${arm}-${LABEL}.txt"
  else
    shasum -a 256 "$OUT/bench-fandhe-2113-${arm}-${LABEL}" >"$OUT/sha-2113-${arm}-${LABEL}.txt"
  fi
}

echo "== build before ==(facade=$BEFORE_FACADE)"
build_arm before "$BEFORE_FACADE" "target-ab-2113-${LABEL}-before"
echo "== build after  ==(facade=$AFTER_FACADE)"
build_arm after "$AFTER_FACADE" "target-ab-2113-${LABEL}-after"
cat "$OUT"/tree-2113-*-"${LABEL}".txt

BIN_BEFORE="$OUT/bench-fandhe-2113-before-${LABEL}"
BIN_AFTER="$OUT/bench-fandhe-2113-after-${LABEL}"

# セル（arm × task × mode）の JSONL 行数を数える。読み取り不能は空文字（呼び出し側が
# 不足扱いにする）。round 対応（各 round で各セル +1 行）の検証と、比較前の
# 5 件検証で共用する（PR #2448 codex P1 指摘）。
count_cell_rows() { # count_cell_rows <arm> <task> <mode> -> 行数 | (空)
  local f="$OUT/results-${1}-${LABEL}-${DEVICE}-${2}.jsonl"
  [[ -f "$f" ]] || return 0
  jq -rs --arg m "$3" '[.[] | select((.mode // "fresh") == $m)] | length' "$f" 2>/dev/null || true
}

run_cell() { # run_cell <arm> <task> <mode> [size]
  local arm=$1 task=$2 mode=$3 size=${4:-64} bin before_n after_n
  if [[ "$arm" == "before" ]]; then bin="$BIN_BEFORE"; else bin="$BIN_AFTER"; fi
  echo "== $task $DEVICE size=$size mode=$mode arm=$arm =="
  before_n="$(count_cell_rows "$arm" "$task" "$mode")"
  if ! "$bin" --task "$task" --size "$size" --device "$DEVICE" --mode "$mode" \
    --out "$OUT/results-${arm}-${LABEL}-${DEVICE}-${task}.jsonl" 2>"$OUT/err-${arm}-${LABEL}-${DEVICE}.tmp"; then
    echo "arm=$arm task=$task size=$size mode=$mode : $(cat "$OUT/err-${arm}-${LABEL}-${DEVICE}.tmp")" >>"$SKIP"
    echo "  -> FAILED (recorded in $SKIP)"
    record_failure "$task"
  else
    # 成功終了でも当該 round のセルが JSONL へ ちょうど 1 行追加されたことを確認する
    # （追記漏れ・二重追記は round 対応が崩れるため fail-closed）。
    after_n="$(count_cell_rows "$arm" "$task" "$mode")"
    if [[ ! "$before_n" =~ ^[0-9]+$ || ! "$after_n" =~ ^[0-9]+$ || "$after_n" -ne $((before_n + 1)) ]]; then
      echo "arm=$arm task=$task size=$size mode=$mode : round の記録行数が不正（before=${before_n:-NA} after=${after_n:-NA}。+1 行を期待）" >>"$SKIP"
      echo "  -> ROW COUNT MISMATCH (recorded in $SKIP)"
      record_failure "$task"
    fi
  fi
  rm -f "$OUT/err-${arm}-${LABEL}-${DEVICE}.tmp"
}

run_pair() { # run_pair <round> <task> <mode> [size]
  local run_i=$1
  shift
  if ((run_i % 2 == 1)); then
    run_cell before "$@"
    run_cell after "$@"
  else
    run_cell after "$@"
    run_cell before "$@"
  fi
}

LOAD_GATE=${AB_LOAD_GATE:-8.0}
# RULE.txt が正式系列の条件とする閾値は 8.0 固定。AB_LOAD_GATE で緩和した系列は
# 「通過」と記録せず、適用閾値とともに参考系列（reference）としてログへ明記する
# （PR #2016 codex-review 指摘。事前登録規則とログの通過判定を一致させる）。
RULE_LOAD_GATE=8.0
if [[ ! "$LOAD_GATE" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
  echo "error: AB_LOAD_GATE must be numeric (got: $LOAD_GATE)" >&2
  exit 1
fi
GATE_SERIES=official
if [[ "$(awk -v a="$LOAD_GATE" -v r="$RULE_LOAD_GATE" 'BEGIN{print (a==r)?1:0}')" != "1" ]]; then
  GATE_SERIES=reference
fi
GATE_LOG="$OUT/load-gate-2113-${DEVICE}-${LABEL}.log"
: >"$GATE_LOG"
echo "threshold=${LOAD_GATE} rule_threshold=${RULE_LOAD_GATE} series=${GATE_SERIES}" >>"$GATE_LOG"
wait_load_gate() { # wait_load_gate <round>
  local waited=0 status=timeout l1 ok
  while ((waited < 1800)); do
    l1=$(sysctl -n vm.loadavg 2>/dev/null | awk '{print $2}')
    [[ -z "$l1" && -r /proc/loadavg ]] && l1=$(awk '{print $1}' /proc/loadavg)
    # 負荷を取得できない（空・非数値）場合は gate=pass に化けさせず unavailable として
    # 記録する（実行は継続し系列は参考扱い。RULE.txt の不通過時と同じ扱い）。
    if [[ -z "$l1" || "$l1" =~ [^0-9.] ]]; then
      status=unavailable
      l1=NA
      break
    fi
    ok=$(awk -v a="$l1" -v t="$LOAD_GATE" 'BEGIN{print (a<t)?1:0}')
    if [[ "$ok" == "1" ]]; then
      status=pass
      break
    fi
    sleep 30
    waited=$((waited + 30))
  done
  # 緩和閾値で通過した round は pass ではなく pass-reference（参考扱い）として記録する
  if [[ "$status" == pass && "$GATE_SERIES" == reference ]]; then status=pass-reference; fi
  echo "round${1} gate=${status} threshold=${LOAD_GATE} load1=${l1} waited_s=${waited} at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$GATE_LOG"
}

: >"$OUT/uptime-2113-${DEVICE}-${LABEL}.log"
uptime | sed 's/.*load/load/' >>"$OUT/uptime-2113-${DEVICE}-${LABEL}.log"

for run_i in $(seq 1 "$ROUNDS"); do
  wait_load_gate "$run_i"
  for task in train infer; do
    for mode in fresh reuse; do
      run_pair "$run_i" "$task" "$mode"
    done
  done
  echo "round $run_i: $(uptime | sed 's/.*load/load/')" | tee -a "$OUT/uptime-2113-${DEVICE}-${LABEL}.log"
done

# 比較レポート・エラーログの名前にも LABEL を含め、別 LABEL の再実行が前系列の
# レポートを上書きしないようにする（JSONL・終了コードログと同じ系列別保持。
# PR #2016 codex-review 指摘）
# 比較前に全セル（arm × task × mode）がちょうど ROUNDS 件であることを検証する。
# 不足・過剰があれば当該 task は判定不能として比較へ進まず非ゼロ終了する
# （少標本の中央値で非後退判定が出るのを防ぐ。PR #2448 codex P1 指摘）。
for task in train infer; do
  CELLS_OK=1
  for arm in before after; do
    for mode in fresh reuse; do
      n="$(count_cell_rows "$arm" "$task" "$mode")"
      if [[ ! "$n" =~ ^[0-9]+$ || "$n" -ne "$ROUNDS" ]]; then
        echo "compare task=$task: 判定不能（arm=$arm mode=$mode の記録は ${n:-NA} 件。${ROUNDS} 件を要求）" >>"$SKIP"
        CELLS_OK=0
      fi
    done
  done
  if [[ "$CELLS_OK" != "1" ]]; then
    echo "compare task=$task exit=NA (判定不能: 標本数不足)" | tee -a "$OUT/compare-exit-2113-${DEVICE}-${LABEL}.log"
    record_failure "$task"
    continue
  fi
  python3 compare_gemm_ab.py --device "$DEVICE" --task "$task" --threshold 1.00 --per-run \
    --require-checksum-exact \
    "$OUT/results-before-${LABEL}-${DEVICE}-${task}.jsonl" "$OUT/results-after-${LABEL}-${DEVICE}-${task}.jsonl" \
    >"compare-${task}-2113-${DEVICE}-${LABEL}.md" 2>"compare-${task}-2113-${DEVICE}-${LABEL}.err"
  COMPARE_EXIT=$?
  # compare_gemm_ab.py の終了コード: 0 = 非後退・3 = 後退セルあり（--require-checksum-exact
  # 指定時は checksum 不一致・判定不能セルも 3 になる）・2 = 入力不正・空データ、それ以外
  # （python 起動失敗等）は比較処理自体の失敗。RULE.txt「checksum 不一致は FAIL として停止」
  # に従い、train の 3 は checksum 非完全一致・判定不能を検出したら非ゼロ終了へ伝播する
  # （純粋な速度後退は判定結果として記録のみ）。infer は record_only のため常に記録のみ
  # で train 判定へ影響させない（PR #2457 codex P1/P2 指摘）。
  echo "compare task=$task exit=$COMPARE_EXIT" | tee -a "$OUT/compare-exit-2113-${DEVICE}-${LABEL}.log"
  case "$COMPARE_EXIT" in
    0) ;;
    3)
      if [[ "$task" != "infer" ]] &&
        { grep -qE '不一致|複合判定 ok|判定不能' "compare-${task}-2113-${DEVICE}-${LABEL}.md" ||
          grep -q 'checksum_exact_match=False' "compare-${task}-2113-${DEVICE}-${LABEL}.err"; }; then
        echo "compare task=$task: checksum 不一致または判定不能（exit=3。RULE.txt により FAIL）" >>"$SKIP"
        ANY_FAILED=$((ANY_FAILED + 1))
      fi
      ;;
    *)
      echo "compare task=$task: 比較不能（exit=$COMPARE_EXIT）。$(tail -3 "compare-${task}-2113-${DEVICE}-${LABEL}.err" | tr '\n' ' ')" >>"$SKIP"
      record_failure "$task"
      ;;
  esac
done

echo "done. results in $OUT ; failures (if any) in $SKIP"
if [[ "$INFER_ISSUES" -gt 0 ]]; then
  echo "note: infer(record_only) に計測不備 ${INFER_ISSUES} 件（train 判定とは分離。詳細は $SKIP）" >&2
fi
if [[ "$ANY_FAILED" -gt 0 ]]; then
  echo "FAILED: $ANY_FAILED run(s) failed; see $SKIP" >&2
  exit 1
fi
