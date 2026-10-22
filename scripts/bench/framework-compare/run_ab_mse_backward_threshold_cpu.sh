#!/usr/bin/env bash
# イシュー #3195（Phase 10）: MSE backward の逐次しきい値
# （`crates/backend-cpu/src/mse.rs` の `MSE_BACKWARD_PARALLEL_MIN_ELEMS`）を
# T = 4096 で再測定する framework-compare cpu A/B（train × fresh/reuse を判定、
# `--phases` 系列は記録のみ）。#1578（T = 1<<18・REJECT 確定）の判定・記録は変えない。
# before = main の `crates/facade`（定数 0）、after = 同一コミットで定数 1 行だけを
# AB_THRESHOLD_T に変えた計測専用 worktree の `crates/facade`。それぞれ
# `[patch.crates-io.fandhe-ai]` path patch した 2 本の `bench-fandhe` を 5 round・
# round 単位で起動順を反転しながら交互実行する（`run_ab_reduction_threshold_cpu.sh`
# 〈#2102〉の派生。ビルド・lock 復元・path patch 解決検証・負荷ゲートの骨格は同一）。
# 判定規則（実測前に固定）: `docs/perf/logs/cpu-mse-backward-threshold-3195/RULE.txt`。
#
# 呼び出し例（AB_HOST_KIND は macmini|gb10 必須。負荷閾値を 3.0／1.0 に固定する）:
#   AB_BEFORE_FACADE_PATH=/path/to/before/crates/facade \
#   AB_AFTER_FACADE_PATH=/path/to/after/crates/facade \
#   AB_THRESHOLD_T=4096 AB_HOST_KIND=macmini AB_ROUND_COOLDOWN_MAX_LOAD1=3.0 \
#     bash run_ab_mse_backward_threshold_cpu.sh 3195-macmini
set -u
cd "$(dirname "$0")" || exit 1
# shellcheck source=./bench_fandhe_lock_restore.sh
source ./bench_fandhe_lock_restore.sh
# イシュー #3184（再開分）: round 前クールダウン hook（既定 OFF。AB_ROUND_COOLDOWN_MAX_LOAD1 設定時のみ待機・記録。
# 継承の閾値・gate_ok・series 判定は不変。docs/perf/logs/optin-ab-macmini-3184/RULE-reopen.txt）
source ./ab_round_cooldown.sh

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
  echo "usage: $0 <label>  (label must match [A-Za-z0-9._-]+, e.g. 3195-macmini)" >&2
  echo "  env AB_BEFORE_FACADE_PATH / AB_AFTER_FACADE_PATH (absolute paths) and AB_THRESHOLD_T (decimal, T must be exactly 4096 per RULE.txt) and AB_HOST_KIND (macmini|gb10) are required" >&2
  exit 1
fi

validate_facade_path() {
  local var_name=$1 path=$2
  if [[ -z "$path" ]]; then
    echo "error: $var_name is required (absolute path to a crates/facade checkout; issue #3195)" >&2
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

# after 側のしきい値 T（RULE.txt: 単一候補 T = 4096 に事前固定）。10 進整数のみ受理し、
# 4096 以外は拒否する（fail-closed）。#1578 で REJECT 済みの 262144 を含め、候補の
# 差し替え・再選定による再判定を防ぐ（640 < T <= 262144 の範囲検査も兼ねる）。
AB_THRESHOLD_T=${AB_THRESHOLD_T:-}
if [[ ! "$AB_THRESHOLD_T" =~ ^[0-9]+$ ]] || ((10#$AB_THRESHOLD_T <= 640 || 10#$AB_THRESHOLD_T >= 262144)); then
  echo "error: AB_THRESHOLD_T must be a decimal integer with 640 < T < 262144 (262144 は #1578 で REJECT 済み。got: ${AB_THRESHOLD_T:-unset})" >&2
  exit 1
fi
AB_THRESHOLD_T=$((10#$AB_THRESHOLD_T))
if [[ "$AB_THRESHOLD_T" != "4096" ]]; then
  echo "error: AB_THRESHOLD_T must be 4096 (RULE.txt の単一候補。got: ${AB_THRESHOLD_T})" >&2
  exit 1
fi

# ホスト種別（RULE.txt: 正式系列の負荷閾値は Mac mini 3.0・GB10 1.0 に固定）。
AB_HOST_KIND=${AB_HOST_KIND:-}
case "$AB_HOST_KIND" in
  macmini) RULE_LOAD_GATE=3.0 ;;
  gb10) RULE_LOAD_GATE=1.0 ;;
  *)
    echo "error: AB_HOST_KIND must be macmini or gb10 (got: ${AB_HOST_KIND:-unset})" >&2
    exit 1
    ;;
esac

# A/B 両 checkout の定数を計測前に検証する（fail-closed）。before は 0（main 既定）、
# after は T。読み取りはアンカー付き正規表現のみで行い、ソースは実行しない。
# 宣言が 1 行形式でない・取得不能・不一致はビルド前に停止する。
mse_rs() { echo "$1/../backend-cpu/src/mse.rs"; }
read_min_elems() { # read_min_elems <facade_path> -> 10 進整数 | (空)。`1 << k` 形式も受理
  local f raw
  f="$(mse_rs "$1")"
  [[ -f "$f" ]] || return 0
  raw="$(sed -nE 's/^[[:space:]]*pub(\(crate\))?[[:space:]]+const[[:space:]]+MSE_BACKWARD_PARALLEL_MIN_ELEMS[[:space:]]*:[[:space:]]*usize[[:space:]]*=[[:space:]]*([0-9_]+|1[[:space:]]*<<[[:space:]]*[0-9]+)[[:space:]]*;.*/\2/p' "$f" | head -n1)"
  raw="${raw//_/}"
  if [[ "$raw" =~ ^1[[:space:]]*\<\<[[:space:]]*([0-9]+)$ ]]; then
    ((BASH_REMATCH[1] < 63)) && echo $((1 << BASH_REMATCH[1]))
  elif [[ "$raw" =~ ^[0-9]+$ ]]; then
    echo "$raw"
  fi
}
BEFORE_MIN="$(read_min_elems "$BEFORE_FACADE")"
AFTER_MIN="$(read_min_elems "$AFTER_FACADE")"
if [[ "$BEFORE_MIN" != "0" ]]; then
  echo "error: before の定数が既定と一致しません (min=${BEFORE_MIN:-unreadable}); expected 0" >&2
  exit 1
fi
if [[ "$AFTER_MIN" != "$AB_THRESHOLD_T" ]]; then
  echo "error: after の定数が不一致です (min=${AFTER_MIN:-unreadable}); expected ${AB_THRESHOLD_T}" >&2
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
  if [[ "$names" != "crates/backend-cpu/src/mse.rs" ]]; then
    echo "error: after checkout の差分が mse.rs のみではありません: ${names:-(空)}" >&2
    exit 1
  fi
  # 変更行は定数宣言の 1 行置換（-1 行・+1 行）だけであること。
  changed="$(git -C "$after_root" diff -U0 HEAD -- crates/backend-cpu/src/mse.rs | grep -E '^[-+][^-+]' || true)"
  if [[ "$(printf '%s\n' "$changed" | wc -l | tr -d ' ')" != "2" ]] ||
    [[ "$(printf '%s\n' "$changed" | grep -cE '^-.*const MSE_BACKWARD_PARALLEL_MIN_ELEMS\b')" != "1" ]] ||
    [[ "$(printf '%s\n' "$changed" | grep -cE '^\+.*const MSE_BACKWARD_PARALLEL_MIN_ELEMS\b')" != "1" ]]; then
    echo "error: after checkout の差分が定数宣言 1 行の置換のみではありません" >&2
    exit 1
  fi
}
verify_same_commit_and_diff

# 判定対象デバイスは cpu 限定（対象は backend-cpu のMSE backward。CUDA・Metal は対象外）。cpu 以外は fail-closed で拒否する。
DEVICE=${AB_DEVICE:-cpu}
case "$DEVICE" in
  cpu) ;;
  *)
    echo "error: AB_DEVICE must be cpu (got: $DEVICE). this A/B targets the CPU MSE backward only" >&2
    exit 1
    ;;
esac

OUT="results/raw"
mkdir -p "$OUT"
SKIP="$OUT/skipped-3195-${DEVICE}-${LABEL}.log"
ANY_FAILED=0

# 同じ LABEL の既存系列は消去しない（RULE.txt: run の差し替え・追加起動はしない）。
# 既存 JSONL を検出したら書き込み前に終了し、再実行は別ラベルで行わせる
# （R4 の orchestrate_m4max.sh と同じ fail-closed）。
for _reset_arm in before after; do
  for _reset_suffix in train phases; do
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
  for _reset_suffix in train phases; do
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
  cp "$exe" "$OUT/bench-fandhe-3195-${arm}-${LABEL}"
  local tree_output
  tree_output="$(cargo tree -p bench-fandhe --depth 1 --config "$patch_config" 2>&1)"
  if ! echo "$tree_output" | grep -qE 'fandhe-ai v[0-9.]+ \(.*crates/facade\)'; then
    echo "error: fandhe-ai did not resolve to the path-patched crates/facade ($arm); cargo tree:" >&2
    echo "$tree_output" >&2
    exit 1
  fi
  echo "$tree_output" >"$OUT/tree-3195-${arm}-${LABEL}.txt"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$OUT/bench-fandhe-3195-${arm}-${LABEL}" >"$OUT/sha-3195-${arm}-${LABEL}.txt"
  else
    shasum -a 256 "$OUT/bench-fandhe-3195-${arm}-${LABEL}" >"$OUT/sha-3195-${arm}-${LABEL}.txt"
  fi
}

echo "== build before ==(facade=$BEFORE_FACADE)"
build_arm before "$BEFORE_FACADE" "target-ab-3195-${LABEL}-before"
echo "== build after  ==(facade=$AFTER_FACADE)"
build_arm after "$AFTER_FACADE" "target-ab-3195-${LABEL}-after"
cat "$OUT"/tree-3195-*-"${LABEL}".txt

BIN_BEFORE="$OUT/bench-fandhe-3195-before-${LABEL}"
BIN_AFTER="$OUT/bench-fandhe-3195-after-${LABEL}"

# セル（arm × task × mode）の JSONL 行数を数える。読み取り不能は空文字（呼び出し側が
# 不足扱いにする）。round 対応（各 round で各セル +1 行）の検証と、比較前の
# 5 件検証で共用する（PR #2448 codex P1 指摘）。
# gemm は size ごとに別セルのため、第 4 引数 size を指定した場合はその size に限って数える
# （省略時は size を問わない。train／infer は単一形状）。
count_cell_rows() { # count_cell_rows <arm> <task> <mode> [size] -> 行数 | (空)
  local f="$OUT/results-${1}-${LABEL}-${DEVICE}-${2}.jsonl" sz="${4:-}"
  [[ -f "$f" ]] || return 0
  jq -rs --arg m "$3" --arg z "$sz" \
    '[.[] | select((.mode // "fresh") == $m and ($z == "" or (.size | tostring) == $z))] | length' \
    "$f" 2>/dev/null || true
}

# run_cell <arm> <task> <mode> [judge|phases]。phases 指定時は `--phases` 系列（記録のみ・
# 判定に使わない）として `-phases.jsonl` へ追記し、行数検証は行わない（終了コードのみ）。
run_cell() {
  local arm=$1 task=$2 mode=$3 kind=${4:-judge} size=64 bin before_n after_n suffix extra=()
  if [[ "$arm" == "before" ]]; then bin="$BIN_BEFORE"; else bin="$BIN_AFTER"; fi
  suffix=$task
  if [[ "$kind" == phases ]]; then
    suffix=phases
    extra=(--phases)
  fi
  echo "== $task $DEVICE size=$size mode=$mode arm=$arm kind=$kind =="
  before_n=""
  [[ "$kind" == judge ]] && before_n="$(count_cell_rows "$arm" "$task" "$mode" "$size")"
  if ! "$bin" --task "$task" --size "$size" --device "$DEVICE" --mode "$mode" ${extra[@]+"${extra[@]}"} \
    --out "$OUT/results-${arm}-${LABEL}-${DEVICE}-${suffix}.jsonl" 2>"$OUT/err-${arm}-${LABEL}-${DEVICE}.tmp"; then
    echo "arm=$arm task=$task size=$size mode=$mode kind=$kind : $(cat "$OUT/err-${arm}-${LABEL}-${DEVICE}.tmp")" >>"$SKIP"
    echo "  -> FAILED (recorded in $SKIP)"
    ANY_FAILED=$((ANY_FAILED + 1))
  elif [[ "$kind" == judge ]]; then
    # 成功終了でも当該 round のセルが JSONL へちょうど 1 行追加されたことを確認する
    # （追記漏れ・二重追記は round 対応が崩れるため fail-closed）。
    after_n="$(count_cell_rows "$arm" "$task" "$mode" "$size")"
    if [[ ! "$before_n" =~ ^[0-9]+$ || ! "$after_n" =~ ^[0-9]+$ || "$after_n" -ne $((before_n + 1)) ]]; then
      echo "arm=$arm task=$task size=$size mode=$mode : round の記録行数が不正（before=${before_n:-NA} after=${after_n:-NA}。+1 行を期待）" >>"$SKIP"
      echo "  -> ROW COUNT MISMATCH (recorded in $SKIP)"
      ANY_FAILED=$((ANY_FAILED + 1))
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

LOAD_GATE=${AB_LOAD_GATE:-$RULE_LOAD_GATE}
# RULE.txt が正式系列の条件とする閾値はホスト別固定（macmini 3.0・gb10 1.0）。AB_LOAD_GATE で緩和した系列は
# 「通過」と記録せず、適用閾値とともに参考系列（reference）としてログへ明記する
# （PR #2016 codex-review 指摘。事前登録規則とログの通過判定を一致させる）。
if [[ ! "$LOAD_GATE" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
  echo "error: AB_LOAD_GATE must be numeric (got: $LOAD_GATE)" >&2
  exit 1
fi
GATE_SERIES=official
if [[ "$(awk -v a="$LOAD_GATE" -v r="$RULE_LOAD_GATE" 'BEGIN{print (a==r)?1:0}')" != "1" ]]; then
  GATE_SERIES=reference
fi
GATE_LOG="$OUT/load_gate-3195-${DEVICE}-${LABEL}.log"
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
  # gate_ok=1 は厳密に「規則閾値で pass」の round のみ（RULE.txt: 5/5 round が
  # gate_ok=1 の系列だけが正式系列）。timeout・unavailable・pass-reference は 0。
  local gate_ok=0
  [[ "$status" == pass ]] && gate_ok=1
  echo "round${1} gate=${status} gate_ok=${gate_ok} threshold=${LOAD_GATE} load1=${l1} waited_s=${waited} at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$GATE_LOG"
}

: >"$OUT/uptime-3195-${DEVICE}-${LABEL}.log"
uptime | sed 's/.*load/load/' >>"$OUT/uptime-3195-${DEVICE}-${LABEL}.log"

ab_round_cooldown_init "$OUT/cooldown-3195-${DEVICE}-${LABEL}.log"
for run_i in $(seq 1 "$ROUNDS"); do
  ab_round_cooldown "$run_i"
  wait_load_gate "$run_i"
  for mode in fresh reuse; do
    run_pair "$run_i" train "$mode" judge
  done
  for mode in fresh reuse; do
    run_pair "$run_i" train "$mode" phases
  done
  echo "round $run_i: $(uptime | sed 's/.*load/load/')" | tee -a "$OUT/uptime-3195-${DEVICE}-${LABEL}.log"
done

# 比較レポート・エラーログの名前にも LABEL を含め、別 LABEL の再実行が前系列の
# レポートを上書きしないようにする（JSONL・終了コードログと同じ系列別保持。
# PR #2016 codex-review 指摘）
# 比較前に全セル（arm × task × mode）がちょうど ROUNDS 件であることを検証する。
# 不足・過剰があれば当該 task は判定不能として比較へ進まず非ゼロ終了する
# （少標本の中央値で非後退判定が出るのを防ぐ。PR #2448 codex P1 指摘）。
# shellcheck disable=SC2043  # 判定対象は train のみ（RULE.txt）。将来拡張用にループ形を維持
for task in train; do
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
    echo "compare task=$task exit=NA (判定不能: 標本数不足)" | tee -a "$OUT/compare-exit-3195-${DEVICE}-${LABEL}.log"
    ANY_FAILED=$((ANY_FAILED + 1))
    continue
  fi
  # checksum 完全一致の独立確認（RULE.txt: 不一致は機構の不具合として FAIL。ADOPT／REJECT に
  # 数えず停止）。compare_gemm_ab.py は性能後退と checksum 不一致を同じ終了コード 3 で返し
  # 区別できないため、比較の前に JSONL からセル（mode）ごとに全行の checksum が数値かつ
  # 完全一致であることを確認する（fail-closed。#2102 ハーネスと同方式）。
  CK_BAD="$(jq -rs '
      [.[] | select(.device == $dev)]
      | group_by([(.mode // "fresh")])
      | map(select(
          ([.[].checksum] | (map(type == "number") | all) | not)
          or ([.[].checksum] | unique | length) != 1))
      | map("mode=\(.[0].mode // "fresh")")
      | join(", ")' --arg dev "$DEVICE" \
    "$OUT/results-before-${LABEL}-${DEVICE}-${task}.jsonl" "$OUT/results-after-${LABEL}-${DEVICE}-${task}.jsonl" 2>/dev/null)"
  CK_RC=$?
  if [[ "$CK_RC" -ne 0 || -n "$CK_BAD" ]]; then
    echo "compare task=$task: FAIL（checksum 完全一致の確認に失敗: ${CK_BAD:-検査不能 jq exit=$CK_RC}。機構の不具合として停止）" >>"$SKIP"
    echo "compare task=$task exit=NA (FAIL: checksum 不一致。性能判定へ進まない)" | tee -a "$OUT/compare-exit-3195-${DEVICE}-${LABEL}.log"
    ANY_FAILED=$((ANY_FAILED + 1))
    continue
  fi
  python3 compare_gemm_ab.py --device "$DEVICE" --task "$task" --threshold 1.00 --per-run \
    --require-checksum-exact \
    "$OUT/results-before-${LABEL}-${DEVICE}-${task}.jsonl" "$OUT/results-after-${LABEL}-${DEVICE}-${task}.jsonl" \
    >"compare-${task}-3195-${DEVICE}-${LABEL}.md" 2>"compare-${task}-3195-${DEVICE}-${LABEL}.err"
  COMPARE_EXIT=$?
  # 終了コード: 0 = 非後退・3 = 後退セルあり（checksum は上で独立確認済み）。いずれも正常な
  # 判定結果。それ以外は比較処理自体の失敗として非ゼロ終了へ伝播する。
  echo "compare task=$task exit=$COMPARE_EXIT" | tee -a "$OUT/compare-exit-3195-${DEVICE}-${LABEL}.log"
  case "$COMPARE_EXIT" in
    0 | 3) ;;
    *)
      echo "compare task=$task: 比較不能（exit=$COMPARE_EXIT）。$(tail -3 "compare-${task}-3195-${DEVICE}-${LABEL}.err" | tr '\n' ' ')" >>"$SKIP"
      ANY_FAILED=$((ANY_FAILED + 1))
      ;;
  esac
done

echo "done. results in $OUT ; failures (if any) in $SKIP"
if [[ "$ANY_FAILED" -gt 0 ]]; then
  echo "FAILED: $ANY_FAILED run(s) failed; see $SKIP" >&2
  exit 1
fi
