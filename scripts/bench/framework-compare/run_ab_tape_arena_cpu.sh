#!/usr/bin/env bash
# イシュー #2104（Phase 3 親 #2099）: ホスト arena（`fandhe_ai_tensor_core::alloc`。
# `HOST_ARENA_DEFAULT_ENABLED`）の framework-compare train／infer cpu A/B。
# before = main の `crates/facade`、after = 同一コミットで
# `crates/tensor-core/src/alloc.rs` の `HOST_ARENA_DEFAULT_ENABLED` だけを `true` に
# した計測専用 worktree の `crates/facade` を、それぞれ `[patch.crates-io.fandhe-ai]`
# path patch した 2 本の `bench-fandhe` として CPU 限定でビルドし、5 round・round 単位で
# 起動順を反転しながら交互実行する（`run_ab_sme_cpu.sh` の派生。ビルド・lock 復元・
# path patch 解決検証・load ゲートは同一。gemm セルは本件の対象外のため持たない）。
# 定数を反転した worktree は計測専用であり main へはコミットしない（既定切替は
# 両機体 ADOPT 後の別 PR。`docs/perf/logs/tape-arena-reuse-2104/RULE.txt`）。
#
# 呼び出し例:
#   AB_BEFORE_FACADE_PATH=/path/to/before/crates/facade \
#   AB_AFTER_FACADE_PATH=/path/to/after/crates/facade \
#     bash run_ab_tape_arena_cpu.sh 2104
set -u
cd "$(dirname "$0")" || exit 1
# shellcheck source=./bench_fandhe_lock_restore.sh
source ./bench_fandhe_lock_restore.sh

LABEL=${1:-}
ROUNDS=${AB_ROUNDS:-5}

# A03 インジェクション対策: ラベルはファイル名・パスに直接埋め込むため、
# 英数字・`._-` のみを許可する allowlist で検証する。
if [[ -z "$LABEL" || ! "$LABEL" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "usage: $0 <label>  (label must match [A-Za-z0-9._-]+, e.g. 2104)" >&2
  echo "  env AB_BEFORE_FACADE_PATH / AB_AFTER_FACADE_PATH (absolute paths) are required" >&2
  exit 1
fi

validate_facade_path() {
  local var_name=$1 path=$2
  if [[ -z "$path" ]]; then
    echo "error: $var_name is required (absolute path to a crates/facade checkout; issue #2104)" >&2
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

# A/B 両 checkout の `HOST_ARENA_DEFAULT_ENABLED` を計測前に検証する（fail-closed。
# PR #2448 codex 指摘）。before=false・after=true 以外（同一 checkout・反転漏れ・
# 定数の取得不能）は比較として無効なため、ビルド前に停止する。
read_arena_default() { # read_arena_default <facade_path> -> true|false|(空)
  local alloc="$1/../tensor-core/src/alloc.rs"
  [[ -f "$alloc" ]] || return 0
  sed -nE 's/^[[:space:]]*pub(\(crate\))?[[:space:]]+const[[:space:]]+HOST_ARENA_DEFAULT_ENABLED[[:space:]]*:[[:space:]]*bool[[:space:]]*=[[:space:]]*(true|false)[[:space:]]*;.*/\2/p' "$alloc" | head -n1
}
BEFORE_ARENA_DEFAULT="$(read_arena_default "$BEFORE_FACADE")"
AFTER_ARENA_DEFAULT="$(read_arena_default "$AFTER_FACADE")"
if [[ "$BEFORE_ARENA_DEFAULT" != "false" || "$AFTER_ARENA_DEFAULT" != "true" ]]; then
  echo "error: HOST_ARENA_DEFAULT_ENABLED mismatch (before=${BEFORE_ARENA_DEFAULT:-unreadable}, after=${AFTER_ARENA_DEFAULT:-unreadable}); expected before=false / after=true" >&2
  exit 1
fi

# 判定対象デバイスは cpu 限定（arena は CPU の host バッファ再利用のため。CUDA・Metal は
# 対象外）。cpu 以外は fail-closed で拒否する。
DEVICE=${AB_DEVICE:-cpu}
case "$DEVICE" in
  cpu) ;;
  *)
    echo "error: AB_DEVICE must be cpu (got: $DEVICE). this A/B targets the CPU host arena only" >&2
    exit 1
    ;;
esac

OUT="results/raw"
mkdir -p "$OUT"
SKIP="$OUT/skipped-2104-${DEVICE}-${LABEL}.log"
ANY_FAILED=0

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
  cp "$exe" "$OUT/bench-fandhe-2104-${arm}-${LABEL}"
  local tree_output
  tree_output="$(cargo tree -p bench-fandhe --depth 1 --config "$patch_config" 2>&1)"
  if ! echo "$tree_output" | grep -qE 'fandhe-ai v[0-9.]+ \(.*crates/facade\)'; then
    echo "error: fandhe-ai did not resolve to the path-patched crates/facade ($arm); cargo tree:" >&2
    echo "$tree_output" >&2
    exit 1
  fi
  echo "$tree_output" >"$OUT/tree-2104-${arm}-${LABEL}.txt"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$OUT/bench-fandhe-2104-${arm}-${LABEL}" >"$OUT/sha-2104-${arm}-${LABEL}.txt"
  else
    shasum -a 256 "$OUT/bench-fandhe-2104-${arm}-${LABEL}" >"$OUT/sha-2104-${arm}-${LABEL}.txt"
  fi
}

echo "== build before ==(facade=$BEFORE_FACADE)"
build_arm before "$BEFORE_FACADE" "target-ab-2104-${LABEL}-before"
echo "== build after  ==(facade=$AFTER_FACADE)"
build_arm after "$AFTER_FACADE" "target-ab-2104-${LABEL}-after"
cat "$OUT"/tree-2104-*-"${LABEL}".txt

BIN_BEFORE="$OUT/bench-fandhe-2104-before-${LABEL}"
BIN_AFTER="$OUT/bench-fandhe-2104-after-${LABEL}"

run_cell() { # run_cell <arm> <task> <mode> [size]
  local arm=$1 task=$2 mode=$3 size=${4:-64} bin
  if [[ "$arm" == "before" ]]; then bin="$BIN_BEFORE"; else bin="$BIN_AFTER"; fi
  echo "== $task $DEVICE size=$size mode=$mode arm=$arm =="
  if ! "$bin" --task "$task" --size "$size" --device "$DEVICE" --mode "$mode" \
    --out "$OUT/results-${arm}-${LABEL}-${DEVICE}-${task}.jsonl" 2>"$OUT/err-${arm}-${LABEL}-${DEVICE}.tmp"; then
    echo "arm=$arm task=$task size=$size mode=$mode : $(cat "$OUT/err-${arm}-${LABEL}-${DEVICE}.tmp")" >>"$SKIP"
    echo "  -> FAILED (recorded in $SKIP)"
    ANY_FAILED=$((ANY_FAILED + 1))
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
GATE_LOG="$OUT/load-gate-2104-${DEVICE}-${LABEL}.log"
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

: >"$OUT/uptime-2104-${DEVICE}-${LABEL}.log"
uptime | sed 's/.*load/load/' >>"$OUT/uptime-2104-${DEVICE}-${LABEL}.log"

for run_i in $(seq 1 "$ROUNDS"); do
  wait_load_gate "$run_i"
  for task in train infer; do
    for mode in fresh reuse; do
      run_pair "$run_i" "$task" "$mode"
    done
  done
  echo "round $run_i: $(uptime | sed 's/.*load/load/')" | tee -a "$OUT/uptime-2104-${DEVICE}-${LABEL}.log"
done

# 比較レポート・エラーログの名前にも LABEL を含め、別 LABEL の再実行が前系列の
# レポートを上書きしないようにする（JSONL・終了コードログと同じ系列別保持。
# PR #2016 codex-review 指摘）
for task in train infer; do
  python3 compare_gemm_ab.py --device "$DEVICE" --task "$task" --threshold 1.00 --per-run \
    --require-checksum-exact \
    "$OUT/results-before-${LABEL}-${DEVICE}-${task}.jsonl" "$OUT/results-after-${LABEL}-${DEVICE}-${task}.jsonl" \
    >"compare-${task}-2104-${DEVICE}-${LABEL}.md" 2>"compare-${task}-2104-${DEVICE}-${LABEL}.err"
  COMPARE_EXIT=$?
  # compare_gemm_ab.py の終了コード: 0 = 非後退・3 = 後退セルあり（いずれも正常な
  # 判定結果で記録のみ）。2 = 入力不正・空データ、それ以外（python 起動失敗等）は
  # 比較処理自体の失敗であり、判定結果と区別して非ゼロ終了へ伝播する。
  echo "compare task=$task exit=$COMPARE_EXIT" | tee -a "$OUT/compare-exit-2104-${DEVICE}-${LABEL}.log"
  case "$COMPARE_EXIT" in
    0 | 3) ;;
    *)
      echo "compare task=$task: 比較不能（exit=$COMPARE_EXIT）。$(tail -3 "compare-${task}-2104-${DEVICE}-${LABEL}.err" | tr '\n' ' ')" >>"$SKIP"
      ANY_FAILED=$((ANY_FAILED + 1))
      ;;
  esac
done

echo "done. results in $OUT ; failures (if any) in $SKIP"
if [[ "$ANY_FAILED" -gt 0 ]]; then
  echo "FAILED: $ANY_FAILED run(s) failed; see $SKIP" >&2
  exit 1
fi
