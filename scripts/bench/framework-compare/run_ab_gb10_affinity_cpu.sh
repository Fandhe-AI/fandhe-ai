#!/usr/bin/env bash
# イシュー #2117: GB10 大コア affinity（#1576。`crates/backend-cpu/src/gb10_affinity.rs` の
# `GB10_AFFINITY_ENABLED`。既定 false）の framework-compare cpu A/B
# （train・infer size=64 × fresh/reuse ＋ gemm N=256/512/1024/2048/4096 × fresh/reuse の 14 セル）。
# before = main の `crates/facade`、after = 同一コミットで上記 1 定数（false -> true）だけを
# 変えた計測専用 worktree の `crates/facade` を、それぞれ `[patch.crates-io.fandhe-ai]` path
# patch した 2 本の `bench-fandhe` としてビルドし、5 round・round 単位で起動順を反転しながら
# 交互実行する（`run_ab_gb10_affinity_cpu.sh`〈#2117〉の派生。ビルド・lock 復元・
# path patch 解決検証・負荷ゲートの骨格は同一）。after worktree は計測専用で main へコミットしない。
# 追加要素: 各 arm で `crates/backend-cpu/tests/gb10_affinity_report.rs` を実行し、専用
# affinity プールが after だけで実際に有効であること（機構発火）をベンチ前に fail-closed で確認する
# （両腕が実質同一動作のまま「非後退」で通る偽 ADOPT の遮断）。
# 判定規則（実測前に固定）: `docs/perf/logs/cpu-gb10-affinity-ab-2117/RULE.txt`。
# 実行手順: `docs/perf/logs/cpu-gb10-affinity-ab-2117/README.md`。
#
# 呼び出し例（GB10 実機。専有ゲート load1 < 1.0）:
#   AB_BEFORE_FACADE_PATH=/path/to/before/crates/facade \
#   AB_AFTER_FACADE_PATH=/path/to/after/crates/facade \
#     bash run_ab_gb10_affinity_cpu.sh 2117-gb10
# x86 等での疎通確認（系列ではなく smoke。機構は発火せず ratio に意味はない）:
#   AB_AFFINITY_PRECHECK=report-only AB_LOAD_GATE=64 ... bash run_ab_gb10_affinity_cpu.sh smoke-x86
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
  echo "usage: $0 <label>  (label must match [A-Za-z0-9._-]+, e.g. 2117-gb10)" >&2
  echo "  env AB_BEFORE_FACADE_PATH / AB_AFTER_FACADE_PATH (absolute paths) are required" >&2
  exit 1
fi

validate_facade_path() {
  local var_name=$1 path=$2
  if [[ -z "$path" ]]; then
    echo "error: $var_name is required (absolute path to a crates/facade checkout; issue #2117)" >&2
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

# RAYON_NUM_THREADS が設定されていると機構が `env_override` で無効化され、after 腕が
# 実質 before と同一になる（偽 ADOPT）ため、設定されていれば拒否する（fail-closed）。
if [[ -n "${RAYON_NUM_THREADS:-}" ]]; then
  echo "error: RAYON_NUM_THREADS が設定されています（affinity 機構が無効化される）。unset して再実行してください" >&2
  exit 1
fi

# 機構発火の事前確認モード。assert（既定・正式系列）: before は pool_active=false、after は
# enabled・大コア 10 個検出・pool_active=true を要求する。report-only は x86 等での疎通確認用で、
# 出力を記録するだけで assert しない（ログに series=smoke と明記し、正式系列としては扱わない）。
PRECHECK=${AB_AFFINITY_PRECHECK:-assert}
case "$PRECHECK" in
  assert | report-only) ;;
  *)
    echo "error: AB_AFFINITY_PRECHECK must be assert|report-only (got: $PRECHECK)" >&2
    exit 1
    ;;
esac

# A/B 両 checkout の `GB10_AFFINITY_ENABLED` を計測前に検証する（fail-closed）。before は
# false（main 既定）、after は true。読み取りはアンカー付き正規表現のみで行い、ソースは
# 実行しない。宣言が 1 行形式でない・取得不能・不一致はビルド前に停止する。
affinity_rs() { echo "$1/../backend-cpu/src/gb10_affinity.rs"; }
read_gate() { # read_gate <facade_path> -> true|false|(空)
  local f
  f="$(affinity_rs "$1")"
  [[ -f "$f" ]] || return 0
  sed -nE 's/^[[:space:]]*pub(\(crate\))?[[:space:]]+const[[:space:]]+GB10_AFFINITY_ENABLED[[:space:]]*:[[:space:]]*bool[[:space:]]*=[[:space:]]*(true|false)[[:space:]]*;.*/\2/p' "$f" | head -n1
}
BEFORE_GATE="$(read_gate "$BEFORE_FACADE")"
AFTER_GATE="$(read_gate "$AFTER_FACADE")"
if [[ "$BEFORE_GATE" != "false" ]]; then
  echo "error: before の GB10_AFFINITY_ENABLED が false ではありません (${BEFORE_GATE:-unreadable})" >&2
  exit 1
fi
if [[ "$AFTER_GATE" != "true" ]]; then
  echo "error: after の GB10_AFFINITY_ENABLED が true ではありません (${AFTER_GATE:-unreadable})" >&2
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
  if [[ "$names" != "crates/backend-cpu/src/gb10_affinity.rs" ]]; then
    echo "error: after checkout の差分が gb10_affinity.rs のみではありません: ${names:-(空)}" >&2
    exit 1
  fi
  # 変更行は `GB10_AFFINITY_ENABLED` の宣言行（-1 行・+1 行）だけであること。
  changed="$(git -C "$after_root" diff -U0 HEAD -- crates/backend-cpu/src/gb10_affinity.rs | grep -E '^[-+][^-+]' || true)"
  if [[ "$(printf '%s\n' "$changed" | wc -l | tr -d ' ')" != "2" ]] ||
    [[ "$(printf '%s\n' "$changed" | grep -cE '^-.*const GB10_AFFINITY_ENABLED\b')" != "1" ]] ||
    [[ "$(printf '%s\n' "$changed" | grep -cE '^\+.*const GB10_AFFINITY_ENABLED\b')" != "1" ]]; then
    echo "error: after checkout の差分が GB10_AFFINITY_ENABLED 宣言 1 行の置換のみではありません" >&2
    exit 1
  fi
}
verify_same_commit_and_diff

# 判定対象デバイスは cpu 限定（対象は backend-cpu の GB10 affinity。CUDA・Metal は対象外）。cpu 以外は fail-closed で拒否する。
DEVICE=${AB_DEVICE:-cpu}
case "$DEVICE" in
  cpu) ;;
  *)
    echo "error: AB_DEVICE must be cpu (got: $DEVICE). this A/B targets the CPU GB10 affinity only" >&2
    exit 1
    ;;
esac

OUT="results/raw"
mkdir -p "$OUT"
SKIP="$OUT/skipped-2117-${DEVICE}-${LABEL}.log"
ANY_FAILED=0

# 同じ LABEL の既存系列は消去しない（RULE.txt: run の差し替え・追加起動はしない）。
# 既存 JSONL を検出したら書き込み前に終了し、再実行は別ラベルで行わせる
# （#2102 と同じ fail-closed）。
for _reset_arm in before after; do
  for _reset_suffix in gemm train infer; do
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
  for _reset_suffix in gemm train infer; do
    : >"$OUT/results-${_reset_arm}-${LABEL}-${DEVICE}-${_reset_suffix}.jsonl"
  done
done

bench_fandhe_setup_lock_restore_trap

# 機構発火の事前確認（RULE.txt「機構発火の前提条件」）。各 arm の checkout ルートで
# `gb10_affinity_report` 結合テストを実行し、出力を保存する。assert モードでは before が
# pool_active=false、after が enabled・大コア 10 個検出・pool_active=true でなければ停止する
# （テスト側が EXPECT_GB10_AFFINITY_POOL_ACTIVE で検査。系列は無効＝判定 undetermined）。
# after では `--lib` を実行しない（`GB10_AFFINITY_ENABLED` が true の単体テストが落ちるため）。
run_affinity_precheck() { # run_affinity_precheck <arm> <facade_path> <expect 0|1>
  local arm=$1 root out
  root="$(git -C "$2" rev-parse --show-toplevel)" || exit 1
  out="$OUT/affinity-report-2117-${arm}-${LABEL}.txt"
  local -a envs=()
  [[ "$PRECHECK" == assert ]] && envs=("EXPECT_GB10_AFFINITY_POOL_ACTIVE=$3")
  if ! (cd "$root" && env ${envs[@]+"${envs[@]}"} cargo test -p fandhe-ai-backend-cpu --release --locked \
    --test gb10_affinity_report -- --ignored --nocapture) >"$out" 2>&1; then
    tail -20 "$out" >&2
    echo "affinity precheck FAILED ($arm): 機構発火の前提が不成立。系列は無効（判定 undetermined）" >>"$SKIP"
    echo "error: affinity precheck failed ($arm); see $out" >&2
    exit 1
  fi
  if ! grep -q '^gb10_affinity_report enabled=' "$out"; then
    echo "error: affinity precheck ($arm) の出力に report 行がありません" >&2
    exit 1
  fi
  echo "precheck($arm,mode=${PRECHECK}): $(grep '^gb10_affinity_report enabled=' "$out" | head -n1)"
}
echo "== affinity precheck (mode=${PRECHECK}) =="
run_affinity_precheck before "$BEFORE_FACADE" 0
run_affinity_precheck after "$AFTER_FACADE" 1

# env_info（ホスト名・$HOME はマスクしてから収録する。RULE.txt「マスク規則」）
{
  echo "uname_m=$(uname -m)"
  echo "rustc=$(rustc --version 2>/dev/null || echo NA)"
  echo "head=$(git -C "$(git -C "$BEFORE_FACADE" rev-parse --show-toplevel)" rev-parse HEAD)"
  echo "cpu_model=$(lscpu 2>/dev/null | sed -nE 's/^Model name:[[:space:]]*//p' | head -n1)"
  echo "nproc=$(nproc 2>/dev/null || echo NA)"
  echo "cpus_allowed_list=$(sed -nE 's/^Cpus_allowed_list:[[:space:]]*//p' /proc/self/status 2>/dev/null)"
  echo "rayon_num_threads=${RAYON_NUM_THREADS:-unset}"
  echo "precheck=${PRECHECK}"
  echo "--- lscpu -e (CPU,MAXMHZ) ---"
  lscpu -e=CPU,MAXMHZ 2>/dev/null || echo NA
} 2>&1 | sed -e "s#${HOME}#<home>#g" -e "s#$(hostname 2>/dev/null || echo __nohost__)#masked#g" >"$OUT/env_info-2117-${LABEL}.txt"

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
  cp "$exe" "$OUT/bench-fandhe-2117-${arm}-${LABEL}"
  local tree_output
  tree_output="$(cargo tree -p bench-fandhe --depth 1 --config "$patch_config" 2>&1)"
  if ! echo "$tree_output" | grep -qE 'fandhe-ai v[0-9.]+ \(.*crates/facade\)'; then
    echo "error: fandhe-ai did not resolve to the path-patched crates/facade ($arm); cargo tree:" >&2
    echo "$tree_output" >&2
    exit 1
  fi
  echo "$tree_output" >"$OUT/tree-2117-${arm}-${LABEL}.txt"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$OUT/bench-fandhe-2117-${arm}-${LABEL}" >"$OUT/sha-2117-${arm}-${LABEL}.txt"
  else
    shasum -a 256 "$OUT/bench-fandhe-2117-${arm}-${LABEL}" >"$OUT/sha-2117-${arm}-${LABEL}.txt"
  fi
}

echo "== build before ==(facade=$BEFORE_FACADE)"
build_arm before "$BEFORE_FACADE" "target-ab-2117-${LABEL}-before"
echo "== build after  ==(facade=$AFTER_FACADE)"
build_arm after "$AFTER_FACADE" "target-ab-2117-${LABEL}-after"
cat "$OUT"/tree-2117-*-"${LABEL}".txt

BIN_BEFORE="$OUT/bench-fandhe-2117-before-${LABEL}"
BIN_AFTER="$OUT/bench-fandhe-2117-after-${LABEL}"

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

run_cell() { # run_cell <arm> <task> <mode> [size]
  local arm=$1 task=$2 mode=$3 size=${4:-64} bin before_n after_n
  if [[ "$arm" == "before" ]]; then bin="$BIN_BEFORE"; else bin="$BIN_AFTER"; fi
  echo "== $task $DEVICE size=$size mode=$mode arm=$arm =="
  before_n="$(count_cell_rows "$arm" "$task" "$mode" "$size")"
  if ! "$bin" --task "$task" --size "$size" --device "$DEVICE" --mode "$mode" \
    --out "$OUT/results-${arm}-${LABEL}-${DEVICE}-${task}.jsonl" 2>"$OUT/err-${arm}-${LABEL}-${DEVICE}.tmp"; then
    echo "arm=$arm task=$task size=$size mode=$mode : $(cat "$OUT/err-${arm}-${LABEL}-${DEVICE}.tmp")" >>"$SKIP"
    echo "  -> FAILED (recorded in $SKIP)"
    ANY_FAILED=$((ANY_FAILED + 1))
  else
    # 成功終了でも当該 round のセルが JSONL へ ちょうど 1 行追加されたことを確認する
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

LOAD_GATE=${AB_LOAD_GATE:-1.0}
# RULE.txt が正式系列の条件とする閾値は 1.0 固定。AB_LOAD_GATE で緩和した系列は
# 「通過」と記録せず、適用閾値とともに参考系列（reference）としてログへ明記する
# （PR #2016 codex-review 指摘。事前登録規則とログの通過判定を一致させる）。
RULE_LOAD_GATE=1.0
if [[ ! "$LOAD_GATE" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
  echo "error: AB_LOAD_GATE must be numeric (got: $LOAD_GATE)" >&2
  exit 1
fi
GATE_SERIES=official
if [[ "$(awk -v a="$LOAD_GATE" -v r="$RULE_LOAD_GATE" 'BEGIN{print (a==r)?1:0}')" != "1" ]]; then
  GATE_SERIES=reference
fi
GATE_LOG="$OUT/load_gate-2117-${DEVICE}-${LABEL}.log"
: >"$GATE_LOG"
echo "threshold=${LOAD_GATE} rule_threshold=${RULE_LOAD_GATE} series=${GATE_SERIES} precheck=${PRECHECK}$([[ "$PRECHECK" == report-only ]] && echo " series_note=smoke（正式系列として扱わない）")" >>"$GATE_LOG"
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
  # gate_ok=1 は厳密に「規則閾値 1.0 で pass」の round のみ（RULE.txt: 5/5 round が
  # gate_ok=1 の系列だけが正式系列）。timeout・unavailable・pass-reference は 0。
  local gate_ok=0
  [[ "$status" == pass ]] && gate_ok=1
  echo "round${1} gate=${status} gate_ok=${gate_ok} threshold=${LOAD_GATE} load1=${l1} waited_s=${waited} at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$GATE_LOG"
}

: >"$OUT/uptime-2117-${DEVICE}-${LABEL}.log"
uptime | sed 's/.*load/load/' >>"$OUT/uptime-2117-${DEVICE}-${LABEL}.log"

for run_i in $(seq 1 "$ROUNDS"); do
  wait_load_gate "$run_i"
  for size in 256 512 1024 2048 4096; do
    for mode in fresh reuse; do
      run_pair "$run_i" gemm "$mode" "$size"
    done
  done
  for task in train infer; do
    for mode in fresh reuse; do
      run_pair "$run_i" "$task" "$mode"
    done
  done
  echo "round $run_i: $(uptime | sed 's/.*load/load/')" | tee -a "$OUT/uptime-2117-${DEVICE}-${LABEL}.log"
done

# 比較レポート・エラーログの名前にも LABEL を含め、別 LABEL の再実行が前系列の
# レポートを上書きしないようにする（JSONL・終了コードログと同じ系列別保持。
# PR #2016 codex-review 指摘）
# 比較前に全セル（arm × task × mode）がちょうど ROUNDS 件であることを検証する。
# 不足・過剰があれば当該 task は判定不能として比較へ進まず非ゼロ終了する
# （少標本の中央値で非後退判定が出るのを防ぐ。PR #2448 codex P1 指摘）。
for task in gemm train infer; do
  CELLS_OK=1
  for arm in before after; do
    for mode in fresh reuse; do
      # gemm は N=256/512/1024/2048/4096 の各セル、train／infer は単一形状（size 不問）を検証する。
      for cell_size in $([[ "$task" == gemm ]] && echo "256 512 1024 2048 4096" || echo ""); do
        n="$(count_cell_rows "$arm" "$task" "$mode" "$cell_size")"
        if [[ ! "$n" =~ ^[0-9]+$ || "$n" -ne "$ROUNDS" ]]; then
          echo "compare task=$task: 判定不能（arm=$arm mode=$mode size=${cell_size:-NA} の記録は ${n:-NA} 件。${ROUNDS} 件を要求）" >>"$SKIP"
          CELLS_OK=0
        fi
      done
      if [[ "$task" != gemm ]]; then
        n="$(count_cell_rows "$arm" "$task" "$mode")"
        if [[ ! "$n" =~ ^[0-9]+$ || "$n" -ne "$ROUNDS" ]]; then
          echo "compare task=$task: 判定不能（arm=$arm mode=$mode の記録は ${n:-NA} 件。${ROUNDS} 件を要求）" >>"$SKIP"
          CELLS_OK=0
        fi
      fi
    done
  done
  if [[ "$CELLS_OK" != "1" ]]; then
    echo "compare task=$task exit=NA (判定不能: 標本数不足)" | tee -a "$OUT/compare-exit-2117-${DEVICE}-${LABEL}.log"
    ANY_FAILED=$((ANY_FAILED + 1))
    continue
  fi
  # checksum 完全一致の独立確認（RULE.txt: checksum 不一致は機構の不具合として FAIL。
  # ADOPT／REJECT に数えず停止）。compare_gemm_ab.py --require-checksum-exact は性能後退と
  # checksum 不一致を同じ終了コード 3 で返し区別できないため、比較の前に JSONL から
  # セル（mode × size）ごとに before・after 全行の checksum が数値かつ完全一致であることを
  # 確認し、不一致・欠損・検査不能なら比較へ進まず FAIL として停止する（fail-closed。PR #2451 codex P1 指摘）。
  CK_BAD="$(jq -rs '
      [.[] | select(.device == $dev)]
      | group_by([(.mode // "fresh"), (.size // "NA" | tostring)])
      | map(select(
          ([.[].checksum] | (map(type == "number") | all) | not)
          or ([.[].checksum] | unique | length) != 1))
      | map("mode=\(.[0].mode // "fresh") size=\(.[0].size // "NA")")
      | join(", ")' --arg dev "$DEVICE" \
    "$OUT/results-before-${LABEL}-${DEVICE}-${task}.jsonl" "$OUT/results-after-${LABEL}-${DEVICE}-${task}.jsonl" 2>/dev/null)"
  CK_RC=$?
  if [[ "$CK_RC" -ne 0 || -n "$CK_BAD" ]]; then
    echo "compare task=$task: FAIL（checksum 完全一致の確認に失敗: ${CK_BAD:-検査不能 jq exit=$CK_RC}。機構の不具合として ADOPT／REJECT に数えず停止）" >>"$SKIP"
    echo "compare task=$task exit=NA (FAIL: checksum 不一致。性能判定へ進まない)" | tee -a "$OUT/compare-exit-2117-${DEVICE}-${LABEL}.log"
    ANY_FAILED=$((ANY_FAILED + 1))
    continue
  fi
  SIZES_ARGS=()
  [[ "$task" == gemm ]] && SIZES_ARGS=(--sizes affinity)
  python3 compare_gemm_ab.py --device "$DEVICE" --task "$task" --threshold 1.00 --per-run \
    --require-checksum-exact ${SIZES_ARGS[@]+"${SIZES_ARGS[@]}"} \
    "$OUT/results-before-${LABEL}-${DEVICE}-${task}.jsonl" "$OUT/results-after-${LABEL}-${DEVICE}-${task}.jsonl" \
    >"compare-${task}-2117-${DEVICE}-${LABEL}.md" 2>"compare-${task}-2117-${DEVICE}-${LABEL}.err"
  COMPARE_EXIT=$?
  # compare_gemm_ab.py の終了コード: 0 = 非後退・3 = 後退セルあり（checksum は上で
  # 独立確認済みのため、ここでの 3 は性能後退のみ。いずれも正常な判定結果で記録のみ）。2 = 入力不正・空データ、それ以外（python 起動失敗等）は
  # 比較処理自体の失敗であり、判定結果と区別して非ゼロ終了へ伝播する。
  echo "compare task=$task exit=$COMPARE_EXIT" | tee -a "$OUT/compare-exit-2117-${DEVICE}-${LABEL}.log"
  case "$COMPARE_EXIT" in
    0 | 3) ;;
    *)
      echo "compare task=$task: 比較不能（exit=$COMPARE_EXIT）。$(tail -3 "compare-${task}-2117-${DEVICE}-${LABEL}.err" | tr '\n' ' ')" >>"$SKIP"
      ANY_FAILED=$((ANY_FAILED + 1))
      ;;
  esac
done

echo "done. results in $OUT ; failures (if any) in $SKIP"
if [[ "$ANY_FAILED" -gt 0 ]]; then
  echo "FAILED: $ANY_FAILED run(s) failed; see $SKIP" >&2
  exit 1
fi
