#!/usr/bin/env bash
# イシュー #3178: Mac mini（M4 = P コア 4 + E コア 6）の CPU スレッド方針 A/B。
# 既定 OFF の 2 ゲートを `AB_GATE` で 1 本ずつ反転して測る（gb10 affinity A/B〈#2117〉の派生）。
#   AB_GATE=big-core : `crates/backend-cpu/src/thread_limit.rs` の `BIG_CORE_LIMIT_ENABLED`
#                      （GEMM の並列度を P コア数へ制限する。#1364 で REJECT 済み）
#   AB_GATE=small-cap: `crates/backend-cpu/src/small_shape_thread_cap.rs` の
#                      `SMALL_SHAPE_CAP_ENABLED` ＋ `ELIGIBLE_BRANDS` への実測 brand 追加
#                      （小形状 GEMM を専用 6 スレッドプールで実行する。#1575 で REJECT 済み）
# before = main の `crates/facade`、after = 同一コミットで上記の宣言行だけを変えた計測専用 checkout
# の `crates/facade`。それぞれ `[patch.crates-io.fandhe-ai]` path patch した 2 本の `bench-fandhe` を
# 5 round・round 単位で起動順を反転して交互実行し、同じ round の中で PyTorch（bench_py.py）も測る。
# セルは gemm N=256/512/1024/2048 と train／infer（size=64）の fresh/reuse（12 セル）。
# main のゲート定数は変更しない（after 腕は計測専用 checkout で、main へコミットしない）。
# macOS 標準の bash 3.2 で動くこと（変数は ${VAR} 形式・mapfile／連想配列を使わない・jq に依存しない）。
# 判定規則（実測前に固定）: `docs/perf/logs/cpu-thread-policy-macmini-3178/RULE.txt`。
# 実行手順: `docs/perf/logs/cpu-thread-policy-macmini-3178/README.md`。
#
# 呼び出し例（Mac mini 実機。専有ゲート load1 < 3.0）:
#   AB_GATE=big-core \
#   AB_BEFORE_FACADE_PATH=/path/to/before/crates/facade \
#   AB_AFTER_FACADE_PATH=/path/to/after/crates/facade \
#   AB_TORCH_PYTHON=/path/to/venv-torch/bin/python \
#     bash run_ab_thread_policy_macmini_cpu.sh 3178-big-core
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
  echo "usage: $0 <label>  (label must match [A-Za-z0-9._-]+, e.g. 3178-big-core)" >&2
  echo "  env AB_GATE=big-core|small-cap, AB_BEFORE_FACADE_PATH / AB_AFTER_FACADE_PATH (absolute paths), AB_TORCH_PYTHON are required" >&2
  exit 1
fi

# A/B の対象ゲート（allowlist。ファイル名・差分検証の分岐に使うため厳格に検証する）。
GATE=${AB_GATE:-}
case "$GATE" in
  big-core)
    GATE_RS="crates/backend-cpu/src/thread_limit.rs"
    GATE_CONST="BIG_CORE_LIMIT_ENABLED"
    PRECHECK_TEST="thread_limit_report"
    PRECHECK_ENV="EXPECT_BIG_CORE_LIMIT"
    PRECHECK_LINE="thread_limit_report current="
    ;;
  small-cap)
    GATE_RS="crates/backend-cpu/src/small_shape_thread_cap.rs"
    GATE_CONST="SMALL_SHAPE_CAP_ENABLED"
    PRECHECK_TEST="small_shape_cap_report"
    PRECHECK_ENV="EXPECT_SMALL_SHAPE_CAP"
    PRECHECK_LINE="small_shape_cap_report current="
    ;;
  *)
    echo "error: AB_GATE must be big-core|small-cap (got: ${GATE:-unset})" >&2
    exit 1
    ;;
esac

# PyTorch 同席計測用の python（venv-torch の絶対パス。bench_py.py が要る torch を持つもの）。
TORCH_PY=${AB_TORCH_PYTHON:-}
if [[ -z "$TORCH_PY" || "$TORCH_PY" != /* || ! -x "$TORCH_PY" ]]; then
  echo "error: AB_TORCH_PYTHON must be an absolute path to an executable python (got: ${TORCH_PY:-unset})" >&2
  exit 1
fi
BENCH_PY="../../../docs/perf/logs/lowlayer-diagnosis-2026-09-12/bench_py.py"
if [[ ! -f "$BENCH_PY" ]]; then
  echo "error: bench_py.py not found (${BENCH_PY})" >&2
  exit 1
fi

validate_facade_path() {
  local var_name=$1 path=$2
  if [[ -z "$path" ]]; then
    echo "error: $var_name is required (absolute path to a crates/facade checkout; issue #3178)" >&2
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

# RAYON_NUM_THREADS が設定されていると両機構が `env_override` で無効化され、after 腕が
# 実質 before と同一になる（偽 ADOPT）ため、設定されていれば拒否する（fail-closed）。
if [[ -n "${RAYON_NUM_THREADS:-}" ]]; then
  echo "error: RAYON_NUM_THREADS が設定されています（スレッド方針の機構が無効化される）。unset して再実行してください" >&2
  exit 1
fi

# 機構発火の事前確認モード。assert（既定・正式系列）: ゲート別の期待値（テスト側の EXPECT_* で
# 検査。big-core は effective が P コア数 4 へ絞られること、small-cap は専用プール発火）を要求する。
# report-only は Mac mini 以外での疎通確認用で、
# 出力を記録するだけで assert しない（ログに series=smoke と明記し、正式系列としては扱わない）。
PRECHECK=${AB_THREADPOLICY_PRECHECK:-assert}
case "$PRECHECK" in
  assert | report-only) ;;
  *)
    echo "error: AB_THREADPOLICY_PRECHECK must be assert|report-only (got: $PRECHECK)" >&2
    exit 1
    ;;
esac

# A/B 両 checkout の対象ゲート定数を計測前に検証する（fail-closed）。before は false（main 既定）、
# after は true。読み取りはアンカー付き正規表現のみで行い、ソースは実行しない。宣言が 1 行形式
# でない・取得不能・不一致はビルド前に停止する。
gate_rs() { echo "$1/../backend-cpu/src/$(basename "${GATE_RS}")"; }
read_gate() { # read_gate <facade_path> -> true|false|(空)
  local f
  f="$(gate_rs "$1")"
  [[ -f "$f" ]] || return 0
  sed -nE "s/^[[:space:]]*pub(\(crate\))?[[:space:]]+const[[:space:]]+${GATE_CONST}[[:space:]]*:[[:space:]]*bool[[:space:]]*=[[:space:]]*(true|false)[[:space:]]*;.*/\2/p" "$f" | head -n1
}
BEFORE_GATE="$(read_gate "$BEFORE_FACADE")"
AFTER_GATE="$(read_gate "$AFTER_FACADE")"
if [[ "$BEFORE_GATE" != "false" ]]; then
  echo "error: before の ${GATE_CONST} が false ではありません (${BEFORE_GATE:-unreadable})" >&2
  exit 1
fi
if [[ "$AFTER_GATE" != "true" ]]; then
  echo "error: after の ${GATE_CONST} が true ではありません (${AFTER_GATE:-unreadable})" >&2
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
  if [[ "$names" != "${GATE_RS}" ]]; then
    echo "error: after checkout の差分が ${GATE_RS} のみではありません: ${names:-(空)}" >&2
    exit 1
  fi
  changed="$(git -C "$after_root" diff -U0 HEAD -- "${GATE_RS}" | grep -E '^[-+][^-+]' || true)"
  if [[ "$GATE" == big-core ]]; then
    # 変更行は `BIG_CORE_LIMIT_ENABLED` の宣言行（-1 行・+1 行）だけであること。
    if [[ "$(printf '%s\n' "$changed" | wc -l | tr -d ' ')" != "2" ]] ||
      [[ "$(printf '%s\n' "$changed" | grep -cE '^-.*const BIG_CORE_LIMIT_ENABLED([^A-Za-z0-9_]|$)')" != "1" ]] ||
      [[ "$(printf '%s\n' "$changed" | grep -cE '^\+.*const BIG_CORE_LIMIT_ENABLED([^A-Za-z0-9_]|$)')" != "1" ]]; then
      echo "error: after checkout の差分が BIG_CORE_LIMIT_ENABLED 宣言 1 行の置換のみではありません" >&2
      exit 1
    fi
  else
    # small-cap: `SMALL_SHAPE_CAP_ENABLED` 宣言 1 行と `ELIGIBLE_BRANDS` 宣言 1 行（-2／+2）だけ。
    # brand 追加は既存の "Apple M4 Max" を残した追加であること（診断用の上書き。本番 allowlist は #3227）。
    if [[ "$(printf '%s\n' "$changed" | wc -l | tr -d ' ')" != "4" ]] ||
      [[ "$(printf '%s\n' "$changed" | grep -cE '^-.*const SMALL_SHAPE_CAP_ENABLED([^A-Za-z0-9_]|$)')" != "1" ]] ||
      [[ "$(printf '%s\n' "$changed" | grep -cE '^\+.*const SMALL_SHAPE_CAP_ENABLED([^A-Za-z0-9_]|$)')" != "1" ]] ||
      [[ "$(printf '%s\n' "$changed" | grep -cE '^-.*const ELIGIBLE_BRANDS([^A-Za-z0-9_]|$)')" != "1" ]] ||
      [[ "$(printf '%s\n' "$changed" | grep -cE '^\+.*const ELIGIBLE_BRANDS([^A-Za-z0-9_]|$).*"Apple M4 Max"')" != "1" ]]; then
      echo "error: after checkout の差分が SMALL_SHAPE_CAP_ENABLED／ELIGIBLE_BRANDS 宣言 2 行の置換のみではありません" >&2
      exit 1
    fi
  fi
}
verify_same_commit_and_diff

# 判定対象デバイスは cpu 限定（対象は backend-cpu のスレッド方針。CUDA・Metal は対象外）。cpu 以外は fail-closed で拒否する。
DEVICE=${AB_DEVICE:-cpu}
case "$DEVICE" in
  cpu) ;;
  *)
    echo "error: AB_DEVICE must be cpu (got: $DEVICE). this A/B targets the CPU thread policy only" >&2
    exit 1
    ;;
esac

OUT="results/raw"
mkdir -p "$OUT"
SKIP="$OUT/skipped-3178-${GATE}-${DEVICE}-${LABEL}.log"
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

# 機構発火の事前確認（RULE.txt「機構発火の前提条件」）。各 arm の checkout ルートでゲートに対応する
# 結合テスト（thread_limit_report／small_shape_cap_report）を実行し、出力を保存する。assert モードでは
# テスト側が EXPECT_* で検査し、失敗した系列は無効（判定 undetermined）としてベンチ前に停止する。
# after では `--lib` を実行しない（ゲートが true の単体テストが落ちるため）。
run_gate_precheck() { # run_gate_precheck <arm> <facade_path> <expect 0|1>
  local arm=$1 root out
  root="$(git -C "$2" rev-parse --show-toplevel)" || exit 1
  out="${OUT}/precheck-3178-${GATE}-${arm}-${LABEL}.txt"
  local -a envs=()
  [[ "$PRECHECK" == assert ]] && envs=("${PRECHECK_ENV}=$3")
  if ! (cd "$root" && env ${envs[@]+"${envs[@]}"} cargo test -p fandhe-ai-backend-cpu --release --locked \
    --test "${PRECHECK_TEST}" -- --ignored --nocapture) >"$out" 2>&1; then
    tail -20 "$out" >&2
    echo "precheck FAILED (${arm}): 機構発火の前提が不成立。系列は無効（判定 undetermined）" >>"$SKIP"
    echo "error: precheck failed (${arm}); see $out" >&2
    exit 1
  fi
  if ! grep -q "^${PRECHECK_LINE}" "$out"; then
    echo "error: precheck (${arm}) の出力に report 行がありません" >&2
    exit 1
  fi
  echo "precheck(${arm},mode=${PRECHECK}): $(grep "^${PRECHECK_LINE}" "$out" | head -n1)"
}
echo "== precheck (gate=${GATE}, mode=${PRECHECK}) =="
run_gate_precheck before "$BEFORE_FACADE" 0
run_gate_precheck after "$AFTER_FACADE" 1

# env_info（ホスト名・$HOME はマスクしてから収録する。RULE.txt「マスク規則」）
{
  echo "gate=${GATE}"
  echo "sw_vers=$(sw_vers 2>/dev/null | tr '\n' ' ')"
  echo "uname_m=$(uname -m)"
  echo "rustc=$(rustc --version 2>/dev/null || echo NA)"
  echo "head=$(git -C "$(git -C "$BEFORE_FACADE" rev-parse --show-toplevel)" rev-parse HEAD)"
  echo "cpu_brand=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo NA)"
  echo "hw.logicalcpu=$(sysctl -n hw.logicalcpu 2>/dev/null || echo NA)"
  echo "hw.perflevel0.logicalcpu=$(sysctl -n hw.perflevel0.logicalcpu 2>/dev/null || echo NA)"
  echo "hw.perflevel1.logicalcpu=$(sysctl -n hw.perflevel1.logicalcpu 2>/dev/null || echo NA)"
  echo "hw.memsize=$(sysctl -n hw.memsize 2>/dev/null || echo NA)"
  echo "rayon_num_threads=${RAYON_NUM_THREADS:-unset}"
  echo "precheck=${PRECHECK}"
  echo "bench_py_sha256=$(bench_fandhe_sha256_of "$BENCH_PY")"
  echo "torch=$("$TORCH_PY" -c 'import torch; print(torch.__version__, torch.get_num_threads())' 2>&1 | tail -n1)"
} 2>&1 | sed -e "s#${HOME}#<home>#g" -e "s#$(hostname 2>/dev/null || echo __nohost__)#masked#g" >"${OUT}/env_info-3178-${GATE}-${LABEL}.txt"

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
  exe="$(python3 -c '
import json, sys
last = ""
for line in open(sys.argv[1]):
    try:
        o = json.loads(line)
    except ValueError:
        continue
    if (o.get("reason") == "compiler-artifact" and o.get("target", {}).get("name") == "bench-fandhe"
            and "bin" in o.get("target", {}).get("kind", []) and o.get("executable")):
        last = o["executable"]
print(last)
' "$msg")"
  rm -f "$msg"
  if [[ -z "$exe" || ! -f "$exe" ]]; then
    echo "error: build $arm: exe not found" >&2
    exit 1
  fi
  cp "$exe" "$OUT/bench-fandhe-3178-${GATE}-${arm}-${LABEL}"
  local tree_output
  tree_output="$(cargo tree -p bench-fandhe --depth 1 --config "$patch_config" 2>&1)"
  if ! echo "$tree_output" | grep -qE 'fandhe-ai v[0-9.]+ \(.*crates/facade\)'; then
    echo "error: fandhe-ai did not resolve to the path-patched crates/facade ($arm); cargo tree:" >&2
    echo "$tree_output" >&2
    exit 1
  fi
  echo "$tree_output" >"$OUT/tree-3178-${GATE}-${arm}-${LABEL}.txt"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$OUT/bench-fandhe-3178-${GATE}-${arm}-${LABEL}" >"$OUT/sha-3178-${GATE}-${arm}-${LABEL}.txt"
  else
    shasum -a 256 "$OUT/bench-fandhe-3178-${GATE}-${arm}-${LABEL}" >"$OUT/sha-3178-${GATE}-${arm}-${LABEL}.txt"
  fi
}

echo "== build before ==(facade=$BEFORE_FACADE)"
build_arm before "$BEFORE_FACADE" "target-ab-3178-${GATE}-${LABEL}-before"
echo "== build after  ==(facade=$AFTER_FACADE)"
build_arm after "$AFTER_FACADE" "target-ab-3178-${GATE}-${LABEL}-after"
cat "$OUT"/tree-3178-${GATE}-*-"${LABEL}".txt

BIN_BEFORE="$OUT/bench-fandhe-3178-${GATE}-before-${LABEL}"
BIN_AFTER="$OUT/bench-fandhe-3178-${GATE}-after-${LABEL}"

# セル（arm × task × mode）の JSONL 行数を数える。読み取り不能は空文字（呼び出し側が
# 不足扱いにする）。round 対応（各 round で各セル +1 行）の検証と、比較前の
# 5 件検証で共用する（PR #2448 codex P1 指摘）。
# gemm は size ごとに別セルのため、第 4 引数 size を指定した場合はその size に限って数える
# （省略時は size を問わない。train／infer は単一形状）。
count_cell_rows() { # count_cell_rows <arm> <task> <mode> [size] -> 行数 | (空)
  local f="$OUT/results-${1}-${LABEL}-${DEVICE}-${2}.jsonl" sz="${4:-}"
  [[ -f "$f" ]] || return 0
  python3 -c '
import json, sys
f, mode, size = sys.argv[1], sys.argv[2], sys.argv[3]
n = 0
for line in open(f):
    line = line.strip()
    if not line:
        continue
    o = json.loads(line)
    if o.get("mode", "fresh") == mode and (size == "" or str(o.get("size")) == size):
        n += 1
print(n)
' "$f" "$3" "$sz" 2>/dev/null || true
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

LOAD_GATE=${AB_LOAD_GATE:-3.0}
# RULE.txt が正式系列の条件とする閾値は 3.0 固定。AB_LOAD_GATE で緩和した系列は
# 「通過」と記録せず、適用閾値とともに参考系列（reference）としてログへ明記する
# （PR #2016 codex-review 指摘。事前登録規則とログの通過判定を一致させる）。
RULE_LOAD_GATE=3.0
if [[ ! "$LOAD_GATE" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
  echo "error: AB_LOAD_GATE must be numeric (got: $LOAD_GATE)" >&2
  exit 1
fi
GATE_SERIES=official
if [[ "$(awk -v a="$LOAD_GATE" -v r="$RULE_LOAD_GATE" 'BEGIN{print (a==r)?1:0}')" != "1" ]]; then
  GATE_SERIES=reference
fi
# report-only（機構発火を assert しない疎通確認）は閾値が 3.0 でも正式系列にしない。
# RULE.txt: 機構発火の確認を経ない系列は正式判定の対象外。series は smoke に固定する
# （PR #2464 codex P1 指摘）。
if [[ "$PRECHECK" == report-only ]]; then
  GATE_SERIES=smoke
fi
GATE_NOT_OK=0
GATE_LOG="$OUT/load_gate-3178-${GATE}-${DEVICE}-${LABEL}.log"
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
  if [[ "$status" == pass && "$GATE_SERIES" != official ]]; then status="pass-${GATE_SERIES}"; fi
  # gate_ok=1 は厳密に「規則閾値 3.0 で pass」の round のみ（RULE.txt: 5/5 round が
  # gate_ok=1 の系列だけが正式系列）。timeout・unavailable・pass-reference は 0。
  local gate_ok=0
  [[ "$status" == pass ]] && gate_ok=1
  # 専有ゲート不成立 round を集約する（smoke は正式判定の対象外のため数えない）。
  # 1 件でも不成立なら比較結果が非後退でも判定不能として非ゼロ終了する（末尾参照）。
  if [[ "$gate_ok" != 1 && "$GATE_SERIES" != smoke ]]; then GATE_NOT_OK=$((GATE_NOT_OK + 1)); fi
  echo "round${1} gate=${status} gate_ok=${gate_ok} threshold=${LOAD_GATE} load1=${l1} waited_s=${waited} at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$GATE_LOG"
}

: >"$OUT/uptime-3178-${GATE}-${DEVICE}-${LABEL}.log"
uptime | sed 's/.*load/load/' >>"$OUT/uptime-3178-${GATE}-${DEVICE}-${LABEL}.log"

# 同じ run の中の PyTorch 計測（R5。bench_py.py は fresh のみ・同一 round で起動する）。
# 行は round 順に追記される（round 番号は行の出現順で aggregate.py が復元する）。
TORCH_OUT="${OUT}/results-torch-3178-${GATE}-${LABEL}-${DEVICE}.jsonl"
if [[ -e "$TORCH_OUT" ]]; then
  echo "error: 既存の計測記録 $TORCH_OUT があります。差し替え禁止のため別の LABEL で実行してください" >&2
  exit 1
fi
: >"$TORCH_OUT"
run_torch_cell() { # run_torch_cell <task> <size>
  echo "== torch $1 ${DEVICE} size=$2 =="
  if ! "$TORCH_PY" "$BENCH_PY" --framework pytorch --task "$1" --device "$DEVICE" --size "$2" --out "$TORCH_OUT" >/dev/null 2>"${OUT}/err-torch-${LABEL}.tmp"; then
    echo "torch task=$1 size=$2 : $(tail -3 "${OUT}/err-torch-${LABEL}.tmp" | tr '\n' ' ')" >>"$SKIP"
    ANY_FAILED=$((ANY_FAILED + 1))
  fi
  rm -f "${OUT}/err-torch-${LABEL}.tmp"
}
run_torch_round() { # run_torch_round <round>
  local s
  for s in 256 512 1024 2048; do run_torch_cell gemm "$s"; done
  run_torch_cell train 64
  run_torch_cell infer 64
}

for run_i in $(seq 1 "$ROUNDS"); do
  wait_load_gate "$run_i"
  for size in 256 512 1024 2048; do
    for mode in fresh reuse; do
      run_pair "$run_i" gemm "$mode" "$size"
    done
  done
  for task in train infer; do
    for mode in fresh reuse; do
      run_pair "$run_i" "$task" "$mode"
    done
  done
  run_torch_round "$run_i"
  echo "round $run_i: $(uptime | sed 's/.*load/load/')" | tee -a "$OUT/uptime-3178-${GATE}-${DEVICE}-${LABEL}.log"
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
      # gemm は N=256/512/1024/2048 の各セル、train／infer は単一形状（size 不問）を検証する。
      for cell_size in $([[ "$task" == gemm ]] && echo "256 512 1024 2048" || echo ""); do
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
    echo "compare task=$task exit=NA (判定不能: 標本数不足)" | tee -a "$OUT/compare-exit-3178-${GATE}-${DEVICE}-${LABEL}.log"
    ANY_FAILED=$((ANY_FAILED + 1))
    continue
  fi
  # checksum 完全一致の独立確認（RULE.txt: checksum 不一致は機構の不具合として FAIL。
  # ADOPT／REJECT に数えず停止）。compare_gemm_ab.py --require-checksum-exact は性能後退と
  # checksum 不一致を同じ終了コード 3 で返し区別できないため、比較の前に JSONL から
  # セル（mode × size）ごとに before・after 全行の checksum が数値かつ完全一致であることを
  # 確認し、不一致・欠損・検査不能なら比較へ進まず FAIL として停止する（fail-closed。PR #2451 codex P1 指摘）。
  CK_BAD="$(python3 -c '
import json, sys
dev = sys.argv[1]
groups = {}
for path in sys.argv[2:]:
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        if o.get("device") != dev:
            continue
        groups.setdefault((o.get("mode", "fresh"), str(o.get("size", "NA"))), []).append(o.get("checksum"))
bad = []
for (mode, size), cs in sorted(groups.items()):
    numeric = all(isinstance(c, (int, float)) and not isinstance(c, bool) for c in cs)
    if not numeric or len(set(cs)) != 1:
        bad.append("mode=%s size=%s" % (mode, size))
print(", ".join(bad))
' "$DEVICE" \
    "${OUT}/results-before-${LABEL}-${DEVICE}-${task}.jsonl" "${OUT}/results-after-${LABEL}-${DEVICE}-${task}.jsonl" 2>/dev/null)"
  CK_RC=$?
  if [[ "$CK_RC" -ne 0 || -n "$CK_BAD" ]]; then
    echo "compare task=$task: FAIL（checksum 完全一致の確認に失敗: ${CK_BAD:-検査不能 python exit=${CK_RC}}。機構の不具合として ADOPT／REJECT に数えず停止）" >>"$SKIP"
    echo "compare task=$task exit=NA (FAIL: checksum 不一致。性能判定へ進まない)" | tee -a "$OUT/compare-exit-3178-${GATE}-${DEVICE}-${LABEL}.log"
    ANY_FAILED=$((ANY_FAILED + 1))
    continue
  fi
  SIZES_ARGS=()
  [[ "$task" == gemm ]] && SIZES_ARGS=(--sizes threadpolicy)
  python3 compare_gemm_ab.py --device "$DEVICE" --task "$task" --threshold 1.00 --per-run \
    --require-checksum-exact ${SIZES_ARGS[@]+"${SIZES_ARGS[@]}"} \
    "$OUT/results-before-${LABEL}-${DEVICE}-${task}.jsonl" "$OUT/results-after-${LABEL}-${DEVICE}-${task}.jsonl" \
    >"compare-${task}-3178-${GATE}-${DEVICE}-${LABEL}.md" 2>"compare-${task}-3178-${GATE}-${DEVICE}-${LABEL}.err"
  COMPARE_EXIT=$?
  # compare_gemm_ab.py の終了コード: 0 = 非後退・3 = 「後退セルあり」または「判定不能セルあり」
  # （evaluate_cell の status != ok・欠測セル・per-run 比の算出不能はいずれも 3 に畳まれる）。
  # checksum は上で独立確認済み。3 を無条件に REJECT 相当の正常判定として扱うと RULE.txt が
  # 区別する undetermined を REJECT と誤認するため、比較レポートのセル別状態を検査し、
  # 「判定不能」行があれば性能判定とは独立に undetermined として記録し非ゼロ終了へ伝播する
  # （PR #2464 codex P1 指摘）。2 = 入力不正・空データ、それ以外（python 起動失敗等）は
  # 比較処理自体の失敗であり、判定結果と区別して非ゼロ終了へ伝播する。
  echo "compare task=$task exit=$COMPARE_EXIT" | tee -a "$OUT/compare-exit-3178-${GATE}-${DEVICE}-${LABEL}.log"
  case "$COMPARE_EXIT" in
    0) ;;
    3)
      if grep -q '判定不能' "compare-${task}-3178-${GATE}-${DEVICE}-${LABEL}.md"; then
        echo "compare task=$task: undetermined（判定不能セルあり。REJECT ではない。詳細は compare-${task}-3178-${GATE}-${DEVICE}-${LABEL}.md の「判定不能」行）" >>"$SKIP"
        echo "compare task=$task undetermined=1 (判定不能セルあり。性能判定 REJECT とは区別)" | tee -a "$OUT/compare-exit-3178-${GATE}-${DEVICE}-${LABEL}.log"
        ANY_FAILED=$((ANY_FAILED + 1))
      fi
      ;;
    *)
      echo "compare task=$task: 比較不能（exit=${COMPARE_EXIT}）。$(tail -3 "compare-${task}-3178-${GATE}-${DEVICE}-${LABEL}.err" | tr '\n' ' ')" >>"$SKIP"
      ANY_FAILED=$((ANY_FAILED + 1))
      ;;
  esac
done

if [[ "$GATE_NOT_OK" -gt 0 ]]; then
  echo "gate: 判定不能（専有ゲート gate_ok=1 でない round が ${GATE_NOT_OK} 件。RULE.txt は 5/5 round が gate_ok=1 の系列のみ正式判定対象。ADOPT／REJECT に数えない。詳細は ${GATE_LOG}）" >>"$SKIP"
  ANY_FAILED=$((ANY_FAILED + 1))
fi

echo "done. results in $OUT ; failures (if any) in $SKIP"
if [[ "$ANY_FAILED" -gt 0 ]]; then
  echo "FAILED: $ANY_FAILED run(s) failed; see $SKIP" >&2
  exit 1
fi
