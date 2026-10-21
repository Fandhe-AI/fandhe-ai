#!/usr/bin/env bash
# イシュー #3176: Mac mini（Apple M4 = P コア 4 + E コア 6）での SME `fmopa` カーネル
# （`SME_PRODUCTION_ENABLED`）の ON／OFF A/B。`run_ab_sme_cpu.sh`（#1978／#2118）の腕構成・
# セル集合・`--require-checksum-exact` を継承した Mac mini 派生版で、`run_ab_thread_policy_macmini_cpu.sh`
# （#3176）を下敷きにしている。派生した理由（RULE.txt に事前登録）:
#   (1) 同じ round 内の PyTorch 計測が無く R5（PyTorch 比）を満たせない
#   (2) 内側の負荷ゲート RULE_LOAD_GATE=8.0 が 16 コア機向けで Mac mini の専有ゲート 3.0 と合わない
#   (3) jq 依存（Mac mini 向けの姉妹スクリプトは jq を使わない）
#   (4) compare の exit 3 が「後退」と「checksum 不一致」を区別しない（FAIL 停止を満たせない）
# `run_ab_sme_cpu.sh` と #2118 の一式は未実測の事前登録記録が依存するため変更しない。
# before = main の `crates/facade`、after = 同一コミットで `crates/backend-cpu/src/gemm_blis/mod.rs`
# の `SME_PRODUCTION_ENABLED` 宣言 1 行だけを true にした計測専用 checkout の `crates/facade`。
# それぞれ `[patch.crates-io.fandhe-ai]` path patch した 2 本の `bench-fandhe` を 5 round・round 単位で
# 起動順を反転して交互実行し、同じ round の中で PyTorch（bench_py.py）も測る。
# セルは gemm N=512/1024/2048 と train／infer（size=64）の fresh/reuse（10 セル）。
# main の定数は変更しない（after 腕は計測専用 checkout で、main へコミットしない）。
# 機構発火（SME が実際に動くこと）の前提確認は orchestrate_macmini.sh の P-R0／P-TREE が担う。
# macOS 標準の bash 3.2 で動くこと（変数は ${VAR} 形式・mapfile／連想配列を使わない・jq に依存しない）。
# 判定規則（実測前に固定）: `docs/perf/logs/sme-macmini-3176/RULE.txt`。
# 実行手順: `docs/perf/logs/sme-macmini-3176/README.md`。
#
# 呼び出し例（Mac mini 実機。専有ゲート load1 < 3.0）:
#   AB_BEFORE_FACADE_PATH=/path/to/before/crates/facade \
#   AB_AFTER_FACADE_PATH=/path/to/after/crates/facade \
#   AB_TORCH_PYTHON=/path/to/venv-torch/bin/python \
#     bash run_ab_sme_macmini_cpu.sh 3176-r1
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
  echo "usage: $0 <label>  (label must match [A-Za-z0-9._-]+, e.g. 3176-r1)" >&2
  echo "  env AB_BEFORE_FACADE_PATH / AB_AFTER_FACADE_PATH (absolute paths), AB_TORCH_PYTHON are required" >&2
  exit 1
fi

# A/B の対象ゲート（SME 本番ゲート定数 1 本に固定）。
GATE=sme
GATE_RS="crates/backend-cpu/src/gemm_blis/mod.rs"
GATE_CONST="SME_PRODUCTION_ENABLED"

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
    echo "error: $var_name is required (absolute path to a crates/facade checkout; issue #3176)" >&2
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

# A/B 両 checkout の対象ゲート定数を計測前に検証する（fail-closed）。before は false（main 既定）、
# after は true。読み取りはアンカー付き正規表現のみで行い、ソースは実行しない。宣言が 1 行形式
# でない・取得不能・不一致はビルド前に停止する。
gate_rs() { echo "$1/../backend-cpu/${GATE_RS#crates/backend-cpu/}"; }
read_gate() { # read_gate <facade_path> -> true|false|(空)
  local f
  f="$(gate_rs "$1")"
  [[ -f "$f" ]] || return 0
  sed -nE "s/^[[:space:]]*(pub(\(crate\))?[[:space:]]+)?const[[:space:]]+${GATE_CONST}[[:space:]]*:[[:space:]]*bool[[:space:]]*=[[:space:]]*(true|false)[[:space:]]*;.*/\3/p" "$f" | head -n1
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
  # 変更行は `SME_PRODUCTION_ENABLED` の宣言行（-1 行・+1 行）だけであること。
  if [[ "$(printf '%s\n' "$changed" | wc -l | tr -d ' ')" != "2" ]] ||
    [[ "$(printf '%s\n' "$changed" | grep -cE '^-.*const SME_PRODUCTION_ENABLED([^A-Za-z0-9_]|$)')" != "1" ]] ||
    [[ "$(printf '%s\n' "$changed" | grep -cE '^\+.*const SME_PRODUCTION_ENABLED([^A-Za-z0-9_]|$)')" != "1" ]]; then
    echo "error: after checkout の差分が SME_PRODUCTION_ENABLED 宣言 1 行の置換のみではありません" >&2
    exit 1
  fi
}
verify_same_commit_and_diff

# 判定対象デバイスは cpu 限定（対象は backend-cpu の SME カーネル。CUDA・Metal は対象外）。cpu 以外は fail-closed で拒否する。
DEVICE=${AB_DEVICE:-cpu}
case "$DEVICE" in
  cpu) ;;
  *)
    echo "error: AB_DEVICE must be cpu (got: $DEVICE). this A/B targets the CPU SME kernel only" >&2
    exit 1
    ;;
esac

OUT="results/raw"
mkdir -p "$OUT"
SKIP="$OUT/skipped-3176-${GATE}-${DEVICE}-${LABEL}.log"
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
  echo "bench_py_sha256=$(bench_fandhe_sha256_of "$BENCH_PY")"
  echo "torch=$("$TORCH_PY" -c 'import torch; print(torch.__version__, torch.get_num_threads())' 2>&1 | tail -n1)"
} 2>&1 | sed -e "s#${HOME}#<home>#g" -e "s#$(hostname 2>/dev/null || echo __nohost__)#masked#g" >"${OUT}/env_info-3176-${GATE}-${LABEL}.txt"

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
    # cargo の非 JSON 行・オブジェクト以外は読み飛ばす（AttributeError で落ちない。PR #3501 codex P2）
    if not isinstance(o, dict):
        continue
    t = o.get("target")
    if (o.get("reason") == "compiler-artifact" and isinstance(t, dict) and t.get("name") == "bench-fandhe"
            and isinstance(t.get("kind"), list) and "bin" in t["kind"]
            and isinstance(o.get("executable"), str) and o["executable"]):
        last = o["executable"]
print(last)
' "$msg")"
  rm -f "$msg"
  if [[ -z "$exe" || ! -f "$exe" ]]; then
    echo "error: build $arm: exe not found" >&2
    exit 1
  fi
  cp "$exe" "$OUT/bench-fandhe-3176-${GATE}-${arm}-${LABEL}"
  local tree_output
  tree_output="$(cargo tree -p bench-fandhe --depth 1 --config "$patch_config" 2>&1)"
  if ! echo "$tree_output" | grep -qE 'fandhe-ai v[0-9.]+ \(.*crates/facade\)'; then
    echo "error: fandhe-ai did not resolve to the path-patched crates/facade ($arm); cargo tree:" >&2
    echo "$tree_output" >&2
    exit 1
  fi
  echo "$tree_output" >"$OUT/tree-3176-${GATE}-${arm}-${LABEL}.txt"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$OUT/bench-fandhe-3176-${GATE}-${arm}-${LABEL}" >"$OUT/sha-3176-${GATE}-${arm}-${LABEL}.txt"
  else
    shasum -a 256 "$OUT/bench-fandhe-3176-${GATE}-${arm}-${LABEL}" >"$OUT/sha-3176-${GATE}-${arm}-${LABEL}.txt"
  fi
}

echo "== build before ==(facade=$BEFORE_FACADE)"
build_arm before "$BEFORE_FACADE" "target-ab-3176-${GATE}-${LABEL}-before"
echo "== build after  ==(facade=$AFTER_FACADE)"
build_arm after "$AFTER_FACADE" "target-ab-3176-${GATE}-${LABEL}-after"
cat "$OUT"/tree-3176-${GATE}-*-"${LABEL}".txt

BIN_BEFORE="$OUT/bench-fandhe-3176-${GATE}-before-${LABEL}"
BIN_AFTER="$OUT/bench-fandhe-3176-${GATE}-after-${LABEL}"

# セル（arm × task × mode）の JSONL 行数を数える。読み取り不能は空文字（呼び出し側が
# 不足扱いにする）。round 対応（各 round で各セル +1 行）の検証と、比較前の
# 5 件検証で共用する（PR #2448 codex P1 指摘）。
# gemm は size ごとに別セルのため、第 4 引数 size を指定した場合はその size に限って数える
# （省略時は size を問わない。train／infer は単一形状）。
# 行がオブジェクトでない・size が bool を除く int でない・mode が str でない場合は python が
# 非ゼロ終了し空文字を返す（呼び出し側が不足＝判定不能として扱う。PR #3501 codex P2）。
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
    if not isinstance(o, dict):
        sys.exit("row is not an object")
    sz, md = o.get("size"), o.get("mode", "fresh")
    if isinstance(sz, bool) or not isinstance(sz, int) or not isinstance(md, str):
        sys.exit("bad size/mode type")
    if md == mode and (size == "" or str(sz) == size):
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
GATE_NOT_OK=0
GATE_LOG="$OUT/load_gate-3176-${GATE}-${DEVICE}-${LABEL}.log"
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
  if [[ "$status" == pass && "$GATE_SERIES" != official ]]; then status="pass-${GATE_SERIES}"; fi
  # gate_ok=1 は厳密に「規則閾値 3.0 で pass」の round のみ（RULE.txt: 5/5 round が
  # gate_ok=1 の系列だけが正式系列）。timeout・unavailable・pass-reference は 0。
  local gate_ok=0
  [[ "$status" == pass ]] && gate_ok=1
  # 専有ゲート不成立 round を集約する。
  # 1 件でも不成立なら比較結果が非後退でも判定不能として非ゼロ終了する（末尾参照）。
  if [[ "$gate_ok" != 1 ]]; then GATE_NOT_OK=$((GATE_NOT_OK + 1)); fi
  echo "round${1} gate=${status} gate_ok=${gate_ok} threshold=${LOAD_GATE} load1=${l1} waited_s=${waited} at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$GATE_LOG"
}

: >"$OUT/uptime-3176-${GATE}-${DEVICE}-${LABEL}.log"
uptime | sed 's/.*load/load/' >>"$OUT/uptime-3176-${GATE}-${DEVICE}-${LABEL}.log"

# 同じ run の中の PyTorch 計測（R5。bench_py.py は fresh のみ・同一 round で起動する）。
# 行は round 順に追記される（round 番号は行の出現順で aggregate.py が復元する）。
TORCH_OUT="${OUT}/results-torch-3176-${GATE}-${LABEL}-${DEVICE}.jsonl"
if [[ -e "$TORCH_OUT" ]]; then
  echo "error: 既存の計測記録 $TORCH_OUT があります。差し替え禁止のため別の LABEL で実行してください" >&2
  exit 1
fi
: >"$TORCH_OUT"
# PyTorch は記録専用（RULE.txt §7「判定には使わない」）。PyTorch の計測失敗は SME/NEON 本体の
# ANY_FAILED・$SKIP・終了コードへ伝えず、専用カウンタ TORCH_FAILED と専用ログ $TORCH_FAIL_LOG にだけ
# 記録する（PR #3501 codex P2。PyTorch だけの失敗で P-RUNAB が不成立になり総合判定が undetermined へ
# 落ちるのを防ぐ）。行形式は `torch round=<r> task=<task> size=<n> : <stderr 末尾>` で、最後に
# `torch_failed_total=<n>` を 1 行だけ書く。aggregate.py がこのログを厳格に読み、PyTorch 欠損の注記へ回す。
TORCH_FAILED=0
TORCH_FAIL_LOG="${OUT}/torch-failed-3176-${GATE}-${DEVICE}-${LABEL}.log"
if [[ -e "$TORCH_FAIL_LOG" ]]; then
  echo "error: 既存の失敗記録 $TORCH_FAIL_LOG があります。差し替え禁止のため別の LABEL で実行してください" >&2
  exit 1
fi
: >"$TORCH_FAIL_LOG"
run_torch_cell() { # run_torch_cell <round> <task> <size>
  echo "== torch round=$1 $2 ${DEVICE} size=$3 =="
  if ! "$TORCH_PY" "$BENCH_PY" --framework pytorch --task "$2" --device "$DEVICE" --size "$3" --out "$TORCH_OUT" >/dev/null 2>"${OUT}/err-torch-${LABEL}.tmp"; then
    echo "torch round=$1 task=$2 size=$3 : $(tail -3 "${OUT}/err-torch-${LABEL}.tmp" | tr '\n' ' ')" >>"$TORCH_FAIL_LOG"
    TORCH_FAILED=$((TORCH_FAILED + 1))
  fi
  rm -f "${OUT}/err-torch-${LABEL}.tmp"
}
run_torch_round() { # run_torch_round <round>
  local s
  for s in 512 1024 2048; do run_torch_cell "$1" gemm "$s"; done
  run_torch_cell "$1" train 64
  run_torch_cell "$1" infer 64
}

for run_i in $(seq 1 "$ROUNDS"); do
  wait_load_gate "$run_i"
  for size in 512 1024 2048; do
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
  echo "round $run_i: $(uptime | sed 's/.*load/load/')" | tee -a "$OUT/uptime-3176-${GATE}-${DEVICE}-${LABEL}.log"
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
      # gemm は N=512/1024/2048 の各セル、train／infer は単一形状（size 不問）を検証する。
      for cell_size in $([[ "$task" == gemm ]] && echo "512 1024 2048" || echo ""); do
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
    echo "compare task=$task exit=NA (判定不能: 標本数不足)" | tee -a "$OUT/compare-exit-3176-${GATE}-${DEVICE}-${LABEL}.log"
    ANY_FAILED=$((ANY_FAILED + 1))
    continue
  fi
  # checksum 完全一致の独立確認（RULE.txt: checksum 不一致は機構の不具合として FAIL。
  # ADOPT／REJECT に数えず停止）。compare_gemm_ab.py --require-checksum-exact は性能後退と
  # checksum 不一致を同じ終了コード 3 で返し区別できないため、比較の前に JSONL から
  # セル（mode × size）ごとに before・after 全行の checksum が数値かつ完全一致であることを
  # 確認し、不一致・欠損・検査不能なら比較へ進まず FAIL として停止する（fail-closed。PR #2451 codex P1 指摘）。
  # 行がオブジェクトでない・識別フィールド（device／mode は str、size は bool を除く int）の型が
  # 不正・checksum が非有限の行は「入力構造不正」を出して exit 2 とし、AttributeError／TypeError
  # の素の例外で落とさない（下の分岐で checksum 不一致〈FAIL〉とは区別し判定不能として停止する。PR #3501 codex P2）。
  CK_BAD="$(python3 -c '
import json, math, sys
dev = sys.argv[1]
groups = {}
def malformed(where, why):
    print("入力構造不正 %s: %s" % (where, why))
    sys.exit(2)
for path in sys.argv[2:]:
    for i, line in enumerate(open(path, encoding="utf-8"), 1):
        line = line.strip()
        if not line:
            continue
        where = "%s:%d" % (path, i)
        try:
            o = json.loads(line)
        except (ValueError, RecursionError) as e:
            malformed(where, "JSON 不正")
        if not isinstance(o, dict):
            malformed(where, "行がオブジェクトではない")
        d, m, sz = o.get("device"), o.get("mode", "fresh"), o.get("size")
        if not isinstance(d, str) or not isinstance(m, str) or isinstance(sz, bool) or not isinstance(sz, int):
            malformed(where, "device／mode／size の型が不正")
        if d != dev:
            continue
        groups.setdefault((m, str(sz)), []).append(o.get("checksum"))
bad = []
for (mode, size), cs in sorted(groups.items()):
    numeric = all(isinstance(c, float) or (isinstance(c, int) and not isinstance(c, bool) and abs(c) < 2 ** 53) for c in cs) and all(math.isfinite(c) for c in cs)
    if not numeric or len(set(cs)) != 1:
        bad.append("mode=%s size=%s" % (mode, size))
print(", ".join(bad))
' "$DEVICE" \
    "${OUT}/results-before-${LABEL}-${DEVICE}-${task}.jsonl" "${OUT}/results-after-${LABEL}-${DEVICE}-${task}.jsonl" 2>/dev/null)"
  CK_RC=$?
  if [[ "$CK_RC" -eq 2 ]]; then
    # 入力構造不正は checksum 不一致（FAIL）ではなく判定不能として停止する（RULE.txt の undetermined）
    echo "compare task=$task: 判定不能（${CK_BAD:-入力構造不正}。checksum 不一致ではない）" >>"$SKIP"
    echo "compare task=$task exit=NA (判定不能: JSONL の入力構造不正)" | tee -a "$OUT/compare-exit-3176-${GATE}-${DEVICE}-${LABEL}.log"
    ANY_FAILED=$((ANY_FAILED + 1))
    continue
  fi
  if [[ "$CK_RC" -ne 0 || -n "$CK_BAD" ]]; then
    echo "compare task=$task: FAIL（checksum 完全一致の確認に失敗: ${CK_BAD:-検査不能 python exit=${CK_RC}}。機構の不具合として ADOPT／REJECT に数えず停止）" >>"$SKIP"
    echo "compare task=$task exit=NA (FAIL: checksum 不一致。性能判定へ進まない)" | tee -a "$OUT/compare-exit-3176-${GATE}-${DEVICE}-${LABEL}.log"
    ANY_FAILED=$((ANY_FAILED + 1))
    continue
  fi
  python3 compare_gemm_ab.py --device "$DEVICE" --task "$task" --threshold 1.00 --per-run \
    --require-checksum-exact \
    "$OUT/results-before-${LABEL}-${DEVICE}-${task}.jsonl" "$OUT/results-after-${LABEL}-${DEVICE}-${task}.jsonl" \
    >"compare-${task}-3176-${GATE}-${DEVICE}-${LABEL}.md" 2>"compare-${task}-3176-${GATE}-${DEVICE}-${LABEL}.err"
  COMPARE_EXIT=$?
  # compare_gemm_ab.py の終了コード: 0 = 非後退・3 = 「後退セルあり」または「判定不能セルあり」
  # （evaluate_cell の status != ok・欠測セル・per-run 比の算出不能はいずれも 3 に畳まれる）。
  # checksum は上で独立確認済み。3 を無条件に REJECT 相当の正常判定として扱うと RULE.txt が
  # 区別する undetermined を REJECT と誤認するため、比較レポートのセル別状態を検査し、
  # 「判定不能」行があれば性能判定とは独立に undetermined として記録し非ゼロ終了へ伝播する
  # （PR #2464 codex P1 指摘）。2 = 入力不正・空データ、それ以外（python 起動失敗等）は
  # 比較処理自体の失敗であり、判定結果と区別して非ゼロ終了へ伝播する。
  echo "compare task=$task exit=$COMPARE_EXIT" | tee -a "$OUT/compare-exit-3176-${GATE}-${DEVICE}-${LABEL}.log"
  case "$COMPARE_EXIT" in
    0) ;;
    3)
      if grep -q '判定不能' "compare-${task}-3176-${GATE}-${DEVICE}-${LABEL}.md"; then
        echo "compare task=$task: undetermined（判定不能セルあり。REJECT ではない。詳細は compare-${task}-3176-${GATE}-${DEVICE}-${LABEL}.md の「判定不能」行）" >>"$SKIP"
        echo "compare task=$task undetermined=1 (判定不能セルあり。性能判定 REJECT とは区別)" | tee -a "$OUT/compare-exit-3176-${GATE}-${DEVICE}-${LABEL}.log"
        ANY_FAILED=$((ANY_FAILED + 1))
      fi
      ;;
    *)
      echo "compare task=$task: 比較不能（exit=${COMPARE_EXIT}）。$(tail -3 "compare-${task}-3176-${GATE}-${DEVICE}-${LABEL}.err" | tr '\n' ' ')" >>"$SKIP"
      ANY_FAILED=$((ANY_FAILED + 1))
      ;;
  esac
done

if [[ "$GATE_NOT_OK" -gt 0 ]]; then
  echo "gate: 判定不能（専有ゲート gate_ok=1 でない round が ${GATE_NOT_OK} 件。RULE.txt は 5/5 round が gate_ok=1 の系列のみ正式判定対象。ADOPT／REJECT に数えない。詳細は ${GATE_LOG}）" >>"$SKIP"
  ANY_FAILED=$((ANY_FAILED + 1))
fi

# PyTorch の失敗件数は記録専用のログにだけ書き、ANY_FAILED と終了コードには加えない（RULE.txt §7）
echo "torch_failed_total=${TORCH_FAILED}" >>"$TORCH_FAIL_LOG"
if [[ "$TORCH_FAILED" -gt 0 ]]; then
  echo "note: PyTorch の計測が ${TORCH_FAILED} 件失敗（記録専用。判定・終了コードには影響しない）; see $TORCH_FAIL_LOG" >&2
fi
echo "done. results in $OUT ; failures (if any) in $SKIP"
if [[ "$ANY_FAILED" -gt 0 ]]; then
  echo "FAILED: $ANY_FAILED run(s) failed; see $SKIP" >&2
  exit 1
fi
