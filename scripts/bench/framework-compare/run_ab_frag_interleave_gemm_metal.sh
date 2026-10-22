#!/usr/bin/env bash
# イシュー #3212: simdgroup 交互配置候補のゲート（`crates/backend-metal/src/tile.rs` の
# `FRAG_INTERLEAVE_ENABLED`。main は false。実装は #3211）の framework-compare gemm Metal セル A/B
# （Mac mini 用）。`run_ab_m64_thin_tile_metal.sh`（#3209）の派生で、ラベル allowlist・
# `validate_facade_path`・同一コミット＋定数 1 行差分の検証・未追跡ビルド入力の検出・
# `bench_fandhe_lock_restore.sh`・`ab_round_cooldown.sh`・ROUNDS=5 固定・before／after バイナリの
# sha256 不一致チェックは継承する。
#
# before = main の `crates/facade`、after = 同一コミットで上記定数だけを `true` にした計測専用
# checkout の `crates/facade` を、それぞれ `[patch.crates-io.fandhe-ai]` path patch した 2 本の
# `bench-fandhe` としてビルドし、5 round・round 単位で起動順を反転しながら交互実行する。
# セルは metal × gemm × N {256,512,1024,2048,4096} × {fresh, reuse}（10 セル）。同じ round で
# candle（`bench-candle --task gemm --device metal`。fresh のみ。record_only）も測り、candle 比
# の算出に使う。
#
# 規則の正は `docs/perf/logs/metal-frag-interleave-ab-macmini-3212/RULE.txt` 項 11:
# gemm セル A/B は「非後退ゲート」であり改善の証拠にしない（ADOPT／REJECT の根拠は純カーネル
# 時間〈`orchestrate_macmini.sh`〉）。本スクリプトは判定しない。`compare_gemm_ab.py
# --sizes metal-square5 --threshold 1.00 --per-run --require-checksum-exact` の出力と終了コードを
# 記録し、checksum 不一致・判定不能は FAIL として非ゼロ終了にする（純粋な速度後退は記録のみ。
# round 別の比は `aggregate.py` が JSONL から直接計算する）。
# 負荷ゲートは RULE 項 9 の load1 < 3.0。before／after のバイナリ sha256 が同一なら定数の反転が
# 効いていないため fail-closed で停止する。after は計測専用でありコミットしない。
#
# 呼び出し例（Mac mini 上。bash 3.2 互換。変数は ${VAR} 形式）:
#   AB_BEFORE_FACADE_PATH=/abs/before/crates/facade \
#   AB_AFTER_FACADE_PATH=/abs/after/crates/facade \
#     bash run_ab_frag_interleave_gemm_metal.sh 3212
set -u
cd "$(dirname "$0")" || exit 1
# shellcheck source=./bench_fandhe_lock_restore.sh
source ./bench_fandhe_lock_restore.sh
# shellcheck source=./ab_round_cooldown.sh
source ./ab_round_cooldown.sh

LABEL=${1:-}
ROUNDS=5
if [[ -n "${AB_ROUNDS:-}" && "${AB_ROUNDS}" != "5" ]]; then
  echo "error: AB_ROUNDS must be 5 (RULE.txt の事前登録は 5 round 固定。got: ${AB_ROUNDS})" >&2
  exit 1
fi
SIZES="256 512 1024 2048 4096"
MODES="fresh reuse"

# A03 インジェクション対策: ラベルはファイル名・パスに直接埋め込むため allowlist で検証する。
if [[ -z "${LABEL}" || ! "${LABEL}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "usage: $0 <label>  (label must match [A-Za-z0-9._-]+, e.g. 3212)" >&2
  echo "  env AB_BEFORE_FACADE_PATH / AB_AFTER_FACADE_PATH (absolute paths) are required" >&2
  exit 1
fi

validate_facade_path() {
  local var_name=$1 path=$2
  if [[ -z "${path}" ]]; then
    echo "error: ${var_name} is required (absolute path to a crates/facade checkout; issue #3212)" >&2
    exit 1
  fi
  if [[ "${path}" != /* ]]; then
    echo "error: ${var_name} must be an absolute path (got: ${path})" >&2
    exit 1
  fi
  if [[ "${path}" == *'"'* || "${path}" == *'\'* ]]; then
    echo "error: ${var_name} must not contain a double quote or backslash" >&2
    exit 1
  fi
  if [[ ! -f "${path}/Cargo.toml" ]]; then
    echo "error: ${var_name}/Cargo.toml not found (${path})" >&2
    exit 1
  fi
  if ! grep -qE '^\s*name\s*=\s*"fandhe-ai"\s*$' "${path}/Cargo.toml"; then
    echo "error: ${var_name}/Cargo.toml does not declare name = \"fandhe-ai\" (${path})" >&2
    exit 1
  fi
}
validate_facade_path AB_BEFORE_FACADE_PATH "${AB_BEFORE_FACADE_PATH:-}"
validate_facade_path AB_AFTER_FACADE_PATH "${AB_AFTER_FACADE_PATH:-}"
BEFORE_FACADE="${AB_BEFORE_FACADE_PATH}"
AFTER_FACADE="${AB_AFTER_FACADE_PATH}"

# A/B 両 checkout の `FRAG_INTERLEAVE_ENABLED` を計測前に検証する（fail-closed）。
# before=false・after=true 以外（同一 checkout・反転漏れ・取得不能）はビルド前に停止する。
# 宣言は `pub(crate) const FRAG_INTERLEAVE_ENABLED: bool = <bool>;`（直前行に #[cfg] 属性）。
read_flag_default() { # read_flag_default <facade_path> -> true|false|(空)
  local tile="$1/../backend-metal/src/tile.rs"
  [[ -f "${tile}" ]] || return 0
  sed -nE 's/^[[:space:]]*pub(\(crate\))?[[:space:]]+const[[:space:]]+FRAG_INTERLEAVE_ENABLED[[:space:]]*:[[:space:]]*bool[[:space:]]*=[[:space:]]*(true|false)[[:space:]]*;.*/\2/p' "${tile}" | head -n1
}
BEFORE_FLAG_DEFAULT="$(read_flag_default "${BEFORE_FACADE}")"
AFTER_FLAG_DEFAULT="$(read_flag_default "${AFTER_FACADE}")"
if [[ "${BEFORE_FLAG_DEFAULT}" != "false" || "${AFTER_FLAG_DEFAULT}" != "true" ]]; then
  echo "error: FRAG_INTERLEAVE_ENABLED mismatch (before=${BEFORE_FLAG_DEFAULT:-unreadable}, after=${AFTER_FLAG_DEFAULT:-unreadable}); expected before=false / after=true" >&2
  exit 1
fi

verify_no_untracked_build_inputs() { # verify_no_untracked_build_inputs <arm> <root>
  local arm=$1 root=$2 untracked ignored_hits
  untracked="$(git -C "${root}" ls-files --others --exclude-standard)" || {
    echo "error: ${arm} checkout の未追跡ファイル一覧を取得できません" >&2
    exit 1
  }
  if [[ -n "${untracked}" ]]; then
    echo "error: ${arm} checkout に未追跡ファイルがあります（ビルド影響の恐れ）:" >&2
    printf '%s\n' "${untracked}" | head -20 >&2
    exit 1
  fi
  ignored_hits="$(git -C "${root}" ls-files --others --ignored --exclude-standard)" || {
    echo "error: ${arm} checkout の無視済みファイル一覧を取得できません" >&2
    exit 1
  }
  ignored_hits="$(printf '%s\n' "${ignored_hits}" | grep -vE '^(target[^/]*|scripts/bench/[^ ]*/target[^/]*)/' |
    grep -E '(^|/)\.cargo/|(^|/)rust-toolchain(\.toml)?$|(^|/)Cargo\.(toml|lock)$|(^|/)build\.rs$|\.rs$' || true)"
  if [[ -n "${ignored_hits}" ]]; then
    echo "error: ${arm} checkout に gitignore 済みのビルド影響ファイルがあります:" >&2
    printf '%s\n' "${ignored_hits}" | head -20 >&2
    exit 1
  fi
}

verify_same_commit_and_diff() {
  local before_root after_root bh ah names changed
  before_root="$(git -C "${BEFORE_FACADE}" rev-parse --show-toplevel 2>/dev/null)" || {
    echo "error: before checkout is not a git worktree (${BEFORE_FACADE})" >&2
    exit 1
  }
  after_root="$(git -C "${AFTER_FACADE}" rev-parse --show-toplevel 2>/dev/null)" || {
    echo "error: after checkout is not a git worktree (${AFTER_FACADE})" >&2
    exit 1
  }
  bh="$(git -C "${before_root}" rev-parse HEAD)" || exit 1
  ah="$(git -C "${after_root}" rev-parse HEAD)" || exit 1
  if [[ -z "${bh}" || "${bh}" != "${ah}" ]]; then
    echo "error: before/after checkout の HEAD コミットが一致しません (before=${bh} after=${ah})" >&2
    exit 1
  fi
  if [[ -n "$(git -C "${before_root}" status --porcelain --untracked-files=no)" ]]; then
    echo "error: before checkout に未コミット変更があります" >&2
    exit 1
  fi
  verify_no_untracked_build_inputs "before" "${before_root}"
  verify_no_untracked_build_inputs "after" "${after_root}"
  names="$(git -C "${after_root}" diff --name-only HEAD)"
  if [[ "${names}" != "crates/backend-metal/src/tile.rs" ]]; then
    echo "error: after checkout の差分が tile.rs のみではありません: ${names:-(空)}" >&2
    exit 1
  fi
  changed="$(git -C "${after_root}" diff -U0 HEAD -- crates/backend-metal/src/tile.rs | grep -E '^[-+][^-+]' || true)"
  if [[ "$(printf '%s\n' "${changed}" | wc -l | tr -d ' ')" != "2" ]] ||
    ! printf '%s\n' "${changed}" | grep -qE '^-.*FRAG_INTERLEAVE_ENABLED.*= false;' ||
    ! printf '%s\n' "${changed}" | grep -qE '^\+.*FRAG_INTERLEAVE_ENABLED.*= true;'; then
    echo "error: after checkout の差分が FRAG_INTERLEAVE_ENABLED の false→true 1 行のみではありません" >&2
    exit 1
  fi
}
verify_same_commit_and_diff

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "error: Metal セル A/B は macOS 限定です" >&2
  exit 1
fi
DEVICE=metal

OUT="results/raw"
mkdir -p "${OUT}"
SKIP="${OUT}/skipped-3212-${DEVICE}-${LABEL}.log"
ANY_FAILED=0
record_failure() {
  ANY_FAILED=$((ANY_FAILED + 1))
}

# 同じ LABEL の既存系列は消去しない（run の差し替え・追加起動はしない。再実行は別 LABEL）。
for arm in before after candle; do
  f="${OUT}/results-${arm}-${LABEL}-${DEVICE}-gemm.jsonl"
  if [[ -e "${f}" ]]; then
    echo "error: 既存の計測記録 ${f} があります。差し替え禁止のため別の LABEL で実行してください" >&2
    exit 1
  fi
done
if [[ -e "${SKIP}" ]]; then
  echo "error: 既存の失敗記録 ${SKIP} があります。差し替え禁止のため別の LABEL で実行してください" >&2
  exit 1
fi
: >"${SKIP}"
for arm in before after candle; do
  : >"${OUT}/results-${arm}-${LABEL}-${DEVICE}-gemm.jsonl"
done

bench_fandhe_setup_lock_restore_trap

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

build_arm() { # build_arm <arm> <facade_path> <target_dir>
  local arm=$1 facade=$2 tdir=$3 patch_config exe tree_output
  patch_config="patch.crates-io.fandhe-ai.path=\"${facade}\""
  if ! cargo build --release -p bench-fandhe --target-dir "${tdir}" \
    --config "${patch_config}" >"${OUT}/build-${arm}-${LABEL}-${DEVICE}.out" 2>"${OUT}/build-${arm}-${LABEL}-${DEVICE}.err"; then
    tail -40 "${OUT}/build-${arm}-${LABEL}-${DEVICE}.err"
    echo "bench-fandhe BUILD FAILED (${arm}): $(tail -3 "${OUT}/build-${arm}-${LABEL}-${DEVICE}.err" | tr '\n' ' ')" >>"${SKIP}"
    exit 1
  fi
  exe="${tdir}/release/bench-fandhe"
  if [[ ! -f "${exe}" ]]; then
    echo "error: build ${arm}: exe not found (${exe})" >&2
    exit 1
  fi
  cp "${exe}" "${OUT}/bench-fandhe-3212-${arm}-${LABEL}"
  tree_output="$(cargo tree -p bench-fandhe --depth 1 --config "${patch_config}" 2>&1)"
  if ! echo "${tree_output}" | grep -qE 'fandhe-ai v[0-9.]+ \(.*crates/facade\)'; then
    echo "error: fandhe-ai did not resolve to the path-patched crates/facade (${arm}); cargo tree:" >&2
    echo "${tree_output}" >&2
    exit 1
  fi
  echo "${tree_output}" >"${OUT}/tree-3212-${arm}-${LABEL}.txt"
  echo "$(sha256_file "${OUT}/bench-fandhe-3212-${arm}-${LABEL}")  bench-fandhe-3212-${arm}-${LABEL}" >"${OUT}/sha-3212-${arm}-${LABEL}.txt"
}

echo "== build before ==(facade=${BEFORE_FACADE})"
build_arm before "${BEFORE_FACADE}" "target-ab-3212-${LABEL}-before"
echo "== build after  ==(facade=${AFTER_FACADE})"
build_arm after "${AFTER_FACADE}" "target-ab-3212-${LABEL}-after"
cat "${OUT}"/tree-3212-*-"${LABEL}".txt

# candle（record_only。candle 比の算出用。fandhe 側 patch とは無関係）をビルドして固定する。
if ! cargo build --release -p bench-candle --target-dir "target-ab-3212-${LABEL}-candle" \
  >"${OUT}/build-candle-${LABEL}-${DEVICE}.out" 2>"${OUT}/build-candle-${LABEL}-${DEVICE}.err"; then
  tail -40 "${OUT}/build-candle-${LABEL}-${DEVICE}.err"
  echo "bench-candle BUILD FAILED: $(tail -3 "${OUT}/build-candle-${LABEL}-${DEVICE}.err" | tr '\n' ' ')" >>"${SKIP}"
  exit 1
fi
cp "target-ab-3212-${LABEL}-candle/release/bench-candle" "${OUT}/bench-candle-3212-${LABEL}"
cargo tree -p bench-candle --depth 1 | grep -E 'candle-core v' >"${OUT}/candle-tree-3212-${LABEL}.txt" || true

BIN_BEFORE="${OUT}/bench-fandhe-3212-before-${LABEL}"
BIN_AFTER="${OUT}/bench-fandhe-3212-after-${LABEL}"
BIN_CANDLE="${OUT}/bench-candle-3212-${LABEL}"

# 反転が効いていない場合（同一バイナリ）を計測前に検出する（fail-closed。A/A を A/B と誤認しない）。
sha_before="$(awk '{print $1}' "${OUT}/sha-3212-before-${LABEL}.txt")"
sha_after="$(awk '{print $1}' "${OUT}/sha-3212-after-${LABEL}.txt")"
if [[ -z "${sha_before}" || "${sha_before}" == "${sha_after}" ]]; then
  echo "error: before/after のバイナリ sha256 が同一または取得不能です（定数の反転が効いていない）: before=${sha_before:-NA} after=${sha_after:-NA}" >&2
  exit 1
fi

# セル（arm × size × mode）の JSONL 行数。読み取り不能は空文字（呼び出し側が不足扱い）。
# round 対応（各 round で各セル +1 行）の検証と比較前の 5 件検証で共用する。
count_cell_rows() { # count_cell_rows <arm> <size> <mode> -> 行数 | (空)
  local f="${OUT}/results-${1}-${LABEL}-${DEVICE}-gemm.jsonl"
  [[ -f "${f}" ]] || return 0
  python3 -I -c '
import json, sys
path, size, mode = sys.argv[1], int(sys.argv[2]), sys.argv[3]
n = 0
for line in open(path, encoding="utf-8"):
    if not line.strip():
        continue
    r = json.loads(line)
    if r.get("task") == "gemm" and r.get("size") == size and (r.get("mode") or "fresh") == mode:
        n += 1
print(n)
' "${f}" "$2" "$3" 2>/dev/null || true
}

run_cell() { # run_cell <arm> <size> <mode>
  local arm=$1 size=$2 mode=$3 bin before_n after_n
  case "${arm}" in
    before) bin="${BIN_BEFORE}" ;;
    after) bin="${BIN_AFTER}" ;;
    candle) bin="${BIN_CANDLE}" ;;
    *) echo "error: unknown arm" >&2; exit 1 ;;
  esac
  echo "== gemm ${DEVICE} size=${size} mode=${mode} arm=${arm} =="
  before_n="$(count_cell_rows "${arm}" "${size}" "${mode}")"
  if ! "${bin}" --task gemm --size "${size}" --device "${DEVICE}" --mode "${mode}" \
    --out "${OUT}/results-${arm}-${LABEL}-${DEVICE}-gemm.jsonl" 2>"${OUT}/err-${arm}-${LABEL}-${DEVICE}.tmp"; then
    echo "arm=${arm} size=${size} mode=${mode} : $(cat "${OUT}/err-${arm}-${LABEL}-${DEVICE}.tmp")" >>"${SKIP}"
    echo "  -> FAILED (recorded in ${SKIP})"
    record_failure
  else
    after_n="$(count_cell_rows "${arm}" "${size}" "${mode}")"
    if [[ ! "${before_n}" =~ ^[0-9]+$ || ! "${after_n}" =~ ^[0-9]+$ || "${after_n}" -ne $((before_n + 1)) ]]; then
      echo "arm=${arm} size=${size} mode=${mode} : round の記録行数が不正（before=${before_n:-NA} after=${after_n:-NA}。+1 行を期待）" >>"${SKIP}"
      echo "  -> ROW COUNT MISMATCH (recorded in ${SKIP})"
      record_failure
    fi
  fi
  rm -f "${OUT}/err-${arm}-${LABEL}-${DEVICE}.tmp"
}

run_pair() { # run_pair <round> <size> <mode>
  local run_i=$1 size=$2 mode=$3
  if ((run_i % 2 == 1)); then
    run_cell before "${size}" "${mode}"
    run_cell after "${size}" "${mode}"
  else
    run_cell after "${size}" "${mode}"
    run_cell before "${size}" "${mode}"
  fi
}

LOAD_GATE=${AB_LOAD_GATE:-3.0}
RULE_LOAD_GATE=3.0
if [[ ! "${LOAD_GATE}" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
  echo "error: AB_LOAD_GATE must be numeric (got: ${LOAD_GATE})" >&2
  exit 1
fi
GATE_SERIES=official
if [[ "$(awk -v a="${LOAD_GATE}" -v r="${RULE_LOAD_GATE}" 'BEGIN{print (a==r)?1:0}')" != "1" ]]; then
  GATE_SERIES=reference
fi
GATE_LOG="${OUT}/load-gate-3212-${DEVICE}-${LABEL}.log"
: >"${GATE_LOG}"
echo "threshold=${LOAD_GATE} rule_threshold=${RULE_LOAD_GATE} series=${GATE_SERIES}" >>"${GATE_LOG}"
wait_load_gate() { # wait_load_gate <round>
  local waited=0 status=timeout l1 ok
  while ((waited < 1800)); do
    l1=$(sysctl -n vm.loadavg 2>/dev/null | awk '{print $2}')
    if [[ -z "${l1}" || "${l1}" =~ [^0-9.] ]]; then
      status=unavailable
      l1=NA
      break
    fi
    ok=$(awk -v a="${l1}" -v t="${LOAD_GATE}" 'BEGIN{print (a<t)?1:0}')
    if [[ "${ok}" == "1" ]]; then
      status=pass
      break
    fi
    sleep 30
    waited=$((waited + 30))
  done
  if [[ "${status}" == pass && "${GATE_SERIES}" == reference ]]; then status=pass-reference; fi
  echo "round${1} gate=${status} threshold=${LOAD_GATE} load1=${l1} waited_s=${waited} at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"${GATE_LOG}"
}

: >"${OUT}/uptime-3212-${DEVICE}-${LABEL}.log"
uptime | sed 's/.*load/load/' >>"${OUT}/uptime-3212-${DEVICE}-${LABEL}.log"

ab_round_cooldown_init "${OUT}/cooldown-3212-${DEVICE}-${LABEL}.log"
for run_i in $(seq 1 "${ROUNDS}"); do
  ab_round_cooldown "${run_i}"
  wait_load_gate "${run_i}"
  for size in ${SIZES}; do
    for mode in ${MODES}; do
      run_pair "${run_i}" "${size}" "${mode}"
    done
    # 同じ round の candle（fresh のみ。record_only）。
    run_cell candle "${size}" fresh
  done
  echo "round ${run_i}: $(uptime | sed 's/.*load/load/')" | tee -a "${OUT}/uptime-3212-${DEVICE}-${LABEL}.log"
done

# 比較前に全セルがちょうど ROUNDS 件であることを検証する（少標本の中央値で判定が出るのを防ぐ）。
CELLS_OK=1
for arm in before after; do
  for size in ${SIZES}; do
    for mode in ${MODES}; do
      n="$(count_cell_rows "${arm}" "${size}" "${mode}")"
      if [[ ! "${n}" =~ ^[0-9]+$ || "${n}" -ne "${ROUNDS}" ]]; then
        echo "compare: 判定不能（arm=${arm} size=${size} mode=${mode} の記録は ${n:-NA} 件。${ROUNDS} 件を要求）" >>"${SKIP}"
        CELLS_OK=0
      fi
    done
  done
done
if [[ "${CELLS_OK}" != "1" ]]; then
  echo "compare exit=NA (判定不能: 標本数不足)" | tee -a "${OUT}/compare-exit-3212-${DEVICE}-${LABEL}.log"
  record_failure
else
  python3 -I compare_gemm_ab.py --device "${DEVICE}" --task gemm --sizes metal-square5 --threshold 1.00 --per-run \
    --require-checksum-exact \
    "${OUT}/results-before-${LABEL}-${DEVICE}-gemm.jsonl" "${OUT}/results-after-${LABEL}-${DEVICE}-gemm.jsonl" \
    >"${OUT}/compare-gemm-3212-${DEVICE}-${LABEL}.md" 2>"${OUT}/compare-gemm-3212-${DEVICE}-${LABEL}.err"
  COMPARE_EXIT=$?
  # 終了コード: 0 = 非後退・3 = 後退セルあり（--require-checksum-exact 指定時は checksum 非完全一致・
  # 判定不能も 3）・2 = 入力不正。RULE 項 11「checksum 不一致は FAIL として停止」に従い、3 のうち
  # checksum 不一致・判定不能と読めるものは FAIL にする。純粋な速度後退は記録のみ。
  echo "compare exit=${COMPARE_EXIT}" | tee -a "${OUT}/compare-exit-3212-${DEVICE}-${LABEL}.log"
  case "${COMPARE_EXIT}" in
    0) ;;
    3)
      if { grep -qE '不一致|複合判定 ok|判定不能' "${OUT}/compare-gemm-3212-${DEVICE}-${LABEL}.md" ||
          grep -q 'checksum_exact_match=False' "${OUT}/compare-gemm-3212-${DEVICE}-${LABEL}.err"; }; then
        echo "compare: checksum 不一致または判定不能（exit=3。RULE 項 11 により FAIL）" >>"${SKIP}"
        record_failure
      fi
      ;;
    *)
      echo "compare: 比較不能（exit=${COMPARE_EXIT}）。$(tail -3 "${OUT}/compare-gemm-3212-${DEVICE}-${LABEL}.err" | tr '\n' ' ')" >>"${SKIP}"
      record_failure
      ;;
  esac
fi

echo "done. results in ${OUT} ; failures (if any) in ${SKIP}"
if [[ "${ANY_FAILED}" -gt 0 ]]; then
  echo "FAILED: ${ANY_FAILED} run(s) failed; see ${SKIP}" >&2
  exit 1
fi
