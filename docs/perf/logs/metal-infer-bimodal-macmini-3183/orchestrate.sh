#!/usr/bin/env bash
# イシュー #3183: Mac mini の infer Metal 二峰性診断の実機計測オーケストレータ。
# 判定規則は同ディレクトリの RULE.txt（第 1 ラウンド）と RULE2.txt（第 2 ラウンド）が正（いずれも実測前に固定）。
# `docs/perf/logs/metal-tape-build-infer-phase-2114/orchestrate.sh` と同型で、
# テストバイナリを 1 回だけビルドし、run ごとに条件の起動順をローテーションして独立 5 回実行する。
#
# 使い方:
#   orchestrate.sh macmini [--out <dir>]        第 1 ラウンド（RULE.txt。HEAD の R/D/Don/G の 4 腕）
#   orchestrate.sh macmini --r2 [--out <dir>]   第 2 ラウンド（RULE2.txt。HEAD 4 腕 + v0.11.0 3 腕 + bench 対照 Bc/BcT）
# 集計は `python3 -I aggregate.py <dir>`（第 1 ラウンド）／`python3 -I aggregate.py --rule 2 <dir>`（第 2 ラウンド）。
# 既存 run ファイルの上書きは拒否する。
#
# 出どころの担保（fail-closed）: 実行ツリー直下の .rev-stamp（1 行目が HEAD）を env_info の git_head に書き、
# .tree-manifest（`sha256  path` の一覧）を `shasum -a 256 -c` で照合する。どちらかが無い・不一致なら計測前に停止する。
# 転送ツリーは worktree 由来の gitfile（.git ファイル）が混入しうるため `git rev-parse` は使わない
# （第 1 ラウンドは git_head が空になった）。ツリーの作り方は stage-tree.sh。
#
# bash 3.2（macOS 標準）互換: 変数展開は ${VAR} 形式、連想配列は使わない。
set -euo pipefail

usage() { echo "usage: $0 macmini [--r2] [--out <dir>]" >&2; exit 2; }

host_kind="${1:-}"
[ -n "${host_kind}" ] || usage
shift
case "${host_kind}" in macmini) ;; *) usage ;; esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../../.." && pwd)"
round=1
out_dir=""
while [ $# -gt 0 ]; do
  case "$1" in
    --r2)
      round=2
      shift
      ;;
    --out)
      [ $# -ge 2 ] || usage
      out_dir="$2"
      shift 2
      ;;
    *) usage ;;
  esac
done
if [ -z "${out_dir}" ]; then
  if [ "${round}" = 2 ]; then out_dir="${script_dir}/${host_kind}-r2"; else out_dir="${script_dir}/${host_kind}"; fi
fi
mkdir -p "${out_dir}"
out_dir="$(cd "${out_dir}" && pwd)"

# 差し替え禁止: 既存の run ファイルがあれば中止する。
for n in 1 2 3 4 5; do
  for f in "${out_dir}/run${n}.jsonl" "${out_dir}/head/run${n}.jsonl" "${out_dir}/v0110/run${n}.jsonl" \
    "${out_dir}/bench_control/run${n}_Bc.jsonl" "${out_dir}/bench_control/run${n}_BcT.jsonl"; do
    if [ -e "${f}" ]; then
      echo "既存の ${f##*/}（run${n}）があるため中止する（差し替え禁止）" >&2
      exit 1
    fi
  done
done

# マスク: ホスト名は masked、$HOME と /Users/<名前> は <home>（RULE.txt・RULE2.txt のマスク規則）
mask() {
  sed -e "s|${HOME}|<home>|g" -e "s|/Users/[^/ ]*|<home>|g" -e "s|$(hostname)|masked|g" -e "s|$(hostname -s)|masked|g"
}

# 実行ツリーの出どころを確かめる。$1 = ツリーの根、$2 = 出力名の接頭辞（空 or v0110_）。
# 成功時に ${tree_head}（.rev-stamp の 1 行目）と ${tree_fp}（.tree-manifest の sha256）を設定する。
verify_tree() {
  local root="$1" prefix="$2" listed actual
  [ -f "${root}/.rev-stamp" ] || { echo "${root##*/}: .rev-stamp が無い。stage-tree.sh で作ったツリーを転送すること" >&2; return 1; }
  [ -f "${root}/.tree-manifest" ] || { echo "${root##*/}: .tree-manifest が無い" >&2; return 1; }
  tree_head="$(head -n 1 "${root}/.rev-stamp")"
  printf '%s' "${tree_head}" | grep -Eq '^[0-9a-f]{40}$' || { echo "${root##*/}: .rev-stamp の 1 行目が 40 桁の SHA でない" >&2; return 1; }
  (cd "${root}" && shasum -a 256 -c .tree-manifest >/dev/null 2>&1) || { echo "${root##*/}: .tree-manifest の照合に失敗（内容の不一致・欠落）" >&2; return 1; }
  # 一覧にないファイルの混入も拒否する（照合は一覧にあるファイルしか見ないため）
  listed="$(sed -e 's/^[0-9a-f]*  //' "${root}/.tree-manifest" | LC_ALL=C sort)"
  actual="$(cd "${root}" && find . -type f ! -name .rev-stamp ! -name .tree-manifest | sed -e 's|^\./||' | LC_ALL=C sort)"
  [ "${listed}" = "${actual}" ] || { echo "${root##*/}: ツリーに一覧外のファイルがある、または一覧のファイルが無い" >&2; return 1; }
  tree_fp="$(shasum -a 256 "${root}/.tree-manifest" | awk '{print $1}')"
  cp "${root}/.rev-stamp" "${out_dir}/${prefix}rev-stamp.txt"
  echo "${tree_fp}  .tree-manifest" >"${out_dir}/${prefix}tree_manifest.sha256"
}

# cargo の JSON メッセージからテストバイナリ（実行可能ファイル）の絶対パスを 1 つ取り出す。
# $1 = target.name、$2 = 種別 (test|lib)。
find_bin() {
  python3 -I -c '
import json, sys
name, kind = sys.argv[1], sys.argv[2]
exe = None
for line in sys.stdin:
    try:
        m = json.loads(line)
    except ValueError:
        continue
    if m.get("reason") != "compiler-artifact" or not m.get("executable"):
        continue
    t = m.get("target", {})
    if t.get("name") != name:
        continue
    if kind == "lib" and not (m.get("profile", {}).get("test") and "lib" in t.get("kind", [])):
        continue
    exe = m["executable"]
if exe is None:
    sys.exit(1)
print(exe)
' "$1" "$2"
}

verify_tree "${repo_root}" ""
head_rev="${tree_head}"
head_fp="${tree_fp}"

cd "${repo_root}"
facade_bin="$(cargo test --release --locked -p fandhe-ai --test metal_infer_bimodal_diag \
  --no-run --message-format=json 2>/dev/null | find_bin metal_infer_bimodal_diag test)"
backend_bin="$(cargo test --release --locked -p fandhe-ai-backend-metal --lib \
  --no-run --message-format=json 2>/dev/null | find_bin fandhe_ai_backend_metal lib)"
[ -x "${facade_bin}" ] || { echo "facade テストバイナリを特定できない" >&2; exit 1; }
[ -x "${backend_bin}" ] || { echo "backend-metal テストバイナリを特定できない" >&2; exit 1; }

v0_rev="" v0_fp="" v0_facade="" bench_bin="" v0_reason=""
if [ "${round}" = 2 ]; then
  mkdir -p "${out_dir}/head" "${out_dir}/v0110" "${out_dir}/bench_control"
  # v0.11.0 ツリー（HEAD の診断テスト 1 ファイルを重ねたもの）。ビルド不能・空き不足は「未実施（理由）」に倒す。
  v0_root="${V0_TREE:-${HOME}/work/rust-ai-library-run-v0110}"
  free_kib="$(df -k "${HOME}" | awk 'NR==2 {print $4}')"
  if [ "${free_kib}" -lt $((20 * 1024 * 1024)) ]; then
    v0_reason="ノードの空き容量が 20 GiB 未満（${free_kib} KiB）"
  elif ! verify_tree "${v0_root}" "v0110_"; then
    v0_reason="v0.11.0 ツリーの出どころ検証に失敗"
  else
    v0_rev="${tree_head}"
    v0_fp="${tree_fp}"
    if v0_json="$(cd "${v0_root}" && CARGO_TARGET_DIR="${V0_TARGET_DIR:-${HOME}/work/target-fandhe-ai-v0110}" \
      cargo test --release --locked -p fandhe-ai --test metal_infer_bimodal_diag \
      --no-run --message-format=json 2>"${out_dir}/v0110/build.err.raw")" \
      && v0_facade="$(printf '%s\n' "${v0_json}" | find_bin metal_infer_bimodal_diag test)" && [ -x "${v0_facade}" ]; then
      :
    else
      v0_reason="v0.11.0 ツリーのビルドに失敗（v0110/build.err 参照）"
      v0_facade=""
    fi
    tail -n 30 "${out_dir}/v0110/build.err.raw" | mask >"${out_dir}/v0110/build.err" || true
    rm -f "${out_dir}/v0110/build.err.raw"
  fi
  if [ -z "${v0_facade}" ]; then
    echo "${v0_reason}" >"${out_dir}/v0110/NOT_RUN.txt"
    echo "v0110 条件は未実施: ${v0_reason}" >&2
  fi
  # bench-fandhe（crates.io fandhe-ai =0.11.0・変更なし。独立 workspace・--locked）。target は別ディレクトリ。
  (cd "${repo_root}/scripts/bench/framework-compare" \
    && CARGO_TARGET_DIR="${FC_TARGET_DIR:-${HOME}/work/target-fandhe-fc}" cargo build --release --locked -p bench-fandhe >/dev/null 2>&1) \
    || { echo "bench-fandhe のビルドに失敗" >&2; exit 1; }
  (cd "${repo_root}/scripts/bench/framework-compare" && cargo tree --locked -p bench-fandhe --depth 1 | grep -E 'fandhe-ai v' >"${out_dir}/bench_control/tree.txt")
  grep -q 'fandhe-ai v0.11.0$' "${out_dir}/bench_control/tree.txt" || { echo "registry pin 0.11.0 を解決できていない" >&2; exit 1; }
  bench_bin="${FC_TARGET_DIR:-${HOME}/work/target-fandhe-fc}/release/bench-fandhe"
  [ -x "${bench_bin}" ] || { echo "bench-fandhe のバイナリが無い" >&2; exit 1; }
fi
build_end_epoch="$(date +%s)"

{
  echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "uname: $(uname -srm)"
  sysctl -n machdep.cpu.brand_string hw.ncpu hw.perflevel0.logicalcpu hw.perflevel1.logicalcpu 2>/dev/null || true
  echo "rustc: $(rustc --version)"
  echo "cargo: $(cargo --version)"
  echo "git_head: ${head_rev}"
  echo "tree_fingerprint: ${head_fp}"
  if [ "${round}" = 2 ]; then
    echo "v0110_git_head: ${v0_rev:-未実施}"
    echo "v0110_tree_fingerprint: ${v0_fp:-未実施}"
  fi
  echo "host_kind: ${host_kind}"
  echo "round: ${round}"
} 2>&1 | mask >"${out_dir}/env_info.txt"

load1() {
  uptime | sed -e 's/.*load averages*: *//' -e 's/,/ /g' | awk '{print $1}'
}
load5() {
  uptime | sed -e 's/.*load averages*: *//' -e 's/,/ /g' | awk '{print $2}'
}

# 負荷ゲート（RULE.txt ゲート節。load1 < 3.0 を 60 秒間隔で最大 20 回待つ）
: >"${out_dir}/load_gate_status.txt"
: >"${out_dir}/load_gate.log"
gate_wait() {
  local n="$1" i=0 l
  while [ "${i}" -lt 20 ]; do
    l="$(load1)"
    if ! printf '%s' "${l}" | grep -Eq '^[0-9]+([.][0-9]+)?$'; then
      echo "run${n} load1=${l} GATE-INVALID" >>"${out_dir}/load_gate.log"
      echo "run${n} gate=GATE-INVALID" >>"${out_dir}/load_gate_status.txt"
      return 1
    fi
    echo "run${n} load1=${l} try=${i}" >>"${out_dir}/load_gate.log"
    if awk -v l="${l}" 'BEGIN { exit !(l < 3.0) }'; then
      echo "run${n} gate=pass load1=${l}" >>"${out_dir}/load_gate_status.txt"
      return 0
    fi
    i=$((i + 1))
    sleep 60
  done
  echo "run${n} gate=GATE-TIMEOUT" >>"${out_dir}/load_gate_status.txt"
  return 1
}

# 第 1 ラウンドの腕の起動順（RULE.txt: run ごとにローテーション）
order_for() {
  case "$1" in
    1 | 5) echo "R D Don G" ;;
    2) echo "D Don G R" ;;
    3) echo "Don G R D" ;;
    4) echo "G R D Don" ;;
  esac
}

# 第 2 ラウンドの条件の起動順（RULE2.txt: 9 条件を 2×(n-1) 個ずつ左へ巡回。v0110 未実施なら v0 系を除く）
order_for_r2() {
  local n="$1" all i k idx out="" c
  all=(R D Don G v0R v0D v0Don Bc BcT)
  k=$(((n - 1) * 2 % 9))
  i=0
  while [ "${i}" -lt 9 ]; do
    idx=$(((i + k) % 9))
    c="${all[${idx}]}"
    case "${c}" in
      v0*) [ -n "${v0_facade}" ] || { i=$((i + 1)); continue; } ;;
    esac
    out="${out}${out:+ }${c}"
    i=$((i + 1))
  done
  echo "${out}"
}

# 診断テスト 1 腕を独立プロセスで実行し、DIAG_JSON 行を $4 へ追記する。$1 = 腕、$2 = run、$3 = facade バイナリ、$4 = 出力先。
run_arm() {
  local arm="$1" n="$2" fbin="$3" dest="$4" raw
  raw="$(mktemp "${TMPDIR:-/tmp}/bimodal3183.XXXXXX")"
  case "${arm}" in
    R) "${fbin}" --ignored --exact metal_infer_bimodal_arm_r --nocapture --test-threads=1 >"${raw}" 2>>"${out_dir}/run${n}.err.raw" ;;
    D) "${fbin}" --ignored --exact metal_infer_bimodal_arm_d --nocapture --test-threads=1 >"${raw}" 2>>"${out_dir}/run${n}.err.raw" ;;
    Don) "${fbin}" --ignored --exact metal_infer_bimodal_arm_d_on --nocapture --test-threads=1 >"${raw}" 2>>"${out_dir}/run${n}.err.raw" ;;
    G) "${backend_bin}" --ignored infer_bimodal_diag_per_iter_gpu_timestamps --nocapture --test-threads=1 >"${raw}" 2>>"${out_dir}/run${n}.err.raw" ;;
  esac
  grep -o 'DIAG_JSON {.*' "${raw}" | sed -e 's/^DIAG_JSON //' >>"${dest}"
  rm -f "${raw}"
}

# bench-fandhe を 1 プロセス実行して結果 1 行を $4 へ追記する。$1 = task、$2 = run、$3 = 条件名（ログ用）、$4 = 出力先。
run_bench() {
  local task="$1" n="$2" cond="$3" dest="$4"
  if ! "${bench_bin}" --task "${task}" --device metal --size 64 --mode reuse --out "${dest}" 2>>"${out_dir}/run${n}.err.raw"; then
    echo "run${n} ${cond}: bench-fandhe が失敗した" >&2
    return 1
  fi
}

run_cond_r2() {
  local c="$1" n="$2"
  case "${c}" in
    R | D | Don | G) run_arm "${c}" "${n}" "${facade_bin}" "${out_dir}/head/run${n}.jsonl" ;;
    v0R) run_arm R "${n}" "${v0_facade}" "${out_dir}/v0110/run${n}.jsonl" ;;
    v0D) run_arm D "${n}" "${v0_facade}" "${out_dir}/v0110/run${n}.jsonl" ;;
    v0Don) run_arm Don "${n}" "${v0_facade}" "${out_dir}/v0110/run${n}.jsonl" ;;
    Bc) run_bench infer "${n}" Bc "${out_dir}/bench_control/run${n}_Bc.jsonl" ;;
    BcT)
      # 系列 A／B の実行順（train reuse の直後に infer reuse）を再現。判定に使うのは infer の行。
      run_bench train "${n}" BcT-train "${out_dir}/bench_control/run${n}_BcT_train.jsonl"
      run_bench infer "${n}" BcT "${out_dir}/bench_control/run${n}_BcT.jsonl"
      ;;
  esac
}

: >"${out_dir}/run_meta.txt"
for n in 1 2 3 4 5; do
  gate_wait "${n}" || { echo "負荷ゲート失敗（run${n}）。系列は無効" >&2; exit 3; }
  if [ "${round}" = 2 ]; then
    order="$(order_for_r2 "${n}")"
  else
    order="$(order_for "${n}")"
  fi
  since_build=$(($(date +%s) - build_end_epoch))
  {
    echo "== run${n} before =="
    uptime
    echo "order: ${order}  since_build_s: ${since_build}"
    pmset -g therm 2>&1 || true
    pmset -g 2>&1 | grep -i 'lowpowermode' || true
    echo "-- CPU 使用率上位 8 プロセス（pcpu comm）--"
    ps -Ao pcpu,comm -r | head -n 9 || true
  } 2>&1 | mask >"${out_dir}/env_run${n}.txt"
  echo "run=${n} order=$(printf '%s' "${order}" | tr ' ' ',') since_build_s=${since_build} load1=$(load1) load5=$(load5)" >>"${out_dir}/run_meta.txt"
  : >"${out_dir}/run${n}.err.raw"
  if [ "${round}" = 2 ]; then
    : >"${out_dir}/head/run${n}.jsonl"
    [ -z "${v0_facade}" ] || : >"${out_dir}/v0110/run${n}.jsonl"
    for c in ${order}; do
      run_cond_r2 "${c}" "${n}"
    done
  else
    : >"${out_dir}/run${n}.jsonl"
    for arm in ${order}; do
      run_arm "${arm}" "${n}" "${facade_bin}" "${out_dir}/run${n}.jsonl"
    done
  fi
  {
    echo "== run${n} after =="
    uptime
    pmset -g therm 2>&1 || true
  } 2>&1 | mask >>"${out_dir}/env_run${n}.txt"
  # powermetrics は sudo なしで通る場合のみ（sudo は使わない。RULE.txt）
  if sudo -n true 2>/dev/null; then
    sudo -n powermetrics --samplers gpu_power -n 1 2>&1 | mask >"${out_dir}/power_run${n}.txt" || echo "未取得" >"${out_dir}/power_run${n}.txt"
  else
    echo "未取得（sudo -n 不可）" >"${out_dir}/power_run${n}.txt"
  fi
  mask <"${out_dir}/run${n}.err.raw" >"${out_dir}/run${n}.err"
  rm -f "${out_dir}/run${n}.err.raw"
done
if [ "${round}" = 2 ]; then
  echo "完了: ${out_dir}（次: python3 -I ${script_dir}/aggregate.py --rule 2 ${out_dir}）"
else
  echo "完了: ${out_dir}（次: python3 -I ${script_dir}/aggregate.py ${out_dir}）"
fi
