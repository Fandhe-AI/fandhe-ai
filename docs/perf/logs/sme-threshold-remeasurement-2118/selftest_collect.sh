#!/bin/bash
# イシュー #2118: lib_trees.sh の sme2118_collect_r1r2 の fail-closed 契約（RULE.txt §13）の self-test。
# 役割: 一時ディレクトリに run_ab_sme_cpu.sh の成果物を模擬し、(1) 全て揃えば rc=0、(2) 必須成果物の欠損・
# 空ファイル・コピー（マスク）失敗のいずれでも非ゼロで返ることを確認する。実機・cargo・git は不要。
# 使い方: bash selftest_collect.sh（成功時 `selftest_collect ok`・失敗時は非ゼロ終了）。
# 検出範囲: 必須成果物の存在／非空とコピー失敗のみ。JSONL の中身の正しさは aggregate.py の責務。
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=./lib_trees.sh
. "${HERE}/lib_trees.sh"
TMP=$(mktemp -d) || exit 1
trap 'rm -rf "${TMP}"' EXIT
LABEL="2118-selftest-k64"
fail() { echo "selftest_collect FAIL: $*" >&2; exit 1; }

# make_bench <dir>: 必須成果物一式（+ 任意の uptime ログ + 収録対象外の bench バイナリ）を作る
make_bench() {
  local b=$1 t a
  mkdir -p "${b}/results/raw" || return 1
  for t in gemm train infer; do
    for a in before after; do echo '{"x":1}' >"${b}/results/raw/results-${a}-${LABEL}-cpu-${t}.jsonl"; done
    echo "| cell |" >"${b}/compare-${t}-1978-cpu-${LABEL}.md"
    : >"${b}/compare-${t}-1978-cpu-${LABEL}.err"
  done
  echo "round1 gate=pass" >"${b}/results/raw/load-gate-1978-cpu-${LABEL}.log"
  echo "compare task=gemm exit=0" >"${b}/results/raw/compare-exit-1978-cpu-${LABEL}.log"
  echo "up" >"${b}/results/raw/uptime-1978-cpu-${LABEL}.log"
  echo "bin" >"${b}/results/raw/bench-fandhe-${LABEL}"
}

run_case() { # run_case <name> <expect: ok|fail> [mutation command...]
  local name=$1 expect=$2 b="${TMP}/$1/bench" d="${TMP}/$1/dest" rc
  shift 2
  make_bench "${b}" || fail "${name}: 模擬成果物の作成失敗"
  mkdir -p "${d}"
  ( cd "${TMP}/${name}" && "$@" ) || fail "${name}: 前処理の失敗"
  sme2118_collect_r1r2 "${b}" "${LABEL}" "${d}" "${TMP}/${name}" 2>/dev/null
  rc=$?
  if [ "${expect}" = "ok" ] && [ "${rc}" -ne 0 ]; then fail "${name}: 全成果物が揃うのに rc=${rc}"; fi
  if [ "${expect}" = "fail" ] && [ "${rc}" -eq 0 ]; then fail "${name}: 失敗を検出できず rc=0"; fi
  [ ! -e "${d}/bench-fandhe-${LABEL}" ] || fail "${name}: bench バイナリを収録した"
}

run_case complete ok true
[ -f "${TMP}/complete/dest/results-before-${LABEL}-cpu-gemm.jsonl" ] || fail "complete: JSONL が収録されていない"
run_case missing_jsonl fail rm "bench/results/raw/results-after-${LABEL}-cpu-train.jsonl"
run_case empty_jsonl fail truncate -s 0 "bench/results/raw/results-before-${LABEL}-cpu-infer.jsonl"
run_case missing_compare_md fail rm "bench/compare-gemm-1978-cpu-${LABEL}.md"
run_case missing_gate_log fail rm "bench/results/raw/load-gate-1978-cpu-${LABEL}.log"
run_case missing_exit_log fail rm "bench/results/raw/compare-exit-1978-cpu-${LABEL}.log"
# コピー失敗の模擬: 収録先に同名のディレクトリを置き、リダイレクトを失敗させる
run_case copy_fails fail mkdir -p "dest/results-before-${LABEL}-cpu-gemm.jsonl"
run_case missing_bench_dir fail rm -rf bench
echo "selftest_collect ok"
