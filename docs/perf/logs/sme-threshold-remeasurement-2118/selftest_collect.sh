#!/bin/bash
# イシュー #2118: lib_trees.sh の fail-closed 契約の self-test。
# 役割:
#   1. sme2118_collect_r1r2（RULE.txt §13・§14）: 一時ディレクトリに run_ab_sme_cpu.sh の成果物を模擬し、
#      全て揃えば rc=0、必須成果物の欠損・空ファイル・コピー（マスク）失敗のいずれでも非ゼロで返ること。
#   2. sme2118_tree_fingerprint／sme2118_prepare_trees（RULE.txt §1・§15）: 差分が mod.rs 1 件のときだけ
#      成功し tree_verify.txt を書くこと。非 C ロケール（SME2118_SELFTEST_LOCALE。既定 C.UTF-8）で、かつ
#      翻訳された文言を出す偽の diff を PATH 先頭に置いた状態でも結果が変わらないこと（diff の文言を解析しない）。
#   実機・cargo は不要（2. は一時 git リポジトリを使い、登録 sha をそのコミットへ差し替えて呼ぶ）。
# 使い方: bash selftest_collect.sh（成功時 `selftest_collect ok`・失敗時は非ゼロ終了）。
#   日本語ロケールでの確認例: localedef -i ja_JP -f UTF-8 <dir>/ja_JP.UTF-8 のうえで
#   LOCPATH=<dir> SME2118_SELFTEST_LOCALE=ja_JP.UTF-8 bash selftest_collect.sh
# 検出範囲: 必須成果物の存在／非空とコピー失敗、ツリー差分エントリの一覧のみ。JSONL の中身の正しさは aggregate.py の責務。
set -uo pipefail
# ロケール固定（RULE.txt §15）: awk の数値比較・case の文字範囲・sort 順・lscpu 等の出力を C に固定する
# （source する lib_trees.sh も同じ export を行う。source 前の処理を含めて固定するためここでも行う）。
export LC_ALL=C
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

# --- 2. ツリーの指紋差分（ロケール非依存）---
MOD_DIR="crates/backend-cpu/src/gemm_blis"
LOC=${SME2118_SELFTEST_LOCALE:-C.UTF-8}
SHIM="${TMP}/shim"
mkdir -p "${SHIM}" || fail "shim 作成失敗"
# 翻訳された diff を模擬する（旧実装は英語の `Files ... differ` を解析していた）。
# $2・$3 は生成する偽 diff スクリプト内で展開させるため、ここでは単引用符のまま書き出す。
# shellcheck disable=SC2016
printf '#!/bin/sh\necho "ファイル $2 と $3 は異なります"\nexit 1\n' >"${SHIM}/diff"
chmod +x "${SHIM}/diff"

# make_trees <dir> <mutation>: before／after を作り after にだけ変更を加える
make_trees() {
  local w=$1 m=$2
  mkdir -p "${w}/before/${MOD_DIR}" "${w}/before/docs" || return 1
  printf 'const SME_PRODUCTION_ENABLED: bool = false;\nconst SME_MIN_K: usize = 64;\n' >"${w}/before/${MOD_DIR}/mod.rs"
  echo doc >"${w}/before/docs/a.md"
  cp -R "${w}/before" "${w}/after" || return 1
  case "${m}" in
    none) ;;
    mod) echo '// x' >>"${w}/after/${MOD_DIR}/mod.rs" ;;
    mod_orig) echo '// x' >>"${w}/after/${MOD_DIR}/mod.rs"; cp "${w}/before/${MOD_DIR}/mod.rs" "${w}/after/${MOD_DIR}/mod.rs.orig" ;;
    mod_other) echo '// x' >>"${w}/after/${MOD_DIR}/mod.rs"; echo y >>"${w}/after/docs/a.md" ;;
    mod_dir) echo '// x' >>"${w}/after/${MOD_DIR}/mod.rs"; mkdir "${w}/after/docs/new" ;;
    mod_type) echo '// x' >>"${w}/after/${MOD_DIR}/mod.rs"; rm "${w}/after/docs/a.md"; mkdir "${w}/after/docs/a.md" ;;
    *) return 1 ;;
  esac
}

# fp_case <name> <mutation> <expect: ok|fail>: C と非 C ロケール（+ 偽 diff）の両方で同じ結果になること
fp_case() {
  local name=$1 m=$2 expect=$3 w="${TMP}/fp-$1" out_c out_l
  make_trees "${w}" "${m}" || fail "${name}: 模擬ツリーの作成失敗"
  out_c=$(sme2118_tree_fingerprint "${w}") || fail "${name}: 列挙が非ゼロ（C）"
  out_l=$(LC_ALL="${LOC}" PATH="${SHIM}:${PATH}" sme2118_tree_fingerprint "${w}") || fail "${name}: 列挙が非ゼロ（${LOC}）"
  [ "${out_c}" = "${out_l}" ] || fail "${name}: ロケールで結果が変わる（C=[${out_c}] ${LOC}=[${out_l}]）"
  if [ "${expect}" = "ok" ]; then
    [ "${out_c}" = "${SME2118_MOD_REL}" ] || fail "${name}: mod.rs 1 件のみなのに [${out_c}]"
  else
    [ "${out_c}" != "${SME2118_MOD_REL}" ] || fail "${name}: mod.rs 以外の差分を検出できない"
  fi
}
fp_case only_mod mod ok
fp_case no_diff none fail
fp_case orig_file mod_orig fail
fp_case other_file mod_other fail
fp_case extra_dir mod_dir fail
fp_case type_change mod_type fail

# prepare_trees の端から端まで（一時 git リポジトリ・合成パッチ K=128）。非 C ロケール + 偽 diff で呼ぶ。
# pt_case <name> <patch_extra: none|other> <expect: ok|fail>
pt_case() {
  local name=$1 extra=$2 expect=$3 r="${TMP}/pt-$1/repo" w="${TMP}/pt-$1/work" o="${TMP}/pt-$1/out" pf="${TMP}/pt-$1/k128.patch" base rc
  mkdir -p "${r}/${MOD_DIR}" "${r}/docs" || fail "${name}: repo 作成失敗"
  printf 'const SME_PRODUCTION_ENABLED: bool = false;\nconst SME_MIN_K: usize = 64;\n' >"${r}/${MOD_DIR}/mod.rs"
  echo doc >"${r}/docs/a.md"
  if ! { git -C "${r}" init -q && git -C "${r}" add -A \
    && git -C "${r}" -c user.name=t -c user.email=t@example.invalid commit -qm init; }; then
    fail "${name}: commit 失敗"
  fi
  base=$(git -C "${r}" rev-parse HEAD)
  if ! { cp -R "${r}/${MOD_DIR}" "${TMP}/pt-$1/b" && cp -R "${r}/${MOD_DIR}" "${TMP}/pt-$1/a"; }; then
    fail "${name}: パッチ作成失敗"
  fi
  printf 'const SME_PRODUCTION_ENABLED: bool = true;\nconst SME_MIN_K: usize = 128;\n' >"${TMP}/pt-$1/a/mod.rs"
  (cd "${TMP}/pt-$1" && diff -u "b/mod.rs" "a/mod.rs" | sed -e "s#^--- b/mod.rs.*#--- a/${MOD_DIR}/mod.rs#" -e "s#^+++ a/mod.rs.*#+++ b/${MOD_DIR}/mod.rs#") >"${pf}"
  if [ "${extra}" = "other" ]; then
    printf -- '--- a/docs/a.md\n+++ b/docs/a.md\n@@ -1 +1 @@\n-doc\n+doc2\n' >>"${pf}"
  fi
  ( SME2118_REGISTERED_BASE=${base}; LC_ALL="${LOC}" PATH="${SHIM}:${PATH}" sme2118_prepare_trees "${r}" "${w}" 128 "${pf}" "${o}" ) >/dev/null 2>&1
  rc=$?
  if [ "${expect}" = "ok" ]; then
    [ "${rc}" -eq 0 ] || fail "${name}: mod.rs 1 件のみなのに prepare_trees rc=${rc}"
    grep -qx "trees_ok head=${base} K=128" "${o}/tree_verify.txt" || fail "${name}: tree_verify.txt が無い／不一致"
    [ "$(cat "${o}/tree_diff.txt")" = "${SME2118_MOD_REL}" ] || fail "${name}: tree_diff.txt が mod.rs 1 行でない"
  else
    [ "${rc}" -ne 0 ] || fail "${name}: mod.rs 以外の差分で prepare_trees が成功した"
    [ ! -e "${o}/tree_verify.txt" ] || fail "${name}: 失敗時に tree_verify.txt を書いた"
  fi
}
pt_case prepare_ok none ok
pt_case prepare_other_file other fail
echo "selftest_collect ok"
