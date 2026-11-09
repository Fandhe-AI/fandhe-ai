#!/usr/bin/env bash
# #4153 の PR-2 の前に 1 回だけ使う xctrace の schema 検出（<metal-node> = Mac mini M4。PR-1 時点では未実行）。
# 役割: N=256 の小さな GEMM を Metal System Trace で記録し、`xctrace export --toc` を env/xctrace_toc.xml に保存する。
#   GPU 実行区間の表の schema 名・列名（RULE の AUX xctrace_schema／xctrace_*_col）を人間が確定するための材料で、
#   計測ではなく、ここで得た値は判定に使わない。xctrace が無い・失敗した場合は env/xctrace_probe.txt に unavailable と書いて非 0 で終わる
#   （RULE は formulas=F2 だけで固定する。README 参照）。
# 環境変数: ROOT=計測ツリー（framework-compare をビルド済みの git archive 展開先）  OUT=記録先 env ディレクトリ（絶対パス。未存在）
# bash 3.2／5 互換。外部入力をシェル文字列へ展開しない。
set -uo pipefail
validate_abs() {
  case "$2" in /*) ;; *) echo "$1 は絶対パスのみ: $2"; exit 2;; esac
  case "$2" in *\"*|*\'*|*\\*|*$'\n'*) echo "$1 に禁止文字を含む"; exit 2;; esac
}
ROOT="${ROOT:?計測ツリー}"; OUT="${OUT:?記録先 env ディレクトリ}"
validate_abs ROOT "${ROOT}"; validate_abs OUT "${OUT}"
[ ! -e "${OUT}" ] || { echo "${OUT} が既にある（上書き禁止）"; exit 2; }
mkdir -p "${OUT}" || exit 1
BC="${ROOT}/scripts/bench/framework-compare/target/release/bench-candle"
unavailable() { echo "unavailable: $1" > "${OUT}/xctrace_probe.txt"; echo "xctrace unavailable: $1"; exit 1; }
xcrun xctrace version > "${OUT}/xctrace_version.txt" 2>&1 || unavailable "xctrace version 失敗"
xcrun xctrace record --help > "${OUT}/xctrace_record_help.txt" 2>&1
[ -x "${BC}" ] || unavailable "bench-candle がない"
xcrun xctrace record --template 'Metal System Trace' --output "${OUT}/probe.trace" \
  --target-stdout "${OUT}/probe_stdout.txt" --launch -- "${BC}" --task gemm-transfer-split --device metal --size 256 --out "${OUT}/probe_results.jsonl" \
  > "${OUT}/xctrace_record.log" 2>&1 || unavailable "xctrace record 失敗（${OUT}/xctrace_record.log）"
xcrun xctrace export --input "${OUT}/probe.trace" --toc --output "${OUT}/xctrace_toc.xml" || unavailable "xctrace export --toc 失敗"
grep -q 'schema=' "${OUT}/xctrace_toc.xml" || unavailable "toc に表がない"
rm -rf "${OUT}/probe.trace"
sed -i.bak -e "s#${HOME}#<home>#g" -e "s#$(hostname)#<metal-node>#g" "${OUT}"/*.txt "${OUT}"/*.log "${OUT}"/*.xml && rm -f "${OUT}"/*.bak
echo "available: toc 保存済み。GPU 実行区間の表の schema 名・列名を ${OUT}/xctrace_toc.xml から確定して RULE の AUX へ書く" > "${OUT}/xctrace_probe.txt"
cat "${OUT}/xctrace_probe.txt"
