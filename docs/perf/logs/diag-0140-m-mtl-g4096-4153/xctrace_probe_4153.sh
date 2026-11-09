#!/usr/bin/env bash
# #4153 の PR-2 の前に 1 回だけ使う xctrace の schema 検出（<metal-node> = Mac mini M4。PR-1 時点では未実行）。
# 役割: N=256 の小さな GEMM を Metal System Trace で記録し、`xctrace export --toc` を env/xctrace_toc.xml に保存する。
#   GPU 実行区間の表の schema 名・列名（RULE の AUX xctrace_schema／xctrace_*_col）を人間が確定するための材料で、
#   計測ではなく、ここで得た値は判定に使わない。xctrace が無い・失敗した場合は env/xctrace_probe.txt に unavailable と書いて非 0 で終わる
#   （RULE は formulas=F2 だけで固定する。README 参照）。
# 環境変数: ROOT=計測ツリー（framework-compare をビルド済みの git archive 展開先）  OUT=記録先 env ディレクトリ（絶対パス。未存在）
# マスク: OUT を作った後は成功・unavailable・失敗・シグナルのどの経路でも EXIT trap（finish）が
#   trace バンドルを消し、`to_values_4153.py --mask-dir` で env/ の全ファイルを置換する（対象と限界は同スクリプト冒頭）。
#   マスクに失敗（識別子が残る等）したら非 0 で終わる。保存するメッセージには OUT の絶対パスを書かない（OUT からの相対名）。
# bash 3.2／5 互換。外部入力をシェル文字列へ展開しない。
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)" || exit 2
TV="${HERE}/to_values_4153.py"
validate_abs() {
  case "$2" in /*) ;; *) echo "$1 は絶対パスのみ: $2"; exit 2;; esac
  case "$2" in *\"*|*\'*|*\\*|*$'\n'*) echo "$1 に禁止文字を含む"; exit 2;; esac
}
ROOT="${ROOT:?計測ツリー}"; OUT="${OUT:?記録先 env ディレクトリ}"
validate_abs ROOT "${ROOT}"; validate_abs OUT "${OUT}"
[ ! -e "${OUT}" ] || { echo "${OUT} が既にある（上書き禁止）"; exit 2; }
# マスクが使えない環境では何も保存しない（記録の後で気づかないよう、OUT を作る前に確かめる）
python3 -I "${TV}" --mask-ready > /dev/null || { echo "マスクの準備に失敗（to_values_4153.py --mask-ready）"; exit 2; }
mkdir -p "${OUT}" || exit 1
finish() {
  local rc=$?
  trap - EXIT
  rm -rf "${OUT}/probe.trace"  # 未マスクのバイナリ（パスを含む）。どの経路でも env/ に残さない
  python3 -I "${TV}" --mask-dir "${OUT}" || { echo "マスク失敗（env/ を公開しない）"; rc=3; }
  exit "${rc}"
}
trap finish EXIT
trap 'exit 130' INT TERM HUP
BC="${ROOT}/scripts/bench/framework-compare/target/release/bench-candle"
unavailable() { echo "unavailable: $1" > "${OUT}/xctrace_probe.txt"; echo "xctrace unavailable: $1"; exit 1; }
xcrun xctrace version > "${OUT}/xctrace_version.txt" 2>&1 || unavailable "xctrace version 失敗"
xcrun xctrace record --help > "${OUT}/xctrace_record_help.txt" 2>&1
[ -x "${BC}" ] || unavailable "bench-candle がない"
xcrun xctrace record --template 'Metal System Trace' --output "${OUT}/probe.trace" \
  --target-stdout "${OUT}/probe_stdout.txt" --launch -- "${BC}" --task gemm-transfer-split --device metal --size 256 --out "${OUT}/probe_results.jsonl" \
  > "${OUT}/xctrace_record.log" 2>&1 || unavailable "xctrace record 失敗（env/xctrace_record.log を参照）"
xcrun xctrace export --input "${OUT}/probe.trace" --toc --output "${OUT}/xctrace_toc.xml" || unavailable "xctrace export --toc 失敗"
grep -q 'schema=' "${OUT}/xctrace_toc.xml" || unavailable "toc に表がない"
# toc は目次のみで列の mnemonic・単位・区間の中身を含まない。GPU／Metal 系の表を --xpath で個別に export して保存し、
# 列定義と GPU 実行区間を人間が確認できる成果物にしてから trace を削除する（取得に失敗した表があれば unavailable）。
schemas=$(grep -o 'schema="[^"]*"' "${OUT}/xctrace_toc.xml" | sed -e 's/^schema="//' -e 's/"$//' | grep -Ei 'metal|gpu' | sort -u)
[ -n "${schemas}" ] || unavailable "toc に metal/gpu 系の表がない"
for sc in ${schemas}; do
  case "${sc}" in *[!A-Za-z0-9_-]*|'') continue;; esac
  xcrun xctrace export --input "${OUT}/probe.trace" \
    --xpath "/trace-toc/run[@number=\"1\"]/data/table[@schema=\"${sc}\"]" --output "${OUT}/table_${sc}.xml" \
    || unavailable "xctrace export 失敗: ${sc}"
done
# trace の削除と env/ のマスクは finish（EXIT trap）が行う。.xml は XML を壊さない形で置換される。
echo "available: toc と GPU／Metal 系の表（table_*.xml）を保存済み。列定義・GPU 実行区間を env/table_*.xml から確定して RULE の AUX へ書く" > "${OUT}/xctrace_probe.txt"
cat "${OUT}/xctrace_probe.txt"
