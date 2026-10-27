#!/usr/bin/env bash
# 0.12.0 スコアボードの一括生成（イシュー #3274）。
#
# 役割: #3273 の実測 JSONL（5 run 中央値と run1〜run5）から、次の 3 つと HTML を同じ生成器 `gen_0120.py` で作る。
#   - gen_0120.out          新条件（正式値）の判定一覧（gen_0120.py の標準出力）
#   - gen_0120_oldarm.out   旧条件（#3272 直前の bench-fandhe）の判定一覧。fandhe-ai 行を旧腕の行に置き換えた
#                           入力（aggregate/swap_fandhe_arm.py の出力）を一時ツリーに並べ、同じ生成器をもう 1 回かける
#   - out/scoreboard_0120.html  新条件の HTML（CSS インライン・1 ファイル）。旧条件・0.11.0 の tally を併記する
# 判定規則・tolerance・baseline には触れない（判定経路は gen_0120.py の 1 本だけ）。
# 呼び出し元: ../README.md の「スコアボード」節。検証は check_0120.py（本スクリプトの最後で実行する）。
# 注記の引数（負荷・日付・skipped）は gen_args.txt の各行（実測ログから転記した値）をそのまま渡す。
# macOS の bash 3.2 でも動くように書く（mapfile・xargs -d を使わない）。
set -euo pipefail

SB="$(cd "$(dirname "$0")" && pwd)"
D="$(cd "${SB}/.." && pwd)"                                   # framework-compare-0.12.0-remeasure
M="$(cd "${D}/../framework-compare-0.12.0-macmini-remeasure" && pwd)"
P110="$(cd "${D}/../framework-compare-0.11.0-macmini-remeasure" && pwd)"
G="${D}/gb10"

# gen_args.txt の「--key=value」行を配列に積む（# で始まる行と空行は飛ばす）
ARGS=()
while IFS= read -r line || [ -n "${line}" ]; do
  case "${line}" in ''|'#'*) continue ;; esac
  ARGS+=("${line}")
done < "${SB}/gen_args.txt"

mkdir -p "${SB}/out"

# 1. 新条件（正式値）。旧条件腕の fandhe-ai 中央値はセル内の注記だけに使う（--m4-old／--gb-old）
python3 "${SB}/gen_0120.py" \
  --m4 "${M}/macmini-series-b/results-macmini-0.12.0-median5.jsonl" \
  --m4-py "${M}/macmini-series-b/results-macmini-py-0.12.0-median5.jsonl" \
  --m4-alt "${M}/macmini-series-a/results-macmini-0.12.0-median5.jsonl" \
  --m4-py-alt "${M}/macmini-series-a/results-macmini-py-0.12.0-median5.jsonl" \
  --m4-runs "${M}/macmini-series-b" --m4-alt-runs "${M}/macmini-series-a" \
  --gb "${G}/results-dgx-0.12.0-median5.jsonl" --gb-extra "${G}/results-dgx-0.12.0-extra-median5.jsonl" \
  --gb-py "${G}/results-dgx-py-0.12.0-median5.jsonl" --gb-runs "${G}" \
  --m4-old "${M}/macmini-series-b/results-macmini-old-0.12.0-median5.jsonl" \
  --gb-old "${G}/results-dgx-old-0.12.0-median5.jsonl" \
  --unreached-tsv "${SB}/unreached_0120.tsv" \
  --oldarm-out "${SB}/gen_0120_oldarm.out" \
  --prev-out "${P110}/scoreboard/gen_0110_macmini.out" \
  "${ARGS[@]}" \
  --out "${SB}/out/scoreboard_0120.html" > "${SB}/gen_0120.out.new"

# 2. 旧条件。run ディレクトリの主 JSONL を旧腕置換版にした一時ツリーを作る。
#    GB10 の置換版（120 行）は extra（CPU GEMM reuse 等）を含むため、extra は空ファイルにする（重複キーは index() が止める）。
#    Mac mini は系列 A・B の両方を置換する（ノイズ帯の判定も旧腕同士で行う）。
T="$(mktemp -d)"
trap 'rm -rf "${T}"' EXIT
for i in 1 2 3 4 5; do
  mkdir -p "${T}/gb10/run${i}"
  cp "${G}/run${i}/results-dgx-oldarm-swapped-0.12.0.jsonl" "${T}/gb10/run${i}/results-dgx-0.12.0.jsonl"
  : > "${T}/gb10/run${i}/results-dgx-0.12.0-extra.jsonl"
  cp "${G}/run${i}/results-dgx-py-0.12.0.jsonl" "${T}/gb10/run${i}/results-dgx-py-0.12.0.jsonl"
  for s in a b; do
    mkdir -p "${T}/series-${s}/run${i}"
    cp "${M}/macmini-series-${s}/run${i}/results-macmini-oldarm-swapped-0.12.0.jsonl" "${T}/series-${s}/run${i}/results-macmini-0.12.0.jsonl"
    cp "${M}/macmini-series-${s}/run${i}/results-macmini-py-0.12.0.jsonl" "${T}/series-${s}/run${i}/results-macmini-py-0.12.0.jsonl"
  done
done
: > "${T}/empty.jsonl"
python3 "${SB}/gen_0120.py" \
  --m4 "${M}/macmini-series-b/results-macmini-oldarm-swapped-0.12.0-median5.jsonl" \
  --m4-py "${M}/macmini-series-b/results-macmini-py-0.12.0-median5.jsonl" \
  --m4-alt "${M}/macmini-series-a/results-macmini-oldarm-swapped-0.12.0-median5.jsonl" \
  --m4-py-alt "${M}/macmini-series-a/results-macmini-py-0.12.0-median5.jsonl" \
  --m4-runs "${T}/series-b" --m4-alt-runs "${T}/series-a" \
  --gb "${G}/results-dgx-oldarm-swapped-0.12.0-median5.jsonl" --gb-extra "${T}/empty.jsonl" \
  --gb-py "${G}/results-dgx-py-0.12.0-median5.jsonl" --gb-runs "${T}/gb10" \
  "${ARGS[@]}" \
  --out "${T}/oldarm.html" > "${SB}/gen_0120_oldarm.out.new"

# 3. コミット済みの判定一覧と一致するか（一致しなければ止める。どちらも直さない）
rc=0
for f in gen_0120.out gen_0120_oldarm.out; do
  if [ -f "${SB}/${f}" ] && ! cmp -s "${SB}/${f}" "${SB}/${f}.new"; then
    echo "MISMATCH: ${f} が再生成結果と一致しない" >&2
    diff "${SB}/${f}" "${SB}/${f}.new" >&2 || true
    rc=1
  fi
done
if [ "${rc}" -ne 0 ]; then
  exit 1
fi
mv "${SB}/gen_0120.out.new" "${SB}/gen_0120.out"
mv "${SB}/gen_0120_oldarm.out.new" "${SB}/gen_0120_oldarm.out"

# 4. HTML と判定一覧・TSV の機械照合
python3 "${SB}/check_0120.py" "${SB}/out/scoreboard_0120.html" "${SB}/gen_0120.out" "${SB}/gen_0120_oldarm.out" \
  "${P110}/scoreboard/gen_0110_macmini.out" "${SB}/unreached_0120.tsv"
