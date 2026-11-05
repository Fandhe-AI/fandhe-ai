#!/usr/bin/env bash
# 0.14.0 スコアボードの一括生成。
#
# 役割: 0.14.0 の実測 JSONL（Mac mini 系列 B・GB10 の 5 ラウンド中央値と run1〜run5）から、判定一覧 gen_0140.out と
# HTML（out/scoreboard_0140.html）を同じ生成器 `gen_0140.py` で作り、`check_0140.py` で機械照合する。
# 0.13.0 版（run_0130.sh）からの変更点: 版・入出力パス・前版（0.13.0）の入力 5 本の sha256 と、機体別の判定一覧との
# byte 一致検査（手順 3。fail-closed）を足した。判定規則・tolerance・baseline には触れない（判定経路は gen_0140.py の 1 本だけ）。
#
# 使い方:
#   bash run_0140.sh              # 本番生成（gen_args.txt・unreached_0140.tsv に <要記入 が残っていれば gen_0140.py が止まる）
#   bash run_0140.sh skeleton     # 判定だけ行い、1.0 以下のセルの雛形 TSV（unreached_0140.template.tsv）を書く。
#                                 # 雛形を unreached_0140.tsv にコピーして、分類・理由・出典を実測に基づいて埋める
# 環境変数（ドライラン・検証用。既定は本番の配置）:
#   RUN_VER   run ディレクトリ内 JSONL 名の版部分・fandhe-ai 行の version の期待値（既定 0.14.0）
#   GB_DIR    GB10 の run1〜run5 と median5 を含むディレクトリ（既定 ../gb10）
#   M4_DIR    Mac mini 系列 B の run1〜run5 と median5 を含むディレクトリ（既定 ../../framework-compare-0.14.0-macmini-remeasure/macmini-series-b）
#   ARGS_FILE gen_args.txt の代わり／UNREACHED 理由表 TSV の代わり／OUT_DIR 出力先（既定は本ディレクトリ。ドライランはリポの外を指す）
# 前版（0.13.0）の入力 5 ファイルは sha256 を RULE.txt と同じ値で照合してから使う（一致しなければ止める）。
# 判定一覧（27 行）は、Mac mini 13 行を framework-compare-0.14.0-macmini-remeasure/aggregate/gen_0140_macmini.out の判定行と、
# GB10 14 行を gb10/judge_0140.out の判定行と byte 一致させる。tally は両機体の tally の和になること。
# macOS の bash 3.2 でも動くように書く（mapfile・xargs -d を使わない）。
set -euo pipefail

MODE="${1:-build}"
SB="$(cd "$(dirname "$0")" && pwd)"
D="$(cd "${SB}/.." && pwd)"                                   # framework-compare-0.14.0-remeasure
L="$(cd "${D}/.." && pwd)"                                    # docs/perf/logs
P130G="${L}/framework-compare-0.13.0-remeasure"
P130M="${L}/framework-compare-0.13.0-macmini-remeasure"
RUN_VER="${RUN_VER:-0.14.0}"
GB_DIR="${GB_DIR:-${D}/gb10}"
M4_DIR="${M4_DIR:-${L}/framework-compare-0.14.0-macmini-remeasure/macmini-series-b}"
M4_OUT="${M4_OUT:-${L}/framework-compare-0.14.0-macmini-remeasure/aggregate/gen_0140_macmini.out}"
GB_OUT="${GB_OUT:-${D}/gb10/judge_0140.out}"
ARGS_FILE="${ARGS_FILE:-${SB}/gen_args.txt}"
UNREACHED="${UNREACHED:-${SB}/unreached_0140.tsv}"
OUT_DIR="${OUT_DIR:-${SB}}"

# 0. 前版（0.13.0）入力の固定（両機体の RULE.txt に書いた sha256 と同じ値）
if command -v shasum >/dev/null 2>&1; then SHA_CHECK="shasum -a 256 -c -"; else SHA_CHECK="sha256sum -c -"; fi
( cd "${L}" && ${SHA_CHECK} <<SHA
6a0e3c7530b353dcb82662f0c08affa72b126533978084b053970239b1e22896  framework-compare-0.13.0-remeasure/gb10/results-dgx-0.13.0-median5.jsonl
12f8c0c607a9b3770e9e860b8b374c095893d52e5778019aff20613078623828  framework-compare-0.13.0-remeasure/gb10/results-dgx-0.13.0-extra-median5.jsonl
4a7416222d8cbe8b56d9aa96f9cf4825b871acaf2d5a8f270663f35dd1e0008a  framework-compare-0.13.0-remeasure/gb10/results-dgx-py-0.13.0-median5.jsonl
e288fd21d72b9d375e89aaed12b4195d1907b2bae71b9c541ff19013366c1f38  framework-compare-0.13.0-macmini-remeasure/macmini-series-b/results-macmini-0.13.0-median5.jsonl
f79622dd0e5bdc4d6de2a6ce18d1e84dba6c2fd74441768a8b0cf0b1644b7eaa  framework-compare-0.13.0-macmini-remeasure/macmini-series-b/results-macmini-py-0.13.0-median5.jsonl
SHA
) > /dev/null || { echo "前版（0.13.0）入力の sha256 が RULE.txt の値と一致しない" >&2; exit 1; }

# gen_args.txt の「--key=value」行を配列に積む（# で始まる行と空行は飛ばす）
ARGS=()
while IFS= read -r line || [ -n "${line}" ]; do
  case "${line}" in ''|'#'*) continue ;; esac
  ARGS+=("${line}")
done < "${ARGS_FILE}"

mkdir -p "${OUT_DIR}/out"

COMMON=(
  --m4 "${M4_DIR}/results-macmini-${RUN_VER}-median5.jsonl"
  --m4-py "${M4_DIR}/results-macmini-py-${RUN_VER}-median5.jsonl"
  --m4-runs "${M4_DIR}"
  --gb "${GB_DIR}/results-dgx-${RUN_VER}-median5.jsonl" --gb-extra "${GB_DIR}/results-dgx-${RUN_VER}-extra-median5.jsonl"
  --gb-py "${GB_DIR}/results-dgx-py-${RUN_VER}-median5.jsonl" --gb-runs "${GB_DIR}"
  --run-ver "${RUN_VER}"
  --prev-out "${P130G}/scoreboard/gen_0130.out"
)

if [ "${MODE}" = "skeleton" ]; then
  python3 "${SB}/gen_0140.py" "${COMMON[@]}" "${ARGS[@]}" \
    --emit-unreached-skeleton "${OUT_DIR}/unreached_0140.template.tsv" > "${OUT_DIR}/gen_0140.skeleton.out"
  echo "雛形: ${OUT_DIR}/unreached_0140.template.tsv（判定一覧: ${OUT_DIR}/gen_0140.skeleton.out）"
  exit 0
fi

# 1. 判定一覧と HTML
python3 "${SB}/gen_0140.py" "${COMMON[@]}" "${ARGS[@]}" \
  --unreached-tsv "${UNREACHED}" \
  --out "${OUT_DIR}/out/scoreboard_0140.html" > "${OUT_DIR}/gen_0140.out.new"

# 1b. 機体別の判定一覧との整合（fail-closed）: 27 行のうち先頭 13 行（Mac mini）は ${M4_OUT} の判定行と、
#     後ろ 14 行（GB10）は ${GB_OUT} の判定行と byte 一致しなければならず、tally 行は両機体の tally の和でなければならない。
#     どちらも直さずに止まる（判定器が別経路の出力と食い違うのは、入力・関数・前版入力のどれかが違う印）
NEW="${OUT_DIR}/gen_0140.out.new"
verdict_lines() { grep -v '^tally ' "$1"; }
tally_of() { grep '^tally ' "$1"; }
[ "$(verdict_lines "${NEW}" | wc -l | tr -d ' ')" = "27" ] || { echo "判定行が 27 行でない" >&2; exit 1; }
verdict_lines "${NEW}" | sed -n '1,13p' | cmp -s - <(verdict_lines "${M4_OUT}") \
  || { echo "MISMATCH: Mac mini の 13 行が ${M4_OUT} と byte 一致しない" >&2; exit 1; }
verdict_lines "${NEW}" | sed -n '14,27p' | cmp -s - <(verdict_lines "${GB_OUT}") \
  || { echo "MISMATCH: GB10 の 14 行が ${GB_OUT} と byte 一致しない" >&2; exit 1; }
SUM="$( { tally_of "${M4_OUT}"; tally_of "${GB_OUT}"; } | awk '{w+=$2; n+=$3; l+=$4; i+=$6; t+=$10; if ($8 != "na") bad=1} END {if (bad || NR != 2) exit 1; print "tally", w, n, l, "invalid", i, "noise na total", t}' )" \
  || { echo "両機体の tally 行が 1 行ずつでない／noise が na でない" >&2; exit 1; }
[ "$(tally_of "${NEW}")" = "${SUM}" ] || { echo "MISMATCH: tally 行 [$(tally_of "${NEW}")] が両機体の和 [${SUM}] と違う" >&2; exit 1; }
echo "整合 ok: Mac mini 13 行・GB10 14 行が機体別の判定一覧と byte 一致・${SUM}"

# 2. コミット済みの判定一覧と一致するか（一致しなければ止める。どちらも直さない）
if [ -f "${OUT_DIR}/gen_0140.out" ] && ! cmp -s "${OUT_DIR}/gen_0140.out" "${OUT_DIR}/gen_0140.out.new"; then
  echo "MISMATCH: gen_0140.out が再生成結果と一致しない" >&2
  diff "${OUT_DIR}/gen_0140.out" "${OUT_DIR}/gen_0140.out.new" >&2 || true
  exit 1
fi
mv "${OUT_DIR}/gen_0140.out.new" "${OUT_DIR}/gen_0140.out"

# 3. HTML と判定一覧・TSV の機械照合
python3 "${SB}/check_0140.py" "${OUT_DIR}/out/scoreboard_0140.html" "${OUT_DIR}/gen_0140.out" \
  "${P130G}/scoreboard/gen_0130.out" "${UNREACHED}"
