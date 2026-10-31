#!/usr/bin/env bash
# 0.13.0 スコアボードの一括生成（計測キット。未実測）。
#
# 役割: 0.13.0 の実測 JSONL（Mac mini 系列 B・GB10 の 5 ラウンド中央値と run1〜run5）から、判定一覧 gen_0130.out と
# HTML（out/scoreboard_0130.html）を同じ生成器 `gen_0130.py` で作り、`check_0130.py` で機械照合する。
# 0.12.0 版（run_0120.sh）からの変更点: 単腕のため旧条件（swap 経路・gen_0120_oldarm.out・一時ツリー）を削除した。
# 判定規則・tolerance・baseline には触れない（判定経路は gen_0130.py の 1 本だけ）。
#
# 使い方:
#   bash run_0130.sh              # 本番生成（gen_args.txt・unreached_0130.tsv に <要記入 が残っていれば gen_0130.py が止まる）
#   bash run_0130.sh skeleton     # 判定だけ行い、1.0 以下のセルの雛形 TSV（unreached_0130.template.tsv）を書く。
#                                 # 雛形を unreached_0130.tsv にコピーして、分類・理由・出典を実測に基づいて埋める
# 環境変数（ドライラン・検証用。既定は本番の配置）:
#   RUN_VER   run ディレクトリ内 JSONL 名の版部分・fandhe-ai 行の version の期待値（既定 0.13.0）
#   GB_DIR    GB10 の run1〜run5 と median5 を含むディレクトリ（既定 ../gb10）
#   M4_DIR    Mac mini 系列 B の run1〜run5 と median5 を含むディレクトリ（既定 ../../framework-compare-0.13.0-macmini-remeasure/macmini-series-b）
#   ARGS_FILE gen_args.txt の代わり／UNREACHED 理由表 TSV の代わり／OUT_DIR 出力先（既定は本ディレクトリ。ドライランはリポの外を指す）
# 前版（0.12.0）の入力 5 ファイルは sha256 を RULE.txt と同じ値で照合してから使う（一致しなければ止める）。
# macOS の bash 3.2 でも動くように書く（mapfile・xargs -d を使わない）。
set -euo pipefail

MODE="${1:-build}"
SB="$(cd "$(dirname "$0")" && pwd)"
D="$(cd "${SB}/.." && pwd)"                                   # framework-compare-0.13.0-remeasure
L="$(cd "${D}/.." && pwd)"                                    # docs/perf/logs
P120G="${L}/framework-compare-0.12.0-remeasure"
P120M="${L}/framework-compare-0.12.0-macmini-remeasure"
RUN_VER="${RUN_VER:-0.13.0}"
GB_DIR="${GB_DIR:-${D}/gb10}"
M4_DIR="${M4_DIR:-${L}/framework-compare-0.13.0-macmini-remeasure/macmini-series-b}"
ARGS_FILE="${ARGS_FILE:-${SB}/gen_args.txt}"
UNREACHED="${UNREACHED:-${SB}/unreached_0130.tsv}"
OUT_DIR="${OUT_DIR:-${SB}}"

# 0. 前版（0.12.0）入力の固定（RULE.txt に書いた sha256 と同じ値）
( cd "${L}" && shasum -a 256 -c - <<SHA
50e6ea58c7d219be1040e2c64438e48d76b794beca6608e8bced0f455c4b0e37  framework-compare-0.12.0-remeasure/gb10/results-dgx-0.12.0-median5.jsonl
081291a4d0ce4f61ce273cfb0ec78f99e52be4280c022eb5395b0a02217d9a51  framework-compare-0.12.0-remeasure/gb10/results-dgx-0.12.0-extra-median5.jsonl
99ec726c99ad73e5c47d8cd7d039357323772138b3f1927662185ba8df27f495  framework-compare-0.12.0-remeasure/gb10/results-dgx-py-0.12.0-median5.jsonl
a74e4f5ff0fed000d9a1b20790593a02c16ecdc5edde3f4ddc7eff10e50923e8  framework-compare-0.12.0-macmini-remeasure/macmini-series-b/results-macmini-0.12.0-median5.jsonl
d42335c701c7a3fc0dc337d080557253ddda95fd9649624fb453a1c35951ab55  framework-compare-0.12.0-macmini-remeasure/macmini-series-b/results-macmini-py-0.12.0-median5.jsonl
SHA
) > /dev/null || { echo "前版（0.12.0）入力の sha256 が RULE.txt の値と一致しない" >&2; exit 1; }

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
  --prev-out "${P120G}/scoreboard/gen_0120.out"
)

if [ "${MODE}" = "skeleton" ]; then
  python3 "${SB}/gen_0130.py" "${COMMON[@]}" "${ARGS[@]}" \
    --emit-unreached-skeleton "${OUT_DIR}/unreached_0130.template.tsv" > "${OUT_DIR}/gen_0130.skeleton.out"
  echo "雛形: ${OUT_DIR}/unreached_0130.template.tsv（判定一覧: ${OUT_DIR}/gen_0130.skeleton.out）"
  exit 0
fi

# 1. 判定一覧と HTML
python3 "${SB}/gen_0130.py" "${COMMON[@]}" "${ARGS[@]}" \
  --unreached-tsv "${UNREACHED}" \
  --out "${OUT_DIR}/out/scoreboard_0130.html" > "${OUT_DIR}/gen_0130.out.new"

# 2. コミット済みの判定一覧と一致するか（一致しなければ止める。どちらも直さない）
if [ -f "${OUT_DIR}/gen_0130.out" ] && ! cmp -s "${OUT_DIR}/gen_0130.out" "${OUT_DIR}/gen_0130.out.new"; then
  echo "MISMATCH: gen_0130.out が再生成結果と一致しない" >&2
  diff "${OUT_DIR}/gen_0130.out" "${OUT_DIR}/gen_0130.out.new" >&2 || true
  exit 1
fi
mv "${OUT_DIR}/gen_0130.out.new" "${OUT_DIR}/gen_0130.out"

# 3. HTML と判定一覧・TSV の機械照合
python3 "${SB}/check_0130.py" "${OUT_DIR}/out/scoreboard_0130.html" "${OUT_DIR}/gen_0130.out" \
  "${P120G}/scoreboard/gen_0120.out" "${UNREACHED}"
