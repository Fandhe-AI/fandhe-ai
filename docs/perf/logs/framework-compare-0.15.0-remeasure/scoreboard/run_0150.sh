#!/usr/bin/env bash
# 0.15.0 スコアボードの一括生成。
#
# 役割: 0.15.0 の実測 JSONL（Mac mini 系列 B・GB10 の 5 ラウンド中央値と run1〜run5）から、判定一覧 gen_0150.out と
# HTML（out/scoreboard_0150.html）を同じ生成器 `gen_0150.py` で作り、`check_0150.py` で機械照合する。
# 0.14.0 版（run_0140.sh）からの変更点: 版・入出力パスと、前版（0.14.0）の入力 5 本＋前版の判定一覧 gen_0140.out の sha256 照合
# （RULE.txt が固定した値）を足し、手順 1b（機体別の判定一覧との byte 一致。機体別 judge は 0.15.0 では置かない）を、
# 次の 2 つに置き換えた（所有者決定・#4179 issuecomment-6887862187）。いずれも fail-closed で、判定規則・tolerance・baseline には触れない。
#   A-1 字句照合: 判定関数（load・key・index・metric・better・paired_ratio・judge）が gen_0140.py と関数単位で同一（ast で抜き出して diff 0）
#   A-2 回帰: gen_0150.py に 0.14.0 の入力（0.14.0 の median5 と run1〜run5・前版 0.13.0 の median5・gen_0130.out・0.14.0 の gen_args.txt と
#       unreached_0140.tsv）を与えて出した判定一覧が、コミット済みの gen_0140.out と byte 一致する（版は --run-ver／--prev-ver で切り替え）
#
# 使い方:
#   bash run_0150.sh              # 本番生成（gen_args.txt・unreached_0150.tsv に <要記入 が残っていれば gen_0150.py が止まる）
#   bash run_0150.sh skeleton     # 判定だけ行い、1 位以外のセルの雛形 TSV（unreached_0150.template.tsv）を書く
# 環境変数（ドライラン・検証用。既定は本番の配置）:
#   RUN_VER   run ディレクトリ内 JSONL 名の版部分・fandhe-ai 行の version の期待値（既定 0.15.0）
#   GB_DIR    GB10 の run1〜run5 と median5 を含むディレクトリ（既定 ../gb10）
#   M4_DIR    Mac mini 系列 B の run1〜run5 と median5 を含むディレクトリ（既定 ../../framework-compare-0.15.0-macmini-remeasure/macmini-series-b）
#   ARGS_FILE gen_args.txt の代わり／UNREACHED 理由表 TSV の代わり／OUT_DIR 出力先（既定は本ディレクトリ。ドライランはリポの外を指す）
# macOS の bash 3.2 でも動くように書く（mapfile・xargs -d を使わない）。
set -euo pipefail

MODE="${1:-build}"
SB="$(cd "$(dirname "$0")" && pwd)"
D="$(cd "${SB}/.." && pwd)"                                   # framework-compare-0.15.0-remeasure
L="$(cd "${D}/.." && pwd)"                                    # docs/perf/logs
P140G="${L}/framework-compare-0.14.0-remeasure"
P140M="${L}/framework-compare-0.14.0-macmini-remeasure"
P130G="${L}/framework-compare-0.13.0-remeasure"
P130M="${L}/framework-compare-0.13.0-macmini-remeasure"
RUN_VER="${RUN_VER:-0.15.0}"
GB_DIR="${GB_DIR:-${D}/gb10}"
M4_DIR="${M4_DIR:-${L}/framework-compare-0.15.0-macmini-remeasure/macmini-series-b}"
ARGS_FILE="${ARGS_FILE:-${SB}/gen_args.txt}"
UNREACHED="${UNREACHED:-${SB}/unreached_0150.tsv}"
OUT_DIR="${OUT_DIR:-${SB}}"

if command -v shasum >/dev/null 2>&1; then SHA_CHECK="shasum -a 256 -c -"; else SHA_CHECK="sha256sum -c -"; fi

# 0. 前版（0.14.0）入力の固定。値は両機体の RULE.txt の「前版比」の節に固定した sha256 と同じ（下で RULE.txt に載っていることも照合する）
H_GB="205129ec14fa79c201c0647928db8fbf74dc3460f5283e7c49275053613897b2"
H_GBX="d838593eb6759f734f7e564f690336e40a176acf4884b357a043a4805df15f9a"
H_GBP="dd217b11d4dbe6beb9d9fb222efa467af42759c0d5a49fd00a895225fb5d7e84"
H_M4="ce89a7772b723714a910c7f1ae134fc30802abb82f2300995de4e169357c2416"
H_M4P="d381443dbe34c51461449ea3bc857d521eabc1bc90fa052d57948c50ad1e7245"
H_OUT="cdec4f544ba8ddd3b8d0e0c3fe5c5573e18a2df28fcb2e228d66b34ee35a715e"
for h in "${H_GB}" "${H_GBX}" "${H_GBP}" "${H_OUT}"; do
  grep -q "${h}" "${D}/RULE.txt" || { echo "RULE.txt（GB10）に前版参照の sha256 が無い: ${h}" >&2; exit 1; }
done
for h in "${H_M4}" "${H_M4P}" "${H_OUT}"; do
  grep -q "${h}" "${L}/framework-compare-0.15.0-macmini-remeasure/RULE.txt" || { echo "RULE.txt（Mac mini）に前版参照の sha256 が無い: ${h}" >&2; exit 1; }
done
( cd "${L}" && ${SHA_CHECK} <<SHA
${H_GB}  framework-compare-0.14.0-remeasure/gb10/results-dgx-0.14.0-median5.jsonl
${H_GBX}  framework-compare-0.14.0-remeasure/gb10/results-dgx-0.14.0-extra-median5.jsonl
${H_GBP}  framework-compare-0.14.0-remeasure/gb10/results-dgx-py-0.14.0-median5.jsonl
${H_M4}  framework-compare-0.14.0-macmini-remeasure/macmini-series-b/results-macmini-0.14.0-median5.jsonl
${H_M4P}  framework-compare-0.14.0-macmini-remeasure/macmini-series-b/results-macmini-py-0.14.0-median5.jsonl
${H_OUT}  framework-compare-0.14.0-remeasure/scoreboard/gen_0140.out
SHA
) > /dev/null || { echo "前版（0.14.0）入力の sha256 が RULE.txt の値と一致しない" >&2; exit 1; }
# 回帰（手順 1c）が使う 0.13.0 の入力 5 本（0.14.0 の run_0140.sh が固定した値）
( cd "${L}" && ${SHA_CHECK} <<SHA
6a0e3c7530b353dcb82662f0c08affa72b126533978084b053970239b1e22896  framework-compare-0.13.0-remeasure/gb10/results-dgx-0.13.0-median5.jsonl
12f8c0c607a9b3770e9e860b8b374c095893d52e5778019aff20613078623828  framework-compare-0.13.0-remeasure/gb10/results-dgx-0.13.0-extra-median5.jsonl
4a7416222d8cbe8b56d9aa96f9cf4825b871acaf2d5a8f270663f35dd1e0008a  framework-compare-0.13.0-remeasure/gb10/results-dgx-py-0.13.0-median5.jsonl
e288fd21d72b9d375e89aaed12b4195d1907b2bae71b9c541ff19013366c1f38  framework-compare-0.13.0-macmini-remeasure/macmini-series-b/results-macmini-0.13.0-median5.jsonl
f79622dd0e5bdc4d6de2a6ce18d1e84dba6c2fd74441768a8b0cf0b1644b7eaa  framework-compare-0.13.0-macmini-remeasure/macmini-series-b/results-macmini-py-0.13.0-median5.jsonl
SHA
) > /dev/null || { echo "0.13.0 入力（回帰用）の sha256 が固定値と一致しない" >&2; exit 1; }
echo "前版参照 sha256 ok: 0.14.0 median5 5 本・gen_0140.out（RULE.txt の値）・0.13.0 median5 5 本（回帰用）"

# gen_args.txt の「--key=value」行を配列に積む（# で始まる行と空行は飛ばす）
read_args() {
  ARGS=()
  while IFS= read -r line || [ -n "${line}" ]; do
    case "${line}" in ''|'#'*) continue ;; esac
    ARGS+=("${line}")
  done < "$1"
}
read_args "${ARGS_FILE}"

mkdir -p "${OUT_DIR}/out"

COMMON=(
  --m4 "${M4_DIR}/results-macmini-${RUN_VER}-median5.jsonl"
  --m4-py "${M4_DIR}/results-macmini-py-${RUN_VER}-median5.jsonl"
  --m4-runs "${M4_DIR}"
  --gb "${GB_DIR}/results-dgx-${RUN_VER}-median5.jsonl" --gb-extra "${GB_DIR}/results-dgx-${RUN_VER}-extra-median5.jsonl"
  --gb-py "${GB_DIR}/results-dgx-py-${RUN_VER}-median5.jsonl" --gb-runs "${GB_DIR}"
  --run-ver "${RUN_VER}"
  --prev-out "${P140G}/scoreboard/gen_0140.out"
)

if [ "${MODE}" = "skeleton" ]; then
  python3 "${SB}/gen_0150.py" "${COMMON[@]}" "${ARGS[@]}" \
    --emit-unreached-skeleton "${OUT_DIR}/unreached_0150.template.tsv" > "${OUT_DIR}/gen_0150.skeleton.out"
  echo "雛形: ${OUT_DIR}/unreached_0150.template.tsv（判定一覧: ${OUT_DIR}/gen_0150.skeleton.out）"
  exit 0
fi

# 1a. 字句照合（A-1。fail-closed）: 判定関数 7 本が gen_0140.py と関数単位で同一であること（ast で関数ブロックの行を抜き出して比べる）
python3 -I - "${P140G}/scoreboard/gen_0140.py" "${SB}/gen_0150.py" <<'PY'
import ast, sys
FUNCS = ('load', 'key', 'index', 'metric', 'better', 'paired_ratio', 'judge')
def funcs(path):
    text = open(path, encoding='utf-8').read()
    lines = text.splitlines()
    return {n.name: lines[n.lineno - 1:n.end_lineno] for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)}
old, new = funcs(sys.argv[1]), funcs(sys.argv[2])
bad = [f for f in FUNCS if f not in old or f not in new or old[f] != new[f]]
if bad:
    raise SystemExit(f'字句照合に失敗: gen_0140.py と違う（または欠けた）関数 {bad}')
print('字句照合 ok（gen_0140.py と diff 0）: ' + ' '.join(FUNCS))
PY

# 1c. 回帰（A-2。fail-closed）: 0.14.0 の入力と引数一式を gen_0150.py に与えた判定一覧が、コミット済みの gen_0140.out と byte 一致すること。
#     版は --run-ver 0.14.0・--prev-ver 0.13.0・前版の入力 0.13.0 で切り替える（判定ロジックは同じ関数のまま）。HTML は捨てる
REG="$(mktemp -d)"
trap 'rm -rf "${REG}"' EXIT
read_args "${P140G}/scoreboard/gen_args.txt"
python3 "${SB}/gen_0150.py" \
  --m4 "${P140M}/macmini-series-b/results-macmini-0.14.0-median5.jsonl" \
  --m4-py "${P140M}/macmini-series-b/results-macmini-py-0.14.0-median5.jsonl" \
  --m4-runs "${P140M}/macmini-series-b" \
  --gb "${P140G}/gb10/results-dgx-0.14.0-median5.jsonl" --gb-extra "${P140G}/gb10/results-dgx-0.14.0-extra-median5.jsonl" \
  --gb-py "${P140G}/gb10/results-dgx-py-0.14.0-median5.jsonl" --gb-runs "${P140G}/gb10" \
  --run-ver 0.14.0 --prev-ver 0.13.0 \
  --m4-prev "${P130M}/macmini-series-b/results-macmini-0.13.0-median5.jsonl" "${P130M}/macmini-series-b/results-macmini-py-0.13.0-median5.jsonl" \
  --gb-prev "${P130G}/gb10/results-dgx-0.13.0-median5.jsonl" "${P130G}/gb10/results-dgx-0.13.0-extra-median5.jsonl" \
  --prev-out "${P130G}/scoreboard/gen_0130.out" \
  "${ARGS[@]}" \
  --unreached-tsv "${P140G}/scoreboard/unreached_0140.tsv" \
  --out "${REG}/regress.html" > "${REG}/regress.out"
cmp -s "${REG}/regress.out" "${P140G}/scoreboard/gen_0140.out" \
  || { echo "MISMATCH: 0.14.0 入力での回帰の判定一覧が gen_0140.out と byte 一致しない" >&2; diff "${REG}/regress.out" "${P140G}/scoreboard/gen_0140.out" >&2 || true; exit 1; }
echo "回帰 ok: 0.14.0 入力の判定一覧（$(grep -vc '^tally ' "${REG}/regress.out") 行＋tally）が gen_0140.out と byte 一致"

# 1. 判定一覧と HTML（0.15.0）
read_args "${ARGS_FILE}"
python3 "${SB}/gen_0150.py" "${COMMON[@]}" "${ARGS[@]}" \
  --unreached-tsv "${UNREACHED}" \
  --out "${OUT_DIR}/out/scoreboard_0150.html" > "${OUT_DIR}/gen_0150.out.new"

NEW="${OUT_DIR}/gen_0150.out.new"
[ "$(grep -v '^tally ' "${NEW}" | wc -l | tr -d ' ')" = "27" ] || { echo "判定行が 27 行でない" >&2; exit 1; }
grep '^tally .* noise na total 27$' "${NEW}" > /dev/null || { echo "tally 行が noise na・total 27 の形でない" >&2; exit 1; }

# 2. コミット済みの判定一覧と一致するか（一致しなければ止める。どちらも直さない）
if [ -f "${OUT_DIR}/gen_0150.out" ] && ! cmp -s "${OUT_DIR}/gen_0150.out" "${OUT_DIR}/gen_0150.out.new"; then
  echo "MISMATCH: gen_0150.out が再生成結果と一致しない" >&2
  diff "${OUT_DIR}/gen_0150.out" "${OUT_DIR}/gen_0150.out.new" >&2 || true
  exit 1
fi
mv "${OUT_DIR}/gen_0150.out.new" "${OUT_DIR}/gen_0150.out"

# 3. HTML と判定一覧・TSV の機械照合
python3 "${SB}/check_0150.py" "${OUT_DIR}/out/scoreboard_0150.html" "${OUT_DIR}/gen_0150.out" \
  "${P140G}/scoreboard/gen_0140.out" "${UNREACHED}"
