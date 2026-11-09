#!/usr/bin/env bash
# イシュー #4159 の Mac mini 計測セッション（RULE.txt が正）。1 本の切り離しセッションで直列に実行する。
#   (1) 実機ロック（外部の mkdir 方式。他イシューの計測と排他）  (2) run_ab_eager_release_cpu.sh（AB_ISSUE=4159・AB_PROFILE=macmini）
# ビルドは run スクリプトが round より前に直列で行うため、計測中のビルド並走は無い。
# 使い方: session-4159-macmini.sh <label>   （label は ^[A-Za-z0-9._-]+$）
# 配置: $HOME/work/r4159/{runner,before,after}
#   runner・before・after = 同一コミットの Git クローン（run スクリプトが rev-parse --show-toplevel と HEAD 一致を要求するため
#   アーカイブ展開木は不可。各ツリーに .rev-stamp を置く）。after だけ 4158 の on-arm.patch を当てる
#   回収: 終了時に runner の results/raw と compare・judge の成果物（ラベル一致分）を out-<label>/artifacts/ へ集約する
# 終了コードは run スクリプトの rc を反映する（終端行 `done. rc=` を steps.log に書く）。
# パスは固定定数のみで組み立て、外部入力（label を除く）を展開しない。ホスト名・ユーザー名は書かない。
export PATH=$HOME/.cargo/bin:$PATH
export LC_ALL=C
unset RAYON_NUM_THREADS CARGO_TARGET_DIR

label="${1:-}"
case "${label}" in '' | *[!A-Za-z0-9._-]*) echo "usage: $0 <label>  (label は [A-Za-z0-9._-]+)" >&2; exit 2 ;; esac

R=$HOME/work/r4159
T=$R/runner
O=$R/out-${label}
LOCK_DIR=$HOME/work/.fandhe-hw-lock
LOG=$O/steps.log

if [ -e "${O}" ]; then echo "output dir exists: ${O}" >&2; exit 5; fi
mkdir -p "${O}" || exit 1
step() { echo "$(date -u +%FT%TZ) $*" >>"${LOG}"; }

# ---- 実機ロック（取れるまで 60 秒おき。最大 3 時間）----
got=0
i=0
while [ "${i}" -lt 180 ]; do
  if mkdir "${LOCK_DIR}" 2>/dev/null; then got=1; break; fi
  sleep 60
  i=$((i + 1))
done
if [ "${got}" != 1 ]; then step "hwlock timeout"; echo "done. rc=lock" >>"${LOG}"; exit 3; fi
echo "#4159 macmini $(date -u +%FT%TZ)" >"${LOCK_DIR}/owner"
trap 'rm -rf "${LOCK_DIR}"; step "hwlock released"' EXIT
step "hwlock acquired"

cd "${T}/scripts/bench/framework-compare" || { step "cd failed"; echo "done. rc=cd" >>"${LOG}"; exit 1; }
AB_ISSUE=4159 AB_PROFILE=macmini \
  AB_BEFORE_FACADE_PATH="${R}/before/crates/facade" AB_AFTER_FACADE_PATH="${R}/after/crates/facade" \
  bash run_ab_eager_release_cpu.sh "4159-${label}" >>"${O}/run.log" 2>&1
rc=$?
step "run_ab rc=${rc}"
# JSONL・負荷ゲートログ・compare・judge は run スクリプトの cwd 配下に出るため、再判定用の証跡として out-<label>/ へ集約する
A=${O}/artifacts
F=${T}/scripts/bench/framework-compare
mkdir -p "${A}/raw"
# 部分一致 glob は label=trial と trial-long のような前方一致するラベルの別系列まで拾うため使わない。
# arm・device・task・拡張子を既知の集合から列挙し、ファイル名全体が一致するものだけを回収する（存在するものだけ）
L="4159-${label}"
collect() { [ -e "$1" ] && cp -p "$1" "$2" 2>>"${LOG}"; return 0; }
for arm in before after; do
  for task in gemm train infer gemmdiag traindiag; do
    collect "${F}/results/raw/results-${arm}-${L}-cpu-${task}.jsonl" "${A}/raw/"
  done
done
for stem in skipped load_gate uptime compare-exit; do
  collect "${F}/results/raw/${stem}-4159-cpu-${L}.log" "${A}/raw/"
done
for arm in before after; do
  collect "${F}/results/raw/tree-4159-${arm}-${L}.txt" "${A}/raw/"
  collect "${F}/results/raw/sha-4159-${arm}-${L}.txt" "${A}/raw/"
  collect "${F}/results/raw/eager-release-report-4159-${arm}-${L}.txt" "${A}/raw/"
done
collect "${F}/results/raw/env_info-4159-${L}.txt" "${A}/raw/"
for task in gemm train infer gemmdiag traindiag; do
  for ext in md err; do
    collect "${F}/compare-${task}-4159-cpu-${L}.${ext}" "${A}/"
  done
done
collect "${F}/judge-4159-${L}.md" "${A}/"
step "artifacts collected: $(ls "${A}/raw" 2>/dev/null | wc -l) raw files"
echo "done. rc=${rc}" >>"${LOG}"
exit "${rc}"
