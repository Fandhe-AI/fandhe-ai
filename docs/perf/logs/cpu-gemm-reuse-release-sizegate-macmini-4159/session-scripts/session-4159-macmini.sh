#!/usr/bin/env bash
# イシュー #4159 の Mac mini 計測セッション（RULE.txt が正）。1 本の切り離しセッションで直列に実行する。
#   (1) 実機ロック（外部の mkdir 方式。他イシューの計測と排他）  (2) run_ab_eager_release_cpu.sh（AB_ISSUE=4159・AB_PROFILE=macmini）
# ビルドは run スクリプトが round より前に直列で行うため、計測中のビルド並走は無い。
# 使い方: session-4159-macmini.sh <label>   （label は ^[A-Za-z0-9._-]+$）
# 配置: $HOME/work/r4159/{runner,before,after}
#   runner = 部品を入れたコミットの git archive 展開木、before = main、after = 同一コミットに 4158 の on-arm.patch を当てた木
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
echo "done. rc=${rc}" >>"${LOG}"
exit "${rc}"
