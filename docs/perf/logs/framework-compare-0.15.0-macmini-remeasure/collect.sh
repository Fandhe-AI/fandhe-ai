#!/usr/bin/env bash
# 0.15.0 同席計測 Mac mini（#4177）: 系列 B の収録スクリプト（計測後に機体で、本スクリプトの置き場所＝リポジトリの clone で実行する）。
# 役割: 作業ディレクトリの生ログを本ディレクトリへ収録する。0.14.0 では手作業だったマスク・漏れ検査・SHA256SUMS を機械化し、
#   計測後の venv の freeze 収録と 0.14.0 の収録との cmp も行う（#4168 は freeze を収録していなかった）。
#   マスクは docs/perf/logs/prepost-0150-4168/recollect.sh の語境界つき版（scripts/mask_lib.sh）:
#   ホーム → <home>、ホスト名 → <metal-node>、ユーザー名 → <user>。実名・ホスト名はここに書かず、実行時に hostname・$HOME・id -un から取る。
# 収録先（本ディレクトリ直下。既にあれば止まる＝差し替えない）:
#   macmini-series-b/（SERIES_DIR の複写: run1〜run5・gate.log・foreign-gate/・build/）・loop-macmini.log・
#   env/（env_info_before／after・tree-fingerprint-before／after・pip-freeze-venv-{torch,tf}.txt・freeze-cmp.txt）・SHA256SUMS
# 手順（fail-closed。どれかが失敗したら何も収録しない）:
#   1. 系列が 5/5 完走（ループログに `ALL-DONE completed=5/5`）か確かめる（ALLOW_INCOMPLETE=1 のときだけ打ち切り系列も収録する。採用は不可）
#   2. ステージへ複写 → venv の freeze を取得 → マスク（txt/log/md/tsv/out/err。jsonl は書き換えない）
#   3. 0.14.0 の収録 freeze と cmp して freeze-cmp.txt に記録（不一致は最後に rc 5）
#   4. 漏れ検査（ホーム・ホスト名は部分一致、ユーザー名は語境界つき、IPv4）。ステージ（今回収録する分）を対象に 1 件でもあれば rc 1
#   5. 収録先へ移し、SHA256SUMS（README.md と自身を除く全ファイル）を書く
# 環境変数: SERIES_DIR=作業ディレクトリの macmini-0150b（絶対パス）・LOOP_LOG=ループログ（絶対パス）・ENV_SERIES_DIR=作業ディレクトリの env-series・
#   PY_VENV_DIR=計測に使った venv の親（macmini-0150-series.sh と同じ値。既定は SERIES_DIR の親）・任意 VENV_TORCH／VENV_TF（個別指定）・FREEZE_REF_DIR・ALLOW_INCOMPLETE。
# 終了コード: 0 = 収録完了・freeze 一致／1 = 失敗（何も収録しない）／5 = 収録完了だが freeze が 0.14.0 と不一致（所有者へ報告）。
# bash 3.2 互換（${VAR} 形式。bash 3.2 は `$VAR` の直後の全角文字を変数名に含めて展開が壊れる前例があるため必ず ${} を使う）。
set -uo pipefail
PROFILE=macmini; NODE_PH='<metal-node>'; SERIES_NAME=macmini-series-b
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
. "${HERE}/scripts/mask_lib.sh"
# shellcheck source=/dev/null
. "${HERE}/scripts/venv_preflight.sh"
die() { echo "error: $*" >&2; exit 1; }
abs() { case "$2" in /*) ;; *) die "$1 は絶対パス" ;; esac; }

SERIES_DIR="${SERIES_DIR:-}"; LOOP_LOG="${LOOP_LOG:-}"; ENV_SERIES_DIR="${ENV_SERIES_DIR:-}"
[ -n "${SERIES_DIR}" ] && [ -n "${LOOP_LOG}" ] && [ -n "${ENV_SERIES_DIR}" ] || die "SERIES_DIR・LOOP_LOG・ENV_SERIES_DIR が要る"
abs SERIES_DIR "${SERIES_DIR}"; abs LOOP_LOG "${LOOP_LOG}"; abs ENV_SERIES_DIR "${ENV_SERIES_DIR}"
PY_VENV_DIR="${PY_VENV_DIR:-$(dirname "${SERIES_DIR}")}"; abs PY_VENV_DIR "${PY_VENV_DIR}"
[ -d "${SERIES_DIR}" ] || die "SERIES_DIR が無い"
[ -f "${LOOP_LOG}" ] || die "LOOP_LOG が無い"
[ -d "${ENV_SERIES_DIR}" ] || die "ENV_SERIES_DIR が無い"
for n in 1 2 3 4 5; do [ -d "${SERIES_DIR}/run${n}" ] || { [ "${ALLOW_INCOMPLETE:-}" = 1 ] || die "run${n} が無い"; }; done
if ! grep -q '^ALL-DONE completed=5/5$' "${LOOP_LOG}"; then
  [ "${ALLOW_INCOMPLETE:-}" = 1 ] || die "ループログに ALL-DONE completed=5/5 が無い（5/5 完走だけを採る。打ち切り系列を残すだけなら ALLOW_INCOMPLETE=1）"
  echo "note: 打ち切り系列を収録する（採用しない）" >&2
fi
[ ! -e "${HERE}/${SERIES_NAME}" ] && [ ! -e "${HERE}/env" ] && [ ! -e "${HERE}/loop-macmini.log" ] && [ ! -e "${HERE}/SHA256SUMS" ] || die "収録先が既にある（差し替えない）"

H="${HOME}"; HN="$(hostname 2> /dev/null || true)"; HNS="$(hostname -s 2> /dev/null || true)"; UN="$(id -un)"
STAGE="$(mktemp -d "${HERE}/.collect-stage.XXXXXX")" || die "ステージを作れない"
trap 'rm -rf "${STAGE}"' EXIT
mkdir -p "${STAGE}/env" || die "mkdir"
cp -R "${SERIES_DIR}" "${STAGE}/${SERIES_NAME}" || die "SERIES_DIR の複写に失敗"
cp "${LOOP_LOG}" "${STAGE}/loop-macmini.log" || die "ループログの複写に失敗"
for f in env_info_before.txt env_info_after.txt tree-fingerprint-before.txt tree-fingerprint-after.txt; do
  [ -f "${ENV_SERIES_DIR}/${f}" ] || die "env に ${f} が無い"
  cp "${ENV_SERIES_DIR}/${f}" "${STAGE}/env/${f}" || die "env の複写に失敗"
done
# venv の freeze（計測後。0.14.0 と同じ形式）
vp_freeze macmini "${VENV_TORCH:-${PY_VENV_DIR}/venv-torch}" "${STAGE}/env/pip-freeze-venv-torch.txt" || die "venv-torch の freeze を取れない"
vp_freeze macmini "${VENV_TF:-${PY_VENV_DIR}/venv-tf}" "${STAGE}/env/pip-freeze-venv-tf.txt" || die "venv-tf の freeze を取れない"
mask_tree "${STAGE}" "${H}" "${HN}" "${HNS}" "${UN}" "${NODE_PH}" || die "マスクの適用に失敗。収録しない"
# 0.14.0 の収録 freeze との cmp（マスク後どうしを比べる）
REF="$(vp_ref_dir "${PROFILE}")"; DIFFS=0
: > "${STAGE}/env/freeze-cmp.txt"
for name in torch tf; do
  if [ -f "${REF}/pip-freeze-venv-${name}.txt" ] && cmp -s "${STAGE}/env/pip-freeze-venv-${name}.txt" "${REF}/pip-freeze-venv-${name}.txt"; then
    echo "venv-${name}: identical（0.14.0 の収録 pip-freeze-venv-${name}.txt と byte 一致）" >> "${STAGE}/env/freeze-cmp.txt"
  else
    echo "venv-${name}: DIFFERENT（0.14.0 の収録と不一致、または参照が無い）" >> "${STAGE}/env/freeze-cmp.txt"; DIFFS=$((DIFFS + 1))
  fi
done
# 漏れ検査（今回収録する分＝ステージ。recollect.sh と同じ範囲。scripts・README・RULE は PR のレビューで確かめる）
leaks="$(leak_count "${STAGE}" "${H}" "${HN}" "${HNS}" "${UN}")" || die "漏れ検査そのものが失敗。収録しない"
[ "${leaks}" = "0" ] || die "漏れ検査に失敗（${leaks} 項目。上の LEAK 行を見る）。収録しない"
mv "${STAGE}/${SERIES_NAME}" "${STAGE}/env" "${STAGE}/loop-macmini.log" "${HERE}/" || die "収録先への移動に失敗"
write_sums "${HERE}" || die "SHA256SUMS を作れない（収録先へは移動済み。中身を確かめる）"
echo "収録完了: ${SERIES_NAME}・env・loop-macmini.log・SHA256SUMS（漏れ 0 件）"
cat "${HERE}/env/freeze-cmp.txt"
[ "${DIFFS}" -eq 0 ] || { echo "freeze が 0.14.0 と不一致（${DIFFS} 件）。所有者へ報告する" >&2; exit 5; }
