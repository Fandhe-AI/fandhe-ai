#!/usr/bin/env bash
# switches.sh — Phase 3 施策スイッチの既定値スナップショット抽出（イシュー #2120）。
#
# 役割: framework-compare 再計測（orchestrate.sh）の腕 B／C のツリーから、Phase 3（親 #2099）で導入された
#   opt-in・既定 OFF スイッチの「ソース上の既定値」を機械抽出し `name=value file:line` で標準出力へ出す。
#   「Phase 3 で ADOPT 結線された施策が計測ツリーで実際に既定 ON か」を計測結果と切り離して証跡化する
#   （RULE.txt の前提のずれ・期待値の根拠。docs/perf/framework-compare-phase3-remeasure.md の採否スナップショット表）。
# 呼び出し元: orchestrate.sh（腕 B・C の各ツリーに対し 1 回。出力は switches-{B,C}.txt）。
# 使い方: switches.sh <tree> [--allow-missing]
#   通常（腕 C・HEAD）: 8 定数と 3 環境変数名のいずれかが見つからなければ非 0 終了（fail-closed）。
#     改名・削除でスナップショットが黙って欠けるのを防ぐ。
#   --allow-missing（腕 B・Phase 3 直前ツリー）: Phase 3 導入前のため定数が存在しないのは正常。
#     見つからない項目は `name=<absent> -` として記録する。
set -uo pipefail
TREE="${1:?usage: switches.sh <tree> [--allow-missing]}"
ALLOW_MISSING=0
[[ "${2:-}" == "--allow-missing" ]] && ALLOW_MISSING=1
[[ -d "${TREE}/crates" ]] || { echo "ERROR: ${TREE}/crates が無い" >&2; exit 1; }

CONSTS=(
  HOST_ARENA_DEFAULT_ENABLED
  REDUCTION_SEQUENTIAL_FALLBACK_ENABLED
  INFER_GRAPH_DEFAULT_ENABLED
  TRAIN_FORWARD_ENCODE_ONLY_DEFAULT_ENABLED
  METAL_DEVICE_VERIFY_CACHE_DEFAULT_ENABLED
  UNROLL_LOAD_ENABLED
  SME_PRODUCTION_ENABLED
  GB10_AFFINITY_ENABLED
)
ENVS=(
  FANDHE_AI_CUDA_READBACK_DEST
  FANDHE_AI_METAL_READBACK_DEST
  FANDHE_AI_CUDA_GRAPH_INFER
)
rc=0
for c in "${CONSTS[@]}"; do
  hit="$(grep -rnE "^[[:space:]]*(pub(\([a-z]+\))? )?const ${c}: bool = (true|false);" "${TREE}/crates" --include='*.rs' | head -1)"
  if [[ -z "${hit}" ]]; then
    if [[ "${ALLOW_MISSING}" -eq 1 ]]; then echo "${c}=<absent> -"; else echo "ERROR: 定数 ${c} が見つからない" >&2; rc=1; fi
    continue
  fi
  file="${hit%%:*}"; rest="${hit#*:}"; line="${rest%%:*}"
  val="$(echo "${hit}" | sed -E 's/.*= (true|false);.*/\1/')"
  echo "${c}=${val} ${file#"${TREE}/"}:${line}"
done
for e in "${ENVS[@]}"; do
  hit="$(grep -rnF "\"${e}\"" "${TREE}/crates" --include='*.rs' | head -1)"
  if [[ -z "${hit}" ]]; then
    if [[ "${ALLOW_MISSING}" -eq 1 ]]; then echo "${e}=<absent> -"; else echo "ERROR: 環境変数 ${e} の参照が見つからない" >&2; rc=1; fi
    continue
  fi
  file="${hit%%:*}"; rest="${hit#*:}"; line="${rest%%:*}"
  echo "${e}=<env opt-in・未設定で OFF> ${file#"${TREE}/"}:${line}"
done
exit "${rc}"
