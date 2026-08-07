#!/usr/bin/env bash
#
# TASK-4.3b（イシュー #116）: v2 baseline
# （crates/guardrail/tests/fixtures/labeled-changes/baseline。tensor-core/autodiff
# 自作コア上のミニ MLP）に対する候補閾値の再実測を再現するための手順スクリプト。
#
# 呼び出し元: 人間・runtime-builder が手動で実行する（CI・Makefile には組み込まない。
# defect-injected な change.patch を実際に適用してビルド・テストする経路のため、
# 通常 CI ジョブに混入させない。ci.md・coding-rust.md の「実機依存分離」と同じ
# 精神を「欠陥注入テストデータの実行分離」に適用したもの）。
#
# 再実測の対象範囲（悉皆ではない。docs/guardrail-threshold-recalibration.md
# 「再実測の対象範囲」節・docs/guardrail-recalibration/v2-measured-signals.json
# 「not_remeasured」節に理由を記録済み）:
#   - G4-large-comment-refactor: lines_changed（`git diff --numstat` のみ。build 不要）
#   - D3-redundant-calc:         bench_median_pct（baseline 版・パッチ適用版を
#                                 各 5 回計測し中央値の変化率を採る。REQ-4「5 回
#                                 以上」・.claude/rules/coding-rust.md「5 回計測の
#                                 中央値」に準拠）
#   - D2-private-method:         build_ok（gate spot-check。cargo build の失敗を確認）
#   - D1-relu-sigmoid-swap:      test_ok（gate spot-check。cargo build 成功・
#                                 cargo test 失敗を確認）
# 他 11 件（D4/D5/G1/G2/G3/G5/S1〜S5）はゲート判定（ブール条件）または
# v1 実測で十分なマージンを持つベンチ改善側であり、候補閾値の合否を左右しない
# ため本スクリプトでは再実測しない。TASK-4.4（ベンチ計測モジュール）が
# 悉皆再実測の担当範囲。
#
# セキュリティ（A03。.claude/rules/security.md）: change_id は
# crates/guardrail/src/eval/dataset.rs::is_valid_change_id と同一の文字クラス
# 契約（[A-Za-z0-9._-]+・64 字以内）を満たす固定リテラルのみを扱う（本スクリプトは
# 外部入力を change_id として受け取らない）。baseline のコピー・パッチ適用は
# すべて $WORKDIR（mktemp -d で作成した一時ディレクトリ）内に限定する。
#
# 使い方:
#   bash scripts/recalibrate-guardrail-thresholds.sh
#
# 出力: $WORKDIR のパスと各シグナルの実測値を標準出力に表示する（記録は
# 呼び出し側が docs/guardrail-recalibration/v2-measured-signals.json へ
# 手動で反映する。本スクリプトは JSON を自動生成しない = 出力を無検証で
# コミットしない。.claude/rules/security.md A08「取り込み判断の根拠を
# 追跡可能にする」と同じ理由で、実測結果は人間の目視確認を経てから記録する）。

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_DIR="${REPO_ROOT}/crates/guardrail/tests/fixtures/labeled-changes"
BASELINE_SRC="${DATASET_DIR}/baseline"

WORKDIR="$(mktemp -d -t guardrail-recalibrate.XXXXXX)"
TARGET_DIR="${WORKDIR}/target"
export CARGO_TARGET_DIR="${TARGET_DIR}"
trap 'rm -rf "${WORKDIR}"' EXIT

echo "== WORKDIR: ${WORKDIR} =="

cp -r "${BASELINE_SRC}/." "${WORKDIR}/"

# baseline/Cargo.toml の path 依存（../../../../../{tensor-core,autodiff,bench-harness}）を
# $WORKDIR からの相対パスでは解決できないため、本リポジトリ内の絶対パスへ書き換える
# （fixture README「パッチの再生成・検証手順」節の手動手順を機械化したもの）。
sed -i.bak \
  -e "s#\.\./\.\./\.\./\.\./\.\./tensor-core#${REPO_ROOT}/crates/tensor-core#" \
  -e "s#\.\./\.\./\.\./\.\./\.\./autodiff#${REPO_ROOT}/crates/autodiff#" \
  -e "s#\.\./\.\./\.\./\.\./\.\./bench-harness#${REPO_ROOT}/crates/bench-harness#" \
  "${WORKDIR}/Cargo.toml"
rm -f "${WORKDIR}/Cargo.toml.bak"

git -C "${WORKDIR}" init -q
git -C "${WORKDIR}" add -A
git -C "${WORKDIR}" -c user.email=recalibrate@local -c user.name=recalibrate commit -q -m baseline

apply_patch() {
  local change_id="$1"
  git -C "${WORKDIR}" apply "${DATASET_DIR}/changes/${change_id}/change.patch"
}

restore_baseline() {
  git -C "${WORKDIR}" checkout -q -- .
}

bench_median_us() {
  # criterion の point estimate（`time: [lo mid hi]` の mid）を 5 回計測し、
  # シェルソート不要な awk のみで中央値を出す（bash 配列ソートの依存を避ける）。
  local mid values=()
  for _i in 1 2 3 4 5; do
    mid="$(cargo bench --bench forward_bench -- --measurement-time 1 --warm-up-time 1 2>/dev/null \
      | grep 'mlp_forward' | grep 'time:' \
      | sed -E 's/.*\[[0-9.]+ [a-zµ]+ ([0-9.]+) ([a-zµ]+) .*/\1 \2/')"
    values+=("${mid}")
  done
  printf '%s\n' "${values[@]}"
}

echo
echo "== G4-large-comment-refactor: lines_changed =="
apply_patch "G4-large-comment-refactor"
git -C "${WORKDIR}" diff --numstat
restore_baseline

echo
echo "== D2-private-method: build_ok gate spot-check =="
apply_patch "D2-private-method"
if (cd "${WORKDIR}" && cargo build) >"${WORKDIR}/d2-build.log" 2>&1; then
  echo "D2 build_ok=true（想定外: reject 期待）"
else
  echo "D2 build_ok=false（想定どおり reject 方向）"
fi
restore_baseline

echo
echo "== D1-relu-sigmoid-swap: test_ok gate spot-check =="
apply_patch "D1-relu-sigmoid-swap"
if (cd "${WORKDIR}" && cargo build) >"${WORKDIR}/d1-build.log" 2>&1; then
  echo "D1 build_ok=true"
  if (cd "${WORKDIR}" && cargo test) >"${WORKDIR}/d1-test.log" 2>&1; then
    echo "D1 test_ok=true（想定外: reject 期待）"
  else
    echo "D1 test_ok=false（想定どおり reject 方向）"
  fi
else
  echo "D1 build_ok=false（想定外: build は通る想定）"
fi
restore_baseline

echo
echo "== D3-redundant-calc: bench_median_pct（5 回計測 x baseline/patched） =="
echo "-- baseline --"
(cd "${WORKDIR}" && bench_median_us)
apply_patch "D3-redundant-calc"
echo "-- patched --"
(cd "${WORKDIR}" && bench_median_us)
restore_baseline

echo
echo "== 完了。実測値は docs/guardrail-recalibration/v2-measured-signals.json へ手動転記する =="
