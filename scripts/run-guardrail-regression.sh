#!/usr/bin/env bash
#
# guardrail 判定器の 2 層検証（TASK-6.1a・docs/spec/05-tasks.md、イシュー #147）を
# 実行する単一ソース。
#
# 背景（`docs/guardrail-self-repair-cli.md` 1.3 節・`.claude/rules/`）:
#   - 1 層目（REQ-4）: `expected_verdict` による判定器単体の検証。`guardrail eval`
#     サブコマンドが担い、除外リスト評価を含まない（同 CLI 仕様の設計制約）。
#   - 2 層目（REQ-5）: `expected_verdict_after_exclusions` による除外リスト適用後・
#     本番相当経路の検証。CLI ではなく回帰テスト側が担う。
# テスト実体（TASK-4.2b/4.3a/5.3a/5.3b/5.4a/5.4b。イシュー #110/#115/#126/#127/#129/#130）
# は既にマージ済みであり、本スクリプトはそれらを「push/PR 契機の CI ジョブ」として
# 一括実行する導線を新設する（テストの追加ではない）。
#
# 呼び出し元:
#   - .github/workflows/ci.yml の guardrail-regression ジョブ（self-test → layer1 → layer2
#     の順で呼ぶ）
#   - Makefile の guardrail-regression ターゲット（CI と同一判定をローカル再現）
#   - #148（TASK-6.1b・schedule 定期実行）が再利用する想定の単一ソース（本イシューでは
#     schedule 自体は追加しない）
#
# サブコマンド:
#   self-test  2 層検証を構成するテストファイル・fixture の存在を fail-closed 検証する
#              （テストの無言削除・リネームによる検証の空洞化を検知。cargo 不要）
#   layer1     `cargo test -p guardrail --locked` で 1 層目（REQ-4・判定器単体）のテストを
#              実行する
#   layer2     `cargo test -p guardrail --locked` で 2 層目（REQ-5・除外適用後）のテストを
#              実行する
#   all        self-test → layer1 → layer2 を直列実行する
#
# スコープ外（out-of-scope-tracking.md に従いイシュー #146 へ記録）:
#   TASK-5.2（除外ルール実装の完成。#123/#124）完了後に追加される「本番相当経路の結合検証
#   テスト」は、追加時に本スクリプトの LAYER2_TESTS へ追記が必要。追記運用は TASK-6.2
#   （#149）の変更時フロー文書に委ねる。
set -euo pipefail

# 1 層目（REQ-4・判定器単体）を構成する結合テストファイル（拡張子なし。--test 引数値）。
LAYER1_TESTS=(
  eval_harness
  labeled_changes_labels
)

# 2 層目（REQ-5・除外適用後）を構成する結合テストファイル。
LAYER2_TESTS=(
  label_invariant_empty_exclusions
  label_invariant_safe_side_monotonicity
  blindspot_g2_regression
  blindspot_g5_regression
)

GUARDRAIL_TESTS_DIR="crates/guardrail/tests"
FIXTURES_DIR="${GUARDRAIL_TESTS_DIR}/fixtures/labeled-changes"
# README「2 層ラベルモデル」節が定義する実 dataset の件数（labeled_changes_labels.rs・
# eval_harness.rs が前提とする固定件数。TASK-4.2b・#110 のピン留め値と同一）。
EXPECTED_FIXTURE_COUNT=15

usage() {
  echo "usage: $0 {self-test|layer1|layer2|all}" >&2
  exit 2
}

# 検査ロジック自体の退行（テストファイル・fixture の無言削除／リネーム）を、実行環境の
# cargo 有無に関わらず常時検出する（scripts/check-forbidden-deps.sh self-test と同一方針）。
cmd_self_test() {
  local failed=0

  for name in "${LAYER1_TESTS[@]}" "${LAYER2_TESTS[@]}"; do
    local path="${GUARDRAIL_TESTS_DIR}/${name}.rs"
    if [ -f "${path}" ]; then
      echo "OK: ${path} が存在します"
    else
      echo "NG: ${path} が見つかりません（2 層検証テストの欠落）" >&2
      failed=1
    fi
  done

  if [ -f "${FIXTURES_DIR}/README.md" ]; then
    echo "OK: ${FIXTURES_DIR}/README.md が存在します"
  else
    echo "NG: ${FIXTURES_DIR}/README.md が見つかりません" >&2
    failed=1
  fi

  if [ -d "${FIXTURES_DIR}/changes" ]; then
    local actual_count
    actual_count=$(find "${FIXTURES_DIR}/changes" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d '[:space:]')
    if [ "${actual_count}" -eq "${EXPECTED_FIXTURE_COUNT}" ]; then
      echo "OK: ${FIXTURES_DIR}/changes 配下に ${actual_count} 件（期待値と一致）"
    else
      echo "NG: ${FIXTURES_DIR}/changes 配下が ${actual_count} 件（期待値 ${EXPECTED_FIXTURE_COUNT} 件と不一致。fixture の追加・削除が未反映の可能性）" >&2
      failed=1
    fi
  else
    echo "NG: ${FIXTURES_DIR}/changes ディレクトリが見つかりません" >&2
    failed=1
  fi

  if [ "${failed}" -ne 0 ]; then
    echo "NG: self-test に失敗しました" >&2
    return 1
  fi
  echo "OK: self-test すべて成功"
}

cmd_layer1() {
  echo "1 層目（REQ-4・判定器単体）: guardrail eval の見逃し率 0%・誤検知率 30% 以下を検証"
  # --locked: Cargo.lock の意図しない書き換え・runner 汚染を防止する（他ジョブと同一方針）。
  cargo test -p guardrail --locked \
    --test eval_harness \
    --test labeled_changes_labels
}

cmd_layer2() {
  echo "2 層目（REQ-5・除外適用後）: 不変条件 (1)(2)・G2/G5 ブラインドスポットの安全側判定を検証"
  cargo test -p guardrail --locked \
    --test label_invariant_empty_exclusions \
    --test label_invariant_safe_side_monotonicity \
    --test blindspot_g2_regression \
    --test blindspot_g5_regression
}

cmd_all() {
  cmd_self_test
  cmd_layer1
  cmd_layer2
}

case "${1:-}" in
self-test)
  cmd_self_test
  ;;
layer1)
  cmd_layer1
  ;;
layer2)
  cmd_layer2
  ;;
all)
  cmd_all
  ;;
*)
  usage
  ;;
esac
