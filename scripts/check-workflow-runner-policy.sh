#!/usr/bin/env bash
#
# self-hosted runner の逆戻り防止を fail-closed で検査する呼び出し面（イシュー #472）。
# #457 Phase 2（#465〜#469）で .github/workflows/ の全ジョブが runs-on: ubuntu-latest へ
# 移行完了した（.claude/rules/ci.md「runner（GitHub ホステッド既定）」節）が、以後
# self-hosted を再導入しても機械的に検知する仕組みがなかったため追加した契約検査。
# 本リポの既存契約検査パターン（scripts/check-forbidden-deps.sh: 単一ソースのスクリプト +
# scripts/testdata/ fixture + self-test サブコマンド + ci.yml ジョブと Makefile の共用）に
# 従う。
#
# 検査本体は scripts/check-workflow-runner-policy.py（python3 標準ライブラリのみ。
# 追加依存なしの自前 YAML サブセットパーサー方式）に分離してある。
# grep/パターン照合ではなく実 YAML パーサー方式を採るのは、PR #626 の codex-review で
# テキスト照合ベースの旧実装が YAML 表記トリック（エスケープ済みキー `"runs-on"`・
# 複数行 double-quoted キー・フローマッピング内キー・明示キー `?`・引用文字列内 `#` に
# よるコメント切り捨て誤動作等）で迂回可能と P0/P1 指摘されたため。デコード後の構造に
# 対して検査すれば表記揺れはパーサーが正規化し、パターン増築なしに構造的に遮断できる。
# PyYAML 等の外部依存は使わない（イシュー #472 再開時の追加指摘。依存の追加はユーザー
# 承認必須〈deps-policy.md〉であり、バージョン固定・導入ステップ・license-matrix 更新を
# 伴わない「ubuntu-latest 標準搭載」前提は fail-closed の趣旨に反する。詳細・対応範囲は
# .py 冒頭のパーサー方針コメントを参照）。
# 検査方針（唯一の許容形 ubuntu-latest 完全一致・runs-on / runner-label /
# post-feedback-runner-label を対象・codex-review の例外の扱い）の詳細は .py 冒頭を参照。
#
# 呼び出し元:
#   - .github/workflows/ci.yml の runner-policy ジョブ（self-test → check の順で呼ぶ）
#   - Makefile の runner-policy ターゲット（CI と同一判定をローカル再現）
#
# サブコマンド:
#   check      .github/workflows/ 配下の *.yml・*.yaml を対象に runner 宣言を検査する
#   self-test  scripts/testdata/ の固定 fixture に対しポジティブ（clean は pass）・
#              ネガティブ（違反・トリック fixture は fail）判定を行い、検査ロジック
#              自体の退行を検出する（受け入れ条件「self-hosted 再導入で CI が fail
#              する」の機械検証）
#
# fail-closed の前提: python3 が利用不可の場合は検査をスキップせず失敗させる
# （GitHub ホステッドランナー〈ubuntu-latest〉には標準搭載。外部パッケージの導入は
# 一切不要）。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKER="${SCRIPT_DIR}/check-workflow-runner-policy.py"

usage() {
  echo "usage: $0 {check|self-test}" >&2
  exit 2
}

# fail-closed: 検査本体の実行手段が無い場合、スキップして成功にせず失敗させる。
ensure_python() {
  if ! command -v python3 >/dev/null 2>&1; then
    echo "::error::python3 が見つかりません。runner 契約検査は実行できないため失敗とします（fail-closed）。python3 を導入してください。" >&2
    return 1
  fi
  if [ ! -f "${CHECKER}" ]; then
    echo "::error::検査本体（${CHECKER}）が見つかりません（fail-closed）" >&2
    return 1
  fi
}

check() {
  python3 "${CHECKER}" check
}

self_test() {
  local testdata="${SCRIPT_DIR}/testdata"
  # ポジティブ fixture: 唯一の許容形（ubuntu-latest 完全一致）のみで構成され pass する
  # こと（コメント内の self-hosted 言及・引用文字列内の `#` を誤検出しないことの検証
  # を兼ねる）。
  local clean_fixtures=(
    "${testdata}/workflow-runner-clean.yml"
  )
  # ネガティブ fixture: いずれも fail すること。素直な違反（forbidden）・許容形外
  # （unknown: larger runner・ブロックシーケンス・式展開）に加え、PR #626 の
  # codex-review が P0/P1 指摘した YAML 表記トリック迂回（trick-*。キー重複による
  # 禁止宣言の後勝ち上書き〈trick-duplicate-*〉を含む）と YAML パース不能
  # （invalid: fail-closed の検証）を含む。
  local violation_fixtures=(
    "${testdata}/workflow-runner-forbidden.yml"
    "${testdata}/workflow-runner-unknown.yml"
    "${testdata}/workflow-runner-trick-escaped-key.yml"
    "${testdata}/workflow-runner-trick-flow-mapping.yml"
    "${testdata}/workflow-runner-trick-multiline-key.yml"
    "${testdata}/workflow-runner-trick-comment-in-string.yml"
    "${testdata}/workflow-runner-trick-duplicate-block-key.yml"
    "${testdata}/workflow-runner-trick-duplicate-flow-key.yml"
    "${testdata}/workflow-runner-trick-duplicate-quoted-key.yml"
    "${testdata}/workflow-runner-invalid.yml"
  )
  local failed=0
  local f

  for f in "${clean_fixtures[@]}" "${violation_fixtures[@]}"; do
    if [ ! -f "${f}" ]; then
      echo "NG: self-test fixture が見つかりません（${f}）" >&2
      return 1
    fi
  done

  for f in "${clean_fixtures[@]}"; do
    if python3 "${CHECKER}" check-file "${f}" >/dev/null; then
      echo "self-test OK: $(basename "${f}") は pass する"
    else
      echo "self-test NG: $(basename "${f}") が誤って fail した（検査ロジックが誤検出している）" >&2
      failed=1
    fi
  done

  for f in "${violation_fixtures[@]}"; do
    if python3 "${CHECKER}" check-file "${f}" >/dev/null 2>&1; then
      echo "self-test NG: $(basename "${f}") が誤って pass した（検査ロジックが退行している）" >&2
      failed=1
    else
      echo "self-test OK: $(basename "${f}") は fail する"
    fi
  done

  if [ "${failed}" -ne 0 ]; then
    return 1
  fi
  echo "OK: self-test すべて pass"
}

main() {
  local subcommand="${1:-}"
  case "${subcommand}" in
    check)
      ensure_python
      check
      ;;
    self-test)
      ensure_python
      self_test
      ;;
    *)
      usage
      ;;
  esac
}

main "$@"
