#!/usr/bin/env bash
# CI と共用する契約検査の一括実行面（make deps-forbidden / runner-policy /
# guardrail-regression / verification-gates / verification-gate-bench）。
# 検査本体は各 scripts/check-*.sh・run-*.sh にあり、ここでは ci.yml の各ジョブと同じ
# 「self-test → 本検査」の順序だけを定義する（検査ロジックは二重管理しない）。
# usage: ci-checks.sh <deps-forbidden|runner-policy|guardrail-regression|verification-gates|verification-gate-bench>
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./lib.sh
source "${DIR}/lib.sh"

# 検査ロジック自体の退行は cargo 有無に関わらず常時検出するため self-test は無条件で先に実行する。
case "${1:-}" in
  deps-forbidden)
    # 依存禁止リスト（deps-policy.md）の混入検査（TASK-1.2）。self-test → lock-all → tree。
    # lock-all の走査対象列挙は check-forbidden-deps.sh 側に集約済み。
    bash scripts/check-forbidden-deps.sh self-test
    bash scripts/check-forbidden-deps.sh lock-all
    if has_cargo; then
      bash scripts/check-forbidden-deps.sh tree
    else
      echo "skip: Cargo.toml 未追加のため cargo tree 検査をスキップ"
    fi
    ;;
  runner-policy)
    # self-hosted runner 逆戻り防止の fail-closed 契約検査（イシュー #472）。
    bash scripts/check-workflow-runner-policy.sh self-test
    bash scripts/check-workflow-runner-policy.sh check
    ;;
  guardrail-regression)
    # guardrail 判定器の 2 層検証（REQ-4/REQ-5。TASK-6.1a・イシュー #147）。
    bash scripts/run-guardrail-regression.sh self-test
    if has_cargo; then
      bash scripts/run-guardrail-regression.sh layer1
      bash scripts/run-guardrail-regression.sh layer2
    else
      echo "skip: Cargo.toml 未追加のため layer1/layer2 をスキップ"
    fi
    ;;
  verification-gates)
    # AI 自律メンテナンス検証ゲート build/test/clippy（TASK-6.1c・イシュー #199）。
    bash scripts/run-verification-gates.sh self-test
    if has_cargo; then
      bash scripts/run-verification-gates.sh build
      bash scripts/run-verification-gates.sh test
      bash scripts/run-verification-gates.sh clippy
    else
      echo "skip: Cargo.toml 未追加のため build/test/clippy をスキップ"
    fi
    ;;
  verification-gate-bench)
    # bench ゲート単体。実行時間が長いため verification-gates には含めず、
    # verification-gate-bench.yml（schedule／workflow_dispatch）と同一内容を再現する。
    bash scripts/run-verification-gates.sh self-test
    if has_cargo; then
      bash scripts/run-verification-gates.sh bench
    else
      echo "skip: Cargo.toml 未追加のため bench をスキップ"
    fi
    ;;
  *)
    echo "usage: $(basename "$0") <deps-forbidden|runner-policy|guardrail-regression|verification-gates|verification-gate-bench>" >&2
    exit 2
    ;;
esac
