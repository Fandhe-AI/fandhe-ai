#!/usr/bin/env bash
#
# AI 自律メンテナンス検証ゲート（`docs/spec/04-requirements.md` の「検証済み AI 自律
# メンテナンス検証ゲート」節〈前提条件節、2026-08-07 時点 L34〉、REQ-1 非機能要件
# 「CI/CD への統合」節「REQ-3〜REQ-6 のガードレール（build/test/clippy/bench）は CI 上で
# 自動実行可能な構成とすること」〈2026-08-07 時点 L292〉）を実行する単一ソース
# （TASK-6.1c・docs/spec/05-tasks.md、イシュー #199）。行番号は spec 更新で変動しうるため
# 節見出しを一次の参照点とする。
#
# 4 ゲートの定義（spec 準拠）:
#   1. build  : `cargo build`（型検査）
#   2. test   : `cargo test --release`（回帰テスト・数値精度検証）
#   3. clippy : `cargo clippy --all-targets -- -D warnings`（静的解析）
#   4. bench  : 性能回帰検出相当（v2 では計測系を bench-harness へ付け替え済み。
#               REQ-3 注記・docs/guardrail-self-repair-cli.md 1.4 節）
#
# 既存 test ジョブ（ci.yml。`cargo test --workspace --all-features`・debug ビルド）とは
# 責務が異なる点に注意する: 本スクリプトの test サブコマンドは spec のゲート定義どおり
# `--release` を使う（回帰テスト・数値精度検証。debug ビルドでは大規模ストレスケースの
# 実行時間が過大になるテストが既に release 実行を前提としている。coding-rust.md）。
#
# bench ゲートは `crates/backend-cpu/examples/gemm_bench.rs`（bench-harness 計測プロトコル・
# 決定的シード 0xC0FFEE・warmup/iters 20/20。TASK-8.1）を実ワークロードとして使う。CPU
# バックエンドのみが対象で CUDA/Metal の bench example は実機必須のため対象外とする
# （Linux CI runner〈GitHub ホステッド〉で実行可能な範囲に限定）。`guardrail check` 自身が 4 ゲートを
# 起動して baseline 比較・劣化率閾値判定を行う結線（#103 残スコープ・TASK-8.2）は本スクリプト
# の範囲外であり、本スクリプトは「CI ジョブとして実行可能にする」導線に留める。
#
# 呼び出し元:
#   - .github/workflows/ci.yml の verification-gates ジョブ（self-test → build → test →
#     clippy の順で呼ぶ。PR/push 契機。実行時間の都合上 bench は含めない）
#   - .github/workflows/verification-gate-bench.yml（schedule／workflow_dispatch 契機で
#     bench サブコマンドのみを呼ぶ。#148〈TASK-6.1b〉の schedule 定期実行と衝突しないよう
#     独立 workflow ファイルに隔離している）
#   - Makefile の verification-gates／verification-gate-bench ターゲット（CI と同一判定を
#     ローカル再現）
#
# サブコマンド:
#   self-test  ゲート定義の空洞化検知（bench ワークロード example の存在等を fail-closed
#              検証。cargo 不要。scripts/check-forbidden-deps.sh self-test と同一方針）
#   build      `cargo build --workspace --locked`
#   test       `cargo test --workspace --release --locked`
#   clippy     `cargo clippy --workspace --all-targets --all-features -- -D warnings`
#   bench      `cargo run --release -p fandhe-ai-backend-cpu --example gemm_bench`
#   all        self-test → build → test → clippy を直列実行する（bench は実行時間が
#              長く push/PR 契機に不向きなため対象外。schedule/手動トリガーで別途実行する）
#
# スコープ外（out-of-scope-tracking.md に従い記録。PR 本文へ切り出し先を記載）:
#   - bench ゲートの baseline 比較・劣化率閾値判定・`guardrail check` 実シグナル計測経路
#     との結線（#103 残スコープ・TASK-8.2）
#   - CUDA/Metal 実機 bench ジョブ（実機 runner 未登録。登録後に runner ラベル明示で追加）
#   - 計測結果の構造化保存（`BenchReport` JSON の artifact 化）
set -euo pipefail

# bench ゲートの実ワークロード（CPU バックエンドのみ。Linux CI runner（GitHub ホステッド）で
# 実行可能な bench example。CUDA/Metal は実機必須のため対象外）。
BENCH_EXAMPLE_PATH="crates/backend-cpu/examples/gemm_bench.rs"

usage() {
  echo "usage: $0 {self-test|build|test|clippy|bench|all}" >&2
  exit 2
}

# 検査ロジック自体の退行（bench ワークロード example の無言削除・リネーム）を、実行環境の
# cargo 有無に関わらず常時検出する（scripts/check-forbidden-deps.sh self-test と同一方針）。
cmd_self_test() {
  local failed=0

  if [ -f "${BENCH_EXAMPLE_PATH}" ]; then
    echo "OK: ${BENCH_EXAMPLE_PATH} が存在します"
  else
    echo "NG: ${BENCH_EXAMPLE_PATH} が見つかりません（bench ゲートの計測ワークロードの欠落）" >&2
    failed=1
  fi

  if [ "${failed}" -ne 0 ]; then
    echo "NG: self-test に失敗しました" >&2
    return 1
  fi
  echo "OK: self-test すべて成功"
}

cmd_build() {
  echo "ゲート 1/4（build）: cargo build（型検査）"
  # --locked: Cargo.lock の意図しない書き換え・runner 汚染を防止する（他ジョブと同一方針）。
  cargo build --workspace --locked
}

cmd_test() {
  echo "ゲート 2/4（test）: cargo test --release（回帰テスト・数値精度検証）"
  # spec のゲート定義どおり --release を使う（ci.yml の test ジョブ〈debug〉とは責務が
  # 異なる。本ファイル冒頭コメント参照）。実機 #[ignore] テストは既定挙動で除外される。
  # --all-features を付けない: バックエンド切替は feature フラグなしの cfg ベース
  # （REQ-2・coding-rust.md）であり、本 workspace に切り替え用 feature 自体が存在しない
  # ため clippy ジョブの --all-features とは異なり付与の要否がない（既存 test ジョブ
  # 〈ci.yml〉が --all-features を付けているのは将来の feature 追加に備えた既定であり、
  # 本ゲートは spec のゲート定義コマンドに忠実に揃える）。
  cargo test --workspace --release --locked
}

cmd_clippy() {
  echo "ゲート 3/4（clippy）: cargo clippy --all-targets -- -D warnings（静的解析）"
  cargo clippy --workspace --all-targets --all-features -- -D warnings
}

cmd_bench() {
  echo "ゲート 4/4（bench）: 性能回帰検出相当（bench-harness 計測プロトコル）"
  cargo run --release -p fandhe-ai-backend-cpu --example gemm_bench
}

cmd_all() {
  cmd_self_test
  cmd_build
  cmd_test
  cmd_clippy
}

case "${1:-}" in
self-test)
  cmd_self_test
  ;;
build)
  cmd_build
  ;;
test)
  cmd_test
  ;;
clippy)
  cmd_clippy
  ;;
bench)
  cmd_bench
  ;;
all)
  cmd_all
  ;;
*)
  usage
  ;;
esac
