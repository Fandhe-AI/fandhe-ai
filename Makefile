# fandhe-ai の開発タスクランナー。
#
# Make は薄い入口であり、処理の定義は scripts/*.sh 側にのみ置く（各レシピは
# `@scripts/<name>.sh [args]` の 1 行呼び出しに限定する）。cargo / docker compose の
# 1 行コマンドで済むものだけは scripts/ に包まず直接書く。複数行・条件分岐・環境依存の
# 処理を足す場合は scripts/ にスクリプトを作ってから呼ぶ。
# Make を使わない場合の同等コマンドは scripts/help.sh を参照。
#
# `make setup` 一発で開発環境（サブモジュール・rustup・lefthook）を構築し、
# `make ci` で CI（.github/workflows/ci.yml）と同一のチェックをローカル実行する。
# Docker で環境非依存に開発する場合は docker-* ターゲットを使う（compose.yaml 参照）。
#
# GNU Make 3.81（macOS 標準）で動作する。3.82+ 専用機能（.ONESHELL・.RECIPEPREFIX 等）は使わない。
#
# 使い方: `make help`（既定ターゲット）でターゲット一覧を表示する。
# 一覧は下記の `## ` コメントから生成されるため、`## ` の無いターゲットは表示されない。
# ターゲットを追加・改名・削除したら scripts/help.sh も同じ変更で更新すること。
# 中身の無いターゲット（`@true` 等）は呼び出し元が終了コードを信用してしまうため置かない。
# 変更時は .claude/rules/makefile.md に従う。

.DEFAULT_GOAL := help

# 計測ハーネス（startup-bench・peak-memory-bench）の入力。`export` して環境変数として
# scripts/bench-harness.sh へ渡す（make の $(VAR) 展開をレシピへ埋め込むとシェル注入が
# 成立するため。検証は bench-harness.sh 側で行う）。BACKEND は cpu|cuda|metal。
BACKEND ?= cpu
TRIALS ?= 5
SIZE ?= 4096
export BACKEND
export TRIALS
export SIZE

.PHONY: help setup rustup submodule hooks \
	fmt fmt-check lint test test-ignored test-ignored-cuda test-ignored-metal test-ignored-metal-facade \
	build-cross build-no-cuda check-cross-metal-tests check-cross-cpu-tests check-cross-rocm-tests check-cross-windows-interop \
	doc-warnings deny deps-forbidden runner-policy guardrail-regression verification-gates verification-gate-bench \
	startup-bench peak-memory-bench ci docker-build docker-shell docker-ci

help: ## ターゲット一覧を表示する（Make 非依存の一覧: scripts/help.sh）
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z0-9_-]+:.*## /{printf "  \033[36m%-30s\033[0m %s\n", $$1, $$2}' $(firstword $(MAKEFILE_LIST))

# --------------------------------------------------
# 環境構築
# --------------------------------------------------

setup: ## 開発環境を一括構築する（サブモジュール → rustup → lefthook の順を保証）。直接: scripts/setup.sh
	@scripts/setup.sh

rustup: ## rustup（cargo）を未導入の場合のみ導入する。直接: scripts/rustup.sh
	@scripts/rustup.sh

submodule: ## docs/spec サブモジュールを初期化・更新する
	git submodule update --init

hooks: ## lefthook の git hooks を導入する（未導入なら lefthook 本体も導入）。直接: scripts/hooks.sh
	@scripts/hooks.sh

# --------------------------------------------------
# 品質チェック（CI と同一内容）
# --------------------------------------------------

fmt: ## cargo fmt --all で整形する
	cargo fmt --all

fmt-check: ## cargo fmt --check（整形差分の検出）
	cargo fmt --all --check

lint: ## cargo clippy -D warnings（lint ゲート）。直接: scripts/lint.sh
	@scripts/lint.sh

test: ## cargo test（実機依存の #[ignore] テストは除く）
	cargo test --workspace --all-features

# 実機（Metal / CUDA）専用の #[ignore] 分離テスト（coding-rust.md・ci.md の実機分離規約）。
# `--all-features` 必須: required-features ゲート付きテストが feature 未指定だと
# ビルド対象から外れ、--ignored でも暗黙に「パス」扱いになる（false-green）。
test-ignored: ## 実機（Metal / CUDA）専用: #[ignore] 分離テストを実行する
	cargo test --workspace --all-features -- --ignored --nocapture

test-ignored-cuda: ## CUDA 実機専用: backend-cuda の #[ignore] 分離テストを実行する（release）。直接: scripts/test-ignored.sh cuda
	@scripts/test-ignored.sh cuda

test-ignored-metal: ## Metal 実機専用: backend-metal の #[ignore] 分離テストを実行する（release）
	cargo test -p fandhe-ai-backend-metal --release --all-features -- --ignored --nocapture

test-ignored-metal-facade: ## Metal 実機専用: facade の DeviceParamStore 契約 #[ignore] テストを実行する（release）。直接: scripts/test-ignored.sh metal-facade
	@scripts/test-ignored.sh metal-facade

build-cross: ## macOS／Linux 両ターゲットの cargo build 検証（TASK-2.1b）。直接: scripts/build-cross.sh
	@scripts/build-cross.sh

build-no-cuda: ## CUDA toolkit／ROCm 非搭載環境でのビルド・実行成立検証（TASK-1.7d・TASK-2.3・#3154。搭載環境ではスキップ）。直接: scripts/build-no-cuda.sh
	@scripts/build-no-cuda.sh

check-cross-metal-tests: ## backend-metal の #[ignore] テストを aarch64-apple-darwin で型検査する。直接: scripts/check-cross.sh metal-tests
	@scripts/check-cross.sh metal-tests

check-cross-cpu-tests: ## backend-cpu の cfg(test) 限定コード（sysctl_ffi 等）を aarch64-apple-darwin で型検査する。直接: scripts/check-cross.sh cpu-tests
	@scripts/check-cross.sh cpu-tests

check-cross-rocm-tests: ## backend-rocm の cfg 分岐（非 Linux の HIP 候補空）を tests 込みで aarch64-apple-darwin 型検査する。直接: scripts/check-cross.sh rocm-tests
	@scripts/check-cross.sh rocm-tests

check-cross-windows-interop: ## onnx-interop・facade（Windows ビルド可能性）を x86_64-pc-windows-msvc で clippy 検査する。直接: scripts/check-cross.sh windows-interop
	@scripts/check-cross.sh windows-interop

doc-warnings: ## cargo doc -D warnings を Linux ホスト・aarch64-apple-darwin・backend-cuda internal-diagnostics の 3 通りで検証する。直接: scripts/doc-warnings.sh
	@scripts/doc-warnings.sh

deny: ## cargo deny check advisories bans licenses sources（依存監査。cargo-deny 未導入なら自動導入）。直接: scripts/deny.sh
	@scripts/deny.sh

deps-forbidden: ## 依存禁止リスト（burn 系等）の混入を検査する（self-test → lock-all → tree）。直接: scripts/ci-checks.sh deps-forbidden
	@scripts/ci-checks.sh deps-forbidden

runner-policy: ## self-hosted runner 逆戻り防止の fail-closed 契約検査を実行する。直接: scripts/ci-checks.sh runner-policy
	@scripts/ci-checks.sh runner-policy

guardrail-regression: ## guardrail 判定器の 2 層検証（REQ-4/REQ-5）を実行する。直接: scripts/ci-checks.sh guardrail-regression
	@scripts/ci-checks.sh guardrail-regression

verification-gates: ## AI 自律メンテナンス検証ゲート（build/test/clippy）を実行する。直接: scripts/ci-checks.sh verification-gates
	@scripts/ci-checks.sh verification-gates

verification-gate-bench: ## bench ゲート（CPU GEMM 計測）を単体実行する。直接: scripts/ci-checks.sh verification-gate-bench
	@scripts/ci-checks.sh verification-gate-bench

startup-bench: ## プロセス起動コスト計測を実行する（既定 CPU。BACKEND=cuda|metal で切替）。直接: scripts/bench-harness.sh startup
	@scripts/bench-harness.sh startup

peak-memory-bench: ## GEMM ピークメモリ計測を実行する（既定 CPU・4096³。BACKEND=cuda|metal・SIZE=N で切替）。直接: scripts/bench-harness.sh peak-memory
	@scripts/bench-harness.sh peak-memory

ci: fmt-check lint build-cross build-no-cuda check-cross-metal-tests check-cross-cpu-tests check-cross-rocm-tests check-cross-windows-interop doc-warnings test deny deps-forbidden runner-policy guardrail-regression verification-gates ## CI（ci.yml）と同一チェックを一括実行する

# --------------------------------------------------
# Docker（環境非依存の開発。CPU バックエンドのみ。詳細は README 参照）
# --------------------------------------------------

docker-build: ## 開発コンテナイメージをビルドする
	docker compose build

docker-shell: ## 開発コンテナのシェルに入る
	docker compose run --rm dev

docker-ci: ## コンテナ内で make ci を実行する（環境非依存の検証）
	docker compose run --rm dev make ci
