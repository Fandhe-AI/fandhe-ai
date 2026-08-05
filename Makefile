# rust-ai-library の開発タスクランナー。
#
# `make setup` 一発で開発環境（rustup・lefthook・サブモジュール）を構築し、
# `make ci` で CI（.github/workflows/ci.yml）と同一のチェックをローカル実行する。
# Cargo.toml 未追加（M0 の TASK-1.1 で workspace 作成予定）の間、cargo 系ターゲットは
# CI と同じ方針でスキップする（エラーにしない）。
# Docker で環境非依存に開発する場合は docker-* ターゲットを使う（compose.yaml 参照）。
# Fandhe-AI/rust-ai-library-v1 の Makefile と同一方針。

.DEFAULT_GOAL := help
SHELL := /bin/bash

# Cargo.toml の有無（CI の detect ステップと同一の判定。無ければ cargo 系をスキップ）
HAS_CARGO := $(wildcard Cargo.toml)
HAS_DENY := $(wildcard deny.toml)

.PHONY: help
help: ## ターゲット一覧を表示する
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# --------------------------------------------------
# 環境構築
# --------------------------------------------------

# 依存ターゲット並記だと -j 実行時に順序が保証されず、cargo フォールバックを持つ hooks が
# rustup より先に走りうるため、再帰 make で「submodule → rustup → hooks」の順を明示する。
.PHONY: setup
setup: ## 開発環境を一括構築する（サブモジュール → rustup → lefthook の順を保証）
	$(MAKE) submodule
	$(MAKE) rustup
	$(MAKE) hooks
	@echo "setup 完了"

.PHONY: rustup
rustup: ## rustup（cargo）を未導入の場合のみ導入する
	@if ! command -v rustup >/dev/null 2>&1 && [ ! -x "$$HOME/.cargo/bin/rustup" ]; then \
		echo "rustup を導入します"; \
		curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable; \
	fi

.PHONY: submodule
submodule: ## docs/spec サブモジュールを初期化・更新する
	git submodule update --init

# make rustup 直後（同一シェルの PATH 未反映）でも cargo を拾えるよう ~/.cargo/bin を PATH に足す。
.PHONY: hooks
hooks: ## lefthook の git hooks を導入する（未導入なら lefthook 本体も導入）
	@export PATH="$$HOME/.cargo/bin:$$PATH"; \
	command -v lefthook >/dev/null 2>&1 || { \
		echo "lefthook を導入します"; \
		if command -v brew >/dev/null 2>&1; then brew install lefthook; \
		elif command -v cargo >/dev/null 2>&1; then cargo install lefthook --locked; \
		else echo "brew / cargo が見つかりません。https://lefthook.dev/installation/ を参照してください" && exit 1; fi; \
	}; \
	lefthook install

# --------------------------------------------------
# 品質チェック（CI と同一内容）
# --------------------------------------------------

.PHONY: fmt
fmt: ## cargo fmt --all で整形する
ifdef HAS_CARGO
	cargo fmt --all
else
	@echo "skip: Cargo.toml 未追加のため fmt をスキップ"
endif

.PHONY: fmt-check
fmt-check: ## cargo fmt --check（整形差分の検出）
ifdef HAS_CARGO
	cargo fmt --all --check
else
	@echo "skip: Cargo.toml 未追加のため fmt-check をスキップ"
endif

.PHONY: lint
lint: ## cargo clippy -D warnings（lint ゲート）
ifdef HAS_CARGO
	cargo clippy --workspace --all-targets --all-features -- -D warnings
else
	@echo "skip: Cargo.toml 未追加のため lint をスキップ"
endif

.PHONY: test
test: ## cargo test（実機依存の #[ignore] テストは除く）
ifdef HAS_CARGO
	cargo test --workspace --all-features
else
	@echo "skip: Cargo.toml 未追加のため test をスキップ"
endif

# CI（Linux self-hosted）では実行不可・実機でのみ使用する #[ignore] 分離テスト
# （coding-rust.md・ci.md の実機分離規約）。Metal は Apple Silicon 実機、
# CUDA は DGX Spark GB10 等の NVIDIA GPU 実機で実行する。
.PHONY: test-ignored
test-ignored: ## 実機（Metal / CUDA）専用: #[ignore] 分離テストを実行する
ifdef HAS_CARGO
	cargo test --workspace -- --ignored --nocapture
else
	@echo "skip: Cargo.toml 未追加のため test-ignored をスキップ"
endif

.PHONY: deny
deny: ## cargo deny check licenses sources（依存ライセンス監査）
ifneq ($(and $(HAS_CARGO),$(HAS_DENY)),)
	cargo deny --locked check licenses sources
else
	@echo "skip: Cargo.toml または deny.toml 未追加のため deny をスキップ"
endif

# 依存禁止リスト（deps-policy.md: burn 系・cubecl・candle・tch・ndarray）の混入検査。
# TASK-1.2 の CI 機械検査と同一の判定をローカルで再現する。Cargo.lock を対象とし、
# name = "<crate>" の完全一致で fail-closed に判定する。
.PHONY: deps-forbidden
deps-forbidden: ## 依存禁止リスト（burn 系等）の混入を Cargo.lock から検査する
	@if [ -f Cargo.lock ]; then \
		forbidden='^name = "(burn|burn-[a-z-]+|cubecl|cubecl-[a-z-]+|candle-core|tch|ndarray)"$$'; \
		if grep -qE "$$forbidden" Cargo.lock; then \
			echo "NG: 依存禁止リストのクレートが Cargo.lock に含まれています:" >&2; \
			grep -E "$$forbidden" Cargo.lock >&2; \
			exit 1; \
		fi; \
		echo "OK: 依存禁止リストの混入なし"; \
	else \
		echo "skip: Cargo.lock 未追加のため deps-forbidden をスキップ"; \
	fi

.PHONY: ci
ci: fmt-check lint test deny deps-forbidden ## CI（ci.yml）と同一チェックを一括実行する

# --------------------------------------------------
# Docker（環境非依存の開発。CPU バックエンドのみ。詳細は README 参照）
# --------------------------------------------------

.PHONY: docker-build
docker-build: ## 開発コンテナイメージをビルドする
	docker compose build

.PHONY: docker-shell
docker-shell: ## 開発コンテナのシェルに入る
	docker compose run --rm dev

.PHONY: docker-ci
docker-ci: ## コンテナ内で make ci を実行する（環境非依存の検証）
	docker compose run --rm dev make ci
