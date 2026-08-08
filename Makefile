# rust-ai-library の開発タスクランナー。
#
# `make setup` 一発で開発環境（rustup・lefthook・サブモジュール）を構築し、
# `make ci` で CI（.github/workflows/ci.yml）と同一のチェックをローカル実行する。
# workspace Cargo.toml（TASK-1.1）・deny.toml（TASK-1.3）はいずれも追加済みのため
# cargo 系ターゲットは deny を含め全て実行される。
# HAS_CARGO／HAS_DENY 判定によるスキップ分岐は、CI の detect ステップと同一の
# 冪等セルフヒール方針（.claude/rules/ci.md）のフェイルセーフとして残置する。
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

# TASK-1.7e（イシュー #36）: `backend-cuda` に限定した実機テスト導線。
# `test-ignored`（workspace 全体）は Metal 側の #[ignore] テストや他クレートの
# perf 系テストも巻き込むため、CUDA 実機（DGX Spark GB10 等・Linux）のみで
# `backend-cuda` を検証したい場合はこちらを使う。
# TASK-11.1e（イシュー #64）: `--release` を既定にする（`test-ignored-metal`
# と同じ理由。`tests/tensor_core_real_device.rs` の M=N=K=4096 TFLOPS 実測・
# `tests/gemm_wmma_tf32_opt.rs` の K=4096 ストレスケースは CPU 参照実装
# 〈`matmul_reference_fma`〉の計算量が大きく debug ビルドでは著しく遅いため）。
.PHONY: test-ignored-cuda
test-ignored-cuda: ## CUDA 実機専用: backend-cuda の #[ignore] 分離テストを実行する（release）
ifdef HAS_CARGO
	cargo test -p backend-cuda --release -- --ignored --nocapture
else
	@echo "skip: Cargo.toml 未追加のため test-ignored-cuda をスキップ"
endif

# TASK-1.8e（イシュー #42）: `backend-cuda` の `test-ignored-cuda`（#36）と対になる
# Metal 実機テスト導線。`--release` を既定にする（`tests/cpu_metal_parity.rs` の
# K=4096 ストレスケース〈`k4096_stress_poc_v2_5`〉が debug ビルドでは著しく遅いため。
# 各テストファイル冒頭コメントの推奨コマンドと一致させる）。
.PHONY: test-ignored-metal
test-ignored-metal: ## Metal 実機専用: backend-metal の #[ignore] 分離テストを実行する（release）
ifdef HAS_CARGO
	cargo test -p backend-metal --release -- --ignored --nocapture
else
	@echo "skip: Cargo.toml 未追加のため test-ignored-metal をスキップ"
endif

# macOS runner 未登録の代替として、aarch64-apple-darwin へのクロスターゲットビルドで
# Metal 有効経路（cfg(target_os = "macos")）のコンパイルを検証する（TASK-2.1b・イシュー #50。
# 全 9 クレートは lib のみでリンク不要なため、macOS SDK が無い環境でもコンパイル検証が成立する。
# 詳細は .github/workflows/ci.yml 冒頭コメント・PoC-v2-5 の cfg ベースバックエンド切替構成を参照）。
.PHONY: build-cross
build-cross: ## macOS／Linux 両ターゲットの cargo build 検証（TASK-2.1b）
ifdef HAS_CARGO
	rustup target list --installed | grep -qx 'aarch64-apple-darwin' || rustup target add aarch64-apple-darwin
	cargo build --workspace --locked
	cargo build --workspace --locked --target aarch64-apple-darwin
else
	@echo "skip: Cargo.toml 未追加のため build-cross をスキップ"
endif

# CUDA toolkit「非搭載」であることを scripts/check-cuda-toolkit-absent.sh（ci.yml の
# build-no-cuda-toolkit ジョブと共用する単一ソース）で検証したうえで cargo build・
# cargo test を実行する（TASK-1.7d・イシュー #35／TASK-2.3・イシュー #56。ビルド・実行
# 成立検証）。CUDA 実機（DGX Spark GB10 等）で toolkit 搭載環境から `make ci` を実行しても
# 壊れないよう、probe でスキップ分岐する（CI ジョブ側は assert で fail-closed のまま。
# toolkit 非搭載が確定した環境での検証は `make docker-ci` を使う）。
# cargo test は非 #[ignore] テストのみを実行し、DriverUnavailable への型付きエラー縮退・
# backend-cpu 経由の演算実行を含む workspace 全体の「実行成立」を ci.yml と同一内容で
# ローカル再現する。
.PHONY: build-no-cuda
build-no-cuda: ## CUDA toolkit 非搭載環境でのビルド・実行成立検証（TASK-1.7d・TASK-2.3。搭載環境ではスキップ）
ifdef HAS_CARGO
	@bash scripts/check-cuda-toolkit-absent.sh self-test
	@if bash scripts/check-cuda-toolkit-absent.sh probe; then \
		bash scripts/check-cuda-toolkit-absent.sh assert; \
		cargo build --workspace --locked; \
		cargo test --workspace --locked; \
	else \
		echo "skip: CUDA toolkit 搭載環境のため非搭載検証をスキップ（make docker-ci で検証可能）"; \
	fi
else
	@echo "skip: Cargo.toml 未追加のため build-no-cuda をスキップ"
endif

# TASK-1.8e（イシュー #42）: `backend-metal` の `#[ignore]` 実機テスト（本ファイルの
# `tests/`）を Linux CI でも型検査する。`--workspace --all-targets` は bench-harness の
# dev-dependencies 経由で criterion → alloca（macOS ターゲットではネイティブ C ビルドが
# 必要）を引き込み、macOS クロスコンパイラ非搭載の self-hosted runner では
# `cc: error: unrecognized command-line option '-arch'` で失敗することを実測済みのため
# 採用しない（`-p backend-metal --tests` に限定すれば bench-harness は
# `[dependencies]`（criterion を含まない）としてのみ解決され、この失敗を回避できる。
# `cargo check` はリンクを行わないため macOS SDK 非搭載でも成立する）。
.PHONY: check-cross-metal-tests
check-cross-metal-tests: ## backend-metal の #[ignore] テストを aarch64-apple-darwin で型検査する
ifdef HAS_CARGO
	rustup target list --installed | grep -qx 'aarch64-apple-darwin' || rustup target add aarch64-apple-darwin
	cargo check -p backend-metal --tests --target aarch64-apple-darwin
else
	@echo "skip: Cargo.toml 未追加のため check-cross-metal-tests をスキップ"
endif

.PHONY: deny
deny: ## cargo deny check advisories bans licenses sources（依存監査。cargo-deny 未導入なら自動導入。#353 で advisories / bans 追加）
ifneq ($(and $(HAS_CARGO),$(HAS_DENY)),)
	@export PATH="$$HOME/.cargo/bin:$$PATH"; \
	command -v cargo-deny >/dev/null 2>&1 || { \
		echo "cargo-deny を導入します"; \
		cargo install cargo-deny --locked; \
	}; \
	cargo deny --locked check advisories bans licenses sources
else
	@echo "skip: Cargo.toml または deny.toml 未追加のため deny をスキップ"
endif

# 依存禁止リスト（deps-policy.md: burn 系・cubecl・candle・tch・ndarray）の混入検査。
# TASK-1.2 の CI 機械検査と同一の判定を scripts/check-forbidden-deps.sh 経由でローカル再現する
# （検査ロジックは ci.yml の deps-forbidden ジョブと同一スクリプトを共用し、二重管理しない）。
# self-test で検査ロジック自体の退行を検出したうえで、Cargo.lock（存在すれば）・
# cargo tree（Cargo.toml があれば）を検査する。
.PHONY: deps-forbidden
deps-forbidden: ## 依存禁止リスト（burn 系等）の混入を検査する（self-test → lock → tree）
	@bash scripts/check-forbidden-deps.sh self-test
	@if [ -f Cargo.lock ]; then \
		bash scripts/check-forbidden-deps.sh lock Cargo.lock; \
	else \
		echo "skip: Cargo.lock 未追加のため lock 検査をスキップ"; \
	fi
ifdef HAS_CARGO
	@bash scripts/check-forbidden-deps.sh tree
else
	@echo "skip: Cargo.toml 未追加のため cargo tree 検査をスキップ"
endif

# guardrail 判定器の 2 層検証（TASK-6.1a・イシュー #147）。
# REQ-4（1 層目・判定器単体）・REQ-5（2 層目・除外適用後）の回帰テストを
# scripts/run-guardrail-regression.sh（ci.yml の guardrail-regression ジョブと共用する
# 単一ソース）経由でローカル実行する。self-test → layer1 → layer2 の順で直列実行する
# `all` サブコマンドを使う（検査ロジック自体の退行を実行環境の cargo 有無に関わらず
# 常時検出したうえでテストを実行する。deps-forbidden ターゲットと同一方針）。
.PHONY: guardrail-regression
guardrail-regression: ## guardrail 判定器の 2 層検証（REQ-4/REQ-5）を実行する
ifdef HAS_CARGO
	@bash scripts/run-guardrail-regression.sh self-test
	@bash scripts/run-guardrail-regression.sh layer1
	@bash scripts/run-guardrail-regression.sh layer2
else
	@bash scripts/run-guardrail-regression.sh self-test
	@echo "skip: Cargo.toml 未追加のため layer1/layer2 をスキップ"
endif

# AI 自律メンテナンス検証ゲート（build/test/clippy。TASK-6.1c・イシュー #199）。
# `docs/spec/04-requirements.md` の検証ゲート定義（L34）・非機能要件「REQ-3〜REQ-6 の
# ガードレール（build/test/clippy/bench）は CI 上で自動実行可能な構成とすること」（L292）
# に対応する。scripts/run-verification-gates.sh（ci.yml の verification-gates ジョブと
# 共用する単一ソース）経由で self-test → build → test（--release）→ clippy を直列実行する
# （guardrail-regression ターゲットと同一方針）。
.PHONY: verification-gates
verification-gates: ## AI 自律メンテナンス検証ゲート（build/test/clippy）を実行する
ifdef HAS_CARGO
	@bash scripts/run-verification-gates.sh self-test
	@bash scripts/run-verification-gates.sh build
	@bash scripts/run-verification-gates.sh test
	@bash scripts/run-verification-gates.sh clippy
else
	@bash scripts/run-verification-gates.sh self-test
	@echo "skip: Cargo.toml 未追加のため build/test/clippy をスキップ"
endif

# bench ゲート単体（TASK-6.1c・イシュー #199）。実行時間が長いため verification-gates
# には含めず、`.github/workflows/verification-gate-bench.yml`（schedule／
# workflow_dispatch）と同一内容をローカルで再現する独立ターゲットとする。
.PHONY: verification-gate-bench
verification-gate-bench: ## bench ゲート（CPU GEMM 計測）を単体実行する
ifdef HAS_CARGO
	@bash scripts/run-verification-gates.sh self-test
	@bash scripts/run-verification-gates.sh bench
else
	@bash scripts/run-verification-gates.sh self-test
	@echo "skip: Cargo.toml 未追加のため bench をスキップ"
endif

# 起動コスト計測ハーネス（TASK-13.1a・イシュー #170）。CI には組み込まない
# （実測の実施・v1 実測値との差分記録は兄弟イシュー #171・TASK-13.1b のスコープ。
# ここでは #171 が再現に使う導線のみを提供する。既定はホスト CPU バックエンド、
# `BACKEND=cuda`／`BACKEND=metal`（実機限定）で切り替える）。
BACKEND ?= cpu
TRIALS ?= 5
.PHONY: startup-bench
startup-bench: ## プロセス起動コスト計測を実行する（既定 CPU。BACKEND=cuda|metal で切替）
ifdef HAS_CARGO
	cargo build -p bench-harness --release --bins
	cargo run -p bench-harness --release --bin startup_bench -- --backend $(BACKEND) --trials $(TRIALS)
else
	@echo "skip: Cargo.toml 未追加のため startup-bench をスキップ"
endif

.PHONY: ci
ci: fmt-check lint build-cross build-no-cuda check-cross-metal-tests test deny deps-forbidden guardrail-regression verification-gates ## CI（ci.yml）と同一チェックを一括実行する

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
