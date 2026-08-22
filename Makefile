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

# CI（Linux ホステッド。実機非搭載）では実行不可・実機でのみ使用する #[ignore] 分離テスト
# （coding-rust.md・ci.md の実機分離規約）。Metal は Apple Silicon 実機、
# CUDA は DGX Spark GB10 等の NVIDIA GPU 実機で実行する。
# `--all-features` 必須（PR #685 codex-review/Bugbot 指摘の是正）: backend-cuda の
# `specialized_mma_parity`（イシュー #531）は `[[test]] required-features =
# ["internal-diagnostics"]`（`crates/backend-cuda/Cargo.toml` 参照）でゲートされて
# おり、feature 未指定では cargo がテストバイナリ自体をビルド対象から外すため
# `--ignored` を渡しても実行されず暗黙に「パス」扱いになる（false-green。
# PR #667 の swizzle 系テストと同型の落とし穴）。`--all-features` を明示することで
# `cargo test --all-features`（CI の test ジョブ・`make test` と同じ feature 集合）と
# 揃え、実機検証で実際にビルド・実行されることを保証する。
.PHONY: test-ignored
test-ignored: ## 実機（Metal / CUDA）専用: #[ignore] 分離テストを実行する
ifdef HAS_CARGO
	cargo test --workspace --all-features -- --ignored --nocapture
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
# `--all-features` 必須（PR #685 codex-review/Bugbot 指摘の是正。`test-ignored`
# 冒頭コメント参照）: `specialized_mma_parity`（イシュー #531）は
# `internal-diagnostics` feature の `required-features` ゲート対象のため、
# feature 未指定だとテストバイナリがビルドされず実機検証で実際には走らない
# （false-green）。
.PHONY: test-ignored-cuda
test-ignored-cuda: ## CUDA 実機専用: backend-cuda の #[ignore] 分離テストを実行する（release）
ifdef HAS_CARGO
	cargo test -p fandhe-ai-backend-cuda --release --all-features -- --ignored --nocapture
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
	cargo test -p fandhe-ai-backend-metal --release -- --ignored --nocapture
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
# 必要）を引き込み、macOS クロスコンパイラ非搭載の Linux CI runner（実測当時は
# self-hosted。#457 で GitHub ホステッドへ移行済みだが技術的前提は不変）では
# `cc: error: unrecognized command-line option '-arch'` で失敗することを実測済みのため
# 採用しない（`-p fandhe-ai-backend-metal --tests` に限定すれば bench-harness は
# `[dependencies]`（criterion を含まない）としてのみ解決され、この失敗を回避できる。
# `cargo check` はリンクを行わないため macOS SDK 非搭載でも成立する）。
.PHONY: check-cross-metal-tests
check-cross-metal-tests: ## backend-metal の #[ignore] テストを aarch64-apple-darwin で型検査する
ifdef HAS_CARGO
	rustup target list --installed | grep -qx 'aarch64-apple-darwin' || rustup target add aarch64-apple-darwin
	cargo check -p fandhe-ai-backend-metal --tests --target aarch64-apple-darwin
else
	@echo "skip: Cargo.toml 未追加のため check-cross-metal-tests をスキップ"
endif

# イシュー #753 レビュー指摘（codex-review P2）: `backend-cpu::gemm_blis::cache_params`
# モジュールは本番未結線のため `#[cfg(test)]` 限定でしかコンパイルされず、その内部の
# `sysctl_ffi`（`cfg(target_os = "macos")` 限定の FFI 宣言）は Linux CI の
# 通常 `cargo test` では `target_os != "macos"` のため到達しない。上記
# check-cross-metal-tests と同じ手法（`--tests` で `cfg(test)` を有効化した状態のまま
# aarch64-apple-darwin へクロス型検査）で `sysctl_ffi` を継続的コンパイル検証の対象に
# 含める。criterion → alloca 問題は check-cross-metal-tests と同じ理由で
# `-p fandhe-ai-backend-cpu --tests` への限定で回避する（`cargo check` はリンクを行わないため
# macOS SDK 非搭載でも成立する）。
.PHONY: check-cross-cpu-tests
check-cross-cpu-tests: ## backend-cpu の cfg(test) 限定コード（sysctl_ffi 等）を aarch64-apple-darwin で型検査する
ifdef HAS_CARGO
	rustup target list --installed | grep -qx 'aarch64-apple-darwin' || rustup target add aarch64-apple-darwin
	cargo check -p fandhe-ai-backend-cpu --tests --target aarch64-apple-darwin
else
	@echo "skip: Cargo.toml 未追加のため check-cross-cpu-tests をスキップ"
endif

# rustdoc 警告ゲート（イシュー #883）。ci.yml の build ジョブに追加した 2 step
# （Linux ホスト分・aarch64-apple-darwin クロス分）とローカルで同一コマンドを共用し、
# 再現手順を一本化する。cfg（aarch64 NEON vs x86 AVX・macOS 限定モジュール）により
# 警告集合がターゲットごとに異なるため両方をゲートする
# （docs/crates-io-publishing-order.md「公開前検証手順と実測記録」参照）。
.PHONY: doc-warnings
doc-warnings: ## cargo doc -D warnings を Linux ホスト分・aarch64-apple-darwin クロス分の両方で検証する
ifdef HAS_CARGO
	rustup target list --installed | grep -qx 'aarch64-apple-darwin' || rustup target add aarch64-apple-darwin
	RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked
	RUSTDOCFLAGS="-D warnings" cargo doc -p fandhe-ai-backend-metal -p fandhe-ai-backend-cpu --no-deps --locked --target aarch64-apple-darwin
else
	@echo "skip: Cargo.toml 未追加のため doc-warnings をスキップ"
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
# self-test で検査ロジック自体の退行を検出したうえで、lock-all（本体 workspace ルート
# Cargo.lock・OSS 直接比較ハーネス scripts/bench/oss-gemm-compare/Cargo.lock。許容依存
# 第 9 区分・イシュー #755。走査対象パスの列挙は scripts/check-forbidden-deps.sh 側の
# lock-all サブコマンドに集約済みでここではハードコードしない）・cargo tree
# （Cargo.toml があれば）を検査する。
.PHONY: deps-forbidden
deps-forbidden: ## 依存禁止リスト（burn 系等）の混入を検査する（self-test → lock-all → tree）
	@bash scripts/check-forbidden-deps.sh self-test
	@bash scripts/check-forbidden-deps.sh lock-all
ifdef HAS_CARGO
	@bash scripts/check-forbidden-deps.sh tree
else
	@echo "skip: Cargo.toml 未追加のため cargo tree 検査をスキップ"
endif

# self-hosted runner 逆戻り防止の fail-closed 契約検査（イシュー #472）。
# ci.yml の runner-policy ジョブと共用する scripts/check-workflow-runner-policy.sh
# （検査本体は python3 標準ライブラリのみの check-workflow-runner-policy.py。追加
# パッケージ導入不要。PyYAML 等の外部依存を使わない理由は .py 冒頭コメント参照）経由で、
# .github/workflows/ 配下への self-hosted 再導入・許容形リスト（標準 GitHub ホステッド
# ランナーの `-latest` ラベル集合: ubuntu-latest・macos-latest・windows-latest）以外の
# runner 宣言をローカルで検査する
# （deps-forbidden ターゲットと同一方針。self-test → check の順で直列実行する）。
.PHONY: runner-policy
runner-policy: ## self-hosted runner 逆戻り防止の fail-closed 契約検査を実行する
	@bash scripts/check-workflow-runner-policy.sh self-test
	@bash scripts/check-workflow-runner-policy.sh check

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
# BACKEND・TRIALS を `export` して環境変数として子シェルへ渡す。make の `$(VAR)` 展開は
# レシピ本文へのテキスト置換であり、値に `"` や改行・`;` 等のシェル構文を含めると
# 二重引用符で囲んでも引用符を閉じられてしまいコマンド注入が成立する（PR #360
# codex-review P1 指摘。二重引用符化だけでは防げない）。環境変数は execve の環境ブロック
# 経由でシェルへ渡るため、テキスト置換段階を経由せず値がレシピの再解釈対象にならない。
# レシピ側では `$$BACKEND`／`$$TRIALS`（シェル変数参照）でのみ読み、許可リスト
# （cpu|cuda|metal）／数値のみを検証したうえで cargo 呼び出しへ二重引用符で渡す
# （StartupBackend::parse 側の許可リスト検証はプロセス起動後の防御であり、
# シェル注入そのものはレシピ側で防ぐ必要がある）。
export BACKEND
export TRIALS
.PHONY: startup-bench
startup-bench: ## プロセス起動コスト計測を実行する（既定 CPU。BACKEND=cuda|metal で切替）
ifdef HAS_CARGO
	@case "$$BACKEND" in \
		cpu|cuda|metal) ;; \
		*) echo "BACKEND は cpu|cuda|metal のいずれかを指定する（指定値: $$BACKEND）" >&2; exit 1 ;; \
	esac
	@case "$$TRIALS" in \
		''|*[!0-9]*) echo "TRIALS は数値を指定する（指定値: $$TRIALS）" >&2; exit 1 ;; \
		*) ;; \
	esac
	cargo build -p bench-harness --release --bins
	cargo run -p bench-harness --release --bin startup_bench -- --backend "$$BACKEND" --trials "$$TRIALS"
else
	@echo "skip: Cargo.toml 未追加のため startup-bench をスキップ"
endif

# GEMM ピークメモリ計測ハーネス（TASK-14.2a・イシュー #178）。CI には組み込まない
# （REQ-14 の代表ワークロード〈M=N=K=4096〉は数百 MB〜数 GB の確保・GEMM 実行を伴い
# 通常 CI ジョブとしては重いため。CI 実行可能なスモークテストは
# `crates/bench-harness/tests/peak_memory_smoke.rs` が別途担う）。既定はホスト CPU
# バックエンド・REQ-14 代表サイズ、`BACKEND=cuda`／`BACKEND=metal`（実機限定）・
# `SIZE=<N>` で切り替える。BACKEND・TRIALS の export・シェル注入対策は
# `startup-bench` ターゲットと同一方針（上記コメント参照）。SIZE も同様に export し、
# レシピ側で数値のみを許可リスト検証してから cargo 呼び出しへ渡す。
SIZE ?= 4096
export SIZE
.PHONY: peak-memory-bench
peak-memory-bench: ## GEMM ピークメモリ計測を実行する（既定 CPU・4096³。BACKEND=cuda|metal・SIZE=N で切替）
ifdef HAS_CARGO
	@case "$$BACKEND" in \
		cpu|cuda|metal) ;; \
		*) echo "BACKEND は cpu|cuda|metal のいずれかを指定する（指定値: $$BACKEND）" >&2; exit 1 ;; \
	esac
	@case "$$TRIALS" in \
		''|*[!0-9]*) echo "TRIALS は数値を指定する（指定値: $$TRIALS）" >&2; exit 1 ;; \
		*) ;; \
	esac
	@case "$$SIZE" in \
		''|*[!0-9]*) echo "SIZE は数値を指定する（指定値: $$SIZE）" >&2; exit 1 ;; \
		*) ;; \
	esac
	cargo build -p bench-harness --release --bins
	cargo run -p bench-harness --release --bin peak_memory_bench -- --backend "$$BACKEND" --size "$$SIZE" --trials "$$TRIALS"
else
	@echo "skip: Cargo.toml 未追加のため peak-memory-bench をスキップ"
endif

.PHONY: ci
ci: fmt-check lint build-cross build-no-cuda check-cross-metal-tests check-cross-cpu-tests doc-warnings test deny deps-forbidden runner-policy guardrail-regression verification-gates ## CI（ci.yml）と同一チェックを一括実行する

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
