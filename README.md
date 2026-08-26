# fandhe-ai

Rust 製 AI/ML ライブラリです。Burn 等の既存フレームワークに依存せず、テンソル・
autodiff・演算グラフ／カーネル融合機構・計算カーネル・バックエンド抽象層を
**完全自作コア**として実装しています。

*A from-scratch Rust AI/ML library — no Burn/candle/tch dependency. See the
[Getting Started](#最小コード例) section below for install & a minimal example.*

## ドキュメント

利用者向けドキュメントサイト（GitHub Pages）は公開済みです。

- https://fandhe-ai.github.io/fandhe-ai/

ローカルでビルドして確認する場合は次のコマンドを実行してください。

```bash
cargo run -p docs-site -- --out dist/
```

生成物は `dist/index.html` から閲覧できます。ビルドには内部リンク検証（linkcheck）が
内蔵されており、リンク切れを検出した場合は非 0 終了します。コンテンツの正は
`site/` 配下、サイト構成は `site/nav.toml` です。

## インストール

本ライブラリは Rust の `stable` チャンネルを前提としています（リポジトリ直下の
[`rust-toolchain.toml`](./rust-toolchain.toml) が単一真実源です）。crates.io（v0.3.0・
2026-08-23 公開済み）から利用できます。

```toml
[dependencies]
fandhe-ai = "0.3.0"
```

公開ドキュメントは以下のとおりです。

- https://docs.rs/fandhe-ai

開発版を試す場合は Git 依存で参照できます。

```toml
[dependencies]
fandhe-ai = { git = "https://github.com/Fandhe-AI/fandhe-ai" }
```

利用者が直接依存すべきクレートは `fandhe-ai` だけです。`fandhe-ai-tensor-core`・
`fandhe-ai-autodiff`・`fandhe-ai-backend-cpu`・`fandhe-ai-backend-cuda`・
`fandhe-ai-backend-metal` は内部クレートであり、直接の依存・利用はサポート対象外
です（下記「クレート構成」参照）。

## 最小コード例

`compat::array`（numpy `np.array` 慣習のテンソル生成）と `compat::Sequential`
（Keras `Sequential` 慣習のレイヤー積み上げ）を使うと、数行でモデルを組み立てて
推論できます。以下は
[`crates/facade/examples/getting_started.rs`](crates/facade/examples/getting_started.rs)
（`cargo run -p fandhe-ai --example getting_started` で実行確認済み）と同一のコードです。

```rust
use fandhe_ai::compat::{Sequential, array};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input = array(vec![
        vec![0.1_f32, 0.2, 0.3, 0.4],
        vec![0.5_f32, 0.6, 0.7, 0.8],
    ])?;

    let model = Sequential::new()
        .add_linear(4, 8, /* seed = */ 42)?
        .add_relu()
        .add_linear(8, 2, /* seed = */ 43)?;

    let output = model.predict(&input)?;

    println!("output shape: {:?}", output.shape());
    Ok(())
}
```

実行コマンドと期待出力:

```bash
cargo run -p fandhe-ai --example getting_started
# => output shape: [2, 2]
```

## クレート構成とサポート境界

内部は複数のクレートに分かれていますが、利用者が直接触れるのは `fandhe-ai`
クレートだけです。

| クレート | 役割 |
|---|---|
| `fandhe-ai` | **唯一のサポートされる公開 API 面**。composition root（`Device` → バックエンドの結線）と compat 公開面（`compat::array`／`compat::Sequential`）を提供します |
| `fandhe-ai-tensor-core`・`fandhe-ai-autodiff`・`fandhe-ai-backend-cpu`・`fandhe-ai-backend-cuda`・`fandhe-ai-backend-metal` | 内部クレート。直接利用はサポート対象外です |

上記 6 クレートが crates.io 公開対象です（[`docs/crates-io-naming-decision.md`](docs/crates-io-naming-decision.md)）。
ディレクトリ名（`crates/tensor-core` 等）はリネーム前のまま維持しており、
`[package] name`（crates.io 公開名）のみ `fandhe-ai` prefix 付きへ変更しています。

このほか、相互運用（`onnx-interop`）・自己修復ループ（`guardrail`・`self-repair`）・
ベンチ計測（`bench-harness`）・ドキュメントサイト生成（`docs-site`）を担う非公開の
内部クレートがあります。compat API のサポート境界の詳細は [`docs/compat-api-scope.md`](docs/compat-api-scope.md) を参照してください。

## バックエンド

バックエンド切替は feature フラグを使わない **cfg ベース**です。CPU は常に利用
可能な既定バックエンドで、CUDA・Metal は実行時にデバイスの存在を検証し、利用
できない場合はエラーを返します（自動フォールバックはしません）。バックエンド
間の数値一致は「相対誤差 1e-3 未満または絶対誤差 1e-5 未満」の複合判定で担保
します（詳細 → [`docs/backend-switching-design.md`](docs/backend-switching-design.md)）。

Metal バインディングは `objc2-metal` 直接（`wgpu` 不採用: 直接比 約 2.3 倍実測・
PoC-v2-4。詳細 → [`docs/backend-metal-wgpu-decision.md`](docs/backend-metal-wgpu-decision.md)）。

## ステータス

- **crates.io 初回公開完了**: v0.3.0（6 クレート: `fandhe-ai`・`fandhe-ai-tensor-core`・
  `fandhe-ai-autodiff`・`fandhe-ai-backend-cpu`・`fandhe-ai-backend-cuda`・
  `fandhe-ai-backend-metal`）を 2026-08-23 に公開済み
- **ドキュメントサイト公開済み**: https://fandhe-ai.github.io/fandhe-ai/
- **実装の進行状況**: コア（テンソル・autodiff・演算グラフ／カーネル融合機構）・
  3 バックエンド（CPU／CUDA／Metal）の実装と性能実測を継続中

## 開発（コントリビュータ向け）

以下はライブラリを clone してコントリビュートする開発者向けの情報です（利用者は上記の
「インストール」「最小コード例」だけで十分です）。

- 依存は許容 9 区分のみを `=x.y.z` 完全固定で管理する。うち第 1〜8 区分（`cudarc`／`objc2` 系／`safetensors`／`prost`／`serde` 系／`rayon`／`half`／`criterion`）は本体 workspace ルート `Cargo.toml` の `[workspace.dependencies]` に一元定義し、第 9 区分（`matrixmultiply`／`gemm`。OSS GEMM ベンチ比較対象）は `scripts/bench/oss-gemm-compare/`（独立 Cargo プロジェクト）限定で本体 workspace への混入を禁止する（詳細 → [`.claude/rules/deps-policy.md`](.claude/rules/deps-policy.md)）
- 依存禁止リスト（`burn` 系一式・`cubecl`・`candle`・`tch`・`ndarray`）を CI で機械検査

### 依存追加・更新フロー

依存の追加・更新は必ずユーザー承認を経て行う（`.claude/rules/deps-policy.md`）。承認後の実施手順:

1. **判断理由の記録**: 許容依存 9 区分以外を新規追加する場合、判断軸 a〜e（数値意味論か境界層か／AI 保守ガードレール対象か／自作コスト対差別化価値／unsafe・FFI 面積／ライセンス適合。`docs/spec/01-brainstorm.md` の「v2 自作範囲の境界定義」節）に基づく判断理由を PR 本文・コミットメッセージに記録する
2. **バージョン固定**: `Cargo.toml` で `=x.y.z` 完全固定とし、`Cargo.lock` を同一コミットでコミットする
3. **`docs/license-matrix.md` の同時更新**: 新規・更新する依存のライセンスを確認し、`docs/license-matrix.md`（未作成の場合は同一 PR で作成）を更新する。MPL-2.0 等コピーレフトの推移的混入は推定で記述せず、有効化しうる feature 組合せごとの `cargo tree` 実測で個別に適合確認する
4. **CI 検査の確認**: `deps-forbidden` ジョブ（禁止リストの `Cargo.lock` 機械検査）・`deny` ジョブ（`deny.toml` によるライセンス監査）が green であることを確認する

## 開発環境セットアップ

### クローンと初期化

仕様書が `docs/spec/` にサブモジュールとして取り込まれているため、
クローン時に `--recurse-submodules` を指定してください。

```bash
git clone --recurse-submodules git@github.com:Fandhe-AI/fandhe-ai.git
cd fandhe-ai
```

既存クローンがある場合は以下で初期化してください。

```bash
git submodule update --init
```

### 最短セットアップ

ローカル開発環境の構築は `make setup` で行います。サブモジュール取得 → rustup（`stable` の導入。rustfmt / clippy は `rust-toolchain.toml` の components で同期）→ lefthook（git hooks）の順に一括構築します。

```bash
make setup
```

その後、以下で基本的なチェック（ビルド・テスト・整形・lint）を実行できます。

```bash
cargo build --workspace
make test
make fmt
make lint
```

全ターゲット一覧は `make help` で確認してください。

仕様書 submodule は private リポジトリのため、アクセス権のない環境では submodule 取得ステップが失敗します。この場合も `docs/spec` 抜き（空ディレクトリ）で `cargo build --workspace`・`cargo test --workspace`（`#[ignore]` を除く通常テスト）は成立することを実測確認済みです（#463）。docs/spec 配下のファイル読み取りに依存するテストは 3 件のみで、いずれも `#[ignore]` により通常実行から分離されています（`crates/autodiff/tests/poc_v2_2_parity.rs`・`crates/backend-cpu/tests/gemm_parity.rs`・`crates/tensor-core/tests/poc_v2_1_parity.rs`）。submodule 未取得の環境でこれら 3 件を `--ignored` 指定で実行すると、evidence ファイルを読めない旨のエラーで失敗することを実測確認済みです。

### Make ターゲット

| ターゲット | 内容 |
|-----------|------|
| `make setup` | 開発環境の一括構築（submodule・rustup・lefthook） |
| `make fmt` / `make fmt-check` | 整形 / 整形差分の検出 |
| `make lint` | `cargo clippy -D warnings` |
| `make test` | `cargo test`（実機依存の `#[ignore]` テストは除く） |
| `make test-ignored` | 実機（Metal / CUDA）専用の `#[ignore]` 分離テスト |
| `make test-ignored-cuda` | CUDA 実機専用: `backend-cuda` の `#[ignore]` 分離テストのみ実行（TASK-1.7e・#36） |
| `make test-ignored-metal` | Metal 実機専用: `backend-metal` の `#[ignore]` 分離テストのみ実行・release（TASK-1.8e・#42） |
| `make deny` | `cargo deny --locked check advisories bans licenses sources`（依存の脆弱性・重複・ライセンス・取得元監査〈#353〉。`cargo-deny` 未導入なら自動導入） |
| `make deps-forbidden` | 依存禁止リスト（burn 系等）の混入検査 |
| `make ci` | CI（`.github/workflows/ci.yml`）と同一チェックの一括実行 |

`Cargo.toml`（TASK-1.1）・`deny.toml`（TASK-1.3）はいずれも追加済みのため cargo 系ターゲットは deny を含め全て実行されます。detect ガード（`HAS_CARGO`／`HAS_DENY` 判定）は、CI の detect ステップと同一の冪等セルフヒール方針（`.claude/rules/ci.md`）のフェイルセーフとして残置しています。`cargo-deny` サブコマンドは `make setup` の対象外ですが、`make deny`（`make ci` 経由も含む）実行時に未導入なら `cargo install cargo-deny --locked` で自動導入するため（CI の `deny` ジョブと同一の冪等セルフヒール方針）、クリーンなホスト環境でも追加手順なしで実行できます。開発コンテナ（後述）にはビルド時に導入済みです。

## Docker 開発環境

ホスト環境（macOS / Linux・aarch64 / amd64）に依存せずビルド・テスト・lint を実行できます。

```bash
make docker-build   # 開発コンテナイメージをビルド
make docker-shell   # 開発コンテナのシェルに入る
make docker-ci      # コンテナ内で make ci を実行（環境非依存の検証）
```

コンテナ内で使えるのは CPU（rayon）バックエンドのみです。Metal はホスト macOS 直接実行、CUDA は実機（DGX Spark GB10 等）で実行します（`cudarc` の動的ロード方式のため、CUDA バックエンドの「ビルド」は CUDA toolkit 無しのコンテナでも成立します）。

## 実機テスト（CUDA / Metal）

### CUDA 実機での `#[ignore]` テスト実行

`backend-cuda` の実機依存テスト（形状網羅・K=4096 ストレス・性能比較・デバイスメタデータ肯定的検証等）は通常 CI（GitHub ホステッド `ubuntu-latest`・CUDA toolkit 非搭載。方針は [`.claude/rules/ci.md`](.claude/rules/ci.md)）では `#[ignore]` により除外されます。CUDA ドライバ搭載の実機（DGX Spark GB10 等）で以下を実行してください（TASK-1.7e・#36）。

```bash
make test-ignored-cuda   # backend-cuda に限定した #[ignore] テスト実行（release）
# 相当コマンド:
cargo test -p fandhe-ai-backend-cuda --release -- --ignored --nocapture
```

`backend-cuda` 以外を含む全 `#[ignore]` テスト（Metal 実機分も含む）をまとめて実行したい場合は `make test-ignored`（`cargo test --workspace -- --ignored --nocapture`）を使ってください。

Tensor Core（WMMA TF32／f16）経路の TFLOPS 実測・複合判定通過の記録手順・記録テンプレートは [`docs/perf/cuda-tensor-core-measurement.md`](docs/perf/cuda-tensor-core-measurement.md) を参照してください（TASK-11.1e・#64）。

### Metal 実機での `#[ignore]` テスト実行

`backend-metal` の実機依存テスト（デバイス・バッファ基盤・naive/tiled/simdgroup GEMM の CPU 参照実装との数値一致・CPU-Metal ペア回帰等）は通常 CI（GitHub ホステッド・Linux）では `cfg(target_os = "macos")` と `#[ignore]` の二重分離により除外されます。Apple Silicon 実機で以下を実行してください（TASK-1.8e・#42。詳細手順・テスト一覧は [`docs/backend-metal-real-device-testing.md`](docs/backend-metal-real-device-testing.md) を参照）。

```bash
make test-ignored-metal   # backend-metal に限定した #[ignore] テスト実行（release）
# 相当コマンド:
cargo test -p fandhe-ai-backend-metal --release -- --ignored --nocapture
```

**実測状況**（Metal 実機検証・ベンチ計測トラッキングツリー、親 #379。2026-08-10 完了）: 上記
`#[ignore]` テスト 52 件は Apple Silicon 実機（M4 Max・macOS 26.6）で green を実測確認済み（#380）。
続けて f32 GEMM 4 段（naive/tiled/simdgroup/dynamic-tile）・境界形状 TFLOPS・f16 GEMM 対 PyTorch MPS
f16・起動コスト・ピークメモリのベンチ実測を完了し（#381〜#385）、REQ-8「Metal f16 初期リリース」の
性能下限を 15% に確定した（#386・人間承認済み）。詳細出典は
[`docs/backend-metal-real-device-testing.md`](docs/backend-metal-real-device-testing.md)・
[`docs/performance-targets.md`](docs/performance-targets.md) を参照。

## CI

- CI ランナー方針は GitHub ホステッド（`ubuntu-latest`）既定へ移行済み（public 区分。例外は codex-review の codex 実行ジョブのみ。#457 Phase 1〜3 完了。逆戻り防止は `runner-policy` ジョブ〈#472〉が fail-closed で検知。詳細 → [`.claude/rules/ci.md`](.claude/rules/ci.md)）
- `ci.yml`: `rust-ci`（fmt / clippy / test / deny の reusable workflow 呼び出し）＋固有ジョブ（build / build-no-cuda-toolkit / deps-forbidden / runner-policy / guardrail-regression / verification-gates）＋集約ジョブ `ci-complete`（fail-closed 集約の核。branch protection の required status check の詳細は `.claude/rules/ci.md`「ワークフロー設計」節を参照。二重管理を避けるため本節では書き写さない）
- `update-external.yml`: `docs/spec` サブモジュールと `.claude/skills` の自動追従（毎日 09:00 JST。PR label: `dependencies`・`automated`）。`docs/spec` は private リポジトリのため、org secret `SUBMODULE_PAT`（visibility=all）を優先参照して取得します（`GITHUB_TOKEN` はフォールバックのみで、public 化後も private submodule は取得できません。#463）

## 仕様

仕様書（ブレスト〜PoC〜要件定義〜タスク分解〜ロードマップ）は [Fandhe-AI/rust-ai-library-spec](https://github.com/Fandhe-AI/rust-ai-library-spec) で管理し、`docs/spec/` にサブモジュールとして取り込んでいます。本リポジトリでは編集しません。

| ドキュメント | 内容 |
|-------------|------|
| `docs/spec/04-requirements.md` | MoSCoW 優先度付き要件・受け入れ基準 |
| `docs/spec/05-tasks.md` | タスク分解（依存関係・工数） |
| `docs/spec/06-roadmap.md` | マイルストーン M0〜M5・着手判定 |

## 開発の進め方

### ロードマップ・タスク

`docs/spec/06-roadmap.md` のマイルストーン（M0〜M5）と `docs/spec/05-tasks.md` のタスク（4h 粒度）に従って実装します。M0（workspace・CI・依存監査ベースライン）は完了し、コア・3 バックエンドの実装と性能実測、crates.io 公開（v0.3.0）・ドキュメントサイト公開まで到達しています（「ステータス」節参照）。未着手・進行中の作業は GitHub Issues で追跡しています。

### Conventional Commits と git hooks

本プロジェクトは Conventional Commits を採用しています（型・スコープの詳細は [`.claude/rules/conventional-commits.md`](.claude/rules/conventional-commits.md) を参照）。

```
feat(scope): 日本語説明
fix(scope): バグ修正の説明
test(scope): テスト追加・修正
...
```

git hooks は [lefthook](https://lefthook.dev/)（`lefthook.yml`）で管理し、commit 時に以下が自動実行されます。

- `cargo fmt --all --check`（整形ガード）
- 簡易シークレット検知（API キー検出）
- Conventional Commits 形式検証

`--no-verify` による bypass は [`.claude/rules/conventional-commits.md`](.claude/rules/conventional-commits.md) で禁止されています。

### CI の実行

`make ci` で本リポジトリの CI（`.github/workflows/ci.yml`）と同一チェック（fmt / clippy / test / deny / deps-forbidden ほか）の一括実行ができます。

```bash
make ci
```

### Claude Code による開発体制

`.claude/` に Claude Code の運用体系（Agents・Rules・Skills・hooks）を整備しています。概要は [CLAUDE.md](./CLAUDE.md) を参照してください。

## ライセンス

本プロジェクトは [MIT ライセンス](./LICENSE-MIT) と [Apache License 2.0](./LICENSE-APACHE) の
デュアルライセンスで提供されます。あなたが本プロジェクトへ提出する Contribution は、明示的な
別段の定めがない限り、上記デュアルライセンスの下で提供されるものとみなされます。
