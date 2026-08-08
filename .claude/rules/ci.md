# CI 規約

## runner

- **CI はすべて self-hosted runner で実行する**（`runs-on: self-hosted`）。GitHub ホステッドランナー（`ubuntu-latest` 等）は使用しない。組織 runner 方針（private リポジトリは self-hosted）の正は Fandhe-AI/actions の `docs/runner-policy.md`
- self-hosted runner はハングしたジョブに無期限占有されうるため、**全ジョブに `timeout-minutes` を必ず設定する**（reusable workflow 呼び出しジョブは共通側の各ジョブが timeout を持つため呼び出し側での設定は不要）

## ベースライン品質ゲート（rust-base-ci reusable workflow）

fmt / clippy / test / cargo-deny の 4 ジョブ実体は Fandhe-AI/actions の **rust-base-ci reusable workflow** へ集約済み（イシュー #325・PR #344。local-llm-server と同型だった重複ジョブ定義の解消）。`.github/workflows/ci.yml` の `rust-ci` ジョブが `uses:` で呼び出す。

- **`uses:` の参照はコミット SHA へ固定する**（`@main` 禁止。更新時は `gh api repos/Fandhe-AI/actions/commits/main --jq '.sha'` で解決した SHA へ差し替え、行末コメントで追跡 ref を残す）
- **呼び出しジョブに `permissions: contents: read` を明示する**。reusable workflow は呼び出し側が付与した以上の権限を持てないため、共通側ジョブが必要とする最小権限を呼び出し側でも付与する
- **`runner-label: self-hosted` は既定値と同値でも省略しない**（組織 runner 方針の明示のため）。**`deny-checks: advisories bans licenses sources`**（共通側既定に bans を追加するため明示指定。イシュー #353。advisories / bans の設定と実測根拠は deny.toml の該当節コメントを正とする）と **`test-timeout-minutes: 45`**（共通側既定 30）も明示指定する。それ以外の入力（fmt / clippy / deny の timeout・`cache` 等）は共通側の既定値が本リポジトリの意図と同一である限り指定しない（既定値との二重管理を避ける）
- **`concurrency` は reusable workflow のトップレベルでは機能しない**ため、従来どおり呼び出し側 ci.yml 冒頭の `concurrency` 設定が同一 ref の直列化・キャンセルを担う
- **前提: リポジトリルートの `rust-toolchain.toml`**（channel = "stable"、components = ["rustfmt", "clippy"]）。共通側の rust-toolchain-setup action が単一真実源として参照する（rustup 冪等セルフヒール・toolchain 同期・component 追加は共通側実装）。components に rustfmt / clippy を列挙するのは本リポジトリ固有の理由による: self-repair の検証ゲート（`crates/self-repair/src/verify_gates.rs`）はリポジトリ外の sandbox（rust-toolchain.toml が及ばない一時ディレクトリ）で cargo clippy を起動するため、clippy component を持たない stable が runner に居ると rust-ci の test ジョブが落ちる（PR #344 CI 実測）。開発コンテナ（Dockerfile。ベースイメージは rust:1.88）にも stable を rustfmt / clippy / aarch64-apple-darwin target ごとイメージビルド時に事前導入してある（root 所有 RUSTUP_HOME への stable 自動インストールが dev ユーザーの権限エラーになるのを回避。この整合を壊さない）
- **Cargo.toml / deny.toml の有無判定（`detect` ステップ）は共通側の各ジョブが実施する**（不在時はステップをスキップしジョブは success のまま）。呼び出し側での判定は不要
- cargo-deny のバージョン固定導入・`deny-checks` 入力の fail-closed 検証など実装詳細は Fandhe-AI/actions の `.github/workflows/rust-base-ci.yml` および `rust-base-ci/README.md` を正とし、本規約では二重管理しない

## codex-review（PR 自動レビュー）

`.github/workflows/codex-review.yml` は Fandhe-AI/actions の codex-review reusable workflow を SHA 固定で呼び出す薄い wrapper（イシュー #326・PR #350。fandhe-backend / local-llm-server と同型）。

- private リポジトリ向けテンプレート（Fandhe-AI/actions `codex-review/templates/codex-review.private.yml`）を正とし、独自改変しない（`<SHA>` の差し替えとコメント追記のみ）
- runner ラベルは指定しない（reusable workflow の既定値: codex ジョブ `codex` / コメント投稿ジョブ `self-hosted` が組織 runner 方針に合致するため）
- 有効化スイッチは Actions variable `CODEX_HOME_DIR`（org 側供給あり。未設定なら codex ジョブが skip される fail-closed 設計）
- レビュー基準は `.github/codex/prompts/review.md`（カスタム版、イシュー #376）を正とする。本リポジトリは AGENTS.md を持たず、CLAUDE.md・`.claude/rules/` から抽出したリポジトリ固有基準（P0/P1 格上げ項目）を prompt へ直接埋め込む方式（fandhe-backend の AGENTS.md 方式との意図的な差分）。規約側（deps-policy / coding-rust / security / ci）を変更した際は prompt の基準との乖離を確認する。制御ファイルは PR の base コミットから読まれるため、変更はマージ後の PR から反映される（当の PR 自身のレビューには反映されない）

## ワークフロー設計（Fandhe-AI/local-llm-server・fandhe-multi-platform と同一方針。本リポ固有ジョブに適用）

本リポ固有の build / build-no-cuda-toolkit / deps-forbidden / guardrail-regression / verification-gates ジョブは引き続き ci.yml で直接定義し、以下に従う。

- `permissions` はワークフロー既定を `contents: read` の最小とし、必要なジョブのみ個別に昇格する
- サードパーティ actions は**コミット SHA に固定**する（タグ参照禁止。`actions/checkout@<sha>` 等）
- checkout 後に GITHUB_TOKEN を使わないジョブは `persist-credentials: false` を指定し、共有 runner の workspace に認証情報を残さない
- `concurrency` で同一 ref の重複実行を直列化・キャンセルする
- **グローバル状態を汚す処理を workflow に書かない**。ツール導入は「未導入の場合のみ導入する」冪等セルフヒール（Ensure rustup / Ensure component パターン）とする
- branch protection の required status check は集約ジョブ（`ci-complete`）のみを指定し、needs の result を明示検査して fail-closed で判定する。`rust-ci`（reusable workflow 呼び出し）の result は共通側の全ジョブ（fmt / clippy / test / deny / rust-base-ci-complete）を集約した結果になるため、`ci-complete` の needs に含めれば共通側の失敗も fail-closed に伝播する
- Cargo.toml 未追加の間は各ジョブの `detect` ステップで判定し cargo 系ステップをスキップする（ジョブは success のまま）。`jobs.<id>.if` は checkout 前に評価され `hashFiles` が使えないため、ステップ単位の `if:` で判定する

## 依存禁止検査（TASK-1.2）

- 依存禁止リスト（`burn` 系一式・`cubecl`・`candle`・`tch`・`ndarray`。deps-policy.md）の混入は CI で機械検査する。検査は `Cargo.lock` を対象とし fail-closed で判定する

## 実機依存

- CUDA 実機（DGX Spark GB10）・Metal 実機依存のテスト・ベンチは `#[ignore]` 分離を前提とし、通常 CI ジョブでは実行しない。実機ジョブを追加する場合は runner ラベルで対象 runner を明示する

## update-external.yml

- `.github/workflows/update-external.yml` は Fandhe-AI/rust-ai-library-v1 の同名ワークフローをほぼ変更せず流用する（docs/spec サブモジュール・.claude/skills の自動追従）。改変時は upstream と差分が出た理由をコメントに残す

## 秘密情報

- workflow に API キー・トークンをハードコードしない。`secrets.*` / `vars.*` 経由のみとする
- self-hosted runner 上に認証情報・キャッシュを残す処理を追加しない
