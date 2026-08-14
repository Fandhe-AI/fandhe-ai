# CI 規約

## runner（GitHub ホステッド既定。#457 Phase 1・イシュー #464 で public 区分へ反転）

- **CI ジョブは GitHub ホステッドランナー（`runs-on: ubuntu-latest` 等の標準スペック）を既定とする**。`runs-on: self-hosted`・larger runner（有料の大型ホステッドランナー）は使わない。本規約は組織 runner 方針（可視性で runner を決める: public は GitHub ホステッド／private は self-hosted）の **public 側**の適用であり、正は Fandhe-AI/actions の `docs/runner-policy.md`（本節では書き写さずドリフト防止）
- **CI 品質ゲート系ジョブにおける唯一の例外**: codex-review reusable workflow の codex 実行ジョブ（`runner-label: codex`。codex-home 方式の認証情報を runner 上に配置する構成のため self-hosted な codex 専用 runner を使用）。例外は codex 実行ジョブに閉じ、PR コメント投稿ジョブ（`post-feedback-runner-label`）は資格情報に触れないため `ubuntu-latest` を明示指定する（詳細は「codex-review」節）。実機（CUDA/Metal）依存ジョブは本節の例外規定と別枠であり、`docs/runner-policy.md` の例外整理に従い将来 self-hosted で追加しうる（詳細は「実機依存」節）
- **移行完了の記録（#457 Phase 1〜3 完了。イシュー #470）**: 本リポジトリの GitHub 可視性は `gh repo view Fandhe-AI/rust-ai-library --json visibility,isPrivate` で `{"isPrivate":false,"visibility":"PUBLIC"}`（イシュー #470 で 2026-08-14 に再実測・確認済み。初回実測は PR #476 コミット 307b453 時点・2026-08-13T20:08:47Z〈UTC〉）。公開前トラッキング（#457）は Phase 1（規約先行反転・#464）→ Phase 2（workflow 実体移行・#465〜#469）→ Phase 3（visibility 確認・#470）の順に完了した。Phase 2 の完了により `.github/workflows/*.yml` 内の `runs-on: self-hosted` は ci-complete を含む全ジョブで解消済み（PR #618・#619 実測。main への push run 31774835151 が `ci-complete` 含む全 11 ジョブ success。codex 専用 runner の例外を除く）。残る追跡事項は Fandhe-AI/actions 側 `docs/runner-policy.md` §2 の対象リポジトリ一覧更新（Fandhe-AI/actions#471）のみであり、別リポジトリの後続作業のため本リポジトリの規約適用には影響しない
- **codex-review 判定との非対称性（一過性・解消済み）**: codex-review の判定基準（`AGENTS.md`・`.github/codex/prompts/review.md`）は PR の base コミットから読まれるため、規約反転を含む PR 自身の変更はその PR 自身の codex-review 判定には反映されない特性がある（詳細は「codex-review」節）。本リポジトリは #464〜#469 のマージにより base 側基準が既に反転後の内容へ追従済みであるため、この非対称は解消している
- runner 種別に依らず、ハングしたジョブが runner を無期限占有するのを防ぐため**全ジョブに `timeout-minutes` を必ず設定する**（reusable workflow 呼び出しジョブは共通側の各ジョブが timeout を持つため呼び出し側での設定は不要）
- **逆戻り防止の機械検査（イシュー #472）**: 上記移行完了後の `self-hosted` 再導入を CI 常時実行で検知する `scripts/check-workflow-runner-policy.sh`（`ci.yml` の `runner-policy` ジョブ・`Makefile` の `runner-policy` ターゲットと共用する単一ソース。本体は追加依存なしの `scripts/check-workflow-runner-policy.py`）を設けている。`.github/workflows/` 配下の非コメント行で `runner-label` 等が許容形（標準 GitHub ホステッドランナーの `-latest` ラベル集合: `ubuntu-latest`・`macos-latest`・`windows-latest`。larger runner・バージョン固定ラベル・`self-hosted` はすべて対象外）のいずれとも完全一致しなければ fail-closed で違反とする（除外リスト方式は取らない）。codex 専用 runner の例外（上記）は本リポ側ファイルに runner 宣言が現れないため自然に検査対象外となる。仕様詳細は同スクリプト冒頭コメントを正とし本節では二重管理しない。実機ジョブを将来 self-hosted で追加する場合は許容形をスクリプト側で明示的に更新する

## ベースライン品質ゲート（rust-base-ci reusable workflow）

fmt / clippy / test / cargo-deny の 4 ジョブ実体は Fandhe-AI/actions の **rust-base-ci reusable workflow** へ集約済み（イシュー #325・PR #344。local-llm-server と同型だった重複ジョブ定義の解消）。`.github/workflows/ci.yml` の `rust-ci` ジョブが `uses:` で呼び出す。

- **`uses:` の参照はコミット SHA へ固定する**（`@main` 禁止。更新時は `gh api repos/Fandhe-AI/actions/commits/main --jq '.sha'` で解決した SHA へ差し替え、行末コメントで追跡 ref を残す）
- **呼び出しジョブに `permissions: contents: read` を明示する**。reusable workflow は呼び出し側が付与した以上の権限を持てないため、共通側ジョブが必要とする最小権限を呼び出し側でも付与する
- **`runner-label: ubuntu-latest` を明示指定する**（共通側 reusable workflow の入力既定値は `self-hosted` のため、public 区分への反転を明示するオーバーライドとして必須。実 yml の切替は Phase 2 #465 で完了済み。PR #618 実測で rust-ci〈fmt / clippy / test / deny・rust-base-ci-complete〉が ubuntu-latest 上で全 green、共通側 rust-toolchain-setup もホステッドで機能することを確認済みのため共通側 Fandhe-AI/actions への追加対応は不要と判断した）。**`deny-checks: advisories bans licenses sources`**（共通側既定に bans を追加するため明示指定。イシュー #353。advisories / bans の設定と実測根拠は deny.toml の該当節コメントを正とする）と **`test-timeout-minutes: 45`**（共通側既定 30）も明示指定する。それ以外の入力（fmt / clippy / deny の timeout・`cache` 等）は共通側の既定値が本リポジトリの意図と同一である限り指定しない（既定値との二重管理を避ける）。**`cache` の既定値**は self-hosted の永続 workspace を前提に設計されているため、ホステッド（ジョブごとに使い捨ての VM）へ移行後のキャッシュ設計（キー構成・restore-keys）は、build 系 3 ジョブは #466 で確立済みだが、`rust-ci` ジョブ（reusable workflow の `cache` 入力）はスコープ外のため #622 で別途確立する
- **`concurrency` は reusable workflow のトップレベルでは機能しない**ため、従来どおり呼び出し側 ci.yml 冒頭の `concurrency` 設定が同一 ref の直列化・キャンセルを担う
- **前提: リポジトリルートの `rust-toolchain.toml`**（channel = "stable"、components = ["rustfmt", "clippy"]）。共通側の rust-toolchain-setup action が単一真実源として参照する（rustup 冪等セルフヒール・toolchain 同期・component 追加は共通側実装）。components に rustfmt / clippy を列挙するのは本リポジトリ固有の理由による: self-repair の検証ゲート（`crates/self-repair/src/verify_gates.rs`）はリポジトリ外の sandbox（rust-toolchain.toml が及ばない一時ディレクトリ）で cargo clippy を起動するため、clippy component を持たない stable が runner に居ると rust-ci の test ジョブが落ちる（PR #344 CI 実測）。開発コンテナ（Dockerfile。ベースイメージは rust:1.88）にも stable を rustfmt / clippy / aarch64-apple-darwin target ごとイメージビルド時に事前導入してある（root 所有 RUSTUP_HOME への stable 自動インストールが dev ユーザーの権限エラーになるのを回避。この整合を壊さない）
- **Cargo.toml / deny.toml の有無判定（`detect` ステップ）は共通側の各ジョブが実施する**（不在時はステップをスキップしジョブは success のまま）。呼び出し側での判定は不要
- cargo-deny のバージョン固定導入・`deny-checks` 入力の fail-closed 検証など実装詳細は Fandhe-AI/actions の `.github/workflows/rust-base-ci.yml` および `rust-base-ci/README.md` を正とし、本規約では二重管理しない

## codex-review（PR 自動レビュー）

`.github/workflows/codex-review.yml` は Fandhe-AI/actions の codex-review reusable workflow を SHA 固定で呼び出す薄い wrapper（イシュー #326・PR #350。fandhe-backend / local-llm-server と同型）。

- **public 区分向けテンプレート**（Fandhe-AI/actions `codex-review/templates/codex-review.public.yml`）を正とし、独自改変しない（`<SHA>` の差し替えとコメント追記のみ。wrapper の実切替は Phase 2 #469 で完了済み〈pin SHA `39d6d5cfd275338ec2e4bcead179b6a017712772` は同 SHA から取得したテンプレートと `diff` 済みで、差異はコメント行のみであることを機械的に確認済み〉）
- runner ラベルは**ジョブごとに分ける**: codex 実行ジョブは指定しない（reusable workflow の既定値 `codex`。codex-home 方式の認証情報を持つ self-hosted 専用 runner が「runner」節の唯一の例外）。**PR コメント投稿ジョブ（`post-feedback-runner-label`）は `ubuntu-latest` を明示指定する**（資格情報に触れないため public 区分の既定に従う）。pin SHA `39d6d5cfd275338ec2e4bcead179b6a017712772` の reusable workflow 側で `post-feedback-runner-label` 入力の既定値は `self-hosted` であるため（実測済み。#469）、`codex-review.public.yml` テンプレート自体が `with:` ブロックで `ubuntu-latest` を明示指定している。この明示指定はテンプレート自体の構成であり「独自改変しない」方針と矛盾しない
- 有効化スイッチは Actions variable `CODEX_HOME_DIR`（org 側供給あり。未設定なら codex ジョブが skip される fail-closed 設計）
- レビュー基準は `.github/codex/prompts/review.md`（カスタム版、イシュー #376）と `AGENTS.md`（リポジトリ観点の正。両者の優先度定義が矛盾する場合は AGENTS.md を優先）を正とする。CLAUDE.md・`.claude/rules/` から抽出したリポジトリ固有基準（P0/P1 格上げ項目）を両ファイルへ整合させて反映する方式。規約側（deps-policy / coding-rust / security / ci）を変更した際は prompt・AGENTS.md 双方との乖離を確認する。制御ファイルは PR の base コミットから読まれるため、変更はマージ後の PR から反映される（当の PR 自身のレビューには反映されない）

## ワークフロー設計（Fandhe-AI/local-llm-server・fandhe-multi-platform と同一方針。本リポ固有ジョブに適用）

本リポ固有の build / build-no-cuda-toolkit / deps-forbidden / guardrail-regression / verification-gates ジョブは引き続き ci.yml で直接定義し、以下に従う。

- `permissions` はワークフロー既定を `contents: read` の最小とし、必要なジョブのみ個別に昇格する
- サードパーティ actions は**コミット SHA に固定**する（タグ参照禁止。`actions/checkout@<sha>` 等）
- checkout 後に GITHUB_TOKEN を使わないジョブは `persist-credentials: false` を指定する。ホステッドランナーはジョブごとに使い捨ての VM だが、codex 専用 runner（唯一の self-hosted 例外）は永続環境のため、認証情報を workspace に残さない多層防御として維持する
- **fork PR 対策（新規。public 化に伴い fork PR が現実化するため明文化）**: fork PR へ secrets を露出するトリガー（`pull_request_target`・secrets を渡す `workflow_run` 等）を新規に追加しない。codex 専用 runner（永続環境）に対する fork PR 実行拒否等の多層防御を弱体化しない
- `concurrency` で同一 ref の重複実行を直列化・キャンセルする
- **グローバル状態を汚す処理を workflow に書かない**。ツール導入は「未導入の場合のみ導入する」冪等セルフヒール（Ensure rustup / Ensure component パターン）とする。ホステッドランナーへの実体移行後（Phase 2）は使い捨て VM のため常に新規導入側へ倒れるが、導入ステップ（バージョン固定・検証）は弱体化しない
- branch protection の required status check は**設計上**集約ジョブ（`ci-complete`）を核とし、needs の result を明示検査して fail-closed で判定する。`rust-ci`（reusable workflow 呼び出し）の result は共通側の全ジョブ（fmt / clippy / test / deny / rust-base-ci-complete）を集約した結果になるため、`ci-complete` の needs に含めれば共通側の失敗も fail-closed に伝播する
  - **実測との差分（イシュー #470 で確認。2026-08-14）**: 本リポジトリは classic branch protection ではなく repository ruleset（`main-protection`・id `20587668`。`gh api repos/.../branches/main/protection` は 404 になり `gh api repos/.../rulesets` で確認する）を使う。`enforcement: active`・`bypass_actors: []`・`current_user_can_bypass: "never"`・`strict_required_status_checks_policy: true` を実測済みで、`required_status_checks` に `ci-complete` は含まれている。ただし実際の一覧には `ci-complete` に加え、`ci.yml` 内で `ci-complete` の `needs` に既に含まれるジョブ（`rust-ci / cargo fmt --check`・`rust-ci / cargo clippy`・`rust-ci / cargo test`・`rust-ci / cargo deny check`・`rust-ci / rust-base-ci-complete`・`cargo build (linux / aarch64-apple-darwin)`・`cargo build / test (CUDA toolkit 非搭載検証)`・`forbidden dependencies check`・`guardrail 2 層検証（REQ-4/REQ-5）`・`検証ゲート（build/test/clippy）`）が個別にも required として重複列挙されている。これは fail-closed 判定をより厳格にする方向の重複であり安全側の逸脱（laxer ではない）。一方 `codex-review / codex`・`codex-review / post_feedback`・`Cursor Bugbot` は `ci.yml` 外の別 workflow／外部 app のチェックのため `needs` で `ci-complete` に集約できず、個別 required 化が必須（設計上の想定内）。`ci.yml` 内ジョブの重複列挙を `ci-complete` のみへ整理するかは ruleset 20587668 への変更（人間承認が要る運用上の設定変更）であり本規約変更だけでは反映されない
- Cargo.toml 未追加の間は各ジョブの `detect` ステップで判定し cargo 系ステップをスキップする（ジョブは success のまま）。`jobs.<id>.if` は checkout 前に評価され `hashFiles` が使えないため、ステップ単位の `if:` で判定する

## 依存禁止検査（TASK-1.2）

- 依存禁止リスト（`burn` 系一式・`cubecl`・`candle`・`tch`・`ndarray`。deps-policy.md）の混入は CI で機械検査する。検査は `Cargo.lock` を対象とし fail-closed で判定する

## 実機依存

- CUDA 実機（DGX Spark GB10）・Metal 実機依存のテスト・ベンチは `#[ignore]` 分離を前提とし、通常 CI ジョブでは実行しない。実機ジョブを将来追加する場合は `docs/runner-policy.md` の例外整理に従い runner ラベルで対象 runner を明示する（例外の無断拡大は不可）

## update-external.yml

- `.github/workflows/update-external.yml` は Fandhe-AI/rust-ai-library-v1 の同名ワークフローをほぼ変更せず流用する（docs/spec サブモジュール・.claude/skills の自動追従）。改変時は upstream と差分が出た理由をコメントに残す

## 秘密情報

- workflow に API キー・トークンをハードコードしない。`secrets.*` / `vars.*` 経由のみとする
- runner 上に認証情報・キャッシュを残す処理を追加しない（codex 専用 runner は永続環境のためとくに注意する）
