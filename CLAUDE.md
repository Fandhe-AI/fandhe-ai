# CLAUDE.md

## Overview

Rust 製 AI/ML ライブラリの実装リポジトリ（v2）。Burn 依存を排した**完全自作コア**（テンソル・autodiff・演算グラフ／カーネル融合機構・計算カーネル・バックエンド抽象層）で実装する。仕様の正本は [Fandhe-AI/fandhe-ai-spec](https://github.com/Fandhe-AI/fandhe-ai-spec)（`docs/spec` submodule）にあり、本リポでは編集しない。本リポジトリ自体は **public**（#457 Phase 1〜3 完了）で、CI は GitHub ホステッド `ubuntu-latest` 既定へ移行済み（self-hosted への逆戻りは `runner-policy` ジョブ〈#472〉が fail-closed で検知。詳細 → `.claude/rules/ci.md`）。仕様 submodule（`docs/spec`）と旧実装（v1）は private を維持する（README「位置づけ」節）。

- 想定クレート 10 個: `tensor-core`・`autodiff`・`backend-cpu`・`backend-cuda`・`backend-metal`・`onnx-interop`・`guardrail`・`self-repair`・`bench-harness`・`facade`（TASK-9.3・イシュー #410 で新設した composition root。TASK-9.4・イシュー #411 で `autodiff::compat` から compat 公開面〈`compat::array`・`compat::Sequential`〉を移設済み）に加え、GitHub Pages 公開ツリー（#865 Phase 1）向けの開発者・CI 専用 SSG クレート `docs-site`（11 個目・`publish = false`。イシュー #868/#869）。**`facade` が唯一のサポートされる公開 API 面**であり `tensor-core`・`autodiff`・`backend-*` は内部クレート（範囲の正は `docs/compat-api-scope.md` §0）。上記の名称はディレクトリ名（`crates/<name>`）であり変更しない
- crates.io 公開済みは 6 クレート（`facade`・`tensor-core`・`autodiff`・`backend-cpu`・`backend-cuda`・`backend-metal`）で、`[package] name` は `fandhe-ai` prefix 付きの公開名（`docs/crates-io-naming-decision.md`）。`onnx-interop` は #1963 のユーザー承認（2026-09-17）を受けた 7 クレート目で、公開準備は完了済み・実 publish は次回リリースサイクル。`guardrail`・`self-repair`・`bench-harness`・`docs-site` は非公開。facade からの ONNX import／export・safetensors save／load の公開範囲は `docs/facade-onnx-import-exposure-decision.md`・`docs/facade-onnx-export-exposure-decision.md`・`docs/facade-safetensors-exposure-decision.md` を正とする
- crates.io への公開は一括リリース `.github/workflows/release-all.yml`（workflow_dispatch 1 回・environment `crates-io-release` 承認 1 回で公開クレートを依存順に publish）を基本とし、単一クレートの再実行・障害復旧には `.github/workflows/release.yml` を使う（いずれも `CARGO_REGISTRY_TOKEN`〈org secret〉・fail-closed ガード群）。手順・版数運用・公開履歴（v0.3.0〜v0.9.0）の正は `docs/crates-io-publishing-order.md` §9〜11・`.claude/rules/ci.md`
- 依存は許容 8 区分のみ・`=x.y.z` 完全固定（`.claude/rules/deps-policy.md`）。禁止リスト（`burn` 系・`cubecl`・`candle`・`tch`・`ndarray`）は CI で機械検査
- バックエンド切替は feature フラグなしの cfg ベース（PoC-v2-5 実証構成）
- 現状 M0 完了（workspace Cargo.toml・クレート雛形・依存禁止検査・deny.toml・license-matrix.md）。crates.io 公開済み（最新 v0.9.0）・GitHub Pages 公開済み（`crates/docs-site` + `site/` + `.github/workflows/docs-site.yml`）。CI・Makefile の cargo 系チェック（fmt / clippy / test / deny / deps-forbidden）は全て有効化済み

## Repository Structure

```
fandhe-ai/
├── CLAUDE.md                # 本ファイル
├── README.md                # 開発環境構築・実装方針の要点
├── LICENSE-APACHE           # Apache License 2.0 全文（MIT/Apache-2.0 デュアルライセンス。#462）
├── LICENSE-MIT              # MIT ライセンス本文（同上）
├── Makefile                 # make setup / ci / docker-* タスクランナー
├── lefthook.yml             # git hooks（rustfmt-check・secrets-guard・commit-msg・pre-push）
├── .editorconfig            # インデント・改行規約
├── Dockerfile / compose.yaml # 環境非依存の開発コンテナ（CPU バックエンドのみ）
├── skills-lock.json         # 導入スキルのハッシュ管理（npx skills）
├── Cargo.toml                # workspace 定義（本体 10 クレート + docs-site〈開発ツール〉・許容依存 8 区分を =x.y.z 固定）
├── Cargo.lock                # 依存解決の完全固定（deps-policy.md）
├── rust-toolchain.toml       # toolchain 単一真実源（stable + rustfmt/clippy。rust-base-ci 前提。#325）
├── deny.toml                 # cargo-deny 設定（licenses 許可リスト・sources = crates.io 限定〈TASK-1.3〉+ advisories / bans〈#353〉）
├── guardrail.toml             # guardrail 判定閾値の確定設定（TASK-4.3c・#117。default プリセット）
├── crates/                  # tensor-core・autodiff・backend-cpu・backend-cuda・backend-metal・
│                             # onnx-interop・guardrail・self-repair・bench-harness・facade（composition root・compat 公開面）・
│                             # docs-site（GitHub Pages 公開ツリー向け SSG。開発者・CI 専用。#868/#869）
├── scripts/
│   ├── check-forbidden-deps.sh # 依存禁止リストの検査ロジック（ci.yml・Makefile 共用。TASK-1.2）
│   ├── check-workflow-runner-policy.sh # self-hosted runner 逆戻り防止の fail-closed 契約検査の呼び出し面（ci.yml・Makefile 共用。#472）
│   ├── check-workflow-runner-policy.py # 同検査の本体（python3 標準ライブラリのみの自前 YAML サブセットパーサー方式。追加依存なしで表記トリック迂回を遮断。#472・PR #626）
│   ├── run-verification-gates.sh # AI 自律メンテナンス検証 4 ゲート（build/test/clippy/bench）の実行ロジック（ci.yml・Makefile 共用。TASK-6.1c）
│   ├── run-guardrail-regression.sh # guardrail 2 層検証ロジック（ci.yml・schedule 共用。TASK-6.1a）
│   ├── report-guardrail-schedule-result.sh # schedule 定期実行失敗時の Issue 起票・復旧クローズ（TASK-6.1b）
│   ├── report-clippy-nocache-schedule-result.sh # キャッシュなしフルビルド clippy 定期検証の失敗時 Issue 起票・復旧クローズ（イシュー #918）
│   ├── testdata/             # 上記の self-test 用固定 fixture
│   └── bench/
│       ├── oss-gemm-compare/ # CPU GEMM OSS 直接比較ハーネス（本体 workspace 外の独立 Cargo パッケージ。matrixmultiply・gemm crate。イシュー #755）
│       ├── gemm_bench_torch_mps_f16.py／gemm_bench_torch_mps_f32.py # PyTorch MPS 参照計測
│       ├── gemm_bench_torch_cpu_f32.py # PyTorch CPU f32 GEMM 参照計測（イシュー #1141）
│       ├── gemm_bench_mlx_f32.py # MLX f32 GEMM 計測（イシュー #755）
│       └── framework-compare/parity_torch_truth.py # PyTorch GEMM の parity fail 要素を厳密真値と突合する診断専用ツール（#1184 の `parity_dump_truth.py` 同型。`bench_py.py` と同じ入力・参照・複合判定を再現し c／線形 K・√K 形の救済表を出力。`--self-test` は torch 不在で torch 経路 skip。イシュー #1985）
├── .github/workflows/
│   ├── ci.yml               # rust-ci（Fandhe-AI/actions rust-base-ci 呼び出し: fmt / clippy / test / deny。#325）+ 固有ジョブ（build / build-no-cuda-toolkit / deps-forbidden / runner-policy / guardrail-regression / verification-gates）+ ci-complete
│   ├── ai-review.yml        # ai-review（provider: codex）による PR 自動レビュー wrapper（Fandhe-AI/actions ai-review を `@latest` 呼び出し。#326。旧 codex-review.yml から移行。public 構成〈post-feedback-runner: ubuntu-latest〉へ切替済み。#469）
│   ├── verification-gate-bench.yml # bench ゲート（schedule／workflow_dispatch。TASK-6.1c）
│   ├── guardrail-regression-schedule.yml # guardrail 2 層検証の schedule 定期実行・失敗時 Issue 可視化（TASK-6.1b）
│   ├── clippy-nocache-schedule.yml # キャッシュなしフルビルド clippy の定期検証・失敗時 Issue 可視化（イシュー #918）
│   ├── release.yml          # crates.io publish（workflow_dispatch + `CARGO_REGISTRY_TOKEN`・environment `crates-io-release` 承認ゲート。#884。手順は `docs/crates-io-publishing-order.md` §9〜11・`.claude/rules/ci.md` release.yml 節）
│   ├── update-external.yml  # docs/spec・.claude/skills の自動追従
│   └── docs-site.yml        # GitHub Pages ビルド・デプロイ（Fandhe-AI/actions `pages-deploy.yml` 呼び出し）
├── site/                    # GitHub Pages 公開原稿（`nav.toml` + Markdown。#873/#874/#875。`docs/spec` の内容は含めない）
├── .claude/
│   ├── agents/              # research / implement / testing / quality / docs
│   ├── rules/               # 委譲・コーディング・依存・CI・セキュリティ等の規約
│   ├── skills/              # npx skills add で導入（skills-lock.json 管理）
│   ├── workflows/           # implement-issue-tree.js（skills への相対 symlink）
│   └── settings.json        # SessionStart / PostToolUse hooks
└── docs/
    ├── README.md            # docs/ 配下（設計判断記録・perf 実測記録・perf/logs 実測ログ）の注釈付き索引。doc の新設・追記に伴う索引更新は本ファイルに対して行い CLAUDE.md には書かない（CLAUDE.md から逐語移設）
    ├── perf/                # 性能実測・下限確定の記録群（`performance-floor-decision.md` ほか。GEMM 最適化ツリー #479 の実測記録を含む。個々の doc の索引は `docs/README.md`）
    └── spec/                # 正本 submodule（fandhe-ai-spec。編集禁止）
        ├── 04-requirements.md  # REQ-1〜14
        ├── 05-tasks.md         # TASK 一覧（4h 粒度）
        ├── 06-roadmap.md       # M0〜M5・全 51 タスク
        └── 03-poc/             # PoC 実測（v2 系は poc-v2-*）
```

## 委譲方針（必読）

main はコンテキスト消費を抑えるため判断と統合に専念し、調査・実装・テスト・レビューは subagent へ委譲する。詳細は `.claude/rules/delegation.md`（調査・設計）・`delegation-impl.md`（作成・編集）を参照。

### model 配分

| 用途 | model |
|------|-------|
| 複雑な横断判断・アーキテクチャ設計 | opus または fable（fable は特に大規模設計・横断判断の最上位 tier） |
| 調査・生成・実装・レビュー | sonnet |
| 機械的集計・lint・ドキュメント更新 | haiku |

## Sub-agents

| カテゴリ | subagent_type | 担当 | model |
|---------|---------------|------|-------|
| research | explorer | コードベース・docs/spec 横断調査（読み取り専用） | sonnet |
| research | reference-researcher | cudarc/CUDA・objc2/Metal・safetensors/ONNX 等の外部仕様調査 | sonnet |
| implement | core-builder | `tensor-core`・`autodiff`・workspace 骨格・`facade`（composition root・compat API 層） | sonnet |
| implement | backend-builder | `backend-cpu`・`backend-cuda`・`backend-metal`・数値一致回帰テスト | sonnet |
| implement | interop-builder | `onnx-interop`（safetensors / prost 自前取り込み） | sonnet |
| implement | runtime-builder | `guardrail`・`self-repair`・`bench-harness` | sonnet |
| testing | test-runner | テスト実行・追加・失敗解析（実機依存は `#[ignore]` 分離） | sonnet |
| testing | bench-runner | ベンチ計測・性能回帰検出（5 回計測中央値・読み取り専用） | sonnet |
| quality | reviewer | コードレビュー（spec 突合・読み取り専用） | sonnet |
| quality | security-auditor | OWASP Top 10・unsafe・ライセンス監査（読み取り専用） | sonnet |
| quality | linter | fmt / clippy / frontmatter lint の機械的実行 | haiku |
| docs | docs-writer | CLAUDE.md・README・license-matrix 等の更新 | haiku |

## Rules

| ファイル | 内容 |
|---------|------|
| `.claude/rules/delegation.md` | 調査・設計フェーズの委譲原則・パスベース切り替え |
| `.claude/rules/delegation-impl.md` | 作成・編集フェーズの委譲マッピング・実装フロー標準 |
| `.claude/rules/coding-rust.md` | 完全自作コア方針・cfg ベースバックエンド・FMA 契約統一・品質基準 |
| `.claude/rules/deps-policy.md` | 許容依存 8 区分・`=x.y.z` 完全固定・禁止リスト・ライセンス要件 |
| `.claude/rules/ci.md` | **CI は GitHub ホステッド（`ubuntu-latest`）既定**（例外は ai-review の codex 実行ジョブのみ）・fork PR 対策・timeout 必須・SHA 固定・fail-closed 集約 |
| `.claude/rules/security.md` | OWASP Top 10・秘密情報混入防止・自己修復ループのガードレール |
| `.claude/rules/japanese-style.md` | 日本語出力スタイル |
| `.claude/rules/conventional-commits.md` | Conventional Commits 詳細規約（`--no-verify` 禁止） |
| `.claude/rules/code-comment-style.md` | コメント規約（役割・責務・呼び出し文脈・spec 根拠を埋め込む） |
| `.claude/rules/out-of-scope-tracking.md` | 実装対象外の追跡規約（スコープ外事項を放置しない） |

## Current Skills

`npx skills add` で導入済み（`skills-lock.json` 管理。更新は update-external.yml が自動追従）。

- **Git/GitHub 運用**: create-commit・create-pr・create-issue・create-issue-tree・update-issue-tree
- **実装フロー**: create-plan・implement-issue・implement-issue-tree・implement-review・implement-review-pr
- **ドキュメント**: update-docs・comment-code
- **スキル管理**: init-claude・update-claude・contribute-skill・sync-skills-lock
- **技術リファレンス**: rust・nvidia-cuda・apple-silicon・amd-rocm・lefthook・editorconfig・commitlint・github-docs

## Conventions

- 日本語でやりとり・報告・コミット・PR を書く（`japanese-style.md`）
- Conventional Commits 厳守・`--no-verify` 禁止（`conventional-commits.md`）
- 依存の追加・更新、ガードレール閾値・テスト許容誤差の変更はユーザー承認必須
- `docs/spec/`（正本 submodule）は編集しない。仕様変更は spec リポ側で対応する
- implement-issue は計画のユーザー承認後に実装する
- スコープ外事項は `out-of-scope-tracking.md` の規約に沿って Issue で追跡する
- `docs/` の doc 新設・イシュー別追記は**各 doc 本文と `docs/README.md`（索引）へ書き、CLAUDE.md には書かない**。CLAUDE.md は毎セッション自動ロードされるため、イシュー別の履歴を積むとコンテキストを圧迫する（`.claude/rules/` も同様に自動ロードされるので規約以外を置かない）

## hooks（settings.json）

- **SessionStart**: 日本語・委譲・完全自作コア・CI は GitHub ホステッド・Conventional Commits のリマインダーを表示する
- **PostToolUse**（Edit|Write）: `.rs` 編集時に rustfmt を自動適用する（Cargo.toml の edition を検出。未追加時は 2021 フォールバック）
