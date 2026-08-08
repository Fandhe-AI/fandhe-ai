# CLAUDE.md

## Overview

Rust 製 AI/ML ライブラリの実装リポジトリ（v2）。Burn 依存を排した**完全自作コア**（テンソル・autodiff・演算グラフ／カーネル融合機構・計算カーネル・バックエンド抽象層）で実装する。仕様の正本は [Fandhe-AI/rust-ai-library-spec](https://github.com/Fandhe-AI/rust-ai-library-spec)（`docs/spec` submodule）にあり、本リポでは編集しない。

- 想定クレート 9 個: `tensor-core`・`autodiff`・`backend-cpu`・`backend-cuda`・`backend-metal`・`onnx-interop`・`guardrail`・`self-repair`・`bench-harness`
- 依存は許容 8 区分のみ・`=x.y.z` 完全固定（`.claude/rules/deps-policy.md`）。禁止リスト（`burn` 系・`cubecl`・`candle`・`tch`・`ndarray`）は CI で機械検査
- バックエンド切替は feature フラグなしの cfg ベース（PoC-v2-5 実証構成）
- 現状 M0 着手中（TASK-1.1a: workspace `Cargo.toml` と 9 クレート雛形を追加済み。TASK-1.1b: 許容依存 8 区分を `[workspace.dependencies]` に `=x.y.z` 完全固定で反映し `Cargo.lock` をコミット済み。TASK-1.2: 依存禁止検査は CI 上で稼働中〈green〉。TASK-1.3: `deny.toml` 導入・`docs/license-matrix.md` 作成済み）。CI・Makefile の cargo 系チェック（fmt / clippy / test / deny / deps-forbidden）は全て有効化済み

## Repository Structure

```
rust-ai-library/
├── CLAUDE.md                # 本ファイル
├── README.md                # 開発環境構築・実装方針の要点
├── Makefile                 # make setup / ci / docker-* タスクランナー
├── lefthook.yml             # git hooks（rustfmt-check・secrets-guard・commit-msg・pre-push）
├── .editorconfig            # インデント・改行規約
├── Dockerfile / compose.yaml # 環境非依存の開発コンテナ（CPU バックエンドのみ）
├── skills-lock.json         # 導入スキルのハッシュ管理（npx skills）
├── Cargo.toml                # workspace 定義（9 クレート・許容依存 8 区分を =x.y.z 固定）
├── Cargo.lock                # 依存解決の完全固定（deps-policy.md）
├── rust-toolchain.toml       # toolchain 単一真実源（stable + rustfmt/clippy。rust-base-ci 前提。#325）
├── deny.toml                 # cargo-deny 設定（licenses 許可リスト・sources = crates.io 限定〈TASK-1.3〉+ advisories / bans〈#353〉）
├── guardrail.toml             # guardrail 判定閾値の確定設定（TASK-4.3c・#117。default プリセット）
├── crates/                  # tensor-core・autodiff・backend-cpu・backend-cuda・backend-metal・
│                             # onnx-interop・guardrail・self-repair・bench-harness（雛形）
├── scripts/
│   ├── check-forbidden-deps.sh # 依存禁止リストの検査ロジック（ci.yml・Makefile 共用。TASK-1.2）
│   ├── run-verification-gates.sh # AI 自律メンテナンス検証 4 ゲート（build/test/clippy/bench）の実行ロジック（ci.yml・Makefile 共用。TASK-6.1c）
│   ├── run-guardrail-regression.sh # guardrail 2 層検証ロジック（ci.yml・schedule 共用。TASK-6.1a）
│   ├── report-guardrail-schedule-result.sh # schedule 定期実行失敗時の Issue 起票・復旧クローズ（TASK-6.1b）
│   └── testdata/             # 上記の self-test 用固定 fixture
├── .github/workflows/
│   ├── ci.yml               # rust-ci（Fandhe-AI/actions rust-base-ci 呼び出し: fmt / clippy / test / deny。#325）+ 固有ジョブ（build / build-no-cuda-toolkit / deps-forbidden / guardrail-regression / verification-gates）+ ci-complete
│   ├── codex-review.yml     # Codex PR 自動レビュー wrapper（Fandhe-AI/actions codex-review を SHA 固定呼び出し。#326）
│   ├── verification-gate-bench.yml # bench ゲート（schedule／workflow_dispatch。TASK-6.1c）
│   ├── guardrail-regression-schedule.yml # guardrail 2 層検証の schedule 定期実行・失敗時 Issue 可視化（TASK-6.1b）
│   └── update-external.yml  # docs/spec・.claude/skills の自動追従
├── .claude/
│   ├── agents/              # research / implement / testing / quality / docs
│   ├── rules/               # 委譲・コーディング・依存・CI・セキュリティ等の規約
│   ├── skills/              # npx skills add で導入（skills-lock.json 管理）
│   ├── workflows/           # implement-issue-tree.js（skills への相対 symlink）
│   └── settings.json        # SessionStart / PostToolUse hooks
└── docs/
    ├── backend-metal-wgpu-decision.md  # Metal バックエンド実装方式（wgpu 非採用）の決定記録
    ├── backend-switching-design.md     # cfg ベースバックエンド切替の設計
    ├── cuda-tensor-core-design.md      # TASK-11.1a WMMA/mma カーネル設計メモ（#60）
    ├── guardrail-change-policy.md    # TASK-6.2 判定器変更時フローの明文化（#149）
    ├── guardrail-self-repair-cli.md  # guardrail／self-repair CLI コマンド仕様（#183）
    ├── kernel-fusion.md     # TASK-12.2b カーネル融合の適用範囲・限界（複合WLで融合を性能目標の前提にしない。#168）
    ├── license-matrix.md    # 許容依存 8 区分のライセンス可否表（TASK-1.3）
    ├── performance-targets.md # REQ-8 段階的下限の全バックエンド横断一覧（TASK-8.4・#159）
    ├── public-api-design.md            # compat API 層の公開 API 設計（REQ-9）
    ├── self-repair-revalidation-plan.md # TASK-3.3a 自己修復ループ再実証の実証計画・題材選定（#140）
    └── spec/                # 正本 submodule（rust-ai-library-spec。編集禁止）
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
| implement | core-builder | `tensor-core`・`autodiff`・workspace 骨格・compat API 層 | sonnet |
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
| `.claude/rules/ci.md` | **CI は self-hosted 必須**・timeout 必須・SHA 固定・fail-closed 集約 |
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

## hooks（settings.json）

- **SessionStart**: 日本語・委譲・完全自作コア・CI self-hosted・Conventional Commits のリマインダーを表示する
- **PostToolUse**（Edit|Write）: `.rs` 編集時に rustfmt を自動適用する（Cargo.toml の edition を検出。未追加時は 2021 フォールバック）
