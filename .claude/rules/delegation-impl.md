# 委譲マッピング（作成・編集フェーズ）

## 原則

- 実装・テスト・レビュー作業は対応する subagent へ委譲し、main は計画・統合・報告に専念する
- 1 タスクの委譲単位は「1 REQ または 1 TASK の結束した変更」とし、細切れにしすぎない

## パスベース・トピック別委譲マッピング

| 対象 | 委譲先 | 補足 |
|------|--------|------|
| `Cargo.toml`・workspace 骨格・`crates/tensor-core`・`crates/autodiff`・`crates/facade`（composition root・compat API 層。TASK-9.4・#411 で `autodiff::compat` から移設） | core-builder | TASK-1.x・REQ-1・REQ-9〜10 系 |
| `crates/backend-cpu`・`crates/backend-cuda`・`crates/backend-metal`・数値一致回帰テスト | backend-builder | TASK-2.x・REQ-2・REQ-11〜13 系 |
| `crates/onnx-interop`（safetensors / prost 自前取り込み） | interop-builder | REQ-7 系 |
| `crates/guardrail`・`crates/self-repair`・`crates/bench-harness` | runtime-builder | TASK-3.x〜6.x・REQ-3〜6・REQ-8 系 |
| テスト実行・受け入れ基準対応テスト追加 | test-runner | 実機依存は `#[ignore]` 分離 |
| ベンチ計測・性能回帰検出 | bench-runner | 5 回計測中央値・読み取り専用 |
| コードレビュー | reviewer | 読み取り専用 |
| セキュリティ・ライセンス監査 | security-auditor | 読み取り専用 |
| fmt/clippy/frontmatter lint | linter | haiku |
| `CLAUDE.md`・`README.md`・license-matrix・チェックリスト類 | docs-writer | haiku |

## 実装フローの標準

1. 実装（core-builder / backend-builder / interop-builder / runtime-builder）
2. lint（linter）→ テスト（test-runner）・必要に応じベンチ（bench-runner）
3. レビュー（reviewer。ガードレール・依存追加を含む変更は security-auditor も並列）
4. main が結果を統合しユーザーへ報告

## 禁止事項

- 実装 Agent に `docs/spec/`（正本サブモジュール）を書き換えさせない（仕様変更は spec リポジトリ側で行う）
- 複数 Agent に同一ファイルを並行編集させない（コンフリクト防止。必要なら worktree 分離）
- 実装 Agent に依存クレートを自己判断で追加させない（deps-policy.md。ユーザー承認必須）
- 実装 Agent にガードレール閾値・テスト許容誤差を緩和させない（ユーザー承認必須）
