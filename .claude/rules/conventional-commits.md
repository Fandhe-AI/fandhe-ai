# Conventional Commits 規約

## 形式

```
<type>(<scope>): <日本語の説明>

<本文（任意・日本語）>
```

## type

| type | 用途 |
|------|------|
| feat | 機能追加 |
| fix | バグ修正 |
| perf | 性能改善（ベンチ根拠を本文に記載） |
| refactor | 挙動を変えない整理 |
| test | テスト・ベンチの追加・修正 |
| docs | ドキュメントのみの変更 |
| build | ビルド・依存関係（Cargo.toml・Dockerfile・Makefile） |
| ci | CI ワークフロー・lefthook |
| chore | 上記以外の雑務（.claude/・スキル・設定） |

## scope

クレート名・領域名を使う（例: `tensor-core`・`autodiff`・`backend`・`interop`・`guardrail`・`claude`・`spec`）。単一クレート確定前は領域名でよい。

## ルール

- 破壊的変更は `!` を付け、本文に `BREAKING CHANGE:` を記載する（公開 API 非破壊はガードレール条件でもある）
- 1 コミット 1 関心事。仕様書サブモジュール更新（`docs/spec`）は単独コミットとする
- **`--no-verify` は禁止**。lefthook の pre-commit / commit-msg を必ず通す
- コミット作成は create-commit スキルのフローに従う（シークレット検査を含む）
