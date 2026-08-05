# rust-ai-library

Rust 製 AI/ML ライブラリの実装リポジトリです。Burn 依存を排した完全自作コア（v2 方針）で実装します。

## 位置づけ

- **仕様・要件定義**: [rust-ai-library-spec](https://github.com/Fandhe-AI/rust-ai-library-spec)（`docs/spec` に submodule 参照）
- **旧実装（v1・Burn ベース）**: [rust-ai-library-v1](https://github.com/Fandhe-AI/rust-ai-library-v1)（アーカイブ済み。資産の引き継ぎ記録は [v1-assets-inventory.md](https://github.com/Fandhe-AI/rust-ai-library-spec/blob/main/v1-assets-inventory.md) を参照）
- **立ち上げ手順**: [v2-repo-migration.md](https://github.com/Fandhe-AI/rust-ai-library-spec/blob/main/v2-repo-migration.md)

## ステータス

M0（リポ基盤: workspace 骨格・依存禁止 CI 検査・ライセンス可否表）は未着手です。タスク定義は spec リポの [`05-tasks.md`](https://github.com/Fandhe-AI/rust-ai-library-spec/blob/main/05-tasks.md)（TASK-1.1〜1.3）、マイルストーンは [`06-roadmap.md`](https://github.com/Fandhe-AI/rust-ai-library-spec/blob/main/06-roadmap.md)（M0〜M5・全 51 タスク）を参照してください。

## 実装方針（要点）

- 想定クレート 9 個: `tensor-core`・`autodiff`・`backend-cpu`・`backend-cuda`・`backend-metal`・`onnx-interop`・`guardrail`・`self-repair`・`bench-harness`
- 許容依存 8 区分（`cudarc`／`objc2` 系／`safetensors`／`prost`／`serde` 系／`rayon`／`half`／`criterion`）を `=x.y.z` 完全固定で管理（workspace ルート `Cargo.toml` の `[workspace.dependencies]` に一元定義。TASK-1.1b）
- 依存禁止リスト（`burn` 系一式・`cubecl`・`candle`・`tch`・`ndarray`）を CI で機械検査（TASK-1.2）
- バックエンド切替は feature フラグなしの cfg ベース（`cudarc` 動的ロード・`objc2` 系は `cfg(target_os = "macos")` 分離。PoC-v2-5 実証構成）

### 依存追加・更新フロー

依存の追加・更新は必ずユーザー承認を経て行う（`.claude/rules/deps-policy.md`）。承認後の実施手順:

1. **判断理由の記録**: 許容依存 8 区分以外を新規追加する場合、判断軸 a〜e（数値意味論か境界層か／AI 保守ガードレール対象か／自作コスト対差別化価値／unsafe・FFI 面積／ライセンス適合。`docs/spec/01-brainstorm.md` の「v2 自作範囲の境界定義」節）に基づく判断理由を PR 本文・コミットメッセージに記録する
2. **バージョン固定**: `Cargo.toml` で `=x.y.z` 完全固定とし、`Cargo.lock` を同一コミットでコミットする
3. **`docs/license-matrix.md` の同時更新**: 新規・更新する依存のライセンスを確認し、`docs/license-matrix.md`（未作成の場合は同一 PR で作成）を更新する。MPL-2.0 等コピーレフトの推移的混入は推定で記述せず、有効化しうる feature 組合せごとの `cargo tree` 実測で個別に適合確認する
4. **CI 検査の確認**: `deps-forbidden` ジョブ（禁止リストの `Cargo.lock` 機械検査）・`deny` ジョブ（`deny.toml` 導入後。ライセンス監査）が green であることを確認する

## 開発環境構築

```bash
git clone git@github.com:Fandhe-AI/rust-ai-library.git
cd rust-ai-library
make setup   # サブモジュール取得 → rustup → lefthook（git hooks）を一括構築
```

主な make ターゲット（`make help` で一覧表示）:

| ターゲット | 内容 |
|-----------|------|
| `make setup` | 開発環境の一括構築（submodule・rustup・lefthook） |
| `make fmt` / `make fmt-check` | 整形 / 整形差分の検出 |
| `make lint` | `cargo clippy -D warnings` |
| `make test` | `cargo test`（実機依存の `#[ignore]` テストは除く） |
| `make test-ignored` | 実機（Metal / CUDA）専用の `#[ignore]` 分離テスト |
| `make deny` | `cargo deny check licenses sources`（依存ライセンス監査） |
| `make deps-forbidden` | 依存禁止リスト（burn 系等）の混入検査 |
| `make ci` | CI（`.github/workflows/ci.yml`）と同一チェックの一括実行 |

Cargo.toml 未追加（M0 の TASK-1.1 で workspace 作成予定）の間、cargo 系ターゲットは CI と同じ方針でスキップされます。

## Docker 開発環境

ホスト環境（macOS / Linux・aarch64 / amd64）に依存せずビルド・テスト・lint を実行できます。

```bash
make docker-build   # 開発コンテナイメージをビルド
make docker-shell   # 開発コンテナのシェルに入る
make docker-ci      # コンテナ内で make ci を実行（環境非依存の検証）
```

コンテナ内で使えるのは CPU（rayon）バックエンドのみです。Metal はホスト macOS 直接実行、CUDA は実機（DGX Spark GB10 等）で実行します（`cudarc` の動的ロード方式のため、CUDA バックエンドの「ビルド」は CUDA toolkit 無しのコンテナでも成立します）。

## CI

- CI はすべて **self-hosted runner** で実行します（`.claude/rules/ci.md`）
- `ci.yml`: fmt / clippy / test / deny / 依存禁止検査（TASK-1.2）＋集約ジョブ `ci-complete`（branch protection にはこれのみを指定）
- `update-external.yml`: `docs/spec` サブモジュールと `.claude/skills` の自動追従（毎日 09:00 JST。PR label: `dependencies`・`automated`）

## 開発体制（Claude Code）

`.claude/` に Claude Code の運用体系（Agents・Rules・Skills・hooks）を整備しています。概要は [CLAUDE.md](./CLAUDE.md) を参照してください。
