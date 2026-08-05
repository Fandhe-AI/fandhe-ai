# CI 規約

## runner

- **CI はすべて self-hosted runner で実行する**（`runs-on: self-hosted`）。GitHub ホステッドランナー（`ubuntu-latest` 等）は使用しない
- self-hosted runner はハングしたジョブに無期限占有されうるため、**全ジョブに `timeout-minutes` を必ず設定する**

## ワークフロー設計（Fandhe-AI/local-llm-server・fandhe-multi-platform と同一方針）

- `permissions` はワークフロー既定を `contents: read` の最小とし、必要なジョブのみ個別に昇格する
- サードパーティ actions は**コミット SHA に固定**する（タグ参照禁止。`actions/checkout@<sha>` 等）
- checkout 後に GITHUB_TOKEN を使わないジョブは `persist-credentials: false` を指定し、共有 runner の workspace に認証情報を残さない
- `concurrency` で同一 ref の重複実行を直列化・キャンセルする
- **グローバル状態を汚す処理を workflow に書かない**。ツール導入は「未導入の場合のみ導入する」冪等セルフヒール（Ensure rustup / Ensure component パターン）とする
- branch protection の required status check は集約ジョブ（`ci-complete`）のみを指定し、needs の result を明示検査して fail-closed で判定する
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
