# fandhe-ai-hf-hub

Hugging Face Hub 連携の補助クレート（`publish = false`・1 段目）。設計は
`docs/hf-hub-integration-design.md` を正とする。facade の公開 API には含まれない。

## テスト構成

| 種別 | 場所 | 外部通信 | CI |
|------|------|----------|----|
| モックテスト（revision 解決・一覧・ステータス写像・3xx 非追従・不正応答・入力検証・キャッシュキー） | `src/api.rs`・`src/lib.rs` の `#[cfg(test)]`（転送層は `JsonTransport` trait 注入） | なし | 実行する |
| 取得経路（pin 一致／不一致・200／404・リダイレクト）のモック | facade 側 `crates/facade/src/model/download.rs`・`model.rs` の既存テスト | なし | 実行する |
| 実通信テスト | `src/live_tests.rs`（全件 `#[ignore]`） | あり（huggingface.co） | 実行しない |

## 実通信テストの実行手順

公開 repo のみを対象とする。直下に `model.safetensors` を持つ公開 repo・その commit sha・
ファイルの sha256（別途取得して `sha256sum` で求める）を用意する。

```sh
FANDHE_AI_HF_TEST_REPO=<namespace/name> \
FANDHE_AI_HF_TEST_COMMIT=<40 桁小文字 16 進> \
FANDHE_AI_HF_TEST_SHA256=<64 桁 16 進> \
cargo test -p fandhe-ai-hf-hub -- --ignored --nocapture live_
```

- 環境変数が未設定のまま `--ignored` で実行すると panic する（黙って成功させない）。
- `live_nonexistent_repo_*` は環境変数を使わない（ネットワークのみ）。

## トークンについて

本クレートは認証トークンを読まない・送らない（private／gated repo は非対応で 401／403 は
fail-closed）。`HF_TOKEN` 等を設定しても使われない。テスト・ログ・リポジトリ・workflow に
トークンを書かない。CI workflow に secrets を追加しない。
