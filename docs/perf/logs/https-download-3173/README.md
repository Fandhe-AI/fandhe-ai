# #3173 HTTPS 取得の実通信とビルドの実機確認（GB10・Mac mini）

状態: **未実測**（事前登録と実行スクリプトのみ。結果は実測後に本ファイルと `docs/model-download-design.md` §22 へ記録する）

## 範囲

- 対象: `ModelRegistry::download`／`download_with` の HTTPS 取得（`ureq =3.4.2`＋`native-tls`＋`vendored`、`NativeTls`＋`PlatformVerifier`。`docs/model-download-design.md` §17〜§19）の実通信・ビルド・リンク・実行
- 変えないもの: 依存ピン（`Cargo.toml`・`Cargo.lock`・`deny.toml`）、本番コード（`crates/facade/src/`）、tolerance・判定規則
- 規則（事前登録）: [`RULE.txt`](./RULE.txt)。宛先は RULE.txt から実行スクリプトが読む

## 手順

1. 実行コミット（本ディレクトリの `RULE.txt` とスクリプトを含むコミット）を `git archive` で展開し、`.rev-stamp`（`git rev-parse HEAD`）を置く。
2. 実機へ rsync（`docs/real-hardware-verification-env.md` §3 の除外フィルタつき）で `~/work/rust-ai-library-run-3173/` へ送る。
3. ノード上で `bash docs/perf/logs/https-download-3173/run_https_download.sh <tree> <out>` を実行する（GB10 は専用 `CARGO_TARGET_DIR` で cold build）。
4. `<out>` を回収し、**無編集で** `gb10/`・`mac-mini/` に置く。ツリー指紋は `run_https_download.sh --fingerprint <dir>` で手元の `git archive` と突き合わせる。
5. マスクはユーザー名・ホスト名・ホームのパスのみ（スクリプトが fail-closed。`--self-test` で自己検査）。

## 結果

| 実行 | GB10 | Mac mini | 備考 |
|---|---|---|---|
| ビルド | 未実測 | 未実測 | |
| R1 取得・pin 検証・再利用 | 未実測 | 未実測 | |
| R2 pin 不一致の拒否 | 未実測 | 未実測 | |
| R3 不正 TLS 証明書 4 種の拒否 | 未実測 | 未実測 | |
| R4 接続タイムアウト | 未実測 | 未実測 | 経過時間は観測値 |
