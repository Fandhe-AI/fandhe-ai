# 共有 target で workspace メンバーの成果物が再利用される条件の再現（#4044）

開発機（Linux）で、実機ノードの共有 `CARGO_TARGET_DIR` で起きた「前の回の workspace メンバー成果物の再利用」を再現した記録。実機は使っていない。結果の解釈と対策は [`docs/real-hardware-shared-target-reuse-decision.md`](../../../real-hardware-shared-target-reuse-decision.md) §2〜§3。

## 実行条件

- 2026-11-05、開発機 Linux（x86_64。ホスト名・パスはマスク済み）、cargo 1.98.1、`--offline`
- コマンド: `bash repro.sh <repo の絶対パス> b1f3dd069 679e86372 <出力先>`（X = `b1f3dd069`、Y = `679e86372`。`crates/tensor-core/src/device.rs` に差分がある 2 コミット）
- 各シナリオは `cargo build -v -p fandhe-ai-tensor-core -p fandhe-ai-backend-cpu`（dev プロファイル）。`CARGO_LOG=cargo::core::compiler::fingerprint=info` の出力も `build_*.log` に入っている
- target と展開先は `$HOME/tmp-build-4044.*` の一時ディレクトリで、終了時に削除される

## ファイル

| ファイル | 内容 |
|---|---|
| `repro.sh` | 再現スクリプト（手動実行専用。S0〜S5） |
| `summary.tsv` | シナリオ別・パッケージ別の `Fresh`／`Compiling` の件数 |
| `rlibs.tsv` | tensor-core の dev rlib の mtime・SHA-256（先頭 16 桁）・ファイル名 |
| `build_*.log` | 各ビルドの出力（マスク済み） |
| `clean_s4.log`・`members.txt`・`after_clean_leftovers.log` | S4 の `cargo clean -p` の出力・対象メンバー・clean 後に残った tensor-core の成果物（空） |
| `SHA256SUMS` | 上記ログ類の SHA-256（`repro.sh` と本 README を除く） |

## 結果（要約）

`summary.tsv`・`rlibs.tsv` のとおり。**S1（同じ展開先）と S2（別の展開先）で、Y を送っても tensor-core は `Fresh` のまま、rlib の SHA-256 は X のままだった**。mtime を保たない転送（S3）・全メンバーの clean（S4）・新しい mtime（S5）では Y の内容に再コンパイルされた。表は決定記録 §2.2。
