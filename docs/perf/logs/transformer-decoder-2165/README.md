# イシュー #2165 実測記録先（TransformerDecoderLayer・Transformer 実機 parity）

## 位置づけ（本 PR 時点では未実測）

本 PR の実行環境（Linux コンテナ／worktree）には DGX Spark GB10・
Apple Silicon（Metal）実機への到達手段がないため、**実測値は一切
含まれていない**。本ディレクトリは実行コマンド・保存すべきログ
一覧のみを提供し、実測は実機を持つセッションへ申し送る
（`docs/perf/logs/mha-options-2163/README.md` と同型の運用）。

## 目的

イシュー #2165「`TransformerDecoderLayer`・`Transformer`（#2068 の
対）」の実機バックエンド（CUDA・Metal）間 forward parity の検証記録
先。`TransformerDecoderLayer` 単体（`S != L` の memory・causal あり）
と小さい `Transformer`（encoder 2 層・decoder 2 層）の forward を
対象とする。CPU 側の bit 一致・parity（CPU vs NaiveOps）は本 PR の
CI 内で検証済み（`crates/autodiff/src/nn/transformer_decoder_layer.rs`・
`crates/autodiff/src/nn/transformer.rs` の単体テスト・
`crates/facade/tests/transformer_decoder_backend_parity.rs` の属性
なしテスト）。

## 実行コマンド

```bash
# facade の TransformerDecoderLayer／Transformer forward REQ-2 複合判定
# （#[ignore] は CUDA 2 テスト・Metal 2 テスト）
cargo test -p fandhe-ai --release --test transformer_decoder_backend_parity -- --ignored --nocapture
```

`make test-ignored-cuda`（`--all-features -- --ignored`）・Metal 実機
（`cfg(target_os = "macos")`）での相当コマンドにも上記が自動的に
含まれる。

## 保存すべきログ

- `transformer_decoder_backend_parity-ignored.log`: 上記コマンドの
  生出力（`fold_bits=` 行を含む。run-to-run 決定性チェック用）
- `env_info.txt`: 実行環境記入欄（内部ホスト名は書かない。GPU 型番・
  driver／CUDA バージョン・`nvidia-smi` 出力の要約程度に留める）

## 事前登録判定規則

- **TransformerDecoderLayer／Transformer forward REQ-2 parity**: CPU
  参照実装（`raw_tape_for(Device::Cpu)`）との複合判定（相対誤差 1e-3
  未満 または絶対誤差 1e-5 未満）が全 fail 0 件であること
- **run-to-run 決定性**: 上記コマンドを 2 回起動し `fold_bits=` 行が
  2 起動間で bit 同一であること
- 判定規則・tolerance は本記録の追記時点で変更しない（変更する場合は
  ユーザー承認が必要。`.claude/rules/coding-rust.md`）

## 記入欄（実機実測後に追記）

未実測（本 PR 時点）。実測後は `docs/autodiff-transformer-decoder-decision.md`
へ結果を追記する。
