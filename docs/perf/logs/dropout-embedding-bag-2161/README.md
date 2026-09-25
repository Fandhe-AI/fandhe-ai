# イシュー #2161 実測記録先（Dropout2d・AlphaDropout・EmbeddingBag 実機 parity）

## 位置づけ（本 PR 時点では未実測）

本 PR の実行環境（Linux コンテナ／worktree）には DGX Spark GB10・
Apple Silicon（Metal）実機への到達手段がないため、**実測値は一切
含まれていない**。本ディレクトリは実行コマンド・保存すべきログ
一覧のみを提供し、実測は実機を持つセッションへ申し送る
（`docs/perf/logs/spatial-layers-2159/README.md` と同型の運用）。

## 目的

イシュー #2161「`nn::Dropout2d`・`nn::AlphaDropout`・
`nn::EmbeddingBag`」の実機バックエンド（CUDA・Metal）間の実機正しさ
検証記録先。以下の `#[ignore]` テストを CUDA（DGX Spark GB10）・
Metal（Apple Silicon）実機で実行して green を確認することが目的。
CPU 側の bit 一致・数値一致・解析的 VJP 突合は本 PR の CI 内で検証済
み（`crates/autodiff/tests/nn_dropout_variants.rs`・
`crates/autodiff/tests/nn_embedding_bag.rs`・
`crates/facade/tests/dropout_variants_backend_parity.rs`・
`crates/facade/tests/embedding_bag_backend_parity.rs` の属性なし
テスト）。

## 実行コマンド

```bash
# Dropout2d／AlphaDropout forward の CUDA／Metal 実機 parity
# （#[ignore] は CUDA 2 テスト・Metal 2 テスト〈cfg(target_os = "macos")〉）
cargo test -p fandhe-ai --release --test dropout_variants_backend_parity -- --ignored --nocapture

# EmbeddingBag(sum) forward の CUDA／Metal 実機 parity
# （#[ignore] は CUDA 1 テスト・Metal 1 テスト）
cargo test -p fandhe-ai --release --test embedding_bag_backend_parity -- --ignored --nocapture
```

`make test-ignored-cuda`（`--all-features -- --ignored`）・Metal 実機
（`cfg(target_os = "macos")`）での相当コマンドにも上記が自動的に
含まれる。

## 保存すべきログ

- `dropout_variants_backend_parity-ignored.log`／
  `embedding_bag_backend_parity-ignored.log`: 上記コマンドの生出力
  （`fold_bits=` 行を含む場合はそれも保存。run-to-run 決定性チェック
  用）
- `env_info.txt`: 実行環境記入欄（内部ホスト名は書かない。GPU 型番・
  driver／CUDA バージョン・`nvidia-smi` 出力の要約程度に留める）

## 事前登録判定規則

- **Dropout2d／AlphaDropout forward**: `BackendOps::mul`（Dropout2d）・
  `BackendOps::mul → BackendOps::add`（AlphaDropout）のみで構成される
  ため、CPU 参照実装（`fandhe_ai_autodiff::Tape::new_with_ops(Box::new(
  CpuBackendOps::new()))`）との比較は**厳密ゼロ fail 判定（bit 完全
  一致）**であること（`assert_parity` も併記するが、主たる判定は bit
  比較）
- **EmbeddingBag(sum) forward**: `BackendOps::gather`→`BackendOps::sum`
  で構成されるため、CPU 参照実装との複合判定（相対誤差 1e-3 未満 または
  絶対誤差 1e-5 未満）が全 fail 0 件であること（Mean／Max も同型だが
  代表として Sum のみを実機テスト対象とする。Max は forward が bit
  完全一致するため理論上は実機でも fail が生じない想定だが、実測で
  裏付けるまでは同型の複合判定を適用する）
- **run-to-run 決定性**: 各コマンドを 2 回起動し出力が 2 起動間で
  一致すること
- 判定規則・tolerance は本記録の追記時点で変更しない（変更する場合は
  ユーザー承認が必要。`.claude/rules/coding-rust.md`）

## 記入欄（実機実測後に追記）

未実測（本 PR 時点）。実測後は `docs/autodiff-dropout-embedding-bag-
decision.md` §7 節へ結果を追記する。
