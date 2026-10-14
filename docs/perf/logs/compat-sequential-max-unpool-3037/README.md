# FunctionalBuilder の MaxPool 索引・MaxUnpool 層経路 実機 parity 申し送り（イシュー #3037）

本イシューは MaxPool 索引・MaxUnpool ノードの保存対応（Functional manifest の op 追加）と、層（Functional
グラフ）経路の `#[ignore]` parity テストの追加まで。実機（CUDA／Metal）の実測と結果記入は Phase 8 の
#3171・#3172 が担う。CPU 側の正しさは `crates/facade/tests/compat_functional_max_unpool.rs`・
`compat_functional_max_unpool_model_io.rs` で検証済み。tolerance・baseline は変更しない。
演算単体の parity は `conv_transpose3d_max_unpool_backend_parity.rs` が担い、本テストは層（Functional グラフ）経由。

テストファイル名: Issue 本文は `compat_sequential_<名前>_backend_parity.rs` としているが、MaxUnpool は
`Sequential` に載せない確定形（決定記録 `docs/compat-sequential-3d-fold-lrn-layers-decision.md` §6 案 A・§12）の
ため実体は `crates/facade/tests/compat_functional_max_unpool_backend_parity.rs` とした。ログ置き場の名前だけ
Issue 指定どおり維持している。

## 判定契約

- SegNet 型 `FunctionalModel`（conv ブロック → `max_pool*_with_indices` → conv ブロック → `max_unpool*`）の
  2D（入力 NCHW `[2, 2, 6, 6]`）・3D（入力 NCDHW `[1, 2, 4, 4, 4]`）について、forward 出力と `sum` 損失に対する
  入力勾配を CPU と REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満。`assert_parity`）で比較する。
- パラメータ勾配は `FunctionalModel::bind` が非公開で公開面から取れないため比較対象に含めない。
- 期待結果の注記: GPU カーネルの有無は演算単体の記録に従う。カーネルが無い場合は既定 `Unsupported` からの
  ホストフォールバック経路の整合の確認になる。fail が出ても tolerance は緩めない。

## 実行コマンド

```
# CUDA（DGX Spark GB10）
cargo test -p fandhe-ai --test compat_functional_max_unpool_backend_parity -- --ignored cuda_
# Metal（M4 Max）
cargo test -p fandhe-ai --test compat_functional_max_unpool_backend_parity -- --ignored metal_
```

## 結果記入欄（未実測）

| 環境 | テスト | 結果 | 日付 |
|---|---|---|---|
| GB10 | cuda_functional_max_unpool_2d_forward_backward_matches_cpu | 未実測（#3171） | |
| GB10 | cuda_functional_max_unpool_3d_forward_backward_matches_cpu | 未実測（#3171） | |
| M4 Max | metal_functional_max_unpool_2d_forward_backward_matches_cpu | 未実測（#3172） | |
| M4 Max | metal_functional_max_unpool_3d_forward_backward_matches_cpu | 未実測（#3172） | |
