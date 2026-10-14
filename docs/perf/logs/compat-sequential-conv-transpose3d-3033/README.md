# compat::Sequential の ConvTranspose3d 層経路 実機 parity 申し送り（イシュー #3033）

本イシューは `conv_transpose3d` の保存対応と、層経路の `#[ignore]` parity テストの追加まで。
実機（CUDA／Metal）の実測と結果記入は Phase 8 の #3171・#3172 が担う。CPU 側の正しさは
`crates/facade/tests/compat_sequential_conv_transpose3d.rs` で検証済み。tolerance・baseline は変更しない。
演算単体の parity は `conv_transpose3d_max_unpool_backend_parity.rs` が担い、本テストは層（`Sequential`）経由。

## 判定契約

- facade `Sequential`（`add_conv_transpose3d(4, 6, [3,3,3], [2,2,2], [1,1,1], [1,1,1], [1,1,1], 2, seed)`。
  入力 NCDHW `[2, 4, 3, 3, 3]` の ramp）の forward 出力と、`sum` 損失に対する全パラメータ勾配
  （weight → bias）を CPU と REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満。
  `assert_parity`）で比較する。
- 期待結果の注記: GPU カーネルの有無は演算単体の記録に従う。カーネルが無い場合は既定 `Unsupported` からの
  ホストフォールバック経路の整合の確認になる。

## 実行コマンド

```
# CUDA（DGX Spark GB10）
cargo test -p fandhe-ai --test compat_sequential_conv_transpose3d_backend_parity -- --ignored cuda_
# Metal（M4 Max）
cargo test -p fandhe-ai --test compat_sequential_conv_transpose3d_backend_parity -- --ignored metal_
```

## 結果記入欄（未実測）

| 環境 | テスト | 結果 | 日付 |
|---|---|---|---|
| GB10 | cuda_conv_transpose3d_forward_backward_matches_cpu | 未実測（#3171） | |
| M4 Max | metal_conv_transpose3d_forward_backward_matches_cpu | 未実測（#3172） | |
