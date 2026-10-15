# compat::Sequential の LocalResponseNorm 層経路 実機 parity 申し送り（イシュー #3045）

本イシューは `local_response_norm` の保存対応と、層経路の `#[ignore]` parity テストの追加まで。
実機（CUDA／Metal）の実測と結果記入は Phase 8 の #3171・#3172 が担う。CPU 側の正しさは
`crates/facade/tests/compat_sequential_lrn.rs` で検証済み。tolerance・baseline は変更しない
（fail が出ても tolerance を緩めない）。演算単体の parity は `lrn_weight_reparam_backend_parity.rs` が担い、
本テストは層（`Sequential`）経由。

## 判定契約

- facade `Sequential`（`add_local_response_norm(5, 1e-4, 0.75, 2.0)`。入力 NCHW `[2, 7, 3, 2]` の ramp）の
  forward 出力と、`sum` 損失に対する入力勾配を CPU と REQ-2 統一複合判定
  （相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満。`assert_parity`）で比較する。
- 期待結果の注記: CUDA／Metal は LRN の 3 フックを override せず `Unsupported` → ホストフォールバック経路を
  通るため、新規カーネルの parity ではなく層経由でもフォールバックが CPU と一致する確認になる。

## 実行コマンド

```
# CUDA（DGX Spark GB10）
cargo test -p fandhe-ai --test compat_sequential_lrn_backend_parity -- --ignored cuda_
# Metal（M4 Max）
cargo test -p fandhe-ai --test compat_sequential_lrn_backend_parity -- --ignored metal_
```

## 結果記入欄（未実測）

| 環境 | テスト | 結果 | 日付 |
|---|---|---|---|
| GB10 | cuda_local_response_norm_forward_backward_matches_cpu | 未実測（#3171） | |
| M4 Max | metal_local_response_norm_forward_backward_matches_cpu | 未実測（#3172） | |
