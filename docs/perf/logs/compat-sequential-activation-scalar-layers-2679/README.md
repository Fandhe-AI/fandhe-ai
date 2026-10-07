# compat::Sequential の活性化 9 層の実機 parity 申し送り（イシュー #2679）

対象は `add_selu`・`add_celu`・`add_softsign`・`add_hardsigmoid`・`add_log_sigmoid`・`add_softmin`・
`add_tanhshrink`・`add_threshold`・`add_rrelu`。本エージェントの実行環境には CUDA／Metal 実機への到達手段が
ないため、実機テストは未実測のまま GB10・M4 Max セッションへ申し送る。CPU 側の正しさは Linux 実行可能な
`crates/facade/tests/compat_sequential_activation_scalar_layers.rs` で検証済み。tolerance・baseline は変更しない。

## 判定契約

facade `Sequential` 経由の `forward` を同じ入力の CPU `predict` と比較し、さらに `Linear → 層 → Linear` の
学習 1 ステップの全パラメータ勾配を CPU と比較する。

- REQ-2 統一複合判定（`fandhe_ai_backend_cpu::parity::assert_parity`。相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）を
  forward・勾配の両方に使う。新しい tolerance 定数は作らない。
- RReLU は推論モード（`eval()`。固定傾き `(lower + upper) / 2`）で比較する。学習時の傾きは乱数のため比較しない。
- 9 層は専用の GPU カーネルを持たない（既存演算の合成、または `scalar_unary`／ホスト参照へのフォールバック）。
  この比較は「フォールバックを含む経路が CPU と同じ結果になること」の確認であり、GPU カーネル自体の parity ではない。
- 常駐経路（`predict_resident`）上での無状態層の GPU 通過は CPU でのみ検証済みで、GPU では未確認。
- Softmin 系 4 層（`add_softmin`・`add_tanhshrink`・`add_threshold`・`add_rrelu`）は `forward_host` を持たず、
  `predict` は tape 経路へフォールバックする（`docs/autodiff-softmin-threshold-ops-decision.md` §7）。

## 実行コマンド

```
# CUDA（DGX Spark GB10）
cargo test -p fandhe-ai --test compat_sequential_activation_scalar_layers_backend_parity -- --ignored cuda_
# Metal（M4 Max）
cargo test -p fandhe-ai --test compat_sequential_activation_scalar_layers_backend_parity -- --ignored metal_
```

## 結果記入欄（未実測）

| 環境 | テスト | 結果 | 日付 |
|---|---|---|---|
| GB10 | cuda_activation_scalar_layers_parity | 未実測 | |
| M4 Max | metal_activation_scalar_layers_parity | 未実測 | |
