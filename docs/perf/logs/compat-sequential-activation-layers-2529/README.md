# compat::Sequential の Mish／Hardtanh／ReLU6／GLU／PReLU 実機 parity 申し送り（イシュー #2529）

本エージェントの実行環境には CUDA／Metal 実機への到達手段がないため、実機テストは未実測のまま
GB10・M4 Max セッションへ申し送る。CPU 側の正しさは Linux 実行可能な
`crates/facade/tests/compat_sequential_activation_layers.rs` で検証済み。tolerance・baseline は変更しない。

## 判定契約

`activation_ops_backend_parity.rs`（#2146）と同じ判定方式の割り当てで、facade `Sequential` 経由の `forward` を同じ入力の CPU `predict` と比較する。

- bit 完全一致: `add_hardtanh(-1.0, 2.0)`・`add_relu6()`・`add_prelu(4, 0.2)`（forward）
- REQ-2 統一複合判定（`fandhe_ai_backend_cpu::parity::assert_parity`。相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）:
  `add_mish()`・`add_glu(1)`（forward）、`Linear → PRelu → Linear` の学習 1 ステップの全パラメータ勾配
  （`weight` 勾配は縮約を経由するため）
- 新規 `Op`／`BackendOps`／カーネルはない。新しい tolerance 定数も作らない。
- 常駐経路上での無状態 4 層（Mish／Hardtanh／Relu6／Glu）の GPU 通過（`predict_resident`）は CPU でのみ検証済みで、GPU では未確認。

## 実行コマンド

```
# CUDA（DGX Spark GB10）
cargo test -p fandhe-ai --test compat_sequential_activation_layers_backend_parity -- --ignored cuda_
# Metal（M4 Max）
cargo test -p fandhe-ai --test compat_sequential_activation_layers_backend_parity -- --ignored metal_
```

## 結果記入欄（未実測）

| 環境 | テスト | 結果 | 日付 |
|---|---|---|---|
| GB10 | cuda_activation_layers_parity | 未実測 | |
| M4 Max | metal_activation_layers_parity | 未実測 | |
