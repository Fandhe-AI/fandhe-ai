# compat::Sequential の Pool3d 層経路 実機 parity 申し送り（イシュー #3029）

本イシューは `max_pool3d`／`avg_pool3d` の保存対応と、層経路の `#[ignore]` parity テストの追加まで。
実機（CUDA／Metal）の実測と結果記入は Phase 8 の #3171・#3172 が担う。CPU 側の正しさは
`crates/facade/tests/compat_sequential_pool3d.rs` で検証済み。tolerance・baseline は変更しない。

## 判定契約

- facade `Sequential`（`add_max_pool3d`→`add_avg_pool3d`）の forward 出力と、`sum` 損失に対する入力勾配
  （pool 層は無パラメータ）を CPU と REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満。
  `assert_parity`）で比較する。入力は全要素が異なる ramp（max の索引タイブレークに依存させない）。
- 期待結果の注記: pool3d の GPU カーネルは存在せず、既定 `Unsupported` からホストフォールバックする。
  したがって本テストが確認するのは GPU カーネルの parity ではなくホストフォールバック経路の整合
  （`pool3d-ops-2643/README.md` と同じ位置づけ）。

## 実行コマンド

```
# CUDA（DGX Spark GB10）
cargo test -p fandhe-ai --test compat_sequential_pool3d_backend_parity -- --ignored cuda_
# Metal（M4 Max）
cargo test -p fandhe-ai --test compat_sequential_pool3d_backend_parity -- --ignored metal_
```

## 結果記入欄（未実測）

| 環境 | テスト | 結果 | 日付 |
|---|---|---|---|
| GB10 | cuda_pool3d_forward_backward_matches_cpu | 未実測（#3171） | |
| M4 Max | metal_pool3d_forward_backward_matches_cpu | 未実測（#3172） | |
