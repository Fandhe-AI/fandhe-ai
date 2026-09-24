# var-f64-gemm-reduction（#2196）CUDA／Metal 実機未実測の申し送り

`docs/autodiff-var-dtype-multiplexing-design.md`「実装記録（#2196）」参照。
本実装エージェント実行環境は CUDA／Metal 実機に到達できないため、
`VarF64::matmul`／`sum`／`mean`／`max`（`crates/autodiff/src/
f64_autograd.rs`）の `crates/facade/tests/dtype_f64_integration.rs`
のうち CUDA（`fandhe_ai::tape_for(Device::Cuda(0))`）・Metal
（`MetalBackendOps::new()`。`cfg(target_os = "macos")` 限定）を対象と
する 2 テストは `#[ignore]` のまま未実測である。

## 測定コマンド案

```sh
# CUDA（DGX Spark GB10 等の実機上で）
cargo test -p fandhe-ai --test dtype_f64_integration -- --ignored --nocapture cuda

# Metal（Apple Silicon 実機上で）
cargo test -p fandhe-ai --test dtype_f64_integration -- --ignored --nocapture metal
```

CPU（`CpuBackendOps`）版は上記テストファイルの属性なしテスト
（`facade_typed_ops_f64_accessor_is_some_on_cpu_and_computes_gemm_sum_max`・
`facade_typed_ops_f64_matches_tape_f64_native_forward`・
`cpu_native_matmul_sum_mean_max_backward_matches_host_reference`）で
既に検証済み（green）。複合 backward と数値微分の突合は
`crates/autodiff/tests/var_f64_integration.rs::
composite_matmul_sum_mean_max_backward_matches_central_difference`
（CPU・`NaiveOps` ホスト経路。CI 実行）で確認済み。

## 期待結果

- **CUDA（`CudaBackendOps`。`typed_ops_f64()` が `Some`。#2060）**:
  `gemm`／`sum`（軸指定）／`max` は本 doc の「bit 一致の境界」節が示す
  結合順序と同じ構造であれば CPU 参照実装（`fandhe_ai_backend_cpu::
  matmul_reference_fma_f64`／`assert_parity_f64`）と一致する見込みだが、
  GPU の縮約カーネルは単一の連続 K ループとは異なる結合順序（分割・
  木縮約等）を取りうるため、厳密ゼロ fail ではなく REQ-2 統一複合判定
  （`assert_parity_f64`。相対誤差 1e-3 未満または絶対誤差 1e-5 未満。
  定数は無変更）で検証すること。全軸 `sum`（`CHUNK` 単位でない可能性が
  高い）は特に構造差が出やすいため、`fail_count` が 0 でなくても
  ただちに regression とはせず統一複合判定の可否で判断する
  （`.claude/rules/coding-rust.md`「結合順序が単一の連続 K ループと
  異なるカーネルの parity テスト判定方式」）
- **Metal（`MetalBackendOps`。`typed_ops_f64()` が常に `None`）**:
  `matmul`／`sum`／`max` はすべてホスト参照実装（`host_gemm_f64`／
  `host_sum_f64`／`host_max_f64`）経由になるため、`RawTape::new()`
  （`NaiveOps`）と **bit 完全一致**するはず（CPU ネイティブとの比較で
  はなく、ホスト経路同士の比較である点に注意）

この前提が崩れる場合（実測で REQ-2 の複合判定を外れる、または Metal の
bit 一致が崩れる等）は、tolerance を単独緩和せず、型付き findings として
PR へ記録すること。
