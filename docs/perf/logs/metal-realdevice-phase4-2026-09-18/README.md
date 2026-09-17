# Metal バックエンド Phase 4 全 `#[ignore]` テスト実機実測（2026-09-18・Apple M4 Max）

本ディレクトリは `crates/backend-metal` 配下のすべての `#[ignore]` テストを
Apple M4 Max で実行した結果を記録する。

## 実行コマンド

```sh
cargo test -p fandhe-ai-backend-metal --release --all-features --no-fail-fast -- --ignored --nocapture
```

## 実測結果

- **総スコア**: 430 pass / 0 FAIL（非後退・新規 FAIL なし）
- **実行日**: 2026-09-18
- **環境**: Apple M4 Max、共有負荷下（record_only）
- **base SHA**: `a1c50f61`（`perf(backend-metal): 協調ロードの threadgroup メモリ XOR swizzle 軸（index 17）を opt-in で実装する (#2008)` のコミット）
- **負荷**: load1 開始前 18.98、終了後 10.58（共有負荷下・他セッション並走）

## by-name 差分（基準との比較）

基準: #1894 の実測結果（`docs/perf/logs/metal-reduce-sum-wiring-1896/` の 411 pass / 1 FAIL）

- 後退テスト: 0 件
- 基準に含まれて今回消失したテスト: 0 件
- 新規追加テスト（本番結線分含む）: 19 件
  - `backend_ops_argmax_argmin_match_cpu_exact`（#1951）
  - `backend_ops_log_softmax_backward_matches_direct_api_call`（#1952）
  - `backend_ops_norm_backward_reaches_metal_override`（#1953）
  - `gemm_smem_swizzle_diag_tests::xor_swizzle_kernel_gpu_ab_production_sizes`（#2008）
  - `gemm::tests::smem_swizzle_bit_match_all_candidates`（#2008）
  - `gemm::tests::smem_swizzle_bit_match_boundary_shape`（#2008）
  - `gemm::tests::smem_swizzle_bit_match_dispatch_auto`（#2008）
  - `gemm::tests::smem_swizzle_default_matches_production_constants`（#2008）
  - `gemm::tests::smem_swizzle_f16_path_is_noop`（#2008）
  - `gemm::tests::smem_swizzle_transposed_bit_match`（#2008）
  - `layer_norm_backward_matches_cpu_reference_across_shapes`（#1953）
  - `rmsnorm_backward_matches_cpu_reference_across_shapes`（#1953）
  - `metal_arg_all_matches_cpu_exact`（#1951）
  - `metal_arg_all_tie_and_nan_match_cpu`（#1951）
  - `metal_arg_axis_matches_cpu_exact`（#1951）
  - `metal_log_softmax_backward_matches_cpu_bit_exact_when_y_is_neg_infinity`（#1952）
  - `metal_log_softmax_backward_matches_cpu_bit_exact_when_y_is_zero`（#1952）
  - `metal_log_softmax_backward_matches_cpu_composite_judgment`（#1952）
  - `command_batching::pool_reuse_zero_fill_does_not_synchronize_open_batch`（基準時は並列干渉で FAIL だったが、今回並列実行のまま pass）

## 判定

**非後退・新規 FAIL なし。合格。**
