# jacobian・hessian（#2670）CUDA／Metal 実機未実測の申し送り

`docs/autodiff-jacobian-hessian-gradcheck-decision.md` 「実装記録（#2670）」参照。本実装エージェント実行環境は
CUDA／Metal 実機に到達できないため、`fandhe_ai_autodiff::jacobian_ops`（`jacobian`／`hessian`）の
`crates/facade/tests/jacobian_hessian_backend_parity.rs` のうち CUDA（`CudaBackendOps`）・Metal
（`MetalBackendOps`。`cfg(target_os = "macos")` 限定）を対象とする 2 テスト
（`cuda_jacobian_hessian_match_cpu_reference`・`metal_jacobian_hessian_match_cpu_reference`）は
`#[ignore]` のまま未実測である。

## 測定コマンド案

```sh
# CUDA（DGX Spark GB10 等の実機上で）
cargo test -p fandhe-ai --test jacobian_hessian_backend_parity -- --ignored --nocapture cuda

# Metal（Apple Silicon 実機上で）
cargo test -p fandhe-ai --test jacobian_hessian_backend_parity -- --ignored --nocapture metal
```

CPU（`CpuBackendOps`）版は同テストファイルの属性なしテスト（`cpu_matches_naive_reference`・
`cpu_matches_hand_computed_values`）で既に検証済み（green）。

## 期待結果

- 新規カーネルは存在しない。`jacobian`／`hessian` は既存 Op（matmul・tanh・sigmoid・sum 等）の VJP と子テープ上の
  再生を繰り返し呼ぶだけの合成であり、確認対象は「既存カーネルの新しい呼び出し形（backward の繰り返し・
  子テープ上の VJP）が CPU tape と同じ結果になること」である。
- 判定は REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満。
  `fandhe_ai_backend_cpu::parity::assert_parity`）。形状は小さく、Metal split-K が発動する形状は使っていない
  （split-K の baseline 方式は適用外）。
- この前提が崩れる場合は本 README の「期待結果」を更新し、REQ-2 判定を外れた事実を PR へ記録すること
  （tolerance の単独緩和は行わない。`.claude/rules/coding-rust.md`）。

## 実測記入欄

| 日付 | 実機 | コマンド | 結果 | 備考 |
|---|---|---|---|---|
| 未実測 | DGX Spark GB10 | 上記 CUDA | 未実測 | |
| 未実測 | Apple Silicon（M4 Max） | 上記 Metal | 未実測 | |
