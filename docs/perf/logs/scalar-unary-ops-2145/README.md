# floor・ceil・round・sign・reciprocal・rsqrt・erf・pow_scalar（#2145）CUDA／Metal 実機未実測の申し送り

`docs/autodiff-scalar-unary-ops-decision.md` §7・§8「実装記録」参照。
本実装エージェント実行環境は CUDA／Metal 実機に到達できないため、
`fandhe_ai_autodiff::scalar_unary_ops`（`floor`／`ceil`／`round`／
`sign`／`reciprocal`／`rsqrt`／`erf`／`pow_scalar` の 8 演算）の
`crates/facade/tests/scalar_unary_ops_backend_parity.rs` のうち CUDA
（`Device::Cuda(0)`）・Metal（`Device::Metal`。`cfg(target_os =
"macos")` 限定）を対象とする**計 4 テスト**（forward 全 8 種 bit 完全
一致の `cuda_forward_matches_cpu_reference`／
`metal_forward_matches_cpu_reference`、backward（`floor`／`ceil`／
`round`／`sign` は恒等的に `0`・`reciprocal`／`rsqrt`／`erf`／
`pow_scalar` は REQ-2 統一複合判定）の
`cuda_backward_matches_cpu_reference`／
`metal_backward_matches_cpu_reference`）は `#[ignore]` のまま未実測
である。

## 測定コマンド案

```sh
# CUDA（DGX Spark GB10 等の実機上で）
cargo test -p fandhe-ai --test scalar_unary_ops_backend_parity -- --ignored --nocapture cuda

# Metal（Apple Silicon 実機上で）
cargo test -p fandhe-ai --test scalar_unary_ops_backend_parity -- --ignored --nocapture metal
```

CPU（`CpuBackendOps`）版は同テストファイルの属性なしテスト
（`cpu_forward_matches_naive_reference`・`cpu_backward_matches_naive_reference`）
で既に検証済み（green。`cargo test -p fandhe-ai --test
scalar_unary_ops_backend_parity` で再現可能）。

## 期待結果

新 7 kind（`Floor`／`Ceil`／`Round`／`Sign`／`Reciprocal`／`Rsqrt`／
`Erf`）は GPU 専用カーネルを実装していない（`crates/backend-cuda/src/
kernels_scalar_op.rs`・`crates/backend-metal/src/scalar_op_source.rs`
がいずれも `unary_kernel_source` で明示 `None` を返す）ため、CUDA・
Metal とも既定の `Unsupported` からホスト参照実装
（`ScalarUnaryOp::apply`。forward 数式の単一情報源）へフォールバック
する。したがって forward は 3 バックエンド間で構造的に bit 完全一致
するはず（CPU 経路も同じホスト参照実装を辿るため）。

backward は次の 2 通り:

- `floor`／`ceil`／`round`／`sign`（区分定数）は `is_piecewise_constant()`
  分岐によりゼロテンソルを直接生成するため、CUDA／Metal でも同じ経路
  （host フォールバック）を辿り bit 完全一致するはず
- `reciprocal`／`rsqrt`／`erf`／`pow_scalar` は `Op::ScalarUnary` の
  一般 VJP（`unary_grad_factor` による係数を `vjp_elementwise_mul` へ
  渡す経路）を辿るため、REQ-2 の統一複合判定（相対誤差 1e-3 未満 または
  絶対誤差 1e-5 未満。`fandhe_ai_backend_cpu::parity::assert_parity`）
  で比較する

この前提が崩れる場合（新 kind に GPU 専用カーネルが追加された、host
フォールバック経路の丸めがバックエンド間で変わった等）は本 README の
「期待結果」を更新し、想定した契約を維持できない事実を型付き findings
として PR へ記録すること（tolerance の単独緩和は行わない。
`.claude/rules/coding-rust.md`）。
