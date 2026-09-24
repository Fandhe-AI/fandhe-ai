# module-freeze（#2137）CUDA／Metal 実機未実測の申し送り

`docs/autodiff-nograd-leaf-dinput-skip-decision.md`「実装記録（#2137）」参照。
本実装エージェント実行環境は CUDA／Metal 実機に到達できないため、層別
`requires_grad` 凍結（`fandhe_ai_autodiff::nn::Module::freeze`／
`set_requires_grad`）の `crates/facade/tests/nn_module_freeze_backend_parity.rs`
のうち CUDA（`CudaBackendOps::new(0)`）・Metal（`MetalBackendOps::new()`。
`cfg(target_os = "macos")` 限定）を対象とする 2 テストは `#[ignore]` のまま
未実測である。

## 測定コマンド案

```sh
# CUDA（DGX Spark GB10 等の実機上で）
cargo test -p fandhe-ai --test nn_module_freeze_backend_parity -- --ignored --nocapture cuda

# Metal（Apple Silicon 実機上で）
cargo test -p fandhe-ai --test nn_module_freeze_backend_parity -- --ignored --nocapture metal
```

CPU（`CpuBackendOps`）版は上記テストファイルの属性なしテスト
（`cpu_freeze_linear_output_and_input_grad_match_unfrozen`）で既に検証済み
（green）。

## 期待結果

`requires_grad` は算術を一切伴わないテープ側メタデータ
（`crates/autodiff/src/tape.rs::TapeNode::requires_grad`）であり、
`backward.rs::accumulate` 呼び出し前のゲート判定以外の経路（融合プラン選択・
forward 演算列）には一切影響しない。したがって CUDA／Metal 上でも、同一
バックエンド内で「凍結あり」と「凍結なし」の `Linear` の出力・x 勾配が
bit 完全一致し、凍結側の weight／bias 勾配は `Err(GradientTrackingDisabled)`
を返すはず（CPU 版と同じ構造の hard assert）。CPU で実証済みの契約が
device 差分なくそのまま成立する見込みであり、この前提が崩れる場合
（融合プラン選択が `requires_grad` を参照している等）は本 README の「期待
結果」を更新し、bit 一致の hard assert を維持できない事実を型付き findings
として PR へ記録すること（tolerance の単独緩和は行わない。
`.claude/rules/coding-rust.md`）。
