# pos_weight 付き BCEWithLogits・HingeEmbedding・SoftMargin・GaussianNLL（#2652）CUDA／Metal 実機未実測の申し送り

`docs/autodiff-elementwise-loss-ops-decision.md` §10 参照。本実装エージェント実行環境は CUDA／Metal 実機に
到達できないため、`crates/facade/tests/elementwise_loss_ops_backend_parity.rs` のうち CUDA（`Device::Cuda(0)`）・
Metal（`Device::Metal`。`cfg(target_os = "macos")` 限定）を対象とする計 8 テストは `#[ignore]` のまま未実測である。

- CUDA: `cuda_bit_exact_forward_matches_cpu_reference`・`cuda_bit_exact_backward_matches_cpu_reference`・
  `cuda_req2_forward_matches_cpu_reference_within_tolerance`・`cuda_req2_backward_matches_cpu_reference_within_tolerance`
- Metal: 上記と対称の `metal_*` 4 件

## 測定コマンド案

```sh
# CUDA（DGX Spark GB10 等の実機上で）
cargo test -p fandhe-ai --test elementwise_loss_ops_backend_parity -- --ignored --nocapture cuda_

# Metal（Apple Silicon 実機上で）
cargo test -p fandhe-ai --test elementwise_loss_ops_backend_parity -- --ignored --nocapture metal_
```

CPU（`CpuBackendOps`）版は同テストファイルの属性なしテスト（`cpu_*`）で既に検証済み（green）。

## 期待結果

- 本実装は新規 GPU カーネルを持たない。新規 Op 4 種は常にホスト参照実装（`eval::elementwise_loss`）で計算されるため、
  CUDA／Metal の tape でも CPU tape と **bit 一致**する見込み（`*_bit_exact_*`）。
- `pos_weight` なしの委譲経路（既存 `Op::BceLoss`）は各バックエンドの融合カーネルを通りうるため、REQ-2 統一複合判定
  （相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）を満たす見込み（`*_req2_*`）。
- 形状は小さく（`[2, 3]`）、Metal の split-K 経路が発動する形状は含まない。

この前提が崩れる場合は本 README の「期待結果」を更新し、判定を外れた事実を PR へ記録すること
（tolerance の単独緩和は行わない。`.claude/rules/coding-rust.md`）。

## 実測記入欄

| 日付 | 実機 | コマンド | 結果 | 備考 |
|---|---|---|---|---|
| 未実測 | DGX Spark GB10 | 上記 CUDA | 未実測 | |
| 未実測 | Apple Silicon（M4 Max） | 上記 Metal | 未実測 | |
