# LocalResponseNorm・weight_norm・spectral_norm（#2646）CUDA／Metal 実機未実測の申し送り

`docs/autodiff-lrn-weight-reparam-decision.md` §10 参照。本実装エージェント実行環境は CUDA／Metal 実機に到達できないため、
`fandhe_ai_autodiff::{lrn_ops, weight_reparam_ops}` の `crates/facade/tests/lrn_weight_reparam_backend_parity.rs` のうち
CUDA（`Device::Cuda(0)`）・Metal（`Device::Metal`。`cfg(target_os = "macos")` 限定）を対象とする 2 テスト
（`cuda_lrn_weight_reparam_match_cpu_reference`・`metal_lrn_weight_reparam_match_cpu_reference`）は `#[ignore]` の
まま未実測である。

## 測定コマンド案

```sh
# CUDA（DGX Spark GB10 等の実機上で）
cargo test -p fandhe-ai --test lrn_weight_reparam_backend_parity -- --ignored --nocapture cuda

# Metal（Apple Silicon 実機上で）
cargo test -p fandhe-ai --test lrn_weight_reparam_backend_parity -- --ignored --nocapture metal
```

CPU（`CpuBackendOps`）版は同テストファイルの属性なしテスト（`cpu_matches_naive_reference`）で既に検証済み（green。
NaiveOps tape のホストフォールバックと bit 一致）。

## 期待結果

本実装は新規 GPU カーネルを持たず、CUDA／Metal は新規 3 フック（`lrn_forward`・`weight_norm_forward`・
`spectral_norm_forward`）を override しない（既定 `Unsupported`）。したがって本テストは **新規 GPU カーネルの parity では
なく、既定 `Unsupported` → ホストフォールバック経路が CPU tape と bit 一致することの確認**である。

- forward は共有ホストカーネルを通り、CPU tape（`CpuBackendOps` の override も同じ共有カーネル）と bit 一致する見込み。
  VJP・`SpectralNormState` の power iteration は両 tape ともホスト側のみで bit 一致する見込み。
- 周辺の演算（`mul`・`sum`・backward の勾配集約）は各バックエンドの既存カーネルを通るため、この部分が外れた場合は本機能
  ではなく当該バックエンドの既存契約違反として扱う。
- **外れた場合は REQ-2 違反として tolerance を緩めず**、原因を特定して別イシューで扱う（`.claude/rules/coding-rust.md`）。
- 将来 GPU 専用の LRN／weight_norm／spectral_norm カーネルを実装する場合は、正規化統計の `f64` 相当契約（Metal は
  `double` 非対応のため補償和または `soft_f64` 系）の適用要否を別イシューの決定記録で判断する。

この前提が崩れる場合は本 README の「期待結果」を更新し、判定を外れた事実を PR へ記録すること。

## 実測記入欄

| 日付 | 実機 | コマンド | 結果 | 備考 |
|---|---|---|---|---|
| 未実測 | DGX Spark GB10 | 上記 CUDA | 未実測 | |
| 未実測 | Apple Silicon（M4 Max） | 上記 Metal | 未実測 | |
