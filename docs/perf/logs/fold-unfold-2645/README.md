# Fold／Unfold（#2645）CUDA／Metal 実機未実測の申し送り

`docs/autodiff-fold-unfold-decision.md` §10 参照。本実装エージェント実行環境は CUDA／Metal 実機に到達できないため、
`fandhe_ai_autodiff::fold_ops` の `crates/facade/tests/fold_unfold_backend_parity.rs` のうち CUDA（`Device::Cuda(0)`）・
Metal（`Device::Metal`。`cfg(target_os = "macos")` 限定）を対象とする 2 テスト
（`cuda_fold_unfold_match_cpu_reference`・`metal_fold_unfold_match_cpu_reference`）は `#[ignore]` のまま未実測である。

## 測定コマンド案

```sh
# CUDA（DGX Spark GB10 等の実機上で）
cargo test -p fandhe-ai --test fold_unfold_backend_parity -- --ignored --nocapture cuda

# Metal（Apple Silicon 実機上で）
cargo test -p fandhe-ai --test fold_unfold_backend_parity -- --ignored --nocapture metal
```

CPU（`CpuBackendOps`）版は同テストファイルの属性なしテスト（`cpu_matches_naive_reference`）で既に検証済み（green。
NaiveOps tape のホストフォールバックと bit 一致）。

## 期待結果

本実装は新規 GPU カーネルを持たない。したがって本テストは **新規 GPU カーネルの parity ではなく、既存の
`im2col`／`col2im` GPU カーネル（CUDA #1766・Metal #1768。Conv2d の forward／VJP 用に実装済み）を通る経路が CPU tape と
bit 一致することの確認**である。

- `unfold` の forward（`im2col`）・`fold` の d_input（`im2col`）は算術を含まないコピーで、bit 一致する見込み。
- `fold` の forward・`unfold` の d_input（`col2im`）は `BackendOps::col2im` の trait doc が 3 バックエンド bit 一致を契約
  している（`f64` アキュムレータ。Metal は `soft_f64` 系）。bit 一致する見込み。
- **外れた場合は REQ-2 違反として tolerance を緩めず**、`col2im` カーネルの契約違反として別イシューで扱う
  （`.claude/rules/coding-rust.md`）。
- 将来 GPU 専用の fold／unfold カーネルを実装する場合は、`col2im` の `f64` アキュムレータ契約の適用要否を別イシューの
  決定記録で判断する。

この前提が崩れる場合は本 README の「期待結果」を更新し、判定を外れた事実を PR へ記録すること。

## 実測記入欄

| 日付 | 実機 | コマンド | 結果 | 備考 |
|---|---|---|---|---|
| 未実測 | DGX Spark GB10 | 上記 CUDA | 未実測 | |
| 未実測 | Apple Silicon（M4 Max） | 上記 Metal | 未実測 | |
