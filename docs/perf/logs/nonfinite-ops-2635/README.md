# 非有限値の判定・置換 4 種（#2635）CUDA／Metal 実機未実測の申し送り

`docs/autodiff-nonfinite-ops-decision.md` §10 参照。本実装エージェント実行環境は CUDA／Metal 実機に
到達できないため、`fandhe_ai_autodiff::nonfinite_ops`（`isnan`／`isinf`／`isfinite`／`nan_to_num`）の
`crates/facade/tests/nonfinite_ops_backend_parity.rs` のうち CUDA（`Device::Cuda(0)`）・Metal
（`Device::Metal`。`cfg(target_os = "macos")` 限定）を対象とする 2 テスト
（`cuda_nonfinite_ops_matches_cpu_reference`・`metal_nonfinite_ops_matches_cpu_reference`）は
`#[ignore]` のまま未実測である。

## 測定コマンド案

```sh
# CUDA（DGX Spark GB10 等の実機上で）
cargo test -p fandhe-ai --test nonfinite_ops_backend_parity -- --ignored --nocapture cuda

# Metal（Apple Silicon 実機上で）
cargo test -p fandhe-ai --test nonfinite_ops_backend_parity -- --ignored --nocapture metal
```

CPU（`CpuBackendOps`）版は同テストファイルの属性なしテスト（`cpu_matches_naive_reference`）で既に検証済み（green）。

## 期待結果

- CUDA／Metal の `scalar_unary` は新 4 kind に対し明示 `None` → `Unsupported` を返すため（GPU カーネル未実装）、
  実機テストは常にホスト参照実装（`ScalarUnaryOp::apply`）へフォールバックする経路になる想定。よって本テストは
  **GPU カーネルの parity ではなく、フォールバック経路が CPU tape と同じ結果になることの確認**であり、判定 3 種は
  bool 完全一致・`nan_to_num` は bit 一致・勾配は REQ-2 統一複合判定を満たす見込み。bool 化は既存 cast カーネル
  （実機があれば GPU 上）を通るため、cast 経路の実機確認も兼ねる。
- 将来 GPU 専用カーネルを実装する場合、`NanToNum` は 3 値ペイロードが必要（別イシューの対象）。

この前提が崩れる場合は本 README の「期待結果」を更新し、REQ-2 判定を外れた事実を PR へ記録すること
（tolerance の単独緩和は行わない。`.claude/rules/coding-rust.md`）。

## 実測記入欄

| 日付 | 実機 | コマンド | 結果 | 備考 |
|---|---|---|---|---|
| 未実測 | DGX Spark GB10 | 上記 CUDA | 未実測 | |
| 未実測 | Apple Silicon（M4 Max） | 上記 Metal | 未実測 | |
