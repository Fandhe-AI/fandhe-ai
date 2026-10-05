# stft／istft（#2633）CUDA／Metal 実機未実測の申し送り

`docs/autodiff-fft-ops-decision.md` §13 参照。本実装エージェント実行環境は CUDA／Metal 実機に
到達できないため、`fandhe_ai_autodiff::fft_ops`（`stft`／`istft`）の
`crates/facade/tests/fft_ops_backend_parity.rs` のうち CUDA（`Device::Cuda(0)`）・Metal
（`Device::Metal`。`cfg(target_os = "macos")` 限定）を対象とする 2 テスト
（`cuda_stft_matches_cpu_reference`・`metal_stft_matches_cpu_reference`。`stft → istft` の
forward・backward）は `#[ignore]` のまま未実測である。

## 測定コマンド案

```sh
# CUDA（DGX Spark GB10 等の実機上で）
cargo test -p fandhe-ai --test fft_ops_backend_parity -- --ignored --nocapture cuda_stft

# Metal（Apple Silicon 実機上で）
cargo test -p fandhe-ai --test fft_ops_backend_parity -- --ignored --nocapture metal_stft
```

CPU（`CpuBackendOps`）版は同テストファイルの属性なしテスト（`cpu_stft_*`）で既に検証済み（green）。

## 期待結果

- `BackendOps::fft_stft`／`fft_istft` は CUDA／Metal では既定 `Unsupported` を返すため
  （GPU カーネル未実装。`crates/tensor-core/src/backend_ops.rs` の既定実装）、実機テストは常に
  共有ホストカーネル（`fandhe_ai_tensor_core::fft`）へフォールバックする経路になる想定。
  よって本テストは **GPU カーネルの parity ではなく、フォールバック経路が CPU tape と同じ結果に
  なることの確認**である。CPU `BackendOps` 実装と同じ共有カーネルを呼ぶため、REQ-2 統一複合判定を
  満たす（実質 bit 一致）見込み。
- 将来の GPU 専用 STFT カーネルは直接 DFT・重畳加算と加算の結合順序が異なりうるため、
  `.claude/rules/coding-rust.md` の「結合順序が単一の連続 K ループと異なるカーネル」に当たる。
  baseline 非後退方式を採るかは GPU 専用カーネルの issue で決める（本 issue の対象外）。

この前提が崩れる場合は本 README の「期待結果」を更新し、REQ-2 判定を外れた事実を型付き findings として
PR へ記録すること（tolerance の単独緩和は行わない。`.claude/rules/coding-rust.md`）。

## 実測記入欄

| 日付 | 実機 | コマンド | 結果 | 備考 |
|---|---|---|---|---|
| 未実測 | DGX Spark GB10 | 上記 CUDA | 未実測 | |
| 未実測 | Apple Silicon（M4 Max） | 上記 Metal | 未実測 | |
