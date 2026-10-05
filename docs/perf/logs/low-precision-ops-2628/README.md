# MatMul・elementwise 5 演算の低精度 forward（#2628）CUDA／Metal 実機未実測の申し送り

`docs/autodiff-low-precision-op-extension-decision.md` 6 節参照。**本書は実測値を含まない**。本実装エージェント実行環境は CUDA／Metal 実機に到達できないため、`crates/facade/tests/low_precision_ops_backend_parity.rs` のうち実機を対象とする 2 テストは `#[ignore]` のまま未実測である。実測は #2629（実機ツリー #2683）が担う。

- `cuda_low_precision_ops_match_cpu_reference`（`Device::Cuda(0)`）
- `metal_low_precision_ops_match_cpu_reference`（`Device::Metal`。`cfg(target_os = "macos")` 限定）

各テストは 6 Op（matmul／add／mul／relu／exp／tanh）× {F16, Bf16} の forward を、CPU tape と同一 dtype・有限出力の入力で `fandhe_ai_backend_cpu::parity::assert_parity` により突合する。

## 測定コマンド案

```sh
# CUDA（DGX Spark GB10 等の実機上で）
cargo test -p fandhe-ai --release --test low_precision_ops_backend_parity -- --ignored --nocapture --test-threads=1 cuda_

# Metal（Apple Silicon 実機上で）
cargo test -p fandhe-ai --release --test low_precision_ops_backend_parity -- --ignored --nocapture --test-threads=1 metal_
```

CPU 版（P1・非有限出力）は同テストファイルの属性なしテストで既に検証済み（green）。

## 事前登録判定規則

- 判定は REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）。tolerance・baseline は変更しない。
- **P6（CUDA／Metal vs CPU。自前バックエンド間）の不一致は「判定不能」にせず、通常の parity 失敗として扱う**。カーネルまたは丸め方針の不具合として修正する（PyTorch 比較〈P4・P5〉の「第三者比較対象の判定不能」とは別）。
- CUDA／Metal は `TypedOps<f16>`／`TypedOps<bf16>` の accessor（`typed_ops_f16`／`typed_ops_bf16`）経由で到達する。accessor が `None` のバックエンドは `Unsupported` で失敗し、それ自体が実測結果として記録される（ホスト計算・f32 へのフォールバックは持たない）。
- **Metal の bf16 の実機可用性は未検証**。

## 実測記入欄

| 日付 | 実機 | コマンド | 結果 | 備考 |
|---|---|---|---|---|
| 未実測 | DGX Spark GB10 | 上記 CUDA | 未実測（verdict=undetermined） | |
| 未実測 | Apple Silicon（M4 Max） | 上記 Metal | 未実測（verdict=undetermined） | |
