# テンソル積・距離・外積（#2640）CUDA／Metal 実機未実測の申し送り

> CUDA は #2992、Metal は #2993 で実測済み（2026-10-11。詳細は `docs/perf/logs/phase4-ops-autodiff-exposure-2678/README.md`）。以下の「未実測」は記入時点の記述（履歴）で、結果は末尾「実測記入欄」を正とする。

`docs/autodiff-tensor-product-ops-decision.md` §10 参照。本実装エージェント実行環境は CUDA／Metal 実機に
到達できないため、`fandhe_ai_autodiff::tensor_product_ops`（`kron`／`tensordot`／`tensordot_axes`／`cdist`／`cross`）の
`crates/facade/tests/tensor_product_ops_backend_parity.rs` のうち CUDA（`Device::Cuda(0)`）・Metal
（`Device::Metal`。`cfg(target_os = "macos")` 限定）を対象とする 2 テスト
（`cuda_tensor_product_ops_match_cpu_reference`・`metal_tensor_product_ops_match_cpu_reference`）は
`#[ignore]` のまま未実測である。

## 測定コマンド案

```sh
# CUDA（DGX Spark GB10 等の実機上で）
cargo test -p fandhe-ai --test tensor_product_ops_backend_parity -- --ignored --nocapture cuda

# Metal（Apple Silicon 実機上で）
cargo test -p fandhe-ai --test tensor_product_ops_backend_parity -- --ignored --nocapture metal
```

CPU（`CpuBackendOps`）版は同テストファイルの属性なしテスト（`cpu_matches_naive_reference`）で既に検証済み（green）。

## 期待結果

- `kron`: forward は GPU の `mul`（乗算 1 回）で bit 一致、backward は broadcast の縮約を通るため REQ-2 統一複合判定。
- `tensordot`: GPU の `gemm`（backward は `gemm_fp32_strict`）で、既存 `matmul` の parity 契約に従う（新しい丸め経路なし）。
  テストの形状は小さく、Metal の split-K 経路が発動する形状は使っていない。
- `cdist`: 減算（`scalar_binary`）は GPU、ノルム（`vector_norm`／`vector_norm_p`）は CUDA／Metal が override していないためホスト参照実装
  （「減算は GPU・ノルムはホスト」の混在）。
- `cross`: `roll` の `gather`／`scatter` と `mul`／`scalar_binary` は GPU カーネルが走る（フォールバックではない）。乗算 2 回・減算 1 回のみで縮約なし。

この前提が崩れる場合は本 README の「期待結果」を更新し、REQ-2 判定を外れた事実を PR へ記録すること
（tolerance の単独緩和は行わない。`.claude/rules/coding-rust.md`）。

## 実測記入欄

| 日付 | 実機 | コマンド | 結果 | 備考 |
|---|---|---|---|---|
| 2026-10-11 | NVIDIA GB10（driver 580.173.02・CUDA 13.0） | `cargo test -p fandhe-ai --release --test tensor_product_ops_backend_parity -- --ignored --nocapture --test-threads=1 cuda_` | pass（`cuda_tensor_product_ops_match_cpu_reference`。1 passed; 0 failed） | 実測は #2992・PR #3006。ログ `docs/perf/logs/phase4-ops-autodiff-exposure-2678/cuda/tensor_product_ops_backend_parity.log`・`cuda/env_info.txt`。対象コミット `784b4e79` |
| 2026-10-11 | Mac mini（Apple M4・macOS 27.0.1） | `cargo test -p fandhe-ai --release --test tensor_product_ops_backend_parity -- --ignored --nocapture --test-threads=1 metal_` | pass（`metal_tensor_product_ops_match_cpu_reference`。1 passed; 0 failed） | 実測は #2993・PR #3008。ログ `docs/perf/logs/phase4-ops-autodiff-exposure-2678/metal/tensor_product_ops_backend_parity.log`・`metal/env_info.txt`。対象コミット `c8e1ceb1` |
