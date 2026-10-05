# ヒストグラム・二分探索系 4 種（#2638）CUDA／Metal 実機未実測の申し送り

`docs/autodiff-binning-ops-decision.md` §10 参照。本実装エージェント実行環境は CUDA／Metal 実機に
到達できないため、`fandhe_ai_autodiff::binning_ops`（`histc`／`bincount`／`searchsorted`／`bucketize`）の
`crates/facade/tests/binning_ops_backend_parity.rs` のうち CUDA（`CudaBackendOps`）・Metal
（`MetalBackendOps`。`cfg(target_os = "macos")` 限定）を対象とする 2 テスト
（`cuda_binning_ops_match_cpu_reference`・`metal_binning_ops_match_cpu_reference`）は
`#[ignore]` のまま未実測である。

## 測定コマンド案

```sh
# CUDA（DGX Spark GB10 等の実機上で）
cargo test -p fandhe-ai --test binning_ops_backend_parity -- --ignored --nocapture cuda

# Metal（Apple Silicon 実機上で）
cargo test -p fandhe-ai --test binning_ops_backend_parity -- --ignored --nocapture metal
```

CPU（`CpuBackendOps`）版は同テストファイルの属性なしテスト（`cpu_matches_naive_reference`）で既に検証済み（green）。

## 期待結果

- CUDA／Metal は `BackendOps::binning_*` 4 メソッドを持たず既定の `Unsupported` を返すため、実機テストは常に共有
  ホストカーネル（`fandhe_ai_tensor_core::binning`）へフォールバックする経路になる想定。よって本テストは
  **GPU カーネルの parity ではなく、フォールバック経路が CPU tape と同じ結果になることの確認**であり、索引・整数カウント・
  `histc` のカウントは完全一致、重み付き `bincount` は REQ-2 統一複合判定を満たす見込み。
- 将来 GPU 専用カーネルを実装する場合、`histc`・重み付き `bincount` は原子加算の順序で結合順序が単一の連続ループと
  異なりうる（重み付きの和は `f64` アキュムレータ契約との整合も要る）。`histc` のビン添字は `f32` の演算順が
  境界要素のビンを決めるため、GPU 側でも `(x - left) * bins / (right - left)` の演算順と FMA 非融合を維持する必要がある。
  baseline 非後退方式の適用要否は決定記録で判断する（別イシューの対象）。

この前提が崩れる場合は本 README の「期待結果」を更新し、REQ-2 判定を外れた事実を PR へ記録すること
（tolerance の単独緩和は行わない。`.claude/rules/coding-rust.md`）。

## 実測記入欄

| 日付 | 実機 | コマンド | 結果 | 備考 |
|---|---|---|---|---|
| 未実測 | DGX Spark GB10 | 上記 CUDA | 未実測 | |
| 未実測 | Apple Silicon（M4 Max） | 上記 Metal | 未実測 | |
