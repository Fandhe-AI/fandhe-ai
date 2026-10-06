# 索引付き更新 4 種（#2641）CUDA／Metal 実機未実測の申し送り

`docs/autodiff-indexed-update-ops-decision.md` §10 参照。本実装エージェント実行環境は CUDA／Metal 実機に
到達できないため、`fandhe_ai_autodiff::indexed_update_ops`（`scatter_reduce`／`index_add`／`index_copy`／
`masked_scatter`）の `crates/facade/tests/indexed_update_ops_backend_parity.rs` のうち CUDA
（`Device::Cuda(0)`）・Metal（`Device::Metal`。`cfg(target_os = "macos")` 限定）を対象とする 2 テスト
（`cuda_indexed_update_ops_match_cpu_reference`・`metal_indexed_update_ops_match_cpu_reference`）は
`#[ignore]` のまま未実測である。

## 測定コマンド案

```sh
# CUDA（DGX Spark GB10 等の実機上で）
cargo test -p fandhe-ai --test indexed_update_ops_backend_parity -- --ignored --nocapture cuda

# Metal（Apple Silicon 実機上で）
cargo test -p fandhe-ai --test indexed_update_ops_backend_parity -- --ignored --nocapture metal
```

CPU（`CpuBackendOps`）版は同テストファイルの属性なしテスト（`cpu_matches_naive_reference`）で既に検証済み（green）。

## 期待結果（演算の系統で意味が異なる）

- **`scatter_reduce`**: CUDA／Metal は `BackendOps::indexed_scatter_reduce` を持たず既定の `Unsupported` を返すため、
  常に共有ホストカーネル（`fandhe_ai_tensor_core::indexed_update`）へフォールバックする経路になる想定。
  よってこの部分は **GPU カーネルの parity ではなく、フォールバック経路が CPU tape と同じ結果になることの
  確認**であり、`Amax`／`Amin` は bit 一致、`Sum`／`Prod`／`Mean` と各 backward は REQ-2 統一複合判定を満たす見込み。
- **`index_add`／`index_copy`／`masked_scatter`**: 既存の `BackendOps::scatter`（CUDA は #1777、Metal は #1778 の
  GPU カーネル）を通るため、実機では **既存 GPU scatter カーネル経由**になる（新しい GPU カーネルは追加していない）。
  `index_add` は `Add`（`f64` 相当の決定的集約契約）、`index_copy`／`masked_scatter` は `Overwrite`（コピーのみ。
  forward は bit 一致）の契約を満たす見込み。Metal の rank 上限超過などで `Unsupported` を返す経路は
  `eval::scatter` へ落ちる既存契約（モック `BackendOps` のテストで到達を確認済み）。
- 将来 `scatter_reduce` の GPU 専用カーネルを実装する場合、`Sum`／`Mean`／`Prod` は結合順序が単一の連続ループと
  異なりうるため、`.claude/rules/coding-rust.md` の baseline 非後退方式の適用要否を決定記録で判断する
  （別イシューの対象）。

この前提が崩れる場合は本 README の「期待結果」を更新し、REQ-2 判定を外れた事実を PR へ記録すること
（tolerance の単独緩和は行わない。`.claude/rules/coding-rust.md`）。

## 実測記入欄

| 日付 | 実機 | コマンド | 結果 | 備考 |
|---|---|---|---|---|
| 未実測 | DGX Spark GB10 | 上記 CUDA | 未実測 | |
| 未実測 | Apple Silicon（M4 Max） | 上記 Metal | 未実測 | |
