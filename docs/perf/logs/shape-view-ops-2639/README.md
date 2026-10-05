# 形状演算 6 種（#2639）CUDA／Metal 実機未実測の申し送り

`docs/autodiff-shape-view-ops-decision.md` §10 参照。本実装エージェント実行環境は CUDA／Metal 実機に
到達できないため、`fandhe_ai_autodiff::shape_view_ops`（`unbind`／`movedim`／`swapaxes`／`tensor_split`／
`meshgrid`／`rot90`）の `crates/facade/tests/shape_view_ops_backend_parity.rs` のうち CUDA（`Device::Cuda(0)`）・Metal
（`Device::Metal`。`cfg(target_os = "macos")` 限定）を対象とする 2 テスト
（`cuda_shape_view_ops_match_cpu_reference`・`metal_shape_view_ops_match_cpu_reference`）は
`#[ignore]` のまま未実測である。

## 測定コマンド案

```sh
# CUDA（DGX Spark GB10 等の実機上で）
cargo test -p fandhe-ai --test shape_view_ops_backend_parity -- --ignored --nocapture cuda

# Metal（Apple Silicon 実機上で）
cargo test -p fandhe-ai --test shape_view_ops_backend_parity -- --ignored --nocapture metal
```

CPU（`CpuBackendOps`）版は同テストファイルの属性なしテスト（`cpu_matches_naive_reference`）で既に検証済み（green）。

## 期待結果

- `swapaxes`／`movedim`／`meshgrid` は view のみでバックエンドを呼ばない。`tensor_split`／`unbind` の backward は `Op::Narrow` の VJP が
  `BackendOps::concat` を呼ぶが、3 バックエンドとも未 override のため常にホスト参照実装へフォールバックする。
- `rot90` は `flip`（`Op::Gather`）経由で `gather`／`scatter` を呼ぶ。**CUDA・Metal はこの 2 つに実カーネルを持つため、実機では
  フォールバックではなく GPU カーネルが走る**。したがって `rot90` の項目は GPU の `gather`／`scatter` の parity 確認になり、
  forward は値 bit 一致（コピーのみ）、backward は REQ-2 統一複合判定を満たす見込み。
- 形状演算 6 種のうち結合順序が単一の連続ループと異なるものはない（`meshgrid` の backward の `reduce_to_shape` 縮約のみ GPU 側の加算順序に
  依存しうるが、既存の `broadcast_to` の VJP と同じ経路で、REQ-2 判定で比較する）。

この前提が崩れる場合は本 README の「期待結果」を更新し、REQ-2 判定を外れた事実を PR へ記録すること
（tolerance の単独緩和は行わない。`.claude/rules/coding-rust.md`）。

## 実測記入欄

| 日付 | 実機 | コマンド | 結果 | 備考 |
|---|---|---|---|---|
| 未実測 | DGX Spark GB10 | 上記 CUDA | 未実測 | |
| 未実測 | Apple Silicon（M4 Max） | 上記 Metal | 未実測 | |
