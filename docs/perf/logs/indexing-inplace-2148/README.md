# indexing_ops（#2148）CUDA／Metal 実機未実測の申し送り

`docs/autodiff-indexing-inplace-design.md` §7・§8「実装記録」参照。
本実装エージェント実行環境は CUDA／Metal 実機に到達できないため、
`fandhe_ai_autodiff::indexing_ops`（`advanced_indexing`／`index_put`）
の `crates/facade/tests/indexing_ops_backend_parity.rs` のうち CUDA
（`Device::Cuda(0)`）・Metal（`Device::Metal`。`cfg(target_os =
"macos")` 限定）を対象とする 4 テストは `#[ignore]` のまま未実測で
ある。

## 測定コマンド案

```sh
# CUDA（DGX Spark GB10 等の実機上で）
cargo test -p fandhe-ai --test indexing_ops_backend_parity -- --ignored --nocapture cuda

# Metal（Apple Silicon 実機上で）
cargo test -p fandhe-ai --test indexing_ops_backend_parity -- --ignored --nocapture metal
```

CPU（`CpuBackendOps`）版は同テストファイルの属性なしテスト
（`cpu_advanced_indexing_forward_bit_matches_naive_reference`・
`cpu_index_put_overwrite_forward_bit_matches_naive_reference`・
`cpu_index_put_accumulate_forward_matches_naive_reference`・
`cpu_indexing_backward_matches_naive_reference`・
`cpu_advanced_indexing_preserves_nan_payload`）で既に検証済み
（green）。

## 期待結果

- `advanced_indexing` forward・`index_put(accumulate=false)` forward
  はコピーのみ（算術を含まない `Op::Gather`／`Op::Scatter(Overwrite)`
  の合成）のため、CUDA／Metal でも CPU と **bit 完全一致**するはず
  （`gather`／`scatter` は CPU・CUDA（#1777）・Metal（#1778）いずれ
  にもネイティブカーネルが実装済みで、CPU 参照実装と bit 同一と
  ドキュメントされている）。
- `index_put(accumulate=true)` forward は `ScatterReduce::Add` の
  `f64` 決定的集約契約に従う。`scatter_add` の Metal ネイティブ
  カーネルは既存の `crates/backend-metal/tests/gather_scatter_
  parity.rs` で CPU と bit 一致を確認済み（本 PR の差分外）のため、
  同じ結果が期待される。CUDA 側も同型の既存カーネル契約に従う想定。
  乖離があれば REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対
  誤差 1e-5 未満）で比較する。

この前提が崩れる場合（想定外の丸めが混入した、フォールバック経路が
機能していない等）は本 README の「期待結果」を更新し、REQ-2 判定を
外れた事実を型付き findings として PR へ記録すること（tolerance の
単独緩和は行わない。`.claude/rules/coding-rust.md`）。
