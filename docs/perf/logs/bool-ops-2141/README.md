# bool_ops（#2141）CUDA／Metal 実機未実測の申し送り

`docs/autodiff-bool-ops-exposure-decision.md` §8「実装記録」参照。
本実装エージェント実行環境は CUDA／Metal 実機に到達できないため、
`fandhe_ai_autodiff::bool_ops`（比較 6 種・logical 3 種・`masked_select`）
の `crates/facade/tests/bool_ops_backend_parity.rs` のうち CUDA
（`Device::Cuda(0)`）・Metal（`Device::Metal`。`cfg(target_os = "macos")`
限定）を対象とする 2 テストは `#[ignore]` のまま未実測である。

## 測定コマンド案

```sh
# CUDA（DGX Spark GB10 等の実機上で）
cargo test -p fandhe-ai --test bool_ops_backend_parity -- --ignored --nocapture cuda

# Metal（Apple Silicon 実機上で）
cargo test -p fandhe-ai --test bool_ops_backend_parity -- --ignored --nocapture metal
```

CPU（`CpuBackendOps`）版は同テストファイルの属性なしテスト
（`cpu_compare_bool_matches_naive_reference`・
`cpu_logical_composition_matches_naive_reference`・
`cpu_masked_select_matches_naive_reference`）で既に検証済み（green）。

## 期待結果

比較 6 種は既存の比較カーネル（`scalar_binary_with_fallback`。イシュー
#1712 で CPU／CUDA／Metal 実装済み）→ 既存の cast カーネル
（`cast_from_f32_with_fallback::<bool>`。イシュー #1751 で CPU／CUDA／
Metal 実装済み）の合成であり、いずれも bit 完全一致契約を持つ
（`docs/tensor-core-cast-design.md`）。`masked_select` はホスト側での
値コピーのみ（算術を含まない）で、比較演算からの出力を経由するため
両カーネルの bit 一致契約が成立すれば CUDA／Metal でも CPU と bit 完全
一致するはず。この前提が崩れる場合（既存カーネルの実装が変わった、
デバイス間の cast/compare で想定外の丸めが混入した等）は本 README の
「期待結果」を更新し、bit 一致の hard assert を維持できない事実を型付き
findings として PR へ記録すること（tolerance の単独緩和は行わない。
`.claude/rules/coding-rust.md`）。
