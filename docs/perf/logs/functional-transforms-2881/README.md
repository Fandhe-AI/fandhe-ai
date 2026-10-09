# vjp・hvp・vmap（#2881）CUDA／Metal 実機未実測の申し送り

`docs/autodiff-functional-transforms-design.md` §6・§21 参照。本実装エージェント実行環境は
CUDA／Metal 実機に到達できないため、`fandhe_ai_autodiff::functional_ops`（`vjp`／`hvp`／`vmap`）の
`crates/facade/tests/functional_ops_backend_parity.rs` のうち CUDA（`CudaBackendOps`）・Metal
（`MetalBackendOps`。`cfg(target_os = "macos")` 限定）を対象とする 2 テスト
（`cuda_functional_ops_match_cpu_reference`・`metal_functional_ops_match_cpu_reference`）は
`#[ignore]` のまま未実測である。

## 測定コマンド案

```sh
# CUDA（DGX Spark GB10 等の実機上で）
cargo test -p fandhe-ai --test functional_ops_backend_parity -- --ignored --nocapture cuda

# Metal（Apple Silicon 実機上で）
cargo test -p fandhe-ai --test functional_ops_backend_parity -- --ignored --nocapture metal
```

ワークスペース全体では `make test-ignored` でも実行される。CPU（`CpuBackendOps`）版は同テストファイルの
属性なしテスト（`cpu_matches_naive_reference`・`cpu_matches_hand_computed_values`）で検証済み（green）。

## 期待結果

- 新規カーネルは存在しない。`vjp`／`hvp`／`vmap` は既存 Op（matmul・tanh・sigmoid・mul・sum・
  narrow／reshape・contiguous・concat 等）の合成であり、確認対象は「既存カーネルの新しい呼び出し形
  （余接重み付き backward・子テープ上の VJP・スライスごとの実行と stack）が CPU tape と同じ結果になること」である。
- 判定は REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満。
  `fandhe_ai_backend_cpu::parity::assert_parity`）。`hvp` の子テープ経路は 1 階 VJP との bit 同一を主張せず、
  vmap も bit 一致を契約にしない（設計記録 §11-2）。形状は小さく、Metal split-K が発動する形状は使っていない。
- この前提が崩れる場合は本 README の「期待結果」を更新し、REQ-2 判定を外れた事実を PR へ記録すること
  （tolerance の単独緩和は行わない。`.claude/rules/coding-rust.md`）。実機上で `Unsupported` 等の型付きエラーが
  出た場合も、推測で直さず事実を記録して申し送る。

## 実測記入欄

| 日付 | 実機 | コマンド | 結果 | 備考 |
|---|---|---|---|---|
| 2026-10-09 | DGX Spark GB10 | `cargo test -p fandhe-ai --test functional_ops_backend_parity -- --ignored --nocapture cuda` | pass 1 / fail 0（`cuda_functional_ops_match_cpu_reference` ok。running 1 test） | main `8bbeb874ceb4748cbcf01b52e7162d02812bf8ba`・`rustc 1.97.0 (2d8144b78 2026-07-07)` |
| 未実測 | Apple Silicon（M4 Max） | 上記 Metal | 未実測 | |
