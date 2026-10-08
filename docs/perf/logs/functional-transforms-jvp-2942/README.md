# jvp・jacfwd（#2942）CUDA／Metal 実機未実測の申し送り

`docs/autodiff-functional-transforms-design.md` §8・§24・§25 参照。本実装エージェント実行環境は
CUDA／Metal 実機に到達できないため、`crates/facade/tests/functional_ops_jvp_backend_parity.rs` のうち
CUDA（`CudaBackendOps`）・Metal（`MetalBackendOps`。`cfg(target_os = "macos")` 限定）を対象とする 2 テスト
（`cuda_double_vjp_jvp_jacfwd_match_cpu_reference`・`metal_double_vjp_jvp_jacfwd_match_cpu_reference`）は
`#[ignore]` のまま未実測である。

`fandhe_ai_autodiff::functional_ops::{jvp, jacfwd}` は `pub(crate)`（facade 非公開）で統合テストから
直接呼べないため、同テストは公開 API だけで #2940 の double-VJP 手順をミラーしている。公開が承認された後は
本物の関数呼び出しへ置き換える（#2941 の担当範囲）。

## 測定コマンド案

```sh
# CUDA（DGX Spark GB10 等の実機上で）
cargo test -p fandhe-ai --test functional_ops_jvp_backend_parity -- --ignored --nocapture cuda

# Metal（Apple Silicon 実機上で）
cargo test -p fandhe-ai --test functional_ops_jvp_backend_parity -- --ignored --nocapture metal
```

ワークスペース全体では `make test-ignored` でも実行される。CPU（`CpuBackendOps`）版は同テストファイルの
属性なしテスト（`cpu_double_vjp_matches_naive_reference`・`cpu_double_vjp_matches_jacobian`・
`cpu_double_vjp_matches_hand_computed_values`）で検証済み（green）。

## 期待結果

- 新規カーネルは存在しない。`jvp`／`jacfwd` は既存 Op（matmul・add・tanh・sigmoid・exp・mul・transpose・
  narrow・reshape 等）の合成であり、確認対象は「既存カーネルの新しい呼び出し形（追跡あり余接葉を含む親
  `backward_create_graph` と子テープ上の第 2 段 VJP）が CPU tape と同じ結果になること」である。
- 判定は REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満。
  `fandhe_ai_backend_cpu::parity::assert_parity`）。子テープ経路は 1 階 VJP・`jacobian` との bit 同一を
  主張しない。形状は小さく、Metal split-K が発動する形状は使っていない。
- この前提が崩れる場合は本 README の「期待結果」を更新し、REQ-2 判定を外れた事実を PR へ記録すること
  （tolerance の単独緩和は行わない。`.claude/rules/coding-rust.md`）。実機上で `Unsupported` 等の型付きエラーが
  出た場合も、推測で直さず事実を記録して申し送る。

## 実測記入欄

| 日付 | 実機 | コマンド | 結果 | 備考 |
|---|---|---|---|---|
| 未実測 | DGX Spark GB10 | 上記 CUDA | 未実測 | |
| 未実測 | Apple Silicon（M4 Max） | 上記 Metal | 未実測 | |
