# speculative decoding・連続バッチング（#2890）CUDA／Metal 実機未実測の申し送り

`docs/facade-speculative-decoding-batching-design.md` §6.4・§16 参照。本実装エージェント実行環境は
x86_64 Linux で NVRTC／CUDA toolkit を持たず CUDA の JIT が成立せず、Metal にも到達できない。
実機以外の GPU は「baseline・実測値は実機実測のみ」の規約により代替にしない。このため
`crates/facade/tests/speculative_batching_backend_parity.rs` のうち CUDA（`CudaBackendOps`）・
Metal（`MetalBackendOps`。`cfg(target_os = "macos")` 限定）を対象とする 4 テストは `#[ignore]` のまま未実測である。

## 測定コマンド案

```sh
# CUDA（DGX Spark GB10 等の実機上で）
cargo test -p fandhe-ai --test speculative_batching_backend_parity -- --ignored --nocapture cuda

# Metal（Apple Silicon 実機上で）
cargo test -p fandhe-ai --test speculative_batching_backend_parity -- --ignored --nocapture metal
```

ワークスペース全体では `make test-ignored` でも実行される。CPU（`CpuBackendOps`）対 NaiveOps は同テストファイルの
属性なしテスト（`cpu_backend_ops_matches_naive_ops_for_speculative_greedy`・
`cpu_backend_ops_matches_naive_ops_for_scheduler_greedy`・`comparison_detects_logits_divergence`）で検証済み（green）。

対象テスト: `cuda_speculative_greedy_matches_cpu`・`cuda_scheduler_greedy_matches_cpu`・
`metal_speculative_greedy_matches_cpu`・`metal_scheduler_greedy_matches_cpu`。

## 期待結果

- 新規カーネルは存在しない。確認対象は既存カーネル（embedding・linear／matmul・MHA の `forward_with_cache`）の
  新しい呼び出し形である: 長さ `k+1` の検証 forward、KV 巻き戻し後の再 forward、要求ごとに B = 1 で交互に進む decode。
- 判定は呼び出しごとの logits に対する REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満。
  `fandhe_ai_backend_cpu::parity::assert_parity`）。bit 一致は契約にしない。形状は小さく Metal split-K は発動しない。
- token 列一致は事前登録の仮説 H1（全行の top1-top2 margin が REQ-2 定数から導いた下限を超える入力で成立）。
  破れた場合は判定を緩めず事実を PR／issue に記録し、設計 §10 論点 2 として承認依頼へ戻す。
- 同一バックエンドでのスケジューラ出力と単独 `generate` の token 列一致（§6.3）はデバイスのカーネル決定性に依存する。
  失敗した場合は所見として記録する。
- `Unsupported` 等の型付きエラーが出た場合も、推測で直さず事実を記録して申し送る（tolerance の単独緩和は行わない。
  `.claude/rules/coding-rust.md`）。
- 対象外: バックエンドをまたぐ TopK／Temperature の比較（ホスト `f64` サンプリングは CDF 境界で反転しうる。設計 §6.2）。

## 実測記入欄

| 日付 | 実機 | コマンド | 結果 | 備考 |
|---|---|---|---|---|
| 未実測 | DGX Spark GB10 | 上記 CUDA（speculative） | 未実測 | |
| 未実測 | DGX Spark GB10 | 上記 CUDA（scheduler） | 未実測 | |
| 未実測 | Apple Silicon（M4 Max） | 上記 Metal（speculative） | 未実測 | |
| 未実測 | Apple Silicon（M4 Max） | 上記 Metal（scheduler） | 未実測 | |
