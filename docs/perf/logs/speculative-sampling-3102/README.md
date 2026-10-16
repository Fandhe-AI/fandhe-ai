# サンプリング版 speculative decoding（#3102）CUDA／Metal 実機未実測の申し送り

`docs/facade-speculative-decoding-batching-design.md` §20.7・§27 参照。本実装エージェント実行環境は
x86_64 Linux で CUDA toolkit を持たず、Metal にも到達できない。実機以外の GPU は「baseline・実測値は
実機実測のみ」の規約により代替にしない。このため
`crates/facade/tests/speculative_batching_backend_parity.rs` のうち CUDA（`CudaBackendOps`）・
Metal（`MetalBackendOps`。`cfg(target_os = "macos")` 限定）を対象とする 2 テストは `#[ignore]` のまま未実測である。

## 測定コマンド案

```sh
# CUDA（DGX Spark GB10 等の実機上で）
cargo test -p fandhe-ai --test speculative_batching_backend_parity -- --ignored --nocapture cuda_speculative_sampling

# Metal（Apple Silicon 実機上で）
cargo test -p fandhe-ai --test speculative_batching_backend_parity -- --ignored --nocapture metal_speculative_sampling
```

対象テスト: `cuda_speculative_sampling_matches_cpu`・`metal_speculative_sampling_matches_cpu`。
CPU（`CpuBackendOps`）対 NaiveOps は属性なしテスト `cpu_backend_ops_matches_naive_ops_for_speculative_sampling`
（と再生ハーネスの検出力を確かめる `replay_comparison_detects_tampering`）で CI 実行される。

## 方式と期待結果

- 設計 §20.7 の推奨方式。参照側（CPU）で `generate_speculative` をサンプリング設定
  （`Temperature(0.8)`・`TopK(3)`）で実行し、target・draft の `forward_step` 入力 id 列を記録する。
  比較側（実機）のモデルへ同じ id 列を再生し、呼び出しごとの logits を比べる。検証 forward（`L = k + 1`）と
  巻き戻し後の再 forward もそのままデバイスで通す
- 判定は呼び出しごとの logits に対する REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満。
  `fandhe_ai_backend_cpu::parity::assert_parity`）。呼び出し数・`seq_before`・ids・shape の一致も確かめる
- **token 列の一致は比べない**（契約でも仮説 H1 でもない。サンプリングの CDF 境界で logits の微差により反転しうる）
- 新規カーネルは無く、tolerance・baseline・係数は新設しない
- `Unsupported` 等の型付きエラーが出た場合も、推測で直さず事実を記録して申し送る

## 実測記入欄

| 日付 | 実機 | コマンド | 結果 | 備考 |
|---|---|---|---|---|
| 未実測 | DGX Spark GB10 | 上記 CUDA コマンド | 未実測 | |
| 未実測 | Apple Silicon | 上記 Metal コマンド | 未実測 | |
