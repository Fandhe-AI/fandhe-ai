# 可変長系列（pack／unpack・RNN 系 packed 実行。#2647）CUDA／Metal 実機未実測の申し送り

> CUDA は #3165 で GB10 実測済み（2026-10-19・04e48feb・pass 1/1）。Metal は未実測（#3166 段 B）。

`docs/autodiff-packed-sequence-decision.md` §10 参照。本実装エージェント実行環境は CUDA／Metal 実機に
到達できないため、`fandhe_ai_autodiff::nn::packed_sequence` の
`crates/facade/tests/packed_sequence_backend_parity.rs` のうち CUDA（`Device::Cuda(0)`）・Metal
（`Device::Metal`。`cfg(target_os = "macos")` 限定）を対象とする 2 テスト
（`cuda_packed_sequence_matches_cpu_reference`・`metal_packed_sequence_matches_cpu_reference`）は
`#[ignore]` のまま未実測である。

## 測定コマンド案

```sh
# CUDA（DGX Spark GB10 等の実機上で）
cargo test -p fandhe-ai --test packed_sequence_backend_parity -- --ignored --nocapture cuda_packed_sequence_matches_cpu_reference

# Metal（Apple Silicon 実機上で）
cargo test -p fandhe-ai --test packed_sequence_backend_parity -- --ignored --nocapture metal_packed_sequence_matches_cpu_reference
```

CPU（`CpuBackendOps`）版は同テストファイルの属性なしテスト（`cpu_matches_naive_reference`）で既に検証済み（green）。

## 期待結果

- 本実装は新規 GPU カーネルを持たない。実機で走るのは **既存カーネル経路**（`gather`／`scatter`・`gemm`・LSTM／GRU の
  pointwise と backward・`concat` のうち各バックエンドが override しているもの）であり、本確認は新規カーネルの parity では
  なく既存カーネルの新しい呼び出し形（可変バッチ幅の `narrow` ビュー・`cat` の連結）の確認である。
- pack／unpack の forward はコピーのみのため bit 一致（`exact_forward = true` のケース）。RNN 系の出力と全 backward は
  REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）を満たす見込み。
- 形状は小さく（`T ≤ 5`・`B ≤ 4`・`H ≤ 4`）、Metal の split-K 経路が発動する形状は含まない。

この前提が崩れる場合は本 README の「期待結果」を更新し、REQ-2 判定を外れた事実を PR へ記録すること
（tolerance の単独緩和は行わない。`.claude/rules/coding-rust.md`）。

## 実測記入欄

| 日付 | 実機 | コマンド | 結果 | 備考 |
|---|---|---|---|---|
| 2026-10-19 | DGX Spark GB10 | `cargo test -p fandhe-ai --test packed_sequence_backend_parity -- --ignored --exact --nocapture --test-threads=1 cuda_packed_sequence_matches_cpu_reference` | pass（1/1・0.73s。`cuda/cuda_packed_sequence_matches_cpu_reference.log`） | GB10（aarch64・sm_121）・ドライバ 580.173.02・CUDA 13.0（nvcc）・rustc 1.97.0。実行コミット 04e48feb（`../cuda-facade-parity-3165/rev-stamp.txt`・`cuda/rev-stamp.txt` とも dirty=0。対象コミット b640875f との `crates/` 等の diff なしを確認）。展開先は専用ツリー（RULE.txt の意図的な逸脱）。ビルドは判定外。GPU 経路（GPU フック経由かホストフォールバックか）は確認していない |
| 未実測 | Mac mini（Apple M4） | 上記 Metal | 未実測 | #3166 で Mac 側へ申し送り（`../metal-facade-parity-3166/`） |
