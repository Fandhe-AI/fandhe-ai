# Softmin・Tanhshrink・Threshold・RReLU（#2650）CUDA／Metal 実機未実測の申し送り

> CUDA は #2992、Metal は #2993 で実測済み（2026-10-11。詳細は `docs/perf/logs/phase4-ops-autodiff-exposure-2678/README.md`）。以下の「未実測」は記入時点の記述（履歴）で、結果は末尾「実測記入欄」を正とする。

`docs/autodiff-softmin-threshold-ops-decision.md` §10 参照。本実装エージェント実行環境は CUDA／Metal 実機に
到達できないため、`crates/facade/tests/softmin_threshold_ops_backend_parity.rs` のうち CUDA（`Device::Cuda(0)`）・
Metal（`Device::Metal`。`cfg(target_os = "macos")` 限定）を対象とする計 8 テストは `#[ignore]` のまま未実測である。

- CUDA: `cuda_bit_exact_forward_matches_cpu_reference`・`cuda_bit_exact_backward_matches_cpu_reference`・
  `cuda_req2_forward_matches_cpu_reference_within_tolerance`・`cuda_req2_backward_matches_cpu_reference_within_tolerance`
- Metal: 上記と対称の `metal_*` 4 件

## 測定コマンド案

```sh
# CUDA（DGX Spark GB10 等の実機上で）
cargo test -p fandhe-ai --test softmin_threshold_ops_backend_parity -- --ignored --nocapture cuda_

# Metal（Apple Silicon 実機上で）
cargo test -p fandhe-ai --test softmin_threshold_ops_backend_parity -- --ignored --nocapture metal_
```

CPU（`CpuBackendOps`）版は同テストファイルの属性なしテスト（`cpu_*`）で既に検証済み（green）。

## 期待結果

- 本実装は新規 GPU カーネルを持たない。実機で走るのは **既存カーネル経路とホストフォールバック経路**
  （`Op::Softmax`・`tanh`・`neg`・`sub`・`mul`・`masked_fill` のうち各バックエンドが override しているもの）であり、
  本確認は新規カーネルの parity ではなく既存カーネルの新しい呼び出し形の確認である。
- `threshold`・`rrelu_with_noise`・`rrelu` 推論は選択と IEEE 乗算 1 回のみのため bit 一致の見込み。
  `softmin`・`tanhshrink` は超越関数を含むため REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）を
  満たす見込み。
- 形状は小さく（要素数 8 の rank 1）、Metal の split-K 経路が発動する形状は含まない。

この前提が崩れる場合は本 README の「期待結果」を更新し、REQ-2 判定を外れた事実を PR へ記録すること
（tolerance の単独緩和は行わない。`.claude/rules/coding-rust.md`）。

## 実測記入欄

| 日付 | 実機 | コマンド | 結果 | 備考 |
|---|---|---|---|---|
| 2026-10-11 | NVIDIA GB10（driver 580.173.02・CUDA 13.0） | `cargo test -p fandhe-ai --release --test softmin_threshold_ops_backend_parity -- --ignored --nocapture --test-threads=1 cuda_` | pass（`cuda_bit_exact_forward／backward_matches_cpu_reference・cuda_req2_forward／backward_matches_cpu_reference_within_tolerance`。4 passed; 0 failed） | 4 テスト（bit 一致 2 件・REQ-2 統一複合判定 2 件）全て pass。実測は #2992・PR #3006。ログ `docs/perf/logs/phase4-ops-autodiff-exposure-2678/cuda/softmin_threshold_ops_backend_parity.log`・`cuda/env_info.txt`。対象コミット `784b4e79` |
| 2026-10-11 | Mac mini（Apple M4・macOS 27.0.1） | `cargo test -p fandhe-ai --release --test softmin_threshold_ops_backend_parity -- --ignored --nocapture --test-threads=1 metal_` | pass（`metal_bit_exact_forward／backward_matches_cpu_reference・metal_req2_forward／backward_matches_cpu_reference_within_tolerance`。4 passed; 0 failed） | 4 テスト（bit 一致 2 件・REQ-2 統一複合判定 2 件）全て pass。実測は #2993・PR #3008。ログ `docs/perf/logs/phase4-ops-autodiff-exposure-2678/metal/softmin_threshold_ops_backend_parity.log`・`metal/env_info.txt`。対象コミット `c8e1ceb1` |
