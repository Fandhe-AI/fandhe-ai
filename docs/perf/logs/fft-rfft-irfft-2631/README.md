# rfft／irfft（#2631）CUDA／Metal 実機未実測の申し送り

> CUDA は #2992、Metal は #2993 で実測済み（2026-10-11。詳細は `docs/perf/logs/phase4-ops-autodiff-exposure-2678/README.md`）。以下の「未実測」は記入時点の記述（履歴）で、結果は末尾「実測記入欄」を正とする。

`docs/autodiff-fft-ops-decision.md` §10 参照。本実装エージェント実行環境は CUDA／Metal 実機に
到達できないため、`fandhe_ai_autodiff::fft_ops`（`rfft`／`irfft`）の
`crates/facade/tests/fft_ops_backend_parity.rs` のうち CUDA（`Device::Cuda(0)`）・Metal
（`Device::Metal`。`cfg(target_os = "macos")` 限定）を対象とする 2 テスト
（`cuda_fft_matches_cpu_reference`・`metal_fft_matches_cpu_reference`。各 rfft／irfft の
forward・backward）は `#[ignore]` のまま未実測である。

## 測定コマンド案

```sh
# CUDA（DGX Spark GB10 等の実機上で）
cargo test -p fandhe-ai --test fft_ops_backend_parity -- --ignored --nocapture cuda

# Metal（Apple Silicon 実機上で）
cargo test -p fandhe-ai --test fft_ops_backend_parity -- --ignored --nocapture metal
```

CPU（`CpuBackendOps`）版は同テストファイルの属性なしテスト（`cpu_rfft_*`・`cpu_irfft_*`・
`cpu_invalid_arguments_*`）で既に検証済み（green）。

## 期待結果

- `BackendOps::fft_rfft`／`fft_irfft` は CUDA／Metal では既定 `Unsupported` を返すため
  （GPU カーネル未実装。`crates/tensor-core/src/backend_ops.rs` の既定実装）、実機テストは常に
  共有ホストカーネル（`fandhe_ai_tensor_core::fft`）へフォールバックする経路になる想定。
  よって本テストは **GPU カーネルの parity ではなく、フォールバック経路が CPU tape と同じ結果に
  なることの確認**である。CPU `BackendOps` 実装と同じ共有カーネルを呼ぶため、REQ-2 統一複合判定を
  満たす（実質 bit 一致）見込み。
- 将来の GPU バタフライカーネル（radix-2 等）は直接 DFT と加算の結合順序が異なるため、
  `.claude/rules/coding-rust.md` の「結合順序が単一の連続 K ループと異なるカーネル」に当たる。
  baseline 非後退方式を採るかは GPU 専用カーネルの issue で決める（本 issue の対象外）。

この前提が崩れる場合（想定外の丸めの混入・フォールバック経路が機能していない等）は本 README の
「期待結果」を更新し、REQ-2 判定を外れた事実を型付き findings として PR へ記録すること
（tolerance の単独緩和は行わない。`.claude/rules/coding-rust.md`）。

## 実測記入欄

| 日付 | 実機 | コマンド | 結果 | 備考 |
|---|---|---|---|---|
| 2026-10-11 | NVIDIA GB10（driver 580.173.02・CUDA 13.0） | `cargo test -p fandhe-ai --release --test fft_ops_backend_parity -- --ignored --nocapture --test-threads=1 cuda_` | pass（`cuda_fft_matches_cpu_reference`。3 passed; 0 failed） | ファイル全体で 3 passed（`fft_ops_backend_parity.rs` の 3 テスト全てが部分一致で走る）。本機能は `*_fft_matches_cpu_reference`。実測は #2992・PR #3006。ログ `docs/perf/logs/phase4-ops-autodiff-exposure-2678/cuda/fft_ops_backend_parity.log`・`cuda/env_info.txt`。対象コミット `784b4e79` |
| 2026-10-11 | Mac mini（Apple M4・macOS 27.0.1） | `cargo test -p fandhe-ai --release --test fft_ops_backend_parity -- --ignored --nocapture --test-threads=1 metal_` | pass（`metal_fft_matches_cpu_reference`。3 passed; 0 failed） | ファイル全体で 3 passed（`fft_ops_backend_parity.rs` の 3 テスト全てが部分一致で走る）。本機能は `*_fft_matches_cpu_reference`。実測は #2993・PR #3008。ログ `docs/perf/logs/phase4-ops-autodiff-exposure-2678/metal/fft_ops_backend_parity.log`・`metal/env_info.txt`。対象コミット `c8e1ceb1` |
