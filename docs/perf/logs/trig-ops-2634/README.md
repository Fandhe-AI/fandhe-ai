# 逆三角関数・双曲線関数 9 種（#2634）CUDA／Metal 実機未実測の申し送り

> CUDA は #2992、Metal は #2993 で実測済み（2026-10-11。詳細は `docs/perf/logs/phase4-ops-autodiff-exposure-2678/README.md`）。以下の「未実測」は記入時点の記述（履歴）で、結果は末尾「実測記入欄」を正とする。

`docs/autodiff-trig-ops-decision.md` §10 参照。本実装エージェント実行環境は CUDA／Metal 実機に
到達できないため、`fandhe_ai_autodiff::trig_ops`（`atan`／`asin`／`acos`／`sinh`／`cosh`／`asinh`／
`acosh`／`atanh`／`atan2`）の `crates/facade/tests/trig_ops_backend_parity.rs` のうち CUDA
（`Device::Cuda(0)`）・Metal（`Device::Metal`。`cfg(target_os = "macos")` 限定）を対象とする 2 テスト
（`cuda_trig_ops_matches_cpu_reference`・`metal_trig_ops_matches_cpu_reference`）は `#[ignore]` のまま未実測である。

## 測定コマンド案

```sh
# CUDA（DGX Spark GB10 等の実機上で）
cargo test -p fandhe-ai --test trig_ops_backend_parity -- --ignored --nocapture cuda

# Metal（Apple Silicon 実機上で）
cargo test -p fandhe-ai --test trig_ops_backend_parity -- --ignored --nocapture metal
```

CPU（`CpuBackendOps`）版は同テストファイルの属性なしテスト（`cpu_unary_*`・`cpu_atan2_*`）で既に検証済み（green）。

## 期待結果

- CUDA／Metal の `scalar_unary`／`scalar_binary` は新 9 kind に対し明示 `None` → `Unsupported` を返すため
  （GPU カーネル未実装）、実機テストは常にホスト参照実装（`ScalarUnaryOp::apply`／`ScalarBinaryOp::apply`）へ
  フォールバックする経路になる想定。よって本テストは **GPU カーネルの parity ではなく、フォールバック経路が
  CPU tape と同じ結果になることの確認**であり、REQ-2 統一複合判定を満たす（実質 bit 一致）見込み。
- 将来 GPU 専用カーネル（`atanf`／`asinhf` 等の超越関数）を実装する場合は、ホスト `f32` 標準ライブラリとの
  ulp 差が出るため REQ-2 複合判定のみで検証する（`Sin`／`Tan` 等と同じ扱い）。別イシューの対象。

この前提が崩れる場合は本 README の「期待結果」を更新し、REQ-2 判定を外れた事実を PR へ記録すること
（tolerance の単独緩和は行わない。`.claude/rules/coding-rust.md`）。

## 実測記入欄

| 日付 | 実機 | コマンド | 結果 | 備考 |
|---|---|---|---|---|
| 2026-10-11 | NVIDIA GB10（driver 580.173.02・CUDA 13.0） | `cargo test -p fandhe-ai --release --test trig_ops_backend_parity -- --ignored --nocapture --test-threads=1 cuda_` | pass（`cuda_trig_ops_matches_cpu_reference`。1 passed; 0 failed） | 実測は #2992・PR #3006。ログ `docs/perf/logs/phase4-ops-autodiff-exposure-2678/cuda/trig_ops_backend_parity.log`・`cuda/env_info.txt`。対象コミット `784b4e79` |
| 2026-10-11 | Mac mini（Apple M4・macOS 27.0.1） | `cargo test -p fandhe-ai --release --test trig_ops_backend_parity -- --ignored --nocapture --test-threads=1 metal_` | pass（`metal_trig_ops_matches_cpu_reference`。1 passed; 0 failed） | 実測は #2993・PR #3008。ログ `docs/perf/logs/phase4-ops-autodiff-exposure-2678/metal/trig_ops_backend_parity.log`・`metal/env_info.txt`。対象コミット `c8e1ceb1` |
