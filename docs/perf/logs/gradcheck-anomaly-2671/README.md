# gradcheck・anomaly detection（#2671）CUDA／Metal 実機未実測の申し送り

> CUDA は #2992、Metal は #2993 で実測済み（2026-10-11。詳細は `docs/perf/logs/phase4-ops-autodiff-exposure-2678/README.md`）。以下の「未実測」は記入時点の記述（履歴）で、結果は末尾「実測記入欄」を正とする。

`docs/autodiff-jacobian-hessian-gradcheck-decision.md` 「実装記録（#2671）」参照。本実装エージェント実行環境は
CUDA／Metal 実機に到達できないため、`fandhe_ai_autodiff::gradcheck::gradcheck`・
`fandhe_ai_autodiff::anomaly::backward_detect_anomaly` の
`crates/facade/tests/gradcheck_anomaly_backend_parity.rs` のうち CUDA（`CudaBackendOps`）・Metal
（`MetalBackendOps`。`cfg(target_os = "macos")` 限定）を対象とする 2 テスト
（`cuda_gradcheck_anomaly_match_cpu_reference`・`metal_gradcheck_anomaly_match_cpu_reference`）は
`#[ignore]` のまま未実測である。

## 測定コマンド案

```sh
# CUDA（DGX Spark GB10 等の実機上で）
cargo test -p fandhe-ai --test gradcheck_anomaly_backend_parity -- --ignored --nocapture cuda

# Metal（Apple Silicon 実機上で）
cargo test -p fandhe-ai --test gradcheck_anomaly_backend_parity -- --ignored --nocapture metal
```

CPU（`CpuBackendOps`）版は同テストファイルの属性なしテスト（`cpu_matches_naive_reference`・
`cpu_matches_hand_computed_values`）で既に検証済み（green）。

## 期待結果

- 新規カーネルは存在しない。gradcheck は既存の `jacobian`（`Tape::backward` の繰り返し）と forward の再評価、
  anomaly は `Tape::backward` と読み取り専用の走査のみで、確認対象は「既存カーネルの新しい呼び出し形が CPU tape と
  同じ合否・同じ検出ノード・同じメッセージになること」である。
- 数値は REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満。
  `fandhe_ai_backend_cpu::parity::assert_parity`）。形状は小さく、Metal split-K が発動する形状は使っていない。
- GPU テープでは anomaly 走査がホスト読み出しを伴う（デバッグ用途）。
- 前提が崩れる場合は本 README の「期待結果」を更新し、REQ-2 判定を外れた事実を PR へ記録すること
  （tolerance の単独緩和は行わない。`.claude/rules/coding-rust.md`）。

## 実測記入欄

| 日付 | 実機 | コマンド | 結果 | 備考 |
|---|---|---|---|---|
| 2026-10-11 | NVIDIA GB10（driver 580.173.02・CUDA 13.0） | `cargo test -p fandhe-ai --release --test gradcheck_anomaly_backend_parity -- --ignored --nocapture --test-threads=1 cuda_` | pass（`cuda_gradcheck_anomaly_match_cpu_reference`。1 passed; 0 failed） | 実測は #2992・PR #3006。ログ `docs/perf/logs/phase4-ops-autodiff-exposure-2678/cuda/gradcheck_anomaly_backend_parity.log`・`cuda/env_info.txt`。対象コミット `784b4e79` |
| 2026-10-11 | Mac mini（Apple M4・macOS 27.0.1） | `cargo test -p fandhe-ai --release --test gradcheck_anomaly_backend_parity -- --ignored --nocapture --test-threads=1 metal_` | pass（`metal_gradcheck_anomaly_match_cpu_reference`。1 passed; 0 failed） | 実測は #2993・PR #3008。ログ `docs/perf/logs/phase4-ops-autodiff-exposure-2678/metal/gradcheck_anomaly_backend_parity.log`・`metal/env_info.txt`。対象コミット `c8e1ceb1` |
