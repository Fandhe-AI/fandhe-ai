# CUDA TF32 opt-in GEMM の複合判定・実測記録（#1042）

イシュー #1042 の受け入れ条件 2「opt-in 時の複合判定（parity）結果を記録す
る」に対応する。実装した公開 API（`fandhe_ai::set_cuda_tf32_gemm_enabled`）・
環境適応スモークテスト（`crates/backend-cuda/src/ops.rs` の
`gemm_routes_to_tf32_path_when_optin_flag_is_enabled_env_adaptive`）は
`docs/cuda-tf32-optin-api-decision.md` を参照。

## 状態: 未実測（本エージェント実行環境に CUDA 実機なし）

本ドキュメントを執筆したエージェント実行環境（macOS worktree）には CUDA
driver・実機がないため、TF32 opt-in 経路の実機複合判定（parity）計測は
未実施である。既存の下記記録が既に検証済みのため、opt-in 経路が呼び出す
カーネル自体（`CudaGemm::run_wmma_tf32`）の誤差分布は新規測定を要しない
（新規に測定が必要なのは「facade の opt-in スイッチ経由で到達した場合も
同じ誤差分布になるか」という配線の正しさのみ）:

- `docs/perf/cuda-tensor-core-tolerance-opt-remeasurement.md`
  （opt 版 WMMA TF32 カーネルの誤差分布再実測。GB10 実機計測完了・sm_86 との
  差分なし。#994・#995）
- `docs/perf/cuda-tensor-core-tolerance-gb10-scale-sweep.md`
  （GB10 実機での入力スケールスイープ再実測。#995）

実測値を捏造せず「未実測」と明記して安全側で本イシューをクローズする
（`.claude/rules/coding-rust.md`「テスト・ベンチ」節・`docs/perf/cuda-fresh-
gemm-n2048-overhead-diagnosis.md` 等の先例と同じ方針）。実機セッションが
持ち帰り次第、本ドキュメントの「実測結果」節を更新する。

## 実機実行手順（DGX Spark GB10 想定。持ち帰り用）

`docs/real-hardware-verification-env.md` の接続・転送手順に従い、以下を
DGX Spark GB10（CUDA 13.0 系）実機で実行する。

### 1. opt-in 経路の環境適応スモークテスト（実機なら本体まで検証）

```bash
cargo test -p fandhe-ai-backend-cuda --lib -- \
  ops::tests::gemm_routes_to_tf32_path_when_optin_flag_is_enabled_env_adaptive \
  ops::tests::gemm_stays_on_fp32_path_when_tf32_optin_flag_is_disabled_env_adaptive \
  --nocapture
```

実機上では `gemm_routes_to_tf32_path_when_optin_flag_is_enabled_env_adaptive`
が `Ok(_)` 分岐（`TF32_OPTIN_GEMM_LAUNCH_COUNT` 増加のアサーション）へ到達
することを確認する。

### 2. `#[ignore]` 実機複合判定（形状網羅）

`make test-ignored-cuda` 導線で `crates/backend-cuda/tests/
parity_nonregression.rs`・`tests/gemm_tf32_optin.rs`（新規追加。§3 参照）を
実行し、統一複合判定「相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満」
（REQ-2・`.claude/rules/coding-rust.md`）で opt-in 経路が CPU 参照実装と
一致することを確認する。

### 3. facade 経由の複合判定（新規テストのひな形）

`crates/backend-cuda/tests/gemm_tf32_optin.rs`
（`docs/cuda-tf32-optin-api-decision.md` 実装計画 §3「テスト」節）が
CUDA 可用時のみ以下を検証する:

1. `crate::precision::set_tf32_gemm_enabled(true)` の状態で
   `CudaBackendOps::gemm` を複数形状（正方・非正方・K 支配的）で実行。
2. 同じ入力を CPU 参照実装（`fandhe_ai_backend_cpu::CpuBackendOps::gemm`）で
   実行し、要素ごとに複合判定を適用。
3. `set_tf32_gemm_enabled(false)` へ戻した直後の `gemm` 出力が、本イシュー
   導入前と bit-exact に一致すること（既定 OFF の非後退契約）。

### 5 回計測中央値・記入欄

`.claude/rules/coding-rust.md`「テスト・ベンチ」節（5 回計測の中央値）に
従い、形状ごとに 5 回計測した最大相対誤差・最大絶対誤差の中央値を記録する。

| 形状 (M×N×K) | 最大相対誤差（中央値） | 最大絶対誤差（中央値） | 複合判定 | 実測日 |
|---|---|---|---|---|
| 512×512×512 | 未実測 | 未実測 | 未実測 | 未実測 |
| 1024×1024×1024 | 未実測 | 未実測 | 未実測 | 未実測 |
| 4096×4096×4096 | 未実測 | 未実測 | 未実測 | 未実測 |
| 非正方（K 支配的。例: 4096×256×4096） | 未実測 | 未実測 | 未実測 | 未実測 |

## 実測結果

（未実施。実機セッション実行後にここへ追記する。）
