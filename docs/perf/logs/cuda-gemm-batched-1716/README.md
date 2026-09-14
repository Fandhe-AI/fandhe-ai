# イシュー #1716 実測記録先（CUDA バッチ行列積・デバイス常駐バッチループ経路）

## 位置づけ（本 PR 時点では未実測）

本 PR の実行環境（Linux コンテナ／worktree）には DGX Spark GB10 実機
への到達手段（`docs/real-hardware-verification-env.local.md`・
`CUDA_NODE` 環境変数・`~/.ssh/config` のいずれも確認できず）がないため、
**実測値は一切含まれていない**。本ディレクトリは実行コマンド・保存
すべきログ一覧のみを提供し、実測は GB10 実機を持つセッションへ申し送る
（`docs/perf/logs/cuda-mse-backward-1692/README.md` と同型の運用）。

## 目的

イシュー #1716「CUDA バッチ行列積のデバイス常駐バッチループ経路
（`CudaGemm::run_tiled_f32_batched`）」の実機正しさ検証記録先。本イシュー
は正しさ・数値契約のみを対象とし（性能 A/B はスコープ外・実装計画 §8）、
以下 3 系統の `#[ignore]` テストを GB10 実機で実行して green を確認する
ことが目的。

## 実行コマンド

```bash
# 1) crates/backend-cuda の bit 同一・REQ-2 parity（形状網羅。整列／非整列
#    ／broadcast／rank 4／退化形状）
cargo test -p fandhe-ai-backend-cuda --release --test gemm_batched_parity -- --ignored --nocapture

# 2) crates/backend-cuda の ops.rs 内 TF32 opt-in カウンタテスト（実機で
#    実際に TF32 経路へルーティングされることまで含めて再確認する場合）
cargo test -p fandhe-ai-backend-cuda --release --lib -- --ignored gemm

# 3) crates/facade の backward（rank≥3 VJP）parity
cargo test -p fandhe-ai --release --test batched_matmul_backend_parity -- --ignored --nocapture
```

`make test-ignored-cuda`（`--all-features -- --ignored`）にも上記が
自動的に含まれる。

## 保存すべきログ

- `gemm_batched_parity-ignored.log`: 上記 1) の生出力（9 形状 ×
  bit 一致・run-to-run 決定性・CPU 参照実装との parity の判定内訳）
- `facade-batched-backward-ignored.log`: 上記 3) の生出力
- `env_info.txt`: 実行環境記入欄（内部ホスト名は書かない。GPU 型番・
  driver／CUDA バージョン・`nvidia-smi` 出力の要約程度に留める）

## 事前登録判定規則

- **bit 同一**: `gemm_batched_fp32_strict` が
  `fandhe_ai_tensor_core::gemm_batched_via_per_batch_gemm_fp32_strict`
  （既定合成実装。オーバーライド導入前の CUDA 既定経路そのもの）と
  全 9 形状で byte 単位完全一致すること（`crates/backend-cuda/tests/
  gemm_batched_parity.rs::gemm_batched_fp32_strict_matches_default_per_batch_composition_bit_exact_on_real_device`）
- **REQ-2 parity**: CPU 参照実装（`CpuBackendOps::gemm_batched`）との
  複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）が全 fail 0
  件であること
- **run-to-run 決定性**: 同一入力で 2 回起動しても bit 同一であること
- 上記いずれも FAIL の場合は本番結線（`ops.rs` の `gemm_batched`／
  `gemm_batched_fp32_strict` オーバーライド）を見直す（イシュー再オープン）
