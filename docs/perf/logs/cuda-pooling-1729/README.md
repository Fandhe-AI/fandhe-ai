# CUDA Pooling（MaxPool／AvgPool／AdaptiveAvgPool）実機実測ログ置き場

イシュー #1729（`docs/pooling-ops-design.md` §15「#1729（CUDA 実装）」）の
実機（DGX Spark GB10 等）実測ログを保存する場所。`cuda-conv2d-1766/README.md`
と同型。本エージェント実行環境に DGX Spark GB10 実機への到達手段がないため
本 PR 時点ではログは未生成（記入欄のみ）。

## 実行コマンド

```sh
# 属性なし単体テスト（正しさ・shape 検査・環境適応スモークを含む。実機不要）
cargo test -p fandhe-ai-backend-cuda --lib pooling -- --nocapture

# 実機 #[ignore] テスト（値・索引 bit 完全一致オラクルは pooling_model.rs）
cargo test -p fandhe-ai-backend-cuda --release --lib \
    pooling::pooling_real_device_tests -- --ignored --nocapture
```

## 保存すべきファイル一覧

- 実機 `#[ignore]` テスト全件の実行ログ（`run-to-run` bit 同一確認のため
  2 回連続実行したログを両方保存する）
- `env_info.txt`（内部ホスト名は含めない。`docs/real-hardware-verification-env.md`
  参照）

## 事前登録判定規則

- CPU 参照実装（未実装。#1728 マージ後）または `pooling_model.rs` の
  ホストモデルと、値・索引とも byte 単位で完全一致すること（NaN はクラス
  一致）。
- run-to-run で bit 完全同一であること（`max_pool2d_run_to_run_bit_identical`）。
- FAIL 時は `CudaPooling::run_*` の実装（特に走査順・更新条件）を見直す。

## 実測記録（2026-09-16・DGX Spark GB10・転送元コミット `3e43bbd0`）

実測ログは `docs/perf/logs/cuda-realdevice-phase2-2026-09-16/`（`backend-cuda_*.log`・`facade_*.log`・`env_info.txt`）に収めた。結果一覧と判定は同ディレクトリ README §2 と design doc の該当節を参照。
