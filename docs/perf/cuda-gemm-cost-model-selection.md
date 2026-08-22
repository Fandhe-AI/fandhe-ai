# CUDA GEMM L1/L2 帯域コストモデルによる静的タイル選定（#527・C-9b）

イシュー #527「perf(backend-cuda): L1/L2 帯域コストモデルによる静的最良構成選定を実装（sm_121 実測定数）」の実機比較記録テンプレート。
GEMM 性能改善ツリー #479 系・Phase C の C-9b。前段 C-9a（#524・`enumerate_tile_candidates`）が列挙したタイル候補群から、実行時ベンチマークを一切行わない静的コストモデル（`crates/backend-cuda/src/gemm_auto.rs::cost_model` モジュール）で最良 1 件を決定的に選ぶ。

## 状態: 未実測・実機実行待ち

本実装セッションは実機接続情報（`docs/real-hardware-verification-env.local.md`）を持たないため、本イシューの受け入れ基準が要求する「実測ベンチ最良構成と 3 形状中 2 形状以上一致」の検証は実行できない（`docs/perf/sm121-device-attributes.md`・A-2・#482 が同じ理由で「未実測・要実機実行」のまま安全側クローズしている先例、および `docs/perf/cuda-gemm-mma-block-tile.md`・#494 の先例と同型）。

本実装セッションで検証済みの事項:

- `cargo build --workspace`
- `cargo fmt --all -- --check` / `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- `cargo test -p fandhe-ai-backend-cuda`（`gemm_auto.rs::cost_model_tests` の GPU 不要ユニットテスト一式。§2 参照）
- `cargo test --workspace`（回帰確認。全 green）

未検証・実機実行待ちの事項:

- `docs/perf/sm121-device-attributes.md` への L1/L2 帯域実測値の記入（A-2・#482）
- 下記 §1 の 3 形状比較・モデル選定と実測最良構成の照合
- `SM121_MEASURED_BANDWIDTH`（`crates/backend-cuda/src/gemm_auto.rs`）の `Some` 化判断

## 0. 安全側判断（本イシューの受け入れ基準そのものが要求するフォールバック）

- sm_121 実測帯域が存在しない間、コストモデル（`cost_model::estimate_candidate_cost`）は帯域定数を **注入パラメータ** `MeasuredBandwidth` として受け取る純関数として実装されており、`SM121_MEASURED_BANDWIDTH: Option<MeasuredBandwidth> = None` である限り `select_tile_config` はコストモデルを一切評価せず、固定選定テーブル（実測裏付けのある現行本番構成 `MMA_BM=64`/`MMA_BN=128`/`MMA_BK=32`・`MMA_STAGES=3`。#494 実測記録が根拠）へ fail-closed にフォールバックする
- DeepGEMM（`sm90.hpp`）の帯域定数（`l2_bandwidth_per_cycle = min(64*num_sms, 8e6/1300)`・`l1 = 128*num_sms`）は H100 実測由来のマジックナンバーであり、本リポジトリでは一切流用していない
- 本ドキュメントの実機手順は「1 回だけの補正」を許容し、補正後も不一致なら `SM121_MEASURED_BANDWIDTH` を `None` に固定して固定選定テーブル採用を確定する（補正ループ禁止。無限に閾値・定数をいじって基準に合わせにいく経路を作らない）

## 1. 実機手順

前提: CUDA driver + NVRTC 搭載・sm_121（DGX Spark GB10）実機（`docs/real-hardware-verification-env.md` の接続手順）。

```sh
git fetch origin
git checkout perf/527-static-cost-model-tile-selection   # 本イシューの実装ブランチ
cargo test -p fandhe-ai-backend-cuda -- --ignored --nocapture       # select_tile_config_for_device の実機結線検証を含む
```

1. **3 形状の候補構成を実測する**（M=N=K = 4096 / 2048 / 1024。`docs/perf/cuda-gemm-mma-block-tile.md` §4 と同じ「定数差し替え → 再計測」手順で、`enumerate_tile_candidates` が列挙する各候補について `kernels_mma.rs` の `MMA_BM`/`MMA_BN`/`MMA_BK`/`MMA_STAGES` 定数とカーネルソース `#define` を一時的に差し替えて `cargo run -p fandhe-ai-backend-cuda --example gemm_mma_bench --release` を実行し、5 回計測中央値の TFLOPS を記録する）
2. **`docs/perf/sm121-device-attributes.md` へ帯域実測値を記入**し、同表を正として `crates/backend-cuda/src/gemm_auto.rs::cost_model::SM121_MEASURED_BANDWIDTH` を `Some(MeasuredBandwidth { l1_bytes_per_cycle_per_sm: ..., l2_bytes_per_cycle_device: ... })` へ更新する
3. **モデル選定と実測最良構成を照合する**（`select_tile_config`／`select_tile_config_for_device` が返す `TileSelection::candidate()` と、§記録欄の実測最良構成の block_m/block_n/block_k を比較する）。**3 形状中 2 形状以上一致**で採用（受け入れ基準）
4. **不一致の場合**: モデル定数（トラフィック係数・帯域値）の実測補正を **1 回だけ** 行い再判定する。補正はコミット・PR に根拠を明記する。**注意**: `cost_model` モジュールのドキュメンテーションコメント（§トラフィックモデル）が示すとおり、`shape.m`/`shape.n` を割り切るタイル構成の間では L1 トラフィックがタイル構成にほぼ不変（`num_blocks * block_m * block_n ≈ shape.m * shape.n` に近似されるため）であり、`l1_bytes_per_cycle_per_sm` を調整してもランキングをほとんど動かせない。不一致が生じている場合、補正はまず L2 側の係数（`l2_bytes_per_cycle_device`）または wave 効率項の解釈を見直すことを優先する
5. **補正後も不一致の場合**: `SM121_MEASURED_BANDWIDTH` を `None` に戻し、固定選定テーブル採用を確定する。判断を本ドキュメントの §記録欄へ記録して完了する（**補正ループ禁止**。2 回目の補正は行わない）

### 記録欄（実機セッションで埋める）

| 形状（M=N=K） | 実測最良構成（block_m/n/k・stages） | コストモデル選定（block_m/n/k・stages） | 一致 | 補正実施 |
|---------------|--------------------------------------|-------------------------------------------|------|----------|
| 4096 | 未実測 | 未実測 | 未実測 | 未実施 |
| 2048 | 未実測 | 未実測 | 未実測 | 未実施 |
| 1024 | 未実測 | 未実測 | 未実測 | 未実施 |

**採用判断（3 形状中 2 形状以上一致で確定）**: 未実測のため未確定。

## 2. GPU 不要ユニットテストの検証範囲（`crates/backend-cuda/src/gemm_auto.rs::cost_model_tests`）

- (a) 単調性: より大きなタイル（`num_blocks` が小さい構成）ほど総トラフィックが減り、コストモデルがより優先する
- (b) L1/L2 律速側の切り替わり: 片側の帯域を極端に絞ると `max()` 側が支配し選定結果に影響しうる
- (c) wave 効率ペナルティ: 端数 wave を生む構成（`num_blocks` が `num_sms` の倍数から外れる）は wave_factor（`num_waves * num_sms`）が大きくなり不利になる
- (d) 決定性: 同一入力 2 回で同一選定
- (e) オーバーフロー安全性: 巨大形状・極端な帯域パラメータでも `panic!` せず `Result` で応答する
- フォールバック: `measured = None` → 固定選定テーブル（`FixedTable`）が返ること／候補ゼロ件（不整列形状）でもフォールバックが機能すること／`None` の間はコストモデル自体を一切評価しないこと（H100 定数流用の不在の機械検査）

## 3. スコープ境界

本イシューは選定結果を既定の本番 GEMM 経路（`CudaGemmAuto::run_f16`・`gemm_mma.rs::CudaMmaGemm::run_f16` の `MMA_STAGES=3` 固定）へ結線・適用しない。カーネルソース（`kernels_mma.rs` 等）・`tests/parity_nonregression.rs`・tolerance 定数は一切変更していないため、既定経路の実行結果・parity ベースラインへの影響はない（§4 参照）。適用判断は実機実測・補正判定完了後の後続タスクに委ねる。

## 4. §1.2 parity 非後退契約の機械確認

```sh
git diff origin/main -- crates/backend-cuda/tests/parity_nonregression.rs crates/backend-cuda/tests/common/parity_baseline crates/backend-cuda/src/kernels_mma.rs crates/backend-cuda/src/kernels_wmma.rs crates/backend-cuda/src/kernels_wmma_opt.rs
```

無差分であることを確認する（カーネルソース・tolerance 定数・parity fixture を一切変更していないことをコミット前に検査する）。
