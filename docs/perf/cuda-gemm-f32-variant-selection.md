# CUDA f32 GEMM の形状別カーネル選択（simple / double-buffer / split-K。#1035）

イシュー #1035「perf(backend-cuda): 形状別カーネル選択を simple / double-buffer / split-K のヒューリスティックへ拡張する」の実機比較記録テンプレート。
親ツリー #1029（GEMM カーネルの candle 超え）・#1007 の Phase 2。cuBLAS フル FP32（candle CUDA）を上回ることが目標で、本イシューは特に小サイズ（N=256〜512。candle 実測 423〜1,109 GFLOP/s @GB10、`docs/perf/cuda-gemm-kernel-vs-frameworks-baseline.md` §3.2）で SM が遊ぶ形状への対処を担う。

## 状態: 未実測・実機実行待ち

本実装セッションは実機接続情報（`docs/real-hardware-verification-env.local.md`）を持たないため、本イシューの受け入れ条件が要求する「(a) 全 N で candle 以上」の検証は実行できない（`docs/perf/cuda-gemm-cost-model-selection.md`・#527 が同じ理由で「未実測・要実機実行」のまま安全側クローズしている先例と同型）。受け入れ条件 (b)「選択ロジックのユニットテスト」は本ランで充足済み（§2）。

本実装セッションで検証済みの事項:

- `cargo build --workspace`
- `cargo fmt --all -- --check` / `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- `cargo test -p fandhe-ai-backend-cuda`（`gemm_variant::tests`・`kernels_gemm_variants::tests` の GPU 不要ユニットテスト一式。§2 参照）
- `cargo test -p fandhe-ai-backend-cuda --features internal-diagnostics`（`gemm_f32_variants.rs` の環境適応スモーク 2 件。`#[ignore]` 2 件は未実行）
- `cargo test --workspace` / `cargo test --workspace --all-features`（回帰確認。全 green）
- `git diff origin/main -- crates/backend-cuda/tests/parity_nonregression.rs crates/backend-cuda/tests/common/parity_baseline crates/backend-cuda/src/kernels.rs crates/backend-cuda/src/kernels_mma.rs crates/backend-cuda/src/kernels_wmma.rs crates/backend-cuda/src/kernels_wmma_opt.rs` が無差分（既存カーネル・fixture・tolerance を一切変更していないことの機械確認）

未検証・実機実行待ちの事項:

- 下記 §1 の実機 A/B（N=256〜4096・K 支配的非正方形状 vs candle）
- `#[ignore]` テスト（`tests/gemm_f32_variants.rs`）による複合判定・決定性・境界形状の実機検証
- 暫定閾値（`SPLITK_MIN_K`・`SPLITK_MAX_SPLITS`・`SPLITK_PARTIAL_MAX_BYTES`・`DOUBLE_BUFFER_MIN_K`）の実測補正
- 本番既定経路（`gemm_auto.rs::CudaGemmAuto::run_f32`・`CudaGemm::new`・`run_tiled_f32`）への結線判断（ユーザー承認必須）

## 0. 安全側判断（opt-in 診断経路に留める理由）

- 本ランは NVRTC 実行不能環境（CUDA toolkit 非搭載）のため、DoubleBuffer／SplitK カーネルの数値検証・性能 A/B がすべて実機待ちになる。よって #1034（PR #740→#758 差し戻しの教訓）と同じ判断で、**未検証カーネルを本番既定経路へ結線しない**。選択ヒューリスティック（`gemm_variant.rs`）とカーネル（`kernels_gemm_variants.rs`）・実行経路（`gemm_variant_selection.rs`）はすべて `internal-diagnostics` feature（既定 off）限定の opt-in とし、本番既定コンストラクタ（`CudaGemm::new`）・`run_tiled_f32`・`kernels::TILED_F32` は一切変更していない
- 選択ヒューリスティックの閾値定数（`SPLITK_MIN_K`=1024・`SPLITK_MIN_K_PER_SPLIT`=32・`SPLITK_MAX_SPLITS`=32・`SPLITK_PARTIAL_MAX_BYTES`=256 MiB・`DOUBLE_BUFFER_MIN_K`=64）は実機実測前の**暫定値**であり、`cuda-gemm-cost-model-selection.md`（#527）と同じ方針で補正は 1 回限りとし、実測を追わない補正ループは行わない

## 1. 実機手順

前提: CUDA driver + NVRTC 搭載実機（DGX Spark GB10 等。`docs/real-hardware-verification-env.md` の接続手順）。

```sh
git fetch origin
git checkout perf/1035-f32-variant-selection   # 本イシューの実装ブランチ
cargo test -p fandhe-ai-backend-cuda --features internal-diagnostics -- --ignored --nocapture
cargo run -p fandhe-ai-backend-cuda --example gemm_f32_variant_bench --release --features internal-diagnostics
```

1. **`#[ignore]` テストを実行**し、複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）・SplitK 決定性（2 回実行 bit 一致）・境界形状（33/65/1000 等の非整列サイズ・K 支配的非正方形状）がすべて green であることを確認する
2. **A/B ベンチを実行**し、下記記録欄へ N=256〜4096（アラインメント済み正方）・K 支配的非正方形状（split-K 対象）ごとの `variant`（選択された変種）・`base_tflops`（`TILED_F32`）・`selected_tflops`（選択された変種）・candle 実測値（`docs/perf/cuda-gemm-kernel-vs-frameworks-baseline.md`）との比を記入する
3. **受け入れ条件 (a)「全 N で candle 以上」を判定する**。満たさない場合は暫定閾値の補正を **1 回だけ** 行い再計測する（補正はコミット・PR に根拠を明記する。補正ループ禁止）
4. **本番既定経路への結線判断**: 受け入れ条件を満たし、かつ数値検証（`#[ignore]` テスト）が全 green であることを確認したうえで、ユーザー承認を得てから後続 Issue として本番結線（`gemm_auto.rs::CudaGemmAuto::run_f32` からの呼び出し）を実施する。本 PR ではこの結線を行わない

### 記録欄（実機セッションで埋める）

| 形状（M, N, K） | 選択された変種 | base_tflops（`TILED_F32`） | selected_tflops | candle 実測（GFLOP/s） | candle 比 |
|-----------------|----------------|------------------------------|-----------------|------------------------|-----------|
| 256, 256, 256 | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| 512, 512, 512 | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| 1024, 1024, 1024 | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| 4096, 4096, 4096 | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| 128, 128, 8192 | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| 256, 256, 16384 | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |

**採用判断（全 N で candle 以上で確定）**: 未実測のため未確定。

## 2. GPU 不要ユニットテストの検証範囲

`crates/backend-cuda/src/gemm_variant.rs::tests`（`select_f32_gemm_variant`・`derive_split_count`・`validate_split_k_launch`）:

- num_sms 判定不能（`None`）・境界条件（0 次元・`num_sms=0`）では常に `Simple`
- SplitK: grid が SM を埋められず K 支配的な形状で選ばれる／候補利用不能・K 閾値未満・K 非支配では選ばれない／部分和バッファが cap を超える場合は `Simple` へ fail-closed で降格する
- DoubleBuffer: grid が SM を十分埋めるアラインメント済み大形状で選ばれる／非整列・候補利用不能・K 閾値未満では選ばれない
- `derive_split_count` が常に 2 冪（または 1）かつ `[1, SPLITK_MAX_SPLITS]` の範囲内であること・`num_blocks`/`num_sms` が 0 の場合は 1 を返すこと
- 選択の決定性（同一入力 2 回で同一結果）・巨大形状（`i32::MAX` 近傍）での非 panic（`u64` オーバーフロー安全性）
- `validate_split_k_launch` の範囲外 `num_splits` 拒否・cap 超過拒否・正常系受理

`crates/backend-cuda/src/kernels_gemm_variants.rs::tests`（カーネルソース構造検査。`kernels_rmsnorm.rs` の split-K テスト群と同型）:

- split-K 部分和カーネルが `c_partial` へ無条件に 1 回だけ書くこと（末尾要素ブロックの扱い）
- split-K の 2 カーネルがいずれも atomics を使わないこと（決定的書き込み）
- 縮約カーネルが `c` へ 1 回だけ書き、`c_partial` へは書き戻さないこと（第 3 パスを作らない契約）
- 縮約の反復順序が `s` 昇順の固定順序であること（決定性の根拠）
- double-buffer カーネルの smem が 2 面であること・C への書き込み時の手動境界チェック（REQ-8）・タイルロードの三項ガードを維持していること
- split-K 部分和カーネルのタイルロードも三項ガードを維持していること

`crates/backend-cuda/tests/gemm_f32_variants.rs`（`internal-diagnostics` feature 限定）:

- 環境適応スモーク（非 ignore）: `CudaGemmF32VariantSelection::new` が CUDA 非搭載環境で panic せず型付きエラーを返すこと・`selected_variant` が panic しないこと
- `#[ignore]`（実機必須。本ランは未実行）: CPU 参照実装との複合判定（アラインメント済み大形状・K 支配的非正方・非整列・境界サイズを網羅）・SplitK の bit 決定性

## 3. スコープ外・追跡事項（`out-of-scope-tracking.md` 準拠）

- 本番既定経路（`CudaGemm::new`・`run_tiled_f32`・`CudaGemmAuto::run_f32`）への選択ヒューリスティック結線は、実機実測（A/B・複合判定・parity 非後退）とユーザー承認後の後続作業とする
- 暫定閾値（`SPLITK_MIN_K`・occupancy 係数・バッファ cap）の実機補正は上記結線判断とセットで実施する（補正は 1 回限り・補正ループ禁止）
- #1033（cp.async 多段パイプライン）の DoubleBuffer 候補への差し替え統合は #1033 マージ後の後続判断とする（`gemm_variant_selection.rs::CudaGemmF32VariantSelection` の候補は `Option` スロットで保持しており差し替え可能な構造）
- resident 経路（`launch_tiled_f32_resident`）への変種適用は本 PR のスコープ外とし、本番既定経路への結線判断と同時に検討する
