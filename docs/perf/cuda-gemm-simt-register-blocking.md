# CUDA FP32 SIMT GEMM のレジスタブロッキング拡大・smem パディング（イシュー #1032）

## 1. 背景・設計根拠

`docs/perf/cuda-gemm-kernel-improvement-policy.md` §1 が指摘するとおり、公開 API
既定の GEMM カーネル（`CudaGemm::run_tiled_f32`。`crates/backend-cuda/src/
kernels.rs::TILED_F32`）は 32x32 共有メモリタイル・**1 スレッド 1 出力**の
素朴な構成のままだった。共有メモリからの 1 ロードにつき積和 1 回しか行わず、
スレッドあたり算術強度が低いままメモリ律速になっていた（#928 ベースライン）。

本イシューは Tensor Core 経路で確立済みのレジスタブロッキング先例（#493
`kernels_mma.rs` の 2x2 warp タイル）・smem パディング先例（#498
`kernels_wmma_opt.rs` の非 2 冪パディング）を FP32 SIMT 経路へ水平展開した。

## 2. 実装

### 2.1 構成（`crates/backend-cuda/src/kernels.rs`）

| 定数 | 値 | 役割 |
|------|----|------|
| `TILED_F32_BM` | 64 | ブロックタイル M 一辺 |
| `TILED_F32_BN` | 64 | ブロックタイル N 一辺 |
| `TILED_F32_BK` | 16 | ブロックタイル K 一辺（1 反復の smem ロード幅） |
| `TILED_F32_TM` | 4 | 1 スレッドが担当する出力タイルの M 方向要素数 |
| `TILED_F32_TN` | 4 | 1 スレッドが担当する出力タイルの N 方向要素数 |
| `TILED_F32_PAD` | 4 | smem タイル（`as_tile`／`bs_tile`）のパディング要素数 |
| `TILED_F32_THREADS_X` | 16 (`BN/TN`) | スレッドブロック x 方向スレッド数 |
| `TILED_F32_THREADS_Y` | 16 (`BM/TM`) | スレッドブロック y 方向スレッド数 |

1 スレッドが `TM x TN`（4x4=16）出力を担当し、共有メモリから読んだ 1 要素を
16 通りの積和で再利用する（旧実装比 16 倍の算術強度。siboehm 系 2D
register-blocked SGEMM と同型の構成）。A タイルは転置格納
（`as_tile[kk][mm]`）し、積和ループ内アクセスが M 方向へ連続するようにする。
smem パディング（+4 要素＝16 byte）は `kernels_wmma_opt.rs` の非 2 冪
パディング先例と同方針。**XOR swizzle は本イシューでは導入せず**、ncu 実測で
コンフリクト残存が確認された場合のみ後続で検討する（実装計画 §3.2 の段階
方針）。

`TILED_BIAS_ACT_F32`（bias/act 融合カーネル）は `TILED_F32` と**同一構造の
アキュムレーション**へ同時に書き換え、bit 完全一致契約（`docs/
kernel-fusion.md` §2.2）を維持した。

### 2.2 ホスト側（`crates/backend-cuda/src/gemm.rs`）

- `TILED_F32_BLOCK_DIM`（16x16=256 スレッド）・`tiled_f32_launch_config`
  （タイル一辺 `BM`/`BN` 基準の `div_ceil` グリッド）を新設し、f32 tiled 系
  6 箇所の launch site（`run_tiled_f32`〈`run_f32_kernel` 経由〉・
  `run_tiled_bias_act_f32`・`launch_tiled_f32`・`launch_tiled_bias_act_f32`・
  `launch_tiled_bias_act_f32_resident`・`launch_tiled_f32_resident`）を
  切り替えた。
- `run_f32_kernel` は `block_dim: (u32,u32,u32)` 引数を `cfg: LaunchConfig`
  へ変更し、呼び出し元（naive／tiled）がそれぞれの launch config を構築する
  形へ一般化した（naive は従来どおり `launch_config(m,n,NAIVE_BLOCK_DIM)`）。
- `validate_tiled_k_bound`（`TILE`=32 基準）は `TILED_F32`（`BK`=16 基準）に
  対しても安全側（より厳しい上限）であるため据え置き、根拠をコメントへ
  追記した（新規関数は追加していない）。
- f16 tiled 系（`TILED_F16`・`TILED_BLOCK_DIM`）は本イシューのスコープ外
  として無変更。

## 3. 検証

- `cargo fmt --all -- --check` / `cargo clippy --workspace --all-targets
  --all-features -- -D warnings` / `cargo test -p fandhe-ai-backend-cuda`
  （環境適応型 + 静的テスト）: いずれも green。
- 新規静的テスト（`kernels.rs::tests`）: `tiled_f32_constants_match_kernel_
  source_defines`（Rust 定数 ⇔ `#define` 突合）・
  `tiled_f32_constants_satisfy_thread_and_tile_invariants`（`BM/TM`・
  `BN/TN`・スレッド数 256・ロード要素数の割り切り検査）。
- 実機（RTX 3060、driver あり・NVRTC は pip wheel からセッション一時提供。
  `docs/perf/cuda-async-sync-removal-rtx3060.md` §1 と同じ手順）での数値
  一致・parity テスト（`--ignored --nocapture`）: 全 green。
  - `tests/gemm_tiled.rs::tiled_f32_matches_cpu_reference_across_shapes`
    （新規境界ケース **65x63x17**〈BM/BN=64・BK=16 いずれも非整数倍〉・
    **64x64x16**〈num_tiles=1・レジスタタイル境界ぴったり〉・
    **96x160x48**〈BM/BN/BK いずれの整数倍でもある複数タイルケース〉を
    含む）
  - `tests/gemm_tiled.rs::tiled_f32_matches_cpu_reference_k_stress`・
    `tiled_f32_zero_k_returns_all_zero`・
    `tiled_f32_zero_dim_shape_returns_empty_without_launch`・
    `tiled_f16_runs_and_returns_expected_shape`・
    `tiled_f32_outperforms_naive_at_4096`
  - `tests/gemm_bias_act_parity.rs::gemm_bias_act_matches_cpu_across_shapes`
    （新規境界ケース 65x63x17・96x160x48 を追加。`TILED_F32`/
    `TILED_BIAS_ACT_F32` の bit 完全一致契約を境界形状でも確認）・
    `elementwise_matches_cpu_across_ops`
  - `tests/cpu_cuda_parity.rs::naive_f32_matches_reference_across_shapes`・
    `naive_f32_k4096_stress_poc_v2_5`（複合判定: 相対誤差 1e-3 未満 または
    絶対誤差 1e-5 未満。`fandhe_ai_backend_cpu::parity` を唯一の参照とし複製
    せず）

## 4. RTX 3060 参考実測（GPU 実行のみ・5 回計測中央値）

`examples/cuda_floor_bench.rs`（`launch_tiled_f32` の GPU 実行 + 同期のみを
計測。H2D/D2H を含まない）で before/after を計測した。**RTX 3060 は
REQ-8 判定機（DGX Spark GB10）と異なる実機のため、以下は参考値であり
REQ-8 段階的下限の判定には用いない。**

| N | before（旧 1 スレッド 1 出力・32x32 タイル） | after（本イシュー・レジスタブロッキング） | 倍率 |
|---|---|---|---|
| 512  | 0.8957 TFLOPS (q1=0.8976, q3=0.8927) | 2.2174 TFLOPS (q1=2.2302, q3=2.2089) | 2.48x |
| 1024 | 0.9134 TFLOPS (q1=0.9155, q3=0.9119) | 2.9026 TFLOPS (q1=2.9075, q3=2.8875) | 3.18x |
| 2048 | 1.0018 TFLOPS (q1=1.0025, q3=0.9933) | 3.4300 TFLOPS (q1=3.4412, q3=3.4252) | 3.42x |
| 4096 | 1.0090 TFLOPS (q1=1.0090, q3=1.0087) | 3.4831 TFLOPS (q1=3.4923, q3=3.4238) | 3.45x |

実測環境: RTX 3060（driver 595.71.05・CUDA 13.2）、NVRTC は
`nvidia-cuda-nvrtc-cu13` pip wheel からセッション一時提供
（`LD_LIBRARY_PATH`/`CUDA_INCLUDE_PATH` 指定のみで本体依存構成は不変）。
compute_capability=(8, 6)（sm_86）。`tiled_f32_outperforms_naive_at_4096`
テストの判定（naive 比 1.1 倍以上）も同実機で green。

## 5. 未実施として残る検証（後続セッション）

- **ncu によるバンクコンフリクト減少確認**: 本ランでは `nsight-compute`
  未導入環境のため未実施。XOR swizzle 導入要否の判断はこの実測後に行う
  （実装計画 §3.2 の段階方針）。
- **DGX Spark GB10（sm_121）実機での性能実測**: REQ-8 段階的下限の判定・
  親イシュー #1031（candle/Burn 比 N=4096 で 2,410 GFLOP/s 超え目標）の
  達成確認は DGX Spark セッションで実施する。
- **Metal M4 Max 実機比較**: 本イシューは CUDA バックエンドのみが対象の
  ためスコープ外。

## 6. 関連

- `docs/perf/cuda-gemm-kernel-improvement-policy.md`（本イシューの動機）
- `docs/kernel-fusion.md` §2.2（`TILED_F32`/`TILED_BIAS_ACT_F32` bit 完全
  一致契約）
- `docs/perf/cuda-async-sync-removal-rtx3060.md` §1（RTX 3060 での NVRTC
  一時提供手順の先例）
- イシュー #493（`kernels_mma.rs` レジスタブロッキング先例）・#498
  （`kernels_wmma_opt.rs` smem パディング先例）
