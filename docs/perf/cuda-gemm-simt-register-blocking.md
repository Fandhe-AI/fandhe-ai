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
  （実装計画 §3.2 の段階方針）。ncu 自体は DGX Spark GB10 実機実測
  セッション（#6 節）でも未導入のため引き続き未実施。
- **DGX Spark GB10（sm_121）実機での性能実測**: #6 節で実施済み。
  SplitK 撤退是正（PR #1111）後の再実測は §7 節を参照。
- **Metal M4 Max 実機比較**: 本イシューは CUDA バックエンドのみが対象の
  ためスコープ外。

## 6. DGX Spark GB10（sm_121）実機実測（#1031 実機実測セッション）

実機: DGX Spark GB10（compute capability (12, 1) = sm_121）・driver 580.173.02・
CUDA 13.0.88（`nvcc --version` 実測）・rustc 1.97.0。計測時 `nvidia-smi
--query-gpu=utilization.gpu --format=csv,noheader` で 0% を確認済み。commit
`10011cd4f8ef097351c0dc1244eb55c8a021040b`。

`examples/cuda_floor_bench.rs --release`（`launch_tiled_f32` の GPU 実行 + 同期のみを
計測。H2D/D2H を含まない。§4 の RTX 3060 実測と同じ計測境界）の `tiled_f32_tflops`
列を転記する。**本カーネル（register-blocked 版）は既に `kernels.rs::TILED_F32`
本体を置換済みであり、HEAD には非 register-blocked 版（before）が存在しないため、
GB10 実機での before/after 倍率は取得できない**（旧版へ revert しての再計測はスコープ外。
§4 の RTX 3060 実機での倍率〈別機種〉と直接比較しない）。よって以下は after 単独の
絶対値と目標（親イシュー #1031「candle/Burn 比 N=4096 で 2,410 GFLOP/s 超え」）との
比較のみを記録する。

| N | tiled_f32 TFLOPS（GPU 実行のみ・5 回計測中央値） |
|---|---|
| 512  | 4.5640 TFLOPS (q1=4.5764, q3=4.5504) |
| 1024 | 6.7469 TFLOPS (q1=6.7633, q3=6.3289) |
| 2048 | 7.5413 TFLOPS (q1=7.5430, q3=7.5398) |
| 4096 | 7.1008 TFLOPS (q1=7.1060, q3=7.0988) |

**q1 > q3 の見え方について（転記ミスではない）**: 上表の q1 は中央値より大きく q3 は
中央値より小さいという、通常の四分位（q1 < median < q3）と逆の並びに見える。これは
`cuda_floor_bench.rs` が時間ドメイン（秒）の分位点（`q1_secs < median_secs < q3_secs`。
速い試行ほど下位 25%）を TFLOPS へ換算してから出力する仕様のためで、逆数変換により
大小関係が反転する（`examples/cuda_floor_bench.rs` の `TflopsSample` 構造体・同ファイル
`#[cfg(test)]` の `assert!(sample.tflops_from_q1_secs > sample.median)` ／
`assert!(sample.median > sample.tflops_from_q3_secs)` がこの仕様を裏付ける）。転記は
ログ出力どおりであり誤りではない。

**目標達成判定**: N=4096 で 7,100.8 GFLOP/s（7.1008 TFLOPS）を記録し、親イシュー
#1031 の目標値 2,410 GFLOP/s を約 2.95 倍上回る。

参考比較（以下はいずれも限定条件付きであり、7,100.8 GFLOP/s 自体の 2,410 GFLOP/s 超え
という上記判定を覆すものではないが、倍率の解釈には注意が必要）:

- `docs/perf/cuda-gemm-kernel-vs-frameworks-baseline.md` §4 の「tiled f32（基準経路）」
  N=4096 行（1972.3 GFLOP/s。candle 比 約 0.87 倍・burn 比 約 0.67 倍で**下回っていた**）
  と本節の 7,100.8 GFLOP/s を単純に並べると 1972.3 → 7100.8（約 3.60 倍）に見えるが、
  **この 1972.3 GFLOP/s は `cuda-optimized-remeasurement.md`（2026-08-18 計測・
  driver 580.159.03）に由来し、本節の実測（2026-08-31 計測・driver 580.173.02）とは
  別セッションかつ GB10 個体の同一性が未確認のクロスセッション比較**である
  （同ベースラインドキュメント §4「限定条件 7」）。加えて本ファイル §6 冒頭のとおり
  HEAD には非 register-blocked 版が残っておらず revert 再計測もしていないため、
  「register blocking 単独の効果として 3.60 倍」と断定はできない（§6 冒頭の
  「before/after 倍率は取得できない」という記述と整合させ、この 3.60 倍は
  register blocking 単独への帰属ではなく、クロスセッション・別 driver 版数の
  参考比較として扱う）。
- GB10 candle 実測（`cuda-gemm-kernel-vs-frameworks-baseline.md` §3.2・§4 経由。
  N=4096: 2265.1 GFLOP/s）・burn 実測（同 N=4096: 2935.5 GFLOP/s）と比べると
  7,100.8 GFLOP/s はそれぞれ約 3.13 倍・約 2.42 倍上回る。ただしこの比較にも
  同ベースラインドキュメント **限定条件 1**（fandhe-ai 側は `cuda_floor_bench` の
  launch-only 同期＝カーネル完了待ちのみで H2D/D2H・ホスト実体化を含まないのに対し、
  candle/burn の framework-compare 実測は matmul 呼び出し＋ホスト実体化を含む。
  同一境界での比較ではなく fandhe-ai 側に有利な方向のバイアスがある）と
  **限定条件 7**（2026-08-18 の cuda-optimized-remeasurement.md 系列と 2026-08-28 の
  framework-compare 系列は別セッション・GB10 個体未確認のクロスセッション比較。
  本節の 2026-08-31 実測はさらに別セッション）が及ぶため、「突合済み」の値の単純比較
  ではなく、境界差・セッション差が未解消のままの参考比較として扱う。
- 上記の限定条件を踏まえても、7,100.8 GFLOP/s が親イシュー目標 2,410 GFLOP/s を
  約 2.95 倍上回るという本節冒頭の**目標達成判定自体は成立する**（この判定は同一
  セッション内の絶対値比較のみに依拠しており、クロスセッション参考比較には依存しない）。

## 7. SplitK 撤退是正（PR #1111）後の GB10 再実測（イシュー #1136）

**状態**: 実機実測完了（2026-09-03 04:06 JST・DGX Spark GB10・sm_121・GPU アイドル
〈計測直前 `nvidia-smi --query-gpu=utilization.gpu` 0%・`--query-compute-apps` は
常駐サービス〈ComfyUI・Kokoro〉のみで計測競合プロセスなし〉）。

### 7.1 計測環境

- GPU: NVIDIA GB10（sm_121）・driver 580.173.02
- nvcc: `Build cuda_13.0.r13.0/compiler.36424714_0`（CUDA 13.0）
- rustc: 1.97.0（2d8144b78 2026-07-07）
- commit: `1a32082e4b521d7a0bed868db3a3b0a65e2bae9a`（転送前後で `.rev-stamp` 一致確認済み。
  #1133 マージ後・PR #1111 の SplitK 撤退是正〈`caa1bdd`〉を含む）
- 実機個体は `<cuda-node>` 表記（実ホスト名は非公開。`docs/real-hardware-verification-env.md` 参照）

### 7.2 §6（2026-08-31・commit `10011cd`）以降の差分確認

```
git diff 10011cd4f8ef097351c0dc1244eb55c8a021040b..1a32082e4b521d7a0bed868db3a3b0a65e2bae9a \
  --stat -- crates/backend-cuda/src/kernels.rs crates/backend-cuda/src/gemm.rs \
  crates/backend-cuda/src/ops.rs crates/backend-cuda/src/gemm_auto.rs
```

結果: `crates/backend-cuda/src/gemm.rs`（41 insertions, 72 deletions）のみ変更あり。
`kernels.rs`・`ops.rs`・`gemm_auto.rs` は無差分。変更元は PR #1111（`caa1bdd`。
FP32 variant selection の SplitK parity 失敗と DoubleBuffer ヒューリスティックの
性能逆転の修正）。`TILED_F32` 本体（`kernels.rs`）は §6 実測時点から不変であり、
本節の計測は「DoubleBuffer バッファ管理是正後」の到達性能を捉える。

### 7.3 parity 結果（受入基準 A）

`--test-threads=1`・`--release --locked` で 6 テストバイナリを実行。全て
`test result: ok`（複合判定 0 fail。`fandhe_ai_backend_cpu::assert_parity` 厳密ゼロ fail 判定）。

| テストバイナリ | 結果 | テスト数 |
|---|---|---|
| `gemm_tiled` | ok | 6 passed（`tiled_f32_outperforms_naive_at_4096` 含む） |
| `gemm_bias_act_parity` | ok | 2 passed |
| `backend_ops_real_device` | ok | 1 passed |
| `gemm_auto` | ok | 3 passed |
| `gemm_resident_real_device` | ok | 2 passed |
| `cpu_cuda_parity` | ok | 2 passed |

実行コマンド（`T` に上記バイナリ名を代入）:

```
cargo test -p fandhe-ai-backend-cuda --release --locked --test "$T" -- \
  --ignored --nocapture --test-threads=1
```

生ログ: `docs/perf/logs/cuda-simt-remeasurement-1136/parity_*.log`

### 7.4 GFLOPS 結果（受入基準 B）

`examples/cuda_floor_bench.rs`（`launch_tiled_f32` の GPU 実行 + 同期のみ計測。
§6 と同一計測境界）を 5 回反復。各 run の `tiled_f32_tflops`（中央値。warmup 20 /
iters 20）と、run 間中央値（＝本節の代表値）:

| N | run1 | run2 | run3 | run4 | run5 | **run 間中央値** | GFLOP/s |
|---|---|---|---|---|---|---|---|
| 512（参考） | 4.5479 | 4.5640 | 4.5677 | 4.5603 | 4.5640 | **4.5640** (q1=4.5739, q3=4.5603) | 4564.0 |
| 1024 | 6.7324 | 6.7589 | 6.7470 | 6.7497 | 6.7459 | **6.7470** (q1=6.7578, q3=6.7375) | 6747.0 |
| 2048 | 7.5301 | 7.5320 | 7.4819 | 7.4742 | 7.4722 | **7.4819** (q1=7.5315, q3=7.4773) | 7481.9 |
| 4096 | 6.7942 | 6.7588 | 6.7291 | 6.7485 | 6.7341 | **6.7485** (q1=6.7517, q3=6.7417) | 6748.5 |

実行コマンド: `cargo run -p fandhe-ai-backend-cuda --example cuda_floor_bench --release --locked`
（PyTorch 参照 env 未設定のため `f32_best_over_pytorch` 等は `n/a`／別実測値・本イシューの対象外）。

生ログ: `docs/perf/logs/cuda-simt-remeasurement-1136/floor_bench_run{1..5}.log`

**§6（2026-08-31・commit `10011cd`）比**:

| N | §6（DoubleBuffer 是正前） | 本節（是正後） | 倍率 |
|---|---|---|---|
| 512 | 4.5640 | 4.5640 | 1.000x（差分なし） |
| 1024 | 6.7469 | 6.7470 | 1.000x（差分なし） |
| 2048 | 7.5413 | 7.4819 | 0.9921x（-0.79%） |
| 4096 | 7.1008 | 6.7485 | 0.9504x（**-4.96%**） |

N=1024 は実質不変、N=2048 は誤差範囲内の軽微な低下。**N=4096 は -4.96% で
「5% 超の劣化」の閾値未満だが僅差**であり、回帰の疑いとして記録する（原因調査・
`kernels.rs::TILED_F32` 本体・DoubleBuffer 閾値のチューニングは本イシューのスコープ外。
§7.2 のとおり `TILED_F32` 本体は §6 から不変のため、この差は同一カーネルの
run-to-run 変動〈driver・熱・スケジューリング等〉である可能性が高く、PR #1111 の
DoubleBuffer バッファ管理是正自体が N=4096 の性能を悪化させたと断定する根拠はない
〈§6 と本節は別セッション・同一 GB10 個体だが連続実行ではない〉）。

### 7.5 参考比較（限定条件付き）

- 親イシュー #1031 目標（candle/Burn 比 N=4096 で 2,410 GFLOP/s 超え）との比較:
  本節 N=4096 は 6,748.5 GFLOP/s で目標を約 2.80 倍上回る（§6 の 2.95 倍からわずかに縮小）。
- v0.6.0 framework-compare 再計測（PR #1127・2026-09-02。
  `scripts/bench/framework-compare/results/summary.md` 環境 10）の `gemm/CUDA`
  fresh N=4096 1928.7 GFLOP/s・reuse N=4096 1931.8 GFLOP/s と比べ、本節の
  6,748.5 GFLOP/s は約 3.5 倍高い。ただしこれは §6 で既述のとおり**計測境界が異なる**
  （本節は `launch_tiled_f32` の GPU 実行 + 同期のみ／launch-only。framework-compare は
  tape 構築・ホスト実体化を含む）ため、fandhe-ai 側に有利な方向のバイアスを含む
  参考比較であり、単純な性能向上の根拠としては扱わない。

### 7.6 opt-in 診断経路（補助・PR #1111 §1b の申し送り。受入基準外）

```
cargo test -p fandhe-ai-backend-cuda --release --locked --features internal-diagnostics \
  --test gemm_f32_variants -- --ignored --nocapture --test-threads=1
cargo test -p fandhe-ai-backend-cuda --release --locked --features internal-diagnostics \
  --test cpu_cuda_tiled_pipeline_parity -- --ignored --nocapture --test-threads=1
```

結果: `gemm_f32_variants`（`run_f32_matches_cpu_reference_across_variant_shapes`・
`split_k_forced_execution_is_bit_deterministic_and_reproduces_gb10_fail` の 2 テストとも
`ok`）・`cpu_cuda_tiled_pipeline_parity`（9 テスト全て `ok`）。いずれもテスト関数自体の
Rust 側判定は green（`split_k_forced_…` はテスト名のとおり「GB10 で複合判定 FAIL を
決定的に再現する」ことを検証する設計のテストであり、テスト結果 `ok` はその再現を
確認できたことを意味する。テスト内部の複合判定そのものの FAIL/PASS 詳細は本節の
スコープ外のため深追いせず、生ログを記録するに留める）。生ログ:
`docs/perf/logs/cuda-simt-remeasurement-1136/parity_gemm_f32_variants.log`。
`cuda-gemm-f32-variant-selection.md` §1b の「記録欄」は別途参照。

### 7.7 不変条件の宣言

本節作成にあたり `crates/`・`Cargo.toml`・`Cargo.lock`・tolerance 定数
（`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`）・parity baseline
（`tests/common/parity_baseline.rs`）はいずれも変更していない
（`git diff origin/main -- crates/ Cargo.toml Cargo.lock` が無差分であることを確認済み）。

### 7.8 未実施・申し送り

- ncu（nsight-compute）は本ノードに未導入のため引き続き未実施（§5 と同じ）
- N=4096 の -4.96% 差分は「回帰の疑い」として記録するのみで、原因調査・
  DoubleBuffer 閾値補正（#1100 §1b・#1035 手順 4）は行わない（ユーザー承認事項）
- cp.async パイプライン（#1033）・スウィズル（#1034）の本番結線は本節の対象外

## 8. 関連

- `docs/perf/cuda-gemm-kernel-improvement-policy.md`（本イシューの動機）
- `docs/kernel-fusion.md` §2.2（`TILED_F32`/`TILED_BIAS_ACT_F32` bit 完全
  一致契約）
- `docs/perf/cuda-async-sync-removal-rtx3060.md` §1（RTX 3060 での NVRTC
  一時提供手順の先例）
- イシュー #493（`kernels_mma.rs` レジスタブロッキング先例）・#498
  （`kernels_wmma_opt.rs` smem パディング先例）
