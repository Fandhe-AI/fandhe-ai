# CUDA GEMM 最終 wave 限定 Stream-K（固定順序 fixup）: 設計・GB10 実測記入欄

イシュー #1358（親 #1357・祖 #1341「CUDA GEMM 構造再設計」・承認元 #1338）。
`docs/cuda-streamk-decision.md`（#812）が保留した Stream-K を、fixup 加算順序の非決定性懸念を設計で
解消したうえで opt-in（`internal-diagnostics` feature 限定・本番結線なし）で実装した記録。GB10 実機
実測・純カーネル時間比較・本番結線可否判断は兄弟イシュー #1359 が担い、その結果を本ドキュメント §6 へ
追記する。

## 1. 背景

- #1347 が persistent タイルキュー版（K 分割なし）を GB10 実機で REJECT と確定した（N=1024・
  `--blocks-per-sm auto` で 64×64=1.0063 倍・128×64=1.0182 倍。事前判定基準 ≥1.05 未達）。末尾（最終
  wave）の遊休 CTA が丸ごと 1 タイルの K 反復を担う限り、persistent 化だけでは末尾時間を縮められない
  ことが実測で示された。
- 本イシューは、末尾のタイルだけ K 反復を全 CTA へ平坦配布する Stream-K を追加することで、末尾時間を
  `Q/nk` タイル時間（`Q` は SK 単位の平坦 K タイル幅・`nk` はタイルあたりの K タイル数）へ縮める候補を
  試す。

## 2. 設計（配布計画・カーネル）

配布計画の記号・導出式・机上値の表は
`crates/backend-cuda/src/gemm.rs::streamk_plan`（GPU 不要の純関数。ドキュメンテーションコメントに
導出式を記載）を単一の真実源とする（本ドキュメントでは重複記載しない）。要点のみ記す:

- 出力タイル総数 `T`・grid 容量 `G`（`num_sms * blocks_per_sm`。`T` へ頭打ちしない生値）・K タイル数
  `nk` から、先頭 `F = T - R` 個を「full タイル」（1 CTA が K 全体を担当。従来どおり）、末尾 `R = T mod
  G` 個を「残タイル」とする。残タイルの K タイルを平坦化した長さ `R*nk` の列を幅 `Q = ceil(R*nk/G)` の
  `U = ceil(R*nk/Q)` 個の「SK 単位」へ連続配布する。
- `R == 0`（端数タイルなし）または `Q >= nk`（分割がタイル境界と完全整列し split-K として無意味）の
  場合は非活性（persistent 相当の動作。出力は persistent 版と bit 同一）。
- 部分和は `slot(r, c) = r * max_contributors + c`（`r` は残タイル番号・`c` は寄与順序）で一意なスロット
  へ書く。fixup は `c` 昇順の固定順序で逐次加算する。
- カーネルソース: `crates/backend-cuda/src/kernels_tiled_pipeline.rs`
  （`TP_SK_KERNEL_PREFIX`／`TP_SK_TILE_CORE`／`TP_SK_KERNEL_SUFFIX`／`TP_SK_FIXUP_KERNEL`）。
  `TP_SK_TILE_CORE` は既存 `TP_TILE_CORE`（非 persistent・persistent 版共有のタイル内計算）から派生し、
  `LOAD_A_STAGE`／`LOAD_B_STAGE` マクロ本文・K 内積 `fmaf` ループ本文は（インデントを除き）文字列として
  同一であることを静的テスト（`tiled_pipeline_streamk_tile_core_shares_load_and_fma_fragments_with_tile_core`）
  で機械検査している。full タイル単位（`kt_begin == 0`）はこの共有により非 Stream-K 版と完全に同一の
  命令列になる。
- ホスト API: `crate::gemm::CudaGemm::compile_tiled_pipeline_streamk_variant`／
  `launch_tiled_pipeline_streamk_f32`／`run_tiled_pipeline_streamk_f32`（いずれも
  `internal-diagnostics` feature 限定）。64×64 タイル固定（128×64 版は対象外）。

## 3. 決定性の根拠

`kernels_tiled_pipeline.rs::TP_SK_FIXUP_KERNEL` ドキュメンテーションコメント「決定性の根拠」節を正とする
（本ドキュメントでは重複記載しない）。要点: (1) full タイルの `c` 書き込みは 1 要素 1 単位、(2) 部分和
スロットは `(r, c)` ごとに一意な書き手、(3) fixup は寄与者昇順の固定順序で逐次加算、(4) `atomicAdd` は
スケジューリング用 `unsigned int` カウンタ 1 箇所のみ（GEMM の数値蓄積には触れない）、(5) Stream-K
カーネルの `c` 書き込み領域（full タイル）と fixup の書き込み領域（残タイル）は互いに素で同一ストリーム
順序（Stream-K カーネル → fixup）で実行される。

**残タイルの値は非 Stream-K 版と bit 同一ではない**（K 連鎖の分割による丸め差。#1100
`splitk_reorder_error_host_model.rs` が示すとおり真値ゼロ近傍で REQ-2 複合判定 fail が出うる）。本
イシューでは統計出力（`fail_count`／`max_abs_diff`／`max_rel_err`）に留め、合否判定・baseline 行追加の
要否は #1359 が判断する。tolerance 定数は変更しない。

配布計画の正しさ（全 `(r, kt)` の網羅性・`(r, c)` スロットの一意性・fixup 側 `C(r)` 式との整合）は GPU
不要のホストシミュレータテスト（`gemm.rs::tests::streamk_plan_host_simulator_covers_units_exactly_once`。
`T ∈ {1,2,3,47,48,49,143,144,145,256,300}`・`G ∈ {1,2,3,47,48,144,200}`・`nk ∈
{0,1,2,3,15,16,63,64,128}` の全組み合わせ）で機械検証済み（`cargo test -p fandhe-ai-backend-cuda --lib`
で実行可能。CUDA 実機不要）。

## 4. opt-in API（`internal-diagnostics` feature 限定）

```rust
let device = CudaDevice::new(0)?;
let mut func = CudaGemm::compile_tiled_pipeline_streamk_variant(&device, /* stages */ 3, /* blocks_per_sm */ None)?;
let gemm = CudaGemm::new(&device)?;
let (c, plan) = gemm.run_tiled_pipeline_streamk_f32(&mut func, &a, &b, m, n, k)?;
// plan.is_active() が false なら分割は発生していない（persistent 相当の動作）。
```

本番既定経路（`CudaGemm::new`・`select_tiled_f32_kernel`）は不変。

## 5. #1359 向け実行コマンド・判定基準の事前宣言

GB10（DGX Spark GB10）実機での実行コマンド:

```sh
cargo test -p fandhe-ai-backend-cuda --release --locked --features internal-diagnostics \
  --test cpu_cuda_tiled_pipeline_streamk_parity -- --ignored --nocapture --test-threads=1
cargo test -p fandhe-ai-backend-cuda --release --locked --features internal-diagnostics \
  --test cpu_cuda_tiled_pipeline_persistent_parity -- --ignored --nocapture --test-threads=1   # 既存回帰の非後退
cargo run -p fandhe-ai-backend-cuda --release --locked --features internal-diagnostics \
  --example gemm_tiled_pipeline_persistent_bench -- --sizes 1024,2048 --tile 64x64 \
  --blocks-per-sm auto --streamk on   # 5 回実行し中央値を取る
```

**判定基準（実測前の事前宣言。実測後に動かさない）**:

- **ゲート A（決定性・正確性）**: `cpu_cuda_tiled_pipeline_streamk_parity.rs` の `#[ignore]` テスト全て
  PASS（`streamk_repeated_launch_is_deterministic`〈AC 本体〉・
  `streamk_full_tiles_match_non_persistent_bit_exact`・`streamk_inactive_matches_non_streamk_bit_exact`
  ほか）。
- **ゲート B（CPU 参照実装との複合判定）**: `streamk_full_tiles_match_non_persistent_bit_exact` が出力
  する残タイルの複合判定統計（`fail_count`／`max_abs_diff`／`max_rel_err`）を記録する。`fail_count ==
  0` なら green。`fail_count > 0` の場合は baseline 行追加（`ParityBaseline`。ユーザー承認必須）の要否を
  #1359 内で判断する。tolerance 定数自体は変更しない。
- **ゲート C（性能。5 回実行中央値）**: `--blocks-per-sm auto` で N=1024 の `streamk_over_pipeline3 ≥
  1.05`・N=2048 の `streamk_over_pipeline3 ≥ 1.00`（#1347 と同じ閾値。N=1024 の Stream-K 分割規模が
  N=2048 より大きい〈§5 末尾の理論上限参照〉ため異なる閾値を設定する判断も #1347 を踏襲する）。

### fixup 固定費の事前見積り

部分和トラフィック（fixup の読み取り＋Stream-K カーネルの書き込み）は `remainder_tiles *
max_contributors * TP_BM * TP_BN * 4 bytes` の往復。N=1024・G=144（machine plan 例。`streamk_plan`
ドキュメンテーションコメント §3.1 表）: `remainder_tiles=112, max_contributors=3` →
`112*3*4096*4 = 5,505,024 bytes ≈ 5.25 MiB` の往復（書き＋読み）。

### 末尾 wave 短縮の理論上限

N=1024・G=144: 末尾 1 wave（tail effect の対象。#1347 の wave quantization 議論と同じ枠組み）が `Q/nk
= 50/64 ≈ 0.781` タイル時間へ縮む。全体は `ceil(256/144) = 2` wave 弱に対し、末尾 wave 短縮分は概ね
10% 前後の改善余地（fixup オーバーヘッドを含まない理論値）。N=2048・G=144: `remainder_tiles=16,
q=15, nk=128` → 末尾 wave が `15/128 ≈ 0.117` タイル時間へ縮むが、`ceil(1024/144) = 8` wave 中の 1 wave
分のみのため全体改善は数% 程度の見込み。**fixup 往復（上記）が短縮分を相殺しうることを判定前に明記する**
（ゲート C の閾値が N=1024/N=2048 で異なる理由の一部）。

## 6. GB10 実機実測記入欄（#1359）

### 0. 結論（先頭）

**REJECT（本番結線〈`select_tiled_f32_kernel`／`CudaGemm::new`〉は行わない。opt-in 実装
〈`internal-diagnostics` feature 限定〉はそのまま維持し、既定経路は不変）**。

GB10（DGX Spark GB10・sm_121）実機で §5 の判定基準（ゲート A〜C）に加え本節で追加した
ゲート D を実測した。ゲート A（決定性・正確性）とゲート B-2（既存経路の非後退）は PASS
したが、**ゲート B-1（残タイル複合判定統計）は 16 行中 12 行で `fail_count > 0`**（全行
0 fail の green ではない）であり、**ゲート C（`streamk_over_pipeline3` 5 回中央値）は
N=1024 で 1.0271 倍（< 1.05）・N=2048 で 0.9395 倍（< 1.00）**といずれも判定基準未達、
**ゲート D（64×64 streamk / 128×64 pipeline3 の 5 回中央値）も N=1024 で 0.9801 倍・
N=2048 で 0.8394 倍**（いずれも < 1.00）で未達だった。§5「結線の総合条件」（A ∧ B-1 全行
0 fail ∧ B-2 ∧ C 合格 ∧ D 合格）のうち B-1・C・D の 3 つが不成立のため、実測は捏造せず
記録した上で **REJECT** と確定する。

### 1. 判定基準の再掲

§5「#1359 向け実行コマンド・判定基準の事前宣言」のゲート A〜C をそのまま採用し、以下のゲート D を
**本番結線可否の前提条件**として追加する（ゲート C 自体の閾値は変更しない）。

- **ゲート A（決定性・正確性・必須）**: `cpu_cuda_tiled_pipeline_streamk_parity -- --ignored` の 8
  テストすべて PASS。1 本でも fail なら REJECT（結線なし）。
- **ゲート B（parity）**:
  - B-1: `streamk_full_tiles_match_non_persistent_bit_exact` が出力する残タイル複合判定統計
    （`fail_count`／`total`／`max_abs_diff`／`max_rel_err`・`remainder_tiles`／`q`／`max_contributors`）
    を全行記録し、全行 `fail_count == 0` なら green。1 行でも `fail_count > 0` の場合は tolerance 定数・
    `ParityBaseline` 行を変更せず、承認候補として記録するに留め結線は保留する。
  - B-2: 既存経路の非後退（`cpu_cuda_tiled_pipeline_parity` 全 PASS・`cpu_cuda_tiled_pipeline_persistent_parity`
    全 PASS）。
- **ゲート C（性能。`--blocks-per-sm auto`・GPU-only・各 5 回実行の中央値）**: N=1024 の
  `streamk_over_pipeline3` 中央値 ≥ 1.05、N=2048 で ≥ 1.00（かつ 0.95 未満の形状がないこと）。
- **ゲート D（結線専用。本イシューで追加）**: N=1024/N=2048 の本番経路実体は 128×64 pipeline
  （`TILED_PIPELINE_128X64_MIN_N/_MIN_K = Some(1024)`）であるため、`--tile both` 実行時に同一 run・
  同一 size の出力から分子を `tile=64x64` 行の `streamk_gpu_only_tflops`、分母を `tile=128x64` 行の
  `pipeline3_gpu_only_tflops` として読み取り、その比の 5 回中央値が N=1024・N=2048 とも ≥ 1.00
  であること。
- **結線の総合条件**: A PASS ∧ B-1 全行 0 fail ∧ B-2 PASS ∧ C 合格 ∧ D 合格。1 つでも欠ければ結線しない。

### 2. 環境

DGX Spark GB10（sm_121）。実行前後で `nvidia-smi --query-gpu=utilization.gpu` 0% を確認
（自身のベンチプロセスによる占有を除く。既知の常駐サービス〈ComfyUI・Kokoro〉は GPU compute を
占有しない）。実行コマンド・rustc/nvcc バージョン・uptime（load average）・破棄 run の有無は
`docs/perf/logs/cuda-tiled-pipeline-streamk-1359/env_info.txt` を参照（内部ホスト名は含めない）。

### 3. ゲート A（決定性・正確性。実機 `#[ignore]` 全 8 テスト）

```sh
cargo test -p fandhe-ai-backend-cuda --release --locked --features internal-diagnostics \
  --test cpu_cuda_tiled_pipeline_streamk_parity -- --ignored --nocapture --test-threads=1
```

**結果: 8 passed; 0 failed**
（`docs/perf/logs/cuda-tiled-pipeline-streamk-1359/gateA_streamk_parity.log`）。
`compile_tiled_pipeline_streamk_variant_rejects_zero_blocks_per_sm`・
`launch_tiled_pipeline_streamk_zero_dim_shape_is_noop_without_launch`・
`streamk_full_tiles_match_non_persistent_bit_exact`・
`streamk_inactive_matches_non_streamk_bit_exact`・`streamk_rejects_misaligned_shape`・
`streamk_rejects_mismatched_context_handle`・**`streamk_repeated_launch_is_deterministic`**
（AC 本体。決定性の実機確認）・`streamk_zero_k_returns_all_zero` の全 8 本が PASS した。
`docs/cuda-streamk-decision.md` §6 の再評価条件 2（fixup のアキュムレート順序変更が
非決定的にならないこと）は本ゲートの実機実行をもって充足した。

### 4. ゲート B-1・B-2

#### B-1: 残タイル複合判定統計（informational。合否ゲートではないが結線の前提条件）

`streamk_full_tiles_match_non_persistent_bit_exact` が出力した全 16 行
（`blocks_per_sm ∈ {Some(1), None}` × 8 形状）:

| m | n | k | blocks_per_sm | remainder_tiles | q | max_contributors | fail_count | total | max_abs_diff | max_rel_err |
|---|---|---|---|---|---|---|---|---|---|---|
| 1024 | 1024 | 1024 | Some(1) | 16 | 22 | 4 | 3 | 65536 | 0.0000553131 | 0.015133 |
| 2048 | 2048 | 2048 | Some(1) | 16 | 43 | 4 | 4 | 65536 | 0.0001201630 | 0.005681 |
| 4096 | 4096 | 4096 | Some(1) | 16 | 86 | 4 | 21 | 65536 | 0.0002059937 | 0.024527 |
| 256 | 256 | 4096 | Some(1) | 16 | 86 | 4 | 15 | 65536 | 0.0001983643 | 0.021975 |
| 448 | 448 | 1024 | Some(1) | 1 | 2 | 33 | 1 | 4096 | 0.0000391006 | 0.007318 |
| 60 | 68 | 36 | Some(1) | 2 | 1 | 4 | 0 | 4080 | 0.0000014305 | 0.000485 |
| 544 | 256 | 2048 | Some(1) | 36 | 96 | 3 | 20 | 139264 | 0.0001144409 | 0.602669 |
| 4100 | 1028 | 64 | Some(1) | 1 | 1 | 5 | 0 | 16 | 0.0000023842 | 0.0000013 |
| 1024 | 1024 | 1024 | None | 112 | 50 | 3 | 9 | 458752 | 0.0000648499 | 0.200332 |
| 2048 | 2048 | 2048 | None | 16 | 15 | 10 | 4 | 65536 | 0.0001182556 | 0.017021 |
| 4096 | 4096 | 4096 | None | 64 | 114 | 4 | 85 | 262144 | 0.0003585815 | 0.085903 |
| 256 | 256 | 4096 | None | 16 | 29 | 10 | 25 | 65536 | 0.0001792908 | 0.064796 |
| 448 | 448 | 1024 | None | 49 | 22 | 4 | 11 | 200704 | 0.0000648499 | 0.139074 |
| 60 | 68 | 36 | None | 2 | 1 | 4 | 0 | 4080 | 0.0000014305 | 0.000485 |
| 544 | 256 | 2048 | None | 36 | 32 | 5 | 24 | 139264 | 0.0001220703 | 0.602669 |
| 4100 | 1028 | 64 | None | 97 | 3 | 3 | 0 | 312592 | 0.0000038147 | 0.009984 |

**16 行中 4 行（`60×68×36`・`4100×1028×64` の各 blocks_per_sm 2 通り）のみ `fail_count == 0`**。
残る 12 行はいずれも `fail_count > 0`（最大 `fail_count=85`／`total=262144`。§5「本イシューでは
統計出力に留め、合否判定・baseline 行追加の要否は #1359 が判断する」で事前宣言したとおり、K 連鎖の
分割による丸め差が真値ゼロ近傍の要素で複合判定 fail を起こす〈§3「決定性の根拠」節で事前に
言及した性質どおり〉）。tolerance 定数・`ParityBaseline` の変更は行っていない。この結果は
「承認候補」として記録するに留め、**結線は保留**する（§5 の事前宣言どおり）。

#### B-2: 既存経路の非後退

```sh
cargo test -p fandhe-ai-backend-cuda --release --locked --features internal-diagnostics \
  --test cpu_cuda_tiled_pipeline_parity -- --ignored --nocapture --test-threads=1
cargo test -p fandhe-ai-backend-cuda --release --locked --features internal-diagnostics \
  --test cpu_cuda_tiled_pipeline_persistent_parity -- --ignored --nocapture --test-threads=1
```

**結果: `cpu_cuda_tiled_pipeline_parity` 18 passed; 0 failed**
（`docs/perf/logs/cuda-tiled-pipeline-streamk-1359/gateB2_pipeline_parity.log`）、
**`cpu_cuda_tiled_pipeline_persistent_parity` 10 passed; 0 failed**
（`docs/perf/logs/cuda-tiled-pipeline-streamk-1359/gateB2_persistent_parity.log`）。
#1358 の Stream-K 追加が非 Stream-K・persistent 両経路の実機挙動を変えていないことを確認した。

### 5. ゲート C・D（純カーネル時間。GPU-only。5 回独立プロセス起動）

```sh
cargo run -p fandhe-ai-backend-cuda --release --locked --features internal-diagnostics \
  --example gemm_tiled_pipeline_persistent_bench -- \
  --sizes 1024,2048,4096 --tile both --blocks-per-sm auto --streamk on
```

5 回独立プロセス起動
（`docs/perf/logs/cuda-tiled-pipeline-streamk-1359/gateC_streamk_auto_run{1..5}.log`。
各 run 前に `nvidia-smi --query-gpu=utilization.gpu` 0% を確認。撮り直しは発生せず）の
集計（`docs/perf/logs/cuda-tiled-pipeline-streamk-1359/aggregate.py`・`aggregate.md`）:

**ゲート C（64×64 タイル `streamk_over_pipeline3`。5 回中央値）**

| N | 5 run 値 | 中央値 | 判定基準 | 結果 |
|---|---|---|---|---|
| 1024 | 1.0169, 1.0271, 1.0189, 1.0295, 1.0350 | **1.0271** | ≥ 1.05 | **FAIL** |
| 2048 | 1.0157, 0.9355, 0.9050, 1.0145, 0.9395 | **0.9395** | ≥ 1.00 | **FAIL** |
| 4096 | 0.9799, 0.9464, 0.9823, 0.9368, 0.9613 | 0.9613 | 参考 | 参考 |

**ゲート D（64×64.`streamk_gpu_only_tflops` / 128×64.`pipeline3_gpu_only_tflops`。5 回中央値）**

| N | 5 run 値 | 中央値 | 判定基準 | 結果 |
|---|---|---|---|---|
| 1024 | 0.9735, 0.9801, 0.9701, 0.9819, 0.9829 | **0.9801** | ≥ 1.00 | **FAIL** |
| 2048 | 1.2295, 0.8394, 0.8132, 0.9015, 0.8377 | **0.8394** | ≥ 1.00 | **FAIL** |
| 4096 | 0.7027, 0.7194, 0.7486, 0.7177, 0.7295 | 0.7194 | 参考 | 参考 |

参考: 各 TFLOPS 列の 5 回中央値（起動ヘッダ: `tile=64x64 num_sms=48 blocks_per_sm=3
grid_capacity=144`・`tile=128x64 num_sms=48 blocks_per_sm=2 grid_capacity=96`。5 run とも同一）:

| N | tile | pipeline3 中央値 | persistent 中央値 | streamk 中央値 |
|---|---|---|---|---|
| 1024 | 64x64 | 11.1875 | 11.2522 | 11.4705 |
| 1024 | 128x64 | 11.7107 | 11.7993 | n/a（64×64 限定実装） |
| 2048 | 64x64 | 12.9367 | 12.9651 | 12.1509 |
| 2048 | 128x64 | 14.4284 | 14.6118 | n/a |
| 4096 | 64x64 | 9.6601 | 9.0707 | 9.1562 |
| 4096 | 128x64 | 12.6254 | 13.0581 | n/a |

N=2048 では `streamk_gpu_only_tflops`（64×64）の中央値（12.1509）が同一タイルの
`pipeline3_gpu_only_tflops`（12.9367）を下回っており、ゲート C の
`streamk_over_pipeline3` 中央値が 1.00 未満（0.9395）になっている一因である。

### 6. 机上見積りとの突合

§5「fixup 固定費の事前見積り」「末尾 wave 短縮の理論上限」の机上値
（N=1024: 理論改善見込み約 10% 前後・N=2048: 数% 程度、fixup 往復 N=1024 で約 5.25 MiB）と実測を
突合する:

- **N=1024**: 実測 `streamk_over_pipeline3` 中央値は 1.0271 倍（約 2.7% 改善）で、机上見積り
  （約 10% 前後）の 3 分の 1 以下に留まった。実測の `streamk_plan` は
  `remainder_tiles=112 q=50 sk_units=144 max_contributors=3 grid_blocks=144`
  （`gateC_streamk_auto_run1.log`）で §5 の想定 machine plan 例（`remainder_tiles=112,
  max_contributors=3`）と一致し、fixup 往復（約 5.25 MiB）自体は事前見積りどおりだった。
  末尾 wave 短縮の理論上限（`Q/nk = 50/64 ≈ 0.781`）に対し実測改善が小さいのは、fixup
  カーネル起動の固定費（追加カーネル起動・`atomicAdd` によるスケジューリングカウンタ経由の
  同期）が、§1「ゲート C の解釈注記」で事前に記録した懸念どおり短縮分の大半を相殺している
  ためと考えられる（#1347 の persistent 化単独でも同様の相殺が観測されており、本イシューの
  スコープでは追加切り分けを行わない）。
- **N=2048**: 実測 `streamk_over_pipeline3` 中央値は 0.9395 倍（**約 6% の後退**）で、机上見積り
  （数% 程度の改善見込み）とは符号が逆転した。`streamk_plan` は
  `remainder_tiles=16 q=15 sk_units=137 max_contributors=10`（`gateC_streamk_auto_run1.log`）
  で `max_contributors=10` と N=1024（3）より大幅に多く、fixup の寄与者数が多いほど固定順序
  逐次加算（§3「決定性の根拠」）のオーバーヘッドが増える構造が、末尾 wave 短縮分
  （`q/nk = 15/128 ≈ 0.117`。全体 8 wave 中の 1 wave 分のみ）を上回ったと考えられる。
- 総じて、**fixup 固定費（カーネル起動・寄与者数に比例する逐次加算コスト）が末尾 wave
  短縮による理論改善を実測では相殺・逆転させており**、§5 で事前に明記した懸念
  「fixup 往復（上記）が短縮分を相殺しうる」が両形状で実現した。

### 7. 採否・結線判断

**REJECT（本番結線は行わない）**。判定根拠:

- ゲート A: PASS（8/8）。
- ゲート B-1: **不成立**（16 行中 12 行で `fail_count > 0`。全行 0 fail という結線の前提条件を
  満たさない）。
- ゲート B-2: PASS（18/18・10/10）。
- ゲート C: **不成立**（N=1024 1.0271 倍 < 1.05・N=2048 0.9395 倍 < 1.00）。
- ゲート D: **不成立**（N=1024 0.9801 倍 < 1.00・N=2048 0.8394 倍 < 1.00）。

§5「結線の総合条件」（A ∧ B-1 全行 0 fail ∧ B-2 ∧ C 合格 ∧ D 合格）はゲート B-1・C・D の
3 つが不成立のため成立しない。`select_tiled_f32_kernel`／`CudaGemm::new` への結線は行わない
（既存方針を変更せず不変のまま）。opt-in 実装（`internal-diagnostics` feature 限定の
`compile_tiled_pipeline_streamk_variant`／`run_tiled_pipeline_streamk_f32` 等）自体は削除せず
そのまま維持する（実機実測の再現・将来の再評価のため）。

### 8. 申し送り・今後の検討候補

- **fixup 固定費の削減**: N=2048（`max_contributors=10`）のように寄与者数が多い形状ほど
  fixup オーバーヘッドが大きい傾向が見られた。fixup カーネル自体の起動オーバーヘッド削減
  （例: 別カーネル方式ではなく同一カーネル内での処理・グリッド構成の見直し）は本イシューの
  スコープ外（§7「申し送り（対象外）」に既出の「最後の寄与 CTA が in-kernel で fixup する」
  方式はデッドロックリスクにより不採用と設計時に判断済み）。
- **B-1 の複合判定 fail の扱い**: `fail_count > 0` の 12 行について、tolerance 緩和や
  `ParityBaseline` 行追加が必要かどうかはユーザー承認事項として本イシューでは判断しない
  （§5 の事前宣言どおり）。仮に将来 Stream-K を再検討する場合は、この統計を出発点に
  K 連鎖分割の丸め差の影響範囲（真値ゼロ近傍の要素数・形状依存性）を先に切り分ける必要がある。
- **128×64 タイルへの Stream-K 拡張**: 本番経路の実体（N≥1024 かつ K≥1024 で 128×64 pipeline）
  に対し Stream-K は 64×64 タイル限定実装のため、ゲート D は「異なるタイル同士の比較」に
  ならざるを得なかった。128×64 版 Stream-K を実装すれば同一タイルでの比較（64×64 pipeline
  改善効果の直接検証）が可能になるが、ゲート C 自体が両形状で FAIL のため優先度は低いと判断する
  （§7「申し送り（対象外）」に既出）。

## 7. 申し送り（対象外）

- 128×64 persistent 版（`kernels_tiled_pipeline_128x64.rs`）の Stream-K 化。
- TF32／f16 Tensor Core 経路・classic tiled f32 経路への横展開。
- tolerance 定数・`ParityBaseline` 行の追加（承認必須。#1359 の判断事項）。
- 「最後の寄与 CTA が in-kernel で fixup する」方式（CTA 間フラグ待ちが co-residency に依存し
  デッドロックリスクがあるため、別カーネル方式のみ実装した）。
- 本番結線（`select_tiled_f32_kernel`／`CudaGemm::new` は不変。結線可否は #1359 の実測後に判断する）。
