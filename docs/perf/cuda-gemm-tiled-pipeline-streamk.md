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

**未実測（ブロッカー: 本エージェント実行環境に CUDA 実機〈GB10〉への接続手段なし）**。§3・§4 に記す
とおり Mac 側で実行可能な範囲（ホストシミュレータテスト・静的テスト・ビルド整合性）はすべて green
であることを確認した。GB10 実機でのゲート A〜D（`#[ignore]` 実機テスト・`--streamk on` ベンチ 5 回計測）
は未実施であり、実測値を捏造せずここに記録する。§7 に再開手順を記す。

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
  `pipeline3_gpu_only_tflops`（`gemm_tiled_pipeline_persistent_bench` は tile 幅ごとに独立した
  `tile=<label>` 行を出力し、128×64 の非 persistent 基準値もラベルに依らず列名
  `pipeline3_gpu_only_tflops` で出す。`pipeline128x64_gpu_only_tflops` という列は存在しない）として
  読み取り、その比の 5 回中央値が N=1024・N=2048 とも ≥ 1.00 であること。
- **結線の総合条件**: A PASS ∧ B-1 全行 0 fail ∧ B-2 PASS ∧ C 合格 ∧ D 合格。1 つでも欠ければ結線しない。

### 2. 環境

- 実行環境: 本エージェントの作業 worktree（Mac、CUDA 実機なし）。
- `docs/real-hardware-verification-env.local.md`（GB10 実機接続情報。Git 管理外）が本 worktree に
  存在しないため `CUDA_NODE` を解決できず、GB10 への rsync 転送・SSH 実行に着手できなかった
  （`docs/real-hardware-verification-env.local.md.example` はテンプレートのみで実値を含まない）。
- ネットワーク経由の CUDA 実機（DGX Spark GB10）への代替アクセス手段も本セッションには与えられていない。

### 3. ゲート A（実機未到達のため未実施）

未実施。Mac 側で代替として `cargo test -p fandhe-ai-backend-cuda --lib --locked` を実行し、Stream-K の
GPU 不要ホストシミュレータテスト・静的テスト（`gemm::tests::streamk_plan_*`・
`kernels_tiled_pipeline::tests::tiled_pipeline_streamk_*`）を含む 696 tests が全て green（0 failed）
であることを確認した（`streamk` 部分一致フィルタで 15 tests 抽出・全 PASS）。これは §3「決定性の根拠」
の設計時静的検査の再確認であり、ゲート A（実機での bit 同一性・繰り返し起動の決定性）の代替にはならない。

### 4. ゲート B（実機未到達のため未実施）

未実施。残タイル複合判定統計・既存経路の非後退確認はいずれも GB10 実機での `#[ignore]` テスト実行を
前提とするため、記録すべき実測値がない。

### 5. ゲート C・D（実機未到達のため未実施）

未実施。`gemm_tiled_pipeline_persistent_bench --streamk on` による 5 回計測・TFLOPS 中央値比較は
GB10 実機を前提とするため、記録すべき実測値がない。

### 6. 机上見積りとの突合

実測値がないため突合不能。§5「fixup 固定費の事前見積り」「末尾 wave 短縮の理論上限」の机上値
（N=1024: 理論改善見込み約 10% 前後・N=2048: 数% 程度、fixup 往復 N=1024 で約 5.25 MiB）は本節時点で
未検証のまま残る。

### 7. 採否・結線判断

**保留（実機未到達のため判定不能）**。ADOPT／REJECT のいずれも実測なしには確定できない。本番結線
（`select_tiled_f32_kernel`／`CudaGemm::new`）は行わない（既存方針を変更せず不変のまま）。

### 8. 申し送り・再開手順

- **再開手順**: GB10（または同等の CUDA 実機）へ SSH 到達可能なセッションで、
  `docs/real-hardware-verification-env.local.md`（`docs/real-hardware-verification-env.local.md.example`
  をコピーして `CUDA_NODE` 等を実値で埋めたもの）を用意したうえで、§5 の実行コマンド（ゲート A・B は
  `--ignored --nocapture --test-threads=1`、ゲート C・D は `--sizes 1024,2048,4096 --tile both
  --blocks-per-sm auto --streamk on` を独立 5 回）を実行し、本節（0〜7 節）を実測値で置き換えること。
  実行前後の `uptime`（load average）・`nvidia-smi --query-gpu=name,driver_version,utilization.gpu`・
  `nvidia-smi --query-compute-apps` を記録し、生ログは
  `docs/perf/logs/cuda-tiled-pipeline-streamk-1359/` 配下へ保存すること（内部ホスト名・ユーザー名を
  含めない）。
- 本節の記入自体は #1359 の受入条件（「parity 結果・5 回中央値・env_info が記録されていること」）を
  満たしていない。再開後の実測完了をもって受入条件を充足させる必要がある。
- `docs/cuda-streamk-decision.md` §6 は本節の状態（未実測・保留）に対応する形で追記した。

## 7. 申し送り（対象外）

- 128×64 persistent 版（`kernels_tiled_pipeline_128x64.rs`）の Stream-K 化。
- TF32／f16 Tensor Core 経路・classic tiled f32 経路への横展開。
- tolerance 定数・`ParityBaseline` 行の追加（承認必須。#1359 の判断事項）。
- 「最後の寄与 CTA が in-kernel で fixup する」方式（CTA 間フラグ待ちが co-residency に依存し
  デッドロックリスクがあるため、別カーネル方式のみ実装した）。
- 本番結線（`select_tiled_f32_kernel`／`CudaGemm::new` は不変。結線可否は #1359 の実測後に判断する）。
