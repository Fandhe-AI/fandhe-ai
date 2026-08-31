# CUDA GEMM tiled pipeline（FP32 SIMT + cp.async 多段パイプライン）計測記録（#1033）

イシュー #1033「perf(backend-cuda): cp.async 多段パイプライン（3〜4 stage）を FP32 SIMT 経路に導入する」の実測記録テンプレート。
親イシュー #1031（FP32 SIMT GEMM 強化）・ルート #1029「GEMM カーネルの candle 超え」Phase 2 の一環。
受け入れ条件「N=4096 での改善値の記録（5 回計測中央値）」「カーネル側の手動境界検査の維持（REQ-8）」に対応する。

## 状態: DGX Spark GB10 実機実測完了（下記「実測結果」節）。以下は実測前の実装セッションの記録として残す

本実装セッションの実行環境には CUDA **driver**（`libcuda`）が実在し、`nvidia-smi` で以下の実機が確認できる:

```
NVIDIA GeForce RTX 3060（compute capability 8.6・Driver Version 595.71.05・CUDA Version 13.2 表記）
```

これは `cp.async`（`cp.async.cg.shared.global`/`.commit_group`/`.wait_group`）が要求する compute capability 8.0 以上（Ampere 以降）
を満たすため、`CudaDevice::new(0)` は成功する。しかし **NVRTC（`libnvrtc`）はこの環境に存在せず**、`compile_ptx` は
`CudaError::NvrtcUnavailable` を返す（`nvcc`／CUDA toolkit 自体が未導入）。

したがって、本ファイルが指すカーネルソース（`crates/backend-cuda/src/kernels_tiled_pipeline.rs::TILED_PIPELINE_F32_BODY`）は
**この実装セッション中に一度も NVRTC の構文検証を通過していない**。`docs/perf/cuda-gemm-mma-pipeline.md`・
`docs/perf/metal-gemm-dynamic-tile.md` の先例（「実機での最初の実行が構文検証を兼ねる」）と同じ位置づけであり、
**RTX 3060（sm_86）を含むいかなる実機でも未検証**である点に注意。sm_121（DGX Spark GB10。設計上のターゲット）での挙動は
さらに別途の検証が必要。

本実装セッションで検証済みの事項:

- `cargo build -p fandhe-ai-backend-cuda`／`cargo test -p fandhe-ai-backend-cuda`／
  `cargo clippy -p fandhe-ai-backend-cuda --all-targets --all-features -- -D warnings` が全て green（Rust 側の型・API
  契約・カーネル定数の内部整合性はコンパイル時 `const _: () = assert!(...)`（`kernels_tiled_pipeline.rs`）で検査済み）
- `cargo run -p fandhe-ai-backend-cuda --example gemm_tiled_pipeline_bench` を実際に実行し、CUDA driver 到達→NVRTC
  不在検出→graceful skip の分岐が意図どおり動作することを確認済み（出力: `CudaGemm::new failed
  (CUDA NVRTC library unavailable: ...); nothing to measure.`）
- `kernels_tiled_pipeline.rs` 内の `#[cfg(test)]`（cp.async 命令実在検査・REQ-8 境界ガード実在検査・Rust 側定数と
  カーネル `#define` の整合検査・`tiled_pipeline_f32_source_with_stages` の範囲検証・commit/wait_group 出現回数検査）が
  全て green
- `cargo test -p fandhe-ai-backend-cuda --test cpu_cuda_tiled_pipeline_parity` の環境適応スモーク
  （`tiled_pipeline_parity_smoke_env_adaptive`）が green（NVRTC 不在検出で early return）。実機必須の残り 6 テストは
  `#[ignore]`（未実行）

未検証の事項（DGX Spark GB10 実機セッションへ引き継ぐ）:

- NVRTC が `cp.async.cg.shared.global`／`.commit_group`／`.wait_group` を受理するかどうか（本カーネルはインライン
  PTX の命令セット自体は `kernels_mma_tf32.rs::mma_tf32_cp_async16` と全く同一のため、その経路がコンパイル可能であれば
  本カーネルも可能である蓋然性は高いが未確認）
- `as_tile`/`bs_tile` の共有メモリレイアウト（転置なし。本ファイル冒頭コメント参照）・レジスタブロッキング
  （4×4 外積）の実際の正しさ（`cargo test -- --ignored` の parity テストでのみ確認可能）
- 3 stage（既定）・4 stage（オンデマンドコンパイル変種）のスループット差・occupancy への影響

## 実測手順（DGX Spark GB10 セッション向け）

1. **正しさ検証を先に実行する**（性能計測より優先）:
   ```sh
   cargo test -p fandhe-ai-backend-cuda --test cpu_cuda_tiled_pipeline_parity -- --ignored
   ```
   全 PASS を確認してから性能計測へ進む。`tiled_pipeline_matches_tiled_f32` が既存 `run_tiled_f32` との相互比較も
   兼ねるため、この 1 テストの PASS だけでも「新カーネルが既存本番経路と同一の GEMM を計算している」ことの追加保証になる。

2. **性能計測**:
   ```sh
   cargo run -p fandhe-ai-backend-cuda --example gemm_tiled_pipeline_bench --release
   ```
   N=1024/2048/4096 それぞれについて、TFLOPS を 5 回計測中央値（`bench_harness::MeasurementConfig::default` の
   warmup/計測 20 回以上）で記録する。**計測区間は 2 段に分かれる**（codex-review P2／Cursor Bugbot 指摘。PR #1071
   `gemm_tiled_pipeline_bench.rs` モジュールコメント「計測区間の統一」参照。異なる区間の TFLOPS を同じ比率へ混ぜると
   「転送有無の違い」が「stage 数増加による改善」として誤計上されるため、実装は比較を分離済み）:
   - **転送込み同士**: `tiled_f32`（本番既定経路） vs `pipeline3`（既定 3 stage・転送込み）。`pipeline3_over_tiled`
     はこの区間の比率
   - **GPU-only 同士**: `pipeline3_gpu_only`（既定 3 stage） vs `pipeline4_gpu_only`（4 stage）。
     `pipeline4_over_pipeline3_gpu_only` はこの区間の比率で、cp.async ステージ数の増加そのものの効果を表す

3. **3 vs 4 stage の追加スイープが必要な場合**: `examples/gemm_tiled_pipeline_bench.rs` の `STAGE_4` 定数を変更するか、
   `CudaGemm::compile_tiled_pipeline_variant(&device, stages)` を直接呼ぶ小さな診断バイナリを追加する
   （`gemm_wmma_tf32_staged_stages_bench.rs` の段数スイープパターンを参考にできるが、tiled pipeline は静的共有メモリ
   予算内（`TP_MAX_STAGES=4` でも約 37KiB < 48KiB）に収まるため、staged 版のような動的共有メモリ opt-in 変種は
   不要である点に注意）。

4. **記録先**: 実測値・TFLOPS・対 tiled f32 比・対 candle 比（`docs/perf/cuda-gemm-kernel-improvement-policy.md` の
   比較基準）は本ファイルの「実測結果」節（プレースホルダ。以下）へ追記する。

## 実測結果（DGX Spark GB10 実機実測完了。#1031）

実機: DGX Spark GB10（compute capability (12, 1) = sm_121）・driver 580.173.02・CUDA 13.0.88
（`nvcc --version` 実測）・rustc 1.97.0。計測時 `nvidia-smi --query-gpu=utilization.gpu
--format=csv,noheader` で 0% を確認済み（他プロセス非同居）。commit
`10011cd4f8ef097351c0dc1244eb55c8a021040b`（rsync 転送元 worktree の HEAD）。

**正当性検証（手順 1）**: `cargo test -p fandhe-ai-backend-cuda --release --features
internal-diagnostics --test cpu_cuda_tiled_pipeline_parity -- --ignored --nocapture`
（`internal-diagnostics` feature 指定が本コマンドに必須である点は本ファイルの手順記載から
更新: feature 未指定では `cargo test` が `requires the features: internal-diagnostics` で
即座に失敗する）。**9 件全て PASS**（`tiled_pipeline_matches_tiled_f32`・
`tiled_pipeline_matches_reference_across_shapes`・`tiled_pipeline_k4096_stress` を含む）。
これは本カーネル（`TILED_PIPELINE_F32_BODY`。`cp.async.cg.shared.global`／
`.commit_group`／`.wait_group` を含む）が sm_121 実機で NVRTC 構文検証を初めて通過し、
かつ CPU 参照実装・`run_tiled_f32` の双方と数値一致（複合判定: 相対誤差 1e-3 未満 または
絶対誤差 1e-5 未満）することを確認した記録である（本ファイル冒頭「状態」節の
「いかなる実機でも未検証」は本実測により解消）。

**性能計測（手順 2）**: `cargo run -p fandhe-ai-backend-cuda --example
gemm_tiled_pipeline_bench --release --features internal-diagnostics`（同様に feature 指定が
必須。既存記載の手順ではエラーになったため実行時に追加した）。

列は `gemm_tiled_pipeline_bench.rs` の出力そのまま（区間を混ぜない。上記「実測手順」手順 2 参照）。

| size (M=N=K) | tiled_f32 TFLOPS（転送込み） | pipeline3 TFLOPS（転送込み） | pipeline3_over_tiled | pipeline3_gpu_only TFLOPS | pipeline4_gpu_only TFLOPS | pipeline4_over_pipeline3_gpu_only |
|---|---|---|---|---|---|---|
| 1024 | 3.9053 | 5.1523 | 1.3193 | 11.2165 | 10.7736 | 0.9605 |
| 2048 | 5.4793 | 7.8827 | 1.4386 | 13.0398 | 12.1491 | 0.9317 |
| 4096 | 0.2120 | 0.2104 | 0.9926 | 9.9062 | 9.7746 | 0.9867 |

**観察**:

- 転送込み区間（`tiled_f32` vs `pipeline3`）では N=1024・2048 で pipeline3 が
  1.32〜1.44 倍改善する一方、**N=4096 では両者とも 0.21 TFLOPS 台まで落ち込み
  `pipeline3_over_tiled` が 0.99 倍（実質横ばい）**になっている。この N=4096 の絶対値の
  低さは `docs/perf/cuda-fresh-gemm-n2048-overhead-diagnosis.md` §6.4 が記録する
  「N=4096 P4（fresh モードの D2H/確保オーバーヘッド）のばらつき（540 ms ↔ 39 ms、原因未特定）」
  と同系統の fresh モード転送込み経路の残存オーバーヘッド事象と整合する挙動であり、
  `tiled_f32`（本番既定経路）自体も同じ落ち込みを示していることから **本ファイルが対象と
  する cp.async パイプライン化固有の劣化ではない**（`tiled_f32` 側もほぼ同水準まで
  落ちているため、両者の相対比較である `pipeline3_over_tiled` の値自体は破綻していない）。
  原因調査は上記診断ドキュメントのスコープであり本ファイルでは追跡のみとする。
- **GPU-only 区間（`pipeline3_gpu_only` vs `pipeline4_gpu_only`）では 3 サイズとも
  4 stage が 3 stage を下回る**（`pipeline4_over_pipeline3_gpu_only` = 0.93〜0.99 倍）。
  cp.async ステージ数を 3→4 に増やしても sm_121 では改善せず、静的共有メモリ予算内
  （§「実測手順」手順 3 の記載どおり TP_MAX_STAGES=4 でも約 37KiB）に収まる範囲でも
  occupancy 等の他要因が支配的である可能性が高い。3 stage（既定）を据え置く判断を
  裏付ける結果であり、4 stage への切替は推奨されない。
- GPU-only 経路の絶対値（3 stage: 9.91〜13.04 TFLOPS）は転送込み経路（0.21〜5.15
  TFLOPS）を大きく上回り、H2D/D2H・出力バッファ確保のオーバーヘッドが依然として
  支配的であることを示す。この転送込みオーバーヘッドの削減自体は本イシューのスコープ外
  （「スコープ外事項」節・本番既定経路への結線判断は #1035 が担う）。

## スコープ外事項（本 PR では対応しない）

- **既定経路への接続・形状別選択**: 本 PR は `run_tiled_f32`（本番既定経路）を切り替えない。tiled pipeline は
  `CudaGemm::run_tiled_pipeline_f32` で明示的に呼べる選択可能な変種として追加するに留める。形状別の経路選択・既定
  切替は兄弟イシュー #1035（simple / double-buffer / split-K ヒューリスティック）が実測を踏まえて担う。
- **レジスタブロッキング拡大・bank conflict 対策**: 兄弟イシュー #1032（並行実装の可能性あり）のスコープ。本 PR は
  #1032 のマージ状況を実装開始時に確認したが未マージだったため、既存 `kernels.rs::TILED_F32` を書き換えず新規
  モジュール（`kernels_tiled_pipeline.rs`）に自己完結した実装を追加した（コンフリクト回避）。

## 関連ドキュメント

- `docs/perf/cuda-gemm-mma-pipeline.md`（TF32 `mma.sync`/`ldmatrix`/`cp.async` 経路の同型記録。本ファイルの
  パイプライン骨格の移植元）
- `docs/perf/cuda-gemm-kernel-improvement-policy.md`（FP32 SIMT 経路の背景・candle 比較基準）
- `crates/backend-cuda/src/kernels_tiled_pipeline.rs`（カーネルソース・設計コメント）
