# CUDA GEMM tiled pipeline（FP32 SIMT + cp.async 多段パイプライン）計測記録（#1033）

イシュー #1033「perf(backend-cuda): cp.async 多段パイプライン（3〜4 stage）を FP32 SIMT 経路に導入する」の実測記録テンプレート。
親イシュー #1031（FP32 SIMT GEMM 強化）・ルート #1029「GEMM カーネルの candle 超え」Phase 2 の一環。
受け入れ条件「N=4096 での改善値の記録（5 回計測中央値）」「カーネル側の手動境界検査の維持（REQ-8）」に対応する。

## 状態: 実測未実施・NVRTC 未検証（実装環境に CUDA driver はあるが NVRTC がない）

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
   N=1024/2048/4096 それぞれについて、tiled f32（本番既定経路）・tiled pipeline（既定 3 stage）・tiled pipeline
   （4 stage・オンデマンドコンパイル）の TFLOPS を 5 回計測中央値（`bench_harness::MeasurementConfig::default` の
   warmup/計測 20 回以上）で記録する。

3. **3 vs 4 stage の追加スイープが必要な場合**: `examples/gemm_tiled_pipeline_bench.rs` の `STAGE_4` 定数を変更するか、
   `CudaGemm::compile_tiled_pipeline_variant(&device, stages)` を直接呼ぶ小さな診断バイナリを追加する
   （`gemm_wmma_tf32_staged_stages_bench.rs` の段数スイープパターンを参考にできるが、tiled pipeline は静的共有メモリ
   予算内（`TP_MAX_STAGES=4` でも約 37KiB < 48KiB）に収まるため、staged 版のような動的共有メモリ opt-in 変種は
   不要である点に注意）。

4. **記録先**: 実測値・TFLOPS・対 tiled f32 比・対 candle 比（`docs/perf/cuda-gemm-kernel-improvement-policy.md` の
   比較基準）は本ファイルの「実測結果」節（プレースホルダ。以下）へ追記する。

## 実測結果（プレースホルダ。DGX Spark GB10 セッションで記入）

| size (M=N=K) | tiled_f32 TFLOPS | tiled_pipeline(3) TFLOPS | tiled_pipeline(4) TFLOPS | pipeline(3)/tiled | pipeline(4)/tiled |
|---|---|---|---|---|---|
| 1024 | (未実測) | (未実測) | (未実測) | (未実測) | (未実測) |
| 2048 | (未実測) | (未実測) | (未実測) | (未実測) | (未実測) |
| 4096 | (未実測) | (未実測) | (未実測) | (未実測) | (未実測) |

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
