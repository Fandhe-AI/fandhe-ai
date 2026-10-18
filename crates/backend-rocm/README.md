# fandhe-ai-backend-rocm

ROCm（AMD GPU）バックエンド（イシュー #3154・#3155）。非公開（`publish = false`）。

- HIP・HIPRTC は `libloading` で実行時にロードする。ビルド時にはリンクしないため、
  ROCm 非搭載の環境（CI の ubuntu-latest・macOS・Windows）でもビルドとテストが通る。
- 公開 API: `WarpGeometry`・`WarpDefines`（wave 幅 32/64 を注入する HIP の `#define` 生成。#3432）・`is_available()`（デバイス 1 台以上で true）・`RocmGemmF32`（HIPRTC でコンパイルした f32 GEMM の実行）・`RocmBackendOps`（`BackendOps` の骨格・R0 #3431。`gemm` は `RocmGemmF32`、`add`〜`max` の 7 メソッドはホスト経路で CPU へ委譲。`device()` は `Device::Cpu` の仮置きで #3157 Phase B で差し替える）・`RocmLinalgLu`／`RocmLinalgFactor`（内部 opt-in ゲート `linalg::set_gpu_linalg_enabled`〈既定 OFF〉配下の `inv`・`solve`・`det`・`cholesky`・`qr`・`eigh`。`double` カーネル。#3444）・`RocmLoss`（loss 5 種〈mse・huber/smooth_l1・bce・nll・kl_div〉と backward の 11 カーネル。wave 幅注入の 2 段縮約。`RocmBackendOps` の 10 メソッドが結線。#3436・R5）・`GfxTarget`・`GemmF32Tile`・`render_gemm_f32_source`・`RocmError`。
- 実機での実行確認は未実施（#3159）。`tests/gemm_real_device.rs`・`tests/linalg_real_device.rs`・`tests/loss_real_device.rs` は `#[ignore]`。
  ホスト側では CPU 上の添字モデル（`src/gemm_model.rs`・`src/linalg_model.rs`・`src/loss_model.rs`）と参照 GEMM・CPU 参照 の照合までを検証する。
- 設計の正本: `docs/backend-rocm-hip-ffi-decision.md`・`docs/backend-rocm-hiprtc-gemm-decision.md`
