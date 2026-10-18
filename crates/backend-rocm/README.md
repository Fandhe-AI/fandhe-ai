# fandhe-ai-backend-rocm

ROCm（AMD GPU）バックエンド（イシュー #3154・#3155）。非公開（`publish = false`）。

- HIP・HIPRTC は `libloading` で実行時にロードする。ビルド時にはリンクしないため、
  ROCm 非搭載の環境（CI の ubuntu-latest・macOS・Windows）でもビルドとテストが通る。
- 公開 API: `WarpGeometry`・`WarpDefines`（wave 幅 32/64 を注入する HIP の `#define` 生成。#3432）・`is_available()`（デバイス 1 台以上で true）・`RocmGemmF32`（HIPRTC でコンパイルした f32 GEMM の実行）・`RocmBackendOps`（`BackendOps` の骨格・R0 #3431。`gemm` は `RocmGemmF32`、`sum`・`max`・`min`・`argmax`・`argmin` は R3〈#3434〉の HIPRTC 縮約カーネル〈`RocmReduce`・`RocmArgReduce`。wave 幅は `WarpGeometry` で注入。空縮約は CPU へ委譲〉。`device()` は `Device::Cpu` の仮置きで #3157 Phase B で差し替える）・`RocmLinalgLu`／`RocmLinalgFactor`（内部 opt-in ゲート `linalg::set_gpu_linalg_enabled`〈既定 OFF〉配下の `inv`・`solve`・`det`・`cholesky`・`qr`・`eigh`。`double` カーネル。#3444）・`RocmLoss`（loss 5 種〈mse・huber/smooth_l1・bce・nll・kl_div〉と backward の 11 カーネル。wave 幅注入の 2 段縮約。`RocmBackendOps` の 10 メソッドが結線。#3436・R5）・`RocmElementwise`／`RocmScalarOp`／`RocmFusedElementwise`（elementwise 一式・R2 #3433。`add`・`mul`・`relu`・`exp`・`tanh`・`where_cond`・`masked_fill`・`scalar_unary`／`scalar_binary`・elementwise 融合・デバイス常駐の `binary_elementwise_device`／`unary_elementwise_device`。ソース先頭の `#pragma clang fp contract(off)` で FMA への暗黙縮約を止める）・`RocmSoftmax`／`RocmLogSoftmaxBackward`（`softmax`・`log_softmax`・`log_softmax_backward` の HIPRTC カーネル。最終軸限定・wave 幅は `WarpGeometry` 注入・2 パス経路のみ。`RocmBackendOps` へ結線済み。#3435）・`GfxTarget`・`GemmF32Tile`・`render_gemm_f32_source`・`RocmError`。
- 実機での実行確認は未実施（#3159）。`tests/gemm_real_device.rs`・`tests/linalg_real_device.rs`・`tests/reduce_real_device.rs`・`tests/loss_real_device.rs`・`tests/softmax_real_device.rs` は `#[ignore]`。
  ホスト側では CPU 上の添字モデル（`src/gemm_model.rs`・`src/linalg_model.rs`・`src/loss_model.rs`）と参照 GEMM の照合までを検証する。
- 設計の正本: `docs/backend-rocm-hip-ffi-decision.md`・`docs/backend-rocm-hiprtc-gemm-decision.md`
