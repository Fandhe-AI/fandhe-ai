# fandhe-ai-backend-rocm

ROCm（AMD GPU）バックエンド（イシュー #3154・#3155）。非公開（`publish = false`）。

- HIP・HIPRTC は `libloading` で実行時にロードする。ビルド時にはリンクしないため、
  ROCm 非搭載の環境（CI の ubuntu-latest・macOS・Windows）でもビルドとテストが通る。
- 公開 API: `is_available()`（デバイス 1 台以上で true）・`RocmGemmF32`（HIPRTC でコンパイルした
  f32 GEMM の実行）・`RocmBackendOps`（`BackendOps` の骨格・R0 #3431。`gemm` は `RocmGemmF32`、`add`〜`max` の 7 メソッドはホスト経路で CPU へ委譲。`device()` は `Device::Cpu` の仮置きで #3157 Phase B で差し替える）・`GfxTarget`・`GemmF32Tile`・`render_gemm_f32_source`・`RocmError`。
- 実機での実行確認は未実施（#3159）。`tests/gemm_real_device.rs` は `#[ignore]`。
  ホスト側では CPU 上の添字モデル（`src/gemm_model.rs`）と参照 GEMM の照合までを検証する。
- 設計の正本: `docs/backend-rocm-hip-ffi-decision.md`・`docs/backend-rocm-hiprtc-gemm-decision.md`
