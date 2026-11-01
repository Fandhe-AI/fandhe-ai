# fandhe-ai-backend-rocm

ROCm（AMD GPU）バックエンド（イシュー #3154・#3155）。**条件付き・実験的**。

> 本クレートへの直接依存はサポート対象外です。サポートされる公開 API 面は [`fandhe-ai`](https://crates.io/crates/fandhe-ai)（facade）のみで、本クレートは facade の依存解決のために crates.io で公開しています。
>
> 実機（AMD GPU）での実行確認は未実施です（#3159）。facade からの `Device::Rocm(0)` は 0.14.0 で追加予定（#3851〜#3853）で、現時点では facade から参照されません。

- HIP・HIPRTC は `libloading` で実行時にロードする。ビルド時にはリンクしないため、
  ROCm 非搭載の環境（CI の ubuntu-latest・macOS・Windows）でもビルドとテストが通る。
- 公開 API: `WarpGeometry`・`WarpDefines`（wave 幅 32/64 を注入する HIP の `#define` 生成。#3432）・`is_available()`（デバイス 1 台以上で true）・`RocmGemmF32`（HIPRTC でコンパイルした f32 GEMM の実行）・`RocmGemmBiasAct`（GEMM + bias + ReLU を 1 本にした融合カーネル。主ループは `RocmGemmF32` とバイト同一。`RocmBackendOps` の `gemm_resident_rhs`・`gemm_resident_lhs`・`gemm_fp32_strict_into`・`linear_forward_device` の 4 メソッドが結線。常駐オペランドを download しない。R11 #3442）・`RocmBackendOps`（`BackendOps` の骨格・R0 #3431。`gemm` は `RocmGemmF32`、`sum`・`max`・`min`・`argmax`・`argmin` は R3〈#3434〉の HIPRTC 縮約カーネル〈`RocmReduce`・`RocmArgReduce`。wave 幅は `WarpGeometry` で注入。空縮約は CPU へ委譲〉。`device()` は `Device::Cpu` の仮置きで #3157 Phase B で差し替える）・`RocmLinalgLu`／`RocmLinalgFactor`（内部 opt-in ゲート `linalg::set_gpu_linalg_enabled`〈既定 OFF〉配下の `inv`・`solve`・`det`・`cholesky`・`qr`・`eigh`。`double` カーネル。#3444）・`RocmLoss`（loss 5 種〈mse・huber/smooth_l1・bce・nll・kl_div〉と backward の 11 カーネル。wave 幅注入の 2 段縮約。`RocmBackendOps` の 10 メソッドが結線。#3436・R5）・`RocmElementwise`／`RocmScalarOp`／`RocmFusedElementwise`（elementwise 一式・R2 #3433。`add`・`mul`・`relu`・`exp`・`tanh`・`where_cond`・`masked_fill`・`scalar_unary`／`scalar_binary`・elementwise 融合・デバイス常駐の `binary_elementwise_device`／`unary_elementwise_device`。ソース先頭の `#pragma clang fp contract(off)` で FMA への暗黙縮約を止める）・`RocmSoftmax`／`RocmLogSoftmaxBackward`（`softmax`・`log_softmax`・`log_softmax_backward` の HIPRTC カーネル。最終軸限定・wave 幅は `WarpGeometry` 注入・2 パス経路のみ。`RocmBackendOps` へ結線済み。#3435）・`RocmRnnCell`（LSTM／GRU の pointwise 段と backward の 5 カーネル。`RocmBackendOps` の 5 メソッドが結線。FMA は CPU の `mul_add` と同じ 3 か所だけ `__builtin_fmaf` で明示。実機 parity は #3159。#3441・R10）・`RocmPagedKv`（デバイス常駐 KV の `paged_kv_append`・`paged_attention_forward` の 2 カーネル。`RocmBackendOps` の 2 メソッドが結線。`j` 軸は単一の連続ループで wave 幅非依存・FMA は score と加重和の 2 か所だけ `__builtin_fmaf`。実機 parity は #3159。#3443・R12）・`GfxTarget`・`GemmF32Tile`・`render_gemm_f32_source`・`RocmError`・`RocmBackendOps` は `sgd_step_device`／`adam_step_device`（デバイス常駐の in-place 更新・同期実行。公開しない内部の `RocmOptim` が実行。R9 #3440）も持つ。
- 実機での実行確認は未実施（#3159）。`tests/gemm_real_device.rs`・`tests/linalg_real_device.rs`・`tests/reduce_real_device.rs`・`tests/loss_real_device.rs`・`tests/softmax_real_device.rs` は `#[ignore]`。
  ホスト側では CPU 上の添字モデル（`src/gemm_model.rs`・`src/linalg_model.rs`・`src/loss_model.rs`・`src/rnn_cell_model.rs`・`src/paged_kv_model.rs`）と参照 GEMM の照合までを検証する。
- 設計の正本: `docs/backend-rocm-hip-ffi-decision.md`・`docs/backend-rocm-hiprtc-gemm-decision.md`

## ライセンス

MIT または Apache-2.0 のデュアルライセンス（`LICENSE-MIT`・`LICENSE-APACHE`）。
