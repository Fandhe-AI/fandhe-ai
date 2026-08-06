//! CUDA バックエンド。
//!
//! `tensor-core` の演算グラフノードを NVRTC 経由でコンパイルした CUDA カーネルへ変換して
//! 実行する。バックエンド切替は feature フラグなしの cfg ベース（PoC-v2-5 実証構成。REQ-2）で、
//! 依存する `cudarc` は無条件依存かつ動的ロード方式を用いるため、CUDA toolkit 非搭載環境でも
//! ビルド自体は成立する（実行時のみ toolkit を要求。`.claude/rules/deps-policy.md`）。
//!
//! `backend-cpu` との数値一致は統一複合判定「相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満」で
//! 検証する。丸め方針（FMA 契約）は NVRTC の既定 FMA 契約を CPU 参照実装（`f32::mul_add`）と
//! 揃える（PoC-v2-5 の K=4096 ストレスケースで実測確認済み。`.claude/rules/coding-rust.md`）。
//! カーネルの手動境界検査は最適化を理由に省略しない（REQ-8）。
//! FFI 境界の `unsafe` は必要最小限に留め理由コメントを付す（`.claude/rules/security.md`）。
//!
//! TASK-1.7a（#32）で、動的ロード・デバイス初期化・NVRTC コンパイルの基盤
//! （`CudaDevice`・`CudaError`・`compile_ptx`）を追加した。`cudarc` の
//! `dynamic-loading` feature は `libcuda`/`libnvrtc` が `dlopen` できない
//! 環境で driver/nvrtc API を直接呼ぶと `Err` ではなく panic するため、
//! 本クレートの初期化入口（`CudaDevice::new`・`CudaDevice::device_count`・
//! `compile_ptx`）は `is_culib_present()` による非 panic プローブで
//! 必ずゲートしてから型付きエラー（`CudaError::DriverUnavailable`／
//! `NvrtcUnavailable`）を返す（`device.rs`／`nvrtc.rs` のドキュメンテーション
//! コメント参照）。これにより CUDA 非搭載環境でも panic しない。
//!
//! TASK-1.9a（#44）で `device` モジュールに [`device::CudaDeviceProvider`]
//! （`tensor_core::device::DeviceProvider` の CUDA 実装）を追加した。上記の
//! `CudaDevice` を内部で経由するため panic 回避ゲートは共通で効く。CPU／Metal
//! 実装（`backend-cpu::CpuDeviceProvider`／`backend-metal::device::MetalDeviceProvider`）
//! と同一 trait で列挙・選択できることを
//! `backend-cpu/tests/device_provider_integration.rs` で検証する。CUDA
//! ドライバ非搭載環境では `is_available() == false`・`enumerate() == Ok(vec![])`
//! を返す（fail-safe。`device.rs` 内コメント参照）。
//!
//! カーネルソース・起動 API は naive 版（#33）・tiled 版（#34。共有メモリ
//! タイリング `TILE=32`）を追加済み。CUDA toolkit 非搭載ビルドの CI 検証は
//! `.github/workflows/ci.yml` の `build-no-cuda-toolkit` ジョブと
//! `scripts/check-cuda-toolkit-absent.sh`（TASK-1.7d・#35）で実装済み。
//! 実機（DGX Spark GB10 等）依存テストの `#[ignore]` 分離は #36 で
//! 完了した（実機での実行導線は `make test-ignored-cuda`。`Makefile`・
//! `README.md` 参照）。f16 向け許容誤差の設計・採用（実質的な許容誤差
//! 変更でありユーザー承認必須）は #36 のスコープ外として未着手のまま
//! 残す（`tests/cpu_cuda_parity.rs` 冒頭コメント参照）。
//! TASK-1.9b（#45）で [`memory`] モジュール（[`memory::CudaMemory`]）を
//! 追加した。`tensor_core::buffer::MemoryOps` の CUDA 実装であり、
//! `CudaDevice` 経由でのみ構築できるため上記の panic 回避ゲートを共有
//! する。既存の `gemm.rs`（`clone_htod`/`alloc_zeros`/`clone_dtoh`）は
//! 演算内部にホスト⇔デバイス転送を抱えたままとし、本イシューでは
//! 載せ替えを行わない（TASK-1.9c・#46 のスコープ）。`BackendOps`
//! トレイト自体（カーネルディスパッチ）へのフルマッピングも
//! TASK-1.9c（#46）のスコープであり、本クレートではまだ扱わない
//! （spec 根拠: `docs/spec/05-tasks.md` TASK-1.7・TASK-1.9）。
//!
//! TASK-11.1b（#61）で f16 Tensor Core（WMMA）GEMM カーネル
//! （[`CudaWmmaGemm`]）を追加した。設計は `docs/cuda-tensor-core-design.md`
//! （#60）で確定済み（方式 A: `#include <mma.h>` の WMMA C++ API・
//! `m16n16k16` fragment・f32 アキュムレート）。naive／tiled 経路
//! （`kernels.rs`／`gemm.rs`）とは別ファイル（`kernels_wmma.rs`／
//! `gemm_wmma.rs`）に分離している。
//!
//! TASK-11.1c（#62）で WMMA（Tensor Core）を用いた TF32/f32 GEMM
//! （[`CudaGemm::run_wmma_tf32`]）を追加した。設計は `docs/cuda-tensor-core-design.md`
//! （#60）を正本とし、fragment `m16n16k8`（TF32 精度・f32 累算）・方式 A
//! （WMMA C++ API `<mma.h>`）を採用する（REQ-11）。TF32 は f32 の仮数部
//! 23bit を 10bit に丸めて Tensor Core へ投入するため、統一複合判定
//! （相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）は TF32 前提の複合指標
//! として適用する（REQ-2、`.claude/rules/coding-rust.md`）。f16 WMMA 経路
//! （#61）とは異なり、TF32 経路は naive／tiled 経路と同じ `kernels.rs`／
//! `gemm.rs` に実装している。
//! TASK-11.1d（#63）で WMMA(TF32)／f16 WMMA の共有メモリ・タイル最適化版
//! カーネル（`kernels_wmma_opt::WMMA_TF32_F32_OPT`／`WMMA_F16_OPT`）を追加
//! した。ブロックタイル 64×64・warp あたり fragment 2×2 個（レジスタ
//! ブロッキング）・バンクコンフリクト回避パディング・`__syncthreads()`
//! ベースのダブルバッファリングを適用する（設計は `docs/cuda-tensor-core-
//! design.md` 4.2 節を正本とする）。公開 API（`CudaGemm::run_wmma_tf32`／
//! `CudaWmmaGemm::run_f16`）のシグネチャは変更せず、opt カーネルが
//! `new` 時点でコンパイル・ロードに成功していれば優先的に使用し、失敗
//! していれば #61/#62 の基本 WMMA カーネルへ自動フォールバックする
//! （`kernels_wmma_opt.rs` 冒頭ドキュメントコメント参照）。実機実測での
//! 数値一致・性能確定は #64 のスコープ。
//!
//! TASK-11.1h（#187）で `mma.sync`/`ldmatrix`/`cp.async` PTX 直叩き経路
//! （[`CudaMmaGemm`]）を追加した。WMMA 経路（cc>=7.0）より厳しい
//! compute capability 8.0+ ゲートを持つ独立経路であり、`kernels_mma.rs`／
//! `gemm_mma.rs` に分離している（並行 issue #62/#63 が `gemm.rs`／
//! `gemm_wmma.rs`／`kernels.rs`／`kernels_wmma.rs` を編集中のため）。
//! XOR swizzle によるバンクコンフリクト低減は未実装（`kernels_mma.rs`
//! 冒頭コメント「タイル構成」参照。コンパイル未検証環境でのリスク
//! 最小化判断）。
//!
//! ディスパッチ規則（naive／tiled／f16 WMMA／TF32 WMMA／`mma.sync` の
//! どの経路をいつ選ぶか）は TASK-11.2（#66）のスコープであり本クレートでは
//! 未実装。
//!
//! TASK-11.2b（#68）で GEMM 自動経路選択の入口（[`CudaGemmAuto`]）を
//! 追加した。`tensor_core::dispatch::select_gemm_kernel`（#67 が設計した
//! 決定的規則。`docs/dispatch-rules-design.md`）の結果に従い、naive／
//! tiled（`CudaGemm`）・WMMA f16（`CudaWmmaGemm`）を呼び分ける。TF32/f32
//! 経路（`CudaGemm::run_wmma_tf32`・#62）・`mma.sync` 経路（`CudaMmaGemm`・
//! #187）は、決定表（設計文書 §4）が TF32 既定採用を #186（TASK-11.1g）の
//! ユーザー承認まで保留と定めているため、現時点の `select_gemm_kernel` の
//! 自動経路には含めない（f32 は常に Tiled）。既存の `CudaGemm`／
//! `CudaWmmaGemm`／`CudaMmaGemm` の直接指定 API はテスト・証跡用途
//! （#70）にそのまま温存する（設計文書 §5.4）。
//!
//! TASK-1.9c（#46）で [`ops`] モジュール（[`ops::CudaBackendOps`]）を追加した。
//! `tensor_core::backend_ops::BackendOps` の CUDA 実装であり、`gemm` は
//! [`CudaGemm::run_tiled_f32`] へ委譲する（既定カーネル変種の選択は保守的に
//! tiled 固定とし、`CudaGemmAuto` を介した Tensor Core 経路の自動選択への
//! 切替は別スコープ）。elementwise・reduction は GPU カーネル未実装のため
//! `tensor_core::device::BackendError::Unsupported` を返す
//! （out-of-scope-tracking.md 対象）。

pub mod device;
mod error;
mod gemm;
mod gemm_auto;
mod gemm_mma;
mod gemm_wmma;
mod kernels;
mod kernels_mma;
mod kernels_wmma;
mod kernels_wmma_opt;
pub mod memory;
mod nvrtc;
mod ops;

pub use device::{CudaDevice, CudaDeviceProvider};
pub use error::CudaError;
pub use gemm::CudaGemm;
pub use gemm_auto::CudaGemmAuto;
pub use gemm_mma::CudaMmaGemm;
pub use gemm_wmma::CudaWmmaGemm;
pub use memory::CudaMemory;
pub use nvrtc::compile_ptx;
pub use ops::CudaBackendOps;
