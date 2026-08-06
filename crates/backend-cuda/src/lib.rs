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
//! `BackendOps`/`BackendError` へのフルマッピング（カーネル起動・メモリ転送）は
//! TASK-1.9c（#46）のスコープであり、本クレートでは扱わない
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
//! `gemm.rs` に実装している（着手時点で #61 が未マージであったため独立に
//! WMMA 共通基盤〈NVRTC `<mma.h>` include パス解決・カーネル起動 API の骨格〉
//! を最小実装した経緯による。ファイル分割方針の統一は必須ではないが、
//! 追加バックエンド作業時の参照コストを避けるため、統一要否の判断は
//! ディスパッチ規則の実装（#66）と併せて検討する）。
//! ディスパッチ規則（naive／tiled／f16 WMMA／TF32 WMMA のどの経路を
//! いつ選ぶか）は TASK-11.2（#66）のスコープであり本クレートでは未実装。
//! 共有メモリ・タイル基本最適化（レジスタブロッキング・ダブルバッファ
//! リング・ベクトル化ロード）は #63、実機実測での数値一致検証は #64、
//! `mma.sync` PTX 直叩きは #187 のスコープであり、いずれも本クレートは
//! 経路の提供までを扱う（`docs/cuda-tensor-core-design.md` 参照）。

pub mod device;
mod error;
mod gemm;
mod gemm_wmma;
mod kernels;
mod kernels_wmma;
mod nvrtc;

pub use device::{CudaDevice, CudaDeviceProvider};
pub use error::CudaError;
pub use gemm::CudaGemm;
pub use gemm_wmma::CudaWmmaGemm;
pub use nvrtc::compile_ptx;
