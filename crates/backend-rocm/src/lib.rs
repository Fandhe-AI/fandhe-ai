//! ROCm（AMD GPU）バックエンドクレート（イシュー #3154・#3155・親 #3150）。
//!
//! 役割: HIP ランタイムの有無を probe し、HIPRTC でコンパイルした f32 GEMM カーネル
//! （`backend-cuda` の `TILED_F32` の移植）を実行する。HIP/HIPRTC は `libloading`（第 12 区分）で
//! 実行時にロードし、`#[link]` を使わないため ROCm 非搭載環境でもビルド・テストが通る
//! （cudarc の dynamic-loading と同じ cfg ベース方針。REQ-2・coding-rust.md）。
//! 縮約系カーネル向けの wave 幅注入部品は `WarpGeometry`（#3432）。
//! BackendOps への統合・facade の Device 追加は #3157、unsafe 監査記録は #3158、
//! 実機 parity は #3159。
//!
//! 設計の正本は `docs/backend-rocm-hip-ffi-decision.md`（FFI）と
//! `docs/backend-rocm-hiprtc-gemm-decision.md`（HIPRTC・GEMM・FMA・wave64）。
//! facade からは参照しない。

mod error;
mod gemm;
#[cfg(test)]
mod gemm_model;
mod hip;
mod hiprtc;
mod kernels_gemm;
mod module;
mod target;
mod warp_geometry;
#[cfg(test)]
mod warp_model;

pub use error::RocmError;
pub use gemm::RocmGemmF32;
pub use kernels_gemm::{GemmF32Tile, GemmSource, render_gemm_f32_source};
pub use target::{GfxTarget, parse_gcn_arch_name};
pub use warp_geometry::{WarpDefines, WarpGeometry};

/// HIP ランタイムをロードでき、デバイスが 1 つ以上あるとき true。
///
/// ROCm 非搭載・ロード失敗・デバイス 0・非対応 OS ではすべて false を返し、panic しない。
/// HIPRTC の有無は含めない（コンパイル経路が型付きエラーを返す）。
pub fn is_available() -> bool {
    hip::load()
        .and_then(|rt| rt.device_count())
        .map(|n| n >= 1)
        .unwrap_or(false)
}
