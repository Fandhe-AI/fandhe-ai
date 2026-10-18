//! ROCm（AMD GPU）バックエンドクレート（イシュー #3154・#3155・親 #3150）。
//!
//! 役割: HIP ランタイムの有無を probe し、HIPRTC でコンパイルした f32 GEMM カーネル
//! （`backend-cuda` の `TILED_F32` の移植）を実行する。HIP/HIPRTC は `libloading`（第 12 区分）で
//! 実行時にロードし、`#[link]` を使わないため ROCm 非搭載環境でもビルド・テストが通る
//! （cudarc の dynamic-loading と同じ cfg ベース方針。REQ-2・coding-rust.md）。
//! 線形代数 6 演算（`inv`・`solve`・`det`・`cholesky`・`qr`・`eigh`。#3444）は内部 opt-in ゲート（[`linalg`]。既定 OFF）の背後にある `f64` カーネルで、`RocmLinalgLu`・`RocmLinalgFactor` が実行する。
//! 縮約（R3・#3434）として sum/max/min（[`RocmReduce`]。wave 幅は `WarpGeometry` で注入）と
//! argmax/argmin（[`RocmArgReduce`]。wave 幅非依存）の HIPRTC カーネルを `RocmBackendOps` へ結線した。
//! `BackendOps` の骨格（必須 9 メソッド。[`RocmBackendOps`]）は R0（#3431）、縮約系カーネル向けの
//! wave 幅注入部品は `WarpGeometry`（#3432）、facade の
//! Device 追加は #3157（所有者判断待ち）、unsafe 監査記録は #3158、実機 parity は #3159。
//!
//! 添字系演算（分解案 R8・#3439）として cumsum/cumprod（[`RocmScan`]）・gather/scatter/one_hot
//! （[`RocmGatherScatter`]）・interpolate の nearest/bilinear（[`RocmInterpolate`]）・sort/topk
//! （[`RocmSort`]）・unique（[`RocmUnique`]）の HIPRTC カーネルを移植した。いずれもホストのスライスを
//! 受けて `Vec` を返す実行型で、`BackendOps` への結線は `RocmBackendOps`（#3431・R0）で行う。
//!
//! 正規化（分解案 R6・#3437）として rmsnorm・layer_norm・両 backward・batch_norm の train/infer の
//! HIPRTC カーネル（[`RocmNorm`]。wave 幅は `WarpGeometry` で注入）を移植し、`RocmBackendOps` の
//! 6 メソッドへ結線した。実機での HIPRTC コンパイル・wave64 parity は #3159。
//!
//! 設計の正本は `docs/backend-rocm-hip-ffi-decision.md`（FFI）と
//! `docs/backend-rocm-hiprtc-gemm-decision.md`（HIPRTC・GEMM・FMA・wave64）。
//! facade からは参照しない。

mod arg_reduce;
#[cfg(test)]
mod arg_reduce_model;
mod error;
mod gather_scatter;
#[cfg(test)]
mod gather_scatter_model;
mod gemm;
#[cfg(test)]
mod gemm_model;
mod hip;
mod hiprtc;
mod interpolate;
#[cfg(test)]
mod interpolate_model;
mod kernel_source;
mod kernels_arg_reduce;
mod kernels_gather_scatter;
mod kernels_gemm;
mod kernels_interpolate;
mod kernels_linalg;
mod kernels_linalg_factor;
mod kernels_norm;
mod kernels_reduce;
mod kernels_scan;
mod kernels_sort;
mod kernels_unique;
mod launch;
pub mod linalg;
#[cfg(test)]
mod linalg_model;
mod module;
mod norm;
#[cfg(test)]
mod norm_model;
mod ops;
mod reduce;
#[cfg(test)]
mod reduce_model;
mod scan;
#[cfg(test)]
mod scan_model;
mod shape_util;
mod sort;
#[cfg(test)]
mod sort_model;
mod target;
mod typed_buffer;
mod unique;
#[cfg(test)]
mod unique_model;
mod warp_geometry;
#[cfg(test)]
mod warp_model;

pub use arg_reduce::RocmArgReduce;
pub use error::RocmError;
pub use gather_scatter::{RocmGatherScatter, ScatterMode};
pub use gemm::RocmGemmF32;
pub use interpolate::RocmInterpolate;
pub use kernels_gemm::{GemmF32Tile, GemmSource, render_gemm_f32_source};
pub use linalg::{LuSolveOutcome, RocmLinalgFactor, RocmLinalgLu};
pub use norm::{BatchNormTrainHost, LayerNormBackwardHost, RmsNormBackwardHost, RocmNorm};
pub use ops::RocmBackendOps;
pub use reduce::RocmReduce;
pub use scan::RocmScan;
pub use sort::RocmSort;
pub use target::{GfxTarget, parse_gcn_arch_name};
pub use unique::RocmUnique;
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
