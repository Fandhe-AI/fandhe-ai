//! `backend-cuda` の型付きエラー。
//!
//! `device.rs`（デバイス初期化）・`nvrtc.rs`（NVRTC コンパイル）が返す
//! 失敗をすべて `CudaError` に集約する。TASK-1.9（#43/#44）で
//! `BackendOps`/`BackendError`（`docs/public-api-design.md` §4.4）が
//! 導入された際は、呼び出し元（backend 抽象層）が本型を
//! `BackendError::CudaUnavailable(String)` 等へマップする想定である
//! （本イシューではマップ側は実装しない）。

use std::fmt;

/// CUDA バックエンドの初期化・コンパイル失敗を表す。
///
/// `#[non_exhaustive]` を付す理由: 公開 API 非破壊はガードレール条件
/// （`.claude/rules/security.md`）であり、#33/#34（カーネル起動・
/// メモリ転送）で variant が増えても呼び出し側の網羅的 match を
/// 破壊しないため。
#[non_exhaustive]
#[derive(Debug)]
pub enum CudaError {
    /// `libcuda` が動的リンカから見つからず `dlopen` できない
    /// （CUDA 非搭載環境。受け入れ条件「CUDA 非搭載環境で実行時に
    /// 型付きエラーが返る（panic しない）」の主役）。
    ///
    /// `cudarc` 0.19.8 の `dynamic-loading` feature は、この状態で
    /// driver API（`CudaContext::new` 等）を直接呼ぶと `Err` ではなく
    /// panic する（`culib()` が `panic_no_lib_found` を呼ぶ。
    /// cudarc-0.19.8/src/driver/sys/mod.rs:16119-16129、src/lib.rs:199-200）。
    /// `CudaDevice::new`/`device_count` は本 variant を返す前に必ず
    /// `cudarc::driver::sys::is_culib_present()` で存在確認し、
    /// 不在ならこの panic 経路に入る前に `Err` を返す（device.rs 参照）。
    DriverUnavailable { detail: String },

    /// `libnvrtc` が動的リンカから見つからず `dlopen` できない。
    ///
    /// driver 側と同様、NVRTC 側にも `is_culib_present` による
    /// panic 回避プローブが用意されている
    /// （cudarc-0.19.8/src/nvrtc/sys/mod.rs:529、540、550）。
    /// `compile_ptx`（nvrtc.rs）はコンパイル呼び出し前に必ずこれを
    /// 確認する。
    NvrtcUnavailable { detail: String },

    /// `cuInit`／コンテキスト生成／デバイス問い合わせ等、driver API
    /// 呼び出し自体が `Err` を返したケース（`libcuda` は存在するが
    /// 実行時エラーが発生した場合。例: ドライババージョン不一致）。
    Driver(cudarc::driver::result::DriverError),

    /// NVRTC コンパイル失敗（構文エラー・`--include-path` 未解決等）。
    Compile(cudarc::nvrtc::CompileError),

    /// GEMM 起動 API（`gemm.rs`）のホスト側形状検証が拒否した入力。
    ///
    /// `run_naive_f32`／`run_naive_f16` は GPU へバッファを転送しカーネルを
    /// 起動する前に、スライス長が `m*k`／`k*n` と一致するか・`m`/`n`/`k` の
    /// 積が `usize`/`i32` の範囲でオーバーフローしないかを検証する
    /// （`gemm.rs::validate_gemm_dims` 参照）。この検証は GPU 起動前の
    /// A03 インジェクション対策（外部由来の形状値を境界チェックなしに
    /// カーネル引数へ渡さない。`.claude/rules/security.md`）を兼ねる。
    InvalidShape { detail: String },
}

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaError::DriverUnavailable { detail } => {
                write!(f, "CUDA driver library unavailable: {detail}")
            }
            CudaError::NvrtcUnavailable { detail } => {
                write!(f, "CUDA NVRTC library unavailable: {detail}")
            }
            // `cudarc::driver::result::DriverError`／`cudarc::nvrtc::CompileError`
            // は本クレートが依存する `cudarc` の feature 構成（driver／nvrtc／
            // dynamic-loading／f16。deps-policy.md 記載の 4 feature に限定し
            // `std` feature は有効化しない）では `std::fmt::Display`／
            // `std::error::Error` を実装しない（`std` feature でのみ
            // `#[cfg(feature = "std")]` 実装が有効になる。cudarc-0.19.8
            // src/driver/result.rs:73-81）。そのため `Debug` 表示に委ねる
            // （PoC-v2-3 の `CudaGemmError` と同じ方針。cuda/mod.rs:48・54）。
            CudaError::Driver(e) => write!(f, "cuda driver error: {e:?}"),
            CudaError::Compile(e) => write!(f, "nvrtc compile error: {e:?}"),
            CudaError::InvalidShape { detail } => {
                write!(f, "invalid GEMM shape: {detail}")
            }
        }
    }
}

// 上記と同じ理由（`cudarc` を `std` feature なしで利用）で
// `DriverError`／`CompileError` は `std::error::Error` を実装しないため
// `source()` で辿ることができない。`Display` に整形済みメッセージを
// 含めているため、既定の `source() -> None` のままで足りる。
impl std::error::Error for CudaError {}

// `?` で `device.rs`／`nvrtc.rs` から `CudaError` へそのまま伝播できるよう、
// cudarc が返す 2 種類のエラー型（driver API 由来・NVRTC コンパイル由来）
// それぞれから個別に変換する（PoC-v2-3 の CudaGemmError と同じ方針。
// `std::error::Error` 全体へのブランケット実装は標準ライブラリの
// `impl<T> From<T> for T` と衝突するため採らない）。
impl From<cudarc::driver::result::DriverError> for CudaError {
    fn from(e: cudarc::driver::result::DriverError) -> Self {
        CudaError::Driver(e)
    }
}

impl From<cudarc::nvrtc::CompileError> for CudaError {
    fn from(e: cudarc::nvrtc::CompileError) -> Self {
        CudaError::Compile(e)
    }
}
