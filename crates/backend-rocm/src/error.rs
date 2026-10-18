//! backend-rocm の型付きエラー。
//!
//! `hip`・`hiprtc`・`gemm` 等のモジュールが返し、`lib.rs` の probe が false へ
//! 畳み込む。本番経路で panic せずエラー値で返す契約
//! （docs/backend-rocm-hip-ffi-decision.md §4-6）。

use std::fmt;

use fandhe_ai_tensor_core::BackendError;

/// ROCm バックエンドのエラー。
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum RocmError {
    /// HIP ランタイムの共有ライブラリを開けなかった（ROCm 非搭載を含む）。
    LibraryUnavailable {
        /// 試した固定ライブラリ名（空なら当該 OS に候補がない）。
        tried: Vec<&'static str>,
        /// ローダーが返したエラー文字列。
        detail: String,
    },
    /// 必須シンボルを解決できなかった。
    SymbolMissing {
        /// シンボル名。
        symbol: &'static str,
        /// ローダーが返したエラー文字列。
        detail: String,
    },
    /// HIP API がエラーコードを返した。
    Hip {
        /// 呼び出した HIP 関数名。
        call: &'static str,
        /// `hipError_t` の値。
        code: i32,
    },
    /// HIPRTC API がエラーコードを返した（コンパイル失敗以外）。
    Hiprtc {
        /// 呼び出した HIPRTC 関数名。
        call: &'static str,
        /// `hiprtcResult` の値。
        code: i32,
    },
    /// HIPRTC のコンパイルが失敗した。
    CompileFailed {
        /// `hiprtcResult` の値。
        code: i32,
        /// コンパイラログ（上限で切り詰め済み）。
        log: String,
    },
    /// コードオブジェクトのサイズが 0 だった。
    CodeObjectEmpty,
    /// カーネルのタイルパラメータが不正（fail-closed）。
    InvalidKernelConfig {
        /// 理由。
        detail: String,
    },
    /// GEMM の形状・スライス長が不正。
    InvalidGemmDims {
        /// 理由。
        detail: String,
    },
    /// 許可リストにない gfx target。
    UnsupportedTarget {
        /// 理由。
        detail: String,
    },
    /// HIP デバイスが 1 つもない。
    NoDevice,
    /// 文字列に NUL を含むなど、C 文字列へ変換できない。
    InvalidCString {
        /// 対象の説明。
        what: &'static str,
    },
}

impl fmt::Display for RocmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LibraryUnavailable { tried, detail } => write!(
                f,
                "HIP ランタイムをロードできません（試行: {tried:?}）: {detail}"
            ),
            Self::SymbolMissing { symbol, detail } => {
                write!(f, "HIP シンボル {symbol} を解決できません: {detail}")
            }
            Self::Hip { call, code } => write!(f, "HIP API {call} がエラー {code} を返しました"),
            Self::Hiprtc { call, code } => {
                write!(f, "HIPRTC API {call} がエラー {code} を返しました")
            }
            Self::CompileFailed { code, log } => {
                write!(f, "HIPRTC のコンパイルに失敗しました（{code}）: {log}")
            }
            Self::CodeObjectEmpty => write!(f, "HIPRTC が空のコードオブジェクトを返しました"),
            Self::InvalidKernelConfig { detail } => {
                write!(f, "カーネルのタイル設定が不正です: {detail}")
            }
            Self::InvalidGemmDims { detail } => write!(f, "GEMM の形状が不正です: {detail}"),
            Self::UnsupportedTarget { detail } => {
                write!(f, "未対応の gfx target です: {detail}")
            }
            Self::NoDevice => write!(f, "HIP デバイスが見つかりません"),
            Self::InvalidCString { what } => {
                write!(f, "{what} に NUL 文字が含まれています")
            }
        }
    }
}

impl std::error::Error for RocmError {}

/// [`RocmError`] を `tensor-core` の `BackendError` へ写す（`ops.rs` の `BackendOps` 実装が使う）。
///
/// 公開の `From` 実装にはしない（公開面を増やさない）。同一クレート内で網羅 `match` とし、
/// variant の追加をコンパイルで検出する。
///
/// | `RocmError` | `BackendError` |
/// |---|---|
/// | `LibraryUnavailable`・`SymbolMissing`・`NoDevice`・`UnsupportedTarget` | `DeviceUnavailable` |
/// | 上記以外（HIP/HIPRTC 失敗・不正な設定・形状） | `KernelLaunchFailed` |
pub(crate) fn to_backend_error(e: RocmError) -> BackendError {
    let msg = e.to_string();
    match e {
        RocmError::LibraryUnavailable { .. }
        | RocmError::SymbolMissing { .. }
        | RocmError::NoDevice
        | RocmError::UnsupportedTarget { .. } => BackendError::DeviceUnavailable(msg),
        RocmError::Hip { .. }
        | RocmError::Hiprtc { .. }
        | RocmError::CompileFailed { .. }
        | RocmError::CodeObjectEmpty
        | RocmError::InvalidKernelConfig { .. }
        | RocmError::InvalidGemmDims { .. }
        | RocmError::InvalidCString { .. } => BackendError::KernelLaunchFailed(msg),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_backend_error_maps_every_variant() {
        let unavailable = [
            RocmError::LibraryUnavailable {
                tried: vec![],
                detail: String::new(),
            },
            RocmError::SymbolMissing {
                symbol: "s",
                detail: String::new(),
            },
            RocmError::NoDevice,
            RocmError::UnsupportedTarget {
                detail: String::new(),
            },
        ];
        for e in unavailable {
            assert!(matches!(
                to_backend_error(e),
                BackendError::DeviceUnavailable(_)
            ));
        }
        let launch = [
            RocmError::Hip { call: "c", code: 1 },
            RocmError::Hiprtc { call: "c", code: 1 },
            RocmError::CompileFailed {
                code: 1,
                log: String::new(),
            },
            RocmError::CodeObjectEmpty,
            RocmError::InvalidKernelConfig {
                detail: String::new(),
            },
            RocmError::InvalidGemmDims {
                detail: String::new(),
            },
            RocmError::InvalidCString { what: "w" },
        ];
        for e in launch {
            assert!(matches!(
                to_backend_error(e),
                BackendError::KernelLaunchFailed(_)
            ));
        }
    }
}
