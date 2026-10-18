//! backend-rocm の型付きエラー。
//!
//! `hip`・`hiprtc`・`gemm` 等のモジュールが返し、`lib.rs` の probe が false へ
//! 畳み込む。本番経路で panic せずエラー値で返す契約
//! （docs/backend-rocm-hip-ffi-decision.md §4-6）。

use std::fmt;

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
    /// GEMM 以外の演算（gather/scatter/sort 等）の引数・形状・スライス長が不正。
    InvalidOpArgs {
        /// 演算名（固定文字列）。
        op: &'static str,
        /// 理由。
        detail: String,
    },
    /// 演算のサイズがカーネル引数型（i32 等）・バッファ上限を超えた。
    SizeLimitExceeded {
        /// 演算名（固定文字列）。
        op: &'static str,
        /// 超過した値。
        n: usize,
        /// 上限。
        limit: usize,
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
            Self::InvalidOpArgs { op, detail } => write!(f, "{op} の引数が不正です: {detail}"),
            Self::SizeLimitExceeded { op, n, limit } => {
                write!(f, "{op} のサイズ {n} が上限 {limit} を超えます")
            }
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
