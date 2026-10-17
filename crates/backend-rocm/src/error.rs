//! backend-rocm の型付きエラー。
//!
//! `hip` モジュール（HIP ランタイムの動的ロード）が返し、`lib.rs` の probe が false へ
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
        }
    }
}

impl std::error::Error for RocmError {}
