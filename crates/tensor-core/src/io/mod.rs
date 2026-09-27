//! NumPy 互換フォーマット（`.npy`／`.npz`）の読み書き（イシュー #2189・
//! 親 #2131）。
//!
//! 本モジュールは `Tensor<f32>` を NumPy の `.npy`（単一配列）・`.npz`
//! （ZIP コンテナに複数配列をまとめたもの）形式で保存・読み込みする
//! ためのホスト側専用 IO を提供する。完全自作コア方針（REQ-1・
//! `.claude/rules/coding-rust.md`）に従い、`tensor-core` の直接依存は
//! `half` のみであるため、ZIP コンテナの解析（EOCD／zip64／central
//! directory／local header）・CRC-32・DEFLATE 伸長はすべて本クレート
//! 内で自作する（依存追加なし）。
//!
//! GPU カーネル・`BackendOps`／`Op`（演算グラフ）とは無関係な、
//! ホスト常駐 `Tensor<f32>` に閉じた IO であり、`autodiff`・
//! `backend-*` に新たな契約を課さない。
//!
//! **facade への公開は保留中**（`crates/facade/src/lib.rs::
//! NpyIoHoldDoctestGuard`。承認事項の詳細は
//! `docs/tensor-core-npy-npz-io-decision.md` を参照）。#2156（RNG 確率
//! 分布サンプラー）の前例に倣い、`Tensor` への inherent メソッドは
//! 追加しない（facade が `pub use fandhe_ai_tensor_core::{..., Tensor,
//! ...};` で `Tensor` を再エクスポートしているため、inherent メソッドは
//! それだけで facade の公開面を広げてしまう。`docs/rng-distributions-
//! generator-decision.md:28` 参照）。
//!
//! 外部フォーマット（ファイル）を扱うため、形式不正・非対応 dtype・
//! 改ざん・巨大サイズ宣言はすべて確保の前に検証し fail-closed で
//! 拒否する（`.claude/rules/security.md` A03/A04/A05/A08）。

pub mod crc32;
pub mod inflate;
pub mod npy;
pub mod npz;

use std::fmt;

use crate::error::ShapeError;

/// npy／npz の読み書きで発生しうるエラー。
///
/// `#[non_exhaustive]` を付す理由は `ShapeError` と同じ（公開 API
/// 非破壊はガードレール条件。`.claude/rules/security.md`）。
/// `std::io::Error` を内包するため `Clone`／`PartialEq` は derive
/// しない（`ShapeError` とは異なる点。呼び出し側は `Display`／
/// `source()` で比較する）。
#[non_exhaustive]
#[derive(Debug)]
pub enum NpyError {
    /// ファイル IO（`std::fs::read`／`std::fs::write`）の失敗。
    Io(std::io::Error),

    /// npy magic（`\x93NUMPY`）が一致しない。
    InvalidMagic,

    /// npy バージョンが (1,0)／(2,0)／(3,0) のいずれでもない。
    UnsupportedVersion { major: u8, minor: u8 },

    /// ヘッダ長が上限（NumPy の `_MAX_HEADER_SIZE` 相当・10000 バイト）
    /// を超える、またはヘッダ全体が入力バイト列に収まらない。
    HeaderTooLarge { len: usize, max: usize },

    /// npy ヘッダ辞書（`descr`／`fortran_order`／`shape`）の構文・
    /// キー集合が専用パーサの受理形（`docs/tensor-core-npy-npz-io-
    /// decision.md` 参照）と一致しない。理由は静的な文字列に限定し、
    /// 入力バイト列そのものは埋め込まない（`.claude/rules/security.md`
    /// A09）。
    InvalidHeader(&'static str),

    /// dtype 記述子が `<f4`／`>f4` 以外（構造化配列・object・他の数値型
    /// を含む）。descr は長さを切り詰めて保持する。
    UnsupportedDtype { descr: String },

    /// shape から求めた要素数バイト長と実データ長が一致しない。
    DataLengthMismatch { expected: usize, actual: usize },

    /// `Tensor::new`／`permute`／`contiguous` 等の shape 検査エラー。
    Shape(ShapeError),

    /// ZIP コンテナの構造（EOCD／central directory／local header 等）が
    /// 不正。理由は静的な文字列に限定する。
    InvalidZip(&'static str),

    /// ZIP エントリの圧縮方式が STORED（0）／DEFLATE（8）以外。
    UnsupportedCompression { method: u16 },

    /// 暗号化・マルチディスク等、本 IO が対応しない ZIP 機能。
    UnsupportedZipFeature(&'static str),

    /// ZIP エントリの CRC-32 が伸長後データと一致しない
    /// （`.claude/rules/security.md` A08。改ざん・破損の検出）。
    CrcMismatch { name: String },

    /// npz 内で同じキー（`.npy` サフィックスを除いたエントリ名）が
    /// 重複している。
    DuplicateEntry { name: String },

    /// npz エントリ名が空・NUL を含む・UTF-8 でない等、キーとして不正。
    InvalidEntryName,

    /// npz の特定メンバの処理中に発生したエラー。`name` はどのメンバで
    /// 失敗したかを示し、`source` は元のエラーを包む。1 メンバでも
    /// 失敗すれば npz 読み込み全体を失敗させ、部分的な `HashMap` は
    /// 返さない（fail-closed）。
    Entry { name: String, source: Box<NpyError> },

    /// npz 書き出し時、1 エントリの非圧縮サイズが `u32::MAX` を超える
    /// （zip64 書き出しは対象外。`docs/tensor-core-npy-npz-io-decision.md`
    /// の制限一覧を参照）。
    EntryTooLarge,

    /// npz 書き出し時、エントリ数が `u16::MAX` を超える。
    TooManyEntries,

    /// DEFLATE ストリームが RFC 1951 の構文・整合性検査に反する
    /// （不正なブロック型・over-subscribed Huffman 符号・距離が出力
    /// 範囲外・宣言長との不一致等）。理由は静的な文字列に限定する。
    InvalidDeflate(&'static str),
}

impl fmt::Display for NpyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NpyError::Io(e) => write!(f, "IO エラー: {e}"),
            NpyError::InvalidMagic => write!(f, "npy magic が不正（\\x93NUMPY ではない）"),
            NpyError::UnsupportedVersion { major, minor } => {
                write!(f, "npy バージョン {major}.{minor} は非対応")
            }
            NpyError::HeaderTooLarge { len, max } => {
                write!(f, "npy ヘッダ長 {len} バイトが上限 {max} バイトを超える")
            }
            NpyError::InvalidHeader(reason) => write!(f, "npy ヘッダが不正: {reason}"),
            NpyError::UnsupportedDtype { descr } => {
                write!(f, "非対応の dtype 記述子: {descr}")
            }
            NpyError::DataLengthMismatch { expected, actual } => write!(
                f,
                "データ長不一致（shape から期待 {expected} バイト、実際 {actual} バイト）"
            ),
            NpyError::Shape(e) => write!(f, "shape エラー: {e}"),
            NpyError::InvalidZip(reason) => write!(f, "ZIP コンテナが不正: {reason}"),
            NpyError::UnsupportedCompression { method } => {
                write!(f, "非対応の圧縮方式（method={method}）")
            }
            NpyError::UnsupportedZipFeature(feature) => {
                write!(f, "非対応の ZIP 機能: {feature}")
            }
            NpyError::CrcMismatch { name } => write!(f, "CRC-32 不一致（エントリ: {name}）"),
            NpyError::DuplicateEntry { name } => write!(f, "重複エントリ: {name}"),
            NpyError::InvalidEntryName => write!(f, "npz エントリ名が不正"),
            NpyError::Entry { name, source } => write!(f, "エントリ {name} の処理に失敗: {source}"),
            NpyError::EntryTooLarge => write!(f, "npz エントリのサイズが上限を超える"),
            NpyError::TooManyEntries => write!(f, "npz エントリ数が上限を超える"),
            NpyError::InvalidDeflate(reason) => write!(f, "DEFLATE ストリームが不正: {reason}"),
        }
    }
}

impl std::error::Error for NpyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            NpyError::Io(e) => Some(e),
            NpyError::Shape(e) => Some(e),
            NpyError::Entry { source, .. } => Some(source),
            _ => None,
        }
    }
}

impl From<std::io::Error> for NpyError {
    fn from(e: std::io::Error) -> Self {
        NpyError::Io(e)
    }
}

impl From<ShapeError> for NpyError {
    fn from(e: ShapeError) -> Self {
        NpyError::Shape(e)
    }
}

/// ヘッダ・ZIP メタデータ解析の境界付き読み取りヘルパ（`pub(crate)`）。
///
/// すべて `checked_add` 相当の範囲検査を先に行ってから読み取り、
/// 範囲外アクセスは `panic` ではなく `Result` で通知する
/// （`.claude/rules/security.md` A03「外部フォーマットパースは長さ・
/// 形状の検証を先に行う」）。
pub(crate) mod bounded {
    use super::NpyError;

    /// `bytes[offset..]` から幅 `len` のスライスを取り出す。範囲外は
    /// `err` を静的理由として `NpyError::InvalidZip` を返す。
    pub(crate) fn slice_at<'a>(
        bytes: &'a [u8],
        offset: usize,
        len: usize,
        err: &'static str,
    ) -> Result<&'a [u8], NpyError> {
        let end = offset.checked_add(len).ok_or(NpyError::InvalidZip(err))?;
        bytes.get(offset..end).ok_or(NpyError::InvalidZip(err))
    }

    /// リトルエンディアン `u16` を `offset` から読む。
    pub(crate) fn read_u16_le(
        bytes: &[u8],
        offset: usize,
        err: &'static str,
    ) -> Result<u16, NpyError> {
        let s = slice_at(bytes, offset, 2, err)?;
        Ok(u16::from_le_bytes([s[0], s[1]]))
    }

    /// リトルエンディアン `u32` を `offset` から読む。
    pub(crate) fn read_u32_le(
        bytes: &[u8],
        offset: usize,
        err: &'static str,
    ) -> Result<u32, NpyError> {
        let s = slice_at(bytes, offset, 4, err)?;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }

    /// リトルエンディアン `u64` を `offset` から読む。
    pub(crate) fn read_u64_le(
        bytes: &[u8],
        offset: usize,
        err: &'static str,
    ) -> Result<u64, NpyError> {
        let s = slice_at(bytes, offset, 8, err)?;
        Ok(u64::from_le_bytes([
            s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7],
        ]))
    }
}
