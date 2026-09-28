//! ONNX external data（initializer／Constant 属性テンソルが `.onnx` 本体の
//! 外に置く `.onnx.data` ファイル。onnx.proto3 `TensorProto.data_location`
//! （tag=14）・`external_data`（tag=13））を fail-closed に解決し、
//! `raw_data` へ inline 化する（イシュー #2347）。
//!
//! ## 背景・呼び出し文脈
//!
//! PyTorch の既定 exporter（`torch.onnx.export(..., dynamo=True)`）は
//! initializer を external data として出力する（432 バイト程度の小さな
//! テンソルでも external になる。#2329 実測）。`graph::decode_tensor` は
//! `data_location`／`external_data` を一切参照しないため（バイト列入口の
//! 契約は不変。下記「不変条件」節）、external data を持つモデルはそのまま
//! では `GraphError::RawDataByteLenMismatch`（raw_data・float_data とも
//! 空）で拒否される。本モジュールは、外部ファイルを解決する基点
//! ディレクトリ（`base_dir`）を受け取る新しい入口
//! （[`build_graph_with_external_data`]）を提供し、`ModelProto` を複製した
//! うえで external なテンソルを `raw_data` へ inline 化してから、変更して
//! いない `graph::build_graph` へ渡す（「全参照を検証 → 範囲を限定して
//! 読込 → raw_data へ inline 化 → 既存の build_graph」という設計。
//! `docs/onnx-external-data-decision.md` が設計判断の正）。
//!
//! ## 不変条件（A6・回帰テスト対象）
//!
//! `graph::decode_tensor` は本モジュールの追加後も
//! 1 バイトも変更しない。`onnx::proto::decode_model` → `graph::build_graph`
//! のバイト列入口は `data_location`／`external_data` を一切参照しないため、
//! external data を持つモデルをバイト列入口へ渡した場合の挙動（従来どおり
//! `RawDataByteLenMismatch` で拒否）は本モジュール導入前後で変わらない。
//!
//! ## 2 パス設計（検証と読み込みの分離。security.md A03／A04）
//!
//! 1. **パス 1（`plan`）**: ファイル内容を一切読まず、`data_location`／
//!    `external_data` のフィールド整合性・キーの重複や未知キー・
//!    `location` のパス安全性（絶対パス・`..`・symlink・base_dir 外への
//!    脱出を拒否）・`offset`／`length` の文法と範囲・期待バイト長との一致・
//!    重複区間・**外部データ合計の上限**（[`ExternalDataOptions::
//!    max_total_bytes`]。確保の前に検査する）・**distinct な external
//!    data ファイル数の上限**（[`ExternalDataOptions::
//!    max_external_files`]。合計バイト数の上限はサイズ 0 のテンソルを
//!    大量の異なる空ファイルへ分散させる入力に対しては無力なため、
//!    ファイルを開いた直後・`files` へ登録する前に別途検査する。fd 枯渇
//!    対策。#2347 P0 是正・PR #2348 コードレビュー対応・
//!    PRRT_kwDOTuUCJc6mlxhy）をすべて検証する。1 件でも失敗すれば `Err`
//!    を返しファイルは一切読まない（A04 資源枯渇対策）。
//! 2. **パス 2（`load`）**: パス 1 が全件成功した場合のみ、パス 1 で
//!    開いたファイルハンドルを再利用して該当区間だけを `read_exact` する
//!    （`.data` ファイル全体は読まない）。読み込み直前に `metadata().len()`
//!    （Unix では dev/ino も）をパス 1 の記録と再照合し、不一致は
//!    [`ExternalDataError::FileChangedDuringLoad`] とする。
//!
//!    **経路解決方式（unix 全般。イシュー #2347 是正版）**: パス 1 の
//!    ファイル解決は `base_dir` を 1 度だけディレクトリ fd として開き
//!    （`no_follow_open::open_base_dir`）、以後の全テンソルがその fd を
//!    起点に `location` を解決する。**Linux** では `openat2(2)`
//!    （`RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS | RESOLVE_NO_MAGICLINKS`。
//!    `no_follow_open::open_chain_openat2`）で経路解決全体を 1 回の
//!    システムコールとしてカーネルへアトミックに封じ込めさせる。
//!    `openat2` が未対応（`ENOSYS`／`EPERM`。古いカーネル・seccomp 等）の
//!    場合のみ、成分ごと逐次 `openat(O_NOFOLLOW)` で辿る方式
//!    （`no_follow_open::open_chain_component_walk`）へフォールバックする。
//!    **macOS・その他 unix**（`openat2` 非対応）では常に逐次方式を使う。
//!    いずれの方式も検証済みの fd をそのまま保持し経路文字列を**一切**
//!    再解決しないため（診断用のシンボリックリンク判別も、既に開いた
//!    親ディレクトリ fd を起点にした単一コンポーネントの `fstatat`
//!    〈`AT_SYMLINK_NOFOLLOW`〉のみを使う）、シンボリックリンク差し替えに
//!    よる TOCTOU 窓は構造的に生じない（P0 是正: PRRT_kwDOTuUCJc6mkUsI・
//!    PRRT_kwDOTuUCJc6mk30J。`docs/onnx-external-data-decision.md` の
//!    残タスクを解消）。
//!
//!    旧実装は `openat` のフラグ定数（`O_DIRECTORY`／`O_NOFOLLOW` 等）を
//!    OS ごとに手書きしていたが、Linux では同じ定数でも CPU アーキテク
//!    チャごとに値が異なり（例: x86 は `O_DIRECTORY=0o200000`・aarch64 は
//!    `O_DIRECTORY=0o40000`）、手書き値のまま aarch64 Linux（DGX Spark
//!    GB10）でビルドするとシンボリックリンク拒否が機能しない実装バグを
//!    生んでいた。`libc`（`.claude/rules/deps-policy.md`「OS FFI」区分。
//!    2026-09-28 ユーザー承認）の定数・`syscall` を使うことでこの
//!    プラットフォーム差異を自作せず解消する。
//!
//!    **非 unix（Windows 等）**: `openat`／`openat2` 相当の安全な経路
//!    解決手段を持たないため、`resolve_and_open` は常に
//!    [`ExternalDataError::UnsupportedPlatformForSecureResolve`] で拒否
//!    する（fail-closed。`docs/onnx-external-data-decision.md` 5 節）。
//!
//! `checksum` キーは黙って無視せず [`ExternalDataError::
//! ChecksumUnsupported`] で fail-closed に拒否する（依存を追加できないため
//! SHA-1 検証は実装しない。no-silent-skip 契約。security.md A08）。
//!
//! ## facade への公開範囲
//!
//! `crate::facade::interop::onnx::OnnxModel::from_path`（本クレート外部）
//! が [`build_graph_with_external_data`] へ委譲する形で 2026-09-28 に
//! 公開済み（`docs/facade-onnx-import-exposure-decision.md` §6.3・
//! `docs/compat-api-scope.md` §5 の承認待ち事項を解消。イシュー #2347）。
//! [`resolve_external_data`]・[`ExternalDataOptions`] 自体は本モジュール
//! （`onnx-interop` 内部限定）のままであり、facade 側は既定オプション
//! （[`ExternalDataOptions::default`]）を渡すラッパーに徹する。

use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Component, Path, PathBuf};

use super::graph::{GraphError, element_count};
use super::proto::{ModelProto, TensorProto, cap_sparse_tensor_diag_name, data_location};

/// external data の合計サイズ上限の既定値（64 GiB）。
///
/// 2026-09-28 ユーザー承認（イシュー #2347。当初の 4 GiB 暫定値から改定）。
/// 変更は本定数 1 行の書き換えで済む。`ExternalDataOptions::default()` が
/// 参照する。
pub const DEFAULT_MAX_EXTERNAL_DATA_TOTAL_BYTES: u64 = 64 * 1024 * 1024 * 1024;

/// 1 モデルあたりに同時オープンを許す external data ファイル数（実体単位。
/// `FileKey` で畳み込んだ後の distinct 数）の既定上限。
///
/// `max_total_bytes` はバイト数のみを制限するため、要素数 0（サイズ 0）の
/// テンソルを大量に並べ、それぞれが別々の空ファイルを `location` で参照
/// するモデルに対しては合計サイズが 0 のまま歯止めが効かず、`plan` が
/// 開いたファイルハンドルだけがプロセスの fd 上限に達しうる（security.md
/// A04・AGENTS.md「外部フォーマットのパース検証」。#2347 P0 是正・
/// PR #2348 コードレビュー対応・PRRT_kwDOTuUCJc6mlxhy）。この上限は
/// distinct なファイル実体（`FileKey`）の数を制限するため、1 ファイルを
/// 複数テンソルが参照する通常の分割形式（同一 `.onnx.data` を initializer
/// 群が共有する構成）は 1 件としてしか数えない。
///
/// **暫定値・ユーザー承認待ち**（`max_total_bytes` と同じ扱い。イシュー
/// #2347 計画 §2「承認待ちの事項」）。変更は本定数 1 行の書き換えで済む。
/// `ExternalDataOptions::default()` が参照する。
pub const DEFAULT_MAX_EXTERNAL_FILES: usize = 4096;

/// [`resolve_external_data`] の挙動を制御するオプション。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExternalDataOptions {
    /// 1 モデルあたりの external data 合計バイト数の上限。超過は
    /// [`ExternalDataError::TotalSizeLimitExceeded`] で拒否する（確保の
    /// 前に検査するため、この上限を超える `length` はメモリを確保しない）。
    pub max_total_bytes: u64,
    /// 1 モデルあたりに同時オープンを許す external data ファイル数
    /// （実体単位・`FileKey` で畳み込んだ後の distinct 数）の上限。
    /// 超過は [`ExternalDataError::TooManyExternalFiles`] で拒否する
    /// （ファイルを開いた直後・`files` へ登録する前に検査するため、この
    /// 上限を超えて開いたハンドルを溜め込まない）。
    pub max_external_files: usize,
}

impl Default for ExternalDataOptions {
    fn default() -> Self {
        ExternalDataOptions {
            max_total_bytes: DEFAULT_MAX_EXTERNAL_DATA_TOTAL_BYTES,
            max_external_files: DEFAULT_MAX_EXTERNAL_FILES,
        }
    }
}

/// `location` 文字列・パス解決が拒否される理由。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum LocationRejectReason {
    Empty,
    ContainsNul,
    TooLong,
    Backslash,
    DrivePrefix,
    Absolute,
    ParentDir,
    Symlink,
    NotRegularFile,
    OutsideBaseDir,
}

impl fmt::Display for LocationRejectReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            LocationRejectReason::Empty => "空文字列",
            LocationRejectReason::ContainsNul => "NUL を含む",
            LocationRejectReason::TooLong => "4096 バイトを超える",
            LocationRejectReason::Backslash => "バックスラッシュを含む",
            LocationRejectReason::DrivePrefix => "ドライブ文字の接頭辞",
            LocationRejectReason::Absolute => "絶対パス",
            LocationRejectReason::ParentDir => "親ディレクトリ参照（..）を含む",
            LocationRejectReason::Symlink => "経路にシンボリックリンクを含む",
            LocationRejectReason::NotRegularFile => "通常ファイルではない",
            LocationRejectReason::OutsideBaseDir => "base_dir の外へ解決される",
        };
        write!(f, "{s}")
    }
}

/// external data 解決時のエラー。`graph::GraphError::ExternalData` に包んで
/// 返す。非信頼文字列（`tensor_name`／`key`）は
/// `cap_sparse_tensor_diag_name` と同じ 256 バイト上限で切り詰める。
/// ホストの絶対パス・canonicalize 後のパスはいずれの variant にも含めない
/// （security.md A05）。
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ExternalDataError {
    /// `data_location` が `DEFAULT`（0）・`EXTERNAL`（1）以外。
    InvalidDataLocation { tensor_name: String, value: i32 },
    /// `EXTERNAL` なのに inline データが非空、または `DEFAULT` なのに
    /// `external_data` が非空。
    InconsistentDataFields {
        tensor_name: String,
        reason: &'static str,
    },
    /// `external_data` に `location` キーが無い。
    MissingLocationKey { tensor_name: String },
    /// `external_data` のキーが重複している。
    DuplicateKey { tensor_name: String, key: String },
    /// `location`／`offset`／`length` 以外のキー。
    UnknownKey { tensor_name: String, key: String },
    /// `checksum` キーは非対応のため fail-closed に拒否する。
    ChecksumUnsupported { tensor_name: String },
    /// `location` のパス検証に失敗した。
    InvalidLocation {
        tensor_name: String,
        reason: LocationRejectReason,
    },
    /// `offset`／`length` の文法・範囲が不正。
    InvalidNumber {
        tensor_name: String,
        field: &'static str,
    },
    /// `offset + length` がファイル長を超える。
    RangeOutOfFile { tensor_name: String },
    /// 実バイト長が `dims`／`data_type` から期待されるバイト長と一致しない。
    LengthMismatch {
        tensor_name: String,
        expected_bytes: u64,
        actual_bytes: u64,
    },
    /// 同一ファイル内で 2 つ以上のテンソルの読み込み区間が重なっている。
    OverlappingRegion {
        tensor_name: String,
        other_tensor_name: String,
    },
    /// external data の合計サイズが上限を超えた。
    TotalSizeLimitExceeded { limit: u64, requested: u64 },
    /// distinct な external data ファイル数（実体単位）が上限を超えた。
    /// `max_total_bytes` はバイト数のみを制限するため、サイズ 0 のテンソル
    /// を大量に異なるファイルへ分散させる攻撃はバイト数の上限をすり抜ける
    /// （A04 資源枯渇対策。`ExternalDataOptions::max_external_files`）。
    TooManyExternalFiles { limit: usize },
    /// ファイル I/O の失敗（`NotFound`／権限エラー等）。
    Io {
        tensor_name: String,
        kind: std::io::ErrorKind,
    },
    /// パス 2 の読み込み直前にファイル長（Unix では dev/ino も）がパス 1 の
    /// 記録と食い違った（TOCTOU 検知）。
    FileChangedDuringLoad { tensor_name: String },
    /// `base_dir` の canonicalize に失敗した。
    InvalidBaseDir { kind: std::io::ErrorKind },
    /// initializer 名が重複している（I/O の前に検出する。`graph::
    /// build_graph` の `DuplicateInitializerName` と同一の欠陥クラス）。
    DuplicateInitializerName { tensor_name: String },
    /// `resolve_and_open` の TOCTOU 非後退実装（ディレクトリ fd 起点の
    /// `O_NOFOLLOW` 追跡拒否オープン）が Linux／macOS 限定であり、それ
    /// 以外の unix ではフラグ定数値を実機実測できていないため安全に
    /// 実装できない。旧来の `symlink_metadata` 検証 → `canonicalize` →
    /// `File::open` 再解決経路は検証とオープンの間にシンボリックリンク
    /// 差し替えの窓が残るため、REQ-1 の完全自作コア方針・security.md
    /// の A08（自己修復ループが取り込む変更の整合性）と同じ fail-closed
    /// 原則に従い、対応不能なプラットフォームでは external data の読み
    /// 込みそのものを拒否する（P0・PRRT_kwDOTuUCJc6mk30J 是正）。
    UnsupportedPlatformForSecureResolve { tensor_name: String },
    /// 内部不変条件違反（本来発生しないはずの状態）。`coding-rust.md`
    /// の「本番経路で `unwrap()`/`expect()` を使わない」方針に従い、
    /// `panic!`／`unreachable!`／`.expect()` の代わりにこの型付きエラーで
    /// 表面化させる（`plan` と `load`／書き戻しの間の内部不変条件——
    /// `plan` が登録したファイルキー・slot は `load`／書き戻し時にも
    /// 存在するはず——が崩れた場合のみ到達する）。
    Internal { reason: &'static str },
}

impl fmt::Display for ExternalDataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExternalDataError::InvalidDataLocation { tensor_name, value } => {
                write!(f, "data_location が不正（tensor={tensor_name}）: {value}")
            }
            ExternalDataError::InconsistentDataFields {
                tensor_name,
                reason,
            } => write!(
                f,
                "data_location と実データフィールドの組み合わせが不正（tensor={tensor_name}）: {reason}"
            ),
            ExternalDataError::MissingLocationKey { tensor_name } => {
                write!(
                    f,
                    "external_data に location キーが無い（tensor={tensor_name}）"
                )
            }
            ExternalDataError::DuplicateKey { tensor_name, key } => {
                write!(
                    f,
                    "external_data のキー重複（tensor={tensor_name} key={key}）"
                )
            }
            ExternalDataError::UnknownKey { tensor_name, key } => {
                write!(
                    f,
                    "external_data の未知キー（tensor={tensor_name} key={key}）"
                )
            }
            ExternalDataError::ChecksumUnsupported { tensor_name } => {
                write!(f, "checksum キーは非対応のため拒否（tensor={tensor_name}）")
            }
            ExternalDataError::InvalidLocation {
                tensor_name,
                reason,
            } => write!(f, "location が不正（tensor={tensor_name}）: {reason}"),
            ExternalDataError::InvalidNumber { tensor_name, field } => {
                write!(f, "{field} の値が不正（tensor={tensor_name}）")
            }
            ExternalDataError::RangeOutOfFile { tensor_name } => {
                write!(
                    f,
                    "offset+length がファイル長を超える（tensor={tensor_name}）"
                )
            }
            ExternalDataError::LengthMismatch {
                tensor_name,
                expected_bytes,
                actual_bytes,
            } => write!(
                f,
                "external data のバイト長不整合（tensor={tensor_name}）: 期待={expected_bytes} 実際={actual_bytes}"
            ),
            ExternalDataError::OverlappingRegion {
                tensor_name,
                other_tensor_name,
            } => write!(
                f,
                "external data の区間重複（tensor={tensor_name} other={other_tensor_name}）"
            ),
            ExternalDataError::TotalSizeLimitExceeded { limit, requested } => write!(
                f,
                "external data 合計サイズが上限を超過: limit={limit} requested={requested}"
            ),
            ExternalDataError::TooManyExternalFiles { limit } => write!(
                f,
                "external data ファイル数（実体単位）が上限を超過: limit={limit}"
            ),
            ExternalDataError::Io { tensor_name, kind } => {
                write!(
                    f,
                    "external data の I/O エラー（tensor={tensor_name}）: {kind:?}"
                )
            }
            ExternalDataError::FileChangedDuringLoad { tensor_name } => write!(
                f,
                "読み込み直前にファイルが変化した（tensor={tensor_name}）: TOCTOU 検知"
            ),
            ExternalDataError::InvalidBaseDir { kind } => {
                write!(f, "base_dir の解決に失敗: {kind:?}")
            }
            ExternalDataError::DuplicateInitializerName { tensor_name } => {
                write!(f, "initializer 名の重複（tensor={tensor_name}）")
            }
            ExternalDataError::UnsupportedPlatformForSecureResolve { tensor_name } => write!(
                f,
                "external data の安全な解決（TOCTOU 非後退のディレクトリ fd 起点オープン）が \
                 このプラットフォームでは未対応のため拒否（tensor={tensor_name}）: Linux／macOS \
                 以外では external data 読み込みをサポートしない"
            ),
            ExternalDataError::Internal { reason } => {
                write!(f, "external_data 内部不変条件違反: {reason}")
            }
        }
    }
}

impl std::error::Error for ExternalDataError {}

/// 診断用テンソル名を `cap_sparse_tensor_diag_name` と同じ上限で切り詰める。
fn cap_name(name: &str) -> String {
    cap_sparse_tensor_diag_name(name.as_bytes())
}

/// `location` 文字列を検証し、`base_dir` へ 1 段ずつ連結できる正規
/// コンポーネント列（`CurDir` は除去済み）を返す（`Path::components` の
/// 判定結果をそのまま返すことで、呼び出し元〈`resolve_and_open`〉が
/// 「ここには `Normal` しか来ないはず」という前提を `unreachable!` で
/// 表現せずに済む。`coding-rust.md` の本番経路 `unwrap`/`expect` 禁止と
/// 同じ理由で `unreachable!` も避ける）。
fn validate_location_string(loc: &str) -> Result<Vec<&std::ffi::OsStr>, LocationRejectReason> {
    if loc.is_empty() {
        return Err(LocationRejectReason::Empty);
    }
    if loc.len() > 4096 {
        return Err(LocationRejectReason::TooLong);
    }
    if loc.contains('\0') {
        return Err(LocationRejectReason::ContainsNul);
    }
    if loc.contains('\\') {
        return Err(LocationRejectReason::Backslash);
    }
    // ドライブ文字接頭辞（`C:` 等）: 先頭 2 バイトが ASCII 英字 + ':'。
    let bytes = loc.as_bytes();
    if bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':' {
        return Err(LocationRejectReason::DrivePrefix);
    }
    let mut parts = Vec::new();
    for component in Path::new(loc).components() {
        match component {
            Component::Normal(part) => parts.push(part),
            Component::CurDir => {}
            Component::RootDir | Component::Prefix(_) => {
                return Err(LocationRejectReason::Absolute);
            }
            Component::ParentDir => return Err(LocationRejectReason::ParentDir),
        }
    }
    Ok(parts)
}

/// 開いたファイルとパス 1 時点のメタデータ（パス 2 の TOCTOU 再照合・
/// 読み込みの両方に使う）。
struct OpenFile {
    file: File,
    len: u64,
    #[cfg(unix)]
    dev_ino: (u64, u64),
}

/// ディレクトリ fd（`base_dir` を開いたハンドル）を起点に `location` を
/// シンボリックリンク拒否で解決してファイルを開く実装（P0 対応。
/// discussion_r4119392011・イシュー #2347 是正版）。
///
/// 旧実装は手書きの `extern "C"` 宣言と手書きの `openat` フラグ定数を
/// 使っていたが、Linux では同じ定数でも CPU アーキテクチャごとに値が
/// 異なり（例: x86 は `O_DIRECTORY=0o200000`・aarch64 は
/// `O_DIRECTORY=0o40000`）、x86 向けの値のまま aarch64 Linux（DGX Spark
/// GB10）でビルドするとシンボリックリンク拒否が機能しない実装バグを
/// 生んでいた。本実装は `libc`（`.claude/rules/deps-policy.md`「OS FFI」
/// 区分。2026-09-28 ユーザー承認）の定数・関数を使い、この種の値の取り
/// 違えを自作せず解消する。
///
/// 経路は文字列として再解決しない。**Linux** は `openat2(2)`
/// （[`open_chain_openat2`]。`RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS |
/// RESOLVE_NO_MAGICLINKS`）で経路解決全体をカーネルへ 1 回のシステム
/// コールとしてアトミックに封じ込めさせる。`openat2` が未対応
/// （`ENOSYS`／`EPERM`）の場合のみ、成分ごと逐次 `openat(dirfd, name,
/// O_NOFOLLOW)` で辿る [`open_chain_component_walk`] へフォールバックする
/// （**macOS・その他 unix**〈`openat2` 非対応〉では常にこちらを使う）。
/// いずれの方式も、対象がシンボリックリンクなら `ELOOP` で即座に失敗する
/// ため検証済みの fd 連鎖以外を辿る余地が生じない。エラー分類用の診断
/// （シンボリックリンクか否かの判別）も、既に開いた親ディレクトリ fd を
/// 起点にした単一コンポーネントの `fstatat(AT_SYMLINK_NOFOLLOW)` のみを
/// 使い、`symlink_metadata`／`canonicalize` によるパス文字列の再解決は
/// 一切行わない（P0 是正: PRRT_kwDOTuUCJc6mkUsI・PRRT_kwDOTuUCJc6mk30J。
/// 旧実装はここで累積パス文字列 `symlink_metadata` を呼んでおり、その
/// 呼び出し自体が検証後の再解決だった）。
///
/// `base_dir` のディレクトリ fd は [`open_base_dir`] で 1 度だけ開き、
/// `plan` が全テンソル分を通して再利用する（呼び出しごとに開き直さない。
/// advisor 指摘）。
#[cfg(unix)]
mod no_follow_open {
    use std::ffi::{CString, OsStr};
    use std::fs::File;
    use std::io;
    #[cfg(target_os = "linux")]
    use std::mem::size_of;
    use std::os::unix::ffi::OsStrExt;
    #[cfg(unix)]
    use std::os::unix::fs::OpenOptionsExt;
    use std::os::unix::io::{AsRawFd, FromRawFd};
    use std::path::Path;

    /// `base_dir_canonical` をディレクトリ fd として 1 度だけ開く。`plan`
    /// が全 external テンソル分を通してこの fd を再利用する。`O_NONBLOCK`
    /// は FIFO 混入時の無期限ハング防止（下記 `openat_no_follow` と同じ
    /// 理由）。`OpenOptionsExt::custom_flags` は安全な std API のため
    /// `unsafe` を要しない。
    pub(super) fn open_base_dir(base_dir_canonical: &Path) -> io::Result<File> {
        std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NONBLOCK)
            .open(base_dir_canonical)
    }

    /// シンボリックリンク検知（`ELOOP`）かどうかを判定する。Linux／macOS
    /// 共通で POSIX が規定する errno のため `libc::ELOOP` をそのまま使う
    /// （アーキテクチャ間の値の取り違えは `libc` が吸収する）。
    fn is_eloop(e: &io::Error) -> bool {
        e.raw_os_error() == Some(libc::ELOOP)
    }

    /// `dir`（親ディレクトリ fd）に対して相対的な単一コンポーネント
    /// `name` の種別を診断する（シンボリックリンクか否か）。
    /// `open_chain_component_walk` が `ENOTDIR`（「シンボリックリンクを
    /// `O_DIRECTORY` 付きで開いた」場合と「単なる非ディレクトリの通常
    /// ファイルを `O_DIRECTORY` 付きで開いた」場合の両方で返り得る）の
    /// どちらのエラー種別として報告するかの分類にのみ使う診断専用の
    /// 関数。`dir` は呼び出し元が既に検証済みの親ディレクトリ fd であり、
    /// `fstatat(AT_SYMLINK_NOFOLLOW)` は `name` 単一コンポーネントのみを
    /// その fd に対して相対的に見るため、経路文字列の再解決（TOCTOU 窓）
    /// を伴わない。
    fn is_symlink_component(dir: &File, name: &OsStr) -> bool {
        let Ok(c_name) = CString::new(name.as_bytes()) else {
            return false;
        };
        // SAFETY: `zeroed()` で得る `libc::stat` はすべてのフィールドが
        // ビット表現 0 でも有効な値になる POD 構造体（POSIX の `stat`
        // 構造体は整数・配列フィールドのみで構成され、`0` 埋めのままでも
        // 不変条件を持たない）。`fstatat` は成功時のみこのバッファへ書き
        // 込むため、`ret == 0` を確認してから中身を読む下の判定は健全。
        let mut st: libc::stat = unsafe { std::mem::zeroed() };
        // SAFETY: `dir.as_raw_fd()` はこの呼び出しの間生存している `dir`
        // が所有する有効な open ディレクトリ fd。`c_name` は直前の
        // `CString::new` が NUL 終端を保証した有効な C 文字列でこの
        // 呼び出しの間生存する。`&mut st` は上で初期化済みの有効な出力
        // バッファへの排他参照。本関数は結果の真偽のみを診断（エラー
        // 種別の分類）に使い、安全性判断（fail-closed 拒否の可否）には
        // 使わない。
        let ret = unsafe {
            libc::fstatat(
                dir.as_raw_fd(),
                c_name.as_ptr(),
                &mut st,
                libc::AT_SYMLINK_NOFOLLOW,
            )
        };
        ret == 0 && (st.st_mode & libc::S_IFMT) == libc::S_IFLNK
    }

    /// `dir` に対して相対的に `name`（単一パス成分。`..`／`/` を含まない
    /// `Path::components()` の `Normal` 由来の値のみを渡す前提）を
    /// `O_NOFOLLOW` で開く。`want_dir` が true なら `O_DIRECTORY` を付け、
    /// 対象がディレクトリでなければ失敗する。
    fn openat_no_follow(dir: &File, name: &OsStr, want_dir: bool) -> io::Result<File> {
        let c_name = CString::new(name.as_bytes())
            .map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
        // `O_NONBLOCK` を無条件で付与する（High・Cursor Bugbot 指摘）:
        // `base_dir` 配下に FIFO（named pipe）等の特殊ファイルが置かれて
        // いた場合、`O_NONBLOCK` 抜きの `open`/`openat` は対向の
        // reader/writer が現れるまで無期限にブロックし得る。`is_file()`
        // による種別検証は open 成功後にしか行えないため、open 自体が
        // ハングすると検証に到達できない。通常ファイル・ディレクトリの
        // open には副作用が無い（POSIX: `O_NONBLOCK` は FIFO・キャラクタ
        // デバイス・ソケットにのみ意味を持つ）ため、中間ディレクトリ・
        // 最終ファイルのどちらの open にも無条件で付与してよい。
        let flags = if want_dir {
            libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_NONBLOCK
        } else {
            libc::O_RDONLY | libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_NONBLOCK
        };
        // SAFETY: `dir.as_raw_fd()` はこの呼び出しの間生存している `dir` が
        // 所有する有効な open ディレクトリ fd。`c_name` は
        // `CString::new` が NUL 終端を保証した有効な C 文字列で、この
        // 呼び出しの間生存する。`libc::openat` は POSIX 標準の可変長引数
        // 関数だが本呼び出しは `O_CREAT` を渡さないため可変長引数は
        // 実際には使わない。返り値が非負なら新規に確保された fd の所有権
        // を呼び出し元へ渡す契約（POSIX `openat(2)`）であり、
        // `File::from_raw_fd` で即座に `File` へ委譲することで二重解放・
        // リークを防ぐ。
        let fd = unsafe { libc::openat(dir.as_raw_fd(), c_name.as_ptr(), flags) };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: 直前の `openat` が返した非負 fd は呼び出し元がここで
        // 一意に所有権を得る新規 fd であり、他のどのコードもまだ
        // 参照していない。
        Ok(unsafe { File::from_raw_fd(fd) })
    }

    /// `base_dir`（[`open_base_dir`] が開いたディレクトリ fd）を起点に
    /// `parts` を 1 段ずつ `O_NOFOLLOW` で辿り、最終ファイルの fd を得る
    /// （`openat2` 非対応環境向けのフォールバック方式。macOS・その他
    /// unix では常にこちらを使う）。中間段は `O_DIRECTORY` 付きで開く
    /// ためディレクトリでなければ失敗し、途中経路のシンボリックリンクは
    /// `ELOOP` で拒否される。エラーはシンボリックリンク検知か否かを
    /// 呼び出し元が判別できるよう `(io::Error, bool /* is_symlink */)`
    /// を返す。
    pub(super) fn open_chain_component_walk(
        base_dir: &File,
        parts: &[&OsStr],
    ) -> Result<File, (io::Error, bool)> {
        if parts.is_empty() {
            return Err((io::Error::from(io::ErrorKind::InvalidInput), false));
        }
        let last_idx = parts.len() - 1;
        // 直前段の fd。最初の反復は呼び出し元が既に開いた `base_dir` を
        // 起点にし、以後は毎段 `openat_no_follow` が返す新規 fd に置き
        // 換わる（経路文字列ではなく fd を起点に辿る）。
        let mut dir_owned: Option<File> = None;
        for (i, part) in parts.iter().enumerate() {
            let want_dir = i != last_idx;
            let current: &File = dir_owned.as_ref().unwrap_or(base_dir);
            match openat_no_follow(current, part, want_dir) {
                Ok(next) => dir_owned = Some(next),
                Err(e) => {
                    let is_sym = is_eloop(&e)
                        || (want_dir
                            && e.raw_os_error() == Some(libc::ENOTDIR)
                            && is_symlink_component(current, part));
                    return Err((e, is_sym));
                }
            }
        }
        // `parts` は上で非空を確認済みのため、ループは必ず 1 回以上
        // `dir_owned` へ書き込んでから正常終了する。`unwrap()` の代わりに
        // 型付きエラーで表面化させる（coding-rust.md「本番経路で
        // unwrap/expect を使わない」。到達しないはずの防御的分岐）。
        dir_owned.ok_or((io::Error::from(io::ErrorKind::InvalidInput), false))
    }

    /// Linux 限定: `openat2(2)`（`RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS |
    /// RESOLVE_NO_MAGICLINKS`）で経路解決全体をカーネルへ 1 回のシステム
    /// コールとしてアトミックに封じ込めさせる。`RESOLVE_NO_XDEV` は
    /// 意図的に付与しない（bind mount 越しのモデルディレクトリ配置を
    /// 壊さないため。`RESOLVE_BENEATH` と `RESOLVE_NO_SYMLINKS` だけで
    /// `base_dir` 外への脱出は既に閉じている）。`libc` は `SYS_openat2`・
    /// `open_how`・`RESOLVE_*` 定数は提供するが `openat2()` の関数
    /// ラッパー自体は未提供のため `libc::syscall` 経由で呼ぶ。
    #[cfg(target_os = "linux")]
    pub(super) enum Openat2Outcome {
        Opened(File),
        Failed(io::Error, bool /* is_symlink */),
        /// `openat2` 自体が未対応（`ENOSYS`／`EPERM`。古いカーネル・
        /// seccomp 等）。呼び出し元は [`open_chain_component_walk`] へ
        /// フォールバックする。
        Unsupported,
    }

    #[cfg(target_os = "linux")]
    pub(super) fn open_chain_openat2(base_dir: &File, parts: &[&OsStr]) -> Openat2Outcome {
        if parts.is_empty() {
            return Openat2Outcome::Failed(io::Error::from(io::ErrorKind::InvalidInput), false);
        }
        let mut rel: Vec<u8> = Vec::new();
        for (i, part) in parts.iter().enumerate() {
            if i > 0 {
                rel.push(b'/');
            }
            rel.extend_from_slice(part.as_bytes());
        }
        let Ok(c_rel) = CString::new(rel) else {
            return Openat2Outcome::Failed(io::Error::from(io::ErrorKind::InvalidInput), false);
        };
        // `libc::open_how` は `#[non_exhaustive]`（将来のカーネル ABI
        // 拡張に備えたフィールド追加余地）のため構造体リテラルで直接
        // 構築できない。POSIX の `open_how` は整数フィールドのみで構成
        // され `0` 埋めのままでも不変条件を持たない POD 構造体のため、
        // ゼロ初期化してから既知フィールドのみを設定する（下の
        // `is_symlink_component` の `libc::stat` ゼロ初期化と同じ根拠）。
        // SAFETY: `libc::open_how` はビット表現 0 が有効な値になる POD
        // 構造体（整数フィールドのみ・不変条件なし）。
        let mut how: libc::open_how = unsafe { std::mem::zeroed() };
        how.flags = (libc::O_RDONLY | libc::O_CLOEXEC | libc::O_NONBLOCK) as u64;
        how.resolve =
            libc::RESOLVE_BENEATH | libc::RESOLVE_NO_SYMLINKS | libc::RESOLVE_NO_MAGICLINKS;
        // rename race 等による一時的な `EAGAIN` のみ有限回再試行する
        // （`openat2(2)` man page: 経路解決が再試行を要して中断された
        // 場合に返る）。無限ループ化を避けるため上限を設ける。
        for _attempt in 0..4 {
            // SAFETY: `base_dir.as_raw_fd()` は生存している有効な
            // ディレクトリ fd。`c_rel` はこの呼び出しの間生存する有効な
            // C 文字列。`&how` は正しく初期化された `open_how`（POSIX
            // `openat2(2)` が要求するレイアウト）への参照で、第 4 引数に
            // 渡す構造体サイズ（`size_of::<libc::open_how>()`）も一致
            // する。返り値が非負なら新規に確保された fd の所有権をこの
            // 呼び出し元が唯一取得する契約（`openat2(2)`）であり、
            // `File::from_raw_fd` で即座に `File` へ委譲することで
            // 二重解放・リークを防ぐ。
            let ret = unsafe {
                libc::syscall(
                    libc::SYS_openat2,
                    base_dir.as_raw_fd(),
                    c_rel.as_ptr(),
                    &how as *const libc::open_how,
                    size_of::<libc::open_how>(),
                )
            };
            if ret >= 0 {
                // SAFETY: 上記と同じ根拠（直前の syscall が返した非負 fd
                // の所有権をここで一意に取得する）。
                return Openat2Outcome::Opened(unsafe { File::from_raw_fd(ret as i32) });
            }
            let e = io::Error::last_os_error();
            match e.raw_os_error() {
                Some(libc::ENOSYS) | Some(libc::EPERM) => return Openat2Outcome::Unsupported,
                Some(libc::EAGAIN) => continue,
                Some(libc::ELOOP) => return Openat2Outcome::Failed(e, true),
                _ => return Openat2Outcome::Failed(e, false),
            }
        }
        Openat2Outcome::Failed(io::Error::from(io::ErrorKind::WouldBlock), false)
    }
}

/// `base_dir_file`（[`no_follow_open::open_base_dir`] が開いたディレクトリ
/// fd）を起点に `location` を検証しながら解決し、ファイルを開く。経路の
/// 途中を含めシンボリックリンクを拒否し、`openat2`／`O_NOFOLLOW` による
/// ディレクトリハンドル連鎖オープンで「検証した経路そのもの」を開くことを
/// 保証する（A2・P0 対応。discussion_r4119392011・イシュー #2347）。
/// 返り値の第 2 要素は `plan` が `file_key_for` のフォールバック（dev/ino
/// を持たないプラットフォーム。実質使われない。下記コメント参照）で使う、
/// `parts` から再構築した正規化済み相対パス。
#[cfg(unix)]
fn resolve_and_open(
    tensor_name: &str,
    base_dir_file: &File,
    loc: &str,
) -> Result<(OpenFile, PathBuf), ExternalDataError> {
    let parts =
        validate_location_string(loc).map_err(|reason| ExternalDataError::InvalidLocation {
            tensor_name: cap_name(tensor_name),
            reason,
        })?;

    #[cfg(target_os = "linux")]
    let open_result = match no_follow_open::open_chain_openat2(base_dir_file, &parts) {
        no_follow_open::Openat2Outcome::Opened(f) => Ok(f),
        no_follow_open::Openat2Outcome::Failed(e, is_symlink) => Err((e, is_symlink)),
        // `openat2` 未対応環境（古いカーネル・seccomp 等）: 成分ごと逐次
        // `openat(O_NOFOLLOW)` 方式へフォールバックする。
        no_follow_open::Openat2Outcome::Unsupported => {
            no_follow_open::open_chain_component_walk(base_dir_file, &parts)
        }
    };
    #[cfg(not(target_os = "linux"))]
    let open_result = no_follow_open::open_chain_component_walk(base_dir_file, &parts);

    let file = open_result.map_err(|(e, is_symlink)| {
        if is_symlink {
            ExternalDataError::InvalidLocation {
                tensor_name: cap_name(tensor_name),
                reason: LocationRejectReason::Symlink,
            }
        } else if e.kind() == std::io::ErrorKind::NotADirectory
            || e.raw_os_error() == Some(libc::ENOTDIR)
        {
            ExternalDataError::InvalidLocation {
                tensor_name: cap_name(tensor_name),
                reason: LocationRejectReason::NotRegularFile,
            }
        } else {
            ExternalDataError::Io {
                tensor_name: cap_name(tensor_name),
                kind: e.kind(),
            }
        }
    })?;

    let meta = file.metadata().map_err(|e| ExternalDataError::Io {
        tensor_name: cap_name(tensor_name),
        kind: e.kind(),
    })?;
    if !meta.is_file() {
        return Err(ExternalDataError::InvalidLocation {
            tensor_name: cap_name(tensor_name),
            reason: LocationRejectReason::NotRegularFile,
        });
    }

    use std::os::unix::fs::MetadataExt;
    let dev_ino = (meta.dev(), meta.ino());

    let normalized_rel: PathBuf = parts.iter().collect();
    Ok((
        OpenFile {
            file,
            len: meta.len(),
            dev_ino,
        },
        normalized_rel,
    ))
}

/// [`resolve_and_open`] の非 unix（Windows 等）向け実装。
///
/// `openat`／`openat2` 相当の、ディレクトリ fd 起点でシンボリックリンクを
/// 拒否しながら経路解決する安全な手段を持たないため、`security.md` の
/// A08（整合性の迂回経路を作らない）・本 crate の fail-closed 方針
/// （イシュー #2347 タイトルのとおり external data 読み込みは fail-closed
/// 前提）に従い、**この関数は常に拒否する**。非 unix で external data を
/// 安全に読み込む対応が必要になった場合は、対象 OS の安全な no-follow
/// open 手段（例: Windows の `FILE_FLAG_OPEN_REPARSE_POINT` ベースの実装）
/// を個別に設計する（`docs/onnx-external-data-decision.md` 参照）。
#[cfg(not(unix))]
fn resolve_and_open(
    tensor_name: &str,
    _base_dir_canonical: &Path,
    loc: &str,
) -> Result<(OpenFile, PathBuf), ExternalDataError> {
    // `location` の文法検証自体は OS 非依存で安全に行えるため、
    // 診断上の一貫性のため先に行う（結果は使わず、後続の fail-closed
    // 判定を先取りしない）。
    let _ = validate_location_string(loc).map_err(|reason| ExternalDataError::InvalidLocation {
        tensor_name: cap_name(tensor_name),
        reason,
    })?;

    Err(ExternalDataError::UnsupportedPlatformForSecureResolve {
        tensor_name: cap_name(tensor_name),
    })
}

/// ASCII 数字のみからなる非空文字列として `u64` を解釈する（符号・空白・
/// 先頭 `+` は拒否。checked 変換）。
fn parse_decimal_u64(
    raw: &str,
    tensor_name: &str,
    field: &'static str,
) -> Result<u64, ExternalDataError> {
    if raw.is_empty() || !raw.bytes().all(|b| b.is_ascii_digit()) {
        return Err(ExternalDataError::InvalidNumber {
            tensor_name: cap_name(tensor_name),
            field,
        });
    }
    raw.parse::<u64>()
        .map_err(|_| ExternalDataError::InvalidNumber {
            tensor_name: cap_name(tensor_name),
            field,
        })
}

/// `data_type` から要素サイズ（バイト）を返す。`graph::decode_tensor` が
/// 対応する 4 型（FLOAT／INT64／BOOL／FLOAT16）とだけ整合させる。
fn element_size(tensor_name: &str, data_type: i32) -> Result<u64, GraphError> {
    match data_type {
        t if t == super::proto::data_type::FLOAT => Ok(4),
        t if t == super::proto::data_type::INT64 => Ok(8),
        t if t == super::proto::data_type::BOOL => Ok(1),
        t if t == super::proto::data_type::FLOAT16 => Ok(2),
        other => Err(GraphError::UnknownDataType {
            tensor_name: tensor_name.to_string(),
            data_type: other,
        }),
    }
}

/// `files`／`regions`（`plan` 内）のファイル識別キー。overlap 検出・
/// ファイルハンドル再利用が「同一ファイル実体」を正しく畳み込めるよう、
/// パス文字列ではなくファイルの実体識別子を使う（Cursor Bugbot 指摘・
/// PR #2348 review thread `PRRT_kwDOTuUCJc6mkYTr`: `base_dir.join(location)` という `location` の
/// 生文字列連結をキーにすると、`Path` の `Eq`/`Hash` はコンポーネント
/// 単位のため `foo.data`／`./foo.data` の表記ゆれ自体は畳み込まれるが、
/// ハードリンクのように**文字列としても正規化後の経路としても異なるが
/// 実体は同一のファイル**は別キーになり overlap 検出をすり抜ける。
/// dev/ino をキーにすることでこの実体単位の同一性を保証する）。
#[cfg(unix)]
type FileKey = (u64, u64);
#[cfg(not(unix))]
type FileKey = PathBuf;

/// `opened`（`resolve_and_open` が返したハンドル）から [`FileKey`] を
/// 作る。Unix では dev/ino（`OpenFile::dev_ino`）を使い、シンボリックリンク
/// や表記ゆれだけでなくハードリンクも実体単位で同一キーへ畳み込む。
/// dev/ino を持たない他プラットフォームでは `normalized_rel`
/// （`resolve_and_open` が `Path::components()` から再構築した正規化済み
/// 相対パス）を `base_dir_canonical` へ連結した値をフォールバックキーに
/// 使う（ハードリンク識別はできないが、表記ゆれの畳み込みは維持する）。
#[cfg(unix)]
fn file_key_for(_base_dir_canonical: &Path, opened: &OpenFile, _normalized_rel: &Path) -> FileKey {
    opened.dev_ino
}
#[cfg(not(unix))]
fn file_key_for(base_dir_canonical: &Path, _opened: &OpenFile, normalized_rel: &Path) -> FileKey {
    base_dir_canonical.join(normalized_rel)
}

/// パス 1 で確定した「どこから何バイト読むか」の 1 件分。
struct LoadPlanEntry {
    slot: TensorSlot,
    tensor_name: String,
    file_key: FileKey,
    offset: u64,
    length: u64,
}

/// external なテンソルの所在（`ModelProto` 内の位置）。パス 2 の mutable
/// 書き戻しで使う。
#[derive(Clone, Copy)]
enum TensorSlot {
    Initializer(usize),
    NodeAttrTensor { node_idx: usize, attr_idx: usize },
}

/// `model` に含まれるすべてのテンソル（initializer・各ノードの
/// `attribute[].t`）を `(slot, tensor_name, &TensorProto)` として列挙する。
/// サブグラフ属性（`g`/`graphs`）は `proto::AttributeProto` に未定義のため
/// 対象外（`proto.rs` 冒頭コメント参照）。
fn enumerate_tensors(model: &ModelProto) -> Vec<(TensorSlot, String, &TensorProto)> {
    let mut out = Vec::new();
    let Some(g) = model.graph.as_ref() else {
        return out;
    };
    for (idx, init) in g.node.iter().enumerate() {
        for (attr_idx, attr) in init.attribute.iter().enumerate() {
            if let Some(t) = attr.t.as_ref() {
                let name = if !t.name.is_empty() {
                    t.name.clone()
                } else {
                    let node_label = if init.name.is_empty() {
                        init.op_type.as_str()
                    } else {
                        init.name.as_str()
                    };
                    format!("{node_label}:{}", attr.name)
                };
                out.push((
                    TensorSlot::NodeAttrTensor {
                        node_idx: idx,
                        attr_idx,
                    },
                    name,
                    t,
                ));
            }
        }
    }
    for (idx, init) in g.initializer.iter().enumerate() {
        out.push((TensorSlot::Initializer(idx), init.name.clone(), init));
    }
    out
}

/// パス 1: 全 external テンソルを検証する（ファイル内容は読まない）。
/// 検証済みの読み込み計画と、開いたファイルハンドルのキャッシュを返す。
fn plan(
    model: &ModelProto,
    base_dir_canonical: &Path,
    options: &ExternalDataOptions,
) -> Result<(Vec<LoadPlanEntry>, HashMap<FileKey, OpenFile>), GraphError> {
    // initializer 名の重複は I/O の前に拒否する（A5）。
    if let Some(g) = model.graph.as_ref() {
        let mut seen = std::collections::HashSet::new();
        for init in &g.initializer {
            if !seen.insert(init.name.as_str()) {
                return Err(GraphError::ExternalData(
                    ExternalDataError::DuplicateInitializerName {
                        tensor_name: cap_name(&init.name),
                    },
                ));
            }
        }
    }

    // `base_dir` のディレクトリ fd は遅延で 1 度だけ開き、以後の全テンソル
    // が `resolve_and_open` 経由でこの fd を再利用する（呼び出しごとに
    // 開き直さない）。external なテンソルが 1 件も無いモデル（`from_path`
    // 経由の通常モデル import を含む）では一切開かない: `std::fs::read`
    // は親ディレクトリの実行〈x〉権限のみを要求するのに対し
    // `open(O_DIRECTORY)` は読み取り〈r〉権限を要求するため、無条件に
    // 開くと「実行のみ許可のディレクトリに置かれた external data 非使用
    // モデル」が `from_path` で読めなくなる後退を招く（advisor 指摘）。
    // 非 unix では `resolve_and_open` が `base_dir_canonical: &Path` を
    // 直接受け取り常に拒否するため不要。
    #[cfg(unix)]
    let mut base_dir_file: Option<File> = None;

    let mut files: HashMap<FileKey, OpenFile> = HashMap::new();
    let mut regions: HashMap<FileKey, Vec<(u64, u64, String)>> = HashMap::new();
    let mut entries = Vec::new();
    let mut total_requested: u64 = 0;

    for (slot, tensor_name, t) in enumerate_tensors(model) {
        match t.data_location {
            v if v == data_location::DEFAULT => {
                if !t.external_data.is_empty() {
                    return Err(GraphError::ExternalData(
                        ExternalDataError::InconsistentDataFields {
                            tensor_name: cap_name(&tensor_name),
                            reason: "data_location=DEFAULT なのに external_data が非空",
                        },
                    ));
                }
                continue;
            }
            v if v == data_location::EXTERNAL => {}
            other => {
                return Err(GraphError::ExternalData(
                    ExternalDataError::InvalidDataLocation {
                        tensor_name: cap_name(&tensor_name),
                        value: other,
                    },
                ));
            }
        }

        if !t.raw_data.is_empty() || !t.float_data.is_empty() || !t.int64_data.is_empty() {
            return Err(GraphError::ExternalData(
                ExternalDataError::InconsistentDataFields {
                    tensor_name: cap_name(&tensor_name),
                    reason: "data_location=EXTERNAL なのに inline データが非空",
                },
            ));
        }

        let mut location: Option<String> = None;
        let mut offset_raw: Option<String> = None;
        let mut length_raw: Option<String> = None;
        let mut seen_keys = std::collections::HashSet::new();
        for entry in &t.external_data {
            if !seen_keys.insert(entry.key.clone()) {
                return Err(GraphError::ExternalData(ExternalDataError::DuplicateKey {
                    tensor_name: cap_name(&tensor_name),
                    key: cap_name(&entry.key),
                }));
            }
            match entry.key.as_str() {
                "location" => location = Some(entry.value.clone()),
                "offset" => offset_raw = Some(entry.value.clone()),
                "length" => length_raw = Some(entry.value.clone()),
                "checksum" => {
                    return Err(GraphError::ExternalData(
                        ExternalDataError::ChecksumUnsupported {
                            tensor_name: cap_name(&tensor_name),
                        },
                    ));
                }
                other => {
                    return Err(GraphError::ExternalData(ExternalDataError::UnknownKey {
                        tensor_name: cap_name(&tensor_name),
                        key: cap_name(other),
                    }));
                }
            }
        }
        let location = location.ok_or_else(|| {
            GraphError::ExternalData(ExternalDataError::MissingLocationKey {
                tensor_name: cap_name(&tensor_name),
            })
        })?;
        let offset = match offset_raw {
            Some(raw) => {
                parse_decimal_u64(&raw, &tensor_name, "offset").map_err(GraphError::ExternalData)?
            }
            None => 0,
        };

        let expected_elements = element_count(&tensor_name, &t.dims)?;
        let esize = element_size(&tensor_name, t.data_type)?;
        let expected_bytes = (expected_elements as u64)
            .checked_mul(esize)
            .ok_or_else(|| GraphError::ElementCountOverflow {
                tensor_name: tensor_name.clone(),
            })?;

        #[cfg(unix)]
        let (opened, normalized_rel) = {
            // 最初の external テンソルに到達した時点でのみ `base_dir` の
            // ディレクトリ fd を開く（上のコメント参照。遅延オープン）。
            if base_dir_file.is_none() {
                let f = no_follow_open::open_base_dir(base_dir_canonical).map_err(|e| {
                    GraphError::ExternalData(ExternalDataError::InvalidBaseDir { kind: e.kind() })
                })?;
                base_dir_file = Some(f);
            }
            let f = base_dir_file.as_ref().ok_or(GraphError::ExternalData(
                ExternalDataError::Internal {
                    reason: "plan: base_dir_file が None のまま resolve_and_open へ到達した",
                },
            ))?;
            resolve_and_open(&tensor_name, f, &location).map_err(GraphError::ExternalData)?
        };
        #[cfg(not(unix))]
        let (opened, normalized_rel) =
            resolve_and_open(&tensor_name, base_dir_canonical, &location)
                .map_err(GraphError::ExternalData)?;

        let length = match length_raw {
            Some(raw) => {
                parse_decimal_u64(&raw, &tensor_name, "length").map_err(GraphError::ExternalData)?
            }
            None => opened.len.checked_sub(offset).ok_or_else(|| {
                GraphError::ExternalData(ExternalDataError::RangeOutOfFile {
                    tensor_name: cap_name(&tensor_name),
                })
            })?,
        };

        let end = offset.checked_add(length).ok_or_else(|| {
            GraphError::ExternalData(ExternalDataError::RangeOutOfFile {
                tensor_name: cap_name(&tensor_name),
            })
        })?;
        if end > opened.len {
            return Err(GraphError::ExternalData(
                ExternalDataError::RangeOutOfFile {
                    tensor_name: cap_name(&tensor_name),
                },
            ));
        }
        if length != expected_bytes {
            return Err(GraphError::ExternalData(
                ExternalDataError::LengthMismatch {
                    tensor_name: cap_name(&tensor_name),
                    expected_bytes,
                    actual_bytes: length,
                },
            ));
        }

        total_requested = total_requested
            .checked_add(length)
            .ok_or(GraphError::ExternalData(
                ExternalDataError::TotalSizeLimitExceeded {
                    limit: options.max_total_bytes,
                    requested: u64::MAX,
                },
            ))?;
        if total_requested > options.max_total_bytes {
            return Err(GraphError::ExternalData(
                ExternalDataError::TotalSizeLimitExceeded {
                    limit: options.max_total_bytes,
                    requested: total_requested,
                },
            ));
        }

        // ファイルキー: `location` の生文字列を `base_dir` へ連結した
        // ものではなく、ファイルの実体識別子（[`FileKey`]。Unix では
        // dev/ino）を使う。生文字列を直接連結すると `foo.data`／
        // `./foo.data`／`foo.data/` のような表記ゆれに加え、ハードリンク
        // のように経路としては異なるが実体が同一のファイルも別キーに
        // なり、overlap 検出（下記）・ハンドル再利用の両方が同一ファイル
        // を見落とす（Cursor Bugbot 指摘・PR #2348 review thread
        // `PRRT_kwDOTuUCJc6mkYTr`）。`file_key_for` が dev/ino を優先し、それが取れない
        // プラットフォームでのみ `normalized_rel`（`resolve_and_open` が
        // `Path::components()` から再構築した正規化済み相対パス）へ
        // フォールバックする。
        let file_key = file_key_for(base_dir_canonical, &opened, &normalized_rel);

        // distinct ファイル数の上限検査（A04）: `max_total_bytes` はバイト
        // 数のみを制限するため、サイズ 0 のテンソルを大量の異なるファイルへ
        // 分散させると合計サイズは 0 のままファイルハンドルだけが増え、
        // プロセスの fd 上限に達しうる（#2347 P0 是正・PR #2348 コード
        // レビュー対応・PRRT_kwDOTuUCJc6mlxhy）。`opened`（この反復で新規に
        // 開いたハンドル）が既知の `file_key` でなければ、`files` へ登録する
        // 前にここで拒否する。拒否時は `opened` をどこにも格納しないため
        // スコープを抜ける際に close される（ハンドルの蓄積を防ぐ）。
        if !files.contains_key(&file_key) && files.len() >= options.max_external_files {
            return Err(GraphError::ExternalData(
                ExternalDataError::TooManyExternalFiles {
                    limit: options.max_external_files,
                },
            ));
        }

        // `FileKey` は Unix では `(u64, u64)`（`Copy`）、それ以外では
        // `PathBuf`（非 `Copy`）と cfg で型が変わる（上記型エイリアス
        // 参照）ため、`.clone()` は環境依存で clippy の
        // `clone_on_copy`（`-D warnings` 対象）に触れ得る。3 箇所で同じ
        // キーを使う必要がある（`regions`・`files` への登録＋
        // `LoadPlanEntry` への格納）ための意図的な複製であり、
        // `#[allow]` はこの cfg 依存の型差分に限定する。
        #[allow(clippy::clone_on_copy)]
        let region_key = file_key.clone();
        let interval_list = regions.entry(region_key).or_default();
        for (s, e, other_name) in interval_list.iter() {
            if offset < *e && *s < end {
                return Err(GraphError::ExternalData(
                    ExternalDataError::OverlappingRegion {
                        tensor_name: cap_name(&tensor_name),
                        other_tensor_name: cap_name(other_name),
                    },
                ));
            }
        }
        interval_list.push((offset, end, tensor_name.clone()));

        #[allow(clippy::clone_on_copy)]
        let files_key = file_key.clone();
        files.entry(files_key).or_insert(opened);

        entries.push(LoadPlanEntry {
            slot,
            tensor_name,
            file_key,
            offset,
            length,
        });
    }

    Ok((entries, files))
}

/// パス 2: パス 1 が確定した計画に従い、該当区間だけを読み込む。TOCTOU
/// 再照合（ファイル長・Unix では dev/ino）を読み込み直前に行う。
fn load(
    entries: &[LoadPlanEntry],
    files: &mut HashMap<FileKey, OpenFile>,
) -> Result<Vec<Vec<u8>>, GraphError> {
    let mut out: Vec<Vec<u8>> = Vec::with_capacity(entries.len());
    for entry in entries.iter() {
        let opened = files
            .get_mut(&entry.file_key)
            .ok_or(GraphError::ExternalData(ExternalDataError::Internal {
                reason: "load: plan が登録したファイルキーが files に存在しない",
            }))?;
        let meta = opened.file.metadata().map_err(|e| {
            GraphError::ExternalData(ExternalDataError::Io {
                tensor_name: cap_name(&entry.tensor_name),
                kind: e.kind(),
            })
        })?;
        let len_ok = meta.len() == opened.len;
        #[cfg(unix)]
        let ident_ok = {
            use std::os::unix::fs::MetadataExt;
            (meta.dev(), meta.ino()) == opened.dev_ino
        };
        #[cfg(not(unix))]
        let ident_ok = true;
        if !len_ok || !ident_ok {
            return Err(GraphError::ExternalData(
                ExternalDataError::FileChangedDuringLoad {
                    tensor_name: cap_name(&entry.tensor_name),
                },
            ));
        }

        opened
            .file
            .seek(SeekFrom::Start(entry.offset))
            .map_err(|e| {
                GraphError::ExternalData(ExternalDataError::Io {
                    tensor_name: cap_name(&entry.tensor_name),
                    kind: e.kind(),
                })
            })?;
        // `entry.length` は `u64`。32bit ターゲット等 `usize` が 64bit
        // 未満の環境では `as usize` の暗黙切り捨てで確保サイズが縮み
        // `read_exact` が誤った短いバッファへ書き込みうるため、
        // `checked` 変換で明示的に拒否する（`plan` の `total_requested`
        // 上限検査より前ではなく後段だが、変換自体の健全性は独立した
        // 契約のため個別に検査する）。
        let buf_len = usize::try_from(entry.length).map_err(|_| {
            GraphError::ExternalData(ExternalDataError::Internal {
                reason: "load: entry.length が usize の範囲を超える",
            })
        })?;
        let mut buf = vec![0u8; buf_len];
        opened.file.read_exact(&mut buf).map_err(|e| {
            GraphError::ExternalData(ExternalDataError::Io {
                tensor_name: cap_name(&entry.tensor_name),
                kind: e.kind(),
            })
        })?;
        out.push(buf);
    }
    Ok(out)
}

/// `model` に含まれる external なテンソル（initializer・Constant 属性
/// テンソル）をすべて検証・読み込み、`raw_data` へ inline 化する（in-place。
/// `data_location` は `DEFAULT` に戻し `external_data` は空にする）。
///
/// 呼び出し元は [`build_graph_with_external_data`]。`base_dir` は
/// canonicalize してから使う（失敗は [`ExternalDataError::InvalidBaseDir`]）。
/// `base_dir` 自体は呼び出し元が与える信頼済み入力として扱う（location
/// 側だけを fail-closed に検証する。モジュール冒頭コメント参照）。
pub fn resolve_external_data(
    model: &mut ModelProto,
    base_dir: &Path,
    options: &ExternalDataOptions,
) -> Result<(), GraphError> {
    let base_dir_canonical = base_dir.canonicalize().map_err(|e| {
        GraphError::ExternalData(ExternalDataError::InvalidBaseDir { kind: e.kind() })
    })?;

    let (entries, mut files) = plan(model, &base_dir_canonical, options)?;
    if entries.is_empty() {
        return Ok(());
    }
    let loaded = load(&entries, &mut files)?;
    if loaded.len() != entries.len() {
        return Err(GraphError::ExternalData(ExternalDataError::Internal {
            reason: "load の戻り値件数が entries と一致しない",
        }));
    }

    // パス 2 完了後にのみ書き戻す（検証・読み込みが全件成功した場合限定）。
    // `entries` が非空の場合、`plan` は `enumerate_tensors` が返した
    // 実在の slot からのみ `entries` を構築しているため、ここで
    // `model.graph` は必ず `Some`（`enumerate_tensors` は graph が
    // `None` なら空 Vec を返す）。`unwrap`/`expect` の代わりに
    // `ok_or_else` で型付きエラーとして表面化させる（内部不変条件が
    // 崩れた場合のみ到達する防御的分岐）。
    let g = model
        .graph
        .as_mut()
        .ok_or(GraphError::ExternalData(ExternalDataError::Internal {
            reason: "entries が非空なのに model.graph が None",
        }))?;
    for (entry, bytes) in entries.into_iter().zip(loaded) {
        let target: &mut TensorProto = match entry.slot {
            TensorSlot::Initializer(idx) => g.initializer.get_mut(idx).ok_or(
                GraphError::ExternalData(ExternalDataError::Internal {
                    reason: "書き戻し先の initializer index が範囲外",
                }),
            )?,
            TensorSlot::NodeAttrTensor { node_idx, attr_idx } => g
                .node
                .get_mut(node_idx)
                .and_then(|n| n.attribute.get_mut(attr_idx))
                .and_then(|a| a.t.as_mut())
                .ok_or(GraphError::ExternalData(ExternalDataError::Internal {
                    reason: "書き戻し先の attribute テンソルが見つからない",
                }))?,
        };
        target.raw_data = bytes;
        target.data_location = data_location::DEFAULT;
        target.external_data.clear();
    }
    Ok(())
}

/// `ModelProto` を複製し external data を inline 化してから
/// `graph::build_graph` へ渡す新しい import 入口（イシュー #2347）。
///
/// `graph::build_graph`（バイト列入口が経由する既存関数）自体は変更しない
/// （A6）。`base_dir` は external な `location` の解決基点。
pub fn build_graph_with_external_data(
    model: &ModelProto,
    base_dir: &Path,
    options: &ExternalDataOptions,
) -> Result<super::graph::Graph, GraphError> {
    let mut cloned = model.clone();
    resolve_external_data(&mut cloned, base_dir, options)?;
    super::graph::build_graph(&cloned)
}

/// `no_follow_open` の 2 方式（Linux 限定 `openat2` と unix 全般の逐次
/// `openat(O_NOFOLLOW)` フォールバック）を直接呼び出す単体テスト。
/// `resolve_and_open` は Linux では既定で `openat2` 側を優先するため、
/// `openat2` が未対応（`ENOSYS`／`EPERM`）にならない通常の開発・CI 環境
/// では統合テスト（`tests/onnx_external_data.rs`）だけではフォールバック
/// 関数（`open_chain_component_walk`）自体が実行されない。ここで両関数を
/// 直接呼び、フォールバック経路も独立して検証する。
#[cfg(all(test, unix))]
mod tests {
    use std::ffi::OsStr;

    use super::no_follow_open;

    /// テスト専用の一時ディレクトリ（`tests/onnx_external_data.rs::
    /// TempDir` と同型。本ファイルはユニットテストのため独立実装する）。
    struct UnitTestDir(std::path::PathBuf);

    impl UnitTestDir {
        fn new(name: &str) -> Self {
            static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let dir = std::env::temp_dir().join(format!(
                "onnx-interop-external-data-unit-{}-{name}-{n}",
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).expect("一時ディレクトリの作成に失敗した");
            UnitTestDir(dir)
        }

        fn path(&self) -> &std::path::Path {
            &self.0
        }
    }

    impl Drop for UnitTestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// 両方式（`open_chain_component_walk`・Linux では `open_chain_openat2`
    /// も）が、通常ファイルへの正常な多段解決を成功させることを確認する。
    #[test]
    fn both_strategies_open_normal_nested_file() {
        let dir = UnitTestDir::new("normal");
        std::fs::create_dir_all(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("sub/f.data"), [1u8, 2, 3, 4]).unwrap();
        let base_dir_file = no_follow_open::open_base_dir(dir.path()).expect("base_dir open 失敗");

        let parts: Vec<&OsStr> = vec![OsStr::new("sub"), OsStr::new("f.data")];

        let walked = no_follow_open::open_chain_component_walk(&base_dir_file, &parts)
            .expect("component_walk は成功するはず");
        assert_eq!(
            std::io::Read::bytes(walked)
                .map(|b| b.unwrap())
                .collect::<Vec<u8>>(),
            vec![1, 2, 3, 4]
        );

        #[cfg(target_os = "linux")]
        {
            match no_follow_open::open_chain_openat2(&base_dir_file, &parts) {
                no_follow_open::Openat2Outcome::Opened(f) => {
                    assert_eq!(
                        std::io::Read::bytes(f)
                            .map(|b| b.unwrap())
                            .collect::<Vec<u8>>(),
                        vec![1, 2, 3, 4]
                    );
                }
                // 本 CI 環境の kernel が openat2 非対応の場合のみ許容する
                // （`ENOSYS`/`EPERM`。`resolve_and_open` 側は
                // `open_chain_component_walk` へフォールバックする経路と
                // 同じ判定）。
                no_follow_open::Openat2Outcome::Unsupported => {}
                no_follow_open::Openat2Outcome::Failed(e, is_symlink) => {
                    panic!("openat2 が失敗した（is_symlink={is_symlink}）: {e}")
                }
            }
        }
    }

    /// 両方式が、経路途中のシンボリックリンクを `ELOOP` 系の失敗として
    /// 拒否することを確認する（`is_symlink` フラグが true になること）。
    #[test]
    fn both_strategies_reject_symlink_component() {
        let dir = UnitTestDir::new("symlink");
        std::fs::write(dir.path().join("real.data"), [9u8; 4]).unwrap();
        std::os::unix::fs::symlink(dir.path().join("real.data"), dir.path().join("link.data"))
            .unwrap();
        let base_dir_file = no_follow_open::open_base_dir(dir.path()).expect("base_dir open 失敗");

        let parts: Vec<&OsStr> = vec![OsStr::new("link.data")];

        let (_e, is_symlink) = no_follow_open::open_chain_component_walk(&base_dir_file, &parts)
            .expect_err("component_walk はシンボリックリンクを拒否するはず");
        assert!(
            is_symlink,
            "component_walk: is_symlink フラグが立っていない"
        );

        #[cfg(target_os = "linux")]
        {
            match no_follow_open::open_chain_openat2(&base_dir_file, &parts) {
                no_follow_open::Openat2Outcome::Failed(_e, is_symlink) => {
                    assert!(is_symlink, "openat2: is_symlink フラグが立っていない");
                }
                no_follow_open::Openat2Outcome::Unsupported => {}
                no_follow_open::Openat2Outcome::Opened(_) => {
                    panic!("openat2 がシンボリックリンクを開いてしまった（TOCTOU 回帰）")
                }
            }
        }
    }
}
