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
//!    max_total_bytes`]。確保の前に検査する）をすべて検証する。1 件でも
//!    失敗すれば `Err` を返しファイルは一切読まない（A04 資源枯渇対策）。
//! 2. **パス 2（`load`）**: パス 1 が全件成功した場合のみ、パス 1 で
//!    開いたファイルハンドルを再利用して該当区間だけを `read_exact` する
//!    （`.data` ファイル全体は読まない）。読み込み直前に `metadata().len()`
//!    （Unix では dev/ino も）をパス 1 の記録と再照合し、不一致は
//!    [`ExternalDataError::FileChangedDuringLoad`] とする。
//!    Linux／macOS（CI ビルド対象）では、そもそもパス 1 のファイル解決
//!    自体が `openat(O_NOFOLLOW)` によるディレクトリハンドル連鎖
//!    （`no_follow_open` モジュール）で行われ、検証済みの fd をそのまま
//!    保持するため経路文字列の再解決が発生せず、シンボリックリンク差し替え
//!    による TOCTOU 窓は構造的に生じない（`docs/
//!    onnx-external-data-decision.md` の残タスクを解消。#2347 P0 是正・
//!    PR #2348 コードレビュー対応）。上記 CI 対象外の他 unix
//!    ターゲットのみ `symlink_metadata` 逐次検証 → `canonicalize` →
//!    `File::open` という経路文字列再解決のフォールバック実装を使い、
//!    この場合に限り本節の窓（`metadata().len()` 再照合による縮小のみ）が
//!    残る。
//!
//! `checksum` キーは黙って無視せず [`ExternalDataError::
//! ChecksumUnsupported`] で fail-closed に拒否する（依存を追加できないため
//! SHA-1 検証は実装しない。no-silent-skip 契約。security.md A08）。
//!
//! ## facade への公開範囲
//!
//! 本モジュールは `onnx-interop` 内部限定であり facade へは公開しない
//! （承認待ち事項。`docs/compat-api-scope.md` §5・`docs/
//! facade-onnx-import-exposure-decision.md` 追補節）。

use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Component, Path, PathBuf};

use super::graph::{GraphError, element_count};
use super::proto::{ModelProto, TensorProto, cap_sparse_tensor_diag_name, data_location};

/// external data の合計サイズ上限の既定値（4 GiB）。
///
/// **暫定値・ユーザー承認待ち**（イシュー #2347 計画 §2「承認待ちの事項」）。
/// 変更は本定数 1 行の書き換えで済む。`ExternalDataOptions::default()` が
/// 参照する。
pub const DEFAULT_MAX_EXTERNAL_DATA_TOTAL_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// [`resolve_external_data`] の挙動を制御するオプション。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExternalDataOptions {
    /// 1 モデルあたりの external data 合計バイト数の上限。超過は
    /// [`ExternalDataError::TotalSizeLimitExceeded`] で拒否する（確保の
    /// 前に検査するため、この上限を超える `length` はメモリを確保しない）。
    pub max_total_bytes: u64,
}

impl Default for ExternalDataOptions {
    fn default() -> Self {
        ExternalDataOptions {
            max_total_bytes: DEFAULT_MAX_EXTERNAL_DATA_TOTAL_BYTES,
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

/// ディレクトリハンドル（fd）を起点に `O_NOFOLLOW` で各パス成分を逐次
/// オープンする実装（P0 対応。discussion_r4119392011）。
///
/// 旧実装は `symlink_metadata` で各段を検証したうえで `canonicalize` →
/// `File::open(&canonical)` とパス文字列から**再度**ファイルを開いていた。
/// この「検証」と「再オープン」の間に窓（TOCTOU）があり、検証後・オープン
/// 前にファイルシステム上でディレクトリ成分がシンボリックリンクへ
/// 差し替えられると、検証済みのはずの経路が `base_dir` の外を指す実体を
/// 開いてしまい得た。
///
/// 本実装は経路を文字列として再解決しない。`base_dir` を開いた
/// ディレクトリ fd を起点に、各パス成分を `openat(dirfd, name,
/// O_NOFOLLOW)` でその fd に対して相対的に開き、得られた fd をそのまま
/// 次段の起点にする。`O_NOFOLLOW` により対象がシンボリックリンクなら
/// `ELOOP` で即座に失敗するため、検証済みの fd 連鎖以外を辿る余地が
/// 生じない（カーネルが 1 段のパス解決をアトミックに行うことに依拠する。
/// `docs/onnx-external-data-decision.md` に残タスクとして記録していた
/// 「std に `O_NOFOLLOW` 相当が無いため完全な排除はできない」という制約は
/// 本実装（std を経由せず `extern "C"` で `openat` を直接呼ぶ）で解消する）。
///
/// Linux／macOS 限定（CI ビルド対象〈`cargo build (linux /
/// aarch64-apple-darwin)`〉と同一。フラグ定数値が OS ごとに異なるため、
/// 実測未検証の他 unix では使わない。対象外の target では下方の
/// `#[cfg(not(any(target_os = "linux", target_os = "macos")))]` 版の
/// `resolve_and_open`（従来実装）にフォールバックする）。
#[cfg(any(target_os = "linux", target_os = "macos"))]
mod no_follow_open {
    use std::ffi::{CString, c_char, c_int};
    use std::fs::File;
    use std::io;
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::io::{AsRawFd, FromRawFd};
    use std::path::Path;

    // `openat(2)` のフラグ値（`libc` crate 相当の定数を手書き。本クレートは
    // 許容依存 9 区分〈deps-policy.md〉に `libc` を含まないためユーザー
    // 承認なしに追加できない。std がリンクする libc は常に存在するため、
    // 新規クレート依存を増やさず `extern "C"` で直接呼ぶ）。値は OS ごとの
    // `fcntl.h` 定義に一致させる。
    #[cfg(target_os = "linux")]
    mod flags {
        pub const O_RDONLY: i32 = 0;
        pub const O_DIRECTORY: i32 = 0o200_000;
        pub const O_NOFOLLOW: i32 = 0o400_000;
        pub const O_CLOEXEC: i32 = 0o2_000_000;
    }
    #[cfg(target_os = "macos")]
    mod flags {
        pub const O_RDONLY: i32 = 0x0000_0000;
        pub const O_DIRECTORY: i32 = 0x0010_0000;
        pub const O_NOFOLLOW: i32 = 0x0000_0100;
        pub const O_CLOEXEC: i32 = 0x0100_0000;
    }
    use flags::{O_CLOEXEC, O_DIRECTORY, O_NOFOLLOW, O_RDONLY};

    // ELOOP（"too many levels of symbolic links"）: Linux/macOS 共通で
    // `O_NOFOLLOW` 指定時に対象がシンボリックリンクだと返る errno。
    #[cfg(target_os = "linux")]
    const ELOOP: i32 = 40;
    #[cfg(target_os = "macos")]
    const ELOOP: i32 = 62;
    // ENOTDIR（"not a directory"）: Linux／macOS 共通で 20。`O_DIRECTORY|
    // O_NOFOLLOW` で開いた対象がシンボリックリンクだった場合、カーネルは
    // `ELOOP` ではなく `ENOTDIR` を返す（実機実測。シンボリックリンクは
    // 「ディレクトリではない」ためこちらが優先される）。この値は「対象が
    // シンボリックリンク」と「対象が単なる非ディレクトリの通常ファイル」の
    // 両方で返るため、下の [`open_chain_no_follow`] は診断用の
    // `symlink_metadata`（open 失敗**後**の分類専用。安全性判断には使わず、
    // 既に fail-closed で拒否済みの結果をどちらのエラー種別として
    // 報告するかにのみ使う）で判別する。呼び出し元（外側の
    // `resolve_and_open`）も同じ値で `NotRegularFile` 判定を行うため
    // `pub(super)` で公開し、値を二重管理しない。
    pub(super) const ENOTDIR: i32 = 20;

    // SAFETY契約: `openat` は POSIX 標準関数で、std バイナリには常に libc が
    // リンクされているため crate 追加なしに呼び出せる。呼び出し側
    // （`openat_no_follow`）が引数の有効性（fd の生存・C 文字列の NUL 終端）
    // を保証する。
    unsafe extern "C" {
        // POSIX の実プロトタイプは `int openat(int, const char *, int, ...)`
        // （`O_CREAT` 指定時のみ第 4 引数 `mode_t` を使う可変長引数）。
        // 固定 3 引数で宣言すると、可変長引数呼び出し規約が固定引数と
        // 異なる ABI（Apple arm64 等）で不一致になり得るため、シグネチャを
        // 可変長引数のまま宣言する（呼び出し側は `O_CREAT` を渡さないため
        // 可変長引数を実際には渡さない）。
        fn openat(dirfd: c_int, pathname: *const c_char, flags: c_int, ...) -> c_int;
    }

    /// シンボリックリンク検知（`ELOOP`）かどうかを判定する。
    fn is_eloop(e: &io::Error) -> bool {
        e.raw_os_error() == Some(ELOOP)
    }

    /// `dir` に対して相対的に `name`（単一パス成分。`..`／`/` を含まない
    /// `Path::components()` の `Normal`由来の値のみを渡す前提）を
    /// `O_NOFOLLOW` で開く。`want_dir` が true なら `O_DIRECTORY` を付け、
    /// 対象がディレクトリでなければ失敗する。
    fn openat_no_follow(dir: &File, name: &std::ffi::OsStr, want_dir: bool) -> io::Result<File> {
        let c_name = CString::new(name.as_bytes())
            .map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
        let flags = if want_dir {
            O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC
        } else {
            O_RDONLY | O_NOFOLLOW | O_CLOEXEC
        };
        // SAFETY: `dir.as_raw_fd()` はこの呼び出しの間生存している `dir` が
        // 所有する有効な open ディレクトリ fd。`c_name` は
        // `CString::new` が NUL 終端を保証した有効な C 文字列で、この
        // 呼び出しの間生存する。返り値が非負なら新規に確保された fd の
        // 所有権を呼び出し元へ渡す契約（POSIX `openat(2)`）であり、
        // `File::from_raw_fd` で即座に `File` へ委譲することで二重解放・
        // リークを防ぐ。
        let fd = unsafe { openat(dir.as_raw_fd(), c_name.as_ptr(), flags) };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: 直前の `openat` が返した非負 fd は呼び出し元がここで
        // 一意に所有権を得る新規 fd であり、他のどのコードもまだ
        // 参照していない。
        Ok(unsafe { File::from_raw_fd(fd) })
    }

    /// `base_dir` を起点に `parts` を 1 段ずつ `O_NOFOLLOW` で辿り、最終
    /// ファイルの fd を得る。中間段は `O_DIRECTORY` 付きで開くためディレクトリ
    /// でなければ失敗し、途中経路のシンボリックリンクは `ELOOP` で拒否される。
    /// エラーはシンボリックリンク検知か否かを呼び出し元が判別できるよう
    /// `(io::Error, bool /* is_symlink */)` を返す。
    pub(super) fn open_chain_no_follow(
        base_dir: &Path,
        parts: &[&std::ffi::OsStr],
    ) -> Result<File, (io::Error, bool)> {
        if parts.is_empty() {
            return Err((io::Error::from(io::ErrorKind::InvalidInput), false));
        }
        // `base_dir` はモジュール冒頭コメントのとおり呼び出し元が与える
        // 信頼済み入力（`resolve_external_data` が canonicalize 済みの値を
        // 渡す）のため、素直に `File::open` する。
        let mut dir = File::open(base_dir).map_err(|e| (e, false))?;
        let last_idx = parts.len() - 1;
        // エラー分類専用（診断用）の累積パス。実際のファイルオープンには
        // 使わない（オープンは常に `dir` の fd を起点にした `openat` 経由）。
        let mut accumulated = base_dir.to_path_buf();
        for (i, part) in parts.iter().enumerate() {
            accumulated.push(part);
            let want_dir = i != last_idx;
            match openat_no_follow(&dir, part, want_dir) {
                Ok(next) => dir = next,
                Err(e) => {
                    let is_sym = is_eloop(&e)
                        || (want_dir
                            && e.raw_os_error() == Some(ENOTDIR)
                            && std::fs::symlink_metadata(&accumulated)
                                .map(|m| m.file_type().is_symlink())
                                .unwrap_or(false));
                    return Err((e, is_sym));
                }
            }
        }
        Ok(dir)
    }
}

/// `base_dir` を起点に `location` を検証しながら解決し、ファイルを開く。
/// 経路の途中を含めシンボリックリンクを拒否し、`O_NOFOLLOW` によるディレクトリ
/// ハンドル連鎖オープン（[`no_follow_open::open_chain_no_follow`]）で
/// 「検証した経路そのもの」を開くことを保証する（A2・P0 対応。
/// discussion_r4119392011）。返り値の第 2 要素は `plan` が `file_key_for`
/// のフォールバック（dev/ino を持たないプラットフォーム）で使う、
/// `parts` から再構築した正規化済み相対パス。
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn resolve_and_open(
    tensor_name: &str,
    base_dir_canonical: &Path,
    loc: &str,
) -> Result<(OpenFile, PathBuf), ExternalDataError> {
    let parts =
        validate_location_string(loc).map_err(|reason| ExternalDataError::InvalidLocation {
            tensor_name: cap_name(tensor_name),
            reason,
        })?;

    let file = no_follow_open::open_chain_no_follow(base_dir_canonical, &parts).map_err(
        |(e, is_symlink)| {
            if is_symlink {
                ExternalDataError::InvalidLocation {
                    tensor_name: cap_name(tensor_name),
                    reason: LocationRejectReason::Symlink,
                }
            } else if e.kind() == std::io::ErrorKind::NotADirectory
                || e.raw_os_error() == Some(no_follow_open::ENOTDIR)
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
        },
    )?;

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

/// [`resolve_and_open`] の Linux／macOS 以外向けフォールバック実装。
/// `symlink_metadata` による逐次検証 → `canonicalize` → `File::open` と
/// パスから再オープンする（旧実装のまま）。CI ビルド対象
/// （linux・aarch64-apple-darwin）はいずれも上の `no_follow_open` 経路を
/// 使うため、本フォールバックは実測未検証の他 unix 向けの保守的な
/// 代替実装であり、TOCTOU 窓の完全な排除は保証しない（コメントに明記して
/// 既知の制約として残す）。
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn resolve_and_open(
    tensor_name: &str,
    base_dir_canonical: &Path,
    loc: &str,
) -> Result<(OpenFile, PathBuf), ExternalDataError> {
    let parts =
        validate_location_string(loc).map_err(|reason| ExternalDataError::InvalidLocation {
            tensor_name: cap_name(tensor_name),
            reason,
        })?;

    let mut cur = base_dir_canonical.to_path_buf();
    for part in &parts {
        cur.push(part);
        let meta = std::fs::symlink_metadata(&cur).map_err(|e| ExternalDataError::Io {
            tensor_name: cap_name(tensor_name),
            kind: e.kind(),
        })?;
        if meta.file_type().is_symlink() {
            return Err(ExternalDataError::InvalidLocation {
                tensor_name: cap_name(tensor_name),
                reason: LocationRejectReason::Symlink,
            });
        }
        if !meta.is_dir() && !meta.is_file() {
            return Err(ExternalDataError::InvalidLocation {
                tensor_name: cap_name(tensor_name),
                reason: LocationRejectReason::NotRegularFile,
            });
        }
    }

    let canonical = cur.canonicalize().map_err(|e| ExternalDataError::Io {
        tensor_name: cap_name(tensor_name),
        kind: e.kind(),
    })?;
    if !canonical.starts_with(base_dir_canonical) {
        return Err(ExternalDataError::InvalidLocation {
            tensor_name: cap_name(tensor_name),
            reason: LocationRejectReason::OutsideBaseDir,
        });
    }

    let file = File::open(&canonical).map_err(|e| ExternalDataError::Io {
        tensor_name: cap_name(tensor_name),
        kind: e.kind(),
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

    #[cfg(unix)]
    let dev_ino = {
        use std::os::unix::fs::MetadataExt;
        (meta.dev(), meta.ino())
    };

    let normalized_rel: PathBuf = parts.iter().collect();
    Ok((
        OpenFile {
            file,
            len: meta.len(),
            #[cfg(unix)]
            dev_ino,
        },
        normalized_rel,
    ))
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
        let mut buf = vec![0u8; entry.length as usize];
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
