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
//!    [`ExternalDataError::FileChangedDuringLoad`] とする（TOCTOU の窓を
//!    縮める。std に `O_NOFOLLOW` 相当が無いため完全な排除はできない。
//!    残るリスクは `docs/onnx-external-data-decision.md` に記録）。
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

/// `base_dir` を起点に `location` を検証しながら解決し、ファイルを開く。
/// 経路の途中を含めシンボリックリンクを拒否し、解決結果が `base_dir` の
/// 外に出ないことを canonicalize で多層防御する（A2）。
fn resolve_and_open(
    tensor_name: &str,
    base_dir_canonical: &Path,
    loc: &str,
) -> Result<OpenFile, ExternalDataError> {
    let parts =
        validate_location_string(loc).map_err(|reason| ExternalDataError::InvalidLocation {
            tensor_name: cap_name(tensor_name),
            reason,
        })?;

    let mut cur = base_dir_canonical.to_path_buf();
    for part in parts {
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

    Ok(OpenFile {
        file,
        len: meta.len(),
        #[cfg(unix)]
        dev_ino,
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

/// パス 1 で確定した「どこから何バイト読むか」の 1 件分。
struct LoadPlanEntry {
    slot: TensorSlot,
    tensor_name: String,
    file_key: PathBuf,
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
) -> Result<(Vec<LoadPlanEntry>, HashMap<PathBuf, OpenFile>), GraphError> {
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

    let mut files: HashMap<PathBuf, OpenFile> = HashMap::new();
    let mut regions: HashMap<PathBuf, Vec<(u64, u64, String)>> = HashMap::new();
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

        let opened = resolve_and_open(&tensor_name, base_dir_canonical, &location)
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

        // ファイルキー: canonicalize 済みパス（`resolve_and_open` が返した
        // ハンドルの実体を特定するため、パスではなくファイルの
        // metadata から再構築せず、単純に `location` 解決の入力に
        // 使った canonical パスを都度計算するのは二重コストになるため、
        // ここでは `resolve_and_open` 呼び出し直後に得られる情報のみで
        // 十分なキーを作る: base_dir + location 文字列の組はファイル
        // 実体と 1:1（symlink を拒否済みのため）なので、これをキーにする。
        let file_key = base_dir_canonical.join(&location);

        let interval_list = regions.entry(file_key.clone()).or_default();
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

        files.entry(file_key.clone()).or_insert(opened);

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
    files: &mut HashMap<PathBuf, OpenFile>,
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
