//! safetensors ワイヤフォーマットのパースと自作 `Tensor<f32>` へのマッピング
//! （TASK-7.1a・#73・REQ-7）。
//!
//! `onnx-interop` クレート（本クレート）は ONNX / safetensors 相互運用層
//! （`crate` ルート `lib.rs` 参照）を担い、本モジュールはその safetensors
//! 経路を提供する。`safetensors` クレートはワイヤフォーマットの読み書き
//! のみに用い、`tensor-core::Tensor<f32>` へのマッピングは自作する
//! （`.claude/rules/deps-policy.md`）。呼び出し元は PyTorch で保存した
//! 重みファイル（`.safetensors`）を本モジュール経由でロードし、
//! `tensor-core::Tensor::transpose_2d()` 等の後続 API へ渡す想定である。
//!
//! ## REQ-7 が明示する 2 つの契約（`docs/spec/04-requirements.md`）
//!
//! 1. **暗黙アダプタを設けない**: PyTorch `nn.Linear.weight` は
//!    `[out_features, in_features]` で保存される。matmul の右辺
//!    （`[in_features, out_features]`）へ変換する転置は本モジュールでは
//!    一切行わない。呼び出し側が `Tensor::transpose_2d()` を明示的に
//!    呼ぶ（PoC-v2-6 `mlp.rs` の `Mlp::from_safetensors` と同型の設計）。
//! 2. **無言 skip 禁止**: 期待するキー集合に対する充足検査
//!    （[`require_keys`]）は不足キーを**全件**収集して返す。呼び出し元が
//!    最初の 1 件だけを見て見落とすことを避ける。
//!
//! ## 検証順序（OWASP A03。`.claude/rules/security.md`）
//!
//! 各テンソルについて、データ変換（バイト列 → `f32`）より**前**に
//! (1) `SafeTensors::deserialize` によるヘッダ・レイアウト整合検査 →
//! (2) dtype 検査（F32 以外は即エラー） → (3) データ長が shape の要素数積
//! と一致するかの検査 → (4) `Tensor::new` の shape 検査（要素数積の
//! オーバーフローは `tensor-core::ShapeError::ElementCountOverflow` に
//! 委譲） の順で行う。

use std::collections::HashMap;
use std::fmt;
use std::path::Path;

use fandhe_ai_tensor_core::{ShapeError, Tensor, cast_from_f32, cast_to_f32};
use half::{bf16, f16};
use safetensors::Dtype;
use safetensors::tensor::SafeTensors;

/// safetensors ロード経路の型付きエラー。
///
/// `#[non_exhaustive]` を付す理由: 公開 API 非破壊はガードレール条件
/// （`.claude/rules/security.md`）であり、後続タスク（ONNX 経路等）で
/// variant が増えても呼び出し側の網羅的 match を破壊しないため
/// （`fandhe_ai_tensor_core::ShapeError` と同じ方針）。
#[non_exhaustive]
#[derive(Debug)]
pub enum LoadError {
    /// ファイル読み込み（`load_safetensors_f32`）の I/O エラー。
    Io(std::io::Error),
    /// `SafeTensors::deserialize`／`tensor()` が返すヘッダ・レイアウト
    /// 不整合（バイト列が safetensors フォーマットとして壊れている）。
    SafetensorsFormat(String),
    /// [`require_keys`] が検出した不足キー（**全件**。無言 skip 禁止の
    /// 契約を診断性の面から強化する）。
    MissingKeys(Vec<String>),
    /// 対応外の dtype（`*_f32` 系では F32 以外、`*_typed` 系では
    /// [`SafetensorsDtype`] の 7 種以外。明示エラーとして扱い無言で飛ばさない）。
    UnsupportedDtype { key: String, dtype: String },
    /// `*_typed` 系で BOOL テンソルに 0／1 以外のバイトが含まれていた
    /// （イシュー #3845・設計記録 §13。不正値を `true` に丸めず拒否する）。
    InvalidBoolValue {
        key: String,
        index: usize,
        value: u8,
    },
    /// テンソルの生データ長（バイト数 / 4）が safetensors ヘッダの shape
    /// が要求する要素数積と一致しない（ヘッダとデータ本体の不整合）。
    ///
    /// 注記: safetensors 0.7.0 は `deserialize` 内部の
    /// `Metadata::validate` で同じ不変条件（`data_offsets` 区間長と
    /// shape 要素数積 × dtype サイズの一致）を検査するため、通常この
    /// variant に到達する前に [`LoadError::SafetensorsFormat`] として
    /// 弾かれる（`tests/st_load.rs` の
    /// `truncated_data_is_rejected_before_conversion` で確認済み）。
    /// 本 variant は将来の safetensors バージョン変更や `SafeTensors`
    /// 以外の入力経路が追加された場合に備えた多層防御として保持する
    /// （OWASP A03「外部入力を信頼しない」。`.claude/rules/security.md`）。
    DataLengthMismatch {
        key: String,
        expected: usize,
        actual: usize,
    },
    /// `Tensor::new` が返す shape 検査エラー（要素数積オーバーフロー等）。
    Shape(ShapeError),
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoadError::Io(e) => write!(f, "IO エラー: {e}"),
            LoadError::SafetensorsFormat(e) => write!(f, "safetensors 形式エラー: {e}"),
            LoadError::MissingKeys(keys) => {
                write!(f, "必須テンソルキーが見つかりません: {}", keys.join(", "))
            }
            LoadError::UnsupportedDtype { key, dtype } => {
                write!(f, "未対応の dtype（key={key}）: {dtype}")
            }
            LoadError::DataLengthMismatch {
                key,
                expected,
                actual,
            } => write!(
                f,
                "テンソルデータ長不整合（key={key}）: shape が要求する要素数 {expected}、実データ要素数 {actual}"
            ),
            LoadError::InvalidBoolValue { key, index, value } => write!(
                f,
                "BOOL テンソルに 0／1 以外の値（key={key}, index={index}）: {value}"
            ),
            LoadError::Shape(e) => write!(f, "shape エラー: {e}"),
        }
    }
}

impl std::error::Error for LoadError {}

/// safetensors バイト列のレイアウト（ヘッダ長・ヘッダ JSON・オフセット連続性・
/// shape×dtype のバイト長・全体長の一致）だけを検証する。dtype は限定せず、テンソルも確保しない。
///
/// facade `model.rs` の公開前検証（`ModelRegistry::download`。#3126・
/// `docs/model-download-design.md` §5 方式 (ii)）から呼ばれる。F32 以外（F16／BF16 等）の
/// 正当なファイルを `load` 側の dtype 制約で拒否しないために、F32 ロード
/// （[`load_safetensors_f32_from_bytes`]）とは別関数として切り出した。facade は
/// `safetensors` に直接依存できない（許容依存の区分上、相互運用ラッパー経由に限る）ため、
/// 公開面への追加（追加のみ・既存関数は不変）とした。
pub fn validate_safetensors_layout_from_bytes(bytes: &[u8]) -> Result<(), LoadError> {
    SafeTensors::read_metadata(bytes)
        .map(|_| ())
        .map_err(|e| LoadError::SafetensorsFormat(format!("{e:?}")))
}

/// safetensors バイト列を `name -> Tensor<f32>` のマップへ変換する。
///
/// [`load_safetensors_f32`] から呼ばれる本体実装。バイト列を直接受け取る
/// 形にしてあるのは、テスト（`tests/st_load.rs`）が破損データ・
/// 未対応 dtype 等の異常系をファイル I/O なしに構築できるようにするため。
///
/// dtype が F32 以外のテンソルは [`LoadError::UnsupportedDtype`] として
/// 報告する（無言でスキップしない。呼び出し元がキー不足を検知できる
/// ようにするため）。転置・キーリネームは一切行わない
/// （モジュール冒頭ドキュメント参照）。
pub fn load_safetensors_f32_from_bytes(
    bytes: &[u8],
) -> Result<HashMap<String, Tensor<f32>>, LoadError> {
    let st = SafeTensors::deserialize(bytes)
        .map_err(|e| LoadError::SafetensorsFormat(format!("{e:?}")))?;

    let mut out = HashMap::new();
    for name in st.names() {
        let view = st
            .tensor(name)
            .map_err(|e| LoadError::SafetensorsFormat(format!("{name}: {e:?}")))?;

        // (2) dtype 検査をデータ変換より先に行う（無言 skip 禁止）。
        if view.dtype() != Dtype::F32 {
            return Err(LoadError::UnsupportedDtype {
                key: name.to_string(),
                dtype: format!("{:?}", view.dtype()),
            });
        }

        let shape: Vec<usize> = view.shape().to_vec();
        let raw = view.data();

        // (3) データ長検査をバイト変換より先に行う。F32 は 4 バイト固定
        // （`chunks_exact(4)` は端数バイトを黙って切り捨てるため、事前に
        // 長さの整合を確認しないと末尾破損データを見逃す）。
        let expected_elems: usize = shape.iter().product();
        if raw.len() % 4 != 0 || raw.len() / 4 != expected_elems {
            return Err(LoadError::DataLengthMismatch {
                key: name.to_string(),
                expected: expected_elems,
                actual: raw.len() / 4,
            });
        }

        // safetensors のテンソルデータは仕様上常にリトルエンディアン。
        // `unsafe` は使わず `from_le_bytes` で変換する。
        let data: Vec<f32> = raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();

        // (4) shape 検査（要素数積オーバーフロー等）は Tensor::new に委譲。
        let tensor = Tensor::new(data, &shape).map_err(LoadError::Shape)?;
        out.insert(name.to_string(), tensor);
    }
    Ok(out)
}

/// `path` の safetensors ファイルを読み込み、`name -> Tensor<f32>` の
/// マップを返す（`std::fs::read` + [`load_safetensors_f32_from_bytes`] の
/// 薄い合成）。
pub fn load_safetensors_f32(path: &Path) -> Result<HashMap<String, Tensor<f32>>, LoadError> {
    let bytes = std::fs::read(path).map_err(LoadError::Io)?;
    load_safetensors_f32_from_bytes(&bytes)
}

/// 期待するキー集合がすべて `map` に存在するか検査する
/// （REQ-7「無言 skip 禁止」契約の実体）。
///
/// 不足キーは**全件**収集して [`LoadError::MissingKeys`] で返す
/// （最初の 1 件のみ報告すると、複数キーが不足している場合に
/// 呼び出し元が 1 件ずつしか気づけず修正が後手に回るため）。
pub fn require_keys(map: &HashMap<String, Tensor<f32>>, keys: &[&str]) -> Result<(), LoadError> {
    let missing: Vec<String> = keys
        .iter()
        .filter(|k| !map.contains_key(**k))
        .map(|k| (*k).to_string())
        .collect();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(LoadError::MissingKeys(missing))
    }
}

// ---------------------------------------------------------------------
// dtype 混在ファイルの読込（イシュー #3845・設計記録 §13 案 C・読込側）
// ---------------------------------------------------------------------

/// `*_typed` 系が扱う safetensors dtype のタグ（7 種）。
///
/// `safetensors::Dtype` は公開面へ出さない（許容依存の型を facade の安定性保証へ
/// 持ち込まないため）。呼び出し元は [`SafetensorsTensor::dtype`] で判定する。
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SafetensorsDtype {
    F32,
    F16,
    BF16,
    F64,
    I32,
    I64,
    Bool,
}

impl SafetensorsDtype {
    /// safetensors ヘッダの `"dtype"` 文字列（PyTorch・safetensors 表記と一致）。
    pub fn header_name(&self) -> &'static str {
        match self {
            SafetensorsDtype::F32 => "F32",
            SafetensorsDtype::F16 => "F16",
            SafetensorsDtype::BF16 => "BF16",
            SafetensorsDtype::F64 => "F64",
            SafetensorsDtype::I32 => "I32",
            SafetensorsDtype::I64 => "I64",
            SafetensorsDtype::Bool => "BOOL",
        }
    }

    /// 1 要素のバイト幅（safetensors の `Dtype::size()` には頼らない）。
    fn element_width(self) -> usize {
        match self {
            SafetensorsDtype::Bool => 1,
            SafetensorsDtype::F16 | SafetensorsDtype::BF16 => 2,
            SafetensorsDtype::F32 | SafetensorsDtype::I32 => 4,
            SafetensorsDtype::F64 | SafetensorsDtype::I64 => 8,
        }
    }

    /// 対応 7 種からワイヤ dtype へ（書出側 `st_save` が使う `from_wire` の逆写像）。
    /// ワイルドカードを置かず、variant 追加時にコンパイルエラーで気付けるようにする。
    pub(crate) fn to_wire(self) -> Dtype {
        match self {
            SafetensorsDtype::F32 => Dtype::F32,
            SafetensorsDtype::F16 => Dtype::F16,
            SafetensorsDtype::BF16 => Dtype::BF16,
            SafetensorsDtype::F64 => Dtype::F64,
            SafetensorsDtype::I32 => Dtype::I32,
            SafetensorsDtype::I64 => Dtype::I64,
            SafetensorsDtype::Bool => Dtype::BOOL,
        }
    }

    /// ワイヤ dtype から対応 7 種へ。`_ => None` で fail-closed
    /// （`Dtype` は `#[non_exhaustive]`、F4 等のサブバイト型は `size()` が 0）。
    fn from_wire(d: Dtype) -> Option<Self> {
        match d {
            Dtype::F32 => Some(SafetensorsDtype::F32),
            Dtype::F16 => Some(SafetensorsDtype::F16),
            Dtype::BF16 => Some(SafetensorsDtype::BF16),
            Dtype::F64 => Some(SafetensorsDtype::F64),
            Dtype::I32 => Some(SafetensorsDtype::I32),
            Dtype::I64 => Some(SafetensorsDtype::I64),
            Dtype::BOOL => Some(SafetensorsDtype::Bool),
            _ => None,
        }
    }
}

/// dtype 混在ファイルの 1 テンソル（値は読み込んだバイト列と bit 単位で一致）。
///
/// f32 への昇格は暗黙に行わず、明示的な [`SafetensorsTensor::to_f32`] のみで行う。
/// 既存の `load_safetensors_f32*` は F32 以外を拒否したままである。
/// `half` は facade から再エクスポートしない（`F16`／`BF16` を名指ししたい利用者は
/// `half` に直接依存する。判定だけなら [`SafetensorsTensor::dtype`] で足りる）。
///
/// 対応 dtype: F32／F16／BF16／F64／I32／I64／BOOL。変換 `to_f32` は
/// F32 恒等・F16／BF16 は完全表現・F64／I64／I32 は丸め（非可逆）・BOOL は 0／1。
///
/// PyTorch 側での確認手順（例）:
///
/// ```text
/// from safetensors.torch import load_file
/// t = load_file("model.safetensors")   # dtype はヘッダ表記のまま保たれる
/// print({k: v.dtype for k, v in t.items()})
/// ```
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum SafetensorsTensor {
    F32(Tensor<f32>),
    F16(Tensor<f16>),
    BF16(Tensor<bf16>),
    F64(Tensor<f64>),
    I32(Tensor<i32>),
    I64(Tensor<i64>),
    Bool(Tensor<bool>),
}

/// f32 の確保量が `isize::MAX` バイトを超えないことを確かめる
/// （`CastElement` 未実装の f16／bf16 向けに `cast` 系と同じ fail-closed 検査を再現）。
fn check_f32_alloc(shape: &[usize]) -> Result<(), ShapeError> {
    // host_slice() は非 contiguous view を実体化する（Vec::with_capacity）ため、
    // 呼ぶ前に shape から checked な要素数積で検査する（panic 回避）。
    let numel = shape
        .iter()
        .try_fold(1usize, |a, &d| a.checked_mul(d))
        .ok_or(ShapeError::ElementCountOverflow)?;
    match numel.checked_mul(std::mem::size_of::<f32>()) {
        Some(b) if b <= isize::MAX as usize => Ok(()),
        _ => Err(ShapeError::ElementCountOverflow),
    }
}

impl SafetensorsTensor {
    /// このテンソルの dtype タグ。
    pub fn dtype(&self) -> SafetensorsDtype {
        match self {
            SafetensorsTensor::F32(_) => SafetensorsDtype::F32,
            SafetensorsTensor::F16(_) => SafetensorsDtype::F16,
            SafetensorsTensor::BF16(_) => SafetensorsDtype::BF16,
            SafetensorsTensor::F64(_) => SafetensorsDtype::F64,
            SafetensorsTensor::I32(_) => SafetensorsDtype::I32,
            SafetensorsTensor::I64(_) => SafetensorsDtype::I64,
            SafetensorsTensor::Bool(_) => SafetensorsDtype::Bool,
        }
    }

    /// テンソルの shape。
    pub fn shape(&self) -> &[usize] {
        match self {
            SafetensorsTensor::F32(t) => t.shape(),
            SafetensorsTensor::F16(t) => t.shape(),
            SafetensorsTensor::BF16(t) => t.shape(),
            SafetensorsTensor::F64(t) => t.shape(),
            SafetensorsTensor::I32(t) => t.shape(),
            SafetensorsTensor::I64(t) => t.shape(),
            SafetensorsTensor::Bool(t) => t.shape(),
        }
    }

    /// 明示的な f32 への変換（opt-in）。F64／I32／I64／BOOL は
    /// `tensor-core::cast_to_f32` に委譲する（単一情報源）。
    pub fn to_f32(&self) -> Result<Tensor<f32>, ShapeError> {
        match self {
            SafetensorsTensor::F32(t) => Ok(t.clone()),
            SafetensorsTensor::F64(t) => cast_to_f32(t),
            SafetensorsTensor::I32(t) => cast_to_f32(t),
            SafetensorsTensor::I64(t) => cast_to_f32(t),
            SafetensorsTensor::Bool(t) => cast_to_f32(t),
            SafetensorsTensor::F16(t) => {
                check_f32_alloc(t.shape())?;
                let src = t.host_slice();
                let data: Vec<f32> = src.iter().map(|v| v.to_f32()).collect();
                Tensor::new(data, t.shape())
            }
            SafetensorsTensor::BF16(t) => {
                check_f32_alloc(t.shape())?;
                let src = t.host_slice();
                let data: Vec<f32> = src.iter().map(|v| v.to_f32()).collect();
                Tensor::new(data, t.shape())
            }
        }
    }

    /// f32 テンソルから指定 dtype へ明示変換する（opt-in）。F16／BF16 は最近接偶数丸め
    /// （範囲超過は ±inf）、BOOL は `v != 0.0`（NaN は true。`CastElement` の規則）。
    pub fn from_f32(t: &Tensor<f32>, dtype: SafetensorsDtype) -> Result<Self, ShapeError> {
        match dtype {
            SafetensorsDtype::F32 => Ok(SafetensorsTensor::F32(t.clone())),
            SafetensorsDtype::F64 => cast_from_f32::<f64>(t).map(SafetensorsTensor::F64),
            SafetensorsDtype::I32 => cast_from_f32::<i32>(t).map(SafetensorsTensor::I32),
            SafetensorsDtype::I64 => cast_from_f32::<i64>(t).map(SafetensorsTensor::I64),
            SafetensorsDtype::Bool => cast_from_f32::<bool>(t).map(SafetensorsTensor::Bool),
            SafetensorsDtype::F16 => {
                check_f32_alloc(t.shape())?;
                let src = t.host_slice();
                let data: Vec<f16> = src.iter().map(|v| f16::from_f32(*v)).collect();
                Tensor::new(data, t.shape()).map(SafetensorsTensor::F16)
            }
            SafetensorsDtype::BF16 => {
                check_f32_alloc(t.shape())?;
                let src = t.host_slice();
                let data: Vec<bf16> = src.iter().map(|v| bf16::from_f32(*v)).collect();
                Tensor::new(data, t.shape()).map(SafetensorsTensor::BF16)
            }
        }
    }
}

/// safetensors バイト列を dtype 混在のまま `name -> SafetensorsTensor` へ変換する
/// （読込側。[`load_safetensors_typed`] から呼ばれる）。
///
/// 検証順: ヘッダ・レイアウト → dtype（要素幅の計算より前に fail-closed）→
/// checked な要素数積と長さ一致 → 変換（BOOL は同じループで 0／1 検査）→ `Tensor::new`。
/// 名前を昇順で走査するため、複数の不正があっても報告されるエラーは決定的。
/// 値は bit 単位で保たれ、f32 への昇格は行わない。
///
/// ```
/// use fandhe_ai_onnx_interop::st_load::{load_safetensors_typed_from_bytes, SafetensorsTensor};
/// let v = safetensors::tensor::TensorView::new(
///     safetensors::Dtype::I64, vec![2], &[1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
/// ).unwrap();
/// let bytes = safetensors::tensor::serialize([("n", v)], None).unwrap();
/// let m = load_safetensors_typed_from_bytes(&bytes).unwrap();
/// assert!(matches!(m["n"], SafetensorsTensor::I64(_)));
/// ```
pub fn load_safetensors_typed_from_bytes(
    bytes: &[u8],
) -> Result<HashMap<String, SafetensorsTensor>, LoadError> {
    let st = SafeTensors::deserialize(bytes)
        .map_err(|e| LoadError::SafetensorsFormat(format!("{e:?}")))?;

    let mut names = st.names();
    names.sort();

    let mut out = HashMap::new();
    for name in names {
        let view = st
            .tensor(name)
            .map_err(|e| LoadError::SafetensorsFormat(format!("{name}: {e:?}")))?;

        let Some(dtype) = SafetensorsDtype::from_wire(view.dtype()) else {
            return Err(LoadError::UnsupportedDtype {
                key: name.to_string(),
                dtype: format!("{:?}", view.dtype()),
            });
        };

        let shape: Vec<usize> = view.shape().to_vec();
        let raw = view.data();
        let width = dtype.element_width();

        let expected = shape
            .iter()
            .try_fold(1usize, |a, &d| a.checked_mul(d))
            .ok_or(LoadError::Shape(ShapeError::ElementCountOverflow))?;
        if raw.len() % width != 0 || raw.len() / width != expected {
            return Err(LoadError::DataLengthMismatch {
                key: name.to_string(),
                expected,
                actual: raw.len() / width,
            });
        }

        // safetensors のデータは常にリトルエンディアン。`unsafe` なしで変換する。
        let tensor = match dtype {
            SafetensorsDtype::F32 => SafetensorsTensor::F32(
                Tensor::new(
                    raw.as_chunks::<4>()
                        .0
                        .iter()
                        .map(|b| f32::from_le_bytes(*b))
                        .collect(),
                    &shape,
                )
                .map_err(LoadError::Shape)?,
            ),
            SafetensorsDtype::F16 => SafetensorsTensor::F16(
                Tensor::new(
                    raw.as_chunks::<2>()
                        .0
                        .iter()
                        .map(|b| f16::from_bits(u16::from_le_bytes(*b)))
                        .collect(),
                    &shape,
                )
                .map_err(LoadError::Shape)?,
            ),
            SafetensorsDtype::BF16 => SafetensorsTensor::BF16(
                Tensor::new(
                    raw.as_chunks::<2>()
                        .0
                        .iter()
                        .map(|b| bf16::from_bits(u16::from_le_bytes(*b)))
                        .collect(),
                    &shape,
                )
                .map_err(LoadError::Shape)?,
            ),
            SafetensorsDtype::F64 => SafetensorsTensor::F64(
                Tensor::new(
                    raw.as_chunks::<8>()
                        .0
                        .iter()
                        .map(|b| f64::from_le_bytes(*b))
                        .collect(),
                    &shape,
                )
                .map_err(LoadError::Shape)?,
            ),
            SafetensorsDtype::I32 => SafetensorsTensor::I32(
                Tensor::new(
                    raw.as_chunks::<4>()
                        .0
                        .iter()
                        .map(|b| i32::from_le_bytes(*b))
                        .collect(),
                    &shape,
                )
                .map_err(LoadError::Shape)?,
            ),
            SafetensorsDtype::I64 => SafetensorsTensor::I64(
                Tensor::new(
                    raw.as_chunks::<8>()
                        .0
                        .iter()
                        .map(|b| i64::from_le_bytes(*b))
                        .collect(),
                    &shape,
                )
                .map_err(LoadError::Shape)?,
            ),
            SafetensorsDtype::Bool => {
                let mut data = Vec::with_capacity(raw.len());
                for (index, &value) in raw.iter().enumerate() {
                    match value {
                        0 => data.push(false),
                        1 => data.push(true),
                        _ => {
                            return Err(LoadError::InvalidBoolValue {
                                key: name.to_string(),
                                index,
                                value,
                            });
                        }
                    }
                }
                SafetensorsTensor::Bool(Tensor::new(data, &shape).map_err(LoadError::Shape)?)
            }
        };
        out.insert(name.to_string(), tensor);
    }
    Ok(out)
}

/// `path` の safetensors ファイルを dtype 混在のまま読む
/// （`std::fs::read` + [`load_safetensors_typed_from_bytes`] の薄い合成）。
pub fn load_safetensors_typed(
    path: &Path,
) -> Result<HashMap<String, SafetensorsTensor>, LoadError> {
    let bytes = std::fs::read(path).map_err(LoadError::Io)?;
    load_safetensors_typed_from_bytes(&bytes)
}

/// [`require_keys`] の typed 版。不足キーを**全件**入力順で [`LoadError::MissingKeys`] に集める。
pub fn require_keys_typed(
    map: &HashMap<String, SafetensorsTensor>,
    keys: &[&str],
) -> Result<(), LoadError> {
    let missing: Vec<String> = keys
        .iter()
        .filter(|k| !map.contains_key(**k))
        .map(|k| (*k).to_string())
        .collect();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(LoadError::MissingKeys(missing))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn require_keys_reports_all_missing_not_just_the_first() {
        let map: HashMap<String, Tensor<f32>> = HashMap::new();
        let err = require_keys(&map, &["fc1.weight", "fc1.bias"]).unwrap_err();
        match err {
            LoadError::MissingKeys(keys) => {
                assert_eq!(keys, vec!["fc1.weight".to_string(), "fc1.bias".to_string()]);
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn safetensors_dtype_inventory_is_exact() {
        let all = [
            (SafetensorsDtype::F32, "F32", 4, Dtype::F32),
            (SafetensorsDtype::F16, "F16", 2, Dtype::F16),
            (SafetensorsDtype::BF16, "BF16", 2, Dtype::BF16),
            (SafetensorsDtype::F64, "F64", 8, Dtype::F64),
            (SafetensorsDtype::I32, "I32", 4, Dtype::I32),
            (SafetensorsDtype::I64, "I64", 8, Dtype::I64),
            (SafetensorsDtype::Bool, "BOOL", 1, Dtype::BOOL),
        ];
        for (d, name, w, wire) in all {
            assert_eq!(d.header_name(), name);
            assert_eq!(d.element_width(), w);
            assert_eq!(wire.bitsize(), w * 8);
            assert_eq!(SafetensorsDtype::from_wire(wire), Some(d));
            assert_eq!(d.to_wire(), wire);
            assert_eq!(SafetensorsDtype::from_wire(d.to_wire()), Some(d));
        }
        assert_eq!(SafetensorsDtype::from_wire(Dtype::U8), None);
        assert_eq!(SafetensorsDtype::from_wire(Dtype::I8), None);
    }

    #[test]
    fn require_keys_ok_when_all_present() {
        let mut map: HashMap<String, Tensor<f32>> = HashMap::new();
        map.insert("fc1.weight".to_string(), Tensor::zeros(&[1]).unwrap());
        require_keys(&map, &["fc1.weight"]).unwrap();
    }

    /// 巨大 broadcast view でも host_slice() 実体化前に ElementCountOverflow で拒否する（panic しない）。
    #[test]
    fn huge_broadcast_view_is_rejected_before_materialization() {
        let n = (isize::MAX as usize) / 4 + 10;
        let f = Tensor::<f32>::zeros(&[1])
            .unwrap()
            .broadcast_to(&[n])
            .unwrap();
        for dt in [SafetensorsDtype::F16, SafetensorsDtype::BF16] {
            assert!(matches!(
                SafetensorsTensor::from_f32(&f, dt),
                Err(ShapeError::ElementCountOverflow)
            ));
        }
        let h = Tensor::<f16>::zeros(&[1])
            .unwrap()
            .broadcast_to(&[n])
            .unwrap();
        assert!(matches!(
            SafetensorsTensor::F16(h).to_f32(),
            Err(ShapeError::ElementCountOverflow)
        ));
        let b = Tensor::<bf16>::zeros(&[1])
            .unwrap()
            .broadcast_to(&[n])
            .unwrap();
        assert!(matches!(
            SafetensorsTensor::BF16(b).to_f32(),
            Err(ShapeError::ElementCountOverflow)
        ));
    }
}
