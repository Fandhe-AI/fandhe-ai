//! `ops` モジュール共通のエラー型（TASK-7.2c）。
//!
//! `tensor-core::ShapeError`（テンソル生成・view 操作の shape 不整合）に加え、
//! ONNX オペ固有の属性検査失敗（軸範囲外・インデックス範囲外・ステップ 0 等）を表す
//! variant を追加する。外部フォーマット由来の属性値（`Gather` の `indices`・`Slice` の
//! `starts`/`ends`/`axes`/`steps` 等）はここで長さ・範囲を検証してから使う
//! （OWASP A03 対策。`.claude/rules/security.md`）。

use std::fmt;

use tensor_core::ShapeError;

/// `ops` モジュールの全公開関数が返す型付きエラー。
///
/// `#[non_exhaustive]`: `tensor-core::ShapeError` と同じ理由（公開 API 非破壊。
/// `.claude/rules/security.md`）で、後続オペ追加時の variant 追加に備える。
#[non_exhaustive]
#[derive(Debug)]
pub enum OpError {
    /// `tensor-core` 側の shape 不整合をそのまま透過する
    /// （`Tensor::new`/`transpose`/`reshape`/`broadcast_to` 等が返す）。
    Shape(ShapeError),

    /// 入力テンソルの rank が要求と一致しない（例: `Gemm` の A/B は 2 次元固定）。
    RankMismatch {
        op: &'static str,
        expected: usize,
        actual: usize,
    },

    /// `Gemm` の内部次元（`A'` の最終軸と `B'` の先頭軸。`'` は `trans_*` 適用後）が一致しない。
    GemmDimMismatch { a: Vec<usize>, b: Vec<usize> },

    /// `tensor-core::Tensor::as_slice` が `None` を返した（呼び出し前に `contiguous()` を
    /// 経由しているため到達しないはずの内部不変条件違反。到達すれば実装バグ）。
    NonContiguousInternal(&'static str),

    /// ONNX の負軸表記（`axis + rank`）を適用しても `[0, rank)` に収まらない軸指定。
    AxisOutOfRange {
        op: &'static str,
        axis: i64,
        rank: usize,
    },

    /// 同一の正規化後軸が複数回指定された（`Unsqueeze`/`Slice` の `axes`）。
    DuplicateAxis { op: &'static str, axis: usize },

    /// `Gather` のインデックスが負軸表記（`index + dim_size`）を適用しても対象軸の
    /// サイズ範囲に収まらない。
    IndexOutOfRange {
        op: &'static str,
        index: i64,
        dim_size: usize,
    },

    /// `Slice` の `steps` に 0 が指定された（ONNX 仕様上不正。無限ループ回避のため
    /// 事前検査で拒否する）。
    InvalidStep { op: &'static str, axis: usize },

    /// 属性配列同士の長さが一致しない（例: `Slice` の `starts`/`ends`/`axes`/`steps`、
    /// `Gather` の `indices` と `indices_shape` の要素数積）。
    LengthMismatch {
        op: &'static str,
        name: &'static str,
        expected: usize,
        actual: usize,
    },

    /// `Concat` に入力が 1 つも渡されなかった。
    EmptyInputs { op: &'static str },

    /// `Concat` の非連結軸の shape が入力間で一致しない。
    ConcatShapeMismatch {
        axis: usize,
        lhs: Vec<usize>,
        rhs: Vec<usize>,
    },

    /// `Cast` の `to`（ONNX `TensorProto.DataType`）が本クレートの対応範囲
    /// （`FLOAT(1)`／`INT64(7)`。TASK-7.3b・#83）外だった。
    UnsupportedDataType { op: &'static str, to: i64 },

    /// `Reshape`／`Squeeze` の shape 指定（`-1` の複数指定・非 1 次元への squeeze 等）が
    /// ONNX 仕様上不正（TASK-7.3b・#83）。`op` は発生元オペ名（`"Reshape"`／`"Squeeze"`）で、
    /// `Display` 実装が固定で `Reshape:` を名乗り `Squeeze` 由来のエラーを誤診断させるのを防ぐ
    /// （PR #275 レビュー指摘）。
    InvalidReshapeSpec {
        op: &'static str,
        reason: &'static str,
    },

    /// `Mod` の `fmod=0`（Python 風・整数専用モード）が `f32` 入力に対して
    /// 要求された（TASK-7.3a・`arith.rs`）。ONNX 仕様上 `fmod=0` は整数入力
    /// のみ有効であり、`f32` に対して `rem_euclid` 等で代替すると異なる数値
    /// 意味論を静かに返すことになるため、明示的に拒否する
    /// （`.claude/rules/security.md` A03 相当の「外部入力の検証」）。
    UnsupportedFmodMode { op: &'static str },

    /// `LayerNormalization` の `epsilon` 属性が非有限値（`NaN`／`inf`）だった
    /// （TASK-7.3d・`layer_norm.rs`）。`epsilon` はモデル属性（外部入力）であり、
    /// 非有限値は分散計算全体を静かに `NaN`／`inf` へ汚染するため事前検査で拒否する
    /// （`.claude/rules/security.md` A03 相当）。
    InvalidEpsilon { op: &'static str, epsilon: f32 },

    /// `LayerNormalization` の正規化集合（`x.shape()[axis..]` の要素数積）が 0 だった
    /// （TASK-7.3d・`layer_norm.rs`。例: `shape=[2,0], axis=1`）。`axis` 自体は
    /// `[0, rank)` の範囲内であり [`OpError::AxisOutOfRange`] とは原因が異なるため、
    /// 分散計算の除数 0 割り（`NaN` を静かに生成する）を専用 variant で区別して拒否する。
    EmptyNormalizedSet { op: &'static str, axis: usize },
}

impl fmt::Display for OpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OpError::Shape(e) => write!(f, "{e}"),
            OpError::RankMismatch {
                op,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "{op}: rank mismatch (expected {expected}, actual {actual})"
                )
            }
            OpError::GemmDimMismatch { a, b } => {
                write!(f, "Gemm: inner dimension mismatch (a {a:?}, b {b:?})")
            }
            OpError::NonContiguousInternal(op) => {
                write!(
                    f,
                    "{op}: internal invariant violated (expected contiguous tensor)"
                )
            }
            OpError::AxisOutOfRange { op, axis, rank } => {
                write!(f, "{op}: axis {axis} out of range for rank {rank}")
            }
            OpError::DuplicateAxis { op, axis } => {
                write!(f, "{op}: axis {axis} specified more than once")
            }
            OpError::IndexOutOfRange {
                op,
                index,
                dim_size,
            } => {
                write!(
                    f,
                    "{op}: index {index} out of range for dim size {dim_size}"
                )
            }
            OpError::InvalidStep { op, axis } => {
                write!(f, "{op}: step for axis {axis} must not be 0")
            }
            OpError::LengthMismatch {
                op,
                name,
                expected,
                actual,
            } => write!(
                f,
                "{op}: `{name}` length mismatch (expected {expected}, actual {actual})"
            ),
            OpError::EmptyInputs { op } => write!(f, "{op}: at least one input is required"),
            OpError::ConcatShapeMismatch { axis, lhs, rhs } => write!(
                f,
                "Concat: shapes differ outside concat axis {axis} (lhs {lhs:?}, rhs {rhs:?})"
            ),
            OpError::UnsupportedDataType { op, to } => {
                write!(f, "{op}: unsupported target data type {to}")
            }
            OpError::InvalidReshapeSpec { op, reason } => write!(f, "{op}: {reason}"),
            OpError::UnsupportedFmodMode { op } => write!(
                f,
                "{op}: fmod=0 (Python-style, integer-only) is not supported for f32 input; use fmod=1"
            ),
            OpError::InvalidEpsilon { op, epsilon } => {
                write!(f, "{op}: epsilon {epsilon} must be finite")
            }
            OpError::EmptyNormalizedSet { op, axis } => {
                write!(
                    f,
                    "{op}: normalized set starting at axis {axis} has 0 elements"
                )
            }
        }
    }
}

impl std::error::Error for OpError {}

impl From<ShapeError> for OpError {
    fn from(e: ShapeError) -> Self {
        OpError::Shape(e)
    }
}
