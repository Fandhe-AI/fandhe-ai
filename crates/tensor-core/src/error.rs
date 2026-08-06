//! `tensor-core` の shape 検査エラー型。
//!
//! `ShapeError` はテンソル生成・view 操作・reshape・演算時 shape 検査
//! （`ops_shape`、TASK-1.4c・#13）の shape 不整合を表す、`tensor-core`
//! の全公開 API が共通して返す型付きエラーである
//! （spec 根拠: `docs/public-api-design.md` §2.1.1）。`autodiff` の
//! `AutodiffError::Shape`・backend 入口の `BackendError::ShapeMismatch`
//! からラップされる想定（本イシューではラップ側は実装しない）。

use std::fmt;

/// テンソル生成・view 操作・reshape の shape 不整合を表す。
///
/// `#[non_exhaustive]` を付す理由: 公開 API 非破壊はガードレール条件
/// （`.claude/rules/security.md`）であり、後続タスクで検査項目が増えても
/// 呼び出し側の網羅的 match を破壊しないため。
#[non_exhaustive]
#[derive(Debug)]
pub enum ShapeError {
    /// 要求される次元数（rank）と実際の次元数が一致しない。
    ///
    /// TASK-1.4a（#11）では型定義のみを提供し構築しなかった。本 variant は
    /// `ops_shape::matmul_out_shape`（TASK-1.4c・#13）が rank ≠ 2 の
    /// 入力を検出した際に初めて構築する（`docs/public-api-design.md`
    /// §3.2 の matmul は 2 次元前提）。
    RankMismatch { expected: usize, actual: usize },

    /// shape の要素数積とデータ長が一致しない
    /// （`Tensor::new`/`from_slice` が `data.len()` と shape の要素数積を
    /// 突き合わせる際に返す）。
    ElementCountMismatch { expected: usize, actual: usize },

    /// 軸番号（`dim`）がテンソルの rank 範囲外
    /// （`transpose`/`narrow` の軸引数）。
    AxisOutOfRange { axis: usize, rank: usize },

    /// `narrow` の `[start, start+len)` が対象軸のサイズを超える。
    NarrowOutOfBounds {
        dim: usize,
        start: usize,
        len: usize,
        dim_size: usize,
    },

    /// shape の要素数積が `usize` の範囲でオーバーフローする
    /// （`zeros`/`ones`/`full`/`Tensor::new`/`from_slice` がアロケーション
    /// 前に検査する）。
    ElementCountOverflow,

    /// 非 contiguous なテンソルに対して `reshape` が呼ばれた。
    ///
    /// `docs/public-api-design.md` §2.2.1 は reshape の非 contiguous
    /// ケースの扱いを「案 A（エラー）」「案 B（暗黙コピー）」の 2 案で
    /// 未決事項としている。本イシュー（TASK-1.4a）では安全側（設計書が
    /// 推奨する案 A）を採用し、この variant で通知する。最終決定は
    /// ユーザー承認が必要な採否論点として残っており、案 B へ変更する
    /// 場合は `Tensor::reshape` 内の分岐 1 箇所の差し替えで足りるよう
    /// 局所化している。設計書の 5 variant には明示されていないが、
    /// `#[non_exhaustive]` は後続タスクでの variant 追加を織り込み済みで
    /// あるため追加する。
    NonContiguousReshape,

    /// 2 つの shape が NumPy 互換のブロードキャスト規則で両立しない
    /// （末尾軸から比較し「両者同一」または「片方が 1」を満たせない）。
    ///
    /// `broadcast_shape`（`broadcast.rs`）・`Tensor::broadcast_to`／
    /// `Tensor::broadcast_with`（`tensor.rs`）が構築する
    /// （#12・TASK-1.4b、`docs/public-api-design.md` §2.1 の
    /// stride 0 ブロードキャスト方針）。`broadcast_to` 呼び出し時は
    /// `lhs` = 自身の shape・`rhs` = target shape として構築する。
    BroadcastIncompatible { lhs: Vec<usize>, rhs: Vec<usize> },

    /// matmul（`ops_shape::matmul_out_shape`）の内部次元（lhs の最終軸と
    /// rhs の先頭軸）が一致しない。rank 検査（`RankMismatch`）を通過した
    /// 2 次元 shape 同士でのみ構築される（TASK-1.4c・#13。
    /// `docs/public-api-design.md` §3.2）。
    MatmulDimMismatch { lhs: Vec<usize>, rhs: Vec<usize> },

    /// 厳密一致を要求する演算（`ops_shape::require_same_shape`。
    /// 例: `mse_loss` の予測値と target）で shape が一致しない
    /// （TASK-1.4c・#13。`docs/public-api-design.md` §3.2）。
    ShapeMismatch { lhs: Vec<usize>, rhs: Vec<usize> },
}

impl fmt::Display for ShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShapeError::RankMismatch { expected, actual } => {
                write!(f, "rank mismatch: expected {expected}, actual {actual}")
            }
            ShapeError::ElementCountMismatch { expected, actual } => write!(
                f,
                "element count mismatch: expected {expected}, actual {actual}"
            ),
            ShapeError::AxisOutOfRange { axis, rank } => {
                write!(f, "axis {axis} out of range for rank {rank}")
            }
            ShapeError::NarrowOutOfBounds {
                dim,
                start,
                len,
                dim_size,
            } => write!(
                f,
                "narrow out of bounds: dim {dim} range [{start}, {}) exceeds size {dim_size}",
                // `start.checked_add(len)` が `None`（オーバーフロー）の
                // 場合でも `Tensor::narrow`（tensor.rs）はこの variant を
                // そのまま構築しうるため、表示側でも非 checked 加算で
                // panic しないよう `saturating_add` を用いる
                // （debug ビルドは overflow-checks=true が既定）。
                start.saturating_add(*len)
            ),
            ShapeError::ElementCountOverflow => {
                write!(f, "element count overflow while computing tensor size")
            }
            ShapeError::NonContiguousReshape => write!(
                f,
                "reshape requires a contiguous tensor; call `.contiguous()` first"
            ),
            ShapeError::BroadcastIncompatible { lhs, rhs } => {
                write!(f, "cannot broadcast shapes {lhs:?} and {rhs:?}")
            }
            ShapeError::MatmulDimMismatch { lhs, rhs } => {
                write!(f, "matmul dimension mismatch: lhs {lhs:?} rhs {rhs:?}")
            }
            ShapeError::ShapeMismatch { lhs, rhs } => {
                write!(f, "shape mismatch: lhs {lhs:?} rhs {rhs:?}")
            }
        }
    }
}

impl std::error::Error for ShapeError {}
