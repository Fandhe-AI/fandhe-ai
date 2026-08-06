//! `tensor-core` の shape 検査エラー型。
//!
//! `ShapeError` はテンソル生成・view 操作・reshape の shape 不整合を
//! 表す、`tensor-core` の全公開 API が共通して返す型付きエラーである
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
    /// 本 variant は `tensor-core` が型定義のみを提供し、本イシュー
    /// （TASK-1.4a）では構築しない。構築は rank 前提を持つ演算・
    /// autodiff／backend 入口側の検査（TASK-1.4c 以降、#13）が行う。
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
        }
    }
}

impl std::error::Error for ShapeError {}
