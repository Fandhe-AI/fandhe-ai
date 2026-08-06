//! `autodiff` クレートの公開エラー型。
//!
//! 順伝播（`Var` の演算メソッド。TASK-1.5a・#16）と逆伝播
//! （`Tape::backward`/`Gradients::get`。`backward.rs`・TASK-1.5c・
//! 本イシュー・#18）双方の失敗経路をここに集約する（spec 根拠:
//! `docs/public-api-design.md` §3.1）。`tensor-core::ShapeError`
//! を包む形でラップし、shape 検査ロジック・エラー variant を二重定義しない。

use std::fmt;

use tensor_core::ShapeError;

/// `autodiff` の公開 API が返すエラー型。
///
/// `#[non_exhaustive]` を付す理由: 公開 API 非破壊はガードレール条件
/// （`.claude/rules/security.md`）であり、TASK-1.5 で演算セットが拡張
/// されるたびに新しい失敗要因（Conv 系の padding 不整合等）の variant
/// 追加を非破壊にするため（`docs/public-api-design.md` §3.1 準拠）。
#[non_exhaustive]
#[derive(Debug)]
pub enum AutodiffError {
    /// 順伝播時の shape 不整合（`matmul`/`add`/`mul` の不正なブロード
    /// キャスト、`sum`/`max`/`mse_loss` の shape 不一致等）。
    /// `tensor-core::ShapeError` をラップする。
    Shape(ShapeError),
    /// `Tape::backward`（`backward.rs`・TASK-1.5c・#18）時のグラフ不整合
    /// 用に予約された variant。本イシューが実装するクロステープ検査は
    /// `TapeMismatch`（下記）で表現でき、`backward()`/`Gradients::get()`
    /// はいずれも構造的に他の失敗要因を持たないため、現時点で構築箇所
    /// はまだない。`Var`/`Tape` 側の演算メソッドと同一エラー型を用いる
    /// 設計（`docs/public-api-design.md` §3.1）に合わせ、将来のグラフ
    /// 不整合検出（例: 循環検知）に備え公開 API 安定化のため先行して
    /// variant を定義しておく。
    Backward(String),
    /// 二項演算（`matmul`/`add`/`mul`/`mse_loss`）に、異なる `Tape` に
    /// 属する `Var` が渡された。ライフタイム `'t` の一致は同一 `Tape`
    /// を指す証明にはならない（同一スコープに複数の `Tape` が存在する
    /// 場合、それぞれの `Var<'t>` は同一の `'t` を持ちうる）ため、
    /// `Var::matmul`/`add`/`mul`/`mse_loss` は shape 検査より前に
    /// `self.tape.id` と相手側の `TapeId` を実行時照合し、不一致なら
    /// 本 variant を返す（`docs/public-api-design.md` §3.1
    /// 「クロステープ安全性」）。
    TapeMismatch,
}

impl From<ShapeError> for AutodiffError {
    fn from(err: ShapeError) -> Self {
        AutodiffError::Shape(err)
    }
}

impl fmt::Display for AutodiffError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AutodiffError::Shape(err) => write!(f, "shape error: {err}"),
            AutodiffError::Backward(msg) => write!(f, "backward error: {msg}"),
            AutodiffError::TapeMismatch => {
                write!(f, "operands belong to different Tape instances")
            }
        }
    }
}

impl std::error::Error for AutodiffError {}
