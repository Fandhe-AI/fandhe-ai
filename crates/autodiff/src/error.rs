//! `autodiff` クレートの公開エラー型。
//!
//! 順伝播（`Var` の演算メソッド。TASK-1.5a・#16）と逆伝播
//! （`Tape::backward`/`Gradients::get`。`backward.rs`・TASK-1.5c・
//! 本イシュー・#18）双方の失敗経路をここに集約する（spec 根拠:
//! `docs/public-api-design.md` §3.1）。`tensor-core::ShapeError`
//! を包む形でラップし、shape 検査ロジック・エラー variant を二重定義しない。

use std::fmt;

use fandhe_ai_tensor_core::ShapeError;

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
    /// 用に予約された variant。当初はクロステープ検査
    /// （`TapeMismatch`。下記）で表現できる範囲のみを想定し構築箇所を
    /// 持たなかったが、TASK-12.1d（#164）の遅延評価統合で
    /// `materialize_fallible`（`tape.rs`）の `OnceCell` 不変条件違反
    /// （`set`/`get` の理論上到達しないはずの `None` 分岐）検出用途に
    /// 転用する（`docs/fusion-graph-design.md` §3.5.2「`OnceCell::set`
    /// の `Err` は通常分岐として扱う」節。`materialize_fallible` は
    /// `&'a Tensor<f32>` を返す関数であり `None` 分岐で捏造した値への
    /// 参照を作れないため、安全側のフォール先を型付きエラーとする）。
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
    /// 呼び出し元が渡した引数がテンソル未生成の段階で構築不可能な
    /// 組み合わせだった（例: `nn::Linear::new` の `in_features == 0`。
    /// `1/√in_features` が非有限になるため shape 検査より前に弾く）。
    ///
    /// `tensor-core::ShapeError`（`#[non_exhaustive]`）へ variant を
    /// 追加する案もあったが、`ShapeError` は「テンソル生成・view 操作・
    /// reshape の shape 不整合」（`tensor-core/src/error.rs` 冒頭）専用の
    /// 型であり、`nn` 層のコンストラクタ引数検証はその責務外
    /// （`tensor-core` は `nn` を知らない下位クレート）。そのため
    /// `autodiff` 側で `ShapeError` をラップしない専用 variant として
    /// 追加する（review 指摘 #91: 既存 `AxisOutOfRange` への転用は
    /// 意味不一致だったため撤回）。
    InvalidArgument(String),
    /// 融合実行・実体化（TASK-12.1d・#164。`materialize_fallible`。
    /// `tape.rs`）で発生した型付きバックエンドエラー。
    /// `fandhe_ai_tensor_core::BackendError` をラップする（`docs/
    /// fusion-graph-design.md` §3.5.2「層 1 は `Unsupported` 以外の
    /// `run_fused` の失敗をフォールバックせずそのまま伝播する」）。
    Backend(fandhe_ai_tensor_core::BackendError),
}

impl From<ShapeError> for AutodiffError {
    fn from(err: ShapeError) -> Self {
        AutodiffError::Shape(err)
    }
}

impl From<fandhe_ai_tensor_core::BackendError> for AutodiffError {
    fn from(err: fandhe_ai_tensor_core::BackendError) -> Self {
        AutodiffError::Backend(err)
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
            AutodiffError::InvalidArgument(message) => {
                write!(f, "invalid argument: {message}")
            }
            AutodiffError::Backend(err) => write!(f, "backend error: {err}"),
        }
    }
}

impl std::error::Error for AutodiffError {}
