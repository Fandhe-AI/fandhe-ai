//! numpy/Keras 慣習の互換 API 層（REQ-9・TASK-9.2a・#95）。
//!
//! `autodiff` の自作コア（`Tape`/`Var`/`nn`）の上に薄いラッパーを被せ、
//! Python 慣習寄りの入口を提供する。テンソル生成は [`array`]
//! （numpy `np.array` 慣習）、レイヤー積み上げは [`Sequential`]
//! （Keras `Sequential` 慣習）。数値ロジック・shape 検査は一切持ち込まず
//! `tensor-core::Tensor::new`／`nn::Module` へ委譲する（REQ-9「互換 API
//! 層は自作コアの上の薄いラッパーに徹する」・`.claude/rules/
//! coding-rust.md`）。
//!
//! **配置の確定（TASK-9.2a）**: `docs/compat-api-scope.md` §4 が
//! 「9 クレート構成に compat 専用クレートは存在しないため、既存クレート
//! （`autodiff` 等）内のモジュールとなる見込み」としていた点を、本
//! モジュール（`autodiff::compat`）として確定する。`Sequential` は
//! `nn::Linear`/`nn::Module` に依存し、`tensor-core` は `autodiff` に
//! 依存できない（下位クレートが上位クレートへ依存すると循環する）ため、
//! `autodiff` 配下以外に置く選択肢はない。新規クレードは作らず、
//! CLAUDE.md の「想定クレート 9 個」を維持する。
//!
//! `lib.rs` クレート doc の「本クレート自体には互換レイヤ固有のロジック
//! を持ち込まない」は、「compat 層は本モジュールに隔離し、コア
//! （`tape`/`var`/`nn`）へ互換慣習を漏らさない」という境界記述として
//! 引き継ぐ（`nn` 側は `compat` を一切 `use` しない。依存方向は
//! `compat` → `nn`/`var`/`tape` の一方向）。
//!
//! **対象範囲**（`docs/compat-api-scope.md` §1〜2）: レイヤーは
//! Linear・ReLU・Sigmoid・Tanh の 3 種限定。`Sequential` 経由の学習
//! （勾配取得・パラメータ更新）は #294 で対応済み（`sequential.rs`
//! 冒頭 doc・[`SequentialVars`] 参照。`fit()`/`compile()` 等の高水準
//! 学習ループ API は引き続き対象外）。

mod array;
mod sequential;

pub use array::{ArrayData, array};
pub use sequential::{Sequential, SequentialVars};
