//! numpy/Keras 慣習の互換 API 層（REQ-9・TASK-9.4・イシュー #411）。
//!
//! `autodiff` の自作コア（`Tape`/`Var`/`nn`）の上に薄いラッパーを被せ、
//! Python 慣習寄りの入口を提供する。テンソル生成は [`array`]
//! （numpy `np.array` 慣習）、レイヤー積み上げは [`Sequential`]
//! （Keras `Sequential` 慣習）。数値ロジック・shape 検査は一切持ち込まず
//! `tensor-core::Tensor::new`／`fandhe_ai_autodiff::nn::Module` へ委譲する（REQ-9
//! 「互換 API 層は自作コアの上の薄いラッパーに徹する」・`.claude/rules/
//! coding-rust.md`）。
//!
//! **配置の確定履歴**: TASK-9.2a（#95）は本モジュールを `fandhe_ai_autodiff::compat`
//! として確定していた（当時は 9 クレート構成で compat 専用クレートが
//! 存在せず、`autodiff` 配下以外に置く選択肢がなかったため）。10 クレート
//! 化（イシュー #52・`facade` の新設）を受け、TASK-9.3（#410）で
//! composition root（`Device` → 具体 `BackendOps` の結線）が `facade` へ
//! 一本化されたのに続き、TASK-9.4（本イシュー・#411）で compat 公開面も
//! `fandhe_ai_autodiff::compat` から本モジュール（`fandhe_ai::compat`）へ移設した。
//!
//! **サポート境界の明文化（TASK-9.4・REQ-9 の 2026-08-08 追記）**: `facade`
//! が唯一のサポートされる公開 API 面であり、`tensor-core`／`autodiff`／
//! `backend-*` は内部クレート（直接利用は非サポート）である。詳細・根拠は
//! `docs/compat-api-scope.md` の「サポート境界」節を参照（`docs/spec/
//! 04-requirements.md:209-210`・`05-tasks.md:322`）。
//!
//! **`predict_with_ops`（任意 `BackendOps` 注入経路）は本移設で公開面から
//! 撤去した**（破壊的変更。REQ-12「任意 `BackendOps` 実装を注入できる
//! 公開 API を設けない」・`crates/facade/tests/api_surface.rs` の機械検査と
//! 整合させるため）。[`Sequential::predict`] は本クレートの [`crate::tape`]
//! （既定 CPU・`CpuBackendOps`・融合有効）で `Tape` を構築して forward する
//! （`docs/public-api-design.md:431`「facade 経由なら既定バックエンドが
//! 透過的に効く」と整合）。ops を明示的に選びたい内部用途は
//! [`Sequential::forward`]（`&fandhe_ai_autodiff::Tape` を受け取るだけで `BackendOps`
//! は受け取らない）へ、呼び出し元が任意に構築した `Tape` を渡せば足りる。
//!
//! `lib.rs` クレート doc の「本クレート自体には互換レイヤ固有のロジック
//! を持ち込まない」は、「compat 層は本モジュールに隔離し、composition
//! root（`crate::tape`/`crate::tape_for`）へ互換慣習を漏らさない」という
//! 境界記述として引き継ぐ（依存方向は `compat` → `crate`（composition
//! root）／`autodiff`／`tensor_core` の一方向）。
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
