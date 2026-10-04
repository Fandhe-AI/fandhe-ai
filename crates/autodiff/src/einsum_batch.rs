//! `Var::einsum` と同一挙動の薄い委譲（イシュー #2149 で導入、
//! #2517 で `Var::einsum` が batch 添字付き縮約を受理するよう拡張
//! されたため、公開済み `fandhe-ai-autodiff 0.10.0` の
//! `einsum_batched` との互換のためだけに維持する）。
//!
//! **新規 `Op`・新規 VJP はゼロ**: 実体は `crate::einsum::einsum`
//! （`Var::einsum` の実装本体）を呼ぶだけで、受理範囲・数値契約・
//! 既知の制約は `crate::einsum` モジュール doc を参照。撤去すると
//! 公開済みクレートの `pub fn` が消える semver 破壊になるため残す
//! （`docs/autodiff-einsum-batch-decision.md` §11）。facade は本
//! モジュールを再エクスポートしない（正ガード
//! `facade_does_not_reexport_or_declare_einsum_batch`）。
//!
//! **数値契約・既知の制約**: `crate::einsum` モジュール doc「数値契約」
//! 「batch 経路の既知の制約」節と完全に同一（rank≥3 `Var::matmul` と
//! 同一の FMA 契約・TF32 opt-in 挙動、create_graph 下では型付き
//! エラーで拒否、size-1 broadcast なし）。

use crate::error::AutodiffError;
use crate::var::Var;

/// [`crate::var::Var::einsum`] と同一挙動の互換入口（内部クレート
/// 限定。facade 非公開）。`Var::einsum` が batch 添字付き縮約を受理
/// するようになった（#2517）ため意味論は完全に同一。
pub fn einsum_batched<'t>(spec: &str, operands: &[&Var<'t>]) -> Result<Var<'t>, AutodiffError> {
    crate::einsum::einsum(spec, operands)
}
