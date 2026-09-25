//! batch 添字を伴う einsum 縮約（例 `"bij,bjk->bik"`）の内部クレート
//! 限定の到達入口（イシュー #2149・親 #2131「PyTorch／TF 置き換えの
//! API 網羅」）。
//!
//! **新規 `Op`・新規 VJP はゼロ**: 実体は `crate::einsum::einsum_with`
//! を `crate::einsum::BatchContraction::Allow` で呼ぶだけの薄い委譲
//! （分解ロジック本体・数値契約・既知の制約は `crate::einsum` モジュール
//! doc を参照）。既存の受理範囲（[`crate::var::Var::einsum`] が対応する
//! 全ケース）に加え、両オペランドと出力に共通する batch 添字を伴う
//! 2 項縮約を rank≥3 `Var::matmul`（`gemm_batched`。イシュー #1715）
//! へ分解して受理する——`Var::einsum` の受理範囲の**上位集合**になる。
//!
//! **facade 非公開（意図的）**: [`crate::bool_ops`]・[`crate::
//! rearrange_ops`]・[`crate::matrix_ops`] と同じ理由・同じ判断枠組みに
//! よる。`Var` は facade（`fandhe_ai` クレート）から直接再エクスポート
//! されるため、`Var::einsum` の挙動を batch 添字受理へ拡張すると
//! それだけで facade 公開面が変わってしまう。イシュー #2149 本文は
//! facade 公開面の拡張（`Var::einsum` の batch 添字対応）を承認事項と
//! して明示し、親 #2131 はこのツリーに限り「設計判断記録 → 承認 →
//! 実装」の 2 段階を定めるため、承認が取れるまでは `Var::einsum` から
//! 独立した自由関数として到達可能にする（`docs/autodiff-einsum-batch-
//! decision.md` §5）。承認後は `crate::einsum::einsum`（`Var::einsum`
//! の実装本体）が `BatchContraction::Allow` を渡すよう 1 行変更し、
//! facade 側の保留ガード（`crates/facade/src/lib.rs::
//! VarEinsumBatchHoldDoctestGuard`）を撤去する（同 doc §6「承認後の
//! 切替手順」）。本モジュール自体は撤去するか維持するかを同 doc の
//! 判断に従う。
//!
//! **数値契約・既知の制約**: `crate::einsum` モジュール doc「数値契約」
//! 「batch 経路の既知の制約」節と完全に同一（rank≥3 `Var::matmul` と
//! 同一の FMA 契約・TF32 opt-in 挙動、create_graph 下では型付き
//! エラーで拒否、size-1 broadcast なし）。

use crate::error::AutodiffError;
use crate::var::Var;

/// batch 添字を伴う einsum 縮約を受理する自由関数（内部クレート限定。
/// facade 非公開）。`spec`／`operands` の意味・受理範囲は
/// [`crate::var::Var::einsum`] と同一だが、両オペランドと出力に共通
/// する batch 添字を伴う 2 項縮約（例 `"bij,bjk->bik"`）も追加で受理
/// する点のみが異なる（モジュール doc 参照）。
pub fn einsum_batched<'t>(spec: &str, operands: &[&Var<'t>]) -> Result<Var<'t>, AutodiffError> {
    crate::einsum::einsum_with(spec, operands, crate::einsum::BatchContraction::Allow)
}
