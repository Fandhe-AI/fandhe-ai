//! `atan`・`asin`・`acos`・`sinh`・`cosh`・`asinh`・`acosh`・`atanh`・
//! `atan2` の 9 種要素演算（イシュー #2634・親 #2625「Phase 4」）。
//!
//! **新規 `Op` はゼロ**: 既存の `Op::ScalarUnary`／`Op::ScalarBinary`
//! （`tensor_core::ScalarUnaryOp` の 8 variant・`ScalarBinaryOp::Atan2`）
//! への薄い委譲のみで構成する（`scalar_unary_ops` と同型）。forward の
//! 実体化・バックエンド dispatch（既定 `Unsupported` → ホスト参照実装
//! フォールバック）・tape 記録は `Var::scalar_unary`／`scalar_binary`
//! （`pub(crate)`）が担う。
//!
//! **公開形は未承認（保留）**: facade（`fandhe_ai`）へは公開しない。
//! 推奨案は `Var` の委譲メソッド（本モジュールへ 1 行委譲）で、承認依頼は
//! #2677・公開は承認後の #2678。保留は facade の `TrigOpsHoldDoctestGuard`
//! と `tests/api_surface.rs` の否定ガードで機械固定している
//! （`docs/autodiff-trig-ops-decision.md` §7・§9）。
//!
//! **数値規約**: forward の単一情報源は `ScalarUnaryOp::apply`／
//! `ScalarBinaryOp::apply`。VJP 係数は `eval::scalar` で `f64` 昇格・
//! 1 回 downcast（`docs/autodiff-trig-ops-decision.md` §3）。
//!
//! **CUDA／Metal**: 専用カーネルはスコープ外で、`unary_kernel_source`／
//! `binary_kernel_source` が明示的に `None` を返し、既定の `Unsupported`
//! からホスト参照実装へフォールバックする。
//!
//! **対象外**: `create_graph`（高階微分）・f64 自動微分経路・f16／bf16。
//!
//! **公開状況（イシュー #2678）**: 承認形どおり公開済み: `Var::{atan,asin,acos,sinh,cosh,asinh,acosh,atanh,atan2}`。本モジュール自体は facade から再エクスポートしない。
//! 上の「未承認」「保留」「承認依頼は #2677」の記述は #2677 時点のもので、承認形の公開は #2678 で行った
//! （ルート #2499 の承認コメント issuecomment-6033824965・`docs/compat-api-scope.md` §5.1）。

use crate::error::AutodiffError;
use crate::var::Var;
use fandhe_ai_tensor_core::{ScalarBinaryOp, ScalarUnaryOp};

/// 要素ごとの逆正接 `atan(x)`（PyTorch `torch.atan`）。IEEE のまま伝播し panic しない。
pub fn atan<'t>(x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    x.scalar_unary(ScalarUnaryOp::Atan)
}

/// 要素ごとの逆正弦 `asin(x)`（PyTorch `torch.asin`）。定義域 `[-1, 1]` の外は `NaN`。IEEE のまま伝播し panic しない。
pub fn asin<'t>(x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    x.scalar_unary(ScalarUnaryOp::Asin)
}

/// 要素ごとの逆余弦 `acos(x)`（PyTorch `torch.acos`）。定義域 `[-1, 1]` の外は `NaN`。IEEE のまま伝播し panic しない。
pub fn acos<'t>(x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    x.scalar_unary(ScalarUnaryOp::Acos)
}

/// 要素ごとの双曲線正弦（PyTorch `torch.sinh`）。IEEE のまま伝播し panic しない。
pub fn sinh<'t>(x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    x.scalar_unary(ScalarUnaryOp::Sinh)
}

/// 要素ごとの双曲線余弦（PyTorch `torch.cosh`）。IEEE のまま伝播し panic しない。
pub fn cosh<'t>(x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    x.scalar_unary(ScalarUnaryOp::Cosh)
}

/// 要素ごとの逆双曲線正弦（PyTorch `torch.asinh`）。IEEE のまま伝播し panic しない。
pub fn asinh<'t>(x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    x.scalar_unary(ScalarUnaryOp::Asinh)
}

/// 要素ごとの逆双曲線余弦（PyTorch `torch.acosh`）。`x < 1` は `NaN`。IEEE のまま伝播し panic しない。
pub fn acosh<'t>(x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    x.scalar_unary(ScalarUnaryOp::Acosh)
}

/// 要素ごとの逆双曲線正接（PyTorch `torch.atanh`）。`±1` は `±inf`、外側は `NaN`。IEEE のまま伝播し panic しない。
pub fn atanh<'t>(x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    x.scalar_unary(ScalarUnaryOp::Atanh)
}

/// 2 引数逆正接 `atan2(y, x)`（PyTorch `torch.atan2(input, other)` と同じ
/// 引数順で `y = 自身側`・`x = other 側`）。NumPy 互換 broadcast に従い、
/// 別 tape の `Var` は型付きエラーで拒否する。
pub fn atan2<'t>(y: &Var<'t>, x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    y.scalar_binary(x, ScalarBinaryOp::Atan2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tape::Tape;
    use fandhe_ai_tensor_core::Tensor;

    #[test]
    fn each_entry_forwards_to_expected_scalar_op() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&Tensor::new(vec![0.1, 0.5, 0.9], &[3]).unwrap());
        assert_eq!(
            atan(&x).unwrap().value().get(&[1]).unwrap().to_bits(),
            ScalarUnaryOp::Atan.apply(0.5).to_bits()
        );
        assert_eq!(
            asin(&x).unwrap().value().get(&[1]).unwrap().to_bits(),
            ScalarUnaryOp::Asin.apply(0.5).to_bits()
        );
        assert_eq!(
            acos(&x).unwrap().value().get(&[1]).unwrap().to_bits(),
            ScalarUnaryOp::Acos.apply(0.5).to_bits()
        );
        assert_eq!(
            sinh(&x).unwrap().value().get(&[1]).unwrap().to_bits(),
            ScalarUnaryOp::Sinh.apply(0.5).to_bits()
        );
        assert_eq!(
            cosh(&x).unwrap().value().get(&[1]).unwrap().to_bits(),
            ScalarUnaryOp::Cosh.apply(0.5).to_bits()
        );
        assert_eq!(
            asinh(&x).unwrap().value().get(&[1]).unwrap().to_bits(),
            ScalarUnaryOp::Asinh.apply(0.5).to_bits()
        );
        assert_eq!(
            acosh(&x).unwrap().value().get(&[1]).unwrap().to_bits(),
            ScalarUnaryOp::Acosh.apply(0.5).to_bits()
        );
        assert_eq!(
            atanh(&x).unwrap().value().get(&[1]).unwrap().to_bits(),
            ScalarUnaryOp::Atanh.apply(0.5).to_bits()
        );
        let y = tape.var(&Tensor::new(vec![1.0, -2.0, 3.0], &[3]).unwrap());
        assert_eq!(
            atan2(&y, &x).unwrap().value().get(&[1]).unwrap(),
            ScalarBinaryOp::Atan2.apply(-2.0, 0.5)
        );
    }
}
