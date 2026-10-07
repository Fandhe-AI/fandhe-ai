//! `selu`・`celu`・`softsign`・`hardsigmoid`・`log_sigmoid` の 5 活性化演算
//! （イシュー #2649・親 #2648「不足している活性化関数」）。
//!
//! **新規 `Op` はゼロ**: いずれも `tensor-core` の `ScalarUnaryOp`
//! （`Selu`／`Celu { alpha }`／`Softsign`／`Hardsigmoid`／`LogSigmoid`）を
//! `Var::scalar_unary`（`Op::ScalarUnary`）へ渡す薄い委譲で、forward の
//! 単一情報源は `ScalarUnaryOp::apply`、VJP 係数は
//! `eval::scalar::unary_grad_factor`。CUDA／Metal は専用カーネル未実装で、
//! バックエンドが `Unsupported` を返したときだけホスト参照実装へ
//! フォールバックして到達可能になる（`.claude/rules/security.md` A08:
//! `Unsupported` 以外のエラーは握りつぶさず伝播する）。
//!
//! **公開形**: `compat::Sequential::add_*` は #2679 で facade へ公開済み
//! （`docs/autodiff-activation-scalar-ops-decision.md` §7・§12）。`Var` 委譲メソッドの
//! 公開は #2678 の担当で、本モジュールは facade から再エクスポートしない
//! （`facade::ActivationScalarOpsHoldDoctestGuard` と `api_surface.rs` の否定ガードが
//! 未承認経路を機械固定する。§9）。
//!
//! **数値契約**（詳細は決定記録 §3）: `NaN` は伝播、`±inf`・巨大有限入力は
//! IEEE のまま（panic・マスクなし）。CELU のみ入口で `alpha == 0`・非有限
//! `alpha` を `AutodiffError::InvalidArgument` で拒否する（tape 操作より前に
//! 検査するため失敗時に孤児ノードを残さない）。
//!
//! **対象外**: `create_graph`（高階微分）・f64 自動微分・f16／bf16・
//! activation checkpoint の対象化。

use crate::error::AutodiffError;
use crate::var::Var;
use fandhe_ai_tensor_core::ScalarUnaryOp;

/// CELU の `alpha` 検査（`celu` と `nn::activation::Celu::new` が共有する
/// 単一実装）。`alpha == 0`（`x / alpha` が破綻）と非有限値を拒否し、負の
/// `alpha` は PyTorch と同じく受理する。
pub(crate) fn validate_celu_alpha(alpha: f32, context: &str) -> Result<(), AutodiffError> {
    if alpha == 0.0 || !alpha.is_finite() {
        return Err(AutodiffError::InvalidArgument(format!(
            "{context}: alpha must be finite and non-zero, got {alpha}"
        )));
    }
    Ok(())
}

/// SELU（`F.selu` 相当）。`x > 0` なら `scale * x`、それ以外は
/// `alpha * scale * expm1(x)`。`input` と同 shape の `Var` を返す。
pub fn selu<'t>(x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    x.scalar_unary(ScalarUnaryOp::Selu)
}

/// CELU（`F.celu(x, alpha)` 相当）。`x > 0` なら `x`、それ以外は
/// `alpha * expm1(x / alpha)`。`alpha == 0`・非有限 `alpha` は
/// `AutodiffError::InvalidArgument`（tape へ何も積まない）。
pub fn celu<'t>(x: &Var<'t>, alpha: f32) -> Result<Var<'t>, AutodiffError> {
    validate_celu_alpha(alpha, "celu")?;
    x.scalar_unary(ScalarUnaryOp::Celu { alpha })
}

/// Softsign（`F.softsign` 相当）。`x / (1 + |x|)`。
pub fn softsign<'t>(x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    x.scalar_unary(ScalarUnaryOp::Softsign)
}

/// Hardsigmoid（`F.hardsigmoid` 相当）。`relu6(x + 3) / 6`。backward は
/// 開区間 `-3 < x < 3` のみ `upstream / 6`、それ以外は厳密に `0`
/// （上流が `inf`／`NaN` でも汚染されない要素選択。決定記録 §3）。
pub fn hardsigmoid<'t>(x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    x.scalar_unary(ScalarUnaryOp::Hardsigmoid)
}

/// LogSigmoid（`F.logsigmoid` 相当）。`min(x, 0) - ln(1 + exp(-|x|))`
/// （数値安定形）。PyTorch の関数名は `logsigmoid` だが、既存の
/// `log_softmax` の snake_case 規則に揃えて `log_sigmoid` とする。
pub fn log_sigmoid<'t>(x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    x.scalar_unary(ScalarUnaryOp::LogSigmoid)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tape::Tape;
    use crate::test_support::test_ops;
    use fandhe_ai_tensor_core::Tensor;

    const XS: [f32; 6] = [-4.0, -1.0, 0.0, 0.5, 2.0, 5.0];

    fn check(op: ScalarUnaryOp, f: for<'t> fn(&Var<'t>) -> Result<Var<'t>, AutodiffError>) {
        let tape = Tape::new_with_ops(test_ops());
        let x = tape.var(&Tensor::new(XS.to_vec(), &[6]).unwrap());
        let y = f(&x).unwrap().to_tensor();
        for (i, &xv) in XS.iter().enumerate() {
            assert_eq!(
                y.get(&[i]).unwrap().to_bits(),
                op.apply(xv).to_bits(),
                "{op:?}"
            );
        }
    }

    #[test]
    fn new_2649_entries_match_scalar_unary_apply() {
        check(ScalarUnaryOp::Selu, selu);
        check(ScalarUnaryOp::Softsign, softsign);
        check(ScalarUnaryOp::Hardsigmoid, hardsigmoid);
        check(ScalarUnaryOp::LogSigmoid, log_sigmoid);
        let tape = Tape::new_with_ops(test_ops());
        let x = tape.var(&Tensor::new(XS.to_vec(), &[6]).unwrap());
        let y = celu(&x, 1.5).unwrap().to_tensor();
        assert_eq!(
            y.get(&[0]).unwrap().to_bits(),
            ScalarUnaryOp::Celu { alpha: 1.5 }.apply(-4.0).to_bits()
        );
    }

    #[test]
    fn new_2649_celu_rejects_invalid_alpha_without_orphan_nodes() {
        let tape = Tape::new_with_ops(test_ops());
        let x = tape.var(&Tensor::new(XS.to_vec(), &[6]).unwrap());
        let before = tape.len();
        for alpha in [0.0, -0.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(matches!(
                celu(&x, alpha),
                Err(AutodiffError::InvalidArgument(_))
            ));
        }
        assert_eq!(tape.len(), before);
        assert!(celu(&x, -1.5).is_ok());
    }
}
