//! `hardshrink`・`softshrink`（PyTorch `F.hardshrink`／`F.softshrink`、`nn.Hardshrink`／
//! `nn.Softshrink` 相当）の自由関数（イシュー #3817・親 #3813。設計記録
//! `docs/nn-shrink-bilinear-channel-shuffle-decision.md` §1）。
//!
//! **新規 `Op`・`BackendOps`・カーネル・`unsafe`・依存はゼロ**（設計 §1.10 の不変条件）。既存の
//! `Var::masked_fill`・`Var::clamp`・`Var::sub` の合成のみで、CPU・CUDA・Metal の全バックエンドに
//! 既存経路で到達する。呼び出し元は `Var::{hardshrink, softshrink}`（1 行委譲）と
//! `nn::shrink::{Hardshrink, Softshrink}` 層。本モジュール自体は facade から再エクスポートしない。
//!
//! **式**:
//! - `hardshrink(x, λ)`: `-λ <= x <= λ` なら 0、それ以外は `x`。範囲内の勾配は 0、範囲外は 1。
//! - `softshrink(x, λ)`: `x - clamp(x, -λ, λ)`。範囲内（境界含む）の勾配は 0、範囲外は 1。
//!
//! **NaN**: `hardshrink` は比較が偽になるため素通し（出力 NaN・勾配 1）。`softshrink` は forward が
//! NaN になり、勾配は `1 - 0 = 1`（`Var::clamp` の NaN 勾配は 0）。
//!
//! **数値契約**: 選択と IEEE 減算のみのため、forward／backward とも bit 完全一致を見込む。
//!
//! **境界検査（REQ-8・`.claude/rules/security.md` A03）**: `lambd` は有限かつ `>= 0`（`-0.0` は受理。
//! #3768 の所有者回答で承認）。保存ファイル経由の値も `Sequential::add_*` を通るため同じ検査を受ける。
//! 検査は tape 操作・mask 構築より前に行い、エラー時に孤児ノードを残さない。

use crate::activation_ops::build_value_mask;
use crate::error::AutodiffError;
use crate::var::Var;

/// `lambd` の検査（有限かつ `>= 0`）。層の構築時と自由関数の先頭から呼ぶ。
pub(crate) fn validate_lambd(op: &str, lambd: f32) -> Result<(), AutodiffError> {
    if !lambd.is_finite() || lambd < 0.0 {
        return Err(AutodiffError::InvalidArgument(format!(
            "shrink_ops::{op}: lambd must be finite and >= 0, got {lambd}"
        )));
    }
    Ok(())
}

/// Hardshrink（PyTorch `F.hardshrink(input, lambd)` 相当）。
///
/// `-lambd <= x <= lambd` の位置を 0 に置換する。`NaN` は述語が偽で素通し。
pub fn hardshrink<'t>(x: &Var<'t>, lambd: f32) -> Result<Var<'t>, AutodiffError> {
    validate_lambd("hardshrink", lambd)?;
    let mask = build_value_mask(x, |v| -lambd <= v && v <= lambd)?;
    x.masked_fill(&mask, 0.0)
}

/// Softshrink（PyTorch `F.softshrink(input, lambd)` 相当）: `x - clamp(x, -lambd, lambd)`。
pub fn softshrink<'t>(x: &Var<'t>, lambd: f32) -> Result<Var<'t>, AutodiffError> {
    validate_lambd("softshrink", lambd)?;
    x.sub(&x.clamp(-lambd, lambd)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tape;
    use fandhe_ai_tensor_core::Tensor;

    fn run<F>(f: F, xs: &[f32]) -> (Vec<f32>, Vec<f32>)
    where
        F: for<'a> Fn(&Var<'a>) -> Result<Var<'a>, AutodiffError>,
    {
        let tape = Tape::new();
        let x = tape.var(&Tensor::new(xs.to_vec(), &[xs.len()]).unwrap());
        let y = f(&x).unwrap();
        let out = y.value().host_slice().to_vec();
        let loss = y.sum(None).unwrap();
        let g = tape.backward(&loss).unwrap();
        (out, g.get(&x).unwrap().unwrap().host_slice().to_vec())
    }

    #[test]
    fn hardshrink_forward_backward_hand_computed() {
        let (y, g) = run(|x| hardshrink(x, 0.5), &[-1.0, -0.5, -0.2, 0.0, 0.5, 0.7]);
        assert_eq!(y, vec![-1.0, 0.0, 0.0, 0.0, 0.0, 0.7]);
        assert_eq!(g, vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn softshrink_forward_backward_hand_computed() {
        let (y, g) = run(|x| softshrink(x, 0.5), &[-1.0, -0.5, -0.2, 0.0, 0.5, 1.25]);
        assert_eq!(y, vec![-0.5, 0.0, 0.0, 0.0, 0.0, 0.75]);
        assert_eq!(g, vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn lambd_zero_and_negative_zero_are_accepted() {
        let (y, _) = run(|x| hardshrink(x, 0.0), &[-1.0, 0.0, 2.0]);
        assert_eq!(y, vec![-1.0, 0.0, 2.0]);
        let (y, _) = run(|x| softshrink(x, -0.0), &[-1.0, 0.0, 2.0]);
        assert_eq!(y, vec![-1.0, 0.0, 2.0]);
    }

    #[test]
    fn nan_passes_through_with_unit_gradient() {
        let (y, g) = run(|x| hardshrink(x, 0.5), &[f32::NAN]);
        assert!(y[0].is_nan());
        assert_eq!(g, vec![1.0]);
        let (y, g) = run(|x| softshrink(x, 0.5), &[f32::NAN]);
        assert!(y[0].is_nan());
        assert_eq!(g, vec![1.0]);
    }

    #[test]
    fn invalid_lambd_is_rejected() {
        let tape = Tape::new();
        let x = tape.var(&Tensor::new(vec![1.0f32, 2.0], &[2]).unwrap());
        for bad in [-0.1f32, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(matches!(
                hardshrink(&x, bad),
                Err(AutodiffError::InvalidArgument(_))
            ));
            assert!(matches!(
                softshrink(&x, bad),
                Err(AutodiffError::InvalidArgument(_))
            ));
        }
    }
}
