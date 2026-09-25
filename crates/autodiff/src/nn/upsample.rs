//! Upsample（`torch.nn.Upsample` 相当。イシュー #2159・親 #2131）。
//!
//! [`crate::var::Var::interpolate`]（イシュー #1757・#1762・#2152）の
//! 薄いラッパー。`size` 直接指定と `scale_factor` 指定の両方を受理
//! し、後者は [`fandhe_ai_tensor_core::interpolate_size_from_scale_factor`]
//! （#2152）で `size` を導出してから `Var::interpolate` へ渡す
//! （新規 `Op`／`BackendOps`／VJP／カーネルは追加しない）。

use fandhe_ai_tensor_core::{
    BackendOps, InterpolateMode, ShapeError, Tensor, interpolate_out_shape_for_mode,
    interpolate_size_from_scale_factor,
};

use crate::error::AutodiffError;
use crate::grad::interpolate_with_fallback;
use crate::var::Var;

/// `Upsample` のサイズ指定方式（PyTorch `nn.Upsample(size=…)`／
/// `nn.Upsample(scale_factor=…)` の二者択一を型で表現する。両方
/// 同時指定・どちらも未指定という状態は型の上で表現できない）。
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum UpsampleSize {
    /// 出力の空間軸サイズを直接指定する。
    Size(Vec<usize>),
    /// 入力の空間軸サイズへ乗じる倍率（軸ごと）を指定する
    /// （`recompute_scale_factor=True` 相当の座標系になる点は
    /// `interpolate_size_from_scale_factor` doc・
    /// `docs/autodiff-spatial-layers-decision.md` §6 参照）。
    ScaleFactor(Vec<f64>),
}

/// リサンプリング層（`size`／`scale_factor` のいずれかと `mode` を
/// 保持する）。
#[derive(Debug, Clone, PartialEq)]
pub struct Upsample {
    spec: UpsampleSize,
    mode: InterpolateMode,
}

impl Upsample {
    /// 出力サイズを直接指定して構築する。`size` が空の場合は
    /// [`AutodiffError::InvalidArgument`]（`mode` と空間軸数の整合は
    /// forward 時に [`interpolate_out_shape_for_mode`] へ委ねる遅延
    /// 検査契約——`Flatten::new` と同型）。
    pub fn with_size(size: Vec<usize>, mode: InterpolateMode) -> Result<Self, AutodiffError> {
        if size.is_empty() {
            return Err(AutodiffError::InvalidArgument(
                "Upsample::with_size: size must not be empty".to_string(),
            ));
        }
        Ok(Self {
            spec: UpsampleSize::Size(size),
            mode,
        })
    }

    /// 倍率を指定して構築する。`scale` が空、または非有限／0 以下の
    /// 要素を含む場合は [`AutodiffError::InvalidArgument`]（構築時点
    /// で即座に判定できる項目のみ検査し、入力 rank に依存する長さ
    /// 整合は forward 時の [`interpolate_size_from_scale_factor`] へ
    /// 委ねる）。
    pub fn with_scale_factor(
        scale: Vec<f64>,
        mode: InterpolateMode,
    ) -> Result<Self, AutodiffError> {
        if scale.is_empty() {
            return Err(AutodiffError::InvalidArgument(
                "Upsample::with_scale_factor: scale_factor must not be empty".to_string(),
            ));
        }
        for (axis, &s) in scale.iter().enumerate() {
            if !s.is_finite() || s <= 0.0 {
                return Err(AutodiffError::InvalidArgument(format!(
                    "Upsample::with_scale_factor: scale_factor[{axis}] = {s} は非有限または 0 \
                     以下で不正"
                )));
            }
        }
        Ok(Self {
            spec: UpsampleSize::ScaleFactor(scale),
            mode,
        })
    }

    /// サイズ指定方式。
    pub fn spec(&self) -> &UpsampleSize {
        &self.spec
    }

    /// リサンプリング方式。
    pub fn mode(&self) -> InterpolateMode {
        self.mode
    }

    /// `in_shape`（入力全体の shape）から `Var::interpolate`／
    /// `Tensor::interpolate` 相当へ渡す `size` を確定する
    /// （[`Self::forward`]／[`Self::forward_host`] で共有し、判定
    /// 基準が食い違う迂回経路を作らない。`.claude/rules/security.md`
    /// A08）。`ScaleFactor` の場合は `in_shape` の末尾
    /// `scale.len()` 軸を空間軸とみなす（rank 不足は
    /// `ShapeError::RankMismatch`）。
    fn resolve_size(&self, in_shape: &[usize]) -> Result<Vec<usize>, AutodiffError> {
        match &self.spec {
            UpsampleSize::Size(size) => Ok(size.clone()),
            UpsampleSize::ScaleFactor(scale) => {
                let rank = in_shape.len();
                if rank < scale.len() {
                    return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                        expected: scale.len(),
                        actual: rank,
                    }));
                }
                let spatial_in = &in_shape[rank - scale.len()..];
                interpolate_size_from_scale_factor(spatial_in, scale)
                    .map_err(|e| AutodiffError::InvalidArgument(format!("Upsample: {e}")))
            } // `UpsampleSize` は `#[non_exhaustive]`（本クレート内
              // 定義のため既知 variant を網羅済み）。将来 variant
              // 追加時はこの match の網羅性チェックが追加を強制する。
        }
    }

    /// `Self::resolve_size` で `size` を確定し [`Var::interpolate`]
    /// へ委譲する。
    pub fn forward<'t>(&self, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        let in_shape = input.shape();
        let size = self.resolve_size(&in_shape)?;
        input.interpolate(&size, self.mode)
    }

    /// [`crate::nn::module::Module::forward_host`]（`Upsample` 実装）
    /// の本体。`Self::resolve_size`・
    /// [`interpolate_out_shape_for_mode`]・`grad::
    /// interpolate_with_fallback`（`Var::interpolate` 自身が使う
    /// ヘルパーと同一）を tape 不要経路で再現する。
    pub fn forward_host(
        &self,
        ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        let size = self.resolve_size(input.shape())?;
        let out_shape = interpolate_out_shape_for_mode(input.shape(), &size, self.mode)
            .map_err(AutodiffError::Shape)?;
        interpolate_with_fallback(ops, input, &size, self.mode, &out_shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::dense_vec;
    use crate::tape::Tape;

    #[test]
    fn with_size_rejects_empty() {
        assert!(Upsample::with_size(vec![], InterpolateMode::Nearest).is_err());
    }

    #[test]
    fn with_scale_factor_rejects_empty_and_non_positive() {
        assert!(Upsample::with_scale_factor(vec![], InterpolateMode::Nearest).is_err());
        assert!(Upsample::with_scale_factor(vec![0.0], InterpolateMode::Nearest).is_err());
        assert!(Upsample::with_scale_factor(vec![-1.0], InterpolateMode::Nearest).is_err());
        assert!(Upsample::with_scale_factor(vec![f64::NAN], InterpolateMode::Nearest).is_err());
    }

    #[test]
    fn forward_nearest_scale_2_matches_hand_computed() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]).unwrap());
        let layer = Upsample::with_scale_factor(vec![2.0, 2.0], InterpolateMode::Nearest).unwrap();
        let y = layer.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 1, 4, 4]);
        assert_eq!(
            dense_vec(&y.to_tensor()),
            vec![
                1.0, 1.0, 2.0, 2.0, //
                1.0, 1.0, 2.0, 2.0, //
                3.0, 3.0, 4.0, 4.0, //
                3.0, 3.0, 4.0, 4.0,
            ]
        );
    }

    #[test]
    fn with_size_and_scale_factor_agree_on_same_resulting_size() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]).unwrap());
        let mode = InterpolateMode::Bilinear {
            align_corners: false,
        };
        let via_size = Upsample::with_size(vec![4, 4], mode)
            .unwrap()
            .forward(&x)
            .unwrap();
        let via_scale = Upsample::with_scale_factor(vec![2.0, 2.0], mode)
            .unwrap()
            .forward(&x)
            .unwrap();
        assert_eq!(via_size.shape(), via_scale.shape());
        assert_eq!(
            dense_vec(&via_size.to_tensor()),
            dense_vec(&via_scale.to_tensor())
        );
    }

    #[test]
    fn forward_rejects_scale_factor_rank_shortage() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&Tensor::new(vec![1.0, 2.0], &[2]).unwrap());
        let layer = Upsample::with_scale_factor(vec![2.0, 2.0], InterpolateMode::Nearest).unwrap();
        assert!(layer.forward(&x).is_err());
    }

    #[test]
    fn forward_rejects_mode_rank_mismatch() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        // Bilinear はちょうど 2 空間軸を要求するため、1 軸指定は拒否される。
        let x = tape.var(&Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 4]).unwrap());
        let layer = Upsample::with_size(
            vec![8],
            InterpolateMode::Bilinear {
                align_corners: false,
            },
        )
        .unwrap();
        assert!(layer.forward(&x).is_err());
    }

    #[test]
    fn forward_bilinear_numeric_gradient_check() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let base = vec![1.0f32, 2.0, 3.0, 4.0];
        let mode = InterpolateMode::Bilinear {
            align_corners: false,
        };
        let eps = 1e-3f32;
        for i in 0..base.len() {
            let mut plus = base.clone();
            plus[i] += eps;
            let mut minus = base.clone();
            minus[i] -= eps;

            let x = tape.var(&Tensor::new(base.clone(), &[1, 1, 2, 2]).unwrap());
            let y = Upsample::with_size(vec![4, 4], mode)
                .unwrap()
                .forward(&x)
                .unwrap();
            let loss = y.sum(None).unwrap();
            let grads = tape.backward(&loss).unwrap();
            let grad = grads
                .get(&x)
                .unwrap()
                .expect("x は backward で到達するはず");
            let analytic = dense_vec(grad)[i];

            let xp = Tensor::new(plus, &[1, 1, 2, 2]).unwrap();
            let xm = Tensor::new(minus, &[1, 1, 2, 2]).unwrap();
            let ops = crate::test_support::test_ops();
            let out_shape = interpolate_out_shape_for_mode(xp.shape(), &[4, 4], mode).unwrap();
            let lp: f32 = dense_vec(
                &interpolate_with_fallback(ops.as_ref(), &xp, &[4, 4], mode, &out_shape).unwrap(),
            )
            .iter()
            .sum();
            let lm: f32 = dense_vec(
                &interpolate_with_fallback(ops.as_ref(), &xm, &[4, 4], mode, &out_shape).unwrap(),
            )
            .iter()
            .sum();
            let numeric = (lp - lm) / (2.0 * eps);
            assert!(
                (analytic - numeric).abs() < 1e-2,
                "index {i}: analytic={analytic} numeric={numeric}"
            );
        }
    }

    #[test]
    fn forward_host_matches_forward() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let host_ops = crate::test_support::test_ops();
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]).unwrap();
        let layer = Upsample::with_scale_factor(vec![2.0, 2.0], InterpolateMode::Nearest).unwrap();

        let via_host = layer.forward_host(host_ops.as_ref(), &x).unwrap();
        let via_tape = layer.forward(&tape.var(&x)).unwrap();

        assert_eq!(via_host.shape(), via_tape.shape());
        assert_eq!(dense_vec(&via_host), dense_vec(&via_tape.to_tensor()));
    }
}
