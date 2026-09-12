//! RMSNorm／LayerNorm 層（TASK-9.1 系列の拡張。イシュー #1596。
//! `docs/compat-api-scope.md` §1.2 Tier 1）。
//!
//! 既存 `nn::Linear`（`linear.rs`）と同じ「本体（`RmsNorm`／
//! `LayerNorm`。`Tensor<f32>` を永続保持する層パラメータ）→
//! `bind(&tape)` で `Var` 化した `RmsNormVars`／`LayerNormVars`
//! （1 ステップ分のテープ登録済みパラメータ）」の分離パターンを
//! 踏襲する（`tape.rs` の「学習ループでの運用」節参照。`Tape` は
//! ステップごとに生成・破棄される前提のため）。
//!
//! 正規化軸は常に最終軸のみ（`fandhe_ai_tensor_core::row_norm_layout`
//! が `(rows, hidden)` を導出する契約。多次元 `normalized_shape` は
//! 本イシューのスコープ外——利用者は `reshape` で最終軸へ畳める）。
//! `weight`（既定 1 初期化）・`bias`（LayerNorm のみ。既定 0 初期化）は
//! PyTorch `nn.LayerNorm`／`nn.RMSNorm` の既定 `elementwise_affine=true`
//! に相当し、`without_affine` はいずれも `None` にする
//! （`Var::rms_norm`／`layer_norm` は `weight`／`bias` が `None` の場合
//! 対応する演算をスキップする）。

use fandhe_ai_tensor_core::{ShapeError, Tensor};

use crate::error::AutodiffError;
use crate::tape::Tape;
use crate::var::Var;

/// RMSNorm 層の既定 `eps`（PyTorch `nn.RMSNorm` の既定値）。
pub const RMS_NORM_DEFAULT_EPS: f32 = 1e-6;
/// LayerNorm 層の既定 `eps`（PyTorch `nn.LayerNorm` の既定値）。
pub const LAYER_NORM_DEFAULT_EPS: f32 = 1e-5;

/// `eps` の fail-closed 検査（`Var::rms_norm`／`layer_norm` と同じ
/// 「有限かつ非負」契約。層コンストラクタの時点で早期に弾くことで、
/// 誤った `eps` を持つ層が forward 実行まで検出されずに残るのを防ぐ）。
fn validate_eps(eps: f32, who: &str) -> Result<(), AutodiffError> {
    if !eps.is_finite() || eps < 0.0 {
        return Err(AutodiffError::InvalidArgument(format!(
            "{who}: eps must be finite and non-negative, got {eps}"
        )));
    }
    Ok(())
}

/// RMSNorm 層のパラメータ本体。`weight` は `Some` の場合 `[hidden]`。
#[derive(Debug)]
pub struct RmsNorm {
    weight: Option<Tensor<f32>>,
    eps: f32,
}

impl RmsNorm {
    /// `weight` を全要素 `1.0` で初期化する
    /// （`elementwise_affine=true` 相当。PyTorch `nn.RMSNorm` 既定）。
    pub fn new(hidden: usize, eps: f32) -> Result<Self, AutodiffError> {
        validate_eps(eps, "RmsNorm::new")?;
        let weight = Tensor::new(vec![1.0f32; hidden], &[hidden])?;
        Ok(Self {
            weight: Some(weight),
            eps,
        })
    }

    /// `weight` を持たない構成（`elementwise_affine=false` 相当）。
    /// `hidden` の妥当性検査は forward 時（`Var::rms_norm` →
    /// `row_norm_layout`）に委ねる（層自体は `weight` を持たないため
    /// ここでは検証対象がない）。
    pub fn without_affine(eps: f32) -> Result<Self, AutodiffError> {
        validate_eps(eps, "RmsNorm::without_affine")?;
        Ok(Self { weight: None, eps })
    }

    /// 明示的な `weight` から構築する（safetensors ロード等向けの入口。
    /// `Linear::from_parameters` と同じ位置付け）。rank 1 を要求する
    /// （A03: 外部由来パラメータを計算前に検証する契約。
    /// `.claude/rules/security.md`）。
    pub fn from_parameters(weight: Tensor<f32>, eps: f32) -> Result<Self, AutodiffError> {
        validate_eps(eps, "RmsNorm::from_parameters")?;
        if weight.rank() != 1 {
            return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: 1,
                actual: weight.rank(),
            }));
        }
        Ok(Self {
            weight: Some(weight),
            eps,
        })
    }

    /// `weight` パラメータ（`elementwise_affine=false`／
    /// [`Self::without_affine`] の場合は `None`）。`Some` の場合は
    /// shape `[hidden]`（[`Self::new`] の引数・`Var::rms_norm` の入力
    /// 最終軸長と一致）。
    pub fn weight(&self) -> Option<&Tensor<f32>> {
        self.weight.as_ref()
    }

    /// `eps`（`sum(x^2)*inv_n + eps` の加算項）。構築時に
    /// [`validate_eps`] で有限かつ非負であることを検証済み。
    pub fn eps(&self) -> f32 {
        self.eps
    }

    /// このステップの `tape` へ `weight`（あれば）を葉ノードとして
    /// 登録し、`forward` を呼べる `RmsNormVars` を返す
    /// （`Linear::bind` と同じ理由。`Tape::var` 経由のため返る `Var` は
    /// この `tape` に属する）。
    pub fn bind<'t>(&self, tape: &'t Tape) -> RmsNormVars<'t> {
        let weight = self.weight.as_ref().map(|w| tape.var(w));
        RmsNormVars {
            weight,
            eps: self.eps,
        }
    }
}

/// `RmsNorm::bind` が返す、1 ステップ分のテープに登録済みパラメータ。
/// `weight` を公開する理由は [`crate::nn::linear::LinearVars`] と同じ
/// （`Tape::backward` 後の `Gradients::get(&vars.weight)` は呼び出し側
/// の責務）。
pub struct RmsNormVars<'t> {
    /// `RmsNorm::bind` 時点の `weight`（`elementwise_affine=false` の
    /// 場合は `None`）をこの `tape` へ登録した `Var`。`Tape::backward`
    /// 後に `Gradients::get(&vars.weight)`（`Some` の場合）で `dweight`
    /// を取得する（呼び出し側の責務。[`crate::nn::linear::LinearVars`]
    /// と同じ理由）。
    pub weight: Option<Var<'t>>,
    eps: f32,
}

impl<'t> RmsNormVars<'t> {
    /// `input.rms_norm(weight, eps)`（`Var::rms_norm`。`var.rs`）への
    /// 委譲。
    pub fn forward(&self, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        input.rms_norm(self.weight.as_ref(), self.eps)
    }
}

/// LayerNorm 層のパラメータ本体。`weight`／`bias` はそれぞれ独立に
/// `Some`/`None` を取りうる（[`Self::new`] は両方 `Some`、
/// [`Self::without_affine`] は両方 `None`）。
#[derive(Debug)]
pub struct LayerNorm {
    weight: Option<Tensor<f32>>,
    bias: Option<Tensor<f32>>,
    eps: f32,
}

impl LayerNorm {
    /// `weight` を全要素 `1.0`・`bias` を全要素 `0.0` で初期化する
    /// （`elementwise_affine=true` 相当。PyTorch `nn.LayerNorm` 既定）。
    pub fn new(hidden: usize, eps: f32) -> Result<Self, AutodiffError> {
        validate_eps(eps, "LayerNorm::new")?;
        let weight = Tensor::new(vec![1.0f32; hidden], &[hidden])?;
        let bias = Tensor::new(vec![0.0f32; hidden], &[hidden])?;
        Ok(Self {
            weight: Some(weight),
            bias: Some(bias),
            eps,
        })
    }

    /// `weight`／`bias` を持たない構成（`elementwise_affine=false`
    /// 相当）。
    pub fn without_affine(eps: f32) -> Result<Self, AutodiffError> {
        validate_eps(eps, "LayerNorm::without_affine")?;
        Ok(Self {
            weight: None,
            bias: None,
            eps,
        })
    }

    /// 明示的な `weight`／`bias` から構築する（safetensors ロード等
    /// 向け）。渡す場合はいずれも rank 1・shape 一致（`weight`／`bias`
    /// を両方渡す場合は同一 shape）を要求する。
    pub fn from_parameters(
        weight: Option<Tensor<f32>>,
        bias: Option<Tensor<f32>>,
        eps: f32,
    ) -> Result<Self, AutodiffError> {
        validate_eps(eps, "LayerNorm::from_parameters")?;
        if let Some(w) = &weight
            && w.rank() != 1
        {
            return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: 1,
                actual: w.rank(),
            }));
        }
        if let Some(b) = &bias
            && b.rank() != 1
        {
            return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: 1,
                actual: b.rank(),
            }));
        }
        if let (Some(w), Some(b)) = (&weight, &bias)
            && w.shape() != b.shape()
        {
            return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                lhs: w.shape().to_vec(),
                rhs: b.shape().to_vec(),
            }));
        }
        Ok(Self { weight, bias, eps })
    }

    /// `weight` パラメータ（`elementwise_affine=false`／
    /// [`Self::without_affine`] の場合は `None`）。`Some` の場合は
    /// shape `[hidden]`（[`Self::new`] の引数・`Var::layer_norm` の
    /// 入力最終軸長と一致。`bias` とは独立に `None` を取りうる）。
    pub fn weight(&self) -> Option<&Tensor<f32>> {
        self.weight.as_ref()
    }

    /// `bias` パラメータ（`elementwise_affine=false`／
    /// [`Self::without_affine`] の場合は `None`）。`Some` の場合は
    /// shape `[hidden]`（`weight` とは独立に `None` を取りうる。
    /// [`Self::from_parameters`] は両方 `Some` の場合のみ shape 一致を
    /// 要求する）。
    pub fn bias(&self) -> Option<&Tensor<f32>> {
        self.bias.as_ref()
    }

    /// `eps`（`var(x) + eps` の加算項）。構築時に [`validate_eps`] で
    /// 有限かつ非負であることを検証済み。
    pub fn eps(&self) -> f32 {
        self.eps
    }

    /// このステップの `tape` へ `weight`／`bias`（あれば）を葉ノードと
    /// して登録し、`forward` を呼べる `LayerNormVars` を返す。
    pub fn bind<'t>(&self, tape: &'t Tape) -> LayerNormVars<'t> {
        let weight = self.weight.as_ref().map(|w| tape.var(w));
        let bias = self.bias.as_ref().map(|b| tape.var(b));
        LayerNormVars {
            weight,
            bias,
            eps: self.eps,
        }
    }
}

/// `LayerNorm::bind` が返す、1 ステップ分のテープに登録済みパラメータ。
pub struct LayerNormVars<'t> {
    /// `LayerNorm::bind` 時点の `weight`（`elementwise_affine=false`
    /// の場合は `None`）をこの `tape` へ登録した `Var`。`Tape::backward`
    /// 後に `Gradients::get(&vars.weight)`（`Some` の場合）で `dweight`
    /// を取得する（呼び出し側の責務）。
    pub weight: Option<Var<'t>>,
    /// `LayerNorm::bind` 時点の `bias`（`elementwise_affine=false` の
    /// 場合は `None`）をこの `tape` へ登録した `Var`。`Tape::backward`
    /// 後に `Gradients::get(&vars.bias)`（`Some` の場合）で `dbias`
    /// を取得する（呼び出し側の責務。`weight` とは独立に `None` を
    /// 取りうる）。
    pub bias: Option<Var<'t>>,
    eps: f32,
}

impl<'t> LayerNormVars<'t> {
    /// `input.layer_norm(weight, bias, eps)`（`Var::layer_norm`）への
    /// 委譲。
    pub fn forward(&self, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        input.layer_norm(self.weight.as_ref(), self.bias.as_ref(), self.eps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tape::Tape;

    fn dense_vec(t: &Tensor<f32>) -> Vec<f32> {
        t.as_slice()
            .expect("test: expected contiguous tensor")
            .to_vec()
    }

    #[test]
    fn rms_norm_new_initializes_weight_to_ones() {
        let layer = RmsNorm::new(4, RMS_NORM_DEFAULT_EPS).unwrap();
        assert_eq!(dense_vec(layer.weight().unwrap()), vec![1.0f32; 4]);
    }

    #[test]
    fn rms_norm_without_affine_has_no_weight() {
        let layer = RmsNorm::without_affine(RMS_NORM_DEFAULT_EPS).unwrap();
        assert!(layer.weight().is_none());
    }

    #[test]
    fn rms_norm_rejects_non_finite_eps() {
        let err = RmsNorm::new(4, f32::NAN).unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn rms_norm_rejects_negative_eps() {
        let err = RmsNorm::new(4, -1e-5).unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn rms_norm_from_parameters_rejects_rank_mismatch() {
        let w = Tensor::new(vec![1.0f32; 4], &[2, 2]).unwrap();
        let err = RmsNorm::from_parameters(w, RMS_NORM_DEFAULT_EPS).unwrap_err();
        assert!(matches!(
            err,
            AutodiffError::Shape(ShapeError::RankMismatch {
                expected: 1,
                actual: 2
            })
        ));
    }

    #[test]
    fn rms_norm_bind_forward_matches_direct_var_call() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&Tensor::new(vec![1.0, 2.0, -1.0, 0.5], &[1, 4]).unwrap());
        let layer = RmsNorm::new(4, RMS_NORM_DEFAULT_EPS).unwrap();

        let via_vars = layer.bind(&tape).forward(&x).unwrap();
        let w = tape.var(layer.weight().unwrap());
        let via_direct = x.rms_norm(Some(&w), RMS_NORM_DEFAULT_EPS).unwrap();

        assert_eq!(
            dense_vec(&via_vars.to_tensor()),
            dense_vec(&via_direct.to_tensor())
        );
    }

    #[test]
    fn layer_norm_new_initializes_weight_ones_bias_zeros() {
        let layer = LayerNorm::new(4, LAYER_NORM_DEFAULT_EPS).unwrap();
        assert_eq!(dense_vec(layer.weight().unwrap()), vec![1.0f32; 4]);
        assert_eq!(dense_vec(layer.bias().unwrap()), vec![0.0f32; 4]);
    }

    #[test]
    fn layer_norm_without_affine_has_no_weight_or_bias() {
        let layer = LayerNorm::without_affine(LAYER_NORM_DEFAULT_EPS).unwrap();
        assert!(layer.weight().is_none());
        assert!(layer.bias().is_none());
    }

    #[test]
    fn layer_norm_rejects_non_finite_eps() {
        let err = LayerNorm::new(4, f32::INFINITY).unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn layer_norm_from_parameters_rejects_weight_bias_shape_mismatch() {
        let w = Tensor::new(vec![1.0f32; 4], &[4]).unwrap();
        let b = Tensor::new(vec![0.0f32; 3], &[3]).unwrap();
        let err = LayerNorm::from_parameters(Some(w), Some(b), LAYER_NORM_DEFAULT_EPS).unwrap_err();
        assert!(matches!(
            err,
            AutodiffError::Shape(ShapeError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn layer_norm_bind_forward_matches_direct_var_call() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&Tensor::new(vec![1.0, 2.0, -1.0, 0.5], &[1, 4]).unwrap());
        let layer = LayerNorm::new(4, LAYER_NORM_DEFAULT_EPS).unwrap();

        let via_vars = layer.bind(&tape).forward(&x).unwrap();
        let w = tape.var(layer.weight().unwrap());
        let b = tape.var(layer.bias().unwrap());
        let via_direct = x
            .layer_norm(Some(&w), Some(&b), LAYER_NORM_DEFAULT_EPS)
            .unwrap();

        assert_eq!(
            dense_vec(&via_vars.to_tensor()),
            dense_vec(&via_direct.to_tensor())
        );
    }
}
