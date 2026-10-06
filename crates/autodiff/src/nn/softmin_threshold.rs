//! `Softmin`・`Tanhshrink`・`Threshold`・`RRelu` の nn 層（イシュー #2650・
//! 親 #2648・Phase 親 #2625）。
//!
//! 各層の `forward` は `crate::softmin_threshold_ops` の自由関数への 1 行
//! 委譲で、`Module` 実装も本ファイルに置く（`nn/dropout.rs` と同型。
//! `nn/activation.rs`・`nn/module.rs` を触らず並列イシューとの競合を避ける）。
//! モード依存の `RRelu` は `set_training`／`training` を両方オーバーライド
//! して実フィールドへ保持する（`Module` trait の契約）。
//!
//! **`forward_host` は 4 層とも未提供**（既定の `Unsupported`・
//! `supports_forward_host() == false`）: tape 経由 forward との bit 一致を
//! 検証していない経路を主張しないため（#2146 の `Mish`／`Glu` と同じ判断）。
//! tape 不要経路は層の公開（#2679）側の検討事項（決定記録 §7）。
//!
//! **facade 非公開**: 本モジュールは facade から再エクスポートしない
//! （承認依頼 #2677。`docs/autodiff-softmin-threshold-ops-decision.md`）。

use crate::error::AutodiffError;
use crate::nn::Module;
use crate::softmin_threshold_ops as ops;
use crate::tape::Tape;
use crate::var::Var;

/// PyTorch `nn.Softmin(dim)` 相当。`dim` の範囲検査は forward 時
/// （入力 rank が構築時に分からないため）。
#[derive(Debug, Clone, Copy)]
pub struct Softmin {
    dim: usize,
}

impl Softmin {
    /// 縮約軸 `dim` を指定して構築する。
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    /// 構築済みの縮約軸。
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// [`ops::softmin`] へ委譲する。
    pub fn forward<'t>(&self, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        ops::softmin(input, self.dim)
    }
}

impl Module for Softmin {
    fn forward<'t>(&self, _tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Softmin::forward(self, input)
    }

    /// モジュール doc 参照: `forward_host` 未提供のため `false`。
    fn supports_forward_host(&self) -> bool {
        false
    }
}

/// PyTorch `nn.Tanhshrink` 相当（`x - tanh(x)`）。
#[derive(Debug, Default, Clone, Copy)]
pub struct Tanhshrink;

impl Tanhshrink {
    /// [`ops::tanhshrink`] へ委譲する。
    pub fn forward<'t>(&self, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        ops::tanhshrink(input)
    }
}

impl Module for Tanhshrink {
    fn forward<'t>(&self, _tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Tanhshrink::forward(self, input)
    }

    /// モジュール doc 参照: `forward_host` 未提供のため `false`。
    fn supports_forward_host(&self) -> bool {
        false
    }
}

/// PyTorch `nn.Threshold(threshold, value)` 相当。PyTorch も必須引数の
/// ため `Default` は提供しない。
#[derive(Debug, Clone, Copy)]
pub struct Threshold {
    threshold: f32,
    value: f32,
}

impl Threshold {
    /// 閾値と置換値を指定して構築する（検証しない。IEEE のまま扱う）。
    pub fn new(threshold: f32, value: f32) -> Self {
        Self { threshold, value }
    }

    /// 構築済みの閾値と置換値 `(threshold, value)`。
    pub fn params(&self) -> (f32, f32) {
        (self.threshold, self.value)
    }

    /// [`ops::threshold`] へ委譲する。
    pub fn forward<'t>(&self, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        ops::threshold(input, self.threshold, self.value)
    }
}

impl Module for Threshold {
    fn forward<'t>(&self, _tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Threshold::forward(self, input)
    }

    /// モジュール doc 参照: `forward_host` 未提供のため `false`。
    fn supports_forward_host(&self) -> bool {
        false
    }
}

/// PyTorch `nn.RReLU(lower=1/8, upper=1/3)` 相当。学習時は乱数の傾き、
/// 推論時は `(lower + upper) / 2` の固定傾き。
#[derive(Debug, Clone, Copy)]
pub struct RRelu {
    lower: f32,
    upper: f32,
    training: bool,
}

impl RRelu {
    /// 傾きの範囲 `[lower, upper]` を指定して構築する（`training = true`。
    /// 有限・`lower <= upper` を要求し、違反は
    /// `AutodiffError::InvalidArgument`）。
    pub fn new(lower: f32, upper: f32) -> Result<Self, AutodiffError> {
        if !lower.is_finite() || !upper.is_finite() || lower > upper {
            return Err(AutodiffError::InvalidArgument(format!(
                "RRelu::new: lower/upper must be finite and lower <= upper, got lower={lower}, upper={upper}"
            )));
        }
        Ok(Self {
            lower,
            upper,
            training: true,
        })
    }

    /// 構築済みの `(lower, upper)`。
    pub fn bounds(&self) -> (f32, f32) {
        (self.lower, self.upper)
    }

    /// [`ops::rrelu`] へ委譲する（`self.training` を渡す）。
    pub fn forward<'t>(&self, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        ops::rrelu(input, self.lower, self.upper, self.training)
    }
}

impl Default for RRelu {
    /// PyTorch の既定値（`lower = 1/8`・`upper = 1/3`・学習モード）。
    /// `new` の検査を自明に満たすため、本番経路の `unwrap` を避けて
    /// フィールドを直接構築する。
    fn default() -> Self {
        Self {
            lower: 1.0 / 8.0,
            upper: 1.0 / 3.0,
            training: true,
        }
    }
}

impl Module for RRelu {
    fn forward<'t>(&self, _tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        RRelu::forward(self, input)
    }

    /// モジュール doc 参照: `forward_host` 未提供のため `false`。
    fn supports_forward_host(&self) -> bool {
        false
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn training(&self) -> bool {
        self.training
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fandhe_ai_tensor_core::Tensor;

    fn tape() -> Tape {
        Tape::new()
    }

    fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
        Tensor::new(data, shape).unwrap()
    }

    fn assert_bits_eq(a: &Var<'_>, b: &Var<'_>) {
        let (a, b) = (a.to_tensor(), b.to_tensor());
        assert_eq!(a.shape(), b.shape());
        for (p, q) in a.host_slice().iter().zip(b.host_slice().iter()) {
            assert_eq!(p.to_bits(), q.to_bits());
        }
    }

    #[test]
    fn layers_match_free_functions_bit_for_bit() {
        let tape = tape();
        let x = tape.var(&t(vec![-1.5, -0.2, 0.0, 0.7, 2.0, 0.5], &[2, 3]));
        assert_bits_eq(
            &Softmin::new(1).forward(&x).unwrap(),
            &ops::softmin(&x, 1).unwrap(),
        );
        assert_bits_eq(
            &Tanhshrink.forward(&x).unwrap(),
            &ops::tanhshrink(&x).unwrap(),
        );
        assert_bits_eq(
            &Threshold::new(0.1, -9.0).forward(&x).unwrap(),
            &ops::threshold(&x, 0.1, -9.0).unwrap(),
        );
        let mut r = RRelu::default();
        r.set_training(false);
        assert_bits_eq(
            &r.forward(&x).unwrap(),
            &ops::rrelu(&x, 1.0 / 8.0, 1.0 / 3.0, false).unwrap(),
        );
    }

    #[test]
    fn rrelu_default_and_new_validation() {
        let r = RRelu::default();
        assert_eq!(r.bounds(), (1.0 / 8.0, 1.0 / 3.0));
        assert!(Module::training(&r));
        assert!(RRelu::new(0.5, 0.1).is_err());
        assert!(RRelu::new(f32::NAN, 0.1).is_err());
        assert!(RRelu::new(0.1, f32::INFINITY).is_err());
        assert!(RRelu::new(0.2, 0.2).is_ok());
    }

    #[test]
    fn rrelu_set_training_round_trips() {
        let mut r = RRelu::default();
        r.set_training(false);
        assert!(!Module::training(&r));
        r.set_training(true);
        assert!(Module::training(&r));
    }

    #[test]
    fn layers_do_not_support_forward_host() {
        assert!(!Softmin::new(0).supports_forward_host());
        assert!(!Tanhshrink.supports_forward_host());
        assert!(!Threshold::new(0.0, 0.0).supports_forward_host());
        assert!(!RRelu::default().supports_forward_host());
    }

    #[test]
    fn rrelu_eval_mode_is_deterministic_and_repeatable() {
        let tape = tape();
        let x = tape.var(&t(vec![-1.0, 2.0, -3.0, 0.0], &[4]));
        let mut r = RRelu::default();
        r.set_training(false);
        let a = r.forward(&x).unwrap().to_tensor();
        let b = r.forward(&x).unwrap().to_tensor();
        assert_eq!(a.host_slice().as_ref(), b.host_slice().as_ref());
    }
}
