//! `Bilinear` の nn 層（PyTorch `nn.Bilinear(in1, in2, out, bias)` 相当。イシュー #3818・親 #3813。
//! 設計記録 `docs/nn-shrink-bilinear-channel-shuffle-decision.md` §2）。
//!
//! 入力が 2 つ（`x1`・`x2`）のため入力 1 つの [`crate::nn::Module`] は実装せず、`Linear`／
//! [`LinearVars`](crate::nn::LinearVars) と同じ「`bind` → `forward`」の構成にする。`forward` は
//! [`Var::bilinear`] を呼ぶだけで、検査と合成は `crate::bilinear_ops` が持つ。
//!
//! 層型は autodiff 内部に留め、facade には再エクスポートしない（承認形。`Sequential` にも載せない
//! ため model_io・ONNX export は不変）。外部利用者は `Var::bilinear` と自前のパラメータで
//! 2 入力の独自構造体を組む。
//!
//! 初期化は `U(-1/√in1, 1/√in1)`（PyTorch のソースの定義に基づく記述で、参照版での実測確認は
//! していない）。シード導出は `nn/init.rs::derive_seed` で weight・bias を独立させる。

use fandhe_ai_tensor_core::{ShapeError, Tensor};

use crate::error::AutodiffError;
use crate::nn::init::{BIAS_SEED_SALT, WEIGHT_SEED_SALT, checked_uniform_init, derive_seed};
use crate::tape::Tape;
use crate::var::Var;

/// 双線形層のパラメータ本体。`weight` は `[out, in1, in2]`、`bias` は `Some` なら `[out]`。
pub struct Bilinear {
    weight: Tensor<f32>,
    bias: Option<Tensor<f32>>,
}

impl Bilinear {
    /// 決定的シードで `U(-1/√in1, 1/√in1)` の一様初期化を行う。`in1 == 0` は bound が非有限に
    /// なるため `InvalidArgument`。`in2 == 0`・`out == 0` は空テンソルとして受理する。
    /// 要素数の積は `checked_mul` で検査する（`ElementCountOverflow`）。
    pub fn new(
        in1: usize,
        in2: usize,
        out: usize,
        bias: bool,
        seed: u64,
    ) -> Result<Bilinear, AutodiffError> {
        if in1 == 0 {
            return Err(AutodiffError::InvalidArgument(
                "Bilinear::new: in1 must be > 0 (1/sqrt(in1) would be non-finite)".to_string(),
            ));
        }
        let len = out
            .checked_mul(in1)
            .and_then(|v| v.checked_mul(in2))
            .ok_or(AutodiffError::Shape(ShapeError::ElementCountOverflow))?;
        let bound = 1.0 / (in1 as f32).sqrt();
        let weight_data = checked_uniform_init(len, bound, derive_seed(seed, WEIGHT_SEED_SALT))?;
        let weight = Tensor::new(weight_data, &[out, in1, in2])?;
        let bias = if bias {
            let data = checked_uniform_init(out, bound, derive_seed(seed, BIAS_SEED_SALT))?;
            Some(Tensor::new(data, &[out])?)
        } else {
            None
        };
        Ok(Bilinear { weight, bias })
    }

    /// 明示的な重み・バイアスから構築する（外部由来パラメータの入口。A03: 計算前に検証する）。
    /// `weight` は rank 3 かつ `in1 > 0`、`bias` は `[out]`。
    pub fn from_parameters(
        weight: Tensor<f32>,
        bias: Option<Tensor<f32>>,
    ) -> Result<Bilinear, AutodiffError> {
        if weight.rank() != 3 {
            return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: 3,
                actual: weight.rank(),
            }));
        }
        let shape = weight.shape().to_vec();
        if shape[1] == 0 {
            return Err(AutodiffError::InvalidArgument(
                "Bilinear::from_parameters: weight.shape()[1] (in1) must be > 0".to_string(),
            ));
        }
        if let Some(b) = &bias
            && b.shape() != [shape[0]]
        {
            return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                lhs: b.shape().to_vec(),
                rhs: vec![shape[0]],
            }));
        }
        Ok(Bilinear { weight, bias })
    }

    /// `weight`（`[out, in1, in2]`）。
    pub fn weight(&self) -> &Tensor<f32> {
        &self.weight
    }

    /// `bias`（`[out]`）。
    pub fn bias(&self) -> Option<&Tensor<f32>> {
        self.bias.as_ref()
    }

    /// パラメータを 1 ステップ分のテープへ葉として登録する。
    pub fn bind<'t>(&self, tape: &'t Tape) -> BilinearVars<'t> {
        BilinearVars {
            weight: tape.var(&self.weight),
            bias: self.bias.as_ref().map(|b| tape.var(b)),
        }
    }
}

/// [`Bilinear::bind`] が返す、1 ステップ分のテープに登録済みパラメータ。
pub struct BilinearVars<'t> {
    pub weight: Var<'t>,
    pub bias: Option<Var<'t>>,
}

impl<'t> BilinearVars<'t> {
    /// `y = bilinear(x1, x2, weight, bias)`。[`Var::bilinear`] を呼ぶだけ。
    pub fn forward(&self, x1: &Var<'t>, x2: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        x1.bilinear(x2, &self.weight, self.bias.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_is_bounded_and_deterministic() {
        let a = Bilinear::new(4, 3, 2, true, 7).unwrap();
        let b = Bilinear::new(4, 3, 2, true, 7).unwrap();
        assert_eq!(a.weight().shape(), [2, 4, 3]);
        assert_eq!(a.bias().unwrap().shape(), [2]);
        assert_eq!(a.weight().host_slice(), b.weight().host_slice());
        let bound = 0.5f32; // 1/sqrt(4)
        assert!(a.weight().host_slice().iter().all(|v| v.abs() <= bound));
        assert!(
            a.bias()
                .unwrap()
                .host_slice()
                .iter()
                .all(|v| v.abs() <= bound)
        );
        assert!(Bilinear::new(4, 3, 2, false, 7).unwrap().bias().is_none());
    }

    #[test]
    fn new_rejects_zero_in1_and_overflow() {
        assert!(matches!(
            Bilinear::new(0, 3, 2, true, 1),
            Err(AutodiffError::InvalidArgument(_))
        ));
        assert!(matches!(
            Bilinear::new(usize::MAX, 2, 2, false, 1),
            Err(AutodiffError::Shape(ShapeError::ElementCountOverflow))
        ));
    }

    #[test]
    fn new_returns_err_on_huge_allocation() {
        // 要素数は usize に収まるが Vec<f32> の容量上限を超える（panic せず Err）
        assert!(Bilinear::new(1, 1, 1usize << 61, false, 0).is_err());
        assert!(Bilinear::new(1, 0, 1usize << 61, true, 0).is_err());
    }

    #[test]
    fn from_parameters_validates() {
        let w2 = Tensor::new(vec![0.0f32; 6], &[2, 3]).unwrap();
        assert!(matches!(
            Bilinear::from_parameters(w2, None),
            Err(AutodiffError::Shape(ShapeError::RankMismatch { .. }))
        ));
        let w0 = Tensor::new(vec![], &[2, 0, 3]).unwrap();
        assert!(matches!(
            Bilinear::from_parameters(w0, None),
            Err(AutodiffError::InvalidArgument(_))
        ));
        let w = Tensor::new(vec![0.0f32; 12], &[2, 2, 3]).unwrap();
        let bad = Tensor::new(vec![0.0f32; 3], &[3]).unwrap();
        assert!(matches!(
            Bilinear::from_parameters(w.clone(), Some(bad)),
            Err(AutodiffError::Shape(ShapeError::ShapeMismatch { .. }))
        ));
        let ok = Tensor::new(vec![0.0f32; 2], &[2]).unwrap();
        assert!(Bilinear::from_parameters(w, Some(ok)).is_ok());
    }

    #[test]
    fn bind_forward_matches_var_bilinear() {
        let layer = Bilinear::new(3, 2, 4, true, 11).unwrap();
        let tape = Tape::new();
        let vars = layer.bind(&tape);
        let x1 = tape.var(&Tensor::new(vec![0.1f32, -0.2, 0.3, 0.4, 0.5, -0.6], &[2, 3]).unwrap());
        let x2 = tape.var(&Tensor::new(vec![1.0f32, -1.0, 0.5, 2.0], &[2, 2]).unwrap());
        let y = vars.forward(&x1, &x2).unwrap();
        let want = x1.bilinear(&x2, &vars.weight, vars.bias.as_ref()).unwrap();
        assert_eq!(y.shape(), vec![2, 4]);
        assert_eq!(y.value().host_slice(), want.value().host_slice());
    }
}
