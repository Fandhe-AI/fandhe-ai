//! 共通 `Module` trait（TASK-9.2a・#95）。
//!
//! `compat::Sequential`（`compat/sequential.rs`）がレイヤーの列を
//! `Vec<Box<dyn Module>>` として保持し、種類の異なる `nn` の部品
//! （`Linear`・活性化関数）を統一シグネチャで呼べるようにするための
//! 最小 trait。`nn/mod.rs` が「共通 `Module` trait の定義は
//! `compat::Sequential` 設計時に確定する」としていた点を本イシューで
//! 確定する。
//!
//! シグネチャに `tape: &'t Tape` を含める理由: `Linear` は
//! `Linear::bind(&tape)` でそのステップの葉ノードを毎回登録してから
//! でないと forward できない（`nn/linear.rs` の `Tape` ライフサイクル
//! 節参照）。活性化関数側は `tape` を使わないが、`Box<dyn Module>` を
//! 均一に扱うため同じ引数を受け取る。この「毎呼び出しで葉ノードを
//! 登録し直す」契約は推論・1 ステップ forward 用であり、学習（勾配
//! 取得・パラメータ更新）には対応しない（`compat::Sequential` 側の
//! スコープ外事項として記録。`docs/compat-api-scope.md` 参照）。

use crate::error::AutodiffError;
use crate::nn::activation::{Relu, Sigmoid, Tanh};
use crate::nn::linear::Linear;
use crate::tape::Tape;
use crate::var::Var;

/// `nn` の部品（層・活性化関数）に共通の forward シグネチャ。
pub trait Module {
    /// このステップの `tape` 上で 1 回分の forward を計算する。
    fn forward<'t>(&self, tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError>;
}

/// `Linear::bind(tape)` で当該ステップの葉ノードを登録してから
/// `LinearVars::forward` を呼ぶ（`nn/linear.rs` 参照）。
impl Module for Linear {
    fn forward<'t>(&self, tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        self.bind(tape).forward(input)
    }
}

/// `Relu::forward`（shape 不変の単項演算のため構造的に失敗しえない）を
/// `Result` へ包むだけの委譲。`tape` は使わない（`nn/activation.rs`
/// 参照）。
impl Module for Relu {
    fn forward<'t>(&self, _tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Ok(Relu::forward(self, input))
    }
}

/// `Sigmoid::forward` への委譲。`Relu` の実装と同じ理由で `tape` は
/// 使わない。
impl Module for Sigmoid {
    fn forward<'t>(&self, _tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Ok(Sigmoid::forward(self, input))
    }
}

/// `Tanh::forward` への委譲。`Relu` の実装と同じ理由で `tape` は
/// 使わない。
impl Module for Tanh {
    fn forward<'t>(&self, _tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Ok(Tanh::forward(self, input))
    }
}

#[cfg(test)]
mod tests {
    //! `Module::forward` が既存の直接呼び出し（`Linear::bind().forward()`・
    //! `Relu::forward()` 等）と同一の値・テープ記録を返すことを検証する
    //! （「薄いラッパー性」の担保）。

    use super::*;
    use crate::eval::dense_vec;
    use tensor_core::Tensor;

    #[test]
    fn linear_module_forward_matches_bind_forward() {
        let linear = Linear::new(3, 2, true, 42).expect("seed=42 は有効な構築引数");
        let tape = Tape::new(crate::test_support::test_ops());
        let x = tape.var(&Tensor::new(vec![1.0, 2.0, 3.0], &[1, 3]).unwrap());

        let via_module = <Linear as Module>::forward(&linear, &tape, &x).unwrap();
        let via_direct = linear.bind(&tape).forward(&x).unwrap();

        assert_eq!(
            dense_vec(&via_module.to_tensor()),
            dense_vec(&via_direct.to_tensor())
        );
    }

    #[test]
    fn relu_module_forward_matches_direct_forward() {
        let tape = Tape::new(crate::test_support::test_ops());
        let x = tape.var(&Tensor::new(vec![-1.0, 2.0], &[2]).unwrap());

        let via_module = <Relu as Module>::forward(&Relu, &tape, &x).unwrap();
        let via_direct = Relu.forward(&x);

        assert_eq!(
            dense_vec(&via_module.to_tensor()),
            dense_vec(&via_direct.to_tensor())
        );
    }

    #[test]
    fn sigmoid_module_forward_matches_direct_forward() {
        let tape = Tape::new(crate::test_support::test_ops());
        let x = tape.var(&Tensor::new(vec![-1.0, 2.0], &[2]).unwrap());

        let via_module = <Sigmoid as Module>::forward(&Sigmoid, &tape, &x).unwrap();
        let via_direct = Sigmoid.forward(&x);

        assert_eq!(
            dense_vec(&via_module.to_tensor()),
            dense_vec(&via_direct.to_tensor())
        );
    }

    #[test]
    fn tanh_module_forward_matches_direct_forward() {
        let tape = Tape::new(crate::test_support::test_ops());
        let x = tape.var(&Tensor::new(vec![-1.0, 2.0], &[2]).unwrap());

        let via_module = <Tanh as Module>::forward(&Tanh, &tape, &x).unwrap();
        let via_direct = Tanh.forward(&x);

        assert_eq!(
            dense_vec(&via_module.to_tensor()),
            dense_vec(&via_direct.to_tensor())
        );
    }
}
