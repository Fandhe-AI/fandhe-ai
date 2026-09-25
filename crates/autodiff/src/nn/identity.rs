//! Identity（`torch.nn.Identity` 相当。イシュー #2159・親 #2131）。
//!
//! 入力をそのまま返すだけの無状態層。`Sequential` の中で「この位置
//! には層を置かない」プレースホルダや、条件分岐で層を差し替える
//! 用途に使う（PyTorch の慣習と同じ）。新規 `Op`／`BackendOps`／
//! VJP／カーネルを一切追加しない（`Var` は `Copy` のため、tape へ
//! 新しいノードを push せず入力をそのまま返す）。

use crate::error::AutodiffError;
use crate::var::Var;
use fandhe_ai_tensor_core::{BackendOps, Tensor};

/// 恒等写像を行う無状態層（ZST）。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Identity;

impl Identity {
    /// 構築する（フィールドを持たないため常に成功する）。
    pub fn new() -> Self {
        Self
    }

    /// 入力をそのまま返す（`Var` は `Copy` のため新規ノードを push
    /// しない。勾配は入力へそのまま流れる）。
    pub fn forward<'t>(&self, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Ok(*input)
    }

    /// [`crate::nn::module::Module::forward_host`]（`Identity` 実装）の
    /// 本体。ホスト `Tensor` を複製して返すだけ（算術・shape 変換
    /// なし）。
    pub fn forward_host(
        &self,
        _ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        Ok(input.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::dense_vec;
    use crate::tape::Tape;
    use fandhe_ai_tensor_core::Tensor;

    #[test]
    fn forward_returns_same_value_as_input() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&Tensor::new(vec![1.0, 2.0, 3.0], &[3]).unwrap());
        let layer = Identity::new();
        let y = layer.forward(&x).unwrap();
        assert_eq!(y.shape(), x.shape());
        assert_eq!(dense_vec(&y.to_tensor()), dense_vec(&x.to_tensor()));
    }

    #[test]
    fn forward_gradient_is_identity() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&Tensor::new(vec![1.0, 2.0, 3.0], &[3]).unwrap());
        let layer = Identity::new();
        let y = layer.forward(&x).unwrap();
        let loss = y.sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let grad = grads
            .get(&x)
            .unwrap()
            .expect("x は backward で到達するはず");
        assert_eq!(dense_vec(grad), vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn forward_host_matches_forward() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let host_ops = crate::test_support::test_ops();
        let x = Tensor::new(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let layer = Identity::new();

        let via_host = layer.forward_host(host_ops.as_ref(), &x).unwrap();
        let via_tape = layer.forward(&tape.var(&x)).unwrap();

        assert_eq!(via_host.shape(), via_tape.shape());
        assert_eq!(dense_vec(&via_host), dense_vec(&via_tape.to_tensor()));
    }
}
