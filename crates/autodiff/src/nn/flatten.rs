//! Flatten（`torch.nn.Flatten` 相当。イシュー #2065・親 #2059）。
//!
//! `Var::flatten(start_dim, end_dim)`（`var.rs`。イシュー #1597 で実装
//! 済み）の薄いラッパー。CNN の `Conv2d`／pooling 出力（`[N, C, H, W]`）
//! を `Linear` へ渡す前に `[N, C*H*W]` へ潰す用途で使う（PyTorch の
//! `nn.Sequential(..., nn.Flatten(), nn.Linear(...))` パターンと同型）。
//! `[start_dim, end_dim]`（両端含む）の連続軸を 1 軸へ潰す規約は
//! `Var::flatten` doc・`tensor-core::flatten_out_shape` doc を正とする。
//!
//! 無状態層（`Relu`／`Softmax` 等と同じく `named_parameters` へ寄与
//! しない）のため `Default`／`Copy` を導出する必要はないが、他の
//! パラメータなし活性化層（`Softmax`／`LogSoftmax`）と同様に
//! `Debug`／`Clone`／`Copy` を導出しておく。

use crate::error::AutodiffError;
use crate::var::Var;

/// `[start_dim, end_dim]`（両端含む）の連続軸を 1 軸へ潰す層。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Flatten {
    start_dim: usize,
    end_dim: usize,
}

impl Flatten {
    /// `start_dim`／`end_dim` を指定して構築する。範囲検査は構築時
    /// ではなく forward 時（`Var::flatten` 内部）に行う（`Softmax::new`
    /// と同じ遅延検査契約——`start_dim`／`end_dim` の妥当性は入力の
    /// rank に依存するため、構築時点では判定できない）。
    pub fn new(start_dim: usize, end_dim: usize) -> Self {
        Self { start_dim, end_dim }
    }

    /// `self.start_dim`／`self.end_dim` を用いて [`Var::flatten`] へ
    /// 委譲する。
    pub fn forward<'t>(&self, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        input.flatten(self.start_dim, self.end_dim)
    }

    /// `nn/module.rs::Module::forward_host` の `Flatten` 実装が
    /// `start_dim`／`end_dim` を読み出すためのクレート内アクセサ
    /// （`nn/activation.rs::Softmax::dim` と同じ理由。フィールド自体は
    /// 非公開のまま）。
    pub(crate) fn start_dim(&self) -> usize {
        self.start_dim
    }

    pub(crate) fn end_dim(&self) -> usize {
        self.end_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::dense_vec;
    use crate::tape::Tape;
    use fandhe_ai_tensor_core::Tensor;

    #[test]
    fn forward_matches_var_flatten_directly() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&Tensor::new((1..=24).map(|v| v as f32).collect(), &[2, 3, 4]).unwrap());

        let layer = Flatten::new(1, 2);
        let via_layer = layer.forward(&x).unwrap();
        let via_direct = x.flatten(1, 2).unwrap();

        assert_eq!(via_layer.shape(), &[2, 12]);
        assert_eq!(
            dense_vec(&via_layer.to_tensor()),
            dense_vec(&via_direct.to_tensor())
        );
    }

    #[test]
    fn forward_propagates_shape_error() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap());

        let layer = Flatten::new(2, 5);
        assert!(layer.forward(&x).is_err());
    }

    #[test]
    fn accessors_return_constructed_values() {
        let layer = Flatten::new(1, 3);
        assert_eq!(layer.start_dim(), 1);
        assert_eq!(layer.end_dim(), 3);
    }
}
