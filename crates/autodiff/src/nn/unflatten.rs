//! Unflatten（`torch.nn.Unflatten` 相当。イシュー #2159・親 #2131）。
//!
//! [`crate::nn::flatten::Flatten`] の逆変換。指定した 1 軸 `dim` を
//! `unflattened_size` の複数軸へ展開する。`Var::reshape`（`var.rs`）
//! の薄いラッパーで、新規 `Op`／`BackendOps`／VJP／カーネルは追加
//! しない。
//!
//! PyTorch の `-1`（1 軸だけ自動推論）は `usize` では負値を表現
//! できないため非対応（`docs/autodiff-spatial-layers-decision.md`
//! §8 スコープ外）。

use fandhe_ai_tensor_core::{BackendOps, ShapeError, Tensor};

use crate::error::AutodiffError;
use crate::var::Var;

/// `in_shape` の軸 `dim` を `sizes` へ展開した出力 shape を計算する
/// （[`Unflatten::forward`]／[`Unflatten::forward_host`] で共有し、
/// 判定基準が食い違う迂回経路を作らない。`.claude/rules/security.md`
/// A08）。
///
/// 検査順序: ①`dim >= rank` を [`ShapeError::AxisOutOfRange`] で拒否
/// → ②`sizes` の要素積を `checked_mul` で求め、オーバーフローは
/// [`ShapeError::ElementCountOverflow`] → ③積が `in_shape[dim]` と
/// 一致しないことを [`ShapeError::ShapeMismatch`] で拒否 → ④
/// `in[..dim] ++ sizes ++ in[dim+1..]` を返す。
pub(crate) fn unflatten_out_shape(
    in_shape: &[usize],
    dim: usize,
    sizes: &[usize],
) -> Result<Vec<usize>, ShapeError> {
    let rank = in_shape.len();
    if dim >= rank {
        return Err(ShapeError::AxisOutOfRange { axis: dim, rank });
    }
    let mut product: usize = 1;
    for &s in sizes {
        product = product
            .checked_mul(s)
            .ok_or(ShapeError::ElementCountOverflow)?;
    }
    if product != in_shape[dim] {
        return Err(ShapeError::ShapeMismatch {
            lhs: sizes.to_vec(),
            rhs: vec![in_shape[dim]],
        });
    }
    let mut out = Vec::with_capacity(rank - 1 + sizes.len());
    out.extend_from_slice(&in_shape[..dim]);
    out.extend_from_slice(sizes);
    out.extend_from_slice(&in_shape[dim + 1..]);
    Ok(out)
}

/// 軸 `dim` を `unflattened_size` の複数軸へ展開する層。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unflatten {
    dim: usize,
    unflattened_size: Vec<usize>,
}

impl Unflatten {
    /// `dim`／`sizes` を指定して構築する。`sizes` が空の場合は
    /// [`AutodiffError::InvalidArgument`]（軸 1 本を 0 本へ展開する
    /// 操作は無意味なため構築時点で拒否する）。`dim` の範囲検査は
    /// 入力の rank に依存するため forward 時（`unflatten_out_shape`
    /// 内部）に遅延させる（`Flatten::new` と同じ遅延検査契約）。
    pub fn new(dim: usize, sizes: Vec<usize>) -> Result<Self, AutodiffError> {
        if sizes.is_empty() {
            return Err(AutodiffError::InvalidArgument(
                "Unflatten::new: unflattened_size must not be empty".to_string(),
            ));
        }
        Ok(Self {
            dim,
            unflattened_size: sizes,
        })
    }

    /// 展開対象の軸番号。
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// 展開後のサイズ列。
    pub fn unflattened_size(&self) -> &[usize] {
        &self.unflattened_size
    }

    /// `unflatten_out_shape` で出力 shape を求め [`Var::reshape`]
    /// へ委譲する。
    pub fn forward<'t>(&self, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        let in_shape = input.shape();
        let out_shape = unflatten_out_shape(&in_shape, self.dim, &self.unflattened_size)
            .map_err(AutodiffError::Shape)?;
        input.reshape(&out_shape)
    }

    /// [`crate::nn::module::Module::forward_host`]（`Unflatten` 実装）
    /// の本体。[`Self::forward`] と同じ `unflatten_out_shape` を
    /// tape 不要経路で再現する。
    pub fn forward_host(
        &self,
        _ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        let out_shape = unflatten_out_shape(input.shape(), self.dim, &self.unflattened_size)
            .map_err(AutodiffError::Shape)?;
        input.reshape(&out_shape).map_err(AutodiffError::Shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::dense_vec;
    use crate::nn::flatten::Flatten;
    use crate::tape::Tape;

    #[test]
    fn flatten_then_unflatten_round_trips_bit_exact() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&Tensor::new((1..=24).map(|v| v as f32).collect(), &[2, 3, 4]).unwrap());

        let flat = Flatten::new(1, 2).forward(&x).unwrap();
        assert_eq!(flat.shape(), &[2, 12]);

        let unflat = Unflatten::new(1, vec![3, 4])
            .unwrap()
            .forward(&flat)
            .unwrap();
        assert_eq!(unflat.shape(), x.shape());
        assert_eq!(dense_vec(&unflat.to_tensor()), dense_vec(&x.to_tensor()));
    }

    #[test]
    fn new_rejects_empty_sizes() {
        assert!(Unflatten::new(0, vec![]).is_err());
    }

    #[test]
    fn forward_rejects_dim_out_of_range() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap());
        let layer = Unflatten::new(5, vec![1, 2]).unwrap();
        assert!(layer.forward(&x).is_err());
    }

    #[test]
    fn forward_rejects_size_product_mismatch() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap());
        let layer = Unflatten::new(1, vec![3]).unwrap();
        assert!(layer.forward(&x).is_err());
    }

    #[test]
    fn forward_rejects_checked_mul_overflow() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap());
        let layer = Unflatten::new(1, vec![usize::MAX, 2]).unwrap();
        assert!(layer.forward(&x).is_err());
    }

    #[test]
    fn forward_rejects_non_contiguous_input_like_flatten() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&Tensor::new((1..=6).map(|v| v as f32).collect(), &[2, 3]).unwrap());
        let xt = x.permute(&[1, 0]).unwrap();
        let flatten_err = Flatten::new(0, 1).forward(&xt).is_err();
        let unflatten_err = Unflatten::new(0, vec![3, 2]).unwrap().forward(&xt).is_err();
        assert_eq!(flatten_err, unflatten_err);
    }

    #[test]
    fn forward_host_matches_forward() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let host_ops = crate::test_support::test_ops();
        let x = Tensor::new((1..=24).map(|v| v as f32).collect(), &[2, 12]).unwrap();
        let layer = Unflatten::new(1, vec![3, 4]).unwrap();

        let via_host = layer.forward_host(host_ops.as_ref(), &x).unwrap();
        let via_tape = layer.forward(&tape.var(&x)).unwrap();

        assert_eq!(via_host.shape(), via_tape.shape());
        assert_eq!(dense_vec(&via_host), dense_vec(&via_tape.to_tensor()));
    }
}
