//! ZeroPad2d（`torch.nn.ZeroPad2d` 相当。イシュー #2159・親 #2131）。
//!
//! `Var::pad`（イシュー #1756。`var.rs`）の薄いラッパー。空間 2 軸
//! （末尾 2 軸。`(H, W)`）だけを 0 値で拡張する用途に絞った層で、
//! 新規 `Op`／`BackendOps`／VJP／カーネルは追加しない（既存
//! `Op::Pad` の再利用）。
//!
//! **`Var::pad` との並び順の違いに注意**: `Var::pad` は先頭次元から
//! 順に `(before, after)` を並べる設計（PyTorch `F.pad` の「末尾軸
//! から逆順」とは異なる）。本層は PyTorch `nn.ZeroPad2d` と同じ
//! `(left, right, top, bottom)` 引数順を採用し、内部で `Var::pad` の
//! 並びへ組み替える（`build_pads`）。

use fandhe_ai_tensor_core::{BackendOps, ShapeError, Tensor, pad_out_shape};

use crate::error::AutodiffError;
use crate::grad::pad_with_fallback;
use crate::var::Var;

/// 入力の rank（3 または 4）から `Var::pad`／`Tensor::pad` へ渡す
/// `pads`（先頭次元から順の `(before, after)` リスト）を組み立てる
/// （`ZeroPad2d::forward`／`forward_host` で共有し、判定基準が
/// 食い違う迂回経路を作らない。`.claude/rules/security.md` A08）。
/// `padding` は `[left, right, top, bottom]`（PyTorch `nn.ZeroPad2d`
/// と同じ並び）。rank 3 は `(C, H, W)`・rank 4 は `(N, C, H, W)`
/// （PyTorch のドキュメント記載の両対応）で、末尾 2 軸だけに
/// `(top, bottom)`／`(left, right)` を適用し他軸は `(0, 0)`。
pub(crate) fn build_pads(
    rank: usize,
    padding: [usize; 4],
) -> Result<Vec<(usize, usize)>, AutodiffError> {
    if rank != 3 && rank != 4 {
        return Err(AutodiffError::Shape(ShapeError::RankMismatch {
            expected: 4,
            actual: rank,
        }));
    }
    let [left, right, top, bottom] = padding;
    let mut pads = vec![(0usize, 0usize); rank - 2];
    pads.push((top, bottom));
    pads.push((left, right));
    Ok(pads)
}

/// `(left, right, top, bottom)` の定数 0 埋めパディング層。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ZeroPad2d {
    /// `[left, right, top, bottom]`（PyTorch `nn.ZeroPad2d` と同じ
    /// 並び）。`usize` のため負パディング（クロップ）は表現できない
    /// （`Var::narrow` を使うこと。`docs/autodiff-spatial-layers-decision.md`
    /// §8 スコープ外）。
    padding: [usize; 4],
}

impl ZeroPad2d {
    /// `[left, right, top, bottom]` を個別に指定して構築する。
    pub fn new(padding: [usize; 4]) -> Self {
        Self { padding }
    }

    /// 全 4 辺へ同じ幅 `p` を適用する（PyTorch `nn.ZeroPad2d(p)` の
    /// スカラー引数相当）。
    pub fn uniform(p: usize) -> Self {
        Self {
            padding: [p, p, p, p],
        }
    }

    /// `[left, right, top, bottom]`。
    pub fn padding(&self) -> [usize; 4] {
        self.padding
    }

    /// `build_pads` で `pads` を組み立ててから [`Var::pad`] へ
    /// 委譲する（`value = 0.0` 固定）。
    pub fn forward<'t>(&self, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        let rank = input.shape().len();
        let pads = build_pads(rank, self.padding)?;
        input.pad(&pads, 0.0)
    }

    /// [`crate::nn::module::Module::forward_host`]（`ZeroPad2d` 実装）
    /// の本体。[`Self::forward`] と同じ `pads` 組み立てヘルパー
    /// （`build_pads`）を使い、`pad_out_shape` → `pad_with_fallback`
    /// を tape 不要経路で再現する。
    pub fn forward_host(
        &self,
        ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        let rank = input.shape().len();
        let pads = build_pads(rank, self.padding)?;
        let out_shape = pad_out_shape(input.shape(), &pads).map_err(AutodiffError::Shape)?;
        pad_with_fallback(ops, input, &pads, 0.0, &out_shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::dense_vec;
    use crate::tape::Tape;

    #[test]
    fn build_pads_rank4_places_top_bottom_left_right_on_trailing_axes() {
        let pads = build_pads(4, [1, 2, 3, 4]).unwrap();
        // [left=1, right=2, top=3, bottom=4] -> [(0,0), (0,0), (top,bottom), (left,right)]
        assert_eq!(pads, vec![(0, 0), (0, 0), (3, 4), (1, 2)]);
    }

    #[test]
    fn build_pads_rank3_omits_batch_axis() {
        let pads = build_pads(3, [1, 2, 3, 4]).unwrap();
        assert_eq!(pads, vec![(0, 0), (3, 4), (1, 2)]);
    }

    #[test]
    fn build_pads_rejects_rank2_and_rank5() {
        assert!(build_pads(2, [0, 0, 0, 0]).is_err());
        assert!(build_pads(5, [0, 0, 0, 0]).is_err());
    }

    #[test]
    fn forward_matches_hand_computed_padding() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]).unwrap());
        let layer = ZeroPad2d::new([1, 0, 0, 1]);
        let y = layer.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 1, 3, 3]);
        assert_eq!(
            dense_vec(&y.to_tensor()),
            vec![0.0, 1.0, 2.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn forward_gradient_flows_only_to_interior_region() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]).unwrap());
        let layer = ZeroPad2d::uniform(1);
        let y = layer.forward(&x).unwrap();
        let loss = y.sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let grad = grads
            .get(&x)
            .unwrap()
            .expect("x は backward で到達するはず");
        // 余白部分は upstream に寄与しないため中央領域のみ 1.0 が伝播する。
        assert_eq!(dense_vec(grad), vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn forward_rejects_rank2_and_rank5() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x2 = tape.var(&Tensor::new(vec![1.0, 2.0], &[1, 2]).unwrap());
        let layer = ZeroPad2d::uniform(1);
        assert!(layer.forward(&x2).is_err());

        let x5 = tape.var(&Tensor::new(vec![1.0; 16], &[1, 1, 2, 2, 4]).unwrap());
        assert!(layer.forward(&x5).is_err());
    }

    #[test]
    fn forward_host_matches_forward() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let host_ops = crate::test_support::test_ops();
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]).unwrap();
        let layer = ZeroPad2d::new([1, 0, 0, 1]);

        let via_host = layer.forward_host(host_ops.as_ref(), &x).unwrap();
        let via_tape = layer.forward(&tape.var(&x)).unwrap();

        assert_eq!(via_host.shape(), via_tape.shape());
        assert_eq!(dense_vec(&via_host), dense_vec(&via_tape.to_tensor()));
    }
}
