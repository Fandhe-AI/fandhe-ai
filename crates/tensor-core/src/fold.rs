//! Fold／Unfold（`F.fold`／`F.unfold`、`nn.Fold`／`nn.Unfold` 相当。空間 2 軸・バッチ入力固定）の
//! shape 検査（イシュー #2645・親 #2625「Phase 4」・ルート #2499）。
//!
//! 役割: `autodiff::fold_ops::{unfold, fold}` の入口が、実体化・確保・tape 操作より前に
//! 「rank・`groups == 1`・`K` の整除性・`L` の一致・出力の確保サイズ」を一括検査するための純関数を
//! 提供する。計算そのものは既存の `BackendOps::im2col`／`col2im`（Conv2d 用。CPU #1764・CUDA #1766・
//! Metal #1768）の再利用で、本モジュールは shape 計算のみを担う。`ops_shape.rs` へ追記せず自己完結
//! モジュールにしたのは、兄弟イシューとの競合回避のため（`conv_transpose3d.rs` と同じ事情）。
//!
//! `fold` の `output_size` は入力テンソルの大きさと独立に呼び出し側が指定できるため、ホスト
//! フォールバックが無検査で出力バッファを確保する前に、出力要素数・バイト数を
//! `checked_numel_for` で検査する（`checked_numel_for` は `pub(crate)` のため autodiff 側では
//! 書けず、本モジュールが担う）。
//!
//! K 軸は `(c, kh, kw)` row-major・L 軸は `(oh, ow)` row-major（PyTorch の `unfold` と同じ並び）。

use crate::backend_ops::Conv2dParams;
use crate::error::ShapeError;
use crate::ops_shape::im2col_out_shape;
use crate::tensor::checked_numel_for;

/// `groups == 1` を要求する（Fold／Unfold に groups の概念はない）。
fn require_single_group(params: &Conv2dParams) -> Result<(), ShapeError> {
    if params.groups() != 1 {
        return Err(ShapeError::ShapeMismatch {
            lhs: vec![params.groups()],
            rhs: vec![1],
        });
    }
    Ok(())
}

/// Unfold の出力 shape `[N, C·kH·kW, L]` を検査・計算する（入力 `[N, C, H, W]`）。
///
/// 検査順序: `groups == 1` → 入力の確保前検査 → `im2col_out_shape`（rank 4・空間軸 0 拒否・窓数 `L` の算出・
/// 出力要素数の確保前検査）。
pub fn unfold_out_shape(
    input_shape: &[usize],
    params: &Conv2dParams,
) -> Result<Vec<usize>, ShapeError> {
    require_single_group(params)?;
    // 入力側の確保前検査。巨大な broadcast view は出力 `L` が小さくても `contiguous()` で全体が
    // 実体化されるため、出力サイズとは独立に入力要素数・バイト数を拒否する。
    checked_numel_for::<f32>(input_shape)?;
    let col = im2col_out_shape(input_shape, params)?;
    Ok(vec![col[0], col[2], col[3]])
}

/// Fold の出力 shape `[N, C, H, W]` を検査・計算する（入力 `[N, C·kH·kW, L]`）。
///
/// 検査順序: rank 3 → `groups == 1` → 入力の確保前検査 → `K % (kH·kW) == 0`（`C = K / (kH·kW)`）→ `output_size` の
/// 0 拒否 → 出力要素数・バイト数の確保前検査 → `im2col_out_shape` で窓数 `P` を導出 → `P == L`。
pub fn fold_out_shape(
    input_shape: &[usize],
    output_size: [usize; 2],
    params: &Conv2dParams,
) -> Result<Vec<usize>, ShapeError> {
    if input_shape.len() != 3 {
        return Err(ShapeError::RankMismatch {
            expected: 3,
            actual: input_shape.len(),
        });
    }
    require_single_group(params)?;
    // 入力側の確保前検査（`unfold_out_shape` と同じ理由。`contiguous()` による全体実体化を防ぐ）。
    checked_numel_for::<f32>(input_shape)?;
    let (n, k, l) = (input_shape[0], input_shape[1], input_shape[2]);
    let [kh, kw] = params.kernel_size();
    let kk = kh.checked_mul(kw).ok_or(ShapeError::ElementCountOverflow)?;
    if kk == 0 || !k.is_multiple_of(kk) {
        return Err(ShapeError::ShapeMismatch {
            lhs: vec![k],
            rhs: vec![kk],
        });
    }
    let c = k / kk;
    let [h, w] = output_size;
    if h == 0 || w == 0 {
        return Err(ShapeError::ShapeMismatch {
            lhs: vec![h, w],
            rhs: vec![1, 1],
        });
    }
    let out_shape = vec![n, c, h, w];
    // 出力側の確保前検査（`im2col_out_shape` は col 側の要素数しか見ないため先に行う）。
    checked_numel_for::<f32>(&out_shape)?;
    let col = im2col_out_shape(&out_shape, params)?;
    if col[3] != l {
        return Err(ShapeError::ShapeMismatch {
            lhs: vec![l],
            rhs: vec![col[3]],
        });
    }
    Ok(out_shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(k: [usize; 2], s: [usize; 2], p: [usize; 2], d: [usize; 2]) -> Conv2dParams {
        Conv2dParams::new(k, s, p, d, 1).unwrap()
    }

    #[test]
    fn unfold_matches_pytorch_l_formula() {
        // 入力 5×5・k=3・s=1・p=0 → L = 3·3 = 9、K = C·9。
        let p = params([3, 3], [1, 1], [0, 0], [1, 1]);
        assert_eq!(unfold_out_shape(&[2, 3, 5, 5], &p).unwrap(), vec![2, 27, 9]);
    }

    #[test]
    fn anisotropic_and_dilation() {
        // H: (7+2·1-2·(2-1)-1)/2+1 = 4、W: (6+0-1·(3-1)-1)/1+1 = 4 → L = 16。
        let p = params([2, 3], [2, 1], [1, 0], [2, 1]);
        assert_eq!(
            unfold_out_shape(&[1, 2, 7, 6], &p).unwrap(),
            vec![1, 12, 16]
        );
        assert_eq!(
            fold_out_shape(&[1, 12, 16], [7, 6], &p).unwrap(),
            vec![1, 2, 7, 6]
        );
    }

    #[test]
    fn empty_batch_is_accepted() {
        let p = params([2, 2], [1, 1], [0, 0], [1, 1]);
        assert_eq!(unfold_out_shape(&[0, 1, 3, 3], &p).unwrap(), vec![0, 4, 4]);
        assert_eq!(
            fold_out_shape(&[0, 4, 4], [3, 3], &p).unwrap(),
            vec![0, 1, 3, 3]
        );
    }

    #[test]
    fn rejects_rank_mismatch() {
        let p = params([2, 2], [1, 1], [0, 0], [1, 1]);
        assert!(unfold_out_shape(&[1, 3, 3], &p).is_err());
        assert!(fold_out_shape(&[1, 1, 4, 4], [3, 3], &p).is_err());
        assert!(fold_out_shape(&[4, 4], [3, 3], &p).is_err());
    }

    #[test]
    fn rejects_non_divisible_k() {
        let p = params([2, 2], [1, 1], [0, 0], [1, 1]);
        assert!(fold_out_shape(&[1, 5, 4], [3, 3], &p).is_err());
    }

    #[test]
    fn rejects_l_mismatch() {
        let p = params([2, 2], [1, 1], [0, 0], [1, 1]);
        assert!(fold_out_shape(&[1, 4, 5], [3, 3], &p).is_err());
    }

    #[test]
    fn rejects_zero_spatial_axes() {
        let p = params([2, 2], [1, 1], [0, 0], [1, 1]);
        assert!(unfold_out_shape(&[1, 1, 0, 3], &p).is_err());
        assert!(fold_out_shape(&[1, 4, 1], [0, 3], &p).is_err());
    }

    #[test]
    fn rejects_kernel_larger_than_padded_input() {
        let p = params([5, 5], [1, 1], [0, 0], [1, 1]);
        assert!(unfold_out_shape(&[1, 1, 3, 3], &p).is_err());
        assert!(fold_out_shape(&[1, 25, 1], [3, 3], &p).is_err());
    }

    #[test]
    fn rejects_groups_other_than_one() {
        let p = Conv2dParams::new([2, 2], [1, 1], [0, 0], [1, 1], 2).unwrap();
        assert!(unfold_out_shape(&[1, 2, 3, 3], &p).is_err());
        assert!(fold_out_shape(&[1, 4, 4], [3, 3], &p).is_err());
    }

    #[test]
    fn rejects_huge_output_size_before_allocation() {
        // kernel 1・stride 2^31 → L = 1 だが出力は 2^31·2^31 要素（バイト数が usize を超える）。
        // 確保前に `checked_numel_for` が拒否する（上限は isize::MAX バイト。実メモリ超過までは見ない）。
        let big = 1usize << 31;
        let p = params([1, 1], [big, big], [0, 0], [1, 1]);
        assert!(fold_out_shape(&[1, 1, 1], [big, big], &p).is_err());
        assert!(fold_out_shape(&[1, 1, 1], [usize::MAX, usize::MAX], &p).is_err());
    }
}
