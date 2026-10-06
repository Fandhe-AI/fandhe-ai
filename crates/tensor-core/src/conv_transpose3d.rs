//! ConvTranspose3d（`F.conv_transpose3d`／`nn.ConvTranspose3d` 相当。NCDHW 固定）の出力 shape 検査
//! （イシュー #2644・親 #2625「Phase 4」・ルート #2499）。
//!
//! 役割: `autodiff::conv_transpose3d_ops::conv_transpose3d` の入口が、実体化・確保・tape 操作より
//! 前に「rank・チャンネル整合・空間軸 0・各軸の出力長・確保サイズ」を一括検査するための純関数を
//! 提供する（`ops_shape::conv_transpose2d_out_shape` の空間 3 軸一般化。既存ファイルへ
//! 追記せず自己完結モジュールにしたのは、兄弟イシューとの `ops_shape.rs` 競合回避のため）。
//! 実際の計算（GEMM＋`col2im3d`）は autodiff 側の合成で、本モジュールは shape 計算のみを担う。
//!
//! **PyTorch との意図的な差分**: `output_padding >= stride`（各軸）は拒否する。PyTorch は
//! `output_padding < max(stride, dilation)` まで許容するが、`col2im3d` の P 軸契約
//! （`conv_out_len(Dout)·conv_out_len(Hout)·conv_out_len(Wout)` と forward の im2col3d 出力 P 軸
//! `D·H·W` の一致）を `op < stride` が保証するため（`docs/conv-ops-design.md` §15・
//! `docs/autodiff-conv-transpose3d-max-unpool-decision.md` §5）。

use crate::backend_ops::Conv3dParams;
use crate::error::ShapeError;
use crate::ops_shape::conv_transpose_out_len;
use crate::tensor::checked_numel_for;

/// ConvTranspose3d の出力 shape `[N, Cout, Dout, Hout, Wout]` を検査・計算する。
///
/// `input_shape: [N, Cin, D, H, W]`・`weight_shape: [Cin, Cout/groups, kD, kH, kW]`
/// （PyTorch `nn.ConvTranspose3d.weight` と同じレイアウト）。
///
/// 検査順序（`conv_transpose2d_out_shape` と同じ）: rank（input／weight とも 5）→
/// `Cin % groups == 0` → `weight_shape[0] == Cin` → `Cout_g >= 1` → `Cout = Cout_g·groups`
/// （checked）→ 空間軸 `D`／`H`／`W == 0` 拒否（`N == 0` は受理）→ 各軸 `conv_transpose_out_len`
/// （`op < stride` ゲート込み）→ 出力要素数・バイト数の確保前検査。
pub fn conv_transpose3d_out_shape(
    input_shape: &[usize],
    weight_shape: &[usize],
    params: &Conv3dParams,
    output_padding: [usize; 3],
) -> Result<Vec<usize>, ShapeError> {
    if input_shape.len() != 5 {
        return Err(ShapeError::RankMismatch {
            expected: 5,
            actual: input_shape.len(),
        });
    }
    if weight_shape.len() != 5 {
        return Err(ShapeError::RankMismatch {
            expected: 5,
            actual: weight_shape.len(),
        });
    }
    let (n, cin) = (input_shape[0], input_shape[1]);
    let (weight_cin, cout_g) = (weight_shape[0], weight_shape[1]);
    let groups = params.groups();
    if !cin.is_multiple_of(groups) {
        return Err(ShapeError::ShapeMismatch {
            lhs: vec![cin],
            rhs: vec![groups],
        });
    }
    if weight_cin != cin {
        return Err(ShapeError::ShapeMismatch {
            lhs: vec![weight_cin],
            rhs: vec![cin],
        });
    }
    if cout_g == 0 {
        return Err(ShapeError::ShapeMismatch {
            lhs: vec![cout_g],
            rhs: vec![1],
        });
    }
    let cout = cout_g
        .checked_mul(groups)
        .ok_or(ShapeError::ElementCountOverflow)?;
    if input_shape[2..].contains(&0) {
        return Err(ShapeError::ShapeMismatch {
            lhs: input_shape[2..].to_vec(),
            rhs: vec![1, 1, 1],
        });
    }
    let stride = params.stride();
    let padding = params.padding();
    let dilation = params.dilation();
    let mut out_shape = vec![n, cout, 0, 0, 0];
    for a in 0..3 {
        out_shape[2 + a] = conv_transpose_out_len(
            input_shape[2 + a],
            weight_shape[2 + a],
            stride[a],
            padding[a],
            dilation[a],
            output_padding[a],
        )?;
    }
    checked_numel_for::<f32>(&out_shape)?;
    Ok(out_shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(
        k: [usize; 3],
        s: [usize; 3],
        p: [usize; 3],
        d: [usize; 3],
        g: usize,
    ) -> Conv3dParams {
        Conv3dParams::new(k, s, p, d, g).unwrap()
    }

    #[test]
    fn matches_pytorch_formula_examples() {
        // 入力 3×3×3・k=3・s=2・p=1・op=1: (3-1)*2 - 2 + 2 + 1 + 1 = 6（PyTorch 2.14.0 実測と一致）。
        let p = params([3, 3, 3], [2, 2, 2], [1, 1, 1], [1, 1, 1], 1);
        let out =
            conv_transpose3d_out_shape(&[1, 2, 3, 3, 3], &[2, 3, 3, 3, 3], &p, [1, 1, 1]).unwrap();
        assert_eq!(out, vec![1, 3, 6, 6, 6]);
    }

    #[test]
    fn anisotropic_axes_are_independent() {
        let p = params([2, 3, 2], [1, 2, 2], [0, 1, 0], [1, 1, 2], 1);
        let out =
            conv_transpose3d_out_shape(&[2, 2, 2, 3, 4], &[2, 2, 2, 3, 2], &p, [0, 1, 1]).unwrap();
        assert_eq!(out, vec![2, 2, 3, 6, 10]);
    }

    #[test]
    fn groups_scale_output_channels() {
        let p = params([2, 2, 2], [1, 1, 1], [0, 0, 0], [1, 1, 1], 2);
        let out =
            conv_transpose3d_out_shape(&[1, 4, 2, 3, 3], &[4, 3, 2, 2, 2], &p, [0, 0, 0]).unwrap();
        assert_eq!(out, vec![1, 6, 3, 4, 4]);
    }

    #[test]
    fn rejects_output_padding_ge_stride_on_each_axis() {
        let p = params([2, 2, 2], [2, 2, 2], [0, 0, 0], [1, 1, 1], 1);
        for op in [[2, 0, 0], [0, 2, 0], [0, 0, 2], [3, 3, 3]] {
            let err =
                conv_transpose3d_out_shape(&[1, 1, 2, 2, 2], &[1, 1, 2, 2, 2], &p, op).unwrap_err();
            assert!(matches!(err, ShapeError::ShapeMismatch { .. }), "{op:?}");
        }
    }

    #[test]
    fn rejects_rank_channel_and_group_mismatch() {
        let p = params([2, 2, 2], [1, 1, 1], [0, 0, 0], [1, 1, 1], 1);
        assert!(matches!(
            conv_transpose3d_out_shape(&[1, 1, 2, 2], &[1, 1, 2, 2, 2], &p, [0; 3]),
            Err(ShapeError::RankMismatch {
                expected: 5,
                actual: 4
            })
        ));
        assert!(matches!(
            conv_transpose3d_out_shape(&[1, 1, 2, 2, 2], &[1, 1, 2, 2], &p, [0; 3]),
            Err(ShapeError::RankMismatch {
                expected: 5,
                actual: 4
            })
        ));
        // weight の Cin 軸が input の Cin と一致しない。
        assert!(matches!(
            conv_transpose3d_out_shape(&[1, 2, 2, 2, 2], &[3, 1, 2, 2, 2], &p, [0; 3]),
            Err(ShapeError::ShapeMismatch { .. })
        ));
        // Cin が groups で割り切れない。
        let pg = params([2, 2, 2], [1, 1, 1], [0, 0, 0], [1, 1, 1], 2);
        assert!(matches!(
            conv_transpose3d_out_shape(&[1, 3, 2, 2, 2], &[3, 1, 2, 2, 2], &pg, [0; 3]),
            Err(ShapeError::ShapeMismatch { .. })
        ));
        // Cout_g == 0。
        assert!(matches!(
            conv_transpose3d_out_shape(&[1, 1, 2, 2, 2], &[1, 0, 2, 2, 2], &p, [0; 3]),
            Err(ShapeError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn rejects_zero_spatial_axis_but_accepts_zero_batch() {
        let p = params([2, 2, 2], [1, 1, 1], [0, 0, 0], [1, 1, 1], 1);
        for shape in [[1, 1, 0, 2, 2], [1, 1, 2, 0, 2], [1, 1, 2, 2, 0]] {
            assert!(matches!(
                conv_transpose3d_out_shape(&shape, &[1, 1, 2, 2, 2], &p, [0; 3]),
                Err(ShapeError::ShapeMismatch { .. })
            ));
        }
        let out =
            conv_transpose3d_out_shape(&[0, 1, 2, 2, 2], &[1, 1, 2, 2, 2], &p, [0; 3]).unwrap();
        assert_eq!(out, vec![0, 1, 3, 3, 3]);
    }

    #[test]
    fn rejects_non_positive_output_length_from_large_padding() {
        let p = params([1, 1, 1], [1, 1, 1], [5, 0, 0], [1, 1, 1], 1);
        assert!(matches!(
            conv_transpose3d_out_shape(&[1, 1, 2, 2, 2], &[1, 1, 1, 1, 1], &p, [0; 3]),
            Err(ShapeError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn rejects_overflowing_shapes_before_allocation() {
        let p = params([2, 2, 2], [1, 1, 1], [0, 0, 0], [1, 1, 1], 1);
        // 出力要素数の積が usize を超える（確保前に型付きエラーで拒否）。
        let big = usize::MAX / 2;
        assert!(matches!(
            conv_transpose3d_out_shape(&[big, 1, 2, 2, 2], &[1, 2, 2, 2, 2], &p, [0; 3]),
            Err(ShapeError::ElementCountOverflow)
        ));
        // Cout_g·groups のオーバーフロー。
        let pg = params([1, 1, 1], [1, 1, 1], [0, 0, 0], [1, 1, 1], 2);
        assert!(matches!(
            conv_transpose3d_out_shape(&[1, 2, 1, 1, 1], &[2, usize::MAX, 1, 1, 1], &pg, [0; 3]),
            Err(ShapeError::ElementCountOverflow)
        ));
        // 空間長のオーバーフロー（(in-1)*s）。
        let ps = params([1, 1, 1], [usize::MAX, 1, 1], [0, 0, 0], [1, 1, 1], 1);
        assert!(matches!(
            conv_transpose3d_out_shape(&[1, 1, 3, 1, 1], &[1, 1, 1, 1, 1], &ps, [0; 3]),
            Err(ShapeError::ElementCountOverflow)
        ));
    }
}
