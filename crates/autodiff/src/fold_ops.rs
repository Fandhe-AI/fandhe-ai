//! `unfold`／`fold` の自由関数（`F.unfold`／`F.fold`、`nn.Unfold`／`nn.Fold` 相当。空間 2 軸・バッチ
//! 入力 `[N, C, H, W]`⇄`[N, C·kH·kW, L]` 固定。イシュー #2645・親 #2625「Phase 4」・ルート #2499）。
//!
//! 既存の `BackendOps::im2col`／`col2im`（Conv2d 用。CPU #1764・CUDA #1766・Metal #1768）を
//! `groups = 1` で再利用するだけで、新規 `BackendOps` メソッド・新規カーネルは追加しない。
//! K 軸は `(c, kh, kw)` row-major・L 軸は `(oh, ow)` row-major で PyTorch と同じ並び。
//!
//! **facade 公開状況**: 入口の `Var::unfold`／`Var::fold`（本関数への 1 行委譲メソッド）は #2851 で facade へ
//! 公開済み（モジュール自体は再エクスポートしない。公開形は `docs/autodiff-fold-unfold-decision.md` §12.1）。
//! 層化（`nn::Fold`／`nn::Unfold`・`Sequential::add_*`）は保留継続。残る保留は `crates/facade/src/lib.rs` の
//! `FoldUnfoldHoldDoctestGuard` と `crates/facade/tests/api_surface.rs` の否定ガードが機械的に固定し、
//! 公開済み側は同ファイルの正ガードが固定する。
//!
//! **経路**: ① `Conv2dParams::new`（0 の kernel／stride／dilation・`2·padding` オーバーフローを拒否）→
//! ② shape 検査（`tensor_core::fold::{unfold,fold}_out_shape`。実体化・tape 操作より前。エラー時に
//! 孤児ノードを残さない）→ ③ 入力の実体化 → ④ `im2col`／`col2im`（`BackendError::Unsupported` の
//! ときだけホスト参照実装へフォールバック。他のエラーは伝播）と reshape → ⑤ 戻り shape 再検証 →
//! ⑥ `Op::Unfold`／`Op::Fold` を `push_eager`。VJP は `grad.rs`（互いの随伴）。
//!
//! **数値契約**: `unfold` は純コピー（bit 一致）。`fold` は重なる窓の加算を `f64` アキュムレータで行い
//! 1 回だけ `f32` へ downcast する（`col2im` の契約。PyTorch は f32 累積のため、重なる窓では bit 一致
//! ではなく統一複合判定で検証する）。高階微分（`create_graph`）・activation checkpoint・低精度経路は
//! 対象外。
//!
//! **PyTorch との意図的な差分**: バッチなし入力（`[C,H,W]`／`[K,L]`）・`torch.Tensor.unfold`（次元方向の
//! スライディング窓。別演算）は対象外。引数順は crate 内の `conv2d` 系（`stride, padding, dilation`）に
//! 揃えており PyTorch の `(dilation, padding, stride)` とは異なる。

use fandhe_ai_tensor_core::{
    BackendError, Conv2dParams, ShapeError, Tensor,
    fold::{fold_out_shape, unfold_out_shape},
};

use crate::error::AutodiffError;
use crate::grad::{col2im_with_fallback, im2col_with_fallback};
use crate::tape::{Op, materialize_fallible};
use crate::var::Var;

/// スライディング窓の列展開（`F.unfold` 相当）。`input`: `[N, C, H, W]` →
/// `[N, C·kH·kW, L]`（`L` は窓の数）。
///
/// 検査順序: ①`Conv2dParams::new` → ②`unfold_out_shape`（rank・空間軸 0・窓数・確保サイズ）→
/// ③実体化 → ④`im2col` → ⑤戻り shape 再検証 → ⑥`push_eager`。
pub fn unfold<'t>(
    input: &Var<'t>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
) -> Result<Var<'t>, AutodiffError> {
    let params = Conv2dParams::new(kernel_size, stride, padding, dilation, 1)
        .map_err(AutodiffError::Backend)?;
    let in_shape = input.shape();
    let out_shape = unfold_out_shape(&in_shape, &params).map_err(AutodiffError::Shape)?;

    let input_val = materialize_one(input)?;
    let value_out = if out_shape.contains(&0) {
        Tensor::zeros(&out_shape).map_err(AutodiffError::Shape)?
    } else {
        let col_shape = [out_shape[0], 1, out_shape[1], out_shape[2]];
        im2col_with_fallback(
            input.tape().ops(),
            &input_val.contiguous(),
            &params,
            &col_shape,
        )?
        .reshape(&out_shape)
        .map_err(AutodiffError::Shape)?
    };
    check_shape(&value_out, &out_shape)?;
    let id = input.tape().push_eager(
        Op::Unfold {
            input: input.node_id(),
            params,
        },
        value_out,
    );
    Ok(Var::from_raw(input.tape(), id))
}

/// 列からの畳み戻し（`F.fold` 相当。重なる窓は加算）。`input`: `[N, C·kH·kW, L]` →
/// `[N, C, output_size[0], output_size[1]]`。
///
/// 検査順序: ①`Conv2dParams::new` → ②`fold_out_shape`（rank・`K` の整除性・`L` の一致・出力の確保前
/// サイズ検査）→ ③実体化 → ④`col2im` → ⑤戻り shape 再検証 → ⑥`push_eager`。
pub fn fold<'t>(
    input: &Var<'t>,
    output_size: [usize; 2],
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
) -> Result<Var<'t>, AutodiffError> {
    let params = Conv2dParams::new(kernel_size, stride, padding, dilation, 1)
        .map_err(AutodiffError::Backend)?;
    let in_shape = input.shape();
    let out_shape =
        fold_out_shape(&in_shape, output_size, &params).map_err(AutodiffError::Shape)?;

    let input_val = materialize_one(input)?;
    let value_out = if out_shape.contains(&0) || in_shape.contains(&0) {
        Tensor::zeros(&out_shape).map_err(AutodiffError::Shape)?
    } else {
        let d_col = input_val
            .contiguous()
            .reshape(&[in_shape[0], 1, in_shape[1], in_shape[2]])
            .map_err(AutodiffError::Shape)?;
        // `col2im_with_fallback` の第 3 引数は「fold の出力 shape」（`col2im` の "input" は仮想
        // conv2d の入力を指す。`conv2d_with_fallback` に慣れた読み手ほど誤読しやすい）。
        col2im_with_fallback(input.tape().ops(), &d_col, &out_shape, &params)?
    };
    check_shape(&value_out, &out_shape)?;
    let id = input.tape().push_eager(
        Op::Fold {
            input: input.node_id(),
            params,
        },
        value_out,
    );
    Ok(Var::from_raw(input.tape(), id))
}

fn materialize_one<'t>(x: &Var<'t>) -> Result<Tensor<f32>, AutodiffError> {
    let nodes = x.tape().nodes.borrow();
    Ok(materialize_fallible(&nodes, x.tape().ops(), x.node_id())?.clone())
}

fn check_shape(value: &Tensor<f32>, expected: &[usize]) -> Result<(), AutodiffError> {
    if value.shape() != expected {
        return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
            ShapeError::ShapeMismatch {
                lhs: value.shape().to_vec(),
                rhs: expected.to_vec(),
            },
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tape::Tape;

    fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
        Tensor::new(data, shape).expect("test fixture: shape 一致")
    }

    #[test]
    fn unfold_extracts_windows_in_pytorch_order() {
        // 入力 [1,1,2,3]（1..6）・k=[1,2]: 窓は (0,0..1),(0,1..2),(1,0..1),(1,1..2) の 4 つ。
        let tape = Tape::new();
        let x = tape.var(&t(vec![1., 2., 3., 4., 5., 6.], &[1, 1, 2, 3]));
        let y = unfold(&x, [1, 2], [1, 1], [0, 0], [1, 1]).unwrap();
        let out = y.to_tensor();
        assert_eq!(out.shape(), &[1, 2, 4]);
        // K 軸 = (kh=0, kw=0),(kh=0, kw=1)。L 軸 = (oh, ow) row-major。
        assert_eq!(
            out.host_slice().into_owned(),
            vec![1., 2., 4., 5., 2., 3., 5., 6.]
        );
    }

    #[test]
    fn fold_sums_overlapping_windows() {
        // k=[1,2]・stride 1・出力 [1,3]: 窓 2 つが中央要素で重なる。
        let tape = Tape::new();
        let x = tape.var(&t(vec![1., 2., 10., 20.], &[1, 2, 2]));
        let y = fold(&x, [1, 3], [1, 2], [1, 1], [0, 0], [1, 1]).unwrap();
        let out = y.to_tensor();
        assert_eq!(out.shape(), &[1, 1, 1, 3]);
        // 窓0 = (k0=1, k1=10) → 位置 0,1、窓1 = (2, 20) → 位置 1,2。
        assert_eq!(out.host_slice().into_owned(), vec![1., 10. + 2., 20.]);
    }

    #[test]
    fn rejects_invalid_arguments_without_orphan_nodes() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![0.0; 9], &[1, 1, 3, 3]));
        let c = tape.var(&t(vec![0.0; 8], &[1, 4, 2]));
        let before = tape.len();
        assert!(unfold(&x, [0, 1], [1, 1], [0, 0], [1, 1]).is_err());
        assert!(unfold(&x, [2, 2], [0, 1], [0, 0], [1, 1]).is_err());
        assert!(unfold(&x, [2, 2], [1, 1], [0, 0], [0, 1]).is_err());
        assert!(unfold(&x, [4, 4], [1, 1], [0, 0], [1, 1]).is_err());
        // L 不一致（3×3 を k=2 で畳むと L=4 だが入力は 2）／K 非整除。
        assert!(fold(&c, [3, 3], [2, 2], [1, 1], [0, 0], [1, 1]).is_err());
        assert!(fold(&c, [3, 3], [3, 1], [1, 1], [0, 0], [1, 1]).is_err());
        assert!(fold(&c, [0, 3], [2, 2], [1, 1], [0, 0], [1, 1]).is_err());
        assert_eq!(tape.len(), before, "孤児ノードを残さない");
    }

    #[test]
    fn empty_batch_yields_empty_output() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![], &[0, 1, 3, 3]));
        let y = unfold(&x, [2, 2], [1, 1], [0, 0], [1, 1]).unwrap();
        assert_eq!(y.to_tensor().shape(), &[0, 4, 4]);
        let c = tape.var(&t(vec![], &[0, 4, 4]));
        let z = fold(&c, [3, 3], [2, 2], [1, 1], [0, 0], [1, 1]).unwrap();
        assert_eq!(z.to_tensor().shape(), &[0, 1, 3, 3]);
    }
}
