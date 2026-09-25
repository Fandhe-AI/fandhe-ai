//! Conv3d（`torch.nn.functional.conv3d`／`nn.Conv3d` 相当の
//! cross-correlation。NCDHW 固定。イシュー #2158・設計
//! `docs/conv-ops-design.md` §16）。`Var::conv2d` の空間 3 軸一般化
//! （im2col3d＋GEMM の段階的合成）。
//!
//! **facade 非公開（意図的）**: [`crate::reduce_ops`] モジュール doc
//! と同じ理由・同じ判断枠組みによる。`Var` は facade（`fandhe_ai`
//! クレート）から直接再エクスポートされるため、`Var` への inherent
//! メソッド追加は即座に facade 公開面へ出てしまう。イシュー #2158 は
//! `Var::conv3d`（委譲メソッド）・`compat::Sequential::add_conv3d`
//! （facade 公開）を承認事項として明示するが、本 PR 時点で承認の記録
//! が無いため、承認が取れるまでは自由関数として `Var` の外に置き
//! 到達不能にする（`docs/conv-ops-design.md` §16「承認事項」）。承認後
//! は `Var::conv3d` の薄い委譲メソッドと `compat::Sequential::
//! add_conv3d` を追加し、facade 側の保留ガード
//! （`crates/facade/src/lib.rs::VarConv3dHoldDoctestGuard`）を撤去する。
//!
//! **新規 `Op`**: `crate::tape::Op::Conv3d`。`Op::Conv2d` と同じ
//! 段階的合成（`ops.conv3d`〈override フック〉→ im2col3d →
//! `gemm_batched`〈常にバックエンド〉→ `add`〈bias〉）で、`col`
//! （im2col3d の中間結果）は `Op::Conv3d` に保持せず backward で
//! 再計算する。低精度 conv3d はスコープ外のため `compute_dtype`
//! フィールドを持たない（`Op::Conv2d` との差異）。
//!
//! **数値契約**: `im2col3d` は算術を含まない純粋なコピー演算のため
//! bit 完全一致、`col2im3d`（backward の d_input）は `f64` アキュム
//! レータへの逐次加算契約（`.claude/rules/coding-rust.md` の勾配の
//! 長軸縮約規約）、GEMM は forward が `gemm_batched`・VJP が
//! `gemm_batched_fp32_strict`（`Op::Conv2d` と同一の数値契約）。

use fandhe_ai_tensor_core::{Conv3dParams, ShapeError, conv3d_out_shape};

use crate::error::AutodiffError;
use crate::grad::conv3d_with_fallback;
use crate::tape::{Op, materialize_fallible};
use crate::var::Var;

/// 3 次元畳み込み（`torch.nn.functional.conv3d` 相当の
/// cross-correlation。NCDHW 固定。イシュー #2158）。`input`:
/// `[N, Cin, D, H, W]`・`weight`: `[Cout, Cin/groups, kD, kH, kW]`
/// （PyTorch 準拠）・`bias`: `Some` なら `[Cout]`。
///
/// 検査順序（`Var::conv2d` と同一。設計 doc §16）: ①同一 tape の検査
/// → ②`weight.shape()` が rank 5 か → ③`weight.shape()[2..5]` から
/// `kernel_size` を導出し [`Conv3dParams::new`]（`stride`／
/// `dilation`／`groups` の 0・`2·padding` オーバーフローを拒否）→
/// ④[`conv3d_out_shape`] で `out_shape` を確定（rank・チャンネル整合・
/// 空間軸 `D`／`H`／`W = 0` 拒否・負分子拒否ゲート）→ ⑤`bias` の
/// shape 検査 → ⑥`input`／`weight`／`bias` を層 1 で実体化（`RefCell`
/// 借用を閉じてから push）→ ⑦`conv3d_with_fallback` → ⑧戻り shape
/// 検証（`.claude/rules/security.md` A08）→ ⑨`push_eager`（非融合・
/// 常実体化。`col` は `Op::Conv3d` 自身に保持しない）。
#[allow(clippy::too_many_arguments)] // `Var::conv2d` と同じ理由（PyTorch nn.Conv3d の全引数を受理する必要があるため）。
pub fn conv3d<'t>(
    input: &Var<'t>,
    weight: &Var<'t>,
    bias: Option<&Var<'t>>,
    stride: [usize; 3],
    padding: [usize; 3],
    dilation: [usize; 3],
    groups: usize,
) -> Result<Var<'t>, AutodiffError> {
    input.check_same_tape(weight)?;
    if let Some(b) = bias {
        input.check_same_tape(b)?;
    }

    let weight_shape = weight.shape();
    if weight_shape.len() != 5 {
        return Err(AutodiffError::Shape(ShapeError::RankMismatch {
            expected: 5,
            actual: weight_shape.len(),
        }));
    }
    let kernel_size = [weight_shape[2], weight_shape[3], weight_shape[4]];
    let params = Conv3dParams::new(kernel_size, stride, padding, dilation, groups)
        .map_err(AutodiffError::Backend)?;

    let in_shape = input.shape();
    let out_shape =
        conv3d_out_shape(&in_shape, &weight_shape, &params).map_err(AutodiffError::Shape)?;
    if let Some(b) = bias {
        let bias_shape = b.shape();
        let cout = weight_shape[0];
        if bias_shape != [cout] {
            return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                lhs: bias_shape,
                rhs: vec![cout],
            }));
        }
    }

    let input_val = {
        let nodes = input.tape().nodes.borrow();
        materialize_fallible(&nodes, input.tape().ops(), input.node_id())?.clone()
    };
    let weight_val = {
        let nodes = weight.tape().nodes.borrow();
        materialize_fallible(&nodes, weight.tape().ops(), weight.node_id())?.clone()
    };
    let bias_val = match bias {
        Some(b) => {
            let nodes = b.tape().nodes.borrow();
            Some(materialize_fallible(&nodes, b.tape().ops(), b.node_id())?.clone())
        }
        None => None,
    };

    let value_out = conv3d_with_fallback(
        input.tape().ops(),
        &input_val,
        &weight_val,
        bias_val.as_ref(),
        &params,
        &out_shape,
    )?;
    if value_out.shape() != out_shape {
        return Err(AutodiffError::Backend(
            fandhe_ai_tensor_core::BackendError::ShapeMismatch(ShapeError::ShapeMismatch {
                lhs: value_out.shape().to_vec(),
                rhs: out_shape,
            }),
        ));
    }
    let id = input.tape().push_eager(
        Op::Conv3d {
            input: input.node_id(),
            weight: weight.node_id(),
            bias: bias.map(|b| b.node_id()),
            params,
        },
        value_out,
    );
    Ok(Var::from_raw(input.tape(), id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tape::Tape;
    use fandhe_ai_tensor_core::Tensor;

    fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
        Tensor::new(data, shape).expect("test fixture: shape 一致")
    }

    #[test]
    fn conv3d_basic_forward_matches_direct_reference() {
        let tape = Tape::new();
        // input [1,1,2,2,2], weight [1,1,2,2,2] (全窓 1 個の kernel=input)
        let x = tape.var(&t(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[1, 1, 2, 2, 2],
        ));
        let w = tape.var(&t(vec![1.0; 8], &[1, 1, 2, 2, 2]));
        let y = conv3d(&x, &w, None, [1, 1, 1], [0, 0, 0], [1, 1, 1], 1).unwrap();
        assert_eq!(y.to_tensor().shape(), &[1, 1, 1, 1, 1]);
        let expected: f32 = (1..=8).map(|v| v as f32).sum();
        assert_eq!(y.to_tensor().host_slice()[0], expected);
    }

    #[test]
    fn conv3d_rejects_weight_rank_mismatch() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![0.0; 8], &[1, 1, 2, 2, 2]));
        let w = tape.var(&t(vec![0.0; 4], &[1, 1, 2, 2]));
        let err = conv3d(&x, &w, None, [1, 1, 1], [0, 0, 0], [1, 1, 1], 1);
        assert!(matches!(
            err,
            Err(AutodiffError::Shape(ShapeError::RankMismatch { .. }))
        ));
    }

    #[test]
    fn conv3d_rejects_bias_shape_mismatch() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![0.0; 8], &[1, 1, 2, 2, 2]));
        let w = tape.var(&t(vec![0.0; 16], &[2, 1, 2, 2, 2]));
        let bias = tape.var(&t(vec![0.0; 3], &[3]));
        let err = conv3d(&x, &w, Some(&bias), [1, 1, 1], [0, 0, 0], [1, 1, 1], 1);
        assert!(matches!(
            err,
            Err(AutodiffError::Shape(ShapeError::ShapeMismatch { .. }))
        ));
    }

    #[test]
    fn conv3d_zero_kernel_rejected_via_params_new() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![0.0; 8], &[1, 1, 2, 2, 2]));
        let w = tape.var(&t(vec![], &[1, 1, 0, 2, 2]));
        let err = conv3d(&x, &w, None, [1, 1, 1], [0, 0, 0], [1, 1, 1], 1);
        assert!(err.is_err());
    }

    #[test]
    fn conv3d_kernel_one_matches_pointwise_scale() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 1, 2, 2]));
        let w = tape.var(&t(vec![2.0], &[1, 1, 1, 1, 1]));
        let y = conv3d(&x, &w, None, [1, 1, 1], [0, 0, 0], [1, 1, 1], 1).unwrap();
        assert_eq!(
            y.to_tensor().host_slice().into_owned(),
            vec![2.0, 4.0, 6.0, 8.0]
        );
    }
}
