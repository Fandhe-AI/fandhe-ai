//! `conv_transpose3d` の自由関数（`F.conv_transpose3d`／`nn.ConvTranspose3d` 相当。NCDHW 固定。
//! イシュー #2644・親 #2625「Phase 4」・ルート #2499）。`Var::conv_transpose2d`（#2067）の空間
//! 3 軸一般化で、新規 `BackendOps` メソッド・新規カーネルは追加しない（既存の `gemm_batched`＋
//! `col2im3d` フックの合成。設計 `docs/conv-ops-design.md` §15・§16）。
//!
//! **facade 公開状況**: `Var::conv_transpose3d`（本モジュールの自由関数への 1 行委譲）は #2850 で
//! 公開済み（公開形は `docs/autodiff-conv-transpose3d-max-unpool-decision.md` §12.1）。モジュール自体は
//! 再エクスポートしない。層化（`nn::ConvTranspose3d`・`Sequential::add_*`）は保留継続で、
//! `crates/facade/src/lib.rs` の `ConvTranspose3dMaxUnpoolHoldDoctestGuard` と
//! `crates/facade/tests/api_surface.rs` の否定ガードが機械的に固定する。
//!
//! **経路**: ① 同一 tape・`Conv3dParams`・`output_padding < stride`・出力 shape・bias shape の検査
//! （実体化・tape 操作より前。エラー時に孤児ノードを残さない）→ ② 入力の実体化 → ③ 段階的合成
//! （`w_matᵀ` と入力の `gemm_batched`〈常にバックエンド。forward は CUDA TF32 opt-in に追従〉→
//! `col2im3d`〈`Unsupported` のときだけホストへフォールバック。他のエラーは伝播〉→ bias は
//! `[1, Cout, 1, 1, 1]` へ reshape して `add`）→ ④ 戻り shape 再検証 → ⑤ `Op::ConvTranspose3d` を
//! `push_eager`。VJP は `grad.rs`（`col` を保持せず `im2col3d` を再計算する）。
//!
//! **数値契約**: GEMM は forward が `gemm_batched`・VJP が `gemm_batched_fp32_strict`（FMA 契約）、
//! `col2im3d` は `f64` アキュムレータ契約、bias 勾配は `f64` 逐次和・1 回 downcast
//! （`.claude/rules/coding-rust.md`）。PyTorch とは総和順が異なるため bit 一致ではなく統一複合判定で
//! 検証する。高階微分（`create_graph`）・activation checkpoint・低精度経路は対象外。
//!
//! **PyTorch との意図的な差分**: `output_padding >= stride`（各軸）は拒否する（`col2im3d` の P 軸契約。
//! PyTorch は `max(stride, dilation)` 未満まで許容）。`N = 0` は受理して空出力を返す。

use fandhe_ai_tensor_core::{
    BackendError, BackendOps, Conv3dParams, ShapeError, Tensor,
    conv_transpose3d::conv_transpose3d_out_shape,
};

use crate::error::AutodiffError;
use crate::grad::col2im3d_with_fallback;
use crate::tape::{Op, materialize_fallible};
use crate::var::Var;

/// 3 次元転置畳み込み（`F.conv_transpose3d` 相当。NCDHW 固定）。`input`: `[N, Cin, D, H, W]`・
/// `weight`: `[Cin, Cout/groups, kD, kH, kW]`（PyTorch `nn.ConvTranspose3d.weight` と同じレイアウト。
/// `conv3d` の `[Cout, Cin/groups, ...]` と先頭 2 軸が逆）・`bias`: `Some` なら `[Cout]`。
///
/// 検査順序（`Var::conv_transpose2d` と同一）: ①同一 tape → ②`weight` rank 5 →
/// ③`Conv3dParams::new`（`stride`／`dilation`／`groups` の 0・`2·padding` オーバーフロー拒否）→
/// ④`output_padding < stride`（各軸）→ ⑤`conv_transpose3d_out_shape`（rank・チャンネル整合・空間軸 0・
/// 確保サイズ）→ ⑥bias shape → ⑦実体化 → ⑧合成 → ⑨戻り shape 再検証 → ⑩`push_eager`。
#[allow(clippy::too_many_arguments)] // `Var::conv_transpose2d` と同じ理由（PyTorch の全引数を受理するため）。
pub fn conv_transpose3d<'t>(
    input: &Var<'t>,
    weight: &Var<'t>,
    bias: Option<&Var<'t>>,
    stride: [usize; 3],
    padding: [usize; 3],
    output_padding: [usize; 3],
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
    if output_padding
        .iter()
        .zip(stride.iter())
        .any(|(op, s)| op >= s)
    {
        return Err(AutodiffError::InvalidArgument(format!(
            "conv_transpose3d: output_padding ({output_padding:?}) must be < stride ({stride:?}) \
             on each axis (col2im3d の P 軸契約による意図的な PyTorch 非互換。\
             docs/autodiff-conv-transpose3d-max-unpool-decision.md §5)"
        )));
    }

    let in_shape = input.shape();
    let out_shape = conv_transpose3d_out_shape(&in_shape, &weight_shape, &params, output_padding)
        .map_err(AutodiffError::Shape)?;
    if let Some(b) = bias {
        let bias_shape = b.shape();
        let cout = out_shape[1];
        if bias_shape != [cout] {
            return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                lhs: bias_shape,
                rhs: vec![cout],
            }));
        }
    }

    let input_val = materialize_one(input)?;
    let weight_val = materialize_one(weight)?;
    let bias_val = match bias {
        Some(b) => Some(materialize_one(b)?),
        None => None,
    };

    let value_out = conv_transpose3d_forward(
        input.tape().ops(),
        &input_val,
        &weight_val,
        bias_val.as_ref(),
        &params,
        &out_shape,
    )?;
    if value_out.shape() != out_shape {
        return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
            ShapeError::ShapeMismatch {
                lhs: value_out.shape().to_vec(),
                rhs: out_shape,
            },
        )));
    }
    let id = input.tape().push_eager(
        Op::ConvTranspose3d {
            input: input.node_id(),
            weight: weight.node_id(),
            bias: bias.map(|b| b.node_id()),
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

/// `Op::ConvTranspose3d` の forward 合成（`grad::conv_transpose2d_with_fallback` の空間 3 軸版）。
///
/// ```text
/// w_mat = weight.reshape([G, Cin_g, K_g])            // K_g = Cout_g·kD·kH·kW
/// x5    = input.reshape([N, G, Cin_g, D·H·W])
/// d_col = ops.gemm_batched(w_matᵀ, x5)               // [N, G, K_g, D·H·W]（常にバックエンド）
/// out   = col2im3d_with_fallback(ops, d_col, out_shape, params)
/// out  += bias.reshape([1, Cout, 1, 1, 1])           // ops.add
/// ```
///
/// **`col2im3d_with_fallback` の `input_shape` 引数には「転置畳み込みの出力 shape」（`out_shape`）を
/// 渡す**——`col2im3d` の "input" は仮想 conv3d の入力（＝転置畳み込みの出力）を指すため、
/// `conv3d_with_fallback` に慣れた読み手ほど誤読しやすい。bias を `[1, Cout, 1, 1, 1]` へ reshape
/// してから加算するのは、右詰め broadcast による `W` 軸誤加算を避けるため。`N = 0` は GEMM・col2im3d を
/// 呼ばず空テンソルを返す。
fn conv_transpose3d_forward(
    ops: &dyn BackendOps,
    input: &Tensor<f32>,
    weight: &Tensor<f32>,
    bias: Option<&Tensor<f32>>,
    params: &Conv3dParams,
    out_shape: &[usize],
) -> Result<Tensor<f32>, AutodiffError> {
    if out_shape[0] == 0 {
        return Tensor::zeros(out_shape).map_err(AutodiffError::Shape);
    }
    let in_shape = input.shape().to_vec();
    let weight_shape = weight.shape().to_vec();
    let groups = params.groups();
    let cin_g = in_shape[1] / groups.max(1);
    let cout = out_shape[1];
    let k_g = weight_shape[1..2]
        .iter()
        .chain(weight_shape[2..].iter())
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or(AutodiffError::Shape(ShapeError::ElementCountOverflow))?;
    let dhw = in_shape[2..]
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or(AutodiffError::Shape(ShapeError::ElementCountOverflow))?;

    let w_mat = weight
        .contiguous()
        .reshape(&[groups, cin_g, k_g])
        .map_err(AutodiffError::Shape)?;
    let x5 = input
        .contiguous()
        .reshape(&[in_shape[0], groups, cin_g, dhw])
        .map_err(AutodiffError::Shape)?;
    let w_mat_t = w_mat.transpose(1, 2).map_err(AutodiffError::Shape)?;
    let d_col = ops
        .gemm_batched(&w_mat_t, &x5)
        .map_err(AutodiffError::Backend)?;
    let out_no_bias = col2im3d_with_fallback(ops, &d_col, out_shape, params)?;

    match bias {
        Some(b) => {
            let bias_reshaped = b
                .contiguous()
                .reshape(&[1, cout, 1, 1, 1])
                .map_err(AutodiffError::Shape)?;
            let out = ops
                .add(&out_no_bias, &bias_reshaped)
                .map_err(AutodiffError::Backend)?;
            if out.shape() != out_shape {
                return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                    ShapeError::ShapeMismatch {
                        lhs: out.shape().to_vec(),
                        rhs: out_shape.to_vec(),
                    },
                )));
            }
            Ok(out)
        }
        None => Ok(out_no_bias),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tape::Tape;

    fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
        Tensor::new(data, shape).expect("test fixture: shape 一致")
    }

    #[test]
    fn scatters_each_input_element_times_kernel() {
        // 入力 [1,1,1,1,2]・kernel [1,1,1,1,2]・stride 1: out[w] = Σ x[i]·k[w-i]。
        let tape = Tape::new();
        let x = tape.var(&t(vec![2.0, 3.0], &[1, 1, 1, 1, 2]));
        let w = tape.var(&t(vec![10.0, 100.0], &[1, 1, 1, 1, 2]));
        let y = conv_transpose3d(&x, &w, None, [1; 3], [0; 3], [0; 3], [1; 3], 1).unwrap();
        let out = y.to_tensor();
        assert_eq!(out.shape(), &[1, 1, 1, 1, 3]);
        assert_eq!(
            out.host_slice().into_owned(),
            vec![20.0, 200.0 + 30.0, 300.0]
        );
    }

    #[test]
    fn bias_is_added_per_output_channel() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 1.0], &[1, 1, 1, 1, 2]));
        // Cout=3 == Wout=3 の形状で、bias が W 軸へ誤加算されないことを固定する。
        let w = tape.var(&t(vec![0.0; 6], &[1, 3, 1, 1, 2]));
        let b = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
        let y = conv_transpose3d(&x, &w, Some(&b), [1; 3], [0; 3], [0; 3], [1; 3], 1).unwrap();
        assert_eq!(y.to_tensor().shape(), &[1, 3, 1, 1, 3]);
        let data = y.to_tensor().host_slice().into_owned();
        assert_eq!(data, vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0]);
    }

    #[test]
    fn rejects_invalid_arguments_without_orphan_nodes() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![0.0; 8], &[1, 1, 2, 2, 2]));
        let w = tape.var(&t(vec![0.0; 8], &[1, 1, 2, 2, 2]));
        let before = tape.len();
        assert!(matches!(
            conv_transpose3d(&x, &w, None, [1; 3], [0; 3], [1, 0, 0], [1; 3], 1),
            Err(AutodiffError::InvalidArgument(_))
        ));
        assert!(conv_transpose3d(&x, &w, None, [0, 1, 1], [0; 3], [0; 3], [1; 3], 1).is_err());
        assert!(conv_transpose3d(&x, &w, None, [1; 3], [0; 3], [0; 3], [0, 1, 1], 1).is_err());
        assert!(conv_transpose3d(&x, &w, None, [1; 3], [0; 3], [0; 3], [1; 3], 0).is_err());
        let bad_bias = tape.var(&t(vec![0.0; 2], &[2]));
        assert!(
            conv_transpose3d(&x, &w, Some(&bad_bias), [1; 3], [0; 3], [0; 3], [1; 3], 1).is_err()
        );
        assert_eq!(
            tape.len(),
            before + 1,
            "孤児ノードを残さない（bad_bias の var 1 件のみ）"
        );
    }

    #[test]
    fn empty_batch_yields_empty_output() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![], &[0, 1, 2, 2, 2]));
        let w = tape.var(&t(vec![0.0; 8], &[1, 1, 2, 2, 2]));
        let y = conv_transpose3d(&x, &w, None, [1; 3], [0; 3], [0; 3], [1; 3], 1).unwrap();
        assert_eq!(y.to_tensor().shape(), &[0, 1, 3, 3, 3]);
    }
}
