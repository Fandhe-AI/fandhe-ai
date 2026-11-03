//! `adaptive_avg_pool3d`・`adaptive_max_pool3d` の自由関数（イシュー #3823・親 #3747「3D 適応
//! プーリング」。設計・実装記録は `docs/autodiff-pool3d-ops-decision.md` §18・§21）。
//!
//! **facade 公開状況**: `Var::adaptive_avg_pool3d`／`Var::adaptive_max_pool3d`（本モジュールの
//! 自由関数への 1 行委譲）として公開する。委譲本体は `crates/facade/tests/api_surface.rs` の
//! `var_adaptive_pool3d_methods_are_thin_delegations` が固定する。モジュール自体は再エクスポート
//! しない。層型・`Sequential::add_adaptive_*_pool3d` は #3824 の担当。
//!
//! **PyTorch 相当**:
//!
//! | 演算 | PyTorch 相当 | 出力 |
//! |---|---|---|
//! | [`adaptive_avg_pool3d`] | `F.adaptive_avg_pool3d(x, output_size)` | 値 `Var` |
//! | [`adaptive_max_pool3d`] | `F.adaptive_max_pool3d(x, output_size, return_indices=True)` | 値 `Var` と索引 `Tensor<i32>` |
//!
//! NCDHW 固定・rank 5 のみ。`output_size` は `[usize; 3]`（`None` 軸は無し）で各軸 1 以上。
//! 窓は軸ごとの `adaptive_window`（2D と共有）。索引は `(n, c)` 平面内 flat 添字
//! `d·H·W + h·W + w`（`Op::MaxPool3d` と同じ意味で `max_unpool3d` へそのまま渡せる）。
//!
//! **経路**: ① 形状・`output_size`・（Max のみ）索引上限 `i32` の検査（実体化・tape 操作より前。
//! エラー時に孤児ノードを残さない）→ ② 入力の実体化 → ③ `BackendOps::adaptive_pool3d_avg`／
//! `adaptive_pool3d_max`（`Unsupported` のときだけ共有ホストカーネル
//! `fandhe_ai_tensor_core::adaptive_pool3d::*_host` へフォールバックし、他のエラーは伝播する。
//! 戻り値 shape も検証する）→ ④ `push_eager`。Max は既存の `Op::MaxPool3d { input, index }` を
//! 再利用し、Avg は `Op::AdaptiveAvgPool3d` を積む。VJP は `grad.rs`。
//!
//! **数値契約**: Max は比較・選択のみ（値は入力要素と bit 一致・タイ先勝ち・NaN 伝播）。Avg は
//! `f64` アキュムレータで 1 回だけ `f32` へ downcast（規則の正は
//! `fandhe_ai_tensor_core::adaptive_pool3d`）。非有限入力は拒否せず伝播する。

use fandhe_ai_tensor_core::adaptive_pool3d::{self, AdaptivePool3dLayout};
use fandhe_ai_tensor_core::{BackendError, ShapeError, Tensor};

use crate::error::AutodiffError;
use crate::tape::{Op, materialize_fallible};
use crate::var::Var;

fn verify_shape(actual: &[usize], expected: &[usize]) -> Result<(), AutodiffError> {
    if actual == expected {
        Ok(())
    } else {
        Err(AutodiffError::Backend(BackendError::ShapeMismatch(
            ShapeError::ShapeMismatch {
                lhs: actual.to_vec(),
                rhs: expected.to_vec(),
            },
        )))
    }
}

fn materialize_one<'t>(x: &Var<'t>) -> Result<Tensor<f32>, AutodiffError> {
    let nodes = x.tape().nodes.borrow();
    let ops = x.tape().ops();
    Ok(materialize_fallible(&nodes, ops, x.node_id())?.clone())
}

fn layout_for(
    x_shape: &[usize],
    output_size: [usize; 3],
) -> Result<AdaptivePool3dLayout, AutodiffError> {
    adaptive_pool3d::adaptive_pool3d_layout(x_shape, output_size).map_err(AutodiffError::Shape)
}

/// 3D adaptive average pooling（`F.adaptive_avg_pool3d` 相当）。
///
/// `input`: `[N, C, D, H, W]`、`output_size`: `[Dout, Hout, Wout]`（各軸 1 以上。入力より大きくてもよい）。
/// 検査順: rank 5 → 空間軸 0 → `output_size` 0 → 出力要素数オーバーフロー → 実体化 →
/// バックエンド／ホスト → 戻り shape 再検証 → `push_eager`。
pub fn adaptive_avg_pool3d<'t>(
    input: &Var<'t>,
    output_size: [usize; 3],
) -> Result<Var<'t>, AutodiffError> {
    let layout = layout_for(&input.shape(), output_size)?;
    let x = materialize_one(input)?;
    let out_shape = layout.out_shape();
    let value = match input.tape().ops().adaptive_pool3d_avg(&x, output_size) {
        Ok(v) => {
            verify_shape(v.shape(), &out_shape)?;
            v
        }
        Err(BackendError::Unsupported(_)) => {
            let data =
                adaptive_pool3d::adaptive_avg_pool3d_host(&x.contiguous().host_slice(), &layout)
                    .map_err(AutodiffError::Shape)?;
            Tensor::new(data, &out_shape).map_err(AutodiffError::Shape)?
        }
        Err(other) => return Err(AutodiffError::Backend(other)),
    };
    let id = input.tape().push_eager(
        Op::AdaptiveAvgPool3d {
            input: input.node_id(),
        },
        value,
    );
    Ok(Var::from_raw(input.tape(), id))
}

/// 3D adaptive max pooling（`F.adaptive_max_pool3d(..., return_indices=True)` 相当）。値と索引を返す。
///
/// 検査順は [`adaptive_avg_pool3d`] に `D·H·W <= i32::MAX`（`N`／`C` が 0 でも検査）を加える。
pub fn adaptive_max_pool3d<'t>(
    input: &Var<'t>,
    output_size: [usize; 3],
) -> Result<(Var<'t>, Tensor<i32>), AutodiffError> {
    let layout = layout_for(&input.shape(), output_size)?;
    layout.check_i32_indices().map_err(AutodiffError::Shape)?;
    let x = materialize_one(input)?;
    let out_shape = layout.out_shape();
    let (value, index) = match input.tape().ops().adaptive_pool3d_max(&x, output_size) {
        Ok((v, i)) => {
            verify_shape(v.shape(), &out_shape)?;
            verify_shape(i.shape(), &out_shape)?;
            (v, i)
        }
        Err(BackendError::Unsupported(_)) => {
            let (v, i) =
                adaptive_pool3d::adaptive_max_pool3d_host(&x.contiguous().host_slice(), &layout)
                    .map_err(AutodiffError::Shape)?;
            (
                Tensor::new(v, &out_shape).map_err(AutodiffError::Shape)?,
                Tensor::new(i, &out_shape).map_err(AutodiffError::Shape)?,
            )
        }
        Err(other) => return Err(AutodiffError::Backend(other)),
    };
    let id = input.tape().push_eager(
        Op::MaxPool3d {
            input: input.node_id(),
            index: index.clone(),
        },
        value,
    );
    Ok((Var::from_raw(input.tape(), id), index))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tape;

    fn leaf<'t>(tape: &'t Tape, shape: &[usize]) -> Var<'t> {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        tape.var(&Tensor::new(data, shape).unwrap())
    }

    /// 検査エラー時は tape にノードを残さない（孤児ノード無し）。
    #[test]
    fn rejected_inputs_leave_no_orphan_nodes() {
        let tape = Tape::new();
        let x = leaf(&tape, &[1, 1, 2, 2, 2]);
        let r4 = leaf(&tape, &[1, 1, 2, 2]);
        let before = tape.nodes.borrow().len();
        assert!(adaptive_avg_pool3d(&x, [0, 1, 1]).is_err());
        assert!(adaptive_max_pool3d(&x, [1, 0, 1]).is_err());
        assert!(adaptive_avg_pool3d(&r4, [1, 1, 1]).is_err());
        assert!(adaptive_max_pool3d(&r4, [1, 1, 1]).is_err());
        assert_eq!(tape.nodes.borrow().len(), before);
    }

    #[test]
    fn avg_and_max_record_one_node_each() {
        let tape = Tape::new();
        let x = leaf(&tape, &[1, 1, 2, 2, 2]);
        let before = tape.nodes.borrow().len();
        let a = adaptive_avg_pool3d(&x, [1, 1, 1]).unwrap();
        assert_eq!(a.to_tensor().host_slice().to_vec(), vec![3.5]);
        let (m, i) = adaptive_max_pool3d(&x, [1, 1, 1]).unwrap();
        assert_eq!(m.to_tensor().host_slice().to_vec(), vec![7.0]);
        assert_eq!(i.host_slice().to_vec(), vec![7]);
        assert_eq!(tape.nodes.borrow().len(), before + 2);
    }
}
