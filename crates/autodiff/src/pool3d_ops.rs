//! `max_pool3d`・`avg_pool3d` の自由関数（イシュー #2643・親 #2625「Phase 4」・ルート
//! #2499）。
//!
//! **facade 非公開（保留）**: 公開形（`Var::max_pool3d`／`Var::avg_pool3d` の委譲メソッド）は
//! 未承認で、承認依頼は #2677（公開自体は承認後の #2678）。層化（`nn::MaxPool3d`／
//! `nn::AvgPool3d`・`Sequential::add_*`）は #2679 の対象で本イシューでは作らない。本モジュールは
//! 内部クレート限定の入口で、`Var` に inherent メソッドを足さない。保留は
//! `crates/facade/src/lib.rs` の `Pool3dOpsHoldDoctestGuard` と
//! `crates/facade/tests/api_surface.rs` の否定ガードが機械的に固定する
//! （`docs/autodiff-pool3d-ops-decision.md`）。
//!
//! **PyTorch 相当**:
//!
//! | 演算 | PyTorch 相当 | 出力 |
//! |---|---|---|
//! | [`max_pool3d`] | `F.max_pool3d(x, k, s, p, d, ceil_mode=False, return_indices=True)` | 値 `Var` と索引 `Tensor<i32>` |
//! | [`avg_pool3d`] | `F.avg_pool3d(x, k, s, p, ceil_mode=False, count_include_pad)` | 値 `Var` |
//!
//! NCDHW 固定・rank 5 のみ。索引は `(n, c)` 平面内 flat 添字 `d·H·W + h·W + w`（`i32`・
//! 勾配は流れない）。`ceil_mode = true`・`divisor_override`・バッチなし入力は未対応。
//!
//! **経路**: ① `ceil_mode`・パラメータ・形状・索引上限（`i32`）の検査（実体化・tape 操作より
//! 前。エラー時に孤児ノードを残さない）→ ② 入力の実体化 → ③ `BackendOps::pool3d_max`／
//! `pool3d_avg`（`Unsupported` のときだけ共有ホストカーネル
//! `fandhe_ai_tensor_core::pool3d::{max,avg}_pool3d_host` へフォールバックし、他のエラーは
//! 伝播する。戻り値 shape も検証する）→ ④ 専用 `Op`（`Op::MaxPool3d`／`Op::AvgPool3d`）を積む。
//! VJP は `grad.rs`。
//!
//! **数値契約**: Max は比較・選択のみ（値は入力要素と bit 一致・タイ先勝ち・NaN 伝播）。Avg は
//! `f64` アキュムレータで 1 回だけ `f32` へ downcast（規則の正は
//! `fandhe_ai_tensor_core::pool3d`）。非有限入力は拒否せず伝播する。高階微分（`create_graph`）・
//! activation checkpoint・f64 自動微分経路は対象外。

use fandhe_ai_tensor_core::pool3d::{self, Pool3dLayout, Pool3dParams};
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

fn layout_for(x_shape: &[usize], params: &Pool3dParams) -> Result<Pool3dLayout, AutodiffError> {
    pool3d::pool3d_layout(x_shape, params).map_err(AutodiffError::Shape)
}

/// 3D max pooling（`F.max_pool3d(..., return_indices=True)` 相当）。値と索引を返す。
///
/// `input`: `[N, C, D, H, W]`。`stride = None` は `kernel_size`。検査順: `ceil_mode == true` 拒否 →
/// `Pool3dParams::new`（0・`padding <= kernel/2`）→ `pool3d_layout`（rank・空間軸 0・負分子・空窓）→
/// `D·H·W <= i32::MAX` → 実体化 → バックエンド／ホスト → 戻り shape 再検証 → `push_eager`。
pub fn max_pool3d<'t>(
    input: &Var<'t>,
    kernel_size: [usize; 3],
    stride: Option<[usize; 3]>,
    padding: [usize; 3],
    dilation: [usize; 3],
    ceil_mode: bool,
) -> Result<(Var<'t>, Tensor<i32>), AutodiffError> {
    if ceil_mode {
        return Err(AutodiffError::InvalidArgument(
            "max_pool3d: ceil_mode=true は v1 で未対応".into(),
        ));
    }
    let params = Pool3dParams::new(kernel_size, stride, padding, dilation)
        .map_err(AutodiffError::Backend)?;
    let layout = layout_for(&input.shape(), &params)?;
    layout.check_i32_indices().map_err(AutodiffError::Shape)?;
    let x = materialize_one(input)?;
    let out_shape = layout.out_shape();
    let (value, index) = match input.tape().ops().pool3d_max(&x, &params) {
        Ok((v, i)) => {
            verify_shape(v.shape(), &out_shape)?;
            verify_shape(i.shape(), &out_shape)?;
            (v, i)
        }
        Err(BackendError::Unsupported(_)) => {
            let data = x.contiguous();
            let (v, i) = pool3d::max_pool3d_host(&data.host_slice(), &layout)
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

/// 3D average pooling（`F.avg_pool3d` 相当）。
///
/// `input`: `[N, C, D, H, W]`。`dilation` は `[1, 1, 1]` 固定。検査順は [`max_pool3d`] と同じ
/// （索引上限の検査は不要）。
pub fn avg_pool3d<'t>(
    input: &Var<'t>,
    kernel_size: [usize; 3],
    stride: Option<[usize; 3]>,
    padding: [usize; 3],
    ceil_mode: bool,
    count_include_pad: bool,
) -> Result<Var<'t>, AutodiffError> {
    if ceil_mode {
        return Err(AutodiffError::InvalidArgument(
            "avg_pool3d: ceil_mode=true は v1 で未対応".into(),
        ));
    }
    let params = Pool3dParams::new(kernel_size, stride, padding, [1, 1, 1])
        .map_err(AutodiffError::Backend)?;
    let layout = layout_for(&input.shape(), &params)?;
    let x = materialize_one(input)?;
    let out_shape = layout.out_shape();
    let value = match input
        .tape()
        .ops()
        .pool3d_avg(&x, &params, count_include_pad)
    {
        Ok(v) => {
            verify_shape(v.shape(), &out_shape)?;
            v
        }
        Err(BackendError::Unsupported(_)) => {
            let data =
                pool3d::avg_pool3d_host(&x.contiguous().host_slice(), &layout, count_include_pad)
                    .map_err(AutodiffError::Shape)?;
            Tensor::new(data, &out_shape).map_err(AutodiffError::Shape)?
        }
        Err(other) => return Err(AutodiffError::Backend(other)),
    };
    let id = input.tape().push_eager(
        Op::AvgPool3d {
            input: input.node_id(),
            params,
            count_include_pad,
        },
        value,
    );
    Ok(Var::from_raw(input.tape(), id))
}
