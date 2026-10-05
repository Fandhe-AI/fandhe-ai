//! `cummax`・`cummin`・`logcumsumexp` の自由関数（イシュー #2636・親 #2625
//! 「Phase 4」・ルート #2499）。
//!
//! **facade 非公開（保留）**: 公開形（`Var::cummax`／`Var::cummin`／
//! `Var::logcumsumexp` の委譲メソッド）は未承認で、承認依頼は #2677（公開自体は
//! 承認後の #2678）。本モジュールは内部クレート限定の入口で、`Var` に inherent
//! メソッドを足さない。保留は `crates/facade/src/lib.rs` の
//! `CumulativeOpsHoldDoctestGuard` と `crates/facade/tests/api_surface.rs` の
//! 否定ガードが機械的に固定する（`docs/autodiff-cumulative-ops-decision.md`）。
//!
//! **PyTorch 相当**:
//!
//! | 演算 | PyTorch 相当 | 出力 |
//! |---|---|---|
//! | [`cummax`] | `torch.cummax(x, dim)` | 値 `Var` と索引 `Tensor<i32>`（入力と同 shape） |
//! | [`cummin`] | `torch.cummin(x, dim)` | 同上 |
//! | [`logcumsumexp`] | `torch.logcumsumexp(x, dim)` | 値 `Var`（入力と同 shape） |
//!
//! `dim` は `usize`（`Var::cumsum` と同じく負の添字は受けない）。索引は既存慣例
//! （`Var::sort`／`topk`）に合わせ `Tensor<i32>` で、勾配は流れない。
//!
//! **経路**: ① `dim`・形状・確保サイズ・索引上限（`i32`）の検査
//! （`fandhe_ai_tensor_core::cumulative::cumulative_layout`。実体化より前）→
//! ② 入力の実体化 → ③ `BackendOps::scan_cummax`／`scan_cummin`／
//! `scan_logcumsumexp`（`Unsupported` のときだけ共有ホストカーネル
//! `cumulative::cummax_host`／`cummin_host`／`logcumsumexp_host` へ
//! フォールバックし、他のエラーは伝播する。戻り値 shape も検証する）→
//! ④ 専用 `Op`（`Op::Cummax`／`Op::Cummin`／`Op::Logcumsumexp`）を積む。VJP は
//! `grad.rs`。
//!
//! **数値契約**: `cummax`／`cummin` は比較・選択のみ（値は入力の要素と bit
//! 一致・タイは後勝ち・NaN は伝播）。`logcumsumexp` は `f64` アキュムレータで
//! 1 回だけ `f32` へ downcast（規則の正は `fandhe_ai_tensor_core::cumulative`）。
//! 非有限入力は拒否せず伝播する。高階微分（`create_graph`）・activation
//! checkpoint・f64 自動微分経路は対象外。

use fandhe_ai_tensor_core::cumulative::{self, CumulativeLayout};
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

/// 2 出力（値・索引）演算の共通本体。`is_max` で `cummax`／`cummin` を切り替える。
fn extremum<'t>(
    x: &Var<'t>,
    dim: usize,
    is_max: bool,
) -> Result<(Var<'t>, Tensor<i32>), AutodiffError> {
    let layout = cumulative::cumulative_layout(&x.shape(), dim).map_err(AutodiffError::Shape)?;
    layout.check_i32_indices().map_err(AutodiffError::Shape)?;
    let input = materialize_one(x)?;
    let ops = x.tape().ops();
    let backend = if is_max {
        ops.scan_cummax(&input, dim)
    } else {
        ops.scan_cummin(&input, dim)
    };
    let (value, index) = match backend {
        Ok((v, i)) => {
            verify_shape(v.shape(), layout.shape())?;
            verify_shape(i.shape(), layout.shape())?;
            (v, i)
        }
        Err(BackendError::Unsupported(_)) => host_extremum(&input, &layout, is_max)?,
        Err(other) => return Err(AutodiffError::Backend(other)),
    };
    let node = if is_max {
        Op::Cummax {
            input: x.node_id(),
            dim,
            index: index.clone(),
        }
    } else {
        Op::Cummin {
            input: x.node_id(),
            dim,
            index: index.clone(),
        }
    };
    let id = x.tape().push_eager(node, value);
    Ok((Var::from_raw(x.tape(), id), index))
}

fn host_extremum(
    input: &Tensor<f32>,
    layout: &CumulativeLayout,
    is_max: bool,
) -> Result<(Tensor<f32>, Tensor<i32>), AutodiffError> {
    let data = input.contiguous();
    let (v, i) = if is_max {
        cumulative::cummax_host(&data.host_slice(), layout)
    } else {
        cumulative::cummin_host(&data.host_slice(), layout)
    }
    .map_err(AutodiffError::Shape)?;
    Ok((
        Tensor::new(v, layout.shape()).map_err(AutodiffError::Shape)?,
        Tensor::new(i, layout.shape()).map_err(AutodiffError::Shape)?,
    ))
}

/// 累積最大値（`torch.cummax` 相当）。値と索引（`dim` 軸上の位置）を返す。
pub fn cummax<'t>(x: &Var<'t>, dim: usize) -> Result<(Var<'t>, Tensor<i32>), AutodiffError> {
    extremum(x, dim, true)
}

/// 累積最小値（`torch.cummin` 相当）。値と索引（`dim` 軸上の位置）を返す。
pub fn cummin<'t>(x: &Var<'t>, dim: usize) -> Result<(Var<'t>, Tensor<i32>), AutodiffError> {
    extremum(x, dim, false)
}

/// 累積 `logsumexp`（`torch.logcumsumexp` 相当）。
pub fn logcumsumexp<'t>(x: &Var<'t>, dim: usize) -> Result<Var<'t>, AutodiffError> {
    let layout = cumulative::cumulative_layout(&x.shape(), dim).map_err(AutodiffError::Shape)?;
    let input = materialize_one(x)?;
    let value = match x.tape().ops().scan_logcumsumexp(&input, dim) {
        Ok(v) => {
            verify_shape(v.shape(), layout.shape())?;
            v
        }
        Err(BackendError::Unsupported(_)) => {
            let data = cumulative::logcumsumexp_host(&input.contiguous().host_slice(), &layout)
                .map_err(AutodiffError::Shape)?;
            Tensor::new(data, layout.shape()).map_err(AutodiffError::Shape)?
        }
        Err(other) => return Err(AutodiffError::Backend(other)),
    };
    let id = x.tape().push_eager(
        Op::Logcumsumexp {
            input: x.node_id(),
            dim,
        },
        value,
    );
    Ok(Var::from_raw(x.tape(), id))
}
