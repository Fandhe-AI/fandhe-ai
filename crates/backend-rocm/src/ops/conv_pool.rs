//! `RocmBackendOps` の `pad`・`im2col`・`col2im`・`max_pool2d`・`avg_pool2d`・`adaptive_avg_pool2d` の
//! 結線（イシュー #3438・分解案 R7）。
//!
//! 役割: `ops.rs` の `BackendOps` impl から 1 行で呼ばれ、(1) `tensor-core` の出力 shape 関数による
//! 再検査（`ShapeMismatch`）、(2) 空出力・空入力の早期リターン、(3) **HIP に触れる前に `Unsupported` と
//! 形状エラーを確定**（サイズ上限超過・avg の dilation ≠ 1。ROCm 非搭載環境で `DeviceUnavailable` が
//! 先に出ると、autodiff の「`Unsupported` のときだけホスト参照実装へフォールバックする」契約が壊れる。
//! PR #1781 の教訓・`ops/ew.rs` と同じ）、(4) engine（Hip／テストの HostModel）の分岐、(5) `Tensor`
//! の組み立て、を担う。カーネル起動そのものは `im2col.rs`／`constant_pad.rs`／`pooling.rs` の実行型。
//! CUDA 版 `backend-cuda::ops` の同名メソッドと同じ順序・同じエラー種別にしてある。
//!
//! # エラー写像（[`map_conv_pool_error`]）
//! 共有の `to_backend_error` は変えず、本族だけ次のように写す（CUDA の `map_im2col_error`／
//! `map_pooling_error`／`map_constant_pad_error` と同じ。`Unsupported` に落とすのはサイズ上限超過だけで、
//! 内部契約違反や HIP の失敗をホストフォールバックで覆い隠さない。security.md A08）。
//! - `SizeLimitExceeded` → `Unsupported`
//! - `InvalidOpArgs` → `ShapeMismatch(ElementCountOverflow)`
//! - それ以外 → `to_backend_error`

use fandhe_ai_tensor_core::{
    BackendError, Conv2dParams, Pool2dParams, ShapeError, Tensor, adaptive_pool2d_out_shape,
    im2col_out_shape, pad_out_shape, pool2d_out_shape,
};

use super::{Engine, RocmBackendOps};
use crate::constant_pad::{cached_constant_pad, plan_pad};
use crate::error::{RocmError, to_backend_error};
use crate::im2col::{cached_im2col, plan_col2im, plan_im2col};
use crate::pooling::{
    PoolPlan, cached_pooling, validate_adaptive_pool, validate_avg_pool, validate_max_pool,
};

/// 本族のエラー写像（モジュール doc 参照）。
fn map_conv_pool_error(e: RocmError) -> BackendError {
    match e {
        RocmError::SizeLimitExceeded { .. } => BackendError::Unsupported(e.to_string()),
        RocmError::InvalidOpArgs { .. } => {
            BackendError::ShapeMismatch(ShapeError::ElementCountOverflow)
        }
        other => to_backend_error(other),
    }
}

fn shape_err(e: ShapeError) -> BackendError {
    BackendError::ShapeMismatch(e)
}

/// 入力 shape の要素数とバイト数（`f32`）を `contiguous()` の前に検査する（巨大な broadcast view で
/// 確保に進んで abort しないため。CUDA の `checked_bytes_for::<f32>` と同じ）。0 軸があれば 0。
fn check_input_numel(shape: &[usize]) -> Result<usize, BackendError> {
    if shape.contains(&0) {
        return Ok(0);
    }
    let numel = shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or(shape_err(ShapeError::ElementCountOverflow))?;
    let bytes = numel
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or(shape_err(ShapeError::ElementCountOverflow))?;
    if bytes > isize::MAX as usize {
        return Err(shape_err(ShapeError::ElementCountOverflow));
    }
    Ok(numel)
}

/// `contiguous()` 直後のスライスを取り出し、plan の期待長と一致することを確かめる。
fn slice_with_len<'a>(
    op: &'static str,
    t: &'a Tensor<f32>,
    expected: usize,
) -> Result<&'a [f32], BackendError> {
    let s = t
        .as_slice()
        .ok_or_else(|| BackendError::KernelLaunchFailed(format!("{op}: tensor not contiguous")))?;
    if s.len() != expected {
        return Err(BackendError::KernelLaunchFailed(format!(
            "{op}: 入力の長さ {} が期待 {expected} と一致しません",
            s.len()
        )));
    }
    Ok(s)
}

/// tensor-core の出力 shape と plan の出力 shape の一致（内部整合の確認）。
fn check_plan_shape(
    op: &'static str,
    out_shape: &[usize],
    plan: &PoolPlan,
) -> Result<(), BackendError> {
    if plan.out_shape.as_slice() != out_shape {
        return Err(BackendError::KernelLaunchFailed(format!(
            "{op}: 出力 shape が一致しません: tensor-core {out_shape:?}, 実行型 {:?}",
            plan.out_shape
        )));
    }
    Ok(())
}

fn as_nchw(shape: &[usize]) -> Result<[usize; 4], BackendError> {
    shape.try_into().map_err(|_| {
        shape_err(ShapeError::RankMismatch {
            expected: 4,
            actual: shape.len(),
        })
    })
}

/// max の索引域検査（`h * w_in + w` が `int` に収まる）。空出力の早期リターンより前に行う。
fn check_max_index_range(shape: &[usize]) -> Result<(), BackendError> {
    let h = shape.get(2).copied().unwrap_or(0);
    let w = shape.get(3).copied().unwrap_or(0);
    let hw = h
        .checked_mul(w)
        .ok_or(shape_err(ShapeError::ElementCountOverflow))?;
    if hw > i32::MAX as usize {
        return Err(shape_err(ShapeError::IndexRangeOverflow { index: hw }));
    }
    Ok(())
}

impl RocmBackendOps {
    /// `BackendOps::pad`（定数 pad）。
    pub(super) fn conv_pool_pad(
        &self,
        input: &Tensor<f32>,
        pads: &[(usize, usize)],
        value: f32,
    ) -> Result<Tensor<f32>, BackendError> {
        let out_shape = pad_out_shape(input.shape(), pads).map_err(shape_err)?;
        // 出力が空なら入力 shape に依らず空。入力の積を取る前に返す。
        if out_shape.contains(&0) {
            return Tensor::new(Vec::new(), &out_shape).map_err(shape_err);
        }
        let plan = plan_pad(input.shape(), pads).map_err(map_conv_pool_error)?;
        let numel_out = plan.numel_out;
        // 入力が空（いずれかの軸が 0）でも出力は非空になりうる。`value` で埋めて返す。
        if input.shape().contains(&0) {
            return Tensor::new(vec![value; numel_out], &out_shape).map_err(shape_err);
        }
        // rank 0: 出力も 1 要素で、デバイスは要らない。
        if input.shape().is_empty() {
            return Ok(input.contiguous());
        }
        check_input_numel(input.shape())?;
        let owned = input.contiguous();
        let src = slice_with_len("pad", &owned, owned.numel())?;
        let out = match self.engine {
            Engine::Hip => cached_constant_pad(self.target)
                .and_then(|k| k.run_pad_f32(src, input.shape(), pads, value)),
            #[cfg(test)]
            Engine::HostModel => crate::constant_pad::validate_pad(src.len(), input.shape(), pads)
                .map(|_| crate::conv_pool_model::pad(src, input.shape(), pads, value)),
        }
        .map_err(map_conv_pool_error)?;
        Tensor::new(out, &out_shape).map_err(shape_err)
    }

    /// `BackendOps::im2col`。
    pub(super) fn conv_pool_im2col(
        &self,
        input: &Tensor<f32>,
        params: &Conv2dParams,
    ) -> Result<Tensor<f32>, BackendError> {
        let out_shape = im2col_out_shape(input.shape(), params).map_err(shape_err)?;
        // 出力が空（N／Cin 系の軸が 0）なら入力を読む必要がない。
        if out_shape.contains(&0) {
            return Tensor::new(Vec::new(), &out_shape).map_err(shape_err);
        }
        check_input_numel(input.shape())?;
        let plan = plan_im2col(input.shape(), &out_shape, params).map_err(map_conv_pool_error)?;
        let owned = input.contiguous();
        let src = slice_with_len("im2col", &owned, plan.numel_in)?;
        let out = match self.engine {
            Engine::Hip => cached_im2col(self.target)
                .and_then(|k| k.run_im2col_f32(src, input.shape(), &out_shape, params)),
            #[cfg(test)]
            Engine::HostModel => Ok(crate::conv_pool_model::im2col(src, &plan)),
        }
        .map_err(map_conv_pool_error)?;
        Tensor::new(out, &out_shape).map_err(shape_err)
    }

    /// `BackendOps::col2im`（im2col の随伴）。
    pub(super) fn conv_pool_col2im(
        &self,
        d_col: &Tensor<f32>,
        input_shape: &[usize],
        params: &Conv2dParams,
    ) -> Result<Tensor<f32>, BackendError> {
        let expected_col = im2col_out_shape(input_shape, params).map_err(shape_err)?;
        if d_col.shape() != expected_col.as_slice() {
            return Err(shape_err(ShapeError::ShapeMismatch {
                lhs: d_col.shape().to_vec(),
                rhs: expected_col,
            }));
        }
        // 戻り値として確保する `input_shape` 自体も、`d_col` とは独立に確保前検査する。
        check_input_numel(input_shape)?;
        if input_shape.contains(&0) {
            return Tensor::new(Vec::new(), input_shape).map_err(shape_err);
        }
        check_input_numel(d_col.shape())?;
        let plan = plan_col2im(input_shape, d_col.shape(), params).map_err(map_conv_pool_error)?;
        let owned = d_col.contiguous();
        let src = slice_with_len("col2im", &owned, plan.numel_col)?;
        let out = match self.engine {
            Engine::Hip => cached_im2col(self.target)
                .and_then(|k| k.run_col2im_f32(src, d_col.shape(), input_shape, params)),
            #[cfg(test)]
            Engine::HostModel => Ok(crate::conv_pool_model::col2im(src, &plan)),
        }
        .map_err(map_conv_pool_error)?;
        Tensor::new(out, input_shape).map_err(shape_err)
    }

    /// `BackendOps::max_pool2d`（値と (n, c) 平面内の flat 索引）。
    pub(super) fn conv_pool_max_pool2d(
        &self,
        input: &Tensor<f32>,
        params: &Pool2dParams,
    ) -> Result<(Tensor<f32>, Tensor<i32>), BackendError> {
        let out_shape = pool2d_out_shape(input.shape(), params).map_err(shape_err)?;
        check_max_index_range(input.shape())?;
        if out_shape.contains(&0) {
            return Ok((
                Tensor::new(Vec::new(), &out_shape).map_err(shape_err)?,
                Tensor::new(Vec::new(), &out_shape).map_err(shape_err)?,
            ));
        }
        check_input_numel(input.shape())?;
        let in_shape = as_nchw(input.shape())?;
        let (kernel, stride, padding, dilation) = (
            params.kernel_size(),
            params.stride(),
            params.padding(),
            params.dilation(),
        );
        let plan = validate_max_pool(in_shape, kernel, stride, padding, dilation)
            .map_err(map_conv_pool_error)?;
        check_plan_shape("max_pool2d", &out_shape, &plan)?;
        let owned = input.contiguous();
        let src = slice_with_len("max_pool2d", &owned, plan.numel_in)?;
        let (values, indices) = match self.engine {
            Engine::Hip => cached_pooling(self.target).and_then(|k| {
                k.run_max_pool2d_f32(src, in_shape, kernel, stride, padding, dilation)
            }),
            #[cfg(test)]
            Engine::HostModel => Ok(crate::conv_pool_model::max_pool2d(
                src, &plan, kernel, stride, padding, dilation,
            )),
        }
        .map_err(map_conv_pool_error)?;
        Ok((
            Tensor::new(values, &out_shape).map_err(shape_err)?,
            Tensor::new(indices, &out_shape).map_err(shape_err)?,
        ))
    }

    /// `BackendOps::avg_pool2d`。dilation ≠ 1 は HIP に触れる前に `Unsupported`（カーネルが dilation を
    /// 持たない。CUDA PR #1888 と同じ）。
    pub(super) fn conv_pool_avg_pool2d(
        &self,
        input: &Tensor<f32>,
        params: &Pool2dParams,
        count_include_pad: bool,
    ) -> Result<Tensor<f32>, BackendError> {
        let out_shape = pool2d_out_shape(input.shape(), params).map_err(shape_err)?;
        if params.dilation() != [1, 1] {
            return Err(BackendError::Unsupported(format!(
                "avg_pool2d: ROCm カーネルは dilation != [1, 1] 非対応です（指定 {:?}）",
                params.dilation()
            )));
        }
        if out_shape.contains(&0) {
            return Tensor::new(Vec::new(), &out_shape).map_err(shape_err);
        }
        check_input_numel(input.shape())?;
        let in_shape = as_nchw(input.shape())?;
        let (kernel, stride, padding) = (params.kernel_size(), params.stride(), params.padding());
        let plan =
            validate_avg_pool(in_shape, kernel, stride, padding).map_err(map_conv_pool_error)?;
        check_plan_shape("avg_pool2d", &out_shape, &plan)?;
        let owned = input.contiguous();
        let src = slice_with_len("avg_pool2d", &owned, plan.numel_in)?;
        let values = match self.engine {
            Engine::Hip => cached_pooling(self.target).and_then(|k| {
                k.run_avg_pool2d_f32(src, in_shape, kernel, stride, padding, count_include_pad)
            }),
            #[cfg(test)]
            Engine::HostModel => Ok(crate::conv_pool_model::avg_pool2d(
                src,
                &plan,
                kernel,
                stride,
                padding,
                count_include_pad,
            )),
        }
        .map_err(map_conv_pool_error)?;
        Tensor::new(values, &out_shape).map_err(shape_err)
    }

    /// `BackendOps::adaptive_avg_pool2d`。
    pub(super) fn conv_pool_adaptive_avg_pool2d(
        &self,
        input: &Tensor<f32>,
        output_size: [usize; 2],
    ) -> Result<Tensor<f32>, BackendError> {
        let out_shape = adaptive_pool2d_out_shape(input.shape(), output_size).map_err(shape_err)?;
        if out_shape.contains(&0) {
            return Tensor::new(Vec::new(), &out_shape).map_err(shape_err);
        }
        check_input_numel(input.shape())?;
        let in_shape = as_nchw(input.shape())?;
        let plan = validate_adaptive_pool(in_shape, output_size).map_err(map_conv_pool_error)?;
        check_plan_shape("adaptive_avg_pool2d", &out_shape, &plan)?;
        let owned = input.contiguous();
        let src = slice_with_len("adaptive_avg_pool2d", &owned, plan.numel_in)?;
        let values = match self.engine {
            Engine::Hip => cached_pooling(self.target)
                .and_then(|k| k.run_adaptive_avg_pool2d_f32(src, in_shape, output_size)),
            #[cfg(test)]
            Engine::HostModel => Ok(crate::conv_pool_model::adaptive_avg_pool2d(src, &plan)),
        }
        .map_err(map_conv_pool_error)?;
        Tensor::new(values, &out_shape).map_err(shape_err)
    }
}

#[cfg(test)]
mod tests;
