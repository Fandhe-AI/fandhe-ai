//! `RocmBackendOps` の elementwise 一式の結線（イシュー #3433・分解案 R2）。
//!
//! 役割: `ops.rs` の `BackendOps` impl から 1 行で呼ばれ、(1) ブロードキャスト・shape 再検査・
//! contiguous 化、(2) **HIP に触れる前に `Unsupported` を確定**（対象外の scalar kind・allowlist 外の
//! 融合プラン。ROCm 非搭載環境で `DeviceUnavailable` を返すと autodiff の「`Unsupported` のときだけ
//! フォールバックする」契約が壊れる。PR #1781 の教訓）、(3) engine（Hip／テストの HostModel）の分岐、
//! (4) `Tensor`／`DeviceBuffer` の組み立て、を担う。カーネル起動そのものは `elementwise.rs` の実行型。
//!
//! # デバイス常駐 API の到達性（要報告事項）
//! `binary_elementwise_device`／`unary_elementwise_device` は実装するが、`memory_ops()` は R0 の方針
//! （被覆計画 §4・既定継承の `None`）のままなので、**crate の外から ROCm の `DeviceBuffer` を作れない**。
//! 判別は `downcast_handle::<RocmBufferHandle>()` の型だけで行う（`device()` は #3157 まで仮置きの
//! `Device::Cpu` で、CPU 由来のバッファと区別できない）。テストは `upload_resident`／
//! `download_resident`（`#[cfg(test)]`）で組み立てる。
//!
//! # 融合の opt-in ゲートは設けない
//! CUDA の融合は `AtomicBool` の opt-in ゲート（facade の Tier B 公開判断に結びつく）の背後にある。
//! 新しいプロセスワイドの setter は公開面の判断で自律的に決められず、ROCm はどの候補リストにも
//! 登録していないため、CPU 版と同じくゲートなしの allowlist にした。facade へ ROCm を露出する
//! #3157 の時点で所有者が揃えるか判断する。

use fandhe_ai_tensor_core::{
    BackendError, BackendOps, BinaryElementwiseOp, DType, DeviceBuffer, FusionPlan, ScalarBinaryOp,
    ScalarUnaryOp, ShapeError, Tensor, UnaryElementwiseOp, elementwise_out_shape,
};

use super::{EW_CACHE, Engine, FUSED_CACHE, RocmBackendOps, SCALAR_CACHE, cached};
use crate::buffer_handle::RocmBufferHandle;
use crate::elementwise::{
    BinaryKind, FusedProgram, RocmElementwise, RocmFusedElementwise, RocmScalarOp, UnaryKind,
    encode_program, fit_i32,
};
use crate::error::{RocmError, to_backend_error};
use crate::kernels_scalar_op::{binary_spec, unary_spec};
use crate::typed_buffer::checked_bytes;

const OP_EW: &str = "elementwise";

/// 契約済みの連続スライスを取り出す（`contiguous()` 直後は常に `Some`）。
fn slice_of(t: &Tensor<f32>) -> Result<&[f32], BackendError> {
    t.as_slice().ok_or_else(|| {
        BackendError::KernelLaunchFailed("elementwise: tensor not contiguous".into())
    })
}

/// 要素数を checked で求め、カーネル引数型（`i32`）への収容を `contiguous()` の前に検査する
/// （巨大 shape で確保に進んで panic・abort しないため）。
fn check_numel(op: &'static str, shape: &[usize]) -> Result<usize, BackendError> {
    let n = shape
        .iter()
        .try_fold(1usize, |acc, &s| acc.checked_mul(s))
        .ok_or(BackendError::ShapeMismatch(
            ShapeError::ElementCountOverflow,
        ))?;
    fit_i32(op, n, "numel").map_err(to_backend_error)?;
    Ok(n)
}

fn shape_mismatch(lhs: &[usize], rhs: &[usize]) -> BackendError {
    BackendError::ShapeMismatch(ShapeError::ShapeMismatch {
        lhs: lhs.to_vec(),
        rhs: rhs.to_vec(),
    })
}

fn handle_of(buf: &DeviceBuffer<f32>) -> Result<&RocmBufferHandle, BackendError> {
    buf.downcast_handle::<RocmBufferHandle>()
        .ok_or(BackendError::DeviceMismatch)
}

fn check_handle_len(h: &RocmBufferHandle, numel: usize) -> Result<(), BackendError> {
    if h.len() != numel {
        return Err(BackendError::ShapeMismatch(
            ShapeError::ElementCountMismatch {
                expected: numel,
                actual: h.len(),
            },
        ));
    }
    Ok(())
}

impl RocmBackendOps {
    // --- engine 分岐（Hip は実機、HostModel はテストのみ） ---

    fn host_binary(&self, kind: BinaryKind, a: &[f32], b: &[f32]) -> Result<Vec<f32>, RocmError> {
        match self.engine {
            Engine::Hip => {
                cached(&EW_CACHE, self.target, RocmElementwise::new)?.run_binary(kind, a, b)
            }
            #[cfg(test)]
            Engine::HostModel => Ok(crate::elementwise_model::binary(kind, a, b)),
        }
    }

    fn host_unary(&self, kind: UnaryKind, a: &[f32]) -> Result<Vec<f32>, RocmError> {
        match self.engine {
            Engine::Hip => cached(&EW_CACHE, self.target, RocmElementwise::new)?.run_unary(kind, a),
            #[cfg(test)]
            Engine::HostModel => Ok(crate::elementwise_model::unary(kind, a)),
        }
    }

    fn host_where(&self, c: &[f32], a: &[f32], b: &[f32]) -> Result<Vec<f32>, RocmError> {
        match self.engine {
            Engine::Hip => {
                cached(&EW_CACHE, self.target, RocmElementwise::new)?.run_where_f32(c, a, b)
            }
            #[cfg(test)]
            Engine::HostModel => Ok(crate::elementwise_model::where_cond(c, a, b)),
        }
    }

    fn host_masked_fill(&self, x: &[f32], m: &[f32], v: f32) -> Result<Vec<f32>, RocmError> {
        match self.engine {
            Engine::Hip => {
                cached(&EW_CACHE, self.target, RocmElementwise::new)?.run_masked_fill_f32(x, m, v)
            }
            #[cfg(test)]
            Engine::HostModel => Ok(crate::elementwise_model::masked_fill(x, m, v)),
        }
    }

    fn host_scalar_unary(&self, op: ScalarUnaryOp, a: &[f32]) -> Result<Vec<f32>, RocmError> {
        match self.engine {
            Engine::Hip => {
                cached(&SCALAR_CACHE, self.target, RocmScalarOp::new)?.run_unary_f32(op, a)
            }
            #[cfg(test)]
            Engine::HostModel => crate::elementwise_model::scalar_unary(op, a).ok_or_else(|| {
                RocmError::InvalidOpArgs {
                    op: "scalar_unary",
                    detail: format!("対象外の kind です: {op:?}"),
                }
            }),
        }
    }

    fn host_scalar_binary(
        &self,
        op: ScalarBinaryOp,
        a: &[f32],
        b: &[f32],
    ) -> Result<Vec<f32>, RocmError> {
        match self.engine {
            Engine::Hip => {
                cached(&SCALAR_CACHE, self.target, RocmScalarOp::new)?.run_binary_f32(op, a, b)
            }
            #[cfg(test)]
            Engine::HostModel => {
                crate::elementwise_model::scalar_binary(op, a, b).ok_or_else(|| {
                    RocmError::InvalidOpArgs {
                        op: "scalar_binary",
                        detail: format!("対象外の kind です: {op:?}"),
                    }
                })
            }
        }
    }

    fn host_fused(
        &self,
        program: &FusedProgram,
        leaves: &[f32],
        numel: usize,
    ) -> Result<Vec<f32>, RocmError> {
        match self.engine {
            Engine::Hip => cached(&FUSED_CACHE, self.target, RocmFusedElementwise::new)?
                .run_encoded(program, leaves, numel),
            #[cfg(test)]
            Engine::HostModel => Ok(crate::fused_elementwise_model::simulate(
                &program.words,
                program.n_ops as i32,
                program.leaf_count as i32,
                leaves,
                numel,
            )),
        }
    }

    // --- ホスト `Tensor` 入出力 ---

    /// `add`／`mul`（NumPy 互換ブロードキャスト）。
    pub(super) fn ew_binary(
        &self,
        kind: BinaryKind,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
    ) -> Result<Tensor<f32>, BackendError> {
        let shape =
            elementwise_out_shape(a.shape(), b.shape()).map_err(BackendError::ShapeMismatch)?;
        let n = check_numel(OP_EW, &shape)?;
        let (ab, bb) = a.broadcast_with(b).map_err(BackendError::ShapeMismatch)?;
        let (ac, bc) = (ab.contiguous(), bb.contiguous());
        let out = if n == 0 {
            Vec::new() // HIP を触らない。
        } else {
            self.host_binary(kind, slice_of(&ac)?, slice_of(&bc)?)
                .map_err(to_backend_error)?
        };
        Tensor::new(out, &shape).map_err(BackendError::ShapeMismatch)
    }

    /// `relu`／`exp`／`tanh`。
    pub(super) fn ew_unary(
        &self,
        kind: UnaryKind,
        a: &Tensor<f32>,
    ) -> Result<Tensor<f32>, BackendError> {
        let n = check_numel(OP_EW, a.shape())?;
        let ac = a.contiguous();
        let out = if n == 0 {
            Vec::new()
        } else {
            self.host_unary(kind, slice_of(&ac)?)
                .map_err(to_backend_error)?
        };
        Tensor::new(out, a.shape()).map_err(BackendError::ShapeMismatch)
    }

    /// `scalar_unary`。対象外 kind は HIP に触れる前に `Unsupported`。
    pub(super) fn ew_scalar_unary(
        &self,
        op: ScalarUnaryOp,
        a: &Tensor<f32>,
    ) -> Result<Tensor<f32>, BackendError> {
        if unary_spec(op).is_none() {
            return Err(BackendError::Unsupported(format!(
                "scalar_unary: {op:?} は ROCm カーネル未対応（ホスト参照実装へフォールバックする）"
            )));
        }
        let n = check_numel("scalar_unary", a.shape())?;
        let ac = a.contiguous();
        let out = if n == 0 {
            Vec::new()
        } else {
            self.host_scalar_unary(op, slice_of(&ac)?)
                .map_err(to_backend_error)?
        };
        Tensor::new(out, a.shape()).map_err(BackendError::ShapeMismatch)
    }

    /// `scalar_binary`（ブロードキャスト対応）。対象外 kind は HIP に触れる前に `Unsupported`。
    pub(super) fn ew_scalar_binary(
        &self,
        op: ScalarBinaryOp,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
    ) -> Result<Tensor<f32>, BackendError> {
        if binary_spec(op).is_none() {
            return Err(BackendError::Unsupported(format!(
                "scalar_binary: {op:?} は ROCm カーネル未対応（ホスト参照実装へフォールバックする）"
            )));
        }
        let shape =
            elementwise_out_shape(a.shape(), b.shape()).map_err(BackendError::ShapeMismatch)?;
        let n = check_numel("scalar_binary", &shape)?;
        let (ab, bb) = a.broadcast_with(b).map_err(BackendError::ShapeMismatch)?;
        let (ac, bc) = (ab.contiguous(), bb.contiguous());
        let out = if n == 0 {
            Vec::new()
        } else {
            self.host_scalar_binary(op, slice_of(&ac)?, slice_of(&bc)?)
                .map_err(to_backend_error)?
        };
        Tensor::new(out, &shape).map_err(BackendError::ShapeMismatch)
    }

    /// `where_cond`。`cond`・`a`・`b` の shape 一致を再検査する（fail-closed）。
    pub(super) fn ew_where(
        &self,
        cond: &Tensor<f32>,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
    ) -> Result<Tensor<f32>, BackendError> {
        if cond.shape() != a.shape() {
            return Err(shape_mismatch(cond.shape(), a.shape()));
        }
        if b.shape() != a.shape() {
            return Err(shape_mismatch(b.shape(), a.shape()));
        }
        let n = check_numel("where_cond", a.shape())?;
        let (cc, ac, bc) = (cond.contiguous(), a.contiguous(), b.contiguous());
        let out = if n == 0 {
            Vec::new()
        } else {
            self.host_where(slice_of(&cc)?, slice_of(&ac)?, slice_of(&bc)?)
                .map_err(to_backend_error)?
        };
        Tensor::new(out, a.shape()).map_err(BackendError::ShapeMismatch)
    }

    /// `masked_fill`。`x`・`mask` の shape 一致を再検査する。
    pub(super) fn ew_masked_fill(
        &self,
        x: &Tensor<f32>,
        mask: &Tensor<f32>,
        value: f32,
    ) -> Result<Tensor<f32>, BackendError> {
        if mask.shape() != x.shape() {
            return Err(shape_mismatch(mask.shape(), x.shape()));
        }
        let n = check_numel("masked_fill", x.shape())?;
        let (xc, mc) = (x.contiguous(), mask.contiguous());
        let out = if n == 0 {
            Vec::new()
        } else {
            self.host_masked_fill(slice_of(&xc)?, slice_of(&mc)?, value)
                .map_err(to_backend_error)?
        };
        Tensor::new(out, x.shape()).map_err(BackendError::ShapeMismatch)
    }

    /// `run_fused`（elementwise の範囲）。allowlist 外は HIP に触れる前に `Unsupported`。
    pub(super) fn ew_run_fused(
        &self,
        plan: &FusionPlan,
        leaves: &[&Tensor<f32>],
    ) -> Result<Tensor<f32>, BackendError> {
        // 1) 静的に確定できる拒否（呼び出し側が演算ごとの実行へフォールバックする）。
        let program = encode_program(plan)
            .map_err(|why| BackendError::Unsupported(format!("run_fused: {why}")))?;
        if plan.dtype() != DType::F32 {
            return Err(BackendError::Unsupported(format!(
                "run_fused: unsupported dtype {:?} (F32 elementwise only)",
                plan.dtype()
            )));
        }
        // 2) 入力の整合性（不一致は ShapeMismatch。fail-closed）。
        if leaves.len() != plan.leaf_count() {
            return Err(BackendError::ShapeMismatch(
                ShapeError::ElementCountMismatch {
                    expected: plan.leaf_count(),
                    actual: leaves.len(),
                },
            ));
        }
        let out_shape = plan.output_shape();
        for leaf in leaves {
            if leaf.shape() != out_shape {
                return Err(shape_mismatch(out_shape, leaf.shape()));
            }
        }
        let numel = check_numel("fused_elementwise", out_shape)?;
        if numel == 0 {
            return Tensor::new(Vec::new(), out_shape).map_err(BackendError::ShapeMismatch);
        }
        // 3) 全 leaf を連結（`[leaf_count * numel]`）。確保サイズは checked で求める。
        let total = leaves
            .len()
            .checked_mul(numel)
            .ok_or(BackendError::ShapeMismatch(
                ShapeError::ElementCountOverflow,
            ))?;
        checked_bytes::<f32>(total).map_err(to_backend_error)?;
        let mut flat = Vec::with_capacity(total);
        for leaf in leaves {
            let c = leaf.contiguous();
            flat.extend_from_slice(slice_of(&c)?);
        }
        let out = self
            .host_fused(&program, &flat, numel)
            .map_err(to_backend_error)?;
        Tensor::new(out, out_shape).map_err(BackendError::ShapeMismatch)
    }

    // --- デバイス常駐 ---

    fn wrap_handle(&self, shape: &[usize], h: RocmBufferHandle) -> DeviceBuffer<f32> {
        DeviceBuffer::new(self.device(), shape.to_vec(), Box::new(h))
    }

    fn device_binary(
        &self,
        kind: BinaryKind,
        a: &RocmBufferHandle,
        b: &RocmBufferHandle,
    ) -> Result<RocmBufferHandle, BackendError> {
        match (self.engine, a, b) {
            (Engine::Hip, RocmBufferHandle::Hip(x), RocmBufferHandle::Hip(y)) => {
                let ew = cached(&EW_CACHE, self.target, RocmElementwise::new)
                    .map_err(to_backend_error)?;
                let out = ew.binary_resident(kind, x, y).map_err(to_backend_error)?;
                Ok(RocmBufferHandle::from(out))
            }
            #[cfg(test)]
            (Engine::HostModel, RocmBufferHandle::Host(x), RocmBufferHandle::Host(y)) => Ok(
                RocmBufferHandle::Host(crate::elementwise_model::binary(kind, x, y)),
            ),
            // engine とハンドル種別の取り違え（Hip engine に Host ハンドル等）は fail-closed。
            _ => Err(BackendError::DeviceMismatch),
        }
    }

    fn device_unary(
        &self,
        kind: UnaryKind,
        a: &RocmBufferHandle,
    ) -> Result<RocmBufferHandle, BackendError> {
        match (self.engine, a) {
            (Engine::Hip, RocmBufferHandle::Hip(x)) => {
                let ew = cached(&EW_CACHE, self.target, RocmElementwise::new)
                    .map_err(to_backend_error)?;
                let out = ew.unary_resident(kind, x).map_err(to_backend_error)?;
                Ok(RocmBufferHandle::from(out))
            }
            #[cfg(test)]
            (Engine::HostModel, RocmBufferHandle::Host(x)) => Ok(RocmBufferHandle::Host(
                crate::elementwise_model::unary(kind, x),
            )),
            _ => Err(BackendError::DeviceMismatch),
        }
    }

    /// `binary_elementwise_device`。`a`・`b` は shape 完全一致限定（ブロードキャスト非対応）。
    pub(super) fn ew_binary_device(
        &self,
        op: BinaryElementwiseOp,
        a: &DeviceBuffer<f32>,
        b: &DeviceBuffer<f32>,
    ) -> Result<DeviceBuffer<f32>, BackendError> {
        let kind = match op {
            BinaryElementwiseOp::Add => BinaryKind::Add,
            BinaryElementwiseOp::Mul => BinaryKind::Mul,
            // `#[non_exhaustive]`。黙って恒等的な値を返さず明示的に拒否する。
            _ => {
                return Err(BackendError::Unsupported(format!(
                    "binary_elementwise_device: unsupported op {op:?}"
                )));
            }
        };
        // `device()` の比較では CPU 由来のバッファと区別できないため、ハンドル型の downcast で判別する。
        let (ha, hb) = (handle_of(a)?, handle_of(b)?);
        if a.shape() != b.shape() {
            return Err(shape_mismatch(a.shape(), b.shape()));
        }
        let numel = a.numel();
        check_handle_len(ha, numel)?;
        check_handle_len(hb, numel)?;
        let out = if numel == 0 {
            RocmBufferHandle::Empty // HIP を触らない。
        } else {
            self.device_binary(kind, ha, hb)?
        };
        Ok(self.wrap_handle(a.shape(), out))
    }

    /// `unary_elementwise_device`。
    pub(super) fn ew_unary_device(
        &self,
        op: UnaryElementwiseOp,
        a: &DeviceBuffer<f32>,
    ) -> Result<DeviceBuffer<f32>, BackendError> {
        let kind = match op {
            UnaryElementwiseOp::Relu => UnaryKind::Relu,
            UnaryElementwiseOp::Exp => UnaryKind::Exp,
            UnaryElementwiseOp::Tanh => UnaryKind::Tanh,
            _ => {
                return Err(BackendError::Unsupported(format!(
                    "unary_elementwise_device: unsupported op {op:?}"
                )));
            }
        };
        let ha = handle_of(a)?;
        let numel = a.numel();
        check_handle_len(ha, numel)?;
        let out = if numel == 0 {
            RocmBufferHandle::Empty
        } else {
            self.device_unary(kind, ha)?
        };
        Ok(self.wrap_handle(a.shape(), out))
    }

    /// テスト専用: ホストの `Tensor` からデバイス常駐バッファを組み立てる。
    ///
    /// `memory_ops()` を `None` のまま変えない（被覆計画 §4 の既定継承）ため、本番経路には
    /// ROCm の `DeviceBuffer` を作る入口がない。
    #[cfg(test)]
    pub(crate) fn upload_resident(
        &self,
        t: &Tensor<f32>,
    ) -> Result<DeviceBuffer<f32>, BackendError> {
        let c = t.contiguous();
        let s = slice_of(&c)?;
        let h = if s.is_empty() {
            RocmBufferHandle::Empty
        } else {
            match self.engine {
                Engine::Hip => {
                    let ew = cached(&EW_CACHE, self.target, RocmElementwise::new)
                        .map_err(to_backend_error)?;
                    RocmBufferHandle::Hip(ew.upload(s).map_err(to_backend_error)?)
                }
                Engine::HostModel => RocmBufferHandle::Host(s.to_vec()),
            }
        };
        Ok(self.wrap_handle(t.shape(), h))
    }

    /// テスト専用: デバイス常駐バッファをホストの `Tensor` へ読み戻す。
    #[cfg(test)]
    pub(crate) fn download_resident(
        &self,
        b: &DeviceBuffer<f32>,
    ) -> Result<Tensor<f32>, BackendError> {
        let v = match handle_of(b)? {
            RocmBufferHandle::Empty => Vec::new(),
            RocmBufferHandle::Hip(x) => x.to_vec().map_err(to_backend_error)?,
            RocmBufferHandle::Host(v) => v.clone(),
        };
        Tensor::new(v, b.shape()).map_err(BackendError::ShapeMismatch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels_scalar_op::tests::{supported_binary, supported_unary};
    use crate::target::GfxTarget;
    use fandhe_ai_backend_cpu::{CpuBackendOps, assert_parity};
    use fandhe_ai_tensor_core::{FusedOpKind, ScalarOpKind};

    fn ops() -> RocmBackendOps {
        RocmBackendOps::with_host_model(GfxTarget::Gfx942)
    }

    /// `[lo, hi]` の決定的データ（`-0.0` を含まない）。
    fn data(len: usize, seed: u32, lo: f32, hi: f32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let r = ((i as u32).wrapping_mul(2654435761).wrapping_add(seed) >> 8) % 10_001;
                let v = lo + (hi - lo) * (r as f32 / 10_000.0);
                if v == 0.0 { 0.25 } else { v }
            })
            .collect()
    }

    fn t(len: usize, shape: &[usize], seed: u32) -> Tensor<f32> {
        Tensor::new(data(len, seed, -1.0, 1.0), shape).unwrap()
    }

    fn pos(len: usize, shape: &[usize], seed: u32) -> Tensor<f32> {
        Tensor::new(data(len, seed, 0.1, 2.1), shape).unwrap()
    }

    fn bits(x: &Tensor<f32>) -> Vec<u32> {
        x.contiguous()
            .as_slice()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect()
    }

    fn same(a: &Tensor<f32>, b: &Tensor<f32>) {
        assert_eq!(a.shape(), b.shape());
        assert_eq!(bits(a), bits(b));
    }

    fn close(label: &str, a: &Tensor<f32>, b: &Tensor<f32>) {
        assert_eq!(a.shape(), b.shape(), "{label}");
        assert_parity(
            label,
            a.contiguous().as_slice().unwrap(),
            b.contiguous().as_slice().unwrap(),
        );
    }

    #[test]
    fn add_mul_match_cpu_bitwise_across_sizes_and_broadcast() {
        let (o, c) = (ops(), CpuBackendOps);
        for n in [1usize, 255, 256, 257, 1000] {
            let (a, b) = (t(n, &[n], 1), t(n, &[n], 2));
            same(&o.add(&a, &b).unwrap(), &c.add(&a, &b).unwrap());
            same(&o.mul(&a, &b).unwrap(), &c.mul(&a, &b).unwrap());
        }
        let a = t(12, &[3, 4], 1);
        for bshape in [vec![1, 4], vec![4], vec![3, 1], vec![1]] {
            let len: usize = bshape.iter().product();
            let b = t(len, &bshape, 2);
            same(&o.add(&a, &b).unwrap(), &c.add(&a, &b).unwrap());
            same(&o.mul(&a, &b).unwrap(), &c.mul(&a, &b).unwrap());
            same(&o.add(&b, &a).unwrap(), &c.add(&b, &a).unwrap());
        }
        // 非 contiguous 入力。
        let x = t(12, &[4, 3], 3).transpose_2d().unwrap();
        same(&o.add(&x, &a).unwrap(), &c.add(&x, &a).unwrap());
        // ブロードキャスト不能。
        let bad = t(6, &[2, 3], 1);
        assert!(matches!(
            o.add(&a, &bad),
            Err(BackendError::ShapeMismatch(_))
        ));
    }

    #[test]
    fn unary_kernels_match_cpu() {
        let (o, c) = (ops(), CpuBackendOps);
        for n in [1usize, 255, 256, 257, 1000] {
            let a = t(n, &[n], 7);
            // relu は三項演算（CUDA と同じ）。データに -0.0 を含めないので CPU と bit 一致。
            same(&o.relu(&a).unwrap(), &c.relu(&a).unwrap());
            // exp・tanh はホスト libm のモデル（構造の検証）。parity は複合判定。
            close("exp", &o.exp(&a).unwrap(), &c.exp(&a).unwrap());
            close("tanh", &o.tanh(&a).unwrap(), &c.tanh(&a).unwrap());
        }
        // NaN は relu で 0（CPU の f32::max と同じ）。
        let nan = Tensor::new(vec![f32::NAN, -2.0, 3.0], &[3]).unwrap();
        same(&o.relu(&nan).unwrap(), &c.relu(&nan).unwrap());
    }

    #[test]
    fn empty_inputs_do_not_touch_hip() {
        // Hip engine（ROCm 非搭載でも）で numel == 0 は Ok。
        let o = RocmBackendOps::new(GfxTarget::Gfx942);
        let e = Tensor::<f32>::zeros(&[0, 3]).unwrap();
        assert_eq!(o.add(&e, &e).unwrap().shape(), &[0, 3]);
        assert_eq!(o.mul(&e, &e).unwrap().shape(), &[0, 3]);
        assert_eq!(o.relu(&e).unwrap().shape(), &[0, 3]);
        assert_eq!(o.exp(&e).unwrap().shape(), &[0, 3]);
        assert_eq!(o.tanh(&e).unwrap().shape(), &[0, 3]);
        assert_eq!(o.where_cond(&e, &e, &e).unwrap().shape(), &[0, 3]);
        assert_eq!(o.masked_fill(&e, &e, 1.0).unwrap().shape(), &[0, 3]);
        assert_eq!(
            o.scalar_unary(ScalarUnaryOp::Sqrt, &e).unwrap().shape(),
            &[0, 3]
        );
        assert_eq!(
            o.scalar_binary(ScalarBinaryOp::Sub, &e, &e)
                .unwrap()
                .shape(),
            &[0, 3]
        );
    }

    #[test]
    fn hip_engine_without_rocm_is_unavailable_for_kernels() {
        if crate::is_available() {
            return;
        }
        let o = RocmBackendOps::new(GfxTarget::Gfx942);
        let a = t(4, &[4], 1);
        for r in [
            o.add(&a, &a),
            o.mul(&a, &a),
            o.relu(&a),
            o.exp(&a),
            o.tanh(&a),
            o.where_cond(&a, &a, &a),
            o.masked_fill(&a, &a, 1.0),
            o.scalar_unary(ScalarUnaryOp::Sqrt, &a),
            o.scalar_binary(ScalarBinaryOp::Sub, &a, &a),
        ] {
            assert!(matches!(r, Err(BackendError::DeviceUnavailable(_))));
        }
    }

    #[test]
    fn scalar_unary_supported_kinds_match_cpu() {
        let (o, c) = (ops(), CpuBackendOps);
        // CPU と bit 一致する（選択・単一演算のみの）kind。
        let exact = ["neg", "abs", "sqrt", "clamp", "leaky_relu", "hardswish"];
        for n in [1usize, 255, 257, 1000] {
            let signed = t(n, &[n], 11);
            let positive = pos(n, &[n], 12);
            for op in supported_unary() {
                let x = match op {
                    ScalarUnaryOp::Sqrt
                    | ScalarUnaryOp::Log
                    | ScalarUnaryOp::Log2
                    | ScalarUnaryOp::Log10 => &positive,
                    _ => &signed,
                };
                let got = o.scalar_unary(op, x).unwrap();
                let want = c.scalar_unary(op, x).unwrap();
                if exact.contains(&op.kind_name()) {
                    same(&got, &want);
                } else {
                    close(op.kind_name(), &got, &want);
                }
            }
        }
    }

    #[test]
    fn scalar_unary_special_values() {
        let (o, c) = (ops(), CpuBackendOps);
        let x = Tensor::new(vec![f32::NAN, -1.0, 0.0, 2.0, f32::INFINITY], &[5]).unwrap();
        for op in [
            ScalarUnaryOp::Clamp {
                min: -0.5,
                max: 1.0,
            },
            // min > max は常に max。
            ScalarUnaryOp::Clamp { min: 2.0, max: 1.0 },
            ScalarUnaryOp::Neg,
            ScalarUnaryOp::Abs,
        ] {
            let (g, w) = (
                o.scalar_unary(op, &x).unwrap(),
                c.scalar_unary(op, &x).unwrap(),
            );
            for (a, b) in bits(&g).iter().zip(bits(&w)) {
                // NaN は payload が処理系依存のためクラス一致で見る。
                assert!(
                    *a == b || (f32::from_bits(*a).is_nan() && f32::from_bits(b).is_nan()),
                    "{op:?}"
                );
            }
        }
    }

    #[test]
    fn scalar_binary_supported_kinds_match_cpu_with_broadcast() {
        let (o, c) = (ops(), CpuBackendOps);
        let exact = ["sub", "div", "gt", "ge", "lt", "le", "eq", "ne"];
        let a = pos(12, &[3, 4], 21);
        for bshape in [vec![3, 4], vec![1, 4], vec![4]] {
            let len: usize = bshape.iter().product();
            let b = pos(len, &bshape, 22);
            for op in supported_binary() {
                let got = o.scalar_binary(op, &a, &b).unwrap();
                let want = c.scalar_binary(op, &a, &b).unwrap();
                if exact.contains(&op.kind_name()) {
                    same(&got, &want);
                } else {
                    close(op.kind_name(), &got, &want);
                }
            }
        }
        // 等値比較で等しい要素を含む。
        let x = Tensor::new(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let y = Tensor::new(vec![1.0, 0.0, 3.0], &[3]).unwrap();
        same(
            &o.scalar_binary(ScalarBinaryOp::Eq, &x, &y).unwrap(),
            &c.scalar_binary(ScalarBinaryOp::Eq, &x, &y).unwrap(),
        );
    }

    #[test]
    fn unsupported_scalar_kinds_are_unsupported_before_touching_hip() {
        // Hip engine（非搭載）でも `DeviceUnavailable` ではなく `Unsupported` になる（PR #1781 の教訓）。
        for o in [ops(), RocmBackendOps::new(GfxTarget::Gfx942)] {
            let a = pos(4, &[4], 1);
            for op in [
                ScalarUnaryOp::Relu,
                ScalarUnaryOp::Floor,
                ScalarUnaryOp::Erf,
                ScalarUnaryOp::Atan,
                ScalarUnaryOp::PowScalar { exponent: 2.0 },
            ] {
                assert!(
                    matches!(o.scalar_unary(op, &a), Err(BackendError::Unsupported(_))),
                    "{op:?}"
                );
            }
            for op in [
                ScalarBinaryOp::Atan2,
                ScalarBinaryOp::Add,
                ScalarBinaryOp::Maximum,
            ] {
                assert!(
                    matches!(
                        o.scalar_binary(op, &a, &a),
                        Err(BackendError::Unsupported(_))
                    ),
                    "{op:?}"
                );
            }
        }
    }

    #[test]
    fn where_and_masked_fill_match_cpu_and_check_shapes() {
        let (o, c) = (ops(), CpuBackendOps);
        for n in [1usize, 255, 256, 257, 1000] {
            let mask = Tensor::new(
                (0..n).map(|i| if i % 3 == 0 { 1.0 } else { 0.0 }).collect(),
                &[n],
            )
            .unwrap();
            let (a, b) = (t(n, &[n], 31), t(n, &[n], 32));
            same(
                &o.where_cond(&mask, &a, &b).unwrap(),
                &c.where_cond(&mask, &a, &b).unwrap(),
            );
            same(
                &o.masked_fill(&a, &mask, -9.5).unwrap(),
                &c.masked_fill(&a, &mask, -9.5).unwrap(),
            );
        }
        let (a, b, m) = (t(6, &[2, 3], 1), t(6, &[2, 3], 2), t(6, &[3, 2], 3));
        assert!(matches!(
            o.where_cond(&m, &a, &b),
            Err(BackendError::ShapeMismatch(_))
        ));
        assert!(matches!(
            o.where_cond(&a, &a, &m),
            Err(BackendError::ShapeMismatch(_))
        ));
        assert!(matches!(
            o.masked_fill(&a, &m, 1.0),
            Err(BackendError::ShapeMismatch(_))
        ));
    }

    #[test]
    fn device_resident_matches_host_bitwise() {
        let o = ops();
        let (a, b) = (t(12, &[3, 4], 41), t(12, &[3, 4], 42));
        let (da, db) = (
            o.upload_resident(&a).unwrap(),
            o.upload_resident(&b).unwrap(),
        );
        for (op, host) in [
            (BinaryElementwiseOp::Add, o.add(&a, &b).unwrap()),
            (BinaryElementwiseOp::Mul, o.mul(&a, &b).unwrap()),
        ] {
            let out = o.binary_elementwise_device(op, &da, &db).unwrap();
            assert_eq!(out.shape(), &[3, 4]);
            same(&o.download_resident(&out).unwrap(), &host);
        }
        for (op, host) in [
            (UnaryElementwiseOp::Relu, o.relu(&a).unwrap()),
            (UnaryElementwiseOp::Exp, o.exp(&a).unwrap()),
            (UnaryElementwiseOp::Tanh, o.tanh(&a).unwrap()),
        ] {
            let out = o.unary_elementwise_device(op, &da).unwrap();
            same(&o.download_resident(&out).unwrap(), &host);
        }
    }

    #[test]
    fn device_resident_checks_shapes_and_device() {
        let o = ops();
        let (a, b) = (t(6, &[2, 3], 1), t(6, &[3, 2], 2));
        let (da, db) = (
            o.upload_resident(&a).unwrap(),
            o.upload_resident(&b).unwrap(),
        );
        assert!(matches!(
            o.binary_elementwise_device(BinaryElementwiseOp::Add, &da, &db),
            Err(BackendError::ShapeMismatch(_))
        ));
        // 他バックエンド（CpuMemory）のバッファは `device()` が同じ Cpu でも DeviceMismatch。
        let cpu_buf = CpuBackendOps.memory_ops().unwrap().upload(&a).unwrap();
        assert!(matches!(
            o.binary_elementwise_device(BinaryElementwiseOp::Add, &cpu_buf, &da),
            Err(BackendError::DeviceMismatch)
        ));
        assert!(matches!(
            o.unary_elementwise_device(UnaryElementwiseOp::Relu, &cpu_buf),
            Err(BackendError::DeviceMismatch)
        ));
        // Hip engine に Host ハンドルを渡すと DeviceMismatch（engine とハンドルの取り違え）。
        let hip = RocmBackendOps::new(GfxTarget::Gfx942);
        assert!(matches!(
            hip.unary_elementwise_device(UnaryElementwiseOp::Relu, &da),
            Err(BackendError::DeviceMismatch)
        ));
    }

    #[test]
    fn device_resident_empty_does_not_touch_hip() {
        let hip = RocmBackendOps::new(GfxTarget::Gfx942);
        let e = Tensor::<f32>::zeros(&[0, 2]).unwrap();
        let de = hip.upload_resident(&e).unwrap();
        let out = hip
            .binary_elementwise_device(BinaryElementwiseOp::Add, &de, &de)
            .unwrap();
        assert_eq!(out.shape(), &[0, 2]);
        let out = hip
            .unary_elementwise_device(UnaryElementwiseOp::Exp, &de)
            .unwrap();
        assert_eq!(out.numel(), 0);
        assert_eq!(hip.download_resident(&out).unwrap().shape(), &[0, 2]);
    }

    // --- run_fused ---

    fn plan(ops: Vec<FusedOpKind>, shape: &[usize], leaf_count: usize) -> FusionPlan {
        FusionPlan::from_ops(ops, shape.to_vec(), DType::F32, leaf_count).unwrap()
    }

    #[test]
    fn run_fused_matches_cpu() {
        use FusedOpKind::*;
        let (o, c) = (ops(), CpuBackendOps);
        let cases: Vec<(Vec<FusedOpKind>, usize)> = vec![
            (
                vec![
                    Input { leaf_index: 0 },
                    Input { leaf_index: 1 },
                    Add { lhs: 0, rhs: 1 },
                    Relu { input: 2 },
                    Exp { input: 3 },
                    Tanh { input: 4 },
                ],
                2,
            ),
            // Mul と fan-in（葉 0 を複数回参照）。
            (
                vec![
                    Input { leaf_index: 0 },
                    Input { leaf_index: 1 },
                    Mul { lhs: 0, rhs: 1 },
                    Add { lhs: 2, rhs: 0 },
                    Mul { lhs: 3, rhs: 3 },
                ],
                2,
            ),
        ];
        for n in [1usize, 255, 256, 257, 1000] {
            for (ops_list, leaf_count) in &cases {
                let p = plan(ops_list.clone(), &[n], *leaf_count);
                let leaves: Vec<Tensor<f32>> = (0..*leaf_count)
                    .map(|i| t(n, &[n], 50 + i as u32))
                    .collect();
                let refs: Vec<&Tensor<f32>> = leaves.iter().collect();
                close(
                    "run_fused",
                    &o.run_fused(&p, &refs).unwrap(),
                    &c.run_fused(&p, &refs).unwrap(),
                );
            }
        }
    }

    #[test]
    fn run_fused_max_plan_matches_cpu() {
        use FusedOpKind::*;
        let (o, c) = (ops(), CpuBackendOps);
        // 演算ノード 12 個・葉 13 個（上限ちょうど）。
        let mut list: Vec<FusedOpKind> = (0..13).map(|l| Input { leaf_index: l }).collect();
        let mut prev = 0usize;
        for l in 1..13 {
            list.push(Add { lhs: prev, rhs: l });
            prev = list.len() - 1;
        }
        let p = plan(list, &[8], 13);
        let leaves: Vec<Tensor<f32>> = (0..13).map(|i| t(8, &[8], 70 + i)).collect();
        let refs: Vec<&Tensor<f32>> = leaves.iter().collect();
        close(
            "run_fused max",
            &o.run_fused(&p, &refs).unwrap(),
            &c.run_fused(&p, &refs).unwrap(),
        );
    }

    #[test]
    fn run_fused_rejects_non_allowlisted_plans_as_unsupported_before_hip() {
        use FusedOpKind::*;
        let unsupported = [
            vec![
                Input { leaf_index: 0 },
                Input { leaf_index: 1 },
                Sub { lhs: 0, rhs: 1 },
            ],
            vec![Input { leaf_index: 0 }, Rsqrt { input: 0 }],
            vec![
                Input { leaf_index: 0 },
                Sum {
                    input: 0,
                    axis: None,
                },
            ],
        ];
        for o in [ops(), RocmBackendOps::new(GfxTarget::Gfx942)] {
            for list in &unsupported {
                let leaf_count = list.iter().filter(|op| matches!(op, Input { .. })).count();
                let p = plan(list.clone(), &[4], leaf_count);
                let leaves: Vec<Tensor<f32>> =
                    (0..leaf_count).map(|i| t(4, &[4], i as u32)).collect();
                let refs: Vec<&Tensor<f32>> = leaves.iter().collect();
                assert!(matches!(
                    o.run_fused(&p, &refs),
                    Err(BackendError::Unsupported(_))
                ));
            }
        }
    }

    #[test]
    fn run_fused_validates_leaves() {
        use FusedOpKind::*;
        let o = ops();
        let p = plan(
            vec![
                Input { leaf_index: 0 },
                Input { leaf_index: 1 },
                Add { lhs: 0, rhs: 1 },
            ],
            &[4],
            2,
        );
        let (a, b) = (t(4, &[4], 1), t(4, &[4], 2));
        // 個数不一致。
        assert!(matches!(
            o.run_fused(&p, &[&a]),
            Err(BackendError::ShapeMismatch(_))
        ));
        // shape 不一致。
        let wrong = t(6, &[6], 3);
        assert!(matches!(
            o.run_fused(&p, &[&a, &wrong]),
            Err(BackendError::ShapeMismatch(_))
        ));
        assert!(o.run_fused(&p, &[&a, &b]).is_ok());
    }

    #[test]
    fn run_fused_empty_output_is_ok_without_hip() {
        use FusedOpKind::*;
        let hip = RocmBackendOps::new(GfxTarget::Gfx942);
        let p = plan(vec![Input { leaf_index: 0 }, Relu { input: 0 }], &[0, 2], 1);
        let e = Tensor::<f32>::zeros(&[0, 2]).unwrap();
        assert_eq!(hip.run_fused(&p, &[&e]).unwrap().shape(), &[0, 2]);
    }

    #[test]
    fn kind_names_used_for_kernels_exist() {
        // カーネル名の導出元（`kind_name`）が仕様表と一致する（ドリフト検出）。
        assert_eq!(ScalarUnaryOp::GeluTanh.kind_name(), "gelu_tanh");
        assert_eq!(ScalarBinaryOp::Sub.kind_name(), "sub");
    }
}
