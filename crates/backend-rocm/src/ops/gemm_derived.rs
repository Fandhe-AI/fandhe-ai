//! `RocmBackendOps` の GEMM 派生 4 メソッドの結線（イシュー #3442・分解案 R11）。
//!
//! 対象: `gemm_resident_rhs`・`gemm_resident_lhs`・`gemm_fp32_strict_into`・`linear_forward_device`。
//! 役割: `ops.rs` の `BackendOps` impl から 1 行で呼ばれ、(1) HIP に触れる前の形状・ハンドル検査
//! （`DeviceMismatch`／`ShapeMismatch`／`InvalidArgument`。ROCm 非搭載環境でも型付きエラーを検査できる
//! ように、`DeviceUnavailable` より前に確定する。PR #1781 の教訓・`ops/ew.rs` と同じ）、(2) 退化形状
//! （空・`k == 0`）の早期リターン、(3) engine（Hip／テストの HostModel）の分岐、(4) `Tensor`／
//! `DeviceBuffer` の組み立て、を担う。カーネル起動そのものは `gemm_bias_act.rs` の `RocmGemmBiasAct`。
//!
//! # 呼び出し元と CUDA との対応
//! 呼び出し元は autodiff の `DeviceParamStore`（resident forward／backward・勾配の staging 直書き・
//! 推論チェーン）。CUDA 版 `backend-cuda::ops` の同名メソッドと同じ検査順序・同じエラー種別にした。
//! 4 メソッドとも**ホストへ黙ってフォールバックしない**（常駐オペランドを download して委譲すると
//! trait doc の D2H 排除契約を破る。security.md A08）。`Unsupported` は未知の `Activation` だけ。
//!
//! # 継承するメソッド
//! `gemm_resident_rhs_act` は既定（`gemm_resident_rhs` → `relu`）のまま継承する。`*_tracked` と
//! `gemm_fp32_strict_into_with_bias_reduce_tracked` も既定のまま: ROCm は呼び出しごとに同期するので
//! 遅延失敗トークンを要さない（CPU／CUDA と同じ判断）。`gemm_bias_act` の専用化は行わない
//! （既定の合成で正しい。融合は性能だけの差で、被覆計画 §9 の R11 記録に申し送る）。
//!
//! # 到達性
//! `memory_ops()` は R0 の方針で `None` のままなので、crate の外から ROCm の `DeviceBuffer` は作れない
//! （`ops/ew.rs` の R2 と同じ制約。#3157）。判別は `downcast_handle::<RocmBufferHandle>()` の型だけで
//! 行う。テストは `upload_resident`／`download_resident` で組み立てる。
//!
//! # 数値契約
//! 主ループは R0 の GEMM とバイト同一（FMA・k 昇順・単一アキュムレータ）で、bias は 1 回の加算、
//! ReLU は `x > 0 ? x : 0`。よって `gemm` → `add` → `relu` の合成と bit 一致する。長軸縮約はなく
//! `f64` アキュムレータ契約は対象外。

use std::cell::RefCell;

use fandhe_ai_tensor_core::{
    Activation, BackendError, BackendOps, DeviceBuffer, DeviceBufferView, ShapeError, Tensor,
    gemm_out_shape,
};

use super::{Cache, Engine, RocmBackendOps, cached};
use crate::buffer_handle::RocmBufferHandle;
use crate::error::to_backend_error;
#[cfg(test)]
use crate::gemm::validate_gemm_dims;
#[cfg(test)]
use crate::gemm_bias_act::operand_range;
use crate::gemm_bias_act::{GemmCall, RocmGemmBiasAct};
#[cfg(test)]
use crate::kernels_gemm::GemmF32Tile;
use crate::typed_buffer::TypedDeviceBuffer;

thread_local! {
    /// target ごとにコンパイル済みの融合 GEMM を遅延保持する（R11）。失敗はキャッシュしない。
    static GEMM_BIAS_ACT_CACHE: Cache<RocmGemmBiasAct> = const { RefCell::new(Vec::new()) };
}

/// テスト専用のホスト模擬ハンドルか（Hip engine では常に取り違え）。本番ビルドでは常に false。
fn is_model_handle(h: &RocmBufferHandle) -> bool {
    #[cfg(test)]
    if matches!(h, RocmBufferHandle::Host(_)) {
        return true;
    }
    let _ = h;
    false
}

/// 入力オペランド: ホストのスライス（アップロードされる）か、常駐ハンドルの要素オフセット。
enum In<'a> {
    Slice(&'a [f32]),
    Res(&'a RocmBufferHandle, usize),
}

/// 出力先: 新規確保か、既存ハンドルの要素オフセットへの上書き。
enum Out<'a> {
    Fresh,
    Into(&'a mut RocmBufferHandle, usize),
}

/// 起動に渡す形状。
#[derive(Clone, Copy)]
struct Dims {
    m: usize,
    n: usize,
    k: usize,
    relu: bool,
}

fn shape_err(e: ShapeError) -> BackendError {
    BackendError::ShapeMismatch(e)
}

fn mismatch(lhs: &[usize], rhs: &[usize]) -> BackendError {
    shape_err(ShapeError::ShapeMismatch {
        lhs: lhs.to_vec(),
        rhs: rhs.to_vec(),
    })
}

fn rank2(shape: &[usize]) -> Result<(usize, usize), BackendError> {
    match shape {
        [r, c] => Ok((*r, *c)),
        _ => Err(shape_err(ShapeError::RankMismatch {
            expected: 2,
            actual: shape.len(),
        })),
    }
}

fn handle_of(buf: &DeviceBuffer<f32>) -> Result<&RocmBufferHandle, BackendError> {
    buf.downcast_handle::<RocmBufferHandle>()
        .ok_or(BackendError::DeviceMismatch)
}

/// ビューの下地ハンドルを取り出し、`offset + numel` がハンドル長に収まることを確かめる
/// （`DeviceBufferView::new` の検査とは独立の二重化。REQ-8）。
fn view_handle<'a>(v: &DeviceBufferView<'a>) -> Result<&'a RocmBufferHandle, BackendError> {
    let h = handle_of(v.buffer())?;
    let end = v.offset().checked_add(v.numel()).ok_or_else(|| {
        BackendError::InvalidArgument("view: offset + numel が usize を超えます".into())
    })?;
    if end > h.len() {
        return Err(shape_err(ShapeError::ElementCountMismatch {
            expected: end,
            actual: h.len(),
        }));
    }
    Ok(h)
}

/// `contiguous()` の前に要素数を checked で求める（巨大な broadcast view で確保に進まないため）。
fn host_numel(shape: &[usize]) -> Result<usize, BackendError> {
    shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or(shape_err(ShapeError::ElementCountOverflow))
}

fn contiguous_slice(t: &Tensor<f32>) -> Result<&[f32], BackendError> {
    t.as_slice()
        .ok_or_else(|| BackendError::KernelLaunchFailed("gemm: tensor not contiguous".into()))
}

fn k_zero() -> BackendError {
    BackendError::InvalidArgument(
        "k == 0 は未対応です（CUDA と同じ。resident bias の D2H なしに epilogue を扱えない）"
            .into(),
    )
}

impl RocmBackendOps {
    fn wrap_gemm_handle(&self, shape: &[usize], h: RocmBufferHandle) -> DeviceBuffer<f32> {
        DeviceBuffer::new(self.device(), shape.to_vec(), Box::new(h))
    }

    /// 4 メソッド共通の実行本体。検査済みの `Dims` で engine ごとに計算し、`Out::Fresh` なら
    /// 新しいハンドルを返す（`Out::Into` は上書きして `None`）。
    fn gemm_derived_run(
        &self,
        a: In<'_>,
        b: In<'_>,
        bias: Option<In<'_>>,
        out: Out<'_>,
        d: Dims,
    ) -> Result<Option<RocmBufferHandle>, BackendError> {
        match self.engine {
            Engine::Hip => self.gemm_derived_hip(a, b, bias, out, d),
            #[cfg(test)]
            Engine::HostModel => self.gemm_derived_model(a, b, bias, out, d),
        }
    }

    fn gemm_derived_hip(
        &self,
        a: In<'_>,
        b: In<'_>,
        bias: Option<In<'_>>,
        out: Out<'_>,
        d: Dims,
    ) -> Result<Option<RocmBufferHandle>, BackendError> {
        // engine とハンドル種別の取り違えは、HIP のロード（`DeviceUnavailable`）より前に確定する。
        let foreign = |i: &In<'_>| matches!(i, In::Res(h, _) if is_model_handle(h));
        if foreign(&a)
            || foreign(&b)
            || bias.as_ref().is_some_and(foreign)
            || matches!(&out, Out::Into(h, _) if is_model_handle(h))
        {
            return Err(BackendError::DeviceMismatch);
        }
        let kern = cached(&GEMM_BIAS_ACT_CACHE, self.target, RocmGemmBiasAct::new)
            .map_err(to_backend_error)?;
        // ホストのスライス（と Empty ハンドル）は一時バッファへ上げ、常駐 Hip ハンドルはそのまま使う。
        let mut temps: Vec<TypedDeviceBuffer<f32>> = Vec::new();
        let sa = stage(&kern, &mut temps, a)?;
        let sb = stage(&kern, &mut temps, b)?;
        let sbias = match bias {
            Some(i) => Some(stage(&kern, &mut temps, i)?),
            None => None,
        };
        let ra = operand(&temps, &sa)?;
        let rb = operand(&temps, &sb)?;
        let rbias = match &sbias {
            Some(s) => Some(operand(&temps, s)?),
            None => None,
        };
        let mn =
            d.m.checked_mul(d.n)
                .ok_or(shape_err(ShapeError::ElementCountOverflow))?;
        let call = |c| GemmCall {
            a: ra,
            b: rb,
            bias: rbias,
            c,
            m: d.m,
            n: d.n,
            k: d.k,
            act_relu: d.relu,
        };
        match out {
            Out::Fresh => {
                let c = kern.alloc(mn).map_err(to_backend_error)?;
                kern.run(&call((&c, 0))).map_err(to_backend_error)?;
                Ok(Some(RocmBufferHandle::from(c)))
            }
            Out::Into(RocmBufferHandle::Hip(c), off) => {
                kern.run(&call((&*c, off))).map_err(to_backend_error)?;
                Ok(None)
            }
            // engine とハンドル種別の取り違えは fail-closed。
            Out::Into(_, _) => Err(BackendError::DeviceMismatch),
        }
    }

    /// テスト専用: `gemm_model::simulate_with_epilogue`（カーネルの逐語モデル）で計算する。
    #[cfg(test)]
    fn gemm_derived_model(
        &self,
        a: In<'_>,
        b: In<'_>,
        bias: Option<In<'_>>,
        out: Out<'_>,
        d: Dims,
    ) -> Result<Option<RocmBufferHandle>, BackendError> {
        let tile = GemmF32Tile::DEFAULT;
        let ovf = || shape_err(ShapeError::ElementCountOverflow);
        let mk = d.m.checked_mul(d.k).ok_or_else(ovf)?;
        let kn = d.k.checked_mul(d.n).ok_or_else(ovf)?;
        let mn = d.m.checked_mul(d.n).ok_or_else(ovf)?;
        validate_gemm_dims(&tile, mk, kn, d.m, d.n, d.k).map_err(to_backend_error)?;
        let av = model_slice("a", &a, mk)?;
        let bv = model_slice("b", &b, kn)?;
        let biasv = match &bias {
            Some(i) => Some(model_slice("bias", i, d.n)?),
            None => None,
        };
        let res = crate::gemm_model::simulate_with_epilogue(
            &tile,
            av,
            bv,
            d.m,
            d.n,
            d.k,
            |_row, col, acc| {
                let mut v = acc;
                if let Some(bs) = biasv {
                    v += bs[col];
                }
                if d.relu {
                    v = if v > 0.0 { v } else { 0.0 };
                }
                v
            },
        );
        match out {
            Out::Fresh => Ok(Some(RocmBufferHandle::Host(res))),
            Out::Into(RocmBufferHandle::Host(v), off) => {
                operand_range("c", v.len(), off, mn).map_err(to_backend_error)?;
                v[off..off + mn].copy_from_slice(&res);
                Ok(None)
            }
            Out::Into(_, _) => Err(BackendError::DeviceMismatch),
        }
    }

    /// 新規出力ハンドルをホストの `Vec` へ読み戻す（`gemm_resident_rhs`／`_lhs` の戻り値用）。
    fn gemm_handle_to_host(&self, h: &RocmBufferHandle) -> Result<Vec<f32>, BackendError> {
        match h {
            RocmBufferHandle::Empty => Ok(Vec::new()),
            RocmBufferHandle::Hip(b) => {
                let kern = cached(&GEMM_BIAS_ACT_CACHE, self.target, RocmGemmBiasAct::new)
                    .map_err(to_backend_error)?;
                kern.download(b).map_err(to_backend_error)
            }
            #[cfg(test)]
            RocmBufferHandle::Host(v) => Ok(v.clone()),
        }
    }

    /// `BackendOps::gemm_resident_rhs`: `y = a @ w (+ bias)`。`w`・`bias` は常駐のまま読む。
    ///
    /// 戻り値の `Tensor` はホスト常駐（trait 契約。`w`／`bias` を download しない）。
    pub(super) fn gemm_derived_rhs(
        &self,
        a: &Tensor<f32>,
        w: DeviceBufferView<'_>,
        bias: Option<DeviceBufferView<'_>>,
    ) -> Result<Tensor<f32>, BackendError> {
        // ハンドル種別の判別を最初に行う（CPU 由来のバッファは空形状でも DeviceMismatch）。
        let hw = view_handle(&w)?;
        let hbias = match &bias {
            Some(v) => Some(view_handle(v)?),
            None => None,
        };
        let (m, k) = rank2(a.shape())?;
        let (w_k, n) = rank2(w.shape()).map_err(|_| mismatch(a.shape(), w.shape()))?;
        if w_k != k {
            return Err(mismatch(a.shape(), w.shape()));
        }
        if let Some(v) = &bias
            && v.shape() != [n]
        {
            return Err(mismatch(v.shape(), &[n]));
        }
        let mn = m
            .checked_mul(n)
            .ok_or(shape_err(ShapeError::ElementCountOverflow))?;
        if k == 0 {
            return Err(k_zero());
        }
        if mn == 0 {
            return Tensor::new(Vec::new(), &[m, n]).map_err(shape_err);
        }
        host_numel(a.shape())?;
        let a_owned = a.contiguous();
        let dims = Dims {
            m,
            n,
            k,
            relu: false,
        };
        let h = self
            .gemm_derived_run(
                In::Slice(contiguous_slice(&a_owned)?),
                In::Res(hw, w.offset()),
                match (&bias, hbias) {
                    (Some(v), Some(h)) => Some(In::Res(h, v.offset())),
                    _ => None,
                },
                Out::Fresh,
                dims,
            )?
            .ok_or_else(|| {
                BackendError::KernelLaunchFailed("gemm_resident_rhs: no output".into())
            })?;
        Tensor::new(self.gemm_handle_to_host(&h)?, &[m, n]).map_err(shape_err)
    }

    /// `BackendOps::gemm_resident_lhs`: `c = w @ b`（`w` が常駐の左辺）。
    ///
    /// NT 転置ビューの高速経路（CUDA の `memory.rs`／`transpose.rs`）は移植せず、`contiguous()` で
    /// 再配置する（数値は同じで、性能だけの差。被覆計画 §9 の R11 記録に申し送る）。
    pub(super) fn gemm_derived_lhs(
        &self,
        w: DeviceBufferView<'_>,
        b: &Tensor<f32>,
    ) -> Result<Tensor<f32>, BackendError> {
        let hw = view_handle(&w)?;
        let (p, q) = rank2(w.shape())?;
        let (b_q, r) = rank2(b.shape()).map_err(|_| mismatch(w.shape(), b.shape()))?;
        if b_q != q {
            return Err(mismatch(w.shape(), b.shape()));
        }
        let pr = p
            .checked_mul(r)
            .ok_or(shape_err(ShapeError::ElementCountOverflow))?;
        if pr == 0 {
            return Tensor::new(Vec::new(), &[p, r]).map_err(shape_err);
        }
        if q == 0 {
            // 縮約長 0 の積は 0 埋め（CUDA と同じ。HIP を使わない）。
            return Tensor::new(vec![0.0; pr], &[p, r]).map_err(shape_err);
        }
        host_numel(b.shape())?;
        let b_owned = b.contiguous();
        let dims = Dims {
            m: p,
            n: r,
            k: q,
            relu: false,
        };
        let h = self
            .gemm_derived_run(
                In::Res(hw, w.offset()),
                In::Slice(contiguous_slice(&b_owned)?),
                None,
                Out::Fresh,
                dims,
            )?
            .ok_or_else(|| {
                BackendError::KernelLaunchFailed("gemm_resident_lhs: no output".into())
            })?;
        Tensor::new(self.gemm_handle_to_host(&h)?, &[p, r]).map_err(shape_err)
    }

    /// `BackendOps::gemm_fp32_strict_into`: `out[off .. off + m*n]` を `a @ b` で**上書き**する。
    ///
    /// 数値は同 shape の `gemm_fp32_strict`（ROCm では既定継承 → `gemm` → R0 カーネル）と bit 同一
    /// （同じ主ループ・bias／ReLU なし）。
    pub(super) fn gemm_derived_into(
        &self,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
        out: &mut DeviceBuffer<f32>,
        out_offset: usize,
    ) -> Result<(), BackendError> {
        gemm_out_shape(a.shape(), b.shape()).map_err(shape_err)?;
        let (m, k) = rank2(a.shape())?;
        let n = b.shape()[1];
        // REQ-8: 書き込み範囲を checked で検査してから out を可変借用する（CPU／CUDA と同じ文言）。
        let mn = m.checked_mul(n).ok_or_else(|| {
            BackendError::InvalidArgument("gemm_fp32_strict_into: m * n overflowed usize".into())
        })?;
        let end = out_offset.checked_add(mn).ok_or_else(|| {
            BackendError::InvalidArgument(
                "gemm_fp32_strict_into: out_offset + m * n overflowed usize".into(),
            )
        })?;
        if end > out.numel() {
            return Err(BackendError::InvalidArgument(format!(
                "gemm_fp32_strict_into: write range [{out_offset}, {end}) exceeds out buffer \
                 length {}",
                out.numel()
            )));
        }
        // 空形状の早期リターンより前に判別する（CPU 由来のバッファは空でも DeviceMismatch）。
        handle_of(out)?;
        if mn == 0 {
            return Ok(());
        }
        host_numel(a.shape())?;
        host_numel(b.shape())?;
        let (a_owned, b_owned) = (a.contiguous(), b.contiguous());
        let handle = out
            .downcast_handle_mut::<RocmBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let dims = Dims {
            m,
            n,
            k,
            relu: false,
        };
        self.gemm_derived_run(
            In::Slice(contiguous_slice(&a_owned)?),
            In::Slice(contiguous_slice(&b_owned)?),
            None,
            Out::Into(handle, out_offset),
            dims,
        )?;
        Ok(())
    }

    /// `BackendOps::linear_forward_device`: `y = act(a @ w (+ bias))` を入出力ともデバイス常駐のまま
    /// 計算する。download しない（推論チェーンの同期点を最終出力の 1 回へ集約する契約）。
    pub(super) fn gemm_derived_linear_forward(
        &self,
        a: &DeviceBuffer<f32>,
        w: DeviceBufferView<'_>,
        bias: Option<DeviceBufferView<'_>>,
        act: Activation,
    ) -> Result<DeviceBuffer<f32>, BackendError> {
        let ha = handle_of(a)?;
        let hw = view_handle(&w)?;
        let hbias = match &bias {
            Some(v) => Some(view_handle(v)?),
            None => None,
        };
        let (m, k) = rank2(a.shape())?;
        if ha.len() != a.numel() {
            return Err(shape_err(ShapeError::ElementCountMismatch {
                expected: a.numel(),
                actual: ha.len(),
            }));
        }
        let (w_k, n) = rank2(w.shape()).map_err(|_| mismatch(a.shape(), w.shape()))?;
        if w_k != k {
            return Err(mismatch(a.shape(), w.shape()));
        }
        if let Some(v) = &bias
            && v.shape() != [n]
        {
            return Err(mismatch(v.shape(), &[n]));
        }
        // 未知の Activation は黙って恒等にせず拒否する（HIP に触れる前に確定）。
        let relu = match act {
            Activation::None => false,
            Activation::Relu => true,
            _ => {
                return Err(BackendError::Unsupported(format!(
                    "linear_forward_device: unsupported activation {act:?}"
                )));
            }
        };
        let mn = m
            .checked_mul(n)
            .ok_or(shape_err(ShapeError::ElementCountOverflow))?;
        if k == 0 {
            return Err(k_zero());
        }
        if mn == 0 {
            return Ok(self.wrap_gemm_handle(&[m, n], RocmBufferHandle::Empty));
        }
        let dims = Dims { m, n, k, relu };
        let h = self
            .gemm_derived_run(
                In::Res(ha, 0),
                In::Res(hw, w.offset()),
                match (&bias, hbias) {
                    (Some(v), Some(h)) => Some(In::Res(h, v.offset())),
                    _ => None,
                },
                Out::Fresh,
                dims,
            )?
            .ok_or_else(|| {
                BackendError::KernelLaunchFailed("linear_forward_device: no output".into())
            })?;
        Ok(self.wrap_gemm_handle(&[m, n], h))
    }
}

/// `stage` の結果: 一時バッファの位置か、常駐バッファへの参照。
enum Staged<'a> {
    Temp(usize, usize),
    Dev(&'a TypedDeviceBuffer<f32>, usize),
}

/// `In` を起動用に整える（ホストのスライスと Empty は一時バッファへ転送する）。
fn stage<'a>(
    kern: &RocmGemmBiasAct,
    temps: &mut Vec<TypedDeviceBuffer<f32>>,
    i: In<'a>,
) -> Result<Staged<'a>, BackendError> {
    match i {
        In::Slice(s) => {
            temps.push(kern.upload(s).map_err(to_backend_error)?);
            Ok(Staged::Temp(temps.len() - 1, 0))
        }
        In::Res(RocmBufferHandle::Empty, off) => {
            temps.push(kern.upload(&[]).map_err(to_backend_error)?);
            Ok(Staged::Temp(temps.len() - 1, off))
        }
        In::Res(RocmBufferHandle::Hip(b), off) => Ok(Staged::Dev(b, off)),
        // Hip engine に Host ハンドルを渡した取り違え。
        #[cfg(test)]
        In::Res(RocmBufferHandle::Host(_), _) => Err(BackendError::DeviceMismatch),
    }
}

/// `Staged` を `(バッファ, オフセット)` へ解決する。
fn operand<'t, 'a: 't>(
    temps: &'t [TypedDeviceBuffer<f32>],
    s: &'t Staged<'a>,
) -> Result<(&'t TypedDeviceBuffer<f32>, usize), BackendError> {
    match s {
        Staged::Temp(i, off) => temps
            .get(*i)
            .map(|t| (t, *off))
            .ok_or_else(|| BackendError::KernelLaunchFailed("gemm: staging index".into())),
        Staged::Dev(b, off) => Ok((*b, *off)),
    }
}

/// テスト専用: HostModel のオペランドを `len` 要素のスライスとして取り出す（範囲は checked）。
#[cfg(test)]
fn model_slice<'a>(what: &str, i: &'a In<'a>, len: usize) -> Result<&'a [f32], BackendError> {
    match i {
        In::Slice(s) => {
            if s.len() != len {
                return Err(shape_err(ShapeError::ElementCountMismatch {
                    expected: len,
                    actual: s.len(),
                }));
            }
            Ok(s)
        }
        In::Res(RocmBufferHandle::Empty, off) => {
            operand_range(what, 0, *off, len).map_err(to_backend_error)?;
            Ok(&[])
        }
        In::Res(RocmBufferHandle::Host(v), off) => {
            operand_range(what, v.len(), *off, len).map_err(to_backend_error)?;
            Ok(&v[*off..*off + len])
        }
        In::Res(RocmBufferHandle::Hip(_), _) => Err(BackendError::DeviceMismatch),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::target::GfxTarget;
    use fandhe_ai_backend_cpu::{CpuBackendOps, assert_parity};

    fn ops() -> RocmBackendOps {
        RocmBackendOps::with_host_model(GfxTarget::Gfx942)
    }

    /// 決定的な疑似乱数（LCG。`[-1, 1)`）。
    fn data(len: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..len)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn t(len: usize, shape: &[usize], seed: u64) -> Tensor<f32> {
        Tensor::new(data(len, seed), shape).unwrap()
    }

    fn bits(x: &Tensor<f32>) -> Vec<u32> {
        x.contiguous()
            .as_slice()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect()
    }

    fn same(x: &Tensor<f32>, y: &Tensor<f32>) {
        assert_eq!(x.shape(), y.shape());
        assert_eq!(bits(x), bits(y));
    }

    fn flat(v: &[f32]) -> Tensor<f32> {
        Tensor::new(v.to_vec(), &[v.len()]).unwrap()
    }

    /// 連結バッファ `[余白 7][w][余白 3][bias][余白 5]` と各オフセットを作る。
    fn layout(w: &[f32], bias: &[f32]) -> (Vec<f32>, usize, usize) {
        let mut v = vec![f32::NAN; 7];
        let w_off = v.len();
        v.extend_from_slice(w);
        v.extend_from_slice(&[f32::NAN; 3]);
        let b_off = v.len();
        v.extend_from_slice(bias);
        v.extend_from_slice(&[f32::NAN; 5]);
        (v, w_off, b_off)
    }

    fn cpu_resident(v: &[f32]) -> DeviceBuffer<f32> {
        CpuBackendOps
            .memory_ops()
            .unwrap()
            .upload(&flat(v))
            .unwrap()
    }

    /// `on` のときだけ bias ビューを作る。
    fn opt_view<'a>(
        on: bool,
        d: &'a DeviceBuffer<f32>,
        off: usize,
        shape: &'a [usize],
    ) -> Option<DeviceBufferView<'a>> {
        on.then(|| DeviceBufferView::new(d, off, shape).unwrap())
    }

    /// `(w ビュー, bias ビュー)`。
    fn rhs_views<'a>(
        d: &'a DeviceBuffer<f32>,
        w: (usize, &'a [usize]),
        bias: (bool, usize, &'a [usize]),
    ) -> (DeviceBufferView<'a>, Option<DeviceBufferView<'a>>) {
        (
            DeviceBufferView::new(d, w.0, w.1).unwrap(),
            opt_view(bias.0, d, bias.1, bias.2),
        )
    }

    fn cpu_download(b: &DeviceBuffer<f32>) -> Tensor<f32> {
        CpuBackendOps.memory_ops().unwrap().download(b).unwrap()
    }

    /// ROCm 自身の `gemm` → bias 加算 → relu の合成（bit 一致の基準）。
    fn composed(
        o: &RocmBackendOps,
        a: &Tensor<f32>,
        w: &Tensor<f32>,
        bias: Option<&Tensor<f32>>,
        relu: bool,
    ) -> Tensor<f32> {
        let mut y = o.gemm(a, w).unwrap();
        if let Some(b) = bias {
            y = o.add(&y, b).unwrap();
        }
        if relu {
            y = o.relu(&y).unwrap();
        }
        y
    }

    /// 64/16 の倍数でない端を含む形状。
    const SHAPES: [(usize, usize, usize); 4] =
        [(1, 1, 1), (63, 65, 17), (130, 70, 33), (5, 129, 200)];

    #[test]
    fn rhs_matches_composition_and_cpu_with_offsets() {
        let o = ops();
        for (idx, &(m, n, k)) in SHAPES.iter().enumerate() {
            let a = t(m * k, &[m, k], 1 + idx as u64);
            let wd = data(k * n, 20 + idx as u64);
            let bd = data(n, 40 + idx as u64);
            let (buf, w_off, b_off) = layout(&wd, &bd);
            let dev = o.upload_resident(&flat(&buf)).unwrap();
            let cpu_dev = cpu_resident(&buf);
            let (ws, bs) = ([k, n], [n]);
            let wt = Tensor::new(wd.clone(), &ws).unwrap();
            let bt = Tensor::new(bd.clone(), &bs).unwrap();
            for with_bias in [false, true] {
                let view = |d| rhs_views(d, (w_off, &ws), (with_bias, b_off, &bs));
                let (w, bias) = view(&dev);
                let got = o.gemm_resident_rhs(&a, w, bias).unwrap();
                let want = composed(&o, &a, &wt, with_bias.then_some(&bt), false);
                same(&got, &want);
                let (cw, cb) = view(&cpu_dev);
                let cpu = CpuBackendOps.gemm_resident_rhs(&a, cw, cb).unwrap();
                assert_parity(
                    &format!("rhs {m}x{n}x{k} bias={with_bias}"),
                    got.as_slice().unwrap(),
                    cpu.as_slice().unwrap(),
                );
                // act 版は既定継承（rhs → relu）。
                let (w, bias) = view(&dev);
                let act = o
                    .gemm_resident_rhs_act(&a, w, bias, Activation::Relu)
                    .unwrap();
                same(&act, &o.relu(&got).unwrap());
            }
        }
    }

    #[test]
    fn rhs_handles_noncontiguous_a() {
        let o = ops();
        let (m, n, k) = (9, 7, 5);
        let a_t = t(k * m, &[k, m], 3).transpose_2d().unwrap();
        assert!(a_t.as_slice().is_none());
        let wd = data(k * n, 4);
        let (buf, w_off, _) = layout(&wd, &[]);
        let dev = o.upload_resident(&flat(&buf)).unwrap();
        let ws = [k, n];
        let got = o
            .gemm_resident_rhs(&a_t, DeviceBufferView::new(&dev, w_off, &ws).unwrap(), None)
            .unwrap();
        same(&got, &o.gemm(&a_t, &Tensor::new(wd, &ws).unwrap()).unwrap());
    }

    #[test]
    fn lhs_matches_gemm_and_cpu() {
        let o = ops();
        for (idx, &(p, q, r)) in SHAPES.iter().enumerate() {
            let wd = data(p * q, 60 + idx as u64);
            let (buf, w_off, _) = layout(&wd, &[]);
            let dev = o.upload_resident(&flat(&buf)).unwrap();
            let cpu_dev = cpu_resident(&buf);
            let ws = [p, q];
            let wt = Tensor::new(wd, &ws).unwrap();
            let b = t(q * r, &[q, r], 70 + idx as u64);
            // 転置ビュー（非連続）も同じ結果。
            let b_tv = t(r * q, &[r, q], 71 + idx as u64).transpose_2d().unwrap();
            for b in [b, b_tv] {
                let got = o
                    .gemm_resident_lhs(DeviceBufferView::new(&dev, w_off, &ws).unwrap(), &b)
                    .unwrap();
                same(&got, &o.gemm(&wt, &b).unwrap());
                let cpu = CpuBackendOps
                    .gemm_resident_lhs(DeviceBufferView::new(&cpu_dev, w_off, &ws).unwrap(), &b)
                    .unwrap();
                assert_parity(
                    &format!("lhs {p}x{q}x{r}"),
                    got.as_slice().unwrap(),
                    cpu.as_slice().unwrap(),
                );
            }
        }
    }

    #[test]
    fn lhs_degenerate_shapes() {
        let o = ops();
        let dev = o.upload_resident(&flat(&data(6, 1))).unwrap();
        // q == 0: 0 埋めの [p, r]。
        let ws = [3, 0];
        let w = DeviceBufferView::new(&dev, 0, &ws).unwrap();
        let b = Tensor::new(Vec::new(), &[0, 4]).unwrap();
        let z = o.gemm_resident_lhs(w, &b).unwrap();
        assert_eq!(z.shape(), &[3, 4]);
        assert!(z.as_slice().unwrap().iter().all(|&v| v == 0.0));
        // p == 0 と r == 0: 空。
        let ws = [0, 2];
        let w = DeviceBufferView::new(&dev, 0, &ws).unwrap();
        let e = o.gemm_resident_lhs(w, &t(6, &[2, 3], 2)).unwrap();
        assert_eq!(e.shape(), &[0, 3]);
        let ws = [3, 2];
        let w = DeviceBufferView::new(&dev, 0, &ws).unwrap();
        let e = o.gemm_resident_lhs(w, &Tensor::new(Vec::new(), &[2, 0]).unwrap());
        assert_eq!(e.unwrap().shape(), &[3, 0]);
    }

    #[test]
    fn linear_forward_matches_composition_and_cpu() {
        let o = ops();
        for (idx, &(m, n, k)) in SHAPES.iter().enumerate() {
            let a = t(m * k, &[m, k], 80 + idx as u64);
            let wd = data(k * n, 90 + idx as u64);
            let bd = data(n, 95 + idx as u64);
            let (buf, w_off, b_off) = layout(&wd, &bd);
            let dev = o.upload_resident(&flat(&buf)).unwrap();
            let cpu_dev = cpu_resident(&buf);
            let da = o.upload_resident(&a).unwrap();
            let cpu_a = CpuBackendOps.memory_ops().unwrap().upload(&a).unwrap();
            let (ws, bs) = ([k, n], [n]);
            let wt = Tensor::new(wd, &ws).unwrap();
            let bt = Tensor::new(bd, &bs).unwrap();
            for (with_bias, act) in [
                (false, Activation::None),
                (true, Activation::None),
                (false, Activation::Relu),
                (true, Activation::Relu),
            ] {
                let relu = act == Activation::Relu;
                let out = o
                    .linear_forward_device(
                        &da,
                        DeviceBufferView::new(&dev, w_off, &ws).unwrap(),
                        opt_view(with_bias, &dev, b_off, &bs),
                        act,
                    )
                    .unwrap();
                assert_eq!(out.shape(), &[m, n]);
                let got = o.download_resident(&out).unwrap();
                same(&got, &composed(&o, &a, &wt, with_bias.then_some(&bt), relu));
                let cpu_out = CpuBackendOps
                    .linear_forward_device(
                        &cpu_a,
                        DeviceBufferView::new(&cpu_dev, w_off, &ws).unwrap(),
                        opt_view(with_bias, &cpu_dev, b_off, &bs),
                        act,
                    )
                    .unwrap();
                let cpu = cpu_download(&cpu_out);
                assert_parity(
                    &format!("linear {m}x{n}x{k} bias={with_bias} relu={relu}"),
                    got.as_slice().unwrap(),
                    cpu.as_slice().unwrap(),
                );
            }
        }
    }

    #[test]
    fn linear_forward_chains_without_download() {
        // 出力ハンドルをそのまま次層の入力に使える。
        let o = ops();
        let (m, k, h, n) = (6, 5, 4, 3);
        let a = t(m * k, &[m, k], 1);
        let (w1, w2) = (data(k * h, 2), data(h * n, 3));
        let da = o.upload_resident(&a).unwrap();
        let d1 = o.upload_resident(&flat(&w1)).unwrap();
        let d2 = o.upload_resident(&flat(&w2)).unwrap();
        let (s1, s2) = ([k, h], [h, n]);
        let y1 = o
            .linear_forward_device(
                &da,
                DeviceBufferView::new(&d1, 0, &s1).unwrap(),
                None,
                Activation::Relu,
            )
            .unwrap();
        let y2 = o
            .linear_forward_device(
                &y1,
                DeviceBufferView::new(&d2, 0, &s2).unwrap(),
                None,
                Activation::None,
            )
            .unwrap();
        let l1 = composed(&o, &a, &Tensor::new(w1, &s1).unwrap(), None, true);
        let want = composed(&o, &l1, &Tensor::new(w2, &s2).unwrap(), None, false);
        same(&o.download_resident(&y2).unwrap(), &want);
    }

    #[test]
    fn linear_forward_empty_shapes_are_empty_handles() {
        let o = ops();
        let w = o.upload_resident(&flat(&data(6, 1))).unwrap();
        let ws = [2, 3];
        let wv = || DeviceBufferView::new(&w, 0, &ws).unwrap();
        let a0 = o
            .upload_resident(&Tensor::new(Vec::new(), &[0, 2]).unwrap())
            .unwrap();
        let out = o
            .linear_forward_device(&a0, wv(), None, Activation::None)
            .unwrap();
        assert_eq!(out.shape(), &[0, 3]);
        assert_eq!(o.download_resident(&out).unwrap().numel(), 0);
        let ws0 = [2, 0];
        let w0 = DeviceBufferView::new(&w, 0, &ws0).unwrap();
        let a = o.upload_resident(&t(4, &[2, 2], 1)).unwrap();
        let out = o
            .linear_forward_device(&a, w0, None, Activation::Relu)
            .unwrap();
        assert_eq!(out.shape(), &[2, 0]);
    }

    #[test]
    fn into_matches_strict_and_overwrites_only_its_range() {
        let o = ops();
        for (idx, &(m, n, k)) in SHAPES.iter().enumerate() {
            let a = t(m * k, &[m, k], 100 + idx as u64);
            let b = t(k * n, &[k, n], 110 + idx as u64);
            let want = o.gemm_fp32_strict(&a, &b).unwrap();
            let (lead, tail) = (5usize, 3usize);
            let len = lead + m * n + tail;
            // 事前のゴミ値（累積なら結果がずれる）と範囲外の番兵（NaN）。
            let mut init = vec![123.0f32; len];
            init[..lead].fill(f32::NAN);
            init[lead + m * n..].fill(f32::NAN);
            let mut out = o.upload_resident(&flat(&init)).unwrap();
            o.gemm_fp32_strict_into(&a, &b, &mut out, lead).unwrap();
            let got = o.download_resident(&out).unwrap();
            let g = got.as_slice().unwrap();
            assert!(g[..lead].iter().all(|v| v.is_nan()));
            assert!(g[lead + m * n..].iter().all(|v| v.is_nan()));
            let mid = Tensor::new(g[lead..lead + m * n].to_vec(), &[m, n]).unwrap();
            same(&mid, &want);
            // CPU とも一致（複合判定）。
            let mut cpu_out = cpu_resident(&init);
            CpuBackendOps
                .gemm_fp32_strict_into(&a, &b, &mut cpu_out, lead)
                .unwrap();
            let cpu = cpu_download(&cpu_out);
            assert_parity(
                &format!("into {m}x{n}x{k}"),
                &g[lead..lead + m * n],
                &cpu.as_slice().unwrap()[lead..lead + m * n],
            );
        }
    }

    #[test]
    fn into_noncontiguous_operands_and_bounds() {
        let o = ops();
        let (m, n, k) = (4, 6, 3);
        let a = t(k * m, &[k, m], 1).transpose_2d().unwrap();
        let b = t(n * k, &[n, k], 2).transpose_2d().unwrap();
        let mut out = o.upload_resident(&flat(&vec![9.0; m * n])).unwrap();
        // 末尾ちょうどは可。
        o.gemm_fp32_strict_into(&a, &b, &mut out, 0).unwrap();
        let got = o.download_resident(&out).unwrap();
        same(
            &Tensor::new(got.as_slice().unwrap().to_vec(), &[m, n]).unwrap(),
            &o.gemm_fp32_strict(&a, &b).unwrap(),
        );
        // 1 つ超え・usize overflow は InvalidArgument。
        for off in [1, usize::MAX] {
            assert!(matches!(
                o.gemm_fp32_strict_into(&a, &b, &mut out, off),
                Err(BackendError::InvalidArgument(_))
            ));
        }
    }

    #[test]
    fn into_degenerate_shapes() {
        let o = ops();
        // k == 0: 範囲を 0 で上書きする。
        let (a, b) = (
            Tensor::new(Vec::new(), &[3, 0]).unwrap(),
            Tensor::new(Vec::new(), &[0, 2]).unwrap(),
        );
        let mut out = o.upload_resident(&flat(&[7.0; 8])).unwrap();
        o.gemm_fp32_strict_into(&a, &b, &mut out, 1).unwrap();
        let g = o.download_resident(&out).unwrap();
        assert_eq!(
            g.as_slice().unwrap(),
            &[7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0]
        );
        // m == 0: 何も書かず Ok。
        let (a, b) = (t(0, &[0, 2], 1), t(6, &[2, 3], 2));
        o.gemm_fp32_strict_into(&a, &b, &mut out, 8).unwrap();
        let g = o.download_resident(&out).unwrap();
        assert_eq!(g.as_slice().unwrap()[0], 7.0);
    }

    #[test]
    fn error_contracts() {
        let o = ops();
        let cpu_buf = cpu_resident(&data(12, 1));
        let dev = o.upload_resident(&flat(&data(12, 1))).unwrap();
        let ws = [3, 4];
        let a = t(6, &[2, 3], 2);
        // CPU 由来のバッファは DeviceMismatch（空形状でも）。
        let cw = DeviceBufferView::new(&cpu_buf, 0, &ws).unwrap();
        assert!(matches!(
            o.gemm_resident_rhs(&a, cw, None),
            Err(BackendError::DeviceMismatch)
        ));
        let cw = DeviceBufferView::new(&cpu_buf, 0, &[0, 4]).unwrap();
        assert!(matches!(
            o.gemm_resident_rhs(&t(0, &[2, 0], 1), cw, None),
            Err(BackendError::DeviceMismatch)
        ));
        let cw = DeviceBufferView::new(&cpu_buf, 0, &ws).unwrap();
        assert!(matches!(
            o.gemm_resident_lhs(cw, &t(8, &[4, 2], 1)),
            Err(BackendError::DeviceMismatch)
        ));
        assert!(matches!(
            o.linear_forward_device(
                &cpu_buf,
                DeviceBufferView::new(&dev, 0, &ws).unwrap(),
                None,
                Activation::None
            ),
            Err(BackendError::DeviceMismatch)
        ));
        let mut cpu_out = cpu_resident(&[0.0; 4]);
        assert!(matches!(
            o.gemm_fp32_strict_into(&t(4, &[2, 2], 1), &t(4, &[2, 2], 2), &mut cpu_out, 0),
            Err(BackendError::DeviceMismatch)
        ));
        let mut empty_cpu = CpuBackendOps
            .memory_ops()
            .unwrap()
            .alloc_zeroed(&[0])
            .unwrap();
        assert!(matches!(
            o.gemm_fp32_strict_into(&t(0, &[0, 2], 1), &t(4, &[2, 2], 2), &mut empty_cpu, 0),
            Err(BackendError::DeviceMismatch)
        ));
        // Hip engine に Host ハンドルを渡すと DeviceMismatch。
        let hip = RocmBackendOps::new(GfxTarget::Gfx942);
        assert!(matches!(
            hip.gemm_resident_rhs(&a, DeviceBufferView::new(&dev, 0, &ws).unwrap(), None),
            Err(BackendError::DeviceMismatch)
        ));
        // rank・shape。
        let w = || DeviceBufferView::new(&dev, 0, &ws).unwrap();
        assert!(matches!(
            o.gemm_resident_rhs(&t(3, &[3], 1), w(), None),
            Err(BackendError::ShapeMismatch(ShapeError::RankMismatch { .. }))
        ));
        assert!(matches!(
            o.gemm_resident_rhs(&t(8, &[2, 4], 1), w(), None),
            Err(BackendError::ShapeMismatch(_))
        ));
        let bs_bad = [3];
        assert!(matches!(
            o.gemm_resident_rhs(
                &a,
                w(),
                Some(DeviceBufferView::new(&dev, 0, &bs_bad).unwrap())
            ),
            Err(BackendError::ShapeMismatch(_))
        ));
        // k == 0 の rhs・linear_forward は InvalidArgument。
        let ws0 = [0, 4];
        let w0 = || DeviceBufferView::new(&dev, 0, &ws0).unwrap();
        assert!(matches!(
            o.gemm_resident_rhs(&t(0, &[2, 0], 1), w0(), None),
            Err(BackendError::InvalidArgument(_))
        ));
        let a0 = o.upload_resident(&t(0, &[2, 0], 1)).unwrap();
        assert!(matches!(
            o.linear_forward_device(&a0, w0(), None, Activation::None),
            Err(BackendError::InvalidArgument(_))
        ));
        // ハンドル長と shape のずれは ElementCountMismatch。
        let wrong = o.wrap_gemm_handle(&[2, 3], RocmBufferHandle::Host(vec![0.0; 5]));
        assert!(matches!(
            o.linear_forward_device(&wrong, w(), None, Activation::None),
            Err(BackendError::ShapeMismatch(
                ShapeError::ElementCountMismatch { .. }
            ))
        ));
        let lying = o.wrap_gemm_handle(&[12], RocmBufferHandle::Host(vec![0.0; 4]));
        assert!(matches!(
            o.gemm_resident_rhs(&a, DeviceBufferView::new(&lying, 0, &ws).unwrap(), None),
            Err(BackendError::ShapeMismatch(
                ShapeError::ElementCountMismatch { .. }
            ))
        ));
    }
}
