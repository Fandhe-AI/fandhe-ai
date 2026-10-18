//! `BackendOps` の ROCm 実装の骨格（イシュー #3431・R0。親 #3150）。
//!
//! 役割: `tensor-core::BackendOps` のうち既定実装を持たない必須 9 メソッド
//! （`device`・`gemm`・`add`・`mul`・`relu`・`exp`・`tanh`・`sum`・`max`）を成立させ、
//! R1〜R13（#3432〜#3444）がメソッドをオーバーライドとして足していける足場にする。
//! 被覆計画の正本は `docs/backend-rocm-ops-coverage-plan.md` §1・§4・§7・§8。
//!
//! - `gemm` は [`RocmGemmF32`]（HIPRTC の f32 GEMM。#3155）へ結線する。HIPRTC ソースと
//!   コンパイルオプションは既存の `GemmSource`・`GfxTarget` 許可リストの経路のままで、
//!   本モジュールは新しいカーネルソースを足さない（A03）。
//! - elementwise 一式（R2・#3433）: `add`・`mul`・`relu`・`exp`・`tanh`・`scalar_unary`・
//!   `scalar_binary`・`where_cond`・`masked_fill`・`run_fused`（elementwise の範囲）・
//!   `binary_elementwise_device`・`unary_elementwise_device` は HIPRTC カーネル
//!   （`kernels_elementwise`・`kernels_scalar_op`・`kernels_fused_elementwise`）で実行する。
//!   結線の本体（ブロードキャスト・形状再検査・`Unsupported` の事前確定）は子モジュール `ew`
//!   （`ops/ew.rs`）に置き、本ファイルのメソッドは 1 行で委譲する（兄弟 issue との衝突回避）。
//! - `sum`・`max`・`min`・`argmax`・`argmin`（R3・#3434）は [`RocmReduce`]／[`RocmArgReduce`]
//!   （HIPRTC。wave 幅は `WarpGeometry` で注入）へ結線する。CUDA 版との対応: sum は `f64`
//!   アキュムレータ（最後に 1 回 `f32`）、max／min は `fmaxf`／`fminf` の厳密選択、argmax／argmin は
//!   チャンク分割 2 段。空縮約（`numel == 0`。sum の 0 埋め・max 等の `EmptyReduction` 文言）は
//!   カーネルを起動せず `CpuBackendOps` へ委譲し、CPU と完全に同じ結果・エラーにする（二重実装を
//!   避ける）。CUDA の `*_lastaxis_f32`（1 warp = 1 行の coalesced 版）は移植せず、`inner == 1` も
//!   汎用の軸縮約で処理する（性能面の follow-up）。
//! - `RocmGemmF32`・`RocmReduce`・`RocmArgReduce` は生ポインタを持つため `!Send`・`!Sync` で、
//!   `unsafe impl` は足さない。
//!   Tape（`Box<dyn BackendOps + Send>`）に載せるため、コンパイル済みカーネルは thread_local の
//!   遅延キャッシュに置き、構造体は `Copy` の値だけを持つ（CUDA の `context_cache` と同じ考え方）。
//!   残るリスク: スレッド終了時に `hipModuleUnload` が走る。プロセス終了時の TLS 破棄順は保証されない。
//! - どの候補リスト（`ops_for`・facade の `resolve_ops`／`available_devices`）にも登録しない。

use std::cell::RefCell;
use std::rc::Rc;
use std::thread::LocalKey;

use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_tensor_core::{
    BackendError, BackendOps, BinaryElementwiseOp, Device, DeviceBuffer, FusionPlan,
    ScalarBinaryOp, ScalarUnaryOp, Tensor, UnaryElementwiseOp, gemm_out_shape, reduce_out_shape,
    row_softmax_layout,
};

use crate::arg_reduce::{ArgReduceKind, RocmArgReduce};
use crate::elementwise::{
    BinaryKind, RocmElementwise, RocmFusedElementwise, RocmScalarOp, UnaryKind,
};
use crate::error::{RocmError, to_backend_error};
use crate::gemm::RocmGemmF32;
use crate::reduce::{ReduceKind, RocmReduce, reduce_axis_layout};
use crate::target::GfxTarget;

/// 実行経路。`HostModel` はテスト専用（実機なしで組み立て経路を検証する）。
#[derive(Debug, Clone, Copy)]
enum Engine {
    Hip,
    #[cfg(test)]
    HostModel,
}

/// target ごとにコンパイル済みの実行型を遅延保持する thread_local キャッシュの型。
type Cache<T> = RefCell<Vec<(GfxTarget, Rc<T>)>>;

thread_local! {
    /// target ごとにコンパイル済みの GEMM を遅延保持する。失敗はキャッシュしない。
    static GEMM_CACHE: Cache<RocmGemmF32> = const { RefCell::new(Vec::new()) };
    /// elementwise 基本カーネル（R2）。
    static EW_CACHE: Cache<RocmElementwise> = const { RefCell::new(Vec::new()) };
    /// scalar-op カーネル群（R2）。
    static SCALAR_CACHE: Cache<RocmScalarOp> = const { RefCell::new(Vec::new()) };
    /// 融合インタプリタカーネル（R2）。
    static FUSED_CACHE: Cache<RocmFusedElementwise> = const { RefCell::new(Vec::new()) };
    /// 縮約（sum/max/min。#3434）。
    static REDUCE_CACHE: Cache<RocmReduce> = const { RefCell::new(Vec::new()) };
    /// 添字縮約（argmax/argmin。#3434）。
    static ARG_REDUCE_CACHE: Cache<RocmArgReduce> = const { RefCell::new(Vec::new()) };
}

/// thread_local キャッシュ `key` から `target` 用の実行型を取得する（無ければ `make` で構築して入れる）。
///
/// 構築は HIPRTC コンパイルを伴い高価なので族ごとに遅延し、失敗はキャッシュしない
/// （ROCm 非搭載のまま後から有効になる環境でも再試行できる）。
fn cached<T>(
    key: &'static LocalKey<Cache<T>>,
    target: GfxTarget,
    make: fn(GfxTarget) -> Result<T, RocmError>,
) -> Result<Rc<T>, RocmError> {
    let hit = key
        .try_with(|c| {
            c.borrow()
                .iter()
                .find(|(t, _)| *t == target)
                .map(|(_, g)| Rc::clone(g))
        })
        .map_err(|_| RocmError::InvalidKernelConfig {
            detail: "カーネルキャッシュ（thread_local）が破棄済みです".into(),
        })?;
    if let Some(g) = hit {
        return Ok(g);
    }
    // 借用を解いてから構築する（構築中に再入しても二重借用にならない）。
    let g = Rc::new(make(target)?);
    let _ = key.try_with(|c| c.borrow_mut().push((target, Rc::clone(&g))));
    Ok(g)
}

/// ROCm（AMD GPU）向け `BackendOps`。必須 9 メソッドに R2 の elementwise 一式・R3 の縮約 5 メソッドを足した実装。
///
/// HIP のデバイス ID は `RocmGemmF32::new` と同じく 0 固定。gfx target の照会は #3157・#3159。
#[derive(Debug, Clone, Copy)]
pub struct RocmBackendOps {
    target: GfxTarget,
    engine: Engine,
    cpu: CpuBackendOps,
}

impl RocmBackendOps {
    /// `target` 向けの ops を作る。失敗せず、ROCm には触れない（初回実行時に遅延コンパイルする）。
    pub fn new(target: GfxTarget) -> Self {
        Self {
            target,
            engine: Engine::Hip,
            cpu: CpuBackendOps,
        }
    }

    /// 対象の gfx target。
    pub fn target(&self) -> GfxTarget {
        self.target
    }

    /// テスト専用: GEMM・縮約を CPU 上の添字モデル（`gemm_model`・`reduce_model`・
    /// `arg_reduce_model`）で計算する。
    #[cfg(test)]
    pub(crate) fn with_host_model(target: GfxTarget) -> Self {
        Self {
            engine: Engine::HostModel,
            ..Self::new(target)
        }
    }

    /// sum／max／min の dispatch（形状検査 → 空縮約の CPU 委譲 → contiguous 化 → カーネル）。
    fn reduce_dispatch(
        &self,
        a: &Tensor<f32>,
        dim: Option<usize>,
        kind: ReduceKind,
    ) -> Result<Tensor<f32>, BackendError> {
        let out_shape = reduce_out_shape(a.shape(), dim).map_err(BackendError::ShapeMismatch)?;
        if a.numel() == 0 {
            // 空縮約（sum の 0 埋め・max/min の EmptyReduction・空出力）は HIP に触れず CPU と
            // 同じ結果・エラー文言にする。
            return match kind {
                ReduceKind::Sum => self.cpu.sum(a, dim),
                ReduceKind::Max => self.cpu.max(a, dim),
                ReduceKind::Min => self.cpu.min(a, dim),
            };
        }
        let c = a.contiguous();
        let x = c.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("reduce: input not contiguous".into())
        })?;
        let data = match dim {
            None => vec![self.run_reduce_all(kind, x).map_err(to_backend_error)?],
            Some(axis) => {
                let (outer, axis_len, inner) =
                    reduce_axis_layout(a.shape(), axis).map_err(to_backend_error)?;
                self.run_reduce_axis(kind, x, outer, axis_len, inner)
                    .map_err(to_backend_error)?
            }
        };
        Tensor::new(data, &out_shape).map_err(BackendError::ShapeMismatch)
    }

    /// argmax／argmin の dispatch（[`Self::reduce_dispatch`] と同じ形。出力は `Tensor<i32>`）。
    fn arg_reduce_dispatch(
        &self,
        a: &Tensor<f32>,
        dim: Option<usize>,
        kind: ArgReduceKind,
    ) -> Result<Tensor<i32>, BackendError> {
        let out_shape = reduce_out_shape(a.shape(), dim).map_err(BackendError::ShapeMismatch)?;
        if a.numel() == 0 {
            return match kind {
                ArgReduceKind::Max => self.cpu.argmax(a, dim),
                ArgReduceKind::Min => self.cpu.argmin(a, dim),
            };
        }
        let c = a.contiguous();
        let x = c.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("arg_reduce: input not contiguous".into())
        })?;
        let data = match dim {
            None => vec![self.run_arg_all(kind, x).map_err(to_backend_error)?],
            Some(axis) => {
                let (outer, axis_len, inner) =
                    reduce_axis_layout(a.shape(), axis).map_err(to_backend_error)?;
                self.run_arg_axis(kind, x, outer, axis_len, inner)
                    .map_err(to_backend_error)?
            }
        };
        Tensor::new(data, &out_shape).map_err(BackendError::ShapeMismatch)
    }

    fn run_reduce_all(&self, kind: ReduceKind, x: &[f32]) -> Result<f32, RocmError> {
        match self.engine {
            Engine::Hip => cached(&REDUCE_CACHE, self.target, RocmReduce::new)?.run_all(kind, x),
            #[cfg(test)]
            Engine::HostModel => {
                crate::reduce::validate_reduce_all(x.len())?;
                crate::reduce_model::simulate_all(
                    crate::warp_geometry::WarpGeometry::for_target(self.target),
                    kind,
                    x,
                )
                .map_err(model_err)
            }
        }
    }

    fn run_reduce_axis(
        &self,
        kind: ReduceKind,
        x: &[f32],
        outer: usize,
        axis_len: usize,
        inner: usize,
    ) -> Result<Vec<f32>, RocmError> {
        match self.engine {
            Engine::Hip => cached(&REDUCE_CACHE, self.target, RocmReduce::new)?
                .run_axis(kind, x, outer, axis_len, inner),
            #[cfg(test)]
            Engine::HostModel => {
                crate::reduce::validate_reduce_axis(x.len(), outer, axis_len, inner)?;
                Ok(crate::reduce_model::simulate_axis(
                    kind, x, outer, axis_len, inner,
                ))
            }
        }
    }

    fn run_arg_all(&self, kind: ArgReduceKind, x: &[f32]) -> Result<i32, RocmError> {
        match self.engine {
            Engine::Hip => {
                cached(&ARG_REDUCE_CACHE, self.target, RocmArgReduce::new)?.run_all(kind, x)
            }
            #[cfg(test)]
            Engine::HostModel => {
                crate::reduce::validate_reduce_all(x.len())?;
                Ok(crate::arg_reduce_model::simulate_all(kind, x))
            }
        }
    }

    fn run_arg_axis(
        &self,
        kind: ArgReduceKind,
        x: &[f32],
        outer: usize,
        axis_len: usize,
        inner: usize,
    ) -> Result<Vec<i32>, RocmError> {
        match self.engine {
            Engine::Hip => cached(&ARG_REDUCE_CACHE, self.target, RocmArgReduce::new)?
                .run_axis(kind, x, outer, axis_len, inner),
            #[cfg(test)]
            Engine::HostModel => {
                crate::reduce::validate_reduce_axis(x.len(), outer, axis_len, inner)?;
                Ok(crate::arg_reduce_model::simulate_axis(
                    kind, x, outer, axis_len, inner,
                ))
            }
        }
    }

    fn run_gemm(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>, RocmError> {
        match self.engine {
            Engine::Hip => cached(&GEMM_CACHE, self.target, RocmGemmF32::new)?.run(a, b, m, n, k),
            #[cfg(test)]
            Engine::HostModel => {
                use crate::kernels_gemm::GemmF32Tile;
                let tile = GemmF32Tile::DEFAULT;
                crate::gemm::validate_gemm_dims(&tile, a.len(), b.len(), m, n, k)?;
                Ok(crate::gemm_model::simulate(&tile, a, b, m, n, k))
            }
        }
    }
}

/// `gemm` の形状検査・contiguous 化・`Tensor` 組み立てを共通化する（Hip／HostModel が同じ経路を通る）。
fn gemm_with(
    a: &Tensor<f32>,
    b: &Tensor<f32>,
    run: impl FnOnce(&[f32], &[f32], usize, usize, usize) -> Result<Vec<f32>, RocmError>,
) -> Result<Tensor<f32>, BackendError> {
    let out_shape = gemm_out_shape(a.shape(), b.shape()).map_err(BackendError::ShapeMismatch)?;
    let (m, k) = (a.shape()[0], a.shape()[1]);
    let n = b.shape()[1];
    let a_owned = a.contiguous();
    let b_owned = b.contiguous();
    let a_slice = a_owned
        .as_slice()
        .ok_or_else(|| BackendError::KernelLaunchFailed("gemm: lhs not contiguous".into()))?;
    let b_slice = b_owned
        .as_slice()
        .ok_or_else(|| BackendError::KernelLaunchFailed("gemm: rhs not contiguous".into()))?;
    let out = if m == 0 || n == 0 {
        // HIP を触らない（`RocmGemmF32::run` も空を返す）。
        Vec::new()
    } else {
        run(a_slice, b_slice, m, n, k).map_err(to_backend_error)?
    };
    Tensor::new(out, &out_shape).map_err(BackendError::ShapeMismatch)
}

mod ew;

/// softmax 系 3 演算の共通入口（最終軸判定・contiguous 化・`Tensor` 組み立て）。
///
/// 非最終軸は `Unsupported`（autodiff がホスト参照実装へフォールバックする合図。迂回経路を作らない。
/// security.md A08）。`run` は検証済みの `(rows, cols)` とスライスを受けて `Vec` を返す。
fn softmax_rows_with(
    op: &'static str,
    x: &Tensor<f32>,
    dim: usize,
    run: impl FnOnce(&[f32], usize, usize) -> Result<Vec<f32>, RocmError>,
) -> Result<Tensor<f32>, BackendError> {
    let Some((rows, cols)) =
        row_softmax_layout(x.shape(), dim).map_err(BackendError::ShapeMismatch)?
    else {
        return Err(BackendError::Unsupported(format!(
            "{op}: ROCm 行カーネルは最終軸限定（非最終軸はホスト参照実装へ委ねる）"
        )));
    };
    let x_owned = x.contiguous();
    let x_slice = x_owned
        .as_slice()
        .ok_or_else(|| BackendError::KernelLaunchFailed(format!("{op}: input not contiguous")))?;
    let out = if rows == 0 || cols == 0 {
        Vec::new()
    } else {
        run(x_slice, rows, cols).map_err(to_backend_error)?
    };
    Tensor::new(out, x.shape()).map_err(BackendError::ShapeMismatch)
}

impl BackendOps for RocmBackendOps {
    /// 仮置き: `Device::Cpu` を返す。`tensor-core::Device` に ROCm の variant がなく、追加は
    /// 破壊的変更で所有者判断待ちのため。承認後に #3157 Phase B で `Device::Rocm(usize)` へ
    /// 差し替える。それまで `ops_for` や autodiff のデバイス一致検査では CPU と区別されない
    /// （このため本型はどの候補リストにも登録しない）。
    fn device(&self) -> Device {
        Device::Cpu
    }

    /// `RocmGemmF32` へ結線した f32 GEMM。ROCm 非搭載では `DeviceUnavailable`（panic しない）。
    fn gemm(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        gemm_with(a, b, |a, b, m, n, k| self.run_gemm(a, b, m, n, k))
    }

    /// ブロードキャスト対応の `a + b`（HIPRTC カーネル。R2）。ROCm 非搭載では `DeviceUnavailable`。
    fn add(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.ew_binary(BinaryKind::Add, a, b)
    }

    /// ブロードキャスト対応の `a * b`（HIPRTC カーネル。R2）。
    fn mul(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.ew_binary(BinaryKind::Mul, a, b)
    }

    /// `x > 0 ? x : 0`（HIPRTC カーネル。R2。NaN は 0）。
    fn relu(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.ew_unary(UnaryKind::Relu, a)
    }

    /// `expf`（HIPRTC カーネル。R2）。
    fn exp(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.ew_unary(UnaryKind::Exp, a)
    }

    /// `tanhf`（HIPRTC カーネル。R2）。
    fn tanh(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.ew_unary(UnaryKind::Tanh, a)
    }

    /// `ScalarUnaryOp` 17 kind（対象外 kind は HIP に触れず `Unsupported`）。
    fn scalar_unary(
        &self,
        op: ScalarUnaryOp,
        a: &Tensor<f32>,
    ) -> Result<Tensor<f32>, BackendError> {
        self.ew_scalar_unary(op, a)
    }

    /// `ScalarBinaryOp` 9 kind（ブロードキャスト対応。対象外 kind は `Unsupported`）。
    fn scalar_binary(
        &self,
        op: ScalarBinaryOp,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
    ) -> Result<Tensor<f32>, BackendError> {
        self.ew_scalar_binary(op, a, b)
    }

    /// `cond != 0 ? a : b`。3 本の shape 一致を再検査し、不一致は `ShapeMismatch`。
    fn where_cond(
        &self,
        cond: &Tensor<f32>,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
    ) -> Result<Tensor<f32>, BackendError> {
        self.ew_where(cond, a, b)
    }

    /// `mask != 0 ? value : x`。shape 不一致は `ShapeMismatch`。
    fn masked_fill(
        &self,
        x: &Tensor<f32>,
        mask: &Tensor<f32>,
        value: f32,
    ) -> Result<Tensor<f32>, BackendError> {
        self.ew_masked_fill(x, mask, value)
    }

    /// elementwise 融合（静的インタプリタカーネル）。allowlist 外のプランは `Unsupported`。
    fn run_fused(
        &self,
        plan: &FusionPlan,
        leaves: &[&Tensor<f32>],
    ) -> Result<Tensor<f32>, BackendError> {
        self.ew_run_fused(plan, leaves)
    }

    /// デバイス常駐の `add`／`mul`。呼び出しごとに同期する（Metal と同じ。CUDA の非同期ストリームとは異なる）。
    fn binary_elementwise_device(
        &self,
        op: BinaryElementwiseOp,
        a: &DeviceBuffer<f32>,
        b: &DeviceBuffer<f32>,
    ) -> Result<DeviceBuffer<f32>, BackendError> {
        self.ew_binary_device(op, a, b)
    }

    /// デバイス常駐の `relu`／`exp`／`tanh`。呼び出しごとに同期する。
    fn unary_elementwise_device(
        &self,
        op: UnaryElementwiseOp,
        a: &DeviceBuffer<f32>,
    ) -> Result<DeviceBuffer<f32>, BackendError> {
        self.ew_unary_device(op, a)
    }

    /// HIPRTC の縮約（R3・#3434）。`f64` アキュムレータで逐次／2 段に加算し最後に 1 回 `f32` へ
    /// downcast する（CUDA 版と同じ契約）。全軸は block/lane の結合順序が CPU と異なるため統一複合判定、
    /// 軸指定は CPU と bit 一致。空縮約は CPU へ委譲。ROCm 非搭載では `DeviceUnavailable`。
    fn sum(&self, a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        self.reduce_dispatch(a, dim, ReduceKind::Sum)
    }

    /// HIPRTC の縮約（R3・#3434）。`fmaxf` の厳密選択（NaN を伝播しない。`f32::max` と同じ）。
    /// 空縮約は CPU の `EmptyReduction` と同じ結果にするため CPU へ委譲する。
    fn max(&self, a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        self.reduce_dispatch(a, dim, ReduceKind::Max)
    }

    /// HIPRTC の縮約（R3・#3434）。`fminf` の厳密選択。契約は [`Self::max`] と対称。
    fn min(&self, a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        self.reduce_dispatch(a, dim, ReduceKind::Min)
    }

    /// HIPRTC の添字縮約（R3・#3434）。タイは最初の添字・NaN は無視・全 NaN は 0
    /// （`backend-cpu::reduction::argmax` と同じ走査契約）。空縮約は CPU へ委譲する。
    fn argmax(&self, a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<i32>, BackendError> {
        self.arg_reduce_dispatch(a, dim, ArgReduceKind::Max)
    }

    /// HIPRTC の添字縮約（R3・#3434）。契約は [`Self::argmax`] と対称（`v < best` で更新）。
    fn argmin(&self, a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<i32>, BackendError> {
        self.arg_reduce_dispatch(a, dim, ArgReduceKind::Min)
    }

    /// HIPRTC の行 softmax（最終軸限定。R4・#3435）。非最終軸は `Unsupported`。
    fn softmax(&self, x: &Tensor<f32>, dim: usize) -> Result<Tensor<f32>, BackendError> {
        softmax_rows_with("softmax", x, dim, |x, rows, cols| match self.engine {
            Engine::Hip => {
                crate::softmax::cached_softmax(self.target)?.run_softmax_f32(x, rows, cols)
            }
            #[cfg(test)]
            Engine::HostModel => {
                crate::softmax::validate_softmax_dims("softmax", x.len(), rows, cols)?;
                Ok(crate::softmax_model::simulate_softmax(
                    self.target.wave_size() as usize,
                    x,
                    rows,
                    cols,
                    false,
                ))
            }
        })
    }

    /// HIPRTC の行 log_softmax（最終軸限定。R4・#3435）。非最終軸は `Unsupported`。
    fn log_softmax(&self, x: &Tensor<f32>, dim: usize) -> Result<Tensor<f32>, BackendError> {
        softmax_rows_with("log_softmax", x, dim, |x, rows, cols| match self.engine {
            Engine::Hip => {
                crate::softmax::cached_softmax(self.target)?.run_log_softmax_f32(x, rows, cols)
            }
            #[cfg(test)]
            Engine::HostModel => {
                crate::softmax::validate_softmax_dims("log_softmax", x.len(), rows, cols)?;
                Ok(crate::softmax_model::simulate_softmax(
                    self.target.wave_size() as usize,
                    x,
                    rows,
                    cols,
                    true,
                ))
            }
        })
    }

    /// HIPRTC の `log_softmax` backward（最終軸限定。R4・#3435）。`out`・`upstream` の shape 不一致は
    /// `ShapeMismatch`、非最終軸は `Unsupported`。
    fn log_softmax_backward(
        &self,
        out: &Tensor<f32>,
        upstream: &Tensor<f32>,
        dim: usize,
    ) -> Result<Tensor<f32>, BackendError> {
        if out.shape() != upstream.shape() {
            return Err(BackendError::ShapeMismatch(
                fandhe_ai_tensor_core::ShapeError::ShapeMismatch {
                    lhs: out.shape().to_vec(),
                    rhs: upstream.shape().to_vec(),
                },
            ));
        }
        let up = upstream.contiguous();
        let up_slice = up.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("log_softmax_backward: upstream not contiguous".into())
        })?;
        softmax_rows_with(
            "log_softmax_backward",
            out,
            dim,
            |y, rows, cols| match self.engine {
                Engine::Hip => crate::softmax::cached_log_softmax_backward(self.target)?
                    .run_log_softmax_backward_f32(y, up_slice, rows, cols),
                #[cfg(test)]
                Engine::HostModel => {
                    crate::softmax::validate_softmax_backward_dims(
                        "log_softmax_backward",
                        y.len(),
                        rows,
                        cols,
                    )?;
                    Ok(crate::softmax_model::simulate_log_softmax_backward(
                        self.target.wave_size() as usize,
                        y,
                        up_slice,
                        rows,
                        cols,
                    ))
                }
            },
        )
    }
}

/// テスト専用: ホストモデルが検出した未定義動作相当の読み書きを `RocmError` へ写す。
#[cfg(test)]
fn model_err(e: crate::reduce_model::ModelError) -> RocmError {
    RocmError::InvalidKernelConfig {
        detail: format!("ホストモデルが不正アクセスを検出: {e:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fandhe_ai_backend_cpu::{assert_parity, gemm::gemm_naive};

    fn ops() -> RocmBackendOps {
        RocmBackendOps::with_host_model(GfxTarget::Gfx942)
    }

    fn data(len: usize, seed: u32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                (((i as u32).wrapping_mul(2654435761).wrapping_add(seed) >> 8) % 2001) as f32
                    / 1000.0
                    - 1.0
            })
            .collect()
    }

    fn same(a: &Tensor<f32>, b: &Tensor<f32>) {
        assert_eq!(a.shape(), b.shape());
        let bits = |x: &Tensor<f32>| -> Vec<u32> {
            x.contiguous()
                .as_slice()
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect()
        };
        assert_eq!(bits(a), bits(b));
    }

    fn t(len: usize, shape: &[usize], seed: u32) -> Tensor<f32> {
        Tensor::new(data(len, seed), shape).unwrap()
    }

    #[test]
    fn is_send_sync_and_boxable_for_tape() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<RocmBackendOps>();
        let _b: Box<dyn BackendOps + Send> = Box::new(RocmBackendOps::new(GfxTarget::Gfx942));
    }

    /// 仮置き。#3157 Phase B で `Device::Rocm(usize)` に差し替える。
    #[test]
    fn device_is_cpu_placeholder() {
        assert_eq!(ops().device(), Device::Cpu);
    }

    #[test]
    fn gemm_host_model_matches_references() {
        let shapes = [
            (1, 1, 1),
            (63, 65, 17),
            (130, 70, 33),
            (5, 129, 200),
            (4, 0, 3),
            (0, 4, 3),
            (4, 3, 0),
        ];
        for (m, k, n) in shapes {
            let a = t(m * k, &[m, k], 1);
            let b = t(k * n, &[k, n], 2);
            let out = ops().gemm(&a, &b).unwrap();
            assert_eq!(out.shape(), &[m, n]);
            let mut want = vec![0.0f32; m * n];
            gemm_naive(
                a.as_slice().unwrap(),
                b.as_slice().unwrap(),
                &mut want,
                m,
                n,
                k,
            )
            .unwrap();
            assert_eq!(out.as_slice().unwrap(), &want[..], "{m}x{k}x{n}");
            let cpu = CpuBackendOps.gemm(&a, &b).unwrap();
            assert_parity(
                "rocm host model vs cpu",
                out.as_slice().unwrap(),
                cpu.as_slice().unwrap(),
            );
        }
    }

    #[test]
    fn gemm_non_contiguous_inputs() {
        let (m, k, n) = (7, 9, 5);
        let a = t(k * m, &[k, m], 3).transpose_2d().unwrap();
        let b = t(n * k, &[n, k], 4).transpose_2d().unwrap();
        let out = ops().gemm(&a, &b).unwrap();
        let cpu = CpuBackendOps.gemm(&a, &b).unwrap();
        assert_parity(
            "non-contiguous",
            out.as_slice().unwrap(),
            cpu.as_slice().unwrap(),
        );
    }

    #[test]
    fn gemm_shape_mismatch() {
        let o = ops();
        let v = t(3, &[3], 1);
        let m = t(6, &[2, 3], 1);
        assert!(matches!(
            o.gemm(&v, &m),
            Err(BackendError::ShapeMismatch(_))
        ));
        assert!(matches!(
            o.gemm(&m, &m),
            Err(BackendError::ShapeMismatch(_))
        ));
    }

    #[test]
    fn inherited_defaults_work() {
        let o = ops();
        let (a, b) = (t(6 * 4, &[6, 4], 1), t(4 * 5, &[4, 5], 2));
        let g = o.gemm(&a, &b).unwrap();
        let s = o.gemm_fp32_strict(&a, &b).unwrap();
        assert_eq!(g.as_slice(), s.as_slice());
        // rank 2 と rank 3（バッチのブロードキャストを含む）。
        let cases = [
            (a.clone(), b.clone()),
            (t(2 * 3 * 4, &[2, 3, 4], 5), t(2 * 4 * 5, &[2, 4, 5], 6)),
            (t(2 * 3 * 4, &[2, 3, 4], 5), t(4 * 5, &[1, 4, 5], 7)),
        ];
        for (x, y) in cases {
            let want = CpuBackendOps.gemm_batched(&x, &y).unwrap();
            let got = o.gemm_batched(&x, &y).unwrap();
            assert_eq!(got.shape(), want.shape());
            assert_parity(
                "gemm_batched",
                got.as_slice().unwrap(),
                want.as_slice().unwrap(),
            );
            let got = o.gemm_batched_fp32_strict(&x, &y).unwrap();
            assert_parity(
                "gemm_batched_strict",
                got.as_slice().unwrap(),
                want.as_slice().unwrap(),
            );
        }
        let x = t(6, &[2, 3], 1);
        // softmax 系は R4（#3435）で実装済み。未実装の代表として gather 系の既定を確かめる。
        assert!(matches!(o.cumsum(&x, 1), Err(BackendError::Unsupported(_))));
        assert!(o.memory_ops().is_none());
        assert!(o.typed_ops_f64().is_none());
    }

    #[test]
    fn hip_engine_without_rocm_is_unavailable() {
        if crate::is_available() {
            return;
        }
        let o = RocmBackendOps::new(GfxTarget::Gfx942);
        let (a, b) = (t(4, &[2, 2], 1), t(4, &[2, 2], 2));
        assert!(matches!(
            o.gemm(&a, &b),
            Err(BackendError::DeviceUnavailable(_))
        ));
        assert!(matches!(
            o.gemm_batched(&a, &b),
            Err(BackendError::DeviceUnavailable(_))
        ));
        // 空の形状は HIP を触らず Ok。
        let (e, f) = (t(0, &[0, 2], 1), t(4, &[2, 2], 2));
        assert_eq!(o.gemm(&e, &f).unwrap().shape(), &[0, 2]);

        // 縮約 5 メソッドも非搭載では DeviceUnavailable（panic しない）。
        for dim in [None, Some(0), Some(1)] {
            assert!(matches!(
                o.sum(&a, dim),
                Err(BackendError::DeviceUnavailable(_))
            ));
            assert!(matches!(
                o.max(&a, dim),
                Err(BackendError::DeviceUnavailable(_))
            ));
            assert!(matches!(
                o.min(&a, dim),
                Err(BackendError::DeviceUnavailable(_))
            ));
            assert!(matches!(
                o.argmax(&a, dim),
                Err(BackendError::DeviceUnavailable(_))
            ));
            assert!(matches!(
                o.argmin(&a, dim),
                Err(BackendError::DeviceUnavailable(_))
            ));
        }
        // 空縮約は HIP に触れず CPU と同じ結果・エラー（Debug 文字列まで一致）。
        let c = CpuBackendOps;
        let empty = Tensor::<f32>::zeros(&[0]).unwrap();
        let empty2 = Tensor::<f32>::zeros(&[2, 0]).unwrap();
        for (x, dim) in [
            (&empty, None),
            (&empty2, None),
            (&empty2, Some(0)),
            (&empty2, Some(1)),
        ] {
            assert_eq!(
                format!("{:?}", o.sum(x, dim)),
                format!("{:?}", c.sum(x, dim))
            );
            assert_eq!(
                format!("{:?}", o.max(x, dim)),
                format!("{:?}", c.max(x, dim))
            );
            assert_eq!(
                format!("{:?}", o.min(x, dim)),
                format!("{:?}", c.min(x, dim))
            );
            assert_eq!(
                format!("{:?}", o.argmax(x, dim)),
                format!("{:?}", c.argmax(x, dim))
            );
            assert_eq!(
                format!("{:?}", o.argmin(x, dim)),
                format!("{:?}", c.argmin(x, dim))
            );
        }
    }

    fn tt(rows: usize, cols: usize, seed: u32) -> Tensor<f32> {
        t(rows * cols, &[rows, cols], seed)
    }

    #[test]
    fn softmax_family_host_model_matches_cpu() {
        for target in [GfxTarget::Gfx942] {
            let o = RocmBackendOps::with_host_model(target);
            for (rows, cols) in [(1, 1), (3, 4), (5, 63), (2, 65), (4, 1000), (0, 4), (4, 0)] {
                let x = tt(rows, cols, 7);
                let c = CpuBackendOps;
                let s = o.softmax(&x, 1).unwrap();
                assert_eq!(s.shape(), x.shape());
                assert_parity(
                    "softmax",
                    s.as_slice().unwrap_or(&[]),
                    c.softmax(&x, 1).unwrap().as_slice().unwrap_or(&[]),
                );
                let l = o.log_softmax(&x, 1).unwrap();
                assert_parity(
                    "log_softmax",
                    l.as_slice().unwrap_or(&[]),
                    c.log_softmax(&x, 1).unwrap().as_slice().unwrap_or(&[]),
                );
                let g = tt(rows, cols, 9);
                let b = o.log_softmax_backward(&l, &g, 1).unwrap();
                assert_eq!(b.shape(), x.shape());
                if rows * cols > 0 {
                    // 参照: f64 逐次和（grad::log_softmax_vjp_along と同式）。
                    let (ys, gs) = (l.as_slice().unwrap(), g.as_slice().unwrap());
                    let mut want = vec![0.0f32; rows * cols];
                    for r in 0..rows {
                        let sum: f64 = (0..cols).map(|i| f64::from(gs[r * cols + i])).sum();
                        for i in 0..cols {
                            let e = f64::from(ys[r * cols + i].exp());
                            want[r * cols + i] = (f64::from(gs[r * cols + i]) - e * sum) as f32;
                        }
                    }
                    assert_parity("log_softmax_backward", b.as_slice().unwrap(), &want);
                }
            }
        }
    }

    #[test]
    fn softmax_family_rejects_non_last_axis_and_bad_shapes() {
        let o = ops();
        let x = tt(4, 6, 1);
        assert!(matches!(
            o.softmax(&x, 0),
            Err(BackendError::Unsupported(_))
        ));
        assert!(matches!(
            o.log_softmax(&x, 0),
            Err(BackendError::Unsupported(_))
        ));
        assert!(matches!(
            o.log_softmax_backward(&x, &x, 0),
            Err(BackendError::Unsupported(_))
        ));
        assert!(matches!(
            o.softmax(&x, 2),
            Err(BackendError::ShapeMismatch(_))
        ));
        let y = tt(4, 5, 2);
        assert!(matches!(
            o.log_softmax_backward(&x, &y, 1),
            Err(BackendError::ShapeMismatch(_))
        ));
    }

    #[test]
    fn softmax_family_handles_non_contiguous_input() {
        let o = ops();
        let x = t(6 * 4, &[6, 4], 3).transpose_2d().unwrap();
        let got = o.softmax(&x, 1).unwrap();
        let want = CpuBackendOps.softmax(&x, 1).unwrap();
        assert_parity(
            "non-contiguous",
            got.as_slice().unwrap(),
            want.as_slice().unwrap(),
        );
    }

    #[test]
    fn softmax_family_hip_engine_without_rocm_is_unavailable() {
        if crate::is_available() {
            return;
        }
        let o = RocmBackendOps::new(GfxTarget::Gfx942);
        let x = tt(2, 3, 1);
        assert!(matches!(
            o.softmax(&x, 1),
            Err(BackendError::DeviceUnavailable(_))
        ));
        assert!(matches!(
            o.log_softmax(&x, 1),
            Err(BackendError::DeviceUnavailable(_))
        ));
        assert!(matches!(
            o.log_softmax_backward(&x, &x, 1),
            Err(BackendError::DeviceUnavailable(_))
        ));
        // 空は HIP を触らず Ok。
        let e = tt(0, 3, 1);
        assert_eq!(o.softmax(&e, 1).unwrap().shape(), &[0, 3]);
    }

    /// sum/max は R3 までホスト経路（CPU 委譲）。add〜tanh は R2 でカーネル化済み（`ops/ew.rs` のテスト）。
    #[test]
    fn host_paths_match_cpu_bitwise() {
        let o = ops();
        let c = CpuBackendOps;
        let a = t(12, &[3, 4], 1);
        let b = t(4, &[1, 4], 2);
        same(&o.add(&a, &b).unwrap(), &c.add(&a, &b).unwrap());
        same(&o.mul(&a, &b).unwrap(), &c.mul(&a, &b).unwrap());
        same(&o.relu(&a).unwrap(), &c.relu(&a).unwrap());
        same(&o.exp(&a).unwrap(), &c.exp(&a).unwrap());
        same(&o.tanh(&a).unwrap(), &c.tanh(&a).unwrap());
        // R3 の縮約は HostModel（reduce_model／arg_reduce_model の逐語モデル）経由で CPU と照合する。
        // 軸指定の sum と max/min は bit 一致、全軸 sum は結合順序が異なるため統一複合判定。
        for dim in [Some(0), Some(1)] {
            same(&o.sum(&a, dim).unwrap(), &c.sum(&a, dim).unwrap());
        }
        assert_parity(
            "sum all",
            o.sum(&a, None).unwrap().as_slice().unwrap(),
            c.sum(&a, None).unwrap().as_slice().unwrap(),
        );
        for dim in [None, Some(0), Some(1)] {
            same(&o.max(&a, dim).unwrap(), &c.max(&a, dim).unwrap());
            same(&o.min(&a, dim).unwrap(), &c.min(&a, dim).unwrap());
            assert_eq!(
                o.argmax(&a, dim).unwrap().as_slice(),
                c.argmax(&a, dim).unwrap().as_slice()
            );
            assert_eq!(
                o.argmin(&a, dim).unwrap().as_slice(),
                c.argmin(&a, dim).unwrap().as_slice()
            );
        }
        let empty = Tensor::<f32>::zeros(&[0]).unwrap();
        let empty2 = Tensor::<f32>::zeros(&[2, 0]).unwrap();
        for (x, dim) in [
            (&empty, None),
            (&empty2, None),
            (&empty2, Some(0)),
            (&empty2, Some(1)),
        ] {
            assert_eq!(
                format!("{:?}", o.sum(x, dim)),
                format!("{:?}", c.sum(x, dim))
            );
            assert_eq!(
                format!("{:?}", o.max(x, dim)),
                format!("{:?}", c.max(x, dim))
            );
            assert_eq!(
                format!("{:?}", o.min(x, dim)),
                format!("{:?}", c.min(x, dim))
            );
            assert_eq!(
                format!("{:?}", o.argmax(x, dim)),
                format!("{:?}", c.argmax(x, dim))
            );
            assert_eq!(
                format!("{:?}", o.argmin(x, dim)),
                format!("{:?}", c.argmin(x, dim))
            );
        }
    }

    #[test]
    fn reduce_ops_non_contiguous_and_bad_dim() {
        let o = ops();
        let c = CpuBackendOps;
        // 非連続（転置 view）入力。
        let a = t(5 * 7, &[5, 7], 8).transpose_2d().unwrap();
        for dim in [Some(0), Some(1)] {
            same(&o.sum(&a, dim).unwrap(), &c.sum(&a, dim).unwrap());
            same(&o.max(&a, dim).unwrap(), &c.max(&a, dim).unwrap());
            same(&o.min(&a, dim).unwrap(), &c.min(&a, dim).unwrap());
            assert_eq!(
                o.argmax(&a, dim).unwrap().as_slice(),
                c.argmax(&a, dim).unwrap().as_slice()
            );
        }
        // 範囲外の dim は ShapeMismatch（CPU と同じ）。
        let x = t(6, &[2, 3], 1);
        assert!(matches!(
            o.sum(&x, Some(2)),
            Err(BackendError::ShapeMismatch(_))
        ));
        assert!(matches!(
            o.max(&x, Some(5)),
            Err(BackendError::ShapeMismatch(_))
        ));
        assert!(matches!(
            o.argmin(&x, Some(2)),
            Err(BackendError::ShapeMismatch(_))
        ));
        // 出力 shape と 3 次元の全軸・各軸。
        let y = t(2 * 3 * 5, &[2, 3, 5], 4);
        for dim in [Some(0), Some(1), Some(2)] {
            let got = o.sum(&y, dim).unwrap();
            assert_eq!(got.shape(), c.sum(&y, dim).unwrap().shape());
            same(&got, &c.sum(&y, dim).unwrap());
        }
        assert_eq!(o.max(&y, None).unwrap().shape(), &[] as &[usize]);
    }
}
