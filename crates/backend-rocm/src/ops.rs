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
//! - `add`〜`tanh` の 5 メソッドは入出力がホスト常駐 `Tensor<f32>` なので `CpuBackendOps` へ
//!   委譲する（CPU と bit 一致）。R2（elementwise）でカーネル化する。
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
//! - 正規化 6 メソッド（`rmsnorm`・`layer_norm`・`rmsnorm_backward`・`layer_norm_backward`・
//!   `batch_norm_train`・`batch_norm_infer`。R6・#3437）は [`RocmNorm`] へ結線する。形状導出
//!   （`row_norm_layout`／`batch_norm_layout`）・contiguous 化・`Tensor` 化は本ファイル、カーネル起動と
//!   起動前検証は `norm.rs` の責務。ROCm 非搭載では `DeviceUnavailable` を返す（`Unsupported` では
//!   ないため autodiff のホスト参照実装へは黙ってフォールバックしない。A08）。
//! - どの候補リスト（`ops_for`・facade の `resolve_ops`／`available_devices`）にも登録しない。

use std::cell::RefCell;
use std::rc::Rc;
use std::thread::LocalKey;

use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_tensor_core::{
    BackendError, BackendOps, BatchNormTrainOutput, Device, LayerNormBackwardOutput, ShapeError,
    Tensor, batch_norm_layout, gemm_out_shape, reduce_out_shape, row_norm_layout,
};

use crate::arg_reduce::{ArgReduceKind, RocmArgReduce};
use crate::error::{RocmError, to_backend_error};
use crate::gemm::RocmGemmF32;
use crate::norm::{
    BatchNormTrainHost, LayerNormBackwardHost, OptLens, RmsNormBackwardHost, RocmNorm,
    batch_norm_train_empty, layer_norm_backward_empty, rmsnorm_backward_empty, validate_batch_norm,
    validate_row_norm,
};
use crate::reduce::{ReduceKind, RocmReduce, reduce_axis_layout};
use crate::target::GfxTarget;
#[cfg(test)]
use crate::warp_geometry::WarpGeometry;

/// 実行経路。`HostModel` はテスト専用（実機なしで組み立て経路を検証する。
/// GEMM は `gemm_model`、縮約は `reduce_model`、正規化は `norm_model` で計算する）。
#[derive(Debug, Clone, Copy)]
enum Engine {
    Hip,
    #[cfg(test)]
    HostModel,
}

type Cache<T> = RefCell<Vec<(GfxTarget, Rc<T>)>>;

thread_local! {
    /// target ごとにコンパイル済みの GEMM を遅延保持する。失敗はキャッシュしない。
    static GEMM_CACHE: Cache<RocmGemmF32> = const { RefCell::new(Vec::new()) };
    /// 正規化（R6・#3437）。
    static NORM_CACHE: Cache<RocmNorm> = const { RefCell::new(Vec::new()) };
    /// 縮約（sum/max/min。#3434）。
    static REDUCE_CACHE: Cache<RocmReduce> = const { RefCell::new(Vec::new()) };
    /// 添字縮約（argmax/argmin。#3434）。
    static ARG_REDUCE_CACHE: Cache<RocmArgReduce> = const { RefCell::new(Vec::new()) };
}

/// thread_local キャッシュ `key` から `target` 用の実行型を取得する（無ければ `ctor` で構築して入れる）。
fn cached<T>(
    key: &'static LocalKey<Cache<T>>,
    target: GfxTarget,
    ctor: impl FnOnce(GfxTarget) -> Result<T, RocmError>,
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
    let g = Rc::new(ctor(target)?);
    let _ = key.try_with(|c| c.borrow_mut().push((target, Rc::clone(&g))));
    Ok(g)
}

/// ROCm（AMD GPU）向け `BackendOps`。必須 9 メソッド・縮約 5 メソッド（R3）・正規化 6 メソッド（R6）を実装する。
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

/// 正規化メソッドの `dy`／`x` の shape 不一致を `ShapeMismatch` に写す。
fn shape_mismatch(lhs: &Tensor<f32>, rhs: &Tensor<f32>) -> BackendError {
    BackendError::ShapeMismatch(ShapeError::ShapeMismatch {
        lhs: lhs.shape().to_vec(),
        rhs: rhs.shape().to_vec(),
    })
}

/// contiguous 化済みテンソルのスライスを取り出す（取り出せなければ `KernelLaunchFailed`）。
fn slice_of<'a>(t: &'a Tensor<f32>, op: &str, what: &str) -> Result<&'a [f32], BackendError> {
    t.as_slice()
        .ok_or_else(|| BackendError::KernelLaunchFailed(format!("{op}: {what} not contiguous")))
}

/// 正規化 6 メソッドの実行経路（Hip は `RocmNorm`、HostModel はテスト専用に `norm_model`）。
/// どちらも先に `norm.rs` の起動前検証と 0 要素契約を通す（0 要素は `cached(&NORM_CACHE, ..)` を呼ばず、
/// HIP のロードにも触れない）。Hip 側の `RocmNorm::run_*` は同じ検証を再度行う（多層防御）。
impl RocmBackendOps {
    fn run_rmsnorm(
        &self,
        x: &[f32],
        w: Option<&[f32]>,
        eps: f32,
        rows: usize,
        hidden: usize,
    ) -> Result<Vec<f32>, RocmError> {
        let lens = OptLens {
            w: w.map(<[f32]>::len),
            ..OptLens::default()
        };
        let d = validate_row_norm("rmsnorm", x.len(), rows, hidden, lens, eps)?;
        if d.numel == 0 {
            return Ok(Vec::new());
        }
        match self.engine {
            Engine::Hip => cached(&NORM_CACHE, self.target, RocmNorm::new)?
                .run_rmsnorm_f32(x, w, eps, rows, hidden),
            #[cfg(test)]
            Engine::HostModel => {
                let geom = WarpGeometry::for_target(self.target);
                Ok(crate::norm_model::rmsnorm(geom, x, w, eps, rows, hidden))
            }
        }
    }

    fn run_layer_norm(
        &self,
        x: &[f32],
        w: Option<&[f32]>,
        b: Option<&[f32]>,
        eps: f32,
        rows: usize,
        hidden: usize,
    ) -> Result<Vec<f32>, RocmError> {
        let lens = OptLens {
            w: w.map(<[f32]>::len),
            b: b.map(<[f32]>::len),
            ..OptLens::default()
        };
        let d = validate_row_norm("layer_norm", x.len(), rows, hidden, lens, eps)?;
        if d.numel == 0 {
            return Ok(Vec::new());
        }
        match self.engine {
            Engine::Hip => cached(&NORM_CACHE, self.target, RocmNorm::new)?
                .run_layer_norm_f32(x, w, b, eps, rows, hidden),
            #[cfg(test)]
            Engine::HostModel => {
                let geom = WarpGeometry::for_target(self.target);
                Ok(crate::norm_model::layer_norm(
                    geom, x, w, b, eps, rows, hidden,
                ))
            }
        }
    }

    fn run_rmsnorm_backward(
        &self,
        x: &[f32],
        w: Option<&[f32]>,
        dy: &[f32],
        eps: f32,
        rows: usize,
        hidden: usize,
    ) -> Result<RmsNormBackwardHost, RocmError> {
        let lens = OptLens {
            w: w.map(<[f32]>::len),
            dy: Some(dy.len()),
            ..OptLens::default()
        };
        let d = validate_row_norm("rmsnorm_backward", x.len(), rows, hidden, lens, eps)?;
        if d.numel == 0 {
            return rmsnorm_backward_empty(w.is_some(), hidden);
        }
        match self.engine {
            Engine::Hip => cached(&NORM_CACHE, self.target, RocmNorm::new)?
                .run_rmsnorm_backward_f32(x, w, dy, eps, rows, hidden),
            #[cfg(test)]
            Engine::HostModel => {
                let geom = WarpGeometry::for_target(self.target);
                Ok(crate::norm_model::rmsnorm_backward(
                    geom, x, w, dy, eps, rows, hidden,
                ))
            }
        }
    }

    #[allow(clippy::too_many_arguments)] // `RocmNorm::run_layer_norm_backward_f32` と同じ引数構成
    fn run_layer_norm_backward(
        &self,
        x: &[f32],
        w: Option<&[f32]>,
        has_bias: bool,
        dy: &[f32],
        eps: f32,
        rows: usize,
        hidden: usize,
    ) -> Result<LayerNormBackwardHost, RocmError> {
        let lens = OptLens {
            w: w.map(<[f32]>::len),
            dy: Some(dy.len()),
            ..OptLens::default()
        };
        let d = validate_row_norm("layer_norm_backward", x.len(), rows, hidden, lens, eps)?;
        if d.numel == 0 {
            return layer_norm_backward_empty(w.is_some(), has_bias, hidden);
        }
        match self.engine {
            Engine::Hip => cached(&NORM_CACHE, self.target, RocmNorm::new)?
                .run_layer_norm_backward_f32(x, w, has_bias, dy, eps, rows, hidden),
            #[cfg(test)]
            Engine::HostModel => {
                let geom = WarpGeometry::for_target(self.target);
                Ok(crate::norm_model::layer_norm_backward(
                    geom, x, w, has_bias, dy, eps, rows, hidden,
                ))
            }
        }
    }

    #[allow(clippy::too_many_arguments)] // `RocmNorm::run_batch_norm_train_f32` と同じ引数構成
    fn run_batch_norm_train(
        &self,
        x: &[f32],
        w: Option<&[f32]>,
        b: Option<&[f32]>,
        eps: f32,
        n: usize,
        c: usize,
        spatial: usize,
    ) -> Result<BatchNormTrainHost, RocmError> {
        let lens = OptLens {
            w: w.map(<[f32]>::len),
            b: b.map(<[f32]>::len),
            ..OptLens::default()
        };
        let d = validate_batch_norm("batch_norm_train", x.len(), (n, c, spatial), lens, eps)?;
        if d.numel == 0 {
            return batch_norm_train_empty(c);
        }
        match self.engine {
            Engine::Hip => cached(&NORM_CACHE, self.target, RocmNorm::new)?
                .run_batch_norm_train_f32(x, w, b, eps, n, c, spatial),
            #[cfg(test)]
            Engine::HostModel => {
                let geom = WarpGeometry::for_target(self.target);
                Ok(crate::norm_model::batch_norm_train(
                    geom, x, w, b, eps, n, c, spatial,
                ))
            }
        }
    }

    #[allow(clippy::too_many_arguments)] // `RocmNorm::run_batch_norm_infer_f32` と同じ引数構成
    fn run_batch_norm_infer(
        &self,
        x: &[f32],
        mean: &[f32],
        var: &[f32],
        w: Option<&[f32]>,
        b: Option<&[f32]>,
        eps: f32,
        n: usize,
        c: usize,
        spatial: usize,
    ) -> Result<Vec<f32>, RocmError> {
        let lens = OptLens {
            w: w.map(<[f32]>::len),
            b: b.map(<[f32]>::len),
            mean: Some(mean.len()),
            var: Some(var.len()),
            ..OptLens::default()
        };
        let d = validate_batch_norm("batch_norm_infer", x.len(), (n, c, spatial), lens, eps)?;
        if d.numel == 0 {
            return Ok(Vec::new());
        }
        match self.engine {
            Engine::Hip => cached(&NORM_CACHE, self.target, RocmNorm::new)?
                .run_batch_norm_infer_f32(x, mean, var, w, b, eps, n, c, spatial),
            #[cfg(test)]
            Engine::HostModel => Ok(crate::norm_model::batch_norm_infer(
                x,
                mean,
                var,
                w,
                b,
                eps,
                c,
                spatial,
                x.len(),
            )),
        }
    }
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

    /// R0 のホスト経路（CPU 委譲）。R2 でカーネル化する。性能は保証しない。
    fn add(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.cpu.add(a, b)
    }

    /// R0 のホスト経路（CPU 委譲）。R2 でカーネル化する。性能は保証しない。
    fn mul(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.cpu.mul(a, b)
    }

    /// R0 のホスト経路（CPU 委譲）。R2 でカーネル化する。性能は保証しない。
    fn relu(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.cpu.relu(a)
    }

    /// R0 のホスト経路（CPU 委譲）。R2 でカーネル化する。性能は保証しない。
    fn exp(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.cpu.exp(a)
    }

    /// R0 のホスト経路（CPU 委譲）。R2 でカーネル化する。性能は保証しない。
    fn tanh(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.cpu.tanh(a)
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

    /// 行ごとの RMSNorm。`row_norm_layout` で `(rows, hidden)` を導出し `RocmNorm` へ委譲する（R6）。
    fn rmsnorm(
        &self,
        x: &Tensor<f32>,
        weight: Option<&Tensor<f32>>,
        eps: f32,
    ) -> Result<Tensor<f32>, BackendError> {
        let (rows, hidden) = row_norm_layout(x.shape()).map_err(BackendError::ShapeMismatch)?;
        let x_owned = x.contiguous();
        let x_s = slice_of(&x_owned, "rmsnorm", "input")?;
        let w_owned = weight.map(|w| w.contiguous());
        let w_s = match &w_owned {
            Some(w) => Some(slice_of(w, "rmsnorm", "weight")?),
            None => None,
        };
        let out = self
            .run_rmsnorm(x_s, w_s, eps, rows, hidden)
            .map_err(to_backend_error)?;
        Tensor::new(out, x.shape()).map_err(BackendError::ShapeMismatch)
    }

    /// 行ごとの LayerNorm（biased ÷N）。形状導出は [`Self::rmsnorm`] と同じ（R6）。
    fn layer_norm(
        &self,
        x: &Tensor<f32>,
        weight: Option<&Tensor<f32>>,
        bias: Option<&Tensor<f32>>,
        eps: f32,
    ) -> Result<Tensor<f32>, BackendError> {
        let (rows, hidden) = row_norm_layout(x.shape()).map_err(BackendError::ShapeMismatch)?;
        let x_owned = x.contiguous();
        let x_s = slice_of(&x_owned, "layer_norm", "input")?;
        let w_owned = weight.map(|w| w.contiguous());
        let w_s = match &w_owned {
            Some(w) => Some(slice_of(w, "layer_norm", "weight")?),
            None => None,
        };
        let b_owned = bias.map(|b| b.contiguous());
        let b_s = match &b_owned {
            Some(b) => Some(slice_of(b, "layer_norm", "bias")?),
            None => None,
        };
        let out = self
            .run_layer_norm(x_s, w_s, b_s, eps, rows, hidden)
            .map_err(to_backend_error)?;
        Tensor::new(out, x.shape()).map_err(BackendError::ShapeMismatch)
    }

    /// RMSNorm の逆伝播 `(dx, dw)`。`dy.shape() != x.shape()` は `ShapeMismatch`（R6）。
    fn rmsnorm_backward(
        &self,
        x: &Tensor<f32>,
        weight: Option<&Tensor<f32>>,
        dy: &Tensor<f32>,
        eps: f32,
    ) -> Result<(Tensor<f32>, Option<Tensor<f32>>), BackendError> {
        let (rows, hidden) = row_norm_layout(x.shape()).map_err(BackendError::ShapeMismatch)?;
        if dy.shape() != x.shape() {
            return Err(shape_mismatch(dy, x));
        }
        let x_owned = x.contiguous();
        let x_s = slice_of(&x_owned, "rmsnorm_backward", "input")?;
        let w_owned = weight.map(|w| w.contiguous());
        let w_s = match &w_owned {
            Some(w) => Some(slice_of(w, "rmsnorm_backward", "weight")?),
            None => None,
        };
        let dy_owned = dy.contiguous();
        let dy_s = slice_of(&dy_owned, "rmsnorm_backward", "dy")?;
        let (dx, dw) = self
            .run_rmsnorm_backward(x_s, w_s, dy_s, eps, rows, hidden)
            .map_err(to_backend_error)?;
        let dx_t = Tensor::new(dx, x.shape()).map_err(BackendError::ShapeMismatch)?;
        let dw_t = match dw {
            Some(dw) => Some(Tensor::new(dw, &[hidden]).map_err(BackendError::ShapeMismatch)?),
            None => None,
        };
        Ok((dx_t, dw_t))
    }

    /// LayerNorm の逆伝播 `(dx, dw, db)`。`db` は `has_bias` のときだけ `Some`（R6）。
    fn layer_norm_backward(
        &self,
        x: &Tensor<f32>,
        weight: Option<&Tensor<f32>>,
        has_bias: bool,
        dy: &Tensor<f32>,
        eps: f32,
    ) -> Result<LayerNormBackwardOutput, BackendError> {
        let (rows, hidden) = row_norm_layout(x.shape()).map_err(BackendError::ShapeMismatch)?;
        if dy.shape() != x.shape() {
            return Err(shape_mismatch(dy, x));
        }
        let x_owned = x.contiguous();
        let x_s = slice_of(&x_owned, "layer_norm_backward", "input")?;
        let w_owned = weight.map(|w| w.contiguous());
        let w_s = match &w_owned {
            Some(w) => Some(slice_of(w, "layer_norm_backward", "weight")?),
            None => None,
        };
        let dy_owned = dy.contiguous();
        let dy_s = slice_of(&dy_owned, "layer_norm_backward", "dy")?;
        let (dx, dw, db) = self
            .run_layer_norm_backward(x_s, w_s, has_bias, dy_s, eps, rows, hidden)
            .map_err(to_backend_error)?;
        let dx_t = Tensor::new(dx, x.shape()).map_err(BackendError::ShapeMismatch)?;
        let dw_t = match dw {
            Some(dw) => Some(Tensor::new(dw, &[hidden]).map_err(BackendError::ShapeMismatch)?),
            None => None,
        };
        let db_t = match db {
            Some(db) => Some(Tensor::new(db, &[hidden]).map_err(BackendError::ShapeMismatch)?),
            None => None,
        };
        Ok((dx_t, dw_t, db_t))
    }

    /// BatchNorm の train（バッチ統計）。チャネル軸は dim 1（`batch_norm_layout`。R6）。
    fn batch_norm_train(
        &self,
        x: &Tensor<f32>,
        weight: Option<&Tensor<f32>>,
        bias: Option<&Tensor<f32>>,
        eps: f32,
    ) -> Result<BatchNormTrainOutput, BackendError> {
        let (n, c, spatial) = batch_norm_layout(x.shape()).map_err(BackendError::ShapeMismatch)?;
        let x_owned = x.contiguous();
        let x_s = slice_of(&x_owned, "batch_norm_train", "input")?;
        let w_owned = weight.map(|w| w.contiguous());
        let w_s = match &w_owned {
            Some(w) => Some(slice_of(w, "batch_norm_train", "weight")?),
            None => None,
        };
        let b_owned = bias.map(|b| b.contiguous());
        let b_s = match &b_owned {
            Some(b) => Some(slice_of(b, "batch_norm_train", "bias")?),
            None => None,
        };
        let (out, mean, var) = self
            .run_batch_norm_train(x_s, w_s, b_s, eps, n, c, spatial)
            .map_err(to_backend_error)?;
        Ok(BatchNormTrainOutput {
            output: Tensor::new(out, x.shape()).map_err(BackendError::ShapeMismatch)?,
            batch_mean: Tensor::new(mean, &[c]).map_err(BackendError::ShapeMismatch)?,
            batch_var: Tensor::new(var, &[c]).map_err(BackendError::ShapeMismatch)?,
        })
    }

    /// BatchNorm の infer（呼び出し元の running stats をそのまま使う。R6）。
    fn batch_norm_infer(
        &self,
        x: &Tensor<f32>,
        mean: &Tensor<f32>,
        var: &Tensor<f32>,
        weight: Option<&Tensor<f32>>,
        bias: Option<&Tensor<f32>>,
        eps: f32,
    ) -> Result<Tensor<f32>, BackendError> {
        let (n, c, spatial) = batch_norm_layout(x.shape()).map_err(BackendError::ShapeMismatch)?;
        let x_owned = x.contiguous();
        let x_s = slice_of(&x_owned, "batch_norm_infer", "input")?;
        let mean_owned = mean.contiguous();
        let mean_s = slice_of(&mean_owned, "batch_norm_infer", "mean")?;
        let var_owned = var.contiguous();
        let var_s = slice_of(&var_owned, "batch_norm_infer", "var")?;
        let w_owned = weight.map(|w| w.contiguous());
        let w_s = match &w_owned {
            Some(w) => Some(slice_of(w, "batch_norm_infer", "weight")?),
            None => None,
        };
        let b_owned = bias.map(|b| b.contiguous());
        let b_s = match &b_owned {
            Some(b) => Some(slice_of(b, "batch_norm_infer", "bias")?),
            None => None,
        };
        let out = self
            .run_batch_norm_infer(x_s, mean_s, var_s, w_s, b_s, eps, n, c, spatial)
            .map_err(to_backend_error)?;
        Tensor::new(out, x.shape()).map_err(BackendError::ShapeMismatch)
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
        assert!(matches!(
            o.softmax(&x, 1),
            Err(BackendError::Unsupported(_))
        ));
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

    /// R6（#3437）の正規化 6 メソッドのテスト（HostModel = gfx942 = wave64）。
    mod norm {
        use super::*;
        use crate::norm_model::tests::{ref_layer_norm_backward, ref_rmsnorm_backward};

        const EPS: f32 = 1e-5;

        fn opt(t: &Option<Tensor<f32>>) -> Option<&Tensor<f32>> {
            t.as_ref()
        }

        fn flat(t: &Tensor<f32>) -> Vec<f32> {
            t.contiguous().as_slice().unwrap().to_vec()
        }

        #[test]
        fn forward_matches_cpu_for_ranks_and_affine() {
            let c = CpuBackendOps;
            for shape in [vec![33usize], vec![5, 70], vec![2, 3, 65], vec![4, 1]] {
                let n: usize = shape.iter().product();
                let hidden = *shape.last().unwrap();
                let x = t(n, &shape, 1);
                let w = t(hidden, &[hidden], 2);
                let b = t(hidden, &[hidden], 3);
                for wo in [None, Some(w.clone())] {
                    let got = ops().rmsnorm(&x, opt(&wo), EPS).unwrap();
                    let want = c.rmsnorm(&x, opt(&wo), EPS).unwrap();
                    assert_eq!(got.shape(), x.shape());
                    assert_parity("rmsnorm", &flat(&got), &flat(&want));
                    for bo in [None, Some(b.clone())] {
                        let got = ops().layer_norm(&x, opt(&wo), opt(&bo), EPS).unwrap();
                        let want = c.layer_norm(&x, opt(&wo), opt(&bo), EPS).unwrap();
                        assert_eq!(got.shape(), x.shape());
                        assert_parity("layer_norm", &flat(&got), &flat(&want));
                    }
                }
            }
        }

        #[test]
        fn non_contiguous_inputs() {
            let c = CpuBackendOps;
            let x = t(7 * 9, &[9, 7], 4).transpose_2d().unwrap();
            let w = t(9, &[9], 5);
            let got = ops().layer_norm(&x, Some(&w), None, EPS).unwrap();
            let want = c.layer_norm(&x, Some(&w), None, EPS).unwrap();
            assert_parity("ln non-contig", &flat(&got), &flat(&want));
            let dy = t(7 * 9, &[9, 7], 6).transpose_2d().unwrap();
            let (dx, dw) = ops().rmsnorm_backward(&x, Some(&w), &dy, EPS).unwrap();
            let (rdx, rdw) =
                ref_rmsnorm_backward(&flat(&x), Some(&flat(&w)), &flat(&dy), EPS, 7, 9);
            assert_parity("rms_bwd non-contig dx", &flat(&dx), &rdx);
            assert_parity("rms_bwd non-contig dw", &flat(&dw.unwrap()), &rdw.unwrap());
        }

        #[test]
        fn backward_matches_reference() {
            for shape in [vec![70usize], vec![5, 70], vec![2, 3, 33]] {
                let n: usize = shape.iter().product();
                let hidden = *shape.last().unwrap();
                let rows = n / hidden;
                let x = t(n, &shape, 11);
                let dy = t(n, &shape, 12);
                let w = t(hidden, &[hidden], 13);
                for wo in [None, Some(w.clone())] {
                    let wf = wo.as_ref().map(flat);
                    let (dx, dw) = ops().rmsnorm_backward(&x, opt(&wo), &dy, EPS).unwrap();
                    let (rdx, rdw) = ref_rmsnorm_backward(
                        &flat(&x),
                        wf.as_deref(),
                        &flat(&dy),
                        EPS,
                        rows,
                        hidden,
                    );
                    assert_eq!(dx.shape(), x.shape());
                    assert_parity("rmsnorm_backward dx", &flat(&dx), &rdx);
                    assert_eq!(dw.is_some(), wo.is_some());
                    if let Some(dw) = dw {
                        assert_eq!(dw.shape(), &[hidden]);
                        assert_parity("rmsnorm_backward dw", &flat(&dw), &rdw.unwrap());
                    }
                    for has_bias in [false, true] {
                        let (dx, dw, db) = ops()
                            .layer_norm_backward(&x, opt(&wo), has_bias, &dy, EPS)
                            .unwrap();
                        let (rdx, rdw, rdb) = ref_layer_norm_backward(
                            &flat(&x),
                            wf.as_deref(),
                            has_bias,
                            &flat(&dy),
                            EPS,
                            rows,
                            hidden,
                        );
                        assert_parity("layer_norm_backward dx", &flat(&dx), &rdx);
                        assert_eq!((dw.is_some(), db.is_some()), (wo.is_some(), has_bias));
                        if let Some(dw) = dw {
                            assert_parity("layer_norm_backward dw", &flat(&dw), &rdw.unwrap());
                        }
                        if let Some(db) = db {
                            assert_eq!(db.shape(), &[hidden]);
                            assert_parity("layer_norm_backward db", &flat(&db), &rdb.unwrap());
                        }
                    }
                }
            }
        }

        #[test]
        fn batch_norm_matches_cpu_for_ncl_and_nchw() {
            let c = CpuBackendOps;
            for shape in [vec![4usize, 3], vec![2, 3, 17], vec![2, 3, 4, 5]] {
                let n: usize = shape.iter().product();
                let ch = shape[1];
                let x = t(n, &shape, 21);
                let w = t(ch, &[ch], 22);
                let b = t(ch, &[ch], 23);
                for wo in [None, Some(w.clone())] {
                    for bo in [None, Some(b.clone())] {
                        let got = ops().batch_norm_train(&x, opt(&wo), opt(&bo), EPS).unwrap();
                        let want = c.batch_norm_train(&x, opt(&wo), opt(&bo), EPS).unwrap();
                        assert_eq!(got.output.shape(), x.shape());
                        assert_eq!(got.batch_mean.shape(), &[ch]);
                        assert_eq!(got.batch_var.shape(), &[ch]);
                        assert_parity("bn out", &flat(&got.output), &flat(&want.output));
                        assert_parity("bn mean", &flat(&got.batch_mean), &flat(&want.batch_mean));
                        assert_parity("bn var", &flat(&got.batch_var), &flat(&want.batch_var));

                        let mean = t(ch, &[ch], 24);
                        let var = Tensor::new(
                            data(ch, 25).iter().map(|v| v.abs() + 0.1).collect(),
                            &[ch],
                        )
                        .unwrap();
                        let got = ops()
                            .batch_norm_infer(&x, &mean, &var, opt(&wo), opt(&bo), EPS)
                            .unwrap();
                        let want = c
                            .batch_norm_infer(&x, &mean, &var, opt(&wo), opt(&bo), EPS)
                            .unwrap();
                        same(&got, &want);
                    }
                }
            }
        }

        #[test]
        fn zero_element_contracts() {
            let o = ops();
            let x = t(0, &[0, 4], 1);
            let w = t(4, &[4], 2);
            assert_eq!(o.rmsnorm(&x, Some(&w), EPS).unwrap().shape(), &[0, 4]);
            assert_eq!(o.layer_norm(&x, None, None, EPS).unwrap().shape(), &[0, 4]);
            let (dx, dw) = o.rmsnorm_backward(&x, Some(&w), &x, EPS).unwrap();
            assert_eq!(dx.shape(), &[0, 4]);
            assert_eq!(flat(&dw.unwrap()), vec![0.0; 4]);
            let (dx, dw, db) = o.layer_norm_backward(&x, Some(&w), true, &x, EPS).unwrap();
            assert_eq!(dx.shape(), &[0, 4]);
            assert_eq!(flat(&dw.unwrap()), vec![0.0; 4]);
            assert_eq!(flat(&db.unwrap()), vec![0.0; 4]);
            let (dx, dw, db) = o.layer_norm_backward(&x, None, false, &x, EPS).unwrap();
            assert_eq!(dx.shape(), &[0, 4]);
            assert!(dw.is_none() && db.is_none());
            // hidden = 0。
            let z = t(0, &[3, 0], 1);
            let (dx, dw) = o.rmsnorm_backward(&z, None, &z, EPS).unwrap();
            assert_eq!(dx.shape(), &[3, 0]);
            assert!(dw.is_none());
            // batch_norm（n = 0）。
            let bx = t(0, &[0, 3, 2], 1);
            let tr = o.batch_norm_train(&bx, None, None, EPS).unwrap();
            assert_eq!(tr.output.shape(), &[0, 3, 2]);
            assert_eq!(flat(&tr.batch_mean), vec![0.0; 3]);
            assert_eq!(flat(&tr.batch_var), vec![0.0; 3]);
            let m = t(3, &[3], 1);
            assert_eq!(
                o.batch_norm_infer(&bx, &m, &m, None, None, EPS)
                    .unwrap()
                    .shape(),
                &[0, 3, 2]
            );
        }

        #[test]
        fn shape_and_argument_errors() {
            let o = ops();
            let x = t(12, &[3, 4], 1);
            let dy_bad = t(12, &[4, 3], 1);
            assert!(matches!(
                o.rmsnorm_backward(&x, None, &dy_bad, EPS),
                Err(BackendError::ShapeMismatch(_))
            ));
            assert!(matches!(
                o.layer_norm_backward(&x, None, false, &dy_bad, EPS),
                Err(BackendError::ShapeMismatch(_))
            ));
            // rank 0 は row_norm_layout が拒否、rank 1 は batch_norm_layout が拒否する。
            let scalar = Tensor::new(vec![1.0f32], &[]).unwrap();
            assert!(matches!(
                o.rmsnorm(&scalar, None, EPS),
                Err(BackendError::ShapeMismatch(_))
            ));
            let v = t(4, &[4], 1);
            assert!(matches!(
                o.batch_norm_train(&v, None, None, EPS),
                Err(BackendError::ShapeMismatch(_))
            ));
            // eps と長さ不一致は起動前に KernelLaunchFailed（InvalidOpArgs の写像）。
            for eps in [f32::NAN, -1.0, f32::INFINITY] {
                assert!(matches!(
                    o.rmsnorm(&x, None, eps),
                    Err(BackendError::KernelLaunchFailed(_))
                ));
                assert!(matches!(
                    o.layer_norm(&x, None, None, eps),
                    Err(BackendError::KernelLaunchFailed(_))
                ));
            }
            let w_bad = t(5, &[5], 1);
            assert!(matches!(
                o.rmsnorm(&x, Some(&w_bad), EPS),
                Err(BackendError::KernelLaunchFailed(_))
            ));
            assert!(matches!(
                o.layer_norm(&x, None, Some(&w_bad), EPS),
                Err(BackendError::KernelLaunchFailed(_))
            ));
            let bx = t(12, &[2, 3, 2], 1);
            assert!(matches!(
                o.batch_norm_train(&bx, Some(&w_bad), None, EPS),
                Err(BackendError::KernelLaunchFailed(_))
            ));
            let st = t(3, &[3], 1);
            assert!(matches!(
                o.batch_norm_infer(&bx, &st, &w_bad, None, None, EPS),
                Err(BackendError::KernelLaunchFailed(_))
            ));
        }

        #[test]
        fn hip_engine_without_rocm_is_unavailable() {
            if crate::is_available() {
                return;
            }
            let o = RocmBackendOps::new(GfxTarget::Gfx942);
            let x = t(12, &[3, 4], 1);
            let st = t(3, &[3], 1);
            let bx = t(12, &[2, 3, 2], 1);
            let unavailable = |r: Result<(), BackendError>| {
                assert!(matches!(r, Err(BackendError::DeviceUnavailable(_))));
            };
            unavailable(o.rmsnorm(&x, None, EPS).map(|_| ()));
            unavailable(o.layer_norm(&x, None, None, EPS).map(|_| ()));
            unavailable(o.rmsnorm_backward(&x, None, &x, EPS).map(|_| ()));
            unavailable(o.layer_norm_backward(&x, None, false, &x, EPS).map(|_| ()));
            unavailable(o.batch_norm_train(&bx, None, None, EPS).map(|_| ()));
            unavailable(
                o.batch_norm_infer(&bx, &st, &st, None, None, EPS)
                    .map(|_| ()),
            );
            // 0 要素は HIP に触らず Ok。
            let e = t(0, &[0, 4], 1);
            assert!(o.rmsnorm(&e, None, EPS).is_ok());
            assert!(o.layer_norm_backward(&e, None, true, &e, EPS).is_ok());
            let be = t(0, &[0, 3, 2], 1);
            assert!(o.batch_norm_train(&be, None, None, EPS).is_ok());
        }
    }
}
