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
//! - R5（#3436）で loss 10 メソッド（`mse_loss`・`huber_loss`・`bce_loss`・`nll_loss`・`kl_div_loss`
//!   と各 backward）を [`RocmLoss`](crate::RocmLoss) へ結線した。shape 検査・`factor` 決定・enum
//!   から `i32` への写像（未知 variant は `Unsupported`）・空入力の早期リターンはデバイス取得より
//!   前に行うので、ROCm 非搭載でも型付きエラーを検査できる。
//! - どの候補リスト（`ops_for`・facade の `resolve_ops`／`available_devices`）にも登録しない。

use std::cell::RefCell;
use std::rc::Rc;
use std::thread::LocalKey;

use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_tensor_core::{
    BackendError, BackendOps, BceKind, Device, Element, HuberKind, KlDivTarget, MseReduction,
    ShapeError, Tensor, gemm_out_shape, reduce_out_shape, require_same_shape,
};

use crate::arg_reduce::{ArgReduceKind, RocmArgReduce};
use crate::error::{RocmError, to_backend_error};
use crate::gemm::RocmGemmF32;
use crate::loss::{cached_loss, first_bad_target};
use crate::reduce::{ReduceKind, RocmReduce, reduce_axis_layout};
use crate::target::GfxTarget;

/// 実行経路。`HostModel` はテスト専用（実機なしで組み立て経路を検証する。GEMM は `gemm_model`、
/// loss 10 メソッドは `loss_model` の逐語モデルを使う）。
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

/// ROCm（AMD GPU）向け `BackendOps`。必須 9 メソッドと縮約 5 メソッド（R3）を実装する。
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

/// contiguous 化済みテンソルのホストスライス（`contiguous()` の戻りは常に `Some`。契約違反は型付きエラー）。
fn host_slice<'a, T: Element>(t: &'a Tensor<T>, op: &'static str) -> Result<&'a [T], BackendError> {
    t.as_slice()
        .ok_or_else(|| BackendError::KernelLaunchFailed(format!("{op}: tensor not contiguous")))
}

/// スカラー損失 `Tensor`（shape `[]`）。
fn scalar(v: f32) -> Result<Tensor<f32>, BackendError> {
    Tensor::new(vec![v], &[]).map_err(BackendError::ShapeMismatch)
}

/// `Mean` は `1/n`（空は `1.0`）、`Sum` は `1.0`。未知の variant（`#[non_exhaustive]`）は `Unsupported`。
fn reduction_factor(
    op: &'static str,
    reduction: MseReduction,
    n: usize,
) -> Result<f32, BackendError> {
    match reduction {
        MseReduction::Mean => Ok(if n == 0 { 1.0 } else { 1.0 / n as f32 }),
        MseReduction::Sum => Ok(1.0),
        _ => Err(BackendError::Unsupported(format!(
            "{op}: unsupported MseReduction variant {reduction:?}"
        ))),
    }
}

/// enum からカーネルの `kind`（`int`）への allowlist 写像。未知の variant は fail-closed。
fn huber_kind_i32(kind: HuberKind) -> Result<i32, BackendError> {
    match kind {
        HuberKind::Huber => Ok(0),
        HuberKind::SmoothL1 => Ok(1),
        _ => Err(BackendError::Unsupported(
            "huber_loss: unsupported HuberKind variant".into(),
        )),
    }
}

fn bce_kind_i32(kind: BceKind) -> Result<i32, BackendError> {
    match kind {
        BceKind::Probabilities => Ok(0),
        BceKind::Logits => Ok(1),
        _ => Err(BackendError::Unsupported(
            "bce_loss: unsupported BceKind variant".into(),
        )),
    }
}

fn kl_div_kind_i32(kind: KlDivTarget) -> Result<i32, BackendError> {
    match kind {
        KlDivTarget::Probabilities => Ok(0),
        KlDivTarget::LogProbabilities => Ok(1),
        _ => Err(BackendError::Unsupported(
            "kl_div_loss: unsupported KlDivTarget variant".into(),
        )),
    }
}

/// `(outer, num_classes, inner)`。`class_dim` が rank 以上なら `AxisOutOfRange`、積の overflow は
/// `ElementCountOverflow`（CPU の `nll_loss` と同じ種別）。
fn nll_layout(shape: &[usize], class_dim: usize) -> Result<(usize, usize, usize), BackendError> {
    if class_dim >= shape.len() {
        return Err(BackendError::ShapeMismatch(ShapeError::AxisOutOfRange {
            axis: class_dim,
            rank: shape.len(),
        }));
    }
    let prod = |s: &[usize]| {
        s.iter()
            .try_fold(1usize, |a, &d| a.checked_mul(d))
            .ok_or(BackendError::ShapeMismatch(
                ShapeError::ElementCountOverflow,
            ))
    };
    Ok((
        prod(&shape[..class_dim])?,
        shape[class_dim],
        prod(&shape[class_dim + 1..])?,
    ))
}

/// nll の `targets` 長（`outer*inner`）と添字値域を CPU の `nll_loss` と同じエラー種別で検査する。
fn check_nll_targets(
    targets: &[i32],
    outer: usize,
    num_classes: usize,
    inner: usize,
) -> Result<usize, BackendError> {
    let n = outer.checked_mul(inner).ok_or(BackendError::ShapeMismatch(
        ShapeError::ElementCountOverflow,
    ))?;
    if targets.len() != n {
        return Err(BackendError::ShapeMismatch(
            ShapeError::ElementCountMismatch {
                expected: n,
                actual: targets.len(),
            },
        ));
    }
    if let Some(bad) = first_bad_target(targets, num_classes) {
        return Err(BackendError::InvalidArgument(format!(
            "nll_loss: target index {bad} out of range [0, {num_classes})"
        )));
    }
    Ok(n)
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

    // ---- R5（#3436）: loss 5 種と backward。検証はすべてデバイス取得より前に行う ----

    /// mse forward（`RocmLoss` の 2 段縮約）。
    fn mse_loss(
        &self,
        pred: &Tensor<f32>,
        target: &Tensor<f32>,
        reduction: MseReduction,
    ) -> Result<Tensor<f32>, BackendError> {
        const OP: &str = "mse_loss";
        require_same_shape(pred.shape(), target.shape()).map_err(BackendError::ShapeMismatch)?;
        let (pc, tc) = (pred.contiguous(), target.contiguous());
        let (p, t) = (host_slice(&pc, OP)?, host_slice(&tc, OP)?);
        let factor = reduction_factor(OP, reduction, p.len())?;
        crate::loss::validate_pair(OP, p.len(), t.len()).map_err(to_backend_error)?;
        if p.is_empty() {
            return scalar(0.0);
        }
        let v = match self.engine {
            Engine::Hip => cached_loss(self.target)
                .and_then(|l| l.run_mse_loss_f32(p, t, factor))
                .map_err(to_backend_error)?,
            #[cfg(test)]
            Engine::HostModel => crate::loss_model::mse(
                crate::warp_geometry::WarpGeometry::for_target(self.target),
                p,
                t,
                factor,
            ),
        };
        scalar(v)
    }

    /// mse backward（`dpred` のみ）。
    fn mse_loss_backward(
        &self,
        pred: &Tensor<f32>,
        target: &Tensor<f32>,
        scale: f32,
    ) -> Result<Tensor<f32>, BackendError> {
        const OP: &str = "mse_loss";
        require_same_shape(pred.shape(), target.shape()).map_err(BackendError::ShapeMismatch)?;
        let (pc, tc) = (pred.contiguous(), target.contiguous());
        let (p, t) = (host_slice(&pc, OP)?, host_slice(&tc, OP)?);
        crate::loss::validate_pair(OP, p.len(), t.len()).map_err(to_backend_error)?;
        let out = if p.is_empty() {
            Vec::new()
        } else {
            match self.engine {
                Engine::Hip => cached_loss(self.target)
                    .and_then(|l| l.run_mse_backward_f32(p, t, scale))
                    .map_err(to_backend_error)?,
                #[cfg(test)]
                Engine::HostModel => crate::loss_model::mse_backward(p, t, scale),
            }
        };
        Tensor::new(out, pred.shape()).map_err(BackendError::ShapeMismatch)
    }

    /// huber／smooth_l1 forward。
    fn huber_loss(
        &self,
        pred: &Tensor<f32>,
        target: &Tensor<f32>,
        kind: HuberKind,
        delta: f32,
        reduction: MseReduction,
    ) -> Result<Tensor<f32>, BackendError> {
        const OP: &str = "huber_loss";
        require_same_shape(pred.shape(), target.shape()).map_err(BackendError::ShapeMismatch)?;
        let (pc, tc) = (pred.contiguous(), target.contiguous());
        let (p, t) = (host_slice(&pc, OP)?, host_slice(&tc, OP)?);
        let kind_i = huber_kind_i32(kind)?;
        let factor = reduction_factor(OP, reduction, p.len())?;
        crate::loss::validate_pair(OP, p.len(), t.len()).map_err(to_backend_error)?;
        if p.is_empty() {
            return scalar(0.0);
        }
        let v = match self.engine {
            Engine::Hip => cached_loss(self.target)
                .and_then(|l| l.run_huber_loss_f32(p, t, kind_i, delta, factor))
                .map_err(to_backend_error)?,
            #[cfg(test)]
            Engine::HostModel => crate::loss_model::huber(
                crate::warp_geometry::WarpGeometry::for_target(self.target),
                p,
                t,
                kind_i,
                delta,
                factor,
            ),
        };
        scalar(v)
    }

    /// huber／smooth_l1 backward（`dpred` のみ）。
    fn huber_loss_backward(
        &self,
        pred: &Tensor<f32>,
        target: &Tensor<f32>,
        kind: HuberKind,
        delta: f32,
        scale: f32,
    ) -> Result<Tensor<f32>, BackendError> {
        const OP: &str = "huber_loss";
        require_same_shape(pred.shape(), target.shape()).map_err(BackendError::ShapeMismatch)?;
        let (pc, tc) = (pred.contiguous(), target.contiguous());
        let (p, t) = (host_slice(&pc, OP)?, host_slice(&tc, OP)?);
        let kind_i = huber_kind_i32(kind)?;
        crate::loss::validate_pair(OP, p.len(), t.len()).map_err(to_backend_error)?;
        let out = if p.is_empty() {
            Vec::new()
        } else {
            match self.engine {
                Engine::Hip => cached_loss(self.target)
                    .and_then(|l| l.run_huber_backward_f32(p, t, kind_i, delta, scale))
                    .map_err(to_backend_error)?,
                #[cfg(test)]
                Engine::HostModel => crate::loss_model::huber_backward(p, t, kind_i, delta, scale),
            }
        };
        Tensor::new(out, pred.shape()).map_err(BackendError::ShapeMismatch)
    }

    /// bce forward。
    fn bce_loss(
        &self,
        input: &Tensor<f32>,
        target: &Tensor<f32>,
        kind: BceKind,
        reduction: MseReduction,
    ) -> Result<Tensor<f32>, BackendError> {
        const OP: &str = "bce_loss";
        require_same_shape(input.shape(), target.shape()).map_err(BackendError::ShapeMismatch)?;
        let (ic, tc) = (input.contiguous(), target.contiguous());
        let (x, t) = (host_slice(&ic, OP)?, host_slice(&tc, OP)?);
        let kind_i = bce_kind_i32(kind)?;
        let factor = reduction_factor(OP, reduction, x.len())?;
        crate::loss::validate_pair(OP, x.len(), t.len()).map_err(to_backend_error)?;
        if x.is_empty() {
            return scalar(0.0);
        }
        let v = match self.engine {
            Engine::Hip => cached_loss(self.target)
                .and_then(|l| l.run_bce_loss_f32(x, t, kind_i, factor))
                .map_err(to_backend_error)?,
            #[cfg(test)]
            Engine::HostModel => crate::loss_model::bce(
                crate::warp_geometry::WarpGeometry::for_target(self.target),
                x,
                t,
                kind_i,
                factor,
            ),
        };
        scalar(v)
    }

    /// bce backward（`dinput` のみ）。
    fn bce_loss_backward(
        &self,
        input: &Tensor<f32>,
        target: &Tensor<f32>,
        kind: BceKind,
        scale: f32,
    ) -> Result<Tensor<f32>, BackendError> {
        const OP: &str = "bce_loss";
        require_same_shape(input.shape(), target.shape()).map_err(BackendError::ShapeMismatch)?;
        let (ic, tc) = (input.contiguous(), target.contiguous());
        let (x, t) = (host_slice(&ic, OP)?, host_slice(&tc, OP)?);
        let kind_i = bce_kind_i32(kind)?;
        crate::loss::validate_pair(OP, x.len(), t.len()).map_err(to_backend_error)?;
        let out = if x.is_empty() {
            Vec::new()
        } else {
            match self.engine {
                Engine::Hip => cached_loss(self.target)
                    .and_then(|l| l.run_bce_backward_f32(x, t, kind_i, scale))
                    .map_err(to_backend_error)?,
                #[cfg(test)]
                Engine::HostModel => crate::loss_model::bce_backward(x, t, kind_i, scale),
            }
        };
        Tensor::new(out, input.shape()).map_err(BackendError::ShapeMismatch)
    }

    /// nll forward（`input` は `[.., C, ..]`、`targets` は class 軸を除いた shape の `i32`）。
    fn nll_loss(
        &self,
        input: &Tensor<f32>,
        targets: &Tensor<i32>,
        class_dim: usize,
        reduction: MseReduction,
    ) -> Result<Tensor<f32>, BackendError> {
        const OP: &str = "nll_loss";
        let (outer, classes, inner) = nll_layout(input.shape(), class_dim)?;
        let (ic, tc) = (input.contiguous(), targets.contiguous());
        let (x, t) = (host_slice(&ic, OP)?, host_slice(&tc, OP)?);
        let n = check_nll_targets(t, outer, classes, inner)?;
        let factor = reduction_factor(OP, reduction, n)?;
        crate::loss::validate_nll(OP, Some(x.len()), t, outer, classes, inner)
            .map_err(to_backend_error)?;
        if n == 0 {
            return scalar(0.0);
        }
        let v = match self.engine {
            Engine::Hip => cached_loss(self.target)
                .and_then(|l| l.run_nll_loss_f32(x, t, outer, classes, inner, factor))
                .map_err(to_backend_error)?,
            #[cfg(test)]
            Engine::HostModel => crate::loss_model::nll(
                crate::warp_geometry::WarpGeometry::for_target(self.target),
                x,
                t,
                outer,
                classes,
                inner,
                factor,
            ),
        };
        scalar(v)
    }

    /// nll backward（`dinput` のみ。ターゲット位置が `-scale`、ほかは 0）。
    fn nll_loss_backward(
        &self,
        input_shape: &[usize],
        targets: &Tensor<i32>,
        class_dim: usize,
        scale: f32,
    ) -> Result<Tensor<f32>, BackendError> {
        const OP: &str = "nll_loss";
        let (outer, classes, inner) = nll_layout(input_shape, class_dim)?;
        let tc = targets.contiguous();
        let t = host_slice(&tc, OP)?;
        let n = check_nll_targets(t, outer, classes, inner)?;
        // `input_shape` は利用者が任意に渡すため、確保の前に `i32` 収容まで検証する。
        let dims = crate::loss::validate_nll(OP, None, t, outer, classes, inner)
            .map_err(to_backend_error)?;
        let out = if n == 0 {
            vec![0.0f32; dims.numel]
        } else {
            match self.engine {
                Engine::Hip => cached_loss(self.target)
                    .and_then(|l| l.run_nll_backward_f32(t, outer, classes, inner, scale))
                    .map_err(to_backend_error)?,
                #[cfg(test)]
                Engine::HostModel => {
                    crate::loss_model::nll_backward(t, outer, classes, inner, scale)
                }
            }
        };
        Tensor::new(out, input_shape).map_err(BackendError::ShapeMismatch)
    }

    /// kl_div forward。
    fn kl_div_loss(
        &self,
        input: &Tensor<f32>,
        target: &Tensor<f32>,
        kind: KlDivTarget,
        reduction: MseReduction,
    ) -> Result<Tensor<f32>, BackendError> {
        const OP: &str = "kl_div_loss";
        require_same_shape(input.shape(), target.shape()).map_err(BackendError::ShapeMismatch)?;
        let (ic, tc) = (input.contiguous(), target.contiguous());
        let (x, t) = (host_slice(&ic, OP)?, host_slice(&tc, OP)?);
        let kind_i = kl_div_kind_i32(kind)?;
        let factor = reduction_factor(OP, reduction, x.len())?;
        crate::loss::validate_pair(OP, x.len(), t.len()).map_err(to_backend_error)?;
        if x.is_empty() {
            return scalar(0.0);
        }
        let v = match self.engine {
            Engine::Hip => cached_loss(self.target)
                .and_then(|l| l.run_kl_div_loss_f32(x, t, kind_i, factor))
                .map_err(to_backend_error)?,
            #[cfg(test)]
            Engine::HostModel => crate::loss_model::kl_div(
                crate::warp_geometry::WarpGeometry::for_target(self.target),
                x,
                t,
                kind_i,
                factor,
            ),
        };
        scalar(v)
    }

    /// kl_div backward（`dinput` のみ）。
    fn kl_div_loss_backward(
        &self,
        input: &Tensor<f32>,
        target: &Tensor<f32>,
        kind: KlDivTarget,
        scale: f32,
    ) -> Result<Tensor<f32>, BackendError> {
        const OP: &str = "kl_div_loss";
        require_same_shape(input.shape(), target.shape()).map_err(BackendError::ShapeMismatch)?;
        let (ic, tc) = (input.contiguous(), target.contiguous());
        let (x, t) = (host_slice(&ic, OP)?, host_slice(&tc, OP)?);
        let kind_i = kl_div_kind_i32(kind)?;
        crate::loss::validate_pair(OP, x.len(), t.len()).map_err(to_backend_error)?;
        let out = if x.is_empty() {
            Vec::new()
        } else {
            match self.engine {
                Engine::Hip => cached_loss(self.target)
                    .and_then(|l| l.run_kl_div_backward_f32(x, t, kind_i, scale))
                    .map_err(to_backend_error)?,
                #[cfg(test)]
                Engine::HostModel => crate::loss_model::kl_div_backward(x, t, kind_i, scale),
            }
        };
        Tensor::new(out, input.shape()).map_err(BackendError::ShapeMismatch)
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

    // ---- R5（#3436）: loss 10 メソッドの結線（HostModel 経路。CPU と照合） ----

    fn bits(x: &Tensor<f32>) -> Vec<u32> {
        x.contiguous()
            .as_slice()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect()
    }

    fn parity(label: &str, got: &Tensor<f32>, want: &Tensor<f32>) {
        assert_eq!(got.shape(), want.shape(), "{label}");
        assert_parity(
            label,
            got.contiguous().as_slice().unwrap(),
            want.contiguous().as_slice().unwrap(),
        );
    }

    fn unit(len: usize, seed: u32) -> Vec<f32> {
        data(len, seed)
            .iter()
            .map(|v| (v + 1.0) / 2.0 * 0.98 + 0.01)
            .collect()
    }

    #[test]
    fn loss_forward_and_backward_match_cpu() {
        let (o, c) = (ops(), CpuBackendOps);
        for (shape, n) in [
            (&[3usize, 5][..], 15usize),
            (&[300, 7][..], 2100),
            (&[1][..], 1),
        ] {
            let p = t(n, shape, 1);
            let tg = t(n, shape, 2);
            let probs = Tensor::new(unit(n, 3), shape).unwrap();
            let tprob = Tensor::new(unit(n, 4), shape).unwrap();
            for red in [MseReduction::Mean, MseReduction::Sum] {
                parity(
                    "mse",
                    &o.mse_loss(&p, &tg, red).unwrap(),
                    &c.mse_loss(&p, &tg, red).unwrap(),
                );
                for (k, d) in [(HuberKind::Huber, 0.7f32), (HuberKind::SmoothL1, 1.3)] {
                    parity(
                        "huber",
                        &o.huber_loss(&p, &tg, k, d, red).unwrap(),
                        &c.huber_loss(&p, &tg, k, d, red).unwrap(),
                    );
                }
                for k in [BceKind::Probabilities, BceKind::Logits] {
                    let x = if matches!(k, BceKind::Probabilities) {
                        &probs
                    } else {
                        &p
                    };
                    parity(
                        "bce",
                        &o.bce_loss(x, &tprob, k, red).unwrap(),
                        &c.bce_loss(x, &tprob, k, red).unwrap(),
                    );
                }
                for k in [KlDivTarget::Probabilities, KlDivTarget::LogProbabilities] {
                    parity(
                        "kl_div",
                        &o.kl_div_loss(&p, &tprob, k, red).unwrap(),
                        &c.kl_div_loss(&p, &tprob, k, red).unwrap(),
                    );
                }
            }
            // forward は shape []、backward は入力 shape。
            let f = o.mse_loss(&p, &tg, MseReduction::Mean).unwrap();
            assert_eq!(f.shape(), &[] as &[usize]);
            let b = o.mse_loss_backward(&p, &tg, 0.3).unwrap();
            assert_eq!(b.shape(), shape);
            assert_eq!(bits(&b), bits(&c.mse_loss_backward(&p, &tg, 0.3).unwrap()));
            for (k, d) in [(HuberKind::Huber, 0.7f32), (HuberKind::SmoothL1, 1.3)] {
                assert_eq!(
                    bits(&o.huber_loss_backward(&p, &tg, k, d, 0.5).unwrap()),
                    bits(&c.huber_loss_backward(&p, &tg, k, d, 0.5).unwrap())
                );
            }
            for k in [BceKind::Probabilities, BceKind::Logits] {
                let x = if matches!(k, BceKind::Probabilities) {
                    &probs
                } else {
                    &p
                };
                assert_eq!(
                    bits(&o.bce_loss_backward(x, &tprob, k, 0.5).unwrap()),
                    bits(&c.bce_loss_backward(x, &tprob, k, 0.5).unwrap())
                );
            }
            for k in [KlDivTarget::Probabilities, KlDivTarget::LogProbabilities] {
                assert_eq!(
                    bits(&o.kl_div_loss_backward(&p, &tprob, k, 0.5).unwrap()),
                    bits(&c.kl_div_loss_backward(&p, &tprob, k, 0.5).unwrap())
                );
            }
        }
    }

    #[test]
    fn nll_matches_cpu_for_each_class_dim() {
        let (o, c) = (ops(), CpuBackendOps);
        let shape = [4usize, 6, 3];
        let numel: usize = shape.iter().product();
        let input = t(numel, &shape, 9);
        for cd in 0..3 {
            let rest: Vec<usize> = shape
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != cd)
                .map(|(_, &d)| d)
                .collect();
            let n: usize = rest.iter().product();
            let tg: Vec<i32> = (0..n).map(|s| ((s * 5 + 1) % shape[cd]) as i32).collect();
            let tt = Tensor::new(tg, &rest).unwrap();
            for red in [MseReduction::Mean, MseReduction::Sum] {
                parity(
                    "nll",
                    &o.nll_loss(&input, &tt, cd, red).unwrap(),
                    &c.nll_loss(&input, &tt, cd, red).unwrap(),
                );
            }
            assert_eq!(
                bits(&o.nll_loss_backward(&shape, &tt, cd, 0.25).unwrap()),
                bits(&c.nll_loss_backward(&shape, &tt, cd, 0.25).unwrap())
            );
        }
    }

    #[test]
    fn loss_accepts_non_contiguous_inputs() {
        let (o, c) = (ops(), CpuBackendOps);
        let a = t(35, &[7, 5], 1).transpose_2d().unwrap();
        let b = t(35, &[7, 5], 2).transpose_2d().unwrap();
        parity(
            "mse nc",
            &o.mse_loss(&a, &b, MseReduction::Sum).unwrap(),
            &c.mse_loss(&a, &b, MseReduction::Sum).unwrap(),
        );
        assert_eq!(
            bits(&o.mse_loss_backward(&a, &b, 0.5).unwrap()),
            bits(&c.mse_loss_backward(&a, &b, 0.5).unwrap())
        );
    }

    #[test]
    fn loss_shape_and_target_errors_are_typed() {
        let o = ops();
        let a = t(6, &[2, 3], 1);
        let b = t(4, &[2, 2], 2);
        assert!(matches!(
            o.mse_loss(&a, &b, MseReduction::Mean),
            Err(BackendError::ShapeMismatch(_))
        ));
        assert!(matches!(
            o.kl_div_loss_backward(&a, &b, KlDivTarget::Probabilities, 1.0),
            Err(BackendError::ShapeMismatch(_))
        ));
        let tg = Tensor::new(vec![0i32, 1], &[2]).unwrap();
        assert!(matches!(
            o.nll_loss(&a, &tg, 2, MseReduction::Sum),
            Err(BackendError::ShapeMismatch(
                ShapeError::AxisOutOfRange { .. }
            ))
        ));
        // targets の長さ不一致。
        let short = Tensor::new(vec![0i32], &[1]).unwrap();
        assert!(matches!(
            o.nll_loss(&a, &short, 1, MseReduction::Sum),
            Err(BackendError::ShapeMismatch(
                ShapeError::ElementCountMismatch { .. }
            ))
        ));
        // 範囲外の添字は CPU と同じ InvalidArgument。
        let bad = Tensor::new(vec![0i32, 3], &[2]).unwrap();
        let neg = Tensor::new(vec![-1i32, 0], &[2]).unwrap();
        for tt in [&bad, &neg] {
            assert!(matches!(
                o.nll_loss(&a, tt, 1, MseReduction::Sum),
                Err(BackendError::InvalidArgument(_))
            ));
            assert!(matches!(
                o.nll_loss_backward(&[2, 3], tt, 1, 1.0),
                Err(BackendError::InvalidArgument(_))
            ));
            assert!(matches!(
                CpuBackendOps.nll_loss(&a, tt, 1, MseReduction::Sum),
                Err(BackendError::InvalidArgument(_))
            ));
        }
    }

    #[test]
    fn nll_backward_rejects_huge_shape_before_allocating() {
        let o = ops();
        let tg = Tensor::new(vec![0i32], &[1]).unwrap();
        for shape in [[usize::MAX, 1], [1usize << 40, 1 << 30]] {
            assert!(o.nll_loss_backward(&shape, &tg, 1, 1.0).is_err());
        }
    }

    #[test]
    fn empty_loss_inputs_do_not_touch_device() {
        // 非搭載環境の Hip 経路でも、空は HIP に触れず Ok を返す。
        let o = RocmBackendOps::new(GfxTarget::Gfx942);
        let e = t(0, &[0], 1);
        for red in [MseReduction::Mean, MseReduction::Sum] {
            assert_eq!(o.mse_loss(&e, &e, red).unwrap().as_slice().unwrap(), &[0.0]);
            assert_eq!(
                o.bce_loss(&e, &e, BceKind::Logits, red)
                    .unwrap()
                    .as_slice()
                    .unwrap(),
                &[0.0]
            );
        }
        assert_eq!(o.mse_loss_backward(&e, &e, 1.0).unwrap().shape(), &[0]);
        let tg = Tensor::new(Vec::<i32>::new(), &[0]).unwrap();
        let out = o.nll_loss_backward(&[0, 3], &tg, 1, 1.0).unwrap();
        assert_eq!(out.shape(), &[0, 3]);
    }

    #[test]
    fn loss_without_rocm_is_unavailable_after_validation() {
        if crate::is_available() {
            return;
        }
        let o = RocmBackendOps::new(GfxTarget::Gfx942);
        let a = t(6, &[2, 3], 1);
        let b = t(6, &[2, 3], 2);
        assert!(matches!(
            o.mse_loss(&a, &b, MseReduction::Mean),
            Err(BackendError::DeviceUnavailable(_))
        ));
        assert!(matches!(
            o.huber_loss_backward(&a, &b, HuberKind::Huber, 1.0, 1.0),
            Err(BackendError::DeviceUnavailable(_))
        ));
        let tg = Tensor::new(vec![0i32, 1], &[2]).unwrap();
        assert!(matches!(
            o.nll_loss(&a, &tg, 1, MseReduction::Mean),
            Err(BackendError::DeviceUnavailable(_))
        ));
        // 検証はデバイスより前: shape 不正・範囲外添字は DeviceUnavailable ではなく型付きエラー。
        let c = t(4, &[2, 2], 3);
        assert!(matches!(
            o.mse_loss(&a, &c, MseReduction::Mean),
            Err(BackendError::ShapeMismatch(_))
        ));
        let bad = Tensor::new(vec![0i32, 9], &[2]).unwrap();
        assert!(matches!(
            o.nll_loss(&a, &bad, 1, MseReduction::Mean),
            Err(BackendError::InvalidArgument(_))
        ));
    }
}
