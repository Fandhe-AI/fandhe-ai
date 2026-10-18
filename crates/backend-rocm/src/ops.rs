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
//! - 正規化 6 メソッド（`rmsnorm`・`layer_norm`・`rmsnorm_backward`・`layer_norm_backward`・
//!   `batch_norm_train`・`batch_norm_infer`。R6・#3437）は [`RocmNorm`] へ結線する。形状導出
//!   （`row_norm_layout`／`batch_norm_layout`）・contiguous 化・`Tensor` 化は本ファイル、カーネル起動と
//!   起動前検証は `norm.rs` の責務。ROCm 非搭載では `DeviceUnavailable` を返す（`Unsupported` では
//!   ないため autodiff のホスト参照実装へは黙ってフォールバックしない。A08）。
//! - R5（#3436）で loss 10 メソッド（`mse_loss`・`huber_loss`・`bce_loss`・`nll_loss`・`kl_div_loss`
//!   と各 backward）を [`RocmLoss`](crate::RocmLoss) へ結線した。shape 検査・`factor` 決定・enum
//!   から `i32` への写像（未知 variant は `Unsupported`）・空入力の早期リターンはデバイス取得より
//!   前に行うので、ROCm 非搭載でも型付きエラーを検査できる。
//! - R10（#3441）で LSTM／GRU の `lstm_pointwise`・`lstm_hidden_backward`・`lstm_cell_backward`・
//!   `gru_pointwise`・`gru_backward` を [`RocmRnnCell`](crate::RocmRnnCell) へ結線した。結線の本体
//!   （rank・形状検査、空入力の CPU 委譲、起動前検証）は子モジュール `rnn_cell`（`ops/rnn_cell.rs`）に
//!   置き、本ファイルのメソッドは 1 行で委譲する。ROCm 非搭載は `DeviceUnavailable`（`Unsupported` ではない）。
//! - R7（#3438）で `pad`・`im2col`・`col2im`・`max_pool2d`・`avg_pool2d`・`adaptive_avg_pool2d` を
//!   HIPRTC カーネル（`RocmConstantPad`・`RocmIm2col`・`RocmPooling`）へ結線した。結線の本体
//!   （shape 再検査・空出力の早期リターン・HIP に触れる前の `Unsupported` 確定・engine 分岐）は子モジュール
//!   `conv_pool`（`ops/conv_pool.rs`）に置き、本ファイルのメソッドは 1 行で委譲する。avg の dilation ≠ 1 と
//!   サイズ上限超過は `Unsupported`（ホスト参照実装へのフォールバック合図）。
//! - R12（#3443）でデバイス常駐 KV の `paged_kv_append`・`paged_attention_forward` を [`RocmPagedKv`](crate::RocmPagedKv)
//!   へ結線した。結線の本体（ハンドル downcast・tensor-core 共通検証・空入力・engine 分岐）は子モジュール
//!   `paged_kv`（`ops/paged_kv.rs`）に置き、本ファイルのメソッドは 1 行で委譲する。被覆計画の「shfl を使う 1 か所」は
//!   現行 CUDA ソースと一致せず（カーネル本体に shuffle なし）、`WarpGeometry` は使わない。`f64` アキュムレータ契約は
//!   該当しない（f32 の逐次和）。`_tracked` 版は trait 既定の委譲のまま。ROCm 非搭載は `DeviceUnavailable`。
//! - どの候補リスト（`ops_for`・facade の `resolve_ops`／`available_devices`）にも登録しない。

use std::cell::RefCell;
use std::rc::Rc;
use std::thread::LocalKey;

use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_tensor_core::{
    BackendError, BackendOps, BatchNormTrainOutput, BceKind, BinaryElementwiseOp, Conv2dParams,
    Device, DeviceBuffer, Element, FusionPlan, GruBackwardOutput, GruPointwiseOutput, HuberKind,
    KlDivTarget, LayerNormBackwardOutput, LstmPointwiseOutput, MseReduction, PagedAttentionArgs,
    PagedKvLayout, Pool2dParams, ScalarBinaryOp, ScalarUnaryOp, ShapeError, Tensor,
    UnaryElementwiseOp, batch_norm_layout, gemm_out_shape, reduce_out_shape, require_same_shape,
    row_norm_layout, row_softmax_layout,
};

use crate::arg_reduce::{ArgReduceKind, RocmArgReduce};
use crate::elementwise::{
    BinaryKind, RocmElementwise, RocmFusedElementwise, RocmScalarOp, UnaryKind,
};
use crate::error::{RocmError, to_backend_error};
use crate::gemm::RocmGemmF32;
use crate::loss::{cached_loss, first_bad_target};
use crate::norm::{
    BatchNormTrainHost, LayerNormBackwardHost, OptLens, RmsNormBackwardHost, RocmNorm,
    batch_norm_train_empty, layer_norm_backward_empty, rmsnorm_backward_empty, validate_batch_norm,
    validate_row_norm,
};
use crate::reduce::{ReduceKind, RocmReduce, reduce_axis_layout};
use crate::target::GfxTarget;
#[cfg(test)]
use crate::warp_geometry::WarpGeometry;

/// 実行経路。`HostModel` はテスト専用（実機なしで組み立て経路を検証する。GEMM は `gemm_model`、
/// 縮約は `reduce_model`、正規化は `norm_model`、loss 10 メソッドは `loss_model` の逐語モデルを使う）。
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
    /// 正規化（R6・#3437）。
    static NORM_CACHE: Cache<RocmNorm> = const { RefCell::new(Vec::new()) };
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

/// ROCm（AMD GPU）向け `BackendOps`。必須 9 メソッド・R2 の elementwise 一式・縮約 5 メソッド（R3）・正規化 6 メソッド（R6）を実装する。
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

mod conv_pool;
mod ew;
mod paged_kv;
mod rnn_cell;

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

    /// 新トークンの K／V 行をデバイス常駐のページ化 KV プールへスロット写像どおりに書く（R12・#3443）。
    /// 同期実行。結線は `ops/paged_kv.rs`。`_tracked` 版は trait 既定の委譲。到達性は #3157 待ち。
    fn paged_kv_append(
        &self,
        k_new: &DeviceBuffer<f32>,
        v_new: &DeviceBuffer<f32>,
        k_pool: &mut DeviceBuffer<f32>,
        v_pool: &mut DeviceBuffer<f32>,
        slot_mapping: &[u32],
        layout: &PagedKvLayout,
    ) -> Result<(), BackendError> {
        self.paged_kv_append_impl(k_new, v_new, k_pool, v_pool, slot_mapping, layout)
    }

    /// ページ化 KV プールに対する attention（R12・#3443）。出力 `[B, L_new, E]`。結線は `ops/paged_kv.rs`。
    fn paged_attention_forward(
        &self,
        q: &DeviceBuffer<f32>,
        k_pool: &DeviceBuffer<f32>,
        v_pool: &DeviceBuffer<f32>,
        block_tables: &[u32],
        seq_lens: &[u32],
        args: &PagedAttentionArgs,
    ) -> Result<DeviceBuffer<f32>, BackendError> {
        self.paged_attention_impl(q, k_pool, v_pool, block_tables, seq_lens, args)
    }

    /// LSTM の pointwise 段（R10・#3441）。結線は `ops/rnn_cell.rs`。
    fn lstm_pointwise(
        &self,
        pre: &Tensor<f32>,
        c_prev: &Tensor<f32>,
    ) -> Result<LstmPointwiseOutput, BackendError> {
        self.rnn_lstm_pointwise(pre, c_prev)
    }

    /// `Op::LstmHidden` の VJP 補助（R10・#3441）。
    fn lstm_hidden_backward(
        &self,
        c: &Tensor<f32>,
        gate_o: &Tensor<f32>,
        dh: &Tensor<f32>,
    ) -> Result<(Tensor<f32>, Tensor<f32>), BackendError> {
        self.rnn_lstm_hidden_backward(c, gate_o, dh)
    }

    /// `Op::LstmCell` の VJP 補助（R10・#3441）。
    fn lstm_cell_backward(
        &self,
        gates_ifg: &Tensor<f32>,
        c_prev: &Tensor<f32>,
        dc: &Tensor<f32>,
    ) -> Result<(Tensor<f32>, Tensor<f32>), BackendError> {
        self.rnn_lstm_cell_backward(gates_ifg, c_prev, dc)
    }

    /// GRU の pointwise 段（R10・#3441）。
    fn gru_pointwise(
        &self,
        pre_i: &Tensor<f32>,
        pre_h: &Tensor<f32>,
        h_prev: &Tensor<f32>,
    ) -> Result<GruPointwiseOutput, BackendError> {
        self.rnn_gru_pointwise(pre_i, pre_h, h_prev)
    }

    /// `Op::GruCell` の VJP 補助（R10・#3441）。
    fn gru_backward(
        &self,
        gates_rzn: &Tensor<f32>,
        q: &Tensor<f32>,
        h_prev: &Tensor<f32>,
        dh: &Tensor<f32>,
    ) -> Result<GruBackwardOutput, BackendError> {
        self.rnn_gru_backward(gates_rzn, q, h_prev, dh)
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

    /// HIPRTC の定数 pad（R7・#3438）。
    fn pad(
        &self,
        input: &Tensor<f32>,
        pads: &[(usize, usize)],
        value: f32,
    ) -> Result<Tensor<f32>, BackendError> {
        self.conv_pool_pad(input, pads, value)
    }

    /// HIPRTC の Conv2d im2col（R7・#3438）。
    fn im2col(
        &self,
        input: &Tensor<f32>,
        params: &Conv2dParams,
    ) -> Result<Tensor<f32>, BackendError> {
        self.conv_pool_im2col(input, params)
    }

    /// HIPRTC の Conv2d col2im（im2col の随伴。R7・#3438）。
    fn col2im(
        &self,
        d_col: &Tensor<f32>,
        input_shape: &[usize],
        params: &Conv2dParams,
    ) -> Result<Tensor<f32>, BackendError> {
        self.conv_pool_col2im(d_col, input_shape, params)
    }

    /// HIPRTC の MaxPool2d（値と平面内索引。R7・#3438）。
    fn max_pool2d(
        &self,
        input: &Tensor<f32>,
        params: &Pool2dParams,
    ) -> Result<(Tensor<f32>, Tensor<i32>), BackendError> {
        self.conv_pool_max_pool2d(input, params)
    }

    /// HIPRTC の AvgPool2d（dilation ≠ 1 は `Unsupported`。R7・#3438）。
    fn avg_pool2d(
        &self,
        input: &Tensor<f32>,
        params: &Pool2dParams,
        count_include_pad: bool,
    ) -> Result<Tensor<f32>, BackendError> {
        self.conv_pool_avg_pool2d(input, params, count_include_pad)
    }

    /// HIPRTC の AdaptiveAvgPool2d（R7・#3438）。
    fn adaptive_avg_pool2d(
        &self,
        input: &Tensor<f32>,
        output_size: [usize; 2],
    ) -> Result<Tensor<f32>, BackendError> {
        self.conv_pool_adaptive_avg_pool2d(input, output_size)
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
