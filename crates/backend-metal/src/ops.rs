//! Metal バックエンドの `BackendOps` 実装（TASK-1.9c・#46）。
//!
//! `tensor_core::backend_ops::BackendOps` の Metal 実装。GEMM は
//! `gemm::MetalGemm::dispatch_auto`（動的タイル選択済み。TASK-1.8c・#40）
//! へ委譲する（既存カーネル・許容誤差・境界検査には触れない）。Metal は
//! 本イシュー時点で GEMM カーネルのみ実装済みのため、elementwise・
//! reduction は [`tensor_core::device::BackendError::Unsupported`] を
//! 返す（GPU 側カーネルの実装自体は本イシューのスコープ外。
//! out-of-scope-tracking.md 対象）。
//!
//! `cfg(target_os = "macos")` 限定（`objc2`／`objc2-foundation`／
//! `objc2-metal` と同じ cfg 境界。`.claude/rules/deps-policy.md`）。
//! 非 macOS 環境ではこのファイル自体がコンパイル対象に入らない
//! （`lib.rs` の cfg 境界と整合。`device.rs` と同方針）。

use tensor_core::device::{BackendError, Device};
use tensor_core::{BackendOps, FusionPlan, ShapeError, Tensor};

use crate::context::MetalContext;
use crate::error::MetalError;
use crate::gemm::MetalGemm;
use crate::memory::map_metal_error;
use crate::rmsnorm::MetalRmsNorm;
use crate::row_kernel::{self, plan_dtype_is_f32};
use crate::softmax::MetalSoftmax;

/// Metal バックエンドの `BackendOps` 実装。`Device::Metal` は ordinal を
/// 持たない単一 variant のため（`docs/public-api-design.md` §4.1・
/// `device.rs::MetalDeviceProvider` と同じ位置付け）、本実装は複数 GPU の
/// 個別選択をサポートしない（システムデフォルトの Metal デバイスに
/// 対応する）。
///
/// `MetalContext`／`MetalGemm` は各メソッド呼び出し時に都度構築する
/// （`backend-cuda::ops::CudaBackendOps` と同じ設計判断。TASK-1.9b の
/// デバイスハンドル常駐が未着地のため。ハンドル常駐化は TASK-1.9b／1.9d
/// 以降の最適化対象）。
#[derive(Debug, Default, Clone, Copy)]
pub struct MetalBackendOps;

impl MetalBackendOps {
    /// 新規 `MetalBackendOps` を構築する。構築自体はデバイス初期化を
    /// 行わないため常に成功する（実際の初期化は各メソッドが
    /// `MetalContext::new` を経由した時点）。
    pub fn new() -> Self {
        Self
    }
}

impl BackendOps for MetalBackendOps {
    fn device(&self) -> Device {
        Device::Metal
    }

    fn gemm(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        let out_shape = tensor_core::matmul_out_shape(a.shape(), b.shape())
            .map_err(BackendError::ShapeMismatch)?;
        let (m, k) = (a.shape()[0], a.shape()[1]);
        let n = b.shape()[1];

        // `dispatch_auto` は contiguous な `&[f32]` を要求する（CPU／CUDA
        // 実装と同じ契約）。
        let a_owned = a.contiguous();
        let b_owned = b.contiguous();
        let a_slice = a_owned
            .as_slice()
            .ok_or_else(|| BackendError::KernelLaunchFailed("gemm: lhs not contiguous".into()))?;
        let b_slice = b_owned
            .as_slice()
            .ok_or_else(|| BackendError::KernelLaunchFailed("gemm: rhs not contiguous".into()))?;

        // `MetalContext::new` の失敗（デバイス不在等）は
        // `MetalDeviceProvider::select`（`device.rs`）と同一分類の
        // `BackendError::DeviceUnavailable` に統一する（`map_metal_error`
        // 経由。誤って `DeviceAllocationFailed`〈VRAM／アロケータ起因〉
        // に分類すると、呼び出し側が Metal デバイス不在を検知する経路が
        // 一方に偏る。Bugbot 指摘対応。PR #262 レビュースレッド）。
        let ctx = MetalContext::new().map_err(map_metal_error)?;
        let gemm = MetalGemm::new(&ctx)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        let out = gemm
            .dispatch_auto(&ctx, a_slice, b_slice, m, n, k)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        Tensor::new(out, &out_shape).map_err(BackendError::ShapeMismatch)
    }

    fn add(&self, _a: &Tensor<f32>, _b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "MetalBackendOps::add: elementwise カーネル未実装（TASK-1.9c スコープ外）".into(),
        ))
    }

    fn mul(&self, _a: &Tensor<f32>, _b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "MetalBackendOps::mul: elementwise カーネル未実装（TASK-1.9c スコープ外）".into(),
        ))
    }

    fn relu(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "MetalBackendOps::relu: elementwise カーネル未実装（TASK-1.9c スコープ外）".into(),
        ))
    }

    fn exp(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "MetalBackendOps::exp: elementwise カーネル未実装（TASK-1.9c スコープ外）".into(),
        ))
    }

    fn tanh(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "MetalBackendOps::tanh: elementwise カーネル未実装（TASK-1.9c スコープ外）".into(),
        ))
    }

    fn sum(&self, _a: &Tensor<f32>, _dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "MetalBackendOps::sum: reduction カーネル未実装（TASK-1.9c スコープ外）".into(),
        ))
    }

    fn max(&self, _a: &Tensor<f32>, _dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "MetalBackendOps::max: reduction カーネル未実装（TASK-1.9c スコープ外）".into(),
        ))
    }

    /// [`tensor_core::BackendOps::run_fused`] のデフォルト実装
    /// （`Unsupported` fail-safe）を、canonical RMSNorm／softmax 融合プラン
    /// 検出時のみ融合カーネル（[`MetalRmsNorm`]／[`MetalSoftmax`]）へ
    /// ルーティングする（イシュー #604）。CUDA 側 `CudaBackendOps::run_fused`
    /// （#592）の fail-closed 検証列をそのまま踏襲する:
    ///
    /// 1. [`row_kernel::match_rmsnorm_plan`]（6 op 列・leaf 1 個・
    ///    `axis: None` のみ受理）→ 一致時 [`MetalRmsNorm`] へ
    /// 2. [`row_kernel::match_softmax_plan`]（8 op 列・leaf 1 個・`axis` が
    ///    最終次元または `None` のみ受理）→ 一致時 [`MetalSoftmax`] へ
    /// 3. どちらにも一致しないプランは `Unsupported` を返し per-op
    ///    フォールバックへ委ねる（allowlist 拒否・迂回経路を作らない。
    ///    `.claude/rules/security.md` A08）
    /// 4. 一致後も `plan.dtype() == DType::F32`・leaf 個数 = 1・
    ///    `leaf.shape() == plan.output_shape()` を明示検証する
    ///    （CUDA 側 codex-review 是正 PR #706 と同等の起動前検証）
    ///
    /// softmax の run_fused 配線は CUDA（G-7・#594）に先行するが、プラン
    /// 形状は `tensor-core` の融合 IR（#588）のテストで固定済みのため
    /// 乖離リスクは低い（`row_kernel.rs` モジュール冒頭コメント・`softmax.rs`
    /// ドキュメンテーションコメント「CUDA との parity 状況」参照）。
    fn run_fused(
        &self,
        plan: &FusionPlan,
        leaves: &[&Tensor<f32>],
    ) -> Result<Tensor<f32>, BackendError> {
        if let Some(hidden) = row_kernel::match_rmsnorm_plan(plan) {
            return self.run_fused_rmsnorm(plan, leaves, hidden);
        }
        if let Some(hidden) = row_kernel::match_softmax_plan(plan) {
            return self.run_fused_softmax(plan, leaves, hidden);
        }
        Err(BackendError::Unsupported(
            "MetalBackendOps::run_fused: プランが canonical RMSNorm（x * rsqrt(sum(x^2))）／\
             softmax（exp(x-max(x))/sum）のいずれの形状にも一致しないため融合カーネルへ\
             ルーティングできない（#604 スコープ。呼び出し元の per-op フォールバックに委ねる）"
                .into(),
        ))
    }
}

impl MetalBackendOps {
    /// 一致した leaf・shape・dtype を検証してから
    /// [`MetalRmsNorm::run_rmsnorm_f32_raw`] を `inv_n = 1.0`・`eps = 0.0`・
    /// `w = None`（`has_weight = 0`）で直接呼ぶ（プランの意味論 `x *
    /// rsqrt(sum(x^2))` に厳密一致させる。`mean` 化・`eps` 加算・`weight`
    /// 乗算を勝手に補わない。`backend-cuda::ops::CudaBackendOps::run_fused`
    /// と同じ検証順序: dtype → leaf 数 → leaf shape の順にカーネル起動前
    /// （デバイスアクセス前）で fail-closed に検証する）。
    fn run_fused_rmsnorm(
        &self,
        plan: &FusionPlan,
        leaves: &[&Tensor<f32>],
        hidden: usize,
    ) -> Result<Tensor<f32>, BackendError> {
        let x = validate_fused_leaf(plan, leaves)?;

        let x_owned = x.contiguous();
        let x_slice = x_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("run_fused: rmsnorm input not contiguous".into())
        })?;

        let ctx = MetalContext::new().map_err(map_metal_error)?;
        let rmsnorm = MetalRmsNorm::new(&ctx)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        let out = rmsnorm
            .run_rmsnorm_f32_raw(&ctx, x_slice, None, 0.0, 1.0, 1, hidden)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        Tensor::new(out, plan.output_shape()).map_err(BackendError::ShapeMismatch)
    }

    /// [`Self::run_fused_rmsnorm`] と同じ検証順序で
    /// [`MetalSoftmax::run_softmax_f32`] を呼ぶ。
    ///
    /// `row_kernel::match_softmax_plan` は `axis: None`（全軸縮約。
    /// `rows = 1` 相当）だけでなく `axis` が最終次元（行方向縮約。
    /// `rows > 1` になりうる）も受理するため、[`Self::run_fused_rmsnorm`]
    /// と異なり `rows` を固定値にできない。`x_slice.len() / hidden`
    /// （`validate_fused_leaf` が `x.shape() == plan.output_shape()` を
    /// 検証済みのため `x_slice.len()` は `plan.output_shape()` の要素数積
    /// と一致する）から導出する。`hidden == 0` は
    /// `MetalSoftmax::run_softmax_f32` 側の 0 要素早期 return 契約に委ね、
    /// ここではゼロ除算を避けるためのみ分岐する。
    fn run_fused_softmax(
        &self,
        plan: &FusionPlan,
        leaves: &[&Tensor<f32>],
        hidden: usize,
    ) -> Result<Tensor<f32>, BackendError> {
        let x = validate_fused_leaf(plan, leaves)?;

        let x_owned = x.contiguous();
        let x_slice = x_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("run_fused: softmax input not contiguous".into())
        })?;
        let rows = if hidden == 0 {
            0
        } else {
            x_slice.len() / hidden
        };

        let ctx = MetalContext::new().map_err(map_metal_error)?;
        let softmax = MetalSoftmax::new(&ctx)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        let out = softmax
            .run_softmax_f32(&ctx, x_slice, rows, hidden)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        Tensor::new(out, plan.output_shape()).map_err(BackendError::ShapeMismatch)
    }
}

/// [`MetalBackendOps::run_fused_rmsnorm`]／[`run_fused_softmax`] 共通の
/// カーネル起動前検証（デバイスアクセス前・fail-closed）。
///
/// `match_rmsnorm_plan`／`match_softmax_plan` は op 列・leaf 数・
/// `row_fusion()` の形状のみを照合し、`FusionPlan::from_ops` が受理しうる
/// 任意の `dtype` を検査しない。カーネル起動前に `plan.dtype() ==
/// DType::F32` を明示検証しないと、例えば `DType::F64` のプランでも f32
/// Metal カーネルとして実行されてしまう（CUDA 側 codex-review 指摘・
/// PR #706 レビューと同種の懸念）。leaf の shape が `plan.output_shape()`
/// と一致することも検証する: canonical プランは leaf 1 個・恒等 shape
/// （`axis: None`〈全軸縮約〉または行方向縮約後に broadcast で復元）の
/// 契約のため、要素数が一致しつつ shape（次元分割）が異なる leaf を
/// 拒否する（`backend-cpu::fused_elementwise` の leaf shape 検証と同じ
/// 契約）。
///
/// 検証済みの唯一の leaf（`&Tensor<f32>`）を戻り値として返す（呼び出し元
/// が `leaves` を再度パターンマッチする必要をなくし、`unreachable!` を
/// 使わずに済ませる。coding-rust.md「本番経路で unwrap/expect を使わない」
/// と同じ理由で、到達しないはずの分岐を panic で表現しない）。
fn validate_fused_leaf<'a>(
    plan: &FusionPlan,
    leaves: &[&'a Tensor<f32>],
) -> Result<&'a Tensor<f32>, BackendError> {
    if !plan_dtype_is_f32(plan) {
        return Err(BackendError::Unsupported(format!(
            "MetalBackendOps::run_fused: unsupported dtype {:?} (canonical fusion kernels \
             support F32 only)",
            plan.dtype()
        )));
    }
    let [x] = leaves else {
        return Err(BackendError::Unsupported(format!(
            "MetalBackendOps::run_fused: canonical プランは leaf 1 個を要求するが {} 個が渡された",
            leaves.len()
        )));
    };
    if x.shape() != plan.output_shape() {
        return Err(BackendError::ShapeMismatch(ShapeError::ShapeMismatch {
            lhs: plan.output_shape().to_vec(),
            rhs: x.shape().to_vec(),
        }));
    }
    Ok(*x)
}
