//! Metal バックエンドの `BackendOps` 実装（TASK-1.9c・#46。イシュー #605 で
//! elementwise 5 演算・`gemm_bias_act` 実融合化を追加）。
//!
//! `fandhe_ai_tensor_core::backend_ops::BackendOps` の Metal 実装。GEMM は
//! `gemm::MetalGemm::dispatch_auto`（動的タイル選択済み。TASK-1.8c・#40）
//! へ委譲する（既存カーネル・許容誤差・境界検査には触れない）。elementwise
//! （`add`／`mul`／`relu`／`exp`／`tanh`）は `elementwise::MetalElementwise`
//! へ委譲する（イシュー #605。CUDA 側 #599 の Metal 対応版）。汎用
//! reduction（`sum`／`max`）は未実装のまま
//! [`fandhe_ai_tensor_core::device::BackendError::Unsupported`] を返す（スコープ外。
//! out-of-scope-tracking.md 対象）。
//!
//! `cfg(target_os = "macos")` 限定（`objc2`／`objc2-foundation`／
//! `objc2-metal` と同じ cfg 境界。`.claude/rules/deps-policy.md`）。
//! 非 macOS 環境ではこのファイル自体がコンパイル対象に入らない
//! （`lib.rs` の cfg 境界と整合。`device.rs` と同方針）。

use fandhe_ai_tensor_core::device::{BackendError, Device};
use fandhe_ai_tensor_core::{Activation, BackendOps, FusionPlan, ShapeError, Tensor};

use crate::context::MetalContext;
use crate::elementwise::MetalElementwise;
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

    /// 二項 elementwise 共通のディスパッチ（`add`／`mul`。イシュー #605）。
    ///
    /// `Tensor::broadcast_with`（NumPy 互換ブロードキャスト）で共通 shape
    /// の view を得たのち `contiguous()` で密なバッファへ実体化してから
    /// `MetalElementwise`（同一長バッファのみを扱う。`elementwise.rs`
    /// 冒頭コメント「ブロードキャスト」参照）へ渡す。`run` は
    /// `MetalElementwise::run_add_f32`／`run_mul_f32` のいずれかを呼ぶ
    /// クロージャとして呼び出し側から注入される（`backend-cuda::ops::
    /// CudaBackendOps::elementwise_binary` と同型の構成）。
    fn elementwise_binary(
        &self,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
        run: impl FnOnce(
            &MetalElementwise,
            &MetalContext,
            &[f32],
            &[f32],
        ) -> Result<Vec<f32>, MetalError>,
    ) -> Result<Tensor<f32>, BackendError> {
        let (a_bc, b_bc) = a.broadcast_with(b).map_err(BackendError::ShapeMismatch)?;
        let out_shape = a_bc.shape().to_vec();

        let a_owned = a_bc.contiguous();
        let b_owned = b_bc.contiguous();
        let a_slice = a_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("elementwise: lhs not contiguous".into())
        })?;
        let b_slice = b_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("elementwise: rhs not contiguous".into())
        })?;

        let ctx = MetalContext::new().map_err(map_metal_error)?;
        let ew = MetalElementwise::new(&ctx)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        let out = run(&ew, &ctx, a_slice, b_slice)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        Tensor::new(out, &out_shape).map_err(BackendError::ShapeMismatch)
    }

    /// 単項 elementwise 共通のディスパッチ（`relu`／`exp`／`tanh`。
    /// イシュー #605）。ブロードキャストが発生しない点を除き
    /// [`Self::elementwise_binary`] と同一構造。
    fn elementwise_unary(
        &self,
        a: &Tensor<f32>,
        run: impl FnOnce(&MetalElementwise, &MetalContext, &[f32]) -> Result<Vec<f32>, MetalError>,
    ) -> Result<Tensor<f32>, BackendError> {
        let out_shape = a.shape().to_vec();
        let a_owned = a.contiguous();
        let a_slice = a_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("elementwise: input not contiguous".into())
        })?;

        let ctx = MetalContext::new().map_err(map_metal_error)?;
        let ew = MetalElementwise::new(&ctx)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        let out = run(&ew, &ctx, a_slice)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        Tensor::new(out, &out_shape).map_err(BackendError::ShapeMismatch)
    }
}

/// [`MetalBackendOps::gemm_bias_act`] が融合カーネル
/// （`gemm::MetalGemm::run_tiled_bias_act_f32`）と
/// `fandhe_ai_tensor_core::backend_ops::BackendOps::gemm_bias_act` のデフォルト実装
/// （非融合 `gemm`→`add`→`relu` 3 段合成）のどちらを経由するかを表す
/// （イシュー #605。CUDA 側 `fandhe_ai_backend_cuda::ops::GemmBiasActRoute`〈#599〉と
/// 同一の意味論）。
///
/// `backend-cpu::ops::CpuBackendOps::gemm_bias_act`・`backend-cuda::ops::
/// CudaBackendOps::gemm_bias_act` の分岐条件（`bias` が `None`、または
/// `bias.shape()` が厳密に `[n]` の場合にのみ融合カーネルへ進む）と同一の
/// 意味論を Metal 側にも適用する（バックエンド間で `gemm_bias_act` の
/// 経路依存の挙動差を作らない。イシュー #203 Review 指摘と同じ理由）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GemmBiasActRoute {
    /// 融合カーネル（epilogue 内で bias 加算・activation を適用）へ進む。
    Fused,
    /// デフォルト実装（`gemm`→`add`→act の非融合合成）へフォールバックする。
    ComposedFallback,
}

/// [`GemmBiasActRoute`] の選択ロジック（純関数。実機なしで単体テスト可能。
/// 本ファイル末尾 `#[cfg(test)]` 参照）。CUDA 側
/// `fandhe_ai_backend_cuda::ops::gemm_bias_act_route` と同一実装（`pub(crate)`:
/// `MetalBackendOps::gemm_bias_act` から呼ばれる）。
pub(crate) fn gemm_bias_act_route(bias_shape: Option<&[usize]>, n: usize) -> GemmBiasActRoute {
    match bias_shape {
        None => GemmBiasActRoute::Fused,
        Some(shape) if shape == [n] => GemmBiasActRoute::Fused,
        Some(_) => GemmBiasActRoute::ComposedFallback,
    }
}

impl BackendOps for MetalBackendOps {
    fn device(&self) -> Device {
        Device::Metal
    }

    fn gemm(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        let out_shape = fandhe_ai_tensor_core::matmul_out_shape(a.shape(), b.shape())
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

    /// [`fandhe_ai_tensor_core::BackendOps::gemm_bias_act`] のデフォルト実装（非融合
    /// `gemm` → `add` → `relu` 合成）を、GEMM epilogue に bias 加算・
    /// activation を融合したカーネル
    /// （[`crate::gemm::MetalGemm::run_tiled_bias_act_f32`]）へ差し替える
    /// （イシュー #605）。CPU／CUDA 実装と同型の分岐（[`gemm_bias_act_route`]
    /// 参照）を採り、`bias` が `None` またはブロードキャストの厳密一致
    /// 形状 `[n]` の場合にのみ融合カーネルを使う。それ以外（`[1]`・
    /// `[1, n]` 等）はデフォルト実装と同じ 3 段合成（`self.gemm` →
    /// `self.add` → `self.relu`）へフォールバックする（本イシューで
    /// `add`／`relu` を実装済みのため CPU／CUDA と異なり `Unsupported` を
    /// 透過しない。モジュール冒頭コメント参照）。
    fn gemm_bias_act(
        &self,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
        bias: Option<&Tensor<f32>>,
        act: Activation,
    ) -> Result<Tensor<f32>, BackendError> {
        let out_shape = fandhe_ai_tensor_core::matmul_out_shape(a.shape(), b.shape())
            .map_err(BackendError::ShapeMismatch)?;
        let (m, k) = (a.shape()[0], a.shape()[1]);
        let n = b.shape()[1];

        let bias_shape = bias.map(|t| t.shape());
        match gemm_bias_act_route(bias_shape, n) {
            GemmBiasActRoute::ComposedFallback => {
                if let Some(bias) = bias {
                    // GEMM 本体を実行する前にブロードキャスト可否を検証
                    // する（CPU／CUDA 実装と同じ「カーネル本体アクセス前に
                    // 検証」の順序契約）。
                    fandhe_ai_tensor_core::broadcast_shape(&out_shape, bias.shape())
                        .map_err(BackendError::ShapeMismatch)?;
                }
                // `self.gemm`（`MetalGemm::dispatch_auto` 経由）は
                // `m`/`n`/`k == 0` を `ZeroDimension` として拒否する
                // （`gemm.rs::validate_dims`）が、CPU／CUDA の `gemm`
                // （`gemm_blis_parallel`・CUDA 側実装）はゼロ次元を合法な
                // 形状として受理しゼロ初期化バッファをそのまま返す。この
                // 非対称のため、ブロードキャスト bias（`[1]`・`[1, n]` 等。
                // `Fused` 経路に乗らない形状）かつゼロ次元の場合に
                // `self.gemm` を呼ぶと Metal のみ `ZeroDimension` で失敗し
                // `gemm_bias_act` の CPU／CUDA と共有される契約が破れる
                // （Cursor Bugbot 指摘。PR #717 レビュースレッド）。
                // `Fused` 経路（本関数末尾）は既にゼロ次元をホスト側
                // epilogue で受理しているため、ここでも `self.gemm` を
                // 経由せず CPU／CUDA と同じゼロ初期化 `m * n` 結果を直接
                // 構築することで両経路の zero-dim 挙動を揃える。
                let mut out = if m == 0 || n == 0 || k == 0 {
                    Tensor::new(vec![0.0f32; m * n], &out_shape)
                        .map_err(BackendError::ShapeMismatch)?
                } else {
                    self.gemm(a, b)?
                };
                if let Some(bias) = bias {
                    out = self.add(&out, bias)?;
                }
                out = match act {
                    Activation::None => out,
                    Activation::Relu => self.relu(&out)?,
                    // `Activation` は `#[non_exhaustive]`。CPU／CUDA 実装と
                    // 同じ方針で未知 variant を黙って恒等関数として扱わず
                    // 明示的に拒否する。
                    _ => {
                        return Err(BackendError::Unsupported(format!(
                            "gemm_bias_act: unsupported activation {act:?} in non-fused fallback path"
                        )));
                    }
                };
                Ok(out)
            }
            GemmBiasActRoute::Fused => {
                let act_relu = match act {
                    Activation::None => false,
                    Activation::Relu => true,
                    _ => {
                        return Err(BackendError::Unsupported(format!(
                            "gemm_bias_act: unsupported activation {act:?} in fused epilogue path"
                        )));
                    }
                };

                let a_owned = a.contiguous();
                let b_owned = b.contiguous();
                let a_slice = a_owned.as_slice().ok_or_else(|| {
                    BackendError::KernelLaunchFailed("gemm_bias_act: lhs not contiguous".into())
                })?;
                let b_slice = b_owned.as_slice().ok_or_else(|| {
                    BackendError::KernelLaunchFailed("gemm_bias_act: rhs not contiguous".into())
                })?;

                let bias_owned;
                let bias_slice = match bias {
                    Some(bias) => {
                        bias_owned = bias.contiguous();
                        Some(bias_owned.as_slice().ok_or_else(|| {
                            BackendError::KernelLaunchFailed(
                                "gemm_bias_act: bias not contiguous".into(),
                            )
                        })?)
                    }
                    None => None,
                };

                let ctx = MetalContext::new().map_err(map_metal_error)?;
                let gemm = MetalGemm::new(&ctx)
                    .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
                let out = gemm
                    .run_tiled_bias_act_f32(&ctx, a_slice, b_slice, bias_slice, act_relu, m, n, k)
                    .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
                Tensor::new(out, &out_shape).map_err(BackendError::ShapeMismatch)
            }
        }
    }

    fn add(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.elementwise_binary(a, b, |ew, ctx, a_s, b_s| ew.run_add_f32(ctx, a_s, b_s))
    }

    fn mul(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.elementwise_binary(a, b, |ew, ctx, a_s, b_s| ew.run_mul_f32(ctx, a_s, b_s))
    }

    fn relu(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.elementwise_unary(a, |ew, ctx, a_s| ew.run_relu_f32(ctx, a_s))
    }

    fn exp(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.elementwise_unary(a, |ew, ctx, a_s| ew.run_exp_f32(ctx, a_s))
    }

    fn tanh(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.elementwise_unary(a, |ew, ctx, a_s| ew.run_tanh_f32(ctx, a_s))
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

    /// [`fandhe_ai_tensor_core::BackendOps::run_fused`] のデフォルト実装
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
    /// ここでは `checked_div`（`hidden == 0` なら `rows = 0`）でゼロ除算
    /// のみを避ける。
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
        // `checked_div` を使い `hidden == 0` を除算前に排除する（clippy
        // `manual_checked_ops`。挙動は従来の if 分岐と同一: `hidden == 0`
        // の場合は `rows = 0`、それ以外は通常の整数除算）。
        let rows = x_slice.len().checked_div(hidden).unwrap_or(0);

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

#[cfg(test)]
mod tests {
    use super::*;

    // --- gemm_bias_act_route（pure・実機不要。イシュー #605） ---
    // CUDA 側 `fandhe_ai_backend_cuda::ops` の同名テスト群と同一の検証項目。

    #[test]
    fn gemm_bias_act_route_selects_fused_when_bias_is_none() {
        assert_eq!(gemm_bias_act_route(None, 8), GemmBiasActRoute::Fused);
    }

    #[test]
    fn gemm_bias_act_route_selects_fused_when_bias_shape_matches_n_exactly() {
        assert_eq!(gemm_bias_act_route(Some(&[8]), 8), GemmBiasActRoute::Fused);
    }

    #[test]
    fn gemm_bias_act_route_falls_back_when_bias_shape_is_broadcastable_but_not_n() {
        assert_eq!(
            gemm_bias_act_route(Some(&[1]), 8),
            GemmBiasActRoute::ComposedFallback
        );
        assert_eq!(
            gemm_bias_act_route(Some(&[1, 8]), 8),
            GemmBiasActRoute::ComposedFallback
        );
    }

    #[test]
    fn gemm_bias_act_route_falls_back_when_bias_len_mismatches_n() {
        assert_eq!(
            gemm_bias_act_route(Some(&[4]), 8),
            GemmBiasActRoute::ComposedFallback
        );
    }
}
