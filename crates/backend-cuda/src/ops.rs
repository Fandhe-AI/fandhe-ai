//! CUDA バックエンドの `BackendOps` 実装（TASK-1.9c・#46。イシュー #599 で
//! elementwise 5 演算・`gemm_bias_act` 実融合化を追加）。
//!
//! `fandhe_ai_tensor_core::backend_ops::BackendOps` の CUDA 実装。GEMM は
//! `gemm::CudaGemm::run_tiled_f32` へ委譲する（既存カーネル・許容誤差・
//! 境界検査には触れない）。elementwise（`add`／`mul`／`relu`／`exp`／
//! `tanh`）は `elementwise::CudaElementwise` へ委譲する（イシュー #599）。
//! 汎用 reduction（`sum`／`max`）は未実装のまま
//! [`fandhe_ai_tensor_core::device::BackendError::Unsupported`] を返す（スコープ外。
//! out-of-scope-tracking.md 対象）。イシュー #592 で `run_fused` を
//! オーバーライドし、canonical RMSNorm 融合プラン（`x * rsqrt(sum(x^2))`）
//! 検出時のみ融合カーネル（[`crate::rmsnorm::CudaRmsNorm`]）へルーティング
//! する（`sum`／`max` 単独 API とは独立した経路）。
//!
//! `device.rs` の「動的ロード panic 回避ゲート」方針をそのまま踏襲する:
//! `CudaDevice::new` は driver 不在を `Err(CudaError::DriverUnavailable)`
//! で返す non-panicking な入口であり、本実装はこれを経由してから
//! `BackendError::CudaUnavailable` へ変換する（panic しない。
//! `.claude/rules/coding-rust.md`）。

use std::sync::Arc;

use fandhe_ai_tensor_core::device::{BackendError, Device};
use fandhe_ai_tensor_core::{Activation, BackendOps, DType, FusionPlan, ShapeError, Tensor};

use crate::context_cache;
use crate::device::CudaDevice;
use crate::elementwise::CudaElementwise;
use crate::error::CudaError;
use crate::rmsnorm::match_rmsnorm_plan;
use crate::softmax::match_softmax_plan;

/// CUDA バックエンドの `BackendOps` 実装。`ordinal` は `Device::Cuda(_)`
/// の一致判定に使う `cudarc` のデバイス番号
/// （`CudaContext::new(ordinal)` に対応。`fandhe_ai_tensor_core::device::Device`
/// の doc コメント参照）。
///
/// イシュー #929: `CudaDevice`／`CudaGemm`／`CudaElementwise`／
/// `CudaRmsNorm`／`CudaSoftmax` は各メソッド呼び出し時に都度構築せず、
/// `crate::context_cache`（`ordinal` キーのプロセス内キャッシュ）経由で
/// 取得する。同一プロセス内の 2 回目以降の呼び出しは `CudaContext` 生成・
/// NVRTC コンパイルを再実行しない（`context_cache` モジュール冒頭コメント
/// 参照。実測根拠: `scripts/bench/framework-compare/results/
/// summary.md:177`）。エラー（driver 不在等）はキャッシュされず毎回
/// 再試行される（fail-fast 契約は不変）ため、`Self::device_handle` の
/// 戻り値型が `Result<..., BackendError>` である点・エラー伝播の意味論
/// 自体は変更しない。
#[derive(Debug, Clone, Copy)]
pub struct CudaBackendOps {
    ordinal: usize,
}

impl CudaBackendOps {
    /// 指定した `ordinal` に対応する `CudaBackendOps` を構築する。
    /// 構築自体は driver 初期化を行わないため常に成功する（実際の
    /// driver 呼び出しは各メソッドが `Self::device_handle`（`context_cache`
    /// 経由）を呼んだ時点）。
    pub fn new(ordinal: usize) -> Self {
        Self { ordinal }
    }

    /// `context_cache::cached_device` を経由してデバイスハンドルを取得
    /// する（イシュー #929。プロセス内キャッシュのヒット時は
    /// `CudaContext::new` を再実行しない）。driver 不在・初期化失敗は
    /// `BackendError::CudaUnavailable` へ変換する（panic 回避ゲートは
    /// `CudaDevice::new` 内部で完結する。`device.rs` 参照）。
    fn device_handle(&self) -> Result<Arc<CudaDevice>, BackendError> {
        context_cache::cached_device(self.ordinal)
            .map_err(|e: CudaError| BackendError::CudaUnavailable(e.to_string()))
    }

    /// 二項 elementwise 共通のディスパッチ（`add`／`mul`）。
    ///
    /// `Tensor::broadcast_with`（NumPy 互換ブロードキャスト。CPU
    /// `elementwise::binary_elementwise` と同じ意味論）で共通 shape の
    /// view を得たのち `contiguous()` で密なバッファへ実体化してから
    /// `CudaElementwise`（同一長バッファのみを扱う。`elementwise.rs` 冒頭
    /// コメント「ブロードキャスト」参照）へ渡す。`run` は
    /// `CudaElementwise::run_add_f32`／`run_mul_f32` のいずれかを呼ぶ
    /// クロージャとして呼び出し側から注入される。
    fn elementwise_binary(
        &self,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
        run: impl FnOnce(&CudaElementwise, &[f32], &[f32]) -> Result<Vec<f32>, CudaError>,
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

        let device = self.device_handle()?;
        let ew = context_cache::cached_elementwise(self.ordinal, &device)
            .map_err(|e: CudaError| BackendError::CudaUnavailable(e.to_string()))?;
        let out = run(&ew, a_slice, b_slice)
            .map_err(|e: CudaError| BackendError::KernelLaunchFailed(e.to_string()))?;
        Tensor::new(out, &out_shape).map_err(BackendError::ShapeMismatch)
    }

    /// 単項 elementwise 共通のディスパッチ（`relu`／`exp`／`tanh`）。
    /// ブロードキャストが発生しない点を除き [`Self::elementwise_binary`]
    /// と同一構造。
    fn elementwise_unary(
        &self,
        a: &Tensor<f32>,
        run: impl FnOnce(&CudaElementwise, &[f32]) -> Result<Vec<f32>, CudaError>,
    ) -> Result<Tensor<f32>, BackendError> {
        let out_shape = a.shape().to_vec();
        let a_owned = a.contiguous();
        let a_slice = a_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("elementwise: input not contiguous".into())
        })?;

        let device = self.device_handle()?;
        let ew = context_cache::cached_elementwise(self.ordinal, &device)
            .map_err(|e: CudaError| BackendError::CudaUnavailable(e.to_string()))?;
        let out = run(&ew, a_slice)
            .map_err(|e: CudaError| BackendError::KernelLaunchFailed(e.to_string()))?;
        Tensor::new(out, &out_shape).map_err(BackendError::ShapeMismatch)
    }

    /// [`BackendOps::run_fused`] の RMSNorm 一致経路（イシュー #592）。
    /// `match_rmsnorm_plan` が一致した後の dtype／leaf 数／leaf shape の
    /// 起動前 fail-closed 検証と、`CudaRmsNorm::run_rmsnorm_f32_raw`
    /// （`inv_n = 1.0`・`eps = 0.0`・`w = None`）への委譲を行う。
    fn run_fused_rmsnorm(
        &self,
        plan: &FusionPlan,
        leaves: &[&Tensor<f32>],
        hidden: usize,
    ) -> Result<Tensor<f32>, BackendError> {
        // `match_rmsnorm_plan` は op 列・leaf 数・`row_fusion()` の形状
        // のみを照合し、`FusionPlan::from_ops` が受理しうる任意の
        // `dtype`（`FusionPlan` の DTO は現状 `DType` を素通しする。
        // `plan.rs` §2.1 参照）を検査しない。カーネル起動前に
        // `plan.dtype() == DType::F32` を明示検証しないと、例えば
        // `DType::F64` のプランでも f32 CUDA カーネルとして実行されて
        // しまう（`backend-cpu::fused_elementwise::run_fused_elementwise`
        // が実施する同種の fail-closed 検証との不整合。codex-review
        // 指摘・PR #706 レビュー）。
        if plan.dtype() != DType::F32 {
            return Err(BackendError::Unsupported(format!(
                "CudaBackendOps::run_fused: unsupported dtype {:?} (canonical RMSNorm fusion \
                 kernel supports F32 only)",
                plan.dtype()
            )));
        }
        let [x] = leaves else {
            return Err(BackendError::Unsupported(format!(
                "CudaBackendOps::run_fused: canonical RMSNorm プランは leaf 1 個を要求するが \
                 {} 個が渡された",
                leaves.len()
            )));
        };
        // leaf の shape が `plan.output_shape()` と一致することも明示
        // 検証する。`match_rmsnorm_plan` は要素数（`row_fusion().row_len()`）
        // のみを照合するため、要素数が一致しつつ shape（次元分割）が
        // 異なる leaf（例: `[8]` に対する `[2, 4]`）を渡しても
        // `run_rmsnorm_f32_raw` の長さ検証だけでは検出できない。canonical
        // プランは `axis: None`（全軸縮約）で `x` と出力の shape が恒等
        // （elementwise 型の最終 Mul）である契約のため、ここで shape 恒等
        // を fail-closed に強制する（`backend-cpu::fused_elementwise` の
        // leaf shape 検証と同じ契約）。
        if x.shape() != plan.output_shape() {
            return Err(BackendError::ShapeMismatch(ShapeError::ShapeMismatch {
                lhs: plan.output_shape().to_vec(),
                rhs: x.shape().to_vec(),
            }));
        }

        let x_owned = x.contiguous();
        let x_slice = x_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("run_fused: rmsnorm input not contiguous".into())
        })?;

        let device = self.device_handle()?;
        let rmsnorm = context_cache::cached_rmsnorm(self.ordinal, &device)
            .map_err(map_fused_kernel_init_error)?;
        let out = rmsnorm
            .run_rmsnorm_f32_raw(x_slice, None, 0.0, 1.0, 1, hidden)
            .map_err(|e: CudaError| BackendError::KernelLaunchFailed(e.to_string()))?;
        Tensor::new(out, plan.output_shape()).map_err(BackendError::ShapeMismatch)
    }

    /// [`BackendOps::run_fused`] の softmax 一致経路（イシュー #594）。
    /// `run_fused_rmsnorm` と同じ起動前 fail-closed 検証パターン（dtype
    /// F32 限定・leaf 1 個・leaf shape 恒等）を踏襲し、
    /// `CudaSoftmax::run_softmax_f32_raw` を `scale = log2(e)` で呼ぶ。
    fn run_fused_softmax(
        &self,
        plan: &FusionPlan,
        leaves: &[&Tensor<f32>],
        rows: usize,
        cols: usize,
    ) -> Result<Tensor<f32>, BackendError> {
        if plan.dtype() != DType::F32 {
            return Err(BackendError::Unsupported(format!(
                "CudaBackendOps::run_fused: unsupported dtype {:?} (canonical softmax fusion \
                 kernel supports F32 only)",
                plan.dtype()
            )));
        }
        let [x] = leaves else {
            return Err(BackendError::Unsupported(format!(
                "CudaBackendOps::run_fused: canonical softmax プランは leaf 1 個を要求するが \
                 {} 個が渡された",
                leaves.len()
            )));
        };
        // `run_fused_rmsnorm` と同じ理由（要素数一致だけでは shape の
        // 取り違えを検出できない）で leaf shape の恒等性を明示検証する。
        if x.shape() != plan.output_shape() {
            return Err(BackendError::ShapeMismatch(ShapeError::ShapeMismatch {
                lhs: plan.output_shape().to_vec(),
                rhs: x.shape().to_vec(),
            }));
        }

        let x_owned = x.contiguous();
        let x_slice = x_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("run_fused: softmax input not contiguous".into())
        })?;

        let device = self.device_handle()?;
        let softmax = context_cache::cached_softmax(self.ordinal, &device)
            .map_err(map_fused_kernel_init_error)?;
        let out = softmax
            .run_softmax_f32_raw(x_slice, std::f32::consts::LOG2_E, rows, cols)
            .map_err(|e: CudaError| BackendError::KernelLaunchFailed(e.to_string()))?;
        Tensor::new(out, plan.output_shape()).map_err(BackendError::ShapeMismatch)
    }
}

/// [`CudaBackendOps::gemm_bias_act`] が融合カーネル
/// （`gemm::CudaGemm::run_tiled_bias_act_f32`）と
/// `fandhe_ai_tensor_core::backend_ops::BackendOps::gemm_bias_act` のデフォルト実装
/// （非融合 `gemm`→`add`→`relu` 3 段合成）のどちらを経由するかを表す。
///
/// `backend-cpu::ops::CpuBackendOps::gemm_bias_act` の分岐条件
/// （`bias` が `None`、または `bias.shape()` が厳密に `[n]`
/// の場合にのみ融合カーネルへ進む）と同一の意味論を CUDA 側にも適用する
/// （バックエンド間で `gemm_bias_act` の経路依存の挙動差を作らない。
/// イシュー #203 Review 指摘と同じ理由）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GemmBiasActRoute {
    /// 融合カーネル（epilogue 内で bias 加算・activation を適用）へ進む。
    Fused,
    /// デフォルト実装（`gemm`→`add`→act の非融合合成）へフォールバックする。
    ComposedFallback,
}

/// [`crate::rmsnorm::CudaRmsNorm::new`]／[`crate::softmax::CudaSoftmax::new`] の初期化失敗を
/// `BackendError` へ変換する（純関数。実機なしで単体テスト可能）。
///
/// `CudaError::DriverUnavailable`／`NvrtcUnavailable` のみを環境不在
/// （`BackendError::CudaUnavailable`。CUDA/NVRTC 非搭載環境での早期
/// フォールバックを想定した variant）として扱う。それ以外
/// （NVRTC コンパイルエラー・関数ロード失敗・デバイス属性負値検出の
/// `InvalidKernelDescriptor` 等）を一律 `CudaUnavailable` に丸めると、
/// CUDA/NVRTC が利用可能な環境でもカーネル実装側の回帰が「環境不在」に
/// 化けて握りつぶされる（`tests/rmsnorm_parity.rs` の env-adaptive
/// スモークテストは `CudaUnavailable` を無条件に成功扱いするため。
/// codex-review 指摘・PR #706 レビュー）。よって環境不在の既知 variant
/// 以外は `BackendError::KernelLaunchFailed` として実装回帰を検出できる
/// ようにする（`memory.rs::map_cuda_error` と同じ variant 分岐方針。
/// `#[non_exhaustive]` の `CudaError` に対する将来 variant 追加への
/// フォールバックとして `KernelLaunchFailed` を wildcard の受け皿とする
/// 点も揃える）。
///
/// イシュー #594: 判定ロジックは RMSNorm 固有ではなく `CudaError` の
/// variant 分岐のみに依るため、`run_fused` の softmax ルーティング
/// （[`crate::softmax::CudaSoftmax::new`] の初期化失敗変換）でもそのまま共用する（実装
/// 計画 §3.4「初期化エラー変換は共通化」。旧名 `map_rmsnorm_init_error`
/// から RMSNorm 専用でない名前へ改名した）。
fn map_fused_kernel_init_error(err: CudaError) -> BackendError {
    match err {
        CudaError::DriverUnavailable { detail } => BackendError::CudaUnavailable(detail),
        CudaError::NvrtcUnavailable { detail } => BackendError::CudaUnavailable(detail),
        other => BackendError::KernelLaunchFailed(other.to_string()),
    }
}

/// [`GemmBiasActRoute`] の選択ロジック（純関数。実機なしで単体テスト可能。
/// 本ファイル末尾 `#[cfg(test)]` 参照）。
///
/// `bias_shape` は呼び出し元の `bias.map(|t| t.shape())`、`n` は
/// `B: [k, n]` の列数。`bias_shape` が `None`（bias 指定なし）または
/// 厳密に `[n]`（行方向複製）の場合にのみ [`GemmBiasActRoute::Fused`] を
/// 返す。`pub(crate)`: `CudaBackendOps::gemm_bias_act` から呼ばれる。
pub(crate) fn gemm_bias_act_route(bias_shape: Option<&[usize]>, n: usize) -> GemmBiasActRoute {
    match bias_shape {
        None => GemmBiasActRoute::Fused,
        Some(shape) if shape == [n] => GemmBiasActRoute::Fused,
        Some(_) => GemmBiasActRoute::ComposedFallback,
    }
}

impl BackendOps for CudaBackendOps {
    fn device(&self) -> Device {
        Device::Cuda(self.ordinal)
    }

    fn gemm(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        let out_shape = fandhe_ai_tensor_core::matmul_out_shape(a.shape(), b.shape())
            .map_err(BackendError::ShapeMismatch)?;
        let (m, k) = (a.shape()[0] as u32, a.shape()[1] as u32);
        let n = b.shape()[1] as u32;

        // `run_tiled_f32` は contiguous な `&[f32]` を要求する（CPU 実装
        // と同じ契約。`ops.rs`（backend-cpu）参照）。
        let a_owned = a.contiguous();
        let b_owned = b.contiguous();
        let a_slice = a_owned
            .as_slice()
            .ok_or_else(|| BackendError::KernelLaunchFailed("gemm: lhs not contiguous".into()))?;
        let b_slice = b_owned
            .as_slice()
            .ok_or_else(|| BackendError::KernelLaunchFailed("gemm: rhs not contiguous".into()))?;

        let device = self.device_handle()?;
        let gemm = context_cache::cached_gemm(self.ordinal, &device)
            .map_err(|e: CudaError| BackendError::CudaUnavailable(e.to_string()))?;
        let out = gemm
            .run_tiled_f32(a_slice, b_slice, m, n, k)
            .map_err(|e: CudaError| BackendError::KernelLaunchFailed(e.to_string()))?;
        Tensor::new(out, &out_shape).map_err(BackendError::ShapeMismatch)
    }

    /// [`fandhe_ai_tensor_core::BackendOps::gemm_bias_act`] のデフォルト実装（非融合
    /// `gemm` → `add` → `relu` 合成）を、GEMM epilogue に bias 加算・
    /// activation を融合したカーネル
    /// （[`crate::gemm::CudaGemm::run_tiled_bias_act_f32`]）へ差し替える
    /// （イシュー #599・TASK-12.1f）。`backend-cpu::ops::CpuBackendOps` の
    /// オーバーライドと同型の分岐（`gemm_bias_act_route` 参照）を採り、
    /// `bias` が `None` またはブロードキャストの厳密一致形状 `[n]`
    /// の場合にのみ融合カーネルを使う。それ以外（`[1]`・`[1, n]` 等の
    /// ブロードキャスト可能だが `[n]` ちょうどでない shape）はデフォルト
    /// 実装と同じ 3 段合成（`self.gemm` → `self.add` → `self.relu`）へ
    /// フォールバックする。両バックエンドは本イシュー時点で `add`／`relu`
    /// が実装済みのため CPU と異なり `Unsupported` を透過しない
    /// （モジュール冒頭コメント参照）。
    ///
    /// フォールバック時も CPU 実装と同じ順序契約（GEMM 本体を実行する前に
    /// `fandhe_ai_tensor_core::broadcast_shape` でブロードキャスト可否のみ先に検証。
    /// REQ-8・OWASP A03）を保つ。
    fn gemm_bias_act(
        &self,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
        bias: Option<&Tensor<f32>>,
        act: Activation,
    ) -> Result<Tensor<f32>, BackendError> {
        let out_shape = fandhe_ai_tensor_core::matmul_out_shape(a.shape(), b.shape())
            .map_err(BackendError::ShapeMismatch)?;
        let (m, k) = (a.shape()[0] as u32, a.shape()[1] as u32);
        let n = b.shape()[1] as u32;

        let bias_shape = bias.map(|t| t.shape());
        match gemm_bias_act_route(bias_shape, n as usize) {
            GemmBiasActRoute::ComposedFallback => {
                if let Some(bias) = bias {
                    // GEMM 本体を実行する前にブロードキャスト可否を検証
                    // する（CPU 実装 `CpuBackendOps::gemm_bias_act` と同じ
                    // 「カーネル本体アクセス前に検証」の順序契約）。
                    fandhe_ai_tensor_core::broadcast_shape(&out_shape, bias.shape())
                        .map_err(BackendError::ShapeMismatch)?;
                }
                let mut out = self.gemm(a, b)?;
                if let Some(bias) = bias {
                    out = self.add(&out, bias)?;
                }
                out = match act {
                    Activation::None => out,
                    Activation::Relu => self.relu(&out)?,
                    // `Activation` は `#[non_exhaustive]`。CPU 実装と同じ
                    // 方針で未知 variant を黙って恒等関数として扱わず
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

                let device = self.device_handle()?;
                let gemm = context_cache::cached_gemm(self.ordinal, &device)
                    .map_err(|e: CudaError| BackendError::CudaUnavailable(e.to_string()))?;
                let out = gemm
                    .run_tiled_bias_act_f32(a_slice, b_slice, bias_slice, act_relu, m, n, k)
                    .map_err(|e: CudaError| BackendError::KernelLaunchFailed(e.to_string()))?;
                Tensor::new(out, &out_shape).map_err(BackendError::ShapeMismatch)
            }
        }
    }

    fn add(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.elementwise_binary(a, b, |ew, a_s, b_s| ew.run_add_f32(a_s, b_s))
    }

    fn mul(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.elementwise_binary(a, b, |ew, a_s, b_s| ew.run_mul_f32(a_s, b_s))
    }

    fn relu(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.elementwise_unary(a, |ew, a_s| ew.run_relu_f32(a_s))
    }

    fn exp(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.elementwise_unary(a, |ew, a_s| ew.run_exp_f32(a_s))
    }

    fn tanh(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.elementwise_unary(a, |ew, a_s| ew.run_tanh_f32(a_s))
    }

    /// [`fandhe_ai_tensor_core::BackendOps::run_fused`] のデフォルト実装
    /// （`Unsupported` fail-safe）を、canonical RMSNorm 融合プラン
    /// （`x * rsqrt(sum(x^2))`。mean 化・eps・weight を含まない厳密形状）
    /// 検出時に [`crate::rmsnorm::CudaRmsNorm`]（イシュー #592）へ、
    /// canonical softmax 融合プラン（`exp(x - max(x)) / sum(exp(x -
    /// max(x)))`。最終軸または全軸縮約の厳密形状）検出時に
    /// [`crate::softmax::CudaSoftmax`]（イシュー #594）へルーティング
    /// する。
    ///
    /// プラン一致判定は `match_rmsnorm_plan`／`match_softmax_plan`
    /// （いずれも純関数。プランの op 列・leaf 数・`row_fusion()` の形状を
    /// 厳密照合する）に委ねる。RMSNorm 判定を先に試し、一致しなければ
    /// softmax 判定を試す（op 列長〈6 vs 8〉が異なるため両方に一致する
    /// プランは存在しない）。どちらにも一致しないプラン
    /// （elementwise-only・中間軸 softmax 等）は本オーバーライドの対象外
    /// としてデフォルト実装（`Unsupported`）へ委ね、呼び出し元
    /// （`fandhe_ai_autodiff::Tape` の実体化経路）の per-op フォールバックへ倒す
    /// （`backend-cpu::fused_elementwise::run_fused_elementwise` の
    /// allowlist 拒否方針と同じ fail-closed。`.claude/rules/security.md`
    /// A08「判定の迂回経路を作らない」）。
    ///
    /// RMSNorm 一致時: プランの意味論 `x * rsqrt(sum(x^2))` に厳密一致
    /// させるため `crate::rmsnorm::CudaRmsNorm::run_rmsnorm_f32_raw`
    /// （`inv_n` を明示できる内部エントリ）を `inv_n = 1.0`・`eps = 0.0`・
    /// `w = None`（`has_weight = 0`）で直接呼ぶ（`mean` 化・`eps` 加算・
    /// `weight` 乗算を勝手に補わない。標準 RMSNorm 用の公開 API
    /// [`crate::rmsnorm::CudaRmsNorm::run_rmsnorm_f32`] は `inv_n =
    /// 1/hidden` を内部導出してしまうため canonical プランには使えない。
    /// `rmsnorm.rs` ドキュメンテーションコメント参照）。`rows = 1` は
    /// canonical プランが `axis: None`（全軸縮約）のみを受理する
    /// （`match_rmsnorm_plan` 参照）ため、行方向融合ではなく単一行として
    /// 扱う。
    ///
    /// softmax 一致時: プランの意味論 `exp(x - max(x)) / sum(...)` に
    /// 厳密一致させるため `crate::softmax::CudaSoftmax::run_softmax_f32_raw`
    /// を `scale = log2(e)` で直接呼ぶ（プランの `Exp` は自然指数だが
    /// カーネルは `exp2(x*log2(e))` を計算する恒等式を用いる。数値的な
    /// 一致判定は per-op 経路と丸めが異なるため REQ-2 複合判定に依る。
    /// `softmax.rs` モジュール冒頭コメント「意味論注記」参照）。
    fn run_fused(
        &self,
        plan: &FusionPlan,
        leaves: &[&Tensor<f32>],
    ) -> Result<Tensor<f32>, BackendError> {
        if let Some(hidden) = match_rmsnorm_plan(plan) {
            return self.run_fused_rmsnorm(plan, leaves, hidden);
        }
        if let Some((rows, cols)) = match_softmax_plan(plan) {
            return self.run_fused_softmax(plan, leaves, rows, cols);
        }
        Err(BackendError::Unsupported(
            "CudaBackendOps::run_fused: プランが canonical RMSNorm 形状（x * \
             rsqrt(sum(x^2))）・canonical softmax 形状（exp(x-max(x))/sum(...)）の \
             いずれにも一致しないため融合カーネルへルーティングできない \
             （#592／#594 スコープ。呼び出し元の per-op フォールバックに委ねる）"
                .into(),
        ))
    }

    /// 汎用 reduction カーネルは未実装のまま（#599 スコープ外・イシュー
    /// #592 でも対象外）。イシュー #592 は融合 RMSNorm カーネル
    /// （[`Self::run_fused`] 経由のみ）に閉じた縮約を実装したが、
    /// `BackendOps::sum`（任意軸・非融合の単独縮約 API）自体の GPU
    /// カーネル化は別イシューのスコープ（out-of-scope-tracking.md 対象）。
    fn sum(&self, _a: &Tensor<f32>, _dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "CudaBackendOps::sum: reduction カーネル未実装（#599 スコープ外）".into(),
        ))
    }

    /// [`Self::sum`] と同じ理由（汎用 reduction 未実装）。
    fn max(&self, _a: &Tensor<f32>, _dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "CudaBackendOps::max: reduction カーネル未実装（#599 スコープ外）".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// [`map_fused_kernel_init_error`]: `DriverUnavailable`／
    /// `NvrtcUnavailable` は環境不在として `BackendError::CudaUnavailable`
    /// へ変換される（env-adaptive スモークテストの早期 return 判定と
    /// 揃う）。RMSNorm／softmax 両経路で共用する（イシュー #594）。
    #[test]
    fn map_fused_kernel_init_error_treats_known_unavailable_variants_as_cuda_unavailable() {
        assert!(matches!(
            map_fused_kernel_init_error(CudaError::DriverUnavailable {
                detail: "no libcuda".into()
            }),
            BackendError::CudaUnavailable(msg) if msg.contains("no libcuda")
        ));
        assert!(matches!(
            map_fused_kernel_init_error(CudaError::NvrtcUnavailable {
                detail: "no libnvrtc".into()
            }),
            BackendError::CudaUnavailable(msg) if msg.contains("no libnvrtc")
        ));
    }

    /// [`map_fused_kernel_init_error`]: 環境不在以外の失敗（NVRTC
    /// コンパイルエラー・デバイス属性負値検出等）は
    /// `BackendError::KernelLaunchFailed` として実装回帰を検出できる状態を
    /// 保つ（`CudaUnavailable` に丸めて env-adaptive テストの早期 return
    /// に握りつぶされるのを防ぐ。codex-review 指摘・PR #706 レビュー）。
    #[test]
    fn map_fused_kernel_init_error_treats_other_variants_as_kernel_launch_failed() {
        let err = map_fused_kernel_init_error(CudaError::InvalidKernelDescriptor {
            detail: "negative SM count".into(),
        });
        assert!(matches!(
            err,
            BackendError::KernelLaunchFailed(msg) if msg.contains("negative SM count")
        ));
    }

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
        // `[1]` は `[n]` へブロードキャスト可能だが厳密一致ではないため
        // フォールバック（CPU 実装と同じ分岐条件）。
        assert_eq!(
            gemm_bias_act_route(Some(&[1]), 8),
            GemmBiasActRoute::ComposedFallback
        );
        // `[1, n]` も同様（2 次元形状は `[n]` と厳密一致しない）。
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

    /// 環境適応（CUDA 非搭載環境でも実行可能。実機なら本体まで検証）:
    /// `gemm_bias_act`（`bias.shape() == [n]`）が実際に融合カーネル
    /// （`gemm::CudaGemm::run_tiled_bias_act_f32`）へ到達し、
    /// `fandhe_ai_tensor_core::backend_ops::BackendOps::gemm_bias_act` のデフォルト
    /// 実装（非融合 3 段合成）を経由していないことを、
    /// [`crate::gemm::BIAS_ACT_FUSED_LAUNCH_COUNT`] の増加で検証する
    /// （実装計画 3.3 節「フォールバックを経由しないことのテスト機構」）。
    /// CUDA 非搭載環境では `BackendError::CudaUnavailable` を確認して
    /// 早期 return する（`tests/backend_ops_real_device.rs` と同じ
    /// 分岐パターン）。
    ///
    /// カウンタはスレッドローカル（`gemm.rs::BIAS_ACT_FUSED_LAUNCH_COUNT`
    /// のドキュメンテーションコメント参照。codex-review 指摘・PR #688）
    /// のため、`cargo test` の既定並列実行下で他スレッドの別テストが
    /// 同じ融合カーネルを起動しても `before`/`after` の差分には混入しない
    /// （直列化・プロセス全体 Mutex は不要）。
    #[test]
    fn gemm_bias_act_fused_path_increments_launch_counter_env_adaptive() {
        use fandhe_ai_tensor_core::Tensor;

        let cuda = CudaBackendOps::new(0);
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("valid tensor");
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).expect("valid tensor");
        let bias = Tensor::new(vec![1.0, 1.0], &[2]).expect("valid tensor");

        let before = crate::gemm::BIAS_ACT_FUSED_LAUNCH_COUNT.with(|c| c.get());
        match cuda.gemm_bias_act(&a, &b, Some(&bias), Activation::Relu) {
            Ok(_) => {
                let after = crate::gemm::BIAS_ACT_FUSED_LAUNCH_COUNT.with(|c| c.get());
                assert!(
                    after > before,
                    "融合カーネルの起動カウンタが増加していない（デフォルト非融合合成へ \
                     フォールバックした疑い）: before={before}, after={after}"
                );
            }
            Err(BackendError::CudaUnavailable(msg)) => {
                assert!(!msg.is_empty(), "error detail message must not be empty");
            }
            Err(other) => panic!("unexpected error variant for gemm_bias_act: {other}"),
        }
    }

    /// `run_fused` の canonical RMSNorm プラン検出（`rmsnorm.rs::
    /// match_rmsnorm_plan`）の型と同型の 6 op 列を組み立てる（`hidden`
    /// のみ差し替え）。`rmsnorm.rs::tests::build_canonical_rmsnorm_plan`
    /// と同じ op 列（`plan.rs::
    /// from_segment_builds_rmsnorm_plan_with_row_fusion_metadata` 参照）。
    fn build_canonical_rmsnorm_plan(
        hidden: usize,
        dtype: fandhe_ai_tensor_core::DType,
    ) -> FusionPlan {
        let ops = vec![
            fandhe_ai_tensor_core::FusedOpKind::Input { leaf_index: 0 },
            fandhe_ai_tensor_core::FusedOpKind::Mul { lhs: 0, rhs: 0 },
            fandhe_ai_tensor_core::FusedOpKind::Sum {
                input: 1,
                axis: None,
            },
            fandhe_ai_tensor_core::FusedOpKind::Rsqrt { input: 2 },
            fandhe_ai_tensor_core::FusedOpKind::Broadcast {
                input: 3,
                axis: None,
            },
            fandhe_ai_tensor_core::FusedOpKind::Mul { lhs: 4, rhs: 0 },
        ];
        FusionPlan::from_ops(ops, vec![hidden], dtype, 1).unwrap()
    }

    /// `run_fused` はカーネル起動（デバイスアクセス）前に `plan.dtype()
    /// == DType::F32` を検証するため、非 F32 プランは
    /// `BackendError::Unsupported` を返す（CUDA 非搭載環境でも決定的に
    /// 実行可能。`match_rmsnorm_plan` が一致した後の検証であることを
    /// 確認するため canonical op 列をそのまま使う。codex-review 指摘・
    /// PR #706 レビュー「融合プランの dtype と leaf shape を起動前に
    /// 検証する」）。
    #[test]
    fn run_fused_rejects_non_f32_dtype_before_device_access() {
        let plan = build_canonical_rmsnorm_plan(8, fandhe_ai_tensor_core::DType::F16);
        let x = Tensor::new(vec![1.0f32; 8], &[8]).expect("valid tensor");
        let cuda = CudaBackendOps::new(0);

        let err = cuda.run_fused(&plan, &[&x]).unwrap_err();
        assert!(
            matches!(err, BackendError::Unsupported(_)),
            "expected Unsupported for non-F32 dtype, got {err:?}"
        );
    }

    /// `run_fused` は leaf の shape が `plan.output_shape()` と厳密一致
    /// することも起動前に検証する。要素数が `row_len` と一致するだけの
    /// 異なる shape（`[8]` に対する `[2, 4]`）は
    /// `BackendError::ShapeMismatch` で拒否する（codex-review 指摘・
    /// PR #706 レビュー同上）。
    #[test]
    fn run_fused_rejects_leaf_shape_mismatch_before_device_access() {
        let plan = build_canonical_rmsnorm_plan(8, fandhe_ai_tensor_core::DType::F32);
        // 要素数（8）は `row_len` と一致するが shape が異なる。
        let x = Tensor::new(vec![1.0f32; 8], &[2, 4]).expect("valid tensor");
        let cuda = CudaBackendOps::new(0);

        let err = cuda.run_fused(&plan, &[&x]).unwrap_err();
        assert!(
            matches!(err, BackendError::ShapeMismatch(_)),
            "expected ShapeMismatch for leaf shape != output_shape, got {err:?}"
        );
    }
}
