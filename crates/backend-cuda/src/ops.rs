//! CUDA バックエンドの `BackendOps` 実装（TASK-1.9c・#46。イシュー #599 で
//! elementwise 5 演算・`gemm_bias_act` 実融合化を追加）。
//!
//! `tensor_core::backend_ops::BackendOps` の CUDA 実装。GEMM は
//! `gemm::CudaGemm::run_tiled_f32` へ委譲する（既存カーネル・許容誤差・
//! 境界検査には触れない）。elementwise（`add`／`mul`／`relu`／`exp`／
//! `tanh`）は `elementwise::CudaElementwise` へ委譲する（イシュー #599）。
//! reduction（`sum`／`max`）は本イシュー時点でも未実装のため
//! [`tensor_core::device::BackendError::Unsupported`] を返す（スコープ外。
//! out-of-scope-tracking.md 対象）。
//!
//! `device.rs` の「動的ロード panic 回避ゲート」方針をそのまま踏襲する:
//! `CudaDevice::new` は driver 不在を `Err(CudaError::DriverUnavailable)`
//! で返す non-panicking な入口であり、本実装はこれを経由してから
//! `BackendError::CudaUnavailable` へ変換する（panic しない。
//! `.claude/rules/coding-rust.md`）。

use tensor_core::device::{BackendError, Device};
use tensor_core::{Activation, BackendOps, Tensor};

use crate::device::CudaDevice;
use crate::elementwise::CudaElementwise;
use crate::error::CudaError;
use crate::gemm::CudaGemm;

/// CUDA バックエンドの `BackendOps` 実装。`ordinal` は `Device::Cuda(_)`
/// の一致判定に使う `cudarc` のデバイス番号
/// （`CudaContext::new(ordinal)` に対応。`tensor_core::device::Device`
/// の doc コメント参照）。
///
/// `CudaDevice`／`CudaGemm`／`CudaElementwise` は各メソッド呼び出し時に
/// 都度構築する（TASK-1.9b の `DeviceBuffer`／デバイスハンドル常駐が
/// 未着地のため。モジュール冒頭 `backend_ops` の突合コメント参照）。
/// ハンドル常駐化・再利用による初期化コスト削減は TASK-1.9b／1.9d 以降の
/// 最適化対象（出典なき性能主張として本イシューでは行わない）。
#[derive(Debug, Clone, Copy)]
pub struct CudaBackendOps {
    ordinal: usize,
}

impl CudaBackendOps {
    /// 指定した `ordinal` に対応する `CudaBackendOps` を構築する。
    /// 構築自体は driver 初期化を行わないため常に成功する（実際の
    /// driver 呼び出しは各メソッドが `CudaDevice::new` を経由した時点）。
    pub fn new(ordinal: usize) -> Self {
        Self { ordinal }
    }

    /// `CudaDevice::new` を経由してデバイスハンドルを取得する。
    /// driver 不在・初期化失敗は `BackendError::CudaUnavailable` へ
    /// 変換する（panic 回避ゲートは `CudaDevice::new` 内部で完結する。
    /// `device.rs` 参照）。
    fn device_handle(&self) -> Result<CudaDevice, BackendError> {
        CudaDevice::new(self.ordinal)
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
        let ew = CudaElementwise::new(&device)
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
        let ew = CudaElementwise::new(&device)
            .map_err(|e: CudaError| BackendError::CudaUnavailable(e.to_string()))?;
        let out = run(&ew, a_slice)
            .map_err(|e: CudaError| BackendError::KernelLaunchFailed(e.to_string()))?;
        Tensor::new(out, &out_shape).map_err(BackendError::ShapeMismatch)
    }
}

/// [`CudaBackendOps::gemm_bias_act`] が融合カーネル
/// （`gemm::CudaGemm::run_tiled_bias_act_f32`）と
/// `tensor_core::backend_ops::BackendOps::gemm_bias_act` のデフォルト実装
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
        let out_shape = tensor_core::matmul_out_shape(a.shape(), b.shape())
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
        let gemm = CudaGemm::new(&device)
            .map_err(|e: CudaError| BackendError::CudaUnavailable(e.to_string()))?;
        let out = gemm
            .run_tiled_f32(a_slice, b_slice, m, n, k)
            .map_err(|e: CudaError| BackendError::KernelLaunchFailed(e.to_string()))?;
        Tensor::new(out, &out_shape).map_err(BackendError::ShapeMismatch)
    }

    /// [`tensor_core::BackendOps::gemm_bias_act`] のデフォルト実装（非融合
    /// `gemm` → `add` → `relu` 合成）を、GEMM epilogue に bias 加算・
    /// activation を融合したカーネル
    /// （[`crate::gemm::CudaGemm::run_tiled_bias_act_f32`]）へ差し替える
    /// （イシュー #599・TASK-12.1f）。`backend-cpu::ops::CpuBackendOps` の
    /// オーバーライドと同型の分岐（[`gemm_bias_act_route`] 参照）を採り、
    /// `bias` が `None` またはブロードキャストの厳密一致形状 `[n]`
    /// の場合にのみ融合カーネルを使う。それ以外（`[1]`・`[1, n]` 等の
    /// ブロードキャスト可能だが `[n]` ちょうどでない shape）はデフォルト
    /// 実装と同じ 3 段合成（`self.gemm` → `self.add` → `self.relu`）へ
    /// フォールバックする。両バックエンドは本イシュー時点で `add`／`relu`
    /// が実装済みのため CPU と異なり `Unsupported` を透過しない
    /// （モジュール冒頭コメント参照）。
    ///
    /// フォールバック時も CPU 実装と同じ順序契約（GEMM 本体を実行する前に
    /// `tensor_core::broadcast_shape` でブロードキャスト可否のみ先に検証。
    /// REQ-8・OWASP A03）を保つ。
    fn gemm_bias_act(
        &self,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
        bias: Option<&Tensor<f32>>,
        act: Activation,
    ) -> Result<Tensor<f32>, BackendError> {
        let out_shape = tensor_core::matmul_out_shape(a.shape(), b.shape())
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
                    tensor_core::broadcast_shape(&out_shape, bias.shape())
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
                let gemm = CudaGemm::new(&device)
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

    fn sum(&self, _a: &Tensor<f32>, _dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "CudaBackendOps::sum: reduction カーネル未実装（#599 スコープ外）".into(),
        ))
    }

    fn max(&self, _a: &Tensor<f32>, _dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "CudaBackendOps::max: reduction カーネル未実装（#599 スコープ外）".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    /// `tensor_core::backend_ops::BackendOps::gemm_bias_act` のデフォルト
    /// 実装（非融合 3 段合成）を経由していないことを、
    /// [`crate::gemm::BIAS_ACT_FUSED_LAUNCH_COUNT`] の増加で検証する
    /// （実装計画 3.3 節「フォールバックを経由しないことのテスト機構」）。
    /// CUDA 非搭載環境では `BackendError::CudaUnavailable` を確認して
    /// 早期 return する（`tests/backend_ops_real_device.rs` と同じ
    /// 分岐パターン）。
    #[test]
    fn gemm_bias_act_fused_path_increments_launch_counter_env_adaptive() {
        use std::sync::atomic::Ordering;

        use tensor_core::Tensor;

        let cuda = CudaBackendOps::new(0);
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("valid tensor");
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).expect("valid tensor");
        let bias = Tensor::new(vec![1.0, 1.0], &[2]).expect("valid tensor");

        let before = crate::gemm::BIAS_ACT_FUSED_LAUNCH_COUNT.load(Ordering::Relaxed);
        match cuda.gemm_bias_act(&a, &b, Some(&bias), Activation::Relu) {
            Ok(_) => {
                let after = crate::gemm::BIAS_ACT_FUSED_LAUNCH_COUNT.load(Ordering::Relaxed);
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
}
