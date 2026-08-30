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

use fandhe_ai_tensor_core::buffer::{DeviceBufferView, MemoryOps};
use fandhe_ai_tensor_core::device::{BackendError, Device};
use fandhe_ai_tensor_core::{
    Activation, BackendOps, DispatchFailureCell, FusionPlan, MseReduction, ShapeError, Tensor,
    require_same_shape,
};

use crate::context::MetalContext;
use crate::context_cache;
use crate::elementwise::MetalElementwise;
use crate::error::MetalError;
use crate::layout::{self, MatrixLayout};
use crate::memory::{MetalBufferHandle, MetalMemory, map_metal_error};
use crate::row_kernel::{self, plan_dtype_is_f32};

std::thread_local! {
    /// [`MetalBackendOps::gemm_resident_lhs`]／[`MetalBackendOps::
    /// gemm_resident_rhs`] が「転置 view の zero-repack 経路」に乗れず
    /// `Tensor::contiguous()`（ホスト側転置コピー）へフォールバックした
    /// 回数（イシュー #1040。`gemm::BIAS_ACT_FUSED_LAUNCH_COUNT` と同型の
    /// 可観測点）。`crate::layout::classify_2d` が `None` を返す入力
    /// （stride 0 のブロードキャスト等の非対応形状）のみがこのフォール
    /// バックへ到達する。`pub(crate)`（`gemm::BIAS_ACT_FUSED_LAUNCH_COUNT`
    /// と同じ可視性方針。クレート境界外の `tests/gemm_resident_parity.rs`
    /// からは参照できないため、「フォールバック非経由」の確認は本ファイル
    /// 内の `#[cfg(test)]` クレート内テスト（macOS 実機・`#[ignore]`）に
    /// 委ね、外部テストファイルは数値一致のみを検証する契約とする）。
    pub(crate) static RESIDENT_HOST_REPACK_COUNT: std::cell::Cell<u64> =
        const { std::cell::Cell::new(0) };
}

/// GEMM オペランド 1 個をアップロードする（イシュー #1040）。
/// `layout::classify_2d` が分類できる view（行優先 contiguous・転置
/// view のいずれか）は [`MetalMemory::upload_view`] 経由で
/// `Tensor::as_view_slice`（借用）をそのままアップロードし、ホスト側の
/// 転置コピーを発生させない。分類できない形状（stride 0 の
/// ブロードキャスト等）のみ、従来どおり `MemoryOps::upload`
/// （`Tensor::contiguous()` 経由）へフォールバックし
/// [`RESIDENT_HOST_REPACK_COUNT`] を増やす。
///
/// 戻り値の [`MatrixLayout`] は `dispatch_strided_bias_act_prepared` へ
/// そのまま渡す（フォールバック時は `contiguous()` 後の実際の行優先
/// 形状に対応する NN レイアウトを返す）。
fn upload_operand_for_resident_gemm(
    mem: &MetalMemory,
    tensor: &Tensor<f32>,
) -> Result<
    (
        fandhe_ai_tensor_core::buffer::DeviceBuffer<f32>,
        MatrixLayout,
    ),
    BackendError,
> {
    if let Some(layout) = layout::classify_2d(tensor.shape(), tensor.strides())
        && let Some(slice) = tensor.as_view_slice()
    {
        let dev_buf = mem
            .upload_view(slice, tensor.shape())
            .map_err(map_metal_error)?;
        return Ok((dev_buf, layout));
    }
    RESIDENT_HOST_REPACK_COUNT.with(|c| c.set(c.get() + 1));
    let dev_buf = mem.upload(tensor)?;
    let (rows, cols) = (tensor.shape()[0], tensor.shape()[1]);
    let layout = MatrixLayout {
        rows,
        cols,
        ld: cols,
        transposed: false,
    };
    Ok((dev_buf, layout))
}

/// Metal バックエンドの `BackendOps` 実装。`Device::Metal` は ordinal を
/// 持たない単一 variant のため（`docs/public-api-design.md` §4.1・
/// `device.rs::MetalDeviceProvider` と同じ位置付け）、本実装は複数 GPU の
/// 個別選択をサポートしない（システムデフォルトの Metal デバイスに
/// 対応する）。
///
/// `MetalContext`／`MetalGemm`／`MetalElementwise`／`MetalRmsNorm`／
/// `MetalSoftmax` はいずれも `crate::context_cache` 経由でプロセス内
/// キャッシュから取得する（イシュー #930 で常駐化完了。診断 #927 が特定
/// した「演算メソッド呼び出しごとの都度構築」固定オーバーヘッド〈約 5 ms・
/// N 非依存〉を解消する。CUDA 側 `backend-cuda::ops::CudaBackendOps`
/// も同時期に同型キャッシュ〈#929〉へ移行済み）。
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

        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        let ew = context_cache::cached_elementwise(&ctx)
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

        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        let ew = context_cache::cached_elementwise(&ctx)
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

/// `Device::Metal`（単一 variant。ordinal なし）に対応する
/// `&'static MetalMemory` をプロセス内シングルトンとして取得する
/// （イシュー #935）。
///
/// `BackendOps::memory_ops(&self) -> Option<&dyn MemoryOps>` は戻り値を
/// `&self`（`MetalBackendOps`。unit struct・`Copy`）の寿命へ束縛できる
/// 型で返す必要がある一方、`AllocationTracker` の計測系列（`docs/
/// device-resident-update-design.md` §3.3d）を維持するには `MetalMemory`
/// をプロセス全体で 1 個だけ共有しなければならない。`backend-cuda::ops::
/// static_cuda_memory` と同型の意図的な `Box::leak`（`context_cache.rs`
/// モジュール冒頭コメント「所有モデル・生存期間」と同じ「エントリは
/// プロセスの生存期間中 evict されない」設計に倣う。`Device::Metal` は
/// 単一デバイスのためキーは不要）。
///
/// `MetalMemory::from_shared`（イシュー #935 レビュー対応・`memory.rs`
/// 参照）が `context_cache::cached_context()` の返す `Arc<MetalContext>`
/// をそのまま受け取れるため、本関数はバッファ確保用の `MetalContext` と
/// カーネルディスパッチ（`sgd.rs::MetalSgd::run` 等）が使う `MetalContext`
/// を同一インスタンスに揃える（`docs/device-resident-update-design.md`
/// §3.3d「`XMemory` が持つ stream/context は必ず既存 `context_cache`
/// 経由で取得（独自初期化禁止）」）。`MetalMemory::new`（所有権を要求する
/// 既存の公開シグネチャ）は crates.io 公開済み API のため変更しない。
fn static_metal_memory() -> Result<&'static MetalMemory, BackendError> {
    use std::sync::{Mutex, OnceLock};

    static CACHE: OnceLock<Mutex<Option<&'static MetalMemory>>> = OnceLock::new();
    let cell = CACHE.get_or_init(|| Mutex::new(None));
    let mut guard = cell.lock().map_err(|_| {
        BackendError::DeviceUnavailable("static_metal_memory: cache mutex poisoned".to_string())
    })?;
    if let Some(mem) = *guard {
        return Ok(mem);
    }
    let ctx = context_cache::cached_context().map_err(map_metal_error)?;
    let mem: &'static MetalMemory = Box::leak(Box::new(MetalMemory::from_shared(ctx)));
    *guard = Some(mem);
    Ok(mem)
}

impl MetalBackendOps {
    /// [`BackendOps::sgd_step_device`]／
    /// [`BackendOps::sgd_step_device_tracked`] 共通の検証・ディスパッチ
    /// 本体（イシュー #1017 でトークン引数を追加する際に二重化を避ける
    /// ため切り出した。それ以前の検証ロジック自体は無変更）。
    fn sgd_step_device_impl(
        &self,
        param: &mut fandhe_ai_tensor_core::buffer::DeviceBuffer<f32>,
        grad: &fandhe_ai_tensor_core::buffer::DeviceBuffer<f32>,
        velocity: Option<&mut fandhe_ai_tensor_core::buffer::DeviceBuffer<f32>>,
        config: &fandhe_ai_tensor_core::SgdStepConfig,
        token: Option<&DispatchFailureCell>,
    ) -> Result<(), BackendError> {
        if param.device() != Device::Metal || grad.device() != Device::Metal {
            return Err(BackendError::DeviceMismatch);
        }
        if param.shape() != grad.shape() {
            return Err(BackendError::ShapeMismatch(ShapeError::ShapeMismatch {
                lhs: param.shape().to_vec(),
                rhs: grad.shape().to_vec(),
            }));
        }
        let use_momentum = config.momentum != 0.0;
        if let Some(v) = &velocity {
            // デバイス不一致とテンソル shape 不一致を同一の
            // `ShapeMismatch` に丸めていた（Review 指摘。`backend-cuda`
            // 側 `ops.rs::sgd_step_device` と同型の問題）。
            // `BackendOps::sgd_step_device` の契約（`param`/`grad` と同じ
            // く、デバイス不一致は `DeviceMismatch` を返す）に velocity
            // も揃えるため、判定を分離する。
            if v.device() != Device::Metal {
                return Err(BackendError::DeviceMismatch);
            }
            if v.shape() != param.shape() {
                return Err(BackendError::ShapeMismatch(ShapeError::ShapeMismatch {
                    lhs: param.shape().to_vec(),
                    rhs: v.shape().to_vec(),
                }));
            }
        }
        if use_momentum && velocity.is_none() {
            return Err(BackendError::Unsupported(
                "sgd_step_device: momentum enabled but no velocity buffer provided".into(),
            ));
        }

        let numel = param.numel();
        if numel == 0 {
            return Ok(());
        }

        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        let sgd = context_cache::cached_sgd(&ctx).map_err(map_metal_error)?;

        let grad_handle = grad
            .downcast_handle::<MetalBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let Some(grad_metal_buf) = grad_handle.buffer.as_ref() else {
            return Err(BackendError::DeviceAllocationFailed(
                "sgd_step_device: grad buffer has numel > 0 but no device allocation".into(),
            ));
        };

        let velocity_metal_buf = match &velocity {
            Some(v) => {
                let handle = v
                    .downcast_handle::<MetalBufferHandle>()
                    .ok_or(BackendError::DeviceMismatch)?;
                Some(handle.buffer.as_ref().ok_or_else(|| {
                    BackendError::DeviceAllocationFailed(
                        "sgd_step_device: velocity buffer has numel > 0 but no device allocation"
                            .into(),
                    )
                })?)
            }
            None => None,
        };

        let param_handle = param
            .downcast_handle::<MetalBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let Some(param_metal_buf) = param_handle.buffer.as_ref() else {
            return Err(BackendError::DeviceAllocationFailed(
                "sgd_step_device: param buffer has numel > 0 but no device allocation".into(),
            ));
        };

        let kernel_params = crate::sgd::SgdKernelParams {
            lr: config.lr,
            momentum: config.momentum,
            dampening: config.dampening,
            weight_decay: config.weight_decay,
            nesterov: config.nesterov,
            is_first_step: config.is_first_step,
        };
        sgd.run(
            &ctx,
            param_metal_buf,
            grad_metal_buf,
            velocity_metal_buf,
            numel,
            &kernel_params,
            token,
        )
        .map_err(map_metal_error)
    }
}

impl BackendOps for MetalBackendOps {
    fn device(&self) -> Device {
        Device::Metal
    }

    /// `static_metal_memory()`（プロセス内シングルトン）を返す
    /// （イシュー #935）。デバイス非対応等で初期化に失敗した場合は
    /// `None`（`memory_ops` のデフォルト契約と同じ fail-safe）。
    fn memory_ops(&self) -> Option<&dyn MemoryOps> {
        static_metal_memory().ok().map(|m| m as &dyn MemoryOps)
    }

    /// SGD の 1 パラメータ分の更新を in-place で実行する（イシュー #935・
    /// `docs/device-resident-update-design.md` §3.2・§5.2）。
    /// `context_cache::cached_sgd`（プロセス内 MSL コンパイル済みパイプ
    /// ラインキャッシュ）を経由するため、学習ループの 2 回目以降の
    /// ステップは再コンパイルを支払わない。
    ///
    /// 実体は `sgd_step_device_impl`（`token: None`）。
    /// [`BackendOps::sgd_step_device_tracked`] のオーバーライド
    /// （下記）とロジックを共有する（イシュー #1017。二重化しない）が、
    /// `token: None` を受けた `sgd.rs::MetalSgd::run` が `encode` 直後に
    /// `ctx.synchronize()` まで行うため、本メソッドは従来どおり
    /// **同期契約**（復帰時点で GPU 実行の完了・成否を返す）を保つ
    /// （PR #1057 レビュー指摘。バッチ化された非同期契約は
    /// `sgd_step_device_tracked` 限定）。
    fn sgd_step_device(
        &self,
        param: &mut fandhe_ai_tensor_core::buffer::DeviceBuffer<f32>,
        grad: &fandhe_ai_tensor_core::buffer::DeviceBuffer<f32>,
        velocity: Option<&mut fandhe_ai_tensor_core::buffer::DeviceBuffer<f32>>,
        config: &fandhe_ai_tensor_core::SgdStepConfig,
    ) -> Result<(), BackendError> {
        self.sgd_step_device_impl(param, grad, velocity, config, None)
    }

    /// [`BackendOps::sgd_step_device_tracked`] の Metal オーバーライド
    /// （イシュー #1017・`docs/backend-metal-command-batching-design.md`
    /// §3.7）。`token` を `sgd_step_device_impl` → `sgd.rs::
    /// MetalSgd::run` → `context.rs::MetalContext::encode` へそのまま
    /// 渡し、encode と同一ロック区間でバッチへ登録させる。`token` が
    /// `Some` のため `MetalSgd::run` は `encode` 後に待たず、遅延実行
    /// （バッチ化）される非同期契約となる（`sgd_step_device` との違いは
    /// 同メソッド doc 参照）。
    /// `fandhe_ai_autodiff::optim::device_store::DeviceParamStore::step`
    /// が呼び出し元となる（`device_store.rs` モジュール冒頭コメント
    /// 「遅延失敗トークン経由の poison」参照）。
    fn sgd_step_device_tracked(
        &self,
        param: &mut fandhe_ai_tensor_core::buffer::DeviceBuffer<f32>,
        grad: &fandhe_ai_tensor_core::buffer::DeviceBuffer<f32>,
        velocity: Option<&mut fandhe_ai_tensor_core::buffer::DeviceBuffer<f32>>,
        config: &fandhe_ai_tensor_core::SgdStepConfig,
        token: &DispatchFailureCell,
    ) -> Result<(), BackendError> {
        self.sgd_step_device_impl(param, grad, velocity, config, Some(token))
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

        // コンテキスト取得（`context_cache::cached_context`）の失敗
        // （デバイス不在等）は `MetalDeviceProvider::select`（`device.rs`）
        // と同一分類の `BackendError::DeviceUnavailable` に統一する
        // （`map_metal_error` 経由。誤って `DeviceAllocationFailed`〈VRAM／
        // アロケータ起因〉に分類すると、呼び出し側が Metal デバイス不在を
        // 検知する経路が一方に偏る。Bugbot 指摘対応。PR #262 レビュー
        // スレッド。イシュー #930 でプロセス内キャッシュ経由へ変更した
        // 後もこの分類契約は不変）。
        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        let gemm = context_cache::cached_gemm(&ctx)
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
    /// （イシュー #605）。CPU／CUDA 実装と同型の分岐（`gemm_bias_act_route`
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

                let ctx = context_cache::cached_context().map_err(map_metal_error)?;
                let gemm = context_cache::cached_gemm(&ctx)
                    .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
                let out = gemm
                    .run_tiled_bias_act_f32(&ctx, a_slice, b_slice, bias_slice, act_relu, m, n, k)
                    .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
                Tensor::new(out, &out_shape).map_err(BackendError::ShapeMismatch)
            }
        }
    }

    /// デバイス常駐 `w`（・`bias`）のまま `y = a @ w (+ bias)` を計算する
    /// （イシュー #1022・#1023「R3」）。`a`（活性化値）のみをホストから
    /// アップロードし、`w`／`bias` は [`crate::gemm::MetalGemm::
    /// dispatch_bias_act_prepared`]（イシュー #1022 で追加した
    /// prepared 版入口。#1023 でオフセット引数を追加し `DeviceBufferView`
    /// の部分範囲をそのまま `setBuffer:offset:` へ渡せるようにした）へ
    /// そのまま渡すことでこれらの download を発生させない（Apple Silicon
    /// の UMA・`StorageModeShared` のため CUDA のような明示同期は不要。
    /// `memory.rs` モジュールコメント参照）。
    fn gemm_resident_rhs(
        &self,
        a: &Tensor<f32>,
        w: DeviceBufferView<'_>,
        bias: Option<DeviceBufferView<'_>>,
    ) -> Result<Tensor<f32>, BackendError> {
        if w.device() != Device::Metal {
            return Err(BackendError::DeviceMismatch);
        }
        let a_shape = a.shape();
        if a_shape.len() != 2 {
            return Err(BackendError::ShapeMismatch(ShapeError::RankMismatch {
                expected: 2,
                actual: a_shape.len(),
            }));
        }
        let (m, k) = (a_shape[0], a_shape[1]);
        let w_shape = w.shape();
        if w_shape.len() != 2 || w_shape[0] != k {
            return Err(BackendError::ShapeMismatch(ShapeError::ShapeMismatch {
                lhs: a_shape.to_vec(),
                rhs: w_shape.to_vec(),
            }));
        }
        let n = w_shape[1];
        if let Some(b) = bias {
            if b.device() != Device::Metal {
                return Err(BackendError::DeviceMismatch);
            }
            if b.shape() != [n] {
                return Err(BackendError::ShapeMismatch(ShapeError::ShapeMismatch {
                    lhs: b.shape().to_vec(),
                    rhs: vec![n],
                }));
            }
        }
        if k == 0 {
            // `fandhe_ai_autodiff::nn::linear::Linear::new` が
            // `in_features == 0` を構築時に拒否するため到達不能
            // （CPU／CUDA 実装の同分岐と同じ判断。`tensor-core::
            // backend_ops::BackendOps::gemm_resident_rhs` doc 参照）。
            return Err(BackendError::InvalidArgument(
                "gemm_resident_rhs: k == 0 is unreachable via Linear::new (in_features == 0 is \
                 rejected at construction); a host epilogue fallback would require downloading \
                 the resident bias, defeating the zero-D2H contract this method exists for"
                    .to_string(),
            ));
        }
        if m == 0 || n == 0 {
            return Tensor::new(Vec::new(), &[m, n]).map_err(BackendError::ShapeMismatch);
        }

        let w_handle = w
            .buffer()
            .downcast_handle::<MetalBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let Some(w_buf) = w_handle.buffer.as_ref() else {
            return Err(BackendError::DeviceAllocationFailed(
                "gemm_resident_rhs: w buffer has numel > 0 but no device allocation".into(),
            ));
        };
        let bias_handle = bias
            .map(|b| {
                b.buffer()
                    .downcast_handle::<MetalBufferHandle>()
                    .ok_or(BackendError::DeviceMismatch)
                    .map(|h| (h, b.offset()))
            })
            .transpose()?;
        let bias_arg = match &bias_handle {
            Some((h, offset)) => {
                let buf = h.buffer.as_ref().ok_or_else(|| {
                    BackendError::DeviceAllocationFailed(
                        "gemm_resident_rhs: bias buffer has numel > 0 but no device allocation"
                            .into(),
                    )
                })?;
                Some((buf, *offset))
            }
            None => None,
        };

        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        let mem = MetalMemory::from_shared(ctx.clone());
        // イシュー #1040: `a` が転置 view（`classify_2d` で分類可能）の
        // 場合は `Tensor::contiguous()`（ホスト側転置コピー）を経由せず
        // アップロードする（`upload_operand_for_resident_gemm` 参照）。
        let (a_dev_buf, a_layout) = upload_operand_for_resident_gemm(&mem, a)?;
        let a_handle = a_dev_buf
            .downcast_handle::<MetalBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let Some(a_buf) = a_handle.buffer.as_ref() else {
            return Err(BackendError::DeviceAllocationFailed(
                "gemm_resident_rhs: a buffer has numel > 0 but no device allocation".into(),
            ));
        };
        let w_layout = MatrixLayout {
            rows: k,
            cols: n,
            ld: n,
            transposed: false,
        };

        let c_dev_buf = mem.alloc_zeroed(&[m, n])?;
        let c_handle = c_dev_buf
            .downcast_handle::<MetalBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let Some(c_buf) = c_handle.buffer.as_ref() else {
            return Err(BackendError::DeviceAllocationFailed(
                "gemm_resident_rhs: output buffer has numel > 0 but no device allocation".into(),
            ));
        };

        let gemm = context_cache::cached_gemm(&ctx)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        gemm.dispatch_strided_bias_act_prepared(
            &ctx,
            a_buf,
            0,
            a_layout,
            w_buf,
            w.offset(),
            w_layout,
            bias_arg,
            false,
            c_buf,
            m,
            n,
            k,
        )
        .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;

        mem.download(&c_dev_buf)
    }

    /// デバイス常駐 `w` のまま `c = w @ b` を計算する（イシュー #1022・
    /// #1023「R3」）。`Op::LinearResident` の VJP が `d_input^T = w @ g^T`
    /// を計算するために使う。`bias` なし（`None`）で [`Self::
    /// gemm_resident_rhs`] と同じ [`crate::gemm::MetalGemm::
    /// dispatch_bias_act_prepared`] を呼ぶ（`has_bias=0`・`act=0` で
    /// 純粋な `w @ b` になる）。
    fn gemm_resident_lhs(
        &self,
        w: DeviceBufferView<'_>,
        b: &Tensor<f32>,
    ) -> Result<Tensor<f32>, BackendError> {
        if w.device() != Device::Metal {
            return Err(BackendError::DeviceMismatch);
        }
        let w_shape = w.shape();
        if w_shape.len() != 2 {
            return Err(BackendError::ShapeMismatch(ShapeError::RankMismatch {
                expected: 2,
                actual: w_shape.len(),
            }));
        }
        let (p, q) = (w_shape[0], w_shape[1]);
        let b_shape = b.shape();
        if b_shape.len() != 2 || b_shape[0] != q {
            return Err(BackendError::ShapeMismatch(ShapeError::ShapeMismatch {
                lhs: w_shape.to_vec(),
                rhs: b_shape.to_vec(),
            }));
        }
        let r = b_shape[1];
        if p == 0 || r == 0 {
            return Tensor::new(Vec::new(), &[p, r]).map_err(BackendError::ShapeMismatch);
        }
        if q == 0 {
            // `w` の縮約次元（`out_features`）が 0 の場合、GEMM の数学的
            // 定義どおり結果は全 0（CPU／CUDA 実装の同分岐と同じ判断）。
            // GPU 起動を回避してホスト側で直接構築する。
            return Tensor::from_shape_fill(&[p, r], |_| 0.0).map_err(BackendError::ShapeMismatch);
        }

        let w_handle = w
            .buffer()
            .downcast_handle::<MetalBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let Some(w_buf) = w_handle.buffer.as_ref() else {
            return Err(BackendError::DeviceAllocationFailed(
                "gemm_resident_lhs: w buffer has numel > 0 but no device allocation".into(),
            ));
        };

        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        let mem = MetalMemory::from_shared(ctx.clone());
        // イシュー #1040: `Op::LinearResident` の VJP は `transpose2d`
        // した upstream 勾配（転置 view）をここへ渡す。`classify_2d` で
        // 分類できる限りホスト側転置コピーなしでアップロードする
        // （`upload_operand_for_resident_gemm` 参照）。
        let (b_dev_buf, b_layout) = upload_operand_for_resident_gemm(&mem, b)?;
        let b_handle = b_dev_buf
            .downcast_handle::<MetalBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let Some(b_buf) = b_handle.buffer.as_ref() else {
            return Err(BackendError::DeviceAllocationFailed(
                "gemm_resident_lhs: b buffer has numel > 0 but no device allocation".into(),
            ));
        };
        let w_layout = MatrixLayout {
            rows: p,
            cols: q,
            ld: q,
            transposed: false,
        };
        let c_dev_buf = mem.alloc_zeroed(&[p, r])?;
        let c_handle = c_dev_buf
            .downcast_handle::<MetalBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let Some(c_buf) = c_handle.buffer.as_ref() else {
            return Err(BackendError::DeviceAllocationFailed(
                "gemm_resident_lhs: output buffer has numel > 0 but no device allocation".into(),
            ));
        };

        let gemm = context_cache::cached_gemm(&ctx)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        gemm.dispatch_strided_bias_act_prepared(
            &ctx,
            w_buf,
            w.offset(),
            w_layout,
            b_buf,
            0,
            b_layout,
            None,
            false,
            c_buf,
            p,
            r,
            q,
        )
        .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;

        mem.download(&c_dev_buf)
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

    /// [`fandhe_ai_tensor_core::BackendOps::mse_loss`] の Metal 実装
    /// （イシュー #1045）。`Self::sum`／`Self::max`（汎用 reduction）とは
    /// 独立した専用融合カーネル（`crate::mse::MetalMse`）へのディスパッチ
    /// （`backend_ops.rs::BackendOps::mse_loss` doc の設計判断参照）。
    /// `reduction` に応じた `factor`（`Mean` は `1.0/n`、`Sum` は `1.0`）は
    /// ここで計算してカーネルへ渡す。未知 `MseReduction` variant は
    /// `backend-cpu`／`backend-cuda` と同じく `Unsupported` として拒否
    /// する。
    fn mse_loss(
        &self,
        pred: &Tensor<f32>,
        target: &Tensor<f32>,
        reduction: MseReduction,
    ) -> Result<Tensor<f32>, BackendError> {
        require_same_shape(pred.shape(), target.shape()).map_err(BackendError::ShapeMismatch)?;
        let pred_owned = pred.contiguous();
        let target_owned = target.contiguous();
        let pred_slice = pred_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("mse_loss: pred not contiguous".into())
        })?;
        let target_slice = target_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("mse_loss: target not contiguous".into())
        })?;
        let numel = pred_slice.len();
        let factor = match reduction {
            MseReduction::Mean => {
                if numel == 0 {
                    1.0
                } else {
                    1.0 / numel as f32
                }
            }
            MseReduction::Sum => 1.0,
            _ => {
                return Err(BackendError::Unsupported(format!(
                    "mse_loss: unsupported MseReduction variant {reduction:?}"
                )));
            }
        };

        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        let mse = context_cache::cached_mse(&ctx)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        let value = mse
            .run_mse_loss_f32(&ctx, pred_slice, target_slice, factor)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        Tensor::new(vec![value], &[]).map_err(BackendError::ShapeMismatch)
    }

    /// [`fandhe_ai_tensor_core::BackendOps::mse_loss_backward`] の Metal
    /// 実装（イシュー #1045）。`dTarget = −dPred` は呼び出し元
    /// （`fandhe_ai_autodiff::grad::vjp`）がホスト側で符号反転して得る
    /// 契約のため、本メソッドは `dPred` のみを計算して返す
    /// （`backend_ops.rs::BackendOps::mse_loss_backward` doc 参照）。
    fn mse_loss_backward(
        &self,
        pred: &Tensor<f32>,
        target: &Tensor<f32>,
        scale: f32,
    ) -> Result<Tensor<f32>, BackendError> {
        require_same_shape(pred.shape(), target.shape()).map_err(BackendError::ShapeMismatch)?;
        let pred_owned = pred.contiguous();
        let target_owned = target.contiguous();
        let pred_slice = pred_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("mse_loss_backward: pred not contiguous".into())
        })?;
        let target_slice = target_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("mse_loss_backward: target not contiguous".into())
        })?;

        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        let mse = context_cache::cached_mse(&ctx)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        let out = mse
            .run_mse_backward_f32(&ctx, pred_slice, target_slice, scale)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        Tensor::new(out, pred.shape()).map_err(BackendError::ShapeMismatch)
    }

    /// [`fandhe_ai_tensor_core::BackendOps::run_fused`] のデフォルト実装
    /// （`Unsupported` fail-safe）を、canonical RMSNorm／softmax 融合プラン
    /// 検出時のみ融合カーネル（[`crate::rmsnorm::MetalRmsNorm`]／
    /// [`crate::softmax::MetalSoftmax`]）へルーティングする（イシュー #604）。
    /// CUDA 側 `CudaBackendOps::run_fused`
    /// （#592）の fail-closed 検証列をそのまま踏襲する:
    ///
    /// 1. `row_kernel::match_rmsnorm_plan`（6 op 列・leaf 1 個・
    ///    `axis: None` のみ受理）→ 一致時 [`crate::rmsnorm::MetalRmsNorm`] へ
    /// 2. `row_kernel::match_softmax_plan`（8 op 列・leaf 1 個・`axis` が
    ///    最終次元または `None` のみ受理）→ 一致時 [`crate::softmax::MetalSoftmax`] へ
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

    /// REQ-14 の明示解放 API（イシュー #1021）。`crate::pool::
    /// MetalAllocator::release_cached`（内部メソッド。設計文書 §3.1
    /// 「2 段構成の命名規約」・§3.6 (2) のバックエンド別フェーズ表
    /// 「Metal」列）へ委譲する。バイト数は
    /// `device_memory_pool_stats()`（`PoolStats::cached_bytes`）から
    /// 確認できるため戻り値からは捨てる（設計文書 §3.1 同節）。
    fn release_cached_device_memory(&self) -> Result<(), BackendError> {
        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        context_cache::cached_allocator(&ctx)
            .map_err(map_metal_error)?
            .release_cached()
            .map(|_bytes| ())
            .map_err(map_metal_error)
    }

    /// デバイスメモリプールの統計スナップショット（イシュー #1021）。
    /// コンテキスト・アロケータ取得自体が失敗した場合（デバイス不在等）
    /// は `None`（プールを持たない扱い。`BackendOps::
    /// device_memory_pool_stats` の既定契約と同じ fail-safe）とし、
    /// panic・`Err` の握り潰しはしない（呼び出し元が診断目的でしか
    /// 使わない値のため、デバイス不在自体は他の演算メソッド呼び出しで
    /// 既に検出できる。ここで `Result` を返す必要はないという既存
    /// trait シグネチャ〈`Option<PoolStats>`〉の制約に従う）。
    fn device_memory_pool_stats(&self) -> Option<fandhe_ai_tensor_core::PoolStats> {
        let ctx = context_cache::cached_context().ok()?;
        let allocator = context_cache::cached_allocator(&ctx).ok()?;
        Some(allocator.stats())
    }
}

impl MetalBackendOps {
    /// `a`（rank >= 2 の `[B0, …, M, K]`）の先頭次元を行次元へ畳み、
    /// `b`（`[K, N]`）との GEMM を `[B0, …, M, N]` として計算する
    /// （イシュー #1040。バッチ matmul の公開 API 化は別イシュー——
    /// `BackendOps` trait は変更せず、本メソッドは `MetalBackendOps` の
    /// inherent メソッドとして追加する）。
    ///
    /// `crate::layout::collapse_leading_dims` が `Some` を返す場合
    /// （先頭次元が連続 view として畳める場合）は `a` を
    /// `contiguous()` せずそのまま `MatrixLayout` へ変換し、
    /// `gemm::MetalGemm::dispatch_strided_bias_act_prepared` へ渡す。
    /// `None`（collapse 不能な非連続 view）の場合は `a.contiguous()`
    /// 後に `[B0*…*M, K]` へ reshape してから同じ入口へ渡す
    /// （collapse 可否に関わらず数値結果は同一——`gemm.metal` の
    /// 添字計算は「連続な行優先バッファ」という前提のみに依存する）。
    ///
    /// `bias`・activation は扱わない（`gemm_bias_act` の trait 実装が
    /// 別途担う）。
    pub fn gemm_collapsed_lhs(
        &self,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
    ) -> Result<Tensor<f32>, BackendError> {
        if a.rank() < 2 {
            return Err(BackendError::ShapeMismatch(ShapeError::RankMismatch {
                expected: 2,
                actual: a.rank(),
            }));
        }
        let b_shape = b.shape();
        if b_shape.len() != 2 {
            return Err(BackendError::ShapeMismatch(ShapeError::RankMismatch {
                expected: 2,
                actual: b_shape.len(),
            }));
        }
        let a_shape = a.shape().to_vec();
        let k = a_shape[a_shape.len() - 1];
        if b_shape[0] != k {
            return Err(BackendError::ShapeMismatch(ShapeError::ShapeMismatch {
                lhs: a_shape.clone(),
                rhs: b_shape.to_vec(),
            }));
        }
        let n = b_shape[1];
        let batch_dims = &a_shape[..a_shape.len() - 1];
        let out_shape: Vec<usize> = batch_dims.iter().copied().chain([n]).collect();
        let m: usize = batch_dims.iter().product();

        // イシュー #1040 是正（codex-review・Cursor Bugbot 指摘）: `m == 0`
        // （バッチ次元のいずれかが 0）または `n == 0` の場合、出力の要素数
        // は 0 であり GPU 起動は不要（`gemm_resident_lhs`／
        // `gemm_resident_rhs` と同じ「numel == 0 は空 Tensor」の判断。
        // Metal context 取得前に判定することで `MetalBufferHandle::
        // buffer == None`（空バッファ）に起因する `DeviceAllocationFailed`
        // を回避する）。`k == 0`（`m`・`n` は非 0）は GEMM の数学的定義
        // どおり結果が全 0（`gemm_resident_lhs` の同分岐と同じ判断）で、
        // こちらも GPU 起動を避けホスト側で直接構築する。
        if m == 0 || n == 0 {
            return Tensor::new(Vec::new(), &out_shape).map_err(BackendError::ShapeMismatch);
        }
        if k == 0 {
            return Tensor::from_shape_fill(&out_shape, |_| 0.0)
                .map_err(BackendError::ShapeMismatch);
        }

        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        let mem = MetalMemory::from_shared(ctx.clone());

        // `a` を collapse 可能なら zero-copy（`as_view_slice`）、
        // 不能なら `contiguous()` 後の実際の行優先形状で NN レイアウトを
        // 構築する（`upload_operand_for_resident_gemm` と同じ
        // 「分類できなければ contiguous へフォールバック」方針だが、
        // 本メソッドは rank >= 2 の先頭次元 collapse を扱うため
        // `collapse_leading_dims` を使う専用ロジックとする）。
        let (a_dev_buf, a_layout, m) = match layout::collapse_leading_dims(a.shape(), a.strides()) {
            Some(collapsed) => {
                let slice = a.as_view_slice().ok_or_else(|| {
                    BackendError::KernelLaunchFailed(
                        "gemm_collapsed_lhs: collapse_leading_dims succeeded but \
                         as_view_slice returned None (non-negative-stride invariant violated)"
                            .into(),
                    )
                })?;
                let dev_buf = mem.upload_view(slice, a.shape()).map_err(map_metal_error)?;
                let m = collapsed.rows;
                (dev_buf, collapsed, m)
            }
            None => {
                let a_owned = a.contiguous();
                let m: usize = batch_dims.iter().product();
                let a_reshaped = a_owned
                    .reshape(&[m, k])
                    .map_err(BackendError::ShapeMismatch)?;
                let dev_buf = mem.upload(&a_reshaped)?;
                let layout = MatrixLayout {
                    rows: m,
                    cols: k,
                    ld: k,
                    transposed: false,
                };
                (dev_buf, layout, m)
            }
        };
        let a_handle = a_dev_buf
            .downcast_handle::<MetalBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let Some(a_buf) = a_handle.buffer.as_ref() else {
            return Err(BackendError::DeviceAllocationFailed(
                "gemm_collapsed_lhs: a buffer has numel > 0 but no device allocation".into(),
            ));
        };

        let (b_dev_buf, b_layout) = upload_operand_for_resident_gemm(&mem, b)?;
        let b_handle = b_dev_buf
            .downcast_handle::<MetalBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let Some(b_buf) = b_handle.buffer.as_ref() else {
            return Err(BackendError::DeviceAllocationFailed(
                "gemm_collapsed_lhs: b buffer has numel > 0 but no device allocation".into(),
            ));
        };

        let c_dev_buf = mem.alloc_zeroed(&[m, n])?;
        let c_handle = c_dev_buf
            .downcast_handle::<MetalBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let Some(c_buf) = c_handle.buffer.as_ref() else {
            return Err(BackendError::DeviceAllocationFailed(
                "gemm_collapsed_lhs: output buffer has numel > 0 but no device allocation".into(),
            ));
        };

        let gemm = context_cache::cached_gemm(&ctx)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        gemm.dispatch_strided_bias_act_prepared(
            &ctx, a_buf, 0, a_layout, b_buf, 0, b_layout, None, false, c_buf, m, n, k,
        )
        .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;

        let c_tensor = mem.download(&c_dev_buf)?;
        c_tensor
            .reshape(&out_shape)
            .map_err(BackendError::ShapeMismatch)
    }

    /// 一致した leaf・shape・dtype を検証してから
    /// [`crate::rmsnorm::MetalRmsNorm::run_rmsnorm_f32_raw`] を `inv_n = 1.0`・`eps = 0.0`・
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

        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        let rmsnorm = context_cache::cached_rmsnorm(&ctx)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        let out = rmsnorm
            .run_rmsnorm_f32_raw(&ctx, x_slice, None, 0.0, 1.0, 1, hidden)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        Tensor::new(out, plan.output_shape()).map_err(BackendError::ShapeMismatch)
    }

    /// [`Self::run_fused_rmsnorm`] と同じ検証順序で
    /// [`crate::softmax::MetalSoftmax::run_softmax_f32`] を呼ぶ。
    ///
    /// `row_kernel::match_softmax_plan` は `axis: None`（全軸縮約。
    /// `rows = 1` 相当）だけでなく `axis` が最終次元（行方向縮約。
    /// `rows > 1` になりうる）も受理するため、[`Self::run_fused_rmsnorm`]
    /// と異なり `rows` を固定値にできない。`x_slice.len() / hidden`
    /// （`validate_fused_leaf` が `x.shape() == plan.output_shape()` を
    /// 検証済みのため `x_slice.len()` は `plan.output_shape()` の要素数積
    /// と一致する）から導出する。`hidden == 0` は
    /// `crate::softmax::MetalSoftmax::run_softmax_f32` 側の 0 要素早期 return 契約に委ね、
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

        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        let softmax = context_cache::cached_softmax(&ctx)
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

    // --- gemm_resident_lhs／gemm_resident_rhs zero-repack 経路（イシュー
    // #1040。Metal 実機依存。`tests/gemm_bias_act_parity.rs` と同じ
    // 「pub(crate) カウンタはクレート内テストで検証」方針） ---

    #[test]
    #[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
    fn gemm_resident_lhs_transposed_b_does_not_increment_repack_counter() {
        let ops = MetalBackendOps::new();
        let mem = ops
            .memory_ops()
            .expect("MetalBackendOps must implement MemoryOps");
        let (p, q, r) = (4usize, 8usize, 5usize);
        let w = Tensor::new((0..p * q).map(|i| i as f32 * 0.1).collect(), &[p, q]).unwrap();
        let w_dev = mem.upload(&w).expect("w upload must succeed");
        let w_shape = [p, q];
        let w_view = DeviceBufferView::new(&w_dev, 0, &w_shape).unwrap();

        // 転置 view（`Op::LinearResident` の VJP が渡す実際の形と同じ:
        // 元 [r, q] 行優先データを `transpose(0, 1)` して [q, r] として
        // 読む）。
        let b_rq = Tensor::new((0..r * q).map(|i| i as f32 * 0.01).collect(), &[r, q]).unwrap();
        let b_t = b_rq.transpose(0, 1).unwrap();
        assert!(
            b_t.as_slice().is_none(),
            "precondition: b_t must be non-contiguous"
        );

        let before = RESIDENT_HOST_REPACK_COUNT.with(|c| c.get());
        let _ = ops
            .gemm_resident_lhs(w_view, &b_t)
            .expect("gemm_resident_lhs must succeed on Metal-equipped test runner");
        let after = RESIDENT_HOST_REPACK_COUNT.with(|c| c.get());
        assert_eq!(
            before, after,
            "transposed view input must not fall back to contiguous() host repack"
        );
    }
}
