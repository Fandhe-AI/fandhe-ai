//! CUDA バックエンドの `BackendOps` 実装（TASK-1.9c・#46。イシュー #599 で
//! elementwise 5 演算・`gemm_bias_act` 実融合化を追加）。
//!
//! `fandhe_ai_tensor_core::backend_ops::BackendOps` の CUDA 実装。GEMM は
//! `gemm::CudaGemm::run_tiled_f32` へ委譲する（既存カーネル・許容誤差・
//! 境界検査には触れない）。`run_tiled_f32` 自体は内部で cp.async 3 stage
//! パイプラインカーネルへ形状条件付きに分岐しうる（整列形状のみ。
//! イシュー #1137・`gemm.rs::CudaGemm::select_tiled_f32_kernel`）ため、
//! 本ファイルのコードはこの分岐を意識せず既定 `run_tiled_f32` を呼ぶだけで
//! よい。elementwise（`add`／`mul`／`relu`／`exp`／
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

use fandhe_ai_tensor_core::buffer::{DeviceBufferView, MemoryOps};
use fandhe_ai_tensor_core::device::{BackendError, Device};
use fandhe_ai_tensor_core::{
    Activation, BackendOps, DType, FusionPlan, MseReduction, ShapeError, Tensor, require_same_shape,
};

use crate::context_cache;
use crate::device::CudaDevice;
use crate::elementwise::CudaElementwise;
use crate::error::CudaError;
use crate::memory::{CudaBufferHandle, CudaMemory, map_cuda_error};
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
    /// GEMM 本体（f32）の FP32 厳密経路（`run_tiled_f32`）のみを実行する
    /// 内部ヘルパー。`crate::precision::tf32_gemm_enabled()` の状態に
    /// 関わらず常に FP32 厳密で計算する（TF32 opt-in フラグを一切見ない）。
    ///
    /// `gemm`（公開経路。opt-in 時は TF32 へ分岐しうる）と
    /// `gemm_bias_act` の `ComposedFallback`（非融合合成経路）の双方から
    /// 呼ばれる。後者が `self.gemm(a, b)` を直接呼ぶと `gemm` 側の TF32
    /// 分岐へ意図せず波及し、`gemm_bias_act` は本イシュー（#1042）の
    /// 適用範囲外のまま FP32 で動作するという `crate::precision` モジュール
    /// 冒頭コメントの契約（「適用範囲は `CudaBackendOps::gemm`（素の f32
    /// GEMM）のみ」）に反するため、`ComposedFallback` は必ずこのヘルパーを
    /// 経由する（codex-review 指摘。PR #1091）。
    fn gemm_fp32_strict(
        &self,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
    ) -> Result<Tensor<f32>, BackendError> {
        let out_shape = fandhe_ai_tensor_core::matmul_out_shape(a.shape(), b.shape())
            .map_err(BackendError::ShapeMismatch)?;
        let (m, k) = (a.shape()[0] as u32, a.shape()[1] as u32);
        let n = b.shape()[1] as u32;

        // `run_tiled_f32`／`run_wmma_tf32` はいずれも contiguous な
        // `&[f32]` を要求する（CPU 実装と同じ契約。`ops.rs`（backend-cpu）
        // 参照）。
        let a_owned = a.contiguous();
        let b_owned = b.contiguous();
        let a_slice = a_owned
            .as_slice()
            .ok_or_else(|| BackendError::KernelLaunchFailed("gemm: lhs not contiguous".into()))?;
        let b_slice = b_owned
            .as_slice()
            .ok_or_else(|| BackendError::KernelLaunchFailed("gemm: rhs not contiguous".into()))?;

        let gemm = self.with_driver_call(
            &[],
            |e| BackendError::CudaUnavailable(e.to_string()),
            || {
                let device = self.device_handle_raw()?;
                context_cache::cached_gemm(&device)
            },
        )?;
        let out = self.with_driver_call(
            &[],
            |e| BackendError::KernelLaunchFailed(e.to_string()),
            || gemm.run_tiled_f32(a_slice, b_slice, m, n, k),
        )?;
        Tensor::new(out, &out_shape).map_err(BackendError::ShapeMismatch)
    }
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
    ///
    /// **poison 検査を経由しない**（codex-review P0 指摘・PR #1064 追補・
    /// `ops.rs:147` 相当: `device_handle()` はキャッシュミス時に
    /// `CudaDevice::new` を呼び実際に driver を操作するが、これを
    /// `with_driver_call`（`begin_driver_call` によるポイズン検査を含む）
    /// より前に呼ぶと、poison 済み ordinal でも拒否前に driver 操作が
    /// 走ってしまい、その失敗も観測されない）。そのため本メソッドは
    /// [`Self::memory_ops`]／[`Self::device_memory_pool_stats`] という
    /// 「driver へは触れず `Option` で fail-safe に縮退する」経路専用に
    /// 限定して使い、driver を実際に操作する演算（`gemm`／`elementwise`
    /// 等）は [`Self::device_handle_raw`] を `with_driver_call` の
    /// クロージャ内部から呼ぶ（poison 検査の後）。
    fn device_handle(&self) -> Result<Arc<CudaDevice>, BackendError> {
        self.device_handle_raw()
            .map_err(|e: CudaError| BackendError::CudaUnavailable(e.to_string()))
    }

    /// [`Self::device_handle`] の `CudaError` 版。`with_driver_call` の
    /// クロージャ内部（＝ `begin_driver_call` によるポイズン検査の後）から
    /// 呼ぶことで、`CudaDevice::new`（キャッシュミス時の driver 初期化）
    /// 自体も poison 検査・sticky エラー観測の対象に含める
    /// （codex-review P0 指摘・PR #1064 追補）。
    fn device_handle_raw(&self) -> Result<Arc<CudaDevice>, CudaError> {
        context_cache::cached_device(self.ordinal)
    }

    /// `BackendOps` の各公開メソッドが唯一の driver 呼び出し境界として
    /// 使う共通ヘルパー（イシュー #1013 設計文書 §9 item 7・9。PR #1064
    /// の Phase C 結線。`memory.rs::CudaMemory::with_driver_call` と同じ
    /// 設計）。
    ///
    /// `context_cache::begin_driver_call` を演算入口で 1 回だけ呼び
    /// （`resource_generations` には当該演算が読み書きするデバイス常駐
    /// バッファ〈`DeviceBuffer`／`DeviceBufferView`〉の
    /// [`fandhe_ai_tensor_core::buffer::DeviceBuffer::generation`] を渡す。
    /// ホスト `Tensor` のみを読み書きする演算〈`gemm`／`add`／`relu` 等〉
    /// には検査対象の既存デバイス常駐バッファがないため空スライスでよく、
    /// これは検査を省略する fail-open ではなく「1 回の呼び出し内で
    /// 完結し、跨ぐ世代が存在しない」ことに対応する）、`f` の内部で
    /// `gemm.rs`／`elementwise.rs`／`softmax.rs`／`rmsnorm.rs`／`sgd.rs`
    /// が行う 1 回以上の driver 呼び出しの結果（`?` で直結しているため
    /// 呼び出し元まで伝播する `CudaError` は常に最初に失敗した 1 回を
    /// 表す）を `observe_cuda_result` で観測し、sticky エラーなら
    /// ordinal を poison する。
    ///
    /// **cold-cache 構築も同じ境界に含める**（Cursor Bugbot 指摘・
    /// PR #1064 追補）: `context_cache::cached_gemm`／`cached_elementwise`／
    /// `cached_rmsnorm`／`cached_softmax`／`cached_sgd` はキャッシュミス時
    /// （初回呼び出し、または将来 `invalidate` が新世代のコンテキストを
    /// 再構築した直後）に NVRTC コンパイル・モジュールロードという実際の
    /// driver 呼び出しを行う。この構築呼び出しを `with_driver_call` の
    /// 外側（`device_handle()` 直後）で素通しに実行すると、構築中に
    /// sticky エラーが発生しても観測されず ordinal が poison されない
    /// まま fail-open になる（構築失敗自体はキャッシュされず毎回
    /// 再試行されるため、poison されない限りこの経路は永久に「観測なしで
    /// 消費される」窓になる）。各呼び出し元は `cached_*` 取得自体も
    /// 本ヘルパーで包む（`f` に `context_cache::cached_gemm(&device)` 等を
    /// 渡す）ことでこの窓を閉じる。`resource_generations` は続く実行部と
    /// 同じ値を渡し（`begin_driver_call` は世代不一致以外の目的では
    /// 副作用を持たないため二重に渡しても安全）、構築用のトークンと
    /// 実行用のトークンは別個に取得・解放する。
    fn with_driver_call<T>(
        &self,
        resource_generations: &[u64],
        map: impl FnOnce(CudaError) -> BackendError,
        f: impl FnOnce() -> Result<T, CudaError>,
    ) -> Result<T, BackendError> {
        let token = context_cache::begin_driver_call(self.ordinal, resource_generations)?;
        context_cache::observe_cuda_result(self.ordinal, &token, f()).map_err(map)
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

        let ew = self.with_driver_call(
            &[],
            |e| BackendError::CudaUnavailable(e.to_string()),
            || {
                let device = self.device_handle_raw()?;
                context_cache::cached_elementwise(&device)
            },
        )?;
        let out = self.with_driver_call(
            &[],
            |e| BackendError::KernelLaunchFailed(e.to_string()),
            || run(&ew, a_slice, b_slice),
        )?;
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

        let ew = self.with_driver_call(
            &[],
            |e| BackendError::CudaUnavailable(e.to_string()),
            || {
                let device = self.device_handle_raw()?;
                context_cache::cached_elementwise(&device)
            },
        )?;
        let out = self.with_driver_call(
            &[],
            |e| BackendError::KernelLaunchFailed(e.to_string()),
            || run(&ew, a_slice),
        )?;
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

        let rmsnorm = self.with_driver_call(&[], map_fused_kernel_init_error, || {
            let device = self.device_handle_raw()?;
            context_cache::cached_rmsnorm(&device)
        })?;
        let out = self.with_driver_call(
            &[],
            |e| BackendError::KernelLaunchFailed(e.to_string()),
            || rmsnorm.run_rmsnorm_f32_raw(x_slice, None, 0.0, 1.0, 1, hidden),
        )?;
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

        let softmax = self.with_driver_call(&[], map_fused_kernel_init_error, || {
            let device = self.device_handle_raw()?;
            context_cache::cached_softmax(&device)
        })?;
        let out = self.with_driver_call(
            &[],
            |e| BackendError::KernelLaunchFailed(e.to_string()),
            || softmax.run_softmax_f32_raw(x_slice, std::f32::consts::LOG2_E, rows, cols),
        )?;
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

/// `ordinal` に対応する `&'static CudaMemory` をプロセス内キャッシュから
/// 取得する（イシュー #935）。
///
/// `BackendOps::memory_ops(&self) -> Option<&dyn MemoryOps>` は戻り値の
/// 参照を `&self`（`CudaBackendOps`。`ordinal: usize` のみを持つ軽量な
/// `Copy` 値で、呼び出しのたびに新規構築されうる）の寿命に束縛できる型で
/// 返す必要がある一方、`AllocationTracker` の計測系列（`docs/
/// device-resident-update-design.md` §3.3d「計測系列単一化」）を維持する
/// には `CudaMemory` 自体をプロセス全体で 1 個だけ共有しなければならない。
/// `context_cache`（`Arc<T>` を返す）はこの用途に使えない（`Arc` の中身は
/// `&self` の寿命へ縮小できるが、`Arc` 自体をどこかに所有し続ける主体が
/// 必要で、`CudaBackendOps` 自身はフィールド追加不可の `Copy` 値のため
/// 保持先がない）ため、本関数は `Box::leak` で `'static` 参照へ格上げして
/// 保持する。`ordinal`（物理 GPU 台数で有界）をキーとする点は
/// `context_cache` と同じ「エントリはプロセスの生存期間中 evict されない」
/// 設計（`context_cache.rs` モジュール冒頭コメント「所有モデル・生存
/// 期間」）に倣った意図的なリークであり、通常のメモリリークとは区別する。
fn static_cuda_memory(
    ordinal: usize,
    device: &CudaDevice,
) -> Result<&'static CudaMemory, BackendError> {
    use std::collections::HashMap;
    use std::sync::Mutex;

    static CACHE: std::sync::OnceLock<Mutex<HashMap<usize, &'static CudaMemory>>> =
        std::sync::OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache.lock().map_err(|_| {
        BackendError::DeviceUnavailable("static_cuda_memory: cache mutex poisoned".to_string())
    })?;
    if let Some(mem) = guard.get(&ordinal) {
        return Ok(mem);
    }
    let mem: &'static CudaMemory = Box::leak(Box::new(CudaMemory::new(device)));
    guard.insert(ordinal, mem);
    Ok(mem)
}

impl BackendOps for CudaBackendOps {
    fn device(&self) -> Device {
        Device::Cuda(self.ordinal)
    }

    /// `context_cache::cached_device` で得たデバイス上に `ordinal` キーの
    /// プロセス内シングルトン `CudaMemory` を構築・共有する
    /// （`static_cuda_memory`。イシュー #935）。driver 不在等で
    /// `device_handle()` が失敗した場合は `None` を返す（`memory_ops`
    /// のデフォルト契約と同じ fail-safe。`tensor-core::backend_ops`
    /// 参照）。
    fn memory_ops(&self) -> Option<&dyn MemoryOps> {
        let device = self.device_handle().ok()?;
        static_cuda_memory(self.ordinal, &device)
            .ok()
            .map(|m| m as &dyn MemoryOps)
    }

    /// SGD の 1 パラメータ分の更新を in-place で実行する（イシュー #935・
    /// `docs/device-resident-update-design.md` §3.2・§5.2）。
    /// `context_cache::cached_sgd`（`ordinal` キーのプロセス内 NVRTC
    /// コンパイル済みカーネルキャッシュ）を経由するため、学習ループの
    /// 2 回目以降のステップは再コンパイルを支払わない。
    fn sgd_step_device(
        &self,
        param: &mut fandhe_ai_tensor_core::buffer::DeviceBuffer<f32>,
        grad: &fandhe_ai_tensor_core::buffer::DeviceBuffer<f32>,
        velocity: Option<&mut fandhe_ai_tensor_core::buffer::DeviceBuffer<f32>>,
        config: &fandhe_ai_tensor_core::SgdStepConfig,
    ) -> Result<(), BackendError> {
        if param.device() != Device::Cuda(self.ordinal)
            || grad.device() != Device::Cuda(self.ordinal)
        {
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
            // `ShapeMismatch` に丸めていた（Review 指摘）。
            // `BackendOps::sgd_step_device` の契約（`param`/`grad` と同じ
            // く、デバイス不一致は `DeviceMismatch` を返す）に velocity
            // も揃えるため、判定を分離する。
            if v.device() != Device::Cuda(self.ordinal) {
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

        // イシュー #1013 設計文書 §9 item 7: `param`／`grad`／`velocity` は
        // 学習ループを跨いで生存するデバイス常駐バッファ（`docs/
        // device-resident-update-design.md` §3.2）であり、`invalidate` に
        // よる回復（poison → 新世代）を跨いで使い回されうる唯一の経路
        // （`gemm`／`elementwise` 等はホスト `Tensor` を都度アップロードし
        // 直すため世代を跨がない）。ハンドルを可変借用する前に、この
        // 時点の世代を収集しておく（`downcast_handle_mut` 後は `param`／
        // `velocity` を再度 `&` で読めないため）。
        let resource_generations: Vec<u64> = std::iter::once(param.generation())
            .chain(std::iter::once(grad.generation()))
            .chain(velocity.as_deref().map(|v| v.generation()))
            .collect();

        let sgd = self.with_driver_call(&resource_generations, map_cuda_error, || {
            let device = self.device_handle_raw()?;
            context_cache::cached_sgd(&device)
        })?;

        let grad_handle = grad
            .downcast_handle::<CudaBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        // `download`（`memory.rs::CudaMemory::download`）と同じ「空バッファ
        // は `slice: None`」契約のため、numel == 0 はカーネル起動前に
        // 早期 return する（`CudaSgd::run` 側の `numel == 0` early-return
        // では `grad_slice` を取り出す前に `param_slice` を要求してしまう
        // ため、ここで先に判定する）。
        let numel = param.numel();
        if numel == 0 {
            return Ok(());
        }
        let Some(grad_slice) = grad_handle.slice.as_ref() else {
            return Err(BackendError::DeviceAllocationFailed(
                "sgd_step_device: grad buffer has numel > 0 but no device allocation".into(),
            ));
        };

        let velocity_handle_slice = match velocity {
            Some(v) => {
                let handle = v
                    .downcast_handle_mut::<CudaBufferHandle>()
                    .ok_or(BackendError::DeviceMismatch)?;
                Some(handle.slice.as_mut().ok_or_else(|| {
                    BackendError::DeviceAllocationFailed(
                        "sgd_step_device: velocity buffer has numel > 0 but no device allocation"
                            .into(),
                    )
                })?)
            }
            None => None,
        };

        let param_handle = param
            .downcast_handle_mut::<CudaBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let Some(param_slice) = param_handle.slice.as_mut() else {
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
        self.with_driver_call(&resource_generations, map_cuda_error, || {
            sgd.run(
                param_slice,
                grad_slice,
                velocity_handle_slice,
                &kernel_params,
            )
        })
    }

    /// GEMM 本体（f32）。既定は FP32 厳密経路（`run_tiled_f32`）で、
    /// 本イシュー導入前と bit-exact に不変（`crate::precision` モジュール
    /// 冒頭コメントの契約）。`crate::precision::tf32_gemm_enabled()` が
    /// opt-in（`true`）の場合のみ WMMA TF32 Tensor Core 経路
    /// （[`crate::gemm::CudaGemm::run_wmma_tf32`]）へ分岐する（イシュー
    /// #1042。親ツリー #1029 Phase 2）。opt-in 時に TF32 カーネルが使用
    /// 不能（cc<8.0・NVRTC コンパイル失敗等）な場合は
    /// `CudaError::WmmaUnavailable` をそのまま `BackendError` へ変換して
    /// 伝播し、FP32 への黙示フォールバックはしない（fail-closed。明示
    /// opt-in の計測条件を静かに崩さない方針。`crate::precision` 参照）。
    ///
    /// **注意**: `gemm_bias_act` の `ComposedFallback` からはこの
    /// メソッドを呼ばない（`gemm_fp32_strict` を使う）。本メソッドは
    /// TF32 opt-in フラグの適用対象である「素の公開 GEMM 入口」専用。
    fn gemm(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        if !crate::precision::tf32_gemm_enabled() {
            return self.gemm_fp32_strict(a, b);
        }

        let out_shape = fandhe_ai_tensor_core::matmul_out_shape(a.shape(), b.shape())
            .map_err(BackendError::ShapeMismatch)?;
        let (m, k) = (a.shape()[0] as u32, a.shape()[1] as u32);
        let n = b.shape()[1] as u32;

        let a_owned = a.contiguous();
        let b_owned = b.contiguous();
        let a_slice = a_owned
            .as_slice()
            .ok_or_else(|| BackendError::KernelLaunchFailed("gemm: lhs not contiguous".into()))?;
        let b_slice = b_owned
            .as_slice()
            .ok_or_else(|| BackendError::KernelLaunchFailed("gemm: rhs not contiguous".into()))?;

        let gemm = self.with_driver_call(
            &[],
            |e| BackendError::CudaUnavailable(e.to_string()),
            || {
                let device = self.device_handle_raw()?;
                context_cache::cached_gemm(&device)
            },
        )?;
        let out = self.with_driver_call(
            &[],
            |e| BackendError::KernelLaunchFailed(e.to_string()),
            || gemm.run_wmma_tf32(a_slice, b_slice, m, n, k),
        )?;
        crate::gemm::TF32_OPTIN_GEMM_LAUNCH_COUNT.with(|c| c.set(c.get() + 1));
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
    /// 実装と同じ 3 段合成（`self.gemm_fp32_strict` → `self.add` →
    /// `self.relu`）へフォールバックする。`self.gemm`（TF32 opt-in 分岐
    /// を持つ公開経路）ではなく `gemm_fp32_strict` を使うのは、
    /// `gemm_bias_act` が `crate::precision` モジュール冒頭コメントの
    /// 契約どおり本イシュー（#1042）のスコープ外のまま常に FP32 で
    /// 動作することを保証するため（codex-review 指摘。PR #1091）。
    /// 両バックエンドは本イシュー時点で `add`／`relu`
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
                let mut out = self.gemm_fp32_strict(a, b)?;
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

                let gemm = self.with_driver_call(
                    &[],
                    |e| BackendError::CudaUnavailable(e.to_string()),
                    || {
                        let device = self.device_handle_raw()?;
                        context_cache::cached_gemm(&device)
                    },
                )?;
                let out = self.with_driver_call(
                    &[],
                    |e| BackendError::KernelLaunchFailed(e.to_string()),
                    || gemm.run_tiled_bias_act_f32(a_slice, b_slice, bias_slice, act_relu, m, n, k),
                )?;
                Tensor::new(out, &out_shape).map_err(BackendError::ShapeMismatch)
            }
        }
    }

    /// デバイス常駐 `w`（・`bias`）のまま `y = a @ w (+ bias)` を計算する
    /// （イシュー #1022・#1023「R3」）。`a`（活性化値）のみをホストから
    /// アップロードし、`w`／`bias` は [`crate::gemm::CudaGemm::
    /// launch_tiled_bias_act_f32_resident`]（`CudaView` 部分ビュー起動。
    /// #1023 のパラメータ横断連結バッファ化後、`w`／`bias` は連結
    /// バッファ内のオフセット範囲としてしか表現できないため、`w`／
    /// `bias` を `DeviceBufferView`（`offset`／`shape` 付き）で受け取り、
    /// `CudaSlice::slice(offset..offset+numel)` で `CudaView` を構築して
    /// カーネルへ渡す）へそのまま渡すことで、これらの download を
    /// 発生させない（`sgd_step_device` と同じ「転送コストを最小化する」
    /// 方針）。
    fn gemm_resident_rhs(
        &self,
        a: &Tensor<f32>,
        w: DeviceBufferView<'_>,
        bias: Option<DeviceBufferView<'_>>,
    ) -> Result<Tensor<f32>, BackendError> {
        if w.device() != Device::Cuda(self.ordinal) {
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
            if b.device() != Device::Cuda(self.ordinal) {
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
            // `in_features == 0` を構築時に拒否するため、この分岐は
            // `Sequential` 経由の forward では到達しない
            // （`tensor-core::backend_ops::BackendOps::gemm_resident_rhs`
            // doc 参照）。CPU 参照実装のようにホスト側 epilogue のみで
            // 済ませるには resident `bias` を download する必要があり、
            // それは本メソッドが排除する対象の D2H そのものになるため、
            // フォールバックを設けず型付きエラーで拒否する。
            return Err(BackendError::InvalidArgument(
                "gemm_resident_rhs: k == 0 is unreachable via Linear::new (in_features == 0 is \
                 rejected at construction); a host epilogue fallback would require downloading \
                 the resident bias, defeating the zero-D2H contract this method exists for"
                    .to_string(),
            ));
        }
        if m == 0 || n == 0 {
            // 早期 return でも poison 状態は fail-closed に検査する
            // （Cursor Bugbot 指摘・PR #1064 追補: 空入力の早期 return は
            // driver へ触れないため、begin_driver_call の poison 検査を明示的に
            // 経由しないと poison 済み ordinal でも「空 step」相当が黙って
            // 成功してしまう）。世代も通常経路（`resident_generations`。
            // 本関数下部参照）と同じ `w`／`bias` の generation を渡す
            // （codex-review P1 指摘・PR #1064 追補: 空スライスのままだと
            // `invalidate` 後の旧世代 `w`／`bias` ビューがこの分岐だけ
            // `StaleDeviceGeneration` を経由せず成功してしまい、「旧世代は
            // 全て拒否する」という公開エラー契約を経路依存に破る）。
            let empty_shape_generations = [
                Some(w.buffer().generation()),
                bias.map(|b| b.buffer().generation()),
            ]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
            context_cache::begin_driver_call(self.ordinal, &empty_shape_generations)?;
            return Tensor::new(Vec::new(), &[m, n]).map_err(BackendError::ShapeMismatch);
        }

        let w_handle = w
            .buffer()
            .downcast_handle::<CudaBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let Some(w_full) = w_handle.slice.as_ref() else {
            return Err(BackendError::DeviceAllocationFailed(
                "gemm_resident_rhs: w buffer has numel > 0 but no device allocation".into(),
            ));
        };
        let w_view = w_full.slice(w.offset()..w.offset() + w.numel());
        let bias_handle = bias
            .map(|b| {
                b.buffer()
                    .downcast_handle::<CudaBufferHandle>()
                    .ok_or(BackendError::DeviceMismatch)
                    .map(|h| (h, b.offset(), b.numel()))
            })
            .transpose()?;
        let bias_view = match &bias_handle {
            Some((h, offset, numel)) => {
                let Some(full) = h.slice.as_ref() else {
                    return Err(BackendError::DeviceAllocationFailed(
                        "gemm_resident_rhs: bias buffer has numel > 0 but no device allocation"
                            .into(),
                    ));
                };
                Some(full.slice(*offset..*offset + *numel))
            }
            None => None,
        };

        // `w`（・`bias`）はデバイス常駐のまま渡す唯一の入力（`a` はこの
        // 呼び出し内で毎回アップロードし直すため世代を跨がない。イシュー
        // #1013 設計文書 §9 item 7）。
        let resident_generations = [
            Some(w.buffer().generation()),
            bias.map(|b| b.buffer().generation()),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
        // `device_handle_raw`（キャッシュミス時の `CudaDevice::new`）自体も
        // poison 検査・観測の対象に含める（codex-review P0 指摘・PR #1064
        // 追補・`ops.rs:147` 相当: `device_handle()` を `with_driver_call`
        // より前に呼ぶと、poison 済み ordinal でも拒否前に driver 初期化が
        // 走ってしまう）。
        let device = self.with_driver_call(
            &resident_generations,
            |e| BackendError::CudaUnavailable(e.to_string()),
            || self.device_handle_raw(),
        )?;
        let mem = CudaMemory::new(&device);
        let a_dev_buf = mem.upload(a)?;
        let a_handle = a_dev_buf
            .downcast_handle::<CudaBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let Some(a_slice) = a_handle.slice.as_ref() else {
            return Err(BackendError::DeviceAllocationFailed(
                "gemm_resident_rhs: a buffer has numel > 0 but no device allocation".into(),
            ));
        };

        let mut c_dev_buf = mem.alloc_zeroed(&[m, n])?;
        let c_handle = c_dev_buf
            .downcast_handle_mut::<CudaBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let Some(c_slice) = c_handle.slice.as_mut() else {
            return Err(BackendError::DeviceAllocationFailed(
                "gemm_resident_rhs: output buffer has numel > 0 but no device allocation".into(),
            ));
        };

        // キャッシュ構築（`cached_gemm`）自体の driver 呼び出し（NVRTC
        // コンパイル・モジュールロード）も同じ観測対象とする（Cursor
        // Bugbot 指摘・PR #1064 追補: cold-cache 構築中の sticky エラーが
        // 観測なしで消費され fail-open になっていた）。
        let gemm = self.with_driver_call(
            &resident_generations,
            |e| BackendError::CudaUnavailable(e.to_string()),
            || context_cache::cached_gemm(&device),
        )?;
        self.with_driver_call(
            &resident_generations,
            |e| BackendError::KernelLaunchFailed(e.to_string()),
            || {
                gemm.launch_tiled_bias_act_f32_resident(
                    a_slice,
                    &w_view,
                    bias_view.as_ref(),
                    false,
                    c_slice,
                    m as u32,
                    n as u32,
                    k as u32,
                )
            },
        )?;

        mem.download(&c_dev_buf)
    }

    /// デバイス常駐 `w` のまま `c = w @ b` を計算する（イシュー #1022・
    /// #1023「R3」）。`Op::LinearResident` の VJP が `d_input^T = w @ g^T`
    /// を計算するために使う。[`Self::gemm_resident_rhs`] と同じく `w` は
    /// download せず [`crate::gemm::CudaGemm::launch_tiled_f32_resident`]
    /// （`CudaView` 部分ビュー起動）へそのまま渡す。
    fn gemm_resident_lhs(
        &self,
        w: DeviceBufferView<'_>,
        b: &Tensor<f32>,
    ) -> Result<Tensor<f32>, BackendError> {
        if w.device() != Device::Cuda(self.ordinal) {
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
            // 早期 return でも poison 状態は fail-closed に検査する
            // （Cursor Bugbot 指摘・PR #1064 追補: 空入力の早期 return は
            // driver へ触れないため、begin_driver_call の poison 検査を明示的に
            // 経由しないと poison 済み ordinal でも「空 step」相当が黙って
            // 成功してしまう）。世代も通常経路と同じ `w` の generation を
            // 渡す（codex-review P1 指摘・PR #1064 追補: 空スライスの
            // ままだと `invalidate` 後の旧世代 `w` ビューがこの分岐だけ
            // `StaleDeviceGeneration` を経由せず成功してしまう）。
            context_cache::begin_driver_call(self.ordinal, &[w.buffer().generation()])?;
            return Tensor::new(Vec::new(), &[p, r]).map_err(BackendError::ShapeMismatch);
        }
        if q == 0 {
            // `w` の縮約次元（`out_features`。`Linear::new` は
            // `out_features == 0` を許容する）が 0 の場合、GEMM の数学的
            // 定義どおり結果は全 0（`gemm`／`run_tiled_f32` の `k == 0`
            // 契約と同じ）。GPU 起動を回避してホスト側で直接構築する。
            // 早期 return でも poison 状態は fail-closed に検査する
            // （Cursor Bugbot 指摘・PR #1064 追補: 空入力の早期 return は
            // driver へ触れないため、begin_driver_call の poison 検査を
            // 明示的に経由しないと poison 済み ordinal でも「空 step」
            // 相当が黙って成功してしまう）。世代も通常経路と同じ `w` の
            // generation を渡す（codex-review P1 指摘・PR #1064 追補）。
            context_cache::begin_driver_call(self.ordinal, &[w.buffer().generation()])?;
            return Tensor::from_shape_fill(&[p, r], |_| 0.0).map_err(BackendError::ShapeMismatch);
        }

        let w_handle = w
            .buffer()
            .downcast_handle::<CudaBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let Some(w_full) = w_handle.slice.as_ref() else {
            return Err(BackendError::DeviceAllocationFailed(
                "gemm_resident_lhs: w buffer has numel > 0 but no device allocation".into(),
            ));
        };
        let w_view = w_full.slice(w.offset()..w.offset() + w.numel());

        // `w` のみがデバイス常駐入力（`b` はこの呼び出し内で毎回
        // アップロードし直すため世代を跨がない。イシュー #1013 設計文書
        // §9 item 7）。`device_handle_raw`（キャッシュミス時の
        // `CudaDevice::new`）自体も poison 検査・観測の対象に含める
        // （codex-review P0 指摘・PR #1064 追補・`ops.rs:147` 相当）。
        let device = self.with_driver_call(
            &[w.buffer().generation()],
            |e| BackendError::CudaUnavailable(e.to_string()),
            || self.device_handle_raw(),
        )?;
        let mem = CudaMemory::new(&device);
        let b_dev_buf = mem.upload(b)?;
        let b_handle = b_dev_buf
            .downcast_handle::<CudaBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let Some(b_slice) = b_handle.slice.as_ref() else {
            return Err(BackendError::DeviceAllocationFailed(
                "gemm_resident_lhs: b buffer has numel > 0 but no device allocation".into(),
            ));
        };

        let mut c_dev_buf = mem.alloc_zeroed(&[p, r])?;
        let c_handle = c_dev_buf
            .downcast_handle_mut::<CudaBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let Some(c_slice) = c_handle.slice.as_mut() else {
            return Err(BackendError::DeviceAllocationFailed(
                "gemm_resident_lhs: output buffer has numel > 0 but no device allocation".into(),
            ));
        };

        // キャッシュ構築（`cached_gemm`）自体の driver 呼び出し（NVRTC
        // コンパイル・モジュールロード）も同じ観測対象とする（Cursor
        // Bugbot 指摘・PR #1064 追補: cold-cache 構築中の sticky エラーが
        // 観測なしで消費され fail-open になっていた）。
        let gemm = self.with_driver_call(
            &[w.buffer().generation()],
            |e| BackendError::CudaUnavailable(e.to_string()),
            || context_cache::cached_gemm(&device),
        )?;
        self.with_driver_call(
            &[w.buffer().generation()],
            |e| BackendError::KernelLaunchFailed(e.to_string()),
            || {
                gemm.launch_tiled_f32_resident(
                    &w_view, b_slice, c_slice, p as u32, r as u32, q as u32,
                )
            },
        )?;

        mem.download(&c_dev_buf)
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

    /// [`fandhe_ai_tensor_core::BackendOps::mse_loss`] の CUDA 実装
    /// （イシュー #1045）。`Self::sum`／`Self::max`（汎用 reduction）とは
    /// 独立した専用融合カーネル（`crate::mse::CudaMse`）へのディスパッチ
    /// であり、`Self::sum` の未実装状態とは無関係にここで実装する
    /// （`backend_ops.rs::BackendOps::mse_loss` doc の設計判断参照:
    /// `Op::MseLoss` は解析形の専用ノードであり融合 IR〈`run_fused`〉を
    /// 経由しない）。
    ///
    /// `reduction` に応じた `factor`（`Mean` は `1.0/n`、`Sum` は `1.0`）は
    /// ここで計算してカーネルへ渡す（`CudaMse::run_mse_loss_f32` は
    /// reduction 種別を知らない。`kernels_mse.rs` 冒頭コメント参照）。
    /// 未知 `MseReduction` variant は `backend-cpu::ops::CpuBackendOps::
    /// mse_loss` と同じく `Unsupported` として拒否する。
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

        let mse = self.with_driver_call(
            &[],
            |e| BackendError::CudaUnavailable(e.to_string()),
            || {
                let device = self.device_handle_raw()?;
                context_cache::cached_mse(&device)
            },
        )?;
        let value = self.with_driver_call(
            &[],
            |e| BackendError::KernelLaunchFailed(e.to_string()),
            || mse.run_mse_loss_f32(pred_slice, target_slice, factor),
        )?;
        Tensor::new(vec![value], &[]).map_err(BackendError::ShapeMismatch)
    }

    /// [`fandhe_ai_tensor_core::BackendOps::mse_loss_backward`] の CUDA
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

        let mse = self.with_driver_call(
            &[],
            |e| BackendError::CudaUnavailable(e.to_string()),
            || {
                let device = self.device_handle_raw()?;
                context_cache::cached_mse(&device)
            },
        )?;
        let out = self.with_driver_call(
            &[],
            |e| BackendError::KernelLaunchFailed(e.to_string()),
            || mse.run_mse_backward_f32(pred_slice, target_slice, scale),
        )?;
        Tensor::new(out, pred.shape()).map_err(BackendError::ShapeMismatch)
    }

    /// [`fandhe_ai_tensor_core::BackendOps::release_cached_device_memory`] の CUDA 実装
    /// （イシュー #1020・REQ-14）。`gemm.rs`／`elementwise.rs`／`softmax.rs`
    /// が `context_cache::cached_allocator` 経由で共有する
    /// `(ordinal, 既定 stream)` 単位のサイズクラス別プールを即座に解放する。
    ///
    /// エラー文字列にはフェーズ識別子（`crate::pool::ReleasePhase`。
    /// "pre-free sync"／"handle release"／"post-free sync"／"driver trim"）
    /// を含める（新しい `BackendError` variant は追加しない設計判断。
    /// `docs/backend-cuda-pool-allocator-decision.md` 参照）。
    fn release_cached_device_memory(&self) -> Result<(), BackendError> {
        // `device_handle_raw`（キャッシュミス時の `CudaDevice::new`）自体も
        // poison 検査・観測の対象に含める（codex-review P0 指摘・PR #1064
        // 追補・`ops.rs:147` 相当）。
        let device = self.with_driver_call(
            &[],
            |e| BackendError::CudaUnavailable(e.to_string()),
            || self.device_handle_raw(),
        )?;

        // `cached_allocator`／`release_cached`（pre/post-free の
        // `stream.synchronize()`・driver トリム）自体も poison 検査・観測
        // の対象に含める（codex-review P0 指摘・PR #1064 追補: これらは
        // 先行する非同期カーネルの sticky エラーを最初に観測しうる同期点
        // であり、`ReleaseCacheError` へ変換されるだけで poison 化されない
        // と、次の演算が Active のまま通ってしまう fail-open 経路になる）。
        // `release_cached` の戻り値型 `ReleaseCacheError` は `CudaError`
        // そのものではないため `with_driver_call` の一律インターフェース
        // には載せず、`begin_driver_call`／`observe_cuda_error_ref` を
        // 直接呼んで分類・poison 化のみを行い、`ReleaseCacheError` が
        // 運ぶフェーズ識別子はそのまま `BackendError` のメッセージへ残す
        // （`pool.rs::ReleaseCacheError` ドキュメンテーションコメント
        // 参照）。
        let token = context_cache::begin_driver_call(self.ordinal, &[])?;
        let allocator = match context_cache::observe_cuda_result(
            self.ordinal,
            &token,
            context_cache::cached_allocator(&device),
        ) {
            Ok(allocator) => allocator,
            Err(e) => return Err(BackendError::CudaUnavailable(e.to_string())),
        };
        match allocator.release_cached() {
            Ok(_freed_bytes) => Ok(()),
            Err(e) => {
                context_cache::observe_cuda_error_ref(self.ordinal, &token, &e.detail);
                Err(BackendError::DeviceAllocationFailed(format!(
                    "release_cached_device_memory: {e}"
                )))
            }
        }
    }

    /// [`fandhe_ai_tensor_core::BackendOps::device_memory_pool_stats`] の CUDA 実装。
    /// driver 不在等で `device_handle()` が失敗した場合は `None` を返す
    /// （`memory_ops` と同じ fail-safe 契約。統計取得の失敗で呼び出し元の
    /// エラー処理を複雑化させない）。
    fn device_memory_pool_stats(&self) -> Option<fandhe_ai_tensor_core::PoolStats> {
        let device = self.device_handle().ok()?;
        let allocator = context_cache::cached_allocator(&device).ok()?;
        Some(allocator.stats())
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

    /// フラグはプロセスグローバル（`crate::precision`）のため、他の
    /// テストとの競合を避けて直列化・原状復帰する RAII ガード
    /// （`precision.rs::tests::FlagGuard` と同型。イシュー #1042）。
    struct Tf32FlagGuard {
        _lock: std::sync::MutexGuard<'static, ()>,
        original: bool,
    }

    impl Tf32FlagGuard {
        fn acquire() -> Self {
            // `precision.rs::tests::FlagGuard` と単一ロックを共有する
            // （codex-review P2・Cursor Bugbot Medium 指摘。別々の
            // `static LOCK` を持つと直列化が効かず `TF32_GEMM_ENABLED`
            // を巡るレースが起こりうる。PR #1091）。
            let lock = crate::precision::test_support::tf32_flag_test_lock()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let original = crate::precision::tf32_gemm_enabled();
            Self {
                _lock: lock,
                original,
            }
        }
    }

    impl Drop for Tf32FlagGuard {
        fn drop(&mut self) {
            crate::precision::set_tf32_gemm_enabled(self.original);
        }
    }

    /// 環境適応（CUDA 非搭載環境でも実行可能。実機なら本体まで検証）:
    /// `crate::precision::tf32_gemm_enabled()` が既定 `false`（OFF）の
    /// 場合、`gemm` が TF32 経路（[`crate::gemm::TF32_OPTIN_GEMM_LAUNCH_COUNT`]）
    /// へ一切到達しないことを検証する（イシュー #1042 実装計画 §2.1
    /// 「既定は OFF（FP32 厳密）」契約。`gemm_bias_act_fused_path_
    /// increments_launch_counter_env_adaptive` と同じ分岐パターン）。
    #[test]
    fn gemm_stays_on_fp32_path_when_tf32_optin_flag_is_disabled_env_adaptive() {
        use fandhe_ai_tensor_core::Tensor;

        let _guard = Tf32FlagGuard::acquire();
        crate::precision::set_tf32_gemm_enabled(false);

        let cuda = CudaBackendOps::new(0);
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("valid tensor");
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).expect("valid tensor");

        let before = crate::gemm::TF32_OPTIN_GEMM_LAUNCH_COUNT.with(|c| c.get());
        let _ = cuda.gemm(&a, &b);
        let after = crate::gemm::TF32_OPTIN_GEMM_LAUNCH_COUNT.with(|c| c.get());
        assert_eq!(
            before, after,
            "既定 OFF のはずが TF32 opt-in 経路のカウンタが増加した（フラグ OFF 時の \
             bit-exact 不変契約違反の疑い）: before={before}, after={after}"
        );
    }

    /// opt-in（`true`）時、CUDA 実機が利用可能で TF32 カーネルが使用可能な
    /// 環境では `gemm` が [`crate::gemm::CudaGemm::run_wmma_tf32`] 経路へ
    /// 実際にルーティングされる（[`crate::gemm::TF32_OPTIN_GEMM_LAUNCH_COUNT`]
    /// の増加で検証）ことを確認する。CUDA 非搭載環境・TF32 カーネル使用
    /// 不能環境（`CudaError::WmmaUnavailable` 由来の
    /// `BackendError::KernelLaunchFailed`）ではエラーの型のみ確認して
    /// 早期 return する（fail-closed 契約: FP32 への黙示フォールバックを
    /// しないことの裏返しとして、エラーはそのまま伝播される）。
    #[test]
    fn gemm_routes_to_tf32_path_when_optin_flag_is_enabled_env_adaptive() {
        use fandhe_ai_tensor_core::Tensor;

        let _guard = Tf32FlagGuard::acquire();
        crate::precision::set_tf32_gemm_enabled(true);

        let cuda = CudaBackendOps::new(0);
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("valid tensor");
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).expect("valid tensor");

        let before = crate::gemm::TF32_OPTIN_GEMM_LAUNCH_COUNT.with(|c| c.get());
        match cuda.gemm(&a, &b) {
            Ok(_) => {
                let after = crate::gemm::TF32_OPTIN_GEMM_LAUNCH_COUNT.with(|c| c.get());
                assert!(
                    after > before,
                    "opt-in 時に TF32 経路の起動カウンタが増加していない（既定 \
                     FP32 経路へ黙示フォールバックした疑い）: before={before}, after={after}"
                );
            }
            Err(BackendError::CudaUnavailable(msg)) => {
                assert!(!msg.is_empty(), "error detail message must not be empty");
            }
            Err(BackendError::KernelLaunchFailed(msg)) => {
                // TF32 カーネル使用不能環境（cc<8.0 等）の fail-closed 伝播。
                // FP32 への黙示フォールバックはしない契約（`crate::precision`
                // モジュール冒頭コメント参照）。
                assert!(!msg.is_empty(), "error detail message must not be empty");
            }
            Err(other) => panic!("unexpected error variant for tf32 opt-in gemm: {other}"),
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

    // ---------------------------------------------------------------
    // Cursor Bugbot 指摘（PR #1064 追補）の回帰テスト。
    //
    // `context_cache` のプロセスワイド static レジストリはテスト間で
    // 共有されるため、他所（`context_cache.rs::poison_state_tests` は
    // 10000 番台、実機依存テストは ordinal 0/1）と衝突しない専用 ordinal
    // を払い出す（`context_cache.rs::poison_state_tests::unique_ordinal`
    // と同方針）。
    // ---------------------------------------------------------------

    fn unique_test_ordinal() -> usize {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static NEXT: AtomicUsize = AtomicUsize::new(30_000);
        NEXT.fetch_add(1, Ordering::SeqCst)
    }

    fn sticky_driver_error() -> cudarc::driver::result::DriverError {
        cudarc::driver::result::DriverError(
            cudarc::driver::sys::CUresult::CUDA_ERROR_ILLEGAL_ADDRESS,
        )
    }

    /// [`CudaBackendOps::with_driver_call`] の回帰テスト（Cursor Bugbot
    /// Medium 指摘・`ops.rs:100` 相当）: cold-cache 構築（`cached_gemm`
    /// 等）を表す最初のクロージャが sticky な driver エラーを返した場合、
    /// 修正前は構築呼び出しが `with_driver_call` の外側で素通しに実行され
    /// 観測されないため ordinal が poison されず、以降の呼び出しも
    /// fail-open のまま成功し続けた。修正後は構築呼び出し自体も
    /// `with_driver_call` で包むため、直後の（実行フェーズ相当の）
    /// 呼び出しが `DeviceContextPoisoned` で拒否されることを確認する。
    #[test]
    fn with_driver_call_poisons_ordinal_when_construction_closure_returns_sticky_error() {
        let ordinal = unique_test_ordinal();
        let cuda = CudaBackendOps::new(ordinal);

        // 注意（CI 実機構成差で 1 度落ちた教訓）: `map` クロージャで
        // `CudaError::Driver(e)` を `e.to_string()`／`{e:?}` で整形すると、
        // `cudarc::driver::result::DriverError` の `Debug` 実装が
        // `cuGetErrorString`（driver API 経由。`culib()` の遅延ロードを
        // 要求する）を呼ぶ（cudarc-0.19.8 `src/driver/result.rs`）。
        // CUDA toolkit 非搭載環境（本テストの前提。CI `build-no-cuda-
        // toolkit` ジョブ）ではこのロードが `panic_no_lib_found` で
        // panic するため、poison 化ロジック自体とは無関係にテストが
        // 落ちる。本テストは poison 検査の副作用のみを検証すればよく
        // 実際の driver エラー詳細文字列は不要なため、`map` では
        // `CudaError` を整形せず固定メッセージにする。
        let construction_result: Result<(), BackendError> = cuda.with_driver_call(
            &[],
            |_e| BackendError::CudaUnavailable("simulated sticky driver error (test)".to_string()),
            || Err(CudaError::Driver(sticky_driver_error())),
        );
        assert!(
            construction_result.is_err(),
            "構築失敗はそのまま Err として伝播するはず"
        );

        let run_result: Result<(), BackendError> = cuda.with_driver_call(
            &[],
            |_e| {
                BackendError::KernelLaunchFailed(
                    "unexpected: should be rejected before this                 map is reached"
                        .to_string(),
                )
            },
            || Ok(()),
        );
        assert!(
            matches!(run_result, Err(BackendError::DeviceContextPoisoned(_))),
            "構築呼び出しで観測された sticky エラーにより ordinal は poison され、             以降の呼び出しは fail-closed に拒否されるはず: {run_result:?}"
        );
    }

    /// テスト専用の最小 `BufferHandle`（`tensor-core::backend_ops::
    /// EmptyHandle` と同型。データの実体は持たず、`DeviceBuffer` を
    /// 構築するためだけの空ハンドル）。
    #[derive(Debug)]
    struct EmptyHandle;

    impl fandhe_ai_tensor_core::buffer::BufferHandle for EmptyHandle {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
            self
        }
    }

    /// [`CudaBackendOps::gemm_resident_rhs`] の回帰テスト（Cursor Bugbot
    /// Low 指摘・`ops.rs:468` 相当の一般化）: `n == 0`（空入力）の早期
    /// return 分岐は、修正前は `device_handle()`／driver 呼び出しの手前で
    /// 無条件に `Ok` を返していたため、poison 済み ordinal でも「空
    /// 出力」が黙って成功していた。修正後は早期 return の直前で
    /// `begin_driver_call` の poison 検査を通すため、この分岐は driver
    /// 呼び出し（＝実機）を要求せずに poison 状態のみで再現・検証できる
    /// （`device_handle()` より手前で拒否されるため CUDA 非搭載環境でも
    /// 実行可能）。
    #[test]
    fn gemm_resident_rhs_rejects_on_poisoned_ordinal_even_via_trivial_empty_shape_early_return() {
        use fandhe_ai_tensor_core::buffer::DeviceBuffer;

        let ordinal = unique_test_ordinal();

        // context_cache の poison 状態機械を直接操作して poison 化する
        // （`context_cache::poison_state_tests` と同じ手法）。
        let token = context_cache::begin_driver_call(ordinal, &[]).expect("begin succeeds");
        let _ = context_cache::observe_cuda_result::<()>(
            ordinal,
            &token,
            Err(CudaError::Driver(sticky_driver_error())),
        );
        drop(token);

        let cuda = CudaBackendOps::new(ordinal);
        // `a` は `[m, k] = [1, 1]`（k != 0 のため k==0 分岐は通らない）。
        let a = Tensor::new(vec![1.0f32], &[1, 1]).expect("valid tensor");
        // `w` は `[k, n] = [1, 0]`（n == 0 のため対象の早期 return 分岐へ
        // 到達する）。ビュー構築は `buffer.numel() >= offset + numel`
        // のみを検査し、`numel == 0` のビューは任意のバッキングバッファに
        // 対して構築できる。
        let w_buffer = DeviceBuffer::new(Device::Cuda(ordinal), vec![1], Box::new(EmptyHandle));
        let w = DeviceBufferView::new(&w_buffer, 0, &[1, 0]).expect("view construction succeeds");

        let cuda_result = cuda.gemm_resident_rhs(&a, w, None);
        assert!(
            matches!(cuda_result, Err(BackendError::DeviceContextPoisoned(_))),
            "poison 済み ordinal では n == 0 の早期 return 分岐も fail-closed に              拒否されるはず: {cuda_result:?}"
        );
    }

    /// [`CudaBackendOps::gemm_resident_rhs`] の回帰テスト（codex-review
    /// P1 指摘・`ops.rs:785` 相当・PR #1064 追補）: `m == 0 || n == 0`
    /// の早期 return 分岐は、修正前は `begin_driver_call` を空スライス
    /// （generation 検査なし）で呼んでいたため、`invalidate` 後の旧世代
    /// `w` ビューでもこの分岐だけ `StaleDeviceGeneration` を経由せず
    /// 成功してしまい、「旧世代のバッファは全て拒否する」という公開
    /// エラー契約を経路依存に破っていた。修正後は通常経路と同じ
    /// `w.buffer().generation()` を渡すため、旧世代スタンプ済みの `w`
    /// では空 shape でも `StaleDeviceGeneration` を返す（実機不要。
    /// `current_generation` は新規 ordinal で既定 `0` のため、`w` を
    /// 意図的にそれと異なる世代でスタンプするだけで再現できる）。
    #[test]
    fn gemm_resident_rhs_rejects_stale_generation_even_via_trivial_empty_shape_early_return() {
        use fandhe_ai_tensor_core::buffer::DeviceBuffer;

        let ordinal = unique_test_ordinal();
        assert_eq!(
            context_cache::current_generation(ordinal),
            0,
            "新規 ordinal の現行世代は既定 0 のはず"
        );

        let cuda = CudaBackendOps::new(ordinal);
        let a = Tensor::new(vec![1.0f32], &[1, 1]).expect("valid tensor");
        // `w` を現行世代（0）とは異なる世代（1）でスタンプする
        // （`invalidate` による回復後に取り残された旧世代バッファを
        // 模す）。`m == 0 || n == 0` の早期 return 分岐（`w` は
        // `[k, n] = [1, 0]` で n == 0）へ到達させる。
        let w_buffer = DeviceBuffer::new_with_generation(
            Device::Cuda(ordinal),
            vec![1],
            Box::new(EmptyHandle),
            1,
        );
        let w = DeviceBufferView::new(&w_buffer, 0, &[1, 0]).expect("view construction succeeds");

        let result = cuda.gemm_resident_rhs(&a, w, None);
        assert!(
            matches!(
                result,
                Err(BackendError::StaleDeviceGeneration {
                    resource_generation: 1,
                    current_generation: 0,
                    ..
                })
            ),
            "旧世代 w ビューは空 shape の早期 return でも StaleDeviceGeneration で              拒否されるはず: {result:?}"
        );
    }

    /// [`CudaBackendOps::gemm_resident_lhs`] の同種回帰テスト（`p == 0 ||
    /// r == 0` 分岐）。
    #[test]
    fn gemm_resident_lhs_rejects_stale_generation_even_via_trivial_empty_shape_early_return() {
        use fandhe_ai_tensor_core::buffer::DeviceBuffer;

        let ordinal = unique_test_ordinal();
        let cuda = CudaBackendOps::new(ordinal);
        // `b` は `[q, r] = [1, 0]`（r == 0 のため `p == 0 || r == 0` 分岐へ
        // 到達する）。`w` は `[p, q] = [1, 1]` で世代 1 にスタンプする。
        let b = Tensor::new(Vec::new(), &[1, 0]).expect("valid tensor");
        let w_buffer = DeviceBuffer::new_with_generation(
            Device::Cuda(ordinal),
            vec![1],
            Box::new(EmptyHandle),
            1,
        );
        let w = DeviceBufferView::new(&w_buffer, 0, &[1, 1]).expect("view construction succeeds");

        let result = cuda.gemm_resident_lhs(w, &b);
        assert!(
            matches!(
                result,
                Err(BackendError::StaleDeviceGeneration {
                    resource_generation: 1,
                    current_generation: 0,
                    ..
                })
            ),
            "旧世代 w ビューは空 shape の早期 return でも StaleDeviceGeneration で              拒否されるはず: {result:?}"
        );
    }

    /// [`CudaBackendOps::gemm_resident_lhs`] の `q == 0` 分岐の同種回帰
    /// テスト。
    #[test]
    fn gemm_resident_lhs_rejects_stale_generation_even_via_trivial_zero_contraction_dim_early_return()
     {
        use fandhe_ai_tensor_core::buffer::DeviceBuffer;

        let ordinal = unique_test_ordinal();
        let cuda = CudaBackendOps::new(ordinal);
        // `w` は `[p, q] = [1, 0]`（q == 0）・`b` は `[q, r] = [0, 1]`。
        let b = Tensor::new(Vec::new(), &[0, 1]).expect("valid tensor");
        let w_buffer = DeviceBuffer::new_with_generation(
            Device::Cuda(ordinal),
            vec![1],
            Box::new(EmptyHandle),
            1,
        );
        let w = DeviceBufferView::new(&w_buffer, 0, &[1, 0]).expect("view construction succeeds");

        let result = cuda.gemm_resident_lhs(w, &b);
        assert!(
            matches!(
                result,
                Err(BackendError::StaleDeviceGeneration {
                    resource_generation: 1,
                    current_generation: 0,
                    ..
                })
            ),
            "旧世代 w ビューは q == 0 の早期 return でも StaleDeviceGeneration で              拒否されるはず: {result:?}"
        );
    }

    /// [`CudaBackendOps::gemm`] の回帰テスト（codex-review P0 指摘・
    /// `ops.rs:147` 相当。PR #1064 追補）: 修正前は `device_handle()`
    /// （キャッシュミス時に `CudaDevice::new` を呼び実際に driver を
    /// 操作する）が `with_driver_call`（`begin_driver_call` による poison
    /// 検査を含む）より前に呼ばれており、poison 済み ordinal でも拒否
    /// される前に driver 初期化が試みられ、その失敗も観測されなかった。
    /// 修正後は `device_handle_raw()` を `with_driver_call` のクロージャ
    /// 内部（poison 検査の後）へ移したため、poison 済み ordinal では
    /// `device_handle_raw()` 自体が一切呼ばれず、`begin_driver_call` の
    /// 拒否がそのまま返る。これは CUDA 非搭載環境でも検証できる: もし
    /// 修正が入っていなければ、この環境では `device_handle_raw()` が
    /// `BackendError::CudaUnavailable` 相当（`CudaError::DriverUnavailable`
    /// 等）を先に返してしまい、`DeviceContextPoisoned` へは到達しない
    /// （＝ poison 状態が観測できないまま別のエラーにすり替わる）。
    #[test]
    fn gemm_rejects_on_poisoned_ordinal_before_device_handle_is_attempted() {
        let ordinal = unique_test_ordinal();

        let token = context_cache::begin_driver_call(ordinal, &[]).expect("begin succeeds");
        let _ = context_cache::observe_cuda_result::<()>(
            ordinal,
            &token,
            Err(CudaError::Driver(sticky_driver_error())),
        );
        drop(token);

        let cuda = CudaBackendOps::new(ordinal);
        let a = Tensor::new(vec![1.0, 2.0], &[1, 2]).expect("valid tensor");
        let b = Tensor::new(vec![1.0, 2.0], &[2, 1]).expect("valid tensor");

        let result = cuda.gemm(&a, &b);
        assert!(
            matches!(result, Err(BackendError::DeviceContextPoisoned(_))),
            "poison 済み ordinal では device_handle_raw() が試行される前に              begin_driver_call が拒否するはず（CUDA 非搭載環境でも              CudaUnavailable にすり替わらないことを確認する）: {result:?}"
        );
    }

    /// イシュー #1014（設計文書 §8 T3b: 呼び出し側の拒否経路）: 上記
    /// `gemm_rejects_on_poisoned_ordinal_before_device_handle_is_attempted`・
    /// `gemm_resident_rhs_rejects_on_poisoned_ordinal_even_via_trivial_empty_shape_early_return`
    /// が既にカバーする `gemm`／`gemm_resident_rhs` を除く、残る公開演算
    /// エントリ（`add`〈`mul` は同じ `elementwise_binary` 経路を通るため
    /// 併せて代表する。`relu`〈`exp`／`tanh` は同じ `elementwise_unary`
    /// 経路を通るため併せて代表する〉は下記
    /// `relu_rejects_on_poisoned_ordinal_before_device_handle_is_attempted`
    /// で個別に検証する（codex-review 指摘・PR #1067。`elementwise_binary`
    /// と `elementwise_unary` はディスパッチ関数自体が分かれているため
    /// `add` 側の検証だけでは `elementwise_unary` 経路を通らない）〉・
    /// `sgd_step_device`・`gemm_resident_lhs`・
    /// `release_cached_device_memory`）が、poison 済み ordinal では実処理
    /// （driver 呼び出し）へ一切入らず `BackendError::DeviceContextPoisoned`
    /// を即座に返すことを確認する（GPU 不要・CI 常時実行）。
    ///
    /// 実 CUDA fault 注入は行わず（`.claude/rules/coding-rust.md`
    /// カーネル境界検査の規約に反するため）、`context_cache` の poison
    /// 状態機械を直接セットする方式を採る（設計文書 §8 T3b が明示的に
    /// 許容する代替経路）。`MemoryOps::upload`／`download` は同一の
    /// `with_driver_call` ゲート（`memory.rs` の `CudaMemory::
    /// with_driver_call`）を通るが、そちらは `context_cache::
    /// poison_state_tests`（GPU 非依存モック）で既に等価な検証がある
    /// ため、ordinal 0/1（実機テストが使う共有 ordinal）を汚染してまで
    /// ここで重複検証しない（イシュー #1014 実装計画 §3 方針 2 の判断）。
    fn poison_ordinal(ordinal: usize) {
        let token = context_cache::begin_driver_call(ordinal, &[]).expect("begin succeeds");
        let _ = context_cache::observe_cuda_result::<()>(
            ordinal,
            &token,
            Err(CudaError::Driver(sticky_driver_error())),
        );
        drop(token);
    }

    #[test]
    fn add_rejects_on_poisoned_ordinal_before_device_handle_is_attempted() {
        let ordinal = unique_test_ordinal();
        poison_ordinal(ordinal);

        let cuda = CudaBackendOps::new(ordinal);
        let a = Tensor::new(vec![1.0, 2.0], &[1, 2]).expect("valid tensor");
        let b = Tensor::new(vec![1.0, 2.0], &[1, 2]).expect("valid tensor");

        let result = cuda.add(&a, &b);
        assert!(
            matches!(result, Err(BackendError::DeviceContextPoisoned(_))),
            "poison 済み ordinal では add（elementwise_binary 経由。mul も同一経路）は             device_handle_raw() が試行される前に拒否されるはず: {result:?}"
        );
    }

    #[test]
    fn relu_rejects_on_poisoned_ordinal_before_device_handle_is_attempted() {
        // codex-review 指摘（PR #1067）: 上記 `add_rejects_on_poisoned_
        // ordinal_before_device_handle_is_attempted` は `elementwise_binary`
        // 経路のみを通り、`relu`／`exp`／`tanh` が使う `elementwise_unary`
        // 経路は未検証だった。`elementwise_binary`／`elementwise_unary` は
        // ともに `with_driver_call` を経由する同一のゲート構造だが、
        // ディスパッチ関数自体が分かれているため実際に両方を通しておく。
        let ordinal = unique_test_ordinal();
        poison_ordinal(ordinal);

        let cuda = CudaBackendOps::new(ordinal);
        let a = Tensor::new(vec![1.0, -2.0], &[1, 2]).expect("valid tensor");

        let result = cuda.relu(&a);
        assert!(
            matches!(result, Err(BackendError::DeviceContextPoisoned(_))),
            "poison 済み ordinal では relu（elementwise_unary 経由。exp／tanh も             同一経路）は device_handle_raw() が試行される前に拒否されるはず:             {result:?}"
        );
    }

    #[test]
    fn mse_loss_rejects_on_poisoned_ordinal_before_device_handle_is_attempted() {
        // イシュー #1045: `mse_loss`／`mse_loss_backward` は `elementwise_
        // binary`／`elementwise_unary` とは別のディスパッチ関数
        // （`context_cache::cached_mse` 経由）のため、`with_driver_call`
        // ゲートが正しく結線されていることを個別に確認する
        // （`relu_rejects_on_poisoned_ordinal...` のコメントと同じ理由）。
        let ordinal = unique_test_ordinal();
        poison_ordinal(ordinal);

        let cuda = CudaBackendOps::new(ordinal);
        let pred = Tensor::new(vec![1.0, 2.0], &[1, 2]).expect("valid tensor");
        let target = Tensor::new(vec![0.0, 0.0], &[1, 2]).expect("valid tensor");

        let forward = cuda.mse_loss(&pred, &target, MseReduction::Mean);
        assert!(
            matches!(forward, Err(BackendError::DeviceContextPoisoned(_))),
            "poison 済み ordinal では mse_loss は device_handle_raw() が試行される前に \
             拒否されるはず: {forward:?}"
        );

        let backward = cuda.mse_loss_backward(&pred, &target, 1.0);
        assert!(
            matches!(backward, Err(BackendError::DeviceContextPoisoned(_))),
            "poison 済み ordinal では mse_loss_backward は device_handle_raw() が試行される \
             前に拒否されるはず: {backward:?}"
        );
    }

    #[test]
    fn sgd_step_device_rejects_on_poisoned_ordinal_before_device_handle_is_attempted() {
        use fandhe_ai_tensor_core::SgdStepConfig;
        use fandhe_ai_tensor_core::buffer::DeviceBuffer;

        let ordinal = unique_test_ordinal();
        poison_ordinal(ordinal);

        let cuda = CudaBackendOps::new(ordinal);
        let mut param =
            DeviceBuffer::<f32>::new(Device::Cuda(ordinal), vec![4], Box::new(EmptyHandle));
        let grad = DeviceBuffer::<f32>::new(Device::Cuda(ordinal), vec![4], Box::new(EmptyHandle));
        let config = SgdStepConfig {
            lr: 0.1,
            momentum: 0.0,
            dampening: 0.0,
            weight_decay: 0.0,
            nesterov: false,
            is_first_step: true,
        };

        // momentum == 0.0 のため velocity は不要（`use_momentum` 分岐を
        // 通らない。`ops.rs::sgd_step_device` 参照）。poison 検査
        // （`cached_sgd` 構築の `with_driver_call`）は device/shape の
        // 事前検証の後・実際のバッファ downcast より前に走る。
        let result = cuda.sgd_step_device(&mut param, &grad, None, &config);
        assert!(
            matches!(result, Err(BackendError::DeviceContextPoisoned(_))),
            "poison 済み ordinal では sgd_step_device は cached_sgd 構築より前に             拒否されるはず: {result:?}"
        );
    }

    #[test]
    fn gemm_resident_lhs_rejects_on_poisoned_ordinal_before_device_handle_is_attempted() {
        use fandhe_ai_tensor_core::buffer::DeviceBuffer;

        let ordinal = unique_test_ordinal();
        poison_ordinal(ordinal);

        let cuda = CudaBackendOps::new(ordinal);
        // `w` は `[p, q] = [1, 0]`（q == 0）の早期 return 分岐（`begin_driver_call`
        // による poison 検査を経由する）へ到達させる。`p`／`r` が非ゼロの
        // 一般経路は poison 検査より先に `w.buffer().downcast_handle::
        // <CudaBufferHandle>()` を呼ぶため、テスト専用の `EmptyHandle`
        // （`CudaBufferHandle` ではない）を渡すと `DeviceContextPoisoned`
        // ではなく `DeviceMismatch` にすり替わってしまう（`gemm_resident_lhs`
        // の実装順序どおり）。早期 return 分岐を使うことでこの問題を回避
        // する（`gemm_resident_lhs_rejects_stale_generation_even_via_trivial_
        // zero_contraction_dim_early_return` と同じ手法）。
        let w_buffer = DeviceBuffer::new(Device::Cuda(ordinal), vec![1], Box::new(EmptyHandle));
        let w = DeviceBufferView::new(&w_buffer, 0, &[1, 0]).expect("view construction succeeds");
        let b = Tensor::new(Vec::new(), &[0, 1]).expect("valid tensor");

        let result = cuda.gemm_resident_lhs(w, &b);
        assert!(
            matches!(result, Err(BackendError::DeviceContextPoisoned(_))),
            "poison 済み ordinal では gemm_resident_lhs は q == 0 の早期 return 分岐でも             拒否されるはず: {result:?}"
        );
    }

    #[test]
    fn release_cached_device_memory_rejects_on_poisoned_ordinal_before_device_handle_is_attempted()
    {
        let ordinal = unique_test_ordinal();
        poison_ordinal(ordinal);

        let cuda = CudaBackendOps::new(ordinal);
        let result = cuda.release_cached_device_memory();
        assert!(
            matches!(result, Err(BackendError::DeviceContextPoisoned(_))),
            "poison 済み ordinal では release_cached_device_memory は             device_handle_raw() が試行される前に拒否されるはず: {result:?}"
        );
    }
}
