//! dtype 変換（`fandhe_ai_tensor_core::cast::CastOps`。イシュー #1751・
//! 親 #1613・依存 #1750）の起動 API（NVRTC コンパイル・保持・実行）と
//! `CastOps for CudaBackendOps` 実装。
//!
//! `unique.rs::CudaUnique` と同じ構成方針を踏襲する: [`CudaCast::new`]
//! が `CudaDevice` から 8 カーネル（`kernels_cast.rs`）を NVRTC
//! コンパイルして保持し、以降は `run_*` へホスト側スライスを渡すだけで
//! GPU 実行できる。`impl CastOps for CudaBackendOps` は CPU 実装
//! （`backend-cpu::cast`）と同じ配置規約（「impl は dtype ファイル・
//! accessor は `ops.rs`」）で本ファイルへ置く。
//!
//! # 数値契約
//!
//! 算術を含まない変換のため **bit 完全一致**（NaN のみクラス一致）。
//! カーネル記述規則の詳細は `kernels_cast.rs` モジュール doc を正とする。
//!
//! # shape 検証・上限検査（呼び出し元 `impl CastOps` の共通前処理）
//!
//! `ops::checked_shape_numel`（PR #1795 系の `[usize::MAX, 2, 0]` 型
//! 中間積オーバーフロー対策。`unique`／`gather` と同じ適用方針）を
//! `.contiguous()`／`host_slice()` より前に適用し、`numel == 0` は
//! GPU 起動なしで空テンソルを返す。カーネル引数 `int numel` に収まらない
//! 要素数（`> i32::MAX`）は [`BackendError::Unsupported`] を返しホスト
//! フォールバックへ委ねる（shape 自体は妥当なので `ShapeMismatch` に
//! しない。`unique` の `UniqueSizeLimitExceeded` → `Unsupported` 写像と
//! 同じ判断）。

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg};

use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{CastOps, Tensor};

use crate::context_cache;
use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::kernels_cast::{self, CAST_BLOCK_DIM};
use crate::memory::readback;
use crate::nvrtc::compile_ptx;
use crate::ops::{CudaBackendOps, checked_shape_numel};

/// `numel` がカーネル引数 `int`（C の 32bit 符号付き整数）へ収まるかを
/// 検証する（`unique.rs::validate_i32_bound` と同じ理由の複製。
/// `CudaError` を経由させず直接 `BackendError::Unsupported` を返す点が
/// 異なる——本モジュールは `checked_shape_numel` 適用後のホスト側
/// `usize` を扱うだけで、それ以上の driver 呼び出しは行わないため
/// 専用エラー variant を新設する必要がない）。
fn checked_i32_numel(numel: usize) -> Result<i32, BackendError> {
    i32::try_from(numel).map_err(|_| {
        BackendError::Unsupported(format!(
            "cast: numel={numel} exceeds i32::MAX (CUDA kernel argument type); \
             falling back to host cast"
        ))
    })
}

/// 8 方向の cast カーネルのコンパイル済みハンドルを保持する。
pub struct CudaCast {
    stream: Arc<CudaStream>,
    /// `unique.rs::CudaUnique::ordinal` と同じ役割（`Self::
    /// with_driver_call` が `context_cache::with_driver_call` を呼ぶ際の
    /// キー）。
    ordinal: usize,
    cast_f32_to_f64: CudaFunction,
    cast_f32_to_i32: CudaFunction,
    cast_f32_to_i64: CudaFunction,
    cast_f32_to_bool: CudaFunction,
    cast_f64_to_f32: CudaFunction,
    cast_i32_to_f32: CudaFunction,
    cast_i64_to_f32: CudaFunction,
    cast_bool_to_f32: CudaFunction,
}

impl CudaCast {
    /// `device` 上で 8 カーネルすべてを NVRTC コンパイルし保持する
    /// ハンドルを構築する（`unique.rs::CudaUnique::new` と同一手順）。
    pub fn new(device: &CudaDevice) -> Result<Self, CudaError> {
        let arch = device.arch();
        let context = device.context();

        macro_rules! compile_and_load {
            ($src:expr, $name:literal) => {{
                let ptx = compile_ptx($src, arch)?;
                context.load_module(ptx)?.load_function($name)?
            }};
        }

        let cast_f32_to_f64 = compile_and_load!(kernels_cast::CAST_F32_TO_F64, "cast_f32_to_f64");
        let cast_f32_to_i32 = compile_and_load!(kernels_cast::CAST_F32_TO_I32, "cast_f32_to_i32");
        let cast_f32_to_i64 = compile_and_load!(kernels_cast::CAST_F32_TO_I64, "cast_f32_to_i64");
        let cast_f32_to_bool =
            compile_and_load!(kernels_cast::CAST_F32_TO_BOOL, "cast_f32_to_bool");
        let cast_f64_to_f32 = compile_and_load!(kernels_cast::CAST_F64_TO_F32, "cast_f64_to_f32");
        let cast_i32_to_f32 = compile_and_load!(kernels_cast::CAST_I32_TO_F32, "cast_i32_to_f32");
        let cast_i64_to_f32 = compile_and_load!(kernels_cast::CAST_I64_TO_F32, "cast_i64_to_f32");
        let cast_bool_to_f32 =
            compile_and_load!(kernels_cast::CAST_BOOL_TO_F32, "cast_bool_to_f32");

        Ok(Self {
            stream: device.stream().clone(),
            ordinal: device.ordinal(),
            cast_f32_to_f64,
            cast_f32_to_i32,
            cast_f32_to_i64,
            cast_f32_to_bool,
            cast_f64_to_f32,
            cast_i32_to_f32,
            cast_i64_to_f32,
            cast_bool_to_f32,
        })
    }

    /// `CudaCast` の driver 呼び出しを CUDA Graph capture 排他へ参加
    /// させる共通ヘルパー（`unique.rs::CudaUnique::with_driver_call` と
    /// 同じ設計）。
    fn with_driver_call<T>(
        &self,
        f: impl FnOnce() -> Result<T, CudaError>,
    ) -> Result<T, CudaError> {
        context_cache::with_driver_call(self.ordinal, f)
    }

    /// 1 入力 1 出力の cast カーネルを起動する共通骨格（8 方向すべてが
    /// 「H2D → 出力バッファ確保 → 起動 → D2H」の同一手続きを踏むため、
    /// カーネルハンドルと型パラメータのみを差し替える。`kernels_cast.rs`
    /// モジュール doc の REQ-8 境界検査はカーネル側で担保済み）。
    fn run<In, Out>(&self, func: &CudaFunction, input: &[In]) -> Result<Vec<Out>, CudaError>
    where
        In: cudarc::driver::DeviceRepr,
        Out: crate::memory::ReadbackSentinel + cudarc::driver::ValidAsZeroBits,
    {
        let numel = input.len();
        self.with_driver_call(|| {
            let input_dev = self.stream.clone_htod(input)?;
            let mut out_dev = self.stream.alloc_zeros::<Out>(numel)?;

            let numel_i = numel as i32;
            let cfg = LaunchConfig {
                grid_dim: ((numel as u32).div_ceil(CAST_BLOCK_DIM), 1, 1),
                block_dim: (CAST_BLOCK_DIM, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: `input_dev` は `numel` 要素確保済み（直上の
            // `clone_htod` の結果）。`out_dev` も `numel` 要素確保済み
            // （直上の `alloc_zeros` の結果）。カーネルは `idx < numel`
            // （REQ-8）を維持したまま各要素を 1 回だけ読み書きする
            // （`kernels_cast.rs` 参照）。呼び出し元がすでに `numel` を
            // `i32` の範囲内であることを検証済み（`checked_i32_numel`）。
            unsafe {
                self.stream
                    .launch_builder(func)
                    .arg(&input_dev)
                    .arg(&mut out_dev)
                    .arg(&numel_i)
                    .launch(cfg)?;
            }
            readback::<Out, _>(&self.stream, &out_dev)
        })
    }

    /// f32→f64（完全表現）。
    pub fn run_f32_to_f64(&self, x: &[f32]) -> Result<Vec<f64>, CudaError> {
        self.run(&self.cast_f32_to_f64, x)
    }

    /// f32→i32（ゼロ方向切り捨て・範囲外は飽和・NaN→0）。
    pub fn run_f32_to_i32(&self, x: &[f32]) -> Result<Vec<i32>, CudaError> {
        self.run(&self.cast_f32_to_i32, x)
    }

    /// f32→i64（同上）。
    pub fn run_f32_to_i64(&self, x: &[f32]) -> Result<Vec<i64>, CudaError> {
        self.run(&self.cast_f32_to_i64, x)
    }

    /// f32→bool（`unsigned char` 0／1 で readback してから
    /// `!= 0` で `bool` へ実体化する。ホスト側で生バイトを
    /// transmute しない）。
    pub fn run_f32_to_bool(&self, x: &[f32]) -> Result<Vec<bool>, CudaError> {
        let raw = self.run::<f32, u8>(&self.cast_f32_to_bool, x)?;
        Ok(raw.into_iter().map(|v| v != 0).collect())
    }

    /// f64→f32（最近接偶数丸め・範囲超過は ±inf）。
    pub fn run_f64_to_f32(&self, x: &[f64]) -> Result<Vec<f32>, CudaError> {
        self.run(&self.cast_f64_to_f32, x)
    }

    /// i32→f32（最近接偶数丸め。`|v| > 2^24` は非可逆）。
    pub fn run_i32_to_f32(&self, x: &[i32]) -> Result<Vec<f32>, CudaError> {
        self.run(&self.cast_i32_to_f32, x)
    }

    /// i64→f32（同上）。
    pub fn run_i64_to_f32(&self, x: &[i64]) -> Result<Vec<f32>, CudaError> {
        self.run(&self.cast_i64_to_f32, x)
    }

    /// bool→f32（`true→1.0`・`false→0.0`。ホスト側で `bool` を
    /// `unsigned char`〈0／1〉へ変換してから転送する）。
    pub fn run_bool_to_f32(&self, x: &[bool]) -> Result<Vec<f32>, CudaError> {
        let raw: Vec<u8> = x.iter().map(|&b| u8::from(b)).collect();
        self.run(&self.cast_bool_to_f32, &raw)
    }
}

/// `x` を稠密化し `f32` スライスとして取り出す（8 方向 cast 共通の
/// 前処理。`unique`／`gather` と同じ `checked_shape_numel` 適用方針。
/// `checked_shape_numel` 自体は要素数積のオーバーフローのみ検査する
/// ため、`Tensor::contiguous()`（内部で無検査の `numel()` 乗算を呼ぶ）
/// より前に必ず呼ぶ）。空テンソル（`numel == 0`）の場合は `Ok(None)`
/// を返し、呼び出し元が GPU 起動なしで空 `Tensor` を構築する。
fn checked_contiguous<T: fandhe_ai_tensor_core::Element>(
    x: &Tensor<T>,
) -> Result<Option<Tensor<T>>, BackendError> {
    checked_shape_numel(x.shape()).map_err(BackendError::ShapeMismatch)?;
    if x.shape().iter().product::<usize>() == 0 {
        return Ok(None);
    }
    Ok(Some(x.contiguous()))
}

/// `CudaCast::run_*` のエラーを `BackendError` へ変換する。
/// `CudaError::InvalidShape`（内部契約違反。呼び出し元の事前検証を
/// 通過した入力からは実質到達しない防御的経路。`ops.rs::
/// checked_f32_bytes` と同種の判断）は `ShapeError::
/// ElementCountOverflow` へ、それ以外（driver 不在等）は既存
/// `crate::memory::map_cuda_error` へ委譲する（`map_unique_error` と
/// 同じ設計判断）。
fn map_cast_error(err: CudaError) -> BackendError {
    match err {
        CudaError::InvalidShape { .. } => {
            BackendError::ShapeMismatch(fandhe_ai_tensor_core::ShapeError::ElementCountOverflow)
        }
        other => crate::memory::map_cuda_error(other),
    }
}

/// `x`（cast 元。要素数はすでに `checked_i32_numel` で検証済み）から
/// `CudaCast` インスタンスを取得して `run` を呼び、結果を `Tensor<Out>`
/// へ包む共通骨格（8 方向の `impl CastOps` メソッドがこの関数へ委譲する。
/// `run_fn` は `CudaCast::run_*` のいずれかを束縛したクロージャ）。
fn dispatch_cast<In, Out>(
    ops: &CudaBackendOps,
    x: &Tensor<In>,
    run_fn: impl FnOnce(&CudaCast, &[In]) -> Result<Vec<Out>, CudaError>,
) -> Result<Tensor<Out>, BackendError>
where
    In: fandhe_ai_tensor_core::Element,
    Out: fandhe_ai_tensor_core::Element,
{
    let shape = x.shape().to_vec();
    let Some(x_owned) = checked_contiguous(x)? else {
        return Tensor::new(Vec::new(), &shape).map_err(BackendError::ShapeMismatch);
    };
    let x_slice = x_owned
        .as_slice()
        .ok_or_else(|| BackendError::KernelLaunchFailed("cast: input not contiguous".into()))?;
    checked_i32_numel(x_slice.len())?;

    let cast = ops.with_driver_call(
        &[],
        |e| BackendError::CudaUnavailable(e.to_string()),
        || {
            let device = ops.device_handle_raw()?;
            context_cache::cached_cast(&device)
        },
    )?;
    let out = ops.with_driver_call(&[], map_cast_error, || run_fn(&cast, x_slice))?;
    Tensor::new(out, &shape).map_err(BackendError::ShapeMismatch)
}

impl CastOps for CudaBackendOps {
    fn cast_f32_to_f64(&self, x: &Tensor<f32>) -> Result<Tensor<f64>, BackendError> {
        dispatch_cast(self, x, CudaCast::run_f32_to_f64)
    }

    fn cast_f32_to_i32(&self, x: &Tensor<f32>) -> Result<Tensor<i32>, BackendError> {
        dispatch_cast(self, x, CudaCast::run_f32_to_i32)
    }

    fn cast_f32_to_i64(&self, x: &Tensor<f32>) -> Result<Tensor<i64>, BackendError> {
        dispatch_cast(self, x, CudaCast::run_f32_to_i64)
    }

    fn cast_f32_to_bool(&self, x: &Tensor<f32>) -> Result<Tensor<bool>, BackendError> {
        dispatch_cast(self, x, CudaCast::run_f32_to_bool)
    }

    fn cast_f64_to_f32(&self, x: &Tensor<f64>) -> Result<Tensor<f32>, BackendError> {
        dispatch_cast(self, x, CudaCast::run_f64_to_f32)
    }

    fn cast_i32_to_f32(&self, x: &Tensor<i32>) -> Result<Tensor<f32>, BackendError> {
        dispatch_cast(self, x, CudaCast::run_i32_to_f32)
    }

    fn cast_i64_to_f32(&self, x: &Tensor<i64>) -> Result<Tensor<f32>, BackendError> {
        dispatch_cast(self, x, CudaCast::run_i64_to_f32)
    }

    fn cast_bool_to_f32(&self, x: &Tensor<bool>) -> Result<Tensor<f32>, BackendError> {
        dispatch_cast(self, x, CudaCast::run_bool_to_f32)
    }
}
