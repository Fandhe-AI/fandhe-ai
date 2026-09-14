//! 累積和／累積積（`torch.cumsum`／`torch.cumprod` 相当。イシュー
//! #1740・親イシュー #1731）の起動 API（NVRTC コンパイル・保持・実行）。
//!
//! `gather_scatter.rs::CudaGatherScatter` と同じ構成方針を踏襲する:
//! [`CudaScan::new`] が `CudaDevice` から 2 カーネル（`kernels_scan.rs`）
//! を NVRTC コンパイルして保持し、以降は [`CudaScan::run_cumsum_f32`]／
//! [`CudaScan::run_cumprod_f32`] へホスト側スライスを渡すだけで GPU
//! 実行できる。`ops.rs::CudaBackendOps::cumsum`／`cumprod` から
//! `BackendOps` の実装として呼ばれる。
//!
//! **数値契約・並列化不可の理由**は `kernels_scan.rs` モジュール doc を
//! 正とする。呼び出し元 `ops.rs` が `reduce_out_shape` で `dim` を
//! 再検査してから稠密化（`contiguous()`）した `outer`／`axis_len`／
//! `inner` を渡す契約のため、本モジュールは `lanes = outer * inner` の
//! カーネル引数 `int` 範囲検査（[`CudaError::ScanSizeLimitExceeded`]。
//! `ops.rs` はこの variant のみを `Unsupported` へ写像しホスト
//! フォールバックへ委ねる）のみを独立に行う。

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg};

use crate::context_cache;
use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::kernels_scan::{self, SCAN_BLOCK_DIM};
use crate::nvrtc::compile_ptx;
use crate::pool::CudaAllocator;

/// `value` が `i32::MAX` に収まることを検証する（カーネル引数 `int` は
/// C の 32bit 符号付き整数のため。`gather_scatter.rs::
/// validate_i32_bound` と同じ理由の複製——モジュールごとに専用の
/// 検証関数を持つ既存方針を踏襲する）。値が上限を超える場合は
/// [`CudaError::ScanSizeLimitExceeded`]（バックエンド固有サイズ上限の
/// 超過。`ops.rs` が `Unsupported` へ写像しホストフォールバックへ
/// 委ねる。`InvalidScanShape`〈内部契約違反〉とは区別する）を返す。
fn validate_i32_bound(value: usize, name: &str) -> Result<i32, CudaError> {
    i32::try_from(value).map_err(|_| CudaError::ScanSizeLimitExceeded {
        detail: format!("scan dimension must fit in i32 (kernel argument type): {name}={value}"),
    })
}

/// scan（累積和／累積積）2 カーネルのコンパイル済みハンドルを保持する。
pub struct CudaScan {
    stream: Arc<CudaStream>,
    /// `gather_scatter.rs::CudaGatherScatter::ordinal` と同じ役割
    /// （`Self::with_driver_call` が `context_cache::with_driver_call`
    /// を呼ぶ際のキー）。
    ordinal: usize,
    allocator: Arc<CudaAllocator>,
    cumsum_f32: CudaFunction,
    cumprod_f32: CudaFunction,
}

/// [`CudaScan::run_cumsum_f32`]／[`run_cumprod_f32`] の演算種別
/// （カーネル選択を 1 箇所へ集約するための内部列挙）。
enum ScanKind {
    Sum,
    Prod,
}

impl CudaScan {
    /// `device` 上で cumsum／cumprod 2 カーネルを NVRTC コンパイルし
    /// 保持するハンドルを構築する（`gather_scatter.rs::
    /// CudaGatherScatter::new` と同一手順）。
    pub fn new(device: &CudaDevice) -> Result<Self, CudaError> {
        let arch = device.arch();

        macro_rules! compile_and_load {
            ($src:expr, $name:literal) => {{
                let ptx = compile_ptx($src, arch)?;
                device.context().load_module(ptx)?.load_function($name)?
            }};
        }

        let cumsum_f32 = compile_and_load!(kernels_scan::CUMSUM_F32, "cumsum_f32");
        let cumprod_f32 = compile_and_load!(kernels_scan::CUMPROD_F32, "cumprod_f32");

        let allocator = context_cache::cached_allocator(device)?;

        Ok(Self {
            stream: device.stream().clone(),
            ordinal: device.ordinal(),
            allocator,
            cumsum_f32,
            cumprod_f32,
        })
    }

    /// `CudaScan` の driver 呼び出しを CUDA Graph capture 排他へ参加
    /// させる共通ヘルパー（`gather_scatter.rs::CudaGatherScatter::
    /// with_driver_call` と同じ設計）。
    fn with_driver_call<T>(
        &self,
        f: impl FnOnce() -> Result<T, CudaError>,
    ) -> Result<T, CudaError> {
        context_cache::with_driver_call(self.ordinal, f)
    }

    /// `torch.cumsum` 相当（`kernels_scan.rs` モジュール doc 参照）。
    /// `x` は `outer * axis_len * inner` 要素の稠密（contiguous）
    /// スライス。
    pub fn run_cumsum_f32(
        &self,
        x: &[f32],
        outer: usize,
        axis_len: usize,
        inner: usize,
    ) -> Result<Vec<f32>, CudaError> {
        self.run_scan(ScanKind::Sum, x, outer, axis_len, inner)
    }

    /// `torch.cumprod` 相当。[`Self::run_cumsum_f32`] と同じ lane 分解・
    /// 検証だが `cumprod_f32` カーネルを起動する。
    pub fn run_cumprod_f32(
        &self,
        x: &[f32],
        outer: usize,
        axis_len: usize,
        inner: usize,
    ) -> Result<Vec<f32>, CudaError> {
        self.run_scan(ScanKind::Prod, x, outer, axis_len, inner)
    }

    /// [`Self::run_cumsum_f32`]／[`run_cumprod_f32`] の共通本体（カーネル
    /// のみ `kind` で分岐する）。
    fn run_scan(
        &self,
        kind: ScanKind,
        x: &[f32],
        outer: usize,
        axis_len: usize,
        inner: usize,
    ) -> Result<Vec<f32>, CudaError> {
        let lanes = outer
            .checked_mul(inner)
            .ok_or(CudaError::InvalidScanShape {
                detail: "run_scan: outer * inner overflowed usize".to_string(),
            })?;
        let numel = lanes
            .checked_mul(axis_len)
            .ok_or(CudaError::InvalidScanShape {
                detail: "run_scan: lanes * axis_len overflowed usize".to_string(),
            })?;
        if x.len() != numel {
            return Err(CudaError::InvalidScanShape {
                detail: format!("run_scan: x.len()={} does not match numel={numel}", x.len()),
            });
        }
        if numel == 0 {
            return Ok(Vec::new());
        }

        let lanes_i = validate_i32_bound(lanes, "lanes")?;
        let axis_len_i = validate_i32_bound(axis_len, "axis_len")?;
        let inner_i = validate_i32_bound(inner, "inner")?;
        validate_i32_bound(numel, "numel")?;

        let func = match kind {
            ScanKind::Sum => &self.cumsum_f32,
            ScanKind::Prod => &self.cumprod_f32,
        };

        self.with_driver_call(|| {
            let x_dev = self.stream.clone_htod(x)?;
            let mut out_dev = self.allocator.alloc_uninit_f32(numel)?;

            let cfg = LaunchConfig {
                grid_dim: ((lanes as u32).div_ceil(SCAN_BLOCK_DIM), 1, 1),
                block_dim: (SCAN_BLOCK_DIM, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: `x_dev` は `numel` 要素の H2D 済みバッファ（直上の
            // 長さ検証でホストスライスと一致することを確認済み）。
            // `out_dev` は `numel` 要素確保済みでカーネルは
            // `gid < lanes`（REQ-8）を維持したまま `axis_len` 回の
            // ループで各 lane の `axis_len` 個の出力要素をちょうど
            // 1 回ずつ書く（`kernels_scan.rs::CUMSUM_F32`／
            // `CUMPROD_F32` 参照）。範囲外アクセスはない。
            unsafe {
                self.stream
                    .launch_builder(func)
                    .arg(&x_dev)
                    .arg(&mut out_dev.as_view_mut())
                    .arg(&lanes_i)
                    .arg(&axis_len_i)
                    .arg(&inner_i)
                    .launch(cfg)?;
            }
            crate::memory::readback(&self.stream, &out_dev.as_view())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_i32_bound_rejects_exceeding_i32_max() {
        let err = validate_i32_bound(i32::MAX as usize + 1, "lanes").unwrap_err();
        assert!(matches!(err, CudaError::ScanSizeLimitExceeded { .. }));
    }

    #[test]
    fn validate_i32_bound_accepts_i32_max() {
        assert!(validate_i32_bound(i32::MAX as usize, "lanes").is_ok());
    }
}
