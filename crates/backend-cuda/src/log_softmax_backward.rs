//! `log_softmax` backward カーネルの起動 API（イシュー #1949。
//! `layer_norm.rs`〈#1596〉と同じ構成方針を踏襲する CUDA 対応版）。
//!
//! [`CudaLogSoftmaxBackward::new`] が `kernels_log_softmax_backward.rs`
//! の単一カーネルを NVRTC コンパイルして保持し、
//! [`CudaLogSoftmaxBackward::run_log_softmax_backward_f32`] へホスト側
//! スライスを渡すだけで H2D → 起動 → D2H を内部で完結できる。
//! `grid_dim = rows`（1 CTA = 1 行）の単純な起動とする
//! （`layer_norm.rs` と同水準）。
//!
//! `crates/autodiff/src/grad.rs::vjp` の `Op::LogSoftmax` 分岐（`ops.rs::
//! CudaBackendOps::log_softmax_backward` 経由）から呼ばれる。

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg};

use crate::context_cache;
use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::kernels_log_softmax_backward::LOG_SOFTMAX_BACKWARD_F32;
use crate::memory::readback;
use crate::nvrtc::compile_ptx;

/// カーネル起動時の block 幅（32 スレッド = 1 warp 固定。
/// `kernels_log_softmax_backward.rs` の「1 CTA = 1 warp」設計と一致
/// させる）。
const LOG_SOFTMAX_BACKWARD_BLOCK_DIM: u32 = 32;

/// 起動前 fail-closed 検証: `rows * cols == y_len == g_len`（checked
/// 乗算）・`rows`／`cols` それぞれが `i32::MAX`（カーネル引数
/// `int rows`／`int cols` 契約）に収まること。
///
/// `validate_softmax_launch`（`softmax.rs`）は `rows*cols`（`numel`）
/// 自体も `i32::MAX` 以下であることを要求するが、これは softmax
/// フォワードカーネルが `numel` を index 計算に使わないためだけの
/// 制約である。本カーネル（`kernels_log_softmax_backward.rs`）は
/// 行内オフセットを `long long row_base = (long long)row * (long
/// long)cols` として `long long` で計算するため `rows*cols` が
/// `i32::MAX` を超えても正しく動作する。`validate_softmax_launch` を
/// そのまま再利用すると `rows`／`cols` が個別に `i32` へ収まる大きな
/// vocab バッチ（例: `rows` は小さいが `cols` が大きく `numel` のみ
/// `i32::MAX` 超）を不要に拒否し、`grad::vjp` の CUDA 経路が
/// `KernelLaunchFailed` を返してホストフォールバック
/// （`log_softmax_vjp_along`）へ委譲する前に失敗してしまう
/// （Cursor Bugbot 指摘・PR #1994）。そのため `numel` の上限検査は
/// 行わず、カーネル引数として実際に `int` へキャストする
/// `rows`／`cols` のみを個別に検証する。
pub(crate) fn validate_log_softmax_backward_launch(
    rows: usize,
    cols: usize,
    y_len: usize,
    g_len: usize,
) -> Result<(), CudaError> {
    let numel = rows
        .checked_mul(cols)
        .ok_or_else(|| CudaError::InvalidSoftmaxShape {
            detail: format!(
                "log_softmax_backward rows*cols overflowed usize: rows={rows}, cols={cols}"
            ),
        })?;
    if numel != y_len {
        return Err(CudaError::InvalidSoftmaxShape {
            detail: format!(
                "log_softmax_backward y length mismatch: rows*cols={numel}, y.len()={y_len}"
            ),
        });
    }
    if g_len != y_len {
        return Err(CudaError::InvalidSoftmaxShape {
            detail: format!(
                "log_softmax_backward: y/g length mismatch: y.len()={y_len}, g.len()={g_len}"
            ),
        });
    }
    if rows > i32::MAX as usize || cols > i32::MAX as usize {
        return Err(CudaError::InvalidSoftmaxShape {
            detail: format!(
                "log_softmax_backward rows/cols must each fit in i32 (kernel argument type): \
                 rows={rows}, cols={cols}"
            ),
        });
    }
    Ok(())
}

/// `log_softmax` backward カーネルのコンパイル済みハンドルを保持する。
pub struct CudaLogSoftmaxBackward {
    stream: Arc<CudaStream>,
    /// 構築元 `CudaDevice` の ordinal（`layer_norm.rs::CudaLayerNorm::
    /// ordinal` と同じ役割。`Self::with_driver_call` が
    /// `context_cache::with_driver_call` を呼ぶ際のキーとして使う）。
    ordinal: usize,
    func: CudaFunction,
}

impl CudaLogSoftmaxBackward {
    /// `device` 上で `log_softmax` backward カーネルを NVRTC コンパイルし
    /// 保持するハンドルを構築する（`layer_norm.rs::CudaLayerNorm::new`
    /// と同一手順）。
    pub fn new(device: &CudaDevice) -> Result<Self, CudaError> {
        let arch = device.arch();
        let ptx = compile_ptx(LOG_SOFTMAX_BACKWARD_F32, arch)?;
        let func = device
            .context()
            .load_module(ptx)?
            .load_function("log_softmax_backward_f32")?;

        Ok(Self {
            stream: device.stream().clone(),
            ordinal: device.ordinal(),
            func,
        })
    }

    /// `CudaLogSoftmaxBackward` の driver 呼び出し（H2D 転送・カーネル
    /// 起動・D2H readback）を CUDA Graph capture 排他へ参加させる共通
    /// ヘルパー（`layer_norm.rs::CudaLayerNorm::with_driver_call` と同じ
    /// 設計）。
    fn with_driver_call<T>(
        &self,
        f: impl FnOnce() -> Result<T, CudaError>,
    ) -> Result<T, CudaError> {
        context_cache::with_driver_call(self.ordinal, f)
    }

    /// `log_softmax` backward（`dx = g − exp(y)·Σ_dim(g)`）を実行する。
    ///
    /// `y`（forward 記録値）・`g`（upstream 勾配）は `[rows, cols]` の
    /// 行優先 1 次元化済みバッファで同一 shape。`rows == 0 || cols == 0`
    /// は空結果の早期 return（`layer_norm.rs::run_layer_norm_f32` と
    /// 同じ 0 要素契約は持たないが `mse.rs` 系と同様の設計）。
    pub fn run_log_softmax_backward_f32(
        &self,
        y: &[f32],
        g: &[f32],
        rows: usize,
        cols: usize,
    ) -> Result<Vec<f32>, CudaError> {
        validate_log_softmax_backward_launch(rows, cols, y.len(), g.len())?;

        if rows == 0 || cols == 0 {
            return Ok(Vec::new());
        }

        let rows_i = rows as i32;
        let cols_i = cols as i32;

        self.with_driver_call(|| {
            let y_dev = self.stream.clone_htod(y)?;
            let g_dev = self.stream.clone_htod(g)?;
            let mut dx_dev = self.stream.alloc_zeros::<f32>(y.len())?;

            let cfg = LaunchConfig {
                grid_dim: (rows_i as u32, 1, 1),
                block_dim: (LOG_SOFTMAX_BACKWARD_BLOCK_DIM, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: `y_dev`／`g_dev` は `rows*cols` 要素の H2D 済み
            // デバイスバッファ（`validate_log_softmax_backward_launch`
            // で長さ一致検証済み）、`dx_dev` は同数要素確保済みで
            // カーネルが行ごとに全要素を書く
            // （`kernels_log_softmax_backward.rs` 参照）。grid 次元は
            // `rows`（1 CTA = 1 行）で `kernels_log_softmax_backward.rs`
            // 側の `if (row >= rows) return;` ガードと合わせて OOB は
            // 起きない。
            unsafe {
                self.stream
                    .launch_builder(&self.func)
                    .arg(&y_dev)
                    .arg(&g_dev)
                    .arg(&mut dx_dev)
                    .arg(&rows_i)
                    .arg(&cols_i)
                    .launch(cfg)?;
            }

            readback(&self.stream, &dx_dev)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_log_softmax_backward_launch_accepts_matching_dims() {
        assert!(validate_log_softmax_backward_launch(3, 8, 24, 24).is_ok());
    }

    #[test]
    fn validate_log_softmax_backward_launch_rejects_y_len_mismatch() {
        let err = validate_log_softmax_backward_launch(3, 8, 23, 24).unwrap_err();
        assert!(matches!(err, CudaError::InvalidSoftmaxShape { .. }));
    }

    #[test]
    fn validate_log_softmax_backward_launch_rejects_g_len_mismatch() {
        let err = validate_log_softmax_backward_launch(3, 8, 24, 23).unwrap_err();
        assert!(matches!(err, CudaError::InvalidSoftmaxShape { .. }));
    }

    #[test]
    fn validate_log_softmax_backward_launch_rejects_dims_over_i32_max() {
        // `rows`（カーネル引数 `int rows`）自体が `i32::MAX` を超える
        // 場合は拒否する（`rows*cols` の上限とは独立の検査）。
        let rows = (i32::MAX as usize) + 1;
        let err = validate_log_softmax_backward_launch(rows, 1, rows, rows).unwrap_err();
        assert!(matches!(err, CudaError::InvalidSoftmaxShape { .. }));
    }

    #[test]
    fn validate_log_softmax_backward_launch_accepts_numel_over_i32_max_when_dims_fit() {
        // `rows`／`cols` は個別に `i32::MAX` に収まるが `rows*cols`
        // （numel）は `i32::MAX` を超える形状。本カーネルは行内
        // オフセットを `long long` で計算するため numel 自体の上限は
        // 不要（Cursor Bugbot 指摘・PR #1994）。
        let rows: usize = 100_000;
        let cols: usize = 100_000;
        let numel = rows * cols;
        assert!(numel > i32::MAX as usize);
        assert!(validate_log_softmax_backward_launch(rows, cols, numel, numel).is_ok());
    }
}
