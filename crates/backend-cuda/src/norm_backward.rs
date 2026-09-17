//! RMSNorm／LayerNorm backward の起動 API（イシュー #1950。`rmsnorm.rs`
//! （#592・#596）・`layer_norm.rs`（#1596）と同じ構成方針: [`CudaNormBackward::
//! new`] が `kernels_norm_backward.rs` の 4 カーネルを一括 NVRTC
//! コンパイルして保持し、[`CudaNormBackward::run_rmsnorm_backward_f32`]／
//! [`CudaNormBackward::run_layer_norm_backward_f32`] へホスト側スライスを
//! 渡すだけで H2D → 起動 → D2H を内部で完結できる。
//!
//! `ops.rs::CudaBackendOps::rmsnorm_backward`／`layer_norm_backward`
//! （`fandhe_ai_tensor_core::BackendOps::rmsnorm_backward`／
//! `layer_norm_backward` の CUDA 実装）から直接呼ばれる。

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg};

use crate::context_cache;
use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::kernels_norm_backward::{
    LAYER_NORM_BWD_DWDB_F32, LAYER_NORM_BWD_DX_F32, RMSNORM_BWD_DW_NEW_F32, RMSNORM_BWD_DX_NEW_F32,
};
use crate::memory::readback;
use crate::nvrtc::compile_ptx;

/// dx カーネル（1 CTA = 1 warp = 1 行）の block 幅。`kernels_layer_norm.rs::
/// LAYER_NORM_BLOCK_DIM`（`layer_norm.rs`）と同じ 32（1 warp）固定。
const NORM_BWD_DX_BLOCK_DIM: u32 = 32;

/// dw／db カーネル（列方向 grid-stride）の block 幅。`rmsnorm.rs::
/// RMSNORM_BWD_DW_BLOCK_DIM` と同じ単純な 256 スレッド（reduction を
/// 伴わないため warp 数の制約はない）。
const NORM_BWD_DWDB_BLOCK_DIM: u32 = 256;

/// `rows`／`hidden` の対を 1 引数へまとめる（`clippy::too_many_arguments`
/// 対策。`rmsnorm.rs::RmsNormShape` と同じ設計判断で
/// [`CudaNormBackward::run_layer_norm_backward_f32`] の引数に使う）。
#[derive(Debug, Clone, Copy)]
pub struct NormBackwardShape {
    /// 正規化対象の行数（`x`／`dy` の長さは `rows * hidden` に一致する
    /// 契約）。
    pub rows: usize,
    /// 1 行あたりの要素数（`w` の長さおよび `dw`／`db` の shape に一致する
    /// 契約）。
    pub hidden: usize,
}

/// [`CudaNormBackward::run_layer_norm_backward_f32`] の戻り値型エイリアス
/// （`clippy::too_many_arguments` 対策と同じ理由の `clippy::
/// type_complexity` 回避。`(dx, dw, db)` の意味は同メソッドの doc を
/// 正本とする）。
pub type LayerNormBackwardHostOutput = (Vec<f32>, Option<Vec<f32>>, Option<Vec<f32>>);

/// dw／db カーネルの grid 次元（`hidden` 列を単純に block 数で割った
/// 1 対 1 マッピング。`derive_persistent_grid_dw`〈rmsnorm.rs〉のような
/// persistent grid 最適化は本イシューのスコープ外——冒頭ドキュメント
/// コメント参照）。
fn dwdb_grid_dim(hidden: usize) -> u32 {
    let blocks = (hidden as u64).div_ceil(NORM_BWD_DWDB_BLOCK_DIM as u64);
    blocks.max(1).min(u32::MAX as u64) as u32
}

/// ホスト側検証（起動前・fail-closed。`rmsnorm.rs::
/// validate_rmsnorm_backward_launch`／`layer_norm.rs::
/// validate_layer_norm_launch` と同型）: `rows * hidden == x_len == dy_len`
/// （checked 乗算）・`w_len == Some(hidden)`（`weight` 指定時のみ）・
/// `rows`／`hidden`／`numel` が `i32::MAX`（カーネル引数 `int rows`／
/// `int hidden` 契約）に収まること・`eps` が有限かつ非負を検証する
/// （`.claude/rules/security.md` A03）。
pub(crate) fn validate_norm_backward_launch(
    rows: usize,
    hidden: usize,
    x_len: usize,
    dy_len: usize,
    w_len: Option<usize>,
    eps: f32,
) -> Result<(), CudaError> {
    if !eps.is_finite() || eps < 0.0 {
        return Err(CudaError::InvalidRmsNormShape {
            detail: format!("norm backward eps must be finite and non-negative: eps={eps}"),
        });
    }
    let numel = rows
        .checked_mul(hidden)
        .ok_or_else(|| CudaError::InvalidRmsNormShape {
            detail: format!(
                "norm backward rows*hidden overflowed usize: rows={rows}, hidden={hidden}"
            ),
        })?;
    if numel != x_len {
        return Err(CudaError::InvalidRmsNormShape {
            detail: format!(
                "norm backward x length mismatch: rows*hidden={numel}, x.len()={x_len}"
            ),
        });
    }
    if numel != dy_len {
        return Err(CudaError::InvalidRmsNormShape {
            detail: format!(
                "norm backward dy length mismatch: rows*hidden={numel}, dy.len()={dy_len}"
            ),
        });
    }
    if let Some(wl) = w_len
        && wl != hidden
    {
        return Err(CudaError::InvalidRmsNormShape {
            detail: format!("norm backward w length mismatch: hidden={hidden}, w.len()={wl}"),
        });
    }
    if rows > i32::MAX as usize || hidden > i32::MAX as usize || numel > i32::MAX as usize {
        return Err(CudaError::InvalidRmsNormShape {
            detail: format!(
                "norm backward dims must fit in i32 (kernel argument type): rows={rows}, \
                 hidden={hidden}, numel={numel}"
            ),
        });
    }
    Ok(())
}

/// RMSNorm／LayerNorm backward の 4 カーネルのコンパイル済みハンドルを
/// 保持する（`CudaRmsNorm`／`CudaLayerNorm` と同じ構成方針）。
pub struct CudaNormBackward {
    stream: Arc<CudaStream>,
    /// 構築元 `CudaDevice` の ordinal（`Self::with_driver_call` が
    /// `context_cache::with_driver_call` を呼ぶ際のキーとして使う）。
    ordinal: usize,
    rmsnorm_dx: CudaFunction,
    rmsnorm_dw: CudaFunction,
    layer_norm_dx: CudaFunction,
    layer_norm_dwdb: CudaFunction,
}

impl CudaNormBackward {
    /// `device` 上で 4 カーネルを NVRTC コンパイルし保持するハンドルを
    /// 構築する（`CudaLayerNorm::new` と同一手順）。
    pub fn new(device: &CudaDevice) -> Result<Self, CudaError> {
        let arch = device.arch();

        let rmsnorm_dx_ptx = compile_ptx(RMSNORM_BWD_DX_NEW_F32, arch)?;
        let rmsnorm_dx = device
            .context()
            .load_module(rmsnorm_dx_ptx)?
            .load_function("rmsnorm_bwd_dx_new_f32")?;

        let rmsnorm_dw_ptx = compile_ptx(RMSNORM_BWD_DW_NEW_F32, arch)?;
        let rmsnorm_dw = device
            .context()
            .load_module(rmsnorm_dw_ptx)?
            .load_function("rmsnorm_bwd_dw_new_f32")?;

        let layer_norm_dx_ptx = compile_ptx(LAYER_NORM_BWD_DX_F32, arch)?;
        let layer_norm_dx = device
            .context()
            .load_module(layer_norm_dx_ptx)?
            .load_function("layer_norm_bwd_dx_f32")?;

        let layer_norm_dwdb_ptx = compile_ptx(LAYER_NORM_BWD_DWDB_F32, arch)?;
        let layer_norm_dwdb = device
            .context()
            .load_module(layer_norm_dwdb_ptx)?
            .load_function("layer_norm_bwd_dwdb_f32")?;

        Ok(Self {
            stream: device.stream().clone(),
            ordinal: device.ordinal(),
            rmsnorm_dx,
            rmsnorm_dw,
            layer_norm_dx,
            layer_norm_dwdb,
        })
    }

    /// `CudaNormBackward` の driver 呼び出しを CUDA Graph capture 排他へ
    /// 参加させる共通ヘルパー（`CudaRmsNorm::with_driver_call` と同じ
    /// 設計）。
    fn with_driver_call<T>(
        &self,
        f: impl FnOnce() -> Result<T, CudaError>,
    ) -> Result<T, CudaError> {
        context_cache::with_driver_call(self.ordinal, f)
    }

    /// [`fandhe_ai_tensor_core::BackendOps::rmsnorm_backward`] の CUDA
    /// 実装本体。`(dx, dw)` を返す（`dw` は `w.is_some()` のときのみ
    /// `Some`）。`rows == 0 || hidden == 0` は空の `dx`・（`w` ありなら）
    /// ゼロ埋めの `dw` を返す早期 return（driver 非接触。`CudaRmsNorm::
    /// run_rmsnorm_f32_inner` と同じ 0 要素契約）。
    pub fn run_rmsnorm_backward_f32(
        &self,
        x: &[f32],
        w: Option<&[f32]>,
        dy: &[f32],
        eps: f32,
        rows: usize,
        hidden: usize,
    ) -> Result<(Vec<f32>, Option<Vec<f32>>), CudaError> {
        validate_norm_backward_launch(rows, hidden, x.len(), dy.len(), w.map(|s| s.len()), eps)?;

        if rows == 0 || hidden == 0 {
            let dw = w.map(|w_slice| vec![0.0f32; w_slice.len()]);
            return Ok((Vec::new(), dw));
        }

        let rows_i = rows as i32;
        let hidden_i = hidden as i32;
        let has_weight = i32::from(w.is_some());

        self.with_driver_call(|| {
            let x_dev = self.stream.clone_htod(x)?;
            let dy_dev = self.stream.clone_htod(dy)?;
            // `w` が `None` の場合もカーネル引数としてポインタは必要だが、
            // predicated load による境界外読み出しを防ぐため `hidden`
            // 要素のダミーバッファを渡す（`layer_norm.rs::
            // run_layer_norm_f32` と同じ理由。1 要素ダミーは不可）。
            let w_dev = match w {
                Some(w_slice) => self.stream.clone_htod(w_slice)?,
                None => self.stream.alloc_zeros::<f32>(hidden)?,
            };
            let mut dx_dev = self.stream.alloc_zeros::<f32>(x.len())?;
            let mut rstd_dev = self.stream.alloc_zeros::<f64>(rows)?;

            let dx_cfg = LaunchConfig {
                grid_dim: (rows_i as u32, 1, 1),
                block_dim: (NORM_BWD_DX_BLOCK_DIM, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: `x_dev`／`dy_dev` は `rows*hidden` 要素の H2D 済み
            // デバイスバッファ、`w_dev` は `has_weight` が真のときのみ
            // 実データ（偽のときは `hidden` 要素ゼロ初期化ダミーで
            // `has_weight == 0` によりデリファレンスされない）、`dx_dev`
            // は `x.len()` 要素確保済みでカーネルが行ごとに全要素を書く、
            // `rstd_dev` は `rows` 要素確保済みでカーネルが `lane==0` の
            // ときのみ `row` の範囲内へ書く。grid 次元は `rows`
            // （1 CTA = 1 行）で `kernels_norm_backward.rs` 側の
            // `if (row >= rows) return;` ガードと合わせて OOB は起きない。
            unsafe {
                self.stream
                    .launch_builder(&self.rmsnorm_dx)
                    .arg(&x_dev)
                    .arg(&w_dev)
                    .arg(&dy_dev)
                    .arg(&mut dx_dev)
                    .arg(&mut rstd_dev)
                    .arg(&rows_i)
                    .arg(&hidden_i)
                    .arg(&eps)
                    .arg(&has_weight)
                    .launch(dx_cfg)?;
            }

            let mut dw_dev = if w.is_some() {
                Some(self.stream.alloc_zeros::<f32>(hidden)?)
            } else {
                None
            };
            if let Some(dw_dev) = dw_dev.as_mut() {
                let grid = dwdb_grid_dim(hidden);
                let dw_cfg = LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (NORM_BWD_DWDB_BLOCK_DIM, 1, 1),
                    shared_mem_bytes: 0,
                };
                // SAFETY: `rstd_dev` は dx カーネルが全 `rows` 行分を
                // 書き終えている（同一ストリーム上の投入順で dx → dw の
                // 順序が保証される）。`dw_dev` は `hidden` 要素確保済みで
                // カーネルが `col < hidden` の範囲のみ書く
                // （`if (col >= hidden) return;` ガード）。
                unsafe {
                    self.stream
                        .launch_builder(&self.rmsnorm_dw)
                        .arg(&x_dev)
                        .arg(&dy_dev)
                        .arg(&rstd_dev)
                        .arg(dw_dev)
                        .arg(&rows_i)
                        .arg(&hidden_i)
                        .launch(dw_cfg)?;
                }
            }

            let dx = readback(&self.stream, &dx_dev)?;
            let dw = match dw_dev {
                Some(dw_dev) => Some(readback(&self.stream, &dw_dev)?),
                None => None,
            };
            Ok((dx, dw))
        })
    }

    /// [`fandhe_ai_tensor_core::BackendOps::layer_norm_backward`] の CUDA
    /// 実装本体。`(dx, dw, db)` を返す（`dw` は `w.is_some()`・`db` は
    /// `has_bias` のときのみ `Some`）。`rows == 0 || hidden == 0` は
    /// [`Self::run_rmsnorm_backward_f32`] と同じ 0 要素契約。
    ///
    /// `rows`／`hidden` を [`NormBackwardShape`] へまとめる
    /// （`clippy::too_many_arguments` 対策。`rmsnorm.rs::RmsNormShape`
    /// と同じ設計判断）。
    pub fn run_layer_norm_backward_f32(
        &self,
        x: &[f32],
        w: Option<&[f32]>,
        has_bias: bool,
        dy: &[f32],
        eps: f32,
        shape: NormBackwardShape,
    ) -> Result<LayerNormBackwardHostOutput, CudaError> {
        let NormBackwardShape { rows, hidden } = shape;
        validate_norm_backward_launch(rows, hidden, x.len(), dy.len(), w.map(|s| s.len()), eps)?;

        if rows == 0 || hidden == 0 {
            let dw = w.map(|w_slice| vec![0.0f32; w_slice.len()]);
            let db = if has_bias {
                Some(vec![0.0f32; hidden])
            } else {
                None
            };
            return Ok((Vec::new(), dw, db));
        }

        let rows_i = rows as i32;
        let hidden_i = hidden as i32;
        let has_weight = i32::from(w.is_some());
        let has_bias_i = i32::from(has_bias);

        self.with_driver_call(|| {
            let x_dev = self.stream.clone_htod(x)?;
            let dy_dev = self.stream.clone_htod(dy)?;
            let w_dev = match w {
                Some(w_slice) => self.stream.clone_htod(w_slice)?,
                None => self.stream.alloc_zeros::<f32>(hidden)?,
            };
            let mut dx_dev = self.stream.alloc_zeros::<f32>(x.len())?;
            let mut mean_dev = self.stream.alloc_zeros::<f64>(rows)?;
            let mut rstd_dev = self.stream.alloc_zeros::<f64>(rows)?;

            let dx_cfg = LaunchConfig {
                grid_dim: (rows_i as u32, 1, 1),
                block_dim: (NORM_BWD_DX_BLOCK_DIM, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: `run_rmsnorm_backward_f32` と同じ根拠（`mean_dev`も
            // `rstd_dev` と同じ `rows` 要素・`lane==0` 限定書き込み）。
            unsafe {
                self.stream
                    .launch_builder(&self.layer_norm_dx)
                    .arg(&x_dev)
                    .arg(&w_dev)
                    .arg(&dy_dev)
                    .arg(&mut dx_dev)
                    .arg(&mut mean_dev)
                    .arg(&mut rstd_dev)
                    .arg(&rows_i)
                    .arg(&hidden_i)
                    .arg(&eps)
                    .arg(&has_weight)
                    .launch(dx_cfg)?;
            }

            let mut dw_dev = if w.is_some() {
                Some(self.stream.alloc_zeros::<f32>(hidden)?)
            } else {
                None
            };
            let mut db_dev = if has_bias {
                Some(self.stream.alloc_zeros::<f32>(hidden)?)
            } else {
                None
            };
            if dw_dev.is_some() || db_dev.is_some() {
                // `w`／`b` いずれか一方のみ必要な場合も、未使用側は
                // `has_weight`／`has_bias == 0` で書かれないダミー
                // （`hidden` 要素）を渡す（dx カーネルの `w` ダミーと同じ
                // predicated-write 対策——書き込みは条件分岐で完全に
                // 抑制されるため 1 要素でも安全だが、`alloc_zeros` の
                // コストは無視できるため hidden 要素で揃える）。
                let mut dw_scratch = match dw_dev.take() {
                    Some(buf) => buf,
                    None => self.stream.alloc_zeros::<f32>(hidden)?,
                };
                let mut db_scratch = match db_dev.take() {
                    Some(buf) => buf,
                    None => self.stream.alloc_zeros::<f32>(hidden)?,
                };
                let grid = dwdb_grid_dim(hidden);
                let dwdb_cfg = LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (NORM_BWD_DWDB_BLOCK_DIM, 1, 1),
                    shared_mem_bytes: 0,
                };
                // SAFETY: `mean_dev`／`rstd_dev` は dx カーネルが全 `rows`
                // 行分を書き終えている（同一ストリーム上の投入順）。
                // `dw_scratch`／`db_scratch` は `hidden` 要素確保済みで
                // `has_weight`／`has_bias` が真の側のみ `col < hidden` の
                // 範囲へ書く。
                unsafe {
                    self.stream
                        .launch_builder(&self.layer_norm_dwdb)
                        .arg(&x_dev)
                        .arg(&dy_dev)
                        .arg(&mean_dev)
                        .arg(&rstd_dev)
                        .arg(&mut dw_scratch)
                        .arg(&mut db_scratch)
                        .arg(&rows_i)
                        .arg(&hidden_i)
                        .arg(&has_weight)
                        .arg(&has_bias_i)
                        .launch(dwdb_cfg)?;
                }
                dw_dev = if w.is_some() { Some(dw_scratch) } else { None };
                db_dev = if has_bias { Some(db_scratch) } else { None };
            }

            let dx = readback(&self.stream, &dx_dev)?;
            let dw = match dw_dev {
                Some(dw_dev) => Some(readback(&self.stream, &dw_dev)?),
                None => None,
            };
            let db = match db_dev {
                Some(db_dev) => Some(readback(&self.stream, &db_dev)?),
                None => None,
            };
            Ok((dx, dw, db))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_norm_backward_launch_accepts_matching_dims() {
        assert!(validate_norm_backward_launch(3, 8, 24, 24, Some(8), 1e-5).is_ok());
        assert!(validate_norm_backward_launch(3, 8, 24, 24, None, 0.0).is_ok());
    }

    #[test]
    fn validate_norm_backward_launch_rejects_x_len_mismatch() {
        let err = validate_norm_backward_launch(3, 8, 23, 24, None, 1e-5).unwrap_err();
        assert!(matches!(err, CudaError::InvalidRmsNormShape { .. }));
    }

    #[test]
    fn validate_norm_backward_launch_rejects_dy_len_mismatch() {
        let err = validate_norm_backward_launch(3, 8, 24, 23, None, 1e-5).unwrap_err();
        assert!(matches!(err, CudaError::InvalidRmsNormShape { .. }));
    }

    #[test]
    fn validate_norm_backward_launch_rejects_w_len_mismatch() {
        let err = validate_norm_backward_launch(3, 8, 24, 24, Some(7), 1e-5).unwrap_err();
        assert!(matches!(err, CudaError::InvalidRmsNormShape { .. }));
    }

    #[test]
    fn validate_norm_backward_launch_rejects_non_finite_eps() {
        let err = validate_norm_backward_launch(3, 8, 24, 24, None, f32::NAN).unwrap_err();
        assert!(matches!(err, CudaError::InvalidRmsNormShape { .. }));
    }

    #[test]
    fn validate_norm_backward_launch_rejects_negative_eps() {
        let err = validate_norm_backward_launch(3, 8, 24, 24, None, -1e-5).unwrap_err();
        assert!(matches!(err, CudaError::InvalidRmsNormShape { .. }));
    }

    #[test]
    fn validate_norm_backward_launch_rejects_multiplication_overflow() {
        let err = validate_norm_backward_launch(usize::MAX, 2, 0, 0, None, 1e-5).unwrap_err();
        assert!(matches!(err, CudaError::InvalidRmsNormShape { .. }));
    }

    #[test]
    fn dwdb_grid_dim_covers_all_columns() {
        assert_eq!(dwdb_grid_dim(0), 1);
        assert_eq!(dwdb_grid_dim(1), 1);
        assert_eq!(dwdb_grid_dim(256), 1);
        assert_eq!(dwdb_grid_dim(257), 2);
    }
}
