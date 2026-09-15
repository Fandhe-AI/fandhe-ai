//! BatchNorm1d／2d 順伝播カーネルの起動 API（イシュー #1735・親
//! #1608。`layer_norm.rs`〈#1596〉と同じ構成方針を踏襲する CUDA 対応
//! 版）。
//!
//! [`CudaBatchNorm::new`] が `kernels_batch_norm.rs` の 2 カーネル
//! （train／infer）を NVRTC コンパイルして保持し、
//! [`CudaBatchNorm::run_batch_norm_train_f32`]／
//! [`CudaBatchNorm::run_batch_norm_infer_f32`] へホスト側スライスを
//! 渡すだけで H2D → 起動 → D2H を内部で完結できる。
//!
//! `ops.rs::CudaBackendOps::batch_norm_train`／`batch_norm_infer`
//! （`BackendOps::batch_norm_train`／`batch_norm_infer` の独立エントリ）
//! から直接呼ばれる。

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg};

use crate::context_cache;
use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::kernels_batch_norm::{BATCH_NORM_INFER_F32, BATCH_NORM_TRAIN_F32};
use crate::memory::readback;
use crate::nvrtc::compile_ptx;

/// train カーネル起動時の block 幅（32 スレッド = 1 warp 固定。
/// `kernels_batch_norm.rs` の「1 CTA = 1 warp = 1 channel」設計と
/// 一致させる）。
const BATCH_NORM_TRAIN_BLOCK_DIM: u32 = 32;

/// infer カーネル起動時の block 幅（grid-stride elementwise。
/// `kernels_batch_norm.rs` doc comment 参照）。
const BATCH_NORM_INFER_BLOCK_DIM: u32 = 256;

/// 起動前 fail-closed 検証（`layer_norm::validate_layer_norm_launch`・
/// `backend-cpu::batch_norm::validate_batch_norm_launch` と同型。
/// OWASP A03・`.claude/rules/security.md`）: `eps` が有限かつ非負・
/// `n*c*spatial == x_len`（checked 乗算）・`w_len`／`b_len`／
/// `mean_len`／`var_len` が指定時のみ `c` と一致・各次元／`numel` が
/// `i32::MAX`（カーネル引数の `int` 契約）に収まること・
/// `c * size_of::<f32>() <= isize::MAX`（`n=0, c=usize::MAX` のような
/// 退化入力での確保時 capacity overflow panic を防ぐ。CPU 側
/// `batch_norm::validate_batch_norm_launch`〈PR #1874 codex-review
/// P1〉と同じ対策）。
#[allow(clippy::too_many_arguments)]
pub(crate) fn validate_batch_norm_launch(
    n: usize,
    c: usize,
    spatial: usize,
    x_len: usize,
    w_len: Option<usize>,
    b_len: Option<usize>,
    mean_len: Option<usize>,
    var_len: Option<usize>,
    eps: f32,
) -> Result<(), CudaError> {
    if !eps.is_finite() || eps < 0.0 {
        return Err(CudaError::InvalidBatchNormShape {
            detail: format!("batch_norm eps must be finite and non-negative: eps={eps}"),
        });
    }

    // `c` 要素の `mean`／`var`（train 経路。`n==0` 早期 return 経路
    // 込み）を確保する前に、確保サイズ自体が `isize::MAX` を超えない
    // ことを検査する（`n*c*spatial` の乗算検査だけでは `n=0,
    // c=usize::MAX` のように積が 0 へ潰れて通過してしまう経路を防げ
    // ない。`backend-cpu::batch_norm::validate_batch_norm_launch` と
    // 同じ対策）。
    if c.checked_mul(std::mem::size_of::<f32>())
        .is_none_or(|bytes| bytes > isize::MAX as usize)
    {
        return Err(CudaError::InvalidBatchNormShape {
            detail: format!("channel count too large to allocate mean/var: c={c}"),
        });
    }

    let numel = n
        .checked_mul(c)
        .and_then(|v| v.checked_mul(spatial))
        .ok_or_else(|| CudaError::InvalidBatchNormShape {
            detail: format!(
                "batch_norm n*c*spatial overflowed usize: n={n}, c={c}, spatial={spatial}"
            ),
        })?;
    if numel != x_len {
        return Err(CudaError::InvalidBatchNormShape {
            detail: format!("batch_norm x length mismatch: n*c*spatial={numel}, x.len()={x_len}"),
        });
    }
    if let Some(wl) = w_len
        && wl != c
    {
        return Err(CudaError::InvalidBatchNormShape {
            detail: format!("batch_norm weight length mismatch: c={c}, w.len()={wl}"),
        });
    }
    if let Some(bl) = b_len
        && bl != c
    {
        return Err(CudaError::InvalidBatchNormShape {
            detail: format!("batch_norm bias length mismatch: c={c}, b.len()={bl}"),
        });
    }
    if let Some(ml) = mean_len
        && ml != c
    {
        return Err(CudaError::InvalidBatchNormShape {
            detail: format!("batch_norm mean length mismatch: c={c}, mean.len()={ml}"),
        });
    }
    if let Some(vl) = var_len
        && vl != c
    {
        return Err(CudaError::InvalidBatchNormShape {
            detail: format!("batch_norm var length mismatch: c={c}, var.len()={vl}"),
        });
    }

    // カーネル引数はいずれも `int`（`n`／`c`／`spatial`／`m`）または
    // `long long`（`numel`。infer カーネルのみ）。`m = n*spatial` も
    // `int` 引数として渡すため `i32::MAX` 上限検査の対象へ含める。
    let m = n
        .checked_mul(spatial)
        .ok_or_else(|| CudaError::InvalidBatchNormShape {
            detail: format!("batch_norm n*spatial overflowed usize: n={n}, spatial={spatial}"),
        })?;
    if n > i32::MAX as usize
        || c > i32::MAX as usize
        || spatial > i32::MAX as usize
        || m > i32::MAX as usize
        || numel > i32::MAX as usize
    {
        return Err(CudaError::BatchNormSizeLimitExceeded {
            detail: format!(
                "batch_norm dims must fit in i32 (kernel argument type): n={n}, c={c}, \
                 spatial={spatial}, m={m}, numel={numel}"
            ),
        });
    }
    Ok(())
}

/// BatchNorm1d／2d 順伝播カーネル（train／infer）のコンパイル済み
/// ハンドルを保持する。
pub struct CudaBatchNorm {
    stream: Arc<CudaStream>,
    /// 構築元 `CudaDevice` の ordinal（`layer_norm.rs::CudaLayerNorm::
    /// ordinal` と同じ役割。`Self::with_driver_call` が `context_cache::
    /// with_driver_call` を呼ぶ際のキーとして使う）。
    ordinal: usize,
    train_func: CudaFunction,
    infer_func: CudaFunction,
}

impl CudaBatchNorm {
    /// `device` 上で train／infer カーネルを NVRTC コンパイルし保持
    /// するハンドルを構築する（`layer_norm.rs::CudaLayerNorm::new` と
    /// 同一手順）。
    pub fn new(device: &CudaDevice) -> Result<Self, CudaError> {
        let arch = device.arch();
        let train_ptx = compile_ptx(BATCH_NORM_TRAIN_F32, arch)?;
        let train_func = device
            .context()
            .load_module(train_ptx)?
            .load_function("batch_norm_train_f32")?;
        let infer_ptx = compile_ptx(BATCH_NORM_INFER_F32, arch)?;
        let infer_func = device
            .context()
            .load_module(infer_ptx)?
            .load_function("batch_norm_infer_f32")?;

        Ok(Self {
            stream: device.stream().clone(),
            ordinal: device.ordinal(),
            train_func,
            infer_func,
        })
    }

    /// `CudaBatchNorm` の driver 呼び出し（H2D 転送・カーネル起動・
    /// D2H readback）を CUDA Graph capture 排他へ参加させる共通
    /// ヘルパー（`layer_norm.rs::CudaLayerNorm::with_driver_call` と
    /// 同じ設計）。
    fn with_driver_call<T>(
        &self,
        f: impl FnOnce() -> Result<T, CudaError>,
    ) -> Result<T, CudaError> {
        context_cache::with_driver_call(self.ordinal, f)
    }

    /// BatchNorm1d／2d train モード（バッチ統計）を実行する。
    ///
    /// `x` は `[n, c, spatial]` 相当の行優先 1 次元化済みバッファ
    /// （`fandhe_ai_tensor_core::batch_norm_layout` が導出する契約と
    /// 同一）。戻り値は `(out, mean, var)`（`mean`／`var` は biased
    /// ÷M・長さ `c`）。`n == 0 || c == 0 || spatial == 0` は driver に
    /// 触れず空結果（`(vec![], vec![0.0; c], vec![0.0; c])`）を返す
    /// （`backend-cpu::batch_norm::run_batch_norm_train_f32` と同じ
    /// 0 要素契約）。
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn run_batch_norm_train_f32(
        &self,
        x: &[f32],
        w: Option<&[f32]>,
        b: Option<&[f32]>,
        eps: f32,
        n: usize,
        c: usize,
        spatial: usize,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), CudaError> {
        validate_batch_norm_launch(
            n,
            c,
            spatial,
            x.len(),
            w.map(|s| s.len()),
            b.map(|s| s.len()),
            None,
            None,
            eps,
        )?;

        if n == 0 || c == 0 || spatial == 0 {
            return Ok((Vec::new(), vec![0.0f32; c], vec![0.0f32; c]));
        }

        let n_i = n as i32;
        let c_i = c as i32;
        let spatial_i = spatial as i32;
        let m_i = (n * spatial) as i32;

        self.with_driver_call(|| {
            let x_dev = self.stream.clone_htod(x)?;
            // `w`／`b` が `None` の場合も `c` 要素ゼロ初期化ダミーを
            // 渡す（`layer_norm.rs::run_layer_norm_f32` の同種コメント
            // 参照。predicated load による OOB 読み出し回避）。
            let (w_dev, has_weight) = match w {
                Some(w_slice) => (self.stream.clone_htod(w_slice)?, 1i32),
                None => (self.stream.alloc_zeros::<f32>(c)?, 0i32),
            };
            let (b_dev, has_bias) = match b {
                Some(b_slice) => (self.stream.clone_htod(b_slice)?, 1i32),
                None => (self.stream.alloc_zeros::<f32>(c)?, 0i32),
            };
            let mut out_dev = self.stream.alloc_zeros::<f32>(x.len())?;
            let mut mean_dev = self.stream.alloc_zeros::<f32>(c)?;
            let mut var_dev = self.stream.alloc_zeros::<f32>(c)?;

            let cfg = LaunchConfig {
                grid_dim: (c_i as u32, 1, 1),
                block_dim: (BATCH_NORM_TRAIN_BLOCK_DIM, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: `x_dev` は `n*c*spatial` 要素の H2D 済みデバイス
            // バッファ、`w_dev`／`b_dev` は `has_weight`／`has_bias` が
            // 真のときのみ `c` 要素の実データ（偽のときはデリファレン
            // スされない `c` 要素ゼロ初期化ダミー）、`out_dev` は
            // `x.len()` 要素・`mean_dev`／`var_dev` は `c` 要素確保済み
            // でカーネルがそれぞれ全要素を書く（`kernels_batch_norm.rs`
            // 参照）。grid 次元は `c`（1 CTA = 1 channel）で
            // `kernels_batch_norm.rs` 側の `if (ch >= c) return;` ガード
            // と合わせて OOB は起きない。
            unsafe {
                self.stream
                    .launch_builder(&self.train_func)
                    .arg(&x_dev)
                    .arg(&w_dev)
                    .arg(&b_dev)
                    .arg(&mut out_dev)
                    .arg(&mut mean_dev)
                    .arg(&mut var_dev)
                    .arg(&n_i)
                    .arg(&c_i)
                    .arg(&spatial_i)
                    .arg(&m_i)
                    .arg(&eps)
                    .arg(&has_weight)
                    .arg(&has_bias)
                    .launch(cfg)?;
            }

            let out = readback(&self.stream, &out_dev)?;
            let mean = readback(&self.stream, &mean_dev)?;
            let var = readback(&self.stream, &var_dev)?;
            Ok((out, mean, var))
        })
    }

    /// BatchNorm1d／2d eval モード（固定統計）を実行する。
    ///
    /// `mean`／`var`（呼び出し元が保持する running stats。長さ `c`）
    /// をバッチから計算し直さずそのまま使う。`n == 0 || c == 0 ||
    /// spatial == 0` は driver に触れず空結果を返す。
    #[allow(clippy::too_many_arguments)]
    pub fn run_batch_norm_infer_f32(
        &self,
        x: &[f32],
        mean: &[f32],
        var: &[f32],
        w: Option<&[f32]>,
        b: Option<&[f32]>,
        eps: f32,
        n: usize,
        c: usize,
        spatial: usize,
    ) -> Result<Vec<f32>, CudaError> {
        validate_batch_norm_launch(
            n,
            c,
            spatial,
            x.len(),
            w.map(|s| s.len()),
            b.map(|s| s.len()),
            Some(mean.len()),
            Some(var.len()),
            eps,
        )?;

        if n == 0 || c == 0 || spatial == 0 {
            return Ok(Vec::new());
        }

        let c_i = c as i32;
        let spatial_i = spatial as i32;
        let numel = x.len();
        let numel_ll = numel as i64;
        // grid-stride ループのため grid 次元は `numel` に対して十分な
        // block 数を確保すればよい（`numel` を block_dim で割った切り
        // 上げ。`kernels_elementwise.rs` 等と同型の起動式）。
        let grid = numel.div_ceil(BATCH_NORM_INFER_BLOCK_DIM as usize) as u32;
        let grid = grid.max(1);

        self.with_driver_call(|| {
            let x_dev = self.stream.clone_htod(x)?;
            let mean_dev = self.stream.clone_htod(mean)?;
            let var_dev = self.stream.clone_htod(var)?;
            let (w_dev, has_weight) = match w {
                Some(w_slice) => (self.stream.clone_htod(w_slice)?, 1i32),
                None => (self.stream.alloc_zeros::<f32>(c)?, 0i32),
            };
            let (b_dev, has_bias) = match b {
                Some(b_slice) => (self.stream.clone_htod(b_slice)?, 1i32),
                None => (self.stream.alloc_zeros::<f32>(c)?, 0i32),
            };
            let mut out_dev = self.stream.alloc_zeros::<f32>(numel)?;

            let cfg = LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (BATCH_NORM_INFER_BLOCK_DIM, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: `x_dev`／`out_dev` は `numel` 要素、`mean_dev`／
            // `var_dev`／`w_dev`／`b_dev` は `c` 要素（`w`／`b` の
            // `None` 時ダミーも `c` 要素ゼロ初期化）。カーネルは
            // grid-stride ループで `idx < numel` を手動ガードし、
            // `ch = (idx/spatial) % c` は常に `[0, c)` に収まるため
            // OOB は起きない（`kernels_batch_norm.rs` 参照）。
            unsafe {
                self.stream
                    .launch_builder(&self.infer_func)
                    .arg(&x_dev)
                    .arg(&mean_dev)
                    .arg(&var_dev)
                    .arg(&w_dev)
                    .arg(&b_dev)
                    .arg(&mut out_dev)
                    .arg(&c_i)
                    .arg(&spatial_i)
                    .arg(&numel_ll)
                    .arg(&eps)
                    .arg(&has_weight)
                    .arg(&has_bias)
                    .launch(cfg)?;
            }

            readback(&self.stream, &out_dev)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_batch_norm_launch_accepts_matching_dims() {
        assert!(
            validate_batch_norm_launch(2, 3, 4, 24, Some(3), Some(3), None, None, 1e-5).is_ok()
        );
        assert!(validate_batch_norm_launch(2, 3, 4, 24, None, None, None, None, 1e-5).is_ok());
        assert!(
            validate_batch_norm_launch(2, 3, 4, 24, None, None, Some(3), Some(3), 1e-5).is_ok()
        );
    }

    #[test]
    fn validate_batch_norm_launch_rejects_x_len_mismatch() {
        let err =
            validate_batch_norm_launch(2, 3, 4, 23, None, None, None, None, 1e-5).unwrap_err();
        assert!(matches!(err, CudaError::InvalidBatchNormShape { .. }));
    }

    #[test]
    fn validate_batch_norm_launch_rejects_weight_len_mismatch() {
        let err =
            validate_batch_norm_launch(2, 3, 4, 24, Some(2), None, None, None, 1e-5).unwrap_err();
        assert!(matches!(err, CudaError::InvalidBatchNormShape { .. }));
    }

    #[test]
    fn validate_batch_norm_launch_rejects_bias_len_mismatch() {
        let err =
            validate_batch_norm_launch(2, 3, 4, 24, None, Some(2), None, None, 1e-5).unwrap_err();
        assert!(matches!(err, CudaError::InvalidBatchNormShape { .. }));
    }

    #[test]
    fn validate_batch_norm_launch_rejects_stats_len_mismatch() {
        let err = validate_batch_norm_launch(2, 3, 4, 24, None, None, Some(2), Some(3), 1e-5)
            .unwrap_err();
        assert!(matches!(err, CudaError::InvalidBatchNormShape { .. }));
    }

    #[test]
    fn validate_batch_norm_launch_rejects_non_finite_eps() {
        for eps in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let err =
                validate_batch_norm_launch(2, 3, 4, 24, None, None, None, None, eps).unwrap_err();
            assert!(matches!(err, CudaError::InvalidBatchNormShape { .. }));
        }
    }

    #[test]
    fn validate_batch_norm_launch_rejects_negative_eps() {
        let err =
            validate_batch_norm_launch(2, 3, 4, 24, None, None, None, None, -1e-5).unwrap_err();
        assert!(matches!(err, CudaError::InvalidBatchNormShape { .. }));
    }

    /// `n=0, c=usize::MAX, spatial=1` は `n*c*spatial` が `usize` 積と
    /// しては `0` に潰れてしまうが、`c*size_of::<f32>()` の確保前上限
    /// 検査により型付きエラーへ収束する（CPU 側 PR #1874 codex-review
    /// P1 是正と同じ対策）。
    #[test]
    fn validate_batch_norm_launch_rejects_huge_channel_count_without_overflow() {
        let err = validate_batch_norm_launch(0, usize::MAX, 1, 0, None, None, None, None, 1e-5)
            .unwrap_err();
        assert!(matches!(err, CudaError::InvalidBatchNormShape { .. }));
    }

    /// 形状パラメータが `i32::MAX` を超える場合は
    /// `BatchNormSizeLimitExceeded` を返す（`ops.rs::map_batch_norm_error`
    /// が `Unsupported` へ写像しホストフォールバックへ委ねる対象）。
    #[test]
    fn validate_batch_norm_launch_rejects_dims_exceeding_i32_max() {
        let big = i32::MAX as usize + 1;
        let err =
            validate_batch_norm_launch(1, big, 1, big, None, None, None, None, 1e-5).unwrap_err();
        assert!(matches!(err, CudaError::BatchNormSizeLimitExceeded { .. }));
    }
}
