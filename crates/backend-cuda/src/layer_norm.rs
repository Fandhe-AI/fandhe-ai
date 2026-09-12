//! LayerNorm 順伝播カーネルの起動 API（イシュー #1596。`rmsnorm.rs`
//! 〈#592〉・`mse.rs`〈#1045〉と同じ構成方針を踏襲する CUDA 対応版）。
//!
//! [`CudaLayerNorm::new`] が `kernels_layer_norm.rs` の単一カーネルを
//! NVRTC コンパイルして保持し、[`CudaLayerNorm::run_layer_norm_f32`]
//! へホスト側スライスを渡すだけで H2D → 起動 → D2H を内部で完結できる。
//! `kernels_layer_norm.rs` 冒頭コメントのとおり、`rmsnorm.rs` の
//! persistent block・occupancy 予算に基づく grid 導出は採用せず、
//! `grid_dim = rows`（1 CTA = 1 行）の単純な起動とする（`mse.rs` と
//! 同水準の簡素さ）。
//!
//! `ops.rs::CudaBackendOps::layer_norm`（`BackendOps::layer_norm` の
//! 独立エントリ）から直接呼ばれる。既存の `run_fused`（canonical 融合
//! プラン一致経路）に LayerNorm 一致経路は追加しない
//! （`docs/norm-ops-design.md`「LayerNorm は本エントリ経由でのみ到達
//! する」）。

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg};

use crate::context_cache;
use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::kernels_layer_norm::LAYER_NORM_F32;
use crate::memory::readback;
use crate::nvrtc::compile_ptx;

/// カーネル起動時の block 幅（32 スレッド = 1 warp 固定。
/// `kernels_layer_norm.rs` の「1 CTA = 1 warp」設計と一致させる）。
const LAYER_NORM_BLOCK_DIM: u32 = 32;

/// 起動前 fail-closed 検証（`rmsnorm.rs::validate_rmsnorm_launch` と
/// 同型。OWASP A03・`.claude/rules/security.md`）: `eps` が有限かつ
/// 非負・`rows * hidden == x_len`（checked 乗算）・`w_len`／`b_len` が
/// 指定時のみ `hidden` と一致・各次元が `i32::MAX`（カーネル引数の
/// `int rows`／`int hidden` 契約）に収まること。
pub(crate) fn validate_layer_norm_launch(
    rows: usize,
    hidden: usize,
    x_len: usize,
    w_len: Option<usize>,
    b_len: Option<usize>,
    eps: f32,
) -> Result<(), CudaError> {
    if !eps.is_finite() || eps < 0.0 {
        return Err(CudaError::InvalidRmsNormShape {
            detail: format!("layer_norm eps must be finite and non-negative: eps={eps}"),
        });
    }

    let numel = rows
        .checked_mul(hidden)
        .ok_or_else(|| CudaError::InvalidRmsNormShape {
            detail: format!(
                "layer_norm rows*hidden overflowed usize: rows={rows}, hidden={hidden}"
            ),
        })?;
    if numel != x_len {
        return Err(CudaError::InvalidRmsNormShape {
            detail: format!("layer_norm x length mismatch: rows*hidden={numel}, x.len()={x_len}"),
        });
    }
    if let Some(wl) = w_len
        && wl != hidden
    {
        return Err(CudaError::InvalidRmsNormShape {
            detail: format!("layer_norm w length mismatch: hidden={hidden}, w.len()={wl}"),
        });
    }
    if let Some(bl) = b_len
        && bl != hidden
    {
        return Err(CudaError::InvalidRmsNormShape {
            detail: format!("layer_norm b length mismatch: hidden={hidden}, b.len()={bl}"),
        });
    }
    if rows > i32::MAX as usize || hidden > i32::MAX as usize || numel > i32::MAX as usize {
        return Err(CudaError::InvalidRmsNormShape {
            detail: format!(
                "layer_norm dims must fit in i32 (kernel argument type): rows={rows}, \
                 hidden={hidden}, numel={numel}"
            ),
        });
    }
    Ok(())
}

/// LayerNorm 順伝播カーネルのコンパイル済みハンドルを保持する。
pub struct CudaLayerNorm {
    stream: Arc<CudaStream>,
    /// 構築元 `CudaDevice` の ordinal（`rmsnorm.rs::CudaRmsNorm::ordinal`
    /// と同じ役割。`Self::with_driver_call` が `context_cache::
    /// with_driver_call` を呼ぶ際のキーとして使う）。
    ordinal: usize,
    func: CudaFunction,
}

impl CudaLayerNorm {
    /// `device` 上で LayerNorm カーネルを NVRTC コンパイルし保持する
    /// ハンドルを構築する（`mse.rs::CudaMse::new` と同一手順）。
    pub fn new(device: &CudaDevice) -> Result<Self, CudaError> {
        let arch = device.arch();
        let ptx = compile_ptx(LAYER_NORM_F32, arch)?;
        let func = device
            .context()
            .load_module(ptx)?
            .load_function("layer_norm_f32")?;

        Ok(Self {
            stream: device.stream().clone(),
            ordinal: device.ordinal(),
            func,
        })
    }

    /// `CudaLayerNorm` の driver 呼び出し（H2D 転送・カーネル起動・D2H
    /// readback）を CUDA Graph capture 排他へ参加させる共通ヘルパー
    /// （`rmsnorm.rs::CudaRmsNorm::with_driver_call`／`mse.rs::CudaMse::
    /// with_driver_call` と同じ設計）。
    fn with_driver_call<T>(
        &self,
        f: impl FnOnce() -> Result<T, CudaError>,
    ) -> Result<T, CudaError> {
        context_cache::with_driver_call(self.ordinal, f)
    }

    /// LayerNorm（`out = (x − mean(x)) · rsqrt(var(x) + eps) · w + b`。
    /// `w`／`b` はそれぞれ `None` の場合は対応する演算をスキップ。分散は
    /// biased ÷N）を実行する。
    ///
    /// `x` は `[rows, hidden]` の行優先 1 次元化済みバッファ。
    /// `rows == 0 || hidden == 0` は空結果の早期 return（`mse.rs::
    /// run_mse_backward_f32` と同じ 0 要素契約）。
    pub fn run_layer_norm_f32(
        &self,
        x: &[f32],
        w: Option<&[f32]>,
        b: Option<&[f32]>,
        eps: f32,
        rows: usize,
        hidden: usize,
    ) -> Result<Vec<f32>, CudaError> {
        validate_layer_norm_launch(
            rows,
            hidden,
            x.len(),
            w.map(|s| s.len()),
            b.map(|s| s.len()),
            eps,
        )?;

        if rows == 0 || hidden == 0 {
            return Ok(Vec::new());
        }

        let rows_i = rows as i32;
        let hidden_i = hidden as i32;

        self.with_driver_call(|| {
            let x_dev = self.stream.clone_htod(x)?;
            // `w`／`b` が `None` の場合もカーネル引数としてポインタは
            // 必要だが `has_weight`／`has_bias == 0` により論理的には
            // デリファレンスされない。ただし `hidden` 要素すべてに対する
            // `(has_weight != 0) ? w[i] : 1.0f` という warp 一様の三項式は
            // nvcc が predicated load（`i` の全域で `w[i]` の読み出し自体は
            // 無条件発行し、書き込みのみ述語化する）へコンパイルしうる
            // ため、1 要素のダミーでは `hidden > 1` のとき境界外読み出しに
            // なりうる（Cursor Bugbot 指摘）。Metal 側 `layer_norm.rs`
            // （`MetalBuffer::alloc_zeroed_pooled(ctx, hidden)`）と同じ
            // `hidden` 要素ゼロ初期化バッファへ揃える（`rmsnorm.rs::
            // run_rmsnorm_f32_inner` の 1 要素ダミーは本 PR のスコープ外
            // として別途記録する）。
            let (w_dev, has_weight) = match w {
                Some(w_slice) => (self.stream.clone_htod(w_slice)?, 1i32),
                None => (self.stream.alloc_zeros::<f32>(hidden)?, 0i32),
            };
            let (b_dev, has_bias) = match b {
                Some(b_slice) => (self.stream.clone_htod(b_slice)?, 1i32),
                None => (self.stream.alloc_zeros::<f32>(hidden)?, 0i32),
            };
            let mut out_dev = self.stream.alloc_zeros::<f32>(x.len())?;

            let cfg = LaunchConfig {
                grid_dim: (rows_i as u32, 1, 1),
                block_dim: (LAYER_NORM_BLOCK_DIM, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: `x_dev` は `rows*hidden` 要素の H2D 済みデバイス
            // バッファ、`w_dev`／`b_dev` は `has_weight`／`has_bias` が
            // 真のときのみ `hidden` 要素の実データ（偽のときは
            // デリファレンスされない `hidden` 要素ゼロ初期化ダミー）、`out_dev` は
            // `x.len()` 要素確保済みでカーネルが行ごとに全要素を書く
            // （`kernels_layer_norm.rs` 参照）。grid 次元は `rows`
            // （1 CTA = 1 行）で `kernels_layer_norm.rs` 側の
            // `if (row >= rows) return;` ガードと合わせて OOB は
            // 起きない。
            unsafe {
                self.stream
                    .launch_builder(&self.func)
                    .arg(&x_dev)
                    .arg(&w_dev)
                    .arg(&b_dev)
                    .arg(&mut out_dev)
                    .arg(&rows_i)
                    .arg(&hidden_i)
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
    fn validate_layer_norm_launch_accepts_matching_dims() {
        assert!(validate_layer_norm_launch(3, 8, 24, Some(8), Some(8), 1e-5).is_ok());
        assert!(validate_layer_norm_launch(3, 8, 24, None, None, 0.0).is_ok());
    }

    #[test]
    fn validate_layer_norm_launch_rejects_x_len_mismatch() {
        let err = validate_layer_norm_launch(3, 8, 23, None, None, 1e-5).unwrap_err();
        assert!(matches!(err, CudaError::InvalidRmsNormShape { .. }));
    }

    #[test]
    fn validate_layer_norm_launch_rejects_w_len_mismatch() {
        let err = validate_layer_norm_launch(3, 8, 24, Some(7), None, 1e-5).unwrap_err();
        assert!(matches!(err, CudaError::InvalidRmsNormShape { .. }));
    }

    #[test]
    fn validate_layer_norm_launch_rejects_b_len_mismatch() {
        let err = validate_layer_norm_launch(3, 8, 24, None, Some(7), 1e-5).unwrap_err();
        assert!(matches!(err, CudaError::InvalidRmsNormShape { .. }));
    }

    #[test]
    fn validate_layer_norm_launch_rejects_non_finite_eps() {
        let err = validate_layer_norm_launch(3, 8, 24, None, None, f32::NAN).unwrap_err();
        assert!(matches!(err, CudaError::InvalidRmsNormShape { .. }));
    }

    #[test]
    fn validate_layer_norm_launch_rejects_negative_eps() {
        let err = validate_layer_norm_launch(3, 8, 24, None, None, -1e-5).unwrap_err();
        assert!(matches!(err, CudaError::InvalidRmsNormShape { .. }));
    }
}
