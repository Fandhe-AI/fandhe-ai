//! Conv2d の im2col／col2im 起動 API（NVRTC コンパイル・保持・実行。
//! イシュー #1766・親 #1643・設計 `docs/conv-ops-design.md`）。
//!
//! `constant_pad.rs::CudaConstantPad`・`scan.rs::CudaScan` と同じ構成
//! 方針を踏襲する: [`CudaIm2col::new`] が `CudaDevice` から
//! `im2col_f32`／`col2im_f32`（`kernels_im2col.rs`）を NVRTC
//! コンパイルして保持し、以降は [`CudaIm2col::run_im2col_f32`]／
//! [`CudaIm2col::run_col2im_f32`] へホスト側スライスを渡すだけで GPU
//! 実行できる。`ops.rs::CudaBackendOps::im2col`／`col2im` から
//! `BackendOps` の実装として呼ばれる。
//!
//! **shape 検証の責務分担**（`gather_scatter.rs`／`constant_pad.rs`
//! モジュール doc・`.claude/rules/security.md` A08 と同じ二重検査
//! 方針）: 呼び出し元 `ops.rs` が
//! [`fandhe_ai_tensor_core::im2col_out_shape`] で `input.shape()`／
//! `params`（`im2col`）または `d_col.shape()`／`input_shape`／`params`
//! （`col2im`）の rank・整合を再検査してから本モジュールへ委譲する
//! 契約のため、本モジュール自身は `in_shape`／`out_shape` の rank を
//! `debug_assert!` でのみ確認する（release ビルドでは無効化される
//! 内部契約検証）。一方でホストスライスの**要素数**（`in_shape`／
//! `out_shape` から導出した期待長との一致）とカーネル引数 `int`
//! 上限（`i32::MAX`）は release ビルドでも独立に検証する
//! （`scan.rs::validate_i32_bound` と同じ理由: ホストスライスが短いと
//! GPU 側で未定義動作になりうるため、H2D 転送前に必須の独立した
//! 安全策）。
//!
//! `kernels_gather_scatter.rs`／`kernels_constant_pad.rs` と異なり
//! rank 可変の shape を配列 H2D で渡す必要がない（`Conv2dParams` は
//! rank 固定 4D NCHW）ため、すべての形状パラメータをスカラー `int`
//! 引数として渡す（`kernels_im2col.rs` モジュール doc 参照）。

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use fandhe_ai_tensor_core::{Conv2dParams, conv_out_len};

use crate::context_cache;
use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::kernels_im2col::{self, IM2COL_BLOCK_DIM};
use crate::memory::readback;
use crate::nvrtc::compile_ptx;
use crate::pool::CudaAllocator;

/// `value` が `i32::MAX` に収まることを検証する（カーネル引数 `int` は
/// C の 32bit 符号付き整数のため。`scan.rs::validate_i32_bound` と同じ
/// 理由の複製——モジュールごとに専用の検証関数を持つ既存方針を踏襲
/// する）。値が上限を超える場合は [`CudaError::Im2colSizeLimitExceeded`]
/// （バックエンド固有サイズ上限の超過。`ops.rs` が `Unsupported` へ
/// 写像しホストフォールバックへ委ねる。`InvalidIm2colShape`〈内部契約
/// 違反〉とは区別する）を返す。
fn validate_i32_bound(value: usize, name: &str) -> Result<i32, CudaError> {
    i32::try_from(value).map_err(|_| CudaError::Im2colSizeLimitExceeded {
        detail: format!(
            "im2col/col2im dimension must fit in i32 (kernel argument type): {name}={value}"
        ),
    })
}

/// shape の要素数積を `checked_mul` の畳み込みで検査する
/// （`constant_pad.rs::checked_numel` と同型。クレート内で `pub(crate)`
/// 共有できないため専用に複製する。理由は同モジュールの doc を参照）。
fn checked_numel(shape: &[usize]) -> Result<usize, CudaError> {
    shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| CudaError::InvalidIm2colShape {
            detail: "checked_numel: element count overflow".to_string(),
        })
}

/// im2col／col2im カーネルのコンパイル済みハンドルを保持する。
pub struct CudaIm2col {
    stream: Arc<CudaStream>,
    /// `constant_pad.rs::CudaConstantPad::ordinal` と同じ役割
    /// （[`Self::with_driver_call`] が `context_cache::with_driver_call`
    /// を呼ぶ際のキー）。
    ordinal: usize,
    allocator: Arc<CudaAllocator>,
    im2col_f32: CudaFunction,
    col2im_f32: CudaFunction,
}

/// [`im2col_out_shape`] 相当の形状パラメータを 1 呼び出し分にまとめた
/// 内部構造体（`run_im2col_f32`／`run_col2im_f32` 双方の起動引数を
/// 共有するための集約。`kernels_im2col.rs` の 19 個のスカラー引数と
/// 1:1 対応する）。
#[derive(Debug)]
struct LaunchShape {
    n_batch: i32,
    cin: i32,
    groups: i32,
    cin_g: i32,
    h_in: i32,
    w_in: i32,
    h_out: i32,
    w_out: i32,
    kh: i32,
    kw: i32,
    sh: i32,
    sw: i32,
    ph: i32,
    pw: i32,
    dh: i32,
    dw: i32,
    k_g: i32,
    p: i32,
}

impl LaunchShape {
    /// `in_shape: [N, Cin, H, W]`・`groups`／`k_g`／`p`（`out_shape`
    /// `[N, G, K_g, P]` から）・`params` から起動引数一式を導出し、
    /// すべて `i32` 範囲検査する。`h_out`／`w_out` は `conv_out_len`
    /// で独立に再計算する（`im2col`／`col2im` いずれも `out_shape`／
    /// `d_col` の `P` 軸だけでは `h_out`／`w_out` 個別の値が復元
    /// できないため。`backend-cpu::im2col::im2col` の
    /// `debug_assert_eq!` と同じ再計算だが、本モジュールはカーネル
    /// 引数として実際に使うため release ビルドでも必ず計算する）。
    fn derive(
        in_shape: &[usize],
        groups: usize,
        k_g: usize,
        p: usize,
        params: &Conv2dParams,
    ) -> Result<Self, CudaError> {
        let (n_batch, cin, h_in, w_in) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
        let cin_g = cin / groups.max(1);
        let [kh, kw] = params.kernel_size();
        let [sh, sw] = params.stride();
        let [ph, pw] = params.padding();
        let [dh, dw] = params.dilation();
        let h_out =
            conv_out_len(h_in, kh, sh, ph, dh).map_err(|e| CudaError::InvalidIm2colShape {
                detail: format!("conv_out_len(h) failed: {e:?}"),
            })?;
        let w_out =
            conv_out_len(w_in, kw, sw, pw, dw).map_err(|e| CudaError::InvalidIm2colShape {
                detail: format!("conv_out_len(w) failed: {e:?}"),
            })?;
        if h_out.checked_mul(w_out) != Some(p) {
            return Err(CudaError::InvalidIm2colShape {
                detail: format!(
                    "P axis mismatch: h_out*w_out={} p={p}",
                    h_out.saturating_mul(w_out)
                ),
            });
        }

        Ok(Self {
            n_batch: validate_i32_bound(n_batch, "n_batch")?,
            cin: validate_i32_bound(cin, "cin")?,
            groups: validate_i32_bound(groups, "groups")?,
            cin_g: validate_i32_bound(cin_g, "cin_g")?,
            h_in: validate_i32_bound(h_in, "h_in")?,
            w_in: validate_i32_bound(w_in, "w_in")?,
            h_out: validate_i32_bound(h_out, "h_out")?,
            w_out: validate_i32_bound(w_out, "w_out")?,
            kh: validate_i32_bound(kh, "kh")?,
            kw: validate_i32_bound(kw, "kw")?,
            sh: validate_i32_bound(sh, "sh")?,
            sw: validate_i32_bound(sw, "sw")?,
            ph: validate_i32_bound(ph, "ph")?,
            pw: validate_i32_bound(pw, "pw")?,
            dh: validate_i32_bound(dh, "dh")?,
            dw: validate_i32_bound(dw, "dw")?,
            k_g: validate_i32_bound(k_g, "k_g")?,
            p: validate_i32_bound(p, "p")?,
        })
    }
}

impl CudaIm2col {
    /// `device` 上で im2col／col2im 2 カーネルを NVRTC コンパイルし
    /// 保持するハンドルを構築する（`scan.rs::CudaScan::new` と同一
    /// 手順）。
    pub fn new(device: &CudaDevice) -> Result<Self, CudaError> {
        let arch = device.arch();

        macro_rules! compile_and_load {
            ($src:expr, $name:literal) => {{
                let ptx = compile_ptx($src, arch)?;
                device.context().load_module(ptx)?.load_function($name)?
            }};
        }

        let im2col_f32 = compile_and_load!(kernels_im2col::IM2COL_F32, "im2col_f32");
        let col2im_f32 = compile_and_load!(kernels_im2col::COL2IM_F32, "col2im_f32");

        let allocator = context_cache::cached_allocator(device)?;

        Ok(Self {
            stream: device.stream().clone(),
            ordinal: device.ordinal(),
            allocator,
            im2col_f32,
            col2im_f32,
        })
    }

    /// `CudaIm2col` の driver 呼び出しを CUDA Graph capture 排他へ
    /// 参加させる共通ヘルパー（`constant_pad.rs::CudaConstantPad::
    /// with_driver_call` と同じ設計）。
    fn with_driver_call<T>(
        &self,
        f: impl FnOnce() -> Result<T, CudaError>,
    ) -> Result<T, CudaError> {
        context_cache::with_driver_call(self.ordinal, f)
    }

    /// Conv2d の im2col（`kernels_im2col.rs` モジュール doc 参照）。
    /// `in_shape: [N, Cin, H, W]`・`out_shape: [N, G, K_g, P]`
    /// （呼び出し元 `ops.rs` が `im2col_out_shape` で検査・確定済み）。
    ///
    /// `out_shape` の要素数が 0 の場合、それは `in_shape` のいずれかの
    /// 軸が 0（`N`／`Cin`。空間軸 `H`／`W` は `im2col_out_shape` が
    /// 事前に拒否する契約）であることに起因し `input` を読む必要が
    /// 一切ないため、空 `Vec` を早期に返す（GPU 計算を伴わない。
    /// `gather_scatter.rs::run_gather_f32` の空出力早期リターンと同じ
    /// 理由）。
    pub fn run_im2col_f32(
        &self,
        input: &[f32],
        in_shape: &[usize],
        out_shape: &[usize],
        params: &Conv2dParams,
    ) -> Result<Vec<f32>, CudaError> {
        debug_assert_eq!(
            in_shape.len(),
            4,
            "run_im2col_f32: caller must pre-validate rank via im2col_out_shape"
        );
        debug_assert_eq!(
            out_shape.len(),
            4,
            "run_im2col_f32: caller must pre-validate rank via im2col_out_shape"
        );

        let numel_out = checked_numel(out_shape)?;
        if numel_out == 0 {
            return Ok(Vec::new());
        }

        let numel_in = checked_numel(in_shape)?;
        if input.len() != numel_in {
            return Err(CudaError::InvalidIm2colShape {
                detail: format!(
                    "im2col: input.len()={} does not match in numel={numel_in}",
                    input.len()
                ),
            });
        }

        let groups = out_shape[1];
        let k_g = out_shape[2];
        let p = out_shape[3];
        let shape = LaunchShape::derive(in_shape, groups, k_g, p, params)?;
        let numel_i = validate_i32_bound(numel_out, "numel_out")?;

        self.with_driver_call(|| {
            let in_dev = self.stream.clone_htod(input)?;
            let mut out_dev = self.allocator.alloc_uninit_f32(numel_out)?;

            let cfg = LaunchConfig {
                grid_dim: ((numel_out as u32).div_ceil(IM2COL_BLOCK_DIM), 1, 1),
                block_dim: (IM2COL_BLOCK_DIM, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: `in_dev` は `numel_in` 要素の H2D 済みバッファ
            // （直上の長さ検証でホストスライスと一致することを確認
            // 済み）。`out_dev` は `numel_out` 要素確保済みでカーネルは
            // `idx < numel`（REQ-8）を維持したまま各出力要素を 1 回
            // だけ書く。`in` への読み出し添字（`in_idx`）はカーネル内で
            // `0 <= h < h_in`・`0 <= w < w_in` を検査してから計算する
            // ため範囲外読み出しはない（`kernels_im2col.rs::
            // IM2COL_F32` 参照）。
            unsafe {
                self.stream
                    .launch_builder(&self.im2col_f32)
                    .arg(&in_dev)
                    .arg(&mut out_dev.as_view_mut())
                    .arg(&shape.n_batch)
                    .arg(&shape.cin)
                    .arg(&shape.groups)
                    .arg(&shape.cin_g)
                    .arg(&shape.h_in)
                    .arg(&shape.w_in)
                    .arg(&shape.h_out)
                    .arg(&shape.w_out)
                    .arg(&shape.kh)
                    .arg(&shape.kw)
                    .arg(&shape.sh)
                    .arg(&shape.sw)
                    .arg(&shape.ph)
                    .arg(&shape.pw)
                    .arg(&shape.dh)
                    .arg(&shape.dw)
                    .arg(&shape.k_g)
                    .arg(&shape.p)
                    .arg(&numel_i)
                    .launch(cfg)?;
            }
            readback(&self.stream, &out_dev.as_view())
        })
    }

    /// Conv2d の col2im（[`Self::run_im2col_f32`] の随伴。
    /// `kernels_im2col.rs` モジュール doc 参照）。`d_col: [N, G, K_g,
    /// P]`・`input_shape: [N, Cin, H, W]`（呼び出し元 `ops.rs` が
    /// `d_col.shape()` との完全一致を再検査済み）。
    ///
    /// `input_shape` の要素数が 0 の場合、空 `Vec` を早期に返す
    /// （`d_col` を読む必要が一切ない。`backend-cpu::im2col::col2im`
    /// と同じ早期リターン契約）。
    pub fn run_col2im_f32(
        &self,
        d_col: &[f32],
        col_shape: &[usize],
        input_shape: &[usize],
        params: &Conv2dParams,
    ) -> Result<Vec<f32>, CudaError> {
        debug_assert_eq!(
            col_shape.len(),
            4,
            "run_col2im_f32: caller must pre-validate rank via im2col_out_shape"
        );
        debug_assert_eq!(
            input_shape.len(),
            4,
            "run_col2im_f32: caller must pre-validate rank"
        );

        let numel_out = checked_numel(input_shape)?;
        if numel_out == 0 {
            return Ok(Vec::new());
        }

        let numel_col = checked_numel(col_shape)?;
        if d_col.len() != numel_col {
            return Err(CudaError::InvalidIm2colShape {
                detail: format!(
                    "col2im: d_col.len()={} does not match col numel={numel_col}",
                    d_col.len()
                ),
            });
        }

        let groups = col_shape[1];
        let k_g = col_shape[2];
        let p = col_shape[3];
        let shape = LaunchShape::derive(input_shape, groups, k_g, p, params)?;
        let numel_i = validate_i32_bound(numel_out, "numel_out")?;

        self.with_driver_call(|| {
            let col_dev = self.stream.clone_htod(d_col)?;
            let mut out_dev = self.allocator.alloc_uninit_f32(numel_out)?;

            let cfg = LaunchConfig {
                grid_dim: ((numel_out as u32).div_ceil(IM2COL_BLOCK_DIM), 1, 1),
                block_dim: (IM2COL_BLOCK_DIM, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: `col_dev` は `numel_col` 要素の H2D 済みバッファ
            // （直上の長さ検証でホストスライスと一致することを確認
            // 済み）。`out_dev` は `numel_out` 要素確保済みでカーネルは
            // `idx < numel`（REQ-8）を維持したまま各出力要素を 1 回
            // だけ書く（atomic 不使用・1 スレッド 1 出力要素の入力位置
            // 定常走査）。`d_col` への読み出し添字（`col_idx`）は
            // カーネル内で `0 <= oh < h_out`・`0 <= ow < w_out` を
            // 検査してから計算するため範囲外読み出しはない
            // （`kernels_im2col.rs::COL2IM_F32` 参照）。
            unsafe {
                self.stream
                    .launch_builder(&self.col2im_f32)
                    .arg(&col_dev)
                    .arg(&mut out_dev.as_view_mut())
                    .arg(&shape.n_batch)
                    .arg(&shape.cin)
                    .arg(&shape.groups)
                    .arg(&shape.cin_g)
                    .arg(&shape.h_in)
                    .arg(&shape.w_in)
                    .arg(&shape.h_out)
                    .arg(&shape.w_out)
                    .arg(&shape.kh)
                    .arg(&shape.kw)
                    .arg(&shape.sh)
                    .arg(&shape.sw)
                    .arg(&shape.ph)
                    .arg(&shape.pw)
                    .arg(&shape.dh)
                    .arg(&shape.dw)
                    .arg(&shape.k_g)
                    .arg(&shape.p)
                    .arg(&numel_i)
                    .launch(cfg)?;
            }
            readback(&self.stream, &out_dev.as_view())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_i32_bound_rejects_exceeding_i32_max() {
        let err = validate_i32_bound(i32::MAX as usize + 1, "h_in").unwrap_err();
        assert!(matches!(err, CudaError::Im2colSizeLimitExceeded { .. }));
    }

    #[test]
    fn validate_i32_bound_accepts_i32_max() {
        assert!(validate_i32_bound(i32::MAX as usize, "h_in").is_ok());
    }

    #[test]
    fn checked_numel_rejects_overflow() {
        let err = checked_numel(&[usize::MAX, 2]).unwrap_err();
        assert!(matches!(err, CudaError::InvalidIm2colShape { .. }));
    }

    #[test]
    fn checked_numel_computes_product() {
        assert_eq!(checked_numel(&[2, 3, 4]).unwrap(), 24);
    }

    #[test]
    fn launch_shape_derive_rejects_p_axis_mismatch() {
        let params = Conv2dParams::new([2, 2], [1, 1], [0, 0], [1, 1], 1).unwrap();
        // in_shape [1,1,4,4] -> h_out=w_out=3 -> P=9 (kernel=2,stride=1).
        // p=1 を渡して意図的に不一致を起こす。
        let err = LaunchShape::derive(&[1, 1, 4, 4], 1, 4, 1, &params).unwrap_err();
        assert!(matches!(err, CudaError::InvalidIm2colShape { .. }));
    }

    #[test]
    fn launch_shape_derive_accepts_consistent_shape() {
        let params = Conv2dParams::new([2, 2], [1, 1], [0, 0], [1, 1], 1).unwrap();
        // in_shape [1,1,4,4] -> h_out=w_out=3 -> P=9.
        let shape = LaunchShape::derive(&[1, 1, 4, 4], 1, 4, 9, &params).unwrap();
        assert_eq!(shape.h_out, 3);
        assert_eq!(shape.w_out, 3);
        assert_eq!(shape.p, 9);
    }

    /// イシュー #1767: 1d 形状（`Var::conv1d` が reshape する
    /// `[N, Cin, 1, L]`・`kernel=[1, k]`）でも `h_out=1`（`H` 軸自体が
    /// 1・`kh=1`・`sh=1`・`dh=1` のため `conv_out_len` は常に 1 を
    /// 返す）・`w_out=lout`・`P=w_out` が正しく導出されることを、
    /// stride／padding／dilation を伴う非自明な値で確認する
    /// （driver 非接触）。
    #[test]
    fn launch_shape_derive_handles_1d_shape() {
        let params = Conv2dParams::new([1, 3], [1, 2], [0, 1], [1, 2], 1).unwrap();
        // L=9, k=3, stride=2, padding=1, dilation=2 ->
        // lout = floor((9 + 2*1 - 2*(3-1) - 1) / 2) + 1 = floor(6/2)+1 = 4.
        let in_shape = [1usize, 1, 1, 9];
        let shape = LaunchShape::derive(&in_shape, 1, 3, 4, &params).unwrap();
        assert_eq!(shape.h_out, 1);
        assert_eq!(shape.w_out, 4);
        assert_eq!(shape.p, 4);
        assert_eq!(shape.h_in, 1);
        assert_eq!(shape.kh, 1);
        assert_eq!(shape.sh, 1);
        assert_eq!(shape.ph, 0);
        assert_eq!(shape.dh, 1);
    }

    /// 1d 形状でも `P` 軸不整合（`h_out*w_out != p`）は 2d と同じ経路で
    /// 拒否される（`launch_shape_derive_rejects_p_axis_mismatch` の 1d
    /// 版。driver 非接触）。
    #[test]
    fn launch_shape_derive_rejects_1d_p_axis_mismatch() {
        let params = Conv2dParams::new([1, 2], [1, 1], [0, 0], [1, 1], 1).unwrap();
        // in_shape [1,1,1,4] -> w_out=3 -> 正しい p=3 のところに 1 を渡す。
        let err = LaunchShape::derive(&[1, 1, 1, 4], 1, 4, 1, &params).unwrap_err();
        assert!(matches!(err, CudaError::InvalidIm2colShape { .. }));
    }
}
