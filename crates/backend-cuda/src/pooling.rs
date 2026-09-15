//! MaxPool／AvgPool／AdaptiveAvgPool（2d。イシュー #1729・親 #1607・
//! 設計 `docs/pooling-ops-design.md`）の起動 API（NVRTC コンパイル・
//! 保持・実行）。
//!
//! `im2col.rs::CudaIm2col`・`sort.rs::CudaSort` と同じ構成方針を
//! 踏襲する: [`CudaPooling::new`] が `CudaDevice` から 3 カーネル
//! （`kernels_pooling.rs`）を NVRTC コンパイルして保持し、以降は
//! [`CudaPooling::run_max_pool2d_f32`]／
//! [`CudaPooling::run_avg_pool2d_f32`]／
//! [`CudaPooling::run_adaptive_avg_pool2d_f32`]
//! へホスト側スライスを渡すだけで GPU 実行できる。
//!
//! **`ops.rs::CudaBackendOps` への配線（イシュー #1729・追従イシュー。
//! #1607 ツリー）**: イシュー #1729 実装時点（2026-09-15）では兄弟
//! イシュー #1728（`backend-cpu`。設計 doc §9 が指す共有基盤
//! `fandhe_ai_tensor_core::backend_ops::BackendOps::max_pool2d`／
//! `avg_pool2d`／`adaptive_avg_pool2d`・`Pool2dParams`・出力 shape
//! 関数）が `main` に未マージだったため、本モジュールは
//! `Pool2dParams` へ依存せずプリミティブ引数（`[usize; 2]`・`bool`）
//! で shape・パラメータを受け取り、**自己完結**で検査・出力 shape
//! 導出を行う設計（`pool_out_len`／`validate_*` 関数群。設計 doc §3／
//! §4 の式をクレート内へ複製）を維持したまま、追従イシューで
//! `ops.rs::CudaBackendOps::max_pool2d`／`avg_pool2d`／
//! `adaptive_avg_pool2d`（`Pool2dParams` を受け取り [`pool2d_out_shape`]
//! 等で shape を再検査してから本モジュールの `run_*` へ委譲する
//! `im2col.rs` と同じ二重検査方針）・`context_cache::cached_pooling`
//! を追加した。本モジュール自体は `Pool2dParams` に非依存のまま
//! （`ops.rs` 側が `Pool2dParams` の getter で分解してから渡す）。
//!
//! [`pool2d_out_shape`]: fandhe_ai_tensor_core::pool2d_out_shape

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg};

use crate::context_cache;
use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::kernels_pooling::{self, POOLING_BLOCK_DIM};
use crate::memory::readback;
use crate::nvrtc::compile_ptx;

/// `value` が `i32::MAX` に収まることを検証する（カーネル引数 `int` は
/// C の 32bit 符号付き整数のため。`im2col.rs::validate_i32_bound` と
/// 同じ理由の複製——モジュールごとに専用の検証関数を持つ既存方針を
/// 踏襲する）。
fn validate_i32_bound(value: usize, name: &str) -> Result<i32, CudaError> {
    i32::try_from(value).map_err(|_| CudaError::PoolingSizeLimitExceeded {
        detail: format!("pooling dimension must fit in i32 (kernel argument type): {name}={value}"),
    })
}

/// shape の要素数積を `checked_mul` の畳み込みで検査する
/// （`im2col.rs::checked_numel` と同型の複製）。
fn checked_numel(shape: &[usize; 4]) -> Result<usize, CudaError> {
    shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| CudaError::InvalidPoolingShape {
            detail: "checked_numel: element count overflow".to_string(),
        })
}

/// `docs/pooling-ops-design.md` §4 の `pool_out_len` を実装する。
/// 分子（`in + 2p − d(k−1) − 1`）が負の場合は floor 除算を行わず直ちに
/// 拒否する（設計 doc §4「実装契約」。Rust の符号付き整数 `/` は
/// ゼロ方向丸めのため、分子が負のまま素通しすると誤った出力長を
/// 返しうる）。全項を `usize` の `checked_*` 演算で連鎖する
/// （`tensor_core::ops_shape::conv_out_len` と同型。呼び出し元
/// `validate_and_shape` は本関数の呼び出し前に `validate_ge_one` で
/// `kernel`／`stride`／`dilation` が `>= 1` であることを保証するが、
/// `padding` は上限のみ検査され巨大な `usize` 値をそのまま受け取り
/// うるため、生の `i64` 演算〈本関数の旧実装〉では `2 * padding` や
/// `dilation * (kernel - 1)` が i64 乗算オーバーフローしうる。
/// `checked_mul`／`checked_add`／`checked_sub` の連鎖により、
/// オーバーフローは `usize` の桁あふれとして `None` に写像され
/// `InvalidPoolingShape` へ fail-closed で変換される）。
fn pool_out_len(
    in_len: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<usize, CudaError> {
    let overflow_err = || CudaError::InvalidPoolingShape {
        detail: format!(
            "pool_out_len: element count overflow (in={in_len}, kernel={kernel}, \
             stride={stride}, padding={padding}, dilation={dilation})"
        ),
    };

    let two_p = padding.checked_mul(2).ok_or_else(overflow_err)?;
    let in_plus_2p = in_len.checked_add(two_p).ok_or_else(overflow_err)?;
    // `validate_ge_one` により `kernel >= 1` が呼び出し前に保証されて
    // いるため `kernel - 1` は桁あふれしない。
    let k_minus_1 = kernel - 1;
    let dk = dilation.checked_mul(k_minus_1).ok_or_else(overflow_err)?;

    let numerator = match in_plus_2p.checked_sub(dk).and_then(|v| v.checked_sub(1)) {
        Some(numerator) => numerator,
        None => {
            return Err(CudaError::InvalidPoolingShape {
                detail: format!(
                    "pool_out_len: negative numerator (in={in_len}, kernel={kernel}, \
                     stride={stride}, padding={padding}, dilation={dilation})"
                ),
            });
        }
    };
    // `stride >= 1` は呼び出し前の `validate_ge_one` により保証済み。
    let out = numerator / stride + 1;
    if out < 1 {
        return Err(CudaError::InvalidPoolingShape {
            detail: format!("pool_out_len: computed non-positive output length {out}"),
        });
    }
    Ok(out)
}

/// `kernel`／`stride`／`dilation` の各成分が `>= 1` であることを検査
/// する（設計 doc §3）。
fn validate_ge_one(name: &str, values: [usize; 2]) -> Result<(), CudaError> {
    if values[0] == 0 || values[1] == 0 {
        return Err(CudaError::InvalidPoolingShape {
            detail: format!("{name} components must be >= 1: {values:?}"),
        });
    }
    Ok(())
}

/// `padding <= floor(kernel / 2)`（dilation に依存しない。設計 doc §3
/// の導出済み結論）を検査する。
fn validate_padding(kernel: [usize; 2], padding: [usize; 2]) -> Result<(), CudaError> {
    for i in 0..2 {
        let limit = kernel[i] / 2;
        if padding[i] > limit {
            return Err(CudaError::InvalidPoolingShape {
                detail: format!(
                    "padding[{i}]={} exceeds floor(kernel[{i}]/2)={limit}",
                    padding[i]
                ),
            });
        }
    }
    Ok(())
}

/// 空間軸（`H`／`W`）を `>= 1` に限定する（設計 doc §3。`N`／`C` は
/// 対象外）。
fn validate_spatial_nonzero(h_in: usize, w_in: usize) -> Result<(), CudaError> {
    if h_in == 0 || w_in == 0 {
        return Err(CudaError::InvalidPoolingShape {
            detail: format!("spatial axes must be >= 1: h_in={h_in}, w_in={w_in}"),
        });
    }
    Ok(())
}

/// `dilation` による空窓（`kernel=2 かつ dilation > H`）を拒否する
/// （設計 doc §3「導出（証明のスケッチ）」）。
fn validate_no_dilation_empty_window(
    kernel: [usize; 2],
    dilation: [usize; 2],
    h_in: usize,
    w_in: usize,
) -> Result<(), CudaError> {
    if kernel[0] == 2 && dilation[0] > h_in {
        return Err(CudaError::InvalidPoolingShape {
            detail: format!(
                "dilation[0]={} exceeds h_in={h_in} with kernel[0]=2 (empty window)",
                dilation[0]
            ),
        });
    }
    if kernel[1] == 2 && dilation[1] > w_in {
        return Err(CudaError::InvalidPoolingShape {
            detail: format!(
                "dilation[1]={} exceeds w_in={w_in} with kernel[1]=2 (empty window)",
                dilation[1]
            ),
        });
    }
    Ok(())
}

/// `MaxPool2d`／`AvgPool2d` 共通の shape・パラメータ検査（設計 doc
/// §3・§4）。検査を通過した場合、出力 shape `[N, C, Hout, Wout]` を
/// 返す。`AvgPool` は呼び出し元が `dilation=[1,1]` を渡す契約
/// （設計 doc §3「Avg は `dilation=1` 固定」）。
fn validate_and_shape(
    in_shape: [usize; 4],
    kernel: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
) -> Result<[usize; 4], CudaError> {
    let [n, c, h_in, w_in] = in_shape;
    validate_ge_one("kernel", kernel)?;
    validate_ge_one("stride", stride)?;
    validate_ge_one("dilation", dilation)?;
    validate_padding(kernel, padding)?;
    validate_spatial_nonzero(h_in, w_in)?;
    validate_no_dilation_empty_window(kernel, dilation, h_in, w_in)?;
    let h_out = pool_out_len(h_in, kernel[0], stride[0], padding[0], dilation[0])?;
    let w_out = pool_out_len(w_in, kernel[1], stride[1], padding[1], dilation[1])?;
    Ok([n, c, h_out, w_out])
}

/// `AdaptiveAvgPool2d` の shape 検査（設計 doc §3「adaptive の
/// `output_size >= 1`」・空間軸ゼロ拒否）。
fn validate_and_shape_adaptive(
    in_shape: [usize; 4],
    output_size: [usize; 2],
) -> Result<[usize; 4], CudaError> {
    let [n, c, h_in, w_in] = in_shape;
    validate_spatial_nonzero(h_in, w_in)?;
    if output_size[0] == 0 || output_size[1] == 0 {
        return Err(CudaError::InvalidPoolingShape {
            detail: format!("adaptive output_size components must be >= 1: {output_size:?}"),
        });
    }
    Ok([n, c, output_size[0], output_size[1]])
}

/// MaxPool の索引値域が `Tensor<i32>` に収まること（`H * W <=
/// i32::MAX`。設計 doc §3「MaxPool の索引値域は `H・W ≤ i32::MAX`」）
/// を検査する。
fn validate_index_domain(h_in: usize, w_in: usize) -> Result<(), CudaError> {
    let hw = h_in
        .checked_mul(w_in)
        .ok_or_else(|| CudaError::PoolingSizeLimitExceeded {
            detail: format!("h_in*w_in overflow: h_in={h_in}, w_in={w_in}"),
        })?;
    validate_i32_bound(hw, "h_in*w_in")?;
    Ok(())
}

/// `MaxPool2d`／`AvgPool2d`／`AdaptiveAvgPool2d` の起動引数（[N, C,
/// H_in, W_in, H_out, W_out] と該当する kernel/stride/padding/dilation
/// 成分）を `i32` 範囲検査済みで保持する共通構造体（`im2col.rs::
/// LaunchShape` と同型）。
#[derive(Debug)]
struct LaunchShape {
    n_batch: i32,
    c: i32,
    h_in: i32,
    w_in: i32,
    h_out: i32,
    w_out: i32,
}

impl LaunchShape {
    fn derive(in_shape: [usize; 4], out_shape: [usize; 4]) -> Result<Self, CudaError> {
        let [n, c, h_in, w_in] = in_shape;
        let [_, _, h_out, w_out] = out_shape;
        Ok(Self {
            n_batch: validate_i32_bound(n, "n_batch")?,
            c: validate_i32_bound(c, "c")?,
            h_in: validate_i32_bound(h_in, "h_in")?,
            w_in: validate_i32_bound(w_in, "w_in")?,
            h_out: validate_i32_bound(h_out, "h_out")?,
            w_out: validate_i32_bound(w_out, "w_out")?,
        })
    }
}

/// [`CudaPooling::run_max_pool2d_f32`] の戻り値（値・索引・出力
/// shape）。clippy `type_complexity` 回避のための型エイリアス
/// （3 要素タプルの意味は関数 doc を参照）。
type MaxPool2dOutput = (Vec<f32>, Vec<i32>, [usize; 4]);

/// `max_pool2d_f32`／`avg_pool2d_f32`／`adaptive_avg_pool2d_f32`
/// カーネルのコンパイル済みハンドルを保持する。
pub struct CudaPooling {
    stream: Arc<CudaStream>,
    /// `im2col.rs::CudaIm2col::ordinal` と同じ役割（[`Self::
    /// with_driver_call`] が `context_cache::with_driver_call` を呼ぶ
    /// 際のキー）。
    ordinal: usize,
    max_pool2d_f32: CudaFunction,
    avg_pool2d_f32: CudaFunction,
    adaptive_avg_pool2d_f32: CudaFunction,
}

impl CudaPooling {
    /// `device` 上で 3 カーネルを NVRTC コンパイルし保持するハンドルを
    /// 構築する（`im2col.rs::CudaIm2col::new` と同一手順）。
    pub fn new(device: &CudaDevice) -> Result<Self, CudaError> {
        let arch = device.arch();

        macro_rules! compile_and_load {
            ($src:expr, $name:literal) => {{
                let ptx = compile_ptx($src, arch)?;
                device.context().load_module(ptx)?.load_function($name)?
            }};
        }

        let max_pool2d_f32 = compile_and_load!(kernels_pooling::MAX_POOL2D_F32, "max_pool2d_f32");
        let avg_pool2d_f32 = compile_and_load!(kernels_pooling::AVG_POOL2D_F32, "avg_pool2d_f32");
        let adaptive_avg_pool2d_f32 = compile_and_load!(
            kernels_pooling::ADAPTIVE_AVG_POOL2D_F32,
            "adaptive_avg_pool2d_f32"
        );

        Ok(Self {
            stream: device.stream().clone(),
            ordinal: device.ordinal(),
            max_pool2d_f32,
            avg_pool2d_f32,
            adaptive_avg_pool2d_f32,
        })
    }

    /// `CudaPooling` の driver 呼び出しを CUDA Graph capture 排他へ
    /// 参加させる共通ヘルパー（`im2col.rs::CudaIm2col::
    /// with_driver_call` と同じ設計）。
    fn with_driver_call<T>(
        &self,
        f: impl FnOnce() -> Result<T, CudaError>,
    ) -> Result<T, CudaError> {
        context_cache::with_driver_call(self.ordinal, f)
    }

    /// MaxPool2d（設計 doc §5・§6。索引は (n, c) 平面内の flat 添字
    /// `h * w_in + w`）。`in_shape: [N, Cin, H, W]`。戻り値は
    /// `(values, indices, out_shape)`。
    ///
    /// `out_shape` の要素数が 0 の場合（`N=0`／`C=0`。空間軸ゼロは
    /// `validate_and_shape` が事前に拒否する）、`input` を読む必要が
    /// 一切ないため空 `Vec` を早期に返す（`im2col.rs::
    /// run_im2col_f32` と同じ理由）。
    pub fn run_max_pool2d_f32(
        &self,
        input: &[f32],
        in_shape: [usize; 4],
        kernel: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> Result<MaxPool2dOutput, CudaError> {
        let out_shape = validate_and_shape(in_shape, kernel, stride, padding, dilation)?;
        validate_index_domain(in_shape[2], in_shape[3])?;

        let numel_out = checked_numel(&out_shape)?;
        if numel_out == 0 {
            return Ok((Vec::new(), Vec::new(), out_shape));
        }

        let numel_in = checked_numel(&in_shape)?;
        if input.len() != numel_in {
            return Err(CudaError::InvalidPoolingShape {
                detail: format!(
                    "max_pool2d: input.len()={} does not match in numel={numel_in}",
                    input.len()
                ),
            });
        }

        let shape = LaunchShape::derive(in_shape, out_shape)?;
        let kh = validate_i32_bound(kernel[0], "kh")?;
        let kw = validate_i32_bound(kernel[1], "kw")?;
        let sh = validate_i32_bound(stride[0], "sh")?;
        let sw = validate_i32_bound(stride[1], "sw")?;
        let ph = validate_i32_bound(padding[0], "ph")?;
        let pw = validate_i32_bound(padding[1], "pw")?;
        let dh = validate_i32_bound(dilation[0], "dh")?;
        let dw = validate_i32_bound(dilation[1], "dw")?;
        let numel_i = validate_i32_bound(numel_out, "numel_out")?;

        self.with_driver_call(|| {
            let in_dev = self.stream.clone_htod(input)?;
            let mut out_dev = self.stream.alloc_zeros::<f32>(numel_out)?;
            let mut idx_dev = self.stream.alloc_zeros::<i32>(numel_out)?;

            let cfg = LaunchConfig {
                grid_dim: ((numel_out as u32).div_ceil(POOLING_BLOCK_DIM), 1, 1),
                block_dim: (POOLING_BLOCK_DIM, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: `in_dev` は `numel_in` 要素の H2D 済みバッファ
            // （直上の長さ検証でホストスライスと一致することを確認
            // 済み）。`out_dev`／`idx_dev` はいずれも `numel_out` 要素
            // 確保済みでカーネルは `idx < numel`（REQ-8）を維持した
            // まま各出力要素を 1 回だけ書く。`in` への読み出し添字は
            // カーネル内で `0 <= h < h_in`・`0 <= w < w_in` を検査
            // してから計算するため範囲外読み出しはない
            // （`kernels_pooling.rs::MAX_POOL2D_F32` 参照）。
            unsafe {
                self.stream
                    .launch_builder(&self.max_pool2d_f32)
                    .arg(&in_dev)
                    .arg(&mut out_dev)
                    .arg(&mut idx_dev)
                    .arg(&shape.n_batch)
                    .arg(&shape.c)
                    .arg(&shape.h_in)
                    .arg(&shape.w_in)
                    .arg(&shape.h_out)
                    .arg(&shape.w_out)
                    .arg(&kh)
                    .arg(&kw)
                    .arg(&sh)
                    .arg(&sw)
                    .arg(&ph)
                    .arg(&pw)
                    .arg(&dh)
                    .arg(&dw)
                    .arg(&numel_i)
                    .launch(cfg)?;
            }
            let values = readback::<f32, _>(&self.stream, &out_dev)?;
            let indices = readback::<i32, _>(&self.stream, &idx_dev)?;
            Ok((values, indices, out_shape))
        })
    }

    /// AvgPool2d（設計 doc §7。`dilation` は常に `[1, 1]`）。
    /// `in_shape: [N, Cin, H, W]`。戻り値は `(values, out_shape)`。
    pub fn run_avg_pool2d_f32(
        &self,
        input: &[f32],
        in_shape: [usize; 4],
        kernel: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
    ) -> Result<(Vec<f32>, [usize; 4]), CudaError> {
        let out_shape = validate_and_shape(in_shape, kernel, stride, padding, [1, 1])?;

        let numel_out = checked_numel(&out_shape)?;
        if numel_out == 0 {
            return Ok((Vec::new(), out_shape));
        }

        let numel_in = checked_numel(&in_shape)?;
        if input.len() != numel_in {
            return Err(CudaError::InvalidPoolingShape {
                detail: format!(
                    "avg_pool2d: input.len()={} does not match in numel={numel_in}",
                    input.len()
                ),
            });
        }

        let shape = LaunchShape::derive(in_shape, out_shape)?;
        let kh = validate_i32_bound(kernel[0], "kh")?;
        let kw = validate_i32_bound(kernel[1], "kw")?;
        let sh = validate_i32_bound(stride[0], "sh")?;
        let sw = validate_i32_bound(stride[1], "sw")?;
        let ph = validate_i32_bound(padding[0], "ph")?;
        let pw = validate_i32_bound(padding[1], "pw")?;
        let numel_i = validate_i32_bound(numel_out, "numel_out")?;
        let count_include_pad_i: i32 = i32::from(count_include_pad);

        self.with_driver_call(|| {
            let in_dev = self.stream.clone_htod(input)?;
            let mut out_dev = self.stream.alloc_zeros::<f32>(numel_out)?;

            let cfg = LaunchConfig {
                grid_dim: ((numel_out as u32).div_ceil(POOLING_BLOCK_DIM), 1, 1),
                block_dim: (POOLING_BLOCK_DIM, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: `in_dev` は `numel_in` 要素の H2D 済みバッファ・
            // `out_dev` は `numel_out` 要素確保済み。カーネルは
            // `idx < numel`（REQ-8）を維持し、`in` への読み出しは
            // `0 <= h < h_in`・`0 <= w < w_in` を検査してから行う
            // （`kernels_pooling.rs::AVG_POOL2D_F32` 参照）。
            unsafe {
                self.stream
                    .launch_builder(&self.avg_pool2d_f32)
                    .arg(&in_dev)
                    .arg(&mut out_dev)
                    .arg(&shape.n_batch)
                    .arg(&shape.c)
                    .arg(&shape.h_in)
                    .arg(&shape.w_in)
                    .arg(&shape.h_out)
                    .arg(&shape.w_out)
                    .arg(&kh)
                    .arg(&kw)
                    .arg(&sh)
                    .arg(&sw)
                    .arg(&ph)
                    .arg(&pw)
                    .arg(&count_include_pad_i)
                    .arg(&numel_i)
                    .launch(cfg)?;
            }
            let values = readback::<f32, _>(&self.stream, &out_dev)?;
            Ok((values, out_shape))
        })
    }

    /// AdaptiveAvgPool2d（設計 doc §4／§7。padding は存在しない）。
    /// `in_shape: [N, Cin, H, W]`。戻り値は `(values, out_shape)`。
    pub fn run_adaptive_avg_pool2d_f32(
        &self,
        input: &[f32],
        in_shape: [usize; 4],
        output_size: [usize; 2],
    ) -> Result<(Vec<f32>, [usize; 4]), CudaError> {
        let out_shape = validate_and_shape_adaptive(in_shape, output_size)?;

        let numel_out = checked_numel(&out_shape)?;
        if numel_out == 0 {
            return Ok((Vec::new(), out_shape));
        }

        let numel_in = checked_numel(&in_shape)?;
        if input.len() != numel_in {
            return Err(CudaError::InvalidPoolingShape {
                detail: format!(
                    "adaptive_avg_pool2d: input.len()={} does not match in numel={numel_in}",
                    input.len()
                ),
            });
        }

        let shape = LaunchShape::derive(in_shape, out_shape)?;
        let numel_i = validate_i32_bound(numel_out, "numel_out")?;

        self.with_driver_call(|| {
            let in_dev = self.stream.clone_htod(input)?;
            let mut out_dev = self.stream.alloc_zeros::<f32>(numel_out)?;

            let cfg = LaunchConfig {
                grid_dim: ((numel_out as u32).div_ceil(POOLING_BLOCK_DIM), 1, 1),
                block_dim: (POOLING_BLOCK_DIM, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: `in_dev` は `numel_in` 要素の H2D 済みバッファ・
            // `out_dev` は `numel_out` 要素確保済み。カーネルは
            // `idx < numel`（REQ-8）を維持し、`in` への読み出しは
            // 窓 `[h_start, h_end) x [w_start, w_end)` が常に
            // `[0, h_in) x [0, w_in)` の部分集合であること（`§4` の
            // 窓式の性質）を根拠に境界内に収まる
            // （`kernels_pooling.rs::ADAPTIVE_AVG_POOL2D_F32` 参照）。
            unsafe {
                self.stream
                    .launch_builder(&self.adaptive_avg_pool2d_f32)
                    .arg(&in_dev)
                    .arg(&mut out_dev)
                    .arg(&shape.n_batch)
                    .arg(&shape.c)
                    .arg(&shape.h_in)
                    .arg(&shape.w_in)
                    .arg(&shape.h_out)
                    .arg(&shape.w_out)
                    .arg(&numel_i)
                    .launch(cfg)?;
            }
            let values = readback::<f32, _>(&self.stream, &out_dev)?;
            Ok((values, out_shape))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_i32_bound_rejects_exceeding_i32_max() {
        let err = validate_i32_bound(i32::MAX as usize + 1, "h_in").unwrap_err();
        assert!(matches!(err, CudaError::PoolingSizeLimitExceeded { .. }));
    }

    #[test]
    fn validate_i32_bound_accepts_i32_max() {
        assert!(validate_i32_bound(i32::MAX as usize, "h_in").is_ok());
    }

    #[test]
    fn checked_numel_rejects_overflow() {
        let err = checked_numel(&[usize::MAX, 2, 1, 1]).unwrap_err();
        assert!(matches!(err, CudaError::InvalidPoolingShape { .. }));
    }

    #[test]
    fn checked_numel_computes_product() {
        assert_eq!(checked_numel(&[1, 2, 3, 4]).unwrap(), 24);
    }

    /// 設計 doc §13 の境界例: `kernel=2, dilation=1, padding=1`
    /// （許可）／`padding=2`（拒否）。
    #[test]
    fn padding_boundary_kernel2_dilation1() {
        assert!(validate_padding([2, 2], [1, 1]).is_ok());
        assert!(validate_padding([2, 2], [2, 2]).is_err());
    }

    /// 設計 doc §13 の境界例: `kernel=3, dilation=2, padding=1`
    /// （許可）／`padding=2`（拒否。`effective_kernel_size` のみの
    /// 検査だけでは許可されてしまう反例）。
    #[test]
    fn padding_boundary_kernel3_dilation2() {
        // padding は dilation に依存しないため kernel のみで決まる。
        assert!(validate_padding([3, 3], [1, 1]).is_ok());
        assert!(validate_padding([3, 3], [2, 2]).is_err());
    }

    /// 設計 doc §4 の負分子拒否: `in=1, k=2, s=2, p=0, d=1` は
    /// `ShapeError` 相当（ゼロ方向丸めで誤った出力長 1 を返さない）。
    #[test]
    fn pool_out_len_rejects_negative_numerator() {
        let err = pool_out_len(1, 2, 2, 0, 1).unwrap_err();
        assert!(matches!(err, CudaError::InvalidPoolingShape { .. }));
    }

    #[test]
    fn pool_out_len_computes_expected_value() {
        // in=9, k=3, s=2, p=1, d=2 -> floor((9+2-2*2-1)/2)+1 = floor(6/2)+1=4.
        assert_eq!(pool_out_len(9, 3, 2, 1, 2).unwrap(), 4);
    }

    /// Review 指摘（イシュー #1729）: `pool_out_len` が生の `i64` 演算
    /// （`2 * padding_i` 等）を使っていた旧実装では、`validate_padding`
    /// による上限検査を経ない巨大な `padding`（`usize` としては有効な
    /// 値）を渡すと `i64` 乗算オーバーフローで panic
    /// （debug ビルド）／誤った出力長へサイレントにラップ
    /// （release ビルド）しうった。`checked_mul`／`checked_add` へ
    /// 是正した後は、同じ巨大 `padding` に対して panic せず
    /// `InvalidPoolingShape` を fail-closed に返すことを検証する
    /// （`tensor_core::ops_shape::conv_out_len` と同じ `checked_*`
    /// 連鎖の複製であることの回帰）。
    #[test]
    fn pool_out_len_rejects_overflowing_padding_without_panicking() {
        let err = pool_out_len(4, 3, 1, usize::MAX / 2, 1).unwrap_err();
        assert!(matches!(err, CudaError::InvalidPoolingShape { .. }));
    }

    /// 上記と対称に、`dilation * (kernel - 1)` 側の乗算オーバーフロー
    /// も fail-closed に検出することを検証する。
    #[test]
    fn pool_out_len_rejects_overflowing_dilation_kernel_product_without_panicking() {
        let err = pool_out_len(4, usize::MAX, 1, 0, usize::MAX / 2).unwrap_err();
        assert!(matches!(err, CudaError::InvalidPoolingShape { .. }));
    }

    /// 設計 doc §3 の空間軸ゼロ拒否（Max／Avg 一律。`N`／`C` は対象外）。
    #[test]
    fn validate_spatial_nonzero_rejects_zero_h_or_w() {
        assert!(validate_spatial_nonzero(0, 4).is_err());
        assert!(validate_spatial_nonzero(4, 0).is_err());
        assert!(validate_spatial_nonzero(4, 4).is_ok());
    }

    /// 設計 doc §3 の `dilation` 空窓拒否: `in=1, kernel=2, stride=1,
    /// padding=1, dilation=2` と `in=1, kernel=2, stride=2, padding=1,
    /// dilation=2` はともに拒否・`in=2` は許可される境界。
    #[test]
    fn validate_and_shape_rejects_dilation_empty_window() {
        let err1 = validate_and_shape([1, 1, 1, 1], [2, 2], [1, 1], [1, 1], [2, 2]).unwrap_err();
        assert!(matches!(err1, CudaError::InvalidPoolingShape { .. }));
        let err2 = validate_and_shape([1, 1, 1, 1], [2, 2], [2, 2], [1, 1], [2, 2]).unwrap_err();
        assert!(matches!(err2, CudaError::InvalidPoolingShape { .. }));

        // 境界の反例: H = dilation (=2) では検査対象外・許可される。
        let out = validate_and_shape([1, 1, 2, 2], [2, 2], [1, 1], [1, 1], [2, 2]).unwrap();
        assert_eq!(out, [1, 1, 2, 2]);
    }

    /// N=0（空バッチ）は拒否されず出力 shape `[0,C,out_h,out_w]` が
    /// 得られる（設計 doc §3 の N/C 対象外方針）。
    #[test]
    fn validate_and_shape_allows_empty_batch() {
        let out = validate_and_shape([0, 3, 4, 4], [2, 2], [2, 2], [0, 0], [1, 1]).unwrap();
        assert_eq!(out, [0, 3, 2, 2]);
    }

    /// adaptive の `output_size` は入力より大きくてもよい（拡大側も
    /// 成立。設計 doc §3）。
    #[test]
    fn validate_and_shape_adaptive_allows_upsampling() {
        let out = validate_and_shape_adaptive([1, 1, 2, 2], [4, 4]).unwrap();
        assert_eq!(out, [1, 1, 4, 4]);
    }

    #[test]
    fn validate_and_shape_adaptive_rejects_zero_output_size() {
        assert!(validate_and_shape_adaptive([1, 1, 2, 2], [0, 4]).is_err());
    }

    /// 1d 併合形状（`[N, C, 1, L]`。`kernel=[1, k]`）でも 2d と同じ経路
    /// で正しく検査・shape 導出される（設計 doc §2）。
    #[test]
    fn validate_and_shape_handles_1d_merged_shape() {
        // L=9, k=3, s=2, p=1, d=2 -> lout=4（H 軸は kernel=1/stride=1/
        // padding=0/dilation=1 で常に 1）。
        let out = validate_and_shape([1, 1, 1, 9], [1, 3], [1, 2], [0, 1], [1, 2]).unwrap();
        assert_eq!(out, [1, 1, 1, 4]);
    }
}

/// `CudaPooling` の実機（DGX Spark GB10 等）検証（イシュー #1729）。
/// `context_cache.rs::poison_recovery_real_device_tests` と同じ配置
/// 理由: `CudaPooling`（本モジュール private）へ `tests/`（統合
/// テスト・クレート外部扱い）からは到達できないため、本ファイルを
/// `#[cfg(test)] #[path]` で子モジュールとして登録する
/// （`pooling_real_device_tests.rs` モジュール doc 参照）。
#[cfg(test)]
#[path = "pooling_real_device_tests.rs"]
mod pooling_real_device_tests;
