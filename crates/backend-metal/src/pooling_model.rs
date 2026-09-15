//! MaxPool／AvgPool／AdaptiveAvgPool（1d／2d。1d は呼び出し元が
//! `[N,C,1,L]` へ reshape 併合する契約——本モジュールは常に rank 4
//! を扱う）の起動前形状検証・ホスト側逐語モデル（イシュー #1730・
//! 親 #1607・設計 `docs/pooling-ops-design.md`）。
//!
//! `im2col_model.rs`／`batch_norm_model.rs` と同じ「`cfg(target_os =
//! "macos")` を付けない純関数層」の設計判断を踏襲する（`objc2` 系
//! FFI に触れないため、本実装環境（Linux・CI）でも単体テストが回る）。
//! [`derive_pool_dims`]／[`derive_adaptive_dims`] は `crate::pooling`
//! （macOS 限定の起動 API）から呼ばれる純関数で、`pub` な起動 API を
//! 経由せず直接呼ばれても安全なよう独立に形状を再検証する
//! （`.claude/rules/security.md` A08・`im2col.rs` と同じ多層防御）。
//!
//! # なぜこれが Metal 側の正しさの根拠になるか
//!
//! [`max_pool2d_model`]／[`avg_pool2d_soft_f64`]／
//! [`adaptive_avg_pool2d_soft_f64`] は `shaders/pooling.metal` の 3
//! カーネルの逐語移植（走査順序・soft-f64 演算列とも本ファイルの
//! `crate::soft_f64` 呼び出しと 1 対 1 対応する）である。本 PR の
//! 実行環境（Linux）には Apple Silicon 実機が無く GPU カーネル自体は
//! 未実測のため、本モジュールの単体テストが「ホスト `f64` 参照実装
//! との bit 完全一致」を Linux 上で機械的に裏付ける唯一の根拠となる
//! （`shaders/pooling.metal` 冒頭コメント「数値契約」参照）。

use crate::soft_f64::{add_f64_bits, div_f64_bits, narrow_f64_bits, widen_f32_bits};

/// [`derive_pool_dims`]／[`derive_adaptive_dims`] の型付きエラー
/// （`crate::im2col_model::Im2colPrepareError`・`crate::
/// batch_norm_model::BatchNormPrepareError` と同型の「小さな enum」
/// 方針）。
#[non_exhaustive]
#[derive(Debug)]
pub enum PoolingPrepareError {
    /// rank 不一致・空間軸ゼロ長・カーネル／ストライド／dilation が
    /// 0・`ceil_mode == true`・padding が `floor(kernel/2)` を超過・
    /// 空窓構成（`kernel == 2 && dilation > 入力長`）・出力長の分子が
    /// 負・shape 積の `usize` overflow・ホストスライス実長の不一致等。
    InvalidShape { detail: String },
    /// 形状パラメータの導出値（`n`／`c`／`h_in`／`w_in`／`h_out`／
    /// `w_out`／`kh`／`kw`／`kh*kw`／`numel_out`／`n*c*h_in*w_in`）の
    /// いずれかがカーネル引数の `uint`（`u32::MAX`）上限、または
    /// MaxPool 索引契約の `plane_in <= i32::MAX` を超過する
    /// （`InvalidShape`〈内部契約違反〉とは区別する: `ops.rs` 側が
    /// 存在すれば本 variant のみ `BackendError::Unsupported`
    /// へ写像しホストフォールバックへ委ねる設計。`crate::im2col_model::
    /// Im2colPrepareError::SizeLimitExceeded` と同型の判断）。
    SizeLimitExceeded { detail: String },
}

impl std::fmt::Display for PoolingPrepareError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PoolingPrepareError::InvalidShape { detail } => {
                write!(f, "pooling invalid shape: {detail}")
            }
            PoolingPrepareError::SizeLimitExceeded { detail } => {
                write!(f, "pooling size limit exceeded: {detail}")
            }
        }
    }
}

impl std::error::Error for PoolingPrepareError {}

/// `shaders/pooling.metal::struct PoolDims` とバイトレイアウトを一致
/// させる（`#[repr(C)]`・全フィールド `u32`。`size_of` 一致を
/// [`tests::pool_dims_size_matches_msl_struct`] が固定する）。
///
/// adaptive 系（[`derive_adaptive_dims`]）では `kh`〜`dw`・
/// `count_include_pad` を `0` で埋める（`adaptive_avg_pool2d_f32`
/// カーネルはこれらのフィールドを読まない）。
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolDims {
    pub n: u32,
    pub c: u32,
    pub h_in: u32,
    pub w_in: u32,
    pub h_out: u32,
    pub w_out: u32,
    pub kh: u32,
    pub kw: u32,
    pub sh: u32,
    pub sw: u32,
    pub ph: u32,
    pub pw: u32,
    pub dh: u32,
    pub dw: u32,
    pub count_include_pad: u32,
    pub numel_out: u32,
    pub plane_in: u32,
    pub plane_out: u32,
}

/// `v` が `u32::MAX` を超えないことを検証して `u32` へ変換する
/// （超過時 [`PoolingPrepareError::SizeLimitExceeded`]）。
fn to_u32_checked(v: usize, what: &str) -> Result<u32, PoolingPrepareError> {
    u32::try_from(v).map_err(|_| PoolingPrepareError::SizeLimitExceeded {
        detail: format!("{what}={v} exceeds u32::MAX"),
    })
}

/// `in_len + 2p - d(k-1) - 1` を `i128`（overflow 耐性）で計算し、
/// 負なら [`PoolingPrepareError::InvalidShape`]（floor 契約の実装
/// 手段。`docs/pooling-ops-design.md` §3 参照）、非負なら `/s + 1`
/// で出力長を返す。
fn compute_out_len(
    in_len: usize,
    k: usize,
    s: usize,
    p: usize,
    d: usize,
    axis: &str,
) -> Result<usize, PoolingPrepareError> {
    let numerator: i128 = in_len as i128 + 2 * (p as i128) - (d as i128) * ((k as i128) - 1) - 1;
    if numerator < 0 {
        return Err(PoolingPrepareError::InvalidShape {
            detail: format!(
                "{axis}: output length numerator is negative (in={in_len}, k={k}, s={s}, p={p}, d={d})"
            ),
        });
    }
    Ok((numerator as usize) / s + 1)
}

/// `MaxPool2d`／`AvgPool2d` の起動前検証・[`PoolDims`] 導出
/// （`docs/pooling-ops-design.md` §3／§4 の全ゲートを逐語実装）。
/// `in_shape` は `[N, C, H, W]`（rank 4 以外は `InvalidShape`。1d は
/// 呼び出し元が `[N, C, 1, L]` へ reshape 併合してから渡す契約）。
#[allow(clippy::too_many_arguments)]
pub fn derive_pool_dims(
    in_shape: &[usize],
    kernel: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    ceil_mode: bool,
    count_include_pad: bool,
) -> Result<PoolDims, PoolingPrepareError> {
    if in_shape.len() != 4 {
        return Err(PoolingPrepareError::InvalidShape {
            detail: format!("expected rank-4 input shape, got rank {}", in_shape.len()),
        });
    }
    let (n, c, h_in, w_in) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
    if h_in == 0 || w_in == 0 {
        return Err(PoolingPrepareError::InvalidShape {
            detail: format!("spatial axes must be non-zero, got h_in={h_in}, w_in={w_in}"),
        });
    }
    let (kh, kw) = kernel;
    let (sh, sw) = stride;
    let (ph, pw) = padding;
    let (dh, dw) = dilation;
    if kh == 0 || kw == 0 || sh == 0 || sw == 0 || dh == 0 || dw == 0 {
        return Err(PoolingPrepareError::InvalidShape {
            detail: format!(
                "kernel/stride/dilation must be non-zero, got kernel=({kh},{kw}) stride=({sh},{sw}) dilation=({dh},{dw})"
            ),
        });
    }
    if ceil_mode {
        return Err(PoolingPrepareError::InvalidShape {
            detail: "ceil_mode=true is not supported".to_string(),
        });
    }
    if ph > kh / 2 || pw > kw / 2 {
        return Err(PoolingPrepareError::InvalidShape {
            detail: format!(
                "padding must not exceed floor(kernel/2), got padding=({ph},{pw}) kernel=({kh},{kw})"
            ),
        });
    }
    // 空窓構成: kernel==2 かつ dilation > 入力長の場合、padding が
    // 分子を非負にしても実際のタップ 2 個がいずれも padding 範囲
    // （入力外）に落ちる（`docs/pooling-ops-design.md` §3 の反例
    // `in=2, k=2, p=1, d=3` 参照）。
    if kh == 2 && dh > h_in {
        return Err(PoolingPrepareError::InvalidShape {
            detail: format!("empty window: kernel==2 && dilation({dh}) > h_in({h_in})"),
        });
    }
    if kw == 2 && dw > w_in {
        return Err(PoolingPrepareError::InvalidShape {
            detail: format!("empty window: kernel==2 && dilation({dw}) > w_in({w_in})"),
        });
    }

    let h_out = compute_out_len(h_in, kh, sh, ph, dh, "h")?;
    let w_out = compute_out_len(w_in, kw, sw, pw, dw, "w")?;

    let plane_in = h_in
        .checked_mul(w_in)
        .ok_or_else(|| PoolingPrepareError::InvalidShape {
            detail: "h_in * w_in overflows usize".to_string(),
        })?;
    let plane_out = h_out
        .checked_mul(w_out)
        .ok_or_else(|| PoolingPrepareError::InvalidShape {
            detail: "h_out * w_out overflows usize".to_string(),
        })?;
    let numel_out = n
        .checked_mul(c)
        .and_then(|v| v.checked_mul(plane_out))
        .ok_or_else(|| PoolingPrepareError::InvalidShape {
            detail: "n * c * h_out * w_out overflows usize".to_string(),
        })?;
    let numel_in = n
        .checked_mul(c)
        .and_then(|v| v.checked_mul(plane_in))
        .ok_or_else(|| PoolingPrepareError::InvalidShape {
            detail: "n * c * h_in * w_in overflows usize".to_string(),
        })?;
    let kernel_area = kh
        .checked_mul(kw)
        .ok_or_else(|| PoolingPrepareError::InvalidShape {
            detail: "kh * kw overflows usize".to_string(),
        })?;

    // MaxPool 索引契約: 平面添字（`h*w_in + w`）は `i32` へ収める
    // （`shaders/pooling.metal::max_pool2d_f32` の `idx` 出力型）。
    if plane_in > i32::MAX as usize {
        return Err(PoolingPrepareError::SizeLimitExceeded {
            detail: format!("plane_in={plane_in} exceeds i32::MAX (MaxPool index contract)"),
        });
    }

    Ok(PoolDims {
        n: to_u32_checked(n, "n")?,
        c: to_u32_checked(c, "c")?,
        h_in: to_u32_checked(h_in, "h_in")?,
        w_in: to_u32_checked(w_in, "w_in")?,
        h_out: to_u32_checked(h_out, "h_out")?,
        w_out: to_u32_checked(w_out, "w_out")?,
        kh: to_u32_checked(kh, "kh")?,
        kw: to_u32_checked(kw, "kw")?,
        sh: to_u32_checked(sh, "sh")?,
        sw: to_u32_checked(sw, "sw")?,
        ph: to_u32_checked(ph, "ph")?,
        pw: to_u32_checked(pw, "pw")?,
        dh: to_u32_checked(dh, "dh")?,
        dw: to_u32_checked(dw, "dw")?,
        count_include_pad: u32::from(count_include_pad),
        numel_out: to_u32_checked(numel_out, "numel_out")?,
        plane_in: {
            let _ = to_u32_checked(kernel_area, "kh*kw")?;
            let _ = to_u32_checked(numel_in, "n*c*h_in*w_in")?;
            to_u32_checked(plane_in, "plane_in")?
        },
        plane_out: to_u32_checked(plane_out, "plane_out")?,
    })
}

/// `AdaptiveAvgPool2d` の起動前検証・[`PoolDims`] 導出。`output_size`
/// は入力長以下であることを要求しない（PyTorch と同じ契約）。窓は
/// カーネル呼び出し時に各出力位置ごとに動的計算するため `kh`〜`dw`・
/// `count_include_pad` は `0` で埋める。
pub fn derive_adaptive_dims(
    in_shape: &[usize],
    output_size: (usize, usize),
) -> Result<PoolDims, PoolingPrepareError> {
    if in_shape.len() != 4 {
        return Err(PoolingPrepareError::InvalidShape {
            detail: format!("expected rank-4 input shape, got rank {}", in_shape.len()),
        });
    }
    let (n, c, h_in, w_in) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
    if h_in == 0 || w_in == 0 {
        return Err(PoolingPrepareError::InvalidShape {
            detail: format!("spatial axes must be non-zero, got h_in={h_in}, w_in={w_in}"),
        });
    }
    let (h_out, w_out) = output_size;
    if h_out == 0 || w_out == 0 {
        return Err(PoolingPrepareError::InvalidShape {
            detail: format!("output_size must be non-zero, got ({h_out},{w_out})"),
        });
    }

    let plane_in = h_in
        .checked_mul(w_in)
        .ok_or_else(|| PoolingPrepareError::InvalidShape {
            detail: "h_in * w_in overflows usize".to_string(),
        })?;
    let plane_out = h_out
        .checked_mul(w_out)
        .ok_or_else(|| PoolingPrepareError::InvalidShape {
            detail: "h_out * w_out overflows usize".to_string(),
        })?;
    let numel_out = n
        .checked_mul(c)
        .and_then(|v| v.checked_mul(plane_out))
        .ok_or_else(|| PoolingPrepareError::InvalidShape {
            detail: "n * c * h_out * w_out overflows usize".to_string(),
        })?;
    let numel_in = n
        .checked_mul(c)
        .and_then(|v| v.checked_mul(plane_in))
        .ok_or_else(|| PoolingPrepareError::InvalidShape {
            detail: "n * c * h_in * w_in overflows usize".to_string(),
        })?;

    Ok(PoolDims {
        n: to_u32_checked(n, "n")?,
        c: to_u32_checked(c, "c")?,
        h_in: to_u32_checked(h_in, "h_in")?,
        w_in: to_u32_checked(w_in, "w_in")?,
        h_out: to_u32_checked(h_out, "h_out")?,
        w_out: to_u32_checked(w_out, "w_out")?,
        kh: 0,
        kw: 0,
        sh: 0,
        sw: 0,
        ph: 0,
        pw: 0,
        dh: 0,
        dw: 0,
        count_include_pad: 0,
        numel_out: to_u32_checked(numel_out, "numel_out")?,
        plane_in: {
            let _ = to_u32_checked(numel_in, "n*c*h_in*w_in")?;
            to_u32_checked(plane_in, "plane_in")?
        },
        plane_out: to_u32_checked(plane_out, "plane_out")?,
    })
}

/// `x`（`[N,C,H,W]` 行優先平坦化・長さ `n*c*plane_in`）の実長が
/// `dims` と整合しているかを検証する（`crate::pooling::MetalPooling`
/// の各 `run_*` から `derive_*` の直後に呼ばれる共通ヘルパー）。
pub fn validate_input_len(x_len: usize, dims: &PoolDims) -> Result<(), PoolingPrepareError> {
    let expected = (dims.n as usize) * (dims.c as usize) * (dims.plane_in as usize);
    if x_len != expected {
        return Err(PoolingPrepareError::InvalidShape {
            detail: format!("input length {x_len} does not match expected {expected}"),
        });
    }
    Ok(())
}

/// `shaders/pooling.metal::max_pool2d_f32` の逐語移植（ホスト側参照
/// 実装）。窓走査は `kh` 外側・`kw` 内側の row-major、最初の有効
/// タップ（padding 除外）で `best`／`best_idx` を初期化し、以後は
/// `v > best || (v.is_nan() && !best.is_nan())` のときのみ更新する
/// （タイは先勝ち・NaN は最初に出現した NaN の索引で確定し以後
/// 更新されない。`docs/pooling-ops-design.md` §5 のタイ規則・NaN
/// 伝播索引契約）。全タップが padding（=有効タップ 0 件）の窓は
/// [`derive_pool_dims`] の空窓ゲートにより到達しない。索引は入力
/// 平面内の row-major 添字 `h*w_in + w`（`i32`）。
pub fn max_pool2d_model(
    x: &[f32],
    dims: &PoolDims,
) -> Result<(Vec<f32>, Vec<i32>), PoolingPrepareError> {
    validate_input_len(x.len(), dims)?;
    let (n, c, h_in, w_in, h_out, w_out) = (
        dims.n as i64,
        dims.c as i64,
        dims.h_in as i64,
        dims.w_in as i64,
        dims.h_out as i64,
        dims.w_out as i64,
    );
    let (kh, kw, sh, sw, ph, pw, dh, dw) = (
        dims.kh as i64,
        dims.kw as i64,
        dims.sh as i64,
        dims.sw as i64,
        dims.ph as i64,
        dims.pw as i64,
        dims.dh as i64,
        dims.dw as i64,
    );
    let numel_out = dims.numel_out as usize;
    let mut out = vec![0.0f32; numel_out];
    let mut idx = vec![0i32; numel_out];

    for nb in 0..n {
        for ch in 0..c {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut best = 0.0f32;
                    let mut best_idx = 0i32;
                    let mut first = true;
                    for kh_i in 0..kh {
                        let ih = oh * sh + kh_i * dh - ph;
                        if ih < 0 || ih >= h_in {
                            continue;
                        }
                        for kw_i in 0..kw {
                            let iw = ow * sw + kw_i * dw - pw;
                            if iw < 0 || iw >= w_in {
                                continue;
                            }
                            let in_idx = ((nb * c + ch) * h_in + ih) * w_in + iw;
                            let v = x[in_idx as usize];
                            let this_idx = (ih * w_in + iw) as i32;
                            if first {
                                best = v;
                                best_idx = this_idx;
                                first = false;
                            } else if v > best || (v.is_nan() && !best.is_nan()) {
                                best = v;
                                best_idx = this_idx;
                            }
                        }
                    }
                    let out_idx = ((nb * c + ch) * h_out + oh) * w_out + ow;
                    out[out_idx as usize] = best;
                    idx[out_idx as usize] = best_idx;
                }
            }
        }
    }
    Ok((out, idx))
}

/// `shaders/pooling.metal::avg_pool2d_f32` の逐語移植。窓内の有効
/// （非 padding）タップを row-major（`kh` 外側・`kw` 内側）に soft-f64
/// アキュムレータへ逐次加算し、`divisor`（`count_include_pad` なら
/// `kh*kw` 固定・でなければ有効タップ数）で soft-f64 除算してから
/// 1 回だけ `f32` へ narrow する（`.claude/rules/coding-rust.md`
/// 「勾配の長軸縮約」節と同型の binary64 ソフトウェアエミュレーション
/// 契約。`divisor`（`u32`）→ binary64 の変換はホスト側では
/// `(divisor as f64).to_bits()` が厳密変換である——`divisor < 2^32`
/// は常に `f64` で正確に表現できるため——ことを利用する。GPU 側は
/// `pool_f64_from_uint` が同じ変換をビット構成で行い、両者が bit
/// 完全一致することは [`tests`] が固定する）。
pub fn avg_pool2d_soft_f64(x: &[f32], dims: &PoolDims) -> Result<Vec<f32>, PoolingPrepareError> {
    validate_input_len(x.len(), dims)?;
    let (n, c, h_in, w_in, h_out, w_out) = (
        dims.n as i64,
        dims.c as i64,
        dims.h_in as i64,
        dims.w_in as i64,
        dims.h_out as i64,
        dims.w_out as i64,
    );
    let (kh, kw, sh, sw, ph, pw, dh, dw) = (
        dims.kh as i64,
        dims.kw as i64,
        dims.sh as i64,
        dims.sw as i64,
        dims.ph as i64,
        dims.pw as i64,
        dims.dh as i64,
        dims.dw as i64,
    );
    let numel_out = dims.numel_out as usize;
    let mut out = vec![0.0f32; numel_out];

    for nb in 0..n {
        for ch in 0..c {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut acc = 0u64; // widen_f32_bits(0.0f32.to_bits()) == 0u64。
                    let mut valid = 0u32;
                    for kh_i in 0..kh {
                        let ih = oh * sh + kh_i * dh - ph;
                        if ih < 0 || ih >= h_in {
                            continue;
                        }
                        for kw_i in 0..kw {
                            let iw = ow * sw + kw_i * dw - pw;
                            if iw < 0 || iw >= w_in {
                                continue;
                            }
                            let in_idx = ((nb * c + ch) * h_in + ih) * w_in + iw;
                            let v = x[in_idx as usize];
                            acc = add_f64_bits(acc, widen_f32_bits(v.to_bits()));
                            valid += 1;
                        }
                    }
                    let divisor = if dims.count_include_pad != 0 {
                        dims.kh * dims.kw
                    } else {
                        valid
                    };
                    let divisor_bits = (divisor as f64).to_bits();
                    let result = f32::from_bits(narrow_f64_bits(div_f64_bits(acc, divisor_bits)));
                    let out_idx = ((nb * c + ch) * h_out + oh) * w_out + ow;
                    out[out_idx as usize] = result;
                }
            }
        }
    }
    Ok(out)
}

/// 適応窓 `[start, end)` を PyTorch `torch.nn.AdaptiveAvgPool2d` と
/// 同じ整数演算で導出する（`start = floor(o*in/out)`・
/// `end = ceil((o+1)*in/out)`）。
fn adaptive_window(o: i64, in_len: i64, out_len: i64) -> (i64, i64) {
    let start = (o * in_len) / out_len;
    let end = ((o + 1) * in_len + out_len - 1) / out_len;
    (start, end)
}

/// `shaders/pooling.metal::adaptive_avg_pool2d_f32` の逐語移植。窓
/// `[start_h,end_h) × [start_w,end_w)` を row-major に soft-f64 総和
/// し `divisor = (end_h-start_h)*(end_w-start_w)` で除算する
/// （[`avg_pool2d_soft_f64`] と同じ soft-f64 契約）。
pub fn adaptive_avg_pool2d_soft_f64(
    x: &[f32],
    dims: &PoolDims,
) -> Result<Vec<f32>, PoolingPrepareError> {
    validate_input_len(x.len(), dims)?;
    let (n, c, h_in, w_in, h_out, w_out) = (
        dims.n as i64,
        dims.c as i64,
        dims.h_in as i64,
        dims.w_in as i64,
        dims.h_out as i64,
        dims.w_out as i64,
    );
    let numel_out = dims.numel_out as usize;
    let mut out = vec![0.0f32; numel_out];

    for nb in 0..n {
        for ch in 0..c {
            for oh in 0..h_out {
                let (start_h, end_h) = adaptive_window(oh, h_in, h_out);
                for ow in 0..w_out {
                    let (start_w, end_w) = adaptive_window(ow, w_in, w_out);
                    let mut acc = 0u64;
                    for ih in start_h..end_h {
                        for iw in start_w..end_w {
                            let in_idx = ((nb * c + ch) * h_in + ih) * w_in + iw;
                            let v = x[in_idx as usize];
                            acc = add_f64_bits(acc, widen_f32_bits(v.to_bits()));
                        }
                    }
                    let divisor = ((end_h - start_h) * (end_w - start_w)) as u32;
                    let divisor_bits = (divisor as f64).to_bits();
                    let result = f32::from_bits(narrow_f64_bits(div_f64_bits(acc, divisor_bits)));
                    let out_idx = ((nb * c + ch) * h_out + oh) * w_out + ow;
                    out[out_idx as usize] = result;
                }
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `PoolDims` は `shaders/pooling.metal::struct PoolDims` と
    /// レイアウトを一致させる（18 × uint）契約を固定する（`pooling.rs::
    /// encode_*_dispatch` の `setBytes_length_atIndex` が本サイズを
    /// 使う）。
    #[test]
    fn pool_dims_size_matches_msl_struct() {
        assert_eq!(std::mem::size_of::<PoolDims>(), 18 * 4);
    }

    #[test]
    fn padding_at_floor_half_kernel_allowed() {
        // k=2, d=1, p=1 は許可（floor(2/2)=1）。
        assert!(
            derive_pool_dims(&[1, 1, 4, 4], (2, 2), (1, 1), (1, 1), (1, 1), false, true).is_ok()
        );
        // k=3, d=2, p=1 は許可（floor(3/2)=1）。
        assert!(
            derive_pool_dims(&[1, 1, 8, 8], (3, 3), (1, 1), (1, 1), (2, 2), false, true).is_ok()
        );
    }

    #[test]
    fn padding_exceeding_floor_half_kernel_rejected() {
        assert!(matches!(
            derive_pool_dims(&[1, 1, 4, 4], (2, 2), (1, 1), (2, 2), (1, 1), false, true),
            Err(PoolingPrepareError::InvalidShape { .. })
        ));
        assert!(matches!(
            derive_pool_dims(&[1, 1, 8, 8], (3, 3), (1, 1), (2, 2), (2, 2), false, true),
            Err(PoolingPrepareError::InvalidShape { .. })
        ));
    }

    #[test]
    fn negative_output_numerator_rejected() {
        // in=1, k=2, s=2, p=0: numerator = 1+0-1-1 = -1 < 0。
        assert!(matches!(
            derive_pool_dims(&[1, 1, 1, 1], (2, 2), (2, 2), (0, 0), (1, 1), false, true),
            Err(PoolingPrepareError::InvalidShape { .. })
        ));
    }

    #[test]
    fn zero_spatial_axis_rejected_zero_batch_allowed() {
        assert!(matches!(
            derive_pool_dims(&[1, 2, 0, 4], (2, 2), (1, 1), (0, 0), (1, 1), false, true),
            Err(PoolingPrepareError::InvalidShape { .. })
        ));
        assert!(
            derive_pool_dims(&[0, 2, 4, 4], (2, 2), (1, 1), (0, 0), (1, 1), false, true).is_ok()
        );
    }

    #[test]
    fn empty_window_via_padding_compensated_dilation_rejected() {
        // in=2, k=2, s=1, p=1, d=3: numerator = 2+2-3-1 = 0 >= 0 だが
        // 両タップとも padding（h=-1, h=in=2）で空窓。
        assert!(matches!(
            derive_pool_dims(&[1, 1, 2, 2], (2, 2), (1, 1), (1, 1), (3, 3), false, true),
            Err(PoolingPrepareError::InvalidShape { .. })
        ));
        // in を 3 に増やせば d=3 でも許可（h=-1 は padding だが
        // h=in-1=2 側のタップは有効）。
        assert!(
            derive_pool_dims(&[1, 1, 3, 3], (2, 2), (1, 1), (1, 1), (3, 3), false, true).is_ok()
        );
    }

    #[test]
    fn ceil_mode_rejected() {
        assert!(matches!(
            derive_pool_dims(&[1, 1, 4, 4], (2, 2), (1, 1), (0, 0), (1, 1), true, true),
            Err(PoolingPrepareError::InvalidShape { .. })
        ));
    }

    #[test]
    fn reshape_merged_1d_shape_accepted() {
        // [N,C,1,L] は 1d 併合契約どおり受理される。
        let dims =
            derive_pool_dims(&[2, 3, 1, 10], (1, 2), (1, 2), (0, 0), (1, 1), false, true).unwrap();
        assert_eq!(dims.h_out, 1);
        assert_eq!(dims.w_out, 5);
    }

    fn naive_f64_avg(x: &[f32], dims: &PoolDims) -> Vec<f32> {
        let (n, c, h_in, w_in, h_out, w_out) = (
            dims.n as i64,
            dims.c as i64,
            dims.h_in as i64,
            dims.w_in as i64,
            dims.h_out as i64,
            dims.w_out as i64,
        );
        let (kh, kw, sh, sw, ph, pw, dh, dw) = (
            dims.kh as i64,
            dims.kw as i64,
            dims.sh as i64,
            dims.sw as i64,
            dims.ph as i64,
            dims.pw as i64,
            dims.dh as i64,
            dims.dw as i64,
        );
        let mut out = vec![0.0f32; dims.numel_out as usize];
        for nb in 0..n {
            for ch in 0..c {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut acc = 0.0f64;
                        let mut valid = 0u32;
                        for kh_i in 0..kh {
                            let ih = oh * sh + kh_i * dh - ph;
                            if ih < 0 || ih >= h_in {
                                continue;
                            }
                            for kw_i in 0..kw {
                                let iw = ow * sw + kw_i * dw - pw;
                                if iw < 0 || iw >= w_in {
                                    continue;
                                }
                                let in_idx = ((nb * c + ch) * h_in + ih) * w_in + iw;
                                acc += x[in_idx as usize] as f64;
                                valid += 1;
                            }
                        }
                        let divisor = if dims.count_include_pad != 0 {
                            dims.kh * dims.kw
                        } else {
                            valid
                        };
                        let out_idx = ((nb * c + ch) * h_out + oh) * w_out + ow;
                        out[out_idx as usize] = (acc / divisor as f64) as f32;
                    }
                }
            }
        }
        out
    }

    #[test]
    fn avg_pool_matches_naive_f64_reference_bit_exact() {
        let dims =
            derive_pool_dims(&[1, 2, 5, 5], (3, 3), (2, 2), (1, 1), (1, 1), false, true).unwrap();
        let x: Vec<f32> = (0..50).map(|i| (i as f32) * 0.37 - 3.0).collect();
        let got = avg_pool2d_soft_f64(&x, &dims).unwrap();
        let want = naive_f64_avg(&x, &dims);
        assert_eq!(got, want);
    }

    #[test]
    fn avg_pool_count_include_pad_false_matches_naive_reference() {
        let dims =
            derive_pool_dims(&[1, 1, 4, 4], (3, 3), (1, 1), (1, 1), (1, 1), false, false).unwrap();
        let x: Vec<f32> = (0..16).map(|i| i as f32 - 8.0).collect();
        let got = avg_pool2d_soft_f64(&x, &dims).unwrap();
        let want = naive_f64_avg(&x, &dims);
        assert_eq!(got, want);
    }

    #[test]
    fn avg_pool_cancelling_sum_matches_naive_f64_reference() {
        // 相殺列（対消滅）: 非結合な f32 単純和では丸め誤差が乗るが、
        // soft-f64 蓄積はホスト `f64` 参照と一致する。
        let dims =
            derive_pool_dims(&[1, 1, 1, 4], (1, 4), (1, 1), (0, 0), (1, 1), false, true).unwrap();
        let x: Vec<f32> = vec![2f32.powi(24), 1.0, -(2f32.powi(24)), 1.0];
        let got = avg_pool2d_soft_f64(&x, &dims).unwrap();
        let want = naive_f64_avg(&x, &dims);
        assert_eq!(got, want);
    }

    #[test]
    fn max_pool_tie_prefers_first_index() {
        let dims =
            derive_pool_dims(&[1, 1, 1, 4], (1, 4), (1, 1), (0, 0), (1, 1), false, true).unwrap();
        let x = vec![5.0f32, 5.0, 5.0, 1.0];
        let (out, idx) = max_pool2d_model(&x, &dims).unwrap();
        assert_eq!(out, vec![5.0]);
        assert_eq!(idx, vec![0]);
    }

    #[test]
    fn max_pool_first_nan_wins_and_stays() {
        let dims =
            derive_pool_dims(&[1, 1, 1, 4], (1, 4), (1, 1), (0, 0), (1, 1), false, true).unwrap();
        let x = vec![1.0f32, f32::NAN, f32::NAN, 9.0];
        let (out, idx) = max_pool2d_model(&x, &dims).unwrap();
        assert!(out[0].is_nan());
        assert_eq!(idx, vec![1]);
    }

    #[test]
    fn max_pool_all_negative_infinity_window() {
        let dims =
            derive_pool_dims(&[1, 1, 1, 3], (1, 3), (1, 1), (0, 0), (1, 1), false, true).unwrap();
        let x = vec![f32::NEG_INFINITY; 3];
        let (out, idx) = max_pool2d_model(&x, &dims).unwrap();
        assert_eq!(out, vec![f32::NEG_INFINITY]);
        assert_eq!(idx, vec![0]);
    }

    #[test]
    fn max_pool_padding_never_wins() {
        // p=1 の窓端では padding 位置が候補に入らないことを確認
        // （padding は常に非勝者——スキップされるため値として現れない）。
        let dims =
            derive_pool_dims(&[1, 1, 1, 3], (1, 3), (1, 1), (0, 1), (1, 1), false, true).unwrap();
        let x = vec![-1.0f32, -2.0, -3.0];
        let (out, idx) = max_pool2d_model(&x, &dims).unwrap();
        // oh=0 window taps: iw=-1(padding,skip),0,1 -> max(-1,-2)=-1 at idx 0
        assert_eq!(out[0], -1.0);
        assert_eq!(idx[0], 0);
    }

    #[test]
    fn adaptive_avg_pool_matches_naive_f64_reference_output_size_le_input() {
        let dims = derive_adaptive_dims(&[1, 1, 5, 5], (3, 3)).unwrap();
        let x: Vec<f32> = (0..25).map(|i| (i as f32) * 0.11 - 1.0).collect();
        let got = adaptive_avg_pool2d_soft_f64(&x, &dims).unwrap();

        // 独立な f64 naive 参照（PyTorch と同じ start/end 整数演算）。
        let mut want = vec![0.0f32; got.len()];
        for oh in 0..3i64 {
            let (sh, eh) = adaptive_window(oh, 5, 3);
            for ow in 0..3i64 {
                let (sw, ew) = adaptive_window(ow, 5, 3);
                let mut acc = 0.0f64;
                for ih in sh..eh {
                    for iw in sw..ew {
                        acc += x[(ih * 5 + iw) as usize] as f64;
                    }
                }
                let divisor = (eh - sh) * (ew - sw);
                want[(oh * 3 + ow) as usize] = (acc / divisor as f64) as f32;
            }
        }
        assert_eq!(got, want);
    }

    #[test]
    fn adaptive_avg_pool_output_size_greater_than_input_allowed() {
        let dims = derive_adaptive_dims(&[1, 1, 2, 2], (4, 4)).unwrap();
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let got = adaptive_avg_pool2d_soft_f64(&x, &dims).unwrap();
        assert_eq!(got.len(), 16);
    }

    #[test]
    fn adaptive_avg_pool_zero_output_size_rejected() {
        assert!(matches!(
            derive_adaptive_dims(&[1, 1, 4, 4], (0, 2)),
            Err(PoolingPrepareError::InvalidShape { .. })
        ));
    }

    #[test]
    fn divisor_u32_to_f64_bits_matches_pool_f64_from_uint_contract() {
        // `avg_pool2d_soft_f64`／`adaptive_avg_pool2d_soft_f64` が使う
        // `(divisor as f64).to_bits()` の厳密変換契約を固定する
        // （`shaders/pooling.metal::pool_f64_from_uint` が同じ変換を
        // ビット構成で行うことの根拠。u32 の全域で f64 は常に厳密
        // 表現できる）。
        for v in [0u32, 1, 2, 9, 255, 256, 1_000_000, u32::MAX] {
            let expected = (v as f64).to_bits();
            // MSL 側の構成規則をここでも複製し bit 一致を検証する
            // （`pool_f64_from_uint` の Rust 側再実装。テスト専用）。
            let got = if v == 0 {
                0u64
            } else {
                let lead = 31 - v.leading_zeros();
                let exp64 = (lead as u64) + 1023;
                let frac = if lead <= 52 {
                    (v as u64) << (52 - lead)
                } else {
                    (v as u64) >> (lead - 52)
                } & 0x000F_FFFF_FFFF_FFFFu64;
                (exp64 << 52) | frac
            };
            assert_eq!(got, expected, "mismatch for v={v}");
        }
    }
}
