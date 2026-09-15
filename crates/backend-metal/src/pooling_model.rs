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
/// `tests::pool_dims_size_matches_msl_struct` が固定する）。
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
    // 全項を `checked_*` で計算し、`kernel`／`dilation`／`padding` に
    // `usize::MAX` 級の巨大値が渡されても `i128` 演算・最終的な
    // `usize` への変換のいずれの段でも panic せず型付きエラーへ
    // 落とす（本番経路 panic 禁止。`.claude/rules/coding-rust.md`）。
    // `k`／`d`／`p` は `usize`（最大 `usize::MAX` 〜 1.8e19）のため
    // `d*(k-1)` は `i128`（最大 〜1.7e38）すら超過しうる
    // （codex-review 指摘・#1885 line 115）。
    let overflow = |detail: String| PoolingPrepareError::SizeLimitExceeded { detail };
    let k_m1 = (k as i128)
        .checked_sub(1)
        .ok_or_else(|| overflow(format!("{axis}: kernel-1 underflows (k={k})")))?;
    let d_term = (d as i128).checked_mul(k_m1).ok_or_else(|| {
        overflow(format!(
            "{axis}: dilation*(kernel-1) overflows i128 (k={k}, d={d})"
        ))
    })?;
    let p_term = (p as i128)
        .checked_mul(2)
        .ok_or_else(|| overflow(format!("{axis}: padding*2 overflows i128 (p={p})")))?;
    let numerator = (in_len as i128)
        .checked_add(p_term)
        .and_then(|v| v.checked_sub(d_term))
        .and_then(|v| v.checked_sub(1))
        .ok_or_else(|| {
            overflow(format!(
                "{axis}: output length numerator overflows i128 (in={in_len}, k={k}, s={s}, p={p}, d={d})"
            ))
        })?;
    if numerator < 0 {
        return Err(PoolingPrepareError::InvalidShape {
            detail: format!(
                "{axis}: output length numerator is negative (in={in_len}, k={k}, s={s}, p={p}, d={d})"
            ),
        });
    }
    // `s` は呼び出し元（`derive_pool_dims`）が非零を保証済みだが、
    // 本関数単体で呼ばれても panic しないよう `checked_div` で防御する。
    let quotient = numerator
        .checked_div(s as i128)
        .ok_or_else(|| overflow(format!("{axis}: division by zero stride (s={s})")))?;
    let out_len = quotient
        .checked_add(1)
        .ok_or_else(|| overflow(format!("{axis}: output length +1 overflows i128")))?;
    usize::try_from(out_len).map_err(|_| {
        overflow(format!(
            "{axis}: output length {out_len} exceeds usize::MAX"
        ))
    })
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
///
/// `n`／`c`／`plane_in` はいずれも `u32`（`PoolDims` フィールド）
/// からの `usize` 拡幅積のため理論上 `usize` を超えうる
/// （`u32::MAX`^3 ≈ 7.9e28 > `usize::MAX`〈64bit〉）。`checked_mul`
/// で防御し、overflow 時は panic ではなく型付きエラーを返す。
pub fn validate_input_len(x_len: usize, dims: &PoolDims) -> Result<(), PoolingPrepareError> {
    let expected = (dims.n as usize)
        .checked_mul(dims.c as usize)
        .and_then(|v| v.checked_mul(dims.plane_in as usize))
        .ok_or_else(|| PoolingPrepareError::SizeLimitExceeded {
            detail: "validate_input_len: n * c * plane_in overflows usize".to_string(),
        })?;
    if x_len != expected {
        return Err(PoolingPrepareError::InvalidShape {
            detail: format!("input length {x_len} does not match expected {expected}"),
        });
    }
    Ok(())
}

/// [`PoolDims`] の全フィールドが内部整合していることを検証する
/// （`n`／`c`／`plane_in` と入力長の一致だけでは、`h_out`／`w_out`／
/// `plane_out`／`numel_out` を手動で不整合な値に構築した
/// `PoolDims`（全フィールド `pub`）をすり抜けさせてしまい、
/// [`max_pool2d_model`]／[`avg_pool2d_soft_f64`]／
/// [`adaptive_avg_pool2d_soft_f64`] が `out[out_idx]` へ書き込む際に
/// 長さ不足の `out` バッファへ out-of-bounds で panic しうる
/// （codex-review 指摘・#1885 line 331）。
///
/// `dims` が持つ生パラメータ（`in_shape`・kernel／stride／padding／
/// dilation・`count_include_pad`。adaptive 系は `kh==0 && kw==0` を
/// マーカーとして扱う——[`derive_adaptive_dims`] が両者を常に `0`
/// で埋める設計に対応）から [`derive_pool_dims`]／
/// [`derive_adaptive_dims`] を再実行し、結果が `dims` 自身と
/// `PartialEq` で完全一致することを要求する（両関数が持つ検証
/// ロジック——`ceil_mode`・空窓ゲート・`checked_mul` 済み積・
/// `u32` 上限等——を再利用でき、二重管理を避けられる）。
fn validate_pool_dims_consistent(dims: &PoolDims) -> Result<(), PoolingPrepareError> {
    let in_shape = [
        dims.n as usize,
        dims.c as usize,
        dims.h_in as usize,
        dims.w_in as usize,
    ];
    let recomputed = if dims.kh == 0 && dims.kw == 0 {
        derive_adaptive_dims(&in_shape, (dims.h_out as usize, dims.w_out as usize))?
    } else {
        derive_pool_dims(
            &in_shape,
            (dims.kh as usize, dims.kw as usize),
            (dims.sh as usize, dims.sw as usize),
            (dims.ph as usize, dims.pw as usize),
            (dims.dh as usize, dims.dw as usize),
            false,
            dims.count_include_pad != 0,
        )?
    };
    if recomputed != *dims {
        return Err(PoolingPrepareError::InvalidShape {
            detail: format!(
                "PoolDims is internally inconsistent: given={dims:?} recomputed={recomputed:?}"
            ),
        });
    }
    Ok(())
}

/// 通常（非 adaptive）Pooling 専用の追加契約検査。[`derive_adaptive_
/// dims`] が返す [`PoolDims`]（`kh == 0 && kw == 0` を adaptive の
/// マーカーとして 0 埋めする設計。§ [`validate_pool_dims_consistent`]
/// 参照）が [`max_pool2d_model`]／[`avg_pool2d_soft_f64`] へそのまま
/// 渡されると、両関数はカーネルサイズ 0 の窓を空ループとして扱い、
/// `valid == 0` から `0/0` の NaN・未初期化の `best` を成功扱いで
/// 返してしまう（[`validate_pool_dims_consistent`] は `dims` 自身が
/// 内部整合しているかのみを検査し、adaptive 用寸法であること自体は
/// 許容してしまうため検出できない。codex-review 指摘・#1885
/// line 414）。
fn require_regular_pool_dims(dims: &PoolDims) -> Result<(), PoolingPrepareError> {
    if dims.kh == 0 || dims.kw == 0 {
        return Err(PoolingPrepareError::InvalidShape {
            detail: format!(
                "kh/kw must be non-zero for max_pool2d/avg_pool2d (got adaptive-style PoolDims: kh={}, kw={})",
                dims.kh, dims.kw
            ),
        });
    }
    Ok(())
}

/// [`require_regular_pool_dims`] の adaptive 版。通常 Pooling 用の
/// `PoolDims`（`kh != 0 && kw != 0`）が [`adaptive_avg_pool2d_soft_f64`]
/// へ渡されるのを拒否する（誤用時に「通常の窓」を無視して adaptive
/// 窓計算が上書きしてしまう入力契約の曖昧さを排除する）。
fn require_adaptive_pool_dims(dims: &PoolDims) -> Result<(), PoolingPrepareError> {
    if dims.kh != 0 || dims.kw != 0 {
        return Err(PoolingPrepareError::InvalidShape {
            detail: format!(
                "kh/kw must be zero for adaptive_avg_pool2d (got non-adaptive PoolDims: kh={}, kw={})",
                dims.kh, dims.kw
            ),
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
    validate_pool_dims_consistent(dims)?;
    require_regular_pool_dims(dims)?;
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
/// 完全一致することは `tests` が固定する）。
pub fn avg_pool2d_soft_f64(x: &[f32], dims: &PoolDims) -> Result<Vec<f32>, PoolingPrepareError> {
    validate_pool_dims_consistent(dims)?;
    require_regular_pool_dims(dims)?;
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
///
/// `o`／`in_len`／`out_len` は `PoolDims`（`u32` フィールド）由来の
/// 値であり adaptive 窓は padding を持たず座標が常に非負のため、
/// 符号付き `i64` ではなく符号なし `u64` で計算する（`derive_adaptive_
/// dims` の `to_u32_checked` が各値を `u32::MAX` 以下に保証するため、
/// 最大の中間積 `(o+1)*in_len` ≈ `u32::MAX * u32::MAX` ≈ 1.8447e19 は
/// `u64::MAX` ≈ 1.8447e19 に収まり overflow しない。`i64::MAX` ≈
/// 9.223e18 では overflow して panic／負値ラップにより負添字での
/// out-of-bounds 読み出しを起こしていた。`shaders/pooling.metal::
/// adaptive_avg_pool2d_f32` の `ulong` 版と 1 対 1 対応する
/// codex-review 是正・#1885）。
fn adaptive_window(o: u64, in_len: u64, out_len: u64) -> (u64, u64) {
    let start = (o * in_len) / out_len;
    // `div_ceil` は `((o + 1) * in_len + out_len - 1) / out_len` と
    // 数学的に同じ整数結果を返す（`shaders/pooling.metal::
    // adaptive_avg_pool2d_f32` は `clippy::manual_div_ceil` の対象外
    // の MSL のため後者の形のまま。値は一致する）。
    let end = ((o + 1) * in_len).div_ceil(out_len);
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
    validate_pool_dims_consistent(dims)?;
    require_adaptive_pool_dims(dims)?;
    validate_input_len(x.len(), dims)?;
    let (n, c, h_in, w_in, h_out, w_out) = (
        dims.n as u64,
        dims.c as u64,
        dims.h_in as u64,
        dims.w_in as u64,
        dims.h_out as u64,
        dims.w_out as u64,
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
        for oh in 0..3u64 {
            let (sh, eh) = adaptive_window(oh, 5, 3);
            for ow in 0..3u64 {
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

    /// codex-review 指摘（#1885 line 115）の回帰: `kernel`／`dilation`
    /// に `usize::MAX` 級の巨大値を渡しても `compute_out_len`
    /// （`derive_pool_dims` 経由）が overflow で panic せず型付き
    /// エラーを返すことを固定する。
    #[test]
    fn huge_kernel_and_dilation_does_not_panic() {
        // `k=usize::MAX, d=1` 単独では `d*(k-1)` 自体は `i128` に
        // 収まる（分子が大きく負になり `InvalidShape` で拒否）が、
        // 呼び出し自体が overflow で panic しないことを固定する。
        assert!(matches!(
            derive_pool_dims(
                &[1, 1, 4, 4],
                (usize::MAX, 1),
                (1, 1),
                (0, 0),
                (1, 1),
                false,
                true,
            ),
            Err(PoolingPrepareError::InvalidShape { .. })
        ));
        // `k`／`d` がともに `usize::MAX` 級だと `d*(k-1)` が `i128`
        // すら超過する（codex-review 指摘の具体例）ため
        // `SizeLimitExceeded` を返す。
        assert!(matches!(
            derive_pool_dims(
                &[1, 1, 4, 4],
                (usize::MAX, usize::MAX),
                (1, 1),
                (0, 0),
                (usize::MAX, usize::MAX),
                false,
                true,
            ),
            Err(PoolingPrepareError::SizeLimitExceeded { .. })
        ));
    }

    /// codex-review 指摘（#1885 line 331）の回帰: `PoolDims` を手動で
    /// 内部不整合（`numel_out` が実際の `n*c*h_out*w_out` より小さい）
    /// に構築すると、`validate_input_len`（`n*c*plane_in` と入力長の
    /// 一致）はすり抜けるが `validate_pool_dims_consistent` が
    /// `InvalidShape` で拒否し、`out[out_idx]` への out-of-bounds
    /// 書き込み panic を未然に防ぐことを固定する。
    #[test]
    fn inconsistent_numel_out_rejected_not_panicking() {
        let mut dims =
            derive_pool_dims(&[1, 1, 4, 4], (2, 2), (2, 2), (0, 0), (1, 1), false, true).unwrap();
        assert_eq!(dims.numel_out, 4); // n=1,c=1,h_out=2,w_out=2
        // numel_out を実際より小さく手動改竄する（`out` バッファが
        // 短くなり、書き込み時に out-of-bounds を起こしうる構成）。
        dims.numel_out = 1;
        let x = vec![0.0f32; 16];
        assert!(matches!(
            max_pool2d_model(&x, &dims),
            Err(PoolingPrepareError::InvalidShape { .. })
        ));
        assert!(matches!(
            avg_pool2d_soft_f64(&x, &dims),
            Err(PoolingPrepareError::InvalidShape { .. })
        ));
    }

    /// 同上（#1885 line 331）の adaptive 系回帰: `PoolDims` を
    /// adaptive マーカー（`kh==0 && kw==0`）のまま `h_out`／`w_out`
    /// を手動改竄しても `adaptive_avg_pool2d_soft_f64` が panic
    /// せず `InvalidShape` を返すことを固定する。
    #[test]
    fn inconsistent_adaptive_dims_rejected_not_panicking() {
        let mut dims = derive_adaptive_dims(&[1, 1, 4, 4], (2, 2)).unwrap();
        assert_eq!(dims.numel_out, 4);
        // `h_out`／`w_out`／`plane_out` は正しい値（2, 2, 4）のまま
        // `numel_out` だけを実際の積（`n*c*plane_out`=4）より小さく
        // 改竄する（`out` バッファが短くなり書き込み時に
        // out-of-bounds を起こしうる構成）。
        dims.numel_out = 1;
        let x = vec![0.0f32; 16];
        assert!(matches!(
            adaptive_avg_pool2d_soft_f64(&x, &dims),
            Err(PoolingPrepareError::InvalidShape { .. })
        ));
    }

    /// codex-review 指摘（#1885。AdaptiveAvgPool の窓座標計算が
    /// 整数オーバーフロー可能）の回帰: `o`／`in_len`／`out_len` が
    /// `u32::MAX` 付近（`PoolDims` の `u32` フィールドが許容する
    /// 上限）でも `adaptive_window` が overflow せず panic しない
    /// （旧 `i64` 実装では `(o + 1) * in_len` が `i64::MAX` を超えて
    /// overflow し、負値へラップした `start`／`end` を境界検査なしに
    /// 返していた）。導出した窓が入力範囲 `[0, in_len)` に収まる
    /// （`end` は排他的上限であり `in_len` を超えないこと）ことも
    /// 併せて確認する。
    #[test]
    fn adaptive_window_no_overflow_near_u32_max() {
        let in_len = u32::MAX as u64;
        let out_len = u32::MAX as u64;
        // `(o + 1) * in_len` が符号付き `i64::MAX`（約 9.223e18）を
        // 超え始める境目付近（約 2.147e9）を狙う（指摘中の具体例
        // `oh=2147483649` と同じオーダー）。
        for o in [0u64, 1, 2_147_483_649, out_len - 2, out_len - 1] {
            let (start, end) = adaptive_window(o, in_len, out_len);
            assert!(start <= end, "start={start} end={end} for o={o}");
            assert!(end <= in_len, "end={end} exceeds in_len={in_len} for o={o}");
        }
    }

    /// codex-review 指摘（#1885 line 414）の回帰: `derive_adaptive_
    /// dims` が返す adaptive 用 `PoolDims`（`kh==0 && kw==0`）を
    /// `avg_pool2d_soft_f64`／`max_pool2d_model`（通常 Pooling 用）へ
    /// 渡すと `InvalidShape` で拒否され、`validate_pool_dims_
    /// consistent` だけでは検出できなかった「空ループ・0/0 NaN が
    /// 成功扱いで返る」誤用を機構的に遮断することを固定する。
    #[test]
    fn regular_pool_models_reject_adaptive_style_dims() {
        let dims = derive_adaptive_dims(&[1, 1, 4, 4], (2, 2)).unwrap();
        assert_eq!(dims.kh, 0);
        assert_eq!(dims.kw, 0);
        let x = vec![1.0f32; 16];
        assert!(matches!(
            avg_pool2d_soft_f64(&x, &dims),
            Err(PoolingPrepareError::InvalidShape { .. })
        ));
        assert!(matches!(
            max_pool2d_model(&x, &dims),
            Err(PoolingPrepareError::InvalidShape { .. })
        ));
    }

    /// [`regular_pool_models_reject_adaptive_style_dims`] の逆方向:
    /// `derive_pool_dims` が返す通常 Pooling 用 `PoolDims`
    /// （`kh != 0 && kw != 0`）を `adaptive_avg_pool2d_soft_f64` へ
    /// 渡すと `InvalidShape` で拒否されることを固定する。
    #[test]
    fn adaptive_pool_model_rejects_regular_style_dims() {
        let dims =
            derive_pool_dims(&[1, 1, 4, 4], (2, 2), (2, 2), (0, 0), (1, 1), false, true).unwrap();
        assert_ne!(dims.kh, 0);
        assert_ne!(dims.kw, 0);
        let x = vec![1.0f32; 16];
        assert!(matches!(
            adaptive_avg_pool2d_soft_f64(&x, &dims),
            Err(PoolingPrepareError::InvalidShape { .. })
        ));
    }
}
