//! Conv2d の im2col／col2im（イシュー #1768・親 #1644・設計
//! `docs/conv-ops-design.md`）のホスト側逐語モデル。`crate::soft_f64`
//! の binary64 ソフトウェアエミュレーション（[`crate::soft_f64::
//! widen_f32_bits`]／[`crate::soft_f64::add_f64_bits`]／
//! [`crate::soft_f64::narrow_f64_bits`]）を `shaders/im2col.metal::
//! im2col_f64_*` と同じ演算列で呼び出すことで、GPU 側カーネルが
//! 正しい binary64 逐次加算列を実行することを Mac 実機に到達できない
//! 環境（本実装環境。Linux・CI）でも機械的に裏付ける（`scan_model.rs`・
//! `unique_model.rs` と同じ設計判断: `objc2` 系 FFI に触れないため
//! `cfg(target_os = "macos")` を付けない）。
//!
//! `Im2colDims`（`#[repr(C)]`・19 × `u32`）は `crate::im2col`（macOS
//! 限定の起動 API）と本モジュールの両方から参照する形状引数一式
//! （`shaders/im2col.metal::struct Im2colDims` とレイアウトを一致させる。
//! `gemm.rs::Dims`／`GemmStrides` と同じ「1 回の `setBytes` で構造体
//! 丸ごと渡す」方式——19 個の個別 `setBytes` を避ける）。
//!
//! # なぜこれが Metal 側の正しさの根拠になるか
//!
//! [`im2col_model`] は `shaders/im2col.metal::im2col_f32` の逐語移植
//! （算術を含まない純粋コピー）、[`col2im_soft_f64`] は
//! `shaders/im2col.metal::col2im_f32` の逐語移植（binary64 ソフトウェア
//! エミュレーションアキュムレータへの逐次加算）である。本モジュールの
//! 単体テストはこれらを `fandhe_ai_backend_cpu::CpuBackendOps`
//! （`backend-cpu::im2col::im2col`／`col2im`。dev-dependency）の
//! `BackendOps::im2col`／`col2im` と **bit 完全一致**で突き合わせる
//! （`.claude/rules/coding-rust.md` 数値契約節）。

use crate::soft_f64::{add_f64_bits, narrow_f64_bits, widen_f32_bits};

/// im2col／col2im カーネルの `u32` 引数（`Im2colDims` の各フィールド）
/// が収まるバックエンド固有上限。CUDA 側（`backend-cuda::im2col::
/// validate_i32_bound`。`int` 引数のため `i32::MAX`）と対になる Metal
/// 側の定数（`uint` 引数のため `u32::MAX`。`scan_model::
/// SCAN_KERNEL_ARG_LIMIT` と同型）。
pub const IM2COL_KERNEL_ARG_LIMIT: usize = u32::MAX as usize;

/// `shaders/im2col.metal::struct Im2colDims` とレイアウトを一致させる
/// （`repr(C)`・19 × `u32` = 76 バイト）。`im2col.rs::MetalIm2col`
/// （macOS 限定）が `setBytes_length_atIndex`（buffer index 2）で
/// 1 回の呼び出しにまとめて渡す。
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Im2colDims {
    pub n_batch: u32,
    pub cin: u32,
    pub groups: u32,
    pub cin_g: u32,
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
    pub k_g: u32,
    pub p: u32,
    pub numel: u32,
}

/// [`derive_im2col_dims`] の失敗理由（`scan_model::ScanPrepareError`
/// と同型のホスト側純関数エラー。`objc2` 非依存）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Im2colPrepareError {
    /// いずれかの形状パラメータが [`IM2COL_KERNEL_ARG_LIMIT`]
    /// （`u32::MAX`）を超過した。`ops.rs::map_im2col_prepare_error` は
    /// 本 variant を `BackendError::Unsupported` へ写像し、
    /// `Var::conv2d` のホストフォールバック（`conv2d_with_fallback`
    /// 経由の `eval::im2col`／`col2im`）へ委ねる（col は入力の
    /// `kH·kW` 倍で現実的形状でも上限に到達しうるため hard fail では
    /// なくフォールバックが妥当。`backend-cuda::im2col::
    /// Im2colSizeLimitExceeded` と同じ設計判断）。
    SizeLimitExceeded {
        what: &'static str,
        value: usize,
        limit: usize,
    },
    /// `in_shape`／`col_shape`／`params` から再計算した `h_out*w_out`
    /// が `col_shape` の `P` 軸と一致しない（内部契約違反。呼び出し元
    /// `ops.rs` が `im2col_out_shape` で事前検証済みの入力からは実質
    /// 到達しない防御的経路。`backend-cuda::im2col::
    /// InvalidIm2colShape` と同型）。
    InvalidShape { detail: String },
}

impl std::fmt::Display for Im2colPrepareError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Im2colPrepareError::SizeLimitExceeded { what, value, limit } => write!(
                f,
                "im2col/col2im dimension exceeds kernel argument limit: {what}={value} (limit={limit})"
            ),
            Im2colPrepareError::InvalidShape { detail } => {
                write!(f, "im2col/col2im shape validation failed: {detail}")
            }
        }
    }
}

/// `value` が [`IM2COL_KERNEL_ARG_LIMIT`] に収まることを検証し `u32`
/// へ変換する（`scan_model::plan_scan` の逐一検証と同型）。
fn checked_u32(value: usize, what: &'static str) -> Result<u32, Im2colPrepareError> {
    u32::try_from(value).map_err(|_| Im2colPrepareError::SizeLimitExceeded {
        what,
        value,
        limit: IM2COL_KERNEL_ARG_LIMIT,
    })
}

/// `in_shape: [N, Cin, H, W]`・`col_shape: [N, G, K_g, P]`（呼び出し元
/// `ops.rs` が [`fandhe_ai_tensor_core::im2col_out_shape`] で検査・確定
/// 済みの形状）・`params` から [`Im2colDims`] を導出する
/// （`backend-cuda::im2col::LaunchShape::derive` の Metal 対応版）。
/// `numel` は呼び出し側が指定する起動グリッドの範囲（im2col は
/// `col_shape` の要素数・col2im は `in_shape` の要素数）。
///
/// `col_shape` は [`fandhe_ai_tensor_core::im2col_out_shape`] で
/// `in_shape`／`params` から再計算した期待値と **4 軸すべて**（`N`・
/// `G`・`K_g`・`P`）を fail-closed 照合する（codex-review P0 是正:
/// 従来は `P` 軸のみを照合しており、`col_shape[0]`〈N〉が過大な場合の
/// `im2col_model` による `input` 範囲外読み出し、`col_shape[1]`／
/// `[2]`〈G／K_g〉が `params`／`cin` と食い違う場合の
/// `col2im_soft_f64` による `d_col` 範囲外読み出しを検出できな
/// かった）。`h_out`／`w_out` は照合後の `col_shape` の `P` 軸だけ
/// では個別の値が復元できないため [`fandhe_ai_tensor_core::
/// conv_out_len`] で独立に再計算する（設計 doc §6.2「訂正 2」・
/// CUDA 側と同じ理由）。全 19 値の `u32` 収容も検査する。
pub fn derive_im2col_dims(
    in_shape: &[usize],
    col_shape: &[usize],
    params: &fandhe_ai_tensor_core::Conv2dParams,
    numel: usize,
) -> Result<Im2colDims, Im2colPrepareError> {
    // rank 検査を通常の条件分岐で行い、`in_shape[0..3]` への添字
    // アクセス（直後の行）より必ず先に完了させる（codex-review P1
    // 是正: 旧実装は `debug_assert_eq!` に依存しており、release
    // ビルドでは rank 不足の `in_shape`（例 `[1, 1, 1]`）に対しても
    // panic せずに範囲外添字アクセスへ進んでしまい `InvalidShape`
    // を返せなかった。`col_shape` は直後で `expected_col_shape.
    // as_slice()` とスライス比較〈長さ不一致なら即 false〉するのみで
    // 添字アクセスしないため rank 不一致でも安全だが、`in_shape` と
    // 対称にここで検査し早期に分かりやすいエラーへ倒す）。
    if in_shape.len() != 4 {
        return Err(Im2colPrepareError::InvalidShape {
            detail: format!(
                "in_shape must be rank 4, got rank {} ({in_shape:?})",
                in_shape.len()
            ),
        });
    }
    if col_shape.len() != 4 {
        return Err(Im2colPrepareError::InvalidShape {
            detail: format!(
                "col_shape must be rank 4, got rank {} ({col_shape:?})",
                col_shape.len()
            ),
        });
    }

    let (n_batch, cin, h_in, w_in) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);

    // `in_shape`／`params` から期待される完全な col shape（`[N, G,
    // K_g, P]`）を独立に再計算し、呼び出し元が渡した `col_shape`
    // 全体（N・G・K_g・P の 4 軸すべて）と照合する（codex-review
    // P0 是正: 従来は P 軸のみを照合しており、N・G・K_g の不一致を
    // 検出できなかった。`col_shape[0]`〈N〉が `in_shape[0]` より
    // 大きいと `im2col_model` が `input` の範囲外を読み出し、
    // `col_shape[1]`〈G〉／`[2]`〈K_g〉が `params.groups()`／
    // `cin_g*kh*kw` と食い違うと `col2im_soft_f64` が `d_col` の
    // 範囲外を読み出す——という 2 つの GPU 側範囲外読み出し経路を
    // 同時に塞ぐ。`im2col_out_shape` 自身が `cin % groups == 0`・
    // `h`／`w` 非ゼロ・`checked_mul` によるオーバーフロー検査を
    // 内包するため、ここで個別に再実装しない）。
    let expected_col_shape =
        fandhe_ai_tensor_core::im2col_out_shape(in_shape, params).map_err(|e| {
            Im2colPrepareError::InvalidShape {
                detail: format!("im2col_out_shape failed: {e:?}"),
            }
        })?;
    if col_shape != expected_col_shape.as_slice() {
        return Err(Im2colPrepareError::InvalidShape {
            detail: format!(
                "col shape mismatch: col_shape={col_shape:?} expected={expected_col_shape:?}"
            ),
        });
    }

    let groups = expected_col_shape[1];
    let k_g = expected_col_shape[2];
    let p = expected_col_shape[3];
    let cin_g = cin / groups.max(1);

    let [kh, kw] = params.kernel_size();
    let [sh, sw] = params.stride();
    let [ph, pw] = params.padding();
    let [dh, dw] = params.dilation();

    let h_out = fandhe_ai_tensor_core::conv_out_len(h_in, kh, sh, ph, dh).map_err(|e| {
        Im2colPrepareError::InvalidShape {
            detail: format!("conv_out_len(h) failed: {e:?}"),
        }
    })?;
    let w_out = fandhe_ai_tensor_core::conv_out_len(w_in, kw, sw, pw, dw).map_err(|e| {
        Im2colPrepareError::InvalidShape {
            detail: format!("conv_out_len(w) failed: {e:?}"),
        }
    })?;
    debug_assert_eq!(
        h_out.checked_mul(w_out),
        Some(p),
        "derive_im2col_dims: P axis already verified equal via expected_col_shape"
    );

    Ok(Im2colDims {
        n_batch: checked_u32(n_batch, "n_batch")?,
        cin: checked_u32(cin, "cin")?,
        groups: checked_u32(groups, "groups")?,
        cin_g: checked_u32(cin_g, "cin_g")?,
        h_in: checked_u32(h_in, "h_in")?,
        w_in: checked_u32(w_in, "w_in")?,
        h_out: checked_u32(h_out, "h_out")?,
        w_out: checked_u32(w_out, "w_out")?,
        kh: checked_u32(kh, "kh")?,
        kw: checked_u32(kw, "kw")?,
        sh: checked_u32(sh, "sh")?,
        sw: checked_u32(sw, "sw")?,
        ph: checked_u32(ph, "ph")?,
        pw: checked_u32(pw, "pw")?,
        dh: checked_u32(dh, "dh")?,
        dw: checked_u32(dw, "dw")?,
        k_g: checked_u32(k_g, "k_g")?,
        p: checked_u32(p, "p")?,
        numel: checked_u32(numel, "numel")?,
    })
}

/// `dims`（[`Im2colDims`]。全フィールド `pub` のため
/// [`derive_im2col_dims`] を経由せず任意の値で直接構築されうる）が
/// `derive_im2col_dims` が生成する値と同じ内部整合性を持つことを
/// 独立に再検証する（[`im2col_model`]／[`col2im_soft_f64`] は `pub`
/// かつ `#[cfg(test)]` の外にあるため、`crate::gather_scatter_model::
/// gather_model`〈イシュー #1799〉・`crate::constant_pad_model::
/// constant_pad_model` と同じ理由でここで検証し、0 除算・範囲外
/// 添字アクセスで panic しないようにする。codex-review 指摘。
/// `derive_im2col_dims` と同じ [`fandhe_ai_tensor_core::
/// im2col_out_shape`]／[`fandhe_ai_tensor_core::conv_out_len`] で
/// `in_shape`／`col_shape` を再構成し `dims` の各フィールドと突き
/// 合わせる。戻り値は `(in_numel, col_numel)`（呼び出し側がスライス
/// 長・`dims.numel` の検証に使う）。
///
/// `in_numel`／`col_numel` が 0 でない限り、この検証を通過した
/// `dims` は `im2col_model`／`col2im_soft_f64` 内で使う全ての除数
/// （`p`／`k_g`／`groups`／`kh`／`kw`／`w_out`／`cin`／`h_in`／
/// `w_in`／`cin_g`／`sh`／`sw`）が 1 以上であることを数学的に含意
/// する（`Conv2dParams::new` が `kernel_size`／`stride`／`dilation`／
/// `groups` の 0 を拒否・`im2col_out_shape` が `H`／`W` の 0 と
/// `Cin % groups != 0` を拒否・`conv_out_len` が常に `h_out`／
/// `w_out >= 1` を返す契約による）。
fn validate_im2col_dims_consistent(
    dims: &Im2colDims,
) -> Result<(usize, usize), Im2colPrepareError> {
    let in_shape = [
        dims.n_batch as usize,
        dims.cin as usize,
        dims.h_in as usize,
        dims.w_in as usize,
    ];
    let params = fandhe_ai_tensor_core::Conv2dParams::new(
        [dims.kh as usize, dims.kw as usize],
        [dims.sh as usize, dims.sw as usize],
        [dims.ph as usize, dims.pw as usize],
        [dims.dh as usize, dims.dw as usize],
        dims.groups as usize,
    )
    .map_err(|e| Im2colPrepareError::InvalidShape {
        detail: format!("Conv2dParams::new failed: {e:?}"),
    })?;

    let expected_col_shape =
        fandhe_ai_tensor_core::im2col_out_shape(&in_shape, &params).map_err(|e| {
            Im2colPrepareError::InvalidShape {
                detail: format!("im2col_out_shape failed: {e:?}"),
            }
        })?;
    let col_shape = [
        dims.n_batch as usize,
        dims.groups as usize,
        dims.k_g as usize,
        dims.p as usize,
    ];
    if col_shape != expected_col_shape.as_slice() {
        return Err(Im2colPrepareError::InvalidShape {
            detail: format!(
                "col shape mismatch: dims implies col_shape={col_shape:?} expected={expected_col_shape:?}"
            ),
        });
    }

    let cin_g = dims.cin as usize / (dims.groups as usize).max(1);
    if cin_g != dims.cin_g as usize {
        return Err(Im2colPrepareError::InvalidShape {
            detail: format!("cin_g mismatch: dims.cin_g={} expected={cin_g}", dims.cin_g),
        });
    }

    let h_out = fandhe_ai_tensor_core::conv_out_len(
        dims.h_in as usize,
        dims.kh as usize,
        dims.sh as usize,
        dims.ph as usize,
        dims.dh as usize,
    )
    .map_err(|e| Im2colPrepareError::InvalidShape {
        detail: format!("conv_out_len(h) failed: {e:?}"),
    })?;
    let w_out = fandhe_ai_tensor_core::conv_out_len(
        dims.w_in as usize,
        dims.kw as usize,
        dims.sw as usize,
        dims.pw as usize,
        dims.dw as usize,
    )
    .map_err(|e| Im2colPrepareError::InvalidShape {
        detail: format!("conv_out_len(w) failed: {e:?}"),
    })?;
    if h_out != dims.h_out as usize || w_out != dims.w_out as usize {
        return Err(Im2colPrepareError::InvalidShape {
            detail: format!(
                "h_out/w_out mismatch: dims=({}, {}) expected=({h_out}, {w_out})",
                dims.h_out, dims.w_out
            ),
        });
    }

    let in_numel = in_shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or(Im2colPrepareError::SizeLimitExceeded {
            what: "in_numel",
            value: usize::MAX,
            limit: IM2COL_KERNEL_ARG_LIMIT,
        })?;
    let col_numel = col_shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or(Im2colPrepareError::SizeLimitExceeded {
            what: "col_numel",
            value: usize::MAX,
            limit: IM2COL_KERNEL_ARG_LIMIT,
        })?;
    Ok((in_numel, col_numel))
}

/// `shaders/im2col.metal::im2col_f32` の逐語モデル（算術を含まない
/// 純粋コピー。`input` は `[N, Cin, H, W]` の稠密スライス、戻り値は
/// `[N, G, K_g, P]` の稠密 `Vec`）。
///
/// 本関数は `pub` かつ `#[cfg(test)]` の外にあるため、
/// [`validate_im2col_dims_consistent`] を入口で呼び `dims` の内部
/// 整合性・`dims.numel`／`input` の実長一致まで検証してから本体
/// ループへ入る（本番経路で panic させない方針。
/// `.claude/rules/coding-rust.md`）。
pub fn im2col_model(input: &[f32], dims: &Im2colDims) -> Result<Vec<f32>, Im2colPrepareError> {
    let (in_numel, col_numel) = validate_im2col_dims_consistent(dims)?;
    if dims.numel as usize != col_numel {
        return Err(Im2colPrepareError::InvalidShape {
            detail: format!(
                "im2col_model: dims.numel={} does not match derived col numel={col_numel}",
                dims.numel
            ),
        });
    }
    if input.len() != in_numel {
        return Err(Im2colPrepareError::InvalidShape {
            detail: format!(
                "im2col_model: input.len()={} does not match derived in numel={in_numel}",
                input.len()
            ),
        });
    }
    if col_numel == 0 {
        return Ok(Vec::new());
    }

    let numel = dims.numel as usize;
    let (groups, k_g, p) = (dims.groups as usize, dims.k_g as usize, dims.p as usize);
    let (kh, kw) = (dims.kh as usize, dims.kw as usize);
    let (sh, sw) = (dims.sh as usize, dims.sw as usize);
    let (ph, pw) = (dims.ph as i64, dims.pw as i64);
    let (dh, dw) = (dims.dh as i64, dims.dw as i64);
    let (h_in, w_in) = (dims.h_in as i64, dims.w_in as i64);
    let (cin, cin_g, w_out) = (dims.cin as usize, dims.cin_g as usize, dims.w_out as usize);

    let mut out = vec![0f32; numel];
    for (idx, slot) in out.iter_mut().enumerate() {
        let mut rem = idx;
        let p_idx = rem % p;
        rem /= p;
        let k_idx = rem % k_g;
        rem /= k_g;
        let g = rem % groups;
        rem /= groups;
        let n = rem;

        let kw_ = k_idx % kw;
        let rest = k_idx / kw;
        let kh_ = rest % kh;
        let c_g = rest / kh;
        let c = g * cin_g + c_g;

        let ow = p_idx % w_out;
        let oh = p_idx / w_out;

        let h = (oh as i64) * (sh as i64) + (kh_ as i64) * dh - ph;
        let w = (ow as i64) * (sw as i64) + (kw_ as i64) * dw - pw;

        let value = if h >= 0 && h < h_in && w >= 0 && w < w_in {
            let in_idx = ((n * cin + c) as i64 * h_in + h) * w_in + w;
            input[in_idx as usize]
        } else {
            0.0
        };
        *slot = value;
    }
    Ok(out)
}

/// `shaders/im2col.metal::col2im_f32` の逐語モデル（binary64
/// ソフトウェアエミュレーションアキュムレータ〈`crate::soft_f64`〉へ
/// `(kh, kw)` row-major の逐次加算・最後に 1 回 downcast。`d_col` は
/// `[N, G, K_g, P]` の稠密スライス、戻り値は `[N, Cin, H, W]` の稠密
/// `Vec`）。
///
/// 本関数は `pub` かつ `#[cfg(test)]` の外にあるため、
/// [`im2col_model`] と同じ理由で入口に
/// [`validate_im2col_dims_consistent`] を呼び `dims` の内部整合性・
/// `dims.numel`／`d_col` の実長一致まで検証してから本体ループへ入る
/// （本番経路で panic させない方針。`.claude/rules/coding-rust.md`）。
pub fn col2im_soft_f64(d_col: &[f32], dims: &Im2colDims) -> Result<Vec<f32>, Im2colPrepareError> {
    let (in_numel, col_numel) = validate_im2col_dims_consistent(dims)?;
    if dims.numel as usize != in_numel {
        return Err(Im2colPrepareError::InvalidShape {
            detail: format!(
                "col2im_soft_f64: dims.numel={} does not match derived in numel={in_numel}",
                dims.numel
            ),
        });
    }
    if d_col.len() != col_numel {
        return Err(Im2colPrepareError::InvalidShape {
            detail: format!(
                "col2im_soft_f64: d_col.len()={} does not match derived col numel={col_numel}",
                d_col.len()
            ),
        });
    }
    if in_numel == 0 {
        return Ok(Vec::new());
    }

    let numel = dims.numel as usize;
    let (cin, w_in, h_in) = (dims.cin as usize, dims.w_in as usize, dims.h_in as usize);
    let cin_g = dims.cin_g as usize;
    let (groups, k_g, p) = (dims.groups as usize, dims.k_g as usize, dims.p as usize);
    let (kh, kw) = (dims.kh as usize, dims.kw as usize);
    let (sh, sw) = (dims.sh as i64, dims.sw as i64);
    let (ph, pw) = (dims.ph as i64, dims.pw as i64);
    let (dh, dw) = (dims.dh as i64, dims.dw as i64);
    let (h_out, w_out) = (dims.h_out as i64, dims.w_out as i64);

    let mut out = vec![0f32; numel];
    for (idx, slot) in out.iter_mut().enumerate() {
        let mut rem = idx;
        let w = rem % w_in;
        rem /= w_in;
        let h = rem % h_in;
        rem /= h_in;
        let c = rem % cin;
        rem /= cin;
        let n = rem;

        let g = c / cin_g;
        let c_g = c % cin_g;

        let mut acc: u64 = widen_f32_bits(0.0f32.to_bits());
        for kh_ in 0..kh {
            let num_h = h as i64 + ph - (kh_ as i64) * dh;
            if num_h < 0 {
                continue;
            }
            if num_h % sh != 0 {
                continue;
            }
            let oh = num_h / sh;
            if oh >= h_out {
                continue;
            }
            for kw_ in 0..kw {
                let num_w = w as i64 + pw - (kw_ as i64) * dw;
                if num_w < 0 {
                    continue;
                }
                if num_w % sw != 0 {
                    continue;
                }
                let ow = num_w / sw;
                if ow >= w_out {
                    continue;
                }

                let k_idx = (c_g * kh + kh_) * kw + kw_;
                let p_idx = (oh as usize) * (w_out as usize) + (ow as usize);
                let col_idx = ((n * groups + g) * k_g + k_idx) * p + p_idx;
                acc = add_f64_bits(acc, widen_f32_bits(d_col[col_idx].to_bits()));
            }
        }
        *slot = f32::from_bits(narrow_f64_bits(acc));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fandhe_ai_backend_cpu::CpuBackendOps;
    use fandhe_ai_tensor_core::{BackendOps, Conv2dParams, Tensor, im2col_out_shape};

    /// テスト専用の最小限 xorshift64* PRNG
    /// （`scan_model.rs::TestRng` と同型の自己完結実装）。
    struct TestRng(u64);
    impl TestRng {
        fn next_u64(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn next_f32(&mut self) -> f32 {
            let bits = (self.next_u64() >> 32) as u32;
            let sign = if bits & 1 == 0 { 1.0 } else { -1.0 };
            let v = f32::from_bits(bits & 0x7fff_ffff) * sign;
            if v.is_finite() { v } else { 1.0 }
        }
        fn fill_vec(&mut self, n: usize) -> Vec<f32> {
            (0..n).map(|_| self.next_f32()).collect()
        }
    }

    struct Case {
        label: &'static str,
        in_shape: [usize; 4],
        kernel: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        groups: usize,
    }

    /// #1768 実装計画の形状網羅（CUDA `im2col_col2im_parity.rs::CASES`
    /// と同じ意図の 6 ケース: 基本・重なり窓・dilation・depthwise・
    /// 2 groups＋batch>1・padding のみの窓・座標アンダーフロー
    /// （kernel=1・大 padding）・stride > kernel extent・非対称
    /// H/W・1d 形状）。
    const CASES: &[Case] = &[
        Case {
            label: "basic no pad",
            in_shape: [1, 2, 5, 5],
            kernel: [3, 3],
            stride: [1, 1],
            padding: [0, 0],
            dilation: [1, 1],
            groups: 1,
        },
        Case {
            label: "overlapping windows",
            in_shape: [1, 1, 6, 6],
            kernel: [3, 3],
            stride: [1, 1],
            padding: [1, 1],
            dilation: [1, 1],
            groups: 1,
        },
        Case {
            label: "dilation",
            in_shape: [1, 1, 9, 9],
            kernel: [3, 3],
            stride: [1, 1],
            padding: [2, 2],
            dilation: [2, 2],
            groups: 1,
        },
        Case {
            label: "groups depthwise",
            in_shape: [1, 4, 6, 6],
            kernel: [3, 3],
            stride: [1, 1],
            padding: [1, 1],
            dilation: [1, 1],
            groups: 4,
        },
        Case {
            label: "groups non-depthwise batch>1",
            in_shape: [2, 6, 5, 5],
            kernel: [3, 3],
            stride: [2, 2],
            padding: [1, 1],
            dilation: [1, 1],
            groups: 2,
        },
        Case {
            label: "padding-only window",
            in_shape: [1, 1, 2, 2],
            kernel: [3, 3],
            stride: [1, 1],
            padding: [5, 5],
            dilation: [1, 1],
            groups: 1,
        },
        Case {
            label: "coordinate underflow",
            in_shape: [1, 1, 3, 3],
            kernel: [1, 1],
            stride: [1, 1],
            padding: [4, 4],
            dilation: [1, 1],
            groups: 1,
        },
        Case {
            label: "stride > kernel extent",
            in_shape: [1, 1, 7, 7],
            kernel: [2, 2],
            stride: [3, 3],
            padding: [0, 0],
            dilation: [1, 1],
            groups: 1,
        },
        Case {
            label: "asymmetric H/W",
            in_shape: [1, 4, 5, 7],
            kernel: [3, 2],
            stride: [2, 1],
            padding: [3, 1],
            dilation: [2, 3],
            groups: 2,
        },
        Case {
            label: "1d shape (H=1, kh=1)",
            in_shape: [1, 2, 1, 9],
            kernel: [1, 3],
            stride: [1, 1],
            padding: [0, 1],
            dilation: [1, 2],
            groups: 1,
        },
    ];

    fn bits_or_nan_class(v: f32) -> Result<u32, ()> {
        if v.is_nan() { Err(()) } else { Ok(v.to_bits()) }
    }

    fn assert_bits_eq(label: &str, actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len(), "{label}: 要素数が一致しない");
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            match (bits_or_nan_class(a), bits_or_nan_class(e)) {
                (Ok(ba), Ok(be)) => {
                    assert_eq!(ba, be, "{label}: 要素 {i} が bit 一致しない (a={a}, e={e})")
                }
                (Err(()), Err(())) => {}
                _ => panic!("{label}: 要素 {i} が NaN クラス一致しない (a={a}, e={e})"),
            }
        }
    }

    /// [`im2col_model`]／[`col2im_soft_f64`] が
    /// `fandhe_ai_backend_cpu::CpuBackendOps::im2col`／`col2im` と
    /// 全ケースで bit 完全一致することを確認する（モジュール冒頭
    /// コメント「なぜこれが Metal 側の正しさの根拠になるか」参照）。
    #[test]
    fn im2col_and_col2im_models_match_cpu_backend_across_shapes() {
        let cpu = CpuBackendOps::new();
        let mut seed = 0x1234_5678_9abc_def0u64;
        for case in CASES {
            seed = seed.wrapping_add(1);
            let params = Conv2dParams::new(
                case.kernel,
                case.stride,
                case.padding,
                case.dilation,
                case.groups,
            )
            .expect("valid Conv2dParams");
            let numel_in: usize = case.in_shape.iter().product();
            let mut rng = TestRng(seed);
            let input_data = rng.fill_vec(numel_in);
            let input = Tensor::new(input_data.clone(), &case.in_shape).expect("valid tensor");

            let col_shape = im2col_out_shape(&case.in_shape, &params).expect("valid out shape");
            let numel_col: usize = col_shape.iter().product();
            let dims = derive_im2col_dims(&case.in_shape, &col_shape, &params, numel_col)
                .unwrap_or_else(|e| panic!("{}: derive_im2col_dims failed: {e}", case.label));

            let model_out = im2col_model(&input_data, &dims).expect("valid dims");
            let cpu_out = cpu.im2col(&input, &params).expect("cpu im2col succeeds");
            assert_bits_eq(
                &format!("im2col({})", case.label),
                &model_out,
                cpu_out.as_slice().expect("contiguous"),
            );

            // col2im: d_col 側も乱数で埋め、入力位置定常の逐次加算が
            // CPU の `f64` 逐次和と bit 一致することを確認する。
            let mut d_col_rng = TestRng(seed.wrapping_add(0x9e37_79b9));
            let d_col_data = d_col_rng.fill_vec(numel_col);
            let d_col = Tensor::new(d_col_data.clone(), &col_shape).expect("valid tensor");

            let dims_back = derive_im2col_dims(&case.in_shape, &col_shape, &params, numel_in)
                .unwrap_or_else(|e| panic!("{}: derive_im2col_dims(back) failed: {e}", case.label));
            let model_back = col2im_soft_f64(&d_col_data, &dims_back).expect("valid dims");
            let cpu_back = cpu
                .col2im(&d_col, &case.in_shape, &params)
                .expect("cpu col2im succeeds");
            assert_bits_eq(
                &format!("col2im({})", case.label),
                &model_back,
                cpu_back.as_slice().expect("contiguous"),
            );
        }
    }

    /// `derive_im2col_dims` は `P` 軸不整合を `InvalidShape` として
    /// 拒否する（`backend-cuda::im2col::LaunchShape::derive` の
    /// `launch_shape_derive_rejects_p_axis_mismatch` と同型）。
    #[test]
    fn derive_im2col_dims_rejects_p_axis_mismatch() {
        let params = Conv2dParams::new([2, 2], [1, 1], [0, 0], [1, 1], 1).unwrap();
        // in_shape [1,1,4,4] -> h_out=w_out=3 -> P=9（kernel=2,stride=1）。
        // 誤った col_shape（P=1）を渡して不一致を起こす。
        let err = derive_im2col_dims(&[1, 1, 4, 4], &[1, 1, 4, 1], &params, 4).unwrap_err();
        assert!(matches!(err, Im2colPrepareError::InvalidShape { .. }));
    }

    /// `derive_im2col_dims` は `N` 軸不整合（`col_shape[0]` が
    /// `in_shape[0]` と異なる）も `InvalidShape` として拒否する
    /// （codex-review P0 是正の回帰テスト。是正前はこの不一致が
    /// 素通りし、`im2col_model` が `n_batch` を超える `n` で
    /// `input` の範囲外を読み出しえた）。
    #[test]
    fn derive_im2col_dims_rejects_n_axis_mismatch() {
        let params = Conv2dParams::new([1, 1], [1, 1], [0, 0], [1, 1], 1).unwrap();
        // in_shape=[1,1,1,1] -> 正しい col_shape=[1,1,1,1]。
        // N=2 に差し替えた col_shape を渡して不一致を起こす。
        let err = derive_im2col_dims(&[1, 1, 1, 1], &[2, 1, 1, 1], &params, 2).unwrap_err();
        assert!(matches!(err, Im2colPrepareError::InvalidShape { .. }));
    }

    /// `derive_im2col_dims` は `G`／`K_g` 軸不整合（`groups` が
    /// `params.groups()` と異なる・`cin_g*kh*kw` と食い違う）も
    /// `InvalidShape` として拒否する（codex-review P0 是正の回帰
    /// テスト。是正前は `col_shape` の生値をそのまま `groups`／
    /// `k_g` として信頼していたため、`col2im_soft_f64` が `d_col`
    /// の範囲外を読み出しえた）。
    #[test]
    fn derive_im2col_dims_rejects_groups_axis_mismatch() {
        let params = Conv2dParams::new([1, 1], [1, 1], [0, 0], [1, 1], 2).unwrap();
        // in_shape=[1,2,1,1] (groups=2) -> 正しい col_shape=[1,2,1,1]。
        // groups=1 に差し替えた col_shape を渡して不一致を起こす。
        let err = derive_im2col_dims(&[1, 2, 1, 1], &[1, 1, 1, 1], &params, 1).unwrap_err();
        assert!(matches!(err, Im2colPrepareError::InvalidShape { .. }));
    }

    /// rank 不足の `in_shape`（例 `[1, 1, 1]`）は release ビルドでも
    /// panic せず `InvalidShape` を返す（codex-review P1 是正の回帰
    /// テスト。是正前は `debug_assert_eq!` にのみ依存しており、
    /// `in_shape[0..3]` への添字アクセスへ進んでしまい release
    /// ビルドでは範囲外添字アクセスで panic しえた）。
    #[test]
    fn derive_im2col_dims_rejects_rank_deficient_in_shape() {
        let params = Conv2dParams::new([1, 1], [1, 1], [0, 0], [1, 1], 1).unwrap();
        let err = derive_im2col_dims(&[1, 1, 1], &[1, 1, 1, 1], &params, 1).unwrap_err();
        assert!(matches!(err, Im2colPrepareError::InvalidShape { .. }));
    }

    /// rank 不足の `col_shape` も同様に `InvalidShape` を返す
    /// （`in_shape` と対称の検査。`run_im2col_f32` へ `out_shape=
    /// [1,1,1,1]` のようなユーザー由来の値がそのまま渡る経路の防御）。
    #[test]
    fn derive_im2col_dims_rejects_rank_deficient_col_shape() {
        let params = Conv2dParams::new([1, 1], [1, 1], [0, 0], [1, 1], 1).unwrap();
        let err = derive_im2col_dims(&[1, 1, 1, 1], &[1, 1, 1], &params, 1).unwrap_err();
        assert!(matches!(err, Im2colPrepareError::InvalidShape { .. }));
    }

    /// `u32::MAX` を超える形状パラメータは `SizeLimitExceeded` を返す。
    #[test]
    fn derive_im2col_dims_rejects_u32_overflow() {
        let params = Conv2dParams::new([2, 2], [1, 1], [0, 0], [1, 1], 1).unwrap();
        let big = IM2COL_KERNEL_ARG_LIMIT + 1;
        let err = derive_im2col_dims(&[1, 1, 4, 4], &[1, 1, 4, 9], &params, big).unwrap_err();
        assert!(matches!(
            err,
            Im2colPrepareError::SizeLimitExceeded { what: "numel", .. }
        ));
    }

    /// 1d 形状（`H=1`・`kh=1`・`sh=1`・`dh=1`）でも `h_out=1` が正しく
    /// 導出される（`backend-cuda::im2col::
    /// launch_shape_derive_handles_1d_shape` と同型）。
    #[test]
    fn derive_im2col_dims_handles_1d_shape() {
        let params = Conv2dParams::new([1, 3], [1, 2], [0, 1], [1, 2], 1).unwrap();
        // L=9, k=3, stride=2, padding=1, dilation=2 ->
        // lout = floor((9 + 2*1 - 2*(3-1) - 1) / 2) + 1 = floor(6/2)+1 = 4.
        let in_shape = [1usize, 1, 1, 9];
        let dims = derive_im2col_dims(&in_shape, &[1, 1, 3, 4], &params, 4).unwrap();
        assert_eq!(dims.h_out, 1);
        assert_eq!(dims.w_out, 4);
        assert_eq!(dims.p, 4);
        assert_eq!(dims.h_in, 1);
        assert_eq!(dims.kh, 1);
        assert_eq!(dims.sh, 1);
        assert_eq!(dims.ph, 0);
        assert_eq!(dims.dh, 1);
    }

    /// `Im2colDims` は `shaders/im2col.metal::struct Im2colDims` と
    /// バイト単位で同一レイアウト（19 × `u32` = 76 バイト）。
    #[test]
    fn im2col_dims_size_matches_msl_struct() {
        assert_eq!(std::mem::size_of::<Im2colDims>(), 19 * 4);
    }

    /// col2im の「寄与なし」位置（アキュムレータが `+0.0` のまま）は
    /// ホスト `f64` 参照実装の `acc: f64 = 0.0` と同じ `+0.0` になる
    /// （`scan_model.rs` の同種の契約証明テストと同じ意図）。
    #[test]
    fn col2im_no_contribution_position_yields_positive_zero() {
        // kernel=1x1・stride=2・padding=0 の pointwise strided conv
        // では奇数添字の入力位置は「窓」からまったく採らない（寄与
        // なし）ため、それらの位置は `+0.0`（ホスト `acc: f64 = 0.0`
        // のまま）になるはず。
        let params = Conv2dParams::new([1, 1], [2, 2], [0, 0], [1, 1], 1).unwrap();
        let in_shape = [1usize, 1, 4, 4];
        let col_shape = im2col_out_shape(&in_shape, &params).expect("valid out shape");
        let numel_in: usize = in_shape.iter().product();
        let dims = derive_im2col_dims(&in_shape, &col_shape, &params, numel_in).unwrap();
        let numel_col: usize = col_shape.iter().product();
        let d_col = vec![1.0f32; numel_col];
        let out = col2im_soft_f64(&d_col, &dims).unwrap();
        // (h, w) が両方偶数の位置のみ寄与を受ける。奇数を含む位置は
        // `+0.0` のはず。
        for h in 0..4usize {
            for w in 0..4usize {
                let idx = h * 4 + w;
                if h % 2 != 0 || w % 2 != 0 {
                    assert_eq!(
                        out[idx].to_bits(),
                        0.0f32.to_bits(),
                        "(h={h}, w={w}) は寄与なしのはず"
                    );
                } else {
                    assert_eq!(out[idx], 1.0, "(h={h}, w={w}) は寄与ありのはず");
                }
            }
        }
    }

    /// [`im2col_model`]／[`col2im_soft_f64`] は `pub` かつ
    /// `#[cfg(test)]` の外にあるため、`derive_im2col_dims` を経由
    /// しない手構築の `Im2colDims`（PR #1871 codex-review 指摘の
    /// 再現ケース: `in_shape`／`col_shape` とも `[1,1,1,1]`・
    /// `kernel`／`stride`／`dilation` とも `[1,1]`・`padding=[0,0]`・
    /// `groups=1`・`numel=1` という一見正常な値に、範囲外読み出しを
    /// 誘発する空スライスを渡すケース）に対しても panic せず
    /// `Err` を返す（is-panic-free の直接回帰。指摘の「例えば入力・
    /// col 形状がともに `[1,1,1,1]`」を字義どおり再現）。
    #[test]
    fn im2col_model_rejects_empty_input_slice_without_panicking() {
        let params = Conv2dParams::new([1, 1], [1, 1], [0, 0], [1, 1], 1).unwrap();
        let dims = derive_im2col_dims(&[1, 1, 1, 1], &[1, 1, 1, 1], &params, 1).unwrap();
        let err = im2col_model(&[], &dims).unwrap_err();
        assert!(matches!(err, Im2colPrepareError::InvalidShape { .. }));
    }

    /// [`col2im_soft_f64`] 側の同型ケース（`d_col` 空スライス）。
    #[test]
    fn col2im_soft_f64_rejects_empty_d_col_slice_without_panicking() {
        let params = Conv2dParams::new([1, 1], [1, 1], [0, 0], [1, 1], 1).unwrap();
        let dims = derive_im2col_dims(&[1, 1, 1, 1], &[1, 1, 1, 1], &params, 1).unwrap();
        let err = col2im_soft_f64(&[], &dims).unwrap_err();
        assert!(matches!(err, Im2colPrepareError::InvalidShape { .. }));
    }

    /// `Im2colDims` の全フィールドが `pub` であることを悪用し、
    /// `p=0`（`groups`／`k_g` は非ゼロのまま）という `derive_im2col_dims`
    /// を経由しては到達しえない値を直接構築するケース。是正前は
    /// `im2col_model` 内の `rem % p` で 0 除算 panic しえた
    /// （codex-review 指摘「`Im2colDims` のフィールドも公開されている
    /// ため、p=0 等によるゼロ除算も可能」の直接再現）。
    #[test]
    fn im2col_model_rejects_hand_built_dims_with_zero_p_without_panicking() {
        let dims = Im2colDims {
            n_batch: 1,
            cin: 1,
            groups: 1,
            cin_g: 1,
            h_in: 1,
            w_in: 1,
            h_out: 1,
            w_out: 1,
            kh: 1,
            kw: 1,
            sh: 1,
            sw: 1,
            ph: 0,
            pw: 0,
            dh: 1,
            dw: 1,
            k_g: 1,
            p: 0,
            numel: 1,
        };
        let err = im2col_model(&[0.0f32], &dims).unwrap_err();
        assert!(matches!(err, Im2colPrepareError::InvalidShape { .. }));
    }
}
