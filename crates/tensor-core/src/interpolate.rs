//! `InterpolateMode::Bilinear` の座標・重み計算の**単一情報源**
//! （イシュー #1762）。
//!
//! `autodiff::eval::interpolate_bilinear`（forward ホスト参照実装）・
//! `autodiff::grad::bilinear_src_index_and_weight_map`（backward VJP
//! index／重み構築）・`backend-cpu::interpolate::interpolate_bilinear`
//! （CPU ネイティブ実装）・`backend-metal::interpolate_model`（Metal
//! ホスト逐語モデル）はいずれも `tensor-core` に依存できるため、本
//! モジュールの関数を直接呼ぶことで「座標式が実装ごとに独立に書かれ
//! 乖離する」リスクを構造的に排除する（`nearest_src_coord` が
//! クレートごとに複製されているのとは異なる設計——bilinear は算術を
//! 含み乖離の実害が大きいため、共有可能な層（`tensor-core`）へ
//! 一本化する）。CUDA／Metal の GPU カーネル自体は文字列（NVRTC／MSL）
//! のためこの共有の対象外だが、同じ式を逐語で書き写し文字列証跡
//! テストで固定する（`kernels_interpolate.rs`／`shaders/
//! interpolate.metal` 冒頭コメント参照）。
//!
//! # FMA 契約
//!
//! ブレンド式の丸めを 3 バックエンド間で可能な限り揃えるため、座標
//! 計算・ブレンドとも `f32::mul_add`（CUDA `fmaf`／Metal `fma` に
//! 対応）で明示的に FMA 化する（`.claude/rules/coding-rust.md`・
//! `crates/backend-cuda/src/kernels_rnn_cell.rs`
//! `fmaf(z, h, (1-z)*n)` と同型の先例）。ただし NVRTC は `fmad`
//! 既定契約（`nvrtc.rs`。上書き禁止）のため素の `a*b+c` も自動的に
//! 契約されうる——受入契約はあくまで REQ-2 統一複合判定であり
//! `Nearest` のような bit 完全一致は断言しない
//! （[`crate::InterpolateMode::Bilinear`] doc 参照）。

/// 出力座標 `dst` に対応する入力側の 2 近傍添字と補間重みを表す
/// （[`bilinear_src_coord`] の戻り値）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BilinearCoord {
    /// 下側近傍の入力添字（`[0, in_size)`）。
    pub i0: usize,
    /// 上側近傍の入力添字（`[0, in_size)`。境界では `i0` と一致しうる）。
    pub i1: usize,
    /// `i1` 側への補間重み（`[0, 1]`）。`i0` 側の重みは `1.0 - lambda1`
    /// として呼び出し側が導出する。
    pub lambda1: f32,
}

/// 1 軸分の `scale`（`bilinear_src_coord` へ渡す係数。PyTorch
/// `area_pixel_compute_scale` 相当）を計算する（イシュー #1762）。
///
/// - `align_corners=false`: `in_size / out_size`（half-pixel 変換で
///   使う倍率。`out_size==0` は呼び出し元が事前に拒否する契約——
///   `interpolate_out_shape` は空間軸 0 を拒否済み——だが防御的に
///   `0.0` を返す）。
/// - `align_corners=true`: `out_size > 1` なら
///   `(in_size-1) / (out_size-1)`、`out_size <= 1` なら `0.0`
///   （四隅一致の定義上、出力が 1 点のみなら傾きは意味を持たない）。
pub fn bilinear_scale(in_size: usize, out_size: usize, align_corners: bool) -> f32 {
    if out_size == 0 {
        return 0.0;
    }
    if align_corners {
        if out_size > 1 {
            // 整数側で `-1` してから `f32` へ変換する（先に `f32` へ
            // 変換してから `1.0` を引くと、大きい `in_size`
            // （例: 16,777,217 = 2^24+1。f32 の仮数部が表現できる
            // 整数の上限 2^24 を超える）で `in_size as f32` が既に
            // 丸められ、四隅一致契約〈`dst=out_size-1` が
            // `src=in_size-1` へ一致する〉が整数丸めにより破れる。
            // `in_size==0`（呼び出し元が事前に拒否する契約——
            // `interpolate_out_shape` が空間軸 0 を検査済み——だが
            // 縦深防御として本関数でも扱う）は `saturating_sub` で
            // `usize` 減算 underflow を回避し `0` として扱う）。
            (in_size.saturating_sub(1) as f32) / ((out_size - 1) as f32)
        } else {
            0.0
        }
    } else {
        in_size as f32 / out_size as f32
    }
}

/// 出力座標 `dst`（`[0, out_size)`）に対応する入力側の 2 近傍添字と
/// 補間重みを求める（イシュー #1762。[`bilinear_scale`] が求めた
/// `scale` を使う）。
///
/// - `align_corners=false`: `src = (dst + 0.5) * scale - 0.5`
///   （half-pixel 変換。`fma(dst + 0.5, scale, -0.5)`。`src < 0` は
///   `0.0` へクランプ——PyTorch と同じく、入力の外側へ出る半ピクセル
///   分は端点で飽和させる）。
/// - `align_corners=true`: `src = dst * scale`（四隅一致。非負・
///   `in_size-1` 以下に収まる設計のため下限クランプは不要だが、
///   `in_size==0` 等の縮退ケースに備え `i0`／`i1` 側で防御的に
///   クランプする）。
///
/// `i0 = min(floor(src), in_size-1)`（REQ-8 の縦深防御クランプ。
/// `in_size==0` は呼び出し元が事前に拒否する契約——`interpolate_
/// out_shape`（`crates/tensor-core/src/ops_shape.rs`）は入力側の
/// 空間軸 0 を `shape[axis] == 0` として明示検査済みのため
/// `Var::interpolate` 経由では本関数へ `in_size==0` は到達しない——
/// が、本関数はそれでも縦深防御として `in_size==0` でも `0` を返し
/// `usize` 減算 underflow を起こさない）。
/// `i1 = min(i0+1, in_size-1)`。`lambda1 = src - i0 as f32`
/// （クランプ後の `src` との差。`in_size==1` や `dst` が上限に達する
/// 境界では `i0==i1` になり `lambda1` が非ゼロでも重複コーナーとして
/// 扱う——重みの和は `(1-lambda1) + lambda1 == 1.0` のまま保たれる）。
pub fn bilinear_src_coord(
    dst: usize,
    in_size: usize,
    scale: f32,
    align_corners: bool,
) -> BilinearCoord {
    if in_size == 0 {
        return BilinearCoord {
            i0: 0,
            i1: 0,
            lambda1: 0.0,
        };
    }
    let src = if align_corners {
        dst as f32 * scale
    } else {
        (dst as f32 + 0.5).mul_add(scale, -0.5).max(0.0)
    };
    let i0f = src.floor();
    // `i0f` は非負（`align_corners=false` は `max(0.0)` 済み・
    // `align_corners=true` は `dst`・`scale` とも非負のため `src>=0`）。
    // `in_size-1` を超えないよう縦深防御クランプする。
    let i0 = (i0f as usize).min(in_size - 1);
    let i1 = (i0 + 1).min(in_size - 1);
    let lambda1 = src - i0 as f32;
    BilinearCoord { i0, i1, lambda1 }
}

/// 4 近傍値と行・列方向の重みから bilinear 出力値を合成する
/// （forward・backward の重み計算双方が使う固定ブレンド式。CUDA
/// `fmaf`／Metal `fma`／Rust `mul_add` へ対応する構成で書く——3
/// バックエンドとも同じ式順序で丸めを揃える意図。[`crate::
/// InterpolateMode::Bilinear`] doc 参照）。
///
/// `v00`/`v01`/`v10`/`v11` は `(y0,x0)`/`(y0,x1)`/`(y1,x0)`/`(y1,x1)`。
/// `l1x`/`l1y` は [`bilinear_src_coord`] が返す `lambda1`（x／y 軸別）。
pub fn bilinear_blend(v00: f32, v01: f32, v10: f32, v11: f32, l1x: f32, l1y: f32) -> f32 {
    let l0x = 1.0 - l1x;
    let l0y = 1.0 - l1y;
    let row0 = l1x.mul_add(v01, l0x * v00);
    let row1 = l1x.mul_add(v11, l0x * v10);
    l1y.mul_add(row1, l0y * row0)
}

/// `dst`（`[0, out_size)`）に対応する `nearest-exact` の入力側添字
/// （イシュー #2152。`torch.nn.functional.interpolate(mode=
/// 'nearest-exact')` 相当）。[`crate::InterpolateMode::NearestExact`]
/// が使う唯一の情報源（`autodiff::eval::interpolate_nearest_exact`・
/// `autodiff::grad::nearest_exact_src_index_map`・
/// `backend_cpu::interpolate::interpolate_nearest_exact` が共有）。
///
/// PyTorch の `nearest-exact` は `min(floor((dst+0.5)*in/out), in-1)`
/// （`f32` 計算）。本関数は float を使わず `((2*dst+1)*in) / (2*out)`
/// を `u128` で計算する整数専用の等価式とする（`(dst+0.5)*in/out ==
/// (2*dst+1)*in / (2*out)` を整数のまま床除算する形。`nearest_src_
/// coord` と同型の overflow 対策——`dst`・`in_size` とも `usize::MAX`
/// でも `u128` の範囲に収まる。実装計画 §3.1）。`Nearest`（既定の
/// `src = (dst*in)/out`）とは異なる添字式であり、`(2*dst+1)` の
/// half-pixel オフセットの有無が主な差（例: `in=out` の恒等写像では
/// 両者とも `src=dst` に一致するが、非整数比では 1 要素ずれうる）。
///
/// `out_size==0`／`in_size==0` は呼び出し元が事前に拒否する契約だが
/// （`interpolate_out_shape` の空間軸 0 検査）、`Nearest` 同様に
/// 縦深防御として `0` を返す（ゼロ除算 panic を避ける）。
pub fn nearest_exact_src_coord(dst: usize, in_size: usize, out_size: usize) -> usize {
    if out_size == 0 || in_size == 0 {
        return 0;
    }
    let numer = (2u128 * dst as u128 + 1) * in_size as u128;
    let denom = 2u128 * out_size as u128;
    let src = numer / denom;
    (src as usize).min(in_size - 1)
}

/// 2 個の入力近傍値と補間重みから linear（1 軸）出力値を合成する
/// （イシュー #2152。`bilinear_blend` の行方向補間と同じ式順序——
/// `l1.mul_add(v1, (1-l1)*v0)`。`torch.nn.functional.interpolate
/// (mode='linear')` 相当）。`v0`/`v1` は [`bilinear_src_coord`]（1 軸
/// 分の呼び出し）が返す `i0`/`i1` 側の値、`l1` は同関数の `lambda1`。
pub fn linear_blend(v0: f32, v1: f32, l1: f32) -> f32 {
    l1.mul_add(v1, (1.0 - l1) * v0)
}

/// 8 個の入力近傍値（`z0` 面 4 個・`z1` 面 4 個）と 3 軸分の補間重みから
/// trilinear 出力値を合成する（イシュー #2152。`torch.nn.functional.
/// interpolate(mode='trilinear')` 相当）。`z0` 面・`z1` 面をそれぞれ
/// [`bilinear_blend`]（`(y,x)` の 2 軸補間）で合成してから、
/// [`linear_blend`] と同じ `fma` 式で `z` 軸方向に結ぶ固定順序
/// （実装計画 §3.3）。
///
/// `v000`/`v001`/`v010`/`v011` は `z0` 面の `(y0,x0)`/`(y0,x1)`/
/// `(y1,x0)`/`(y1,x1)`、`v100`/`v101`/`v110`/`v111` は `z1` 面の同順。
/// `l1x`/`l1y`/`l1z` は [`bilinear_src_coord`]（x／y／z 軸それぞれ独立
/// に呼ぶ）が返す `lambda1`。
#[allow(clippy::too_many_arguments)]
pub fn trilinear_blend(
    v000: f32,
    v001: f32,
    v010: f32,
    v011: f32,
    v100: f32,
    v101: f32,
    v110: f32,
    v111: f32,
    l1x: f32,
    l1y: f32,
    l1z: f32,
) -> f32 {
    let p0 = bilinear_blend(v000, v001, v010, v011, l1x, l1y);
    let p1 = bilinear_blend(v100, v101, v110, v111, l1x, l1y);
    linear_blend(p0, p1, l1z)
}

/// bicubic convolution の重み係数（`A = -0.75`。PyTorch
/// `cubic_convolution1`/`cubic_convolution2` と同一。`|x|<=1` 側の式）。
fn cubic_convolution1(x: f32, a: f32) -> f32 {
    ((a + 2.0) * x - (a + 3.0)).mul_add(x * x, 1.0)
}

/// bicubic convolution の重み係数（`1<|x|<2` 側の式）。
fn cubic_convolution2(x: f32, a: f32) -> f32 {
    (((a * x - 5.0 * a) * x + 8.0 * a) * x) - 4.0 * a
}

/// bicubic の 1 軸 4-tap 重み（PyTorch `get_cubic_upsample_
/// coefficients` と同一の `A=-0.75` 固定係数。`t` は `floor(src)` から
/// の小数部 `[0,1)`）。
fn cubic_upsample_coefficients(t: f32) -> [f32; 4] {
    const A: f32 = -0.75;
    [
        cubic_convolution2(t + 1.0, A),
        cubic_convolution1(t, A),
        cubic_convolution1(1.0 - t, A),
        cubic_convolution2(2.0 - t, A),
    ]
}

/// [`bicubic_src_taps`] の戻り値: 1 軸分の bicubic 4-tap 入力添字と重み。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BicubicTaps {
    /// 4 個の入力側 tap 添字（`[0, in_size)` にクランプ済み。順序は
    /// `floor(src)-1 ..= floor(src)+2`）。
    pub idx: [usize; 4],
    /// 対応する重み（`cubic_upsample_coefficients`。和は理論上 1）。
    pub w: [f32; 4],
}

/// 出力座標 `dst` に対応する bicubic の 1 軸 4-tap 添字・重みを求める
/// （イシュー #2152。PyTorch `upsample_get_value_bounded` 相当の境界
/// クランプ + `cubic_interp1d` の係数。実装計画 §3.4）。
///
/// PyTorch `area_pixel_compute_source_index(..., cubic=true)` は
/// `align_corners=false` でも `src<0` を**クランプしない**
/// （`UpSample.h`: `(!cubic && src_idx < 0) ? 0 : src_idx` —— cubic
/// 補間は `[-1,0,1,2]` の近傍参照が必要なため。実装前に PyTorch
/// ソースで確認済み。`bilinear_src_coord` の `max(0.0)` クランプとは
/// 異なる契約）。
///
/// `i = floor(src)`（`isize`。負値になりうる）、`t = src - i`。4 tap
/// `i-1..=i+2` を `[0, in_size-1]` へ個別にクランプする（REQ-8 の
/// 境界検査。クランプで重複した tap もそのまま加算する——PyTorch と
/// 同じ端点飽和）。`in_size==0` は呼び出し元が事前に拒否する契約だが
/// 縦深防御として `idx` を全て `0`・`t=0` 相当（重み `[0,1,0,0]`）
/// として扱う。
pub fn bicubic_src_taps(
    dst: usize,
    in_size: usize,
    scale: f32,
    align_corners: bool,
) -> BicubicTaps {
    if in_size == 0 {
        return BicubicTaps {
            idx: [0; 4],
            w: [0.0, 1.0, 0.0, 0.0],
        };
    }
    let src = if align_corners {
        dst as f32 * scale
    } else {
        (dst as f32 + 0.5).mul_add(scale, -0.5)
    };
    let i = src.floor();
    let t = src - i;
    // `in_size` は `usize`（>=1 に確定済み）なので `isize` へ変換
    // しても `in_size - 1` は非負に収まる（通常の interpolate shape
    // 上限内であれば `isize` の範囲を超えない——巨大 shape は
    // `interpolate_out_shape` のバイトサイズ上限が事前に拒否する）。
    let i_isize = i as isize;
    let last = (in_size - 1) as isize;
    let mut idx = [0usize; 4];
    for (k, slot) in idx.iter_mut().enumerate() {
        let tap = i_isize - 1 + k as isize;
        *slot = tap.clamp(0, last) as usize;
    }
    BicubicTaps {
        idx,
        w: cubic_upsample_coefficients(t),
    }
}

/// 4x4 近傍値（`v[y][x]`。`y`/`x` は [`bicubic_src_taps`] が返す
/// `idx` の並び順）と x／y 軸別の重みから bicubic 出力値を合成する
/// （イシュー #2152。行方向〈x〉の 4-tap 補間を 4 行分行ってから、
/// 列方向〈y〉の 4-tap 補間を行う固定順序。PyTorch `cubic_interp1d`
/// の入れ子順と同じ。`fma` 連鎖で構成する——[`bilinear_blend`] と
/// 同じ FMA 契約の方針）。出力値はクランプしない（PyTorch と同じく
/// overshoot を許す）。
pub fn bicubic_blend(v: [[f32; 4]; 4], wx: [f32; 4], wy: [f32; 4]) -> f32 {
    let mut rows = [0f32; 4];
    for (j, row) in v.iter().enumerate() {
        rows[j] = wx[3].mul_add(
            row[3],
            wx[2].mul_add(row[2], wx[1].mul_add(row[1], wx[0] * row[0])),
        );
    }
    wy[3].mul_add(
        rows[3],
        wy[2].mul_add(rows[2], wy[1].mul_add(rows[1], wy[0] * rows[0])),
    )
}

/// [`interpolate_size_from_scale_factor`] のエラー型（イシュー
/// #2152）。`tensor-core` は内部クレートのため（`docs/compat-api-
/// scope.md` §0）facade からは再エクスポートされず、facade 公開面の
/// 拡張にはならない（承認事項なしの判断根拠。実装計画 §3.6）。
///
/// `#[non_exhaustive]`: 将来の検査項目追加に備え非破壊を保つ
/// （`ScatterReduce`／`Activation` と同方針）。
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScaleFactorError {
    /// `spatial_in.len() != scale_factor.len()`。
    LengthMismatch {
        /// 空間軸数。
        spatial_rank: usize,
        /// `scale_factor` の長さ。
        scale_factor_len: usize,
    },
    /// `scale_factor[axis]` が非有限（NaN／inf）または `<= 0.0`。
    InvalidScaleFactor {
        /// 軸番号。
        axis: usize,
        /// 拒否した値。
        value: f64,
    },
    /// `floor(in * scale_factor[axis])` が `usize` へ収まらない
    /// （非有限になる場合を含む）。
    Overflow {
        /// 軸番号。
        axis: usize,
    },
    /// 導出した出力サイズが `0`（`interpolate_out_shape` が事前に
    /// 拒否する空間軸 0 と同じ契約を scale_factor 経路でも守るため）。
    ZeroOutputSize {
        /// 軸番号。
        axis: usize,
    },
}

impl std::fmt::Display for ScaleFactorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LengthMismatch {
                spatial_rank,
                scale_factor_len,
            } => write!(
                f,
                "scale_factor の長さ {scale_factor_len} が空間軸数 {spatial_rank} と一致しない"
            ),
            Self::InvalidScaleFactor { axis, value } => write!(
                f,
                "scale_factor[{axis}] = {value} は非有限または 0 以下で不正"
            ),
            Self::Overflow { axis } => {
                write!(f, "軸 {axis} の出力サイズ導出が usize の範囲を超える")
            }
            Self::ZeroOutputSize { axis } => {
                write!(f, "軸 {axis} の導出後の出力サイズが 0")
            }
        }
    }
}

impl std::error::Error for ScaleFactorError {}

/// `scale_factor` から interpolate の `size` 引数を導出する（イシュー
/// #2152。PyTorch `torch.nn.functional.interpolate(scale_factor=…)`
/// 相当。`Var::interpolate` に新しいメソッドは追加せず、本関数の戻り
/// 値を既存の `size` 引数へそのまま渡す設計——実装計画 §3.6「配置
/// 判断」）。
///
/// `out[axis] = floor(spatial_in[axis] as f64 * scale_factor[axis])`
/// （PyTorch と同じ）。これは `recompute_scale_factor=True` と同じ
/// 座標系になる——既定（`None`／`False`）では座標計算に
/// `1/scale_factor` をそのまま使うため、`in*s` が整数でない場合は
/// 結果がわずかに異なりうる（整数倍では一致する。実装計画・対象外
/// §7 に明記）。
///
/// 検査は fail-closed: 長さ不一致・非有限／0 以下の `scale_factor`・
/// `usize` への変換オーバーフロー・導出後の出力サイズ 0 をそれぞれ
/// 拒否する。`out_f >= usize::MAX as f64` を **`as usize` へ変換する
/// 前に**検査する（`as` は範囲外を無言で飽和させるため、先に検査
/// しないと fail-closed の意図に反する。実装計画・セキュリティ考慮
/// §6）。
pub fn interpolate_size_from_scale_factor(
    spatial_in: &[usize],
    scale_factor: &[f64],
) -> Result<Vec<usize>, ScaleFactorError> {
    if spatial_in.len() != scale_factor.len() {
        return Err(ScaleFactorError::LengthMismatch {
            spatial_rank: spatial_in.len(),
            scale_factor_len: scale_factor.len(),
        });
    }
    let mut out = Vec::with_capacity(spatial_in.len());
    for (axis, (&in_sz, &s)) in spatial_in.iter().zip(scale_factor.iter()).enumerate() {
        if !s.is_finite() || s <= 0.0 {
            return Err(ScaleFactorError::InvalidScaleFactor { axis, value: s });
        }
        let out_f = in_sz as f64 * s;
        // `as usize` へ変換する前に範囲を検査する（`as` は範囲外を
        // 無言で飽和させ fail-closed の意図に反するため）。
        if !out_f.is_finite() || out_f >= usize::MAX as f64 {
            return Err(ScaleFactorError::Overflow { axis });
        }
        let out_sz = out_f.floor() as usize;
        if out_sz == 0 {
            return Err(ScaleFactorError::ZeroOutputSize { axis });
        }
        out.push(out_sz);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bilinear_scale_align_corners_false_matches_pytorch_ratio() {
        // PyTorch: scale = in / out（align_corners=False）。
        assert_eq!(bilinear_scale(4, 8, false), 0.5);
        assert_eq!(bilinear_scale(8, 4, false), 2.0);
    }

    #[test]
    fn bilinear_scale_align_corners_true_matches_pytorch_ratio() {
        // PyTorch: scale = (in-1) / (out-1)（align_corners=True）。
        assert_eq!(bilinear_scale(5, 9, true), 4.0 / 8.0);
        assert_eq!(bilinear_scale(9, 5, true), 8.0 / 4.0);
    }

    #[test]
    fn bilinear_scale_align_corners_true_single_output_is_zero() {
        assert_eq!(bilinear_scale(4, 1, true), 0.0);
        assert_eq!(bilinear_scale(4, 0, true), 0.0);
    }

    #[test]
    fn bilinear_scale_align_corners_true_large_in_size_preserves_corner_match() {
        // in_size=16,777,217（2^24+1）は f32 の仮数部が正確に表現できる
        // 整数の上限（2^24）を超えるため、`in_size as f32` へ先に変換
        // してから `1.0` を引くと丸めで `in_size-1` と一致しなくなる。
        // 整数側で `-1` してから変換すれば `dst=out_size-1` が
        // `src=in_size-1` へ厳密に一致する（四隅一致契約）ことを固定する。
        let in_size = 16_777_217usize;
        let out_size = in_size;
        let scale = bilinear_scale(in_size, out_size, true);
        let c_last = bilinear_src_coord(out_size - 1, in_size, scale, true);
        assert_eq!(c_last.i0, in_size - 1);
        assert_eq!(c_last.i1, in_size - 1);
        assert_eq!(c_last.lambda1, 0.0);
    }

    #[test]
    fn bilinear_scale_align_corners_true_in_size_zero_no_underflow_panic() {
        // `in_size==0` は呼び出し元（`interpolate_out_shape`）が事前に
        // 拒否する契約だが、`saturating_sub` により `usize` 減算
        // underflow で panic しないことを縦深防御として固定する。
        assert_eq!(bilinear_scale(0, 4, true), 0.0);
    }

    #[test]
    fn bilinear_src_coord_align_corners_false_center_pixel() {
        // in=out=1: scale=1、src=(0+0.5)*1-0.5=0.0 -> i0=i1=0, lambda1=0.
        let c = bilinear_src_coord(0, 1, bilinear_scale(1, 1, false), false);
        assert_eq!(c.i0, 0);
        assert_eq!(c.i1, 0);
        assert_eq!(c.lambda1, 0.0);
    }

    #[test]
    fn bilinear_src_coord_align_corners_false_negative_clamped_to_zero() {
        // in=1,out=4: scale=0.25。dst=0: src=(0.5)*0.25-0.5 = -0.375 -> 0 にクランプ。
        let scale = bilinear_scale(1, 4, false);
        let c = bilinear_src_coord(0, 1, scale, false);
        assert_eq!(c.i0, 0);
        assert_eq!(c.i1, 0);
        assert_eq!(c.lambda1, 0.0);
    }

    #[test]
    fn bilinear_src_coord_align_corners_true_endpoints_match_input_endpoints() {
        // in=5,out=9,align_corners=true: dst=0 -> src=0; dst=8(last) -> src=4(last input idx)。
        let scale = bilinear_scale(5, 9, true);
        let c0 = bilinear_src_coord(0, 5, scale, true);
        assert_eq!(c0.i0, 0);
        assert_eq!(c0.lambda1, 0.0);
        let c_last = bilinear_src_coord(8, 5, scale, true);
        assert_eq!(c_last.i0, 4);
        assert_eq!(c_last.i1, 4);
        assert_eq!(c_last.lambda1, 0.0);
    }

    #[test]
    fn bilinear_src_coord_in_size_one_all_map_to_same_single_source() {
        // `in_size==1` では `i0==i1==0` が常に成立する（唯一の入力
        // 要素以外を参照しえない）。`lambda1` は half-pixel 変換上
        // 非ゼロになりうるが、`i0==i1` のため `bilinear_blend` の
        // 出力には影響しない（`lerp(v,v,lambda)==v`）——本テストは
        // その添字側の不変条件のみを検証する。
        let scale = bilinear_scale(1, 5, false);
        for dst in 0..5 {
            let c = bilinear_src_coord(dst, 1, scale, false);
            assert_eq!(c.i0, 0);
            assert_eq!(c.i1, 0);
        }
    }

    #[test]
    fn bilinear_src_coord_in_size_zero_does_not_underflow() {
        let c = bilinear_src_coord(0, 0, 1.0, false);
        assert_eq!(c.i0, 0);
        assert_eq!(c.i1, 0);
        assert_eq!(c.lambda1, 0.0);
    }

    #[test]
    fn bilinear_blend_identity_when_all_corners_equal() {
        assert_eq!(bilinear_blend(3.0, 3.0, 3.0, 3.0, 0.37, 0.81), 3.0);
    }

    #[test]
    fn bilinear_blend_pure_corner_selection_at_extremes() {
        // l1x=0,l1y=0 -> v00; l1x=1,l1y=1 -> v11.
        assert_eq!(bilinear_blend(1.0, 2.0, 3.0, 4.0, 0.0, 0.0), 1.0);
        assert_eq!(bilinear_blend(1.0, 2.0, 3.0, 4.0, 1.0, 1.0), 4.0);
    }

    #[test]
    fn bilinear_blend_weights_sum_to_one_midpoint() {
        // 全コーナー同値でない場合の中点補間: 単純平均になるはず。
        let out = bilinear_blend(0.0, 2.0, 4.0, 6.0, 0.5, 0.5);
        assert_eq!(out, 3.0);
    }

    #[test]
    fn nearest_exact_src_coord_matches_pytorch_hand_computed() {
        // PyTorch nearest-exact: floor((dst+0.5)*in/out)。in=out=4 は恒等。
        for dst in 0..4 {
            assert_eq!(nearest_exact_src_coord(dst, 4, 4), dst);
        }
        // in=8,out=3: floor((dst+0.5)*8/3): dst=0->1, dst=1->4, dst=2->6。
        assert_eq!(nearest_exact_src_coord(0, 8, 3), 1);
        assert_eq!(nearest_exact_src_coord(1, 8, 3), 4);
        assert_eq!(nearest_exact_src_coord(2, 8, 3), 6);
    }

    #[test]
    fn nearest_exact_src_coord_differs_from_nearest_for_non_integer_ratio() {
        // nearest-exact は half-pixel オフセットを持つため Nearest
        // （floor(dst*in/out)）と一般に異なる添字を返す。
        // Nearest（`src = (dst*in)/out`）は dst=0 で常に src=0。
        assert_ne!(nearest_exact_src_coord(0, 8, 3), 0);
    }

    #[test]
    fn nearest_exact_src_coord_huge_in_size_does_not_overflow() {
        let in_size = 1usize << 63;
        let src = nearest_exact_src_coord(2, in_size, 3);
        assert!(src < in_size);
    }

    #[test]
    fn nearest_exact_src_coord_zero_sizes_return_zero() {
        assert_eq!(nearest_exact_src_coord(0, 0, 4), 0);
        assert_eq!(nearest_exact_src_coord(0, 4, 0), 0);
    }

    #[test]
    fn linear_blend_matches_bilinear_row_blend() {
        assert_eq!(linear_blend(1.0, 3.0, 0.5), 2.0);
        assert_eq!(linear_blend(1.0, 3.0, 0.0), 1.0);
        assert_eq!(linear_blend(1.0, 3.0, 1.0), 3.0);
    }

    #[test]
    fn trilinear_blend_identity_when_all_corners_equal() {
        assert_eq!(
            trilinear_blend(5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 0.3, 0.6, 0.9),
            5.0
        );
    }

    #[test]
    fn trilinear_blend_pure_corner_selection_at_extremes() {
        // l1x=l1y=l1z=0 -> v000; 全て1 -> v111。
        let vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert_eq!(
            trilinear_blend(
                vals[0], vals[1], vals[2], vals[3], vals[4], vals[5], vals[6], vals[7], 0.0, 0.0,
                0.0
            ),
            vals[0]
        );
        assert_eq!(
            trilinear_blend(
                vals[0], vals[1], vals[2], vals[3], vals[4], vals[5], vals[6], vals[7], 1.0, 1.0,
                1.0
            ),
            vals[7]
        );
    }

    #[test]
    fn cubic_upsample_coefficients_sum_to_one() {
        for t in [0.0f32, 0.25, 0.5, 0.75, 0.9999] {
            let w = cubic_upsample_coefficients(t);
            let sum: f32 = w.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "t={t} sum={sum}");
        }
    }

    #[test]
    fn cubic_upsample_coefficients_at_zero_selects_second_tap() {
        // t=0: PyTorch cubic_interp1d は w1=1・他 0 になる（中央 tap 一致）。
        let w = cubic_upsample_coefficients(0.0);
        assert!((w[1] - 1.0).abs() < 1e-6);
        assert!(w[0].abs() < 1e-6);
        assert!(w[2].abs() < 1e-6);
        assert!(w[3].abs() < 1e-6);
    }

    #[test]
    fn bicubic_src_taps_align_corners_false_allows_negative_src_no_clamp() {
        // align_corners=false・cubic=true は PyTorch と同じく src<0 を
        // クランプしない（UpSample.h `area_pixel_compute_source_index`
        // の `!cubic` 条件。実装計画 §3.4 で確認済み）。in=4,out=8:
        // scale=0.5。dst=0: src=(0.5)*0.5-0.5=-0.25 -> floor=-1 ->
        // タップは [-2,-1,0,1] を [0,3] へクランプ -> [0,0,0,1]。
        let taps = bicubic_src_taps(0, 4, 0.5, false);
        assert_eq!(taps.idx, [0, 0, 0, 1]);
    }

    #[test]
    fn bicubic_src_taps_clamps_to_input_bounds() {
        // 右端付近でも 4 タップが [0, in_size-1] にクランプされる。
        let taps = bicubic_src_taps(7, 4, 0.5, true);
        for &i in &taps.idx {
            assert!(i < 4);
        }
    }

    #[test]
    fn bicubic_blend_identity_when_all_values_equal() {
        let v = [[3.0f32; 4]; 4];
        let wx = cubic_upsample_coefficients(0.3);
        let wy = cubic_upsample_coefficients(0.7);
        let out = bicubic_blend(v, wx, wy);
        assert!((out - 3.0).abs() < 1e-4);
    }

    #[test]
    fn interpolate_size_from_scale_factor_doubles_each_axis() {
        let out = interpolate_size_from_scale_factor(&[4, 6], &[2.0, 0.5]).unwrap();
        assert_eq!(out, vec![8, 3]);
    }

    #[test]
    fn interpolate_size_from_scale_factor_floors_non_integer_result() {
        let out = interpolate_size_from_scale_factor(&[5], &[1.5]).unwrap();
        assert_eq!(out, vec![7]); // floor(5*1.5)=floor(7.5)=7
    }

    #[test]
    fn interpolate_size_from_scale_factor_rejects_length_mismatch() {
        let err = interpolate_size_from_scale_factor(&[4, 4], &[2.0]).unwrap_err();
        assert!(matches!(err, ScaleFactorError::LengthMismatch { .. }));
    }

    #[test]
    fn interpolate_size_from_scale_factor_rejects_non_finite_and_non_positive() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -1.0] {
            let err = interpolate_size_from_scale_factor(&[4], &[bad]).unwrap_err();
            assert!(matches!(err, ScaleFactorError::InvalidScaleFactor { .. }));
        }
    }

    #[test]
    fn interpolate_size_from_scale_factor_rejects_zero_output_size() {
        let err = interpolate_size_from_scale_factor(&[1], &[0.1]).unwrap_err();
        assert!(matches!(err, ScaleFactorError::ZeroOutputSize { .. }));
    }

    #[test]
    fn interpolate_size_from_scale_factor_rejects_usize_overflow_without_saturating() {
        // `in * s` が usize::MAX を大きく超える組合せは `as usize` の
        // 無言飽和に頼らず Overflow を返す（`as` を検査前に使わない
        // fail-closed 契約）。
        let err = interpolate_size_from_scale_factor(&[usize::MAX], &[2.0]).unwrap_err();
        assert!(matches!(err, ScaleFactorError::Overflow { .. }));
    }
}
