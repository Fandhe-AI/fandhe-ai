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
            (in_size as f32 - 1.0) / (out_size as f32 - 1.0)
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
/// out_shape` は入力側の空間軸 0 を明示検査しないが `Var::interpolate`
/// は `in_shape` の空間軸が 0 の場合も出力を確定できる——本関数は
/// `in_size==0` でも `0` を返し `usize` 減算 underflow を起こさない）。
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
}
