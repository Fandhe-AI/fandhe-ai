//! `kernels_pooling.rs`（[`MAX_POOL2D_F32`]／[`AVG_POOL2D_F32`]／
//! [`ADAPTIVE_AVG_POOL2D_F32`]）の逐語ホストモデル（GPU 非依存な純
//! Rust 関数群。`sort_model.rs`／`unique_model.rs` と同じ「意図的複製」
//! 方針）。
//!
//! [`docs/pooling-ops-design.md`] §5／§7 の走査順・更新規則・`f64`
//! 縮約契約をそのまま Rust へ書き下し、CUDA 実機 `#[ignore]` テスト
//! （`pooling_real_device_tests.rs`。本モジュールと同じく `pooling.rs`
//! 末尾から `#[path]` 登録）が GPU 出力（値・索引）との **bit 完全
//! 一致**オラクルとして使う。3 関数とも本モジュール専用の
//! `#[cfg(test)]`（`CudaPooling::run_*` は本番の CUDA 実行経路自体を
//! 持つため cfg(test) にしないが、本モジュールはホスト検証専用の
//! ため cfg(test) が適切。`unique_model.rs::bitonic_step_host` と同型）。
//!
//! [`MAX_POOL2D_F32`]: crate::kernels_pooling::MAX_POOL2D_F32
//! [`AVG_POOL2D_F32`]: crate::kernels_pooling::AVG_POOL2D_F32
//! [`ADAPTIVE_AVG_POOL2D_F32`]: crate::kernels_pooling::ADAPTIVE_AVG_POOL2D_F32

#![cfg(test)]

/// [`crate::kernels_pooling::MAX_POOL2D_F32`] の逐語モデル。`in_shape`／
/// `out_shape` は `[N, C, H, W]`。戻り値は `(values, indices)`
/// （row-major・`values.len() == indices.len() == numel(out_shape)`）。
///
/// 先勝ち決定的タイ規則・NaN 伝播（最初に現れた NaN の索引を保持）・
/// 「`-INFINITY` 番兵ではなく最初の有効タップで初期化」というカーネル
/// 側の実装契約をそのまま再現する（設計 doc §5）。呼び出し元
/// （`pooling.rs::CudaPooling::run_max_pool2d_f32` の shape 検証済み
/// パラメータを想定）が「すべての窓が少なくとも 1 つの有効タップを
/// 含む」ことを保証する契約のため、本モデルは有効タップが 1 つも
/// 見つからない窓に遭遇した場合 `panic!` する（起こり得ない契約違反を
/// 沈黙で見逃さない）。
pub(crate) fn max_pool2d_model(
    input: &[f32],
    in_shape: [usize; 4],
    out_shape: [usize; 4],
    kernel: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
) -> (Vec<f32>, Vec<i32>) {
    let [n, c, h_in, w_in] = in_shape;
    let [_, _, h_out, w_out] = out_shape;
    let numel_out = n * c * h_out * w_out;
    let mut values = vec![0.0f32; numel_out];
    let mut indices = vec![0i32; numel_out];

    for nn in 0..n {
        for cc in 0..c {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut best: Option<(f32, i32)> = None;
                    for kh_ in 0..kernel[0] {
                        let h = oh as i64 * stride[0] as i64 + kh_ as i64 * dilation[0] as i64
                            - padding[0] as i64;
                        if h < 0 || h >= h_in as i64 {
                            continue;
                        }
                        let h = h as usize;
                        for kw_ in 0..kernel[1] {
                            let w = ow as i64 * stride[1] as i64 + kw_ as i64 * dilation[1] as i64
                                - padding[1] as i64;
                            if w < 0 || w >= w_in as i64 {
                                continue;
                            }
                            let w = w as usize;
                            let v = input[((nn * c + cc) * h_in + h) * w_in + w];
                            let cur_idx = (h * w_in + w) as i32;
                            best = Some(match best {
                                None => (v, cur_idx),
                                Some((b, bi)) => {
                                    if v > b || (v.is_nan() && !b.is_nan()) {
                                        (v, cur_idx)
                                    } else {
                                        (b, bi)
                                    }
                                }
                            });
                        }
                    }
                    let (v, idx) = best.expect(
                        "max_pool2d_model: window has no valid tap; caller must \
                         validate shape/dilation via pooling.rs::validate_and_shape",
                    );
                    let out_idx = ((nn * c + cc) * h_out + oh) * w_out + ow;
                    values[out_idx] = v;
                    indices[out_idx] = idx;
                }
            }
        }
    }
    (values, indices)
}

/// [`crate::kernels_pooling::AVG_POOL2D_F32`] の逐語モデル（`f64`
/// 逐次加算・1 回 downcast。設計 doc §7）。`dilation` は常に `[1, 1]`。
pub(crate) fn avg_pool2d_model(
    input: &[f32],
    in_shape: [usize; 4],
    out_shape: [usize; 4],
    kernel: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    count_include_pad: bool,
) -> Vec<f32> {
    let [n, c, h_in, w_in] = in_shape;
    let [_, _, h_out, w_out] = out_shape;
    let mut values = vec![0.0f32; n * c * h_out * w_out];

    for nn in 0..n {
        for cc in 0..c {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut acc = 0.0f64;
                    let mut valid_count: i64 = 0;
                    for kh_ in 0..kernel[0] {
                        let h = oh as i64 * stride[0] as i64 + kh_ as i64 - padding[0] as i64;
                        if h < 0 || h >= h_in as i64 {
                            continue;
                        }
                        let h = h as usize;
                        for kw_ in 0..kernel[1] {
                            let w = ow as i64 * stride[1] as i64 + kw_ as i64 - padding[1] as i64;
                            if w < 0 || w >= w_in as i64 {
                                continue;
                            }
                            let w = w as usize;
                            let v = input[((nn * c + cc) * h_in + h) * w_in + w];
                            acc += f64::from(v);
                            valid_count += 1;
                        }
                    }
                    let divisor: i64 = if count_include_pad {
                        (kernel[0] * kernel[1]) as i64
                    } else {
                        valid_count
                    };
                    let out_idx = ((nn * c + cc) * h_out + oh) * w_out + ow;
                    values[out_idx] = (acc / divisor as f64) as f32;
                }
            }
        }
    }
    values
}

/// [`crate::kernels_pooling::ADAPTIVE_AVG_POOL2D_F32`] の逐語モデル
/// （`start = floor(o*in/out)`・`end = ceil((o+1)*in/out)`。設計 doc
/// §4／§7）。
pub(crate) fn adaptive_avg_pool2d_model(
    input: &[f32],
    in_shape: [usize; 4],
    out_shape: [usize; 4],
) -> Vec<f32> {
    let [n, c, h_in, w_in] = in_shape;
    let [_, _, h_out, w_out] = out_shape;
    let mut values = vec![0.0f32; n * c * h_out * w_out];

    for nn in 0..n {
        for cc in 0..c {
            for oh in 0..h_out {
                let h_start = (oh * h_in) / h_out;
                let h_end = ((oh + 1) * h_in).div_ceil(h_out);
                for ow in 0..w_out {
                    let w_start = (ow * w_in) / w_out;
                    let w_end = ((ow + 1) * w_in).div_ceil(w_out);

                    let mut acc = 0.0f64;
                    for h in h_start..h_end {
                        for w in w_start..w_end {
                            let v = input[((nn * c + cc) * h_in + h) * w_in + w];
                            acc += f64::from(v);
                        }
                    }
                    let divisor = ((h_end - h_start) * (w_end - w_start)) as f64;
                    let out_idx = ((nn * c + cc) * h_out + oh) * w_out + ow;
                    values[out_idx] = (acc / divisor) as f32;
                }
            }
        }
    }
    values
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_pool2d_model_first_match_tie_rule() {
        // 2x2 の全要素同値 -> タイは窓先頭 (0,0) が勝つ。
        let input = vec![5.0f32, 5.0, 5.0, 5.0];
        let (values, indices) = max_pool2d_model(
            &input,
            [1, 1, 2, 2],
            [1, 1, 1, 1],
            [2, 2],
            [2, 2],
            [0, 0],
            [1, 1],
        );
        assert_eq!(values, vec![5.0]);
        assert_eq!(indices, vec![0]);
    }

    #[test]
    fn max_pool2d_model_nan_propagation_first_index() {
        let input = vec![1.0f32, f32::NAN, 3.0, f32::NAN];
        let (values, indices) = max_pool2d_model(
            &input,
            [1, 1, 2, 2],
            [1, 1, 1, 1],
            [2, 2],
            [2, 2],
            [0, 0],
            [1, 1],
        );
        assert!(values[0].is_nan());
        // row-major (0,0)->(0,1)->(1,0)->(1,1) 走査で最初の NaN は index 1。
        assert_eq!(indices, vec![1]);
    }

    #[test]
    fn avg_pool2d_model_count_include_pad_true_uses_kernel_area_divisor() {
        // in=[1,1,1,1] 値=4.0、kernel=2, padding=1 -> 窓は1要素のみ有効
        // だが divisor は kh*kw=4。
        let input = vec![4.0f32];
        let values = avg_pool2d_model(
            &input,
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [2, 2],
            [1, 1],
            [1, 1],
            true,
        );
        assert_eq!(values, vec![1.0]);
    }

    #[test]
    fn avg_pool2d_model_count_include_pad_false_uses_valid_count_divisor() {
        let input = vec![4.0f32];
        let values = avg_pool2d_model(
            &input,
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [2, 2],
            [1, 1],
            [1, 1],
            false,
        );
        assert_eq!(values, vec![4.0]);
    }

    #[test]
    fn adaptive_avg_pool2d_model_global_average() {
        let input = vec![1.0f32, 2.0, 3.0, 4.0];
        let values = adaptive_avg_pool2d_model(&input, [1, 1, 2, 2], [1, 1, 1, 1]);
        assert_eq!(values, vec![2.5]);
    }

    #[test]
    fn adaptive_avg_pool2d_model_upsampling_window_covers_input() {
        // in=2x2 -> out=4x4 は各出力が入力 1 要素と一致する（拡大）。
        let input = vec![1.0f32, 2.0, 3.0, 4.0];
        let values = adaptive_avg_pool2d_model(&input, [1, 1, 2, 2], [1, 1, 4, 4]);
        assert_eq!(values.len(), 16);
        // (0,0) の窓は h_start=floor(0*2/4)=0,h_end=ceil(1*2/4)=1,
        // w も同様 -> {input[0]} のみ。
        assert_eq!(values[0], 1.0);
    }
}
