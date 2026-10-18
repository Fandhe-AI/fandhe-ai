//! `kernels_im2col.rs`・`kernels_constant_pad.rs`・`kernels_pooling.rs` の 6 カーネルを CPU 上で
//! スレッドごとに逐語再現するテスト専用モデル（イシュー #3438・R7）。
//!
//! 役割: AMD 実機がない環境で、添字展開・剰余前の符号判定・窓のタップ走査順・`double` アキュムレータ
//! の固定順加算と 1 回 downcast・max の更新規則（NaN 伝播・先勝ち）・adaptive の窓端クランプ・端数
//! ブロックのガード・書き込み回数（各出力がちょうど 1 回）を検証する。`RocmBackendOps` の
//! `Engine::HostModel` から呼ばれ、`CpuBackendOps` と bit 比較される。HIPRTC のコンパイル結果そのもの
//! の検証ではない（実機確認は #3159）。float の積和は 6 カーネルのどこにもなく、FMA 契約の対象外。

use crate::constant_pad::validate_pad;
use crate::im2col::Im2colPlan;
use crate::launch::for_each_thread_gid;
use crate::pooling::PoolPlan;
use crate::shape_util::{row_major_strides_i32, shape_to_i32};

/// 出力 `numel` 個のスレッド（端数ブロック込み）を回し、各出力がちょうど 1 回書かれることを確かめる。
fn run_threads<T: Copy>(numel: usize, init: T, mut f: impl FnMut(i64) -> T) -> Vec<T> {
    let mut out = vec![init; numel];
    let mut writes = vec![0u8; numel];
    for_each_thread_gid(numel, |idx| {
        if idx >= numel as i64 {
            return;
        }
        out[idx as usize] = f(idx);
        writes[idx as usize] += 1;
    });
    assert!(writes.iter().all(|&w| w == 1));
    out
}

pub(crate) fn im2col(input: &[f32], plan: &Im2colPlan) -> Vec<f32> {
    let d = plan.dims.map_i64();
    run_threads(plan.numel_col, f32::NAN, |idx| {
        let mut rem = idx;
        let p_idx = rem % d.p;
        rem /= d.p;
        let k_idx = rem % d.k_g;
        rem /= d.k_g;
        let g = rem % d.groups;
        rem /= d.groups;
        let n = rem;

        let kw_ = k_idx % d.kw;
        let rest = k_idx / d.kw;
        let kh_ = rest % d.kh;
        let c_g = rest / d.kh;
        let c = g * d.cin_g + c_g;

        let ow = p_idx % d.w_out;
        let oh = p_idx / d.w_out;
        let h = oh * d.sh + kh_ * d.dh - d.ph;
        let w = ow * d.sw + kw_ * d.dw - d.pw;

        if h >= 0 && h < d.h_in && w >= 0 && w < d.w_in {
            input[(((n * d.cin + c) * d.h_in + h) * d.w_in + w) as usize]
        } else {
            0.0
        }
    })
}

pub(crate) fn col2im(d_col: &[f32], plan: &Im2colPlan) -> Vec<f32> {
    let d = plan.dims.map_i64();
    run_threads(plan.numel_in, f32::NAN, |idx| {
        let mut rem = idx;
        let w = rem % d.w_in;
        rem /= d.w_in;
        let h = rem % d.h_in;
        rem /= d.h_in;
        let c = rem % d.cin;
        rem /= d.cin;
        let n = rem;

        let g = c / d.cin_g;
        let c_g = c % d.cin_g;

        let mut acc = 0.0f64;
        for kh_ in 0..d.kh {
            let num_h = h + d.ph - kh_ * d.dh;
            if num_h < 0 {
                continue;
            }
            if num_h % d.sh != 0 {
                continue;
            }
            let oh = num_h / d.sh;
            if oh >= d.h_out {
                continue;
            }
            for kw_ in 0..d.kw {
                let num_w = w + d.pw - kw_ * d.dw;
                if num_w < 0 {
                    continue;
                }
                if num_w % d.sw != 0 {
                    continue;
                }
                let ow = num_w / d.sw;
                if ow >= d.w_out {
                    continue;
                }
                let k_idx = (c_g * d.kh + kh_) * d.kw + kw_;
                let p_idx = oh * d.w_out + ow;
                let col_idx = ((n * d.groups + g) * d.k_g + k_idx) * d.p + p_idx;
                acc += f64::from(d_col[col_idx as usize]);
            }
        }
        acc as f32
    })
}

pub(crate) fn pad(
    input: &[f32],
    in_shape: &[usize],
    pads: &[(usize, usize)],
    value: f32,
) -> Vec<f32> {
    let plan = validate_pad(input.len(), in_shape, pads).expect("ops 層が検証済み");
    let rank = in_shape.len() as i64;
    let out_shape = shape_to_i32("m", &plan.out_shape).unwrap();
    let in_shape_i = shape_to_i32("m", in_shape).unwrap();
    let in_strides = row_major_strides_i32("m", in_shape).unwrap();
    let before: Vec<i32> = pads.iter().map(|&(b, _)| b as i32).collect();
    run_threads(plan.numel_out, f32::NAN, |idx| {
        let mut rem = idx;
        let mut in_flat: i64 = 0;
        let mut inside = true;
        for a in (0..rank).rev() {
            let a = a as usize;
            let axis_size = i64::from(out_shape[a]);
            let c = rem % axis_size;
            rem /= axis_size;
            let src_c = c - i64::from(before[a]);
            if src_c < 0 || src_c >= i64::from(in_shape_i[a]) {
                inside = false;
                break;
            }
            in_flat += src_c * i64::from(in_strides[a]);
        }
        if inside {
            input[in_flat as usize]
        } else {
            value
        }
    })
}

/// 出力 idx を (n, cc, oh, ow) へ分解する（pooling 3 カーネル共通の前半）。
fn split_nchw(idx: i64, plan: &PoolPlan) -> (i64, i64, i64, i64) {
    let (c, h_out, w_out) = (
        i64::from(plan.c),
        i64::from(plan.h_out),
        i64::from(plan.w_out),
    );
    let mut rem = idx;
    let ow = rem % w_out;
    rem /= w_out;
    let oh = rem % h_out;
    rem /= h_out;
    let cc = rem % c;
    (rem / c, cc, oh, ow)
}

pub(crate) fn max_pool2d(
    input: &[f32],
    plan: &PoolPlan,
    kernel: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
) -> (Vec<f32>, Vec<i32>) {
    let (c, h_in, w_in) = (
        i64::from(plan.c),
        i64::from(plan.h_in),
        i64::from(plan.w_in),
    );
    let [kh, kw] = kernel.map(|v| v as i64);
    let [sh, sw] = stride.map(|v| v as i64);
    let [ph, pw] = padding.map(|v| v as i64);
    let [dh, dw] = dilation.map(|v| v as i64);
    let pairs = run_threads(plan.numel_out, (f32::NAN, -1i32), |idx| {
        let (n, cc, oh, ow) = split_nchw(idx, plan);
        let mut best = 0.0f32;
        let mut best_idx = 0i32;
        let mut found = false;
        for kh_ in 0..kh {
            let h = oh * sh + kh_ * dh - ph;
            if h < 0 || h >= h_in {
                continue;
            }
            for kw_ in 0..kw {
                let w = ow * sw + kw_ * dw - pw;
                if w < 0 || w >= w_in {
                    continue;
                }
                let v = input[(((n * c + cc) * h_in + h) * w_in + w) as usize];
                let cur_idx = (h * w_in + w) as i32;
                if !found {
                    best = v;
                    best_idx = cur_idx;
                    found = true;
                } else if v > best || (v.is_nan() && !best.is_nan()) {
                    best = v;
                    best_idx = cur_idx;
                }
            }
        }
        (best, best_idx)
    });
    pairs.into_iter().unzip()
}

pub(crate) fn avg_pool2d(
    input: &[f32],
    plan: &PoolPlan,
    kernel: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    count_include_pad: bool,
) -> Vec<f32> {
    let (c, h_in, w_in) = (
        i64::from(plan.c),
        i64::from(plan.h_in),
        i64::from(plan.w_in),
    );
    let [kh, kw] = kernel.map(|v| v as i64);
    let [sh, sw] = stride.map(|v| v as i64);
    let [ph, pw] = padding.map(|v| v as i64);
    run_threads(plan.numel_out, f32::NAN, |idx| {
        let (n, cc, oh, ow) = split_nchw(idx, plan);
        let mut acc = 0.0f64;
        let mut valid_count: i64 = 0;
        for kh_ in 0..kh {
            let h = oh * sh + kh_ - ph;
            if h < 0 || h >= h_in {
                continue;
            }
            for kw_ in 0..kw {
                let w = ow * sw + kw_ - pw;
                if w < 0 || w >= w_in {
                    continue;
                }
                acc += f64::from(input[(((n * c + cc) * h_in + h) * w_in + w) as usize]);
                valid_count += 1;
            }
        }
        let divisor = if count_include_pad {
            kh * kw
        } else {
            valid_count
        };
        (acc / divisor as f64) as f32
    })
}

pub(crate) fn adaptive_avg_pool2d(input: &[f32], plan: &PoolPlan) -> Vec<f32> {
    let (c, h_in, w_in) = (
        i64::from(plan.c),
        i64::from(plan.h_in),
        i64::from(plan.w_in),
    );
    let (h_out, w_out) = (i64::from(plan.h_out), i64::from(plan.w_out));
    run_threads(plan.numel_out, f32::NAN, |idx| {
        let (n, cc, oh, ow) = split_nchw(idx, plan);
        let h_start = (oh * h_in) / h_out;
        let mut h_end = ((oh + 1) * h_in + h_out - 1) / h_out;
        let w_start = (ow * w_in) / w_out;
        let mut w_end = ((ow + 1) * w_in + w_out - 1) / w_out;
        if h_end > h_in {
            h_end = h_in;
        }
        if w_end > w_in {
            w_end = w_in;
        }
        let mut acc = 0.0f64;
        for h in h_start..h_end {
            for w in w_start..w_end {
                acc += f64::from(input[(((n * c + cc) * h_in + h) * w_in + w) as usize]);
            }
        }
        let divisor = (h_end - h_start) * (w_end - w_start);
        (acc / divisor as f64) as f32
    })
}
