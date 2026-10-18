//! `kernels_interpolate.rs` の 2 カーネルを CPU 上でスレッドごとに逐語再現するテスト専用モデル
//! （イシュー #3439・R8）。
//!
//! 役割: AMD 実機がない環境で、座標展開・nearest の整数写像と空間軸クランプ・bilinear の座標／重み／
//! 明示 FMA の演算順序・端数ブロックのガード・書き込み回数（各出力がちょうど 1 回）を検証する。
//! bilinear の積和は `f32::mul_add`（`__builtin_fmaf` と同じ単一丸め）で再現し、
//! `CpuBackendOps::interpolate` と bit 比較する。HIPRTC のコンパイル結果そのものの検証ではない
//! （実機確認は #3159）。

use crate::interpolate::bilinear_scale;
use crate::launch::for_each_thread_gid;
use crate::shape_util::{row_major_strides_i32, shape_to_i32};

pub(crate) fn nearest(
    input: &[f32],
    in_shape: &[usize],
    out_shape: &[usize],
    spatial_start: usize,
) -> Vec<f32> {
    let numel: usize = out_shape.iter().product();
    let out_shape_i = shape_to_i32("m", out_shape).unwrap();
    let in_shape_i = shape_to_i32("m", in_shape).unwrap();
    let in_strides = row_major_strides_i32("m", in_shape).unwrap();
    let rank = in_shape.len() as i64;
    let mut out = vec![f32::NAN; numel];
    let mut writes = vec![0u8; numel];
    for_each_thread_gid(numel, |idx| {
        if idx >= numel as i64 {
            return;
        }
        let mut rem = idx;
        let mut in_flat: i64 = 0;
        for a in (0..rank).rev() {
            let axis_size = i64::from(out_shape_i[a as usize]);
            let c = rem % axis_size;
            rem /= axis_size;
            let src_c = if a as usize >= spatial_start {
                let in_size = i64::from(in_shape_i[a as usize]);
                let s = (c * in_size) / i64::from(out_shape_i[a as usize]);
                s.min(in_size - 1)
            } else {
                c
            };
            in_flat += src_c * i64::from(in_strides[a as usize]);
        }
        out[idx as usize] = input[in_flat as usize];
        writes[idx as usize] += 1;
    });
    assert!(writes.iter().all(|&w| w == 1));
    out
}

pub(crate) fn bilinear(
    input: &[f32],
    in_shape: &[usize],
    out_shape: &[usize],
    align_corners: bool,
) -> Vec<f32> {
    let numel: usize = out_shape.iter().product();
    let rank = in_shape.len();
    let out_shape_i = shape_to_i32("m", out_shape).unwrap();
    let in_shape_i = shape_to_i32("m", in_shape).unwrap();
    let in_strides = row_major_strides_i32("m", in_shape).unwrap();
    let scale_h = bilinear_scale(in_shape[rank - 2], out_shape[rank - 2], align_corners);
    let scale_w = bilinear_scale(in_shape[rank - 1], out_shape[rank - 1], align_corners);
    let mut out = vec![f32::NAN; numel];
    let mut writes = vec![0u8; numel];
    for_each_thread_gid(numel, |idx| {
        if idx >= numel as i64 {
            return;
        }
        let mut rem = idx;
        let mut base_flat: i64 = 0;
        let (mut cy, mut cx) = (0i64, 0i64);
        for a in (0..rank).rev() {
            let axis_size = i64::from(out_shape_i[a]);
            let c = rem % axis_size;
            rem /= axis_size;
            if a == rank - 1 {
                cx = c;
            } else if a == rank - 2 {
                cy = c;
            } else {
                base_flat += c * i64::from(in_strides[a]);
            }
        }
        let in_h = i64::from(in_shape_i[rank - 2]);
        let in_w = i64::from(in_shape_i[rank - 1]);
        let stride_h = i64::from(in_strides[rank - 2]);
        let stride_w = i64::from(in_strides[rank - 1]);

        let (srcy, srcx) = if align_corners {
            (cy as f32 * scale_h, cx as f32 * scale_w)
        } else {
            let mut sy = (cy as f32 + 0.5).mul_add(scale_h, -0.5);
            if sy < 0.0 {
                sy = 0.0;
            }
            let mut sx = (cx as f32 + 0.5).mul_add(scale_w, -0.5);
            if sx < 0.0 {
                sx = 0.0;
            }
            (sy, sx)
        };
        let axis = |src: f32, size: i64| {
            let i0 = (src.floor() as i64).min(size - 1);
            let i1 = (i0 + 1).min(size - 1);
            let l1 = src - i0 as f32;
            (i0, i1, l1, 1.0f32 - l1)
        };
        let (i0y, i1y, l1y, l0y) = axis(srcy, in_h);
        let (i0x, i1x, l1x, l0x) = axis(srcx, in_w);
        let at = |y: i64, x: i64| input[(base_flat + y * stride_h + x * stride_w) as usize];
        let (v00, v01, v10, v11) = (at(i0y, i0x), at(i0y, i1x), at(i1y, i0x), at(i1y, i1x));
        let row0 = l1x.mul_add(v01, l0x * v00);
        let row1 = l1x.mul_add(v11, l0x * v10);
        out[idx as usize] = l1y.mul_add(row1, l0y * row0);
        writes[idx as usize] += 1;
    });
    assert!(writes.iter().all(|&w| w == 1));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use fandhe_ai_backend_cpu::CpuBackendOps;
    use fandhe_ai_tensor_core::{BackendOps, InterpolateMode, Tensor};

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    fn data(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 6.0 - 3.0
            })
            .collect()
    }

    fn cpu(input: &[f32], in_shape: &[usize], size: &[usize], mode: InterpolateMode) -> Vec<f32> {
        let ops = CpuBackendOps::new();
        let t = Tensor::new(input.to_vec(), in_shape).unwrap();
        let r = BackendOps::interpolate(&ops, &t, size, mode).unwrap();
        r.contiguous().as_slice().unwrap().to_vec()
    }

    fn out_shape(in_shape: &[usize], size: &[usize]) -> Vec<usize> {
        let keep = in_shape.len() - size.len();
        let mut o = in_shape[..keep].to_vec();
        o.extend_from_slice(size);
        o
    }

    #[test]
    fn nearest_matches_cpu_bit_exact() {
        // (入力 shape, 空間 size)。拡大・縮小・非整数比・1 軸・3 軸・256 を跨ぐ出力を含む。
        let cases: [(&[usize], &[usize]); 7] = [
            (&[1, 1, 4, 4], &[8, 8]),
            (&[2, 3, 5, 7], &[3, 4]),
            (&[2, 3, 5, 7], &[11, 13]),
            (&[2, 6], &[10]),
            (&[1, 2, 3, 3, 3], &[5, 4, 2]),
            (&[1, 2, 17, 19], &[40, 33]),
            (&[3, 1], &[1]),
        ];
        for (n, &(in_shape, size)) in cases.iter().enumerate() {
            let input = data(in_shape.iter().product(), 3 + n as u64);
            let o = out_shape(in_shape, size);
            let spatial_start = in_shape.len() - size.len();
            let model = nearest(&input, in_shape, &o, spatial_start);
            let reference = cpu(&input, in_shape, size, InterpolateMode::Nearest);
            assert_eq!(bits(&model), bits(&reference), "{in_shape:?} -> {size:?}");
        }
    }

    #[test]
    fn bilinear_matches_cpu_bit_exact() {
        let cases: [(&[usize], &[usize]); 7] = [
            (&[1, 1, 4, 4], &[8, 8]),
            (&[2, 3, 5, 7], &[3, 4]),
            (&[2, 3, 5, 7], &[11, 13]),
            (&[4, 4], &[1, 1]),
            (&[1, 5], &[1, 9]),
            (&[2, 1, 17, 19], &[40, 33]),
            (&[1, 1, 3, 3], &[3, 3]),
        ];
        for (n, &(in_shape, size)) in cases.iter().enumerate() {
            let input = data(in_shape.iter().product(), 13 + n as u64);
            let o = out_shape(in_shape, size);
            for align_corners in [false, true] {
                let model = bilinear(&input, in_shape, &o, align_corners);
                let reference = cpu(
                    &input,
                    in_shape,
                    size,
                    InterpolateMode::Bilinear { align_corners },
                );
                assert_eq!(
                    bits(&model),
                    bits(&reference),
                    "{in_shape:?} -> {size:?} align_corners={align_corners}"
                );
            }
        }
    }

    #[test]
    fn bilinear_align_corners_maps_corners_exactly() {
        let input = [1.0f32, 2.0, 3.0, 4.0];
        let out = bilinear(&input, &[2, 2], &[3, 3], true);
        assert_eq!(out[0], 1.0);
        assert_eq!(out[2], 2.0);
        assert_eq!(out[6], 3.0);
        assert_eq!(out[8], 4.0);
    }
}
