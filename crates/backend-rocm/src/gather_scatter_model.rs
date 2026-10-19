//! `kernels_gather_scatter.rs` の 4 カーネルを CPU 上でスレッドごとに逐語再現するテスト専用
//! モデル（イシュー #3439・R8）。
//!
//! 役割: AMD 実機がない環境で、座標展開・dim 軸の置換・非 dim 軸の範囲判定・index 走査順・
//! `f64` アキュムレータ・端数ブロックのガード・書き込み回数（各出力がちょうど 1 回）を検証する。
//! `CpuBackendOps` の出力と bit 比較する。HIPRTC のコンパイル結果そのものの検証ではない
//! （実機確認は #3159）。

use crate::launch::for_each_thread_gid;
use crate::shape_util::{row_major_strides_i32, shape_to_i32};

fn finish(out: Vec<f32>, writes: Vec<u8>) -> Vec<f32> {
    assert!(
        writes.iter().all(|&w| w == 1),
        "各出力はちょうど 1 回書かれる"
    );
    out
}

pub(crate) fn gather(
    input: &[f32],
    in_shape: &[usize],
    index: &[i32],
    out_shape: &[usize],
    dim: usize,
) -> Vec<f32> {
    let numel: usize = out_shape.iter().product();
    let out_shape_i = shape_to_i32("m", out_shape).unwrap();
    let in_strides = row_major_strides_i32("m", in_shape).unwrap();
    let in_dim_size = in_shape[dim] as i64;
    let rank = in_shape.len() as i64;
    let mut out = vec![f32::NAN; numel];
    let mut writes = vec![0u8; numel];
    for_each_thread_gid(numel, |idx| {
        if idx >= numel as i64 {
            return;
        }
        let index_val = index[idx as usize];
        if index_val >= 0 && i64::from(index_val) < in_dim_size {
            let mut rem = idx;
            let mut in_flat: i64 = 0;
            for a in (0..rank).rev() {
                let axis_size = i64::from(out_shape_i[a as usize]);
                let c = rem % axis_size;
                rem /= axis_size;
                let coord = if a as usize == dim {
                    i64::from(index_val)
                } else {
                    c
                };
                in_flat += coord * i64::from(in_strides[a as usize]);
            }
            out[idx as usize] = input[in_flat as usize];
        } else {
            out[idx as usize] = 0.0;
        }
        writes[idx as usize] += 1;
    });
    finish(out, writes)
}

pub(crate) fn scatter(
    input: &[f32],
    in_shape: &[usize],
    index: &[i32],
    index_shape: &[usize],
    src: &[f32],
    dim: usize,
    add: bool,
) -> Vec<f32> {
    let numel: usize = in_shape.iter().product();
    let out_shape_i = shape_to_i32("m", in_shape).unwrap();
    let index_shape_i = shape_to_i32("m", index_shape).unwrap();
    let index_strides = row_major_strides_i32("m", index_shape).unwrap();
    let rank = in_shape.len() as i64;
    let mut out = vec![f32::NAN; numel];
    let mut writes = vec![0u8; numel];
    for_each_thread_gid(numel, |idx| {
        if idx >= numel as i64 {
            return;
        }
        let mut rem = idx;
        let mut index_base: i64 = 0;
        let mut dim_coord: i64 = 0;
        let mut in_range = true;
        for a in (0..rank).rev() {
            let axis_size = i64::from(out_shape_i[a as usize]);
            let c = rem % axis_size;
            rem /= axis_size;
            if a as usize == dim {
                dim_coord = c;
            } else if c >= i64::from(index_shape_i[a as usize]) {
                in_range = false;
                break;
            } else {
                index_base += c * i64::from(index_strides[a as usize]);
            }
        }
        let value = if add {
            let mut acc = f64::from(input[idx as usize]);
            if in_range {
                let dim_len = i64::from(index_shape_i[dim]);
                let dim_stride = i64::from(index_strides[dim]);
                for j in 0..dim_len {
                    let p = (index_base + j * dim_stride) as usize;
                    if i64::from(index[p]) == dim_coord {
                        acc += f64::from(src[p]);
                    }
                }
            }
            acc as f32
        } else {
            let mut result = input[idx as usize];
            if in_range {
                let dim_len = i64::from(index_shape_i[dim]);
                let dim_stride = i64::from(index_strides[dim]);
                for j in 0..dim_len {
                    let p = (index_base + j * dim_stride) as usize;
                    if i64::from(index[p]) == dim_coord {
                        result = src[p];
                    }
                }
            }
            result
        };
        out[idx as usize] = value;
        writes[idx as usize] += 1;
    });
    finish(out, writes)
}

pub(crate) fn one_hot(index: &[i32], num_classes: usize) -> Vec<f32> {
    let numel = index.len() * num_classes;
    let mut out = vec![f32::NAN; numel];
    let mut writes = vec![0u8; numel];
    for_each_thread_gid(numel, |idx| {
        if idx >= numel as i64 {
            return;
        }
        let row = idx / num_classes as i64;
        let c = idx % num_classes as i64;
        let index_val = index[row as usize];
        out[idx as usize] = if index_val >= 0
            && i64::from(index_val) < num_classes as i64
            && i64::from(index_val) == c
        {
            1.0
        } else {
            0.0
        };
        writes[idx as usize] += 1;
    });
    finish(out, writes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fandhe_ai_backend_cpu::CpuBackendOps;
    use fandhe_ai_tensor_core::{BackendOps, ScatterReduce, Tensor};

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
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 8.0 - 4.0
            })
            .collect()
    }

    fn idx_data(n: usize, limit: usize, seed: u64) -> Vec<i32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as usize % limit) as i32
            })
            .collect()
    }

    fn flat(t: Tensor<f32>) -> Vec<f32> {
        t.contiguous().as_slice().unwrap().to_vec()
    }

    #[test]
    fn gather_matches_cpu_bit_exact() {
        let cases: [(&[usize], &[usize], usize); 6] = [
            (&[5], &[3], 0),
            (&[3, 4], &[3, 2], 1),
            (&[3, 4], &[2, 4], 0),
            (&[2, 3, 4], &[2, 3, 5], 2),
            // CPU の `gather_out_shape` は非 dim 軸の一致を要求する（本層の検証は `<=` で上位互換）。
            (&[2, 3, 4], &[2, 2, 4], 1),
            // 出力要素数が 256 を跨ぐ（複数ブロック・端数ブロック）。
            (&[4, 300], &[4, 321], 1),
        ];
        for (n, &(in_shape, out_shape, dim)) in cases.iter().enumerate() {
            let input = data(in_shape.iter().product(), 7 + n as u64);
            let index = idx_data(out_shape.iter().product(), in_shape[dim], 70 + n as u64);
            let model = gather(&input, in_shape, &index, out_shape, dim);
            let ops = CpuBackendOps::new();
            let cpu = BackendOps::gather(
                &ops,
                &Tensor::new(input.clone(), in_shape).unwrap(),
                dim,
                &Tensor::new(index.clone(), out_shape).unwrap(),
            )
            .unwrap();
            assert_eq!(bits(&model), bits(&flat(cpu)), "{in_shape:?} dim={dim}");
        }
    }

    #[test]
    fn scatter_matches_cpu_bit_exact_including_duplicates() {
        let cases: [(&[usize], &[usize], usize); 6] = [
            (&[5], &[8], 0),
            (&[3, 4], &[3, 6], 1),
            (&[3, 4], &[2, 4], 0),
            (&[2, 3, 4], &[2, 2, 7], 2),
            (&[2, 3, 4], &[1, 3, 4], 0),
            (&[4, 300], &[3, 400], 1),
        ];
        for (n, &(in_shape, idx_shape, dim)) in cases.iter().enumerate() {
            let input = data(in_shape.iter().product(), 17 + n as u64);
            let src = data(idx_shape.iter().product(), 170 + n as u64);
            let index = idx_data(idx_shape.iter().product(), in_shape[dim], 700 + n as u64);
            for (add, reduce) in [
                (false, ScatterReduce::Overwrite),
                (true, ScatterReduce::Add),
            ] {
                let model = scatter(&input, in_shape, &index, idx_shape, &src, dim, add);
                let ops = CpuBackendOps::new();
                let cpu = BackendOps::scatter(
                    &ops,
                    &Tensor::new(input.clone(), in_shape).unwrap(),
                    dim,
                    &Tensor::new(index.clone(), idx_shape).unwrap(),
                    &Tensor::new(src.clone(), idx_shape).unwrap(),
                    reduce,
                )
                .unwrap();
                assert_eq!(
                    bits(&model),
                    bits(&flat(cpu)),
                    "{in_shape:?} dim={dim} add={add}"
                );
            }
        }
    }

    #[test]
    fn scatter_add_uses_f64_accumulation() {
        // f32 逐次加算なら 1e8 + 1 + (-1e8) は 0 になるが、f64 蓄積なら最終値は 1。
        let input = [0.0f32];
        let src = [1e8f32, 1.0, -1e8];
        let index = [0, 0, 0];
        let out = scatter(&input, &[1], &index, &[3], &src, 0, true);
        assert_eq!(out, vec![1.0]);
        // Overwrite は index 走査順で最後の値が残る。
        let out = scatter(&input, &[1], &index, &[3], &src, 0, false);
        assert_eq!(out, vec![-1e8]);
    }

    #[test]
    fn one_hot_matches_cpu() {
        for (n, classes) in [(1usize, 1usize), (5, 3), (40, 7), (300, 2)] {
            let index = idx_data(n, classes, 5 + n as u64);
            let model = one_hot(&index, classes);
            let ops = CpuBackendOps::new();
            let cpu =
                BackendOps::one_hot(&ops, &Tensor::new(index.clone(), &[n]).unwrap(), classes)
                    .unwrap();
            assert_eq!(bits(&model), bits(&flat(cpu)), "n={n} classes={classes}");
        }
    }
}
