//! `kernels_unique.rs` のビットニックカーネルを CPU 上でスレッドごとに逐語再現するテスト専用
//! モデル（イシュー #3439・R8）。
//!
//! 役割: AMD 実機がない環境で、キー変換・パディング・ビットニック整列・隣接重複除去の全工程が
//! `CpuBackendOps::unique` と bit 一致する（NaN の payload・`-0.0` の代表選択を含む）ことを検証する。
//! HIPRTC のコンパイル結果そのものの検証ではない（実機確認は #3159）。

use crate::launch::for_each_thread_gid;
use crate::unique::{finish_unique, padded_len, total_order_key};

fn bitonic_step(keys: &mut [u32], j: i64, k: i64, n: i64) {
    for_each_thread_gid(n as usize, |i| {
        if i >= n {
            return;
        }
        let ixj = i ^ j;
        if ixj <= i || ixj >= n {
            return;
        }
        let a = keys[i as usize];
        let b = keys[ixj as usize];
        let ascending = (i & k) == 0;
        let should_swap = if ascending { a > b } else { a < b };
        if should_swap {
            keys[i as usize] = b;
            keys[ixj as usize] = a;
        }
    });
}

pub(crate) fn unique_model(x: &[f32]) -> Vec<f32> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![x[0]];
    }
    let padded = padded_len(n).unwrap();
    let mut keys: Vec<u32> = x.iter().map(|&v| total_order_key(v)).collect();
    keys.resize(padded, u32::MAX);
    let mut k = 2i64;
    while k <= padded as i64 {
        let mut j = k / 2;
        while j >= 1 {
            bitonic_step(&mut keys, j, k, padded as i64);
            j /= 2;
        }
        k *= 2;
    }
    finish_unique(&keys, n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fandhe_ai_backend_cpu::CpuBackendOps;
    use fandhe_ai_tensor_core::{BackendOps, Tensor};

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    fn cpu(x: &[f32]) -> Vec<f32> {
        let ops = CpuBackendOps::new();
        let t = Tensor::new(x.to_vec(), &[x.len()]).unwrap();
        BackendOps::unique(&ops, &t)
            .unwrap()
            .contiguous()
            .as_slice()
            .unwrap()
            .to_vec()
    }

    fn data(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (((s >> 40) % 11) as f32) - 5.0
            })
            .collect()
    }

    #[test]
    fn matches_cpu_across_sizes_with_specials() {
        for (seed, n) in [2usize, 3, 5, 8, 17, 255, 256, 257, 1000]
            .into_iter()
            .enumerate()
        {
            let mut d = data(n, 40 + seed as u64);
            let specials = [
                f32::NAN,
                -f32::NAN,
                f32::from_bits(f32::NAN.to_bits() | 1),
                0.0,
                -0.0,
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::NAN,
            ];
            for (m, s) in specials.iter().enumerate() {
                if m * 2 + 1 < n {
                    d[m * 2 + 1] = *s;
                }
            }
            assert_eq!(bits(&unique_model(&d)), bits(&cpu(&d)), "n={n}");
        }
    }

    #[test]
    fn trivial_inputs() {
        assert!(unique_model(&[]).is_empty());
        assert_eq!(unique_model(&[3.0]), vec![3.0]);
        assert_eq!(bits(&unique_model(&[0.0, -0.0, 0.0])), bits(&[-0.0]));
        // NaN は == が偽のためすべて残る。
        let out = unique_model(&[f32::NAN, 1.0, f32::NAN]);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0], 1.0);
    }
}
