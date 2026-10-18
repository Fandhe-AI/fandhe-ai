//! `kernels_sort.rs` の 3 カーネルを CPU 上でスレッドごとに逐語再現するテスト専用モデル
//! （イシュー #3439・R8）。
//!
//! 役割: AMD 実機がない環境で、キー構築（NaN 最大・±0 同値・降順の反転・添字による安定化・
//! パディング）・ビットニックの比較ステップ（ホスト側 `j`／`k` ループを含む）・仕上げの gather と
//! 各ガードを検証する。`CpuBackendOps::sort`／`topk` と values は bit・index は完全一致で比較する。
//! HIPRTC のコンパイル結果そのものの検証ではない（実機確認は #3159）。

use crate::launch::for_each_thread_gid;
use crate::sort::validate_sort;

fn build_keys(
    input: &[f32],
    lines: i64,
    dim_size: i64,
    inner: i64,
    padded: i64,
    descending: bool,
) -> Vec<u64> {
    let total = lines * padded;
    let mut keys = vec![0u64; total as usize];
    let mut writes = vec![0u8; total as usize];
    for_each_thread_gid(total as usize, |gid| {
        if gid >= total {
            return;
        }
        let line = gid / padded;
        let i = gid % padded;
        let out_pos = (line * padded + i) as usize;
        if i < dim_size {
            let in_pos = (line / inner) * dim_size * inner + (line % inner) + i * inner;
            assert!(
                in_pos >= 0 && (in_pos as usize) < input.len(),
                "範囲外読み出し"
            );
            let v = input[in_pos as usize];
            let vk: u32 = if v.is_nan() {
                0xFFFF_FFFF
            } else {
                let vv = if v == 0.0 { 0.0f32 } else { v };
                let bits = vv.to_bits();
                if bits >> 31 == 1 {
                    !bits
                } else {
                    bits | 0x8000_0000
                }
            };
            let hi = if descending { !vk } else { vk };
            keys[out_pos] = (u64::from(hi) << 32) | (i as u32 as u64);
        } else {
            keys[out_pos] = u64::MAX;
        }
        writes[out_pos] += 1;
    });
    assert!(writes.iter().all(|&w| w == 1));
    keys
}

fn bitonic_step(keys: &mut [u64], j: i64, k: i64, padded: i64, lines: i64) {
    let total = lines * padded;
    for_each_thread_gid(total as usize, |gid| {
        if gid >= total {
            return;
        }
        let line = gid / padded;
        let i = gid % padded;
        let ixj = i ^ j;
        if ixj <= i || ixj >= padded {
            return;
        }
        let base = line * padded;
        let a = keys[(base + i) as usize];
        let b = keys[(base + ixj) as usize];
        let ascending = (i & k) == 0;
        let should_swap = if ascending { a > b } else { a < b };
        if should_swap {
            keys[(base + i) as usize] = b;
            keys[(base + ixj) as usize] = a;
        }
    });
}

/// build → ビットニック（ホストの j/k ループ）→ finalize を通しで実行する。
pub(crate) fn sort_model(
    input: &[f32],
    shape: &[usize],
    dim: usize,
    descending: bool,
    out_len: usize,
) -> (Vec<f32>, Vec<i32>) {
    let run = validate_sort(input.len(), shape, dim, out_len).unwrap();
    let (lines, dim_size, inner, padded) = (
        run.plan.lines as i64,
        run.plan.dim_size as i64,
        run.plan.inner as i64,
        run.plan.padded as i64,
    );
    let total_out = run.total_out;
    if total_out == 0 {
        return (Vec::new(), Vec::new());
    }
    let mut keys = build_keys(input, lines, dim_size, inner, padded, descending);
    if padded > 1 {
        let mut k = 2i64;
        while k <= padded {
            let mut j = k / 2;
            while j >= 1 {
                bitonic_step(&mut keys, j, k, padded, lines);
                j /= 2;
            }
            k *= 2;
        }
    }
    let out_len_i = out_len as i64;
    let mut values = vec![f32::NAN; total_out];
    let mut index = vec![-1i32; total_out];
    let mut writes = vec![0u8; total_out];
    let total = lines * out_len_i;
    for_each_thread_gid(total as usize, |gid| {
        if gid >= total {
            return;
        }
        let line = gid / out_len_i;
        let o = gid % out_len_i;
        if o >= padded {
            return;
        }
        let key = keys[(line * padded + o) as usize];
        let idxu = key & 0xFFFF_FFFF;
        if idxu >= dim_size as u64 {
            return;
        }
        let idx = idxu as i64;
        let in_pos = (line / inner) * dim_size * inner + (line % inner) + idx * inner;
        let out_pos = (line / inner) * out_len_i * inner + (line % inner) + o * inner;
        assert!(in_pos >= 0 && (in_pos as usize) < input.len());
        assert!(out_pos >= 0 && (out_pos as usize) < total_out);
        values[out_pos as usize] = input[in_pos as usize];
        index[out_pos as usize] = idx as i32;
        writes[out_pos as usize] += 1;
    });
    assert!(
        writes.iter().all(|&w| w == 1),
        "各出力はちょうど 1 回書かれる"
    );
    (values, index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fandhe_ai_backend_cpu::CpuBackendOps;
    use fandhe_ai_tensor_core::{BackendOps, Tensor};

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
                // 同値を多く作るため粗く量子化する（安定性の検証）。
                (((s >> 40) % 9) as f32) - 4.0
            })
            .collect()
    }

    fn cpu(
        data: &[f32],
        shape: &[usize],
        dim: usize,
        descending: bool,
        out_len: usize,
    ) -> (Vec<f32>, Vec<i32>) {
        let ops = CpuBackendOps::new();
        let t = Tensor::new(data.to_vec(), shape).unwrap();
        let (v, i) = if out_len == shape[dim] {
            BackendOps::sort(&ops, &t, dim, descending).unwrap()
        } else {
            BackendOps::topk(&ops, &t, dim, out_len, descending).unwrap()
        };
        (
            v.contiguous().as_slice().unwrap().to_vec(),
            i.contiguous().as_slice().unwrap().to_vec(),
        )
    }

    #[test]
    fn matches_cpu_for_sort_and_topk() {
        let shapes: [(&[usize], usize); 13] = [
            (&[4], 0),
            (&[1, 4], 1),
            (&[3, 4], 1),
            (&[4, 3], 0),
            (&[2, 3, 4], 0),
            (&[2, 3, 4], 1),
            (&[2, 3, 4], 2),
            (&[1, 7], 1),
            (&[1, 8], 1),
            (&[1, 255], 1),
            (&[1, 256], 1),
            (&[1, 257], 1),
            (&[3, 1, 5], 1),
        ];
        for (seed, &(shape, dim)) in shapes.iter().enumerate() {
            let numel: usize = shape.iter().product();
            let mut d = data(numel, 900 + seed as u64);
            // NaN（符号・payload 違い）・±0・±inf を注入する。
            let specials = [
                f32::NAN,
                -f32::NAN,
                0.0,
                -0.0,
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::from_bits(f32::NAN.to_bits() | 1),
            ];
            for (n, s) in specials.iter().enumerate() {
                if n * 3 < numel {
                    d[n * 3] = *s;
                }
            }
            let dim_size = shape[dim];
            for descending in [false, true] {
                for out_len in [0usize, 1, dim_size / 2, dim_size] {
                    if out_len > dim_size {
                        continue;
                    }
                    let (mv, mi) = sort_model(&d, shape, dim, descending, out_len);
                    let (cv, ci) = cpu(&d, shape, dim, descending, out_len);
                    assert_eq!(
                        bits(&mv),
                        bits(&cv),
                        "values: {shape:?} dim={dim} desc={descending} out_len={out_len}"
                    );
                    assert_eq!(
                        mi, ci,
                        "index: {shape:?} dim={dim} desc={descending} out_len={out_len}"
                    );
                }
            }
        }
    }

    #[test]
    fn explicit_descending_nan_and_zero_case() {
        // 降順では NaN が添字順で先頭、次に +inf、最後に ±0 が添字順（NaN は最大・±0 は同値）。
        let d = [-0.0f32, 0.0, f32::NAN, -f32::NAN, f32::INFINITY];
        let (_, index) = sort_model(&d, &[5], 0, true, 5);
        assert_eq!(index, vec![2, 3, 4, 0, 1]);
        let (_, index) = sort_model(&d, &[5], 0, false, 5);
        assert_eq!(index, vec![0, 1, 4, 2, 3]);
    }

    #[test]
    fn empty_outputs_return_empty() {
        let (v, i) = sort_model(&[1.0, 2.0, 3.0], &[3], 0, false, 0);
        assert!(v.is_empty() && i.is_empty());
    }
}
