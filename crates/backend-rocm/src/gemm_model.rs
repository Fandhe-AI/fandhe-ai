//! `kernels_gemm.rs` のカーネルの添字計算を CPU 上で逐語に再現するテスト専用モデル。
//!
//! 役割: AMD 実機がない環境で、タイル分担・転置ロード・境界ガード・書き込み範囲・
//! 積和順序（単一アキュムレータ・k 昇順・FMA）の正しさを検証する。実機での実行結果の確認は
//! #3159（本モデルは HIPRTC のコンパイル結果そのものの検証ではない）。

use crate::kernels_gemm::GemmF32Tile;

/// カーネルを (block, thread) ごとに逐語再現する。範囲外アクセスは panic で検出し、
/// 各出力要素がちょうど 1 回書かれることも検査する。
pub(crate) fn simulate(
    t: &GemmF32Tile,
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    let (bm, bn, bk) = (t.bm() as usize, t.bn() as usize, t.bk() as usize);
    let (tm, tn) = (t.tm() as usize, t.tn() as usize);
    let (tx_n, ty_n) = (t.threads_x() as usize, t.threads_y() as usize);
    let threads = t.threads() as usize;
    let pad = t.pad() as usize;
    let mut c = vec![f32::NAN; m * n];
    let mut writes = vec![0u8; m * n];
    if m == 0 || n == 0 {
        return c;
    }
    let grid_x = n.div_ceil(bn);
    let grid_y = m.div_ceil(bm);
    let num_tiles = if k > 0 { (k - 1) / bk + 1 } else { 0 };
    for by in 0..grid_y {
        for bx in 0..grid_x {
            let block_row = by * bm;
            let block_col = bx * bn;
            let mut acc = vec![vec![vec![0.0f32; tn]; tm]; threads];
            let mut as_tile = vec![vec![0.0f32; bm + pad]; bk];
            let mut bs_tile = vec![vec![0.0f32; bn + pad]; bk];
            for tile in 0..num_tiles {
                let k_base = tile * bk;
                // 共有メモリロード（全スレッド分。__syncthreads 前に完了する）。
                for tid in 0..threads {
                    for i in 0..(bm * bk) / threads {
                        let idx = tid + i * threads;
                        let (mm, kk) = (idx % bm, idx / bm);
                        let (row, col) = (block_row + mm, k_base + kk);
                        as_tile[kk][mm] = if row < m && col < k {
                            a[row * k + col]
                        } else {
                            0.0
                        };
                    }
                    for i in 0..(bn * bk) / threads {
                        let idx = tid + i * threads;
                        let (nn, kk) = (idx % bn, idx / bn);
                        let (b_row, b_col) = (k_base + kk, block_col + nn);
                        bs_tile[kk][nn] = if b_row < k && b_col < n {
                            b[b_row * n + b_col]
                        } else {
                            0.0
                        };
                    }
                }
                for ty in 0..ty_n {
                    for tx in 0..tx_n {
                        let tid = ty * tx_n + tx;
                        for (as_row, bs_row) in as_tile.iter().zip(&bs_tile) {
                            for (i, acc_row) in acc[tid].iter_mut().enumerate() {
                                let af = as_row[ty * tm + i];
                                for (j, v) in acc_row.iter_mut().enumerate() {
                                    *v = af.mul_add(bs_row[tx * tn + j], *v);
                                }
                            }
                        }
                    }
                }
            }
            for ty in 0..ty_n {
                for tx in 0..tx_n {
                    let tid = ty * tx_n + tx;
                    for (i, acc_row) in acc[tid].iter().enumerate() {
                        let row = block_row + ty * tm + i;
                        if row < m {
                            for (j, &v) in acc_row.iter().enumerate() {
                                let col = block_col + tx * tn + j;
                                if col < n {
                                    c[row * n + col] = v;
                                    writes[row * n + col] += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(
        writes.iter().all(|&w| w == 1),
        "各出力はちょうど 1 回書かれる"
    );
    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use fandhe_ai_backend_cpu::{assert_parity, gemm_naive};

    /// 決定的な疑似乱数（LCG）。
    fn data(len: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..len)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn matches_cpu_reference() {
        let shapes = [
            (1, 1, 1),
            (1, 1, 0),
            (64, 64, 16),
            (63, 65, 17),
            (130, 70, 33),
            (5, 129, 200),
            (128, 128, 64),
        ];
        let t = GemmF32Tile::DEFAULT;
        for (idx, &(m, n, k)) in shapes.iter().enumerate() {
            let a = data(m * k, 1 + idx as u64);
            let b = data(k * n, 100 + idx as u64);
            let mut expected = vec![0.0f32; m * n];
            gemm_naive(&a, &b, &mut expected, m, n, k).unwrap();
            let got = simulate(&t, &a, &b, m, n, k);
            assert_parity(&format!("model {m}x{n}x{k}"), &got, &expected);
            // 理論上は bit 一致（同じ k 昇順の FMA。末尾のゼロ埋めは acc を変えない）。
            assert!(
                got.iter()
                    .zip(&expected)
                    .all(|(x, y)| x.to_bits() == y.to_bits()),
                "bit mismatch {m}x{n}x{k}"
            );
        }
    }

    #[test]
    fn other_valid_tile_also_matches() {
        let t = GemmF32Tile::new((32, 128, 16), (2, 4), 1, (32, 16)).unwrap();
        let (m, n, k) = (37, 131, 19);
        let a = data(m * k, 7);
        let b = data(k * n, 8);
        let mut expected = vec![0.0f32; m * n];
        gemm_naive(&a, &b, &mut expected, m, n, k).unwrap();
        assert_parity("alt tile", &simulate(&t, &a, &b, m, n, k), &expected);
    }
}
