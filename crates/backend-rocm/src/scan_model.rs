//! `kernels_scan.rs` のカーネルを CPU 上で (block, thread) ごとに逐語再現するテスト専用モデル
//! （イシュー #3439・R8）。
//!
//! 役割: AMD 実機がない環境で、lane 分解・境界ガード・`f64` アキュムレータ・各ステップの
//! `f32` スナップショット書き出し・書き込み回数（各出力がちょうど 1 回）を検証する。
//! `CpuBackendOps::cumsum`／`cumprod` と bit 比較する。HIPRTC のコンパイル結果そのものの検証では
//! ない（実機確認は #3159）。

use crate::launch::{INDEX_OPS_BLOCK_DIM, grid_1d};

#[derive(Clone, Copy)]
pub(crate) enum ScanKind {
    Sum,
    Prod,
}

/// 範囲外アクセスは panic、書き込み回数は検査する。
pub(crate) fn simulate(
    kind: ScanKind,
    x: &[f32],
    outer: usize,
    axis_len: usize,
    inner: usize,
) -> Vec<f32> {
    let lanes = outer * inner;
    let numel = lanes * axis_len;
    assert_eq!(x.len(), numel);
    let mut out = vec![f32::NAN; numel];
    let mut writes = vec![0u8; numel];
    if numel == 0 {
        return out;
    }
    let (grid, block) = grid_1d("scan_model", lanes, INDEX_OPS_BLOCK_DIM).unwrap();
    for b in 0..grid.0 as i64 {
        for t in 0..block.0 as i64 {
            let gid = b * block.0 as i64 + t;
            if gid >= lanes as i64 {
                continue;
            }
            let o = gid / inner as i64;
            let i = gid % inner as i64;
            let mut acc: f64 = match kind {
                ScanKind::Sum => 0.0,
                ScanKind::Prod => 1.0,
            };
            for a in 0..axis_len as i64 {
                let idx = ((o * axis_len as i64 + a) * inner as i64 + i) as usize;
                let v = x[idx] as f64;
                acc = match kind {
                    ScanKind::Sum => acc + v,
                    ScanKind::Prod => acc * v,
                };
                out[idx] = acc as f32;
                writes[idx] += 1;
            }
        }
    }
    assert!(
        writes.iter().all(|&w| w == 1),
        "各出力はちょうど 1 回書かれる"
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use fandhe_ai_backend_cpu::CpuBackendOps;
    use fandhe_ai_tensor_core::{BackendOps, Tensor};

    fn cpu(kind: ScanKind, x: &[f32], outer: usize, axis_len: usize, inner: usize) -> Vec<f32> {
        let t = Tensor::new(x.to_vec(), &[outer, axis_len, inner]).unwrap();
        let ops = CpuBackendOps::new();
        let r = match kind {
            ScanKind::Sum => BackendOps::cumsum(&ops, &t, 1),
            ScanKind::Prod => BackendOps::cumprod(&ops, &t, 1),
        }
        .unwrap();
        r.contiguous().as_slice().unwrap().to_vec()
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    fn lcg_data(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0
            })
            .collect()
    }

    #[test]
    fn matches_cpu_bit_exact_across_shapes() {
        // lanes が 1・ブロック境界（256）の前後・複数ブロックになる形状を含む。
        let shapes = [
            (1, 1, 1),
            (1, 5, 1),
            (2, 4, 3),
            (1, 7, 255),
            (1, 7, 256),
            (1, 7, 257),
            (3, 9, 100),
            (300, 3, 1),
        ];
        for (n, &(outer, axis_len, inner)) in shapes.iter().enumerate() {
            let x = lcg_data(outer * axis_len * inner, 100 + n as u64);
            for kind in [ScanKind::Sum, ScanKind::Prod] {
                assert_eq!(
                    bits(&simulate(kind, &x, outer, axis_len, inner)),
                    bits(&cpu(kind, &x, outer, axis_len, inner)),
                    "shape=({outer},{axis_len},{inner})"
                );
            }
        }
    }

    #[test]
    fn f64_accumulator_keeps_cancellation_case() {
        // f32 アキュムレータなら [1e8, 1, -1e8] の最終値は 0 になるが、f64 なら 1 を保つ。
        let x = [1e8f32, 1.0, -1e8];
        let out = simulate(ScanKind::Sum, &x, 1, 3, 1);
        assert_eq!(out, vec![1e8f32, 1e8, 1.0]);
        assert_eq!(bits(&out), bits(&cpu(ScanKind::Sum, &x, 1, 3, 1)));
    }

    #[test]
    fn prod_underflow_and_special_values_match_cpu() {
        let cases: [&[f32]; 3] = [
            &[1e-30, 1e-30, 1e30],
            &[2.0, f32::INFINITY, 0.0, 3.0],
            &[1.0, f32::NAN, 2.0, -0.0],
        ];
        for x in cases {
            for kind in [ScanKind::Sum, ScanKind::Prod] {
                assert_eq!(
                    bits(&simulate(kind, x, 1, x.len(), 1)),
                    bits(&cpu(kind, x, 1, x.len(), 1))
                );
            }
        }
    }

    #[test]
    fn empty_shapes_return_empty() {
        assert!(simulate(ScanKind::Sum, &[], 2, 0, 3).is_empty());
        assert!(simulate(ScanKind::Prod, &[], 0, 4, 3).is_empty());
    }
}
