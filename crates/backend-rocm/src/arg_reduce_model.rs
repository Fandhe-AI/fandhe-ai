//! `kernels_arg_reduce.rs` のカーネルを CPU 上で (block, thread) ごとに逐語再現するテスト専用モデル
//! （イシュー #3434・R3）。
//!
//! 役割: AMD 実機がない環境で、チャンク分割（`arg_all_plan`）・NaN の bit 検査による無視・
//! 厳密比較によるタイ（最初の添字が勝つ）・未確定チャンク（`pidx = -1`）の skip・全 NaN 時の添字 0・
//! 各出力が 1 回だけ書かれることを検証する。`CpuBackendOps::argmax`／`argmin` と `i32` 完全一致を
//! 確認する。wave 幅に依存しないカーネル群なので `WarpGeometry` は取らない。`RocmBackendOps` の
//! HostModel 経路（`ops.rs`）もこのモデルを呼ぶ。HIPRTC の結果の検証ではない（実機確認は #3159）。

use crate::arg_reduce::{ArgReduceKind, arg_all_plan};
use crate::launch::{INDEX_OPS_BLOCK_DIM, grid_1d};

/// カーネルの NaN 判定（`(__float_as_uint(v) & 0x7fffffffu) > 0x7f800000u`）。
fn is_nan_bits(v: f32) -> bool {
    (v.to_bits() & 0x7fff_ffff) > 0x7f80_0000
}

fn better(kind: ArgReduceKind, v: f32, best: f32) -> bool {
    match kind {
        ArgReduceKind::Max => v > best,
        ArgReduceKind::Min => v < best,
    }
}

/// 全軸 2 段（partial はチャンク 1 つ = スレッド 1 つ、finalize は単一スレッド）。`x` は非空。
pub(crate) fn simulate_all(kind: ArgReduceKind, x: &[f32]) -> i32 {
    assert!(
        !x.is_empty(),
        "全軸縮約の入力は非空（空は ops 層が CPU へ委譲）"
    );
    let numel = x.len();
    let (chunk_len, num_chunks) = arg_all_plan(numel);
    let mut pval = vec![f32::NAN; num_chunks];
    let mut pidx: Vec<Option<i32>> = vec![None; num_chunks];
    let (grid, block) = grid_1d("arg_model", num_chunks, INDEX_OPS_BLOCK_DIM).unwrap();
    for b in 0..grid.0 as usize {
        for th in 0..block.0 as usize {
            let t = b * block.0 as usize + th;
            if t >= num_chunks {
                continue;
            }
            let start = t * chunk_len;
            let end = (start + chunk_len).min(numel);
            let mut has_best = false;
            let mut best_val = 0.0f32;
            let mut best_idx = -1i32;
            for (i, &v) in x.iter().enumerate().take(end).skip(start) {
                if is_nan_bits(v) {
                    continue;
                }
                if !has_best || better(kind, v, best_val) {
                    has_best = true;
                    best_val = v;
                    best_idx = i as i32;
                }
            }
            assert!(pidx[t].is_none(), "チャンク {t} は 1 回だけ書かれる");
            pval[t] = best_val;
            pidx[t] = Some(best_idx);
        }
    }
    // finalize: blockIdx.x == 0 && threadIdx.x == 0 の単一スレッド。
    let mut has_best = false;
    let mut best_val = 0.0f32;
    let mut best_idx = 0i32;
    for (slot, &v) in pidx.iter().zip(&pval) {
        let idx = slot.expect("partial は全て書かれる");
        if idx < 0 {
            continue;
        }
        if !has_best || better(kind, v, best_val) {
            has_best = true;
            best_val = v;
            best_idx = idx;
        }
    }
    best_idx
}

/// 軸縮約（1 スレッド = 1 出力）。各出力がちょうど 1 回書かれることを検査する。
pub(crate) fn simulate_axis(
    kind: ArgReduceKind,
    x: &[f32],
    outer: usize,
    axis_len: usize,
    inner: usize,
) -> Vec<i32> {
    let total = outer * inner;
    assert_eq!(x.len(), total * axis_len);
    if total == 0 {
        return Vec::new();
    }
    let mut out: Vec<Option<i32>> = vec![None; total];
    let (grid, block) = grid_1d("arg_model", total, INDEX_OPS_BLOCK_DIM).unwrap();
    for b in 0..grid.0 as usize {
        for th in 0..block.0 as usize {
            let idx = b * block.0 as usize + th;
            if idx >= total {
                continue;
            }
            let (o, i) = (idx / inner, idx % inner);
            let mut has_best = false;
            let mut best_val = 0.0f32;
            let mut best_idx = 0i32;
            for a in 0..axis_len {
                let v = x[(o * axis_len + a) * inner + i];
                if is_nan_bits(v) {
                    continue;
                }
                if !has_best || better(kind, v, best_val) {
                    has_best = true;
                    best_val = v;
                    best_idx = a as i32;
                }
            }
            assert!(out[idx].is_none(), "出力 {idx} は 1 回だけ書かれる");
            out[idx] = Some(best_idx);
        }
    }
    out.into_iter()
        .map(|v| v.expect("全出力が書かれる"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arg_reduce::{ARG_MAX_CHUNKS, ARG_MIN_CHUNK};
    use crate::reduce::reduce_axis_layout;
    use fandhe_ai_backend_cpu::CpuBackendOps;
    use fandhe_ai_tensor_core::{BackendOps, Tensor};

    const KINDS: [ArgReduceKind; 2] = [ArgReduceKind::Max, ArgReduceKind::Min];

    fn cpu_all(kind: ArgReduceKind, x: &[f32]) -> i32 {
        let t = Tensor::new(x.to_vec(), &[x.len()]).unwrap();
        let ops = CpuBackendOps::new();
        let r = match kind {
            ArgReduceKind::Max => ops.argmax(&t, None),
            ArgReduceKind::Min => ops.argmin(&t, None),
        }
        .unwrap();
        r.as_slice().unwrap()[0]
    }

    fn cpu_axis(kind: ArgReduceKind, x: &[f32], shape: &[usize], axis: usize) -> Vec<i32> {
        let t = Tensor::new(x.to_vec(), shape).unwrap();
        let ops = CpuBackendOps::new();
        let r = match kind {
            ArgReduceKind::Max => ops.argmax(&t, Some(axis)),
            ArgReduceKind::Min => ops.argmin(&t, Some(axis)),
        }
        .unwrap();
        r.contiguous().as_slice().unwrap().to_vec()
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
    fn all_matches_cpu_across_sizes_and_chunk_boundaries() {
        let sizes = [
            1,
            2,
            255,
            4095,
            ARG_MIN_CHUNK,
            ARG_MIN_CHUNK + 1,
            3 * ARG_MIN_CHUNK + 7,
            ARG_MIN_CHUNK * ARG_MAX_CHUNKS / 64,
        ];
        for (n, &len) in sizes.iter().enumerate() {
            let x = lcg_data(len, 11 + n as u64);
            for kind in KINDS {
                assert_eq!(
                    simulate_all(kind, &x),
                    cpu_all(kind, &x),
                    "{kind:?} n={len}"
                );
            }
        }
    }

    #[test]
    fn ties_nan_and_all_nan_match_cpu() {
        let mut cases: Vec<Vec<f32>> = vec![
            vec![1.0, 3.0, 3.0, 2.0],
            vec![3.0, 3.0, 3.0],
            vec![f32::NAN, 1.0, f32::NAN, 5.0, f32::NAN],
            vec![f32::NAN; 10],
            vec![0.0, -0.0, 0.0],
            vec![-0.0, 0.0],
            vec![f32::INFINITY, f32::INFINITY, 1.0],
            vec![f32::NEG_INFINITY, f32::NEG_INFINITY],
        ];
        // 最大値がチャンク境界をまたぐ位置・複数チャンクにタイがある入力（最初の添字が勝つ）。
        let len = 3 * ARG_MIN_CHUNK + 100;
        let mut v = lcg_data(len, 5);
        v[ARG_MIN_CHUNK - 1] = 99.0;
        v[ARG_MIN_CHUNK] = 99.0;
        v[2 * ARG_MIN_CHUNK + 3] = 99.0;
        v[7] = f32::NAN;
        cases.push(v);
        // 先頭チャンクが全 NaN（pidx = -1 を skip）。
        let mut v = lcg_data(2 * ARG_MIN_CHUNK, 6);
        for e in v.iter_mut().take(ARG_MIN_CHUNK) {
            *e = f32::NAN;
        }
        cases.push(v);
        for x in &cases {
            for kind in KINDS {
                assert_eq!(
                    simulate_all(kind, x),
                    cpu_all(kind, x),
                    "{kind:?} n={}",
                    x.len()
                );
            }
        }
    }

    #[test]
    fn axis_matches_cpu_including_ties_and_nan() {
        let shapes: Vec<Vec<usize>> = vec![
            vec![3, 4],
            vec![2, 3, 5],
            vec![1, 300],
            vec![300, 1],
            vec![257, 3],
            vec![4, 1, 7],
        ];
        for (n, shape) in shapes.iter().enumerate() {
            let numel: usize = shape.iter().product();
            let mut x = lcg_data(numel, 70 + n as u64);
            // タイと NaN を混ぜる。
            if numel > 6 {
                x[1] = x[0];
                x[4] = f32::NAN;
                x[numel - 1] = x[numel - 2];
            }
            for axis in 0..shape.len() {
                let (outer, axis_len, inner) = reduce_axis_layout(shape, axis).unwrap();
                for kind in KINDS {
                    assert_eq!(
                        simulate_axis(kind, &x, outer, axis_len, inner),
                        cpu_axis(kind, &x, shape, axis),
                        "{kind:?} shape={shape:?} axis={axis}"
                    );
                }
            }
        }
    }

    #[test]
    fn all_nan_axis_returns_zero_like_cpu() {
        let x = vec![f32::NAN; 6];
        for kind in KINDS {
            assert_eq!(simulate_axis(kind, &x, 2, 3, 1), vec![0, 0]);
            assert_eq!(
                simulate_axis(kind, &x, 2, 3, 1),
                cpu_axis(kind, &x, &[2, 3], 1)
            );
        }
    }

    #[test]
    fn empty_output_returns_empty() {
        assert!(simulate_axis(ArgReduceKind::Max, &[], 0, 4, 3).is_empty());
        assert!(simulate_axis(ArgReduceKind::Min, &[], 2, 0, 0).is_empty());
    }
}
