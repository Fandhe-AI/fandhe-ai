//! 2 段 block 縮約（wave 内 butterfly → `warp_sums` → wave 0 が再縮約）のホスト側モデル（テスト専用）。
//!
//! 役割: ROCm 非搭載環境で、`WarpGeometry` の導出値（`WARPS_PER_BLOCK` からのバッファ長・
//! 読み出し境界）の正しさと、CUDA 由来の 8 warp 固定が wave64 で破綻することを固定する。
//! マクロ意味論のモデルであり、実カーネルの逐語モデルは R3 以降が書く。HIPRTC・実機の確認は #3159。

use crate::warp_geometry::WarpGeometry;

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum ModelError {
    UninitRead(usize),
    OutOfBounds(usize),
}

/// `__shfl_xor(v, off, width)`: 全 lane が旧値を同時に読む。区間外ソースは自値。
fn butterfly(vals: &mut [f32], width: usize, half: usize) {
    let mut off = half;
    while off > 0 {
        let old = vals.to_vec();
        for (i, v) in vals.iter_mut().enumerate() {
            let src = i ^ off;
            let same_group = src / width == i / width && src < old.len();
            let other = if same_group { old[src] } else { old[i] };
            *v = old[i] + other;
        }
        off >>= 1;
    }
}

/// `buf_len` と読み出し境界 `read_bound`（CUDA 版では 8 固定）を明示できる縮約。戻りは wave 0 の lane 0 の値。
pub(crate) fn reduce_with(
    g: WarpGeometry,
    input: &[f32],
    buf_len: usize,
    read_bound: usize,
) -> Result<f32, ModelError> {
    let w = g.width() as usize;
    let mut v = input.to_vec();
    butterfly(&mut v, w, g.half_width() as usize);
    let mut warp_sums: Vec<Option<f32>> = vec![None; buf_len];
    for warp in 0..input.len() / w {
        *warp_sums
            .get_mut(warp)
            .ok_or(ModelError::OutOfBounds(warp))? = Some(v[warp * w]);
    }
    let mut second = vec![0.0f32; w];
    for (lane, slot) in second.iter_mut().enumerate() {
        if lane < read_bound {
            *slot = warp_sums
                .get(lane)
                .ok_or(ModelError::OutOfBounds(lane))?
                .ok_or(ModelError::UninitRead(lane))?;
        }
    }
    butterfly(&mut second, w, g.half_width() as usize);
    Ok(second[0])
}

/// 導出値（`WARPS_PER_BLOCK`）を使う正しい縮約。
pub(crate) fn reduce(g: WarpGeometry, input: &[f32]) -> Result<f32, ModelError> {
    let n = g
        .warps_per_block(input.len() as u32)
        .expect("テスト入力は有効な block_dim") as usize;
    reduce_with(g, input, n, n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fandhe_ai_backend_cpu::assert_parity;

    const DIMS: [usize; 5] = [64, 128, 256, 512, 1024];

    #[test]
    fn integer_inputs_match_sequential_sum_exactly() {
        for w in [32, 64] {
            let g = WarpGeometry::new(w).unwrap();
            for d in DIMS {
                let input: Vec<f32> = (0..d).map(|i| (i % 7) as f32 - 3.0).collect();
                let expected: f32 = input.iter().sum();
                assert_eq!(reduce(g, &input).unwrap(), expected, "w{w} d{d}");
            }
        }
    }

    #[test]
    fn pseudo_random_inputs_pass_unified_parity() {
        let mut s: u32 = 0x1234_5678;
        for w in [32, 64] {
            let g = WarpGeometry::new(w).unwrap();
            for d in DIMS {
                let input: Vec<f32> = (0..d)
                    .map(|_| {
                        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        (s >> 8) as f32 / (1u32 << 24) as f32 - 0.5
                    })
                    .collect();
                let expected: f32 = input.iter().map(|&x| x as f64).sum::<f64>() as f32;
                let got = reduce(g, &input).unwrap();
                assert_parity(&format!("w{w} d{d}"), &[got], &[expected]);
            }
        }
    }

    #[test]
    fn cuda_fixed_eight_warps_breaks_on_wave64() {
        let g64 = WarpGeometry::new(64).unwrap();
        let input = vec![1.0f32; 256];
        // バッファ長 8 固定・lane < 8 読みだと、書かれたのは 4 要素だけで未初期化を読む。
        assert_eq!(
            reduce_with(g64, &input, 8, 8),
            Err(ModelError::UninitRead(4))
        );
        assert_eq!(reduce(g64, &input).unwrap(), 256.0);
        // wave32 では 8 固定が偶然成立する（CUDA の前提）。
        let g32 = WarpGeometry::new(32).unwrap();
        assert_eq!(reduce_with(g32, &input, 8, 8).unwrap(), 256.0);
    }

    #[test]
    fn too_small_buffer_is_out_of_bounds() {
        let g = WarpGeometry::new(32).unwrap();
        assert_eq!(
            reduce_with(g, &[1.0; 256], 4, 4),
            Err(ModelError::OutOfBounds(4))
        );
    }
}
