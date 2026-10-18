//! `kernels_softmax.rs`・`kernels_log_softmax_backward.rs` の 3 カーネルを CPU 上で lane ごとに
//! 逐語再現するテスト専用モデル（イシュー #3435・R4）。
//!
//! 役割: AMD 実機がない環境で、wave 幅 32／64 の両方について (1) vec 経路／スカラー経路の担当要素順、
//! (2) `__shfl_xor` butterfly（全 lane が旧値を同時に読む）での `(m, l)` 結合、(3) grid-stride が
//! 全行を 1 回ずつ処理すること、(4) 出力の書き出し範囲（REQ-8 ガード）、(5) backward の lane 0 逐次
//! `f64` 和と CTA ガードを検証する。`RocmBackendOps` のテスト用エンジン（`HostModel`）にも使う。
//! 演算は FMA 縮約なし（カーネルの `fp contract(off)` と同じ）。HIPRTC のコンパイル結果そのものの
//! 検証ではない（実機確認は #3159）。

use crate::kernels_softmax::SOFTMAX_MAX_GRID;
use crate::softmax::softmax_grid;

const MASK_E2: f32 = -3.402_823_5e38;

/// 担当要素 1 個ぶんの online 更新（カーネルの `m_new = fmaxf(m, raw)` 以降と同一）。
fn update(m: &mut f32, l: &mut f32, raw: f32, scale: f32) {
    let m_new = m.max(raw);
    if m_new > *m {
        *l *= ((*m - m_new) * scale).exp2();
    }
    *l += ((raw - m_new) * scale).exp2();
    *m = m_new;
}

/// 1 行ぶんの全 lane の `(m, l)`（担当要素走査 → butterfly 結合後）。
fn row_state(width: usize, row: &[f32], scale: f32) -> Vec<(f32, f32)> {
    let cols = row.len();
    let vec_cols = if cols.is_multiple_of(4) { cols } else { 0 };
    let mut st: Vec<(f32, f32)> = (0..width)
        .map(|lane| {
            let (mut m, mut l) = (MASK_E2, 0.0f32);
            let mut base = lane * 4;
            while base < vec_cols {
                if base + 3 < cols {
                    for k in 0..4 {
                        update(&mut m, &mut l, row[base + k], scale);
                    }
                }
                base += width * 4;
            }
            let mut i = vec_cols + lane;
            while i < cols {
                update(&mut m, &mut l, row[i], scale);
                i += width;
            }
            (m, l)
        })
        .collect();
    // `WARP_SHFL_XOR`: 全 lane が旧値を同時に読む。
    let mut off = width / 2;
    while off > 0 {
        let old = st.clone();
        for (lane, s) in st.iter_mut().enumerate() {
            let (m, l) = old[lane];
            let (m_o, l_o) = old[lane ^ off];
            let m_t = m.max(m_o);
            let l_self = if m_t > m {
                l * ((m - m_t) * scale).exp2()
            } else {
                l
            };
            let l_peer = if m_t > m_o {
                l_o * ((m_o - m_t) * scale).exp2()
            } else {
                l_o
            };
            *s = (m_t, l_self + l_peer);
        }
        off >>= 1;
    }
    st
}

/// softmax／log_softmax カーネルのモデル。`grid = softmax_grid(rows)` の grid-stride で全行を処理し、
/// 各出力要素がちょうど 1 回書かれることを assert する。
pub(crate) fn simulate_softmax(
    width: usize,
    x: &[f32],
    rows: usize,
    cols: usize,
    log: bool,
) -> Vec<f32> {
    assert_eq!(x.len(), rows * cols);
    let scale = std::f32::consts::LOG2_E;
    let mut out = vec![f32::NAN; rows * cols];
    let mut writes = vec![0u8; rows * cols];
    let grid = softmax_grid(rows.max(1)) as usize;
    assert!(grid <= SOFTMAX_MAX_GRID as usize);
    let vec_cols = if cols.is_multiple_of(4) { cols } else { 0 };
    for block in 0..grid {
        let mut r = block;
        while r < rows {
            let row = &x[r * cols..(r + 1) * cols];
            let st = row_state(width, row, scale);
            for (lane, &(m, l)) in st.iter().enumerate() {
                let inv_l = 1.0f32 / l;
                let log_l = l.ln();
                let val = |v: f32| {
                    if log {
                        (v - m) - log_l
                    } else {
                        ((v - m) * scale).exp2() * inv_l
                    }
                };
                let mut put = |idx: usize, v: f32| {
                    assert!(idx < cols, "範囲外書き込み");
                    out[r * cols + idx] = val(v);
                    writes[r * cols + idx] += 1;
                };
                let mut base = lane * 4;
                while base < vec_cols {
                    if base + 3 < cols {
                        for k in 0..4 {
                            put(base + k, row[base + k]);
                        }
                    }
                    base += width * 4;
                }
                let mut i = vec_cols + lane;
                while i < cols {
                    put(i, row[i]);
                    i += width;
                }
            }
            r += grid;
        }
    }
    assert!(writes.iter().all(|&w| w == 1), "全要素が 1 回だけ書かれる");
    out
}

/// `log_softmax_backward` カーネルのモデル（grid = rows。lane 0 が `i` 昇順 `f64` 逐次和）。
pub(crate) fn simulate_log_softmax_backward(
    width: usize,
    y: &[f32],
    g: &[f32],
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    assert_eq!(y.len(), rows * cols);
    assert_eq!(g.len(), rows * cols);
    let mut out = vec![f32::NAN; rows * cols];
    let mut writes = vec![0u8; rows * cols];
    for row in 0..rows {
        let base = row * cols;
        let mut acc = 0.0f64;
        for i in 0..cols {
            acc += f64::from(g[base + i]);
        }
        for lane in 0..width {
            let mut i = lane;
            while i < cols {
                let exp_y = f64::from(y[base + i].exp());
                out[base + i] = (f64::from(g[base + i]) - exp_y * acc) as f32;
                writes[base + i] += 1;
                i += width;
            }
        }
    }
    assert!(writes.iter().all(|&w| w == 1));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use fandhe_ai_backend_cpu::{CpuBackendOps, assert_parity};
    use fandhe_ai_tensor_core::{BackendOps, Tensor};

    fn data(len: usize, seed: u32, amp: f32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let h = (i as u32).wrapping_mul(2654435761).wrapping_add(seed);
                ((h >> 8) % 2001) as f32 / 1000.0 * amp - amp
            })
            .collect()
    }

    fn cpu(x: &[f32], rows: usize, cols: usize, log: bool) -> Vec<f32> {
        let t = Tensor::new(x.to_vec(), &[rows, cols]).unwrap();
        let o = if log {
            CpuBackendOps.log_softmax(&t, 1)
        } else {
            CpuBackendOps.softmax(&t, 1)
        }
        .unwrap();
        o.as_slice().unwrap().to_vec()
    }

    /// テスト内に移植した `grad::log_softmax_vjp_along` と同式（f64 逐次和・最後に 1 回 f32）。
    fn cpu_backward(y: &[f32], g: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * cols];
        for r in 0..rows {
            let b = r * cols;
            let sum: f64 = (0..cols).map(|i| f64::from(g[b + i])).sum();
            for i in 0..cols {
                let e = f64::from(y[b + i].exp());
                out[b + i] = (f64::from(g[b + i]) - e * sum) as f32;
            }
        }
        out
    }

    const COLS: [usize; 9] = [1, 3, 4, 63, 64, 65, 128, 1000, 4096];

    #[test]
    fn forward_matches_cpu_for_both_wave_widths() {
        for width in [32usize, 64] {
            for &cols in &COLS {
                for rows in [1usize, 7] {
                    let x = data(rows * cols, 11, 5.0);
                    for log in [false, true] {
                        let got = simulate_softmax(width, &x, rows, cols, log);
                        let want = cpu(&x, rows, cols, log);
                        assert_parity(&format!("w{width} {rows}x{cols} log={log}"), &got, &want);
                    }
                }
            }
        }
    }

    #[test]
    fn grid_stride_covers_rows_beyond_max_grid() {
        let rows = SOFTMAX_MAX_GRID as usize + 5;
        let cols = 3;
        let x = data(rows * cols, 5, 3.0);
        for width in [32usize, 64] {
            let got = simulate_softmax(width, &x, rows, cols, false);
            assert_parity("grid-stride", &got, &cpu(&x, rows, cols, false));
        }
    }

    #[test]
    fn extreme_values_stay_finite() {
        for width in [32usize, 64] {
            let rows = vec![
                vec![f32::MAX, -f32::MAX, 0.0, f32::MAX, 1.0],
                vec![-f32::MAX; 5],
                vec![f32::MAX; 5],
                vec![3.0, -f32::MAX, -f32::MAX, -f32::MAX, -f32::MAX],
            ];
            for row in rows {
                let cols = row.len();
                for log in [false, true] {
                    // log_softmax の `v - m` は MAX と -MAX が同居する行で真に -inf になる
                    // （CPU 参照も同じ）ため、その行は softmax のみ検査する。
                    if log && row.contains(&f32::MAX) && row.contains(&-f32::MAX) {
                        continue;
                    }
                    let got = simulate_softmax(width, &row, 1, cols, log);
                    assert!(got.iter().all(|v| v.is_finite()), "{row:?} {got:?}");
                    assert_parity(
                        &format!("extreme w{width} log={log}"),
                        &got,
                        &cpu(&row, 1, cols, log),
                    );
                }
            }
        }
    }

    #[test]
    fn softmax_rows_sum_to_one() {
        let x = data(4 * 100, 3, 8.0);
        let got = simulate_softmax(64, &x, 4, 100, false);
        for r in 0..4 {
            let s: f32 = got[r * 100..(r + 1) * 100].iter().sum();
            assert!((s - 1.0).abs() < 1e-4, "{s}");
        }
    }

    #[test]
    fn backward_matches_reference_and_survives_cancellation() {
        for width in [32usize, 64] {
            for &cols in &COLS {
                for rows in [1usize, 5] {
                    let x = data(rows * cols, 21, 4.0);
                    let y = cpu(&x, rows, cols, true);
                    let g = data(rows * cols, 22, 1.0);
                    let got = simulate_log_softmax_backward(width, &y, &g, rows, cols);
                    assert_parity(
                        &format!("bwd w{width} {rows}x{cols}"),
                        &got,
                        &cpu_backward(&y, &g, rows, cols),
                    );
                }
            }
            // 相殺入力（lane 0 逐次和でないと桁落ちの位置がホストとずれる）。
            let y = vec![-(4.0f32).ln(); 4];
            let g = vec![1e20f32, -1e20, 1.0, 0.0];
            let got = simulate_log_softmax_backward(width, &y, &g, 1, 4);
            assert_parity("cancel", &got, &cpu_backward(&y, &g, 1, 4));
            // 大きな上流勾配でも overflow しない。
            let g = vec![2e38f32, 2e38, -2e38, 1.0];
            let got = simulate_log_softmax_backward(width, &y, &g, 1, 4);
            assert!(got.iter().all(|v| v.is_finite()), "{got:?}");
            assert_parity("big-g", &got, &cpu_backward(&y, &g, 1, 4));
        }
    }
}
